"""
邮件处理流水线

组装 Gmail → Agent → Embedding → Milvus 的完整链路。
每个环节都是独立模块，Pipeline 只负责编排调用顺序。

使用方式:
    from rag_modules import Pipeline
    pipeline = Pipeline()
    pipeline.process_inbox(max_emails=10)       # 处理收件箱
    pipeline.process_inbox_v2(max_emails=10)    # 并行编排版本
    pipeline.analyze_sent_tone(max_emails=10)   # 分析发件箱语气
    pipeline.store_context("项目背景文本...")   # 存入长期记忆
"""

import json
import logging
import re
import time
from collections import Counter
from typing import Optional

import config
from rag_modules.email import GmailAuth, EmailFetcher, EmailParser
from rag_modules.agents import (
    SummarizerAgent, ClassifierAgent, ContextSummaryAgent, OrchestratorAgent,
)
from rag_modules.storage import embed, VectorStore

logger = logging.getLogger(__name__)


class Pipeline:
    """
    邮件处理流水线

    依赖链：GmailAuth → EmailFetcher/EmailParser → Agent → Embedding → VectorStore
    """

    def __init__(self):
        self.store = VectorStore()
        self.summarizer = SummarizerAgent()
        self.classifier = ClassifierAgent()
        self.context_compressor = ContextSummaryAgent()
        self.orchestrator: Optional[OrchestratorAgent] = None

    def _get_orchestrator(self) -> OrchestratorAgent:
        if self.orchestrator is None:
            self.orchestrator = OrchestratorAgent()
        return self.orchestrator

    def _connect_gmail(self) -> tuple[EmailFetcher, EmailParser]:
        service = GmailAuth().authenticate()
        return EmailFetcher(service), EmailParser(service)

    # ── 收件箱 ──────────────────────────────────────────────

    def process_inbox(self, max_emails: int = 5):
        """处理收件箱邮件：拉取 → 摘要 → 嵌入 → 分类 → 存入 Milvus"""
        fetcher, parser = self._connect_gmail()
        messages = fetcher.fetch_inbox(max_results=max_emails)
        logger.info(f"拉取到 {len(messages)} 封收件箱邮件")

        for i, msg in enumerate(messages, start=1):
            parsed = parser.parse(msg["id"])
            if parsed is None:
                continue

            email_data = {
                "id": msg["id"], "threadId": msg.get("threadId"),
                "snippet": msg.get("snippet"),
                "subject": parsed["subject"], "from": parsed["from"],
                "date": parsed["date"], "body": parsed["body"],
                "attachments": parsed["attachments"],
            }

            summary = self.summarizer.run(email_data)
            email_data["embedding"] = (
                embed(summary.data) if summary.success else embed(email_data["body"])
            )

            classify = self.classifier.run(email_data)
            if classify.success:
                email_data["priority"] = classify.data.get("priority", "Low")
                email_data["reason"] = classify.data.get("reason", "")
            else:
                email_data["priority"] = "Low"
                email_data["reason"] = ""

            self.store.insert_inbox_email(email_data)
            logger.info(f"邮件 {email_data['id']} 处理完成 [{i}/{len(messages)}] "
                        f"优先级: {email_data['priority']}")

    # ── Orchestrator 版本（并行 + 自动分支）─────────────────

    def process_inbox_v2(self, max_emails: int = 5):
        """【推荐】OrchestratorAgent 并行编排版本"""
        fetcher, parser = self._connect_gmail()
        messages = fetcher.fetch_inbox(max_results=max_emails)

        email_list = []
        for msg in messages:
            parsed = parser.parse(msg["id"])
            if parsed is None:
                continue
            email_list.append({
                "id": msg["id"], "threadId": msg.get("threadId"),
                "snippet": msg.get("snippet"),
                "sender": parsed["from"], "from": parsed["from"],
                "date": parsed["date"], "subject": parsed["subject"],
                "body": parsed["body"], "attachments": parsed["attachments"],
            })

        result = self._get_orchestrator().process_inbox(email_list, parallel=True)
        if not result.success:
            logger.error(f"编排失败: {result.error}")
            return

        for i, email in enumerate(email_list):
            results = result.data.get("results", [])
            agent_results = results[i] if i < len(results) else {}
            classify_data = agent_results.get("ClassifierAgent", {})
            if isinstance(classify_data, str):
                try:
                    classify_data = json.loads(classify_data)
                except json.JSONDecodeError:
                    classify_data = {}

            summary_text = agent_results.get("SummarizerAgent", "")
            self.store.insert_inbox_email({
                **email,
                "embedding": embed(str(summary_text) or email["body"]),
                "priority": classify_data.get("priority", "Low"),
                "reason": classify_data.get("reason", ""),
            })

    # ── 发件箱 ──────────────────────────────────────────────

    def analyze_sent_tone(self, max_emails: int = 5):
        """分析发件箱邮件语气，存入向量库"""
        fetcher, parser = self._connect_gmail()
        messages = fetcher.fetch_sent(max_results=max_emails)
        logger.info(f"拉取到 {len(messages)} 封发件箱邮件")

        for i, msg in enumerate(messages, start=1):
            parsed = parser.parse(msg["id"])
            if parsed is None:
                continue

            email_data = {
                "id": msg["id"], "threadId": msg.get("threadId"),
                "snippet": msg.get("snippet"),
                "subject": parsed["subject"], "from": parsed["from"],
                "sender": parsed["from"], "to": parsed.get("to", ""),
                "date": parsed["date"], "body": parsed["body"],
                "attachments": parsed["attachments"],
            }

            summary = self.summarizer.run(email_data)
            email_data["embedding"] = (
                embed(summary.data) if summary.success else embed(email_data["body"])
            )
            self.store.insert_sent_email(email_data)
            logger.info(f"发件箱 {email_data['id']} 处理完成 [{i}/{len(messages)}]")

    # ── 联系人 ──────────────────────────────────────────────

    def top_contacts(self, n: int | None = None) -> dict:
        """统计最近联系人"""
        if n is None:
            n = config.SmartEmailAgentConfig.number_of_common_contacts

        fetcher, parser = self._connect_gmail()

        def _extract(parsed, field):
            val = parsed.get(field, "")
            if not val:
                return ""
            m = re.search(r"<([^>]+)>", val)
            return m.group(1).lower() if m else val.strip().lower()

        senders = Counter()
        for msg in fetcher.fetch_inbox(max_results=10):
            p = parser.parse(msg["id"])
            if p:
                addr = _extract(p, "from")
                if addr:
                    senders[addr] += 1

        receivers = Counter()
        for msg in fetcher.fetch_sent(max_results=10):
            p = parser.parse(msg["id"])
            if p:
                addr = _extract(p, "to")
                if addr:
                    receivers[addr] += 1

        return {
            "top_senders": senders.most_common(n),
            "top_receivers": receivers.most_common(n),
        }

    # ── 长期记忆 ────────────────────────────────────────────

    def store_context(self, context: str):
        """压缩上下文 → 嵌入 → 存入 Milvus 长期记忆"""
        result = self.context_compressor.run({"context": context})
        compressed = result.data if result.success else context

        self.store.insert_context(
            content=compressed,
            embedding=embed(compressed),
            timestamp=time.time(),
        )
        logger.info(f"长期记忆已存储 ({len(context)} → {len(compressed)} 字符)")
