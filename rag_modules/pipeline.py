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
    QueryAgent,
)
from rag_modules.agents.base import AgentResult
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
        self.querier: Optional[QueryAgent] = None

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

    # ── RAG 问答 ────────────────────────────────────────────

    def _get_querier(self) -> QueryAgent:
        """懒加载 QueryAgent"""
        if self.querier is None:
            self.querier = QueryAgent()
        return self.querier

    def query(
        self,
        question: str,
        top_k: int = 5,
        scope: str = "all",
        history: list | None = None,
    ):
        """
        RAG 问答入口 — 语义搜索 + LLM 回答

        Args:
            question: 用户自然语言提问
            top_k:    Milvus 向量搜索返回条数
            scope:    "all" | "inbox" | "context" — 搜索范围
            history:  对话历史，格式 [{"question": "...", "answer": "..."}, ...]

        Returns:
            AgentResult.data = {
                "answer": "带引用标注的自然语言回答",
                "confidence": "high|medium|low",
                "cited_count": N
            }
            AgentResult.metadata["sources"] = 邮件详细列表
        """
        logger.info(f"RAG 查询: {question[:80]}...")

        # Step 1: Embed 问题
        q_vector = embed(question)
        if not q_vector:
            return AgentResult(
                success=False,
                error="文本向量化失败，请检查 Embedding API 配置",
            )

        # Step 2: 搜索向量库
        emails = []
        contexts = []

        if scope in ("all", "inbox"):
            try:
                emails = self.store.search_inbox(q_vector, top_k)
            except Exception as e:
                logger.warning(f"收件箱搜索失败: {e}")

        if scope in ("all", "context"):
            try:
                contexts = self.store.search_context(q_vector, top_k)
            except Exception as e:
                logger.warning(f"长期记忆搜索失败: {e}")

        # Step 3: 格式化上下文
        context_str = self._format_rag_context(emails, contexts)

        # Step 4: LLM 生成回答
        result = self._get_querier().run({
            "question": question,
            "context": context_str,
            "history": history or [],
        })

        # Step 5: 附加来源元数据（用于前端展示引用卡片）
        result.metadata["sources"] = {
            "emails": [
                {
                    "subject":     e.get("entity", {}).get("subject", ""),
                    "from_address": e.get("entity", {}).get("from_address", ""),
                    "date":        e.get("entity", {}).get("date", ""),
                    "priority":    e.get("entity", {}).get("priority", ""),
                    "body":        (e.get("entity", {}).get("body", "") or "")[:200],
                    "distance":    round(e.get("distance", 0), 4),
                }
                for e in emails
            ],
            "contexts": [
                {
                    "content":  (c.get("entity", {}).get("content", "") or "")[:200],
                    "timestamp": c.get("entity", {}).get("timestamp", 0),
                    "distance": round(c.get("distance", 0), 4),
                }
                for c in contexts
            ],
        }

        logger.info(
            f"RAG 查询完成 — 回答长度: {len(result.data.get('answer', '')) if result.data else 0}, "
            f"邮件来源: {len(emails)}, 记忆来源: {len(contexts)}"
        )
        return result

    # ── 统一入口：LLM 自动路由 ─────────────────────────────

    def chat(self, user_input: str, top_k: int = 5, history: list = None):
        """
        唯一用户入口 — LLM 自动判断走哪条路径。

        - 查询类 ("有没有紧急邮件")    → RAG 问答
        - 处理类 ("处理收件箱")        → 多 Agent 批量处理
        - 回复类 ("帮我回复张三")       → 摘要 + 起草回复
        - 记忆类 ("记住项目背景")       → 上下文压缩存储

        原理：OrchestratorAgent.plan() 生成执行计划，
              Pipeline 根据计划中的 Agent 类型自动路由。
        """
        logger.info(f"用户输入: {user_input[:80]}...")

        # 调用编排器生成执行计划（plan 是一个 step 列表，每个 step 含 "agent" 等字段）
        plan = self._get_orchestrator().plan(user_input)

        # 提取本次计划中涉及的所有 Agent 名称（集合去重）
        agent_names = {s["agent"] for s in plan}

        # 路由判断：若只用到 QueryAgent，走轻量查询链路；否则走完整处理链路
        route = "query" if agent_names == {"QueryAgent"} else "process"

        logger.info(f"路由: {route} | Agent: {agent_names} | 计划: {len(plan)} 步")

        # 2. 纯查询 → RAG 路径
        if agent_names == {"QueryAgent"}:
            return self.query(question=user_input, top_k=top_k, history=history)

        # 3. 上下文压缩 → 直接压缩
        if agent_names == {"ContextSummaryAgent"}:
            result = self.context_compressor.run({"context": user_input})
            if result.success:
                self.store.insert_context(
                    content=result.data,
                    embedding=embed(result.data),
                    timestamp=time.time(),
                )
                logger.info(f"长期记忆已存储 ({len(result.data)} 字符)")
            return result

        # 4. 处理/回复类 → 拉邮件 → Orchestrator 多 Agent 并行
        fetcher, parser = self._connect_gmail()
        messages = fetcher.fetch_inbox(max_results=top_k)
        logger.info(f"拉取到 {len(messages)} 封邮件，开始编排处理...")

        emails = []
        for msg in messages:
            parsed = parser.parse(msg["id"])
            if parsed is None:
                continue
            emails.append({
                "id": msg["id"], "threadId": msg.get("threadId"),
                "snippet": msg.get("snippet"),
                "sender": parsed["from"], "from": parsed["from"],
                "date": parsed["date"], "subject": parsed["subject"],
                "body": parsed["body"], "attachments": parsed["attachments"],
            })

        if not emails:
            return AgentResult(success=False, error="收件箱为空，没有邮件可处理")

        # 执行编排
        result = self._get_orchestrator().run({
            "goal": user_input,
            "emails": emails,
            "plan": plan,
            "parallel": True,
        })

        # 处理结果存入向量库
        if result.success and result.data.get("results"):
            for i, email in enumerate(emails):
                agent_results = result.data["results"][i] if i < len(result.data["results"]) else {}
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
            logger.info(f"批量处理完成: {len(emails)} 封邮件已存入向量库")

        return result

    def _format_rag_context(
        self,
        emails: list[dict],
        contexts: list[dict],
        max_body_len: int = 300,
    ) -> str:
        """
        将 Milvus 搜索结果格式化为 LLM 可消费的 markdown 文本。

        Args:
            emails:     search_inbox 返回的邮件列表
            contexts:   search_context 返回的记忆列表
            max_body_len: 邮件正文最大保留字符数（防 token 溢出）

        Returns:
            格式化的上下文字符串
        """
        parts = []

        if emails:
            parts.append("## 相关邮件\n")
            for i, e in enumerate(emails, start=1):
                entity = e.get("entity", {})
                body = (entity.get("body", "") or "")[:max_body_len]
                if len(entity.get("body", "") or "") > max_body_len:
                    body += "..."

                parts.append(
                    f"### [邮件{i}] {entity.get('subject', '无主题')}\n"
                    f"- 发件人: {entity.get('from_address', '未知')}\n"
                    f"- 日期: {entity.get('date', '未知')}\n"
                    f"- 优先级: {entity.get('priority', '未分类')}\n"
                    f"- 相似度: {round(e.get('distance', 0), 4)}\n"
                    f"\n{body}\n"
                )

        if contexts:
            parts.append("## 长期记忆\n")
            for i, c in enumerate(contexts, start=1):
                entity = c.get("entity", {})
                ts = entity.get("timestamp", 0)
                time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "未知时间"

                parts.append(
                    f"### [记忆{i}] {time_str}\n"
                    f"- 相似度: {round(c.get('distance', 0), 4)}\n"
                    f"\n{entity.get('content', '')[:max_body_len]}\n"
                )

        if not parts:
            return "（当前没有任何邮件数据）"

        return "\n".join(parts)