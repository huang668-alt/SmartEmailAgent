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
from typing import Generator
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import asyncio

import config
from rag_modules.email import GmailAuth, EmailFetcher, EmailParser
from rag_modules.agents import (
    SummarizerAgent, ClassifierAgent, ContextSummaryAgent, OrchestratorAgent,
    QueryAgent,
)
from rag_modules.agents.base import AgentResult
from rag_modules.storage import embed, VectorStore, get_mem0_store

MAX_QUESTION_LENGTH = 1000  # 限制问题最大 1000 字符（防止超长文本撑爆 Embedding 或恶意注入）
MAX_TOP_K = 50  # 限制最大检索 K 值（防止恶意传百万级数据导致向量数据库内存溢出）
VALID_SCOPES = {"all", "inbox", "sent", "context"}  # 严格的 Scope 白名单
MAX_HISTORY_COUNT = 20  # 限制最多允许带入的上下文轮数（防止撑爆 LLM Context Window）

logger = logging.getLogger(__name__)


class Pipeline:
    """
    邮件处理流水线

    依赖链：GmailAuth → EmailFetcher/EmailParser → Agent → Embedding → VectorStore
    """


    def __init__(self):
        self.store = VectorStore()                          # Milvus 向量库操作封装
        self.summarizer = SummarizerAgent()                 # 邮件摘要 Agent
        self.classifier = ClassifierAgent()                 # 优先级分类 Agent
        self.context_compressor = ContextSummaryAgent()     # 长期记忆压缩 Agent
        self.orchestrator: Optional[OrchestratorAgent] = None   # 多 Agent 编排器（懒加载）
        self.querier: Optional[QueryAgent] = None               # RAG 问答 Agent（懒加载）
        self._reply_agent = None                                 # ReplyAgent（懒加载）
        self._chat_agent = None                                  # ChatAgent（懒加载）
        self._mem0 = None                                       # mem0ai Memory（懒加载）
        self.executor = ThreadPoolExecutor(max_workers=4)

    @property
    def mem0(self):
        """懒加载 mem0ai Memory 实例"""
        if self._mem0 is None:
            self._mem0 = get_mem0_store()
        return self._mem0

    def _get_orchestrator(self) -> OrchestratorAgent:
        """懒加载 OrchestratorAgent，首次调用时才实例化，避免启动开销"""
        if self.orchestrator is None:
            self.orchestrator = OrchestratorAgent()
        return self.orchestrator

    def _connect_gmail(self) -> tuple[EmailFetcher, EmailParser]:
        """OAuth 认证 → 返回 (邮件拉取器, 邮件解析器) 二元组"""
        service = GmailAuth().authenticate()
        return EmailFetcher(service), EmailParser(service)

    # ── 收件箱 ──────────────────────────────────────────────

    def process_inbox(self, max_emails: int = 5):
        """
        串行处理收件箱：拉取 → 摘要 → 嵌入 → 分类 → 存入 Milvus

        每封邮件独立调用 SummarizerAgent + ClassifierAgent，
        适合调试；生产环境推荐使用 process_inbox_v2（并行版）。
        """
        fetcher, parser = self._connect_gmail()
        messages = fetcher.fetch_inbox(max_results=max_emails)
        logger.info(f"拉取到 {len(messages)} 封收件箱邮件")

        for i, msg in enumerate(messages, start=1):
            parsed = parser.parse(msg["id"])
            if parsed is None:
                continue  # 解析失败（如纯附件邮件），跳过

            email_data = {
                "id": msg["id"], "threadId": msg.get("threadId"),
                "snippet": msg.get("snippet"),
                "subject": parsed["subject"], "from": parsed["from"],
                "date": parsed["date"], "body": parsed["body"],
                "attachments": parsed["attachments"],
            }

            # 摘要成功则嵌入摘要，失败则降级为嵌入原文
            summary = self.summarizer.run(email_data)
            email_data["embedding"] = (
                embed(summary.data) if summary.success else embed(email_data["body"])
            )

            # 分类失败时降级为 Low 优先级
            classify = self.classifier.run(email_data)
            if classify.success:
                email_data["priority"] = classify.data.get("priority", "Low")
                email_data["reason"] = classify.data.get("reason", "")
            else:
                email_data["priority"] = "Low"
                email_data["reason"] = ""

            self.store.insert_inbox_email(email_data)

    # ── Orchestrator 版本（并行 + 自动分支）─────────────────

    def process_inbox_v2(self, max_emails: int = 5):
        """
        【推荐】OrchestratorAgent 并行编排版本

        相较 process_inbox：
        - 多封邮件的 Summarize + Classify 并行执行，减少总耗时
        - 编排逻辑内聚于 OrchestratorAgent，Pipeline 只负责前后数据装配
        """
        fetcher, parser = self._connect_gmail()
        messages = fetcher.fetch_inbox(max_results=max_emails)

        # 解析所有邮件，构造统一结构
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

        # 编排器并行执行所有邮件的 Agent 任务
        result = self._get_orchestrator().process_inbox(email_list, parallel=True)
        if not result.success:
            logger.error(f"编排失败: {result.error}")
            return

        # 将编排结果（摘要 + 分类）逐封写入 Milvus
        for i, email in enumerate(email_list):
            results = result.data.get("results", [])
            agent_results = results[i] if i < len(results) else {}

            # ClassifierAgent 可能返回 JSON 字符串，需反序列化
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
        """
        分析发件箱邮件语气，摘要后嵌入向量库

        用于构建"用户写作风格"记忆，供回复起草时参考。
        """
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

            # 摘要失败则降级嵌入原文
            summary = self.summarizer.run(email_data)
            email_data["embedding"] = (
                embed(summary.data) if summary.success else embed(email_data["body"])
            )
            self.store.insert_sent_email(email_data)
            logger.info(f"发件箱 {email_data['id']} 处理完成 [{i}/{len(messages)}]")

    # ── 联系人 ──────────────────────────────────────────────

    def historical_chat(self, n: int | None = None) -> dict:
        """
        统计最近 n 个高频联系人（发件人 + 收件人分开统计）

        Returns:
            {
                "top_senders":   [(email_addr, count), ...],  # 收件箱高频发件人
                "top_receivers": [(email_addr, count), ...],  # 发件箱高频收件人
            }
        """
        if n is None:
            n = config.SmartEmailAgentConfig.number_of_common_contacts

        fetcher, parser = self._connect_gmail()

        def _extract(parsed, field):
            """从 'Name <email>' 格式中提取纯邮箱地址，统一转小写"""
            val = parsed.get(field, "")
            if not val:
                return ""
            m = re.search(r"<([^>]+)>", val)
            return m.group(1).lower() if m else val.strip().lower()

        # 统计收件箱发件人频次
        senders = Counter()
        for msg in fetcher.fetch_inbox(max_results=10):
            p = parser.parse(msg["id"])
            if p:
                addr = _extract(p, "from")
                if addr:
                    senders[addr] += 1

        # 统计发件箱收件人频次
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
        """
        存储长期记忆 — 优先使用 mem0ai，降级到 Milvus

        mem0ai 自动提取关键信息、去重、建立语义索引；
        如果 mem0 不可用，降级为 ContextSummaryAgent 压缩 + Milvus 存储。
        """
        # 优先使用 mem0ai（自动提取关键记忆）
        try:
            self.mem0.add(content=context, infer=True, memory_type="context")
            logger.info(f"mem0 长期记忆已存储 ({len(context)} 字符)")
            return
        except Exception as e:
            logger.error(f"mem0 存储记忆失败: {e}，降级到旧版存储")

        # 降级：ContextSummaryAgent 压缩 + Milvus
        result = self.context_compressor.run({"context": context})
        compressed = result.data if result.success else context

        self.store.insert_context(
            content=compressed,
            embedding=embed(compressed),
            timestamp=time.time(),
        )
        logger.info(f"Milvus 长期记忆已存储 (降级, {len(context)} → {len(compressed)} 字符)")

    # ── RAG 问答 ────────────────────────────────────────────

    def _get_querier(self) -> QueryAgent:
        """懒加载 QueryAgent，首次调用时才实例化"""
        if self.querier is None:
            self.querier = QueryAgent()
        return self.querier

    def _get_router_agent(self):
        """返回路由 Agent（复用 OrchestratorAgent 的 predict 方法做意图分类）"""
        return self._get_orchestrator()

    def _get_reply_agent(self):
        """懒加载 ReplyAgent"""
        if self._reply_agent is None:
            from rag_modules.agents.reply_agent import ReplyAgent
            self._reply_agent = ReplyAgent()
        return self._reply_agent

    def _get_chat_agent(self):
        """懒加载 ChatAgent — 通用闲聊兜底"""
        if self._chat_agent is None:
            from rag_modules.agents.chat_agent import ChatAgent
            self._chat_agent = ChatAgent()
        return self._chat_agent

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

        # Step 1: 将问题向量化，用于后续相似度搜索
        q_vector = embed(question)
        if not q_vector:
            return AgentResult(
                success=False,
                error="文本向量化失败，请检查 Embedding API 配置",
            )

        # Step 2: 按 scope 决定搜索哪些集合
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

        # mem0ai 语义搜索记忆
        mem0_results = []
        if scope in ("all", "context"):
            try:
                mem0_results = self.mem0.search(query=question, top_k=top_k)
            except Exception as e:
                logger.warning(f"mem0 搜索失败: {e}")

        # Step 3: 将搜索结果格式化为 LLM 可消费的 markdown 上下文
        context_str = self._format_rag_context(emails, contexts, mem0_results)

        # Step 4: 将问题 + 上下文 + 历史传给 LLM 生成最终回答
        result = self._get_querier().run({
            "question": question,
            "context": context_str,
            "history": history or [],
        })

        # Step 5: 将来源元数据挂载到 result，供前端渲染引用卡片
        result.metadata["sources"] = {
            "emails": [
                {
                    "subject":      e.get("entity", {}).get("subject", ""),
                    "from_address": e.get("entity", {}).get("from_address", ""),
                    "date":         e.get("entity", {}).get("date", ""),
                    "priority":     e.get("entity", {}).get("priority", ""),
                    "body":         (e.get("entity", {}).get("body", "") or "")[:200],  # 截断防溢出
                    "distance":     round(e.get("distance", 0), 4),
                }
                for e in emails
            ],
            "contexts": [
                {
                    "content":   (c.get("entity", {}).get("content", "") or "")[:200],
                    "timestamp": c.get("entity", {}).get("timestamp", 0),
                    "distance":  round(c.get("distance", 0), 4),
                }
                for c in contexts
            ],
        }

        logger.info(
            f"RAG 查询完成 — 回答长度: {len(result.data.get('answer', '')) if result.data else 0}, "
            f"邮件来源: {len(emails)}, 记忆来源: {len(contexts)}"
        )
        return result

    import logging
    from typing import Generator

    logger = logging.getLogger(__name__)

    def chat_stream(
            self,
            user_input: str,
            top_k: int = 5,
            history: list | None = None
    ) -> Generator[str, None, None]:
        """
        Agent 统一对话流式方法 — 意图识别 + 动态路由流式输出

        Args:
            user_input (str): 用户输入的自然语言（指令、问题或待记忆文本）
            top_k (int): 向量检索的召回条数
            history (list): 聊天上下文历史记录
        """
        # ── Step 1: 基础入参校验 ───────────────────────────────────────
        try:
            if not user_input or not user_input.strip():
                raise ValueError("输入内容 (user_input) 不能为空或纯空格。")
            if len(user_input) > MAX_QUESTION_LENGTH:
                raise ValueError(f"输入内容过长（当前 {len(user_input)} 字）。")

            # 柔性降级 top_k
            if not isinstance(top_k, int) or top_k < 1:
                top_k = 5
            elif top_k > MAX_TOP_K:
                top_k = MAX_TOP_K

            history = history or []
            if not isinstance(history, list):
                raise ValueError("历史记录 (history) 必须是列表类型。")

        except ValueError as val_err:
            logger.error(f"Chat 输入校验未通过: {val_err}")
            yield f"（安全拒绝：{str(val_err)}）"
            return

        logger.info(f"Agent 统一流式接收请求: {user_input[:80]}...")

        # ── Step 2: 意图识别与大模型路由规划 (Intent Routing) ─────────────
        # 💡 核心设计：这里采用快速同步/非流式调用，先让大模型给出一个明确的意图标签
        try:
            # 假设你有一个 _router_agent，它通过 Prompt 让模型返回 "QUERY", "STORE", "REPLY", "PURE_CHAT" 之一
            intent: str = self._get_router_agent().predict(
                user_input=user_input,
                history=history
            )
            intent = intent.strip().upper()
        except Exception as e:
            logger.warning(f"Agent 路由规划失败，自动降级至 PURE_CHAT 模式: {e}")
            intent = "PURE_CHAT"

        logger.info(f"Agent 路由决策结果: [{intent}]")

        # ── Step 3: 根据路由结果，分流至对应的流式生成器 ───────────────────

        # 路线 A：查询类意图 -> 直接复用你写好的 query_stream 逻辑
        if intent == "QUERY":
            # 完美对接到你的 query_stream 方法中
            yield from self.query_stream(
                question=user_input,
                top_k=top_k,
                scope="all",
                history=history
            )
            return

        # 路线 C：处理类/起草回信类意图
        elif intent == "REPLY":
            # 这里结合发件箱语气等上下文进行流式输出
            try:
                # 捞取一点用户的已发送邮件做风格参考 (Tone Study)
                tone_context = ""
                try:
                    q_vec = embed(user_input)
                    if q_vec and (sent_emails := self.store.search_sent(q_vec, top_k=3)):
                        tone_context = "\n".join(
                            f"主题: {e.subject}\n内容: {e.body}" for e in sent_emails
                        )
                except Exception as embed_err:
                    logger.warning(f"发件箱语气检索失败，跳过: {embed_err}")

                yield from self._get_reply_agent().stream({
                    "instruction": user_input,
                    "tone_reference": tone_context,
                    "history": history
                })
            except Exception as e:
                logger.error(f"起草回信失败: {e}")
                yield f"（写信 Agent 报错: {e}）"
            return

        # 路线 D：常规闲聊或未知意图兜底
        else:
            # 直接调用普通聊天大模型的流式接口
            yield from self._get_chat_agent().stream({
                "user_input": user_input,
                "history": history
            })

            # 日常闲聊事后异步沉淀记忆（fire-and-forget 线程，不阻塞流式输出）
        try:
            self.executor.submit(
                self.mem0.add, f"用户说：{user_input}", user_id="default"
            )
        except Exception:
            pass  # 记忆沉淀失败不影响主流程






    def query_stream(
            self,
            question: str,
            top_k: int = 5,
            scope: str = "all",
            history: list | None = None
    ) -> Generator[str, None, None]:
        """
        RAG 流式问答 — 语义搜索 + 逐 token 产出 LLM 回答
        """

        try:
            # 1. 校验 question (空值与超长文本)
            if not question or not question.strip():
                raise ValueError("查询问题 (question) 不能为空或纯空格。")
            if len(question) > MAX_QUESTION_LENGTH:
                raise ValueError(f"查询问题过长（当前 {len(question)} 字，最大限制 {MAX_QUESTION_LENGTH} 字）。")

            # 2. 校验 top_k (类型、下界、上界)
            if not isinstance(top_k, int) or top_k < 1:
                raise ValueError(f"top_k 必须为正整数，当前输入为: {top_k}")
            if top_k > MAX_TOP_K:
                logger.warning(f"用户请求的 top_k={top_k} 超过系统上限，已强制平滑降级为 {MAX_TOP_K}")
                top_k = MAX_TOP_K  # 柔性策略：不报错，直接限流截断

            # 3. 校验 scope (严格白名单)
            if scope not in VALID_SCOPES:
                raise ValueError(f"非法的检索范围 (scope='{scope}')。允许的值为: {VALID_SCOPES}")

            # 4. 校验并修剪 history (防止上下文窗口爆炸)
            history = history or []
            if not isinstance(history, list):
                raise ValueError("历史记录 (history) 必须是列表 (list) 类型。")
            if len(history) > MAX_HISTORY_COUNT:
                logger.warning(
                    f"历史记录轮数 ({len(history)}) 超过最大限制，系统已自动截取最近的 {MAX_HISTORY_COUNT} 条。")
                history = history[-MAX_HISTORY_COUNT:]  # 柔性策略：截取最新的对话，保障系统可用

        except ValueError as val_err:
            logger.error(f"输入校验未通过: {val_err}")
            yield f"（安全拒绝：{str(val_err)}）"
            return

        logger.info(f"RAG 流式查询: {question[:80]}...")

        # Step 1: 向量化问题
        q_vector = embed(question)
        if not q_vector:
            yield "（错误：文本向量化失败）"
            return

        history = history or []
        emails = []
        mem0_results = []

        # Step 2: 多源检索向量库与记忆库
        # 2.1 检索收件箱
        if scope in ("all", "inbox"):
            try:
                if inbox_emails := self.store.search_inbox(q_vector, top_k):
                    emails.extend(inbox_emails)
            except Exception as e:
                logger.warning(f"收件箱搜索失败: {e}")

        # 2.2 检索发送箱
        if scope in ("all", "sent"):
            try:
                if sent_emails := self.store.search_sent(q_vector, top_k):
                    emails.extend(sent_emails)
            except Exception as e:
                logger.warning(f"发送箱搜索失败: {e}")

        # 2.3 检索知识库/长期记忆 (统一由 scope="context" 或 "all" 控制，避免重复请求)
        if scope in ("all", "context"):
            try:
                # 统一只查一次 mem0
                mem0_results = self.mem0.search(query=question, top_k=top_k)
            except Exception as e:
                logger.warning(f"mem0 检索失败，降级到 Milvus: {e}")
                try:
                    mem0_results = self.store.search_context(q_vector, top_k)
                except Exception as e2:
                    logger.warning(f"Milvus 长期记忆搜索也失败: {e2}")

        contexts = []

        # Step 3: 格式化上下文
        # 移除原先重复的 contexts 参数，保持入参清晰
        context_str = self._format_rag_context(emails, contexts, mem0_results)
        if not context_str.strip():
            context_str = "没有找到与当前问题相关的邮件或背景记忆。"

        # Step 4: 流式调用 LLM 并实时透传 Token
        yield from self._get_querier().stream({
            "question": question,
            "context": context_str,
            "history": mem0_results
        })

        # Step 5: 真正的异步/事后沉淀记忆
        # 如果处于 Async 环境，建议使用 asyncio.create_task；
        # 如果是同步多线程环境，建议提交给线程池 (ThreadPoolExecutor)
        try:
            loop = asyncio.get_running_loop()
            # 避免阻塞当前正在结束的生成器响应
            asyncio.create_task(asyncio.to_thread(self.mem0.add, f"用户问：{question}", user_id="default"))
        except RuntimeError:
            self.mem0.add(f"用户问：{question}", user_id="default")

    # ── 统一入口：LLM 自动路由 ─────────────────────────────

    def chat(self, user_input: str, top_k: int = 5, history: list = None):
        """
        唯一用户入口 — LLM 自动判断走哪条路径。

        路由规则（由 OrchestratorAgent.plan 决定）：
        - 仅 QueryAgent          → RAG 向量搜索问答
        - 仅 ContextSummaryAgent → 压缩存入长期记忆
        - 其他（含 Classifier 等）→ 拉取邮件 + 多 Agent 并行处理

        原理：OrchestratorAgent.plan() 生成执行计划（step 列表），
              Pipeline 根据涉及的 Agent 类型自动分发。
        """
        logger.info(f"用户输入: {user_input[:80]}...")

        # 调用编排器生成执行计划，plan 是 step 列表，每个 step 含 "agent" 字段
        plan = self._get_orchestrator().plan(user_input)

        # 提取本次计划中所有 Agent 名称（集合去重，用于路由判断）
        agent_names = {s["agent"] for s in plan}

        route = "query" if agent_names == {"QueryAgent"} else "process"
        logger.info(f"路由: {route} | Agent: {agent_names} | 计划: {len(plan)} 步")

        # 路径 A：纯查询 → 走 RAG 链路，无需拉取新邮件
        if agent_names == {"QueryAgent"}:
            result = self.query(question=user_input, top_k=top_k, history=history)

            # 对话后自动提取记忆到 mem0
            if result.success:
                try:
                    answer = result.data.get("answer", "") if result.data else ""
                    self.mem0.add_from_conversation(
                        user_message=user_input,
                        assistant_message=answer,
                    )
                except Exception as e:
                    logger.warning(f"自动记忆提取失败 (非致命): {e}")

            return result

        # 路径 B：记忆存储 → 压缩后直接写入向量库
        if agent_names == {"ContextSummaryAgent"}:
            # 优先使用 mem0ai 存储
            try:
                self.mem0.add(content=user_input, infer=True, memory_type="context")
                logger.info(f"mem0 长期记忆已存储 ({len(user_input)} 字符)")
                return AgentResult(success=True, data={"message": "记忆已存入 mem0"})
            except Exception as e:
                logger.error(f"mem0 存储失败，降级: {e}")

            result = self.context_compressor.run({"context": user_input})
            if result.success:
                self.store.insert_context(
                    content=result.data,
                    embedding=embed(result.data),
                    timestamp=time.time(),
                )
                logger.info(f"Milvus 长期记忆已存储 (降级, {len(result.data)} 字符)")
            return result

        # 路径 C：处理/回复类 → 拉取最新邮件 → Orchestrator 并行编排
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

        # 将用户目标、邮件列表、执行计划一并传给编排器
        result = self._get_orchestrator().run({
            "goal": user_input,
            "emails": emails,
            "plan": plan,
            "parallel": True,
        })

        # 编排成功后，将每封邮件的处理结果持久化到 Milvus
        if result.success and result.data.get("results"):
            for i, email in enumerate(emails):
                agent_results = result.data["results"][i] if i < len(result.data["results"]) else {}

                # ClassifierAgent 有时返回 JSON 字符串，需反序列化
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
        mem0_results: list[dict] | None = None,
        max_body_len: int = 300,
    ) -> str:
        """
        将 Milvus 搜索结果格式化为 LLM 可消费的 markdown 文本。

        邮件正文截断至 max_body_len 字符，防止超出 LLM context 窗口。

        Args:
            emails:       search_inbox 返回的邮件列表
            contexts:     search_context 返回的记忆列表
            max_body_len: 邮件/记忆正文最大保留字符数

        Returns:
            markdown 格式的上下文字符串，无数据时返回占位提示
        """
        parts = []

        if emails:
            parts.append("## 相关邮件\n")
            for i, e in enumerate(emails, start=1):
                entity = e.get("entity", {})
                body = (entity.get("body", "") or "")[:max_body_len]
                if len(entity.get("body", "") or "") > max_body_len:
                    body += "..."  # 截断标记，让 LLM 知晓内容不完整

                parts.append(
                    f"### [邮件{i}] {entity.get('subject', '无主题')}\n"
                    f"- 发件人: {entity.get('from_address', '未知')}\n"
                    f"- 日期: {entity.get('date', '未知')}\n"
                    f"- 优先级: {entity.get('priority', '未分类')}\n"
                    f"- 相似度: {round(e.get('distance', 0), 4)}\n"
                    f"\n{body}\n"
                )

        if not parts:
            return "（当前没有任何邮件数据或相关记忆）"
        return "\n".join(parts)


# ── 模块级懒加载（供 API 层复用） ──────────────────────────

_pipeline = None

def get_pipeline() -> Pipeline:
    """
    延迟初始化全局 Pipeline 单例。

    避免 import 时立即连接 Milvus（在测试或冷启动场景下会报错）；
    API 层通过此函数获取实例，保证全程只初始化一次。
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline