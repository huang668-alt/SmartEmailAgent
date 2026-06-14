"""
RAG 问答 Agent

负责：
- 接收用户自然语言提问
- 基于预检索的邮件/上下文生成有据可查的回答
- 支持多轮对话历史
- 输出结构化回答 + 来源引用
"""
import logging
from typing import Any, Dict, List

from .base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)

# ── System Prompt ───────────────────────────────────────────

QUERY_SYSTEM_PROMPT = """你是一个智能邮件助手，名叫 SmartEmailAgent。你的任务是：根据系统提供的"相关邮件和上下文记忆"，回答用户的问题。

# 核心准则 (CORE PRINCIPLES)
1. **基于证据回答**：你的回答必须建立在提供的上下文之上。每条关键信息都应在上下文中能找到依据。
2. **诚实透明**：如果上下文中没有足够信息来回答某个问题，请明确说"根据现有邮件记录，未找到相关信息"，绝对不要编造。
3. **标注来源**：每条事实陈述后面用方括号标注来源编号，例如 [邮件1]、[邮件3]。
4. **结构清晰**：用标题、列表和分段组织信息，便于用户快速阅读。

# 回答格式 (RESPONSE FORMAT)
请按以下结构组织回答：

## 摘要
用 1-2 句话概括你的发现。

## 详细回答
分段展开说明，每条引用标注来源编号。

## 来源清单
列出你引用的所有来源：
- [邮件1] 发件人 | 日期 | 主题
- [邮件2] 发件人 | 日期 | 主题
...

# 边缘情况处理
- 如果上下文完全为空（没有任何邮件），说："当前没有任何邮件数据。请先同步收件箱后再提问。"
- 如果用户问题与邮件无关，礼貌地引导用户提出与邮件相关的问题。
- 对于多轮对话，结合历史记录理解用户的"它"、"那个"等指代。
- 如果用户问"最近有什么重要的事"，从所有上下文中提取最重要的 3-5 条。

# 禁止事项
- 不要输出 JSON 格式的回答（保持自然语言）
- 不要捏造未在上下文中出现的邮件内容、发件人、日期
- 不要对邮件中的商业决策发表个人意见"""

USER_PROMPT_TEMPLATE = """请根据以下检索到的相关邮件和上下文，回答用户的问题。

<conversation_history>
{history}
</conversation_history>

<retrieved_context>
{context}
</retrieved_context>

<user_question>
{question}
</user_question>

请按系统提示的格式组织你的回答。"""

class QueryAgent(BaseAgent):
    """
    RAG 问答 Agent

    输入:
        - question: 用户自然语言提问
        - context: 预检索的邮件/上下文文本（由 Pipeline 格式化）
        - history: 可选，对话历史列表

    输出:
        - answer: 带来源标注的自然语言回答
        - confidence: 置信度 (high/medium/low)
        - cited_count: 引用的来源数量
    """

    def __init__(self, temperature: float = 0.3, **kwargs):
        super().__init__(
            name="QueryAgent",
            system_prompt=QUERY_SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=4096,  # 回答可能较长，需要更大的输出空间
            **kwargs,
        )

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """
        将对话历史格式化为文本

        支持两种格式：
        1. 对话格式: [{"question": "...", "answer": "..."}]
        2. mem0 历史格式: [{"event": "ADD", "old_memory": "...", "new_memory": "..."}]
        """
        if not history:
            return "（无历史记录）"

        lines = []

        # 判断是哪种格式
        if history and "event" in history[0]:
            # mem0 记忆变更历史
            lines.append("## 记忆变更历史\n")
            for i, event in enumerate(history, start=1):
                event_type = event.get("event", "未知")
                timestamp = event.get("timestamp", "")

                if event_type == "ADD":
                    lines.append(f"{i}. [新增] {event.get('new_memory', '')}")
                elif event_type == "UPDATE":
                    lines.append(f"{i}. [更新] {event.get('old_memory', '')} → {event.get('new_memory', '')}")
                elif event_type == "DELETE":
                    lines.append(f"{i}. [删除] {event.get('old_memory', '')}")

                if timestamp:
                    lines[-1] += f" ({timestamp})"

        elif history and "question" in history[0]:
            # 对话历史
            recent = history[-6:]  # 最近3轮（每轮包含question和answer）
            for i, turn in enumerate(recent, start=1):
                question = turn.get('question', '')
                answer = turn.get('answer', '')

                lines.append(f"Q{i}: {question}")

                # 截断过长的回答
                if answer and len(answer) > 200:
                    answer = answer[:200] + "..."
                lines.append(f"A{i}: {answer or '(无回答)'}")

        else:
            # 未知格式，尝试通用处理
            for i, item in enumerate(history[-6:], start=1):
                if isinstance(item, dict):
                    # 取第一个有意义的字段
                    content = item.get('content') or item.get('text') or str(item)
                    lines.append(f"{i}. {content[:200]}")

        return "\n".join(lines) if lines else "（无历史记录）"

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """基于检索上下文回答用户问题"""
        question = input_data.get("question", "")
        context = input_data.get("context", "")
        history = input_data.get("history", [])

        if not question:
            return AgentResult(success=False, error="缺少 question 输入")

        if not context:
            return AgentResult(
                success=True,
                data={
                    "answer": "当前没有任何邮件数据。请先使用 `process_inbox()` 同步收件箱后再提问。",
                    "confidence": "low",
                    "cited_count": 0,
                },
            )

        chain = self._build_chain(USER_PROMPT_TEMPLATE)
        try:
            response = chain.invoke({
                "history": self._format_history(history),
                "context": context,
                "question": question,
            })

            # 统计引用数量（匹配 [邮件N] 和 [记忆N] 模式）
            import re
            cited = len(set(re.findall(r'\[邮件(\d+)\]|\[记忆(\d+)\]', response)))

            return AgentResult(
                success=True,
                data={
                    "answer": response,
                    "confidence": "high" if cited >= 2 else "medium",
                    "cited_count": cited,
                },
                metadata={
                    "question": question,
                    "context_length": len(context),
                    "cited_sources": cited,
                },
            )
        except Exception as e:
            return AgentResult(success=False, error=str(e))

    from typing import Dict, Any, Generator

    def stream(self, input_data: Dict[str, Any]) -> Generator[str, None, None]:
        """
        流式执行 RAG (检索增强生成) 问答，逐 token 产出大模型生成的文本。

        参数:
            input_data (Dict[str, Any]): 输入的数据字典，包含问题、上下文和历史对话。
                - "question" (str): 用户当前提问。
                - "context" (str): 检索出来的相关邮件上下文。
                - "history" (List[Dict]): 历史对话轮次。

        Yields:
            str: 大模型生成的文本片段 (Chunk/Token)
        """
        # 1. 解析输入数据，设置默认值防止 Key 缺失报错
        question = input_data.get("question", "")
        context = input_data.get("context", "")
        history = input_data.get("history", [])

        # 2. 前置边界校验：检查核心参数是否为空
        if not question:
            yield "（错误：缺少问题输入）"
            return

        # 3. 业务逻辑校验：如果上下文为空（未检索到或未同步邮件），进行友好提示
        if not context:
            yield "当前没有任何邮件数据。请先同步收件箱后再提问。"
            return

        # 4. 初始化 LangChain 或者是自定义的执行链 (Chain)
        # USER_PROMPT_TEMPLATE 是预定义的提示词模板
        chain = self._build_chain(USER_PROMPT_TEMPLATE)

        # 5. 格式化变量并流式调用大模型
        # _format_history: 将历史对话格式化为模型可接受的结构 (如字符串或消息对象)
        # _stream_chain: 负责调用模型并返回一个可迭代的生成器 (Generator)
        for chunk in self._stream_chain(chain, {
            "history": self._format_history(history),
            "context": context,
            "question": question,
        }):
            # 6. 逐个 token/片段实时向前端(或调用方)推送数据
            yield chunk