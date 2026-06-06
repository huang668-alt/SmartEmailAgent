"""
RAG 问答 Agent

负责：
- 接收用户自然语言提问
- 基于预检索的邮件/上下文生成有据可查的回答
- 支持多轮对话历史
- 输出结构化回答 + 来源引用
"""

import json
import logging
from typing import Any, Dict, List, Optional

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
        """将对话历史格式化为文本"""
        if not history:
            return "（无历史对话）"

        lines = []
        for i, turn in enumerate(history[-6:], start=1):  # 最近 3 轮
            lines.append(f"Q{i}: {turn.get('question', '')}")
            lines.append(f"A{i}: {turn.get('answer', '')[:200]}")  # 截断旧回答
        return "\n".join(lines)

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
