"""
通用闲聊 Agent

负责处理与邮件系统无关的日常对话，作为 chat_stream 的兜底路由。
"""

import logging
from typing import Any, Dict, Generator

from .base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)

# ── System Prompt ───────────────────────────────────────────

CHAT_SYSTEM_PROMPT = """你是一个智能邮件助手，名叫 SmartEmailAgent。除了邮件处理功能外，你也可以进行日常对话。

# 对话准则
1. 友好、专业、简洁
2. 如果用户问题与邮件系统无关，自然地闲聊回应
3. 如果用户表达了需要邮件相关功能的意图，引导用户明确需求
4. 用中文回复（除非用户主动使用其他语言）"""

CHAT_USER_TEMPLATE = """{user_input}

{history_hint}"""


class ChatAgent(BaseAgent):
    """
    通用闲聊 Agent — chat_stream 的 PURE_CHAT 兜底路由。

    输入:
        - user_input: 用户自然语言输入
        - history: 对话历史列表

    输出:
        - 自然语言回复文本（流式）
    """

    def __init__(self, temperature: float = 0.7, **kwargs):
        super().__init__(
            name="ChatAgent",
            system_prompt=CHAT_SYSTEM_PROMPT,
            temperature=temperature,
            **kwargs,
        )

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """非流式闲聊（兜底用）"""
        user_input = input_data.get("user_input", "")
        history = input_data.get("history", [])

        if not user_input:
            return AgentResult(success=False, error="缺少 user_input 输入")

        history_hint = self._format_history(history)
        chain = self._build_chain(CHAT_USER_TEMPLATE)
        try:
            response = chain.invoke({
                "user_input": user_input,
                "history_hint": history_hint,
            })
            return AgentResult(success=True, data={"answer": response})
        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def stream(self, input_data: Dict[str, Any]) -> Generator[str, None, None]:
        """
        流式闲聊，逐 token 输出回复文本。

        Args:
            input_data: {"user_input": str, "history": list}

        Yields:
            str: 模型生成的文本片段
        """
        user_input = input_data.get("user_input", "")
        history = input_data.get("history", [])

        if not user_input:
            yield "（错误：输入为空）"
            return

        history_hint = self._format_history(history)
        chain = self._build_chain(CHAT_USER_TEMPLATE)
        for chunk in self._stream_chain(chain, {
            "user_input": user_input,
            "history_hint": history_hint,
        }):
            yield chunk

    @staticmethod
    def _format_history(history: list) -> str:
        """将对话历史格式化为提示词片段"""
        if not history:
            return ""
        recent = history[-6:]  # 最近 6 条
        lines = ["【对话历史】"]
        for h in recent:
            if isinstance(h, dict):
                role = h.get("role", "")
                content = h.get("content", h.get("text", str(h)))
                if role == "user":
                    lines.append(f"用户: {content}")
                elif role == "assistant":
                    lines.append(f"助手: {content}")
                else:
                    lines.append(str(content)[:200])
        return "\n".join(lines) if len(lines) > 1 else ""
