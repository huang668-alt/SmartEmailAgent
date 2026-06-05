"""
邮件回复起草 Agent

负责：
- 根据原始邮件 + 用户指令起草回复
- 支持指定语气、长度、语言
- 输出 JSON：{"subject": "Re: ...", "body": "..."}
"""

import json
import logging
from typing import Any, Dict

from .base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)

# ── System Prompt ───────────────────────────────────────────

REPLY_SYSTEM_PROMPT = """你是一个高情商且专业的邮件起草助手。你的任务是根据用户提供的"原始邮件内容"以及"回复指令"，撰写一封得体、准确的回信。

# 起草要求 (DRAFTING REQUIREMENTS)
1. 严格遵循指令：你必须完全按照用户指定的【回复意图】、【语气】、【长度】和【语言】进行撰写。
2. 上下文连贯：准确识别原始邮件中的发件人姓名，并在回信中使用得体的称呼。结合原邮件内容，确保回复逻辑自洽、不生硬。
3. 格式规范：邮件正文必须包含恰当的问候语（如：您好）、核心正文、礼貌的结尾（如：祝好）以及签名占位符（如：[您的名字]）。
4. 绝不捏造：不要在邮件中编造用户未提供的虚假事实、日期、金额或承诺。如果缺少关键信息，可以使用占位符（如：[请确认具体时间]）提醒用户补全。

# 输出要求 (OUTPUT FORMAT)
你必须且只能输出合法的 JSON 格式。不要包含任何 Markdown 标记（如 ```json ），也不要包含任何额外的解释性文字。

期望的 JSON 结构如下：
{
    "subject": "Re: [原始邮件的主题] 或 [你优化的新主题]",
    "body": "这里是完整的邮件正文内容。支持使用换行符 \\n 来保持段落排版。"
}"""

USER_PROMPT_TEMPLATE = """请根据以下信息起草回信：

【原始邮件信息】
发件人: {sender}
时间: {date}
主题: {subject}
正文: {body}

【起草指令】
- 回复意图/内容: {user_instruction}
- 语气 (Tone): {tone}
- 长度 (Length): {length}
- 目标语言 (Language): {language}

请直接输出 JSON 结果。"""


class ReplyAgent(BaseAgent):
    """
    邮件回复起草 Agent

    输入:
        - email_data 字典 (sender/from, date, subject, body)
        - reply_requirements 字典 (instruction, tone, length, language)

    输出: JSON {"subject": "...", "body": "..."}
    """

    def __init__(self, temperature: float = 0.7, **kwargs):
        super().__init__(
            name="ReplyAgent",
            system_prompt=REPLY_SYSTEM_PROMPT,
            temperature=temperature,  # 回复需要一定的创造性
            **kwargs,
        )

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """起草邮件回复"""
        sender = input_data.get("sender") or input_data.get("from", "未知")
        date = input_data.get("date", "未知")
        subject = input_data.get("subject", "无主题")
        body = input_data.get("body", "")

        # 回复要求（带默认值）
        requirements = input_data.get("requirements", input_data)
        instruction = requirements.get("instruction",
            input_data.get("user_instruction", "根据原始邮件内容起草一封得体的回复"))
        tone = requirements.get("tone", "专业礼貌")
        length = requirements.get("length", "中等")
        language = requirements.get("language", "中文")

        chain = self._build_chain(USER_PROMPT_TEMPLATE)
        try:
            response = chain.invoke({
                "sender": sender,
                "date": date,
                "subject": subject,
                "body": body,
                "user_instruction": instruction,
                "tone": tone,
                "length": length,
                "language": language,
            })

            # 尝试解析 JSON
            parsed = json.loads(response)
            return AgentResult(
                success=True,
                data={
                    "subject": parsed.get("subject", f"Re: {subject}"),
                    "body": parsed.get("body", response),
                },
                metadata={
                    "tone": tone,
                    "language": language,
                    "raw_response": response,
                },
            )
        except json.JSONDecodeError:
            # 降级：直接返回原始文本作为 body
            logger.warning("ReplyAgent JSON 解析失败，使用原始文本")
            return AgentResult(
                success=True,
                data={
                    "subject": f"Re: {subject}",
                    "body": response,
                },
                metadata={"tone": tone, "language": language, "raw_response": response},
            )
        except Exception as e:
            return AgentResult(success=False, error=str(e))
