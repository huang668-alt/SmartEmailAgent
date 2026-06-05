"""
邮件摘要 Agent

负责：
- 将邮件内容浓缩为 3-5 句核心摘要
- 提取关键实体（时间、地点、人物、金额）
- 提取 Action Items（待办事项）
"""

from typing import Any, Dict

from .base import BaseAgent, AgentResult

# ── System Prompt ───────────────────────────────────────────

SUMMARIZER_SYSTEM_PROMPT = """你是一个名为 SmartEmailAgent 的高级人工智能邮件助手。你的核心目标是帮助用户高效、准确地处理、总结、分析和起草电子邮件。

# 核心能力 (CORE CAPABILITIES)
1. 邮件总结：从冗长或混乱的邮件对话中提取核心信息、关键实体（时间、地点、人物、金额）以及待办事项（Action Items）。
2. 意图识别：准确对邮件的目的进行分类（例如：会议邀请、任务分配、业务询价、自动回复、垃圾/钓鱼邮件）。
3. 智能起草：根据用户的简短指令，撰写专业、得体、符合上下文语境的回复。
4. 噪音过滤：自动忽略邮件签名档、法律免责声明、历史引用的重复文本以及HTML排版残留信息。

# 严格准则与限制 (RULES & CONSTRAINTS)
- 绝不捏造 (NO HALLUCINATIONS)：绝对不能凭空捏造、猜测或假设邮件中未明确提及的细节（如日期、联系人、链接、金额）。如果缺少关键信息，请明确指出"未提及"。
- 隐私与安全：将所有邮件内容视为高度机密。除了完成当前任务所必需的信息外，不要主动重复或暴露敏感的个人隐私信息（PII）。
- 语气与风格：保持客观、专业、礼貌和简洁。不使用幽默、讽刺或过于口语化的表达，除非用户明确要求某种特定的语气。
- 事实中立：你只负责分析和处理邮件内容，不要对邮件中的商业决策或人际纠纷发表个人意见。

# 处理边缘情况 (EDGE CASES)
- 如果邮件正文为空或只有无法读取的附件，请直接告知："该邮件正文为空或内容无法读取。"
- 如果识别到邮件具有明显的"网络钓鱼"、"诈骗"或"恶意链接"特征，请在回复的最开头用醒目的方式发出安全警告。
- 对于多语言邮件，除非用户另有指定，否则请使用与用户提问时相同的语言进行总结和回复。

# 输出要求 (OUTPUT FORMAT)
请始终保持输出结构清晰、排版易读。当需要列举多个待办事项或要点时，请使用项目符号（Bullet points）。如果系统要求你输出特定格式（如 JSON），你必须严格遵守该格式，绝对不要输出任何多余的解释性文字。"""

USER_PROMPT_TEMPLATE = """请根据以下提供的邮件详细信息，完成我指定的任务。

<email_metadata>
    发件人: {sender}
    时间: {date}
    主题: {subject}
</email_metadata>

<email_body>
    {body}
</email_body>

<attachments>
    附件: {attachments}
</attachments>

<task_instruction>
    {user_instruction}
</task_instruction>

请确保你的回答严格遵循系统提示词中的所有准则。"""


class SummarizerAgent(BaseAgent):
    """
    邮件摘要 Agent

    输入: email_data 字典 (sender, date, subject, body, attachments)
    输出: 3-5 句邮件摘要 + Action Items
    """

    def __init__(self, temperature: float = 0.1, **kwargs):
        super().__init__(
            name="SummarizerAgent",
            system_prompt=SUMMARIZER_SYSTEM_PROMPT,
            temperature=temperature,
            **kwargs,
        )

    def _build_attachments_str(self, attachments: list) -> str:
        """构建附件字符串"""
        if not attachments:
            return "无"
        if isinstance(attachments, list):
            names = [
                att.get("save_filename", att.get("filename", "未知文件"))
                if isinstance(att, dict)
                else str(att)
                for att in attachments
            ]
            return ", ".join(names)
        return str(attachments)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """生成邮件摘要"""
        sender = input_data.get("sender") or input_data.get("from", "未知")
        date = input_data.get("date", "未知")
        subject = input_data.get("subject", "无主题")
        body = input_data.get("body", "")
        attachments = input_data.get("attachments", [])
        user_instruction = input_data.get("user_instruction",
            "请用50字以内总结这封邮件的核心意图，并提取 Action Items。")

        chain = self._build_chain(USER_PROMPT_TEMPLATE)
        try:
            response = chain.invoke({
                "sender": sender,
                "date": date,
                "subject": subject,
                "body": body,
                "attachments": self._build_attachments_str(attachments),
                "user_instruction": user_instruction,
            })
            return AgentResult(
                success=True,
                data=response,
                metadata=
                {
                    "email_subject": subject,
                    "sender": sender
                },
            )
        except Exception as e:
            return AgentResult(success=False, error=str(e))