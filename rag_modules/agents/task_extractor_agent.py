"""
任务与会议提取 Agent

负责：
- 从邮件中提取待办事项 (Tasks)：action / assignee / deadline
- 从邮件中提取会议信息 (Meetings)：topic / time / location
- 输出 JSON：{"tasks": [...], "meetings": [...]}
"""

import json
import logging
from typing import Any, Dict

from .base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)

# ── System Prompt ───────────────────────────────────────────

TASK_EXTRACTOR_SYSTEM_PROMPT = """你是一个专业的邮件信息提取助手。你的核心任务是从邮件内容中精准提取出"待办事项（Action Items）"和"会议安排（Meetings）"。

# 提取规则 (EXTRACTION RULES)
1. 待办事项 (Tasks):
   - action: 具体的任务内容。
   - assignee: 任务的负责人（如果邮件中未明确指出，请填 "未明确" 或发件人/收件人名字）。
   - deadline: 截止时间（如果没有提到时间，请填 "无"）。
2. 会议信息 (Meetings):
   - topic: 会议主题或目的。
   - time: 会议时间（尽量提取具体的日期和时刻）。
   - location: 会议地点（如实体会议室、Zoom/Teams 链接，或填 "未明确"）。
3. 绝不捏造 (No Hallucinations): 仅提取邮件中明确提及的信息。如果邮件中既没有待办事项也没有会议安排，请返回空列表 []。

# 输出要求 (OUTPUT FORMAT)
你必须且只能输出合法的 JSON 格式。不要包含任何 Markdown 标记（如 ```json ），也不要包含任何额外的解释性文字。

期望的 JSON 结构如下：
{
    "tasks": [
        {
            "action": "完成项目架构设计文档",
            "assignee": "张三",
            "deadline": "下周二下班前"
        }
    ],
    "meetings": [
        {
            "topic": "Q3 季度进度同步会",
            "time": "2023-11-05 下午 2:00",
            "location": "2号会议室 / 腾讯会议链接"
        }
    ]
}"""

USER_PROMPT_TEMPLATE = """请分析以下邮件，提取待办事项和会议信息，并输出指定的 JSON 格式：

<email_metadata>
    发件人: {sender}
    时间: {date}
    主题: {subject}
</email_metadata>

<email_body>
    {body}
</email_body>

请直接输出 JSON，不要包含任何其他文字。"""


class TaskExtractorAgent(BaseAgent):
    """
    任务与会议提取 Agent

    输入: email_data 字典 (sender/from, date, subject, body)
    输出: JSON {"tasks": [...], "meetings": [...]}
    """

    def __init__(self, temperature: float = 0.1, **kwargs):
        super().__init__(
            name="TaskExtractorAgent",
            system_prompt=TASK_EXTRACTOR_SYSTEM_PROMPT,
            temperature=temperature,
            **kwargs,
        )

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """提取任务和会议信息"""
        sender = input_data.get("sender") or input_data.get("from", "未知")
        date = input_data.get("date", "未知")
        subject = input_data.get("subject", "无主题")
        body = input_data.get("body", "")

        chain = self._build_chain(USER_PROMPT_TEMPLATE)
        try:
            response = chain.invoke({
                "sender": sender,
                "date": date,
                "subject": subject,
                "body": body,
            })

            parsed = json.loads(response)
            tasks = parsed.get("tasks", [])
            meetings = parsed.get("meetings", [])

            return AgentResult(
                success=True,
                data={
                    "tasks": tasks,
                    "meetings": meetings,
                },
                metadata={
                    "task_count": len(tasks),
                    "meeting_count": len(meetings),
                    "raw_response": response,
                },
            )

        except json.JSONDecodeError as e:
            logger.warning(f"TaskExtractorAgent JSON 解析失败: {response[:200]}")
            return AgentResult(
                success=True,
                data={"tasks": [], "meetings": []},
                metadata={"raw_response": response, "parse_error": str(e)},
            )
        except Exception as e:
            return AgentResult(success=False, error=str(e))
