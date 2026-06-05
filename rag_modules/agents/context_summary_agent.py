"""
上下文记忆压缩 Agent

负责：
- 将冗长的沟通记录 / 项目背景 / 联系人偏好压缩为结构化核心上下文
- 输出适合存储为向量记忆的紧凑文本
"""

from typing import Any, Dict

from .base import BaseAgent, AgentResult

# ── System Prompt ───────────────────────────────────────────

CONTEXT_SUMMARY_SYSTEM_PROMPT = """你是一个专门负责"记忆压缩与知识图谱构建"的 AI 助手。你的核心任务是将用户提供的冗长、杂乱的沟通记录、项目背景或联系人偏好，提炼成高度浓缩、结构化的"核心上下文（Context）"，以便将其作为 AI 的长期记忆存储。

# 压缩与提炼规则 (COMPRESSION RULES)
1. 极简主义：剔除所有问候语、客套话、邮件签名、语气词及冗余的解释。只保留"干货"。
2. 聚焦实体与事实：重点提取并保留以下关键信息：
   - 人物与角色（Who）：谁负责什么，谁是决策者，谁的邮箱是什么。
   - 项目与代号（What）：项目名称、阶段、预算、核心目标。
   - 时间与节点（When）：关键的 Deadline、里程碑。
   - 偏好与规则（Rules）：特定的沟通习惯、禁忌、格式要求。
3. 保持客观绝对：将代词（如"他/她"、"我"）替换为具体的实体名称，确保该文本在脱离原语境后依然能被准确理解。
4. 绝不捏造：只能基于提供的文本进行总结，不得脑补或推测。

# 输出格式要求 (OUTPUT FORMAT)
请直接输出纯文本（支持适当的换行和无序列表 -），不要输出多余的解释。格式应尽可能紧凑，例如：
- 项目[代号/名称]：[状态/截止日期]，负责人：[姓名/邮箱]
- 偏好设定：[偏好细节]
- 关键规则：[规则细节]"""

USER_PROMPT_TEMPLATE = """请对以下原始文本进行"记忆压缩"：

<raw_context>
{context}
</raw_context>

请输出提炼后的简化版本："""

class ContextSummaryAgent(BaseAgent):
    """
    上下文记忆压缩 Agent

    输入: {"context": "原始文本内容"}
    输出: 结构化压缩文本（适合存入向量数据库作为长期记忆）
    """

    def __init__(self, temperature: float = 0.1, **kwargs):
        super().__init__(
            name="ContextSummaryAgent",
            system_prompt=CONTEXT_SUMMARY_SYSTEM_PROMPT,
            temperature=temperature,
            **kwargs,
        )

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """压缩上下文为结构化长期记忆"""
        context = input_data.get("context", "")

        if not context:
            return AgentResult(success=False, error="输入上下文为空")

        chain = self._build_chain(USER_PROMPT_TEMPLATE)
        try:
            response = chain.invoke({"context": context})
            return AgentResult(
                success=True,
                data=response,
                metadata={
                    "input_length": len(context),
                    "output_length": len(response)
                },
            )

        except Exception as e:
            return AgentResult(success=False, error=str(e))
