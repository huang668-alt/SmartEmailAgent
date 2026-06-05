"""
邮件优先级分类 Agent

负责：
- 读取邮件内容，判定紧急程度（High/Medium/Low/Spam）
- 输出 JSON：{"priority": "...", "reason": "..."}
"""

import json
import logging
from typing import Any, Dict

from .base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)

# ── System Prompt ───────────────────────────────────────────

CLASSIFIER_SYSTEM_PROMPT = """你是一个专业的智能邮件分拣助手。你的核心任务是阅读用户的电子邮件，并评估该邮件的"紧急程度（Priority）"，同时简要说明判断理由。

# 紧急程度分类标准 (PRIORITY LEVELS)
请严格在以下四个级别中选择其一：
1. "High" (高优先级)：
   - 包含明确的紧急截止日期（如：今天内、尽快、ASAP）。
   - 重要的系统故障、安全警告或异常通知。
   - 核心客户或关键业务伙伴的紧急请求。
   - 上级领导指派的紧急任务。
2. "Medium" (中优先级)：
   - 日常工作沟通、任务安排或进度汇报（无马上到期的截止时间）。
   - 几天后才需要处理的会议邀请。
   - 正常的服务咨询或业务问询。
3. "Low" (低优先级)：
   - 行业资讯、内部通讯、Newsletter（订阅邮件）。
   - 抄送（CC）给你仅作知悉，无需你采取行动的邮件。
   - 自动生成的常规报表或系统日志。
4. "Spam" (垃圾/无效邮件)：
   - 明显的推销、广告、网络钓鱼或完全无关的内容。

# 输出要求 (OUTPUT FORMAT)
你必须且只能输出合法的 JSON 格式。不要包含任何 Markdown 标记（如 ```json ），也不要包含任何额外的解释性文字。

期望的 JSON 结构如下：
{
    "priority": "High / Medium / Low / Spam",
    "reason": "用一句话（20字以内）简要解释为什么判定为这个优先级"
}"""

USER_PROMPT_TEMPLATE = """请分析以下邮件并输出 JSON 格式的优先级评估：

<email_metadata>
    发件人: {sender}
    时间: {date}
    主题: {subject}
</email_metadata>

<email_body>
    {body}
</email_body>

请直接输出 JSON，不要包含任何其他文字。"""


class ClassifierAgent(BaseAgent):
    """
    邮件优先级分类 Agent

    输入: email_data 字典 (sender/from, date, subject, body)
    输出: JSON 字符串 {"priority": "High|Medium|Low|Spam", "reason": "..."}
    """

    def __init__(self, temperature: float = 0.1, **kwargs):

        super().__init__(  # 调用父类的构造方法进行初始化
            name="ClassifierAgent",  # 设定代理名称为 "ClassifierAgent"，用于日志标识与调试追踪
            system_prompt=CLASSIFIER_SYSTEM_PROMPT,  # 注入预定义的分类器系统提示词，定义代理的角色与行为准则
            temperature=temperature,  # 透传温度参数，调控模型输出的随机性（值越低越确定）
            **kwargs,  # 将额外的关键字参数原样传递给父类，确保扩展配置项不被遗漏
        )

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:  # 定义执行方法，接收字典输入，返回标准化的代理结果对象
        """分类邮件优先级"""  # 方法文档字符串，说明功能：对邮件进行优先级分类

        # ---------- 字段提取与容错 ----------
        sender = input_data.get("sender") or input_data.get("from", "未知")
        # 优先取 sender 字段，若不存在则尝试 from 字段（兼容不同数据源），都缺失时默认"未知"

        date = input_data.get("date", "未知")
        # 提取邮件日期，缺失时默认"未知"

        subject = input_data.get("subject", "无主题")
        # 提取邮件主题，缺失时默认"无主题"

        body = input_data.get("body", "")
        # 提取邮件正文，缺失时为空字符串（允许空内容）

        # ---------- 构建并调用 LLM 链 ----------
        chain = self._build_chain(USER_PROMPT_TEMPLATE)
        # 根据用户提示模板构建 LangChain 调用链（内部封装了 prompt + LLM 的组合逻辑）

        try:
            response = chain.invoke({  # 同步调用 LLM 链，传入模板所需的变量
                "sender": sender,  # 发件人信息
                "date": date,  # 日期信息
                "subject": subject,  # 邮件主题
                "body": body,  # 邮件正文
            })

            # ---------- 解析与校验 LLM 输出 ----------
            parsed = json.loads(response)  # 将 LLM 返回的字符串解析为 JSON 字典
            priority = parsed.get("priority", "Low")  # 提取优先级字段，默认 Low（保守策略）
            reason = parsed.get("reason", "未提供理由")  # 提取分类理由字段，默认提示信息

            # ---------- 标准化优先级值 ----------
            valid_priorities = {"High", "Medium", "Low", "Spam"}
            # 合法的优先级枚举集合（高/中/低/垃圾邮件）

            if priority not in valid_priorities:  # 若 LLM 返回了非法优先级值
                priority = "Low"  # 强制降级为 Low（安全兜底）
                reason = f"分类异常（原始值: {priority}）"  # 在理由中记录原始异常值，便于排查

            # ---------- 构造成功结果 ----------
            return AgentResult(
                success=True,  # 标记本次执行成功
                data={
                    "priority": priority,  # 标准化后的优先级
                    "reason": reason,  # 分类理由说明
                },
                metadata={"raw_response": response},  # 元数据中保留 LLM 原始响应，方便审计回溯
            )

        # ---------- JSON 解析异常：降级处理 ----------
        except json.JSONDecodeError as e:
            logger.warning(  # 记录警告日志
                f"ClassifierAgent JSON 解析失败: {response[:200]}"  # 仅截取前 200 字符防止日志过长
            )
            return AgentResult(
                success=True,  # 虽然解析失败，但系统仍可用，标记为成功（降级策略）
                data={
                    "priority": "Low",  # 默认低优先级，不阻断流程
                    "reason": "JSON解析失败，默认低优先级",  # 明确告知降级原因
                },
                metadata={
                    "raw_response": response,  # 保留原始响应便于排查问题
                    "parse_error": str(e),  # 记录具体的解析错误信息
                },
            )

        # ---------- 其他未知异常：标记失败 ----------
        except Exception as e:
            return AgentResult(
                success=False,  # 标记执行失败，需上层处理
                error=str(e)  # 传递异常描述供上层日志记录或告警
            )
