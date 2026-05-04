import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

from config import SmartEmailAgentConfig


def _summarizer_agent_invoke(chain, parsed_data, attachments_str, post):

    if attachments_str:
        attachments_summary = _summarizer_agent_attachments(attachments_str)
        response = chain.invoke({
            "sender": parsed_data["from"],
            "date": parsed_data["date"],
            "subject": parsed_data["subject"],
            "body": parsed_data["body"],
            "attachments": attachments_summary,
            "user_instruction": post
        })
        return response
    else:
        response = chain.invoke({
            "sender": parsed_data["from"],
            "date": parsed_data["date"],
            "subject": parsed_data["subject"],
            "body": parsed_data["body"],
            "attachments": attachments_str,
            "user_instruction": post
        })
        return response

def _summarizer_agent_attachments(attachments_str):
    attachment_list = attachments_str.split(",")
    save_folder = "attachment"
    for attachment in attachment_list:
        _, file_extension = os.path.splitext(attachment)
        file_extension = file_extension.lower()
        if file_extension == ".pdf":
            file_path = os.path.join(save_folder, attachment)
            with open(file_path, "rb") as f:
                file_content = f.read()

        elif file_extension in [".txt", ".csv", ".json", ".md"]:
            file_path = os.path.join(save_folder, attachment)
            with open(file_path, "rb") as f:
                file_content = f.read()

        elif file_extension in [".jpg", ".jpeg", ".png"]:
            file_path = os.path.join(save_folder, attachment)
            with open(file_path, "rb") as f:
                file_content = f.read()

        else:
            print(f"其他类型的文件: {file_extension}")

    return None

class AiAnalysisCoreModule:

    def __init__(self):
        self.model = ChatOpenAI(
            model=SmartEmailAgentConfig.summarizer_agent_module_name,
            temperature=SmartEmailAgentConfig.temperature,
            base_url=SmartEmailAgentConfig.summarizer_agent_module_base_url,
            api_key=SecretStr(SmartEmailAgentConfig.summarizer_agent_module_api_key),
        )
        self.email_access_and_synchronization_module = None
        self.output_parser = StrOutputParser()

    def orchestrator_agent(self):
        """协调Agent决定执行流程"""
        pass

    def summarizer_agent(self, parsed_data: dict) -> str:
        """生成邮件摘要（3-5句）"""

        system_prompt = """你是一个名为 SmartEmailAgent 的高级人工智能邮件助手。你的核心目标是帮助用户高效、准确地处理、总结、分析和起草电子邮件。

            # 核心能力 (CORE CAPABILITIES)
            1. 邮件总结：从冗长或混乱的邮件对话中提取核心信息、关键实体（时间、地点、人物、金额）以及待办事项（Action Items）。
            2. 意图识别：准确对邮件的目的进行分类（例如：会议邀请、任务分配、业务询价、自动回复、垃圾/钓鱼邮件）。
            3. 智能起草：根据用户的简短指令，撰写专业、得体、符合上下文语境的回复。
            4. 噪音过滤：自动忽略邮件签名档、法律免责声明、历史引用的重复文本以及HTML排版残留信息。
            
            # 严格准则与限制 (RULES & CONSTRAINTS)
            - 绝不捏造 (NO HALLUCINATIONS)：绝对不能凭空捏造、猜测或假设邮件中未明确提及的细节（如日期、联系人、链接、金额）。如果缺少关键信息，请明确指出“未提及”。
            - 隐私与安全：将所有邮件内容视为高度机密。除了完成当前任务所必需的信息外，不要主动重复或暴露敏感的个人隐私信息（PII）。
            - 语气与风格：保持客观、专业、礼貌和简洁。不使用幽默、讽刺或过于口语化的表达，除非用户明确要求某种特定的语气。
            - 事实中立：你只负责分析和处理邮件内容，不要对邮件中的商业决策或人际纠纷发表个人意见。
            
            # 处理边缘情况 (EDGE CASES)
            - 如果邮件正文为空或只有无法读取的附件，请直接告知：“该邮件正文为空或内容无法读取。”
            - 如果识别到邮件具有明显的“网络钓鱼”、“诈骗”或“恶意链接”特征，请在回复的最开头用醒目的方式发出安全警告。
            - 对于多语言邮件，除非用户另有指定，否则请使用与用户提问时相同的语言进行总结和回复。
            
            # 输出要求 (OUTPUT FORMAT)
            请始终保持输出结构清晰、排版易读。当需要列举多个待办事项或要点时，请使用项目符号（Bullet points）。如果系统要求你输出特定格式（如 JSON），你必须严格遵守该格式，绝对不要输出任何多余的解释性文字。"""
        user_prompt = """请根据以下提供的邮件详细信息，完成我指定的任务。

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
            
            请确保你的回答严格遵循系统提示词中的所有准则。
            """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
        self.model.temperature = 0.1
        chain = prompt | self.model | self.output_parser
        attachments_list = parsed_data.get("attachments", [])
        if attachments_list:
            names_list = [att.get("save_filename", "未知文件") for att in attachments_list]
            attachments_str = ", ".join(names_list)
        else:
            attachments_str = "无"
        post = "请用50字以内总结这封邮件的核心意图，并提取 Action Items。"
        response = _summarizer_agent_invoke(chain, parsed_data, attachments_str, post)
        return response

    def reply_agent(self, require, parsed_data: dict)  -> str:
        """生成回复草稿（支持指定语气、长度、语言）"""
        system_prompt = """你是一个高情商且专业的邮件起草助手。你的任务是根据用户提供的“原始邮件内容”以及“回复指令”，撰写一封得体、准确的回信。

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
               }
               """

        user_prompt = """请根据以下信息起草回信：

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

               请直接输出 JSON 结果。
               """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
        self.model.temperature = 0.7
        chain = prompt | self.model | self.output_parser
        post = "生成回复草稿（支持指定语气、长度、语言）"
        response = chain.invoke({
            "sender": parsed_data["from"],
            "date": parsed_data["date"],
            "subject": parsed_data["subject"],
            "body": parsed_data["body"],
            "attachments": None,
            "user_instruction": post,
            "tone": require["tone"],
            "language": require["language"],
            "length": require["length"],
        })
        return response

    def classifier_agent(self, parsed_data: dict) -> str:
        """分类邮件并判断紧急程度"""

        system_prompt = """你是一个专业的智能邮件分拣助手。你的核心任务是阅读用户的电子邮件，并评估该邮件的“紧急程度（Priority）”，同时简要说明判断理由。

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
        }
        """

        user_prompt = """请分析以下邮件并输出 JSON 格式的优先级评估：

        <email_metadata>
            发件人: {sender}
            时间: {date}
            主题: {subject}
        </email_metadata>

        <email_body>
            {body}  
        </email_body>
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
        self.model.temperature = 0.1
        chain = prompt | self.model | self.output_parser
        post = "判断优先级，并且用一句话（20字以内）简要解释为什么判定为这个优先级"
        response = _summarizer_agent_invoke(chain, parsed_data, None, post)
        return response

    def task_extractor_agent(self, parsed_data: dict) -> str:
        """提取待办事项、截止日期、负责人、会议信息"""

        system_prompt = """你是一个专业的邮件信息提取助手。你的核心任务是从邮件内容中精准提取出“待办事项（Action Items）”和“会议安排（Meetings）”。

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
                }
                """

        user_prompt = """请分析以下邮件，提取待办事项和会议信息，并输出指定的 JSON 格式：

                <email_metadata>
                    发件人: {sender}
                    时间: {date}
                    主题: {subject}
                </email_metadata>

                <email_body>
                    {body}  
                </email_body>
                """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
        self.model.temperature = 0.1
        chain = prompt | self.model | self.output_parser
        post = "提取待办事项、截止日期、负责人、会议信息"
        response = _summarizer_agent_invoke(chain, parsed_data, None, post)
        return response

    def context_summary_agent(self, context: str) -> str:
        """
        提炼并压缩上下文，生成适合存储为向量记忆的简化版结构化文本
        """
        system_prompt = """你是一个专门负责“记忆压缩与知识图谱构建”的 AI 助手。你的核心任务是将用户提供的冗长、杂乱的沟通记录、项目背景或联系人偏好，提炼成高度浓缩、结构化的“核心上下文（Context）”，以便将其作为 AI 的长期记忆存储。

        # 压缩与提炼规则 (COMPRESSION RULES)
        1. 极简主义：剔除所有问候语、客套话、邮件签名、语气词及冗余的解释。只保留“干货”。
        2. 聚焦实体与事实：重点提取并保留以下关键信息：
           - 人物与角色（Who）：谁负责什么，谁是决策者，谁的邮箱是什么。
           - 项目与代号（What）：项目名称、阶段、预算、核心目标。
           - 时间与节点（When）：关键的 Deadline、里程碑。
           - 偏好与规则（Rules）：特定的沟通习惯、禁忌、格式要求。
        3. 保持客观绝对：将代词（如“他/她”、“我”）替换为具体的实体名称，确保该文本在脱离原语境后依然能被准确理解。
        4. 绝不捏造：只能基于提供的文本进行总结，不得脑补或推测。

        # 输出格式要求 (OUTPUT FORMAT)
        请直接输出纯文本（支持适当的换行和无序列表 -），不要输出多余的解释。格式应尽可能紧凑，例如：
        - 项目[代号/名称]：[状态/截止日期]，负责人：[姓名/邮箱]
        - 偏好设定：[偏好细节]
        - 关键规则：[规则细节]
        """

        user_prompt = """请对以下原始文本进行“记忆压缩”：

        <raw_context>
        {context}
        </raw_context>

        请输出提炼后的简化版本：
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])

        self.model.temperature = 0.1
        chain = prompt | self.model | self.output_parser

        response = chain.invoke({
            "context": context
        })

        return response
