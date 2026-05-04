import logging
import os

from rag_modules import EmailAccessAndSynchronizationModule
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

from config import SmartEmailAgentConfig


def _summarizer_agent_invoke(chain, parsed_data, attachments_str):

    if attachments_str:
        attachments_summary = _summarizer_agent_attachments(attachments_str)
        response = chain.invoke({
            "sender": parsed_data["from"],
            "date": parsed_data["date"],
            "subject": parsed_data["subject"],
            "body": parsed_data["body"],
            "attachments": attachments_summary,
            "user_instruction": "请用50字以内总结这封邮件的核心意图，并提取 Action Items。"
        })
        return response
    else:
        response = chain.invoke({
            "sender": parsed_data["from"],
            "date": parsed_data["date"],
            "subject": parsed_data["subject"],
            "body": parsed_data["body"],
            "attachments": attachments_str,
            "user_instruction": "请用50字以内总结这封邮件的核心意图，并提取 Action Items。"
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
        self.category = None
        self.priority = None
        self.reason = None

    def orchestrator_agent(self):
        """协调Agent决定执行流程"""
        pass



    @staticmethod
    def summarizer_agent():
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

        model = ChatOpenAI(
            model=SmartEmailAgentConfig.summarizer_agent_module_name,
            temperature=SmartEmailAgentConfig.temperature,
            base_url=SmartEmailAgentConfig.summarizer_agent_module_base_url,
            api_key=SecretStr(SmartEmailAgentConfig.summarizer_agent_module_api_key),
        )
        email_access_and_synchronization_module = EmailAccessAndSynchronizationModule()
        try:
            email_access_and_synchronization_module.authenticate()
        except Exception as e:
            logging.error(f"授权失败: {e}")
            return []

        max_emails = 5
        messages = email_access_and_synchronization_module.fetch_new_emails(max_results=max_emails)
        all_summaries = []
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
        output_parser = StrOutputParser()
        chain = prompt | model | output_parser
        for index, msg in enumerate(messages, start=1):
            msg_id = msg['id']
            parsed_data = email_access_and_synchronization_module.parse_email(msg_id)
            summary = {
                'id': msg_id,
                'summary' : None
            }

            attachments_list = parsed_data.get("attachments", [])
            if attachments_list:
                names_list = [att.get("save_filename", "未知文件") for att in attachments_list]
                attachments_str = ", ".join(names_list)
            else:
                attachments_str = "无"
            response = _summarizer_agent_invoke(chain, parsed_data, attachments_str)
            summary['summary'] = response
            all_summaries.append(summary)
        return all_summaries

    def reply_agent(self):
        """生成回复草稿（支持指定语气、长度、语言）"""
        pass

    def task_extractor_agent(self):
        """提取待办事项、截止日期、负责人、会议信息"""
        pass

    def classifier_agent(self):
        """分类邮件"""
        pass