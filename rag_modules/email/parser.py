"""Gmail 邮件解析模块（MIME + 附件）"""

import base64
import logging
import os

from googleapiclient.discovery import Resource

logger = logging.getLogger(__name__)

ATTACHMENT_SAVE_DIR = "attachments"


class EmailParser:
    """解析单封邮件的 MIME 结构：headers、正文、附件"""

    def __init__(self, service: Resource):
        self.service = service

    def parse(self, msg_id: str) -> dict | None:
        """
        完整解析一封邮件，返回结构化字典：
        {
            id, threadId, snippet,
            subject, from, date, body,
            attachments: [{filename, save_filename, filepath, text}]
        }
        """
        try:
            msg = (
                self.service.users()
                .messages()
                .get(userId="me", id=msg_id, format="full")
                .execute()
            )
        except Exception as e:
            logger.error(f"获取邮件 {msg_id} 失败: {e}")
            return None

        headers = msg["payload"]["headers"]
        email_data = {
            "id": msg["id"],
            "threadId": msg.get("threadId"),
            "snippet": msg.get("snippet"),
            "subject": "",
            "from": "",
            "to": "",
            "date": "",
            "body": "",
            "attachments": [],
        }

        # 1. 解析邮件头
        for header in headers:
            name = header["name"].lower()
            if name == "subject":
                email_data["subject"] = header["value"]
            elif name == "from":
                email_data["from"] = header["value"]
            elif name == "to":
                email_data["to"] = header["value"]
            elif name == "date":
                email_data["date"] = header["value"]

        # 2. 解析 multipart 结构
        payload = msg["payload"]
        html_body = ""  # HTML 兜底

        if "parts" in payload:
            for part in payload["parts"]:
                mime = part.get("mimeType", "")
                filename = part.get("filename")

                # 纯文本正文（优先）
                if mime == "text/plain" and not filename:
                    data = part["body"].get("data", "")
                    if data:
                        email_data["body"] = self._decode_base64(data)

                # HTML 正文（兜底）
                elif mime == "text/html" and not filename:
                    data = part["body"].get("data", "")
                    if data:
                        html_body = self._decode_base64(data)

                # 附件
                elif filename:
                    attachment_info = self._extract_attachment(msg_id, part)
                    if attachment_info:
                        email_data["attachments"].append(attachment_info)
        else:
            # 非 multipart：正文直接在 payload.body 里
            data = payload["body"].get("data", "")
            if data:
                email_data["body"] = self._decode_base64(data)

        # 如果没有纯文本正文，从 HTML 中提取文本
        if not email_data["body"] and html_body:
            email_data["body"] = self._strip_html(html_body)

        return email_data

    # ── 附件提取 ────────────────────────────────────────

    def _extract_attachment(self, message_id: str, part: dict) -> dict | None:
        """下载并保存单个附件"""
        original_filename = part.get("filename")
        if not original_filename:
            return None

        safe_filename = original_filename.replace("/", "_").replace("\\", "_")
        save_filename = f"{message_id}_{safe_filename}"

        # 获取附件数据
        body = part.get("body", {})
        attachment_id = body.get("attachmentId")
        data_b64 = body.get("data")

        if attachment_id and not data_b64:
            try:
                attachment = (
                    self.service.users()
                    .messages()
                    .attachments()
                    .get(userId="me", messageId=message_id, id=attachment_id)
                    .execute()
                )
                data_b64 = attachment.get("data")
            except Exception as e:
                logger.error(f"下载附件 {safe_filename} 失败: {e}")
                return None

        if not data_b64:
            return None

        file_data = base64.urlsafe_b64decode(data_b64)

        # 保存到本地
        if not os.path.exists(ATTACHMENT_SAVE_DIR):
            os.makedirs(ATTACHMENT_SAVE_DIR)
        filepath = os.path.join(ATTACHMENT_SAVE_DIR, save_filename)
        with open(filepath, "wb") as f:
            f.write(file_data)

        # 文本文件尝试提取内容
        if original_filename.lower().endswith((".txt", ".md", ".csv", ".json")):
            try:
                extracted_text = file_data.decode("utf-8")
            except UnicodeDecodeError:
                extracted_text = "无法解码文本内容 (非 UTF-8 编码)"
        else:
            extracted_text = f"[非纯文本附件，保存在: {filepath}]"

        return {
            "filename": original_filename,
            "save_filename": save_filename,
            "filepath": filepath,
            "text": extracted_text,
        }

    # ── 工具 ────────────────────────────────────────────

    @staticmethod
    def _decode_base64(data: str) -> str:
        """URL-safe Base64 → UTF-8 字符串"""
        try:
            return base64.urlsafe_b64decode(data).decode("utf-8")
        except Exception:
            return ""

    @staticmethod
    def _strip_html(html: str) -> str:
        """去除 HTML 标签，提取纯文本"""
        import re
        # 移除非断空格 &nbsp;
        text = html.replace("&nbsp;", " ").replace("&amp;", "&")
        text = text.replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&quot;", '"').replace("&#39;", "'")
        # 移除所有 HTML 标签
        text = re.sub(r"<[^>]+>", "", text)
        # 合并连续空白
        text = re.sub(r"\s+", " ", text).strip()
        return text