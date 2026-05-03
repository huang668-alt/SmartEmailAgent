import base64
import datetime
import logging
import os
import pickle
import time
from urllib.request import Request

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

class EmailAccessAndSynchronizationModule:

    def __init__(self):
        self.service = None
        self.last_sync_time = None
        self.credentials = None

    def authenticate(self):
        """执行 OAuth2 授权（Gmail）"""

        SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json',
                    SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        self.service = build('gmail', 'v1', credentials=creds)
        self.credentials = creds

        self.last_sync_time = time.time()
        return self.service

    def fetch_new_emails(self, max_results):
        """增量拉取新邮件（推荐使用）"""

        if self.last_sync_time is None:
            query = "is:inbox"
        else:
            last_date = datetime.datetime.fromtimestamp(self.last_sync_time)
            safe_date = last_date - datetime.timedelta(days=1)
            after_str = safe_date.strftime("%Y/%m/%d")
            query = f"is:inbox after:{after_str}"
        try:
            results = self.service.users().messages().list(
                userId='me',
                maxResults=max_results,
                q=query
            ).execute()
            messages = results.get('messages', [])
            self.last_sync_time = time.time()
            return messages
        except Exception as e:
            logging.error(f"拉取邮件失败: {e}")
            return []

    def parse_email(self, msg_id):
        """完整解析单封邮件（包含正文和附件）"""

        try:
            msg = self.service.users().messages().get(
                userId='me',
                id=msg_id,
                format='full'
            ).execute()
            headers = msg['payload']['headers']
            email_data = {
                'id': msg['id'],
                'threadId': msg.get('threadId'),
                'snippet': msg.get('snippet'),
                'subject': '',
                'from': '',
                'date': '',
                'body': '',
                'attachments': []
            }
            # 1. 遍历并解析邮件头 (Headers)
            # headers 包含了邮件的元数据，如发件人、收件人、主题、时间等
            for header in headers:
                if header['name'].lower() == 'subject':
                    email_data['subject'] = header['value']  # 提取邮件主题 (标题)
                elif header['name'].lower() == 'from':
                    email_data['from'] = header['value']  # 提取发件人信息 (如 "John Doe <john@example.com>")
                elif header['name'].lower() == 'date':
                    email_data['date'] = header['value']  # 提取邮件发送日期和时间

            # 2. 解析邮件的正文和附件 (处理 Multipart 结构)
            # 如果邮件内容是复杂结构（比如既有正文又有附件，或者既有HTML又有纯文本），它会包含 'parts' 列表
            if 'parts' in msg['payload']:
                # 遍历这封邮件的每一个组成部分 (part)
                for part in msg['payload']['parts']:

                    # 场景 A: 提取纯文本正文
                    # 条件：MIME类型是纯文本，并且没有文件名（有文件名说明它是文本格式的附件，而不是正文）
                    if part['mimeType'] == 'text/plain' and not part.get('filename'):
                        data = part['body'].get('data', '')
                        if data:
                            # Gmail 的数据是 URL-Safe Base64 编码的，需要解码转换成 utf-8 字符串
                            email_data['body'] = base64.urlsafe_b64decode(data).decode('utf-8')

                    # 场景 B: 提取附件
                    # 条件：只要这个 part 包含了 'filename' 属性，就说明它是一个附件文件
                    elif part.get('filename'):
                        # 调用提取附件的专用方法，传入当前邮件ID和当前的 part 数据
                        attachment_info = self.extract_attachments(msg_id, part)
                        # 如果附件下载/解析成功，将其信息（路径、解析出的文本等）追加到邮件数据字典中
                        if attachment_info:
                            email_data['attachments'].append(attachment_info)
            else:
                data = msg['payload']['body'].get('data', '')
                if data:
                    email_data['body'] = base64.urlsafe_b64decode(data).decode('utf-8')
            return email_data
        except Exception as e:
            logging.error(f"获取邮件失败 {msg_id}: {e}")
            return None

    def extract_attachments(self, message_id, part):
        """保存并提取附件文本"""

        original_filename = part.get("filename")
        safe_filename = original_filename.replace('/', '_').replace('\\', '_')
        save_filename = f"{message_id}_{safe_filename}"
        if not original_filename:
            return None
        body = part.get("body", {})
        attachment_id = body.get("attachmentId")
        data_b64 = body.get("data")
        if attachment_id and not data_b64:
            try:
                attachment = self.service.users().messages().attachments().get(
                    userId='me',
                    messageId=message_id,
                    id=attachment_id
                ).execute()
                data_b64 = attachment.get("data")
            except Exception as e:
                logging.error(f"下载附件 {save_filename} 失败: {e}")
                return None

        if not data_b64:
            return None
        file_data = base64.urlsafe_b64decode(data_b64)
        save_dir = "attachments"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, save_filename)
        with open(filepath, 'wb') as f:
            f.write(file_data)
        if original_filename.lower().endswith(('.txt', '.md', '.csv', '.json')):
            try:
                extracted_text = file_data.decode('utf-8')
            except UnicodeDecodeError:
                extracted_text = "无法解码文本内容 (可能不是 UTF-8 编码)"
        else:
            extracted_text = f"[非纯文本附件，保存在: {filepath}]"
        return {
            "filename": original_filename,
            "save_filename": save_filename,
            "filepath": filepath,
            "text": extracted_text
        }