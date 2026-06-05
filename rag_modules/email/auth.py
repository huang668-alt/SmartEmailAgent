"""Gmail OAuth2 认证模块"""

import logging
import os
import pickle

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource

logger = logging.getLogger(__name__)

DEFAULT_SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


class GmailAuth:
    """Gmail OAuth2 认证，返回 authenticated Google API service"""

    def __init__(
        self,
        credentials_path: str = "credentials.json",
        token_path: str = "token.pickle",
        scopes: list[str] | None = None,
    ):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.scopes = scopes or DEFAULT_SCOPES

    def authenticate(self) -> Resource:
        """
        执行 OAuth2 授权流程：
        1. 尝试从本地 pickle 加载已有 token
        2. 过期则尝试 refresh
        3. 无效则启动浏览器交互授权
        4. 保存新 token 到本地
        """
        creds: Credentials | None = None

        if os.path.exists(self.token_path):
            with open(self.token_path, "rb") as token:
                creds = pickle.load(token)
            logger.debug("已加载本地 token")

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Token 已过期，尝试刷新...")
                creds.refresh(Request())
            else:
                logger.info("启动交互式 OAuth 授权...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.scopes
                )
                creds = flow.run_local_server(port=0)

            with open(self.token_path, "wb") as token:
                pickle.dump(creds, token)
            logger.info("新 token 已保存")

        return build("gmail", "v1", credentials=creds)
