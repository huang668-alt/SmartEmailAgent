"""Gmail 邮件列表拉取模块"""

import datetime
import logging
import time

from googleapiclient.discovery import Resource

logger = logging.getLogger(__name__)

# 增量同步安全重叠天数（防止时区/延迟导致漏信）
SAFETY_OVERLAP_DAYS = 1


class EmailFetcher:
    """拉取 Gmail 邮件列表（收件箱 + 发件箱），支持增量同步"""

    def __init__(self, service: Resource):
        self.service = service
        self._last_sync_time: float | None = None

    @property
    def last_sync_time(self) -> float | None:
        return self._last_sync_time

    # ── 收件箱 ──────────────────────────────────────────

    def fetch_inbox(self, max_results: int = 5) -> list[dict]:
        """增量拉取收件箱邮件（仅返回 message 摘要列表）"""
        query = self._build_query("is:inbox")
        return self._fetch(query, max_results)

    # ── 发件箱 ──────────────────────────────────────────

    def fetch_sent(self, max_results: int = 5) -> list[dict]:
        """增量拉取发件箱邮件"""
        query = self._build_query("is:sent")
        return self._fetch(query, max_results)

    # ── 内部 ────────────────────────────────────────────

    def _build_query(self, base: str) -> str:
        """构建 Gmail 搜索查询，加入时间过滤"""
        if self._last_sync_time is None:
            return base
        last_date = datetime.datetime.fromtimestamp(self._last_sync_time)
        safe_date = last_date - datetime.timedelta(days=SAFETY_OVERLAP_DAYS)
        after_str = safe_date.strftime("%Y/%m/%d")
        return f"{base} after:{after_str}"

    def _fetch(self, query: str, max_results: int) -> list[dict]:
        """执行 Gmail API list 请求"""
        try:
            results = (
                self.service.users()
                .messages()
                .list(userId="me", maxResults=max_results, q=query)
                .execute()
            )
            messages = results.get("messages", [])
            self._last_sync_time = time.time()
            logger.info(f"拉取到 {len(messages)} 封邮件 (query={query})")
            return messages
        except Exception as e:
            logger.error(f"拉取邮件失败: {e}")
            return []
