"""Milvus 向量存储封装"""

import logging
from typing import Any

from pymilvus import MilvusClient

from rag_modules import MilvusConnectionModule
from .schema import create_inbox_collection, create_sent_collection, create_context_collection

logger = logging.getLogger(__name__)

COLLECTION_INBOX = "accept_email_collection"
COLLECTION_SENT = "send_email_collection"
COLLECTION_CONTEXT = "context_historical"


class VectorStore:
    """Milvus 向量存储：连接管理 + Collection 自动创建 + 插入"""

    def __init__(self):
        self.client: MilvusClient = MilvusConnectionModule().connection()
        self._collections_ready: set[str] = set()

    # ── Collection 管理 ──────────────────────────────────

    def ensure_inbox(self) -> str:
        name = COLLECTION_INBOX
        if name not in self._collections_ready:
            create_inbox_collection(self.client, name)
            self._collections_ready.add(name)
        return name

    def ensure_sent(self) -> str:
        name = COLLECTION_SENT
        if name not in self._collections_ready:
            create_sent_collection(self.client, name)
            self._collections_ready.add(name)
        return name

    def ensure_context(self) -> str:
        name = COLLECTION_CONTEXT
        if name not in self._collections_ready:
            create_context_collection(self.client, name)
            self._collections_ready.add(name)
        return name

    # ── 插入 ────────────────────────────────────────────

    def insert_inbox_email(self, data: dict[str, Any]):
        """插入收件箱邮件（字段自动映射）"""
        self.client.insert(collection_name=self.ensure_inbox(), data={
            "id": data["id"],
            "threadId": data.get("threadId", ""),
            "snippet": data.get("snippet", ""),
            "subject": data["subject"],
            "from_address": data["from"],
            "date": data["date"],
            "body": data["body"],
            "attachments": data.get("attachments", []),
            "priority": data.get("priority", "Low"),
            "reason": data.get("reason", ""),
            "embedding": data.get("embedding", []),
        })

    def insert_sent_email(self, data: dict[str, Any]):
        """插入发件箱邮件"""
        self.client.insert(collection_name=self.ensure_sent(), data={
            "id": data["id"],
            "threadId": data.get("threadId", ""),
            "snippet": data.get("snippet", ""),
            "subject": data["subject"],
            "to_address": data.get("to", ""),
            "date": data["date"],
            "body": data["body"],
            "attachments": data.get("attachments", []),
            "embedding": data.get("embedding", []),
        })

    def insert_context(self, content: str, embedding: list[float], timestamp: float):
        """插入长期记忆"""
        self.client.insert(collection_name=self.ensure_context(), data=[{
            "embedding": embedding,
            "content": content,
            "timestamp": timestamp,
        }])

    # ── 查询 ────────────────────────────────────────────

    def search_inbox(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        """语义搜索收件箱"""
        results = self.client.search(
            collection_name=COLLECTION_INBOX,
            data=[query_vector], limit=top_k,
            output_fields=["subject", "from_address", "date", "body", "priority"],
        )
        return results[0] if results else []

    def search_context(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        """语义搜索长期记忆"""
        results = self.client.search(
            collection_name=COLLECTION_CONTEXT,
            data=[query_vector], limit=top_k,
            output_fields=["content", "timestamp"],
        )
        return results[0] if results else []
