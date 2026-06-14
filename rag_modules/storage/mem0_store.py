"""
mem0ai 记忆存储封装

基于 mem0ai 2.0.5，使用 Milvus 作为向量后端，复用项目已有的 LLM / Embedding 配置。
提供记忆的 CRUD 和语义搜索能力。

集成点:
    - Pipeline.store_context() → mem0_store.add()
    - Pipeline 聊天后自动提取记忆 → mem0_store.add_from_conversation()
    - RAG 问答搜索记忆 → mem0_store.search()
"""

import logging
from typing import Any, Dict, List, Optional

from mem0 import Memory
from mem0.configs.base import MemoryConfig
from mem0.vector_stores.configs import VectorStoreConfig
from mem0.llms.configs import LlmConfig
from mem0.embeddings.configs import EmbedderConfig

from config import SmartEmailAgentConfig as Config

logger = logging.getLogger(__name__)

# ── 默认用户/Agent ID ─────────────────────────────────────
DEFAULT_USER_ID = "smartemail_user"
DEFAULT_AGENT_ID = "smartemail_agent"


class Mem0Store:
    """
    mem0ai Memory 封装

    初始化时自动连接 Milvus（复用 config 中的 URL），
    后续记忆操作均通过 mem0ai 的 Memory 实例完成。
    """

    def __init__(
        self,
        user_id: str = DEFAULT_USER_ID,
        agent_id: str = DEFAULT_AGENT_ID,
    ):
        self.user_id = user_id
        self.agent_id = agent_id

        # ── 构建 mem0ai MemoryConfig ─────────────────────
        # 向量存储：复用项目已有的 Milvus
        vector_config_dict = {
            "url": Config.milvus_url,
            "collection_name": "mem0_memories",
            "embedding_model_dims": Config.embeddings_dimension,
            "metric_type": "COSINE",
        }
        vector_config = VectorStoreConfig(
            provider="milvus",
            config=vector_config_dict,
        )

        # LLM：复用项目已有的 OpenAI 配置
        llm_config = LlmConfig(
            provider="openai",
            config={
                "model": Config.summarizer_agent_module_name or "gpt-4o",
                "api_key": Config.summarizer_agent_module_api_key,
                "openai_base_url": Config.summarizer_agent_module_base_url,
                "temperature": Config.temperature,
                "max_tokens": Config.max_tokens,
            },
        )

        # Embedder：复用项目已有的 Embedding 配置
        embedder_config = EmbedderConfig(
            provider="openai",
            config={
                "model": Config.embeddings_model_name or "text-embedding-3-small",
                "api_key": Config.embeddings_api_key,
                "openai_base_url": Config.embeddings_url,
                "embedding_dims": Config.embeddings_dimension,
            },
        )

        memory_config = MemoryConfig(
            vector_store=vector_config,
            llm=llm_config,
            embedder=embedder_config,
        )

        self.memory = Memory(config=memory_config)
        logger.info("mem0ai Memory 已初始化 (Milvus: %s)", Config.milvus_url)

    # ── 公开 API ──────────────────────────────────────────

    def add(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        存储文本记忆

        Args:
            content:   待存储的文本
            metadata:  附加元数据（如来源、标签等）
            infer:     是否让 LLM 自动从文本中提取关键记忆（默认 True）
            memory_type: 记忆类型标记

        Returns:
            mem0ai add 的返回列表
        """
        messages = [{"role": "user", "content": content}]
        try:
            results = self.memory.add(
                messages,
                user_id=self.user_id,
                agent_id=self.agent_id,
                metadata=metadata or {},
                infer=infer,
                memory_type=memory_type,
            )
            logger.info("mem0 记忆已存储，结果数: %d", len(results.get("results", [])))
            return results
        except Exception as e:
            logger.error("mem0 存储记忆失败: %s", e)
            raise

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.1,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        语义搜索记忆

        Args:
            query:     搜索查询
            top_k:     返回条数
            threshold: 相似度阈值
            filters:   过滤条件

        Returns:
            匹配的记忆列表
        """
        try:
            results = self.memory.search(
                query,
                user_id=self.user_id,
                top_k=top_k,
                threshold=threshold,
                filters=filters,
            )
            return results.get("results", [])
        except Exception as e:
            logger.warning("mem0 搜索记忆失败: %s", e)
            return []

    def get_all(self, top_k: int = 50) -> List[Dict[str, Any]]:
        """获取所有记忆"""
        try:
            results = self.memory.get_all(
                user_id=self.user_id,
                top_k=top_k,
            )
            return results.get("results", [])
        except Exception as e:
            logger.warning("mem0 获取全部记忆失败: %s", e)
            return []

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """获取单条记忆"""
        try:
            return self.memory.get(memory_id)
        except Exception as e:
            logger.warning("mem0 获取记忆失败 (%s): %s", memory_id, e)
            return None

    def update(
        self,
        memory_id: str,
        data: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """更新记忆"""
        return self.memory.update(memory_id, data, metadata=metadata)

    def delete(self, memory_id: str) -> None:
        """删除单条记忆"""
        self.memory.delete(memory_id)
        logger.info("已删除记忆: %s", memory_id)

    def delete_all(self) -> None:
        """删除当前用户所有记忆"""
        self.memory.delete_all(user_id=self.user_id)
        logger.info("已删除用户 %s 的所有记忆", self.user_id)

    def history(self, memory_id: str) -> List[Dict[str, Any]]:
        """获取记忆变更历史"""
        return self.memory.history(memory_id)

    def reset(self) -> None:
        """重置记忆存储"""
        self.memory.reset()

    # ── 便捷方法 ──────────────────────────────────────────

    def add_from_conversation(
        self,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        从一轮对话中提取并存储记忆

        Args:
            user_message:      用户提问
            assistant_message: AI 回答
            metadata:          附加元数据

        Returns:
            提取的记忆列表
        """
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        try:
            results = self.memory.add(
                messages,
                user_id=self.user_id,
                agent_id=self.agent_id,
                metadata=metadata or {},
                infer=True,
            )
            count = len(results.get("results", []))
            logger.info("从对话中提取了 %d 条记忆", count)
            return results
        except Exception as e:
            logger.error("从对话提取记忆失败: %s", e)
            return {"results": []}

    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """
        将 mem0 搜索结果格式化为 LLM 可消费的文本

        用于 RAG 问答时拼接上下文。
        """
        if not results:
            return ""

        lines = ["## 长期记忆 (mem0)\n"]
        for i, item in enumerate(results, start=1):
            memory_text = item.get("memory", "")
            score = item.get("score", 0)
            lines.append(f"### [记忆{i}] 相似度: {round(score, 4)}")
            lines.append(f"{memory_text}\n")

        return "\n".join(lines)


# ── 模块级懒加载 ──────────────────────────────────────────

_mem0_store: Optional[Mem0Store] = None

def get_mem0_store() -> Mem0Store:
    """懒加载 Mem0Store 单例"""
    global _mem0_store
    if _mem0_store is None:
        _mem0_store = Mem0Store()
    return _mem0_store

def reset_mem0_store() -> None:
    """重置 Mem0Store 单例（配置变更后调用）"""
    global _mem0_store
    _mem0_store = None
    logger.info("Mem0Store 单例已重置")
