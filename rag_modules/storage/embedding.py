"""文本 → 向量 Embedding（模块级懒加载单例）"""

import logging
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from config import SmartEmailAgentConfig

logger = logging.getLogger(__name__)

_embedding_model: Optional[OpenAIEmbeddings] = None


def _get_model() -> OpenAIEmbeddings:
    """懒加载 Embedding 模型单例，避免每次调用都实例化"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = OpenAIEmbeddings(
            model=SmartEmailAgentConfig.embeddings_model_name,
            dimensions=SmartEmailAgentConfig.embeddings_dimension,
            base_url=SmartEmailAgentConfig.embeddings_url,
            api_key=SecretStr(SmartEmailAgentConfig.embeddings_api_key),
        )
        logger.info("Embedding 模型已初始化（单例）")
    return _embedding_model


def embed(text: str) -> list[float]:
    """将文本转为 512 维向量（复用同一个模型实例）"""
    if not text:
        return []
    try:
        return _get_model().embed_query(text)
    except Exception as e:
        logger.error(f"Embedding 失败: {e}")
        return []


def reset_embedding_model() -> None:
    """重置 Embedding 模型实例（配置变更后调用）"""
    global _embedding_model
    _embedding_model = None
    logger.info("Embedding 模型实例已重置")
