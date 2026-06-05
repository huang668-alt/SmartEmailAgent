"""文本 → 向量 Embedding"""

import logging

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from config import SmartEmailAgentConfig

logger = logging.getLogger(__name__)


def embed(text: str) -> list[float]:
    """将文本转为 512 维向量"""
    if not text:
        return []
    try:
        model = OpenAIEmbeddings(
            model=SmartEmailAgentConfig.embeddings_model_name,
            dimensions=SmartEmailAgentConfig.embeddings_dimension,
            base_url=SmartEmailAgentConfig.embeddings_url,
            api_key=SecretStr(SmartEmailAgentConfig.embeddings_api_key),
        )
        return model.embed_query(text)
    except Exception as e:
        logger.error(f"Embedding 失败: {e}")
        return []
