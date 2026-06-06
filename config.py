"""
SmartEmailAgent 系统配置

配置优先级：环境变量 > 代码中显式赋值 > 类默认值

环境变量:
    SMART_EMAIL_LLM_MODEL      — LLM 模型名
    SMART_EMAIL_LLM_BASE_URL   — LLM API 地址
    SMART_EMAIL_LLM_API_KEY    — LLM API Key
    SMART_EMAIL_EMBED_MODEL    — Embedding 模型名
    SMART_EMAIL_EMBED_BASE_URL — Embedding API 地址（默认同 LLM_BASE_URL）
    SMART_EMAIL_EMBED_API_KEY  — Embedding API Key（默认同 LLM_API_KEY）
    MILVUS_URL                 — Milvus 地址

快速开始:
    # Windows PowerShell
    $env:SMART_EMAIL_LLM_API_KEY = "sk-xxx"
    $env:SMART_EMAIL_LLM_BASE_URL = "https://api.openai.com/v1"

    # Linux / Mac
    export SMART_EMAIL_LLM_API_KEY="sk-xxx"
    export SMART_EMAIL_LLM_BASE_URL="https://api.openai.com/v1"

    python main.py
"""

import os
from dataclasses import dataclass, asdict
from typing import Any, Dict


def _env(key: str, fallback: str = "") -> str:
    val = os.environ.get(key, "")
    return val if val else fallback


@dataclass
class SmartEmailAgentConfig:
    """SmartEmailAgent 系统配置类 — 类属性可直接读写，兼容旧用法"""

    # ── Milvus ──────────────────────────────────────────────
    milvus_url: str = "http://localhost:19530"
    milvus_dimension: int = 512

    # ── LLM ────────────────────────────────────────────────
    summarizer_agent_module_name: str = ""
    summarizer_agent_module_base_url: str = ""
    summarizer_agent_module_api_key: str = ""

    # ── Embedding ──────────────────────────────────────────
    embeddings_model_name: str = ""
    embeddings_url: str = ""
    embeddings_dimension: int = 512
    embeddings_api_key: str = ""

    # ── 通用参数 ───────────────────────────────────────────
    temperature: float = 0.1
    max_tokens: int = 2048
    top_k: int = 5
    number_of_common_contacts: int = 5

    def __post_init__(self):
        """初始化后的处理（预留）"""
        pass

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SmartEmailAgentConfig":
        """从字典创建配置对象"""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


# ── 从环境变量自动装配（覆盖类默认值） ──────────────────────

_env_map = {
    "milvus_url":                        ("MILVUS_URL", "http://localhost:19530"),
    "summarizer_agent_module_name":      ("SMART_EMAIL_LLM_MODEL", "gpt-4o"),
    "summarizer_agent_module_base_url":  ("SMART_EMAIL_LLM_BASE_URL", "https://api.openai.com/v1"),
    "summarizer_agent_module_api_key":   ("SMART_EMAIL_LLM_API_KEY", ""),
    "embeddings_model_name":             ("SMART_EMAIL_EMBED_MODEL", "text-embedding-3-small"),
    "embeddings_api_key":                ("SMART_EMAIL_EMBED_API_KEY", ""),
}

for _field, (_env_key, _default) in _env_map.items():
    _value = _env(_env_key, _default)
    if _value:
        setattr(SmartEmailAgentConfig, _field, _value)

# Embedding URL 特殊处理：如果没单独配，就用 LLM 的 URL
_embed_url = _env("SMART_EMAIL_EMBED_BASE_URL")
if not _embed_url:
    _embed_url = getattr(SmartEmailAgentConfig, "summarizer_agent_module_base_url", "https://api.openai.com/v1")
setattr(SmartEmailAgentConfig, "embeddings_url", _embed_url)

# Embedding API Key 特殊处理：如果没单独配，就用 LLM 的
_embed_key = _env("SMART_EMAIL_EMBED_API_KEY")
if not _embed_key:
    _embed_key = getattr(SmartEmailAgentConfig, "summarizer_agent_module_api_key", "")
setattr(SmartEmailAgentConfig, "embeddings_api_key", _embed_key)

# ── 模块级默认实例 ──────────────────────────────────────────

DEFAULT_CONFIG = SmartEmailAgentConfig()
