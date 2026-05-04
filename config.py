from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class SmartEmailAgentConfig:
    """SmartEmailAgent 系统配置类"""

    milvus_url: str = "http://localhost:19530"
    milvus_dimension: int = 512

    summarizer_agent_module_name: str = ""
    summarizer_agent_module_base_url: str = ""
    summarizer_agent_module_api_key: str = ""

    top_k: int = 5

    temperature: float = 0.1
    max_tokens: int = 2048

    def __post_init__(self):
        """初始化后的处理（预留）"""

        pass

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SmartEmailAgentConfig':
        """从字典创建配置对象"""

        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""

        return asdict(self)

DEFAULT_CONFIG = SmartEmailAgentConfig()
