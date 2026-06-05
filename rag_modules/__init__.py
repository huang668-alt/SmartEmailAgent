# Milvus 连接
from .milvus_connection_module import MilvusConnectionModule

# Gmail 模块
from .email import GmailAuth, EmailFetcher, EmailParser

# 向量存储
from .storage import VectorStore, embed

# 流水线
from .pipeline import Pipeline

# 多 Agent 系统
from .agents import (
    BaseAgent, AgentState, AgentResult,
    SummarizerAgent, ClassifierAgent, ReplyAgent,
    TaskExtractorAgent, ContextSummaryAgent, OrchestratorAgent,
)

# Agent 消息总线
from .bus import MessageBus, AgentMessage, AgentContext

__all__ = [
    # 基础设施
    "MilvusConnectionModule",
    # Gmail
    "GmailAuth", "EmailFetcher", "EmailParser",
    # 向量存储
    "VectorStore", "embed",
    # 流水线
    "Pipeline",
    # Agent
    "BaseAgent", "AgentState", "AgentResult",
    "SummarizerAgent", "ClassifierAgent", "ReplyAgent",
    "TaskExtractorAgent", "ContextSummaryAgent", "OrchestratorAgent",
    # 消息总线
    "MessageBus", "AgentMessage", "AgentContext",
]
