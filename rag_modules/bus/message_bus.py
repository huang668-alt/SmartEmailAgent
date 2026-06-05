"""
Agent 间消息总线 (Message Bus)

多 Agent 系统通过消息总线实现松耦合通信：
- AgentContext: 共享的上下文数据容器（黑板模式）
- AgentMessage: Agent 之间传递的消息
- MessageBus: 消息路由与上下文管理

模式：Orchestrator 通过 MessageBus 向各个 Agent 分发输入，收集输出。
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """消息角色"""
    ORCHESTRATOR = "orchestrator"   # 编排器发出的指令
    AGENT = "agent"                 # Agent 产出的结果
    SYSTEM = "system"               # 系统级通知


@dataclass
class AgentMessage:
    """Agent 间传递的消息"""
    id: str                          # 消息唯一标识
    sender: str                      # 发送者名称
    receiver: str                    # 接收者名称（"*" 表示广播）
    role: MessageRole                # 消息角色
    content: Any                     # 消息内容
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class AgentContext:
    """
    共享上下文容器（黑板模式）

    所有 Agent 共享此上下文，Orchestrator 负责管理：
    - 写入：Agent 产出结果后写入
    - 读取：Orchestrator 汇总后分发
    - 生命周期：每次编排会话创建一个实例
    """

    def __init__(self, session_id: str = ""):
        self.session_id = session_id or f"session_{int(time.time())}"
        self._data: Dict[str, Any] = {}
        self._history: List[AgentMessage] = []

    # ── 数据存取 ──────────────────────────────────────

    def put(self, key: str, value: Any) -> None:
        """写入上下文数据"""
        self._data[key] = value
        logger.debug(f"[{self.session_id}] 上下文写入: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """读取上下文数据"""
        return self._data.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._data

    # ── 消息历史 ──────────────────────────────────────

    def record(self, message: AgentMessage) -> None:
        """记录一条消息到历史"""
        self._history.append(message)

    @property
    def history(self) -> List[AgentMessage]:
        return list(self._history)

    def last_message_from(self, sender: str) -> Optional[AgentMessage]:
        """获取某个发送者的最后一条消息"""
        for msg in reversed(self._history):
            if msg.sender == sender:
                return msg
        return None

    # ── 快照 ──────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """返回当前上下文快照"""
        return {
            "session_id": self.session_id,
            "data_keys": list(self._data.keys()),
            "message_count": len(self._history),
            "data": dict(self._data),
        }

    def __repr__(self) -> str:
        return f"<AgentContext session={self.session_id} keys={list(self._data.keys())}>"


class MessageBus:
    """
    Agent 消息总线

    职责：
    1. 为每次编排会话创建独立的 AgentContext
    2. 路由消息（支持点名和广播）
    3. 维护消息历史
    """

    def __init__(self):
        self._contexts: Dict[str, AgentContext] = {}
        self._message_counter = 0

    def create_session(self, session_id: str = "") -> AgentContext:
        """创建新的编排会话"""
        ctx = AgentContext(session_id)
        self._contexts[ctx.session_id] = ctx
        logger.info(f"消息总线创建会话: {ctx.session_id}")
        return ctx

    def get_session(self, session_id: str) -> Optional[AgentContext]:
        """获取已有会话"""
        return self._contexts.get(session_id)

    def close_session(self, session_id: str) -> None:
        """关闭会话"""
        self._contexts.pop(session_id, None)
        logger.info(f"消息总线关闭会话: {session_id}")

    def send(
        self,
        sender: str,
        receiver: str,
        content: Any,
        role: MessageRole = MessageRole.AGENT,
        context: Optional[AgentContext] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """
        发送消息

        Args:
            sender: 发送者名称
            receiver: 接收者名称（"*" 表示广播）
            content: 消息内容
            role: 消息角色
            context: 目标会话上下文
            metadata: 附加元数据

        Returns:
            AgentMessage: 创建的消息对象
        """
        self._message_counter += 1
        msg = AgentMessage(
            id=f"msg_{self._message_counter}",
            sender=sender,
            receiver=receiver,
            role=role,
            content=content,
            metadata=metadata or {},
        )
        if context:
            context.record(msg)
        logger.debug(f"消息路由: {sender} → {receiver} [{role.value}]")
        return msg

    def broadcast(
        self,
        sender: str,
        content: Any,
        context: AgentContext,
        role: MessageRole = MessageRole.SYSTEM,
    ) -> AgentMessage:
        """广播消息到所有 Agent"""
        return self.send(sender, "*", content, role, context)
