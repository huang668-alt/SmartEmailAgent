"""
Agent 间消息总线 (Message Bus)

多 Agent 系统通过消息总线实现松耦合通信：
- AgentContext : 共享的上下文数据容器（黑板模式）
                 所有 Agent 共享同一个实例，通过 put/get 读写数据，
                 通过 record/history 追踪消息流转。
- AgentMessage : Agent 之间传递的消息，携带发送者、接收者、角色和内容。
- MessageBus   : 消息路由与会话管理，负责创建/关闭 AgentContext 实例，
                 并为每条消息生成唯一 ID、记录到上下文历史。

通信模式：
  Orchestrator ──send()──▶ Agent
  Agent        ──send()──▶ Orchestrator  （结果回写）
  Orchestrator ──broadcast()──▶ *        （广播系统通知）
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════
#  消息角色枚举
# ══════════════════════════════════════════════════════

class MessageRole(Enum):
    """
    标识消息的来源语义，便于消费方区分处理逻辑：
      ORCHESTRATOR : 编排器下发的任务指令
      AGENT        : Agent 产出的执行结果
      SYSTEM       : 总线/系统级通知（如会话开始、广播事件）
    """
    ORCHESTRATOR = "orchestrator"
    AGENT        = "agent"
    SYSTEM       = "system"


# ══════════════════════════════════════════════════════
#  消息数据类
# ══════════════════════════════════════════════════════

@dataclass
class AgentMessage:
    """
    Agent 间传递的基本消息单元。

    设计为不可变值对象：创建后字段不应被修改，
    历史记录依赖其稳定性。

    Fields:
        id        : 全局唯一消息 ID（由 MessageBus 自增生成）
        sender    : 发送者注册名称
        receiver  : 接收者注册名称；"*" 表示广播给所有 Agent
        role      : 消息语义角色，见 MessageRole
        content   : 任意消息内容（字符串、字典、结构体均可）
        timestamp : 消息创建时的 Unix 时间戳（秒），默认自动填充
        metadata  : 可选的附加元数据（如 step_id、trace_id 等）
    """
    id        : str
    sender    : str
    receiver  : str
    role      : MessageRole
    content   : Any
    timestamp : float            = field(default_factory=time.time)
    metadata  : Dict[str, Any]  = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可 JSON 化的字典，role 转为字符串值"""
        return {
            "id"       : self.id,
            "sender"   : self.sender,
            "receiver" : self.receiver,
            "role"     : self.role.value,   # Enum → str，便于序列化
            "content"  : self.content,
            "timestamp": self.timestamp,
            "metadata" : self.metadata,
        }


# ══════════════════════════════════════════════════════
#  共享上下文容器（黑板模式）
# ══════════════════════════════════════════════════════

class AgentContext:
    """
    单次编排会话的共享状态容器，采用"黑板"模式：

    黑板模式简介：
      所有参与者（Agent）共享同一块"黑板"（本类），
      任意 Agent 可读写黑板上的数据，Orchestrator 协调读写顺序，
      避免 Agent 之间直接耦合。

    生命周期：
      由 MessageBus.create_session() 创建 → 编排过程中持续使用
      → 编排结束后由 MessageBus.close_session() 释放。

    线程安全：
      当前实现不加锁，假设同一会话内操作是串行的。
      若需并发写入，应在 put/record 处加锁。
    """

    def __init__(self, session_id: str = ""):
        # 若未指定 session_id，使用时间戳生成唯一 ID，确保不同会话不冲突
        self.session_id = session_id or f"session_{int(time.time())}"
        self._data   : Dict[str, Any]     = {}  # 键值数据区：Agent 产出的结果存储于此
        self._history: List[AgentMessage] = []  # 消息历史区：按时序记录所有消息

    # ── 数据存取 ──────────────────────────────────────

    def put(self, key: str, value: Any) -> None:
        """
        写入或覆盖上下文中的一个键值对。

        通常由 Orchestrator 在 Agent 执行后调用，
        将 AgentResult 的输出写入，供后续步骤读取。
        """
        self._data[key] = value
        logger.debug(f"[{self.session_id}] 上下文写入: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        读取上下文中的键值，key 不存在时返回 default（默认 None）。

        Agent 通过此方法获取前置步骤的输出；
        建议调用前先用 has() 确认 key 存在，避免静默使用默认值。
        """
        return self._data.get(key, default)

    def has(self, key: str) -> bool:
        """检查某个 key 是否已写入上下文，用于前置依赖校验"""
        return key in self._data

    # ── 消息历史 ──────────────────────────────────────

    def record(self, message: AgentMessage) -> None:
        """
        将消息追加到历史记录。

        由 MessageBus.send() 自动调用，外部通常不需要直接调用。
        历史记录用于审计、调试和会话回放。
        """
        self._history.append(message)

    @property
    def history(self) -> List[AgentMessage]:
        """返回消息历史的浅拷贝，防止外部意外修改内部列表"""
        return list(self._history)

    def last_message_from(self, sender: str) -> Optional[AgentMessage]:
        """
        从历史记录中倒序查找，返回指定发送者的最后一条消息。

        常用场景：Orchestrator 查询某 Agent 最近一次的执行结果。
        若该发送者没有任何消息，返回 None。
        """
        for msg in reversed(self._history):
            if msg.sender == sender:
                return msg
        return None

    # ── 快照 ──────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """
        返回当前上下文的只读快照，用于日志、调试或持久化。

        注意：data 字段是浅拷贝，嵌套的可变对象仍为引用。
        """
        return {
            "session_id"   : self.session_id,
            "data_keys"    : list(self._data.keys()),  # 仅暴露键名，隐藏值（避免日志过大）
            "message_count": len(self._history),
            "data"         : dict(self._data),         # 浅拷贝，防止外部修改
        }

    def __repr__(self) -> str:
        return f"<AgentContext session={self.session_id} keys={list(self._data.keys())}>"


# ══════════════════════════════════════════════════════
#  消息总线
# ══════════════════════════════════════════════════════

class MessageBus:
    """
    Agent 消息总线——多 Agent 系统的通信枢纽。

    职责：
      1. 会话管理：为每次编排创建独立的 AgentContext，隔离不同会话的状态。
      2. 消息路由：构造消息对象并写入目标会话的历史，支持点对点和广播。
      3. 消息计数：全局自增 ID 保证消息可追溯、可排序。

    设计说明：
      MessageBus 本身不持有 Agent 引用，也不执行任何业务逻辑；
      它只是"邮局"，负责编号、投递和存档，解耦 Agent 间的直接依赖。
    """

    def __init__(self):
        self._contexts        : Dict[str, AgentContext] = {}  # session_id → AgentContext
        self._message_counter : int = 0                       # 全局消息序号，单调递增

    def create_session(self, session_id: str = "") -> AgentContext:
        """
        创建并注册一个新的编排会话上下文。

        若 session_id 为空，由 AgentContext 内部使用时间戳自动生成。
        返回的 AgentContext 实例由调用方（通常是 Orchestrator）持有并传递给各 Agent。
        """
        ctx = AgentContext(session_id)
        self._contexts[ctx.session_id] = ctx
        logger.info(f"消息总线创建会话: {ctx.session_id}")
        return ctx

    def get_session(self, session_id: str) -> Optional[AgentContext]:
        """
        按 session_id 查找已有会话上下文。

        返回 None 表示该会话不存在或已被关闭，调用方应做判空处理。
        """
        return self._contexts.get(session_id)

    def close_session(self, session_id: str) -> None:
        """
        关闭并从注册表中移除会话，释放对 AgentContext 的引用。

        调用后 AgentContext 对象若无其他引用，将被 GC 回收。
        若 session_id 不存在，静默忽略（pop 的默认行为）。
        """
        self._contexts.pop(session_id, None)
        logger.info(f"消息总线关闭会话: {session_id}")

    def send(
        self,
        sender  : str,
        receiver: str,
        content : Any,
        role    : MessageRole               = MessageRole.AGENT,
        context : Optional[AgentContext]    = None,
        metadata: Optional[Dict[str, Any]]  = None,
    ) -> AgentMessage:
        """
        构造并发送一条消息。

        Args:
            sender  : 发送者名称
            receiver: 接收者名称（"*" 表示广播）
            content : 消息内容，任意类型
            role    : 消息角色，默认为 AGENT（结果消息）
            context : 目标会话上下文；若提供，消息将被记录到该会话历史
            metadata: 附加元数据（如 step_id、retry_count 等调试信息）

        Returns:
            构造好的 AgentMessage 实例（已写入 context.history，若 context 非 None）
        """
        # 消息序号自增，生成全局唯一 ID（格式: msg_1, msg_2, ...）
        self._message_counter += 1
        msg = AgentMessage(
            id      =f"msg_{self._message_counter}",
            sender  =sender,
            receiver=receiver,
            role    =role,
            content =content,
            metadata=metadata or {},
        )

        # 若传入了会话上下文，将消息写入其历史记录
        if context:
            context.record(msg)

        logger.debug(f"消息路由: {sender} → {receiver} [{role.value}]")
        return msg

    def broadcast(
        self,
        sender : str,
        content: Any,
        context: AgentContext,
        role   : MessageRole = MessageRole.SYSTEM,
    ) -> AgentMessage:
        """
        向所有 Agent 广播消息（receiver 固定为 "*"）。

        常用于 Orchestrator 发出会话开始/结束通知，
        或系统级事件（如超时、取消）的全局通知。

        本方法是 send() 的语义包装，receiver="*" 由调用方约定解释，
        MessageBus 本身不维护 Agent 订阅列表。
        """
        return self.send(sender, "*", content, role, context)