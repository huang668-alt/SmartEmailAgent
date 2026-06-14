"""
多 Agent 系统基类

定义统一的 Agent 抽象：
- 生命周期状态机 (IDLE → THINKING → ACTING → DONE/ERROR)
- 每个 Agent 拥有独立的 LLM 实例和配置
- 工具注册与调用
- Agent 上下文管理
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from config import SmartEmailAgentConfig

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent 生命周期状态"""
    IDLE = "idle"           # 就绪，等待任务
    THINKING = "thinking"   # 正在推理/调用 LLM
    ACTING = "acting"       # 正在执行工具调用
    DONE = "done"           # 任务完成
    ERROR = "error"         # 执行出错


from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class AgentResult:
    """
    Agent执行结果

    用于统一封装Agent运行后的状态、返回数据、
    错误信息以及额外元数据。
    """

    # 是否执行成功
    # True：成功
    # False：失败
    success: bool

    # Agent返回的结果数据
    # 类型为Any，可以是字符串、字典、列表、对象等任意类型
    data: Any = None

    # 错误信息
    # 当success=False时通常会填写具体错误原因
    error: str = ""

    # 附加元数据
    # 用于存储日志、耗时、执行步骤等额外信息
    # field(default_factory=dict)保证每个实例拥有独立字典
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """
        定义对象的布尔值行为

        当执行：
            if result:

        实际调用：
            result.__bool__()

        返回success字段的值。
        """
        return self.success

class BaseAgent(ABC):
    """
    Agent 基类 — 所有专业 Agent 的抽象父类

    每个 Agent 拥有：
    - 独立名称和状态
    - 独立的 LLM 模型实例（可配置不同 model/temperature/api_key）
    - 工具注册表
    - 上下文字典（Agent 间共享数据的桥梁）

    子类必须实现 execute() 方法。
    """

    def __init__(
            self,
            name: str,
            system_prompt: str,
            model_name: Optional[str] = None,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            temperature: float = 0.1,
            max_tokens: int = 2048,
    ):
        """
        Agent初始化方法

        Args:
            name: Agent名称，用于区分不同Agent
                  例如：SummarizerAgent、RetrieverAgent

            system_prompt: 系统提示词
                           定义Agent的角色、职责和行为规范

            model_name: 大模型名称
                        例如：
                        - gpt-4o
                        - qwen-plus
                        - deepseek-chat
                        如果为空则使用配置文件中的默认模型

            base_url: 模型服务地址
                      例如：
                      https://api.openai.com/v1
                      如果为空则读取配置文件

            api_key: API密钥
                     用于访问模型服务
                     如果为空则读取配置文件

            temperature: 生成温度
                         值越小结果越稳定
                         值越大结果越发散
                         默认0.1适合业务场景

            max_tokens: 最大输出Token数
                        控制模型单次生成长度
        """

        # Agent名称
        self.name = name

        # 系统提示词
        # 使用下划线表示内部变量
        self._system_prompt = system_prompt

        # Agent当前状态
        # 初始状态为IDLE（空闲）
        self.state = AgentState.IDLE

        # 工具注册表
        #
        # 保存Agent可调用的工具
        #
        # 结构：
        # {
        #     "search": search_tool,
        #     "calculator": calculator_tool
        # }
        #
        self._tools: Dict[str, Callable] = {}

        # Agent上下文信息
        #
        # 用于存储运行过程中共享的数据
        #
        # 例如：
        # {
        #     "user_id": 123,
        #     "query": "机器学习是什么"
        # }
        #
        self._context: Dict[str, Any] = {}

        # 创建独立的大模型实例
        #
        # 每个Agent拥有自己的LLM对象
        #
        self.llm = ChatOpenAI(

            # 模型名称
            model=(
                    model_name
                    or SmartEmailAgentConfig.summarizer_agent_module_name
            ),

            # 采样温度
            temperature=temperature,

            # API服务地址
            base_url=(
                    base_url
                    or SmartEmailAgentConfig.summarizer_agent_module_base_url
            ),

            # API密钥
            # 使用SecretStr避免日志中泄露密钥
            api_key=SecretStr(
                api_key
                or SmartEmailAgentConfig.summarizer_agent_module_api_key
            ),

            # 最大输出长度
            max_tokens=max_tokens,
        )

        # 输出解析器
        #
        # 将LLM返回结果解析成纯字符串
        #
        # 例如：
        #
        # AIMessage(
        #     content="这是答案"
        # )
        #
        # 转换后：
        #
        # "这是答案"
        #
        self.output_parser = StrOutputParser()

    # ── 工具系统 ──────────────────────────────────────────

    def register_tool(self, name: str, func: Callable) -> None:
        """注册一个工具函数"""
        self._tools[name] = func
        logger.debug(f"Agent [{self.name}] 注册工具: {name}")

    def use_tool(self, name: str, **kwargs) -> Any:
        """调用已注册的工具"""
        if name not in self._tools:
            raise KeyError(f"工具 '{name}' 未在 Agent [{self.name}] 中注册")
        logger.info(f"Agent [{self.name}] 调用工具: {name}")
        return self._tools[name](**kwargs)

    @property
    def available_tools(self) -> list[str]:
        """返回已注册的工具名称列表"""
        return list(self._tools.keys())

    # ── 上下文管理 ─────────────────────────────────────────

    def update_context(self, key: str, value: Any) -> None:
        """更新 Agent 上下文"""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """读取 Agent 上下文"""
        return self._context.get(key, default)

    def clear_context(self) -> None:
        """清空上下文"""
        self._context.clear()

    # ── LLM 调用链 ─────────────────────────────────────────

    def _build_chain(self, user_prompt_template: str):
        """
        构建 LangChain 调用链：system prompt → user prompt → LLM → output parser
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._system_prompt),
            ("user", user_prompt_template),
        ])
        return prompt | self.llm | self.output_parser

    def _invoke_chain(self, chain, variables: Dict[str, Any]) -> str:
        """安全地调用 LLM 链（非流式）"""
        try:
            return chain.invoke(variables)
        except Exception as e:
            logger.error(f"Agent [{self.name}] LLM 调用失败: {e}")
            raise

    from typing import Dict, Any, Generator

    def _stream_chain(self, chain, variables: Dict[str, Any]) -> Generator[str, None, None]:
        """
        内部辅助方法：流式调用 LLM 链并安全地透传生成的文本片段。

        核心原理:
            利用 LangChain 的 .stream() 方法，将模型生成的 Response 拆分为多个
            Chunk（片段）。本方法充当“中转站”，在迭代过程中捕获可能发生的网络或模型错误。

        参数:
            chain: 已构建好的 LangChain 运行链 (Runnable) 对象。
            variables (Dict[str, Any]): 注入提示词模板的变量字典，例如 {"context": "...", "question": "..."}。

        Yields:
            str: 从 LLM 实时产出的字符串片段。

        Raises:
            Exception: 当模型调用失败或网络连接中断时，记录日志并向上层重新抛出异常，
                      以便上层 stream 方法能向用户展示具体的报错提示。
        """
        try:
            # 使用 LangChain 的流式接口，逐个获取生成的文本块
            for chunk in chain.stream(variables):
                # 这里根据 Chain 的类型不同，chunk 可能已经是字符串，也可能是包含了 content 的对象
                # 这种直接 yield 的方式要求 chain 的输出类型已在构建时处理为字符串 (StrOutputParser)
                yield chunk

        except Exception as e:
            # 记录详细的错误日志，包含 Agent 的名称以便在多智能体环境下定位问题
            logger.error(f"Agent [{self.name}] 流式调用 LLM 链路时发生异常: {e}")
            # 将异常继续抛出，确保上层业务逻辑能感知到“流中断”
            raise

    # ── 核心接口 ───────────────────────────────────────────

    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        执行 Agent 的核心任务（子类必须实现）

        Args:
            input_data: 输入数据字典，结构由各 Agent 自行定义

        Returns:
            AgentResult: 执行结果
        """
        ...

    def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        运行 Agent — 带状态管理和异常保护

        子类不应覆写此方法；应实现 execute() 来定义具体行为。
        """
        try:
            self.state = AgentState.THINKING
            logger.info(f"Agent [{self.name}] 开始执行")
            result = self.execute(input_data)
            self.state = AgentState.DONE if result.success else AgentState.ERROR
            if result.success:
                logger.info(f"Agent [{self.name}] 执行成功")
            else:
                logger.warning(f"Agent [{self.name}] 执行失败: {result.error}")
            return result
        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"Agent [{self.name}] 异常: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def reset(self) -> None:
        """重置 Agent 状态和上下文"""
        self.state = AgentState.IDLE
        self._context.clear()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}' state={self.state.value}>"
