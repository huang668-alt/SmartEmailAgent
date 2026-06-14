"""
编排器 Agent (Orchestrator Agent)

多 Agent 系统的大脑，负责：
1. 接收高层用户目标 → LLM 分解为子任务列表
2. 根据子任务类型路由到对应的专业 Agent
3. 管理 Agent 间上下文（通过 MessageBus）
4. 收集各 Agent 结果并聚合为最终输出
5. 支持条件分支（如：高优先级邮件自动起草回复）

支持的流水线：
- process_email: summarize → classify → [extract_tasks] → [draft_reply]
- process_batch: 并行处理多封邮件
- store_context: 压缩上下文 → 存入向量记忆
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .base import BaseAgent, AgentResult, AgentState
from .summarizer_agent import SummarizerAgent
from .classifier_agent import ClassifierAgent
from .reply_agent import ReplyAgent
from .task_extractor_agent import TaskExtractorAgent
from .context_summary_agent import ContextSummaryAgent
from .query_agent import QueryAgent

logger = logging.getLogger(__name__)

# ── Orchestrator 自身的 System Prompt（用于任务分解） ────────

ORCHESTRATOR_SYSTEM_PROMPT = """你是一个多 Agent 系统的编排器，也是用户唯一的入口。你管理以下专业 Agent：

1. **QueryAgent** — RAG 问答：从向量库中语义搜索邮件，基于邮件内容回答用户问题
2. **SummarizerAgent** — 邮件摘要（3-5句 + Action Items）
3. **ClassifierAgent** — 邮件优先级分类（High/Medium/Low/Spam）
4. **ReplyAgent** — 起草邮件回复（支持指定语气/长度/语言）
5. **TaskExtractorAgent** — 提取待办事项和会议信息
6. **ContextSummaryAgent** — 压缩上下文为长期记忆

你的首要职责：**判断用户意图，选择正确的执行路径**。

# 路由规则（非常重要）

## 路径A：查询/问答类 → 只用 QueryAgent
用户的意图是"问一个问题"、"查信息"、"了解情况"时，只返回 QueryAgent。
关键词：有没有、是什么、怎么样、什么时候、谁发的、最近、进展、总结一下、帮我看看
示例：
- "我有没有什么紧急邮件"         → [QueryAgent]
- "张三最近发了什么"              → [QueryAgent]
- "Q3项目进展如何"                → [QueryAgent]
- "服务器宕机的事情谁在处理"       → [QueryAgent]
- "帮我总结一下最近的邮件"         → [QueryAgent]

## 路径B：处理/操作类 → 用处理 Agent（Summarizer + Classifier + TaskExtractor）
用户的意图是"对邮件执行操作"时，使用处理类 Agent。需要先有邮件数据。
关键词：处理、分析、分类、提取、批量、收件箱
示例：
- "处理今天的收件箱"              → [SummarizerAgent, ClassifierAgent, TaskExtractorAgent]
- "分析这批邮件的优先级"           → [SummarizerAgent, ClassifierAgent]

## 路径C：回复类 → SummarizerAgent → ReplyAgent
用户明确要回复某封邮件时。
关键词：回复、回信、答复、起草
示例：
- "帮我回复张三那封邮件"          → [SummarizerAgent, ReplyAgent]

## 路径D：记忆类 → ContextSummaryAgent
用户要保存或压缩上下文时。
关键词：记住、保存、压缩、存档、记忆
示例：
- "把这个项目背景记住"            → [ContextSummaryAgent]

## 路径E：混合类 → QueryAgent + 其他
用户既要查又要操作时。
示例：
- "看看有没有重要的，有的话帮我回复" → QueryAgent 先查 → 找到后路由到 ReplyAgent

# 输出格式
{{
    "goal": "用户目标的一句话总结",
    "route": "query | process | reply | memory | mixed",
    "steps": [
        {{
            "agent": "Agent名称",
            "action": "描述该步要做什么",
            "depends_on": [],
            "input_from": "从哪里取输入数据"
        }}
    ]
}}

# 规则
- **查询类问题只用 QueryAgent，不要加处理 Agent**
- 步骤必须使用上述 6 个 Agent 之一
- 如果步骤之间无依赖，标记为并行执行
- 保持计划简洁，不超过 5 步"""

ORCHESTRATOR_USER_PROMPT = """用户目标：{goal}

可用的上下文信息：
{context}

请输出执行计划的 JSON。"""


class OrchestratorAgent(BaseAgent):
    """
    编排器 Agent

    输入:
        - goal: 用户高层目标（自然语言）
        - emails: （可选）邮件数据列表
        - context: （可选）额外上下文

    输出:
        - plan: 执行计划
        - results: 各 Agent 的执行结果聚合
    """

    def __init__(self, temperature: float = 0.2, **kwargs):
        super().__init__(  # 调用父类 BaseAgent 的构造方法
            name="OrchestratorAgent",  # 设定代理名称为 "OrchestratorAgent"，用于日志标识
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,  # 注入编排器专用的系统提示词，定义其任务分解的角色行为
            temperature=temperature,  # 透传温度参数，控制 LLM 生成计划的随机性
            **kwargs,  # 透传其余可选关键字参数，保证父类其他配置项也能正常接收
        )

        # 初始化子 Agent 注册表（员工花名册）
        # 类型注解：键为 Agent 名称字符串，值为 BaseAgent 子类实例
        self._agents: Dict[str, BaseAgent] = {}  # 创建空字典，准备存放所有已注册的专业 Agent

        # 立即填充花名册，将 5 个默认专业 Agent 实例化并注册
        self._register_default_agents()  # 调用私有方法完成默认团队的组建

    def _register_default_agents(self) -> None:
        """注册默认的专业 Agent（私有方法，仅在初始化时调用一次）"""

        # 定义默认的 5 个专业 Agent 映射表
        # 键：Agent 的注册名称（用于后续路由查找）
        # 值：对应 Agent 类的实例（即刻创建，完成各自初始化）
        defaults = {
            "QueryAgent": QueryAgent(),  # RAG 问答专家实例
            "SummarizerAgent": SummarizerAgent(),  # 邮件摘要专家实例
            "ClassifierAgent": ClassifierAgent(),  # 邮件优先级分类专家实例
            "ReplyAgent": ReplyAgent(),  # 邮件回复起草专家实例
            "TaskExtractorAgent": TaskExtractorAgent(),  # 待办事项提取专家实例
            "ContextSummaryAgent": ContextSummaryAgent(),  # 上下文压缩专家实例
        }

        # 遍历默认映射表，逐一登记到注册表中
        for name, agent in defaults.items():
            self.register_agent(name, agent)  # 调用注册方法完成登记（附日志记录）

    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """注册一个专业 Agent（对外公开接口，支持动态扩展）

        Args:
            name: Agent 的注册名称，后续通过该名称路由任务
            agent: 已实例化的 Agent 对象，必须继承自 BaseAgent
        """
        self._agents[name] = agent  # 将 Agent 实例存入花名册字典
        logger.info(f"编排器注册 Agent: {name}")  # 记录注册日志，方便追踪系统配置



    def unregister_agent(self, name: str) -> None:
        """从注册表中移除一个 Agent（支持热卸载）

        Args:
            name: 要移除的 Agent 注册名称

        说明:
            - 使用 dict.pop(key, default) 方法，键不存在时返回 None 而不抛出 KeyError
            - 移除后该 Agent 不再参与任务路由，后续调用会返回"未注册"错误
            - 与 register_agent 配对使用，支持动态增减专业 Agent
        """
        self._agents.pop(name, None)
        # pop 方法的第二个参数 None 是默认值，当 name 不存在时静默返回 None
        # 等价于: if name in self._agents: del self._agents[name]
        # 但 pop 写法更简洁且线程安全

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """根据名称获取已注册的 Agent 实例（安全查询，不存在返回 None）

        Args:
            name: Agent 的注册名称（如 "SummarizerAgent"）

        Returns:
            找到则返回对应的 BaseAgent 实例，未找到返回 None

        说明:
            - 使用 dict.get() 而非 dict[key]，键不存在时返回 None 而不是抛出 KeyError
            - 返回类型为 Optional[BaseAgent]，调用方需检查 None 后再使用
            - 通常配合 _route_to_agent 方法使用，用于任务路由前的 Agent 查找
        """
        return self._agents.get(name)
        # dict.get(key) 等价于:
        #   if key in self._agents: return self._agents[key]
        #   else: return None

    @property
    def registered_agents(self) -> List[str]:
        """获取当前所有已注册 Agent 的名称列表（只读属性）

        Returns:
            包含所有已注册 Agent 名称的列表，如 ["SummarizerAgent", "ClassifierAgent", ...]

        说明:
            - @property 装饰器使其可以像访问属性一样调用，无需加括号: obj.registered_agents
            - 返回的是 self._agents 字典键的副本（list() 创建新列表），外部修改不影响内部注册表
            - 常用于调试、日志输出、或向用户展示当前可用的 Agent 列表
            - 配合 _route_to_agent 中的错误提示使用: f"可用: {self.registered_agents}"
        """
        return list(self._agents.keys())
        # self._agents.keys() 返回字典键的视图对象
        # list() 将其转为独立列表，防止外部意外修改

    # ── 任务分解 ────────────────────────────────────────────

    def plan(self, goal: str, context: str = "") -> List[Dict[str, Any]]:
        """
        用 LLM 将高层目标分解为执行计划

        Args:
            goal: 用户输入的高层目标（自然语言描述）
            context: 可选的补充上下文信息，默认为空字符串

        Returns:
            steps: 步骤列表，每个步骤为字典，包含以下字段：
                - agent: 负责执行该步骤的 Agent 名称
                - action: 该步骤需要执行的具体动作
                - depends_on: 依赖的前置步骤列表（用于控制执行顺序）
                - input_from: 输入数据的来源步骤
        """
        # 使用编排器的用户 Prompt 模板构建 LLM 调用链
        chain = self._build_chain(ORCHESTRATOR_USER_PROMPT)
        try:
            # 将 goal 与 context 注入 Prompt，调用 LLM 获取 JSON 格式的计划
            response = chain.invoke({
                "goal": goal,
                "context": context or "无额外上下文",  # context 为空时使用占位文本，避免 Prompt 中出现空字段
            })

            # 解析 LLM 返回的 JSON 字符串为 Python 对象
            plan = json.loads(response)

            # 提取步骤列表，若 LLM 输出中缺少 "steps" 字段则返回空列表
            steps = plan.get("steps", [])

            logger.info(f"编排器生成计划: {len(steps)} 步")
            return steps

        except (json.JSONDecodeError, Exception) as e:
            # JSON 解析失败或其他异常时（如 LLM 返回格式异常、网络超时等），
            # 降级到默认预设流程，保证系统可用性
            logger.warning(f"编排器计划生成失败: {e}，使用默认流程")
            return self._default_plan(goal)

    def _default_plan(self, goal: str) -> List[Dict[str, Any]]:
        """
        LLM 计划生成失败时的降级兜底流程。

        通过关键词匹配 goal 来选择预设的步骤模板，
        确保系统在 LLM 不可用时仍能完成基本处理。

        Args:
            goal: 用户输入的高层目标（自然语言描述）

        Returns:
            steps: 预设的步骤列表，结构与 plan() 返回值相同
        """
        # 统一转小写，使关键词匹配不区分大小写
        goal_lower = goal.lower()

        # 查询类关键词 → 纯 RAG 问答
        _query_kw = ("有没有", "是什么", "怎么样", "什么时候", "谁发的", "最近",
                     "进展", "总结一下", "帮我看看", "有没有什么", "怎么", "哪些", "哪个")
        if any(kw in goal_lower for kw in _query_kw):
            return [
                {"agent": "QueryAgent", "action": "语义检索并回答问题", "depends_on": []},
            ]

        # 场景一：包含"回复"相关关键词 → 先摘要、再起草回复
        if "回复" in goal_lower or "reply" in goal_lower:
            return [
                {"agent": "SummarizerAgent", "action": "总结邮件", "depends_on": []},
                {"agent": "ReplyAgent", "action": "起草回复", "depends_on": [0]},
            ]

        # 场景二：包含"记忆 / context / 压缩"关键词 → 仅执行上下文压缩
        if "记忆" in goal_lower or "context" in goal_lower or "压缩" in goal_lower:
            return [
                {"agent": "ContextSummaryAgent", "action": "压缩上下文", "depends_on": []},
            ]

        # 默认场景：通用邮件处理，三个 Agent 并行执行
        return [
            {"agent": "SummarizerAgent", "action": "总结邮件", "depends_on": []},
            {"agent": "ClassifierAgent", "action": "分类优先级", "depends_on": []},
            {"agent": "TaskExtractorAgent", "action": "提取任务", "depends_on": []},
        ]

    # ── Agent 路由 ───────────────────────────────────────────

    def _route_to_agent(
            self,
            agent_name: str,
            input_data: Dict[str, Any],
    ) -> AgentResult:
        """
        将任务路由到指定 Agent 并执行。

        在 Agent 注册表中查找目标 Agent，若不存在则立即返回失败结果，
        避免抛出异常打断整个编排流程。

        Args:
            agent_name:  目标 Agent 的注册名称（需与 _agents 中的 key 一致）
            input_data:  传递给 Agent 的输入数据字典

        Returns:
            AgentResult: Agent 执行结果，包含 success 标志、输出数据或错误信息
        """
        # 从注册表中查找 Agent 实例；未注册时返回 None
        agent = self._agents.get(agent_name)

        if not agent:
            # Agent 未注册：返回失败结果而非抛出异常，
            # 让上层编排逻辑决定是跳过、重试还是中止整个计划
            return AgentResult(
                success=False,
                error=f"Agent '{agent_name}' 未注册。可用: {self.registered_agents}",
            )

        logger.info(f"编排器路由: {agent_name}")

        # 调用 Agent 的标准入口 run()，异常由 Agent 内部捕获并封装到 AgentResult 中
        return agent.run(input_data)

    # ── 并行执行 ────────────────────────────────────────────

    def _execute_parallel(
        self,
        task_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, AgentResult]:
        """
        并行执行多个 Agent 任务

        Args:
            task_map: {agent_name: input_data}

        Returns:
            {agent_name: AgentResult}
        """

        results = {}
        with ThreadPoolExecutor(max_workers=len(task_map)) as executor:
            futures = {
                executor.submit(self._route_to_agent, name, data): name
                for name, data in task_map.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = AgentResult(success=False, error=str(e))
        return results

    # ── 核心执行 ────────────────────────────────────────────

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        执行编排

        输入结构:
        {
            "goal": "处理今天的新邮件",
            "emails": [ {...}, {...} ],      # 可选
            "context": "...",                # 可选
            "plan": [ {...}, {...} ],        # 可选，跳过 LLM 计划生成
            "parallel": true                 # 可选，开启并行
        }
        """
        goal = input_data.get("goal", "")
        emails = input_data.get("emails", [])
        context = input_data.get("context", "")
        use_parallel = input_data.get("parallel", False)

        if not goal and not emails:
            return AgentResult(success=False, error="缺少 goal 或 emails 输入")

        # 1. 生成执行计划
        steps = input_data.get("plan")
        if steps is None:
            steps = self.plan(goal, context)

        # 2. 如果有邮件列表，为每封邮件执行流水线
        if emails:
            all_results = []
            for i, email in enumerate(emails):
                logger.info(f"编排器处理邮件 {i+1}/{len(emails)}")
                email_result = self._process_single_email(email, steps, use_parallel)
                all_results.append(email_result)
            return AgentResult(
                success=True,
                data={"processed_count": len(all_results), "results": all_results},
                metadata={"goal": goal, "steps": steps, "parallel": use_parallel},
            )

        # 3. 否则处理单个上下文
        single_result = self._process_single_email(input_data, steps, use_parallel)
        return AgentResult(
            success=True,
            data=single_result,
            metadata={"goal": goal, "steps": steps, "parallel": use_parallel},
        )

    def _process_single_email(
        self,
        email_data: Dict[str, Any],
        steps: List[Dict[str, Any]],
        parallel: bool = False,
    ) -> Dict[str, Any]:
        """
        为单封邮件执行编排流水线

        智能路由：
        - 无依赖的步骤可以并行执行
        - 有依赖的步骤串行，且可以从 upstream 结果中读取数据
        """
        results: Dict[str, Any] = {}
        step_results: Dict[int, AgentResult] = {}

        # 分离出独立步骤和有依赖步骤
        independent = [s for s in steps if not s.get("depends_on")]
        dependent = [s for s in steps if s.get("depends_on")]

        # 第一步：并行执行所有独立步骤
        if parallel and len(independent) > 1:
            task_map = {s["agent"]: email_data for s in independent}
            parallel_results = self._execute_parallel(task_map)
            for s in independent:
                result = parallel_results.get(s["agent"])
                if result:
                    step_idx = steps.index(s)
                    step_results[step_idx] = result
                    results[s["agent"]] = result.data
        else:
            for s in independent:
                result = self._route_to_agent(s["agent"], email_data)
                step_idx = steps.index(s)
                step_results[step_idx] = result
                results[s["agent"]] = result.data

        # 第二步：串行执行有依赖的步骤
        for s in dependent:
            # 合并上游结果到输入数据
            enriched_input = dict(email_data)
            for dep_idx in s.get("depends_on", []):
                dep_result = step_results.get(dep_idx)
                if dep_result and dep_result.success:
                    # 将上游 Agent 结果注入当前输入
                    dep_agent = steps[dep_idx]["agent"]
                    enriched_input[f"upstream_{dep_agent}"] = dep_result.data

            result = self._route_to_agent(s["agent"], enriched_input)
            step_idx = steps.index(s)
            step_results[step_idx] = result
            results[s["agent"]] = result.data

        # 检查是否需要自动起草回复（高优先级触发）
        classifier_data = results.get("ClassifierAgent")
        if isinstance(classifier_data, dict) and classifier_data.get("priority") == "High":
            if "ReplyAgent" not in results:
                logger.info("检测到高优先级邮件，自动起草回复...")
                reply_input = dict(email_data)
                reply_input["requirements"] = {
                    "instruction": "该邮件被标记为高优先级，请起草一封紧急回复",
                    "tone": "专业礼貌",
                    "length": "简短",
                    "language": "中文",
                }
                reply_result = self._route_to_agent("ReplyAgent", reply_input)
                results["ReplyAgent"] = reply_result.data

        return results

    # ── 便捷方法 ────────────────────────────────────────────

    def process_inbox(
        self,
        emails: List[Dict[str, Any]],
        parallel: bool = True,
    ) -> AgentResult:
        """
        处理收件箱邮件（标准流程：总结 + 分类 + 提取任务）

        这是一个预设流水线，跳过 LLM 计划生成步骤。
        """
        return self.run({
            "goal": "处理收件箱的新邮件：总结内容、分类优先级、提取待办事项",
            "emails": emails,
            "plan": [
                {"agent": "SummarizerAgent", "action": "总结邮件", "depends_on": []},
                {"agent": "ClassifierAgent", "action": "分类优先级", "depends_on": []},
                {"agent": "TaskExtractorAgent", "action": "提取任务", "depends_on": []},
            ],
            "parallel": parallel,
        })

    def draft_reply(
        self,
        email_data: Dict[str, Any],
        requirements: Optional[Dict[str, str]] = None,
    ) -> AgentResult:
        """
        为单封邮件起草回复

        先总结原邮件以获取上下文，再起草回复。
        """
        return self.run({
            "goal": "总结邮件并起草回复",
            "plan": [
                {"agent": "SummarizerAgent", "action": "总结邮件", "depends_on": []},
                {"agent": "ReplyAgent", "action": "起草回复", "depends_on": [0]},
            ],
            **email_data,
            "requirements": requirements or {
                "instruction": "根据邮件内容起草回复",
                "tone": "专业礼貌",
                "length": "中等",
                "language": "中文",
            },
        })

    def compress_context(self, raw_text: str) -> AgentResult:
        """压缩上下文为长期记忆"""
        return self.run({
            "goal": "压缩上下文",
            "plan": [
                {"agent": "ContextSummaryAgent", "action": "压缩上下文", "depends_on": []},
            ],
            "context": raw_text,
        })

    # ── 意图路由 ────────────────────────────────────────────

    INTENT_ROUTING_PROMPT = """你是一个意图路由器。分析用户输入，返回单一意图标签。

# 意图标签（严格四选一）
- QUERY: 用户在提问、查询信息、了解情况。例如："有没有紧急邮件"、"张三发了什么"、"最近进展如何"
- STORE: 用户要保存/记住/存储信息。例如："记住这个项目背景"、"把这个存下来"
- REPLY: 用户要回复/起草邮件。例如："帮我回复张三"、"起草一封回信"
- PURE_CHAT: 闲聊、问候、与邮件系统无关的对话。例如："你好"、"今天天气怎么样"

# 输出格式
只输出一个单词：QUERY / STORE / REPLY / PURE_CHAT，不要加任何标点或解释。"""

    def predict(self, user_input: str, history: list = None) -> str:
        """
        快速意图分类，用于 chat_stream 路由。

        Args:
            user_input: 用户输入的自然语言
            history: 对话历史（当前仅用于上下文，不影响路由逻辑）

        Returns:
            str: 意图标签 — "QUERY" | "STORE" | "REPLY" | "PURE_CHAT"
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.INTENT_ROUTING_PROMPT),
            ("user", "{input}"),
        ])
        chain = prompt | self.llm | StrOutputParser()
        try:
            result = chain.invoke({"input": user_input})
            intent = result.strip().upper()
            # 白名单校验：如果 LLM 返回了非法标签，降级为 PURE_CHAT
            if intent not in ("QUERY", "STORE", "REPLY", "PURE_CHAT"):
                logger.warning(f"路由返回非法意图 [{intent}]，降级为 PURE_CHAT")
                return "PURE_CHAT"
            return intent
        except Exception as e:
            logger.warning(f"意图路由 LLM 调用失败: {e}，降级为 PURE_CHAT")
            return "PURE_CHAT"

    def reset(self) -> None:
        """重置编排器和所有子 Agent"""
        super().reset()
        for agent in self._agents.values():
            agent.reset()