"""SmartEmailAgent — API 路由"""

import logging
from fastapi import APIRouter, HTTPException

from rag_modules.pipeline import get_pipeline
import json
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Request / Response 模型 ──────────────────────────────────
# 使用 Pydantic 定义请求体和响应体，FastAPI 自动完成 JSON 序列化/反序列化
# 以及 Swagger 文档生成

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    统一对话请求

    自动路由到对应 Agent：查询类→RAG问答、处理类→批量处理、
    回复类→起草回信、记忆类→压缩存储
    """
    message: str = Field(
        ...,
        min_length=1,
        description="用户输入的自然语言，可以为问题、指令或待记忆文本",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Milvus 向量搜索返回条数，越大召回越全但 LLM 上下文越长",
    )
    history: list[dict] = Field(
        default_factory=list,
        description="多轮对话历史 [{'question': '...', 'answer': '...'}, ...]，最多保留最近 6 轮",
    )


class QueryRequest(BaseModel):
    """
    直接 RAG 问答请求（不走自动路由）

    适合明确已知要检索邮件库的场景，性能优于 /chat 因为跳过了 LLM 规划步骤
    """
    question: str = Field(
        ...,
        min_length=1,
        description="自然语言查询问题",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="向量搜索返回条数",
    )
    scope: str = Field(
        default="all",
        description="搜索范围：all=全部, inbox=仅收件箱, context=仅长期记忆",
    )
    history: list[dict] = Field(
        default_factory=list,
        description="多轮对话历史",
    )


class SyncRequest(BaseModel):
    """
    同步收件箱请求

    拉取最新邮件 → AI 摘要 → 优先级分类 → 向量化 → 存入 Milvus
    """
    max_emails: int = Field(
        default=20,
        ge=1,
        le=100,
        description="从 Gmail 拉取的最大邮件数量，受 Google API 配额限制",
    )


class ContextRequest(BaseModel):
    """
    存储长期记忆请求

    文本经 LLM 压缩后存入 Milvus context_historical 集合，
    后续问答时自动召回相关记忆作为辅助上下文
    """
    content: str = Field(
        ...,
        min_length=1,
        description="待存储的知识/背景文本，如项目说明、客户信息、团队约定等",
    )


class AnalyzeSentRequest(BaseModel):
    """
    分析发件箱语气请求

    拉取已发送邮件 → AI 摘要 → 向量化存储，
    用于学习用户回复风格，辅助回信起草
    """
    max_emails: int = Field(
        default=20,
        ge=1,
        le=100,
        description="从发件箱拉取的最大邮件数量",
    )


class ChatResponse(BaseModel):
    """
    对话/问答响应

    适用于 /chat、/query 两个接口的返回格式
    """
    success: bool = Field(
        ...,
        description="请求是否成功处理",
    )
    answer: str = Field(
        default="",
        description="LLM 生成的回答文本（含引用标注）",
    )
    confidence: str = Field(
        default="",
        description="回答置信度：high / medium / low",
    )
    cited_count: int = Field(
        default=0,
        description="回答中引用的邮件/记忆数量",
    )
    sources: dict = Field(
        default_factory=dict,
        description="引用来源详情 {emails: [...], contexts: [...]}，用于前端展示引用卡片",
    )
    error: str = Field(
        default="",
        description="失败时的错误信息，success=false 时此字段非空",
    )


class SyncResponse(BaseModel):
    """
    同步操作响应

    适用于 /sync、/analyze-sent 等批量处理接口
    """
    success: bool = Field(
        ...,
        description="同步是否成功",
    )
    count: int = Field(
        default=0,
        description="实际处理的邮件数量",
    )
    error: str = Field(
        default="",
        description="失败时的错误信息",
    )


class ContactsResponse(BaseModel):
    """
    高频联系人统计响应

    从收件箱/发件箱提取 Top-N 联系人及其出现次数
    """
    success: bool = Field(
        ...,
        description="查询是否成功",
    )
    top_senders: list = Field(
        default_factory=list,
        description="高频发件人 [{'address': 'xxx@mail.com', 'count': 15}, ...]",
    )
    top_receivers: list = Field(
        default_factory=list,
        description="高频收件人 [{'address': 'xxx@mail.com', 'count': 20}, ...]",
    )


# ── 路由定义 ─────────────────────────────────────────────────

from fastapi.responses import StreamingResponse


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    统一对话与智能体路由接口 (Agent Hub)

    这是系统的主聊框入口。接收用户的自然语言输入，通过底层 LLM 动态判断用户意图：
    - 如果是提问 -> 自动检索 Milvus 邮件库并回答 (RAG)
    - 如果是指令 -> 自动路由至写信、摘要、记忆存储等对应 Agent 执行

    Args:
        req (ChatRequest): 包含用户输入消息、检索条数(top_k)及多轮历史记录的请求体。

    Returns:
        ChatResponse: 包含 AI 回答、置信度以及引用来源(sources)的统一响应体。

    Raises:
        HTTPException: 当底层 Pipeline 抛出未捕获异常时，返回 500 系统错误。
    """
    try:
        # 1. 获取业务层管道的单例对象（解耦路由层与核心业务逻辑）
        pipeline = get_pipeline()

        # 2. 将前端请求参数透传给底层逻辑，触发 Agent 意图识别与执行链路
        result = pipeline.chat(
            user_input=req.message,
            top_k=req.top_k,
            history=req.history
        )

        # 3. 业务逻辑层容错：判断业务是否成功执行（如：模型拒绝回答、逻辑前置校验未通过等）
        if not result.success:
            # 返回标准的失败结构，将底层抛出的错误信息（或兜底文字）塞入 error 字段
            return ChatResponse(success=False, error=result.error or "未知错误")

        # 4. 业务执行成功，提取核心返回数据（若 data 为 None 则防御性赋予空字典）
        data = result.data or {}

        # 5. 组装并返回标准的 ChatResponse 模型，FastAPI 会严格按照此结构进行 JSON 序列化
        return ChatResponse(
            success=True,
            answer=data.get("answer", ""),  # AI 生成的最终文本回答
            confidence=data.get("confidence", ""),  # 回答置信度评估 (high/medium/low)
            cited_count=data.get("cited_count", 0),  # 本次回答援引的历史邮件/知识数量
            sources=result.metadata.get("sources", {}),  # 详细的引用来源详情（包含邮件 ID、高亮片段等，供前端渲染卡片）
        )

    except Exception as e:
        # 6. 系统级崩溃兜底（如数据库断连、网络超时等硬报错）
        # logger.exception 会自动捕获并记录完整的错误堆栈信息，是半夜排查 Bug 的核心依据
        logger.exception("chat 接口内部发生未知异常")

        # 向前端抛出标准的 HTTP 500 状态码，避免长连接挂起，同时把具体错误暴露给调用方
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream", tags=["Agent"])
async def chat_stream(req: ChatRequest):
    """
    统一对话与智能体路由接口 — 流式响应 (SSE)

    接收用户的自然语言输入，实时单字/单 Token 推送 AI 的思考与回答。
    适合主聊天框的“打字机”流式交互。

    数据传输约定：
    - 普通文本：以 `data: {"text": "某个字"}` 形式高频推送
    - 最终元数据（单次对话结束）：以 `data: {"status": "done", "sources": {...}, ...}` 形式推送
    - 异常错误：以 `data: {"error": "错误信息"}` 形式推送
    """

    def generate():
        # 1. 获取业务层管道的单例对象
        pipeline = get_pipeline()

        try:
            # 2. 调用底层的流式聊天生成器
            # 预期底层 pipeline.chat_stream 每次 yield 的是一个 chunk（字典或对象）
            for chunk in pipeline.chat_stream(
                    user_input=req.message,
                    top_k=req.top_k,
                    history=req.history
            ):
                # 💡 情况 A：底层抛出了业务层错误
                if hasattr(chunk, "success") and not chunk.success:
                    err_msg = chunk.error or "未知内部错误"
                    yield f"data: {json.dumps({'error': err_msg}, ensure_ascii=False)}\n\n"
                    return

                # 💡 情况 B：如果是最终的结算数据（包含 sources, confidence 等）
                # 假设底层在结束时会返回一个带有特定标记或包含完整 data 的结构
                if isinstance(chunk, dict) and "is_final" in chunk:
                    final_payload = {
                        "status": "done",
                        "confidence": chunk.get("confidence", ""),
                        "cited_count": chunk.get("cited_count", 0),
                        "sources": chunk.get("sources", {})
                    }
                    yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
                    return

                # 💡 情况 C：正常的文本 Token 流
                # 如果 chunk 直接是字符串，或者是一个包含文字的字典
                text_token = chunk if isinstance(chunk, str) else chunk.get("text", "")
                if text_token:
                    yield f"data: {json.dumps({'text': text_token}, ensure_ascii=False)}\n\n"

            # 3. 正常流结束，发送标准 done 事件
            yield f"data: {json.dumps({'status': 'done'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            # 4. 运行期间硬报错捕获
            logger.exception("chat/stream 内部迭代发生异常")
            err_payload = {"error": f"服务器内部错误: {str(e)}"}
            yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"

    # 5. 返回流式响应，配置标准的免缓存、长连接请求头
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 极其关键：防止 Nginx 等代理服务器缓存流式数据
        },
    )




@router.post("/query", response_model=ChatResponse)
def query(req: QueryRequest):
    try:
        pipeline = get_pipeline()
        result = pipeline.query(question=req.question, top_k=req.top_k, scope=req.scope, history=req.history)
        if not result.success:
            return ChatResponse(success=False, error=result.error or "未知错误")
        data = result.data or {}
        return ChatResponse(
            success=True,
            answer=data.get("answer", ""),
            confidence=data.get("confidence", ""),
            cited_count=data.get("cited_count", 0),
            sources=result.metadata.get("sources", {}),
        )
    except Exception as e:
        logger.exception("query 异常")
        raise HTTPException(status_code=500, detail=str(e))

# ── 流式端点（SSE） ─────────────────────────────────────────

@router.post("/query/stream")
async def query_stream(req: QueryRequest):
    """
    流式 RAG 问答 — Server-Sent Events

    每产生一个 token 就推送给客户端，适合前端"打字机效果"展示。
    """
    def generate():
        pipeline = get_pipeline()
        try:
            for chunk in pipeline.query_stream(
                question=req.question,
                top_k=req.top_k,
                scope=req.scope,
                history=req.history,
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception("query/stream 异常")
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
