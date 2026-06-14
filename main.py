"""
SmartEmailAgent — AI 邮件助手 API 服务

基于 FastAPI 的 RESTful API 服务，对外暴露 Gmail 邮件智能处理能力。
路由定义在 rag_modules/api.py，配置集中在 config.py。

启动:
    python main.py

配置方式（优先级：环境变量 > config.py 默认值）:
    $env:SMART_EMAIL_HOST = "127.0.0.1"   # 仅本地访问
    $env:SMART_EMAIL_PORT = "8080"        # 自定义端口
    $env:SMART_EMAIL_LLM_API_KEY = "sk-xxx"
    python main.py

API 文档:
    启动后访问 http://localhost:8000/docs 查看 Swagger UI
"""

import logging
import os

# ── 内网代理清理 ────────────────────────────────────────────
# 开发环境常设系统代理；调用 Gmail / LLM API 时代理会导致连接失败，启动前强制清除
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# ── 日志配置 ─────────────────────────────────────────────────
# 统一格式：时间 + 模块名 + 级别 + 消息，便于多模块日志溯源
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("main")  # 当前模块专属 logger

# ── FastAPI 应用初始化 ──────────────────────────────────────
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_modules.api import router   # 路由模块：接口定义与业务逻辑分离
from config import SmartEmailAgentConfig as Config  # 统一配置：环境变量自动装配







@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理（替代已废弃的 on_event）。
    yield 前：应用启动时执行（可在此初始化数据库连接、预加载模型等）。
    yield 后：应用关闭时执行（释放资源）。
    """
    logger.info("SmartEmailAgent API 服务启动")
    yield
    logger.info("SmartEmailAgent API 服务关闭")


# 创建 FastAPI 应用实例，注入元信息供 Swagger UI 展示
app = FastAPI(
    title="SmartEmailAgent",
    description="AI 驱动的 Gmail 智能邮件助手 — RESTful API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS 中间件 ─────────────────────────────────────────────
# 开发阶段放开所有来源；生产环境应将 allow_origins 限定为前端域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 允许的前端来源，* 表示不限制
    allow_credentials=True,    # 允许携带 Cookie / Authorization 头
    allow_methods=["*"],       # 允许所有 HTTP 方法
    allow_headers=["*"],       # 允许所有请求头
)

# ── 注册路由 ─────────────────────────────────────────────────
# 将 rag_modules/api.py 中定义的所有路由挂载到主应用
app.include_router(router)

# ── 启动入口 ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    host = Config.api_host   # 绑定地址，默认 0.0.0.0（可通过环境变量覆盖）
    port = Config.api_port   # 监听端口，默认 8000

    print(f"🚀 SmartEmailAgent API: http://{host}:{port}")
    print(f"   Swagger 文档: http://{host}:{port}/docs")

    # reload=True：代码变更后自动重启，仅用于开发环境
    # 生产部署应去掉 reload=True，改用 gunicorn 或 supervisor 管理进程
    uvicorn.run("main:app", host=host, port=port, reload=True)