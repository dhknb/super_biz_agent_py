"""FastAPI 应用入口

主应用程序，配置路由、中间件、静态文件等
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from contextlib import asynccontextmanager
import os

from prometheus_fastapi_instrumentator import Instrumentator

from app.config import config
from loguru import logger
from app.api import aiops, chat, chat_v2, file, health, protocol_pdf
from app.core.exception_handlers import register_exception_handlers
from app.core.metrics import render_metrics
from app.core.middleware import RequestIdMiddleware
from app.core.milvus_client import milvus_manager
from app.core.request_context import REQUEST_ID_HEADER


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("=" * 60)
    logger.info(f"🚀 {config.app_name} v{config.app_version} 启动中...")
    logger.info(f"📝 环境: {'开发' if config.debug else '生产'}")
    logger.info(f"🌐 监听地址: http://{config.host}:{config.port}")
    logger.info(f"📚 API 文档: http://{config.host}:{config.port}/docs")
    
    # 连接 Milvus
    logger.info("🔌 正在连接 Milvus...")
    milvus_manager.connect()
    logger.info("✅ Milvus 连接成功")
    
    logger.info("=" * 60)
    
    yield
    
    # 关闭时执行
    logger.info("🔌 正在关闭 Milvus 连接...")
    milvus_manager.close()
    logger.info(f"👋 {config.app_name} 关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description="基于 LangChain 的智能oncall运维系统",
    lifespan=lifespan
)

# 配置 CORS：从配置读取允许的来源，避免生产环境裸奔
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # 让浏览器里的 JS 能读到 X-Request-ID：CORS 默认只暴露 6 个安全响应头，
    # 不显式 expose 的话前端 `response.headers.get("X-Request-ID")` 拿到 null，
    # 用户报障时就给不出这个 id，链路追踪的最后一公里断在浏览器里。
    expose_headers=[REQUEST_ID_HEADER],
)

# request_id 中间件放在 CORS 之后添加。
# Starlette 的 add_middleware 是 insert(0)，构建时再 reversed —— 也就是
# **最后添加的位于最外层、最先执行**。放最外层的理由：
#   1. ContextVar 要在任何业务代码（含异常处理器）之前设好，否则那些日志拿不到 rid。
#   2. 响应头由它最后回写，连 CORS 预检和错误响应都会带上 X-Request-ID。
app.add_middleware(RequestIdMiddleware)

# 注册全局异常处理器。
# 在包含路由之前还是之后调用都可以（异常处理器是应用级的，与路由注册顺序无关），
# 放这里是为了让「中间件 → 异常处理 → 路由」的阅读顺序与请求实际经过的层次一致。
register_exception_handlers(app)

# 注册路由
app.include_router(health.router, tags=["健康检查"])
app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(chat_v2.router, prefix="/api", tags=["对话v2-多查询RAG"])
app.include_router(file.router, prefix="/api", tags=["文件管理"])
app.include_router(protocol_pdf.router, prefix="/api", tags=["协议PDF入库"])
app.include_router(aiops.router, prefix="/api", tags=["AIOps智能运维"])

# ── 可观测性：HTTP 指标 + /metrics ────────────────────────────
#
# instrument(app) 装的是 HTTP 层指标（请求数、耗时、请求/响应体大小）。
# 它必须在**路由注册之后**调用 —— Instrumentator 要遍历 app.routes
# 才知道有哪些路由模板，在之前调用会漏掉全部业务接口。
#
# 关键收益是 label 按**路由模板**聚合：
#   http_requests_total{handler="/api/aiops/trace/{trace_id}", status="2xx"}
# handler 是 `{trace_id}` 而不是真实 id。自己写中间件拿到的是
# request.url.path（已展开），每个 trace_id 一条时间序列，
# 几万次查询就把 Prometheus 的内存打爆。这是引这个库的**唯一实质理由**
# （对比熔断器：那六十行没有这种坑，所以没引 pybreaker）。
#
# 这一行也是第三批的收尾：exception_handlers.py 的文档里写着
# 「Prometheus 的 http_requests_total{status="5xx"}」是状态码的消费者之一，
# 而在此之前那个消费者并不存在 —— 状态码说了真话，但没有任何东西在听。
Instrumentator().instrument(app)


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus 抓取端点。

    为什么不用 Instrumentator 自带的 `.expose(app)`：
    我们需要在每次抓取前刷一遍熔断器状态（见 app/core/metrics.py 的
    refresh_circuit_breaker_metrics —— OPEN→HALF_OPEN 是惰性转换，
    不刷就会永远显示 OPEN）。`.expose()` 直接吐 registry，没有这个钩子。

    ⚠️ 安全：这个端点**无鉴权**，吐出的是内部形状（有哪些节点、
    耗时分布、降级率、熔断器状态、请求量）。对侦察者有用 ——
    比如从 degrade_total 的跳变推断我们哪个依赖正在挂。
    **生产环境必须在网关/ACL 层限制来源**，只让内网的 Prometheus 可达。
    不在应用里加 token 是因为那会让抓取端配置复杂化（密码要发给
    Prometheus 并跟着轮转），而来源限制在网关层做既标准又彻底。
    config.enable_metrics_endpoint 是最后一道保险：出事了不用发版就能关。

    include_in_schema=False：不进 OpenAPI 文档。它不是给人调的业务接口，
    列在 /docs 里只会让读文档的人以为这是一个可以对外提供的能力。
    """
    if not config.enable_metrics_endpoint:
        # 404 而不是 403：关掉之后这个端点在语义上就**不存在**。
        # 403 会告诉扫描者「这里有个 /metrics，只是你没权限」，
        # 反而确认了目标的存在。
        return Response(status_code=404)

    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)

# 挂载静态文件
static_dir = "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    """返回首页"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": f"Welcome to {config.app_name} API",
        "version": config.app_version,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    )
