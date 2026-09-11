"""pytest 共享 fixture

所有 fixture 都基于 mock，绝不访问外部服务（LLM / Milvus / MCP）。

关键：pytest_configure 钩子在测试收集前运行，抢在模块级单例
（vector_store_manager / milvus_manager）尝试连接 Milvus 前就把它们 mock 掉。
"""

import os

# [最高优先级] 抢在任何 app.* import 之前设置假环境变量，
# 确保真实 API Key 绝不进入测试进程日志（pydantic: env var > .env）。
os.environ["DASHSCOPE_API_KEY"] = "sk-test-fake-key-for-testing-only"
os.environ["MILVUS_HOST"] = "localhost"
os.environ["MILVUS_PORT"] = "19530"
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage


# ──────────────────────────────────────────────────────────────
# 会话级 patch —— 在 pytest 收集测试之前运行
# 解决 app/services/vector_store_manager.py 模块级单例
# 在 import 时就尝试连 Milvus 导致收集失败的问题
# ──────────────────────────────────────────────────────────────
_SESSION_PATCHERS: list = []


def pytest_configure(config: pytest.Config) -> None:
    """测试收集前执行，mock 掉所有会连外部服务的模块级单例"""
    # ── mock MilvusClientManager 单例 ──
    mock_mgr = MagicMock()
    mock_mgr.connect = MagicMock(return_value=MagicMock())
    mock_mgr.close = MagicMock()
    mock_mgr.health_check = MagicMock(return_value=True)
    mock_mgr.get_collection = MagicMock(return_value=MagicMock())

    p1 = patch("app.core.milvus_client.milvus_manager", mock_mgr)
    p1.start()
    _SESSION_PATCHERS.append(p1)

    # ── mock langchain_milvus.Milvus，防止 VectorStoreManager 构造时连接 DB ──
    p_milvus = patch("langchain_milvus.Milvus", MagicMock())
    p_milvus.start()
    _SESSION_PATCHERS.append(p_milvus)

    # ── mock pymilvus 连接层，防止 ORM 式 connect 卡住 ──
    p_pm_conn = patch("app.core.milvus_client.connections", MagicMock())
    p_pm_conn.start()
    _SESSION_PATCHERS.append(p_pm_conn)

    # ── mock RAG agent 中的 MemorySaver ──
    p3 = patch("app.services.rag_agent_service.MemorySaver", MagicMock())
    p3.start()
    _SESSION_PATCHERS.append(p3)

    # ── mock ChatQwen ──
    # 打在源头 langchain_qwq.ChatQwen 上，而不是某个 service 模块的引用上。
    # 原因：ChatQwen 现在统一由 app/core/llm_factory.py 的 create_qwen_model
    # 在函数体内延迟 import（`from langchain_qwq import ChatQwen`），
    # 属性在**调用时**才解析，所以打源头一处即可覆盖所有调用方。
    # 反过来说，打在 `app.services.xxx.ChatQwen` 上是脆弱的 ——
    # 一旦某个 service 不再直接 import 它，patch 就会以
    # AttributeError 的形式在 pytest_configure 阶段炸掉整个测试会话。
    p4 = patch("langchain_qwq.ChatQwen", MagicMock())
    p4.start()
    _SESSION_PATCHERS.append(p4)


def pytest_unconfigure(config: pytest.Config) -> None:
    """测试结束后清理所有 session 级 patch"""
    for p in reversed(_SESSION_PATCHERS):
        p.stop()
    _SESSION_PATCHERS.clear()


# ──────────────────────────────────────────────────────────────
# 隔离环境变量：测试期间不读取开发 .env
# ──────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _isolate_env() -> None:
    """确保测试不通过真实配置泄露到环境"""
    os.environ.pop("DEBUG", None)
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("MILVUS_HOST", "localhost")
    os.environ.setdefault("MILVUS_PORT", "19530")


# ──────────────────────────────────────────────────────────────
# Mock LLM —— 每次调用返回固定的可控回答，避免烧 token
# ──────────────────────────────────────────────────────────────
@pytest.fixture
def mock_llm() -> MagicMock:
    """返回一个 AsyncMock，模拟 ChatOpenAI / ChatQwen 的行为"""
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="mock response"))
    llm.astream = AsyncMock(return_value=AsyncMock())
    return llm


# ──────────────────────────────────────────────────────────────
# Mock MCP 客户端 —— 避免异步初始化连远端
# ──────────────────────────────────────────────────────────────
@pytest.fixture
def mock_mcp_client() -> AsyncMock:
    """返回 Mock MCP 客户端，get_tools() 返回空列表"""
    client = AsyncMock()
    client.get_tools = AsyncMock(return_value=[])
    client.close = AsyncMock()
    return client


# ──────────────────────────────────────────────────────────────
# FastAPI TestClient（依赖已在 pytest_configure 中 mock）
# ──────────────────────────────────────────────────────────────
@pytest.fixture
def client():
    """FastAPI TestClient，milvus / vector_store / MemorySaver 已提前 mock"""
    from fastapi.testclient import TestClient as TC

    from app.core.database import get_db
    from app.main import app

    def override_get_db():
        yield MagicMock()

    app.dependency_overrides[get_db] = override_get_db
    try:
        return TC(app)
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def client_no_raise():
    """和 `client` 相同，但不把服务端异常抛回测试 —— 用于验证错误响应本身。

    为什么需要单独一个 fixture：

    Starlette 的 `ServerErrorMiddleware` 在调用完 `Exception` 处理器之后
    **总是 `raise exc`**（源码注释写的理由是「让服务器能记日志、让测试客户端
    可以选择在用例内抛出」）。而 `TestClient` 默认 `raise_server_exceptions=True`，
    于是 `client.post(...)` 会直接抛 RuntimeError，根本拿不到 response 对象 ——
    想断言状态码是 500 还是 504 就无从下手。

    关掉这个开关后，TestClient 的行为与真实 uvicorn 一致：把处理器生成的响应
    如实返回给调用方。这正是我们要验证的东西。

    保留默认 `client` 仍然抛异常是有意的：其他用例里冒出的意外异常应该
    响亮地失败，而不是被悄悄吞成一个 500 响应。
    """
    from fastapi.testclient import TestClient as TC

    from app.core.database import get_db
    from app.main import app

    def override_get_db():
        yield MagicMock()

    app.dependency_overrides[get_db] = override_get_db
    try:
        return TC(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.pop(get_db, None)
