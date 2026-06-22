"""pytest 共享 fixture

所有 fixture 都基于 mock，绝不访问外部服务（LLM / Milvus / MCP）。

关键：pytest_configure 钩子在测试收集前运行，抢在模块级单例
（vector_store_manager / milvus_manager）尝试连接 Milvus 前就把它们 mock 掉。
"""

import os
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

    # ── mock RAG agent 中的 ChatQwen 和 MemorySaver ──
    p3 = patch("app.services.rag_agent_service.MemorySaver", MagicMock())
    p3.start()
    _SESSION_PATCHERS.append(p3)

    p4 = patch("app.services.rag_agent_service.ChatQwen", MagicMock())
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
