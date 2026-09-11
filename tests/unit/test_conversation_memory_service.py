"""会话记忆：滚动摘要、预算与持久化游标测试。"""

import asyncio
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models import conversation  # noqa: F401
from app.repositories.conversation_repository import ConversationRepository
from app.services.conversation_memory_service import (
    ConversationMemoryService,
    estimate_tokens,
)


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def test_memory_uses_recent_messages_without_summary_when_under_threshold() -> None:
    engine, SessionLocal = _session()
    with SessionLocal() as db:
        repo = ConversationRepository(db)
        repo.append_exchange("s1", user_content="服务叫 order-api", assistant_content="已记录")
        summarizer = AsyncMock()
        service = ConversationMemoryService(
            summarizer=summarizer,
            context_token_budget=200,
            compact_threshold_tokens=500,
            recent_turns=3,
        )

        context = asyncio.run(service.build_context(db, "s1"))

        assert "服务叫 order-api" in context.text
        assert context.used_summary is False
        summarizer.ainvoke.assert_not_awaited()
    engine.dispose()


def test_memory_compacts_old_messages_and_persists_snapshot() -> None:
    engine, SessionLocal = _session()
    with SessionLocal() as db:
        repo = ConversationRepository(db)
        for index in range(4):
            repo.append_exchange(
                "s1",
                user_content=f"第 {index} 轮用户说明：服务 order-api 的重要背景信息。",
                assistant_content=f"第 {index} 轮助手回答：继续处理 order-api。",
            )

        summarizer = AsyncMock()
        summarizer.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="【明确事实】\n- 服务是 order-api\n【任务进展】\n- 已分析 CPU\n"
                "【约束与偏好】\n- 中文\n【待确认项】\n- 实时指标"
            )
        )
        service = ConversationMemoryService(
            summarizer=summarizer,
            context_token_budget=180,
            summary_token_budget=80,
            compact_threshold_tokens=20,
            recent_turns=1,
        )

        context = asyncio.run(service.build_context(db, "s1"))
        snapshot = repo.get_memory_snapshot("s1")

        assert snapshot is not None
        assert snapshot.summarized_through_message_id
        assert "【会话滚动摘要】" in context.text
        assert "order-api" in context.text
        assert context.estimated_tokens <= 180
        summarizer.ainvoke.assert_awaited_once()
    engine.dispose()


def test_existing_summary_is_reused_without_repeated_compaction() -> None:
    engine, SessionLocal = _session()
    with SessionLocal() as db:
        repo = ConversationRepository(db)
        repo.append_exchange("s1", user_content="旧消息一", assistant_content="旧回答一")
        messages = repo.list_active_messages("s1")
        repo.save_memory_snapshot(
            "s1",
            summary="【明确事实】\n- 服务是 redis\n【任务进展】\n- 无\n"
            "【约束与偏好】\n- 无\n【待确认项】\n- 无",
            summarized_through_message_id=messages[-1].id,
        )
        repo.append_exchange("s1", user_content="新问题", assistant_content="新回答")
        summarizer = AsyncMock()
        service = ConversationMemoryService(
            summarizer=summarizer,
            context_token_budget=200,
            compact_threshold_tokens=500,
            recent_turns=2,
        )

        context = asyncio.run(service.build_context(db, "s1"))

        assert "服务是 redis" in context.text
        assert "新问题" in context.text
        summarizer.ainvoke.assert_not_awaited()
    engine.dispose()


def test_clear_session_also_deletes_memory_snapshot() -> None:
    engine, SessionLocal = _session()
    with SessionLocal() as db:
        repo = ConversationRepository(db)
        repo.append_exchange("s1", user_content="hello", assistant_content="world")
        message = repo.list_active_messages("s1")[-1]
        repo.save_memory_snapshot("s1", summary="摘要", summarized_through_message_id=message.id)

        repo.clear_session("s1")

        assert repo.get_memory_snapshot("s1") is None
        assert repo.list_session_history("s1") == []
    engine.dispose()


def test_token_estimate_is_conservative_for_chinese() -> None:
    assert estimate_tokens("中文测试") == 4
    assert estimate_tokens("hello world") >= 2


def test_rendered_memory_including_headers_never_exceeds_budget() -> None:
    engine, SessionLocal = _session()
    with SessionLocal() as db:
        repo = ConversationRepository(db)
        repo.append_exchange("s1", user_content="中文内容" * 20, assistant_content="回答内容" * 20)
        service = ConversationMemoryService(
            context_token_budget=30,
            compact_threshold_tokens=999,
            recent_turns=3,
        )

        context = asyncio.run(service.build_context(db, "s1"))

        assert context.estimated_tokens <= 30
    engine.dispose()
