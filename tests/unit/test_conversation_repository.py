"""Tests for conversation persistence repository."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models import conversation  # noqa: F401
from app.repositories.conversation_repository import ConversationRepository


def test_append_list_and_clear_session_history() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

    with SessionLocal() as db:
        repo = ConversationRepository(db)
        repo.append_exchange(
            "session-1",
            user_content="hello",
            assistant_content="hi there",
        )

        history = repo.list_session_history("session-1")
        assert [item["role"] for item in history] == ["user", "assistant"]
        assert [item["content"] for item in history] == ["hello", "hi there"]
        assert all(item["timestamp"] for item in history)

        assert repo.clear_session("session-1") is True
        assert repo.list_session_history("session-1") == []

    engine.dispose()
