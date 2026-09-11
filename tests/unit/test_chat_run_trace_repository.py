"""Tests for chat run trace persistence."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models import chat_run_trace  # noqa: F401
from app.models import conversation  # noqa: F401
from app.repositories.chat_run_trace_repository import ChatRunTraceRepository
from app.repositories.conversation_repository import ConversationRepository


def test_create_trace_marks_blocked_answer_as_bad_case() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

    with SessionLocal() as db:
        ConversationRepository(db).get_or_create_session("session-1")
        repo = ChatRunTraceRepository(db)
        trace = repo.create_trace(
            session_id="session-1",
            source="chat_v2",
            question="redis没响应怎么办",
            answer="根据已有资料无法回答该问题。",
            sub_queries=["redis 超时", "redis 排查步骤"],
            retrieved_count=2,
            used_documents=[],
            validation={
                "blocked": True,
                "coverage_score": 0.3,
                "groundedness_score": 0.4,
                "thresholds": {"coverage_min": 0.6, "groundedness_min": 0.75},
            },
        )

        assert trace.is_bad_case is True
        assert trace.coverage_score == 0.3
        assert trace.groundedness_score == 0.4

    engine.dispose()
