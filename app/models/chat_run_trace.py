"""Persistent run trace models for chat_v2 execution."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class ChatRunTraceStatus(StrEnum):
    SUCCEEDED = "succeeded"
    ERROR = "error"


class ChatRunTrace(Base):
    __tablename__ = "chat_run_traces"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    # request_id：把这条 trace 和「请求执行期间的日志」缝在一起。
    # 为什么不能只靠 id：id 是落库时才生成的，请求跑了 30 秒才写库，
    # 这 30 秒里的所有日志都没有任何字段能关联到这条记录。
    # nullable=True：历史数据没有这个值，且中间件之外的调用路径（脚本、测试）也允许为空。
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("conversation_sessions.id"),
        nullable=False,
        index=True,
    )
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    sub_queries: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    retrieved_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    used_documents: Mapped[list[dict]] = mapped_column(JSON, nullable=False, default=list)
    validation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    coverage_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    groundedness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_bad_case: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    status: Mapped[ChatRunTraceStatus] = mapped_column(
        Enum(ChatRunTraceStatus, values_callable=lambda items: [item.value for item in items]),
        nullable=False,
        default=ChatRunTraceStatus.SUCCEEDED,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_naive, index=True)

    session = relationship("ConversationSession")
