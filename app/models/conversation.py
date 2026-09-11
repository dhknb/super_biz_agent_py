"""Conversation and project context persistence models."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from sqlalchemy import DateTime, Enum, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _utcnow_naive() -> datetime:
    """Return a naive UTC timestamp for current DB column definitions."""
    return datetime.now(UTC).replace(tzinfo=None)


class ConversationMessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ContextProject(Base):
    __tablename__ = "context_projects"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utcnow_naive,
        onupdate=_utcnow_naive,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    contexts: Mapped[list["ProjectContextItem"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    sessions: Mapped[list["ConversationSession"]] = relationship(back_populates="project")


class ProjectContextItem(Base):
    __tablename__ = "project_context_items"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("context_projects.id"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utcnow_naive,
        onupdate=_utcnow_naive,
    )

    project: Mapped[ContextProject] = relationship(back_populates="contexts")


class ConversationSession(Base):
    __tablename__ = "conversation_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("context_projects.id"),
        nullable=True,
        index=True,
    )
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utcnow_naive,
        onupdate=_utcnow_naive,
    )
    cleared_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    project: Mapped[ContextProject | None] = relationship(back_populates="sessions")
    messages: Mapped[list["ConversationMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    memory_snapshot: Mapped["ConversationMemorySnapshot | None"] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        uselist=False,
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("conversation_sessions.id"),
        nullable=False,
        index=True,
    )
    role: Mapped[ConversationMessageRole] = mapped_column(
        Enum(ConversationMessageRole, values_callable=lambda items: [item.value for item in items]),
        nullable=False,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    message_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_naive, index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    session: Mapped[ConversationSession] = relationship(back_populates="messages")


class ConversationMemorySnapshot(Base):
    """一段会话的压缩记忆快照。

    原始消息始终保留在 ``conversation_messages`` 中；本表只保存供模型注入的
    滚动摘要和已压缩到哪个消息的游标，从而避免每轮把完整历史发送给模型。
    """

    __tablename__ = "conversation_memory_snapshots"

    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("conversation_sessions.id"),
        primary_key=True,
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    summarized_through_message_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    version: Mapped[int] = mapped_column(default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utcnow_naive,
        onupdate=_utcnow_naive,
    )

    session: Mapped[ConversationSession] = relationship(back_populates="memory_snapshot")
