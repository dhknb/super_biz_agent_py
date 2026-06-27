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
