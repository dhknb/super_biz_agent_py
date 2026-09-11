"""Repository for conversation and project context persistence."""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.conversation import (
    ConversationMemorySnapshot,
    ConversationMessage,
    ConversationMessageRole,
    ConversationSession,
)


class ConversationRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_or_create_session(
        self,
        session_id: str,
        *,
        project_id: str | None = None,
    ) -> ConversationSession:
        session = self.db.get(ConversationSession, session_id)
        if session is None:
            session = ConversationSession(id=session_id, project_id=project_id)
            self.db.add(session)
        elif project_id and session.project_id != project_id:
            session.project_id = project_id
        session.cleared_at = None
        return session

    def append_message(
        self,
        session_id: str,
        *,
        role: ConversationMessageRole,
        content: str,
        project_id: str | None = None,
        message_metadata: dict | None = None,
    ) -> ConversationMessage:
        self.get_or_create_session(session_id, project_id=project_id)
        message = ConversationMessage(
            session_id=session_id,
            role=role,
            content=content,
            message_metadata=message_metadata,
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message

    def append_exchange(
        self,
        session_id: str,
        *,
        user_content: str,
        assistant_content: str,
        project_id: str | None = None,
        message_metadata: dict | None = None,
    ) -> None:
        self.get_or_create_session(session_id, project_id=project_id)
        self.db.add(
            ConversationMessage(
                session_id=session_id,
                role=ConversationMessageRole.USER,
                content=user_content,
                message_metadata=message_metadata,
            )
        )
        self.db.add(
            ConversationMessage(
                session_id=session_id,
                role=ConversationMessageRole.ASSISTANT,
                content=assistant_content,
                message_metadata=message_metadata,
            )
        )
        self.db.commit()

    def list_session_history(self, session_id: str) -> list[dict]:
        messages = self.list_active_messages(session_id)
        return [
            {
                "role": str(message.role.value),
                "content": message.content,
                "timestamp": message.created_at.isoformat(),
                "message_metadata": message.message_metadata,
            }
            for message in messages
        ]

    def list_active_messages(self, session_id: str) -> list[ConversationMessage]:
        """返回未软删除的原始消息，供会话记忆服务压缩和注入。"""
        stmt = (
            select(ConversationMessage)
            .where(ConversationMessage.session_id == session_id)
            .where(ConversationMessage.deleted_at.is_(None))
            .order_by(ConversationMessage.created_at.asc(), ConversationMessage.id.asc())
        )
        return list(self.db.scalars(stmt).all())

    def get_memory_snapshot(self, session_id: str) -> ConversationMemorySnapshot | None:
        return self.db.get(ConversationMemorySnapshot, session_id)

    def save_memory_snapshot(
        self,
        session_id: str,
        *,
        summary: str,
        summarized_through_message_id: str,
    ) -> ConversationMemorySnapshot:
        """创建或更新滚动摘要，并推进已压缩消息游标。"""
        self.get_or_create_session(session_id)
        snapshot = self.get_memory_snapshot(session_id)
        if snapshot is None:
            snapshot = ConversationMemorySnapshot(
                session_id=session_id,
                summary=summary,
                summarized_through_message_id=summarized_through_message_id,
            )
            self.db.add(snapshot)
        else:
            snapshot.summary = summary
            snapshot.summarized_through_message_id = summarized_through_message_id
            snapshot.version += 1
        self.db.commit()
        self.db.refresh(snapshot)
        return snapshot

    def clear_session(self, session_id: str) -> bool:
        now = datetime.now(UTC).replace(tzinfo=None)
        session = self.db.get(ConversationSession, session_id)
        if session is None:
            session = ConversationSession(id=session_id, cleared_at=now)
            self.db.add(session)
        else:
            session.cleared_at = now

        stmt = (
            select(ConversationMessage)
            .where(ConversationMessage.session_id == session_id)
            .where(ConversationMessage.deleted_at.is_(None))
        )
        for message in self.db.scalars(stmt).all():
            message.deleted_at = now

        snapshot = self.get_memory_snapshot(session_id)
        if snapshot is not None:
            self.db.delete(snapshot)

        self.db.commit()
        return True
