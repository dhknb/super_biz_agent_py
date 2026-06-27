"""create conversation context tables

Revision ID: 20260624_0002
Revises: 20260612_0001
Create Date: 2026-06-24
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260624_0002"
down_revision: Union[str, None] = "20260612_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


message_role = sa.Enum("user", "assistant", "system", name="conversationmessagerole")


def upgrade() -> None:
    op.create_table(
        "context_projects",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "project_context_items",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["context_projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_project_context_items_project_id"),
        "project_context_items",
        ["project_id"],
    )

    op.create_table(
        "conversation_sessions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.String(length=64), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("cleared_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["context_projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_conversation_sessions_project_id"),
        "conversation_sessions",
        ["project_id"],
    )

    op.create_table(
        "conversation_messages",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("role", message_role, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("message_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["conversation_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_conversation_messages_created_at"),
        "conversation_messages",
        ["created_at"],
    )
    op.create_index(
        op.f("ix_conversation_messages_role"),
        "conversation_messages",
        ["role"],
    )
    op.create_index(
        op.f("ix_conversation_messages_session_id"),
        "conversation_messages",
        ["session_id"],
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_conversation_messages_session_id"), table_name="conversation_messages")
    op.drop_index(op.f("ix_conversation_messages_role"), table_name="conversation_messages")
    op.drop_index(op.f("ix_conversation_messages_created_at"), table_name="conversation_messages")
    op.drop_table("conversation_messages")
    op.drop_index(op.f("ix_conversation_sessions_project_id"), table_name="conversation_sessions")
    op.drop_table("conversation_sessions")
    op.drop_index(op.f("ix_project_context_items_project_id"), table_name="project_context_items")
    op.drop_table("project_context_items")
    op.drop_table("context_projects")
    message_role.drop(op.get_bind(), checkfirst=True)
