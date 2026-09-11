"""create chat run traces table

Revision ID: 20260629_0004
Revises: 20260624_0003
Create Date: 2026-06-29
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260629_0004"
down_revision: Union[str, None] = "20260624_0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


chat_run_trace_status = sa.Enum("succeeded", "error", name="chattracestatus")


def upgrade() -> None:
    op.create_table(
        "chat_run_traces",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("sub_queries", sa.JSON(), nullable=False),
        sa.Column("retrieved_count", sa.Integer(), nullable=False),
        sa.Column("used_documents", sa.JSON(), nullable=False),
        sa.Column("validation", sa.JSON(), nullable=True),
        sa.Column("coverage_score", sa.Float(), nullable=True),
        sa.Column("groundedness_score", sa.Float(), nullable=True),
        sa.Column("is_bad_case", sa.Boolean(), nullable=False),
        sa.Column("status", chat_run_trace_status, nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["conversation_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_chat_run_traces_session_id"), "chat_run_traces", ["session_id"])
    op.create_index(op.f("ix_chat_run_traces_source"), "chat_run_traces", ["source"])
    op.create_index(op.f("ix_chat_run_traces_is_bad_case"), "chat_run_traces", ["is_bad_case"])
    op.create_index(op.f("ix_chat_run_traces_status"), "chat_run_traces", ["status"])
    op.create_index(op.f("ix_chat_run_traces_created_at"), "chat_run_traces", ["created_at"])


def downgrade() -> None:
    op.drop_index(op.f("ix_chat_run_traces_created_at"), table_name="chat_run_traces")
    op.drop_index(op.f("ix_chat_run_traces_status"), table_name="chat_run_traces")
    op.drop_index(op.f("ix_chat_run_traces_is_bad_case"), table_name="chat_run_traces")
    op.drop_index(op.f("ix_chat_run_traces_source"), table_name="chat_run_traces")
    op.drop_index(op.f("ix_chat_run_traces_session_id"), table_name="chat_run_traces")
    op.drop_table("chat_run_traces")
    chat_run_trace_status.drop(op.get_bind(), checkfirst=True)
