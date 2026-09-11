"""create aiops diagnosis task tables

Revision ID: 20260723_0006
Revises: 20260707_0005
Create Date: 2026-07-23
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260723_0006"
down_revision: Union[str, None] = "20260707_0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


diagnosis_task_status = sa.Enum(
    "new",
    "planning",
    "retrieving",
    "diagnosing",
    "done",
    "failed",
    name="diagnosistaskstatus",
)


def upgrade() -> None:
    op.create_table(
        "aiops_diagnosis_tasks",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("alert_id", sa.String(length=128), nullable=True),
        sa.Column("alert_name", sa.String(length=255), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("status", diagnosis_task_status, nullable=False),
        sa.Column("current_phase", sa.String(length=64), nullable=True),
        sa.Column("alarm_event", sa.JSON(), nullable=False),
        sa.Column("report", sa.JSON(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["conversation_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_aiops_diagnosis_tasks_alert_id"),
        "aiops_diagnosis_tasks",
        ["alert_id"],
    )
    op.create_index(
        op.f("ix_aiops_diagnosis_tasks_session_id"),
        "aiops_diagnosis_tasks",
        ["session_id"],
    )
    op.create_index(
        op.f("ix_aiops_diagnosis_tasks_status"),
        "aiops_diagnosis_tasks",
        ["status"],
    )
    op.create_index(
        op.f("ix_aiops_diagnosis_tasks_created_at"),
        "aiops_diagnosis_tasks",
        ["created_at"],
    )

    op.create_table(
        "aiops_diagnosis_events",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("phase", sa.String(length=64), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["task_id"], ["aiops_diagnosis_tasks.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_aiops_diagnosis_events_task_id"),
        "aiops_diagnosis_events",
        ["task_id"],
    )
    op.create_index(
        op.f("ix_aiops_diagnosis_events_created_at"),
        "aiops_diagnosis_events",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_aiops_diagnosis_events_created_at"),
        table_name="aiops_diagnosis_events",
    )
    op.drop_index(
        op.f("ix_aiops_diagnosis_events_task_id"),
        table_name="aiops_diagnosis_events",
    )
    op.drop_table("aiops_diagnosis_events")
    op.drop_index(
        op.f("ix_aiops_diagnosis_tasks_created_at"),
        table_name="aiops_diagnosis_tasks",
    )
    op.drop_index(
        op.f("ix_aiops_diagnosis_tasks_status"),
        table_name="aiops_diagnosis_tasks",
    )
    op.drop_index(
        op.f("ix_aiops_diagnosis_tasks_session_id"),
        table_name="aiops_diagnosis_tasks",
    )
    op.drop_index(
        op.f("ix_aiops_diagnosis_tasks_alert_id"),
        table_name="aiops_diagnosis_tasks",
    )
    op.drop_table("aiops_diagnosis_tasks")
    diagnosis_task_status.drop(op.get_bind(), checkfirst=True)
