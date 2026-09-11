"""add request_id to chat_run_traces and aiops_diagnosis_tasks

Revision ID: 20260823_0008
Revises: 20260823_0007
Create Date: 2026-08-23

为什么要加这两列：
落库的 trace 只有自己的主键 id，而 id 是**请求跑完才生成**的。
一个请求执行 30 秒，这 30 秒里的日志没有任何字段能关联回这条记录。
加上 request_id 之后，「用户报障给的 id」→「日志」→「trace 记录」三者可互查。

为什么建索引：
这两列的唯一用途就是按 request_id 精确查一条/一组记录（排障场景）。
不建索引就是全表扫描，表一大排障就变慢 —— 而排障恰恰是最不能慢的场景。

关于线上执行：
两列都是 nullable，加列本身不重写表数据（PostgreSQL 11+ 的 ADD COLUMN NULL 是 O(1)）。
CREATE INDEX 会短暂持有写锁；表体量大的话可改用
`op.create_index(..., postgresql_concurrently=True)` 并把迁移设为非事务模式。
当前数据量下用普通建索引即可（YAGNI）。
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "20260823_0008"
down_revision: Union[str, None] = "20260823_0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chat_run_traces",
        sa.Column("request_id", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_chat_run_traces_request_id",
        "chat_run_traces",
        ["request_id"],
    )

    op.add_column(
        "aiops_diagnosis_tasks",
        sa.Column("request_id", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_aiops_diagnosis_tasks_request_id",
        "aiops_diagnosis_tasks",
        ["request_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_aiops_diagnosis_tasks_request_id", table_name="aiops_diagnosis_tasks")
    op.drop_column("aiops_diagnosis_tasks", "request_id")

    op.drop_index("ix_chat_run_traces_request_id", table_name="chat_run_traces")
    op.drop_column("chat_run_traces", "request_id")
