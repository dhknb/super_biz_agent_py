"""create chat_run_spans

Revision ID: 20260824_0009
Revises: 20260823_0008
Create Date: 2026-08-24

为什么要这张表：
`chat_run_traces` 记的是终态（最后的答案、用过哪些文档、质检分数），
回答不了排障时最常问的那个问题 ——「慢在哪一步」。
一次 chat_v2 请求串行调三次 LLM，中间并行跑 N 条检索；
用户报「等了 40 秒」，现有 trace 只能确认「确实跑了 40 秒」。

有了 span，一句 SQL 就能定位：
    SELECT node, count(*), avg(duration_ms), max(duration_ms)
    FROM chat_run_spans WHERE trace_id = ? GROUP BY node;

为什么不上 OpenTelemetry：
完整 OTel 要引 SDK、起 Collector、部署 Jaeger/Tempo，换来的是跨服务追踪。
当前需求是单进程内的节点耗时，四个字段就够（YAGNI）。
借它的概念，不引它的实现。

关于索引：
- trace_id：最主要的查询入口（看某次请求的完整时间线），且是外键
- request_id：冗余一份，让「日志里捞到一个 id」能直接查 span，不必先查 trace
- node：按节点做聚合统计（哪个节点平均最慢）
- status：只查失败/降级的 span
- created_at：按时间窗口做趋势统计

关于 ondelete="CASCADE"：
span 是 trace 的从属明细，trace 删了 span 就是孤儿数据。
交给数据库级联，而不是指望应用层记得先删 span —— 应用层总会有人忘。
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "20260824_0009"
down_revision: Union[str, None] = "20260823_0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chat_run_spans",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("node", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.Enum("ok", "degraded", "error", name="spanstatus"),
            nullable=False,
            server_default="ok",
        ),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["trace_id"],
            ["chat_run_traces.id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_chat_run_spans_trace_id", "chat_run_spans", ["trace_id"])
    op.create_index("ix_chat_run_spans_request_id", "chat_run_spans", ["request_id"])
    op.create_index("ix_chat_run_spans_node", "chat_run_spans", ["node"])
    op.create_index("ix_chat_run_spans_status", "chat_run_spans", ["status"])
    op.create_index("ix_chat_run_spans_created_at", "chat_run_spans", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_chat_run_spans_created_at", table_name="chat_run_spans")
    op.drop_index("ix_chat_run_spans_status", table_name="chat_run_spans")
    op.drop_index("ix_chat_run_spans_node", table_name="chat_run_spans")
    op.drop_index("ix_chat_run_spans_request_id", table_name="chat_run_spans")
    op.drop_index("ix_chat_run_spans_trace_id", table_name="chat_run_spans")
    op.drop_table("chat_run_spans")
    # PostgreSQL 的 Enum 是独立的类型对象，drop_table 不会带走它。
    # 不显式删除的话，下一次 upgrade 会撞上「type spanstatus already exists」。
    sa.Enum(name="spanstatus").drop(op.get_bind(), checkfirst=True)
