"""create protocol pdf ingestion tables

Revision ID: 20260624_0003
Revises: 20260624_0002
Create Date: 2026-06-24
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260624_0003"
down_revision: Union[str, None] = "20260624_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


protocol_ingestion_status = sa.Enum(
    "pending",
    "extracting",
    "structuring",
    "validating",
    "awaiting_confirmation",
    "writing",
    "completed",
    "failed",
    "rejected",
    name="protocolingestionstatus",
)
protocol_ingestion_job_status = sa.Enum(
    "queued",
    "running",
    "succeeded",
    "failed",
    name="protocolingestionjobstatus",
)


def upgrade() -> None:
    op.create_table(
        "protocol_pdf_ingestions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("file_path", sa.String(length=1024), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("status", protocol_ingestion_status, nullable=False),
        sa.Column("current_phase", sa.String(length=64), nullable=True),
        sa.Column("extracted_text", sa.Text(), nullable=True),
        sa.Column("structured_data", sa.JSON(), nullable=True),
        sa.Column("validation_result", sa.JSON(), nullable=True),
        sa.Column("dry_run_plan", sa.JSON(), nullable=True),
        sa.Column("state_trace", sa.JSON(), nullable=False),
        sa.Column("confirmed_by", sa.String(length=255), nullable=True),
        sa.Column("confirmed_at", sa.DateTime(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_protocol_pdf_ingestions_status"),
        "protocol_pdf_ingestions",
        ["status"],
    )

    op.create_table(
        "protocol_pdf_ingestion_jobs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("ingestion_id", sa.String(length=64), nullable=False),
        sa.Column("rq_job_id", sa.String(length=255), nullable=True),
        sa.Column("status", protocol_ingestion_job_status, nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["ingestion_id"], ["protocol_pdf_ingestions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_protocol_pdf_ingestion_jobs_ingestion_id"),
        "protocol_pdf_ingestion_jobs",
        ["ingestion_id"],
    )
    op.create_index(
        op.f("ix_protocol_pdf_ingestion_jobs_rq_job_id"),
        "protocol_pdf_ingestion_jobs",
        ["rq_job_id"],
    )
    op.create_index(
        op.f("ix_protocol_pdf_ingestion_jobs_status"),
        "protocol_pdf_ingestion_jobs",
        ["status"],
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_protocol_pdf_ingestion_jobs_status"), table_name="protocol_pdf_ingestion_jobs")
    op.drop_index(op.f("ix_protocol_pdf_ingestion_jobs_rq_job_id"), table_name="protocol_pdf_ingestion_jobs")
    op.drop_index(op.f("ix_protocol_pdf_ingestion_jobs_ingestion_id"), table_name="protocol_pdf_ingestion_jobs")
    op.drop_table("protocol_pdf_ingestion_jobs")
    op.drop_index(op.f("ix_protocol_pdf_ingestions_status"), table_name="protocol_pdf_ingestions")
    op.drop_table("protocol_pdf_ingestions")
    protocol_ingestion_job_status.drop(op.get_bind(), checkfirst=True)
    protocol_ingestion_status.drop(op.get_bind(), checkfirst=True)
