"""Protocol PDF ingestion persistence models."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class ProtocolIngestionStatus(StrEnum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    STRUCTURING = "structuring"
    VALIDATING = "validating"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    WRITING = "writing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class ProtocolIngestionJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ProtocolPdfIngestion(Base):
    __tablename__ = "protocol_pdf_ingestions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    schema_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="protocol_pdf_v1",
    )
    status: Mapped[ProtocolIngestionStatus] = mapped_column(
        Enum(ProtocolIngestionStatus, values_callable=lambda items: [item.value for item in items]),
        nullable=False,
        default=ProtocolIngestionStatus.PENDING,
        index=True,
    )
    current_phase: Mapped[str | None] = mapped_column(String(64), nullable=True)
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    structured_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    validation_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    dry_run_plan: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    state_trace: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    confirmed_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    jobs: Mapped[list["ProtocolPdfIngestionJob"]] = relationship(back_populates="ingestion")


class ProtocolPdfIngestionJob(Base):
    __tablename__ = "protocol_pdf_ingestion_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    ingestion_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("protocol_pdf_ingestions.id"),
        nullable=False,
        index=True,
    )
    rq_job_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    status: Mapped[ProtocolIngestionJobStatus] = mapped_column(
        Enum(
            ProtocolIngestionJobStatus,
            values_callable=lambda items: [item.value for item in items],
        ),
        nullable=False,
        default=ProtocolIngestionJobStatus.QUEUED,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    ingestion: Mapped[ProtocolPdfIngestion] = relationship(back_populates="jobs")
