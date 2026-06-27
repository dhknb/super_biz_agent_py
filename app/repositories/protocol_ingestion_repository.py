"""Repository for protocol PDF ingestion state."""

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.protocol_ingestion import (
    ProtocolIngestionJobStatus,
    ProtocolIngestionStatus,
    ProtocolPdfIngestion,
    ProtocolPdfIngestionJob,
)


class ProtocolIngestionRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_ingestion(
        self,
        *,
        filename: str,
        original_filename: str,
        file_path: str,
        file_size: int,
        content_hash: str,
    ) -> ProtocolPdfIngestion:
        ingestion = ProtocolPdfIngestion(
            filename=filename,
            original_filename=original_filename,
            file_path=file_path,
            file_size=file_size,
            content_hash=content_hash,
            status=ProtocolIngestionStatus.PENDING,
        )
        self.db.add(ingestion)
        self.db.commit()
        self.db.refresh(ingestion)
        return ingestion

    def create_job(self, ingestion_id: str) -> ProtocolPdfIngestionJob:
        job = ProtocolPdfIngestionJob(
            ingestion_id=ingestion_id,
            status=ProtocolIngestionJobStatus.QUEUED,
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def bind_rq_job(self, job_id: str, rq_job_id: str) -> None:
        job = self.get_job(job_id)
        if job is None:
            return
        job.rq_job_id = rq_job_id
        self.db.commit()

    def get_ingestion(self, ingestion_id: str) -> ProtocolPdfIngestion | None:
        return self.db.get(ProtocolPdfIngestion, ingestion_id)

    def get_job(self, job_id: str) -> ProtocolPdfIngestionJob | None:
        return self.db.get(ProtocolPdfIngestionJob, job_id)

    def list_ingestions(self) -> list[ProtocolPdfIngestion]:
        stmt = select(ProtocolPdfIngestion).order_by(ProtocolPdfIngestion.created_at.desc())
        return list(self.db.scalars(stmt).all())

    def set_status(
        self,
        ingestion: ProtocolPdfIngestion,
        status: ProtocolIngestionStatus,
        *,
        phase: str | None = None,
        error_message: str | None = None,
    ) -> ProtocolPdfIngestion:
        ingestion.status = status
        ingestion.current_phase = phase
        ingestion.error_message = error_message
        self.db.commit()
        self.db.refresh(ingestion)
        return ingestion

    def save_dry_run(
        self,
        ingestion: ProtocolPdfIngestion,
        *,
        extracted_text: str,
        structured_data: dict[str, Any],
        validation_result: dict[str, Any],
        dry_run_plan: dict[str, Any],
    ) -> ProtocolPdfIngestion:
        ingestion.extracted_text = extracted_text
        ingestion.structured_data = structured_data
        ingestion.validation_result = validation_result
        ingestion.dry_run_plan = dry_run_plan
        ingestion.status = ProtocolIngestionStatus.AWAITING_CONFIRMATION
        ingestion.current_phase = "awaiting_confirmation"
        ingestion.error_message = None
        self.db.commit()
        self.db.refresh(ingestion)
        return ingestion

    def confirm_with_state_trace(
        self,
        ingestion: ProtocolPdfIngestion,
        *,
        confirmed_by: str,
        state_trace: list[dict[str, Any]],
    ) -> ProtocolPdfIngestion:
        ingestion.status = ProtocolIngestionStatus.COMPLETED
        ingestion.current_phase = "thresholds_saved"
        ingestion.confirmed_by = confirmed_by
        ingestion.confirmed_at = datetime.utcnow()
        ingestion.state_trace = state_trace
        ingestion.error_message = None
        self.db.commit()
        self.db.refresh(ingestion)
        return ingestion

    def reject(
        self,
        ingestion: ProtocolPdfIngestion,
        *,
        rejected_by: str,
        reason: str | None = None,
    ) -> ProtocolPdfIngestion:
        trace = list(ingestion.state_trace or [])
        trace.append(
            {
                "phase": "rejected",
                "status": "rejected",
                "actor": rejected_by,
                "reason": reason,
                "at": datetime.utcnow().isoformat(),
            }
        )
        ingestion.status = ProtocolIngestionStatus.REJECTED
        ingestion.current_phase = "rejected"
        ingestion.state_trace = trace
        ingestion.error_message = reason
        self.db.commit()
        self.db.refresh(ingestion)
        return ingestion
