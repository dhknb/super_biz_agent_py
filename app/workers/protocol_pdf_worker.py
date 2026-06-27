"""RQ worker task for protocol PDF ingestion."""

from datetime import datetime

from loguru import logger

from app.core.database import SessionLocal
from app.models.protocol_ingestion import (
    ProtocolIngestionJobStatus,
    ProtocolIngestionStatus,
)
from app.repositories.protocol_ingestion_repository import ProtocolIngestionRepository
from app.services.protocol_pdf_ingestion_service import protocol_pdf_ingestion_service


def run_protocol_pdf_ingestion_job(job_id: str) -> None:
    db = SessionLocal()
    repo = ProtocolIngestionRepository(db)
    try:
        job = repo.get_job(job_id)
        if job is None:
            logger.warning(f"协议 PDF 入库任务不存在: {job_id}")
            return

        job.status = ProtocolIngestionJobStatus.RUNNING
        job.started_at = datetime.utcnow()
        db.commit()

        protocol_pdf_ingestion_service.process_ingestion(db, job.ingestion_id)

        job.status = ProtocolIngestionJobStatus.SUCCEEDED
        job.finished_at = datetime.utcnow()
        db.commit()
        logger.info(f"协议 PDF 入库任务完成: job_id={job_id}")

    except Exception as exc:
        db.rollback()
        job = repo.get_job(job_id)
        if job is not None:
            job.status = ProtocolIngestionJobStatus.FAILED
            job.error_message = str(exc)
            job.finished_at = datetime.utcnow()
            ingestion = repo.get_ingestion(job.ingestion_id)
            if ingestion is not None:
                ingestion.status = ProtocolIngestionStatus.FAILED
                ingestion.error_message = str(exc)
            db.commit()
        logger.exception(f"协议 PDF 入库任务失败: job_id={job_id}")
        raise
    finally:
        db.close()
