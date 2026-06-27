"""Protocol PDF ingestion API."""

import hashlib
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.file import _sanitize_filename
from app.config import config
from app.core.database import get_db
from app.models.protocol_ingestion import ProtocolPdfIngestion
from app.repositories.protocol_ingestion_repository import ProtocolIngestionRepository
from app.services.protocol_pdf_ingestion_service import protocol_pdf_ingestion_service
from app.services.protocol_pdf_job_queue import enqueue_protocol_pdf_ingestion_job
from loguru import logger

router = APIRouter(prefix="/protocol-pdfs")

PROTOCOL_UPLOAD_DIR = Path(config.upload_dir) / "protocol_pdfs"
MAX_PDF_SIZE = 30 * 1024 * 1024


class ConfirmProtocolPdfRequest(BaseModel):
    confirmed_by: str = Field(default="manual-reviewer", min_length=1, max_length=255)


class RejectProtocolPdfRequest(BaseModel):
    rejected_by: str = Field(default="manual-reviewer", min_length=1, max_length=255)
    reason: str | None = Field(default=None, max_length=2000)


@router.post("/upload", status_code=202)
async def upload_protocol_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    safe_filename = _sanitize_filename(file.filename)
    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 协议文件")

    content = await file.read()
    if len(content) > MAX_PDF_SIZE:
        raise HTTPException(status_code=400, detail=f"文件大小超过限制（最大 {MAX_PDF_SIZE} 字节）")

    PROTOCOL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = PROTOCOL_UPLOAD_DIR / safe_filename
    if file_path.exists():
        stem = file_path.stem
        suffix = file_path.suffix
        file_path = PROTOCOL_UPLOAD_DIR / f"{stem}_{hashlib.sha256(content).hexdigest()[:8]}{suffix}"

    file_path.write_bytes(content)
    logger.info(f"协议 PDF 上传成功: {file_path}")

    repo = ProtocolIngestionRepository(db)
    ingestion = repo.create_ingestion(
        filename=file_path.name,
        original_filename=file.filename,
        file_path=str(file_path.resolve()),
        file_size=len(content),
        content_hash=hashlib.sha256(content).hexdigest(),
    )
    job = repo.create_job(ingestion.id)

    try:
        rq_job = enqueue_protocol_pdf_ingestion_job(job.id)
        repo.bind_rq_job(job.id, rq_job.id)
    except Exception as exc:
        logger.exception(f"协议 PDF 入库任务入队失败: ingestion_id={ingestion.id}, job_id={job.id}")
        raise HTTPException(
            status_code=503,
            detail=f"文件已保存，但 PDF 入库任务入队失败，请检查 Redis/RQ: {exc}",
        ) from exc

    return JSONResponse(
        status_code=202,
        content={
            "code": 202,
            "message": "accepted",
            "data": {
                "ingestion_id": ingestion.id,
                "job_id": job.id,
                "rq_job_id": rq_job.id,
                "status": ingestion.status.value,
                "filename": ingestion.filename,
                "file_path": str(file_path),
                "size": len(content),
            },
        },
    )


@router.get("")
async def list_protocol_pdf_ingestions(db: Session = Depends(get_db)):
    repo = ProtocolIngestionRepository(db)
    return {
        "code": 200,
        "message": "success",
        "data": [_serialize_summary(item) for item in repo.list_ingestions()],
    }


@router.get("/{ingestion_id}")
async def get_protocol_pdf_ingestion(
    ingestion_id: str,
    db: Session = Depends(get_db),
):
    repo = ProtocolIngestionRepository(db)
    ingestion = repo.get_ingestion(ingestion_id)
    if ingestion is None:
        raise HTTPException(status_code=404, detail="协议 PDF 入库任务不存在")
    return {"code": 200, "message": "success", "data": _serialize_detail(ingestion)}


@router.post("/{ingestion_id}/confirm")
async def confirm_protocol_pdf_ingestion(
    ingestion_id: str,
    request: ConfirmProtocolPdfRequest,
    db: Session = Depends(get_db),
):
    try:
        ingestion = protocol_pdf_ingestion_service.confirm_ingestion(
            db,
            ingestion_id,
            confirmed_by=request.confirmed_by,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"code": 200, "message": "success", "data": _serialize_detail(ingestion)}


@router.post("/{ingestion_id}/reject")
async def reject_protocol_pdf_ingestion(
    ingestion_id: str,
    request: RejectProtocolPdfRequest,
    db: Session = Depends(get_db),
):
    repo = ProtocolIngestionRepository(db)
    ingestion = repo.get_ingestion(ingestion_id)
    if ingestion is None:
        raise HTTPException(status_code=404, detail="协议 PDF 入库任务不存在")
    ingestion = repo.reject(
        ingestion,
        rejected_by=request.rejected_by,
        reason=request.reason,
    )
    return {"code": 200, "message": "success", "data": _serialize_detail(ingestion)}


def _serialize_summary(ingestion: ProtocolPdfIngestion) -> dict:
    return {
        "id": ingestion.id,
        "filename": ingestion.filename,
        "original_filename": ingestion.original_filename,
        "status": ingestion.status.value,
        "current_phase": ingestion.current_phase,
        "file_size": ingestion.file_size,
        "content_hash": ingestion.content_hash,
        "created_at": ingestion.created_at.isoformat(),
        "updated_at": ingestion.updated_at.isoformat(),
        "error_message": ingestion.error_message,
    }


def _serialize_detail(ingestion: ProtocolPdfIngestion) -> dict:
    data = _serialize_summary(ingestion)
    data.update(
        {
            "schema_version": ingestion.schema_version,
            "structured_data": ingestion.structured_data,
            "validation_result": ingestion.validation_result,
            "dry_run_plan": ingestion.dry_run_plan,
            "state_trace": ingestion.state_trace,
            "confirmed_by": ingestion.confirmed_by,
            "confirmed_at": ingestion.confirmed_at.isoformat() if ingestion.confirmed_at else None,
        }
    )
    return data
