"""文件上传接口模块"""

import hashlib
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.config import config
from app.core.database import get_db
from app.models.knowledge_base import DocumentStatus
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.index_job_queue import enqueue_index_job
from app.services.vector_index_service import vector_index_service
from loguru import logger

router = APIRouter()

# 文件上传后存储的路径
UPLOAD_DIR = Path(config.upload_dir)
# 支持的文件类型
ALLOWED_EXTENSIONS = ["txt", "md"]
# 单个文件支持最大大小
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@router.post("/upload", status_code=202)
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    上传文件并创建异步索引任务

    Args:
        file: 上传的文件

    Returns:
        JSONResponse: 上传结果
    """
    try:
        # 1. 验证文件
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        # 2. 规范化文件名（去除空格，处理 Windows 上传的文件）
        safe_filename = _sanitize_filename(file.filename)

        # 3. 验证文件扩展名
        file_extension = _get_file_extension(safe_filename)
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式，仅支持: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        # 4. 创建上传目录
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # 5. 保存文件
        file_path = UPLOAD_DIR / safe_filename

        # 如果文件已存在，先删除旧文件（实现覆盖更新）
        if file_path.exists():
            logger.info(f"文件已存在，将覆盖: {file_path}")
            file_path.unlink()

        # 读取并保存文件内容
        content = await file.read()

        # 验证文件大小
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"文件大小超过限制（最大 {MAX_FILE_SIZE} 字节）")

        file_path.write_bytes(content)

        logger.info(f"文件上传成功: {file_path}")

        repo = KnowledgeRepository(db)
        content_hash = hashlib.sha256(content).hexdigest()
        resolved_path = str(file_path.resolve())

        # 去重:按 sanitize 后的 filename 查同名(未软删)文档
        existing = repo.get_document_by_filename(safe_filename)

        if existing is None:
            # 全新文档 → 正常创建
            document = repo.create_document(
                filename=safe_filename,
                original_filename=file.filename,
                file_path=resolved_path,
                file_ext=file_extension,
                file_size=len(content),
                content_hash=content_hash,
            )
            reused = False
        elif (
            existing.content_hash == content_hash
            and existing.status == DocumentStatus.INDEXED
        ):
            # 同名+内容没变+已索引完成 → 复用,跳过索引(省 embedding)
            logger.info(f"复用现有文档(同名同内容已索引): id={existing.id}")
            return JSONResponse(
                status_code=200,
                content={
                    "code": 200,
                    "message": "reused",
                    "data": {
                        "document_id": existing.id,
                        "job_id": None,
                        "rq_job_id": None,
                        "status": existing.status.value,
                        "filename": safe_filename,
                        "file_path": str(file_path),
                        "size": len(content),
                        "version": existing.version,
                        "reused": True,
                    },
                },
            )
        else:
            # 同名但内容变了 / 之前失败 / pending → 更新现有记录,版本号 +1
            logger.info(
                f"同名重传,更新现有文档: id={existing.id}, old_version={existing.version}"
            )
            document = repo.update_document_for_reupload(
                existing,
                original_filename=file.filename,
                file_path=resolved_path,
                file_size=len(content),
                content_hash=content_hash,
            )
            reused = False

        job = repo.create_index_job(document.id)

        try:
            rq_job = enqueue_index_job(job.id)
            repo.bind_rq_job(job.id, rq_job.id)
        except Exception as e:
            logger.exception(f"索引任务入队失败: document_id={document.id}, job_id={job.id}")
            raise HTTPException(
                status_code=503,
                detail=f"文件已保存，但索引任务入队失败，请检查 Redis/RQ: {e}",
            ) from e

        return JSONResponse(
            status_code=202,
            content={
                "code": 202,
                "message": "accepted",
                "data": {
                    "document_id": document.id,
                    "job_id": job.id,
                    "rq_job_id": job.rq_job_id,
                    "status": document.status.value,
                    "filename": safe_filename,
                    "file_path": str(file_path),
                    "size": len(content),
                    "version": document.version,
                    "reused": reused,
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {e}")


@router.get("/documents")
async def list_documents(db: Session = Depends(get_db)):
    repo = KnowledgeRepository(db)
    documents = repo.list_documents()
    return {
        "code": 200,
        "message": "success",
        "data": [
            {
                "id": document.id,
                "filename": document.filename,
                "original_filename": document.original_filename,
                "status": document.status.value,
                "version": document.version,
                "file_size": document.file_size,
                "content_hash": document.content_hash,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat(),
                "error_message": document.error_message,
            }
            for document in documents
        ],
    }


@router.post("/documents/{document_id}/reindex", status_code=202)
async def reindex_document(
    document_id: str,
    db: Session = Depends(get_db),
):
    repo = KnowledgeRepository(db)
    document = repo.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    job = repo.create_index_job(document.id)
    try:
        rq_job = enqueue_index_job(job.id)
        repo.bind_rq_job(job.id, rq_job.id)
    except Exception as e:
        logger.exception(f"重建索引任务入队失败: document_id={document.id}, job_id={job.id}")
        raise HTTPException(status_code=503, detail=f"索引任务入队失败，请检查 Redis/RQ: {e}") from e

    return {
        "code": 202,
        "message": "accepted",
        "data": {
            "document_id": document.id,
            "job_id": job.id,
            "rq_job_id": rq_job.id,
        },
    }


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
):
    repo = KnowledgeRepository(db)
    document = repo.mark_document_deleted(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    from app.services.vector_store_manager import vector_store_manager
#删除collection中符合条件的document_id（在medata）
    deleted_vectors = vector_store_manager.delete_by_document_id(document.id)
    return {
        "code": 200,
        "message": "success",
        "data": {
            "document_id": document.id,
            "status": document.status.value,
            "deleted_vectors": deleted_vectors,
        },
    }


@router.post("/index_directory")
async def index_directory(directory_path: str = None):
    """
    索引指定目录下的所有文件

    Args:
        directory_path: 目录路径（可选，默认使用 uploads 目录）

    Returns:
        JSONResponse: 索引结果
    """
    try:
        logger.info(f"开始索引目录: {directory_path or 'uploads'}")

        # 执行索引
        result = vector_index_service.index_directory(directory_path)

        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "message": "success" if result.success else "partial_success",
                "data": result.to_dict(),
            },
        )

    except Exception as e:
        logger.error(f"索引目录失败: {e}")
        raise HTTPException(status_code=500, detail=f"索引目录失败: {e}")


def _get_file_extension(filename: str) -> str:
    """
    获取文件扩展名

    Args:
        filename: 文件名

    Returns:
        str: 扩展名（小写，不含点）
    """
    parts = filename.rsplit(".", 1)
    if len(parts) == 2:
        return parts[1].lower()
    return ""


def _sanitize_filename(filename: str) -> str:
    """
    规范化文件名，去除空格和特殊字符

    Args:
        filename: 原始文件名

    Returns:
        str: 规范化后的文件名
    """
    # 去除空格
    sanitized = filename.replace(" ", "_")
    # 去除其他可能导致问题的字符
    for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        sanitized = sanitized.replace(char, "_")
    return sanitized
