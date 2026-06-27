"""RQ worker task for indexing knowledge documents."""

from datetime import datetime
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger

from app.core.database import SessionLocal
from app.models.knowledge_base import DocumentStatus, IndexJobStatus, KnowledgeChunk
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.document_splitter_service import document_splitter_service
from app.services.metadata_enricher import metadata_enricher
from app.services.text_cleaner import text_cleaner
from app.services.vector_store_manager import vector_store_manager


def run_index_job(job_id: str) -> None:
    db = SessionLocal()
    repo = KnowledgeRepository(db)
    try:
        job = repo.get_index_job(job_id)
        if job is None:
            logger.warning(f"索引任务不存在: {job_id}")
            return

        document = repo.get_document(job.document_id)
        if document is None:
            job.status = IndexJobStatus.FAILED
            job.error_message = "document not found"
            job.finished_at = datetime.utcnow()
            db.commit()
            return

        job.status = IndexJobStatus.RUNNING
        job.started_at = datetime.utcnow()
        document.status = DocumentStatus.INDEXING
        document.error_message = None
        db.commit()

        file_path = Path(document.file_path)
        content = file_path.read_text(encoding="utf-8")
        content = text_cleaner.clean(content)
        docs = document_splitter_service.split_document(content, document.file_path)
        docs = metadata_enricher.enrich_documents(docs, document.file_path)

        vector_store_manager.delete_by_document_id(document.id)

        chunks: list[KnowledgeChunk] = []
        for index, doc in enumerate(docs):
            chunk_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
            chunks.append(
                KnowledgeChunk(
                    document_id=document.id,
                    chunk_index=index,
                    content=doc.page_content,
                    content_hash=chunk_hash,
                    token_count=len(doc.page_content),
                )
            )

        repo.replace_chunks(document.id, chunks)

        vector_docs = [
            Document(
                page_content=chunk.content,
                metadata={
                    # 继承 enricher 写入的 doc_tags / chunk_summary / h1 / h2 等
                    **docs[index].metadata,
                    # 覆盖/补充标准字段(确保一致性)
                    "document_id": document.id,
                    "chunk_id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "version": document.version,
                    "_source": document.file_path,
                    "_file_name": document.filename,
                    "_extension": document.file_ext,
                },
            )
            for index, chunk in enumerate(chunks)
        ]

        if vector_docs:
            vector_store_manager.add_documents(vector_docs)

        document.status = DocumentStatus.INDEXED
        document.error_message = None
        job.status = IndexJobStatus.SUCCEEDED
        job.chunk_count = len(vector_docs)
        job.finished_at = datetime.utcnow()
        db.commit()
        logger.info(f"索引任务完成: job_id={job_id}, chunks={len(vector_docs)}")

    except Exception as exc:
        db.rollback()
        job = repo.get_index_job(job_id)
        if job is not None:
            job.status = IndexJobStatus.FAILED
            job.error_message = str(exc)
            job.finished_at = datetime.utcnow()
            document = repo.get_document(job.document_id)
            if document is not None:
                document.status = DocumentStatus.FAILED
                document.error_message = str(exc)
            db.commit()
        logger.exception(f"索引任务失败: job_id={job_id}")
        raise
    finally:
        db.close()
