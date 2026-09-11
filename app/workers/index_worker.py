"""RQ worker task for indexing knowledge documents."""

from contextlib import suppress
from datetime import UTC, datetime
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.errors import error_code_of
from app.core.job_context import with_job_request_id
from app.core.job_failure import (
    apply_retry_policy,
    format_failure_message,
    record_job_failure,
)
from app.models.knowledge_base import DocumentStatus, IndexJobStatus, KnowledgeChunk
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.document_splitter_service import document_splitter_service
from app.services.metadata_enricher import metadata_enricher
from app.services.text_cleaner import text_cleaner
from app.services.vector_store_manager import vector_store_manager


@with_job_request_id
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
            job.finished_at = datetime.now(UTC).replace(tzinfo=None)
            db.commit()
            return

        job.status = IndexJobStatus.RUNNING
        job.started_at = datetime.now(UTC).replace(tzinfo=None)
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
        job.finished_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()
        logger.info(f"索引任务完成: job_id={job_id}, chunks={len(vector_docs)}")

    except Exception as exc:
        # 顺序是刻意的：先裁决重试，再落盘，最后原样 raise。
        #
        # 先裁决是因为 error_message 里要写清「还会不会重试」——
        # 否则页面上写着 FAILED、后台其实还排着两次重试，
        # 值班同学会立刻开始排查一个十秒后可能自己就好了的问题。
        will_retry = apply_retry_policy(exc, job_id=job_id)
        message = format_failure_message(exc, will_retry=will_retry)

        # 原 session 的连接可能已经废了，rollback 本身都可能抛。
        # 包进 suppress：它只是尽力释放事务，失败不该影响后面的抢救。
        with suppress(Exception):
            db.rollback()

        def _write_failed(session: Session) -> None:
            # 注意必须用新 session 重新构造 repo、重新取行。
            # 原 session 里的 ORM 对象绑在那个可能已经断掉的连接上，
            # 跨 session 复用会在 flush 时炸开。
            failure_repo = KnowledgeRepository(session)
            failed_job = failure_repo.get_index_job(job_id)
            if failed_job is None:
                return
            failed_job.status = IndexJobStatus.FAILED
            failed_job.error_message = message
            failed_job.finished_at = datetime.now(UTC).replace(tzinfo=None)
            failed_document = failure_repo.get_document(failed_job.document_id)
            if failed_document is not None:
                failed_document.status = DocumentStatus.FAILED
                failed_document.error_message = message

        # 走独立 session，且内部绝不外抛 —— 抢救失败最多丢一条状态，
        # 不能让二次异常顶掉真正的原因。
        record_job_failure(_write_failed, job_id=job_id)

        logger.opt(exception=exc).error(
            f"索引任务失败: job_id={job_id}, error_code={error_code_of(exc)}, "
            f"will_retry={will_retry}"
        )
        # 必须原样 raise：RQ 靠这个异常判定任务失败并决定是否重排队。
        raise
    finally:
        with suppress(Exception):
            db.close()
