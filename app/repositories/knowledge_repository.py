"""Repository for knowledge document metadata."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.knowledge_base import (
    DocumentStatus,
    IndexJob,
    IndexJobStatus,
    KnowledgeChunk,
    KnowledgeDocument,
)


class KnowledgeRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_document(
        self,
        *,
        filename: str,
        original_filename: str,
        file_path: str,
        file_ext: str,
        file_size: int,
        content_hash: str,
    ) -> KnowledgeDocument:
        document = KnowledgeDocument(
            filename=filename,
            original_filename=original_filename,
            file_path=file_path,
            file_ext=file_ext,
            file_size=file_size,
            content_hash=content_hash,
            status=DocumentStatus.PENDING,
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def create_index_job(self, document_id: str) -> IndexJob:
        job = IndexJob(document_id=document_id, status=IndexJobStatus.QUEUED)
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def bind_rq_job(self, job_id: str, rq_job_id: str) -> None:
        job = self.get_index_job(job_id)
        if job is None:
            return
        job.rq_job_id = rq_job_id
        self.db.commit()

    def get_document(self, document_id: str) -> KnowledgeDocument | None:
        return self.db.get(KnowledgeDocument, document_id)

    def get_document_by_filename(self, filename: str) -> KnowledgeDocument | None:
        """按 sanitize 后的文件名查找未删除的文档。用于上传时的 upsert 判断。"""
        stmt = (
            select(KnowledgeDocument)
            .where(KnowledgeDocument.filename == filename)
            .where(KnowledgeDocument.status != DocumentStatus.DELETED)
            .order_by(KnowledgeDocument.created_at.desc())
        )
        return self.db.scalars(stmt).first()

    def update_document_for_reupload(
        self,
        document: KnowledgeDocument,
        *,
        original_filename: str,
        file_path: str,
        file_size: int,
        content_hash: str,
    ) -> KnowledgeDocument:
        """同名重传时更新现有文档:版本号 +1,状态重置为 PENDING,清掉旧错误。"""
        document.original_filename = original_filename
        document.file_path = file_path
        document.file_size = file_size
        document.content_hash = content_hash
        document.version = (document.version or 1) + 1
        document.status = DocumentStatus.PENDING
        document.error_message = None
        self.db.commit()
        self.db.refresh(document)
        return document

    def get_index_job(self, job_id: str) -> IndexJob | None:
        return self.db.get(IndexJob, job_id)

    def list_documents(self) -> list[KnowledgeDocument]:
        stmt = (
            select(KnowledgeDocument)
            .where(KnowledgeDocument.status != DocumentStatus.DELETED)
            .order_by(KnowledgeDocument.created_at.desc())
        )
        return list(self.db.scalars(stmt).all())

    def replace_chunks(self, document_id: str, chunks: list[KnowledgeChunk]) -> None:
        old_chunks = self.db.scalars(
            select(KnowledgeChunk).where(KnowledgeChunk.document_id == document_id)
        ).all()
        for chunk in old_chunks:
            self.db.delete(chunk)
        for chunk in chunks:
            self.db.add(chunk)
        self.db.commit()

    def mark_document_deleted(self, document_id: str) -> KnowledgeDocument | None:
        document = self.get_document(document_id)
        if document is None:
            return None
        document.status = DocumentStatus.DELETED
        self.db.commit()
        self.db.refresh(document)
        return document
