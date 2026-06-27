# 计划 1：知识库工程化实施文档

## 1. 目标

当前项目已经能完成“上传文件 -> 文档切分 -> 生成向量 -> 写入 Milvus -> RAG 检索”。这条链路适合演示和学习，但企业落地还需要补齐文档生命周期、索引任务、失败重试、版本追踪、可删除可重建等能力。

本计划的目标是把当前知识库从“文件上传接口”升级为“知识库管理系统”。

你完成本计划后，系统应该具备这些能力：

- 文档有独立元数据记录，不只存在于 `uploads/` 和 Milvus。
- 上传文件不会阻塞 HTTP 请求，索引由后台任务执行。
- 每个文档有状态：`pending`、`indexing`、`indexed`、`failed`、`deleted`。
- 每次索引有任务记录，可以查看耗时、错误、分片数量。
- 支持文档重建索引、软删除、失败重试。
- Milvus 只负责向量检索，业务状态由关系数据库管理。

## 2. 当前项目现状

重点文件：

- `app/api/file.py`：上传文件后同步调用 `vector_index_service.index_single_file()`。
- `app/services/vector_index_service.py`：读取文件、删除旧数据、切分、写入 Milvus。
- `app/services/document_splitter_service.py`：负责 Markdown 和文本切分。
- `app/services/vector_store_manager.py`：负责写入 Milvus。
- `app/core/milvus_client.py`：负责 Milvus 连接和 collection 初始化。

当前主要问题：

- 上传接口同步做 embedding，文件稍大或模型慢时接口会长时间等待。
- 文档没有数据库记录，无法查询“系统里有哪些文档”。
- 索引失败后没有任务记录，排障只能看日志。
- Milvus metadata 里只有 `_source`、`_extension`、`_file_name`，缺少 `document_id`、`chunk_id`、`version`。
- 删除旧数据依赖文件路径，文件重名、路径变化、租户隔离都会变复杂。

## 3. 目标架构

推荐先做一个轻量版架构：

```text
上传文件
  -> 保存文件到 uploads/
  -> 写入 documents 表，状态 pending
  -> 写入 index_jobs 表，状态 queued
  -> 返回 document_id 和 job_id

后台索引任务
  -> 读取 index_jobs
  -> 更新 job 状态 running
  -> 读取文件
  -> 切分 chunks
  -> 写入 document_chunks 表
  -> 写入 Milvus，metadata 绑定 document_id/chunk_id/version
  -> 更新 document 状态 indexed
  -> 更新 job 状态 succeeded
```

初期你可以先用 FastAPI `BackgroundTasks`，后续再升级到 Redis + Celery/RQ。

## 4. 推荐新增依赖

在 `pyproject.toml` 中新增：

```toml
dependencies = [
    # existing dependencies...
    "sqlalchemy>=2.0.0",
    "alembic>=1.13.0",
]
```

如果你准备用 PostgreSQL，再加：

```toml
"asyncpg>=0.29.0"
```

学习阶段可以先用 SQLite，方便本地跑通。

## 5. 新增配置

修改 `app/config.py`，给 `Settings` 增加数据库配置：

```python
database_url: str = "sqlite:///./super_biz_agent.db"
upload_dir: str = "./uploads"
```

`.env.example` 增加：

```bash
# Database
DATABASE_URL=sqlite:///./super_biz_agent.db
UPLOAD_DIR=./uploads
```

生产环境建议使用：

```bash
DATABASE_URL=postgresql+psycopg://user:password@postgres:5432/super_biz_agent
```

## 6. 新增数据库基础模块

新增文件：`app/core/database.py`

```python
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import config


class Base(DeclarativeBase):
    pass


engine = create_engine(
    config.database_url,
    connect_args={"check_same_thread": False}
    if config.database_url.startswith("sqlite")
    else {},
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

学习阶段可以先不接 Alembic，直接在启动时创建表。等你理解后，再引入 Alembic 做迁移管理。

## 7. 新增 ORM 模型

新增文件：`app/models/knowledge_base.py`

```python
from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class DocumentStatus(StrEnum):
    PENDING = "pending"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


class IndexJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_ext: Mapped[str] = mapped_column(String(32), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus),
        nullable=False,
        default=DocumentStatus.PENDING,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    chunks: Mapped[list["KnowledgeChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    document_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("knowledge_documents.id"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped[KnowledgeDocument] = relationship(back_populates="chunks")


class IndexJob(Base):
    __tablename__ = "index_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    document_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("knowledge_documents.id"),
        nullable=False,
        index=True,
    )
    status: Mapped[IndexJobStatus] = mapped_column(
        Enum(IndexJobStatus),
        nullable=False,
        default=IndexJobStatus.QUEUED,
    )
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

为什么需要 `content_hash`：

- 判断同一个文件是否重复上传。
- 做幂等索引。
- 后续可以只重建变化的 chunk。

## 8. 启动时创建表

学习阶段可以在 `app/main.py` 的 lifespan 中临时创建表。

```python
from app.core.database import Base, engine
from app.models import knowledge_base


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    milvus_manager.connect()
    yield
    milvus_manager.close()
```

注意：生产环境不要长期使用 `create_all` 管理迁移，后续要切换到 Alembic。

## 9. 新增 Repository 层

新增文件：`app/repositories/knowledge_repository.py`

```python
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
        doc = KnowledgeDocument(
            filename=filename,
            original_filename=original_filename,
            file_path=file_path,
            file_ext=file_ext,
            file_size=file_size,
            content_hash=content_hash,
            status=DocumentStatus.PENDING,
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def create_index_job(self, document_id: str) -> IndexJob:
        job = IndexJob(document_id=document_id, status=IndexJobStatus.QUEUED)
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def get_document(self, document_id: str) -> KnowledgeDocument | None:
        return self.db.get(KnowledgeDocument, document_id)

    def list_documents(self) -> list[KnowledgeDocument]:
        stmt = select(KnowledgeDocument).where(
            KnowledgeDocument.status != DocumentStatus.DELETED
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
```

Repository 的作用是把数据库细节从业务服务里隔离出来。以后你换数据库、加过滤条件、加分页，都优先改 Repository。

## 10. 改造上传接口

当前 `app/api/file.py` 上传成功后立刻索引。建议改成：

- 保存文件。
- 写入 `KnowledgeDocument`。
- 创建 `IndexJob`。
- 用 `BackgroundTasks` 后台索引。
- 立即返回 `document_id` 和 `job_id`。

示例代码：

```python
import hashlib
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.knowledge_index_task_service import knowledge_index_task_service


@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    safe_filename = _sanitize_filename(file.filename)
    file_extension = _get_file_extension(safe_filename)
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="不支持的文件格式")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="文件大小超过限制")

    content_hash = hashlib.sha256(content).hexdigest()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / safe_filename
    file_path.write_bytes(content)

    repo = KnowledgeRepository(db)
    document = repo.create_document(
        filename=safe_filename,
        original_filename=file.filename,
        file_path=str(file_path.resolve()),
        file_ext=file_extension,
        file_size=len(content),
        content_hash=content_hash,
    )
    job = repo.create_index_job(document.id)

    background_tasks.add_task(
        knowledge_index_task_service.run_index_job,
        job.id,
    )

    return {
        "code": 202,
        "message": "accepted",
        "data": {
            "document_id": document.id,
            "job_id": job.id,
            "status": document.status,
        },
    }
```

这里返回 `202` 更符合语义：请求已接受，索引还在后台执行。

## 11. 新增后台索引任务服务

新增文件：`app/services/knowledge_index_task_service.py`

```python
from datetime import datetime
import hashlib

from langchain_core.documents import Document

from app.core.database import SessionLocal
from app.models.knowledge_base import (
    DocumentStatus,
    IndexJobStatus,
    KnowledgeChunk,
)
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.document_splitter_service import document_splitter_service
from app.services.vector_store_manager import vector_store_manager


class KnowledgeIndexTaskService:
    def run_index_job(self, job_id: str) -> None:
        db = SessionLocal()
        try:
            repo = KnowledgeRepository(db)
            job = db.get(IndexJob, job_id)
            if job is None:
                return

            document = repo.get_document(job.document_id)
            if document is None:
                return

            job.status = IndexJobStatus.RUNNING
            job.started_at = datetime.utcnow()
            document.status = DocumentStatus.INDEXING
            db.commit()

            content = Path(document.file_path).read_text(encoding="utf-8")
            docs = document_splitter_service.split_document(content, document.file_path)

            vector_store_manager.delete_by_document_id(document.id)

            chunks: list[KnowledgeChunk] = []
            vector_docs: list[Document] = []

            for index, doc in enumerate(docs):
                chunk_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
                chunk = KnowledgeChunk(
                    document_id=document.id,
                    chunk_index=index,
                    content=doc.page_content,
                    content_hash=chunk_hash,
                    token_count=len(doc.page_content),
                )
                chunks.append(chunk)

            repo.replace_chunks(document.id, chunks)

            for chunk in chunks:
                vector_docs.append(
                    Document(
                        page_content=chunk.content,
                        metadata={
                            "document_id": document.id,
                            "chunk_id": chunk.id,
                            "chunk_index": chunk.chunk_index,
                            "version": document.version,
                            "_source": document.file_path,
                            "_file_name": document.filename,
                            "_extension": document.file_ext,
                        },
                    )
                )

            if vector_docs:
                vector_store_manager.add_documents(vector_docs)

            document.status = DocumentStatus.INDEXED
            document.error_message = None
            job.status = IndexJobStatus.SUCCEEDED
            job.chunk_count = len(vector_docs)
            job.finished_at = datetime.utcnow()
            db.commit()

        except Exception as exc:
            db.rollback()
            job = db.get(IndexJob, job_id)
            if job is not None:
                job.status = IndexJobStatus.FAILED
                job.error_message = str(exc)
                job.finished_at = datetime.utcnow()
                document = repo.get_document(job.document_id)
                if document is not None:
                    document.status = DocumentStatus.FAILED
                    document.error_message = str(exc)
                db.commit()
            raise
        finally:
            db.close()


knowledge_index_task_service = KnowledgeIndexTaskService()
```

上面代码里需要补两个 import：

```python
from pathlib import Path
from app.models.knowledge_base import IndexJob
```

我故意把它拆开写，因为你学习时要注意：后台任务里不能复用请求里的 `db`，要重新创建 `SessionLocal()`。

## 12. 改造 Milvus 删除逻辑

当前 `VectorStoreManager.delete_by_source()` 根据文件路径删除。建议新增：

```python
def delete_by_document_id(self, document_id: str) -> int:
    try:
        collection = milvus_manager.get_collection()
        expr = f'metadata["document_id"] == "{document_id}"'
        result = collection.delete(expr)
        deleted_count = result.delete_count if hasattr(result, "delete_count") else 0
        logger.info(f"删除文档向量: document_id={document_id}, count={deleted_count}")
        return deleted_count
    except Exception as e:
        logger.warning(f"删除文档向量失败: {e}")
        return 0
```

保留旧的 `delete_by_source()`，这样历史功能不受影响。

## 13. 新增文档查询接口

在 `app/api/file.py` 增加：

```python
@router.get("/documents")
async def list_documents(db: Session = Depends(get_db)):
    repo = KnowledgeRepository(db)
    docs = repo.list_documents()
    return {
        "code": 200,
        "message": "success",
        "data": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "status": doc.status,
                "version": doc.version,
                "file_size": doc.file_size,
                "created_at": doc.created_at.isoformat(),
                "updated_at": doc.updated_at.isoformat(),
                "error_message": doc.error_message,
            }
            for doc in docs
        ],
    }
```

再增加重试索引：

```python
@router.post("/documents/{document_id}/reindex")
async def reindex_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    repo = KnowledgeRepository(db)
    document = repo.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    job = repo.create_index_job(document.id)
    background_tasks.add_task(knowledge_index_task_service.run_index_job, job.id)

    return {
        "code": 202,
        "message": "accepted",
        "data": {"document_id": document.id, "job_id": job.id},
    }
```

## 14. 后续升级到真正任务队列

`BackgroundTasks` 的优点是简单，缺点是：

- 服务重启任务会丢。
- 多 worker 下任务管理混乱。
- 不适合大规模文件索引。

企业版建议升级为 Redis + RQ 或 Celery。

推荐目录：

```text
app/
  workers/
    __init__.py
    index_worker.py
```

任务设计：

```python
def enqueue_index_job(job_id: str) -> None:
    queue.enqueue("app.workers.index_worker.run_index_job", job_id)
```

学习顺序建议：

1. 先用 `BackgroundTasks` 跑通。
2. 再把 `run_index_job(job_id)` 抽成纯函数。
3. 最后把调用方式换成队列。

## 15. 测试建议

新增测试文件：`tests/integration/test_knowledge_document_api.py`

测试点：

```python
def test_upload_returns_document_and_job(client):
    response = client.post(
        "/api/upload",
        files={"file": ("test.md", b"# Title\n\nhello", "text/markdown")},
    )
    assert response.status_code in (200, 202)
    data = response.json()["data"]
    assert "document_id" in data
    assert "job_id" in data
```

新增 Repository 单元测试：

```python
def test_create_document(db_session):
    repo = KnowledgeRepository(db_session)
    doc = repo.create_document(
        filename="a.md",
        original_filename="a.md",
        file_path="/tmp/a.md",
        file_ext="md",
        file_size=10,
        content_hash="hash",
    )
    assert doc.id
    assert doc.status == DocumentStatus.PENDING
```

## 16. 验收标准

你可以按下面清单检查是否完成：

- 上传文件接口能立即返回，不等待 embedding 完成。
- 数据库里能看到 document 记录。
- 数据库里能看到 index_job 记录。
- 索引成功后 document 状态变为 `indexed`。
- 索引失败后 document 状态变为 `failed`，并保存错误信息。
- Milvus metadata 中包含 `document_id` 和 `chunk_id`。
- 可以查询文档列表。
- 可以对失败文档重新索引。

## 17. 学习重点

这一阶段你要重点掌握：

- FastAPI `Depends` 和数据库 Session 生命周期。
- SQLAlchemy ORM 模型设计。
- 文档、chunk、索引任务的领域建模。
- 为什么 HTTP 请求不适合直接执行耗时索引。
- 为什么 Milvus 不应该承担业务元数据主库职责。
- 幂等索引：同一个 job 重跑不会产生脏数据。

