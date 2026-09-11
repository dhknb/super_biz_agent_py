"""RQ worker task for protocol PDF ingestion."""

from contextlib import suppress
from datetime import UTC, datetime

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
from app.models.protocol_ingestion import (
    ProtocolIngestionJobStatus,
    ProtocolIngestionStatus,
)
from app.repositories.protocol_ingestion_repository import ProtocolIngestionRepository
from app.services.protocol_pdf_ingestion_service import protocol_pdf_ingestion_service


@with_job_request_id
def run_protocol_pdf_ingestion_job(job_id: str) -> None:
    db = SessionLocal()
    repo = ProtocolIngestionRepository(db)
    try:
        job = repo.get_job(job_id)
        if job is None:
            logger.warning(f"协议 PDF 入库任务不存在: {job_id}")
            return

        job.status = ProtocolIngestionJobStatus.RUNNING
        job.started_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()

        protocol_pdf_ingestion_service.process_ingestion(db, job.ingestion_id)

        job.status = ProtocolIngestionJobStatus.SUCCEEDED
        job.finished_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()
        logger.info(f"协议 PDF 入库任务完成: job_id={job_id}")

    except Exception as exc:
        # 失败状态用**独立的短连接 session** 落库，不复用上面这个 db。
        #
        # 原来的写法是 `db.rollback()` 之后继续用同一个 db 写 FAILED 再 commit。
        # 问题在于走到这里的常见原因本身就是连接断了(数据库重启、网络抖动、
        # 事务被服务端 kill)。此时 rollback/commit 会抛出**第二个**异常，
        # 它从 except 块里逃出去，把原始的 exc 顶掉 —— 最后 RQ 记下的
        # 失败原因是 `OperationalError: server closed the connection`，
        # 真正的原因(比如 PDF 解析失败)彻底消失。
        #
        # 顺序也很关键：先裁决重试、再落库。
        # error_message 里要写清「还会不会重试」，否则页面上写着 FAILED、
        # 后台其实还排着两次重试，值班同学会去排查一个可能自己就好了的问题。
        will_retry = apply_retry_policy(exc, job_id=job_id)
        message = format_failure_message(exc, will_retry=will_retry)

        # rollback 本身也可能抛（连接已断）。它只是尽力释放事务，
        # 失败不该影响后面的抢救动作。
        with suppress(Exception):
            db.rollback()

        def _write_failed(session: Session) -> None:
            # 必须用新 session 重新构造 repo、重新取行：原 session 里的
            # ORM 对象绑在那个可能已经断掉的连接上，跨 session 复用会在 flush 时炸。
            failure_repo = ProtocolIngestionRepository(session)
            failed_job = failure_repo.get_job(job_id)
            if failed_job is None:
                return
            failed_job.status = ProtocolIngestionJobStatus.FAILED
            failed_job.error_message = message
            failed_job.finished_at = datetime.now(UTC).replace(tzinfo=None)

            ingestion = failure_repo.get_ingestion(failed_job.ingestion_id)
            if ingestion is not None:
                ingestion.status = ProtocolIngestionStatus.FAILED
                ingestion.error_message = message

        record_job_failure(_write_failed, job_id=job_id)

        logger.opt(exception=exc).error(
            f"协议 PDF 入库任务失败: job_id={job_id}, "
            f"error_code={error_code_of(exc)}, will_retry={will_retry}"
        )
        # 必须原样 raise：RQ 靠这个异常判定失败并决定是否重排队。
        raise
    finally:
        with suppress(Exception):
            db.close()
