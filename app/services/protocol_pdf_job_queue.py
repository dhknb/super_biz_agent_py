"""RQ enqueue helper for protocol PDF ingestion."""

from rq.job import Job

from app.core.task_queue import protocol_pdf_queue
from app.workers.protocol_pdf_worker import run_protocol_pdf_ingestion_job


def enqueue_protocol_pdf_ingestion_job(job_id: str) -> Job:
    return protocol_pdf_queue.enqueue(
        run_protocol_pdf_ingestion_job,
        job_id,
        job_timeout=600,
        result_ttl=86400,
        failure_ttl=86400,
    )
