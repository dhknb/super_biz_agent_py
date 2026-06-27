"""Index job enqueue helpers."""

from rq.job import Job

from app.core.task_queue import index_queue
from app.workers.index_worker import run_index_job


def enqueue_index_job(job_id: str) -> Job:
    return index_queue.enqueue(
        run_index_job,
        job_id,
        job_timeout="30m",
        failure_ttl=7 * 24 * 60 * 60,
        result_ttl=7 * 24 * 60 * 60,
    )
