"""RQ enqueue helper for protocol PDF ingestion."""

from rq.job import Job

from app.core.job_context import enqueue_with_request_id
from app.core.job_failure import default_job_retry
from app.core.task_queue import protocol_pdf_queue
from app.workers.protocol_pdf_worker import run_protocol_pdf_ingestion_job


def enqueue_protocol_pdf_ingestion_job(job_id: str) -> Job:
    # 同 index_job_queue：把 request_id 透传给 worker，让上传接口的日志
    # 和几分钟后 worker 里的解析失败日志能用同一个 id 串起来。
    #
    # retry 同样在入队时声明。这条链路上「值得重试」与「不值得重试」的对比特别鲜明：
    # LLM 限流（429）重试一次就过了，而 PDF 本身损坏重试三次是三次同样的失败。
    # 两者的区分交给 worker 侧的 apply_retry_policy，这里只负责给出额度。
    return enqueue_with_request_id(
        protocol_pdf_queue,
        run_protocol_pdf_ingestion_job,
        job_id,
        job_timeout=600,
        result_ttl=86400,
        failure_ttl=86400,
        retry=default_job_retry(),
    )
