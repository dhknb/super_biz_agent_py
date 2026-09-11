"""Index job enqueue helpers."""

from rq.job import Job

from app.core.job_context import enqueue_with_request_id
from app.core.job_failure import default_job_retry
from app.core.task_queue import index_queue
from app.workers.index_worker import run_index_job


def enqueue_index_job(job_id: str) -> Job:
    # 走 enqueue_with_request_id 而不是 index_queue.enqueue：
    # 把当前 HTTP 请求的 request_id 写进 job.meta，worker 侧再取回来。
    # 否则 API 日志有 rid、worker 日志全是 "-"，而索引失败恰恰都发生在 worker 侧。
    #
    # retry 必须在**入队时**声明：RQ 把重试额度存进 job 记录，worker 执行时
    # 只能减少它，不能凭空补上。也就是说没有这个参数的任务，一次失败就是终局，
    # 哪怕失败原因只是 Milvus 正在选主这种几秒钟就恢复的抖动。
    # 至于「哪些失败值得重试」，由 worker 侧的 apply_retry_policy 按异常类型裁决。
    return enqueue_with_request_id(
        index_queue,
        run_index_job,
        job_id,
        job_timeout="30m",
        failure_ttl=7 * 24 * 60 * 60,
        result_ttl=7 * 24 * 60 * 60,
        retry=default_job_retry(),
    )
