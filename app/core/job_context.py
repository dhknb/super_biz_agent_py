"""RQ 任务的 request_id 传递。

这是 request_id 全链路的最后一段：
    中间件 → ContextVar → 日志 → trace → **RQ meta → worker 日志**

为什么这一段单独存在：
ContextVar 只在**同一个进程的同一个执行上下文**里有效。
入队时 HTTP 请求进程持有 rid，而真正干活的是另一台机器上的 worker 进程 ——
两者之间唯一的通道是 Redis 里的那条 job 记录。
所以必须显式地「序列化进去、反序列化出来」：
入队时写进 `job.meta`，worker 执行时读回来重新绑定到自己的 ContextVar。

不这么做的后果：上传一个文档触发索引任务，API 侧日志有 rid、
worker 侧日志全是 "-"。文档索引失败时，你手里有 API 的 rid，
却没法把它和 worker 的报错日志连起来 —— 而失败恰恰都发生在 worker 侧。

为什么放在独立模块而不是 task_queue.py：
task_queue.py 在导入时会构造 Redis 连接与 Queue 单例。worker 模块只需要
「读回 rid」这一个能力，不该为此被迫拉起队列单例 —— 那会让 worker 的单测
必须先准备好 redis 配置。这里只依赖 rq 的类型与 get_current_job。
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

from rq import Queue, get_current_job
from rq.job import Job

from app.core.request_context import get_request_id_or_none, request_id_scope

# job.meta 里存放 request_id 的键名。定义成常量而非字面量：
# 入队侧和 worker 侧必须用同一个键，写错一个字母就会静默失效
# —— 没有任何报错，只是 rid 永远取不到。
JOB_META_REQUEST_ID = "request_id"

F = TypeVar("F", bound=Callable[..., Any])


def enqueue_with_request_id(queue: Queue, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Job:
    """入队并把当前 request_id 带进 job.meta。

    用法与 `queue.enqueue` 完全一致，只是多做一件事。保持签名兼容是刻意的：
    调用点只需把 `queue.enqueue(...)` 换成 `enqueue_with_request_id(queue, ...)`，
    其余参数（job_timeout / result_ttl / retry）原样透传。

    显式传入的 meta 会被保留，只在缺 request_id 键时补上（setdefault 语义）。
    """
    meta: dict[str, Any] = dict(kwargs.pop("meta", None) or {})
    meta.setdefault(JOB_META_REQUEST_ID, get_request_id_or_none())
    return queue.enqueue(func, *args, meta=meta, **kwargs)


@contextmanager
def job_request_id_scope() -> Iterator[str]:
    """worker 侧：从当前 job 的 meta 恢复 request_id，退出作用域自动还原。

    取不到 job（比如直接以普通函数调用做测试）或 meta 里没有 rid 时，
    `request_id_scope(None)` 会生成一个新的 —— 后台任务哪怕是独立触发的，
    也该有自己的追踪标识，否则它那一段日志等于没有 id。
    """
    request_id: str | None = None
    try:
        job = get_current_job()
    except Exception:  # pragma: no cover - RQ 上下文缺失时不该影响任务本身
        job = None
    if job is not None:
        request_id = (job.meta or {}).get(JOB_META_REQUEST_ID)

    with request_id_scope(request_id) as bound:
        yield bound


def with_job_request_id(func: F) -> F:
    """装饰 RQ 任务函数，让函数体内所有日志自动带上原始请求的 rid。

    为什么用装饰器而不是在每个 worker 函数里手写 `with`：
    worker 函数体都是「try 一大段 / except 落库失败状态」的结构，
    手写 with 会让整块代码再缩进一层，diff 噪音大且容易漏。
    装饰器是零侵入的切面（DRY）。

    RQ 通过「模块路径 + 函数名」反序列化任务函数，functools.wraps
    保留了 __name__ / __qualname__ / __module__，所以入队与执行都能正确解析。
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with job_request_id_scope():
            return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
