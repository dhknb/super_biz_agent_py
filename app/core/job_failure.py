"""RQ 任务失败处置：状态落盘 + 重试裁决。

## 为什么需要这一层

改之前两个 worker 的 except 块是同一个形状（`app/workers/index_worker.py`
与 `app/workers/protocol_pdf_worker.py`）：

    except Exception as exc:
        db.rollback()
        job = repo.get_index_job(job_id)
        if job is not None:
            job.status = FAILED
            job.error_message = str(exc)
            db.commit()          # ← 问题就在这一行
        raise

这里藏着一个很隐蔽的 bug：**except 块里又 commit 了一次**。

如果这次 commit 也失败，抛出的新异常会顺着 except 块往外走，把原始异常
**顶掉**。Python 的 traceback 里确实会留一句 "During handling of the above
exception, another exception occurred"，但 RQ 落到 `job.exc_info` 的、
告警里弹出来的、你排障第一眼看到的，全是后面那个
`OperationalError: server closed the connection`。真正的原因
（比如「PDF 第 3 页解析失败」）沉在下面，非常容易被漏掉。

而且这不是罕见情况，恰恰是**最常见**的组合：原始异常本身就是数据库连接断开
或网络故障时，同一个 session 上的 commit 必然也失败。也就是说 ——
越是基础设施故障，越会丢失真实原因，而那正是最需要看清原因的时候。

第二个后果同样致命：失败状态没写进去，job 行永远停在 RUNNING，
document 永远停在 INDEXING。用户在页面上看着「索引中」转了三天，
没有任何人知道它其实早就死了。

## 处置

1. 用一个**独立的短生命周期 session** 写失败状态。原 session 的连接可能
   已经废了（连 rollback 都可能抛），在同一个连接上抢救等于在漏水的船上补漏。
2. 整段抢救逻辑包在 try 里，**任何二次失败只记日志，绝不外抛**。
   原始异常必须原样 raise 出去 —— 那才是要修的东西。
3. 重试裁决与「是否可重试」的判定复用 `app/core/errors.py` 的
   `is_retryable`，不在这里重写一遍分类逻辑（DRY）。
"""

from __future__ import annotations

from contextlib import suppress
from typing import Callable

from loguru import logger
from rq import Retry, get_current_job
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.errors import error_code_of, is_retryable

# 默认重试策略：最多 3 次，间隔递增。
#
# 为什么间隔要递增而不是固定：瞬时故障（Milvus 选主、网络抖动、LLM 限流）
# 的恢复时间是不确定的。固定 10 秒重试三次，总共只覆盖 30 秒窗口；
# 递增到 60 秒能覆盖到分钟级抖动，而代价只是失败任务晚一点定性。
#
# 为什么是 3 次而不是更多：超过 3 次基本说明不是抖动而是真故障，
# 继续重试只是在推迟暴露问题，同时占着 worker 不干别的活。
_DEFAULT_RETRY_MAX = 3
_DEFAULT_RETRY_INTERVALS = [10, 30, 60]


def default_job_retry() -> Retry:
    """入队侧用的默认重试策略。

    定义成函数而不是模块级常量：`Retry` 实例携带可变状态，
    两个队列共享同一个对象是自找麻烦。
    """
    return Retry(max=_DEFAULT_RETRY_MAX, interval=_DEFAULT_RETRY_INTERVALS)


def apply_retry_policy(exc: BaseException, *, job_id: str) -> bool:
    """裁决这次失败是否值得让 RQ 重排队，并在不值得时就地否决。

    返回 True 表示 RQ 还会重试，False 表示这次失败是终局。

    ## 为什么需要裁决

    RQ 的 `Retry(max=N)` 是**无条件**的：只要任务抛异常就重排，不区分
    「网络抖动」和「PDF 文件损坏」。后者重试三次的结果是三次一模一样的失败 ——
    白烧三倍资源，还把最终失败推迟了一分多钟，用户盯着「索引中」多等一轮。

    所以这里按异常类型裁决：不可重试的异常直接把 `retries_left` 清零，
    RQ 在 `handle_job_failure` 里看到 `should_retry` 为 False，
    就走正常的失败落盘路径，不再重排。

    ## 为什么改内存属性是可靠的（照 rq 2.9 源码确认过）

    - `Job.perform()` 里 `_job_stack.push(self)`，所以 `get_current_job()`
      拿到的就是 worker 正在执行的**同一个 Job 对象**，不是副本。
    - `Worker.perform_job()` 的 except 分支把这个对象直接传给
      `handle_job_failure(job=job, ...)`。
    - `handle_job_failure` 里 `retry = job.should_retry and not job_is_stopped`，
      而 `should_retry` 读的是内存属性 `retries_left`，中间没有
      `job.refresh()`，不会从 Redis 重新取值。

    这三点连起来，任务函数里的赋值就能确定地影响重试决策。

    脱离 RQ 上下文时（单测直接调函数）拿不到 job，此时只返回判定结果，
    不做任何副作用 —— 函数在测试里是纯的。
    """
    retryable = is_retryable(exc)

    job = None
    with suppress(Exception):  # pragma: no cover - RQ 上下文缺失不该影响任务本身
        job = get_current_job()

    if job is None:
        return retryable

    retries_left = getattr(job, "retries_left", None) or 0

    if not retryable and retries_left > 0:
        # 清零而不是设成 None：`should_retry` 判的是 `is not None and > 0`，
        # 两种写法都能生效，但 0 更能表达「额度用完了」而非「没配过重试」。
        job.retries_left = 0
        logger.warning(
            f"任务失败且不可重试，已取消剩余 {retries_left} 次重试: "
            f"job_id={job_id}, error_code={error_code_of(exc)}"
        )
        return False

    return retryable and retries_left > 0


def format_failure_message(exc: BaseException, *, will_retry: bool) -> str:
    """拼装落库用的失败原因。

    带上 `error_code` 是为了可聚合 —— 自由文本没法做「同一类失败出现了多少次」
    的统计，而 `error_code` 是有限枚举。

    带上重试状态是为了**不误导看页面的人**：状态列写着 FAILED，但后台其实
    还排着两次重试。不说清楚的话，值班同学会立刻开始手工排查一个
    十秒后可能自己就好了的问题。
    """
    code = error_code_of(exc)
    suffix = "，RQ 将自动重试" if will_retry else ""
    return f"[{code}] {exc}{suffix}"


def record_job_failure(
    write_status: Callable[[Session], None],
    *,
    job_id: str,
) -> None:
    """用独立 session 写入任务失败状态；二次失败只记日志，绝不外抛。

    `write_status` 拿到的是一个全新的 Session，需要在里面用这个 session
    重新构造 repository 并重新取行 —— 原 session 里那些 ORM 对象属于
    一个可能已经废掉的连接，不能跨 session 复用。

    这个函数的全部价值在于「绝不 raise」：抢救失败是坏消息，
    但让抢救过程中的异常顶掉原始异常，是更坏的消息。
    """
    try:
        session = SessionLocal()
    except Exception as exc:
        # 连建 session 都失败（连接池耗尽 / 数据库彻底不可达），
        # 那就只能认了 —— 至少把这件事记下来。
        logger.error(f"任务失败状态落盘失败（无法建立 session）: job_id={job_id}: {exc}")
        return

    try:
        write_status(session)
        session.commit()
    except Exception as secondary:
        # 关键：这里绝不 raise。
        logger.error(
            f"任务失败状态落盘失败（原始异常已保留）: job_id={job_id}, "
            f"二次异常={type(secondary).__name__}: {secondary}"
        )
        with suppress(Exception):
            session.rollback()
    finally:
        # close 在连接已断时同样可能抛。若让它从 finally 里逃出去，
        # 就又变成了「二次异常顶掉原始异常」—— 正是本模块要消灭的那个 bug。
        with suppress(Exception):
            session.close()
