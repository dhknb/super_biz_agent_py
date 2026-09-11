"""请求上下文：request_id 全链路传递。

为什么需要它：
现在的日志长这样 —— `[rag_v2.rewrite] 原始问题: ...`，只有节点名。
10 个并发请求的日志会完全交织在一起，想把某一个请求的完整链路捞出来是不可能的。
而 `ChatRunTrace.id` 只在最后落库时才存在，请求执行期间的日志根本无从关联。

解决办法是给每个请求一个 id，从 HTTP 入口一路带到：
    中间件 → ContextVar → 每条日志 → 落库的 trace → RQ 异步任务 → worker 日志

ContextVar 的关键性质：它是**协程隔离**的。
asyncio 里多个请求在同一线程上交替执行，普通全局变量会被互相覆盖，
而 ContextVar 的值绑定在当前执行上下文上，各协程互不干扰。
`asyncio.to_thread` 派生的线程会复制当前 context，所以线程池里也读得到。
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator

# 请求头名称：对外契约，网关/前端可传入自己的 id 以便跨系统串联
REQUEST_ID_HEADER = "X-Request-ID"

# 没有 request_id 时的占位符。
# 用 "-" 而不是空字符串，是为了让日志列宽稳定、肉眼容易扫。
NO_REQUEST_ID = "-"

# default 必须给值：脱离请求上下文的调用（脚本、worker 启动阶段、测试）
# 直接读也不会抛 LookupError。
_request_id: ContextVar[str] = ContextVar("request_id", default=NO_REQUEST_ID)


def new_request_id() -> str:
    """生成一个新的 request_id。

    用 uuid4 的 hex 前 16 位：足够避免碰撞，又比完整 uuid 短得多 ——
    日志每行都要带它，短一半就是可观的可读性收益。
    """
    return uuid.uuid4().hex[:16]


def get_request_id() -> str:
    """读取当前上下文的 request_id，不存在时返回 NO_REQUEST_ID。"""
    return _request_id.get()


def get_request_id_or_none() -> str | None:
    """读取当前 request_id，脱离请求上下文时返回 None。

    专门给「落库」用，和 `get_request_id()` 的区别只在没有值时的表示：
    - 日志用 `"-"`：展示层需要固定列宽，肉眼扫起来才整齐。
    - 数据库用 `NULL`：NULL 的语义是「没有这个值」，而 `"-"` 会被当成一个
      真实取值参与 `GROUP BY` 与关联查询 —— 所有无上下文的记录会被聚成
      一个假的「请求 -」分组，污染统计。

    这个函数存在的意义是让这条判断只有一处实现（DRY）：
    chat_run_trace 和 aiops_diagnosis 两个 repository 都要落 request_id，
    各自写一遍 `None if rid == NO_REQUEST_ID else rid` 就是等着两边跑偏。
    """
    rid = _request_id.get()
    return None if rid == NO_REQUEST_ID else rid


def set_request_id(request_id: str | None) -> Token[str]:
    """写入 request_id，返回 Token 以便精确还原。

    返回 Token 而非依赖 set 覆盖，是因为嵌套场景下
    只有 reset(token) 能把上下文还原到**进入前**的确切状态。
    """
    return _request_id.set(request_id or NO_REQUEST_ID)


def reset_request_id(token: Token[str]) -> None:
    """还原到 set_request_id 之前的状态。"""
    _request_id.reset(token)


@contextmanager
def request_id_scope(request_id: str | None = None) -> Iterator[str]:
    """在一段作用域内绑定 request_id，退出时自动还原。

    给 RQ worker 和后台任务用：
        with request_id_scope(job.meta.get("request_id")):
            ...   # 这里面所有日志都会带上原始请求的 id

    传 None 会生成新 id —— 独立触发的后台任务也该有自己的追踪标识，
    否则它的日志会全部落在 "-" 上，等于没有。
    """
    rid = request_id or new_request_id()
    token = set_request_id(rid)
    try:
        yield rid
    finally:
        reset_request_id(token)
