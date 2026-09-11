"""span 的内存收集与落盘 —— 在图跑完之后一次性写库。

## 为什么是「内存收集 + 一次 flush」，不是节点直接写库

节点直接写库有三个硬伤：

1. **拿不到 trace_id。** `ChatRunTrace` 的主键是**跑完落库时**才生成的。
   节点执行期间那行记录还不存在，span 无处挂靠。
2. **节点拿不到 Session。** LangGraph 的节点签名是 `state -> patch`，
   要塞一个 db session 进去，就得让它穿过 state 或改所有节点的签名 ——
   而 state 是要被序列化的，往里塞 Session 是自找麻烦。
3. **写库次数爆炸。** 一次请求 8 个节点（含 4 条并行检索），
   每个节点各 commit 一次就是 8 次往返。这正是 AIOps 时间线犯过的错。

所以改成：跑的时候只往内存 list 里 append（几乎零成本），
跑完拿到 trace_id 之后一次 `bulk_save_objects` + 一次 commit。

## ContextVar 在并行分支下的语义（这段是本模块能成立的关键）

`asyncio.to_thread` 和 LangGraph 内部 spawn 任务时，都会
`contextvars.copy_context()`。拷贝的是**变量到对象的绑定关系**，
不是对象本身 —— 父子上下文里的 `_SPAN_COLLECTOR` 指向**同一个 list**。

推论有两条，方向相反，必须都记住：
- 子上下文里 `list.append(...)` → 父上下文**看得见**（同一个对象被改了）✓
- 子上下文里 `_SPAN_COLLECTOR.set(...)` → 父上下文**看不见**（只改了子的绑定）✗

所以本模块的纪律是：**只在请求入口 set 一次，之后全程只 append。**
`record_span` 里绝不调 `.set()`。违反这条，并行检索分支的 span 就会
静默丢失 —— 而且是那种「本地单测能过、并发下才丢」的 bug。

## 为什么收集失败绝不能影响主流程

span 是可观测性设施，不是业务功能。用户要的是答案。
「因为记不下耗时，所以这次回答失败了」是荒谬的因果。
所以本模块所有对外函数都不抛异常 —— 与 job_failure 同一个原则。
"""

from __future__ import annotations

import time
from contextlib import suppress
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any, Iterator

from loguru import logger
from sqlalchemy.orm import Session

from app.core.errors import error_code_of
from app.core.request_context import get_request_id_or_none
from app.models.chat_run_span import ChatRunSpan, SpanStatus

# 值是「本次运行已记录的 span 明细」列表；None 表示当前不在被追踪的运行里
# （比如脚本直调节点、单测），此时 record_span 直接空转。
_SPAN_COLLECTOR: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "chat_run_span_collector", default=None
)


def start_span_collection() -> Token:
    """开始收集本次运行的 span，返回用于还原的 token。

    必须在请求入口调用（而不是在图内部）：见模块文档里 ContextVar 的语义 ——
    只有在**父上下文**里 set，后续 fan-out 出去的所有子上下文才能共享同一个 list。
    """
    return _SPAN_COLLECTOR.set([])


def collected_spans() -> list[dict[str, Any]]:
    """取本次已收集的 span 明细（副本，防止调用方误改内部状态）。"""
    spans = _SPAN_COLLECTOR.get()
    return list(spans) if spans else []


def reset_span_collection(token: Token) -> None:
    """还原到收集开始前的状态。

    用 reset(token) 而不是 set(None)：嵌套调用时后者会把外层的收集器也清掉。
    """
    with suppress(Exception):
        _SPAN_COLLECTOR.reset(token)


def record_span(
    node: str,
    *,
    duration_ms: int,
    status: SpanStatus,
    started_at: datetime,
    payload: dict[str, Any] | None = None,
) -> None:
    """记录一条 span。不在收集上下文里时静默跳过。

    注意这里只 append，不 set —— 这是并行分支下不丢数据的前提。
    """
    spans = _SPAN_COLLECTOR.get()
    if spans is None:
        return
    spans.append(
        {
            "node": node,
            "status": status,
            "started_at": started_at,
            "duration_ms": duration_ms,
            "payload": payload,
        }
    )


class span_scope:
    """给一段代码计时并在退出时记一条 span。

    用类而不是 `@contextmanager` 装饰的生成器，是因为要暴露
    `set_payload` / `mark_degraded` 给 with 体内调用 ——
    节点跑完才知道自己召回了几条、有没有降级。

    异常路径：记 ERROR 并带上 error_code，然后**原样放行异常**。
    span 是旁路观测，不改变控制流。
    """

    def __init__(self, node: str, payload: dict[str, Any] | None = None):
        self._node = node
        self._payload: dict[str, Any] = dict(payload or {})
        self._status = SpanStatus.OK
        self._started_at = datetime.now(UTC).replace(tzinfo=None)
        # 用 perf_counter 而不是两个 datetime 相减：
        # 后者受系统时钟调整（NTP 校时）影响，可能算出负数耗时。
        self._perf_start = time.perf_counter()

    def set_payload(self, **values: Any) -> None:
        self._payload.update(values)

    def mark_degraded(self, reason: str | None = None) -> None:
        """标记「节点没抛异常，但也没干成活」。"""
        self._status = SpanStatus.DEGRADED
        if reason:
            self._payload.setdefault("degrade_reason", reason)

    def __enter__(self) -> "span_scope":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        duration_ms = int((time.perf_counter() - self._perf_start) * 1000)
        if exc is not None:
            self._status = SpanStatus.ERROR
            self._payload["error_code"] = error_code_of(exc)
            self._payload["error"] = str(exc)[:500]
        with suppress(Exception):  # 观测失败不能影响业务
            record_span(
                self._node,
                duration_ms=duration_ms,
                status=self._status,
                started_at=self._started_at,
                payload=self._payload or None,
            )
        return False  # 不吞异常


def flush_spans(
    db: Session,
    *,
    trace_id: str,
    spans: list[dict[str, Any]] | None = None,
) -> int:
    """把收集到的 span 一次性写库，返回写入条数。

    失败只记日志，绝不抛 —— 主流程（答案已经生成、trace 已经落库）
    不该因为写不进耗时明细而失败。
    """
    payload_spans = collected_spans() if spans is None else spans
    if not payload_spans:
        return 0

    request_id = get_request_id_or_none()
    try:
        rows = [
            ChatRunSpan(
                trace_id=trace_id,
                request_id=request_id,
                node=str(item.get("node") or "unknown"),
                status=item.get("status") or SpanStatus.OK,
                started_at=item.get("started_at") or datetime.now(UTC).replace(tzinfo=None),
                duration_ms=int(item.get("duration_ms") or 0),
                payload=item.get("payload"),
            )
            for item in payload_spans
        ]
        db.add_all(rows)
        db.commit()
        return len(rows)
    except Exception as exc:
        logger.warning(f"span 落盘失败（不影响主流程）: trace_id={trace_id}: {exc}")
        with suppress(Exception):
            db.rollback()
        return 0


def iter_span_payloads(spans: list[ChatRunSpan]) -> Iterator[dict[str, Any]]:
    """把 ORM 行转成 API 可序列化的 dict。"""
    for span in spans:
        yield {
            "node": span.node,
            "status": span.status.value if hasattr(span.status, "value") else str(span.status),
            "started_at": span.started_at.isoformat() if span.started_at else None,
            "duration_ms": span.duration_ms,
            "payload": span.payload,
        }
