"""节点级耗时追踪 —— 把「记了终点，丢了过程」补成一条时间线。

## 为什么需要这张表

现有的 `chat_run_traces` 记的是**终态**：最后的答案、用过哪些文档、
质检分数。它回答不了排障时最常问的那个问题 ——「慢在哪一步」。

一次 chat_v2 请求串行调三次 LLM（rewrite → generate → validate），
中间还并行跑 N 条检索。用户报「这个问题要等 40 秒」，
现在的 trace 只能告诉你「确实跑了 40 秒，答案是这个」。
到底是改写慢、Milvus 慢、还是生成慢，只能去翻日志按时间戳人肉对齐 ——
而日志是按行输出的，并行的 4 条检索分支交织在一起，根本对不出每条的耗时。

有了 span 就变成一句 SQL：
    SELECT node, avg(duration_ms), max(duration_ms) FROM chat_run_spans
    WHERE trace_id = ? GROUP BY node;

## 为什么不直接上 OpenTelemetry

OTel 的完整方案要引 SDK、起 Collector、部署 Jaeger/Tempo，
换来的是跨服务的分布式追踪。而当前需求是**单进程内**的节点耗时，
四个字段（node / duration_ms / status / payload）就够了。

所以这里借用 OTel 的**概念**（span 是一段有始有终、可嵌套归属的工作），
但不引它的**实现**（YAGNI）。真到了需要跨服务串联的那天，
这张表的字段能平移成 OTel 的 span 属性，不算白做。

## 为什么没有 parent_span_id

OTel 的 span 有父子关系，用来还原调用树。这里刻意不做：
当前图是「rewrite → fan-out 检索 → dedup → generate → validate」，
是一条扁平序列 + 一层扇出，用 `node` 名字加 `started_at` 排序
就能看清全貌。加上 parent_span_id 就要在节点间传递 span 上下文，
而节点是纯函数（拿 state 返回 patch），传递链路会污染所有节点签名。
代价大于收益。
"""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class SpanStatus(StrEnum):
    """span 的三态。

    为什么要有 DEGRADED 这个中间态：
    OK / ERROR 两态分不出「节点自己兜住了故障并给了降级结果」这种情况 ——
    比如检索分支挂了但被 retrieve_each 接住，节点**返回成功**（没抛异常），
    可它其实没干成活。只有 OK/ERROR 的话这次会被记成 OK，
    于是「检索失败率」这个指标永远是 0，而用户明明拿到了残缺证据。
    """

    OK = "ok"
    DEGRADED = "degraded"
    ERROR = "error"


class ChatRunSpan(Base):
    __tablename__ = "chat_run_spans"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    # ondelete="CASCADE"：span 是 trace 的从属明细，trace 没了 span 就是垃圾数据。
    # 让数据库来级联，而不是在应用层记得先删 span —— 应用层总会有人忘。
    trace_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("chat_run_traces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # request_id 冗余一份在这里，不是为了省 join。
    # 是为了让「日志里捞到一个 request_id」能直接查 span，
    # 而不必先查 trace 拿 id 再查 span —— 排障时少一跳就是少一次犯错机会。
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    node: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[SpanStatus] = mapped_column(
        Enum(SpanStatus, values_callable=lambda items: [item.value for item in items]),
        nullable=False,
        default=SpanStatus.OK,
        index=True,
    )
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow_naive)
    # 存毫秒整数而不是「结束时间」：
    # 耗时是要被 avg / max / 分位数聚合的，存成两个时间戳每次查询都得相减。
    # 也不用浮点秒 —— 毫秒整数在 SQL 里聚合不会有精度漂移。
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 每个节点的关键计数放这里（文档数、子查询数、失败数、错误码……）。
    # 用 JSON 而不是给每种计数开一列：不同节点关心的量完全不同，
    # 开成列的话表会长出十几个大部分为 NULL 的字段。
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_naive, index=True)

    trace = relationship("ChatRunTrace")
