"""轻量熔断器 —— 连续失败到阈值就停止呼叫下游，冷却后放一个探针试水。

**为什么需要它**

现有的防护是超时 + 重试 + 降级：调用挂了等 60 秒，重试 2 次，然后降级。
这套东西在**偶发**故障上很好用，在**持续**故障上是灾难。

设想 Milvus 宕机，QPS 10：

    每个请求 → 检索超时 60s（× 重试 2 次）→ 降级返回
    60 秒里堆积 600 个请求，每个都占着一个线程池 worker、
    一个数据库连接、一个 SSE 连接，全都在等一个**已经确定不会回来**的响应。

问题不在单个请求 —— 它最终会降级，行为是对的。问题在于这 600 次尝试
每一次都要先白等 60 秒才知道「哦，还是不行」。下游已经明确挂了，
我们却坚持每来一个请求就再去撞一次墙，而墙上的坑是我们自己的资源。

熔断器把「重复确认一个已知的坏消息」这件事的成本降到零：
连续失败 N 次之后，直接不打了，立刻走降级路径。等冷却期过去，
放**一个**探针试试 —— 好了就恢复，还不行就再等一轮。

顺带还有一个常被忽略的收益：不打了，下游就少了 10 QPS 的压力。
下游往往正因为过载才挂，而所有客户端一起坚持重试，
恰好在它最虚弱的时候给它最大的负载。熔断是给下游留出恢复窗口。

**为什么不引 pybreaker**

它是个成熟的库，但我们要的是六十行逻辑：一个计数器、一个时间戳、
三个状态。引一个依赖来换这六十行，代价是多一个需要跟版本、
跟安全公告、跟 Python 版本兼容性的第三方包，而它的绝大部分功能
（Redis 共享状态、事件监听器、排除异常列表）本项目一个都用不上。
YAGNI 不只是「别写用不到的代码」，也包括「别引用不到的库」。

**为什么用「连续失败 N 次」而不是「窗口内失败率 X%」**

失败率是更精细的指标，但它需要配一条「最小样本数」规则，否则
第一个请求失败就是 1/1 = 100% 失败率，熔断器立刻打开 ——
对本项目这种低 QPS 的内部服务，这种误触发会非常频繁。

而「连续失败 N 次」天然自带最小样本：想触发就必须真的失败 N 次，
中间任何一次成功都会清零。它对偶发抖动免疫，对持续故障灵敏，
恰好是我们想要的形状。代价是对「一半成功一半失败」的半死状态
不敏感 —— 那种情况留给指标告警去发现，不该由熔断器自动处置，
因为此时下游仍在提供一半的服务能力，切断它反而是净损失。

**状态机**

    CLOSED ──连续失败达到阈值──> OPEN ──冷却期满──> HALF_OPEN
      ^                            ^                   │
      │                            └──探针失败─────────┤
      └──────────探针成功────────────────────────────┘

HALF_OPEN 只放**一个**探针过去。这是关键：如果放行所有请求，
冷却期一到就是几百个请求同时涌向一个可能还没恢复的下游，
把它二次打死（惊群）。一个探针的成本是有界的，信息量却是完整的。

关联：app/core/errors.py 的 CircuitOpenError 与 DegradeReason.CIRCUIT_OPEN。
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import contextmanager
from enum import StrEnum
from typing import Iterator, NamedTuple

from loguru import logger

from app.core.errors import CircuitOpenError


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class _Admission(NamedTuple):
    """一次准入决定。

    比单纯的 bool 多带一个 is_probe。guard() 需要它,因为在
    「不记成功也不记失败」的退出路径上,探针名额必须被归还 ——
    理由见 guard() 的文档,那是一个会让熔断器永久拒绝的坑。
    """

    allowed: bool
    is_probe: bool


class CircuitBreaker:
    """单个下游依赖的熔断器。

    线程安全：用 `threading.Lock` 而不是 `asyncio.Lock`。
    原因是调用点跨两种执行模型 —— 检索走 `asyncio.to_thread`（真线程），
    LLM 调用走协程。`asyncio.Lock` 保护不了线程池里的并发访问，
    而 `threading.Lock` 在两种场景下都正确。临界区只有几个整数赋值，
    不做 IO，所以在协程里持锁也不会阻塞事件循环。
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int,
        cooldown_seconds: float,
    ) -> None:
        self.name = name
        self._failure_threshold = max(1, failure_threshold)
        self._cooldown_seconds = max(0.0, cooldown_seconds)

        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        # monotonic 而不是 time.time()：后者会被 NTP 校准往回拨，
        # 那会让「冷却是否结束」的判断出现负数间隔，熔断器卡在 OPEN 出不来。
        self._opened_at = 0.0
        # HALF_OPEN 期间是否已经有探针在路上。防惊群靠的就是这个标志。
        self._probe_in_flight = False

        _REGISTRY[name] = self

    # ── 查询 ────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        """当前状态。会顺带把「冷却期已满的 OPEN」推进到 HALF_OPEN。"""
        with self._lock:
            self._maybe_half_open()
            return self._state

    def snapshot(self) -> dict[str, object]:
        """给指标 / 排障接口用的状态快照。"""
        with self._lock:
            self._maybe_half_open()
            return {
                "name": self.name,
                "state": self._state.value,
                "consecutive_failures": self._consecutive_failures,
                "failure_threshold": self._failure_threshold,
                "cooldown_seconds": self._cooldown_seconds,
            }

    # ── 决策 ────────────────────────────────────────────────

    def allow(self) -> bool:
        """是否放行这次调用。

        HALF_OPEN 下只有第一个调用者拿到 True —— 它就是那个探针。

        **裸用这个方法要自己负责收尾**:拿到 True 的调用者必须最终调用
        record_success / record_failure 之一,否则半开期的探针名额不会归还,
        熔断器会永久停在「半开且已有探针在路上」的状态,拒绝一切请求。
        推荐直接用 guard(),它把这件事处理干净了。
        """
        return self._admit().allowed

    def _admit(self) -> _Admission:
        """准入判断的唯一实现,顺带告诉调用方「你是不是那个探针」。"""
        with self._lock:
            self._maybe_half_open()

            if self._state is CircuitState.CLOSED:
                return _Admission(allowed=True, is_probe=False)

            if self._state is CircuitState.OPEN:
                return _Admission(allowed=False, is_probe=False)

            # HALF_OPEN：只放一个探针
            if self._probe_in_flight:
                return _Admission(allowed=False, is_probe=False)
            self._probe_in_flight = True
            return _Admission(allowed=True, is_probe=True)

    def _release_probe(self) -> None:
        """归还探针名额,但**不动**状态和失败计数。

        用在那些「这次调用既不算成功也不算失败」的退出路径上
        (目前只有协程取消)。只清标志不改状态,是因为我们对下游
        的健康状况一无所知 —— 探针被取消了,什么都没测出来。
        留在 HALF_OPEN 等下一个请求来当探针,才是诚实的处置。
        """
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._probe_in_flight = False

    def record_success(self) -> None:
        """记一次成功。任何状态下的成功都直接闭合电路。

        为什么探针一次成功就闭合，而不要求连续成功 M 次：
        探针成功说明下游能应答了。此时若继续限流，被拒的请求
        本来是能成功的 —— 那是我们自己造成的失败。真的没恢复，
        下一轮连续失败会重新打开，代价只是一个冷却周期。
        """
        with self._lock:
            was = self._state
            self._consecutive_failures = 0
            self._probe_in_flight = False
            self._state = CircuitState.CLOSED

        if was is not CircuitState.CLOSED:
            logger.info(f"熔断器已闭合（下游恢复）: name={self.name}, 原状态={was.value}")

    def record_failure(self) -> None:
        """记一次失败。达到阈值或探针失败都会打开电路。"""
        with self._lock:
            self._consecutive_failures += 1
            failures = self._consecutive_failures

            if self._state is CircuitState.HALF_OPEN:
                # 探针失败：直接回到 OPEN，重新开始冷却。
                self._trip_locked()
                opened = True
            elif failures >= self._failure_threshold:
                self._trip_locked()
                opened = True
            else:
                opened = False

        if opened:
            logger.error(
                f"熔断器已打开: name={self.name}, 连续失败={failures}/"
                f"{self._failure_threshold}, 冷却={self._cooldown_seconds}s"
            )

    def reset(self) -> None:
        """强制回到初始状态。给测试与人工干预用。"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._opened_at = 0.0
            self._probe_in_flight = False

    # ── 使用姿势 ─────────────────────────────────────────────

    @contextmanager
    def guard(self) -> Iterator[None]:
        """包住一次下游调用：熔断时抛 CircuitOpenError，正常时自动记录成败。

        为什么做成抛异常而不是返回布尔值：调用点原本就有
        `except Exception → 降级` 的分支（比如 SOP 检索的 FAILED 分支）。
        抛一个 AppError 子类进去，那条分支不用改一行就能正确处理熔断 ——
        `error_code_of` 会给出 `circuit_open`，日志和降级原因都是对的。

        注意 CircuitOpenError **不**计入失败：这次调用根本没发出去，
        把它算作下游的一次失败，会让熔断器自己制造的拒绝不断刷新
        失败计数，形成「一旦打开永不闭合」的自锁。
        """
        admission = self._admit()
        if not admission.allowed:
            raise CircuitOpenError(
                f"熔断器打开，已跳过 {self.name} 调用",
                context={"breaker": self.name},
            )
        try:
            yield
        except CircuitOpenError:
            # 内层还有一个熔断器,且它是打开的 —— 这次调用同样没发出去。
            # 不计失败(理由同上),但如果自己是探针,名额必须还回来。
            if admission.is_probe:
                self._release_probe()
            raise
        except asyncio.CancelledError:
            # 取消**不计失败**，但原样往上抛。
            #
            # 取消的来源通常是总预算耗尽（chat_total_budget_seconds），
            # 而预算是整条链路一起花掉的：generate 花了 85 秒，
            # validate 刚开始 5 秒就被取消 —— 这笔账记到 validate 的
            # 下游头上，等于用「别人慢」去熔断一个健康的依赖。
            # 反复几次就把 LLM 熔断器打开了，而 DashScope 一直是好的。
            #
            # 但名额要还:探针被取消时既没 record_success 也没 record_failure,
            # 而那两个方法是唯一会清 _probe_in_flight 的地方。不还的话
            # 熔断器就永久停在「半开 + 探针在路上」——allow() 恒为 False,
            # 下游恢复了也再没有第二个探针能去发现它。
            #
            # 这条路径在总预算耗尽时会**成批**触发(fan-out 的分支一起被取消),
            # 所以它不是理论上的边角情况。
            if admission.is_probe:
                self._release_probe()
            raise
        except BaseException:
            # 用 BaseException 而不是 Exception：真正的超时（TimeoutError）
            # 同样是「这次调用没成功」，熔断器该看见它。
            # 但仍然原样往上抛，不改变控制流。
            self.record_failure()
            raise
        else:
            self.record_success()

    # ── 内部（均需持锁调用）──────────────────────────────────

    def _maybe_half_open(self) -> None:
        if self._state is not CircuitState.OPEN:
            return
        if time.monotonic() - self._opened_at < self._cooldown_seconds:
            return
        self._state = CircuitState.HALF_OPEN
        self._probe_in_flight = False
        logger.warning(f"熔断器进入半开，放行一个探针: name={self.name}")

    def _trip_locked(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        self._probe_in_flight = False


# ── 全局注册表 ────────────────────────────────────────────────
# 存在的唯一目的：让 /metrics 与排障接口能枚举所有熔断器的状态，
# 不必在每个模块里手工维护一份清单。

_REGISTRY: dict[str, CircuitBreaker] = {}


def all_breakers() -> list[CircuitBreaker]:
    return list(_REGISTRY.values())


def reset_all_breakers() -> None:
    """重置全部熔断器。测试用 —— 熔断器有跨用例残留的状态。"""
    for breaker in _REGISTRY.values():
        breaker.reset()
