"""Prometheus 指标 —— 项目里**唯一**的指标定义处。

## 为什么需要它:第三批的活其实还没干完

第三批把 HTTP 状态码改成了说真话的(见 app/core/exception_handlers.py),
那个模块的文档里列了一串「只看状态码」的消费者:

    Prometheus 的 `http_requests_total{status="5xx"}`

问题是 —— 那个消费者**当时并不存在**。状态码说了真话,但没有任何东西在听。
本模块加上 `Instrumentator`,那一行注释才从「设想」变成「事实」。

## 为什么指标和日志不能互相替代

它们回答的是两个不同的问题,而且**互相答不了**:

- 日志回答「**这一次**为什么失败」——  有 request_id,能顺着 trace 一路看到
  是 rewrite 挂了还是 validate 挂了。但你没法从日志问出
  「过去一小时降级率是多少」,那需要遍历几百万行做聚合。
- 指标回答「**整体**现在什么形状」—— 降级率、P99 耗时、熔断器开着没有。
  但它答不了「张三那次查询为什么没给出 SOP」,因为指标里没有任何个体信息。

值班的实际动线是:**指标发现异常 → 日志定位个体**。
只有指标就知道「有问题」却不知道问题在哪;只有日志就得先有人报障才会去看。

## 为什么全部指标定义集中在这一个模块

跟 app/core/breakers.py 同一个理由,但多一条硬约束:
**prometheus_client 的默认 registry 是全局单例,同名指标注册两次直接抛
`Duplicated timeseries in CollectorRegistry`。**

散在各模块里定义的话,两个模块各写一个 `Counter("degrade_total", ...)`
就会在 import 期炸掉 —— 而且是那种「本地跑单个测试没事、
一起 import 才炸」的故障。集中定义之后,这个错误在物理上不可能发生。

## label 的基数纪律(这是指标最容易出的事故)

每一个 label 值的组合都会在 Prometheus 里生成一条**独立的时间序列**,
每条序列常驻内存。所以 label 的取值集合必须是**有界且很小**的。

本模块的 label 全部来自枚举或固定字符串:
- `node`   → 五个节点名 + 少量服务名,有界
- `reason` → DegradeReason 枚举,十来个值,有界
- `code`   → 错误码,来自异常类名转换,有界

**绝对不能**做 label 的:request_id、trace_id、用户问题、文件名、
告警实例 IP。这些是无界的,加进去就是内存泄漏 —— 那类信息属于日志。

同理,HTTP 指标必须按**路由模板**聚合(`/api/aiops/trace/{trace_id}`),
而不是真实路径。手写中间件拿到的 `request.url.path` 是展开后的路径,
每个 trace_id 一条序列,几万次查询就把 Prometheus 打爆。
`prometheus-fastapi-instrumentator` 会从 Starlette 的 route 对象取模板 ——
这是引这个库、而不是自己写一个中间件的**唯一实质理由**。
(对比:熔断器我们没引 pybreaker,因为那六十行没有这种坑。)

## 为什么用 Histogram 而不是 Gauge/Summary 记耗时

- Gauge 只记「最后一次」,平均值和 P99 都算不出来,故障时最没用。
- Summary 在客户端算分位数,多实例部署时**无法聚合**
  (两个实例各自的 P99 求平均没有数学意义)。
- Histogram 存的是分桶计数,可以跨实例相加再算分位数。

代价是要预先定好桶边界。下面的桶按本项目的实际形状选:
LLM 单次超时是 60s(config.llm_timeout_seconds),总预算 90s,
所以桶要一直铺到 60 以上,否则所有慢调用都挤在 `+Inf` 里,
而「慢到什么程度」恰好是排障最需要的信息。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from app.core.circuit_breaker import CircuitState, all_breakers
from app.core.errors import CircuitOpenError, error_code_of

# ── 耗时 ────────────────────────────────────────────────────

# 桶边界:密在 1~10s(正常区间,要能看出 P50/P90 的移动),
# 稀在 30~120s(异常区间,只需要知道「非常慢」的量级)。
_LLM_BUCKETS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0)

llm_call_duration_seconds = Histogram(
    "llm_call_duration_seconds",
    "LLM 单次调用耗时(秒),按调用点区分",
    labelnames=("node",),
    buckets=_LLM_BUCKETS,
)

# ── 工具调用 ─────────────────────────────────────────────────

# 桶边界比 LLM 那套密在低位。直接复用 _LLM_BUCKETS 是不行的:
# 它最小刻度 0.5s,而工具调用(向量检索、MCP 查一段日志)正常在
# 几十毫秒到几秒,九成样本会挤进第一个桶,P50 的移动完全看不出来 ——
# 而「工具开始变慢」恰好是链路一出问题时最早出现的症状。
# 上限仍然铺到 60s,因为工具背后是 MCP 的 HTTP 调用,也会超时。
_TOOL_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)

tool_call_duration_seconds = Histogram(
    "tool_call_duration_seconds",
    "工具调用耗时(秒),按工具名区分",
    labelnames=("tool",),
    buckets=_TOOL_BUCKETS,
)

# label 基数:tool × code。工具名来自本地工具(2 个)加 MCP 服务暴露的
# 工具表 —— 都是**代码里写死的**,不随请求变化,所以有界。
# 反过来说,绝对不能把「模型传进来的工具入参」当 label:
# 那是用户问题的一部分,无界。
tool_failures_total = Counter(
    "tool_failures_total",
    "工具调用失败次数,按工具名与错误码区分",
    labelnames=("tool", "code"),
)

# 为什么工具「列举」失败要单独一个指标,不并进 tool_failures_total:
#
# 两者的修复方向不同,而这正是本项目划分指标与降级原因的判据。
# 列举失败 = MCP 进程没起来/连不上,处置是去起进程;
# 调用失败 = 进程活着但那个工具报错,处置是去看工具实现或入参。
# 并进一个指标的话,值班同学看到 tool_failures_total 上涨,
# 得先翻日志才知道该找运维还是找开发。
#
# 还有一个更实际的理由:列举失败是**静默**的 —— 链路会降级成
# 「只有本地工具」继续正常回答,用户和调用方都看不出异常。
# 没有这个指标,一个进程从此只剩本地工具的状态在监控上完全不可见。
#
# label 基数:server 来自配置(2 个),code 来自 error_code_of,都有界。
mcp_tool_load_failures_total = Counter(
    "mcp_tool_load_failures_total",
    "MCP 工具列举失败次数,按服务名与错误码区分",
    labelnames=("server", "code"),
)

# 为什么列举**耗时**也要记,光有失败计数不够:
#
# 列举发生在请求路径里(agent 构建时 await),而且 provider 用一把
# asyncio.Lock 把并发请求串起来了 —— 一次 30 秒的连接超时就是
# 30 秒的用户可见延迟,还会连带堵住同时到达的其他请求。
# 失败计数只告诉你「列举失败了」,回答不了「用户等了多久」。
#
# 复用 _TOOL_BUCKETS 而不是新开一套:列举也是连接 + 握手 + list 这类
# HTTP 往返,量级和工具调用同一个数量级(正常几十毫秒到几秒,
# 坏的时候顶到超时上限),没有理由再造一组边界。
mcp_tool_load_duration_seconds = Histogram(
    "mcp_tool_load_duration_seconds",
    "MCP 工具列举耗时(秒),按服务名区分",
    labelnames=("server",),
    buckets=_TOOL_BUCKETS,
)

# ── 降级与失败 ───────────────────────────────────────────────

degrade_total = Counter(
    "degrade_total",
    "降级发生次数,按原因区分",
    labelnames=("reason",),
)

retrieval_failures_total = Counter(
    "retrieval_failures_total",
    "检索失败次数,按错误码区分",
    labelnames=("code",),
)

# ── 熔断器 ──────────────────────────────────────────────────

# 用 Gauge 而不是 Counter:熔断状态是一个**当前值**(现在开着还是关着),
# 不是一个累计量。Counter 只能增,表达不了「又闭合了」。
circuit_breaker_state = Gauge(
    "circuit_breaker_state",
    "熔断器状态:0=closed, 1=half_open, 2=open",
    labelnames=("name",),
)

# 状态到数值的映射。为什么按「严重程度」递增而不是随便编号:
# 这样 `max_over_time(circuit_breaker_state[1h]) >= 2` 就能直接写成告警规则,
# 不需要在告警表达式里做字符串匹配。
_STATE_VALUES = {
    CircuitState.CLOSED: 0,
    CircuitState.HALF_OPEN: 1,
    CircuitState.OPEN: 2,
}


def observe_llm_duration(node: str, seconds: float) -> None:
    """记一次 LLM 调用耗时。

    失败的调用也要记 —— 超时是最重要的耗时样本。把它排除掉会得到一个
    「一切正常」的 P99:所有慢请求都超时了,剩下的当然都很快。
    """
    llm_call_duration_seconds.labels(node=node).observe(max(0.0, seconds))


@contextmanager
def observe_llm_call(node: str) -> Iterator[None]:
    """给一次 LLM 调用计时,**无论成败都记**。

    为什么抽成上下文管理器:四个 LLM 调用点(rewrite / generate /
    validate_answer / first_response)都需要计时,各写一遍
    `t0 = perf_counter()` ... `observe(perf_counter() - t0)` 是四份重复,
    而且异常路径上那一行极容易漏 —— 漏掉的恰好是超时那次,
    也就是唯一真正需要被记下来的那次(DRY)。

    用 try/finally 而不是 try/except:异常原样往上抛,
    指标是旁路观测,不改变控制流 —— 与 span_scope 同一个纪律。
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        observe_llm_duration(node, time.perf_counter() - start)


def count_tool_failure(tool: str, code: str) -> None:
    """记一次工具调用失败。"""
    tool_failures_total.labels(tool=tool, code=code or "unknown").inc()


def count_mcp_tool_load_failure(server: str, code: str) -> None:
    """记一次 MCP 工具列举失败(区别于工具调用失败,理由见指标定义处)。"""
    mcp_tool_load_failures_total.labels(server=server, code=code or "unknown").inc()


@contextmanager
def observe_mcp_tool_load(server: str) -> Iterator[None]:
    """给一次 MCP 工具列举计时,**无论成败都记**。

    为什么这个只计时,而下面的 observe_tool_call 把计时和失败计数
    合成了一个 —— 看着像不一致,其实是同一条「记账点唯一」的结论:

    列举只有一个调用点(MCPToolProvider._load_server),而那里的
    except 块**本来就必须存在** —— 它要拿错误码填 last_error_code
    (供 unavailable_servers() 上报),要写 last_failure_at 进冷却期。
    失败计数放在那个非可选的 except 里,不存在「异常路径上漏一笔」的风险。
    反过来,如果这里也数一遍,就变成两个记账点,每次列举失败计两次 ——
    那比漏记更糟,因为指标看起来是自洽的,只是数值全错。

    工具调用的处境相反:两个形态不同的调用点,且它们的 except 块
    并非必须存在(拦截器完全可以直接透传异常),所以合成一个才安全。
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        mcp_tool_load_duration_seconds.labels(server=server).observe(
            max(0.0, time.perf_counter() - start)
        )


@contextmanager
def observe_tool_call(tool: str) -> Iterator[None]:
    """给一次工具调用计时并在失败时计数,**耗时无论成败都记**。

    为什么把「计时」和「记失败」合成一个上下文管理器,而 LLM 那边是分开的:

    工具调用点有两处形态完全不同的实现 —— 本地 @tool 装饰的同步函数,
    和 MCP 的异步拦截器。如果计时和失败计数分成两个 API,这两处各要写
    一个 try/except 把它们串起来,而**异常路径上漏掉一笔**是必然会发生的
    (漏掉的通常是失败计数,于是耗时曲线上有一个 60s 的尖峰,
    failures 却是 0,看指标的人会以为「只是慢,没有失败」)。
    合成一个之后,两笔账在物理上不可能只记一半 —— 这就是「记账点唯一」。

    错误码走 error_code_of:分类逻辑在 app/core/errors.py 只有一处实现,
    这里不自己 isinstance 一遍(DRY)。

    CircuitOpenError 单独排除在失败计数之外。理由跟熔断器自己
    不把 CircuitOpenError 计入失败是同一条:这次调用根本没发出去,
    工具和它的下游都没被碰到。计进去的话,熔断器一打开,
    `tool_failures_total` 会按请求量疯涨,看起来像工具在大面积报错,
    而真相是我们主动跳过了它 —— 那笔账应该记在
    `circuit_breaker_state` 和 `degrade_total{reason="circuit_open"}` 上。
    耗时仍然记(那是一次真实发生的、极快的返回)。

    用 try/except/raise 而不是吞掉异常:指标是旁路观测,不改控制流。
    """
    start = time.perf_counter()
    try:
        yield
    except CircuitOpenError:
        raise
    except BaseException as exc:
        # BaseException 而不是 Exception:超时(TimeoutError)和取消
        # 同样是「这次工具调用没成功」,指标该看见它们。
        count_tool_failure(tool, error_code_of(exc))
        raise
    finally:
        tool_call_duration_seconds.labels(tool=tool).observe(
            max(0.0, time.perf_counter() - start)
        )


def count_degrade(reason: str | None) -> None:
    """记一次降级。

    reason 为 None 时归到 `unknown`,而不是丢弃:
    「降级了但没说原因」本身就是一个需要被看见的信号 ——
    它意味着某条降级路径忘了填 degrade_reason。丢弃的话,
    降级总数会小于真实值,而这个偏差没有任何地方能发现。
    """
    degrade_total.labels(reason=reason or "unknown").inc()


def count_retrieval_failure(code: str) -> None:
    """记一次检索失败。"""
    retrieval_failures_total.labels(code=code or "unknown").inc()


def refresh_circuit_breaker_metrics() -> None:
    """把所有熔断器的当前状态刷进 Gauge。

    为什么在**抓取时**刷新,而不是在状态变化时刷新:

    熔断器的状态转换有一条是**惰性**的 —— OPEN 到 HALF_OPEN 不是由
    某个事件触发,而是「冷却期满之后被人读到」才发生
    (见 CircuitBreaker._maybe_half_open)。如果只在 record_failure /
    record_success 里更新 Gauge,那么一个打开后再没有流量的熔断器
    会永远显示 OPEN,即使它早就该进半开了。

    抓取时刷新则顺带调用了 snapshot(),那个调用本身会推进状态机,
    所以指标和真实状态永远一致。
    """
    for breaker in all_breakers():
        snapshot = breaker.snapshot()
        state = CircuitState(str(snapshot["state"]))
        circuit_breaker_state.labels(name=breaker.name).set(_STATE_VALUES[state])


def render_metrics() -> tuple[bytes, str]:
    """生成 Prometheus 文本格式的指标,返回 `(body, content_type)`。

    这里先刷一次熔断器状态,理由见 refresh_circuit_breaker_metrics。
    """
    refresh_circuit_breaker_metrics()
    return generate_latest(), CONTENT_TYPE_LATEST
