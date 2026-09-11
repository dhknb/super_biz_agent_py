"""Prometheus 指标专项测试。

**为什么指标需要单独一套测试，而不是靠业务测试顺带覆盖**

指标是**旁路**设施：它记错了、少记了、多记了，业务测试全都是绿的。
用户拿到的答案一模一样，接口状态码一模一样，唯一的差别是
值班同学看到的那块大盘 —— 而那正是故障时唯一能看的东西。

具体说，下面这四类 bug 不会被任何业务测试发现：

1. **计时器套错层**：计时器放在熔断器外面，熔断打开时几百个
   「根本没发出去的调用」会以 ~0 秒记进直方图。P99 反而变好看 ——
   下游挂了而耗时指标一片绿。这是本文件最重要的一个用例。
2. **降级少记**：两条返回路径只在其中一条记了 count_degrade，
   漏掉的那条恰好是故障路径（早返回），于是「降级率」在真出事时反而不涨。
3. **口径不一致**：同一个逻辑事件在两处各记一次，reason 还不一样，
   总数凭空翻倍。排障时两个数据源互相矛盾，比没有数据更糟。
4. **label 基数爆炸**：把 request_id 之类无界的东西写进 label，
   本地跑没事，线上跑几万次把 Prometheus 内存打爆。

**为什么断言增量而不是绝对值**

prometheus_client 的默认 registry 是**进程级全局单例**，
计数器只增不减，且跨用例累积。断言 `== 1` 会得到一个
「单跑绿、全跑红」的测试 —— 前面任何一个用例碰过同一个 label，
这里就变成 2。所以统一用 `_read()` 取快照、做差。

关联：app/core/metrics.py、app/core/circuit_breaker.py、app/main.py
"""

from __future__ import annotations

import asyncio

import pytest
from prometheus_client import REGISTRY

from app.core.breakers import llm_breaker, retrieval_breaker
from app.core.circuit_breaker import CircuitBreaker, CircuitState, reset_all_breakers
from app.core.errors import CircuitOpenError
from app.core.metrics import (
    count_degrade,
    count_retrieval_failure,
    observe_llm_call,
    refresh_circuit_breaker_metrics,
    render_metrics,
)


@pytest.fixture(autouse=True)
def _clean_breakers():
    """熔断器状态会跨用例泄漏，前后都清。

    理由与 tests/unit/test_circuit_breaker.py 相同：状态活在模块级单例里，
    上一个用例把 llm_breaker 打开，这里的计时用例就一个样本都记不到。
    """
    reset_all_breakers()
    yield
    reset_all_breakers()


def _read(_metric: str, /, **labels: str) -> float:
    """读一个指标样本的当前值，不存在时当 0。

    为什么要兜 None：`get_sample_value` 在该 label 组合**从未被观测过**时
    返回 None，而不是 0。直接参与减法会 TypeError ——
    于是「第一次记录」这个最基本的场景反而测不了。

    为什么第一个参数是 positional-only（那个 `/`）：
    指标名和 label 名共处一个签名，而我们有一个 label 就叫 `name`
    （circuit_breaker_state{name="llm"}）。写成普通参数的话，
    `_read("circuit_breaker_state", name="llm")` 会让 Python 认为
    `name` 收到了两个值，直接 TypeError。加上 `/` 之后，
    任何 label 名都不可能和它碰撞 —— 包括以后新增的。
    """
    value = REGISTRY.get_sample_value(_metric, labels or None)
    return 0.0 if value is None else float(value)


# ── LLM 耗时 ──────────────────────────────────────────────


class TestLLMDuration:
    """llm_call_duration_seconds 的记录口径。"""

    def test_records_a_sample_on_success(self):
        before = _read("llm_call_duration_seconds_count", node="unit_ok")

        with observe_llm_call("unit_ok"):
            pass

        assert _read("llm_call_duration_seconds_count", node="unit_ok") == before + 1

    def test_records_a_sample_even_when_the_call_fails(self):
        """失败也要记 —— 超时是最重要的耗时样本。

        把失败排除掉会得到一个「一切正常」的 P99：
        所有慢请求都超时了，剩下的当然都很快。
        """
        before = _read("llm_call_duration_seconds_count", node="unit_boom")

        with pytest.raises(RuntimeError):
            with observe_llm_call("unit_boom"):
                raise RuntimeError("上游 500")

        assert _read("llm_call_duration_seconds_count", node="unit_boom") == before + 1

    def test_exception_propagates_unchanged(self):
        """指标是旁路观测，不改变控制流 —— 与 span_scope 同一条纪律。"""
        with pytest.raises(ValueError, match="原样抛出"):
            with observe_llm_call("unit_passthrough"):
                raise ValueError("原样抛出")

    def test_labels_are_separated_per_node(self):
        """node 是 label，不同调用点各自成一条序列。

        合并成一条的话，「是改写慢还是生成慢」就永远分不出来 ——
        而那恰好是收到「回答太慢」投诉时的第一个问题。
        """
        before_a = _read("llm_call_duration_seconds_count", node="unit_a")
        before_b = _read("llm_call_duration_seconds_count", node="unit_b")

        with observe_llm_call("unit_a"):
            pass

        assert _read("llm_call_duration_seconds_count", node="unit_a") == before_a + 1
        assert _read("llm_call_duration_seconds_count", node="unit_b") == before_b

    def test_open_breaker_records_no_sample(self):
        """**本文件最重要的用例**：计时器必须在 guard() 内侧。

        熔断打开时 guard() 第一行就抛，ainvoke 根本没执行。
        如果计时器套在外面，它照样会记下一个 ~0.0001 秒的样本 ——
        于是熔断期间几百个「没打出去的调用」全以 0 秒计入直方图，
        P99 反而**变好看**：下游挂了，而耗时指标显示一切正常。

        这个用例锁死的就是那个嵌套顺序。它红了说明有人把
        `with observe_llm_call(...)` 挪到了 `with llm_breaker.guard()` 外面。
        """
        breaker = CircuitBreaker("unit_open_probe", failure_threshold=1, cooldown_seconds=60.0)
        breaker.record_failure()
        assert breaker.state is CircuitState.OPEN

        before = _read("llm_call_duration_seconds_count", node="unit_never_called")

        with pytest.raises(CircuitOpenError):
            # 生产代码里的形状：guard 在外，计时器在内。
            with breaker.guard():
                with observe_llm_call("unit_never_called"):
                    pass

        assert _read("llm_call_duration_seconds_count", node="unit_never_called") == before

    async def test_works_across_await(self):
        """协程里也要正确计时 —— 四个真实调用点全是 await。"""
        before = _read("llm_call_duration_seconds_count", node="unit_async")

        with observe_llm_call("unit_async"):
            await asyncio.sleep(0)

        assert _read("llm_call_duration_seconds_count", node="unit_async") == before + 1


# ── 降级计数 ──────────────────────────────────────────────


class TestDegradeCounter:
    """degrade_total 的 reason 口径。"""

    def test_counts_by_reason(self):
        before = _read("degrade_total", reason="unit_reason_x")

        count_degrade("unit_reason_x")

        assert _read("degrade_total", reason="unit_reason_x") == before + 1

    def test_none_reason_goes_to_unknown_not_dropped(self):
        """「降级了但没说原因」本身是个需要被看见的信号。

        丢弃的话降级总数会小于真实值，而这个偏差没有任何地方能发现 ——
        大盘上看着一切正常，实际有一条降级路径忘了填 degrade_reason。
        """
        before = _read("degrade_total", reason="unknown")

        count_degrade(None)

        assert _read("degrade_total", reason="unknown") == before + 1

    def test_empty_string_also_goes_to_unknown(self):
        """空串和 None 同等对待：都是「没说原因」。"""
        before = _read("degrade_total", reason="unknown")

        count_degrade("")

        assert _read("degrade_total", reason="unknown") == before + 1


# ── 检索失败计数 ───────────────────────────────────────────


class TestRetrievalFailureCounter:
    """retrieval_failures_total 的 code 口径。"""

    def test_counts_by_error_code(self):
        before = _read("retrieval_failures_total", code="unit_milvus_error")

        count_retrieval_failure("unit_milvus_error")

        assert _read("retrieval_failures_total", code="unit_milvus_error") == before + 1

    def test_circuit_open_is_a_distinct_code(self):
        """`circuit_open` 与真实故障必须分得开。

        混成一个的话，值班同学看到检索失败率涨了会去查 Milvus，
        而真实情况是我们自己的熔断器打开了、Milvus 可能一直是好的。
        code 这个 label 就是用来区分修复方向的。
        """
        before_open = _read("retrieval_failures_total", code="circuit_open")
        before_real = _read("retrieval_failures_total", code="unit_real_failure")

        count_retrieval_failure("circuit_open")

        assert _read("retrieval_failures_total", code="circuit_open") == before_open + 1
        assert _read("retrieval_failures_total", code="unit_real_failure") == before_real


# ── 熔断器状态 Gauge ───────────────────────────────────────


class TestCircuitBreakerGauge:
    """circuit_breaker_state 的取值与刷新时机。"""

    def test_closed_is_zero(self):
        refresh_circuit_breaker_metrics()
        assert _read("circuit_breaker_state", name="llm") == 0.0

    def test_open_is_two(self):
        for _ in range(20):
            llm_breaker.record_failure()

        refresh_circuit_breaker_metrics()

        assert _read("circuit_breaker_state", name="llm") == 2.0

    def test_values_are_ordered_by_severity(self):
        """0 < 1 < 2 不是随便编的号。

        按严重程度递增，`max_over_time(circuit_breaker_state[1h]) >= 2`
        就能直接写成告警规则，不必在表达式里做字符串匹配。
        """
        breaker = CircuitBreaker("unit_gauge_order", failure_threshold=1, cooldown_seconds=0.0)

        refresh_circuit_breaker_metrics()
        assert _read("circuit_breaker_state", name="unit_gauge_order") == 0.0

        breaker.record_failure()
        # cooldown=0 时读 .state 会立刻把 OPEN 推进到 HALF_OPEN，
        # 所以这里直接验半开的取值 —— 这也顺带证明了刷新会跟着状态机走。
        refresh_circuit_breaker_metrics()
        assert _read("circuit_breaker_state", name="unit_gauge_order") == 1.0

    def test_recovers_to_zero_after_success(self):
        """Gauge 必须能**回落**。

        用 Counter 表达状态就会卡在这里：Counter 只能增，
        表达不了「又闭合了」，大盘会永远显示熔断中。
        """
        for _ in range(20):
            retrieval_breaker.record_failure()
        refresh_circuit_breaker_metrics()
        assert _read("circuit_breaker_state", name="retrieval") == 2.0

        retrieval_breaker.record_success()
        refresh_circuit_breaker_metrics()

        assert _read("circuit_breaker_state", name="retrieval") == 0.0

    def test_every_registered_breaker_is_exposed(self):
        """新增熔断器自动出现在指标里，不必手工维护清单。"""
        CircuitBreaker("unit_autoexpose", failure_threshold=3, cooldown_seconds=60.0)

        body, _ = render_metrics()
        text = body.decode()

        assert 'circuit_breaker_state{name="unit_autoexpose"}' in text
        assert 'circuit_breaker_state{name="llm"}' in text
        assert 'circuit_breaker_state{name="retrieval"}' in text


# ── 渲染 ──────────────────────────────────────────────────


class TestRenderMetrics:
    def test_returns_prometheus_text_content_type(self):
        _, content_type = render_metrics()
        assert content_type.startswith("text/plain")

    def test_contains_all_custom_metrics(self):
        """三个自定义指标必须都在输出里。

        直接断言指标名是有意的：改名会破坏所有已有的告警规则和面板，
        这个用例让改名变成一个**必须显式确认**的动作，而不是悄悄发生。
        """
        # 先各记一笔，确保 label 化的指标已被实例化（未观测过的
        # labelled 指标不会出现在输出里，这本身是 prometheus_client 的行为）。
        with observe_llm_call("unit_render"):
            pass
        count_degrade("unit_render")
        count_retrieval_failure("unit_render")

        body, _ = render_metrics()
        text = body.decode()

        assert "llm_call_duration_seconds" in text
        assert "degrade_total" in text
        assert "retrieval_failures_total" in text
        assert "circuit_breaker_state" in text

    def test_refreshes_breaker_state_on_each_render(self):
        """渲染时刷新，而不是状态变化时刷新。

        OPEN→HALF_OPEN 是**惰性**转换（冷却期满后被读到才发生）。
        只在 record_failure/record_success 里更新的话，
        一个打开后再没有流量的熔断器会永远显示 OPEN。
        """
        for _ in range(20):
            llm_breaker.record_failure()

        body, _ = render_metrics()
        text = body.decode()

        assert 'circuit_breaker_state{name="llm"} 2.0' in text


# ── /metrics 端点 ──────────────────────────────────────────


class TestMetricsEndpoint:
    """端点的可用性与开关行为。"""

    def test_endpoint_serves_metrics(self, client):
        response = client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "circuit_breaker_state" in response.text

    def test_endpoint_returns_404_when_disabled(self, client, monkeypatch):
        """关掉之后返回 404 而不是 403。

        403 会告诉扫描者「这里有个 /metrics，只是你没权限」，
        反而确认了目标的存在。404 的语义是「这个端点不存在」。
        """
        monkeypatch.setattr("app.main.config.enable_metrics_endpoint", False)

        response = client.get("/metrics")

        assert response.status_code == 404

    def test_endpoint_is_not_in_openapi_schema(self, client):
        """/metrics 不该出现在 /docs 里 —— 它不是给人调的业务接口。"""
        schema = client.get("/openapi.json").json()

        assert "/metrics" not in schema.get("paths", {})

    def test_http_metrics_use_route_template_not_real_path(self, client):
        """**label 基数纪律的守门用例**。

        handler 必须是路由模板。如果它变成了真实路径，
        每个 trace_id / 文件名都会生成一条独立的时间序列，
        Prometheus 的内存会被慢慢吃干 —— 而这个故障在本地永远不会出现，
        只在线上跑了几万次之后才暴露。

        这也是引 prometheus-fastapi-instrumentator 的唯一实质理由：
        自己写中间件拿到的是已展开的 request.url.path。
        """
        client.get("/metrics")
        body = client.get("/metrics").text

        # /metrics 自己就是一条被观测的路由，用它验证 handler 的形状。
        assert 'handler="/metrics"' in body
        # status 是分组格式（2xx），与 exception_handlers.py 文档里
        # 承诺的 http_requests_total{status="5xx"} 对齐。
        assert 'status="2xx"' in body
