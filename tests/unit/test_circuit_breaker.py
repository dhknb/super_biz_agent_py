"""熔断器与运行时降级开关专项测试。

**为什么熔断器需要单独一套测试**

熔断器是唯一一个**带跨请求记忆**的降级组件。前面几批做的降级都是无状态的：
同一个输入永远得到同一个输出，出错就降级，与上一个请求无关。
熔断器不是 —— 它的行为取决于「前面几次调用成没成」，
于是多了一类别处不会出现的 bug：

1. **状态机走不回来**：能打开、回不到闭合。表现是下游早就恢复了，
   业务却一直降级，而日志里一条错误都没有（因为根本没去调）。
2. **自锁**：熔断器把自己抛出的拒绝也算成一次下游失败，
   失败计数被自己的拒绝不断刷新，永远到不了闭合。
3. **误伤**：把不该算失败的东西算成失败（协程取消、总预算耗尽），
   于是一个健康的依赖被别人的慢拖到熔断。
4. **惊群**：冷却期一到放行全部请求，把刚喘上气的下游二次打死。

这四条都不是「功能对不对」的问题，而是「状态迁移对不对」的问题，
只能靠显式构造迁移序列来验。

**为什么每个用例都要 reset**

熔断器的状态活在模块级单例里，会跨用例泄漏。
用例 A 把 llm_breaker 打开，用例 B 里所有 LLM 调用就都被跳过 ——
B 失败的原因和 B 测的东西毫无关系，而且**取决于用例执行顺序**，
单跑绿、全跑红，是最难查的一类测试污染。autouse fixture 从根上掐掉它。

关联：app/core/circuit_breaker.py、app/core/breakers.py、app/config.py
"""

import asyncio

import pytest
from langchain_core.documents import Document

from app.agent.rag_v2.nodes import (
    generate_node,
    retrieve_each_node,
    rewrite_node,
    validate_answer_node,
)
from app.config import config
from app.core.breakers import llm_breaker, retrieval_breaker
from app.core.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    all_breakers,
    reset_all_breakers,
)
from app.core.errors import (
    CircuitOpenError,
    DegradeReason,
    RetrievalStatus,
    TotalBudgetExceededError,
    degrade_reason_of,
    error_code_of,
    wrap_llm_exception,
)
from app.models.aiops import AlarmEvent, AlarmSeverity
from app.services.sop_retrieval_service import SopRetrievalService


@pytest.fixture(autouse=True)
def _clean_breakers():
    """每个用例前后都清空全部熔断器状态。

    前后都做而不是只做一次：前面清是防止上一个用例污染，
    后面清是防止本用例污染下一个 —— 包括那些不在本文件里的用例。
    """
    reset_all_breakers()
    yield
    reset_all_breakers()


def _breaker(threshold: int = 3, cooldown: float = 60.0) -> CircuitBreaker:
    """造一个独立熔断器。

    名字带 test_ 前缀且每次都不同,避免覆盖注册表里的生产实例 ——
    构造函数会写 _REGISTRY[name],重名会把 retrieval_breaker 顶掉。
    """
    _breaker.counter = getattr(_breaker, "counter", 0) + 1
    return CircuitBreaker(
        f"test_{_breaker.counter}",
        failure_threshold=threshold,
        cooldown_seconds=cooldown,
    )


def _doc(content: str = "证据内容", doc_id: str = "d1") -> Document:
    return Document(page_content=content, metadata={"id": doc_id})


def _alarm() -> AlarmEvent:
    return AlarmEvent(
        alert_name="HighCPUUsage",
        severity=AlarmSeverity.CRITICAL,
        service="order-api",
        summary="CPU 持续超过 90%",
    )


class _BoomRetriever:
    def retrieve_documents(self, query, top_k=3):
        raise RuntimeError("Milvus 连接超时")


class _OkRetriever:
    def retrieve_documents(self, query, top_k=3):
        return [_doc()]


# ── 状态机 ────────────────────────────────────────────────


class TestStateMachine:
    """CLOSED → OPEN → HALF_OPEN → CLOSED 的完整闭环。

    重点是**回得来**：只验「能打开」的测试会漏掉最坏的那个 bug ——
    熔断器打开后再也不闭合,下游恢复了业务还在降级,
    而且日志里一条错误都没有,因为根本没去调用。
    """

    def test_starts_closed(self):
        assert _breaker().state is CircuitState.CLOSED

    def test_opens_at_threshold_not_before(self):
        breaker = _breaker(threshold=3)

        breaker.record_failure()
        breaker.record_failure()
        # 差一次就到阈值:此刻必须仍然放行。
        # 提前熔断等于把「偶发抖动」当成「持续故障」,
        # 而阈值这个参数存在的意义就是区分这两者。
        assert breaker.state is CircuitState.CLOSED
        assert breaker.allow() is True

        breaker.record_failure()
        assert breaker.state is CircuitState.OPEN
        assert breaker.allow() is False

    def test_success_resets_consecutive_count(self):
        """一次成功清零计数 —— 这是「连续失败」区别于「累计失败」的关键。

        没有这条,阈值 5 的熔断器在一天里零散失败 5 次就会打开,
        而那五次之间有几千次成功 —— 下游明明是健康的。
        """
        breaker = _breaker(threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        breaker.record_failure()

        assert breaker.state is CircuitState.CLOSED

    def test_half_open_after_cooldown(self):
        breaker = _breaker(threshold=1, cooldown=0.0)
        breaker.record_failure()

        # 冷却期为 0,下一次查询就该自动推进到半开。
        assert breaker.state is CircuitState.HALF_OPEN

    def test_stays_open_during_cooldown(self):
        breaker = _breaker(threshold=1, cooldown=60.0)
        breaker.record_failure()

        assert breaker.state is CircuitState.OPEN
        assert breaker.allow() is False

    def test_probe_success_closes_circuit(self):
        breaker = _breaker(threshold=1, cooldown=0.0)
        breaker.record_failure()

        assert breaker.allow() is True  # 这是探针
        breaker.record_success()

        assert breaker.state is CircuitState.CLOSED
        assert breaker.allow() is True

    def test_probe_failure_reopens_immediately(self):
        """探针失败直接回 OPEN,不等再攒够阈值。

        半开态下已经知道「刚才连续失败到熔断」,探针再失败就是
        「还没恢复」的确证 —— 此时要求再失败 N 次才熔断,
        等于在下游明确没恢复的情况下又放 N 个请求去白等。
        """
        # 冷却期刻意取非零：cooldown=0.0 时读一次 `state` 就会顺带把
        # OPEN 推进到 HALF_OPEN（`_maybe_half_open` 的正常行为），
        # 于是「刚熔断时是 OPEN」这个断言必然失败 —— 那是观测动作
        # 自己改变了被观测的状态。用非零冷却 + 手动推进才测得准。
        breaker = _breaker(threshold=5, cooldown=60.0)
        for _ in range(5):
            breaker.record_failure()
        assert breaker.state is CircuitState.OPEN

        # 手动把冷却期推到已满，等价于「等了 60 秒」，但不必真的 sleep。
        breaker._opened_at -= 61.0
        assert breaker.state is CircuitState.HALF_OPEN

        assert breaker.allow() is True  # 探针
        breaker.record_failure()

        # 探针失败 → 立刻回 OPEN，重新开始冷却。
        # 关键在「立刻」：此时连续失败数已是 6，若要求再攒满 5 次，
        # 等于在下游明确没恢复的情况下又放 5 个请求去白等。
        assert breaker.state is CircuitState.OPEN

    def test_reset_returns_to_initial(self):
        breaker = _breaker(threshold=1)
        breaker.record_failure()
        breaker.reset()

        assert breaker.state is CircuitState.CLOSED
        assert breaker.allow() is True


class TestThunderingHerd:
    """半开态只放一个探针。

    这条不测就会在生产上以最难看的方式暴露:冷却期一到,
    积压的几百个请求同时判定「可以试试了」,一起涌向一个
    可能还没恢复的下游,把它二次打死,然后熔断器再打开 ——
    形成一个自激的震荡,而且每一轮都比上一轮打得更狠。
    """

    def test_only_first_caller_gets_the_probe(self):
        breaker = _breaker(threshold=1, cooldown=0.0)
        breaker.record_failure()

        assert breaker.allow() is True
        # 后续调用者全部被拒:探针还在路上,结果未知,
        # 此刻放行第二个请求得不到任何新信息,只增加下游负载。
        assert breaker.allow() is False
        assert breaker.allow() is False

    def test_probe_slot_released_after_result(self):
        """探针有结果之后,槽位要放开,否则半开态会永久卡死。"""
        breaker = _breaker(threshold=5, cooldown=0.0)
        for _ in range(5):
            breaker.record_failure()

        assert breaker.allow() is True  # 探针 1
        breaker.record_failure()  # 失败 → 回 OPEN,冷却 0 → 立刻又半开

        assert breaker.allow() is True  # 探针 2 拿得到槽位


# ── guard() 的异常语义 ─────────────────────────────────────


class TestGuardSemantics:
    """guard() 必须记录成败,但绝不改变控制流。"""

    def test_raises_circuit_open_when_open(self):
        breaker = _breaker(threshold=1, cooldown=60.0)
        breaker.record_failure()

        with pytest.raises(CircuitOpenError):
            with breaker.guard():
                pytest.fail("熔断打开时不该进入代码块")

    def test_records_failure_and_reraises_unchanged(self):
        """异常原样往上抛 —— 调用点现有的 except 分支才能照旧工作。"""
        breaker = _breaker(threshold=1)

        with pytest.raises(RuntimeError, match="原始异常"):
            with breaker.guard():
                raise RuntimeError("原始异常")

        assert breaker.state is CircuitState.OPEN

    def test_records_success_on_clean_exit(self):
        breaker = _breaker(threshold=2)
        breaker.record_failure()

        with breaker.guard():
            pass

        assert breaker.state is CircuitState.CLOSED

    def test_circuit_open_error_does_not_count_as_failure(self):
        """自锁防护:熔断器自己的拒绝不能刷新失败计数。

        没有这条,一旦打开就永不闭合:冷却期满 → 半开 → 探针被
        某个更外层的 guard 拒绝(抛 CircuitOpenError)→ 记一次失败
        → 回到 OPEN → 重新冷却。业务永久降级,而下游一直是好的。
        """
        breaker = _breaker(threshold=2, cooldown=0.0)

        # 手动构造:代码块里抛 CircuitOpenError(比如嵌套的内层熔断)
        with pytest.raises(CircuitOpenError):
            with breaker.guard():
                raise CircuitOpenError("内层熔断", context={})

        # 计数没涨,状态还是闭合
        assert breaker.state is CircuitState.CLOSED
        assert breaker.snapshot()["consecutive_failures"] == 0

    def test_timeout_counts_as_failure(self):
        """超时必须算失败 —— 它正是熔断器最该捕捉的那种故障。

        用 except BaseException 而不是 Exception 的理由就在这:
        asyncio.TimeoutError 在 3.11+ 是 OSError 的别名(算 Exception),
        但真正被取消的协程抛的是 CancelledError(BaseException)。
        这里验的是前者 —— 一次实实在在的白等。
        """
        breaker = _breaker(threshold=1)

        with pytest.raises(asyncio.TimeoutError):
            with breaker.guard():
                raise asyncio.TimeoutError("等待响应超时")

        assert breaker.state is CircuitState.OPEN

    def test_cancellation_does_not_count_as_failure(self):
        """取消不算失败 —— 这是防误伤的核心一条。

        取消的来源通常是总预算耗尽:generate 花了 85 秒,
        validate 刚开始 5 秒就被取消。这笔账记到 validate 的下游头上,
        等于用「别人慢」去熔断一个健康的依赖 ——
        反复几次 LLM 熔断器就打开了,而模型服务一直是好的。
        """
        breaker = _breaker(threshold=1)

        with pytest.raises(asyncio.CancelledError):
            with breaker.guard():
                raise asyncio.CancelledError()

        assert breaker.state is CircuitState.CLOSED
        assert breaker.snapshot()["consecutive_failures"] == 0


# ── 与错误分类体系的衔接 ────────────────────────────────────


class TestErrorClassification:
    """熔断必须落到正确的 code / degrade_reason,否则排障方向被带反。"""

    def test_circuit_open_maps_to_circuit_open_reason(self):
        exc = CircuitOpenError("熔断", context={})
        assert error_code_of(exc) == "circuit_open"
        assert degrade_reason_of(exc) is DegradeReason.CIRCUIT_OPEN

    def test_circuit_open_is_not_retryable(self):
        """熔断期内立刻重试毫无意义 —— 下一次调用同样会被拒。"""
        assert CircuitOpenError("x", context={}).retryable is False

    def test_wrap_llm_exception_preserves_circuit_open(self):
        """回归测试:wrap_llm_exception 不能把 circuit_open 重包成 llm_error。

        它原本写的是 `isinstance(exc, LLMError)` 才放行,
        而 CircuitOpenError 继承 AppError,于是被重新分类成 llm_error ——
        「熔断跳过了这次调用」被记成「模型调用失败了」。
        值班同学会去查模型配额,而模型一直是好的。
        """
        exc = CircuitOpenError("熔断", context={})
        wrapped = wrap_llm_exception(exc)

        assert wrapped is exc
        assert wrapped.code == "circuit_open"
        assert wrapped.degrade_reason is DegradeReason.CIRCUIT_OPEN

    def test_wrap_llm_exception_preserves_total_budget(self):
        """同一个漏洞的另一面:预算耗尽也不能被重包成 llm_error。"""
        exc = TotalBudgetExceededError("预算耗尽")
        wrapped = wrap_llm_exception(exc)

        assert wrapped is exc
        assert wrapped.degrade_reason is DegradeReason.TOTAL_BUDGET_EXCEEDED

    def test_wrap_llm_exception_still_classifies_plain_errors(self):
        """放宽判据之后,普通异常的分类不能跟着松掉。"""
        assert wrap_llm_exception(RuntimeError("500")).code == "llm_error"
        assert wrap_llm_exception(asyncio.TimeoutError()).code == "llm_timeout"


# ── 注册表 ────────────────────────────────────────────────


class TestRegistry:
    def test_production_breakers_registered(self):
        names = {b.name for b in all_breakers()}
        assert {"retrieval", "llm"} <= names

    def test_snapshot_shape(self):
        """/metrics 与排障接口依赖这几个键,少一个就少一块可观测性。"""
        snap = retrieval_breaker.snapshot()
        assert set(snap) == {
            "name",
            "state",
            "consecutive_failures",
            "failure_threshold",
            "cooldown_seconds",
        }
        assert snap["state"] == CircuitState.CLOSED.value

    def test_breakers_use_configured_threshold(self):
        assert (
            retrieval_breaker.snapshot()["failure_threshold"]
            == config.circuit_breaker_failure_threshold
        )
        assert (
            llm_breaker.snapshot()["cooldown_seconds"]
            == config.circuit_breaker_cooldown_seconds
        )

    def test_retrieval_and_llm_are_separate(self):
        """两个熔断器必须独立:Milvus 挂了不该切断 LLM。

        合成一个是很自然的偷懒(「反正都是外部依赖」),
        后果是两个健康状况完全独立的依赖被绑在一起 ——
        向量库宕机时连降级答案都生成不出来,纯误伤。
        """
        retrieval_breaker.record_failure()
        assert llm_breaker.snapshot()["consecutive_failures"] == 0


# ── 接入点:SOP 检索 ──────────────────────────────────────


class TestSopRetrievalIntegration:
    def test_failure_returns_failed_status(self):
        service = SopRetrievalService(retriever=_BoomRetriever())
        evidence, status = service.retrieve_sop_evidence(_alarm())

        assert evidence == []
        assert status is RetrievalStatus.FAILED

    def test_repeated_failures_trip_the_breaker(self):
        """连续失败到阈值后,后续调用不再碰下游 —— 熔断器的全部意义。"""
        service = SopRetrievalService(retriever=_BoomRetriever())
        threshold = config.circuit_breaker_failure_threshold

        for _ in range(threshold):
            service.retrieve_sop_evidence(_alarm())

        assert retrieval_breaker.state is CircuitState.OPEN

    def test_open_breaker_still_degrades_not_raises(self):
        """熔断打开后仍然返回 FAILED,而不是把异常抛给调用方。

        这条是熔断器接入是否成功的判据:熔断的意义是**更快地降级**,
        如果它把降级变成了 500,就是净负收益。
        """
        service = SopRetrievalService(retriever=_BoomRetriever())
        for _ in range(config.circuit_breaker_failure_threshold):
            service.retrieve_sop_evidence(_alarm())

        # 换一个健康的检索器,但熔断器还是开的 —— 不该有任何调用发出去
        service_ok = SopRetrievalService(retriever=_OkRetriever())
        evidence, status = service_ok.retrieve_sop_evidence(_alarm())

        assert status is RetrievalStatus.FAILED
        assert evidence == []

    def test_switch_off_returns_failed_not_empty(self):
        """开关关闭要报 FAILED,不能报 EMPTY。

        EMPTY 的含义是「知识库里确实没有」—— 那是业务结论。
        开关关着的时候我们根本没去查,说「库里没有」就是撒谎,
        会让人跑去补 SOP 文档,而实际要做的是把开关打开。
        """
        service = SopRetrievalService(retriever=_OkRetriever())
        original = config.enable_sop_retrieval
        try:
            config.enable_sop_retrieval = False
            evidence, status = service.retrieve_sop_evidence(_alarm())
        finally:
            config.enable_sop_retrieval = original

        assert status is RetrievalStatus.FAILED
        assert evidence == []

    def test_switch_off_does_not_touch_retriever(self):
        """关掉就是真的不调 —— 否则开关只是个装饰品。"""
        service = SopRetrievalService(retriever=_BoomRetriever())
        original = config.enable_sop_retrieval
        try:
            config.enable_sop_retrieval = False
            service.retrieve_sop_evidence(_alarm())
        finally:
            config.enable_sop_retrieval = original

        # 检索器会抛异常;没被调用,所以熔断计数是 0
        assert retrieval_breaker.snapshot()["consecutive_failures"] == 0


# ── 接入点:rag_v2 节点 ───────────────────────────────────


class TestRagNodesIntegration:
    @pytest.mark.asyncio
    async def test_retrieve_each_records_circuit_open_as_failure_detail(self):
        """熔断打开时,检索分支返回一条 code=circuit_open 的失败明细。

        重点是 code:排障时「Milvus 连不上」和「我们自己熔断了」
        指向完全不同的动作,压成同一个 retrieval_error 就分不出来。
        """
        retrieval_breaker.record_failure()  # 走到阈值
        for _ in range(config.circuit_breaker_failure_threshold):
            retrieval_breaker.record_failure()

        patch = await retrieve_each_node({"query": "CPU 高怎么查"})

        assert patch["documents"] == []
        assert patch["retrieve_failures"][0]["code"] == "circuit_open"

    @pytest.mark.asyncio
    async def test_rewrite_falls_back_to_single_query_on_circuit_open(self):
        """改写被熔断 → 退回原始问题当唯一子查询,不是 500。

        这个节点原本**没有兜底**,一炸整张图就崩。
        它其实是全链路里最该降级的一步:改写的价值是「多几个检索角度」,
        不是「能不能检索」—— 拿原始问题照样能召回证据。
        """
        for _ in range(config.circuit_breaker_failure_threshold):
            llm_breaker.record_failure()

        patch = await rewrite_node({"question": "CPU 高怎么查"})

        assert patch["sub_queries"] == ["CPU 高怎么查"]
        assert patch["degrade_reasons"] == [DegradeReason.CIRCUIT_OPEN.value]

    @pytest.mark.asyncio
    async def test_generate_degrades_with_circuit_open_reason(self):
        for _ in range(config.circuit_breaker_failure_threshold):
            llm_breaker.record_failure()

        patch = await generate_node(
            {"question": "CPU 高怎么查", "deduped_documents": [_doc()]}
        )

        assert patch["degrade_reasons"] == [DegradeReason.CIRCUIT_OPEN.value]
        assert patch["answer"]  # 有可用的降级答案

    @pytest.mark.asyncio
    async def test_validate_fails_open_on_circuit_open(self):
        """质检被熔断 → 答案照发,标 validated=False。

        fail-open 而不是 fail-closed:质检是第二道防线,
        拿「质检器不可用」去拦用户的答案,等于用假原因(证据不够)
        掩盖真原因(质检挂了)。
        """
        for _ in range(config.circuit_breaker_failure_threshold):
            llm_breaker.record_failure()

        patch = await validate_answer_node(
            {
                "question": "CPU 高怎么查",
                "answer": "先看 top 再看火焰图",
                "deduped_documents": [_doc()],
            }
        )

        assert patch["validation"]["blocked"] is False
        assert patch["validation"]["validated"] is False
        assert patch["degrade_reasons"] == [DegradeReason.CIRCUIT_OPEN.value]
        # 答案没有被换成保守回答
        assert "answer" not in patch


# ── 运行时开关:证据校验 ─────────────────────────────────


class TestValidationSwitch:
    @pytest.mark.asyncio
    async def test_switch_off_passes_answer_through(self):
        """关掉一道**校验**不该让**答案**消失。"""
        original = config.enable_answer_validation
        try:
            config.enable_answer_validation = False
            patch = await validate_answer_node(
                {
                    "question": "CPU 高怎么查",
                    "answer": "先看 top 再看火焰图",
                    "deduped_documents": [_doc()],
                }
            )
        finally:
            config.enable_answer_validation = original

        assert "answer" not in patch
        assert patch["validation"]["blocked"] is False
        assert patch["validation"]["validated"] is False

    @pytest.mark.asyncio
    async def test_switch_off_uses_feature_disabled_reason(self):
        """降级原因是 feature_disabled,不是 llm_error。

        复用 llm_error 会把值班同学送去查模型配额和鉴权,
        那边一切正常,排障就此卡住 —— 又一次用假原因掩盖真原因。
        """
        original = config.enable_answer_validation
        try:
            config.enable_answer_validation = False
            patch = await validate_answer_node(
                {
                    "question": "q",
                    "answer": "a",
                    "deduped_documents": [_doc()],
                }
            )
        finally:
            config.enable_answer_validation = original

        assert patch["degrade_reasons"] == [DegradeReason.FEATURE_DISABLED.value]

    @pytest.mark.asyncio
    async def test_structural_checks_still_apply_when_switched_off(self):
        """开关只关掉那个 LLM 调用,不该顺手关掉不花钱的结构性检查。

        「答案为空」和「没有证据」的判断不依赖被关掉的模型调用,
        该继续生效 —— 否则关一个开关会连带放过一批本该拦住的空答案。
        """
        original = config.enable_answer_validation
        try:
            config.enable_answer_validation = False
            patch = await validate_answer_node(
                {"question": "q", "answer": "", "deduped_documents": [_doc()]}
            )
        finally:
            config.enable_answer_validation = original

        assert patch["validation"]["blocked"] is True

    @pytest.mark.asyncio
    async def test_disabled_and_failed_share_the_same_shape(self):
        """两条 fail-open 路径的 validation 形状必须完全一致。

        读侧靠 validated=False 统计「有多少答案没过质检就发出去了」,
        两处字段对不齐就会漏统计 —— 这也是把它抽成
        _unvalidated_validation 的理由。
        """
        state = {
            "question": "q",
            "answer": "a",
            "deduped_documents": [_doc()],
        }

        original = config.enable_answer_validation
        try:
            config.enable_answer_validation = False
            disabled = await validate_answer_node(state)
        finally:
            config.enable_answer_validation = original

        for _ in range(config.circuit_breaker_failure_threshold):
            llm_breaker.record_failure()
        failed = await validate_answer_node(state)

        assert set(disabled["validation"]) == set(failed["validation"])
