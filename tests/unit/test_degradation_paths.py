"""降级路径专项测试 —— 七条「出了问题但仍要给出可用结果」的链路。

**为什么要单独一个文件**

降级逻辑的测试和功能测试的关注点不一样。功能测试问「正常时对不对」，
降级测试问的是三件更容易被改坏的事：

1. **降不降**：出错时是抛异常炸穿整条链路，还是返回一个可用的降级结果。
2. **说不说实话**：降级原因是否如实反映真实故障。检索服务挂了却说
   「知识库里没有相关文档」，会让值班同学跑去补文档，而真正该修的是 Milvus。
3. **记不记账**：`degrade_reasons` 是不是恰好一份 —— 少记了指标看不见，
   多记了指标翻倍。

这三件事都是「不看代码就想不到」的约定，重构时最容易无声破坏，
所以每一条都要有测试钉住。

覆盖的七条路径：
- 检索全失败    → retrieval_failed（不是 retrieval_empty）
- 检索部分失败  → partial_retrieval，且仍在残缺证据上继续生成
- 检索正常为空  → retrieval_empty
- LLM 超时      → llm_timeout（与 llm_error 分开）
- LLM 报错      → llm_error
- 解析失败      → parse_failed，且降级后仍保留 SOP 证据
- 证据不足拦截  → evidence_insufficient
- Worker 不可重试错误 → 清零剩余重试次数

关联：app/core/errors.py、app/agent/rag_v2/nodes.py、app/core/job_failure.py
"""

import asyncio

import pytest
from langchain_core.documents import Document

from app.agent.rag_v2.nodes import dedup_node, generate_node, validate_answer_node
from app.core.errors import (
    DegradeReason,
    LLMError,
    LLMTimeoutError,
    ParseError,
    RetrievalError,
    RetrievalStatus,
    degrade_reason_of,
    error_code_of,
    is_retryable,
    wrap_llm_exception,
)
from app.core.job_failure import apply_retry_policy, format_failure_message
from app.models.aiops import AlarmEvent, AlarmSeverity
from app.models.aiops_report import EvidenceSource, parse_report
from app.services.first_response_service import FirstResponseService

# ── 测试替身 ──────────────────────────────────────────────


def _doc(content: str, doc_id: str, sub_query: str = "q") -> Document:
    return Document(
        page_content=content,
        metadata={"id": doc_id, "_sub_query": sub_query},
    )


def _failure(query: str, code: str = "retrieval_error") -> dict:
    """构造一条 retrieve_each 失败记录，形状与 nodes.py:157-160 一致。"""
    return {"query": query, "error": "Milvus 连接超时", "code": code}


class _TimeoutLLM:
    """超时的 LLM。用 asyncio.TimeoutError 而不是自定义异常，
    因为要验的正是「标准库超时能否被正确分类成 llm_timeout」。"""

    async def ainvoke(self, messages):
        raise asyncio.TimeoutError("等待 DashScope 响应超时")


class _ErrorLLM:
    async def ainvoke(self, messages):
        raise RuntimeError("DashScope 返回 500")


class _FakeSop:
    def __init__(self, evidence, status=None):
        self._evidence = evidence
        if status is not None:
            self._status = status
        else:
            self._status = RetrievalStatus.OK if evidence else RetrievalStatus.EMPTY

    def retrieve_sop_evidence(self, alarm):
        return self._evidence, self._status


def _alarm() -> AlarmEvent:
    return AlarmEvent(
        alert_name="HighCPUUsage",
        severity=AlarmSeverity.CRITICAL,
        service="order-api",
        summary="CPU 持续超过 90%",
    )


# ── 路径 1：检索全部失败 ──────────────────────────────────


class TestRetrievalFullFailure:
    """全部子查询失败 → retrieval_failed。

    这条和「知识库为空」必须分开，是整套降级体系里最重要的一处区分：
    前者要修 Milvus，后者要补文档，方向完全相反。
    """

    def test_all_branches_failed_marks_retrieval_failed(self):
        patch = dedup_node(
            {
                "documents": [],
                "retrieve_failures": [_failure("q1"), _failure("q2")],
            }
        )
        assert patch["degrade_reasons"] == [DegradeReason.RETRIEVAL_FAILED.value]
        assert patch["deduped_documents"] == []

    def test_not_confused_with_retrieval_empty(self):
        """有失败记录时绝不能报 retrieval_empty —— 那是在用假原因掩盖真原因。"""
        patch = dedup_node(
            {"documents": [], "retrieve_failures": [_failure("q1")]}
        )
        assert DegradeReason.RETRIEVAL_EMPTY.value not in patch["degrade_reasons"]

    @pytest.mark.asyncio
    async def test_generate_says_service_unavailable_not_no_documents(self):
        """无证据 + 有失败 → 措辞必须指向「检索服务不可用」。

        这里断言的是**给用户看的那句话**。它是排障方向的第一个路标：
        说错了，读到的人会从一开始就走错方向。
        """
        patch = await generate_node(
            {
                "question": "CPU 高怎么查",
                "deduped_documents": [],
                "retrieve_failures": [_failure("q1"), _failure("q2")],
            }
        )
        reason = patch["validation"]["reason"]
        assert "检索服务不可用" in reason
        # 明确否认「知识库里没有」，避免误导
        assert "不代表知识库中没有相关文档" in reason

    @pytest.mark.asyncio
    async def test_generate_does_not_duplicate_degrade_reason(self):
        """generate 不再记一次 retrieval_failed —— dedup 已经记过了。

        这个约定看着像疏漏，其实是刻意的分工（nodes.py:238 无 degrade_reasons）。
        两个节点都记的话，degrade_total{reason="retrieval_failed"} 会翻倍，
        「今天检索挂了多少次」这个数直接不可信。
        """
        patch = await generate_node(
            {
                "question": "CPU 高怎么查",
                "deduped_documents": [],
                "retrieve_failures": [_failure("q1")],
            }
        )
        assert "degrade_reasons" not in patch


# ── 路径 2：检索部分失败 ──────────────────────────────────


class TestRetrievalPartialFailure:
    """部分分支失败但仍有证据 → partial_retrieval，且继续生成。

    关键在「继续」：证据不完整不等于不能回答，但必须让模型知道证据是残缺的。
    """

    def test_partial_failure_marks_partial_retrieval(self):
        patch = dedup_node(
            {
                "documents": [_doc("CPU 排查步骤", "c1")],
                "retrieve_failures": [_failure("q2")],
            }
        )
        assert patch["degrade_reasons"] == [DegradeReason.PARTIAL_RETRIEVAL.value]
        # 有证据就要留着用
        assert len(patch["deduped_documents"]) == 1

    @pytest.mark.asyncio
    async def test_prompt_warns_model_evidence_is_incomplete(self):
        """证据残缺必须写进 prompt。

        不说的话，模型会把手上这几条当成全部事实，给出过度自信的结论 ——
        这是幻觉最常见的来源之一：不是模型乱编，是我们没告诉它信息不全。
        """
        captured = {}

        class _CapturingLLM:
            async def ainvoke(self, messages):
                captured["user"] = messages[1].content

                class _Resp:
                    content = "答案"

                return _Resp()

        import app.agent.rag_v2.nodes as nodes_module

        original = nodes_module.llm_factory.create_chat_model
        nodes_module.llm_factory.create_chat_model = lambda **kw: _CapturingLLM()
        try:
            await generate_node(
                {
                    "question": "CPU 高怎么查",
                    "deduped_documents": [_doc("先用 top 定位", "c1")],
                    "retrieve_failures": [_failure("q2")],
                }
            )
        finally:
            nodes_module.llm_factory.create_chat_model = original

        assert "证据不完整" in captured["user"]
        assert "不要基于残缺证据下确定结论" in captured["user"]


# ── 路径 3：检索正常但为空 ────────────────────────────────


class TestRetrievalEmpty:
    def test_no_failure_no_docs_marks_retrieval_empty(self):
        patch = dedup_node({"documents": [], "retrieve_failures": []})
        assert patch["degrade_reasons"] == [DegradeReason.RETRIEVAL_EMPTY.value]

    def test_success_path_records_no_degrade_reason(self):
        """正常路径不能有 degrade_reasons 键。

        降级指标的分母是全部请求，分子只该是真降级的那些。
        正常路径塞一个空列表进去，读侧一个 `if "degrade_reasons" in patch`
        就会把它当成降级。
        """
        patch = dedup_node({"documents": [_doc("正常内容", "c1")], "retrieve_failures": []})
        assert "degrade_reasons" not in patch

    @pytest.mark.asyncio
    async def test_generate_says_knowledge_base_has_nothing(self):
        patch = await generate_node(
            {"question": "问题", "deduped_documents": [], "retrieve_failures": []}
        )
        reason = patch["validation"]["reason"]
        assert "知识库中没有检索到" in reason
        # 不能反过来说成服务不可用
        assert "检索服务不可用" not in reason


# ── 路径 4 & 5：LLM 超时 / 报错 ───────────────────────────


class TestLLMFailureClassification:
    """超时与一般错误必须分开：前者调超时阈值/换模型，后者查配额与鉴权。"""

    def test_timeout_maps_to_llm_timeout(self):
        wrapped = wrap_llm_exception(asyncio.TimeoutError("超时"))
        assert isinstance(wrapped, LLMTimeoutError)
        assert wrapped.degrade_reason == DegradeReason.LLM_TIMEOUT
        # 504 让网关能正确统计网关超时，而不是混进 500
        assert wrapped.http_status == 504

    def test_generic_error_maps_to_llm_error(self):
        wrapped = wrap_llm_exception(RuntimeError("500"))
        assert isinstance(wrapped, LLMError)
        assert not isinstance(wrapped, LLMTimeoutError)
        assert wrapped.degrade_reason == DegradeReason.LLM_ERROR
        assert wrapped.http_status == 502

    def test_already_wrapped_is_returned_as_is(self):
        """幂等：已经是 LLMError 就原样返回，不套第二层。

        套两层会让 message 变成「LLM 调用失败: LLM 调用超时: ...」，
        也会把 LLMTimeoutError 降级成 LLMError，丢掉超时这个分类。
        """
        original = LLMTimeoutError("超时了")
        assert wrap_llm_exception(original) is original

    @pytest.mark.asyncio
    async def test_generate_timeout_records_llm_timeout(self):
        import app.agent.rag_v2.nodes as nodes_module

        original = nodes_module.llm_factory.create_chat_model
        nodes_module.llm_factory.create_chat_model = lambda **kw: _TimeoutLLM()
        try:
            patch = await generate_node(
                {
                    "question": "问题",
                    "deduped_documents": [_doc("证据", "c1")],
                    "retrieve_failures": [],
                }
            )
        finally:
            nodes_module.llm_factory.create_chat_model = original

        assert patch["degrade_reasons"] == [DegradeReason.LLM_TIMEOUT.value]
        # 降级但仍有可读答案，不是抛异常
        assert patch["answer"]
        assert patch["validation"]["blocked"] is True

    @pytest.mark.asyncio
    async def test_generate_error_records_llm_error(self):
        import app.agent.rag_v2.nodes as nodes_module

        original = nodes_module.llm_factory.create_chat_model
        nodes_module.llm_factory.create_chat_model = lambda **kw: _ErrorLLM()
        try:
            patch = await generate_node(
                {
                    "question": "问题",
                    "deduped_documents": [_doc("证据", "c1")],
                    "retrieve_failures": [],
                }
            )
        finally:
            nodes_module.llm_factory.create_chat_model = original

        assert patch["degrade_reasons"] == [DegradeReason.LLM_ERROR.value]

    @pytest.mark.asyncio
    async def test_first_response_llm_failure_keeps_sop_evidence(self):
        """LLM 挂了，SOP 证据仍然有价值，必须保留。

        降级的定义是「给出更差但仍可用的结果」。SOP 片段本身就是可用的 ——
        丢掉它等于把「部分可用」降成「完全不可用」。
        """
        from app.models.aiops_report import Evidence, EvidenceType

        sop = [
            Evidence(
                type=EvidenceType.VERIFIED_FACT,
                source=EvidenceSource.SOP,
                content="SOP 命中",
                source_title="cpu_high_usage.md",
                excerpt="先用 top 定位",
            )
        ]
        service = FirstResponseService(llm=_TimeoutLLM(), sop_service=_FakeSop(sop))
        report = await service.analyze(_alarm())

        assert report.is_degraded is True
        assert report.degrade_reason == DegradeReason.LLM_TIMEOUT
        assert len(report.evidence) == 1
        assert report.evidence[0].source == EvidenceSource.SOP


# ── 路径 6：解析失败 ──────────────────────────────────────


class TestParseFailure:
    def test_parse_error_is_not_retryable(self):
        """同样的 prompt 重试还是解析失败，真正的修复是改 prompt。"""
        assert is_retryable(ParseError("不是 JSON")) is False
        assert ParseError("x").degrade_reason == DegradeReason.PARSE_FAILED

    def test_non_json_output_degrades_with_parse_failed(self):
        report = parse_report("CPU 有点高，先看日志吧")
        assert report.is_degraded is True
        assert report.degrade_reason == DegradeReason.PARSE_FAILED
        # 原始文本要留着：排查 prompt 问题时唯一的现场证据
        assert report.raw_text

    def test_parse_report_never_raises(self):
        """无论输入多离谱都不抛 —— 这是 aiops_report.py:206 的既定范式。"""
        for bad in ["", "null", "[]", "{", '{"alert_summary": ', "「」"]:
            report = parse_report(bad)
            assert report.is_degraded is True

    @pytest.mark.asyncio
    async def test_parse_failure_still_keeps_sop_evidence(self):
        from app.models.aiops_report import Evidence, EvidenceType

        class _NonJsonLLM:
            async def ainvoke(self, messages):
                class _Resp:
                    content = "这不是 JSON"

                return _Resp()

        sop = [
            Evidence(
                type=EvidenceType.VERIFIED_FACT,
                source=EvidenceSource.SOP,
                content="SOP 命中",
                source_title="cpu_high_usage.md",
                excerpt="先用 top",
            )
        ]
        service = FirstResponseService(llm=_NonJsonLLM(), sop_service=_FakeSop(sop))
        report = await service.analyze(_alarm())

        assert report.is_degraded is True
        assert any(ev.source == EvidenceSource.SOP for ev in report.evidence)


# ── 路径 7：证据不足拦截 ──────────────────────────────────


class TestEvidenceInsufficientBlock:
    """质检拦截 → evidence_insufficient。

    它与 llm_error / retrieval_failed 的修复方向完全不同：
    这条说明模型和检索都正常，是知识库覆盖不够，处置是补文档。
    """

    @pytest.mark.asyncio
    async def test_low_scores_block_and_record_reason(self):
        low_score_json = (
            '{"coverage_score": 0.2, "groundedness_score": 0.3, '
            '"coverage_pass": false, "groundedness_pass": false, '
            '"needs_second_retrieval": true, "unsupported_claims": ["无依据断言"], '
            '"missing_aspects": ["缺少配置说明"], "reason": "证据不足"}'
        )

        class _LowScoreLLM:
            async def ainvoke(self, messages):
                class _Resp:
                    content = low_score_json

                return _Resp()

        import app.agent.rag_v2.nodes as nodes_module

        original = nodes_module.llm_factory.create_chat_model
        nodes_module.llm_factory.create_chat_model = lambda **kw: _LowScoreLLM()
        try:
            patch = await validate_answer_node(
                {
                    "question": "问题",
                    "answer": "一个可能没依据的答案",
                    "deduped_documents": [_doc("证据", "c1")],
                }
            )
        finally:
            nodes_module.llm_factory.create_chat_model = original

        assert patch["validation"]["blocked"] is True
        assert patch["degrade_reasons"] == [DegradeReason.EVIDENCE_INSUFFICIENT.value]
        # 拦截后给的是保守回答，不是原答案
        assert patch["answer"]

    @pytest.mark.asyncio
    async def test_validator_failure_is_fail_open_not_blocked(self):
        """质检器自己挂了 → 放行答案 + 标注未校验，而不是拦成「证据不足」。

        这是本项目里一个刻意的 fail-open 取舍（nodes.py:331-343）：
        拦掉看似安全，实际是质检模型一抖动，每个用户都收到
        「证据还不够支持下结论」—— 而证据明明是好的，挂的是质检器。
        这跟「SOP 检索失败却说未检索到 SOP」是同一类错误：用假原因掩盖真原因。
        """
        import app.agent.rag_v2.nodes as nodes_module

        original = nodes_module.llm_factory.create_chat_model
        nodes_module.llm_factory.create_chat_model = lambda **kw: _ErrorLLM()
        try:
            patch = await validate_answer_node(
                {
                    "question": "问题",
                    "answer": "一个正常生成的答案",
                    "deduped_documents": [_doc("证据", "c1")],
                }
            )
        finally:
            nodes_module.llm_factory.create_chat_model = original

        validation = patch["validation"]
        # 放行
        assert validation["blocked"] is False
        # 但如实声明没校验过
        assert validation["validated"] is False
        assert patch["degrade_reasons"] == [DegradeReason.LLM_ERROR.value]
        # 不能报成证据不足 —— 那是假原因
        assert DegradeReason.EVIDENCE_INSUFFICIENT.value not in patch["degrade_reasons"]
        # 答案没被替换成保守回答
        assert "answer" not in patch


# ── Worker 不可重试错误 ───────────────────────────────────


class TestWorkerNonRetryableError:
    """不可重试的错误要清零剩余重试次数，别白烧三倍资源。"""

    def test_non_retryable_types(self):
        assert is_retryable(ValueError("参数错")) is False
        assert is_retryable(FileNotFoundError("PDF 不存在")) is False
        assert is_retryable(ParseError("解析失败")) is False

    def test_retryable_types(self):
        assert is_retryable(TimeoutError("超时")) is True
        assert is_retryable(ConnectionError("连不上")) is True
        assert is_retryable(RetrievalError("Milvus 挂了")) is True

    def test_unknown_exception_defaults_to_no_retry(self):
        """未知异常兜底不重试。

        默认重试听起来更「稳」，实际是把「PDF 损坏」这类确定性失败
        重试三次：白烧三倍资源，还把失败暴露时间往后推。
        """

        class _WeirdError(Exception):
            pass

        assert is_retryable(_WeirdError("没见过")) is False

    def test_apply_retry_policy_outside_rq_is_pure(self):
        """脱离 RQ 上下文时只返回 is_retryable 的判定，无副作用。

        「纯」在这里有具体含义：没有 job 可改，就不改任何东西，
        返回值等于分类结果本身。这让单测能直接调它而不必假造 RQ 上下文。

        注意两个取值方向都要断言：ValueError 是确定性失败（不重试），
        TimeoutError 是瞬时故障（该重试）。只测一个方向的话，
        一个「永远返回 False」的实现也能通过。
        """
        assert apply_retry_policy(ValueError("x"), job_id="j1") is False
        assert apply_retry_policy(TimeoutError("x"), job_id="j1") is True

    def test_failure_message_carries_error_code(self):
        """错误码可聚合，自由文本不行。"""
        msg = format_failure_message(ValueError("坏参数"), will_retry=False)
        assert "[value_error]" in msg
        assert "将自动重试" not in msg

    def test_failure_message_announces_pending_retry(self):
        """状态是 FAILED 但后台还排着重试，必须说清楚。

        不说的话，值班同学会立刻开始手工排查一个十秒后可能自己就好了的问题。
        """
        msg = format_failure_message(TimeoutError("超时"), will_retry=True)
        assert "RQ 将自动重试" in msg

    def test_error_code_is_finite_enum_not_free_text(self):
        """错误码必须是有限集合，否则指标标签会爆炸（高基数）。"""
        assert error_code_of(RetrievalError("x")) == "retrieval_error"
        assert error_code_of(LLMTimeoutError("x")) == "llm_timeout"
        # 非 AppError 退回类名的 snake 形式，仍然有限可枚举
        assert error_code_of(ValueError("x")) == "value_error"

    def test_degrade_reason_of_never_returns_none(self):
        """任意异常都要能映射到一个降级原因，读侧不必处理 None。"""
        for exc in [
            asyncio.TimeoutError("t"),
            RuntimeError("r"),
            ValueError("v"),
            RetrievalError("m"),
        ]:
            assert isinstance(degrade_reason_of(exc), DegradeReason)
