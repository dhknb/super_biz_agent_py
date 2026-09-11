"""告警首响分析服务单元测试。

用假 LLM + 假 SOP 服务，不连真实服务，覆盖：
- 正常闭环：AlarmEvent → 结构化报告
- SOP 证据被注入报告且置于最前（事实优先）
- LLM 返回非 JSON → 降级报告
- LLM 抛异常 → 降级但仍保留 SOP 证据
- prompt 里包含告警名与 SOP 片段
- **检索失败 vs 知识库为空**：两者必须产生不同的 prompt 与报告状态
  （检索挂了却说「未检索到相关 SOP」= 撒谎，会把排障方向带反）
"""

import pytest

from app.core.errors import DegradeReason, RetrievalStatus
from app.models.aiops import AlarmEvent, AlarmSeverity
from app.models.aiops_report import (
    Evidence,
    EvidenceSource,
    EvidenceType,
)
from app.services.first_response_service import FirstResponseService


def _alarm() -> AlarmEvent:
    return AlarmEvent(
        alert_name="HighCPUUsage",
        severity=AlarmSeverity.CRITICAL,
        service="order-api",
        summary="CPU 持续超过 90%",
    )


class _FakeSop:
    """假 SOP 检索。

    status 默认按 evidence 是否为空推断（OK / EMPTY），
    需要测「检索服务挂了」时显式传 RetrievalStatus.FAILED ——
    正是这个区分让「没找到」和「查不了」不再是同一个信号。
    """

    def __init__(self, evidence, status: RetrievalStatus | None = None):
        self._evidence = evidence
        if status is not None:
            self._status = status
        else:
            self._status = RetrievalStatus.OK if evidence else RetrievalStatus.EMPTY
        self.called_with = None

    def retrieve_sop_evidence(self, alarm):
        self.called_with = alarm
        return self._evidence, self._status


class _FakeLLM:
    def __init__(self, content):
        self._content = content
        self.last_messages = None

    async def ainvoke(self, messages):
        self.last_messages = messages

        class _Resp:
            content = self._content

        return _Resp()


class _BrokenLLM:
    async def ainvoke(self, messages):
        raise RuntimeError("DashScope 超时")


def _sop_evidence():
    return [
        Evidence(
            type=EvidenceType.VERIFIED_FACT,
            source=EvidenceSource.SOP,
            content="知识库 SOP 命中：cpu_high_usage.md",
            source_title="cpu_high_usage.md · CPU 排查",
            excerpt="先用 top 定位高 CPU 进程",
        )
    ]


_VALID_JSON = """
{
  "alert_summary": "order-api CPU 超 90%",
  "current_judgment": "疑似流量突增",
  "recommended_checks": [
    {"order": 1, "action": "看 QPS", "reason": "确认流量"}
  ],
  "evidence": [
    {"type": "model_inference", "source": "model", "content": "可能慢查询"}
  ],
  "root_cause_hypotheses": ["流量突增"],
  "pending_confirmations": ["是否大促"]
}
"""


@pytest.mark.asyncio
async def test_full_pipeline_produces_structured_report():
    service = FirstResponseService(
        llm=_FakeLLM(_VALID_JSON),
        sop_service=_FakeSop(_sop_evidence()),
    )
    report = await service.analyze(_alarm())

    assert report.is_degraded is False
    assert report.alert_summary == "order-api CPU 超 90%"
    # SOP 证据 + 模型推断证据都在
    assert len(report.evidence) == 2
    # 事实优先：第一条是 VERIFIED_FACT
    assert report.evidence[0].type == EvidenceType.VERIFIED_FACT
    assert report.evidence[0].source == EvidenceSource.SOP
    assert report.evidence[1].type == EvidenceType.MODEL_INFERENCE


@pytest.mark.asyncio
async def test_sop_evidence_injected_even_without_model_evidence():
    no_ev_json = '{"alert_summary": "x", "current_judgment": "y"}'
    service = FirstResponseService(
        llm=_FakeLLM(no_ev_json),
        sop_service=_FakeSop(_sop_evidence()),
    )
    report = await service.analyze(_alarm())
    assert len(report.evidence) == 1
    assert report.evidence[0].source == EvidenceSource.SOP


@pytest.mark.asyncio
async def test_non_json_output_degrades():
    service = FirstResponseService(
        llm=_FakeLLM("CPU 有点高，先看日志吧"),
        sop_service=_FakeSop(_sop_evidence()),
    )
    report = await service.analyze(_alarm())
    assert report.is_degraded is True
    # 降级也要保留 SOP 证据
    assert any(ev.source == EvidenceSource.SOP for ev in report.evidence)


@pytest.mark.asyncio
async def test_llm_exception_degrades_but_keeps_sop():
    service = FirstResponseService(
        llm=_BrokenLLM(),
        sop_service=_FakeSop(_sop_evidence()),
    )
    report = await service.analyze(_alarm())
    assert report.is_degraded is True
    assert len(report.evidence) == 1
    assert report.evidence[0].source == EvidenceSource.SOP


@pytest.mark.asyncio
async def test_prompt_contains_alarm_and_sop():
    fake_llm = _FakeLLM(_VALID_JSON)
    service = FirstResponseService(
        llm=fake_llm,
        sop_service=_FakeSop(_sop_evidence()),
    )
    await service.analyze(_alarm())
    # 取 user prompt
    user_msg = fake_llm.last_messages[1][1]
    assert "HighCPUUsage" in user_msg
    assert "cpu_high_usage.md" in user_msg


@pytest.mark.asyncio
async def test_no_sop_still_works():
    service = FirstResponseService(
        llm=_FakeLLM(_VALID_JSON),
        sop_service=_FakeSop([]),
    )
    report = await service.analyze(_alarm())
    assert report.is_degraded is False
    # 只有模型推断证据
    assert all(ev.type == EvidenceType.MODEL_INFERENCE for ev in report.evidence)
