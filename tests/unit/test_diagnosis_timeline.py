"""诊断时间线在**真实边界**推进的单元测试。

修复前的毛病：编排层在调 `analyze()` 之前就把 planning / retrieving /
diagnosing 三个状态一路推完了，而检索和 LLM 调用都发生在 `analyze()` 内部。
于是时间线上前三条事件的时间戳只差几毫秒，`done` 却在几十秒之后 ——
这条时间线记录的是「编排层写代码的顺序」，不是「系统真实的执行顺序」。
排障时问「慢在哪一步」，它答不上来。

本文件锁两件事：
1. `FirstResponseService.analyze` 在真实边界回调 `on_phase`，且带上耗时与关键计数；
2. 编排层把这些回调翻译成状态流转 + 时间线事件，且状态与事件在同一个事务里落库。
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.core.errors import RetrievalStatus
from app.models.aiops import AlarmEvent, AlarmSeverity
from app.models.aiops_diagnosis import DiagnosisTaskStatus
from app.models.aiops_report import (
    Evidence,
    EvidenceSource,
    EvidenceType,
    FirstResponseReport,
)
from app.repositories.aiops_diagnosis_repository import AiopsDiagnosisRepository
from app.services.alert_diagnosis_orchestrator import AlertDiagnosisOrchestrator
from app.services.first_response_service import (
    PHASE_DIAGNOSED,
    PHASE_DIAGNOSING,
    PHASE_RETRIEVED,
    PHASE_RETRIEVING,
    FirstResponseService,
)


# ── 夹具与替身 ─────────────────────────────────────────────
@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _alarm() -> AlarmEvent:
    return AlarmEvent(
        alert_name="HighCPUUsage",
        severity=AlarmSeverity.CRITICAL,
        service="order-api",
        summary="CPU 持续超过 90%",
    )


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


class _FakeSop:
    def __init__(self, evidence, status: RetrievalStatus | None = None):
        self._evidence = evidence
        if status is not None:
            self._status = status
        else:
            self._status = RetrievalStatus.OK if evidence else RetrievalStatus.EMPTY

    def retrieve_sop_evidence(self, alarm):
        return self._evidence, self._status


class _FakeLLM:
    def __init__(self, content):
        self._content = content

    async def ainvoke(self, messages):
        class _Resp:
            content = self._content

        return _Resp()


class _BrokenLLM:
    async def ainvoke(self, messages):
        raise TimeoutError("DashScope 读超时")


_VALID_JSON = """
{
  "alert_summary": "order-api CPU 超 90%",
  "current_judgment": "疑似流量突增",
  "root_cause_hypotheses": ["流量突增"]
}
"""


class _Recorder:
    """把 on_phase 的每次回调原样记下来，供断言检查顺序与载荷。"""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, phase: str, payload: dict) -> None:
        self.calls.append((phase, payload))

    @property
    def phases(self) -> list[str]:
        return [phase for phase, _ in self.calls]

    def payload_of(self, phase: str) -> dict:
        for name, payload in self.calls:
            if name == phase:
                return payload
        raise AssertionError(f"没有 {phase} 阶段的回调")


# ── analyze 的阶段回调契约 ──────────────────────────────────
@pytest.mark.asyncio
async def test_phases_fire_in_real_order():
    recorder = _Recorder()
    service = FirstResponseService(
        llm=_FakeLLM(_VALID_JSON),
        sop_service=_FakeSop(_sop_evidence()),
    )
    await service.analyze(_alarm(), on_phase=recorder)

    assert recorder.phases == [
        PHASE_RETRIEVING,
        PHASE_RETRIEVED,
        PHASE_DIAGNOSING,
        PHASE_DIAGNOSED,
    ]


@pytest.mark.asyncio
async def test_retrieved_payload_carries_duration_and_counts():
    recorder = _Recorder()
    service = FirstResponseService(
        llm=_FakeLLM(_VALID_JSON),
        sop_service=_FakeSop(_sop_evidence()),
    )
    await service.analyze(_alarm(), on_phase=recorder)

    payload = recorder.payload_of(PHASE_RETRIEVED)
    assert payload["duration_ms"] >= 0
    assert payload["sop_hit_count"] == 1
    # 检索状态一起带上：0 命中有两种含义（挂了 / 库里没这篇），
    # 只报数字会把这两件事重新压成同一个信号。
    assert payload["retrieval_status"] == RetrievalStatus.OK.value


@pytest.mark.asyncio
async def test_retrieval_failure_is_distinguishable_from_empty():
    failed = _Recorder()
    await FirstResponseService(
        llm=_FakeLLM(_VALID_JSON),
        sop_service=_FakeSop([], status=RetrievalStatus.FAILED),
    ).analyze(_alarm(), on_phase=failed)

    empty = _Recorder()
    await FirstResponseService(
        llm=_FakeLLM(_VALID_JSON),
        sop_service=_FakeSop([]),
    ).analyze(_alarm(), on_phase=empty)

    # 两者命中数都是 0，但时间线上必须能分辨
    assert failed.payload_of(PHASE_RETRIEVED)["sop_hit_count"] == 0
    assert empty.payload_of(PHASE_RETRIEVED)["sop_hit_count"] == 0
    assert (
        failed.payload_of(PHASE_RETRIEVED)["retrieval_status"]
        != empty.payload_of(PHASE_RETRIEVED)["retrieval_status"]
    )


@pytest.mark.asyncio
async def test_diagnosing_payload_carries_prompt_tokens():
    recorder = _Recorder()
    service = FirstResponseService(
        llm=_FakeLLM(_VALID_JSON),
        sop_service=_FakeSop(_sop_evidence()),
    )
    await service.analyze(_alarm(), on_phase=recorder)

    assert recorder.payload_of(PHASE_DIAGNOSING)["prompt_tokens"] > 0


@pytest.mark.asyncio
async def test_llm_failure_still_reports_diagnosed_duration():
    """LLM 挂了也要报耗时 —— 那正是「白等了多久才放弃」，调超时阈值的唯一依据。"""
    recorder = _Recorder()
    service = FirstResponseService(
        llm=_BrokenLLM(),
        sop_service=_FakeSop(_sop_evidence()),
    )
    report = await service.analyze(_alarm(), on_phase=recorder)

    assert report.is_degraded is True
    payload = recorder.payload_of(PHASE_DIAGNOSED)
    assert payload["duration_ms"] >= 0
    assert payload["is_degraded"] is True
    assert payload["error_code"]


@pytest.mark.asyncio
async def test_on_phase_is_optional():
    """不传回调就是修复前的行为，现有调用点一行都不用改。"""
    service = FirstResponseService(
        llm=_FakeLLM(_VALID_JSON),
        sop_service=_FakeSop(_sop_evidence()),
    )
    report = await service.analyze(_alarm())
    assert report.is_degraded is False


@pytest.mark.asyncio
async def test_broken_callback_does_not_fail_analysis():
    """观测设施不该让业务失败：回调自己炸了，报告照样出。"""

    def boom(phase, payload):
        raise RuntimeError("时间线数据库连不上")

    service = FirstResponseService(
        llm=_FakeLLM(_VALID_JSON),
        sop_service=_FakeSop(_sop_evidence()),
    )
    report = await service.analyze(_alarm(), on_phase=boom)
    assert report.is_degraded is False
    assert report.alert_summary == "order-api CPU 超 90%"


# ── repository：状态与事件同一个事务 ─────────────────────────
def _new_task(repo: AiopsDiagnosisRepository):
    return repo.create_task(
        alarm_event={"alert_name": "HighCPUUsage"},
        alert_name="HighCPUUsage",
        severity="critical",
        source="prometheus",
    )


def test_record_phase_advances_and_logs_together(db):
    repo = AiopsDiagnosisRepository(db)
    task = _new_task(repo)
    repo.record_phase(
        task,
        phase="planning",
        status=DiagnosisTaskStatus.PLANNING,
        message="构造告警上下文",
    )

    assert task.status == DiagnosisTaskStatus.PLANNING
    events = repo.list_events(task.id)
    assert [e.phase for e in events] == ["planning"]


def test_record_phase_without_status_only_logs_event(db):
    """「检索结束」不是新状态：任务仍在 RETRIEVING，只是耗时终于能落库了。"""
    repo = AiopsDiagnosisRepository(db)
    task = _new_task(repo)
    repo.record_phase(task, phase="planning", status=DiagnosisTaskStatus.PLANNING)
    repo.record_phase(task, phase="retrieving", status=DiagnosisTaskStatus.RETRIEVING)
    repo.record_phase(task, phase="retrieved", payload={"duration_ms": 12})

    assert task.status == DiagnosisTaskStatus.RETRIEVING
    events = repo.list_events(task.id)
    assert [e.phase for e in events] == ["planning", "retrieving", "retrieved"]
    assert events[-1].payload == {"duration_ms": 12}


def test_record_phase_rejects_illegal_transition_without_writing_event(db):
    """非法跃迁要在写事件之前就拦住，否则时间线上会留一条「其实没发生」的事件。"""
    repo = AiopsDiagnosisRepository(db)
    task = _new_task(repo)
    with pytest.raises(ValueError, match="非法状态跃迁"):
        repo.record_phase(
            task,
            phase="diagnosing",
            status=DiagnosisTaskStatus.DIAGNOSING,
        )
    db.rollback()
    assert repo.list_events(task.id) == []


def test_save_report_can_inline_done_event(db):
    repo = AiopsDiagnosisRepository(db)
    task = _new_task(repo)
    for status in (
        DiagnosisTaskStatus.PLANNING,
        DiagnosisTaskStatus.RETRIEVING,
        DiagnosisTaskStatus.DIAGNOSING,
    ):
        repo.record_phase(task, phase=status.value, status=status)

    repo.save_report(
        task,
        report={"summary": "x"},
        summary="x",
        event_message="首响报告已生成",
        event_payload={"is_degraded": False},
    )

    assert task.status == DiagnosisTaskStatus.DONE
    events = repo.list_events(task.id)
    assert events[-1].phase == "done"
    assert events[-1].payload == {"is_degraded": False}


def test_mark_failed_can_inline_failed_event(db):
    repo = AiopsDiagnosisRepository(db)
    task = _new_task(repo)
    repo.mark_failed(
        task,
        error_message="MCP 工具超时",
        event_message="MCP 工具超时",
    )

    assert task.status == DiagnosisTaskStatus.FAILED
    events = repo.list_events(task.id)
    assert [e.phase for e in events] == ["failed"]


# ── 编排层：端到端时间线 ────────────────────────────────────
@pytest.mark.asyncio
async def test_orchestrator_timeline_covers_all_real_phases(db):
    orchestrator = AlertDiagnosisOrchestrator(
        response_service=FirstResponseService(
            llm=_FakeLLM(_VALID_JSON),
            sop_service=_FakeSop(_sop_evidence()),
        )
    )
    task, report = await orchestrator.run(
        db,
        payload={"alert_name": "HighCPUUsage", "severity": "critical"},
        source="manual",
    )

    assert task.status == DiagnosisTaskStatus.DONE
    assert report.is_degraded is False

    repo = AiopsDiagnosisRepository(db)
    phases = [e.phase for e in repo.list_events(task.id)]
    # 契约：老的四个阶段名一个都不能少（详情接口和前端都认它们），
    # 新增两个「结束」事件带着耗时。
    assert phases == [
        "planning",
        "retrieving",
        "retrieved",
        "diagnosing",
        "diagnosed",
        "done",
    ]


@pytest.mark.asyncio
async def test_orchestrator_records_durations_on_timeline(db):
    orchestrator = AlertDiagnosisOrchestrator(
        response_service=FirstResponseService(
            llm=_FakeLLM(_VALID_JSON),
            sop_service=_FakeSop(_sop_evidence()),
        )
    )
    task, _ = await orchestrator.run(
        db,
        payload={"alert_name": "HighCPUUsage", "severity": "critical"},
        source="manual",
    )

    repo = AiopsDiagnosisRepository(db)
    by_phase = {e.phase: (e.payload or {}) for e in repo.list_events(task.id)}
    assert "duration_ms" in by_phase["retrieved"]
    assert "sop_hit_count" in by_phase["retrieved"]
    assert "prompt_tokens" in by_phase["diagnosing"]
    assert "duration_ms" in by_phase["diagnosed"]


@pytest.mark.asyncio
async def test_orchestrator_timeline_ordering_is_not_all_upfront(db):
    """核心回归：diagnosing 必须发生在 retrieved **之后**。

    修复前三个状态在 analyze 前一口气推完，这个断言会挂 ——
    那时 diagnosing 排在 retrieved 前面。
    """
    orchestrator = AlertDiagnosisOrchestrator(
        response_service=FirstResponseService(
            llm=_FakeLLM(_VALID_JSON),
            sop_service=_FakeSop(_sop_evidence()),
        )
    )
    task, _ = await orchestrator.run(
        db,
        payload={"alert_name": "HighCPUUsage", "severity": "critical"},
        source="manual",
    )

    repo = AiopsDiagnosisRepository(db)
    phases = [e.phase for e in repo.list_events(task.id)]
    assert phases.index("retrieved") < phases.index("diagnosing")


@pytest.mark.asyncio
async def test_orchestrator_degraded_llm_still_completes(db):
    """LLM 挂了是降级不是失败：任务照样 DONE，时间线照样完整。"""
    orchestrator = AlertDiagnosisOrchestrator(
        response_service=FirstResponseService(
            llm=_BrokenLLM(),
            sop_service=_FakeSop(_sop_evidence()),
        )
    )
    task, report = await orchestrator.run(
        db,
        payload={"alert_name": "HighCPUUsage", "severity": "critical"},
        source="manual",
    )

    assert task.status == DiagnosisTaskStatus.DONE
    assert report.is_degraded is True

    repo = AiopsDiagnosisRepository(db)
    events = repo.list_events(task.id)
    by_phase = {e.phase: (e.payload or {}) for e in events}
    assert by_phase["diagnosed"]["is_degraded"] is True
    assert by_phase["done"]["is_degraded"] is True
