"""AIOps 诊断任务 repository 单元测试。

用内存 SQLite 跑真实 SQL，验证：
- 创建任务默认 NEW
- 合法状态流转 / 非法跃迁防护
- 事件留痕
- 保存报告置 DONE
- mark_failed 任意状态可转
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.aiops_diagnosis import DiagnosisTaskStatus, can_transition
from app.repositories.aiops_diagnosis_repository import AiopsDiagnosisRepository


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


def _new_task(repo: AiopsDiagnosisRepository):
    return repo.create_task(
        alarm_event={"alert_name": "HighCPUUsage", "severity": "critical"},
        alert_name="HighCPUUsage",
        severity="critical",
        source="prometheus",
        alert_id="fp-123",
    )


def test_create_task_defaults_to_new(db):
    repo = AiopsDiagnosisRepository(db)
    task = _new_task(repo)
    assert task.id
    assert task.status == DiagnosisTaskStatus.NEW
    assert task.alert_name == "HighCPUUsage"
    assert task.alarm_event["severity"] == "critical"


def test_full_status_flow(db):
    repo = AiopsDiagnosisRepository(db)
    task = _new_task(repo)
    repo.advance_status(task, DiagnosisTaskStatus.PLANNING)
    repo.advance_status(task, DiagnosisTaskStatus.RETRIEVING)
    repo.advance_status(task, DiagnosisTaskStatus.DIAGNOSING)
    assert task.status == DiagnosisTaskStatus.DIAGNOSING
    assert task.current_phase == "diagnosing"


def test_illegal_transition_raises(db):
    repo = AiopsDiagnosisRepository(db)
    task = _new_task(repo)
    # NEW 不能直接跳到 DIAGNOSING
    with pytest.raises(ValueError, match="非法状态跃迁"):
        repo.advance_status(task, DiagnosisTaskStatus.DIAGNOSING)


def test_can_transition_matrix():
    assert can_transition(DiagnosisTaskStatus.NEW, DiagnosisTaskStatus.PLANNING)
    assert not can_transition(DiagnosisTaskStatus.NEW, DiagnosisTaskStatus.DONE)
    # 相同状态幂等允许
    assert can_transition(DiagnosisTaskStatus.PLANNING, DiagnosisTaskStatus.PLANNING)
    # 终态不可再流转
    assert not can_transition(DiagnosisTaskStatus.DONE, DiagnosisTaskStatus.PLANNING)


def test_add_event_and_list(db):
    repo = AiopsDiagnosisRepository(db)
    task = _new_task(repo)
    repo.add_event(task.id, phase="planning", message="制定计划")
    repo.add_event(task.id, phase="retrieving", message="检索 SOP", payload={"hits": 3})
    events = repo.list_events(task.id)
    assert [e.phase for e in events] == ["planning", "retrieving"]
    assert events[1].payload == {"hits": 3}


def test_save_report_sets_done(db):
    repo = AiopsDiagnosisRepository(db)
    task = _new_task(repo)
    repo.advance_status(task, DiagnosisTaskStatus.PLANNING)
    repo.advance_status(task, DiagnosisTaskStatus.RETRIEVING)
    repo.advance_status(task, DiagnosisTaskStatus.DIAGNOSING)
    repo.save_report(task, report={"summary": "CPU 飙高"}, summary="CPU 飙高")
    assert task.status == DiagnosisTaskStatus.DONE
    assert task.report["summary"] == "CPU 飙高"
    assert task.summary == "CPU 飙高"


def test_mark_failed_from_any_state(db):
    repo = AiopsDiagnosisRepository(db)
    task = _new_task(repo)
    repo.advance_status(task, DiagnosisTaskStatus.PLANNING)
    repo.mark_failed(task, error_message="MCP 工具超时")
    assert task.status == DiagnosisTaskStatus.FAILED
    assert task.error_message == "MCP 工具超时"


def test_list_tasks_filters_by_status(db):
    repo = AiopsDiagnosisRepository(db)
    t1 = _new_task(repo)
    t2 = _new_task(repo)
    repo.advance_status(t2, DiagnosisTaskStatus.PLANNING)
    new_tasks = repo.list_tasks(status=DiagnosisTaskStatus.NEW)
    assert t1.id in [t.id for t in new_tasks]
    assert t2.id not in [t.id for t in new_tasks]
