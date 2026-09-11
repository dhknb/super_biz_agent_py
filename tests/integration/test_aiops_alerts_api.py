"""告警首响分析 API 集成测试。

覆盖 Day6 的端到端闭环（不连真实 LLM / Milvus）：
- POST /api/aiops/alerts/analyze：三条样例告警都能产出结构化报告并落库
- GET  /api/aiops/tasks：能列出任务
- GET  /api/aiops/tasks/{id}：能查详情（含事件时间线）
- 未知来源 → 400

关键隔离手段：
- 用内存 SQLite 覆盖 get_db，跑真实 SQL（验证任务确实落库）
- 用假 LLM + 假 SOP 服务替换 first_response_service，不烧 token、不连 Milvus
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.main import app
from app.models.aiops_report import (
    Evidence,
    EvidenceSource,
    EvidenceType,
    FirstResponseReport,
)
from app.services.first_response_service import (
    PHASE_DIAGNOSED,
    PHASE_DIAGNOSING,
    PHASE_RETRIEVED,
    PHASE_RETRIEVING,
)


_FIXTURES = Path(__file__).parent.parent / "fixtures" / "aiops_alerts"


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


class _FakeResponseService:
    """假首响服务：返回固定的结构化报告，带一条 SOP 证据。

    必须如实回调 on_phase —— 时间线上的 retrieving / diagnosing 现在由
    analyze 在真实边界推进，编排层不再预先推进。假服务不回调的话，
    任务会停在 PLANNING，后面 save_report 直接撞上非法跃迁。

    这个耦合是设计意图而非意外：它逼着替身走一遍真实的阶段顺序，
    集成测试因此真的在验证新链路，而不是绕开它。
    """

    async def analyze(self, alarm, *, on_phase=None):
        emit = on_phase or (lambda phase, payload: None)
        emit(PHASE_RETRIEVING, {})
        emit(PHASE_RETRIEVED, {"duration_ms": 1, "sop_hit_count": 1})
        emit(PHASE_DIAGNOSING, {"prompt_tokens": 42})
        report = FirstResponseReport(
            alert_summary=f"{alarm.alert_name} 首响分析",
            current_judgment="疑似资源打满",
            recommended_checks=[],
            evidence=[
                Evidence(
                    type=EvidenceType.VERIFIED_FACT,
                    source=EvidenceSource.SOP,
                    content="命中 SOP",
                    source_title="cpu_high_usage.md",
                    excerpt="先用 top 定位",
                )
            ],
            root_cause_hypotheses=["资源不足"],
            pending_confirmations=["确认是否有突发流量"],
        )
        emit(PHASE_DIAGNOSED, {"duration_ms": 2, "is_degraded": False})
        return report


@pytest.fixture
def integration_client():
    """带真实内存 SQLite + 假首响服务的 TestClient。"""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    # 替换编排器内部的首响服务，避免真调 LLM / Milvus
    from app.services import alert_diagnosis_orchestrator as orch_module

    original = orch_module.alert_diagnosis_orchestrator._response_service
    orch_module.alert_diagnosis_orchestrator._response_service = _FakeResponseService()

    try:
        yield TestClient(app)
    finally:
        orch_module.alert_diagnosis_orchestrator._response_service = original
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()


@pytest.mark.parametrize(
    "fixture_name",
    ["01_cpu_high.json", "02_disk_full.json", "03_service_unavailable.json"],
)
def test_analyze_alert_produces_report(integration_client, fixture_name):
    fixture = _load(fixture_name)
    response = integration_client.post(
        "/api/aiops/alerts/analyze",
        json={"payload": fixture["payload"], "source": fixture["source"]},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["code"] == 201
    data = body["data"]
    assert data["status"] == "done"
    assert data["task_id"]
    # 结构化报告字段
    report = data["report"]
    assert "alert_summary" in report
    assert report["is_degraded"] is False
    # SOP 证据被溯源
    assert any(ev["source"] == "sop" for ev in report["evidence"])
    # markdown 可读
    assert "告警首响分析报告" in data["report_markdown"]


def test_list_and_get_task(integration_client):
    fixture = _load("01_cpu_high.json")
    created = integration_client.post(
        "/api/aiops/alerts/analyze",
        json={"payload": fixture["payload"], "source": fixture["source"]},
    ).json()
    task_id = created["data"]["task_id"]

    # 列表
    listed = integration_client.get("/api/aiops/tasks")
    assert listed.status_code == 200
    assert any(t["task_id"] == task_id for t in listed.json()["data"])

    # 详情：含事件时间线
    detail = integration_client.get(f"/api/aiops/tasks/{task_id}")
    assert detail.status_code == 200
    detail_data = detail.json()["data"]
    assert detail_data["task_id"] == task_id
    phases = [ev["phase"] for ev in detail_data["events"]]
    assert "planning" in phases
    assert "retrieving" in phases
    assert "diagnosing" in phases
    assert "done" in phases


def test_unknown_source_returns_400(integration_client):
    response = integration_client.post(
        "/api/aiops/alerts/analyze",
        json={"payload": {"alert_name": "x"}, "source": "no_such_source"},
    )
    assert response.status_code == 400


def test_get_missing_task_returns_404(integration_client):
    response = integration_client.get("/api/aiops/tasks/does-not-exist")
    assert response.status_code == 404


def test_legacy_aiops_endpoint_is_removed(integration_client):
    response = integration_client.post("/api/aiops", json={"session_id": "legacy"})
    assert response.status_code == 404
