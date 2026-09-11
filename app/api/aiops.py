"""
AIOps 智能运维接口
"""

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.aiops_diagnosis import AiopsDiagnosisTask, DiagnosisTaskStatus
from app.repositories.aiops_diagnosis_repository import AiopsDiagnosisRepository
from app.services.alert_diagnosis_orchestrator import alert_diagnosis_orchestrator

router = APIRouter()


# ──────────────────────────────────────────────────────────────
# 告警首响分析：AlarmEvent → 落库诊断任务 + 结构化报告
# ──────────────────────────────────────────────────────────────
class AnalyzeAlertRequest(BaseModel):
    """单条告警首响分析请求。

    payload 是告警原始数据；source 决定用哪个归一化器（manual / prometheus）。
    """

    model_config = {
        "json_schema_extra": {
            "example": {
                "source": "manual",
                "session_id": None,
                "payload": {
                    "alert_name": "HighCPUUsage",
                    "severity": "critical",
                    "service": "order-api",
                    "instance": "10.0.0.12:9100",
                    "summary": "CPU 使用率持续超过 90%",
                    "labels": {"env": "prod"},
                },
            }
        }
    }

    payload: dict[str, Any] = Field(description="告警原始数据（单条）")
    source: str = Field(default="manual", description="告警来源：manual / prometheus")
    session_id: Optional[str] = Field(default=None, description="可选会话 ID")


@router.post("/aiops/alerts/analyze", status_code=201)
async def analyze_alert(
    request: AnalyzeAlertRequest,
    db: Session = Depends(get_db),
):
    """对一条告警做首响分析，产出可追踪的诊断任务与结构化报告。"""
    try:
        task, report = await alert_diagnosis_orchestrator.run(
            db,
            payload=request.payload,
            source=request.source,
            session_id=request.session_id,
        )
    except ValueError as exc:  # 未知来源等归一化错误
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "code": 201,
        "message": "success",
        "data": {
            "task_id": task.id,
            "status": task.status.value,
            "report": report.model_dump(mode="json"),
            "report_markdown": report.to_markdown(),
        },
    }


@router.get("/aiops/tasks")
async def list_diagnosis_tasks(
    status: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """列出诊断任务，可按状态过滤。"""
    repo = AiopsDiagnosisRepository(db)
    status_enum = None
    if status:
        try:
            status_enum = DiagnosisTaskStatus(status)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"未知状态: {status}") from exc

    tasks = repo.list_tasks(limit=limit, status=status_enum)
    return {
        "code": 200,
        "message": "success",
        "data": [_serialize_task_summary(task) for task in tasks],
    }


@router.get("/aiops/tasks/{task_id}")
async def get_diagnosis_task(
    task_id: str,
    db: Session = Depends(get_db),
):
    """查询单个诊断任务详情（含报告与事件时间线）。"""
    repo = AiopsDiagnosisRepository(db)
    task = repo.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="诊断任务不存在")

    events = repo.list_events(task_id)
    data = _serialize_task_summary(task)
    data.update(
        {
            "alarm_event": task.alarm_event,
            "report": task.report,
            "error_message": task.error_message,
            "events": [
                {
                    "phase": ev.phase,
                    "message": ev.message,
                    "payload": ev.payload,
                    "created_at": ev.created_at.isoformat(),
                }
                for ev in events
            ],
        }
    )
    return {"code": 200, "message": "success", "data": data}


def _serialize_task_summary(task: AiopsDiagnosisTask) -> dict[str, Any]:
    """诊断任务的摘要序列化（列表与详情共用）。"""
    return {
        "task_id": task.id,
        "alert_id": task.alert_id,
        "alert_name": task.alert_name,
        "severity": task.severity,
        "source": task.source,
        "status": task.status.value,
        "current_phase": task.current_phase,
        "summary": task.summary,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
    }
