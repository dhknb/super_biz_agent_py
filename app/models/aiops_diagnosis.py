"""AIOps 诊断任务持久化模型。

把「一次告警首响诊断」从内存里跑完即焚的临时流程，
升级为一条可追踪、可查询、可审计的任务记录。

设计对齐：
- docs/adr/000-scope-first-response.md（两周范围）
- docs/adr/002-diagnosis-task-persistence.md（本次决策）
- 复用既有范式：chat_run_trace.py / protocol_ingestion.py
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, Enum, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _utcnow_naive() -> datetime:
    """与项目其余模型保持一致：存 naive UTC 时间。"""
    return datetime.now(UTC).replace(tzinfo=None)


class DiagnosisTaskStatus(StrEnum):
    """诊断任务状态机。

    流转：
        NEW ──▶ PLANNING ──▶ RETRIEVING ──▶ DIAGNOSING ──▶ DONE
                  │             │              │
                  └─────────────┴──────────────┴──────────▶ FAILED

    - NEW: 任务已创建，尚未开始
    - PLANNING: 正在制定诊断计划 / 构造告警上下文
    - RETRIEVING: 正在检索 SOP 与采集证据
    - DIAGNOSING: 正在生成结构化首响报告
    - DONE: 成功产出报告
    - FAILED: 任一阶段异常，error_message 记录原因
    """

    NEW = "new"
    PLANNING = "planning"
    RETRIEVING = "retrieving"
    DIAGNOSING = "diagnosing"
    DONE = "done"
    FAILED = "failed"


# 允许的状态跃迁表（用于 repository 层做非法跳转防护）
_ALLOWED_TRANSITIONS: dict[DiagnosisTaskStatus, set[DiagnosisTaskStatus]] = {
    DiagnosisTaskStatus.NEW: {DiagnosisTaskStatus.PLANNING, DiagnosisTaskStatus.FAILED},
    DiagnosisTaskStatus.PLANNING: {
        DiagnosisTaskStatus.RETRIEVING,
        DiagnosisTaskStatus.FAILED,
    },
    DiagnosisTaskStatus.RETRIEVING: {
        DiagnosisTaskStatus.DIAGNOSING,
        DiagnosisTaskStatus.FAILED,
    },
    DiagnosisTaskStatus.DIAGNOSING: {
        DiagnosisTaskStatus.DONE,
        DiagnosisTaskStatus.FAILED,
    },
    DiagnosisTaskStatus.DONE: set(),
    DiagnosisTaskStatus.FAILED: set(),
}


def can_transition(src: DiagnosisTaskStatus, dst: DiagnosisTaskStatus) -> bool:
    """判断状态跃迁是否合法。相同状态视为幂等允许。"""
    if src == dst:
        return True
    return dst in _ALLOWED_TRANSITIONS.get(src, set())


class AiopsDiagnosisTask(Base):
    """一次告警首响诊断任务。"""

    __tablename__ = "aiops_diagnosis_tasks"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid4())
    )
    # request_id：串联「创建任务的 HTTP 请求」与「后续异步执行的日志」。
    # 告警诊断是跨进程的（API 建任务 → worker 跑分析），
    # 没有这个字段就无法把 worker 里的日志追回到最初那次调用。
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # 告警标识：alert_id 可来自外部系统的 fingerprint，也可为空（手工触发）
    alert_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    alert_name: Mapped[str] = mapped_column(String(255), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")

    # 关联会话（复用现有 conversation_sessions；可为空以便独立触发）
    session_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("conversation_sessions.id"),
        nullable=True,
        index=True,
    )

    status: Mapped[DiagnosisTaskStatus] = mapped_column(
        Enum(
            DiagnosisTaskStatus,
            values_callable=lambda items: [item.value for item in items],
        ),
        nullable=False,
        default=DiagnosisTaskStatus.NEW,
        index=True,
    )
    current_phase: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # 归一化后的告警事件全量留档
    alarm_event: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    # 结构化首响报告（Day4 产出），可为空
    report: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # 诊断摘要（人可读的一句话）
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow_naive, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utcnow_naive,
        onupdate=_utcnow_naive,
    )

    events: Mapped[list["AiopsDiagnosisEvent"]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="AiopsDiagnosisEvent.created_at",
    )


class AiopsDiagnosisEvent(Base):
    """诊断过程中的一条事件（时间线留痕）。

    Day9 的 SSE 时间线会读取这些事件；这里先把「留痕」的地基打好。
    """

    __tablename__ = "aiops_diagnosis_events"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid4())
    )
    task_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("aiops_diagnosis_tasks.id"),
        nullable=False,
        index=True,
    )
    phase: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow_naive, index=True
    )

    task: Mapped[AiopsDiagnosisTask] = relationship(back_populates="events")
