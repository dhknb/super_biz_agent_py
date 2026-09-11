"""Repository for AIOps 诊断任务与事件。

职责：只负责「怎么存取诊断任务」，不含任何业务编排逻辑。
业务流程（planning→retrieving→...）由 service 层驱动。
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.request_context import get_request_id_or_none
from app.models.aiops_diagnosis import (
    AiopsDiagnosisEvent,
    AiopsDiagnosisTask,
    DiagnosisTaskStatus,
    can_transition,
)
from app.models.conversation import ConversationSession


class AiopsDiagnosisRepository:
    def __init__(self, db: Session):
        self.db = db

    # ── 创建 ────────────────────────────────────────────────
    def create_task(
        self,
        *,
        alarm_event: dict[str, Any],
        alert_name: str,
        severity: str = "unknown",
        source: str = "manual",
        alert_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> AiopsDiagnosisTask:
        """创建一个 NEW 状态的诊断任务。

        request_id 默认从当前请求上下文自动取（与 chat trace 同一套机制），
        这样「触发告警分析的那次 HTTP 请求」和「后续 worker 里的诊断日志」
        能用同一个 id 串起来。
        """
        if session_id:
            self._ensure_session(session_id)
        task = AiopsDiagnosisTask(
            alert_id=alert_id,
            alert_name=alert_name,
            severity=severity,
            source=source,
            session_id=session_id,
            request_id=request_id or get_request_id_or_none(),
            alarm_event=alarm_event,
            status=DiagnosisTaskStatus.NEW,
        )
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        return task

    # ── 查询 ────────────────────────────────────────────────
    def get_task(self, task_id: str) -> AiopsDiagnosisTask | None:
        return self.db.get(AiopsDiagnosisTask, task_id)

    def list_tasks(
        self,
        *,
        limit: int = 50,
        status: DiagnosisTaskStatus | None = None,
    ) -> list[AiopsDiagnosisTask]:
        stmt = select(AiopsDiagnosisTask)
        if status is not None:
            stmt = stmt.where(AiopsDiagnosisTask.status == status)
        stmt = stmt.order_by(AiopsDiagnosisTask.created_at.desc()).limit(limit)
        return list(self.db.scalars(stmt).all())

    def list_events(self, task_id: str) -> list[AiopsDiagnosisEvent]:
        stmt = (
            select(AiopsDiagnosisEvent)
            .where(AiopsDiagnosisEvent.task_id == task_id)
            .order_by(AiopsDiagnosisEvent.created_at)
        )
        return list(self.db.scalars(stmt).all())

    # ── 状态流转 ─────────────────────────────────────────────
    def _apply_status(
        self,
        task: AiopsDiagnosisTask,
        status: DiagnosisTaskStatus,
        *,
        phase: str | None = None,
    ) -> None:
        """就地推进状态，**不提交**。非法跃迁抛 ValueError。

        抽成不提交的私有方法，是为了让「推进状态」和「落一条事件」
        能被组合进同一个事务（见 record_phase）。
        状态跃迁的合法性校验只有这一处实现（DRY）。
        """
        if not can_transition(task.status, status):
            raise ValueError(
                f"非法状态跃迁: {task.status.value} → {status.value}"
            )
        task.status = status
        task.current_phase = phase or status.value

    def _new_event(
        self,
        task_id: str,
        *,
        phase: str,
        message: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> AiopsDiagnosisEvent:
        """构造并 add 一条事件，**不提交**。"""
        event = AiopsDiagnosisEvent(
            task_id=task_id,
            phase=phase,
            message=message,
            payload=payload,
        )
        self.db.add(event)
        return event

    def advance_status(
        self,
        task: AiopsDiagnosisTask,
        status: DiagnosisTaskStatus,
        *,
        phase: str | None = None,
    ) -> AiopsDiagnosisTask:
        """推进任务状态并提交，非法跃迁抛 ValueError。"""
        self._apply_status(task, status, phase=phase)
        self.db.commit()
        self.db.refresh(task)
        return task

    def record_phase(
        self,
        task: AiopsDiagnosisTask,
        *,
        phase: str,
        status: DiagnosisTaskStatus | None = None,
        message: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> AiopsDiagnosisEvent:
        """一次事务内完成「推进状态 + 追加时间线事件」。

        **为什么要有这个方法**

        原来编排层每个阶段都要写两行：`advance_status(...)` 再 `add_event(...)`，
        两个方法各自 `commit()`。一次告警诊断因此产生 9 次提交，
        而真正的业务边界只有 5 个。多出来的往返除了拖慢链路，
        还制造了一个可观测的不一致窗口：状态已经是 RETRIEVING，
        但时间线上还没有对应的事件 —— 此刻查详情接口会看到一个
        「正在检索但没人记录它开始了」的任务。

        状态与事件是同一件事的两个面（一个给程序判断，一个给人阅读），
        本就该原子地一起落库。

        `status=None` 表示只记事件不改状态 —— 用于阶段**结束**时补记耗时，
        那不是一次新的状态跃迁。
        """
        if status is not None:
            self._apply_status(task, status, phase=phase)
        event = self._new_event(task.id, phase=phase, message=message, payload=payload)
        self.db.commit()
        # 不 refresh：SessionLocal 配了 expire_on_commit=False，
        # 提交后对象属性依然可读，refresh 只是白跑一次 SELECT。
        return event

    def mark_failed(
        self,
        task: AiopsDiagnosisTask,
        *,
        error_message: str,
        event_message: str | None = None,
        event_payload: dict[str, Any] | None = None,
    ) -> AiopsDiagnosisTask:
        """把任务标记为 FAILED，记录错误原因。任何状态都可转 FAILED。

        `event_message` 非空时，顺手在**同一个事务**里落一条 `failed` 事件。
        失败路径上这一点尤其重要：原来是 mark_failed 提交完、再 add_event 提交，
        中间那一瞬如果进程被 kill，任务已经是 FAILED 但时间线上找不到原因 ——
        排障时最需要的那条记录，恰好是最容易丢的一条。
        """
        task.status = DiagnosisTaskStatus.FAILED
        task.current_phase = "failed"
        task.error_message = error_message
        if event_message is not None:
            self._new_event(
                task.id,
                phase="failed",
                message=event_message,
                payload=event_payload,
            )
        self.db.commit()
        return task

    def save_report(
        self,
        task: AiopsDiagnosisTask,
        *,
        report: dict[str, Any],
        summary: str | None = None,
        event_message: str | None = None,
        event_payload: dict[str, Any] | None = None,
    ) -> AiopsDiagnosisTask:
        """保存结构化报告并把任务置为 DONE。

        `event_message` 非空时，`done` 事件与报告在**同一个事务**里落库 ——
        理由和 mark_failed 相同：报告已存但时间线停在 diagnosing 的任务，
        看起来像是「卡住了」，而它其实早就跑完了。
        """
        if not can_transition(task.status, DiagnosisTaskStatus.DONE):
            raise ValueError(
                f"非法状态跃迁: {task.status.value} → done"
            )
        task.report = report
        task.summary = summary
        task.status = DiagnosisTaskStatus.DONE
        task.current_phase = "done"
        task.error_message = None
        if event_message is not None:
            self._new_event(
                task.id,
                phase="done",
                message=event_message,
                payload=event_payload,
            )
        self.db.commit()
        return task

    # ── 事件留痕 ─────────────────────────────────────────────
    def add_event(
        self,
        task_id: str,
        *,
        phase: str,
        message: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> AiopsDiagnosisEvent:
        """向任务追加一条时间线事件（独立事务）。

        当「记事件」不伴随任何状态变化、也不和别的写操作同批时用它；
        否则优先用 record_phase / save_report / mark_failed 的内联事件参数，
        少一次提交，也少一个不一致窗口。
        """
        event = self._new_event(
            task_id,
            phase=phase,
            message=message,
            payload=payload,
        )
        self.db.commit()
        return event

    # ── 内部 ────────────────────────────────────────────────
    def _ensure_session(self, session_id: str) -> None:
        """会话不存在时补一条，避免外键约束失败（复用现有范式）。"""
        session = self.db.get(ConversationSession, session_id)
        if session is None:
            self.db.add(ConversationSession(id=session_id))
            self.db.flush()
