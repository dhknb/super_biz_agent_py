"""告警诊断编排服务 —— 把「一条告警」跑成「一个落库的诊断任务 + 结构化报告」。

这是 Day6 的收口：把前几天的零件（归一化 / 任务化 / 首响分析）串成
一条端到端可调用、可追踪、可审计的链路。

    payload + source
      → normalize_alarm（Day2）
      → 创建诊断任务 NEW（Day3）
      → planning → retrieving → diagnosing（状态流转 + 事件留痕）
      → FirstResponseService.analyze（Day4/5：SOP 检索 + 结构化报告）
      → save_report → DONE

分层说明：
- 本编排层负责「流程」，依赖 repository（怎么存）与 first_response_service（怎么分析）。
- API 层只负责收参数、调本服务、序列化返回，保持很薄。
- first_response_service.analyze 内部已对所有异常降级，不会抛；
  本层只需处理归一化阶段的 ValueError（未知来源）与兜底异常。

关联 ADR：docs/adr/002-diagnosis-task-persistence.md、
         docs/adr/004-sop-retrieval-and-evidence-tracing.md
"""

from __future__ import annotations

from typing import Any, Callable

from loguru import logger
from sqlalchemy.orm import Session

from app.models.aiops import AlarmEvent, AlarmSource, normalize_alarm
from app.models.aiops_diagnosis import AiopsDiagnosisTask, DiagnosisTaskStatus
from app.models.aiops_report import FirstResponseReport
from app.repositories.aiops_diagnosis_repository import AiopsDiagnosisRepository
from app.services.first_response_service import (
    PHASE_DIAGNOSED,
    PHASE_DIAGNOSING,
    PHASE_RETRIEVED,
    PHASE_RETRIEVING,
    FirstResponseService,
    first_response_service,
)

# analyze 回调的阶段名 → 该阶段对应的状态跃迁。
#
# 只有「开始做某件事」才是新状态；「某件事做完了」不是。
# retrieved / diagnosed 不在这张表里，因为检索结束的那一刻任务仍在
# RETRIEVING —— 它只是终于知道了这一步花了多久，可以补记一条事件。
# 强行给它们发明状态，会让状态机从 5 个状态膨胀到 7 个，
# 而多出来的两个没有任何代码需要分支判断（YAGNI）。
_PHASE_STATUS: dict[str, DiagnosisTaskStatus] = {
    PHASE_RETRIEVING: DiagnosisTaskStatus.RETRIEVING,
    PHASE_DIAGNOSING: DiagnosisTaskStatus.DIAGNOSING,
}

# 阶段名 → 给人看的中文描述。放在编排层而不是 analyze 里，
# 是因为 analyze 只该知道「我到哪一步了」，不该知道「这一步在时间线上叫什么」。
_PHASE_MESSAGES: dict[str, str] = {
    PHASE_RETRIEVING: "检索 SOP 与采集证据",
    PHASE_RETRIEVED: "SOP 检索完成",
    PHASE_DIAGNOSING: "生成结构化首响报告",
    PHASE_DIAGNOSED: "报告生成完成",
}


class AlertDiagnosisOrchestrator:
    """协调 repository 与首响分析服务，产出可追踪的诊断任务。"""

    def __init__(self, response_service: FirstResponseService | None = None):
        self._response_service = response_service or first_response_service

    async def run(
        self,
        db: Session,
        *,
        payload: dict[str, Any],
        source: AlarmSource | str = AlarmSource.MANUAL,
        session_id: str | None = None,
    ) -> tuple[AiopsDiagnosisTask, FirstResponseReport]:
        """跑完整条首响链路，返回 (诊断任务, 结构化报告)。

        归一化失败（未知来源等）会抛 ValueError，交由 API 层转 400。
        分析阶段的异常已被 first_response_service 降级，不会中断链路；
        本层仍用 try/except 兜底，确保任务状态一定落到 DONE 或 FAILED。
        """
        alarm: AlarmEvent = normalize_alarm(payload, source)
        repo = AiopsDiagnosisRepository(db)

        task = repo.create_task(
            alarm_event=alarm.model_dump(mode="json"),
            alert_name=alarm.alert_name,
            severity=alarm.severity.value,
            source=alarm.source.value,
            alert_id=str(payload.get("alert_id")) if payload.get("alert_id") else None,
            session_id=session_id,
        )
        logger.info(f"诊断任务已创建: task_id={task.id}, alert={alarm.alert_name}")

        try:
            # planning 是唯一「在 analyze 之前就真的发生完了」的阶段：
            # 归一化 + 任务落库确实已经做完，此刻推进它不算撒谎。
            repo.record_phase(
                task,
                phase="planning",
                status=DiagnosisTaskStatus.PLANNING,
                message="构造告警上下文",
            )

            # 其余阶段交给 analyze 在真实边界回调，编排层不再预先推进。
            report = await self._response_service.analyze(
                alarm,
                on_phase=self._make_phase_handler(repo, task),
            )

            repo.save_report(
                task,
                report=report.model_dump(mode="json"),
                summary=report.alert_summary,
                event_message="首响报告已生成",
                event_payload={"is_degraded": report.is_degraded},
            )
            logger.info(f"诊断任务完成: task_id={task.id}, degraded={report.is_degraded}")
            return task, report

        except Exception as exc:  # 兜底：任何意外都让任务落到 FAILED，不留悬空态
            logger.exception(f"诊断任务失败: task_id={task.id}")
            repo.mark_failed(
                task,
                error_message=str(exc),
                event_message=str(exc),
            )
            raise

    def _make_phase_handler(
        self,
        repo: AiopsDiagnosisRepository,
        task: AiopsDiagnosisTask,
    ) -> Callable[[str, dict[str, Any]], None]:
        """造一个把 analyze 的阶段回调翻译成「状态流转 + 时间线事件」的处理器。

        `_PHASE_STATUS` 里有的阶段是真实的状态跃迁；没有的（retrieved /
        diagnosed）只记事件，因为「检索结束」不是一个新状态 —— 任务此刻
        仍在 RETRIEVING，只是这一步的耗时终于可以落库了。
        """

        def handle(phase: str, payload: dict[str, Any]) -> None:
            repo.record_phase(
                task,
                phase=phase,
                status=_PHASE_STATUS.get(phase),
                message=_PHASE_MESSAGES.get(phase),
                payload=payload or None,
            )

        return handle


# 全局单例
alert_diagnosis_orchestrator = AlertDiagnosisOrchestrator()
