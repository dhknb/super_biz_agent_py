"""Repository for chat_v2 run trace persistence."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.request_context import get_request_id_or_none
from app.core.used_documents import slim_used_documents
from app.models.chat_run_trace import ChatRunTrace, ChatRunTraceStatus
from app.models.conversation import ConversationSession


class ChatRunTraceRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_trace(
        self,
        *,
        session_id: str,
        source: str,
        question: str,
        answer: str | None,
        sub_queries: list[str],
        retrieved_count: int,
        used_documents: list[dict],
        validation: dict | None,
        error_message: str | None = None,
        request_id: str | None = None,
    ) -> ChatRunTrace:
        # request_id 默认从当前请求上下文自动取，调用方不必显式传。
        # 这样做的理由：现有调用点（chat_v2 的正常路径与异常路径）一行都不用改，
        # 同时又保留显式传参的能力给 worker / 测试用。
        request_id = request_id or get_request_id_or_none()
        # 瘦身放在 repository 而不是三个调用点里：非流式成功、非流式失败、
        # 流式，三条路都要经过这里。放在这一处，将来新增调用点也不会漏掉
        # （DRY）。调用方传全文还是传瘦身格式都行 —— slim_used_documents 幂等。
        used_documents = slim_used_documents(used_documents)
        self._ensure_session(session_id)
        coverage_score = _coerce_optional_float((validation or {}).get("coverage_score"))
        groundedness_score = _coerce_optional_float((validation or {}).get("groundedness_score"))
        is_bad_case = _is_bad_case(answer=answer, validation=validation, error_message=error_message)
        trace = ChatRunTrace(
            session_id=session_id,
            request_id=request_id,
            source=source,
            question=question,
            answer=answer,
            sub_queries=sub_queries,
            retrieved_count=retrieved_count,
            used_documents=used_documents,
            validation=validation,
            coverage_score=coverage_score,
            groundedness_score=groundedness_score,
            is_bad_case=is_bad_case,
            status=ChatRunTraceStatus.ERROR if error_message else ChatRunTraceStatus.SUCCEEDED,
            error_message=error_message,
        )
        self.db.add(trace)
        self.db.commit()
        self.db.refresh(trace)
        return trace

    def get_trace(self, trace_id: str) -> ChatRunTrace | None:
        return self.db.get(ChatRunTrace, trace_id)

    def list_bad_cases(
        self,
        *,
        limit: int = 50,
        session_id: str | None = None,
    ) -> list[ChatRunTrace]:
        stmt = select(ChatRunTrace).where(ChatRunTrace.is_bad_case.is_(True))
        if session_id:
            stmt = stmt.where(ChatRunTrace.session_id == session_id)
        stmt = stmt.order_by(ChatRunTrace.created_at.desc()).limit(limit)
        return list(self.db.scalars(stmt).all())

    def _ensure_session(self, session_id: str) -> None:
        session = self.db.get(ConversationSession, session_id)
        if session is None:
            self.db.add(ConversationSession(id=session_id))
            self.db.flush()


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_bad_case(*, answer: str | None, validation: dict | None, error_message: str | None) -> bool:
    if error_message:
        return True
    if not answer or "根据已有资料无法回答" in answer:
        return True
    if not validation:
        return False
    if bool(validation.get("blocked")):
        return True
    coverage_score = _coerce_optional_float(validation.get("coverage_score"))
    groundedness_score = _coerce_optional_float(validation.get("groundedness_score"))
    thresholds = validation.get("thresholds") or {}
    coverage_min = _coerce_optional_float(thresholds.get("coverage_min"))
    groundedness_min = _coerce_optional_float(thresholds.get("groundedness_min"))
    if coverage_score is not None and coverage_min is not None and coverage_score < coverage_min:
        return True
    if groundedness_score is not None and groundedness_min is not None and groundedness_score < groundedness_min:
        return True
    return False
