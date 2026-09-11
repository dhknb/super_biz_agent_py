"""Chat API routes."""

import json

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from app.core.database import get_db
from app.core.errors import degrade_reason_of, error_code_of
from app.core.request_context import get_request_id_or_none
from app.models.request import ChatRequest, ClearRequest
from app.models.response import ApiResponse, SessionInfoResponse
from app.repositories.conversation_repository import ConversationRepository
from app.services.rag_agent_service import rag_agent_service

router = APIRouter()


@router.post("/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Return a complete chat answer.

    这里**没有** try/except —— 这不是遗漏，是刻意的。

    原来的写法是 `except Exception: return {"code": 500, ...}`，
    HTTP 状态码却是 200。整条监控链路只读状态码：Nginx 的 $status、
    负载均衡的健康检查、APM 的错误率、Prometheus 的
    http_requests_total{status="5xx"} —— 没有一个会去解析响应体里的
    "code": 500。结果就是接口全挂了，错误率仍然显示 0%。

    而这个 except 块除了拼一个响应体之外**没有任何副作用**（不像
    chat_v2 那样要落 trace），所以最干净的处置是整块删掉，让异常
    自然冒泡到 app/core/exception_handlers.py：
      - 状态码按异常类型裁决(超时→504、上游不可用→502、其余→500)
      - 日志由全局处理器打，还多了完整堆栈和请求路径
      - 响应体结构与原来完全一致，另外附上
        error_code / degrade_reason / request_id

    错误响应体的构造从此只有一处实现(DRY)，新接口不会再漏掉它。
    """
    logger.info(f"[session {request.id}] chat request received: {request.question}")
    answer = await rag_agent_service.query(
        request.question,
        session_id=request.id,
    )
    ConversationRepository(db).append_exchange(
        request.id,
        user_content=request.question,
        assistant_content=answer,
    )

    logger.info(f"[session {request.id}] chat request completed")

    return {
        "code": 200,
        "message": "success",
        "data": {
            "success": True,
            "answer": answer,
            "errorMessage": None,
        },
    }


@router.post("/chat_stream")
async def chat_stream(request: ChatRequest, db: Session = Depends(get_db)):
    """Stream a chat answer over SSE."""
    logger.info(f"[session {request.id}] chat stream request received: {request.question}")

    async def event_generator():
        answer_parts = []
        try:
            async for chunk in rag_agent_service.query_stream(
                request.question,
                session_id=request.id,
            ):
                chunk_type = chunk.get("type", "unknown")
                chunk_data = chunk.get("data", None)

                if chunk_type == "debug":
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "type": "debug",
                                "node": chunk.get("node", "unknown"),
                                "message_type": chunk.get("message_type", "unknown"),
                            },
                            ensure_ascii=False,
                        ),
                    }
                elif chunk_type == "tool_call":
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"type": "tool_call", "data": chunk_data},
                            ensure_ascii=False,
                        ),
                    }
                elif chunk_type == "search_results":
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"type": "search_results", "data": chunk_data},
                            ensure_ascii=False,
                        ),
                    }
                elif chunk_type == "content":
                    if chunk_data:
                        answer_parts.append(str(chunk_data))
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"type": "content", "data": chunk_data},
                            ensure_ascii=False,
                        ),
                    }
                elif chunk_type == "complete":
                    answer = "".join(answer_parts)
                    if answer:
                        ConversationRepository(db).append_exchange(
                            request.id,
                            user_content=request.question,
                            assistant_content=answer,
                        )
                        answer_parts.clear()
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"type": "done", "data": chunk_data},
                            ensure_ascii=False,
                        ),
                    }
                elif chunk_type == "error":
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"type": "error", "data": str(chunk_data)},
                            ensure_ascii=False,
                        ),
                    }

            logger.info(f"[session {request.id}] chat stream request completed")

        except Exception as e:
            logger.error(f"chat stream endpoint error: {e}")
            # 与 chat_v2_stream 同一套字段：流式接口改不了状态码，
            # 真话只能写进事件体。前端老的 type/data 分支不受影响。
            yield {
                "event": "message",
                "data": json.dumps(
                    {
                        "type": "error",
                        "data": str(e),
                        "error_code": error_code_of(e),
                        "degrade_reason": degrade_reason_of(e).value,
                        "request_id": get_request_id_or_none(),
                    },
                    ensure_ascii=False,
                ),
            }

    return EventSourceResponse(event_generator())


@router.post("/chat/clear", response_model=ApiResponse)
async def clear_session(
    request: ClearRequest,
    db: Session = Depends(get_db),
) -> ApiResponse:
    """Clear a chat session."""
    try:
        success = rag_agent_service.clear_session(request.session_id)
        if success:
            ConversationRepository(db).clear_session(request.session_id)
        logger.info(f"clear session: {request.session_id}, success={success}")

        return ApiResponse(
            status="success" if success else "error",
            message="\u4f1a\u8bdd\u5df2\u6e05\u7a7a" if success else "\u6e05\u7a7a\u4f1a\u8bdd\u5931\u8d25",
            data=None,
        )

    except Exception as e:
        logger.error(f"clear session endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(
    session_id: str,
    db: Session = Depends(get_db),
) -> SessionInfoResponse:
    """Return persisted chat session history."""
    try:
        history = ConversationRepository(db).list_session_history(session_id)
        if not history:
            history = rag_agent_service.get_session_history(session_id)

        return SessionInfoResponse(
            session_id=session_id,
            message_count=len(history),
            history=history,
        )

    except Exception as e:
        logger.error(f"get session endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
