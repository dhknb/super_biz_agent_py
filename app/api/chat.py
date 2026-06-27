"""Chat API routes."""

import json

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from app.core.database import get_db
from app.models.request import ChatRequest, ClearRequest
from app.models.response import ApiResponse, SessionInfoResponse
from app.repositories.conversation_repository import ConversationRepository
from app.services.rag_agent_service import rag_agent_service

router = APIRouter()


@router.post("/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Return a complete chat answer."""
    try:
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

    except Exception as e:
        logger.error(f"chat endpoint error: {e}")
        return {
            "code": 500,
            "message": "error",
            "data": {
                "success": False,
                "answer": None,
                "errorMessage": str(e),
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
            yield {
                "event": "message",
                "data": json.dumps({"type": "error", "data": str(e)}, ensure_ascii=False),
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
