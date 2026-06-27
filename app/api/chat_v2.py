"""Chat v2 API routes for multi-query RAG."""

import json

from fastapi import APIRouter, Depends
from loguru import logger
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from app.agent.rag_v2 import rag_v2_service
from app.core.database import get_db
from app.models.request import ChatRequest
from app.repositories.conversation_repository import ConversationRepository

router = APIRouter()


@router.post("/chat_v2")
async def chat_v2(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        logger.info(f"[session {request.id}] chat v2 request received: {request.question}")
        result = await rag_v2_service.query(request.question, session_id=request.id)
        ConversationRepository(db).append_exchange(
            request.id,
            user_content=request.question,
            assistant_content=result["answer"],
            message_metadata={"source": "chat_v2"},
        )
        logger.info(f"[session {request.id}] chat v2 request completed")

        return {
            "code": 200,
            "message": "success",
            "data": {
                "success": True,
                "answer": result["answer"],
                "sub_queries": result["sub_queries"],
                "retrieved_count": result["retrieved_count"],
                "used_documents": result["used_documents"],
                "validation": result.get("validation", {}),
                "errorMessage": None,
            },
        }

    except Exception as e:
        logger.error(f"chat v2 endpoint error: {e}")
        return {
            "code": 500,
            "message": "error",
            "data": {
                "success": False,
                "answer": None,
                "errorMessage": str(e),
            },
        }


@router.post("/chat_v2_stream")
async def chat_v2_stream(request: ChatRequest, db: Session = Depends(get_db)):
    logger.info(f"[session {request.id}] chat v2 stream request received: {request.question}")

    async def event_generator():
        answer = ""
        try:
            async for chunk in rag_v2_service.query_stream(
                request.question,
                session_id=request.id,
            ):
                chunk_type = chunk.get("type", "unknown")
                payload = chunk.get("data")
                if chunk_type == "answer" and payload:
                    answer = str(payload)

                if chunk_type == "complete":
                    if answer:
                        ConversationRepository(db).append_exchange(
                            request.id,
                            user_content=request.question,
                            assistant_content=answer,
                            message_metadata={"source": "chat_v2_stream"},
                        )
                    yield {
                        "event": "message",
                        "data": json.dumps({"type": "done"}, ensure_ascii=False),
                    }
                    continue

                yield {
                    "event": "message",
                    "data": json.dumps(
                        {"type": chunk_type, "data": payload},
                        ensure_ascii=False,
                    ),
                }

            logger.info(f"[session {request.id}] chat v2 stream request completed")

        except Exception as e:
            logger.error(f"chat v2 stream endpoint error: {e}")
            yield {
                "event": "message",
                "data": json.dumps({"type": "error", "data": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())
