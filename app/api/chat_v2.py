"""Chat v2 API routes for multi-query RAG."""

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from app.agent.rag_v2 import rag_v2_service
from app.core.database import get_db
from app.core.errors import degrade_reason_of, error_code_of
from app.core.request_context import get_request_id_or_none
from app.core.span_context import (
    flush_spans,
    reset_span_collection,
    start_span_collection,
)
from app.core.used_documents import normalize_used_documents
from app.models.request import ChatRequest
from app.repositories.chat_run_trace_repository import ChatRunTraceRepository
from app.repositories.conversation_repository import ConversationRepository
from app.services.conversation_memory_service import conversation_memory_service

router = APIRouter()


async def _load_conversation_memory(db: Session, session_id: str) -> str:
    """记忆加载失败时降级为空上下文，不能让可选记忆阻断聊天主路径。"""
    try:
        return (await conversation_memory_service.build_context(db, session_id)).text
    except Exception as exc:
        logger.warning(f"[session {session_id}] 会话记忆加载失败，降级为空记忆: {exc}")
        return ""


@router.get("/chat_v2/traces/bad-cases")
async def list_chat_v2_bad_cases(
    limit: int = Query(default=50, ge=1, le=200),
    session_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Return recent bad-case traces for chat_v2 debugging and replay."""
    traces = ChatRunTraceRepository(db).list_bad_cases(
        limit=limit,
        session_id=session_id,
    )
    return {
        "code": 200,
        "message": "success",
        "data": [_serialize_trace_summary(trace) for trace in traces],
    }


@router.get("/chat_v2/traces/{trace_id}")
async def get_chat_v2_trace(
    trace_id: str,
    db: Session = Depends(get_db),
):
    """Return a single run trace with full RAG process details."""
    trace = ChatRunTraceRepository(db).get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="chat run trace not found")
    return {
        "code": 200,
        "message": "success",
        "data": _serialize_trace_detail(trace),
    }


@router.post("/chat_v2")
async def chat_v2(request: ChatRequest, db: Session = Depends(get_db)):
    trace_repo = ChatRunTraceRepository(db)
    # span 收集必须在**进图之前**开启。
    #
    # 原因是 ContextVar 的拷贝语义：LangGraph fan-out 并行分支时会
    # copy_context()，子上下文拿到的是「同一个 list 对象」的引用。
    # 所以只有在父上下文（也就是这里）set 一次，各分支 append 的 span
    # 才能汇总到同一个篮子里。如果在图内部才 set，每个分支会各自
    # 建一个 list，父上下文一条都看不到。
    span_token = start_span_collection()
    try:
        logger.info(f"[session {request.id}] chat v2 request received: {request.question}")
        memory_context = await _load_conversation_memory(db, request.id)
        result = await rag_v2_service.query(
            request.question,
            session_id=request.id,
            conversation_context=memory_context,
        )
        high_precision_payload = _build_high_precision_payload(result)
        ConversationRepository(db).append_exchange(
            request.id,
            user_content=request.question,
            assistant_content=result["answer"],
            message_metadata={"source": "chat_v2", "high_precision": high_precision_payload},
        )
        trace = trace_repo.create_trace(
            session_id=request.id,
            source="chat_v2",
            question=request.question,
            answer=result["answer"],
            sub_queries=result.get("sub_queries", []),
            retrieved_count=int(result.get("retrieved_count", 0) or 0),
            used_documents=result.get("used_documents", []),
            validation=result.get("validation"),
        )
        # trace 落库之后才有 trace_id，span 才有地方挂 —— 这是「跑完一次性
        # flush」而不是「节点各自写库」的根本原因。
        flush_spans(db, trace_id=trace.id)
        logger.info(f"[session {request.id}] chat v2 request completed")

        return {
            "code": 200,
            "message": "success",
            "data": {
                "success": True,
                "answer": result["answer"],
                "sub_queries": high_precision_payload["sub_queries"],
                "retrieved_count": high_precision_payload["retrieved_count"],
                "used_documents": high_precision_payload["used_documents"],
                "validation": high_precision_payload["validation"],
                "errorMessage": None,
            },
        }

    except Exception as e:
        # 先落 trace，再往上抛。
        #
        # 顺序很关键：trace 是排障的唯一线索，必须在异常离开这个函数之前写完。
        # 交给全局处理器再落库是做不到的 —— 那里拿不到 session_id / question，
        # 也没有这个请求的 db session。
        logger.error(f"chat v2 endpoint error: {e}")
        trace = trace_repo.create_trace(
            session_id=request.id,
            source="chat_v2",
            question=request.question,
            answer=None,
            sub_queries=[],
            retrieved_count=0,
            used_documents=[],
            validation=None,
            error_message=str(e),
        )
        # 失败路径的 span 比成功路径更值钱：它记着「跑到哪一步炸的、
        # 前面几步各花了多久」。这恰恰是排障第一个要问的问题，
        # 所以这里必须和成功路径一样 flush。
        flush_spans(db, trace_id=trace.id)
        # 改成 raise 而不是 return 200 + code:500。
        # 状态码由 app/core/exception_handlers.py 按异常类型裁决
        # （超时→504、上游不可用→502、其余→500），响应体结构不变。
        raise
    finally:
        # 用 reset(token) 还原，而不是留着不管：
        # 事件循环里的 ContextVar 在同一个 task 内是复用的，
        # 不还原会让下一个请求继承上一个请求的 span 篮子。
        reset_span_collection(span_token)


@router.post("/chat_v2_stream")
async def chat_v2_stream(request: ChatRequest, db: Session = Depends(get_db)):
    logger.info(f"[session {request.id}] chat v2 stream request received: {request.question}")
    memory_context = await _load_conversation_memory(db, request.id)

    async def event_generator():
        answer = ""
        sub_queries: list[str] = []
        used_documents: list[dict] = []
        validation: dict | None = None
        retrieved_count = 0
        trace_repo = ChatRunTraceRepository(db)
        # 收集器必须在**生成器体内**开，不能开在外层 chat_v2_stream 里。
        #
        # 原因是执行时机：外层函数只是构造出 EventSourceResponse 就返回了，
        # 生成器体要等 sse-starlette 真正迭代它时才开始跑 —— 那时外层的栈帧
        # 早已退出，它 set 的 ContextVar 也已随之失效。
        # 开在这里，收集器的生命周期才和「图真正在跑」的那段时间对齐。
        span_token = start_span_collection()
        try:
            async for chunk in rag_v2_service.query_stream(
                request.question,
                session_id=request.id,
                conversation_context=memory_context,
            ):
                chunk_type = chunk.get("type", "unknown")
                payload = chunk.get("data")

                if chunk_type == "sub_queries" and isinstance(payload, list):
                    sub_queries = [str(item) for item in payload]
                elif chunk_type == "used_documents" and isinstance(payload, list):
                    used_documents = payload
                elif chunk_type == "validation" and isinstance(payload, dict):
                    validation = payload
                elif chunk_type == "retrieved" and isinstance(payload, dict):
                    retrieved_count += int(payload.get("count", 0) or 0)
                elif chunk_type == "answer" and payload:
                    answer = str(payload)

                if chunk_type == "complete":
                    if answer:
                        high_precision_payload = _build_high_precision_payload(
                            {
                                "sub_queries": sub_queries,
                                "retrieved_count": retrieved_count,
                                "used_documents": used_documents,
                                "validation": validation,
                            }
                        )
                        ConversationRepository(db).append_exchange(
                            request.id,
                            user_content=request.question,
                            assistant_content=answer,
                            message_metadata={
                                "source": "chat_v2_stream",
                                "high_precision": high_precision_payload,
                            },
                        )
                    trace = trace_repo.create_trace(
                        session_id=request.id,
                        source="chat_v2_stream",
                        question=request.question,
                        answer=answer or None,
                        sub_queries=sub_queries,
                        retrieved_count=retrieved_count,
                        used_documents=used_documents,
                        validation=validation,
                    )
                    # 和非流式路径同一时机：拿到 trace_id 之后一次性落盘。
                    flush_spans(db, trace_id=trace.id)
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
            trace = trace_repo.create_trace(
                session_id=request.id,
                source="chat_v2_stream",
                question=request.question,
                answer=answer or None,
                sub_queries=sub_queries,
                retrieved_count=retrieved_count,
                used_documents=used_documents,
                validation=validation,
                error_message=str(e),
            )
            # 失败路径同样要 flush：span 记着「跑到哪一步断的」，
            # 而流式接口连状态码都改不了，span 是这里唯一的过程线索。
            flush_spans(db, trace_id=trace.id)
            # SSE 不能像普通接口那样 raise。
            #
            # 原因是物理性的：EventSourceResponse 一开始流式输出，响应头
            # （含 200 状态码）就已经发给客户端了。此刻再抛异常，全局处理器
            # 想写 504/502 也写不进去——状态行早就在网线上了。
            #
            # 所以流式接口的「说真话」只能落在**事件体**里:
            # 除了 data(给人看的消息)，额外带上 error_code / degrade_reason
            # / request_id，让前端能区分「超时」和「上游挂了」，
            # 也让排障的人拿着 request_id 直接去日志里捞全链路。
            #
            # 保留 type/data 两个老字段不动 —— 前端现有的
            # `if (msg.type === "error")` 分支照样能跑，新字段是加法。
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
        finally:
            # 生成器也可能被客户端提前断开而不走完 —— 那时 GeneratorExit
            # 会在这里经过，收集器同样要还原，否则这个篮子会一直挂在
            # 当前上下文上，被后续复用同一 task 的请求继承。
            reset_span_collection(span_token)

    return EventSourceResponse(event_generator())


def _serialize_trace_summary(trace) -> dict:
    return {
        "id": trace.id,
        "session_id": trace.session_id,
        "source": trace.source,
        "question": trace.question,
        "answer": trace.answer,
        "coverage_score": trace.coverage_score,
        "groundedness_score": trace.groundedness_score,
        "is_bad_case": trace.is_bad_case,
        "status": trace.status.value if hasattr(trace.status, "value") else str(trace.status),
        "error_message": trace.error_message,
        "created_at": trace.created_at.isoformat() if trace.created_at else None,
    }


def _serialize_trace_detail(trace) -> dict:
    data = _serialize_trace_summary(trace)
    data.update(
        {
            "sub_queries": trace.sub_queries,
            "retrieved_count": trace.retrieved_count,
            # 读侧统一过 normalize：库里同时存在瘦身格式（新）和全文格式（旧），
            # 详情接口不该让前端自己分辨。normalize 后两者都带 content 键，
            # 新数据额外用 excerpt_only=True 声明「这是摘录不是全文」。
            "used_documents": normalize_used_documents(trace.used_documents),
            "validation": trace.validation,
        }
    )
    return data


def _build_high_precision_payload(result: dict) -> dict:
    return {
        "sub_queries": result.get("sub_queries", []),
        "retrieved_count": int(result.get("retrieved_count", 0) or 0),
        "used_documents": result.get("used_documents", []),
        "validation": result.get("validation") or {},
    }
