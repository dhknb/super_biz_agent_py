"""RAG v2 服务层

对外暴露 query / query_stream 两种调用模式,
风格对齐 app.services.rag_agent_service.rag_agent_service。

## 关于总预算（chat_total_budget_seconds）

llm_factory 里的 `timeout` 管的是**单次** LLM 调用。
但一次 chat_v2 请求会串行调用 LLM 三次（rewrite → generate → validate），
再加上并行检索的耗时 —— 最坏情况远超单次超时值。
所以这里必须再套一层**整体预算**，它回答的是另一个问题：
「这个用户最多等多久」，而单次超时回答的是「这次调用最多挂多久」。
两者层级不同，缺一不可。

## 为什么用 astream 累积状态，而不是 ainvoke

`asyncio.timeout` 触发时会取消里面的任务。如果用 `ainvoke`，
取消意味着**整个结果全丢**，我们只能告诉用户「超时了」，
而实际上改写和检索可能早已完成 —— 那些工作白做了，
排查时也看不到「到底卡在哪一步」。

`astream(stream_mode="values")` 每个超步结束都会吐出**完整快照**，
我们把最后一个快照留在手里。超时时就能回答两个关键问题：
    1. 已经拿到了什么（子查询、召回文档，可以返回给用户）
    2. 卡在了哪一步（最后一次快照缺哪个字段）
这才是「返回部分结果 + 降级说明」，而不是干巴巴一句超时。
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List

from langchain_core.documents import Document
from loguru import logger

from app.agent.rag_v2.graph import rag_v2_graph
from app.config import config
from app.core.errors import DegradeReason


class RagV2Service:
    """多查询改写并行检索 RAG"""

    def __init__(self):
        self.graph = rag_v2_graph
        logger.info("RAG v2 服务初始化完成 (multi-query rewrite)")

    async def query(
        self,
        question: str,
        session_id: str = "",
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        logger.info(f"[rag_v2:{session_id}] 收到查询: {question}")

        inputs = {"question": question, "conversation_context": conversation_context}
        # 保存最后一个完整状态快照：超时被取消时，这是我们唯一还握在手里的东西。
        last_state: Dict[str, Any] = {}
        timed_out = False

        try:
            async with asyncio.timeout(config.chat_total_budget_seconds):
                async for state in self.graph.astream(inputs, stream_mode="values"):
                    if isinstance(state, dict):
                        last_state = state
        except asyncio.TimeoutError:
            # 不向上抛：预算耗尽是**可降级**的失败，我们有部分结果可以交付。
            # 抛异常会让用户既等满了预算又什么都拿不到，是最差的结果。
            timed_out = True
            logger.warning(
                f"[rag_v2:{session_id}] 总预算 {config.chat_total_budget_seconds}s 耗尽，"
                f"返回部分结果; 已有字段={sorted(last_state.keys())}"
            )

        result = _build_query_result(last_state)
        if timed_out:
            result["degrade_reason"] = DegradeReason.TOTAL_BUDGET_EXCEEDED.value
            result["answer"] = result["answer"] or _budget_exceeded_answer(last_state)
        return result

    async def query_stream(
        self,
        question: str,
        session_id: str = "",
        conversation_context: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        logger.info(f"[rag_v2:{session_id}] 收到流式查询: {question}")

        inputs = {"question": question, "conversation_context": conversation_context}
        try:
            # 流式路径同样受总预算约束。没有它，SSE 连接会一直挂着 ——
            # 而流式场景更危险：客户端看到连接是「开着」的，会一直等下去，
            # 不像普通请求至少有客户端侧超时兜底。
            async with asyncio.timeout(config.chat_total_budget_seconds):
                async for update in self.graph.astream(inputs, stream_mode="updates"):
                    for node_name, patch in update.items():
                        if not isinstance(patch, dict):
                            continue

                        if "sub_queries" in patch:
                            yield {"type": "sub_queries", "data": patch["sub_queries"]}
                        if "documents" in patch:
                            yield {
                                "type": "retrieved",
                                "data": {"node": node_name, "count": len(patch["documents"])},
                            }
                        if "deduped_documents" in patch:
                            yield {
                                "type": "used_documents",
                                "data": _serialize_docs(patch["deduped_documents"]),
                            }
                        if "answer" in patch:
                            yield {"type": "answer", "data": patch["answer"]}
                        if "validation" in patch:
                            yield {"type": "validation", "data": patch["validation"]}

            yield {"type": "complete"}

        except asyncio.TimeoutError:
            # SSE 已经返回了 HTTP 200，改不了状态码，只能用事件告知降级。
            # 关键是带上 degrade_reason：前端和日志都能区分
            # 「预算耗尽」和「LLM 报错」，两者的处置完全不同。
            logger.warning(
                f"[rag_v2:{session_id}] 流式查询总预算 "
                f"{config.chat_total_budget_seconds}s 耗尽"
            )
            yield {
                "type": "error",
                "data": f"本次回答超出 {config.chat_total_budget_seconds} 秒预算，已中断",
                "degrade_reason": DegradeReason.TOTAL_BUDGET_EXCEEDED.value,
            }
            # 不 raise：前面已经 yield 过若干有效事件，
            # 此时抛异常会让 SSE 连接以错误方式断开，客户端可能丢掉已收到的内容。

        except Exception as e:
            logger.error(f"[rag_v2:{session_id}] 流式查询失败: {e}")
            yield {"type": "error", "data": str(e)}
            raise


def _build_query_result(state: Dict[str, Any]) -> Dict[str, Any]:
    """从图状态构造对外返回结构。

    抽出来是因为正常完成和预算耗尽两条路径都要用它 ——
    字段名和取值逻辑只能有一处（DRY），否则超时路径迟早和正常路径长得不一样。
    """
    return {
        "answer": state.get("answer", "") or "",
        "sub_queries": state.get("sub_queries", []) or [],
        "retrieved_count": len(state.get("documents", []) or []),
        "used_documents": _serialize_docs(state.get("deduped_documents", []) or []),
        "validation": state.get("validation", {}) or {},
    }


def _budget_exceeded_answer(state: Dict[str, Any]) -> str:
    """预算耗尽且还没生成答案时，给用户一句诚实的说明。

    刻意说清「卡在哪一步」而不是只说「超时」：
    检索到文档但没出答案 → 生成阶段慢，可能要调模型或加大预算；
    连文档都没有 → 检索阶段慢，该去看向量库。
    对用户是交代，对值班同学是线索。
    """
    doc_count = len(state.get("documents", []) or [])
    if doc_count:
        return (
            f"本次回答超出 {config.chat_total_budget_seconds} 秒预算，"
            f"已检索到 {doc_count} 条相关资料但未能完成答案生成，请重试或缩小问题范围。"
        )
    return (
        f"本次回答超出 {config.chat_total_budget_seconds} 秒预算，"
        "检索阶段未能完成，请稍后重试。"
    )


def _serialize_docs(docs: List[Document]) -> List[Dict[str, Any]]:
    return [
        {"content": doc.page_content, "metadata": doc.metadata or {}}
        for doc in docs
    ]


rag_v2_service = RagV2Service()
