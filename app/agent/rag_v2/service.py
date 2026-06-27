"""RAG v2 服务层

对外暴露 query / query_stream 两种调用模式,
风格对齐 app.services.rag_agent_service.rag_agent_service。
"""

from typing import Any, AsyncGenerator, Dict, List

from langchain_core.documents import Document
from loguru import logger

from app.agent.rag_v2.graph import rag_v2_graph


class RagV2Service:
    """多查询改写并行检索 RAG"""

    def __init__(self):
        self.graph = rag_v2_graph
        logger.info("RAG v2 服务初始化完成 (multi-query rewrite)")

    async def query(self, question: str, session_id: str = "") -> Dict[str, Any]:
        logger.info(f"[rag_v2:{session_id}] 收到查询: {question}")

        result = await self.graph.ainvoke({"question": question})

        return {
            "answer": result.get("answer", ""),
            "sub_queries": result.get("sub_queries", []),
            "retrieved_count": len(result.get("documents", []) or []),
            "used_documents": _serialize_docs(result.get("deduped_documents", []) or []),
            "validation": result.get("validation", {}),
        }

    async def query_stream(
        self,
        question: str,
        session_id: str = "",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        logger.info(f"[rag_v2:{session_id}] 收到流式查询: {question}")

        try:
            async for update in self.graph.astream(
                {"question": question},
                stream_mode="updates",
            ):
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

        except Exception as e:
            logger.error(f"[rag_v2:{session_id}] 流式查询失败: {e}")
            yield {"type": "error", "data": str(e)}
            raise


def _serialize_docs(docs: List[Document]) -> List[Dict[str, Any]]:
    return [
        {"content": doc.page_content, "metadata": doc.metadata or {}}
        for doc in docs
    ]


rag_v2_service = RagV2Service()
