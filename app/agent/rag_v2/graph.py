"""RAG v2 LangGraph ??"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.agent.rag_v2.nodes import (
    dedup_node,
    generate_node,
    retrieve_each_node,
    rewrite_node,
    validate_answer_node,
)
from app.agent.rag_v2.state import RAGState


def _fanout_to_retrieve(state: RAGState) -> list[Send]:
    sub_queries = state.get("sub_queries", []) or []
    return [Send("retrieve_each", {"query": q}) for q in sub_queries]


def build_rag_v2_graph():
    graph = StateGraph(RAGState)

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve_each", retrieve_each_node)
    graph.add_node("dedup", dedup_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate_answer", validate_answer_node)

    graph.add_edge(START, "rewrite")
    graph.add_conditional_edges("rewrite", _fanout_to_retrieve, ["retrieve_each"])
    graph.add_edge("retrieve_each", "dedup")
    graph.add_edge("dedup", "generate")
    graph.add_edge("generate", "validate_answer")
    graph.add_edge("validate_answer", END)

    return graph.compile()


rag_v2_graph = build_rag_v2_graph()
