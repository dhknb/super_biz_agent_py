"""RAG v2 LangGraph ??"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.agent.rag_v2.instrumentation import instrument_node
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

    # 每个节点都过一遍 instrument_node，拿到节点级耗时 span。
    #
    # 为什么统一在这里包，而不是在节点体内计时：
    # 五个节点写五份计时代码是重复，且新增节点必然有人忘记加。
    # 这里是「所有节点注册进图」的唯一入口，包在这里漏不掉。
    # 节点实现完全不知道 span 的存在，仍是纯粹的 state -> patch。
    #
    # 注意 retrieve_each 是被 Send fan-out 成 N 个并行实例的，
    # 每个实例各记一条 span —— 这正是我们要的：以前日志里
    # 4 条分支交织在一起，根本对不出每条各自花了多久。
    graph.add_node("rewrite", instrument_node("rewrite", rewrite_node))
    graph.add_node("retrieve_each", instrument_node("retrieve_each", retrieve_each_node))
    graph.add_node("dedup", instrument_node("dedup", dedup_node))
    graph.add_node("generate", instrument_node("generate", generate_node))
    graph.add_node(
        "validate_answer", instrument_node("validate_answer", validate_answer_node)
    )

    graph.add_edge(START, "rewrite")
    graph.add_conditional_edges("rewrite", _fanout_to_retrieve, ["retrieve_each"])
    graph.add_edge("retrieve_each", "dedup")
    graph.add_edge("dedup", "generate")
    graph.add_edge("generate", "validate_answer")
    graph.add_edge("validate_answer", END)

    return graph.compile()


rag_v2_graph = build_rag_v2_graph()
