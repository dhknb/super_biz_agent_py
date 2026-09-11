import asyncio
import json
from unittest.mock import AsyncMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.agent.rag_v2.nodes import (
    _build_fallback_answer,
    _format_context,
    generate_node,
    rewrite_node,
    validate_answer_node,
)


def test_build_fallback_answer_uses_ops_triage_tone() -> None:
    answer = _build_fallback_answer(
        {
            "reason": "缺少实例日志和资源水位。",
            "missing_aspects": ["实例状态", "最近变更"],
            "unsupported_claims": ["Redis 已宕机"],
        }
    )

    assert "继续排查" in answer
    assert "时间窗口" in answer
    assert "证据缺口" in answer
    assert "Redis 已宕机" in answer
    assert "根据已有资料无法回答" not in answer


def test_rewrite_injects_memory_to_resolve_references() -> None:
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content='["order-api CPU 排查"]'))
    with patch("app.agent.rag_v2.nodes.llm_factory.create_chat_model", return_value=llm):
        result = asyncio.run(
            rewrite_node(
                {
                    "question": "它为什么 CPU 高？",
                    "conversation_context": "【明确事实】\n- 服务是 order-api",
                }
            )
        )

    prompt = llm.ainvoke.await_args.args[0][1].content
    assert "服务是 order-api" in prompt
    assert "不要把历史信息改写为实时事实" in prompt
    assert "它为什么 CPU 高？" in result["sub_queries"]


def test_generate_and_validate_mark_memory_as_non_evidence() -> None:
    doc = Document(page_content="CPU 高时检查进程。", metadata={"_sub_query": "CPU 排查"})
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content="请检查 order-api 的进程 CPU。"),
            AIMessage(
                content=json.dumps(
                    {
                        "coverage_score": 0.9,
                        "groundedness_score": 0.9,
                        "coverage_pass": True,
                        "groundedness_pass": True,
                        "needs_second_retrieval": False,
                        "unsupported_claims": [],
                        "missing_aspects": [],
                        "reason": "证据充分",
                    }
                )
            ),
        ]
    )
    state = {
        "question": "它为什么 CPU 高？",
        "conversation_context": "【明确事实】\n- 服务是 order-api，昨天 CPU 90%",
        "deduped_documents": [doc],
    }
    with patch("app.agent.rag_v2.nodes.llm_factory.create_chat_model", return_value=llm):
        generated = asyncio.run(generate_node(state))
        validated = asyncio.run(validate_answer_node({**state, **generated}))

    generate_prompt = llm.ainvoke.await_args_list[0].args[0][1].content
    validate_prompt = llm.ainvoke.await_args_list[1].args[0][1].content
    assert "昨天 CPU 90%" in generate_prompt
    assert "不得将历史指标写成当前实时值" in generate_prompt
    assert "仅将 <evidence> 中的内容作为知识库可验证证据" in validate_prompt
    assert validated["validation"]["blocked"] is False


def test_retrieval_context_is_limited_by_token_budget(monkeypatch) -> None:
    monkeypatch.setattr("app.agent.rag_v2.nodes.config.rag_document_context_token_budget", 20)
    docs = [
        Document(page_content="中文内容" * 30, metadata={"_sub_query": "q1"}),
        Document(page_content="第二篇文档" * 30, metadata={"_sub_query": "q2"}),
    ]

    context = _format_context(docs)

    assert "[1]" in context
    assert len(context) < len(docs[0].page_content) + len(docs[1].page_content)
