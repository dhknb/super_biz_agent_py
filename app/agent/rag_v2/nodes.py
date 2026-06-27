"""RAG v2 各节点实现

rewrite       → 让 LLM 把原始问题改写为 N 个角度互补的子查询
retrieve_each → 单条子查询走一次混合检索 (fan-out 时每个分支一个实例)
dedup         → 按内容指纹去重并裁剪到 top_k
generate      → 把上下文喂给 LLM 生成最终答案
validate      → 校验证据覆盖和 groundedness，必要时降级
"""

import asyncio
import hashlib
import json
import re
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from app.agent.rag_v2.state import RAGState, RetrieveTask
from app.core.llm_factory import llm_factory
from app.services.vector_search_service import vector_search_service

NUM_SUB_QUERIES = 3
RETRIEVE_TOP_K = 4
FINAL_TOP_K = 6
REWRITE_TEMPERATURE = 0.3
MIN_GROUNDEDNESS_SCORE = 0.75
MIN_COVERAGE_SCORE = 0.60


REWRITE_SYSTEM_PROMPT = '''你是一个RAG查询改写助手。

任务: 把用户问题改写成 {n} 个角度互补的子查询,用于并行检索向量库。

要求:
1. 每个子查询聚焦一个角度(定义/原理/对比/使用场景/常见问题等)
2. 保留关键实体和术语,不要漫无边际地泛化
3. 用陈述句或关键词短语,不要带"请问"之类的填充词
4. 只输出严格的 JSON 数组,不要任何额外解释

输出格式示例:
["子查询1", "子查询2", "子查询3"]'''


GENERATE_SYSTEM_PROMPT = '''你是一个基于检索结果回答问题的助手。

要求:
1. 严格基于下方<context>中的内容回答,不要编造
2. 如果上下文不足以回答,直接说明"根据已有资料无法回答"
3. 回答简洁、结构化,可以用要点
4. 不要复述"根据上下文..."之类的开场白'''


VALIDATION_SYSTEM_PROMPT = '''你是一个RAG答案质检器，负责做证据校验与幻觉拦截。

你会拿到:
1. 用户问题
2. 候选答案
3. 检索到的证据片段

请只输出严格 JSON，对答案做如下评估:
{
  "coverage_score": 0.0,
  "groundedness_score": 0.0,
  "coverage_pass": true,
  "groundedness_pass": true,
  "needs_second_retrieval": false,
  "unsupported_claims": ["..."],
  "missing_aspects": ["..."],
  "reason": "一句话说明结论"
}

判定原则:
1. coverage_score: 问题关键点被证据覆盖的程度
2. groundedness_score: 答案中的陈述被证据直接支撑的程度
3. 如果答案包含证据中找不到依据的断言，unsupported_claims 要列出来
4. 如果问题的关键维度没有被当前证据覆盖，missing_aspects 要列出来
5. 当 coverage_score 明显不足时，needs_second_retrieval=true
6. 不要输出额外解释，不要使用 markdown'''


async def rewrite_node(state: RAGState) -> Dict[str, Any]:
    question = state["question"]
    logger.info(f"[rag_v2.rewrite] 原始问题: {question}")

    llm = llm_factory.create_chat_model(
        temperature=REWRITE_TEMPERATURE,
        streaming=False,
    )

    messages = [
        SystemMessage(content=REWRITE_SYSTEM_PROMPT.format(n=NUM_SUB_QUERIES)),
        HumanMessage(content=question),
    ]

    response = await llm.ainvoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)

    sub_queries = _parse_sub_queries(raw, fallback=question)
    if question not in sub_queries:
        sub_queries.insert(0, question)
    sub_queries = sub_queries[:NUM_SUB_QUERIES + 1]

    logger.info(f"[rag_v2.rewrite] 改写为 {len(sub_queries)} 个子查询: {sub_queries}")
    return {"sub_queries": sub_queries}


async def retrieve_each_node(task: RetrieveTask) -> Dict[str, Any]:
    query = task["query"]
    logger.info(f"[rag_v2.retrieve_each] 检索子查询: {query}")
#把同步线程丢到线程池里面，避免堵塞
    docs: List[Document] = await asyncio.to_thread(
        vector_search_service.retrieve_documents, query, RETRIEVE_TOP_K
    )

    for doc in docs:
        doc.metadata = {**(doc.metadata or {}), "_sub_query": query}

    logger.info(f"[rag_v2.retrieve_each] 子查询 '{query}' 召回 {len(docs)} 条")
    return {"documents": docs}


def dedup_node(state: RAGState) -> Dict[str, Any]:
    documents = state.get("documents", []) or []
    logger.info(f"[rag_v2.dedup] 入参文档数: {len(documents)}")

    seen: set[str] = set()
    unique_docs: List[Document] = []

    for doc in documents:
        key = doc.metadata.get("id") if doc.metadata else None
        if not key:
            key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)

    unique_docs = unique_docs[:FINAL_TOP_K]
    logger.info(f"[rag_v2.dedup] 去重后文档数: {len(unique_docs)}")
    return {"deduped_documents": unique_docs}


async def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    docs: List[Document] = state.get("deduped_documents", []) or []
    logger.info(f"[rag_v2.generate] 使用 {len(docs)} 条上下文生成答案")

    if not docs:
        return {"answer": "根据已有资料无法回答该问题。"}

    context = _format_context(docs)
    user_prompt = f"<context>\n{context}\n</context>\n\n问题: {question}"

    llm = llm_factory.create_chat_model(temperature=0.3, streaming=False)
    messages = [
        SystemMessage(content=GENERATE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    response = await llm.ainvoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)

    logger.info(f"[rag_v2.generate] 生成完成,长度 {len(answer)}")
    return {"answer": answer}


async def validate_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    answer = (state.get("answer") or "").strip()
    docs: List[Document] = state.get("deduped_documents", []) or []
    logger.info(f"[rag_v2.validate] 开始校验答案, context={len(docs)}")

    if not answer:
        validation = _default_validation(
            reason="答案为空，已视为未通过校验。",
            needs_second_retrieval=not docs,
        )
        return {"validation": validation}

    if not docs:
        validation = _default_validation(
            reason="没有可用证据片段，已降级为保守回答。",
            needs_second_retrieval=True,
        )
        return {"answer": "根据已有资料无法回答该问题。", "validation": validation}

    context = _format_context(docs)
    user_prompt = (
        f"<question>\n{question}\n</question>\n\n"
        f"<answer>\n{answer}\n</answer>\n\n"
        f"<evidence>\n{context}\n</evidence>"
    )

    llm = llm_factory.create_chat_model(temperature=0.0, streaming=False)
    messages = [
        SystemMessage(content=VALIDATION_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)
    validation = _parse_validation_result(raw)

    coverage = float(validation.get("coverage_score", 0.0))
    groundedness = float(validation.get("groundedness_score", 0.0))
    coverage_pass = bool(validation.get("coverage_pass", coverage >= MIN_COVERAGE_SCORE))
    groundedness_pass = bool(validation.get("groundedness_pass", groundedness >= MIN_GROUNDEDNESS_SCORE))
    blocked = (not coverage_pass) or (not groundedness_pass)

    validation.update(
        {
            "coverage_score": coverage,
            "groundedness_score": groundedness,
            "coverage_pass": coverage_pass,
            "groundedness_pass": groundedness_pass,
            "blocked": blocked,
            "thresholds": {
                "coverage_min": MIN_COVERAGE_SCORE,
                "groundedness_min": MIN_GROUNDEDNESS_SCORE,
            },
        }
    )

    if blocked:
        validation["reason"] = validation.get("reason") or "证据覆盖或事实支撑不足，已触发保守降级。"
        logger.warning(f"[rag_v2.validate] 未通过: coverage={coverage:.2f} groundedness={groundedness:.2f}")
        return {"answer": _build_fallback_answer(validation), "validation": validation}

    logger.info(f"[rag_v2.validate] 通过: coverage={coverage:.2f}, groundedness={groundedness:.2f}")
    return {"validation": validation}


def _parse_sub_queries(raw: str, fallback: str) -> List[str]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if fenced:
        text = fenced.group(1)
    else:
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            text = match.group(0)

    try:
        data = json.loads(text)
        if isinstance(data, list):
            cleaned = [str(x).strip() for x in data if str(x).strip()]
            if cleaned:
                return cleaned
    except json.JSONDecodeError:
        logger.warning(f"[rag_v2.rewrite] 解析子查询失败, raw={raw!r}")

    return [fallback]


def _parse_validation_result(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fenced:
        text = fenced.group(1)
    else:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {
                "coverage_score": _coerce_score(data.get("coverage_score")),
                "groundedness_score": _coerce_score(data.get("groundedness_score")),
                "coverage_pass": bool(data.get("coverage_pass", False)),
                "groundedness_pass": bool(data.get("groundedness_pass", False)),
                "needs_second_retrieval": bool(data.get("needs_second_retrieval", False)),
                "unsupported_claims": _coerce_str_list(data.get("unsupported_claims")),
                "missing_aspects": _coerce_str_list(data.get("missing_aspects")),
                "reason": str(data.get("reason", "")).strip(),
            }
    except json.JSONDecodeError:
        logger.warning(f"[rag_v2.validate] 解析校验结果失败, raw={raw!r}")

    return _default_validation(
        reason="质检器输出不可解析，已按未通过处理。",
        needs_second_retrieval=False,
    )


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _coerce_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _default_validation(reason: str, needs_second_retrieval: bool) -> Dict[str, Any]:
    return {
        "coverage_score": 0.0,
        "groundedness_score": 0.0,
        "coverage_pass": False,
        "groundedness_pass": False,
        "needs_second_retrieval": needs_second_retrieval,
        "unsupported_claims": [],
        "missing_aspects": [],
        "blocked": True,
        "reason": reason,
        "thresholds": {
            "coverage_min": MIN_COVERAGE_SCORE,
            "groundedness_min": MIN_GROUNDEDNESS_SCORE,
        },
    }


def _build_fallback_answer(validation: Dict[str, Any]) -> str:
    unsupported = validation.get("unsupported_claims", []) or []
    missing = validation.get("missing_aspects", []) or []

    parts = ["根据当前检索到的资料，暂时无法给出可充分证据支撑的回答。"]
    if missing:
        parts.append(f"证据覆盖不足的方面：{'；'.join(missing[:3])}。")
    if unsupported:
        parts.append(f"已拦截的未证实断言：{'；'.join(unsupported[:3])}。")
    return "".join(parts)


def _format_context(docs: List[Document]) -> str:
    parts = []
    for idx, doc in enumerate(docs, start=1):
        src = (doc.metadata or {}).get("_sub_query", "")
        parts.append(f"[{idx}] (源子查询: {src})\n{doc.page_content}")
    return "\n\n".join(parts)
