"""RAG v2 各节点实现

rewrite       → 让 LLM 把原始问题改写为 N 个角度互补的子查询
retrieve_each → 单条子查询走一次混合检索 (fan-out 时每个分支一个实例)
tool_facts    → 单轮工具选择,查实时事实 (与检索并行,不占串行预算)
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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from loguru import logger

from app.agent.mcp_tool_provider import mcp_tool_provider
from app.agent.rag_v2.state import RAGState, RetrieveTask
from app.config import config
from app.core.breakers import llm_breaker, retrieval_breaker
from app.core.errors import DegradeReason, error_code_of, wrap_llm_exception
from app.core.llm_factory import llm_factory
from app.core.metrics import count_retrieval_failure, observe_llm_call, observe_tool_call
from app.services.vector_search_service import vector_search_service
from app.tools.time_tool import get_current_time

NUM_SUB_QUERIES = 3
RETRIEVE_TOP_K = 4
FINAL_TOP_K = 6
REWRITE_TEMPERATURE = 0.3
MIN_GROUNDEDNESS_SCORE = 0.75
MIN_COVERAGE_SCORE = 0.60

# 单轮里最多并发执行几个工具调用。
#
# 为什么要有上限:工具调用参数是模型给的,它可以在一轮里请求 20 个调用。
# 每个都是一次 MCP 往返,并发打满会同时压垮 MCP 进程和我们的总预算。
# 取 4:够覆盖「查监控 + 查日志」这类组合,又不至于让单个节点
# 变成压测客户端。超出的部分直接丢弃并如实记录 —— 见 _run_tool_calls。
MAX_TOOL_CALLS_PER_TURN = 4

# 单个工具结果注入 prompt 的字符上限。
#
# 日志类工具能吐几万字符,原样拼进 prompt 会把知识库证据和会话记忆
# 一起挤出上下文窗口 —— 那等于用实时数据换掉了 SOP 证据。
# 截断而不是丢弃:一段日志的头部通常就包含关键错误行。
MAX_TOOL_CONTENT_CHARS = 1200


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
2. 如果上下文不足,不要使用客服式套话,而是先给可执行的排查方向,再说明证据缺口
3. 回答要像运维值班助手: 先结论/风险,再给最短命令链路或检查步骤
4. 回答简洁、结构化,可以用要点
5. 不要复述"根据上下文..."之类的开场白'''


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

MEMORY_POLICY = '''会话记忆只用于理解用户已明确说明的对象、任务延续和约束。
它不是知识库证据，更不是实时监控、日志或工具查询结果；不得将历史指标写成当前实时值。'''


TOOL_FACTS_SYSTEM_PROMPT = '''你是一个运维助手的工具调度器。

任务: 判断回答用户问题**是否需要**查询实时数据,需要就调用对应工具。

判定原则:
1. 只有问题涉及**当前状态**时才调用工具:实时指标、当前时间、最近日志、
   某个实例现在的水位。
2. 概念、原理、操作步骤、SOP 类问题**不要**调用工具 —— 那些由知识库负责。
3. 不确定就不调用。少查一次只是少一份事实,乱查一次会拖慢整个回答。
4. 不要为了「补充信息」而调用工具,只在缺了它就答不了的时候调用。

如果不需要任何工具,直接回复空内容,不要解释你为什么不调用。'''


TOOL_FACTS_POLICY = '''<tool_facts> 是工具查到的**实时事实**,与 <context> 的知识库证据性质不同:
它反映查询那一刻的真实状态,优先级高于知识库里可能过期的示例数值。
标记为「查询失败」的条目表示这项数据本次没拿到,不要猜测它的值。'''


async def rewrite_node(state: RAGState) -> Dict[str, Any]:
    question = state["question"]
    conversation_context = (state.get("conversation_context") or "").strip()
    logger.info(f"[rag_v2.rewrite] 原始问题: {question}")

    llm = llm_factory.create_chat_model(
        temperature=REWRITE_TEMPERATURE,
        streaming=False,
    )

    memory_block = (
        f"\n\n<conversation_memory>\n{conversation_context}\n</conversation_memory>"
        if conversation_context
        else ""
    )
    messages = [
        SystemMessage(content=REWRITE_SYSTEM_PROMPT.format(n=NUM_SUB_QUERIES)),
        HumanMessage(
            content=(
                f"{MEMORY_POLICY}{memory_block}\n\n"
                f"当前问题：{question}\n"
                "请利用记忆消解代词和省略的对象，但不要把历史信息改写为实时事实。"
            )
        ),
    ]

    try:
        # 这一步的 LLM 调用原本**没有兜底** —— 一炸整张图就崩,用户拿到 500。
        # 而它其实是全链路里最该降级的一步:改写的价值是「多几个检索角度」,
        # 不是「能不能检索」。拿原始问题当唯一子查询照样能召回证据,
        # 只是召回面窄一些 —— 这是一个真正意义上的「更差但可用」。
        #
        # 顺序上必须先有 try/except 再装熔断器,不能反:
        # 熔断打开时抛的 CircuitOpenError 如果没人接,就变成 500。
        # 熔断器的全部意义是**快速降级**,把降级路径炸成 500 属于本末倒置。
        #
        # 计时器在 guard() **内侧**,顺序不能反。
        # 熔断打开时 guard() 第一行就抛,ainvoke 根本没执行 —— 如果计时器
        # 在外侧,它照样会记下一个 ~0.0001 秒的样本。于是熔断期间几百个
        # 「没打出去的调用」全以 0 秒计入直方图,P99 反而**变好看**:
        # 下游挂了,耗时指标显示一切正常。那正是这四批要消灭的谎。
        # 「熔断了多少次」由 circuit_breaker_state 和
        # degrade_total{reason="circuit_open"} 负责,各司其职。
        with llm_breaker.guard():
            with observe_llm_call("rewrite"):
                response = await llm.ainvoke(messages)
    except Exception as exc:
        wrapped = wrap_llm_exception(exc)
        logger.error(
            f"[rag_v2.rewrite] 改写失败[{wrapped.code}],退回单查询: {wrapped.message}"
        )
        return {
            "sub_queries": [question],
            "degrade_reasons": [wrapped.degrade_reason.value],
        }

    raw = response.content if hasattr(response, "content") else str(response)

    sub_queries = _parse_sub_queries(raw, fallback=question)
    if question not in sub_queries:
        sub_queries.insert(0, question)
    sub_queries = sub_queries[:NUM_SUB_QUERIES + 1]

    logger.info(f"[rag_v2.rewrite] 改写为 {len(sub_queries)} 个子查询: {sub_queries}")
    return {"sub_queries": sub_queries}


async def retrieve_each_node(task: RetrieveTask) -> Dict[str, Any]:
    """单条子查询检索。**任何异常都不向上抛**，只记为一条失败明细。

    为什么必须在这里兜住异常：
    graph.py 用 `Send` 把 N 个子查询并行 fan-out 到本节点的 N 个实例。
    LangGraph 的语义是「任一并行分支抛异常 → 整个 super-step 失败 → 整张图崩」。
    也就是说改造前只要 Milvus 抖一下，哪怕另外 3 个分支已经召回了充足证据，
    用户拿到的仍然是一个 500 —— 已经做完的工作被整批丢掉。

    兜住之后语义变成：一个分支失败 = 证据少一份，而不是整个请求失败。
    失败明细进 `retrieve_failures`，由 dedup 节点汇总判定降级原因，
    这样「这次答案是在证据不完整的情况下生成的」这个事实不会丢。
    """
    query = task["query"]
    logger.info(f"[rag_v2.retrieve_each] 检索子查询: {query}")

    try:
        # 熔断器包在 to_thread **外面**,而不是塞进线程里。
        #
        # 包在外面,熔断打开时连线程池 worker 都不必占用 —— 这正是熔断器
        # 想省下的那份成本(Milvus 挂了,600 个请求各占一个 worker 白等 60 秒)。
        # 塞进线程里的话,拒绝发生在已经拿到 worker 之后,省下的只有网络往返,
        # 最紧张的那个资源反而没保住。
        #
        # 注意 guard() 用的是 threading.Lock,所以跨 to_thread 边界是安全的:
        # 临界区在当前协程里执行完(几个整数赋值),真正的 IO 才进线程池。
        with retrieval_breaker.guard():
            # 把同步检索丢到线程池里，避免阻塞事件循环
            docs: List[Document] = await asyncio.to_thread(
                vector_search_service.retrieve_documents, query, RETRIEVE_TOP_K
            )
    except Exception as exc:
        # 这里刻意捕获宽泛的 Exception：下游检索栈会抛什么由 pymilvus /
        # 网络层决定，穷举类型既不可能也没必要 —— 我们要的是「这个分支没成功」
        # 这一个事实，以及足够排障的原因文本。
        #
        # 不吞掉 CancelledError：它在 Python 3.8+ 继承自 BaseException，
        # 不在 Exception 分支里，总预算超时取消协程时能正常向上传播。
        code = error_code_of(exc)
        # 指标埋在这里而不是 instrumentation.py 里:那一层看到的是
        # 「本节点降级了」这个笼统事实,拿不到错误码 —— 而 code 正是
        # 区分「Milvus 连不上」和「熔断器打开」的那个 label。
        count_retrieval_failure(code)
        logger.warning(f"[rag_v2.retrieve_each] 子查询 '{query}' 检索失败[{code}]: {exc}")
        return {
            "documents": [],
            "retrieve_failures": [{"query": query, "error": str(exc), "code": code}],
        }

    for doc in docs:
        doc.metadata = {**(doc.metadata or {}), "_sub_query": query}

    logger.info(f"[rag_v2.retrieve_each] 子查询 '{query}' 召回 {len(docs)} 条")
    return {"documents": docs}


async def tool_facts_node(state: RAGState) -> Dict[str, Any]:
    """单轮工具选择 + 执行,拿实时事实。**任何异常都不向上抛**。

    ## 为什么是「单轮」而不是 agent 循环

    链路一(/api/chat)用的是 create_agent,模型可以反复「调工具 → 看结果 →
    再调工具」直到自己认为够了。那种循环在这里**不能用**:本节点跑在与检索
    并行的分支上,共享同一个总预算(chat_total_budget_seconds)。循环轮数
    由模型决定,也就是说预算什么时候被吃光由模型决定 —— 而超出预算会取消
    **所有**分支,包括那条已经召回了完整证据的检索链。
    一轮工具调用换来的是可预测的耗时上界,这是并行分支必须付的代价。

    ## 为什么失败不抛异常

    与 retrieve_each_node 同一条纪律:这是并行分支,抛异常会让整个
    super-step 失败,把检索侧已经做完的工作一起丢掉。工具事实是**增强**,
    不是回答的前提 —— 拿不到就少一份事实,不该让整个请求失败。

    ## 为什么工具选择的 LLM 调用失败也只记降级、不退回「全都调一遍」

    退回全调等于在模型判断不可用时反而加大下游压力,方向正好是反的。
    这一步的价值是「判断要不要查」,判断不了就当作不需要查 ——
    知识库证据仍然在,答案照样能生成。
    """
    question = state["question"]

    # 运行时开关。关掉之后如实记 FEATURE_DISABLED,不假装「没有实时数据」。
    if not config.enable_tool_facts:
        logger.warning("[rag_v2.tool_facts] 工具事实已被开关关闭(enable_tool_facts=false)")
        return {
            "tool_facts": [],
            "degrade_reasons": [DegradeReason.FEATURE_DISABLED.value],
        }

    # provider.get_tools() 承诺不抛异常,MCP 全挂时返回空列表。
    mcp_tools = await mcp_tool_provider.get_tools()
    # 刻意**不含** retrieve_knowledge:知识库检索已经在并行的 retrieve_each
    # 分支里按子查询跑了 N 次,这里再放一个入口只会让模型重复检索一遍,
    # 拿到的还是同一批文档,白花一次 Milvus 往返和一份上下文预算。
    tools: List[BaseTool] = [get_current_time, *mcp_tools]

    logger.info(f"[rag_v2.tool_facts] 可用工具 {len(tools)} 个,判断是否需要实时数据")

    llm = llm_factory.create_chat_model(temperature=0.0, streaming=False)
    messages = [
        SystemMessage(content=TOOL_FACTS_SYSTEM_PROMPT),
        HumanMessage(content=f"用户问题：{question}"),
    ]

    try:
        with llm_breaker.guard():
            with observe_llm_call("tool_facts"):
                response = await llm.bind_tools(tools).ainvoke(messages)
    except Exception as exc:
        wrapped = wrap_llm_exception(exc)
        logger.error(
            f"[rag_v2.tool_facts] 工具选择失败[{wrapped.code}],本次不查实时数据: "
            f"{wrapped.message}"
        )
        return {
            "tool_facts": [],
            "degrade_reasons": [wrapped.degrade_reason.value],
        }

    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        logger.info("[rag_v2.tool_facts] 模型判断无需实时数据")
        return {"tool_facts": []}

    facts = await _run_tool_calls(tool_calls, tools)

    failed = [fact for fact in facts if not fact["ok"]]
    result: Dict[str, Any] = {"tool_facts": facts}
    if failed:
        # 工具失败也要记降级原因:答案是在少了这几份实时事实的情况下生成的。
        # 用每个工具自己的 code 去分类,而不是笼统记一个 tool_failed ——
        # MCP 进程没起来(mcp_unavailable)和工具报错(tool_failed)
        # 归属不同的人处理。
        reasons = sorted({_degrade_reason_for(fact["code"]) for fact in failed})
        result["degrade_reasons"] = reasons
        logger.warning(
            f"[rag_v2.tool_facts] {len(failed)}/{len(facts)} 个工具调用失败,"
            f"降级原因={reasons}"
        )

    logger.info(f"[rag_v2.tool_facts] 完成,{len(facts) - len(failed)}/{len(facts)} 个工具成功")
    return result


async def _run_tool_calls(
    tool_calls: List[Dict[str, Any]], tools: List[BaseTool]
) -> List[Dict[str, Any]]:
    """并发执行模型请求的工具调用,返回事实列表。**不抛异常**。

    并发而不是串行:两个工具调用之间没有依赖(单轮里模型是一次性给出全部
    调用的),串行执行会让耗时变成两次 MCP 往返之和 —— 而本节点的耗时
    直接计入总预算。

    超出 MAX_TOOL_CALLS_PER_TURN 的调用被丢弃并记成失败条目,而不是静默截断:
    模型请求了 6 个工具却只有 4 个结果,它会以为剩下两个「没查到数据」,
    进而在答案里说「未发现异常」—— 那是用假原因掩盖真原因。
    """
    by_name = {tool.name: tool for tool in tools}

    accepted = tool_calls[:MAX_TOOL_CALLS_PER_TURN]
    dropped = tool_calls[MAX_TOOL_CALLS_PER_TURN:]

    facts = list(
        await asyncio.gather(*(_run_one_tool_call(call, by_name) for call in accepted))
    )

    for call in dropped:
        name = call.get("name", "unknown")
        logger.warning(f"[rag_v2.tool_facts] 超出单轮上限,丢弃工具调用: {name}")
        facts.append(
            {
                "tool": name,
                "args": call.get("args", {}) or {},
                "content": (
                    f"未执行：单轮工具调用数超过上限 {MAX_TOOL_CALLS_PER_TURN}。"
                ),
                "ok": False,
                "code": "tool_calls_truncated",
            }
        )

    return facts


async def _run_one_tool_call(
    call: Dict[str, Any], by_name: Dict[str, BaseTool]
) -> Dict[str, Any]:
    """执行单个工具调用。**不抛异常**,失败也返回一条事实。

    为什么用 ainvoke 而不是 invoke:MCP 工具是异步的,而本地
    get_current_time 是同步的 —— BaseTool.ainvoke 对同步工具会自动
    走线程池,一个入口覆盖两种形态,不必在这里 isinstance 分流。
    """
    name = call.get("name", "")
    args = call.get("args", {}) or {}

    tool = by_name.get(name)
    if tool is None:
        # 模型幻觉出一个不存在的工具名。如实记下来 ——
        # 静默忽略会让「工具表里没有这个能力」这个事实消失,
        # 而它恰好说明 prompt 或工具描述需要调整。
        logger.warning(f"[rag_v2.tool_facts] 模型请求了不存在的工具: {name}")
        return {
            "tool": name,
            "args": args,
            "content": f"未执行：工具 {name} 不存在。",
            "ok": False,
            "code": "tool_not_found",
        }

    try:
        # observe_tool_call 同时负责计时和失败计数(记账点唯一)。
        # 这里**没有**熔断器:MCP 工具的熔断在 mcp_client 的拦截器里,
        # 按服务各一个;本地工具各自在实现里带(见 knowledge_tool)。
        # 在这里再套一层会让同一次失败被两个熔断器各记一笔,
        # 灵敏度凭空翻倍。
        with observe_tool_call(name):
            content = await tool.ainvoke(args)
    except asyncio.CancelledError:
        # 总预算耗尽时会成批触发,必须原样上抛,不能变成一条「失败事实」——
        # 那会让被取消的请求看起来像是「工具报错了」。
        raise
    except Exception as exc:
        code = error_code_of(exc)
        logger.warning(f"[rag_v2.tool_facts] 工具 {name} 执行失败[{code}]: {exc}")
        return {
            "tool": name,
            "args": args,
            "content": f"查询失败（{code}）：{exc}",
            "ok": False,
            "code": code,
        }

    text = content if isinstance(content, str) else str(content)
    if len(text) > MAX_TOOL_CONTENT_CHARS:
        text = text[:MAX_TOOL_CONTENT_CHARS] + "…（内容过长已截断）"

    logger.info(f"[rag_v2.tool_facts] 工具 {name} 成功,返回 {len(text)} 字符")
    return {"tool": name, "args": args, "content": text, "ok": True, "code": ""}


def _degrade_reason_for(code: str) -> str:
    """把工具错误码映射到降级原因。

    只有 MCP 进程级失败单独归类,其余一律 tool_failed —— 划分标准仍然是
    **修复方向**:mcp_unavailable 去起进程,tool_failed 去看工具实现。
    """
    if code == DegradeReason.MCP_UNAVAILABLE.value:
        return DegradeReason.MCP_UNAVAILABLE.value
    return DegradeReason.TOOL_FAILED.value


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

    # 在这里判定检索侧的降级原因，而不是在 retrieve_each 里。
    # 原因是单个分支看不到全局：它只知道「我失败了」，判断不了
    # 「是全都失败了，还是只有我失败」—— 而这两者的处置方向完全不同。
    # dedup 是 fan-in 之后的第一个节点，是唯一能看到全部分支结果的位置。
    failures = state.get("retrieve_failures", []) or []
    degrade_reasons: List[str] = []
    if failures:
        if unique_docs:
            # 部分分支失败但仍有证据：结果可用，只是证据不完整。
            degrade_reasons.append(DegradeReason.PARTIAL_RETRIEVAL.value)
            logger.warning(
                f"[rag_v2.dedup] 部分检索失败({len(failures)} 条)，"
                f"在 {len(unique_docs)} 条证据上继续生成"
            )
        else:
            # 全部失败，一条证据都没有：这是基础设施问题，不是知识库缺文档。
            # 必须与 RETRIEVAL_EMPTY 区分开 —— 前者要修 Milvus，后者要补文档。
            degrade_reasons.append(DegradeReason.RETRIEVAL_FAILED.value)
            logger.error(f"[rag_v2.dedup] 全部检索分支失败({len(failures)} 条)，无可用证据")
    elif not unique_docs:
        # 没有失败却也没有结果：知识库里确实没有相关内容。
        degrade_reasons.append(DegradeReason.RETRIEVAL_EMPTY.value)
        logger.warning("[rag_v2.dedup] 检索正常但未命中任何文档")

    result: Dict[str, Any] = {"deduped_documents": unique_docs}
    if degrade_reasons:
        result["degrade_reasons"] = degrade_reasons
    return result


async def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    docs: List[Document] = state.get("deduped_documents", []) or []
    failures = state.get("retrieve_failures", []) or []
    tool_facts: List[Dict[str, Any]] = state.get("tool_facts", []) or []
    conversation_context = (state.get("conversation_context") or "").strip()
    # 成功的工具事实才算证据。失败条目(ok=False)的 content 是
    # 「查询失败（xxx）」这类说明文字,它能帮模型如实告知用户,
    # 但不能拿来充当「我们有证据」—— 否则一个 MCP 全挂的请求会
    # 因为「有 3 条 tool_facts」而绕过下面的无证据兜底。
    usable_facts = [fact for fact in tool_facts if fact.get("ok")]
    logger.info(
        f"[rag_v2.generate] 使用 {len(docs)} 条上下文 + "
        f"{len(usable_facts)}/{len(tool_facts)} 条工具事实生成答案"
    )

    # 判据是「一份可用证据都没有」,而不是原来的「没有知识库文档」。
    #
    # 为什么必须放宽:问「现在 CPU 多少」时知识库**本来就该**是空的 ——
    # 那不是故障,是问题类型决定的。按原来的条件走,工具已经查到的水位
    # 会被整个丢掉,换成一句「当前证据还不够支持直接下结论」,
    # 而正确答案就在手里。这是新加的并行分支必须同时修掉的一处。
    if not docs and not usable_facts:
        # 无证据的原因有两种，措辞必须区分开 ——
        # 「知识库里没有」让人去补文档，「检索服务挂了」让人去修服务，
        # 两者的处置方向完全相反。统一说成「没有检索到」就是在误导排障方向。
        if failures:
            reason = (
                f"检索服务不可用（{len(failures)} 个子查询全部失败），"
                "本次未能使用知识库证据。这不代表知识库中没有相关文档。"
            )
        else:
            reason = "知识库中没有检索到与该问题相关的证据片段。"
        # 工具也全军覆没时补一句,否则「检索失败」会掩盖掉
        # 「实时数据同样没拿到」—— 值班同学只会去查 Milvus。
        if tool_facts:
            reason += f" 另有 {len(tool_facts)} 个实时数据查询也未成功。"
        validation = _default_validation(reason=reason, needs_second_retrieval=True)
        return {"answer": _build_fallback_answer(validation), "validation": validation}

    context = _format_context(docs)
    memory_block = (
        f"<conversation_memory>\n{conversation_context}\n</conversation_memory>\n\n"
        if conversation_context
        else ""
    )
    # 工具事实单独成块,不拼进 <context>。
    # 混进去的话模型无从区分「SOP 文档说阈值是 80%」和「现在实测 92%」,
    # 而这两句话在排障里的角色完全不同 —— 一个是标准,一个是现状。
    facts_block = _format_tool_facts(tool_facts)
    # 证据不完整时必须在 prompt 里说明，否则模型会把「手上这几条」
    # 当成全部事实，进而给出过度自信的结论。
    partial_note = (
        f"\n\n注意：本次检索有 {len(failures)} 个子查询失败，下方证据不完整。"
        "请在信息不足处明确说明，不要基于残缺证据下确定结论。"
        if failures
        else ""
    )
    # 知识库为空但有工具事实:必须点明,否则模型会照着
    # GENERATE_SYSTEM_PROMPT 里「严格基于 <context> 回答」的要求,
    # 在 context 空着的时候拒绝作答 —— 明明实时数据就在下面。
    if not docs:
        partial_note += (
            "\n\n注意：本次没有知识库证据，请基于下方实时数据回答，"
            "并说明这是实时查询结果而非文档结论。"
        )
    user_prompt = (
        f"{MEMORY_POLICY}{partial_note}\n\n{memory_block}"
        f"<context>\n{context}\n</context>\n\n{facts_block}问题: {question}"
    )

    llm = llm_factory.create_chat_model(temperature=0.3, streaming=False)
    messages = [
        SystemMessage(content=GENERATE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    try:
        with llm_breaker.guard():
            with observe_llm_call("generate"):
                response = await llm.ainvoke(messages)
    except Exception as exc:
        # LLM 挂掉时给出可用的降级答案，而不是让整张图抛异常。
        # 降级原因写清是 llm_timeout 还是 llm_error：前者要调超时/换模型，
        # 后者要查配额与鉴权 —— 混成一个 "生成失败" 就没法下钻。
        wrapped = wrap_llm_exception(exc)
        logger.error(f"[rag_v2.generate] 生成失败[{wrapped.code}]: {wrapped.message}")
        validation = _default_validation(
            reason=f"答案生成失败（{wrapped.code}），已降级为排查建议。",
            needs_second_retrieval=False,
        )
        return {
            "answer": _build_fallback_answer(validation),
            "validation": validation,
            "degrade_reasons": [wrapped.degrade_reason.value],
        }
    answer = response.content if hasattr(response, "content") else str(response)

    logger.info(f"[rag_v2.generate] 生成完成,长度 {len(answer)}")
    return {"answer": answer}


async def validate_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    answer = (state.get("answer") or "").strip()
    docs: List[Document] = state.get("deduped_documents", []) or []
    facts: List[Dict[str, Any]] = state.get("tool_facts", []) or []
    conversation_context = (state.get("conversation_context") or "").strip()
    logger.info(
        f"[rag_v2.validate] 开始校验答案, context={len(docs)}, tool_facts={len(facts)}"
    )

    if not answer:
        validation = _default_validation(
            reason="答案为空，已视为未通过校验。",
            needs_second_retrieval=not docs and not facts,
        )
        return {"validation": validation}

    # 判据是「有没有任何证据」,不是「有没有知识库文档」。
    # 只看 docs 会把「知识库为空但工具查到了实时数据」误判成无证据,
    # 于是一个正确回答了「现在 CPU 多少」的答案被换成保守回答 ——
    # 与 generate_node 里同一处放宽,理由见那边的注释。
    if not docs and not facts:
        validation = _default_validation(
            reason="没有可用证据片段，已降级为保守回答。",
            needs_second_retrieval=True,
        )
        return {"answer": _build_fallback_answer(validation), "validation": validation}

    # 运行时开关：关掉质检走 fail-open,和「质检器挂了」同一条路径。
    #
    # 为什么不用 _default_validation(blocked=True):那会把每个答案都换成
    # 「当前证据还不够支持直接下结论」—— 用户以为是自己的问题缺证据,
    # 实际是我们把质检关了。关掉一道**校验**不该让**答案**消失。
    #
    # 降级原因用 FEATURE_DISABLED 而不是 LLM_ERROR:后者会把值班同学
    # 送去查模型配额和鉴权,而那边一切正常,排障就此卡住。
    #
    # 放在这里而不是函数入口:上面两个 guard(答案为空、没有证据)是不花钱的
    # 结构性检查,不依赖被关掉的那个 LLM 调用,该继续生效。
    if not config.enable_answer_validation:
        logger.warning(
            "[rag_v2.validate] 证据校验已被开关关闭(enable_answer_validation=false)"
        )
        validation = _unvalidated_validation(
            "证据校验已被运行时开关关闭，本次答案未经过证据校验。"
        )
        return {
            "validation": validation,
            "degrade_reasons": [DegradeReason.FEATURE_DISABLED.value],
        }

    context = _format_context(docs)
    memory_block = (
        f"\n\n<conversation_memory>\n{conversation_context}\n</conversation_memory>"
        if conversation_context
        else ""
    )
    # 工具事实必须进 <evidence>,而且必须和知识库证据**分块**。
    #
    # 为什么必须进:质检器看不到工具结果的话,答案里「当前 CPU 使用率 91%」
    # 这类陈述在证据中找不到出处,会被列进 unsupported_claims,
    # groundedness 判低 → 答案被换成保守回答。也就是说工具查对了,
    # 反而因为质检器没看见而被拦掉 —— 这是接入工具后最容易踩的一脚。
    #
    # 为什么必须分块:coverage_score 评的是「问题的关键点被证据覆盖多少」。
    # 混在一起时,质检器无法分辨哪部分是可复核的静态文档、哪部分是
    # 一次性的实时快照,两种证据的可信期限完全不同。分块之后
    # 下面那行提示才能对它们提出不同的要求。
    tool_block = _format_tool_facts(facts)
    tool_section = f"\n\n<tool_facts>\n{tool_block}\n</tool_facts>" if tool_block else ""
    user_prompt = (
        f"<question>\n{question}\n</question>\n\n"
        f"<answer>\n{answer}\n</answer>\n\n"
        f"<evidence>\n{context}\n</evidence>{tool_section}{memory_block}\n\n"
        f"{MEMORY_POLICY}\n"
        "仅将 <evidence> 与 <tool_facts> 中的内容作为可验证证据；"
        "会话记忆只能帮助判断对象指代。\n"
        "<tool_facts> 是本次实时查询的结果，与 <evidence> 同等可信："
        "答案中的实时数值只要能在其中找到出处，就算有据可依，不要计入 unsupported_claims。"
        "标注为「查询失败」的条目不构成证据。"
    )

    llm = llm_factory.create_chat_model(temperature=0.0, streaming=False)
    messages = [
        SystemMessage(content=VALIDATION_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        # 熔断器守在这里。质检的 fail-open 分支恰好是熔断器最理想的落点:
        # 熔断打开时抛的 CircuitOpenError 被下面同一个 except 接住,
        # 答案照发、原因如实记成 circuit_open —— 一条分支都不用加。
        with llm_breaker.guard():
            with observe_llm_call("validate"):
                response = await llm.ainvoke(messages)
    except Exception as exc:
        # 质检器挂掉时**放行答案并标注未校验**，而不是拦成「证据不足」。
        #
        # 这里的取舍值得说清楚：
        # 拦掉（fail-closed）看似更安全，实际后果是质检模型一抖动，
        # 每个请求都会收到「当前证据还不够支持直接下结论」——
        # 而证据明明是好的，挂的是质检器。这跟 SOP 检索失败却说
        # 「未检索到相关 SOP」是同一类错误：**用一个假原因掩盖真原因**，
        # 把排障方向引向补文档/补证据，而真正该修的是模型调用。
        #
        # 放行（fail-open）的前提是它是第二道防线：答案本身已经是在
        # 检索证据 + 「严格基于上下文、不要编造」的约束下生成的。
        # 所以这里保留答案，但把 validated=False 与降级原因如实记下来，
        # 让 trace / 指标能统计「有多少答案是没过质检就发出去的」。
        wrapped = wrap_llm_exception(exc)
        logger.error(f"[rag_v2.validate] 质检失败[{wrapped.code}]: {wrapped.message}")
        validation = _unvalidated_validation(
            f"质检服务不可用（{wrapped.code}），本次答案未经过证据校验。"
        )
        return {
            "validation": validation,
            "degrade_reasons": [wrapped.degrade_reason.value],
        }

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
        # 质检拦截也是一种降级，必须记原因。
        # 它与 llm_error / retrieval_failed 的修复方向完全不同：
        # 这条说明「模型和检索都正常，是知识库覆盖不够」——处置是补文档。
        # 只看 blocked=True 这个布尔值分不出这一点。
        return {
            "answer": _build_fallback_answer(validation),
            "validation": validation,
            "degrade_reasons": [DegradeReason.EVIDENCE_INSUFFICIENT.value],
        }

    logger.info(f"[rag_v2.validate] 通过: coverage={coverage:.2f}, groundedness={groundedness:.2f}")
    # 显式写 validated=True，与质检失败时的 validated=False 成对出现。
    # 读侧就能区分「过了质检」和「没质检」，而不用靠字段缺失来猜。
    validation["validated"] = True
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


def _unvalidated_validation(reason: str) -> Dict[str, Any]:
    """构造「这次没做质检」的 validation —— fail-open:答案照发,但如实声明未校验。

    与 _default_validation 只差一个字段,语义却完全相反:
    - _default_validation  → blocked=True,答案被换成保守回答(fail-closed)
    - 本函数               → blocked=False,答案原样发出(fail-open)

    用哪个,取决于「没通过」的原因是**证据不足**还是**质检器不可用**:
    证据不足是一个真实的业务结论,该拦;质检器不可用是我方故障,
    拿它去拦用户的答案,等于用一个假原因(证据不够)掩盖真原因(质检挂了)。

    抽成函数是因为有两处需要它(质检器异常、开关关闭),而这两处必须给出
    **完全一致**的形状 —— 读侧靠 validated=False 统计「有多少答案没过质检
    就发出去了」,两处字段对不齐就会漏统计(DRY)。
    """
    return {
        "coverage_score": 0.0,
        "groundedness_score": 0.0,
        "coverage_pass": False,
        "groundedness_pass": False,
        "needs_second_retrieval": False,
        "unsupported_claims": [],
        "missing_aspects": [],
        # 关键：blocked=False —— 答案照发；validated=False —— 但声明没校验过。
        "blocked": False,
        "validated": False,
        "reason": reason,
        "thresholds": {
            "coverage_min": MIN_COVERAGE_SCORE,
            "groundedness_min": MIN_GROUNDEDNESS_SCORE,
        },
    }


def _build_fallback_answer(validation: Dict[str, Any]) -> str:
    unsupported = validation.get("unsupported_claims", []) or []
    missing = validation.get("missing_aspects", []) or []
    reason = str(validation.get("reason") or "").strip()

    parts = [
        "当前证据还不够支持直接下结论，建议先按下面方向继续排查：",
        "\n1. 先确认告警对象、时间窗口和影响范围是否一致。",
        "\n2. 再查看最近变更、服务日志、错误率、延迟和资源水位。",
        "\n3. 如果是组件故障，优先补充对应实例状态、连接数、慢请求和重启记录。",
    ]
    if reason:
        parts.append("\n\n证据缺口：" + reason)
    if missing:
        parts.append("\n仍缺少：" + "；".join(missing[:3]) + "。")
    if unsupported:
        parts.append("\n已拦截未证实判断：" + "；".join(unsupported[:3]) + "。")
    parts.append("\n补齐这些信息后，再继续判断根因和处置步骤。")
    return "".join(parts)


def _format_context(docs: List[Document]) -> str:
    """在固定预算内组装检索文档，避免知识库内容挤占会话记忆和问题空间。"""
    parts: List[str] = []
    used_tokens = 0
    budget = config.rag_document_context_token_budget
    for idx, doc in enumerate(docs, start=1):
        src = (doc.metadata or {}).get("_sub_query", "")
        prefix = f"[{idx}] (源子查询: {src})\n"
        available = budget - used_tokens - _estimate_tokens(prefix)
        if available <= 0:
            break
        content = _truncate_to_token_budget(doc.page_content, available)
        item = f"{prefix}{content}"
        parts.append(item)
        used_tokens += _estimate_tokens(item)
    return "\n\n".join(parts)


def _format_tool_facts(facts: List[Dict[str, Any]]) -> str:
    """把工具事实组装成 prompt 块。空列表返回空串（调用方据此决定是否注入）。

    失败的条目**也要进去**，而且要标明「查询失败」。
    换句话说，不是过滤掉 ok=False，而是让模型看见「这个数据没查到」——
    过滤掉的后果是模型以为自己拿到了全部实时数据，
    进而在答案里写「监控未发现异常」，而真相是监控压根没查到。
    那是这一整轮改造反复要消灭的同一种谎。

    单独一个函数而不是复用 _format_context：两者的输入类型不同
    （Document vs dict），预算也该分开算 —— 工具事实已经在
    _run_one_tool_call 里按 MAX_TOOL_CONTENT_CHARS 截过，
    这里只负责排版，不再二次截断。
    """
    if not facts:
        return ""

    parts: List[str] = []
    for idx, fact in enumerate(facts, start=1):
        status = "成功" if fact.get("ok") else "查询失败"
        args = fact.get("args") or {}
        args_text = json.dumps(args, ensure_ascii=False) if args else "{}"
        parts.append(
            f"[{idx}] 工具: {fact.get('tool', '')} | 参数: {args_text} | 状态: {status}\n"
            f"{fact.get('content', '')}"
        )
    return "\n\n".join(parts)


def _estimate_tokens(text: str) -> int:
    """中文按字符、其他内容按约四字符一 Token 的保守估算。"""
    cjk = sum("\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff" for char in text)
    return cjk + max(0, (len(text) - cjk + 3) // 4)


def _truncate_to_token_budget(text: str, budget: int) -> str:
    if _estimate_tokens(text) <= budget:
        return text
    if budget <= 0:
        return ""
    result: List[str] = []
    used = 0.0
    for char in text:
        cost = 1.0 if ("\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff") else 0.25
        if used + cost > budget:
            break
        result.append(char)
        used += cost
    return "".join(result).rstrip() + "…"
