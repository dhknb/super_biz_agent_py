"""知识检索工具 —— 链路一(Agent Tool Calling)的知识库入口。

**这个工具为什么需要熔断器和指标**

它和 rag_v2 的检索节点、AIOps 的 SOP 检索打的是**同一个 Milvus**。
那两处都接了 retrieval_breaker,只有这里裸奔。后果是 Milvus 挂掉时:

- 每一次 /api/chat 请求都要白等一次完整的连接超时,而答案是注定拿不到的;
- `retrieval_failures_total` 一动不动 —— 值班同学看指标会以为
  检索一切正常,因为另外两条链路的失败计数被这里的沉默稀释了;
- 更糟的是熔断器的**灵敏度**:失败次数是按下游依赖累计的,
  这条链路的失败不计入,同一个 Milvus 故障要靠另外两处攒够 N 次才熔断。

**措辞为什么要区分「没找到」和「检索不可用」**

返回值会被模型当作事实读进上下文。检索服务挂了却回一句
「没有找到相关信息」,模型会据此回答「知识库中没有相关内容」——
一个彻底的谎,而且用户和值班同学都无法从答案里看出真实原因。
这跟 app/core/errors.py 里反复强调的是同一条纪律:
**不许用假原因掩盖真原因**(参见 SopRetrievalService 的 FAILED / EMPTY)。
"""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.core.breakers import retrieval_breaker
from app.core.errors import error_code_of
from app.core.metrics import count_retrieval_failure, observe_tool_call
from app.services.vector_search_service import vector_search_service

# 工具名 —— 同时用作指标 label 与日志前缀。
# 抽成常量是为了让「指标里的名字」和「注册给模型的名字」不会各自漂移。
TOOL_NAME = "retrieve_knowledge"

# 检索服务不可用时给模型的话。
#
# 为什么明确写「请如实告知」:模型看到工具返回错误,默认行为常常是
# 自己编一个答案或者含糊过去。把处置方式写进返回值里,
# 是在这个「只能通过一个字符串跟模型通信」的接口上唯一能施加约束的地方。
_UNAVAILABLE_HINT = (
    "检索服务当前不可用（错误码: {code}），本次未能查询知识库。"
    "请如实告知用户知识库暂时无法访问，不要凭记忆作答。"
)

# 检索正常但确实没命中时给模型的话 —— 这是一个真实的业务结论。
_EMPTY_HINT = "知识库中没有与该问题相关的内容。"


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题

    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。

    Args:
        query: 用户的问题或查询

    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    logger.info(f"知识检索工具被调用: query='{query}'")

    try:
        # 嵌套顺序:熔断在外,计时在内。
        #
        # 反过来(计时在外)的话,熔断打开期间每次请求都会往耗时直方图里
        # 塞一个 ≈0 秒的样本 —— Milvus 挂得越久,P99 看起来越好,
        # 因为分母全是熔断器的快速拒绝。耗时曲线只应该包含**真正发出去的**
        # 那些调用(参见 metrics.py 里计时器必须在 guard 内侧的那条纪律)。
        #
        # guard() 放在调用点而不是检索服务内部:检索内部有自己的
        # 局部降级(混合检索失败退回纯向量检索),那种退回是**成功**,
        # 熔断器不该看见。理由同 SopRetrievalService。
        with retrieval_breaker.guard():
            with observe_tool_call(TOOL_NAME):
                docs = vector_search_service.retrieve_documents(
                    query, top_k=config.rag_top_k
                )
    except Exception as exc:
        # 这里**不抛**,返回一个说实话的字符串。
        #
        # 为什么工具层选择不抛异常:抛出去的话 LangChain 的 agent 循环
        # 会中断整轮对话,用户拿到 500。而工具失败其实是可降级的 ——
        # 模型还能用别的工具、还能如实告知用户。所以这里做的是
        # 「fail-open + 如实说明」,和 rag_v2 里质检器挂掉时的处置同一个取向。
        #
        # 熔断打开走的也是这条分支,一行不用改:CircuitOpenError 是
        # AppError 子类,error_code_of 给出 circuit_open。
        code = error_code_of(exc)
        # 与 rag_v2、SOP 检索共用同一个指标:三处打的是同一个 Milvus。
        count_retrieval_failure(code)
        logger.error(f"知识检索失败[{code}]，本次未使用知识库证据: {exc}")
        return _UNAVAILABLE_HINT.format(code=code), []

    if not docs:
        logger.info("知识检索完成，知识库中无相关内容")
        return _EMPTY_HINT, []

    logger.info(f"检索到 {len(docs)} 个相关文档")
    return format_docs(docs), docs


def format_docs(docs: List[Document]) -> str:
    """
    格式化文档列表为上下文文本

    Args:
        docs: 文档列表

    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []

    for i, doc in enumerate(docs, 1):
        # 提取元数据
        metadata = doc.metadata
        source = metadata.get("_file_name", "未知来源")

        # 提取标题信息 (如果有)
        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])

        header_str = " > ".join(headers) if headers else ""

        # 构建格式化文本
        formatted = f"【参考资料 {i}】"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"

        formatted_parts.append(formatted)

    return "\n".join(formatted_parts)
