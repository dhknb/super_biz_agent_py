"""RAG v2 图状态定义"""

import operator
from typing import Annotated, Any, Dict, List

from langchain_core.documents import Document
from typing_extensions import TypedDict


class RAGState(TypedDict, total=False):
    """RAG v2 流程的共享状态"""

    question: str
    # 有界会话记忆：滚动摘要 + 最近原文，由 API 在进入图前构造。
    conversation_context: str
    sub_queries: List[str]
    documents: Annotated[List[Document], operator.add]
    # 检索失败明细，与 documents 用同一套 fan-in 语义（operator.add 合并各分支）。
    #
    # 为什么失败必须进 state 而不是只打日志：
    # graph.py 用 Send 把 N 个子查询并行 fan-out，任一分支抛异常整张图就崩 ——
    # 即使另外几个分支已经召回了足够证据。把失败收进 state，
    # 就把「一个分支炸掉 = 整个请求失败」变成「一个分支炸掉 = 证据少一份」，
    # 同时下游还能知道「这次结果是不完整的」，从而标出 partial_retrieval 降级。
    #
    # 每个元素形如 {"query": str, "error": str, "code": str}。
    retrieve_failures: Annotated[List[Dict[str, Any]], operator.add]
    deduped_documents: List[Document]
    # 工具查到的实时事实,每个元素形如
    # {"tool": str, "args": dict, "content": str, "ok": bool, "code": str}。
    #
    # 为什么**不能**并进 documents:
    # documents 是知识库检索证据(静态文档),工具结果是实时事实
    # (当前 CPU 水位、刚才那段日志)。混成一个之后,
    # validate_answer 会拿实时数据去评「知识库覆盖度」——
    # 一个正确回答了「现在 CPU 多少」的答案会因为知识库里没有
    # 这个数字而被判 coverage 不足,然后被换成保守回答。
    # 那是用假原因(证据不够)掩盖真原因(问题本来就要问实时数据)。
    #
    # 不用 operator.add:工具节点只有一个实例(单轮工具选择),
    # 不存在 fan-in 合并。加了反而会掩盖「同一个节点被跑了两次」这种 bug。
    tool_facts: List[Dict[str, Any]]
    answer: str
    validation: Dict[str, Any]
    # 本次运行的降级原因（DegradeReason 的字符串值），无降级时不写入。
    # 用列表而非单值：一次运行可能同时命中多个原因
    # （例如部分检索失败 + 证据不足），压成一个会丢信息。
    degrade_reasons: Annotated[List[str], operator.add]


class RetrieveTask(TypedDict):
    """fan-out 时传入单个 retrieve_each 节点的载荷"""

    query: str
