"""RAG v2 图状态定义"""

import operator
from typing import Annotated, Any, Dict, List

from langchain_core.documents import Document
from typing_extensions import TypedDict


class RAGState(TypedDict, total=False):
    """RAG v2 流程的共享状态"""

    question: str
    sub_queries: List[str]
    documents: Annotated[List[Document], operator.add]
    deduped_documents: List[Document]
    answer: str
    validation: Dict[str, Any]


class RetrieveTask(TypedDict):
    """fan-out 时传入单个 retrieve_each 节点的载荷"""

    query: str
