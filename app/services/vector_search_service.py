"""混合检索服务模块"""

from typing import Any, Dict, List

from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from loguru import logger

from app.core.milvus_client import milvus_manager
from app.services.vector_store_manager import vector_store_manager


class SearchResult:
    """搜索结果类"""

    def __init__(
        self,
        id: str,
        content: str,
        score: float,
        metadata: Dict[str, Any],
    ):
        self.id = id
        self.content = content
        self.score = score
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


class VectorSearchService:
    """混合检索服务 - 使用 Milvus 向量检索 + BM25 关键词检索"""

    VECTOR_WEIGHT = 0.7
    BM25_WEIGHT = 0.3
    BM25_BATCH_SIZE = 1000

    def __init__(self):
        """初始化混合检索服务"""
        self._bm25_retriever: BM25Retriever | None = None
        self._bm25_doc_count = -1
        logger.info("混合检索服务初始化完成")

    def retrieve_documents(
        self,
        query: str,
        top_k: int = 3,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
    ) -> List[Document]:
        """
        使用 EnsembleRetriever 执行混合检索

        Args:
            query: 查询文本
            top_k: 返回最相关的 K 个文档
            vector_weight: 向量检索权重(可选,默认走类常量 VECTOR_WEIGHT)
            bm25_weight: BM25 检索权重(可选,默认走类常量 BM25_WEIGHT)
                          —— 评测 / A/B 时可临时覆盖,生产代码调用不需要传

        Returns:
            List[Document]: 融合排序后的文档列表

        Raises:
            RuntimeError: 检索失败时抛出
        """
        vw = vector_weight if vector_weight is not None else self.VECTOR_WEIGHT
        bw = bm25_weight if bm25_weight is not None else self.BM25_WEIGHT

        try:
            logger.info(
                f"开始混合检索, 查询: {query}, topK: {top_k}, "
                f"weights=(vector={vw}, bm25={bw})"
            )

            candidate_k = max(top_k * 3, top_k, 10)

            vector_store = vector_store_manager.get_vector_store()
            vector_retriever = vector_store.as_retriever(
                search_kwargs={"k": candidate_k}
            )

            bm25_retriever = self._get_bm25_retriever(candidate_k)
            if bm25_retriever is None:
                logger.warning("BM25 语料为空，降级为纯向量检索")
                return vector_retriever.invoke(query)[:top_k]
            #RRF混合排序
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[vw, bw],
            )

            docs = ensemble_retriever.invoke(query)[:top_k]
            logger.info(f"混合检索完成, 找到 {len(docs)} 个相关文档")
            return docs

        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            raise RuntimeError(f"混合检索失败: {e}") from e

    def search_similar_documents(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        搜索相似文档

        Args:
            query: 查询文本
            top_k: 返回最相关的 K 个结果

        Returns:
            List[SearchResult]: 搜索结果列表

        Raises:
            RuntimeError: 搜索失败时抛出
        """
        try:
            docs = self.retrieve_documents(query, top_k)
            return [
                SearchResult(
                    id=str(doc.metadata.get("id", doc.metadata.get("pk", index))),
                    content=doc.page_content,
                    # EnsembleRetriever 不暴露 RRF 分数，这里用排序名次提供稳定的相关性分数。
                    score=1.0 / index,
                    metadata=doc.metadata,
                )
                for index, doc in enumerate(docs, start=1)
            ]

        except Exception as e:
            logger.error(f"搜索相似文档失败: {e}")
            raise RuntimeError(f"搜索失败: {e}") from e

    def _get_bm25_retriever(self, top_k: int) -> BM25Retriever | None:
        """获取 BM25 检索器，并在 Milvus 文档数量变化时重建缓存"""
        collection = milvus_manager.get_collection()
        doc_count = collection.num_entities

        if self._bm25_retriever is not None and self._bm25_doc_count == doc_count:
            self._bm25_retriever.k = top_k
            return self._bm25_retriever

        documents = self._load_documents_from_milvus()
        if not documents:
            self._bm25_retriever = None
            self._bm25_doc_count = doc_count
            return None

        self._bm25_retriever = BM25Retriever.from_documents(
            documents,
            preprocess_func=self._tokenize_for_bm25,
        )
        self._bm25_retriever.k = top_k
        self._bm25_doc_count = doc_count

        logger.info(f"BM25 检索器已重建, 语料数: {len(documents)}")
        return self._bm25_retriever

    def _load_documents_from_milvus(self) -> List[Document]:
        """从 Milvus 读取所有文档分片，作为 BM25 关键词检索语料"""
        collection = milvus_manager.get_collection()
        documents: List[Document] = []

        iterator = collection.query_iterator(
            batch_size=self.BM25_BATCH_SIZE,
            expr='id != ""',
            output_fields=["id", "content", "metadata"],
        )

        try:
            while True:
                rows = iterator.next()
                if not rows:
                    break

                for row in rows:
                    content = row.get("content") or ""
                    if not content.strip():
                        continue

                    metadata = row.get("metadata") or {}
                    if not isinstance(metadata, dict):
                        metadata = {"raw_metadata": metadata}
                    metadata["id"] = row.get("id")

                    documents.append(
                        Document(
                            page_content=content,
                            metadata=metadata,
                        )
                    )
        finally:
            iterator.close()

        return documents

    @staticmethod
    def _tokenize_for_bm25(text: str) -> List[str]:
        """适配中英文混合内容的简单 BM25 分词"""
        tokens: List[str] = []
        buffer: List[str] = []

        def flush_buffer() -> None:
            if buffer:
                tokens.append("".join(buffer).lower())
                buffer.clear()

        for char in text:
            if char.isascii() and (char.isalnum() or char in {"_", "-"}):
                buffer.append(char)
            elif "\u4e00" <= char <= "\u9fff":
                flush_buffer()
                tokens.append(char)
            else:
                flush_buffer()

        flush_buffer()
        return tokens


# 全局单例
vector_search_service = VectorSearchService()
