"""向量检索服务单元测试

mock 掉 Milvus 和 vector_store_manager，只验证检索流程逻辑。
"""

from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


class FakeVectorStore:
    def as_retriever(self, **kwargs):  # type: ignore[no-untyped-def]
        # 用一个极小的 fake 替代真实向量库，让测试只聚焦“检索结果如何被处理”。
        retriever = MagicMock()
        retriever.invoke = MagicMock(
            return_value=[
                Document(page_content="doc1 content", metadata={"id": "1"}),
                Document(page_content="doc2 content", metadata={"id": "2"}),
            ]
        )
        return retriever


@pytest.fixture
def mock_dependencies() -> None:
    # 这里同时 patch vector store 和 milvus collection，
    # 这样 retrieve_documents 不会碰真实 Milvus，也不会触发 BM25 构建。
    with patch(
        "app.services.vector_search_service.vector_store_manager"
    ) as mock_vsm, patch(
        "app.services.vector_search_service.milvus_manager"
    ) as mock_mm:
        mock_vsm.get_vector_store.return_value = FakeVectorStore()
        mock_mm.get_collection.return_value.num_entities = 0  # no BM25 corpus
        yield


class TestRetrieveDocuments:
    def test_returns_documents(self, mock_dependencies: None) -> None:
        from app.services.vector_search_service import vector_search_service

        docs = vector_search_service.retrieve_documents("test query", top_k=2)
        assert len(docs) > 0
        assert isinstance(docs[0], Document)

    def test_respects_top_k(self, mock_dependencies: None) -> None:
        from app.services.vector_search_service import vector_search_service

        docs = vector_search_service.retrieve_documents("test query", top_k=1)
        assert len(docs) == 1


class TestSearchSimilarDocuments:
    def test_returns_search_results(self, mock_dependencies: None) -> None:
        from app.services.vector_search_service import vector_search_service

        results = vector_search_service.search_similar_documents("test", top_k=2)
        assert len(results) > 0
        result = results[0]
        assert hasattr(result, "id")
        assert hasattr(result, "content")
        assert hasattr(result, "score")
        assert hasattr(result, "metadata")


class TestTokenizeForBM25:
    def test_chinese_characters_are_individual_tokens(self) -> None:
        from app.services.vector_search_service import VectorSearchService

        # 中文按字切分是 BM25 能否命中中文查询的关键行为。
        tokens = VectorSearchService._tokenize_for_bm25("你好世界")
        assert "你" in tokens
        assert "好" in tokens
        assert "世" in tokens
        assert "界" in tokens

    def test_english_words_are_lowercased(self) -> None:
        from app.services.vector_search_service import VectorSearchService

        tokens = VectorSearchService._tokenize_for_bm25("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_mixed_cn_en(self) -> None:
        from app.services.vector_search_service import VectorSearchService

        tokens = VectorSearchService._tokenize_for_bm25("AI人工智能")
        assert "ai" in tokens
