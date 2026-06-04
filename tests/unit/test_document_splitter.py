"""文档分割器单元测试"""

import pytest
from langchain_core.documents import Document

from app.services.document_splitter_service import DocumentSplitterService


@pytest.fixture
def splitter() -> DocumentSplitterService:
    # 每个测试拿到一个全新的 splitter，避免前一个用例污染后一个。
    return DocumentSplitterService()


class TestTextSplit:
    def test_empty_content_returns_empty(self, splitter: DocumentSplitterService) -> None:
        result = splitter.split_text("", "empty.txt")
        assert result == []

    def test_whitespace_only_returns_empty(self, splitter: DocumentSplitterService) -> None:
        result = splitter.split_text("   \n  ", "ws.txt")
        assert result == []

    def test_short_text_produces_single_document(self, splitter: DocumentSplitterService) -> None:
        content = "Hello, this is a short document."
        docs = splitter.split_text(content, "short.txt")
        assert len(docs) >= 1
        assert any("Hello" in d.page_content for d in docs)

    def test_metadata_is_set(self, splitter: DocumentSplitterService) -> None:
        # 这类 metadata 后面会被检索、去重、溯源逻辑依赖，所以单独锁住。
        docs = splitter.split_text("some content", "myfile.txt")
        for d in docs:
            assert d.metadata["_source"] == "myfile.txt"
            assert d.metadata["_file_name"] == "myfile.txt"


class TestMarkdownSplit:
    def test_empty_md_returns_empty(self, splitter: DocumentSplitterService) -> None:
        result = splitter.split_markdown("", "empty.md")
        assert result == []

    def test_md_with_single_header(self, splitter: DocumentSplitterService) -> None:
        content = "# Header One\n\nSome paragraph text here."
        docs = splitter.split_markdown(content, "test.md")
        assert len(docs) >= 1

    def test_md_metadata_has_extension(self, splitter: DocumentSplitterService) -> None:
        content = "# Title\n\nBody text."
        docs = splitter.split_markdown(content, "doc.md")
        for d in docs:
            assert d.metadata["_extension"] == ".md"


class TestSplitDocumentRouter:
    def test_md_extension_routes_to_markdown(self, splitter: DocumentSplitterService) -> None:
        # split_document 是一个分发入口；这里验证它会按扩展名选对底层 splitter。
        content = "# Title\n\nText."
        docs = splitter.split_document(content, "readme.md")
        for d in docs:
            assert d.metadata["_extension"] == ".md"

    def test_txt_extension_routes_to_text(self, splitter: DocumentSplitterService) -> None:
        docs = splitter.split_document("plain text", "notes.txt")
        assert len(docs) >= 1
