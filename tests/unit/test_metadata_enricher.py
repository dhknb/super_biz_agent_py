from langchain_core.documents import Document

from app.services.metadata_enricher import metadata_enricher


def test_infer_tags_from_filename() -> None:
    tags = metadata_enricher._infer_tags("/data/cpu_high_usage.md")
    assert "aiops" in tags

    tags = metadata_enricher._infer_tags("/data/redis_connection_failed.md")
    assert "aiops" in tags

    tags = metadata_enricher._infer_tags("/data/unknown_file.md")
    assert tags == ["general"]


def test_enrich_documents_adds_doc_tags() -> None:
    docs = [Document(page_content="# CPU 排查\n用 top 定位进程。", metadata={})]
    result = metadata_enricher.enrich_documents(docs, "/data/cpu_high_usage.md")
    assert result[0].metadata["doc_tags"] == ["aiops"]


def test_enrich_documents_adds_chunk_summary() -> None:
    docs = [Document(page_content="### 排查步骤\n用 top 定位高 CPU 进程。继续...")]
    result = metadata_enricher.enrich_documents(docs, "/data/whatever.md")
    # 首句去掉 ### 标记后是 "排查步骤"
    assert result[0].metadata["chunk_summary"] == "排查步骤"


def test_enrich_documents_empty_list() -> None:
    result = metadata_enricher.enrich_documents([], "/data/whatever.md")
    assert result == []


def test_summary_truncates_long_content() -> None:
    long_text = "A" * 200
    s = metadata_enricher._extract_summary(long_text, max_chars=50)
    assert len(s) <= 50
    assert s.endswith("…")


def test_summary_handles_fenced_code() -> None:
    s = metadata_enricher._extract_summary("```python\nx=1\n```")
    assert s == ""
