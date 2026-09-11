"""`used_documents` 存储格式的写侧瘦身 / 读侧还原测试。

这一批测试守着三条线：

1. **写侧真的瘦了**：落库的行里不再有全文，只有定位信息 + 摘录。
2. **读侧新旧都能吃**：老 trace（全文格式）和新 trace（瘦身格式）
   读出来都是同一副形状，前端不必判断数据是哪个年代写的。
3. **老字段的类型没变**：`content` 依然是 str。这是向后兼容的底线 ——
   多几个新键前端可以忽略，`content` 变成 None 会让线上页面白屏。

第 3 条是这批测试里最该盯的一条：它是「兼容」和「悄悄破坏」的分界线。
"""

import pytest

from app.core.used_documents import (
    EXCERPT_MAX_CHARS,
    normalize_used_documents,
    slim_used_documents,
)


def _legacy_row(content: str = "分片正文", **metadata) -> dict:
    """老格式的一行：全文 + metadata。"""
    md = {
        "id": "chunk-1",
        "document_id": "doc-1",
        "_file_name": "cpu_high_usage.md",
        "_source": "/aiops-docs/cpu_high_usage.md",
        "_sub_query": "CPU 为什么飙高",
    }
    md.update(metadata)
    return {"content": content, "metadata": md}


# ── 写侧：瘦身 ────────────────────────────────────────────


def test_slim_drops_full_content():
    long_text = "啊" * 3000
    rows = slim_used_documents([_legacy_row(long_text)])

    assert len(rows) == 1
    row = rows[0]
    # 全文不再落库 —— 这就是这次改动的全部目的
    assert "content" not in row
    assert len(row["excerpt"]) <= EXCERPT_MAX_CHARS
    # 但原文有多长这个事实要留着
    assert row["content_length"] == 3000


def test_slim_keeps_locating_fields():
    row = slim_used_documents([_legacy_row()])[0]
    assert row["chunk_id"] == "chunk-1"
    assert row["document_id"] == "doc-1"
    # 文件名优先于完整路径
    assert row["source"] == "cpu_high_usage.md"
    assert row["sub_query"] == "CPU 为什么飙高"


def test_slim_falls_back_to_source_path_when_no_file_name():
    legacy = _legacy_row()
    del legacy["metadata"]["_file_name"]
    row = slim_used_documents([legacy])[0]
    assert row["source"] == "/aiops-docs/cpu_high_usage.md"


def test_slim_score_is_none_on_ensemble_path():
    """rag_v2 走 EnsembleRetriever，拿不到融合分数。

    如实存 None，而不是拿名次倒数伪造一个「看起来像相关性分数」的值 ——
    后者会让读者据此比较分片质量，而那个比较毫无意义。
    """
    row = slim_used_documents([_legacy_row()])[0]
    assert row["score"] is None


def test_slim_picks_up_score_from_either_level():
    top_level = _legacy_row()
    top_level["score"] = 0.87
    assert slim_used_documents([top_level])[0]["score"] == pytest.approx(0.87)

    in_metadata = _legacy_row(score=0.42)
    assert slim_used_documents([in_metadata])[0]["score"] == pytest.approx(0.42)


def test_slim_ignores_unparsable_score():
    bad = _legacy_row()
    bad["score"] = "很高"
    assert slim_used_documents([bad])[0]["score"] is None


def test_slim_is_idempotent():
    """流式路径的 used_documents 来自 SSE payload，可能被重复加工。

    幂等让调用方不必判断「这份数据到底瘦没瘦」。
    """
    once = slim_used_documents([_legacy_row("啊" * 500)])
    twice = slim_used_documents(once)
    assert twice == once


def test_slim_survives_missing_and_malformed_metadata():
    rows = slim_used_documents(
        [
            {"content": "没有 metadata"},
            {"content": "metadata 不是 dict", "metadata": "oops"},
            {},
        ]
    )
    assert len(rows) == 3
    assert all(row["chunk_id"] is None for row in rows)
    assert rows[2]["content_length"] == 0


def test_slim_skips_non_dict_rows():
    assert slim_used_documents(["字符串", None, 42]) == []


@pytest.mark.parametrize("empty", [None, []])
def test_slim_handles_empty(empty):
    assert slim_used_documents(empty) == []


def test_short_content_is_not_truncated():
    row = slim_used_documents([_legacy_row("很短")])[0]
    assert row["excerpt"] == "很短"
    assert "…" not in row["excerpt"]


def test_long_content_gets_ellipsis():
    row = slim_used_documents([_legacy_row("啊" * (EXCERPT_MAX_CHARS + 1))])[0]
    assert row["excerpt"].endswith("…")
    assert len(row["excerpt"]) == EXCERPT_MAX_CHARS


# ── 读侧：还原 ────────────────────────────────────────────


def test_normalize_legacy_row_keeps_full_content():
    """老 trace 里全文还在，读出来就该是全文，并如实标 excerpt_only=False。"""
    long_text = "啊" * 3000
    row = normalize_used_documents([_legacy_row(long_text)])[0]

    assert row["content"] == long_text
    assert row["excerpt_only"] is False
    assert row["content_length"] == 3000
    # 新字段也补齐了，读侧不必分辨数据年代
    assert row["chunk_id"] == "chunk-1"
    assert row["source"] == "cpu_high_usage.md"


def test_normalize_slim_row_exposes_excerpt_as_content():
    """新 trace 没有全文，content 位放摘录，并且必须标明这是节选。

    不标的话就是在悄悄提供残缺数据 —— 读者会以为分片本来就这么短。
    """
    slim = slim_used_documents([_legacy_row("啊" * 3000)])
    row = normalize_used_documents(slim)[0]

    assert row["content"] == row["excerpt"]
    assert row["excerpt_only"] is True
    # content_length 记的仍是原文长度，不是摘录长度
    assert row["content_length"] == 3000


def test_normalize_output_shape_is_identical_for_old_and_new():
    """新旧两种格式读出来的键集合必须一致。

    这是「读侧不必判断数据年代」的形式化表达：只要键集合相同，
    前端的一套渲染逻辑就能同时吃两个年代的数据。
    """
    legacy = normalize_used_documents([_legacy_row("啊" * 800)])[0]
    slim = normalize_used_documents(slim_used_documents([_legacy_row("啊" * 800)]))[0]
    assert set(legacy) == set(slim)


def test_normalize_never_changes_content_type():
    """兼容底线：content 永远是 str，绝不是 None。

    多几个新键前端可以忽略；content 变成 None 会让线上页面白屏。
    """
    cases = [
        _legacy_row(""),
        {"content": None, "metadata": {}},
        {},
        slim_used_documents([_legacy_row("")])[0],
    ]
    for row in normalize_used_documents(cases):
        assert isinstance(row["content"], str)


def test_normalize_slim_row_without_content_length():
    """手工构造的 / 早期瘦身格式可能缺 content_length，退化成摘录长度。"""
    row = normalize_used_documents([{"excerpt": "一小段", "chunk_id": "c1"}])[0]
    assert row["content_length"] == len("一小段")


def test_normalize_skips_non_dict_rows():
    assert normalize_used_documents(["字符串", None]) == []


@pytest.mark.parametrize("empty", [None, []])
def test_normalize_handles_empty(empty):
    assert normalize_used_documents(empty) == []
