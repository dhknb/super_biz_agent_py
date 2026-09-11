"""`chat_run_traces.used_documents` 的存储格式：写侧瘦身 + 读侧还原。

**为什么需要这个模块**

`ChatRunTrace.used_documents`（app/models/chat_run_trace.py:42）原来存的是
检索命中分片的**全文**：

    {"content": "<page_content 全文>", "metadata": {...}}

`FINAL_TOP_K = 6`，单个分片常有 1000+ 字符，于是每一轮对话都往 Postgres 里
写下 6KB 上下的正文。而这份正文早已存在两处 —— Milvus 的 `content` 字段一份、
`knowledge_chunks` 表一份。trace 表这**第三份拷贝**没有带来任何新信息，
只是让「查一次对话引用了哪些资料」这个需求付出了全量正文的存储代价。
一天一万轮对话就是 60MB 纯冗余，而且它们躺在业务库里，跟着每一次
`pg_dump`、每一次主从同步、每一次备份走。

trace 详情页真正要回答的是「这次回答引用了哪些分片」。读者需要的是
**定位信息**（哪个 chunk、属于哪个文档、由哪个子查询召回）加一小段
**认脸用的摘录**。要看全文，顺着 `chunk_id` 回源即可 —— 那才是全文该待的地方。

**为什么切点在写库前，而不是改 `_serialize_docs`**

`_serialize_docs`（app/agent/rag_v2/service.py:175）的产物同时喂给三处：
SSE 的 `used_documents` 事件、HTTP 响应体的 `used_documents` 字段、以及
trace 落库。前两处是前端契约，动了就是破坏性变更。只有落库这一路该瘦身，
所以瘦身发生在写库前，而不是在序列化时。

**关于 score**

rag_v2 的检索走 `EnsembleRetriever.invoke()`，它不暴露 RRF 融合分数
（见 app/services/vector_search_service.py:129 的注释）。所以这条链路上
`score` 拿不到，如实存 `None`，不用名次伪造一个分数 —— 一个看起来像
相关性分数、实际是排名倒数的值，比没有分数更危险。保留这个键是为了
将来接上 rerank 时不必再改格式。
"""

from __future__ import annotations

from typing import Any

# 摘录长度。200 字符够认出「这是哪一段」，又不至于变成第二份全文。
EXCERPT_MAX_CHARS = 200

# 瘦身格式的判定键。选 excerpt 而不是 chunk_id：chunk_id 可能为 None
# （metadata 里没 id 的历史分片），而 excerpt 键在瘦身格式里一定存在。
_SLIM_MARKER = "excerpt"


def slim_used_documents(rows: list[dict] | None) -> list[dict]:
    """把 used_documents 瘦身成落库格式。

    对已经是瘦身格式的输入是**幂等**的 —— 流式路径的 used_documents 来自
    SSE payload，可能被重复加工，幂等能省掉调用方的「这个到底瘦没瘦」判断。
    """
    if not rows:
        return []
    return [_slim_one(row) for row in rows if isinstance(row, dict)]


def normalize_used_documents(rows: list[dict] | None) -> list[dict]:
    """把落库的 used_documents 还原成读侧形状，新旧格式都能吃。

    输出刻意做成**超集**：新字段（chunk_id / excerpt / ...）与老字段
    （content / metadata）同时在。这样依赖 `content` 的前端读出来依然是
    字符串，不会 KeyError —— 只是新数据的 content 是 200 字摘录而非全文，
    并由 `excerpt_only` 如实标注这一点。

    宁愿多返回几个键，也不改变已有键的类型：前者前端可以忽略，
    后者会让线上页面白屏。
    """
    if not rows:
        return []
    return [_normalize_one(row) for row in rows if isinstance(row, dict)]


# ── 内部 ────────────────────────────────────────────────


def _slim_one(row: dict) -> dict:
    if _SLIM_MARKER in row:
        return row  # 已经是瘦身格式，原样放行（幂等）

    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    content = row.get("content") or ""

    return {
        # Milvus 主键即 chunk 的身份。回源查全文靠它。
        "chunk_id": metadata.get("id"),
        "document_id": metadata.get("document_id"),
        # 文件名优先于完整路径：给人看的时候前者更短也更有信息量。
        "source": metadata.get("_file_name") or metadata.get("_source"),
        # 哪个子查询召回的。排查「改写把问题带偏了」时就靠这个字段。
        "sub_query": metadata.get("_sub_query"),
        "score": _pick_score(row, metadata),
        "excerpt": _excerpt(content),
        # 留着原文长度：能看出摘录是完整的还是截断的，
        # 也让「平均命中分片有多长」这类问题不必回源就能答。
        "content_length": len(content),
    }


def _normalize_one(row: dict) -> dict:
    if _SLIM_MARKER in row:
        excerpt = row.get("excerpt") or ""
        content_length = row.get("content_length")
        return {
            **row,
            # content 是给老读者的兼容位，值是摘录而非全文，
            # 所以必须配上 excerpt_only 说明，否则就是在悄悄提供残缺数据。
            "content": excerpt,
            "excerpt_only": True,
            "metadata": row.get("metadata") or {},
            "content_length": content_length
            if isinstance(content_length, int)
            else len(excerpt),
        }

    # 老格式：全文还在，补齐新字段后原样返回。
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    content = row.get("content") or ""
    return {
        "chunk_id": metadata.get("id"),
        "document_id": metadata.get("document_id"),
        "source": metadata.get("_file_name") or metadata.get("_source"),
        "sub_query": metadata.get("_sub_query"),
        "score": _pick_score(row, metadata),
        "excerpt": _excerpt(content),
        "content_length": len(content),
        "content": content,
        "excerpt_only": False,
        "metadata": metadata,
    }


def _excerpt(content: str) -> str:
    if len(content) <= EXCERPT_MAX_CHARS:
        return content
    # 末尾加省略号：让读者一眼看出这是节选，不会误以为分片就这么短。
    return content[: EXCERPT_MAX_CHARS - 1] + "…"


def _pick_score(row: dict, metadata: dict) -> float | None:
    """顶层和 metadata 两处都找一下 score，取不到就 None。

    两处都看是因为不同检索路径把分数放在不同层级
    （SearchResult.to_dict 放顶层，将来的 rerank 大概会写进 metadata）。
    """
    for candidate in (row.get("score"), metadata.get("score")):
        if candidate is None:
            continue
        try:
            return float(candidate)
        except (TypeError, ValueError):
            continue
    return None
