"""看一眼检索结果 chunk 的真实 metadata 键名和正文长度分布。"""
from app.services.vector_search_service import vector_search_service

docs = vector_search_service.retrieve_documents(
    "线上机器 CPU 一直九十多下不来,我先看哪里?", top_k=30
)
print(f"候选数 = {len(docs)}")
if docs:
    print("\n=== 第一个 chunk 的 metadata 全部键 ===")
    for key, value in (docs[0].metadata or {}).items():
        text = str(value)
        print(f"  {key!r} = {text[:80]}")

print("\n=== chunk_id / id 键的覆盖情况 ===")
has_chunk_id = sum(1 for d in docs if (d.metadata or {}).get("chunk_id"))
has_id = sum(1 for d in docs if (d.metadata or {}).get("id"))
print(f"  有 'chunk_id' 的: {has_chunk_id}/{len(docs)}")
print(f"  有 'id' 的      : {has_id}/{len(docs)}")

print("\n=== chunk 正文长度分布 ===")
lengths = sorted(len(d.page_content or "") for d in docs)
print(f"  min={lengths[0]} p50={lengths[len(lengths) // 2]} max={lengths[-1]}")
over_500 = sum(1 for n in lengths if n > 500)
print(f"  超过 500 字的 chunk: {over_500}/{len(lengths)}  (这些被截断了)")

print("\n=== 同一文档的不同 chunk 内容是否有区分度 ===")
from collections import defaultdict

by_doc = defaultdict(list)
for d in docs:
    file_name = (d.metadata or {}).get("_file_name") or ""
    doc_id = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
    by_doc[doc_id].append(d.page_content or "")

for doc_id, texts in list(by_doc.items())[:2]:
    print(f"\n[{doc_id}] {len(texts)} chunks")
    for i, text in enumerate(texts):
        print(f"  chunk{i} ({len(text)}字) 前80: {text[:80]}")
