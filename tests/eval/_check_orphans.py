"""查看孤儿向量的 metadata 详情"""
from app.core.milvus_client import milvus_manager
import json

milvus_manager.connect()
c = milvus_manager.get_collection()
rows = c.query(expr='id != ""', output_fields=["metadata", "id"], limit=9999)

print(f"Total rows: {len(rows)}")
print()

for r in rows:
    md = r.get("metadata", {})
    if isinstance(md, str):
        md = json.loads(md)
    did = md.get("document_id", "MISSING")
    if did == "MISSING" or not did:
        fn = md.get("_file_name", "?")
        vid = r.get("id", "?")
        keys = sorted(md.keys())
        print(f"ORPHAN: id={vid[:30]}  file={fn}  keys={keys}")
        print(f"        md={json.dumps(md, ensure_ascii=False)[:200]}")
        print()
