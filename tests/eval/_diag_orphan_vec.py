"""找出 Milvus 里 DB 不认识的孤儿向量(残留的旧文档)"""
from app.core.milvus_client import milvus_manager
from app.core.database import SessionLocal
from app.models.knowledge_base import KnowledgeDocument
from sqlalchemy import select
import json

milvus_manager.connect()
c = milvus_manager.get_collection()

# 获取所有已知的 document_id
db = SessionLocal()
known_ids = set()
all_docs = db.scalars(select(KnowledgeDocument)).all()
for d in all_docs:
    known_ids.add(d.id)
db.close()

# 遍历 Milvus 所有向量,按 _file_name 分组
all_rows = c.query(expr='id != ""', output_fields=["metadata"], limit=9999)
orphan_by_file = {}
orphan_by_did = {}
for r in all_rows:
    md = r.get("metadata", {})
    if isinstance(md, str):
        md = json.loads(md)
    fn = md.get("_file_name", "?")
    did = md.get("document_id", "?")
    if did not in known_ids:
        orphan_by_file[fn] = orphan_by_file.get(fn, 0) + 1
        orphan_by_did[did] = orphan_by_did.get(did, 0) + 1

print(f"Total vectors: {len(all_rows)}")
print(f"Known document_ids: {len(known_ids)}")
print(f"Orphan vectors: {sum(orphan_by_file.values())} across {len(orphan_by_file)} files")
print(f"Orphan document_ids: {len(orphan_by_did)}")

if orphan_by_file:
    print("\nOrphan vectors by file:")
    for fn, cnt in sorted(orphan_by_file.items(), key=lambda x: -x[1]):
        print(f"  {fn:50s} {cnt:4d} chunks")

if orphan_by_did:
    print("\nOrphan document_ids (full UUID):")
    for did, cnt in sorted(orphan_by_did.items()):
        print(f"  {did}  {cnt} chunks")
