"""对照 DB 和 Milvus,找出 DB 里 indexed 但 Milvus 里缺向量的文档"""
from app.core.milvus_client import milvus_manager
from app.core.database import SessionLocal
from app.models.knowledge_base import KnowledgeDocument, DocumentStatus
from sqlalchemy import select
import json

milvus_manager.connect()
c = milvus_manager.get_collection()

db = SessionLocal()
active = db.scalars(
    select(KnowledgeDocument).where(KnowledgeDocument.status != DocumentStatus.DELETED)
).all()
db.close()

# 按 document_id 统计 Milvus chunk
milvus_by_doc = {}
all_rows = c.query(expr='id != ""', output_fields=["metadata"], limit=9999)
for r in all_rows:
    md = r.get("metadata", {})
    if isinstance(md, str):
        md = json.loads(md)
    did = md.get("document_id", "?")
    milvus_by_doc[did] = milvus_by_doc.get(did, 0) + 1

print(f"{'filename':45s} {'DB':12s} {'Milvus_chunks':>6s}")
missing = []
for d in sorted(active, key=lambda x: x.filename):
    mc = milvus_by_doc.get(d.id, 0)
    flag = "⚠️ " if mc == 0 else ""
    print(f"{flag}{d.filename:45s} {d.status.value:12s} {str(mc):>6s}")
    if mc == 0 and d.status == DocumentStatus.INDEXED:
        missing.append(d)

print(f"\nTotal active: {len(active)}, missing vectors: {len(missing)}")
if missing:
    print("⚠️  以下文档 DB 显示 indexed 但 Milvus 里没有向量:")
    for d in missing:
        print(f"  id={d.id}  filename={d.filename}  version={d.version}")
