"""临时脚本:检查 Milvus 当前实体数"""
from app.core.milvus_client import milvus_manager

milvus_manager.connect()
c = milvus_manager.get_collection()
print(f"collection: {c.name}, num_entities: {c.num_entities}")
