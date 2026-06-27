"""RAG v2 - 多查询改写并行检索版本

链路: 问题 → LLM 改写多个子查询 → 并行检索 → 去重 → 生成
"""

from app.agent.rag_v2.service import rag_v2_service

__all__ = ["rag_v2_service"]
