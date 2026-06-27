"""chunk metadata 增强服务。

在 splitter 产出 chunks 之后、写向量库之前,给每个 chunk 的 metadata 补:
1. doc_tags:从文件名 + h1 推断的文档级分类标签
2. chunk_summary:chunk 首句(≤120 字),用于 Rerank / 前端预览

纯规则驱动,无 LLM 调用,30ms 内跑完。
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from langchain_core.documents import Document
from loguru import logger


class MetadataEnricher:
    """给 Document.metadata 补充业务字段。"""

    # 文件名关键词 → 标签映射(优先级从上到下,第一个命中就停止)
    TAG_RULES: ClassVar[list[tuple[str, list[str]]]] = [
        # 一级:领域
        ("aiops", ["cpu", "memory", "disk", "service", "slow", "latency", "alert",
                   "告警", "监控", "运维", "api_5xx", "high_api", "dependency",
                   "redis", "postgres", "milvus", "rq_worker", "mcp", "dashscope",
                   "document_index", "vector_search", "connection_failed",
                   "spike", "timeout", "unavailable", "stopped", "failed"]),
    ]

    def enrich_documents(
        self,
        documents: list[Document],
        file_path: str,
    ) -> list[Document]:
        """主入口。给每个 Document.metadata 注入 doc_tags + chunk_summary。"""
        if not documents:
            return documents

        doc_tags = self._infer_tags(file_path)
        file_name = Path(file_path).name

        for doc in documents:
            md = doc.metadata
            md["doc_tags"] = doc_tags
            md.setdefault("_file_name", file_name)
            md.setdefault("_source", file_path)
            md["chunk_summary"] = self._extract_summary(doc.page_content)

        logger.debug(f"MetadataEnricher: {len(documents)} chunks, tags={doc_tags}, file={file_name}")
        return documents

    # ---- 内部规则 ----

    def _infer_tags(self, file_path: str) -> list[str]:
        """从文件名 + 路径推断文档级标签。"""
        name = Path(file_path).stem.lower()  # cpu_high_usage
        tags: list[str] = []
        for tag_name, keywords in self.TAG_RULES:
            for kw in keywords:
                if kw in name:
                    if tag_name not in tags:
                        tags.append(tag_name)
                    break  # 已命中这个 tag,继续看其他规则
        return tags or ["general"]

    @staticmethod
    def _extract_summary(content: str, max_chars: int = 120) -> str:
        """取 chunk 首句(跳过代码块和纯标记行)作为摘要。"""
        if not content:
            return ""
        first_line = content.split("\n", 1)[0].strip()

        # 代码块标记行不取
        if first_line.startswith("```"):
            return ""

        # 去掉 markdown 标题标记(# ## ###)
        for prefix in ("### ", "## ", "# "):
            if first_line.startswith(prefix):
                first_line = first_line[len(prefix):].strip()
        if len(first_line) > max_chars:
            first_line = first_line[:max_chars - 1] + "…"
        return first_line


# 全局单例
metadata_enricher = MetadataEnricher()
