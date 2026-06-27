"""RAG eval CLI.

用法:
    python -m tests.eval.run_eval                              # 跑 baseline (默认 cheat 占位)
    python -m tests.eval.run_eval --retriever vector --tag baseline-real
    python -m tests.eval.run_eval --retriever local-rerank --tag local-rerank-v1
    python -m tests.eval.run_eval --filter domain --only-names cpu_high_usage,service_unavailable
    python -m tests.eval.run_eval --top-k 5 --compare a b

流程:
    golden_set.jsonl
        ↓
    Retriever(question)  ──→  retrieved_doc_ids / retrieved_chunk_ids
        ↓
    metrics.aggregate(...) ──→ Hit@K / MRR / Recall@K
        ↓
    reports/<tag>-<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol

from tests.eval.metrics import AggregateReport, RetrievalCase, aggregate


EVAL_DIR = Path(__file__).parent
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.jsonl"
REPORTS_DIR = EVAL_DIR / "reports"


# =============================================================================
# 评测集加载
# =============================================================================

def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"golden_set.jsonl 第 {line_no} 行 JSON 解析失败: {exc}") from exc
    return items


# =============================================================================
# Retriever 协议
# =============================================================================

class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> tuple[list[str], list[str]]:
        """返回 (retrieved_doc_ids, retrieved_chunk_ids),按相关度从高到低。"""
        ...


class CheatRetriever:
    def __init__(self, golden_items: list[dict]) -> None:
        self._lookup = {item["id"]: item for item in golden_items}

    def search(self, query: str, top_k: int) -> tuple[list[str], list[str]]:
        for item in self._lookup.values():
            if item["question"] == query:
                return (
                    item.get("expected_doc_ids", [])[:top_k],
                    item.get("expected_chunk_ids", [])[:top_k],
                )
        return [], []


# =============================================================================
# 真实检索器(向量 + BM25 RRF 混合)
# =============================================================================

class VectorServiceRetriever:
    def __init__(
        self,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
        doc_id_filter: DocIdFilter | None = None,
    ) -> None:
        from app.services.vector_search_service import vector_search_service

        self._service = vector_search_service
        self._vector_weight = vector_weight
        self._bm25_weight = bm25_weight
        self._filter = doc_id_filter

    def search(self, query: str, top_k: int) -> tuple[list[str], list[str]]:
        candidate_k = max(top_k * 3, 10)
        docs = self._service.retrieve_documents(
            query,
            top_k=candidate_k,
            vector_weight=self._vector_weight,
            bm25_weight=self._bm25_weight,
        )

        chunk_ids: list[str] = []
        doc_ids_in_order: list[str] = []
        seen_docs: set[str] = set()

        for doc in docs:
            md = doc.metadata or {}
            chunk_id = md.get("chunk_id")
            file_name = md.get("_file_name") or ""
            doc_id = file_name.rsplit(".", 1)[0] if "." in file_name else file_name

            # 应用 doc_id 过滤器(领域过滤/黑名单/白名单)
            if self._filter and not self._filter.allow(doc_id):
                continue

            if chunk_id:
                chunk_ids.append(str(chunk_id))
            if doc_id and doc_id not in seen_docs:
                seen_docs.add(doc_id)
                doc_ids_in_order.append(doc_id)
            if len(doc_ids_in_order) >= top_k:
                break

        return doc_ids_in_order[:top_k], chunk_ids[:top_k]


def build_vector_retriever(
    vector_weight: float | None = None,
    bm25_weight: float | None = None,
    doc_id_filter: DocIdFilter | None = None,
) -> Retriever:
    return VectorServiceRetriever(
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        doc_id_filter=doc_id_filter,
    )


# =============================================================================
# DocIdFilter —— 候选文档领域过滤
# =============================================================================

class DocIdFilter:
    """对粗排结果的 doc_id 做领域相关性过滤。

    三种模式:
    - none:    不过滤(默认)
    - deny:    拒绝名单(doc_id 在黑名单里的直接踢掉)
    - allowlist: 白名单(不在白名单的直接踢掉)
    """

    def __init__(
        self,
        mode: str = "none",
        names: Iterable[str] = (),
    ) -> None:
        self.mode = mode
        self._names = set(names)

    def allow(self, doc_id: str) -> bool:
        if self.mode == "none":
            return True
        if self.mode == "deny":
            return doc_id not in self._names
        if self.mode == "allowlist":
            return doc_id in self._names
        return True

    @classmethod
    def from_args(cls, filter_mode: str | None, only_names: str | None) -> "DocIdFilter":
        """从 CLI 参数构造。"""
        if not filter_mode or filter_mode == "none":
            return cls(mode="none")

        names: list[str] = []
        if only_names:
            names = [n.strip() for n in only_names.split(",") if n.strip()]

        return cls(mode=filter_mode, names=names)


# =============================================================================
# 本地 Rerank 检索器
# =============================================================================

class LocalRerankedRetriever:
    MODEL_NAME = "BAAI/bge-reranker-v2-m3"

    def __init__(
        self,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
        candidate_k: int = 30,
        doc_id_filter: DocIdFilter | None = None,
    ) -> None:
        from app.services.vector_search_service import vector_search_service

        self._service = vector_search_service
        self._vector_weight = vector_weight
        self._bm25_weight = bm25_weight
        self._candidate_k = candidate_k
        self._filter = doc_id_filter
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            from loguru import logger
            logger.info(f"加载本地 Rerank 模型: {self.MODEL_NAME}")
            self._model = CrossEncoder(self.MODEL_NAME)
        return self._model

    def search(self, query: str, top_k: int) -> tuple[list[str], list[str]]:
        docs = self._service.retrieve_documents(
            query,
            top_k=self._candidate_k,
            vector_weight=self._vector_weight,
            bm25_weight=self._bm25_weight,
        )
        if not docs:
            return [], []

        doc_passages: list[tuple[str, str]] = []
        seen: set[str] = set()
        for doc in docs:
            md = doc.metadata or {}
            file_name = md.get("_file_name") or ""
            doc_id = file_name.rsplit(".", 1)[0] if "." in file_name else file_name

            if self._filter and not self._filter.allow(doc_id):
                continue

            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                passage = doc.page_content[:500] if doc.page_content else doc_id
                doc_passages.append((doc_id, passage))

        if not doc_passages or len(doc_passages) <= top_k:
            return [dp[0] for dp in doc_passages], []

        try:
            model = self._get_model()
            pairs = [(query, passage) for _, passage in doc_passages]
            scores = model.predict(pairs, show_progress_bar=False)
            ranked = sorted(zip(doc_passages, scores), key=lambda x: x[1], reverse=True)
            reranked_ids = [dp[0] for dp, _ in ranked]
            return reranked_ids[:top_k], []
        except Exception:
            import traceback
            from loguru import logger
            logger.warning(f"本地 Rerank 失败,降级为粗排: {traceback.format_exc()}")
            return [dp[0] for dp in doc_passages][:top_k], []


def build_local_reranked_retriever(
    vector_weight: float | None = None,
    bm25_weight: float | None = None,
    doc_id_filter: DocIdFilter | None = None,
) -> Retriever:
    return LocalRerankedRetriever(
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        doc_id_filter=doc_id_filter,
    )


# =============================================================================
# DashScope Rerank 检索器(API 权限需开通 gte-rerank)
# =============================================================================

class RerankedRetriever:
    RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    def __init__(
        self,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
        candidate_k: int = 30,
        doc_id_filter: DocIdFilter | None = None,
    ) -> None:
        import httpx
        from app.config import config
        from app.services.vector_search_service import vector_search_service

        self._service = vector_search_service
        self._vector_weight = vector_weight
        self._bm25_weight = bm25_weight
        self._candidate_k = candidate_k
        self._filter = doc_id_filter
        self._api_key = config.dashscope_api_key
        self._http = httpx.Client(timeout=httpx.Timeout(30.0))

    def search(self, query: str, top_k: int) -> tuple[list[str], list[str]]:
        docs = self._service.retrieve_documents(
            query,
            top_k=self._candidate_k,
            vector_weight=self._vector_weight,
            bm25_weight=self._bm25_weight,
        )
        if not docs:
            return [], []

        doc_passages: list[tuple[str, str]] = []
        seen: set[str] = set()
        for doc in docs:
            md = doc.metadata or {}
            file_name = md.get("_file_name") or ""
            doc_id = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
            if self._filter and not self._filter.allow(doc_id):
                continue
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                passage = doc.page_content[:500] if doc.page_content else doc_id
                doc_passages.append((doc_id, passage))

        if not doc_passages or len(doc_passages) <= top_k:
            return [dp[0] for dp in doc_passages], []

        try:
            doc_ids = [dp[0] for dp in doc_passages]
            passages = [dp[1] for dp in doc_passages]
            resp = self._http.post(
                self.RERANK_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gte-rerank",
                    "input": {"query": query, "documents": passages},
                    "parameters": {
                        "top_n": min(top_k, len(passages)),
                        "return_documents": False,
                    },
                },
            )
            resp.raise_for_status()
            body = resp.json()
            results = body.get("output", {}).get("results", [])
            reranked_ids = [doc_ids[item["index"]] for item in results]
            return reranked_ids[:top_k], []
        except Exception:
            import traceback
            from loguru import logger
            logger.warning(f"Rerank 调用失败,降级为粗排: {traceback.format_exc()}")
            return [dp[0] for dp in doc_passages][:top_k], []


def build_reranked_retriever(
    vector_weight: float | None = None,
    bm25_weight: float | None = None,
    doc_id_filter: DocIdFilter | None = None,
) -> Retriever:
    return RerankedRetriever(
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        doc_id_filter=doc_id_filter,
    )


# =============================================================================
# 执行评测 + 报告
# =============================================================================

def run(
    retriever: Retriever,
    golden_items: list[dict],
    top_k: int = 10,
) -> tuple[AggregateReport, list[RetrievalCase]]:
    cases: list[RetrievalCase] = []
    for item in golden_items:
        retrieved_docs, retrieved_chunks = retriever.search(item["question"], top_k=top_k)
        cases.append(
            RetrievalCase(
                case_id=item["id"],
                retrieved_doc_ids=retrieved_docs,
                expected_doc_ids=item.get("expected_doc_ids", []),
                retrieved_chunk_ids=retrieved_chunks,
                expected_chunk_ids=item.get("expected_chunk_ids", []),
                tags=item.get("tags", []),
            )
        )
    report = aggregate(cases, ks=(1, 3, 5, 10))
    return report, cases


def save_report(report: AggregateReport, cases: list[RetrievalCase], tag: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    out_path = REPORTS_DIR / f"{tag}-{timestamp}.json"
    payload = {
        "tag": tag,
        "timestamp": timestamp,
        "summary": asdict(report),
        "cases": [asdict(c) for c in cases],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def print_report(report: AggregateReport, tag: str) -> None:
    print(f"\n=== Eval Report [{tag}] ===")
    print(f"Total cases : {report.total}")
    print(f"MRR         : {report.mrr:.4f}")
    print("Hit@K       :", {k: f"{v:.4f}" for k, v in report.hit_at_k.items()})
    print("Recall@K    :", {k: f"{v:.4f}" for k, v in report.recall_at_k.items()})
    if report.by_tag:
        print("\n-- by tag --")
        for tag_name, sub in sorted(report.by_tag.items()):
            print(f"  [{tag_name}] n={sub.total} "
                  f"Hit@3={sub.hit_at_k.get(3, 0):.2f} "
                  f"MRR={sub.mrr:.2f}")


def load_report(report_name: str) -> dict:
    candidate = REPORTS_DIR / f"{report_name}.json"
    if not candidate.exists():
        matches = sorted(REPORTS_DIR.glob(f"{report_name}*.json"))
        if not matches:
            raise FileNotFoundError(f"找不到报告: {report_name}")
        candidate = matches[-1]
    return json.loads(candidate.read_text(encoding="utf-8"))


def compare_reports(name_a: str, name_b: str) -> None:
    a = load_report(name_a)
    b = load_report(name_b)
    print(f"\n=== Compare [{a['tag']}] vs [{b['tag']}] ===")
    sa = a["summary"]
    sb = b["summary"]
    print(f"{'Metric':<14}{'A':>10}{'B':>10}{'Δ':>10}")
    print(f"{'MRR':<14}{sa['mrr']:>10.4f}{sb['mrr']:>10.4f}{sb['mrr'] - sa['mrr']:>+10.4f}")
    for k in sorted({*sa['hit_at_k'].keys(), *sb['hit_at_k'].keys()}, key=int):
        va = sa['hit_at_k'].get(str(k), 0)
        vb = sb['hit_at_k'].get(str(k), 0)
        print(f"{'Hit@' + str(k):<14}{va:>10.4f}{vb:>10.4f}{vb - va:>+10.4f}")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG offline eval")
    parser.add_argument("--tag", default="baseline", help="本次实验名")
    parser.add_argument("--top-k", type=int, default=10, help="检索 Top-K")
    parser.add_argument("--retriever", default="cheat",
                        choices=["cheat", "vector", "rerank", "local-rerank"],
                        help="检索器选择")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="对比两份历史报告")
    parser.add_argument("--vector-weight", type=float, default=None)
    parser.add_argument("--bm25-weight", type=float, default=None)
    # 领域过滤
    parser.add_argument("--filter", dest="filter_mode", default=None,
                        choices=["none", "deny", "allowlist"],
                        help="候选 doc_id 过滤模式: deny=踢掉 --only-names 里的; allowlist=只保留")
    parser.add_argument("--only-names", type=str, default=None,
                        help="逗号分隔的 doc_id 名单,配合 --filter 使用")
    args = parser.parse_args()

    if args.compare:
        compare_reports(*args.compare)
        return

    golden_items = load_golden_set()
    print(f"Loaded {len(golden_items)} golden cases")

    doc_filter = DocIdFilter.from_args(args.filter_mode, args.only_names)
    if doc_filter.mode != "none":
        print(f"[FILTER] mode={doc_filter.mode} names={doc_filter._names}")

    if args.retriever == "cheat":
        retriever: Retriever = CheatRetriever(golden_items)
        print("[WARN] Using CheatRetriever (placeholder).")
    elif args.retriever == "rerank":
        retriever = build_reranked_retriever(
            vector_weight=args.vector_weight,
            bm25_weight=args.bm25_weight,
            doc_id_filter=doc_filter,
        )
    elif args.retriever == "local-rerank":
        retriever = build_local_reranked_retriever(
            vector_weight=args.vector_weight,
            bm25_weight=args.bm25_weight,
            doc_id_filter=doc_filter,
        )
    else:
        retriever = build_vector_retriever(
            vector_weight=args.vector_weight,
            bm25_weight=args.bm25_weight,
            doc_id_filter=doc_filter,
        )

    report, cases = run(retriever, golden_items, top_k=args.top_k)
    print_report(report, args.tag)
    out_path = save_report(report, cases, args.tag)
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
