"""RAG eval CLI.

用法:
    python -m tests.eval.run_eval                              # 跑 baseline (默认 cheat 占位)
    python -m tests.eval.run_eval --retriever vector --tag baseline-real
    python -m tests.eval.run_eval --retriever local-rerank --tag local-rerank-v1
    python -m tests.eval.run_eval --retriever multi-query --tag mq-rrf
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

## 四个检索器的关系(读报告前必须知道)

    cheat         从 golden set 直接抄答案 —— 只验管道,指标无意义
    vector        原始问题查一次(向量+BM25 RRF)
    local-rerank  vector 的候选 + 本地 CrossEncoder 重排
    multi-query   rewrite 出 N 个子查询各查一次 + RRF 融合   ← 线上 chat_v2 走的就是这条

**只有 multi-query 与线上链路同构**(app/agent/rag_v2/graph.py:43-48)。
前三个测的都是单查询召回,拿它们的指标推断线上表现是不成立的。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol

from tests.eval.metrics import AggregateReport, RetrievalCase, aggregate
from tests.eval.multi_query import (
    RERANK_CHUNK_CHARS,
    RERANK_MODEL_NAME,
    RERANK_PASSAGE_CHARS,
    RRF_K,
    build_multi_query_retriever,
    doc_id_of,
    load_cross_encoder,
    max_pool_to_docs,
)


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
            # doc_id 口径收敛到 multi_query.doc_id_of:四个检索器共用一套推导,
            # 否则某处漂移会让历史报告的对比静默失效,而没有任何断言会失败。
            doc_id = doc_id_of(doc)

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
    # 模型名只在 multi_query 里定义一份。写死成字面量的话,改了那边不会有任何
    # 断言失败,只会让 B 组和 D 组静默用上不同模型,而报告看起来照样正常。
    MODEL_NAME = RERANK_MODEL_NAME

    def __init__(
        self,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
        candidate_k: int = 30,
        doc_id_filter: DocIdFilter | None = None,
        rerank_granularity: str = "chunk",
    ) -> None:
        from app.services.vector_search_service import vector_search_service

        self._service = vector_search_service
        self._vector_weight = vector_weight
        self._bm25_weight = bm25_weight
        self._candidate_k = candidate_k
        self._filter = doc_id_filter
        # "chunk" = 逐 chunk 打分后 max-pool 到文档;"doc" = 每篇文档只用第一个
        # 命中 chunk 的前 500 字。默认 chunk,但保留 doc 以便原样复现历史报告。
        self._rerank_granularity = rerank_granularity
        self._model = None

    def _get_model(self):
        # 复用 multi_query.load_cross_encoder:两条 rerank 路径必须是同一个模型
        # 实例、同一套本地权重解析,否则 B 组和 D 组的报告不可比。
        if self._model is None:
            self._model = load_cross_encoder(self.MODEL_NAME)
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
        # (doc_id, chunk 正文):同一篇文档的多个命中 chunk 全部保留 —— 这正是
        # doc 级口径丢掉的两个信号(命中次数、以及非首个 chunk 的正文)。
        chunk_inputs: list[tuple[str, str]] = []
        seen: set[str] = set()
        seen_chunks: set[str] = set()
        for doc in docs:
            doc_id = doc_id_of(doc)

            if self._filter and not self._filter.allow(doc_id):
                continue

            if not doc_id:
                continue

            chunk_id = (doc.metadata or {}).get("chunk_id")
            if chunk_id and str(chunk_id) not in seen_chunks:
                seen_chunks.add(str(chunk_id))
                chunk_inputs.append(
                    (
                        doc_id,
                        doc.page_content[:RERANK_CHUNK_CHARS]
                        if doc.page_content
                        else doc_id,
                    )
                )

            if doc_id not in seen:
                seen.add(doc_id)
                passage = (
                    doc.page_content[:RERANK_PASSAGE_CHARS]
                    if doc.page_content
                    else doc_id
                )
                doc_passages.append((doc_id, passage))

        # 这道早退保持不动:B3 组的历史数字是在它之下测出来的,动了 F 组
        # 就不再是「只换了打分粒度」的对照。
        if not doc_passages or len(doc_passages) <= top_k:
            return [dp[0] for dp in doc_passages], []

        # 粗排的文档级名次:既是降级兜底,也是 max-pool 的 tie-break 顺序。
        coarse_ids = [dp[0] for dp in doc_passages]

        if self._rerank_granularity == "chunk":
            scoring_inputs = chunk_inputs
        else:
            scoring_inputs = doc_passages

        if len(scoring_inputs) <= 1:
            return coarse_ids[:top_k], []

        pairs = [(query, passage) for _, passage in scoring_inputs]
        owner_docs = [doc_id for doc_id, _ in scoring_inputs]

        try:
            model = self._get_model()
            scores = model.predict(pairs, show_progress_bar=False)
        except Exception:
            import traceback
            from loguru import logger
            logger.warning(f"本地 Rerank 失败,降级为粗排: {traceback.format_exc()}")
            return coarse_ids[:top_k], []

        scored = [(doc_id, float(score)) for doc_id, score in zip(owner_docs, scores)]
        # doc 级走这里时每篇文档只有一条得分,max-pool 退化成单纯排序,
        # 且 tie-break 与原先「stable sort 保持粗排顺序」等价 —— 所以 B3
        # 的报告仍可原样复现,不需要为两个口径养两份排序代码。
        return max_pool_to_docs(scored, tie_break_order=coarse_ids)[:top_k], []


def build_local_reranked_retriever(
    vector_weight: float | None = None,
    bm25_weight: float | None = None,
    doc_id_filter: DocIdFilter | None = None,
    rerank_granularity: str = "chunk",
) -> Retriever:
    return LocalRerankedRetriever(
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        doc_id_filter=doc_id_filter,
        rerank_granularity=rerank_granularity,
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
            doc_id = doc_id_of(doc)
            if self._filter and not self._filter.allow(doc_id):
                continue
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                passage = (
                    doc.page_content[:RERANK_PASSAGE_CHARS]
                    if doc.page_content
                    else doc_id
                )
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

def _search_with_retry(
    retriever: Retriever,
    question: str,
    top_k: int,
    max_attempts: int = 3,
) -> tuple[list[str], list[str], int, bool]:
    """带退避重试的单次检索。

    Returns:
        (doc_ids, chunk_ids, 实际尝试次数, 是否最终失败)

    为什么需要这个:本项目的嵌入走 DashScope 公网 API,一次瞬时
    Connection error 就会让整轮 82 条样本作废(实际发生过两次,都是跑到
    一半死掉、20 分钟白费)。评测跑一轮要几分钟到几十分钟,让它被单次
    网络抖动清零是不可接受的。

    失败样本**不跳过**,而是记为空召回并计数。跳过会让分母变小,
    82 条里死了 30 条也能算出一个漂亮的 MRR;记空则会把指标拉低,
    再配合 meta 里的 failed_cases 让读报告的人一眼看出这轮数据脏了。
    """
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            docs, chunks = retriever.search(question, top_k=top_k)
            return docs, chunks, attempt, False
        except Exception:
            import traceback

            from loguru import logger

            if attempt >= max_attempts:
                logger.error(
                    f"检索连续 {max_attempts} 次失败,该样本记为空召回: "
                    f"{traceback.format_exc()}"
                )
                return [], [], attempt, True

            logger.warning(
                f"检索第 {attempt} 次失败,{delay:.0f}s 后重试: "
                f"{traceback.format_exc(limit=1)}"
            )
            time.sleep(delay)
            delay *= 2  # 指数退避:瞬时抖动 2s 够了,真断网时不要疯狂打接口
    return [], [], max_attempts, True


def run(
    retriever: Retriever,
    golden_items: list[dict],
    top_k: int = 10,
    max_attempts: int = 3,
) -> tuple[AggregateReport, list[RetrievalCase], dict]:
    """跑完整评测,返回 (聚合报告, 逐样本明细, 运行元信息)。

    第三个返回值是**新增**的。为什么必须有它:

    multi-query 相比单查询多了一次 LLM 调用和 N 倍检索次数。如果报告里
    只有 Hit@K 上升了几个点,那是一份缺了一半的报告 —— 决定要不要上线
    的人必须同时看到「涨了多少」和「贵了多少」。只报收益不报成本,
    等于替对方把权衡做掉了。

    这里的耗时是**单进程串行**的墙钟时间,不等于线上耗时:线上 fan-out
    是 LangGraph Send 并行的(graph.py:19),N 个子查询取 max 而非 sum。
    所以这个数字应读作「成本的上界」,不是「用户会等这么久」。
    """
    cases: list[RetrievalCase] = []
    # 只收**一次就成功**的样本耗时。重试里含 2s/4s 退避 sleep,混进来会让
    # s/case 凭空变大,而这一列是要跨组对比的 —— 一次网络抖动就能让某组
    # 看起来「慢了 30%」,那是测量噪声冒充结论。
    latencies: list[float] = []
    retried_case_ids: list[str] = []
    failed_case_ids: list[str] = []

    for item in golden_items:
        started = time.perf_counter()
        retrieved_docs, retrieved_chunks, attempts, failed = _search_with_retry(
            retriever, item["question"], top_k=top_k, max_attempts=max_attempts
        )
        elapsed = time.perf_counter() - started

        if attempts == 1:
            latencies.append(elapsed)
        else:
            retried_case_ids.append(str(item["id"]))
        if failed:
            failed_case_ids.append(str(item["id"]))

        cases.append(
            RetrievalCase(
                case_id=item["id"],
                retrieved_doc_ids=retrieved_docs,
                expected_doc_ids=item.get("expected_doc_ids", []),
                retrieved_chunk_ids=retrieved_chunks,
                expected_chunk_ids=item.get("expected_chunk_ids", []),
                tags=item.get("tags", []),
                # 只有 multi-query 检索器有这个属性;单查询检索器留空。
                # 用 getattr 而不是 isinstance 判断,是为了不让 run() 认识
                # 任何具体检索器类型 —— Retriever 协议只约定了 search()。
                sub_queries=list(getattr(retriever, "last_sub_queries", []) or []),
            )
        )

    report = aggregate(cases, ks=(1, 3, 5, 10))

    meta = {
        "total_seconds": round(sum(latencies), 3),
        "avg_seconds_per_case": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "max_seconds_per_case": round(max(latencies), 3) if latencies else 0.0,
        # 耗时统计的样本数。它小于 total_cases 就说明有样本重试过,
        # 上面三个耗时数字是在这个子集上算的 —— 不写出来的话,读报告的人
        # 会以为分母就是全部样本。
        "timed_cases": len(latencies),
        # 重试过、以及三次都失败的样本 id。这两个列表哪怕都是空的也要写进
        # 报告:「本轮没有网络失败」本身就是这份数据可信的前提之一。
        # 非空时,failed 样本被记为空召回,指标是被拉低过的,不能与
        # 干净轮次直接比较。
        "retried_cases": retried_case_ids,
        "failed_cases": failed_case_ids,
        "failed_case_count": len(failed_case_ids),
        # 平均子查询数:改写实际展开了几路。等于 1 说明**全部退化成单查询**,
        # 此时报告标题即使写着 multi-query 也测的是单查询。
        "avg_sub_queries": (
            round(sum(len(c.sub_queries) for c in cases) / len(cases), 2) if cases else 0.0
        ),
        # 改写降级样本数。这个字段哪怕是 0 也要写进报告 ——
        # 「本次没有降级」本身就是报告可信度的一部分。
        "rewrite_degraded_cases": int(getattr(retriever, "rewrite_degraded", 0) or 0),
    }
    return report, cases, meta


def save_report(
    report: AggregateReport,
    cases: list[RetrievalCase],
    tag: str,
    meta: dict | None = None,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    out_path = REPORTS_DIR / f"{tag}-{timestamp}.json"
    payload = {
        "tag": tag,
        "timestamp": timestamp,
        "run_meta": meta or {},
        "summary": asdict(report),
        "cases": [asdict(c) for c in cases],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def print_report(report: AggregateReport, tag: str, meta: dict | None = None) -> None:
    print(f"\n=== Eval Report [{tag}] ===")
    print(f"Total cases : {report.total}")
    print(f"MRR         : {report.mrr:.4f}")
    print("Hit@K       :", {k: f"{v:.4f}" for k, v in report.hit_at_k.items()})
    print("Recall@K    :", {k: f"{v:.4f}" for k, v in report.recall_at_k.items()})

    if meta:
        print(f"Avg latency : {meta.get('avg_seconds_per_case', 0):.3f}s/case "
              f"(max {meta.get('max_seconds_per_case', 0):.3f}s, "
              f"total {meta.get('total_seconds', 0):.1f}s)")
        avg_subs = meta.get("avg_sub_queries", 0)
        if avg_subs:
            print(f"Sub-queries : {avg_subs:.2f} avg/case")
        degraded = meta.get("rewrite_degraded_cases", 0)
        if degraded:
            # 这条必须显眼:降级过的报告不能当 multi-query 的成绩来读。
            print(f"[WARN] {degraded} 条样本改写降级为单查询,本报告不能完整代表多查询效果")

        # 重试与失败必须在**终端**露出,不能只躺在 JSON 里。
        # 一份被网络污染的报告和一份干净的报告,指标格式长得一模一样;
        # 如果读的人得先去翻文件才知道有没有脏,那多数时候他不会翻。
        retried = meta.get("retried_cases") or []
        if retried:
            print(f"[INFO] {len(retried)} 条样本重试后成功: {', '.join(retried[:8])}"
                  f"{' ...' if len(retried) > 8 else ''}")
        failed = meta.get("failed_cases") or []
        if failed:
            print(f"[WARN] {len(failed)} 条样本重试耗尽仍失败,已记为空召回 —— "
                  f"本报告的 MRR/Hit@K 被拉低了,不可与干净的报告直接对比")
            print(f"       失败样本: {', '.join(failed[:8])}"
                  f"{' ...' if len(failed) > 8 else ''}")

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
    for k in sorted({*sa['recall_at_k'].keys(), *sb['recall_at_k'].keys()}, key=int):
        va = sa['recall_at_k'].get(str(k), 0)
        vb = sb['recall_at_k'].get(str(k), 0)
        print(f"{'Recall@' + str(k):<14}{va:>10.4f}{vb:>10.4f}{vb - va:>+10.4f}")

    # 成本对比。老报告没有 run_meta,如实显示 n/a 而不是填 0 ——
    # 0 秒会被读成「不花时间」,而真相是「这个数当时没测」。
    ma = a.get("run_meta") or {}
    mb = b.get("run_meta") or {}
    fa = f"{ma['avg_seconds_per_case']:.3f}" if "avg_seconds_per_case" in ma else "n/a"
    fb = f"{mb['avg_seconds_per_case']:.3f}" if "avg_seconds_per_case" in mb else "n/a"
    print(f"{'Latency/case':<14}{fa:>10}{fb:>10}{'':>10}")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG offline eval")
    parser.add_argument("--tag", default="baseline", help="本次实验名")
    parser.add_argument(
        "--golden-set",
        default=str(GOLDEN_SET_PATH),
        help="评测集 JSONL 路径，默认使用 tests/eval/golden_set.jsonl",
    )
    parser.add_argument("--top-k", type=int, default=10, help="检索 Top-K")
    parser.add_argument("--retriever", default="cheat",
                        choices=["cheat", "vector", "rerank", "local-rerank",
                                 "multi-query", "multi-query-rerank"],
                        help="检索器选择;multi-query* 与线上 chat_v2 链路同构")
    parser.add_argument("--rrf-k", type=int, default=RRF_K,
                        help=f"multi-query 的 RRF 平滑常数,默认 {RRF_K}")
    parser.add_argument("--rerank-granularity", default="chunk",
                        choices=["chunk", "doc"],
                        help="rerank 打分粒度: chunk=逐 chunk 打分后 max-pool 到文档级"
                             "(默认); doc=每篇只取首个命中 chunk 前 500 字"
                             "(旧口径,用于复现历史报告)")
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

    golden_items = load_golden_set(Path(args.golden_set))
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
            rerank_granularity=args.rerank_granularity,
        )
        # 必须打印:B3(doc 级)与 F(chunk 级)只差这一个开关,报告里
        # 没有它的话两份数字看起来就是同一个配置跑出了不同结果。
        print(f"[LOCAL-RERANK] granularity={args.rerank_granularity}")
    elif args.retriever in ("multi-query", "multi-query-rerank"):
        do_rerank = args.retriever == "multi-query-rerank"
        retriever = build_multi_query_retriever(
            vector_weight=args.vector_weight,
            bm25_weight=args.bm25_weight,
            doc_id_filter=doc_filter,
            rrf_k=args.rrf_k,
            rerank=do_rerank,
            rerank_granularity=args.rerank_granularity,
        )
        # 粒度只在真的 rerank 时才有意义,不 rerank 时打印它会误导读者
        # 以为这份报告与粒度有关。
        gran = f" granularity={args.rerank_granularity}" if do_rerank else ""
        print(f"[MULTI-QUERY] rrf_k={args.rrf_k} "
              f"rerank={do_rerank}{gran} "
              f"(每条样本会多调一次 LLM 做改写)")
    else:
        retriever = build_vector_retriever(
            vector_weight=args.vector_weight,
            bm25_weight=args.bm25_weight,
            doc_id_filter=doc_filter,
        )

    report, cases, meta = run(retriever, golden_items, top_k=args.top_k)
    print_report(report, args.tag, meta)
    out_path = save_report(report, cases, args.tag, meta)
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
