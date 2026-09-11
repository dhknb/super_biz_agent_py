"""Retrieval & generation metrics for RAG evaluation.

仅实现"零依赖"的检索类指标(Hit@K / MRR / Recall@K),
生成类指标(Faithfulness / Answer Relevance)留 ragas 接口预留位。

设计原则:
- 单个函数只算一个指标,纯函数,易于单测
- 输入统一是"已检索结果 + 期望项",不耦合具体检索实现
- aggregate() 把单样本指标聚合成总报告
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class RetrievalCase:
    """单条评测样本在某次检索下的结果。"""

    case_id: str
    retrieved_doc_ids: list[str]
    expected_doc_ids: list[str]
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    expected_chunk_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    # 多查询检索器实际用到的子查询,单查询检索器留空。
    #
    # 只进报告、不参与任何指标计算 —— 但没有它就无法解释某条样本
    # 为什么变好或变坏:报告里只剩一个数字,而数字看不出改写是否合理。
    # 一次 bad case review 的第一个问题永远是「它到底查了什么」。
    sub_queries: list[str] = field(default_factory=list)


@dataclass
class CaseMetrics:
    """单样本的检索指标。"""

    case_id: str
    hit_at_k: dict[int, bool]
    recall_at_k: dict[int, float]
    reciprocal_rank: float


@dataclass
class AggregateReport:
    """整个评测集的聚合报告。"""

    total: int
    hit_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    mrr: float
    by_tag: dict[str, "AggregateReport"] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 单样本指标
# ---------------------------------------------------------------------------

def _first_hit_rank(retrieved: list[str], expected: Iterable[str]) -> int | None:
    """期望项在 retrieved 里第一次出现的位置(1-indexed),没命中返回 None。"""
    expected_set = set(expected)
    for idx, item in enumerate(retrieved, start=1):
        if item in expected_set:
            return idx
    return None


def hit_at_k(retrieved: list[str], expected: Iterable[str], k: int) -> bool:
    """Top-K 里是否至少命中一项。"""
    expected_set = set(expected)
    return any(item in expected_set for item in retrieved[:k])


def recall_at_k(retrieved: list[str], expected: Iterable[str], k: int) -> float:
    """Top-K 里命中的期望项数 / 期望项总数。"""
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    hits = sum(1 for item in retrieved[:k] if item in expected_set)
    return hits / len(expected_set)


def reciprocal_rank(retrieved: list[str], expected: Iterable[str]) -> float:
    """期望项第一次出现位置的倒数。没命中算 0。"""
    rank = _first_hit_rank(retrieved, expected)
    return 1.0 / rank if rank else 0.0


def compute_case(case: RetrievalCase, ks: Iterable[int] = (1, 3, 5, 10)) -> CaseMetrics:
    """对单样本算所有检索指标。"""
    ks = tuple(ks)

    # 优先用 chunk 级评测(更精细),退化到 doc 级
    if case.expected_chunk_ids and case.retrieved_chunk_ids:
        retrieved = case.retrieved_chunk_ids
        expected = case.expected_chunk_ids
    else:
        retrieved = case.retrieved_doc_ids
        expected = case.expected_doc_ids

    return CaseMetrics(
        case_id=case.case_id,
        hit_at_k={k: hit_at_k(retrieved, expected, k) for k in ks},
        recall_at_k={k: recall_at_k(retrieved, expected, k) for k in ks},
        reciprocal_rank=reciprocal_rank(retrieved, expected),
    )


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------

def aggregate(
    cases: list[RetrievalCase],
    ks: Iterable[int] = (1, 3, 5, 10),
) -> AggregateReport:
    """把多条样本聚合成总报告 + 按 tag 切片。"""
    return _aggregate(cases, tuple(ks), include_by_tag=True)


def _aggregate(
    cases: list[RetrievalCase],
    ks: tuple[int, ...],
    include_by_tag: bool,
) -> AggregateReport:
    """内部递归实现,by_tag 切片只在外层算一次,避免无限递归。"""
    case_metrics = [compute_case(c, ks) for c in cases]

    if not case_metrics:
        return AggregateReport(
            total=0,
            hit_at_k={k: 0.0 for k in ks},
            recall_at_k={k: 0.0 for k in ks},
            mrr=0.0,
        )

    report = AggregateReport(
        total=len(case_metrics),
        hit_at_k={
            k: mean(1.0 if cm.hit_at_k[k] else 0.0 for cm in case_metrics) for k in ks
        },
        recall_at_k={
            k: mean(cm.recall_at_k[k] for cm in case_metrics) for k in ks
        },
        mrr=mean(cm.reciprocal_rank for cm in case_metrics),
    )

    if not include_by_tag:
        return report

    # 按 tag 切片,但切片内部不再递归算 by_tag
    tag_to_cases: dict[str, list[RetrievalCase]] = {}
    for c in cases:
        for tag in c.tags:
            tag_to_cases.setdefault(tag, []).append(c)

    for tag, sub_cases in tag_to_cases.items():
        report.by_tag[tag] = _aggregate(sub_cases, ks=ks, include_by_tag=False)

    return report


# ---------------------------------------------------------------------------
# 生成类指标(预留位,后续接 ragas)
# ---------------------------------------------------------------------------

def faithfulness_placeholder(*_args, **_kwargs) -> float:
    """TODO(week 2): 接入 ragas.metrics.faithfulness。

    思路:用 LLM 判断 generated_answer 里每个 claim 是否都能由 retrieved_chunks 支撑,
    返回 [支撑的 claim 数 / 总 claim 数]。
    """
    raise NotImplementedError("faithfulness 尚未接入,见 README week 2 计划")


def answer_relevance_placeholder(*_args, **_kwargs) -> float:
    """TODO(week 2): 接入 ragas.metrics.answer_relevancy。"""
    raise NotImplementedError("answer_relevance 尚未接入,见 README week 2 计划")
