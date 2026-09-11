"""AIOps 首响报告质量评估器。

现有 tests/eval/ 已覆盖**检索类**指标（Hit@K / MRR / Recall@K，见 metrics.py）。
本模块补的是它没覆盖的维度 —— **首响报告本身的质量**（ADR-007）：

1. SOP 命中率（sop_hit）：报告证据里是否含期望的 SOP 关键词。
2. 报告要点覆盖率（point_coverage）：报告文本是否覆盖期望排查要点。
3. 幻觉控制（hallucination_controlled）：没有已验证事实证据时，
   是否老实把不确定结论放进 pending_confirmations，而不是硬给确定性根因。

设计原则（对齐 metrics.py）：
- 打分函数都是纯函数，输入「报告 + 期望」，不耦合 LLM / 检索实现，易单测。
- 评估集就用 tests/fixtures/aiops_alerts/ 里的三条告警（已带 expected_* 字段）。

关联 ADR：docs/adr/007-aiops-first-response-eval.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from app.models.aiops_report import EvidenceType, FirstResponseReport


_FIXTURES = Path(__file__).parent.parent / "fixtures" / "aiops_alerts"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ReportCaseScore:
    """单条告警的首响报告评分。"""

    case_id: str
    sop_hit: bool
    point_coverage: float
    hallucination_controlled: bool


@dataclass
class ReportEvalReport:
    """整个评估集的聚合结果。"""

    total: int
    sop_hit_rate: float
    avg_point_coverage: float
    hallucination_control_rate: float
    cases: list[ReportCaseScore] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 单指标（纯函数）
# ---------------------------------------------------------------------------

def score_sop_hit(report: FirstResponseReport, expected_keywords: list[str]) -> bool:
    """报告的证据里是否命中任一期望 SOP 关键词。

    在所有证据的 content / source_title / excerpt 里找关键词，命中一个即算 hit。
    expected_keywords 为空视为「不要求命中」，返回 True。
    """
    if not expected_keywords:
        return True
    haystack = " ".join(
        f"{ev.content} {ev.source_title or ''} {ev.excerpt or ''}"
        for ev in report.evidence
    )
    return any(kw in haystack for kw in expected_keywords)


def score_point_coverage(
    report: FirstResponseReport,
    expected_points: list[str],
) -> float:
    """报告全文覆盖了多少比例的期望排查要点。

    把报告渲染成 markdown 全文，逐个要点做子串匹配。
    expected_points 为空返回 1.0（无要求即满分）。
    """
    if not expected_points:
        return 1.0
    text = report.to_markdown()
    hits = sum(1 for point in expected_points if point in text)
    return hits / len(expected_points)


def score_hallucination_controlled(report: FirstResponseReport) -> bool:
    """幻觉控制：没有已验证事实时，不能硬给确定性根因。

    规则：
    - 若报告含至少一条 VERIFIED_FACT 证据 → 允许给根因假设，视为受控。
    - 若一条已验证事实都没有，却给出了 root_cause_hypotheses，
      同时 pending_confirmations 为空 → 判定为「无据编造」，不受控。
    """
    has_verified = any(
        ev.type == EvidenceType.VERIFIED_FACT for ev in report.evidence
    )
    if has_verified:
        return True
    # 无已验证事实：只要没有「有根因却无待确认」的情况，就算受控
    if report.root_cause_hypotheses and not report.pending_confirmations:
        return False
    return True


def score_case(
    case_id: str,
    report: FirstResponseReport,
    expected_keywords: list[str],
    expected_points: list[str],
) -> ReportCaseScore:
    """对单条告警的报告算三个维度。"""
    return ReportCaseScore(
        case_id=case_id,
        sop_hit=score_sop_hit(report, expected_keywords),
        point_coverage=score_point_coverage(report, expected_points),
        hallucination_controlled=score_hallucination_controlled(report),
    )


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------

def aggregate(cases: list[ReportCaseScore]) -> ReportEvalReport:
    """把多条报告评分聚合成总报告。"""
    if not cases:
        return ReportEvalReport(
            total=0,
            sop_hit_rate=0.0,
            avg_point_coverage=0.0,
            hallucination_control_rate=0.0,
        )
    return ReportEvalReport(
        total=len(cases),
        sop_hit_rate=mean(1.0 if c.sop_hit else 0.0 for c in cases),
        avg_point_coverage=mean(c.point_coverage for c in cases),
        hallucination_control_rate=mean(
            1.0 if c.hallucination_controlled else 0.0 for c in cases
        ),
        cases=cases,
    )


# ---------------------------------------------------------------------------
# 评估集加载
# ---------------------------------------------------------------------------

def load_alarm_cases(fixtures_dir: Path = _FIXTURES) -> list[dict]:
    """加载 tests/fixtures/aiops_alerts/ 里的告警评估集。"""
    cases: list[dict] = []
    for path in sorted(fixtures_dir.glob("*.json")):
        cases.append(json.loads(path.read_text(encoding="utf-8")))
    return cases


def format_report(report: ReportEvalReport) -> str:
    """把聚合结果格式化成人可读的一段文本。"""
    lines = [
        "=== AIOps 首响报告质量评估 ===",
        f"样本数:            {report.total}",
        f"SOP 命中率:        {report.sop_hit_rate:.0%}",
        f"报告要点覆盖率:    {report.avg_point_coverage:.0%}",
        f"幻觉受控率:        {report.hallucination_control_rate:.0%}",
        "",
        "逐条:",
    ]
    for c in report.cases:
        lines.append(
            f"  [{c.case_id}] SOP命中={c.sop_hit} "
            f"要点覆盖={c.point_coverage:.0%} 幻觉受控={c.hallucination_controlled}"
        )
    return "\n".join(lines)
