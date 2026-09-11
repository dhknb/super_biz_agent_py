"""AIOps 首响报告质量评估器的单元测试。

覆盖三个评分维度的纯函数逻辑 + 聚合：
- score_sop_hit：命中 / 未命中 / 空期望
- score_point_coverage：全覆盖 / 部分 / 空期望
- score_hallucination_controlled：有事实 / 无事实但有待确认 / 无事实且硬给根因
- aggregate：空集 / 多条聚合
"""

from app.models.aiops_report import (
    Evidence,
    EvidenceSource,
    EvidenceType,
    FirstResponseReport,
    RecommendedCheck,
)
from tests.eval.aiops_first_response_eval import (
    aggregate,
    score_case,
    score_hallucination_controlled,
    score_point_coverage,
    score_sop_hit,
)


def _report_with_sop() -> FirstResponseReport:
    return FirstResponseReport(
        alert_summary="CPU 高",
        current_judgment="疑似进程死循环",
        evidence=[
            Evidence(
                type=EvidenceType.VERIFIED_FACT,
                source=EvidenceSource.SOP,
                content="命中 SOP",
                source_title="cpu_high_usage.md · CPU 排查",
                excerpt="先用 top 定位高 CPU 进程",
            )
        ],
        recommended_checks=[
            RecommendedCheck(order=1, action="用 top 看进程", reason="定位")
        ],
        root_cause_hypotheses=["进程死循环"],
    )


# ── score_sop_hit ──────────────────────────────────────────────
def test_sop_hit_matches_keyword_in_excerpt():
    report = _report_with_sop()
    assert score_sop_hit(report, ["top"]) is True


def test_sop_hit_matches_keyword_in_title():
    report = _report_with_sop()
    assert score_sop_hit(report, ["CPU"]) is True


def test_sop_miss_when_no_keyword():
    report = _report_with_sop()
    assert score_sop_hit(report, ["磁盘", "内存"]) is False


def test_sop_hit_empty_expected_is_true():
    report = _report_with_sop()
    assert score_sop_hit(report, []) is True


# ── score_point_coverage ───────────────────────────────────────
def test_point_coverage_full():
    report = _report_with_sop()
    # markdown 里含 "CPU"、"进程"、"top"
    cov = score_point_coverage(report, ["CPU", "进程", "top"])
    assert cov == 1.0


def test_point_coverage_partial():
    report = _report_with_sop()
    cov = score_point_coverage(report, ["CPU", "内存不存在的词"])
    assert cov == 0.5


def test_point_coverage_empty_is_full():
    report = _report_with_sop()
    assert score_point_coverage(report, []) == 1.0


# ── score_hallucination_controlled ─────────────────────────────
def test_hallucination_controlled_with_verified_fact():
    report = _report_with_sop()  # 有 VERIFIED_FACT
    assert score_hallucination_controlled(report) is True


def test_hallucination_uncontrolled_when_no_fact_but_root_cause():
    report = FirstResponseReport(
        alert_summary="x",
        current_judgment="y",
        evidence=[
            Evidence(
                type=EvidenceType.MODEL_INFERENCE,
                source=EvidenceSource.MODEL,
                content="瞎猜的",
            )
        ],
        root_cause_hypotheses=["硬给的根因"],
        pending_confirmations=[],  # 没有待确认 → 无据编造
    )
    assert score_hallucination_controlled(report) is False


def test_hallucination_controlled_when_no_fact_but_has_pending():
    report = FirstResponseReport(
        alert_summary="x",
        current_judgment="y",
        evidence=[],
        root_cause_hypotheses=["假设"],
        pending_confirmations=["需人工确认"],  # 老实标注 → 受控
    )
    assert score_hallucination_controlled(report) is True


def test_hallucination_controlled_when_no_root_cause():
    report = FirstResponseReport(
        alert_summary="x",
        current_judgment="y",
        evidence=[],
        root_cause_hypotheses=[],
    )
    assert score_hallucination_controlled(report) is True


# ── score_case + aggregate ─────────────────────────────────────
def test_score_case_combines_three_dims():
    report = _report_with_sop()
    score = score_case("q-cpu", report, ["top"], ["CPU", "进程"])
    assert score.case_id == "q-cpu"
    assert score.sop_hit is True
    assert score.point_coverage == 1.0
    assert score.hallucination_controlled is True


def test_aggregate_empty():
    report = aggregate([])
    assert report.total == 0
    assert report.sop_hit_rate == 0.0


def test_aggregate_multiple_cases():
    r = _report_with_sop()
    cases = [
        score_case("a", r, ["top"], ["CPU"]),            # hit, cov=1.0, controlled
        score_case("b", r, ["磁盘"], ["不存在"]),          # miss, cov=0.0, controlled
    ]
    agg = aggregate(cases)
    assert agg.total == 2
    assert agg.sop_hit_rate == 0.5
    assert agg.avg_point_coverage == 0.5
    assert agg.hallucination_control_rate == 1.0
