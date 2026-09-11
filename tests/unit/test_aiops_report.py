"""AIOps 首响报告 schema 与解析的单元测试。

覆盖：
- 正常 JSON 解析成结构化报告
- ```json 代码块包裹的 JSON
- 纯文本 / 非法 JSON / 缺必填字段 → 降级为纯文本报告（永不抛异常）
- Markdown 渲染区分「已验证事实 / 推断」
"""

import json

from app.models.aiops_report import (
    Evidence,
    EvidenceSource,
    EvidenceType,
    FirstResponseReport,
    RecommendedCheck,
    parse_report,
)


def _valid_payload() -> dict:
    return {
        "alert_summary": "order-api CPU 持续超过 90%",
        "current_judgment": "疑似流量突增导致 CPU 打满",
        "severity_assessment": "高",
        "recommended_checks": [
            {
                "order": 1,
                "action": "查看 order-api 最近 10 分钟 QPS",
                "reason": "确认是否流量突增",
                "command_hint": "curl /metrics | grep qps",
                "requires_human": False,
            }
        ],
        "evidence": [
            {
                "type": "verified_fact",
                "source": "metrics",
                "content": "CPU 使用率 95%",
                "source_title": "monitor 工具",
            },
            {
                "type": "model_inference",
                "source": "model",
                "content": "可能是慢查询堆积",
            },
        ],
        "root_cause_hypotheses": ["流量突增", "慢查询堆积"],
        "pending_confirmations": ["确认是否有大促活动"],
        "risk_notes": "若持续可能雪崩，建议扩容",
    }


def test_parse_plain_json():
    report = parse_report(json.dumps(_valid_payload(), ensure_ascii=False))
    assert report.is_degraded is False
    assert report.alert_summary.startswith("order-api")
    assert len(report.recommended_checks) == 1
    assert report.recommended_checks[0].order == 1
    assert len(report.evidence) == 2
    assert report.evidence[0].type == EvidenceType.VERIFIED_FACT
    assert report.evidence[1].type == EvidenceType.MODEL_INFERENCE


def test_parse_json_in_code_fence():
    raw = f"这是模型的思考...\n```json\n{json.dumps(_valid_payload(), ensure_ascii=False)}\n```\n完成"
    report = parse_report(raw)
    assert report.is_degraded is False
    assert report.alert_summary.startswith("order-api")


def test_parse_json_embedded_in_text():
    raw = f"前言废话 {json.dumps(_valid_payload(), ensure_ascii=False)} 后记废话"
    report = parse_report(raw)
    assert report.is_degraded is False
    assert len(report.evidence) == 2


def test_plain_text_degrades():
    report = parse_report("CPU 有点高，建议看看日志。")
    assert report.is_degraded is True
    assert report.raw_text == "CPU 有点高，建议看看日志。"
    assert "降级" in report.to_markdown()


def test_invalid_json_degrades():
    report = parse_report("{ 这不是合法 json ")
    assert report.is_degraded is True


def test_missing_required_field_degrades():
    # 缺 current_judgment（必填），应降级而非抛异常
    bad = {"alert_summary": "只有摘要"}
    report = parse_report(json.dumps(bad, ensure_ascii=False))
    assert report.is_degraded is True


def test_empty_text_degrades():
    report = parse_report("")
    assert report.is_degraded is True
    assert report.raw_text == "（模型返回为空）"


def test_to_markdown_distinguishes_evidence():
    report = FirstResponseReport(
        alert_summary="摘要",
        current_judgment="判断",
        evidence=[
            Evidence(
                type=EvidenceType.VERIFIED_FACT,
                source=EvidenceSource.SOP,
                content="SOP 说要重启",
                source_title="CPU排查手册",
            ),
            Evidence(
                type=EvidenceType.MODEL_INFERENCE,
                source=EvidenceSource.MODEL,
                content="可能是内存泄漏",
            ),
        ],
    )
    md = report.to_markdown()
    assert "✅ 已验证" in md
    assert "🤔 推断" in md
    assert "CPU排查手册" in md


def test_to_markdown_sorts_checks_by_order():
    report = FirstResponseReport(
        alert_summary="摘要",
        current_judgment="判断",
        recommended_checks=[
            RecommendedCheck(order=2, action="第二步", reason="r2"),
            RecommendedCheck(order=1, action="第一步", reason="r1"),
        ],
    )
    md = report.to_markdown()
    assert md.index("第一步") < md.index("第二步")


def test_model_inference_with_human_source_label_stays_structured():
    payload = _valid_payload()
    payload["evidence"] = [
        {
            "type": "model_inference",
            "source": "SOP 3: cpu_high_usage.md",
            "content": "可能存在流量突增",
        }
    ]

    report = parse_report(json.dumps(payload, ensure_ascii=False))

    assert report.is_degraded is False
    assert report.evidence[0].type == EvidenceType.MODEL_INFERENCE
    assert report.evidence[0].source == EvidenceSource.MODEL
