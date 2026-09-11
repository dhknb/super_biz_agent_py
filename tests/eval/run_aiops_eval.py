"""AIOps 首响评估 CLI — 一条命令跑出质量基线。

用法（在项目根目录）：
    wsl .venv/bin/python -m tests.eval.run_aiops_eval

流程：
    tests/fixtures/aiops_alerts/*.json（告警 + expected_*）
        ↓
    normalize_alarm → FirstResponseService.analyze（注入假 LLM + 假 SOP，离线可跑）
        ↓
    aiops_first_response_eval.score_case → aggregate
        ↓
    打印 SOP命中率 / 要点覆盖率 / 幻觉受控率

设计：默认**全离线**（假 LLM + 假 SOP 检索），产出一个可复现的结构基线，
不烧 token、不连 Milvus。目的是验证「评估管道 + 打分逻辑」本身正确，
以及在「检索理想命中」假设下报告结构能达到的上限。
真实质量评估（接 Milvus + 真 LLM）是后续工作，接口已通过依赖注入预留。
"""

from __future__ import annotations

import asyncio
import json

from app.core.errors import RetrievalStatus
from app.models.aiops import AlarmEvent, normalize_alarm
from app.models.aiops_report import (
    Evidence,
    EvidenceSource,
    EvidenceType,
)
from app.services.first_response_service import FirstResponseService
from tests.eval.aiops_first_response_eval import (
    aggregate,
    format_report,
    load_alarm_cases,
    score_case,
)


class _StubLLM:
    """离线假 LLM：产出一个结构完整、含期望要点的报告。

    把期望排查要点回填进 recommended_checks，模拟「检索理想命中」下
    模型应该产出的结构化报告，用于跑通评估管道并给出结构基线。
    """

    def __init__(self, expected_points: list[str]):
        self._points = expected_points

    async def ainvoke(self, messages):
        checks = [
            {"order": i + 1, "action": f"检查：{p}", "reason": "来自期望排查要点"}
            for i, p in enumerate(self._points)
        ] or [{"order": 1, "action": "初步检查", "reason": "基线"}]
        payload = {
            "alert_summary": "离线基线报告",
            "current_judgment": "基于 SOP 与告警的初步判断",
            "recommended_checks": checks,
            "evidence": [{"type": "model_inference", "source": "model", "content": "基线推断"}],
            "root_cause_hypotheses": [],
            "pending_confirmations": ["离线基线，待人工确认"],
        }

        class _R:
            content = json.dumps(payload, ensure_ascii=False)

        return _R()


class _StubSop:
    """离线假 SOP 检索：按期望关键词造一条 VERIFIED_FACT 证据。

    模拟「检索理想命中」——让 SOP 命中率评估在离线也有意义。
    """

    def __init__(self, keywords: list[str]):
        self._keywords = keywords

    def retrieve_sop_evidence(
        self, alarm: AlarmEvent
    ) -> tuple[list[Evidence], RetrievalStatus]:
        if not self._keywords:
            # 离线基线里「没关键词」表示这条用例不期望命中 SOP，
            # 属于检索正常但知识库无内容 —— 是 EMPTY，不是 FAILED。
            return [], RetrievalStatus.EMPTY
        return [
            Evidence(
                type=EvidenceType.VERIFIED_FACT,
                source=EvidenceSource.SOP,
                content="命中 SOP：" + " / ".join(self._keywords),
                source_title=f"{alarm.alert_name}_sop.md",
                excerpt=" ".join(self._keywords),
            )
        ], RetrievalStatus.OK


async def _run() -> None:
    cases = load_alarm_cases()
    scores = []

    for case in cases:
        payload = case.get("payload", case)
        source = case.get("source", "manual")
        keywords = case.get("expected_sop_keywords", [])
        points = case.get("expected_report_points", [])

        alarm = normalize_alarm(payload, source)
        service = FirstResponseService(
            llm=_StubLLM(points),
            sop_service=_StubSop(keywords),
        )
        report = await service.analyze(alarm)
        scores.append(
            score_case(alarm.alert_name, report, keywords, points)
        )

    print(format_report(aggregate(scores)))


if __name__ == "__main__":
    asyncio.run(_run())
