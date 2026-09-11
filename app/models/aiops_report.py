"""AIOps 首响分析报告的结构化 schema。

设计目标（对齐 docs/adr/000-scope-first-response.md 与产品方向文档）：
- 让模型输出**稳定的结构化 JSON**，而不是一大段散文，方便前端渲染、程序读取、审计。
- 每条证据明确区分「已验证事实」和「模型推断」——这是 Day5 SOP 检索
  接入后「证据溯源」的地基，也是本项目区别于通用聊天机器人的核心。
- 解析失败必须能**降级为纯文本报告**，绝不让整条诊断链路崩掉。

关联 ADR：docs/adr/003-structured-first-response-report.md
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.core.errors import DegradeReason


class EvidenceType(StrEnum):
    """证据类型 —— 首响报告可信度的核心区分。

    - VERIFIED_FACT: 来自工具查询或知识库的**已验证事实**（监控指标、日志、SOP 片段）。
      这类证据可以带来源（工具名 / SOP 标题），可被复核。
    - MODEL_INFERENCE: 模型基于常识或已有证据做出的**推断**。
      不能当作确定结论，报告里必须与事实区分开。
    """

    VERIFIED_FACT = "verified_fact"
    MODEL_INFERENCE = "model_inference"


class EvidenceSource(StrEnum):
    """证据来源渠道。"""

    METRICS = "metrics"  # 监控指标（MCP monitor 工具）
    LOGS = "logs"  # 日志（MCP cls 工具）
    SOP = "sop"  # 知识库 SOP / 历史文档（RAG）
    MODEL = "model"  # 模型推断
    UNKNOWN = "unknown"


class Evidence(BaseModel):
    """一条证据。

    Day5 SOP 检索接入后，verified_fact 类型的证据会带上 source_title
    （如 SOP 文档标题）和 excerpt（命中片段），实现「证据溯源」。
    """

    type: EvidenceType = Field(description="证据类型：已验证事实 / 模型推断")
    source: EvidenceSource = Field(
        default=EvidenceSource.UNKNOWN, description="证据来源渠道"
    )
    content: str = Field(description="证据内容（一句话说清这条证据是什么）")
    source_title: Optional[str] = Field(
        default=None, description="来源标题，如 SOP 文档名 / 工具名"
    )
    excerpt: Optional[str] = Field(default=None, description="原文命中片段（可选）")

    @model_validator(mode="before")
    @classmethod
    def normalize_model_evidence_source(cls, value: Any) -> Any:
        """兼容模型把来源写成人类标签的情况。

        模型推断只能标记为 ``model``。真实 SOP 证据由检索代码注入，
        因而模型写出的 ``SOP 3: ...`` 之类标签不能当作可溯源事实，
        统一归为模型推断，避免一个无效枚举值导致整份报告降级。
        """
        if not isinstance(value, dict):
            return value

        if value.get("type") != EvidenceType.MODEL_INFERENCE:
            return value

        normalized = dict(value)
        normalized["source"] = EvidenceSource.MODEL
        return normalized


class RecommendedCheck(BaseModel):
    """一条建议排障检查项。

    注意（ADR-000 边界）：第一版只产出「检查 / 观察」类建议，
    不产出高风险变更命令。requires_human 标记那些需要人工判断的项。
    """

    order: int = Field(description="排障顺序，从 1 开始")
    action: str = Field(description="建议做什么，如「查看 order-api 最近 10 分钟错误日志」")
    reason: str = Field(description="为什么建议这么做")
    command_hint: Optional[str] = Field(
        default=None, description="可选的只读检查命令提示（不自动执行）"
    )
    requires_human: bool = Field(default=False, description="是否需要人工确认后再操作")


class FirstResponseReport(BaseModel):
    """结构化首响分析报告 —— 诊断链路的最终产物。

    字段对齐产品方向文档 5.3 节推荐结构，但用结构化对象承载，
    既能给人看（可渲染成 Markdown），也能被程序读取（前端 / 评测脚本）。
    """

    alert_summary: str = Field(description="告警摘要：一句话说清发生了什么")
    current_judgment: str = Field(description="当前判断：目前能确定 / 怀疑的方向")
    severity_assessment: str = Field(default="", description="严重性评估")

    recommended_checks: list[RecommendedCheck] = Field(
        default_factory=list, description="建议排障顺序（首响核心产物）"
    )
    evidence: list[Evidence] = Field(
        default_factory=list, description="证据列表（区分事实 / 推断）"
    )

    root_cause_hypotheses: list[str] = Field(
        default_factory=list, description="暂定根因假设（推断，非定论）"
    )
    pending_confirmations: list[str] = Field(
        default_factory=list, description="待人工确认项"
    )
    risk_notes: str = Field(default="", description="风险与升级建议")

    # 元信息：标记这份报告是正常结构化产出还是降级产物
    is_degraded: bool = Field(
        default=False, description="是否为降级报告（解析失败时的纯文本兜底）"
    )
    # 为什么 is_degraded 这个布尔值不够：
    # 它只说「这份报告不完整」，不说**为什么**不完整。而不同原因的处置方向完全相反：
    #   llm_timeout      → 模型调用超时，该看模型服务 / 调超时配置
    #   parse_failed     → 模型没按格式输出，该调 prompt
    #   retrieval_failed → 向量库挂了，该看 Milvus
    #   retrieval_empty  → 知识库真的缺这篇 SOP，该补文档
    # 只有最后一种才该让值班同学去写文档。混在一个布尔里，
    # 质量运营就没法按原因下钻，也没法统计「降级里有多少是我们自己的故障」。
    degrade_reason: Optional[DegradeReason] = Field(
        default=None, description="降级原因（与 is_degraded 配对，可聚合可下钻）"
    )
    raw_text: Optional[str] = Field(
        default=None, description="降级时保留的原始模型文本"
    )

    def to_markdown(self) -> str:
        """渲染成值班同学可直接阅读的 Markdown。"""
        if self.is_degraded and self.raw_text:
            return f"# 告警首响分析报告（降级）\n\n> ⚠️ 结构化解析失败，以下为原始输出\n\n{self.raw_text}"

        lines: list[str] = ["# 告警首响分析报告", ""]
        lines += ["## 告警摘要", "", self.alert_summary, ""]
        lines += ["## 当前判断", "", self.current_judgment, ""]
        if self.severity_assessment:
            lines += ["## 严重性评估", "", self.severity_assessment, ""]

        if self.recommended_checks:
            lines += ["## 建议排障顺序", ""]
            for check in sorted(self.recommended_checks, key=lambda c: c.order):
                human = "（需人工确认）" if check.requires_human else ""
                lines.append(f"{check.order}. **{check.action}**{human}")
                lines.append(f"   - 原因：{check.reason}")
                if check.command_hint:
                    lines.append(f"   - 参考命令：`{check.command_hint}`")
            lines.append("")

        if self.evidence:
            lines += ["## 证据", ""]
            for ev in self.evidence:
                tag = "✅ 已验证" if ev.type == EvidenceType.VERIFIED_FACT else "🤔 推断"
                src = f"（来源：{ev.source_title}）" if ev.source_title else ""
                lines.append(f"- {tag} {ev.content}{src}")
            lines.append("")

        if self.root_cause_hypotheses:
            lines += ["## 暂定根因（假设）", ""]
            lines += [f"- {h}" for h in self.root_cause_hypotheses]
            lines.append("")

        if self.pending_confirmations:
            lines += ["## 待确认项", ""]
            lines += [f"- {p}" for p in self.pending_confirmations]
            lines.append("")

        if self.risk_notes:
            lines += ["## 风险与升级建议", "", self.risk_notes, ""]

        return "\n".join(lines).strip()


# ──────────────────────────────────────────────────────────────
# 解析：把 LLM 的输出安全解析为 FirstResponseReport
# ──────────────────────────────────────────────────────────────
def _extract_json_block(text: str) -> Optional[str]:
    """从模型输出里尽力抠出 JSON 块。

    兼容三种情况：
    1. 整段就是 JSON
    2. 被 ```json ... ``` 代码块包裹
    3. 文本中嵌了一个 {...} 对象
    """
    text = text.strip()
    if not text:
        return None

    # 情况 2：markdown 代码块
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)

    # 情况 1 / 3：找第一个 { 到最后一个 } 的最大区间
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return None


def parse_report(raw_text: str) -> FirstResponseReport:
    """把模型输出解析为结构化报告；任何失败都降级为纯文本报告。

    这是「AI 输出必须可降级」纪律的落点：无论模型返回什么，
    调用方永远拿到一个可用的 FirstResponseReport，不会抛异常。
    """
    json_block = _extract_json_block(raw_text)
    if json_block is None:
        return _degraded(raw_text)

    try:
        data: Any = json.loads(json_block)
    except (json.JSONDecodeError, ValueError):
        return _degraded(raw_text)

    if not isinstance(data, dict):
        return _degraded(raw_text)

    try:
        return FirstResponseReport.model_validate(data)
    except ValidationError:
        # 结构不完整也不崩：能塞的字段尽量塞，其余降级
        return _degraded(raw_text)


def _degraded(raw_text: str) -> FirstResponseReport:
    """构造降级报告。

    这里统一标 PARSE_FAILED：模型答了，只是没按约定格式答。
    它与 llm_error（模型没答上来）、retrieval_failed（检索挂了）
    的处置完全不同 —— 该修的是 prompt 或输出约束，不是扩容也不是补文档。
    """
    return FirstResponseReport(
        alert_summary="（结构化解析失败，见原始输出）",
        current_judgment="模型未按结构化格式返回，已降级为纯文本。",
        is_degraded=True,
        degrade_reason=DegradeReason.PARSE_FAILED.value,
        raw_text=raw_text.strip() or "（模型返回为空）",
    )
