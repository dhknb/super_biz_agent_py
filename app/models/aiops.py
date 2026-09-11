"""
AIOps 数据模型

包含：
- AlarmEvent：标准化告警事件模型（首响分析的统一输入）
- normalize_alarm：把不同来源（手工 JSON / Prometheus Alertmanager）的
  原始 payload 归一化为同一个 AlarmEvent

设计参考 docs/adr/000-scope-first-response.md 与
docs/product-direction-aiops-first-response.md。
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ──────────────────────────────────────────────────────────────
# 告警首响：标准化告警事件模型
# ──────────────────────────────────────────────────────────────
class AlarmSeverity(StrEnum):
    """告警级别（统一口径）

    不同来源的级别名称五花八门（Prometheus 用 critical/warning，
    Zabbix 用 disaster/high/average...），归一化时统一映射到这套枚举。
    """

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    UNKNOWN = "unknown"


class AlarmSource(StrEnum):
    """告警来源。第一版只实现 manual 和 prometheus，其余预留。"""

    MANUAL = "manual"
    PROMETHEUS = "prometheus"
    ZABBIX = "zabbix"
    CLOUD = "cloud"


# 各来源级别 → 统一级别的映射表
_SEVERITY_ALIASES: Dict[str, AlarmSeverity] = {
    # 通用
    "critical": AlarmSeverity.CRITICAL,
    "crit": AlarmSeverity.CRITICAL,
    "fatal": AlarmSeverity.CRITICAL,
    "warning": AlarmSeverity.WARNING,
    "warn": AlarmSeverity.WARNING,
    "info": AlarmSeverity.INFO,
    "information": AlarmSeverity.INFO,
    # Zabbix 风格（为将来预留，先放进映射不吃亏）
    "disaster": AlarmSeverity.CRITICAL,
    "high": AlarmSeverity.CRITICAL,
    "average": AlarmSeverity.WARNING,
    "not classified": AlarmSeverity.UNKNOWN,
}


def normalize_severity(raw: str | None) -> AlarmSeverity:
    """把任意来源的级别字符串归一化为 AlarmSeverity。识别不了就 UNKNOWN。"""
    if not raw:
        return AlarmSeverity.UNKNOWN
    return _SEVERITY_ALIASES.get(raw.strip().lower(), AlarmSeverity.UNKNOWN)


class AlarmEvent(BaseModel):
    """标准化告警事件 —— 首响分析的统一输入。

    无论告警来自手工录入还是 Prometheus Alertmanager，最终都归一化为本模型，
    后续所有节点（证据采集、SOP 检索、报告生成）只认这一个结构。
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "alert_name": "HighCPUUsage",
                "severity": "critical",
                "source": "prometheus",
                "service": "order-api",
                "instance": "10.0.0.12:9100",
                "summary": "CPU 使用率持续超过 90%",
                "labels": {"env": "prod", "cluster": "cn-hangzhou"},
                "fired_at": "2026-07-23T10:00:00Z",
            }
        }
    )

    alert_name: str = Field(description="告警名称，如 HighCPUUsage")
    severity: AlarmSeverity = Field(
        default=AlarmSeverity.UNKNOWN, description="归一化后的告警级别"
    )
    source: AlarmSource = Field(default=AlarmSource.MANUAL, description="告警来源")

    # 资源定位：service / instance 是人看的，resource_type / resource_id 是机器用的
    service: Optional[str] = Field(default=None, description="受影响服务")
    instance: Optional[str] = Field(default=None, description="受影响实例")
    resource_type: Optional[str] = Field(
        default=None, description="资源类型，如 host / pod / service"
    )
    resource_id: Optional[str] = Field(default=None, description="资源唯一标识")

    # 指标（可选，Prometheus 类告警常带）
    metric_name: Optional[str] = Field(default=None, description="触发告警的指标名")
    metric_value: Optional[float] = Field(default=None, description="触发时的指标值")

    summary: Optional[str] = Field(default=None, description="告警摘要")
    labels: Dict[str, str] = Field(default_factory=dict, description="业务/环境标签")

    fired_at: Optional[datetime] = Field(default=None, description="首次触发时间")
    resolved_at: Optional[datetime] = Field(
        default=None, description="恢复时间，未恢复为空"
    )

    # 原始 payload 全量留档，方便审计和回放
    raw_payload: Dict[str, Any] = Field(default_factory=dict, description="原始告警数据")

    @property
    def is_firing(self) -> bool:
        """告警是否仍处于触发状态（未恢复）。"""
        return self.resolved_at is None

    def retrieval_query(self) -> str:
        """构造用于 RAG / SOP 检索的查询语句。

        把告警名、服务、摘要拼成一句自然语言，交给知识库检索最相关的 SOP。
        """
        parts = [self.alert_name]
        if self.service:
            parts.append(self.service)
        if self.summary:
            parts.append(self.summary)
        return " ".join(parts)


# ──────────────────────────────────────────────────────────────
# 归一化：不同来源 → AlarmEvent
# ──────────────────────────────────────────────────────────────
def _parse_time(value: Any) -> Optional[datetime]:
    """尽力把各种时间表示解析为 datetime，失败返回 None。"""
    if value in (None, "", "0001-01-01T00:00:00Z"):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        text = value.strip()
        try:
            # 兼容末尾 Z 的 ISO8601
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _normalize_manual(payload: Dict[str, Any]) -> AlarmEvent:
    """归一化「手工 JSON」告警。

    约定字段名尽量贴近 AlarmEvent 本身，缺失字段用兜底。
    alert_name 支持 alert_name / alertname 两种写法。
    """
    alert_name = payload.get("alert_name") or payload.get("alertname") or "UnknownAlert"
    return AlarmEvent(
        alert_name=alert_name,
        severity=normalize_severity(payload.get("severity")),
        source=AlarmSource.MANUAL,
        service=payload.get("service"),
        instance=payload.get("instance"),
        resource_type=payload.get("resource_type"),
        resource_id=payload.get("resource_id"),
        metric_name=payload.get("metric_name"),
        metric_value=payload.get("metric_value"),
        summary=payload.get("summary") or payload.get("description"),
        labels=dict(payload.get("labels") or {}),
        fired_at=_parse_time(payload.get("fired_at") or payload.get("started_at")),
        resolved_at=_parse_time(payload.get("resolved_at") or payload.get("ended_at")),
        raw_payload=payload,
    )


def _normalize_prometheus(payload: Dict[str, Any]) -> AlarmEvent:
    """归一化「Prometheus Alertmanager」单条告警。

    Alertmanager webhook 的单条 alert 结构大致为：
    {
      "status": "firing",
      "labels": {"alertname": "HighCPUUsage", "severity": "critical",
                 "service": "order-api", "instance": "10.0.0.12:9100"},
      "annotations": {"summary": "...", "description": "..."},
      "startsAt": "2026-07-23T10:00:00Z",
      "endsAt": "0001-01-01T00:00:00Z"
    }
    """
    labels = dict(payload.get("labels") or {})
    annotations = dict(payload.get("annotations") or {})

    alert_name = labels.get("alertname") or "UnknownAlert"
    status = (payload.get("status") or "").strip().lower()

    # 从 labels 里提取常用维度后，把它们从 labels 副本里移除，
    # 剩下的才是真正的「业务标签」，避免污染。
    service = labels.get("service") or labels.get("job")
    instance = labels.get("instance")
    business_labels = {
        k: v
        for k, v in labels.items()
        if k not in {"alertname", "severity", "service", "job", "instance"}
    }

    resolved_at = _parse_time(payload.get("endsAt"))
    # Alertmanager 用 status=resolved 明确表示已恢复；firing 时 endsAt 是零值
    if status == "firing":
        resolved_at = None

    return AlarmEvent(
        alert_name=alert_name,
        severity=normalize_severity(labels.get("severity")),
        source=AlarmSource.PROMETHEUS,
        service=service,
        instance=instance,
        resource_type="host" if instance else None,
        resource_id=instance,
        metric_name=labels.get("metric") or labels.get("__name__"),
        summary=annotations.get("summary") or annotations.get("description"),
        labels=business_labels,
        fired_at=_parse_time(payload.get("startsAt")),
        resolved_at=resolved_at,
        raw_payload=payload,
    )


# 来源 → 归一化函数 的分发表（新增来源时在这里登记即可，符合开闭原则）
_NORMALIZERS = {
    AlarmSource.MANUAL: _normalize_manual,
    AlarmSource.PROMETHEUS: _normalize_prometheus,
}


def normalize_alarm(
    payload: Dict[str, Any],
    source: AlarmSource | str = AlarmSource.MANUAL,
) -> AlarmEvent:
    """把原始告警 payload 归一化为 AlarmEvent。

    Args:
        payload: 原始告警数据（单条）。
        source: 告警来源，决定用哪个归一化函数。

    Raises:
        ValueError: 传入了尚未支持的来源。
    """
    if isinstance(source, str):
        try:
            source = AlarmSource(source.strip().lower())
        except ValueError as exc:
            raise ValueError(f"未知的告警来源: {source}") from exc

    normalizer = _NORMALIZERS.get(source)
    if normalizer is None:
        raise ValueError(f"暂不支持的告警来源: {source.value}")
    return normalizer(payload)


def normalize_prometheus_webhook(body: Dict[str, Any]) -> List[AlarmEvent]:
    """归一化 Alertmanager webhook 的整包（含多条 alerts）。

    Alertmanager 推送的结构是 {"alerts": [ {...}, {...} ]}，
    这里拆成多个 AlarmEvent。
    """
    alerts = body.get("alerts") or []
    return [_normalize_prometheus(alert) for alert in alerts]
