"""AlarmEvent 告警模型与归一化逻辑的单元测试。

覆盖：
- 手工 JSON 与 Prometheus 两种来源归一化为同一模型
- 级别别名映射
- 时间解析容错
- 未知来源报错
- 检索查询构造
"""

from datetime import datetime

import pytest

from app.models.aiops import (
    AlarmEvent,
    AlarmSeverity,
    AlarmSource,
    normalize_alarm,
    normalize_prometheus_webhook,
    normalize_severity,
)


class TestNormalizeSeverity:
    def test_common_aliases(self) -> None:
        assert normalize_severity("critical") == AlarmSeverity.CRITICAL
        assert normalize_severity("CRIT") == AlarmSeverity.CRITICAL
        assert normalize_severity("warning") == AlarmSeverity.WARNING
        assert normalize_severity("Info") == AlarmSeverity.INFO

    def test_zabbix_aliases(self) -> None:
        assert normalize_severity("disaster") == AlarmSeverity.CRITICAL
        assert normalize_severity("average") == AlarmSeverity.WARNING

    def test_unknown_and_empty(self) -> None:
        assert normalize_severity("something-weird") == AlarmSeverity.UNKNOWN
        assert normalize_severity(None) == AlarmSeverity.UNKNOWN
        assert normalize_severity("") == AlarmSeverity.UNKNOWN


class TestNormalizeManual:
    def test_full_manual_payload(self) -> None:
        payload = {
            "alert_name": "HighCPUUsage",
            "severity": "critical",
            "service": "order-api",
            "instance": "10.0.0.12:9100",
            "summary": "CPU 使用率持续超过 90%",
            "labels": {"env": "prod"},
            "fired_at": "2026-07-23T10:00:00Z",
        }
        event = normalize_alarm(payload, source="manual")

        assert event.alert_name == "HighCPUUsage"
        assert event.severity == AlarmSeverity.CRITICAL
        assert event.source == AlarmSource.MANUAL
        assert event.service == "order-api"
        assert event.instance == "10.0.0.12:9100"
        assert event.labels == {"env": "prod"}
        assert event.fired_at == datetime.fromisoformat("2026-07-23T10:00:00+00:00")
        assert event.is_firing is True
        assert event.raw_payload == payload

    def test_alertname_alias_and_description_fallback(self) -> None:
        # 手工 payload 用了 alertname / description 两个别名
        payload = {"alertname": "DiskFull", "description": "磁盘快满了"}
        event = normalize_alarm(payload)  # 默认 manual

        assert event.alert_name == "DiskFull"
        assert event.summary == "磁盘快满了"
        assert event.severity == AlarmSeverity.UNKNOWN

    def test_missing_name_falls_back(self) -> None:
        event = normalize_alarm({}, source="manual")
        assert event.alert_name == "UnknownAlert"


class TestNormalizePrometheus:
    def test_firing_alert(self) -> None:
        payload = {
            "status": "firing",
            "labels": {
                "alertname": "HighCPUUsage",
                "severity": "critical",
                "service": "order-api",
                "instance": "10.0.0.12:9100",
                "env": "prod",
            },
            "annotations": {"summary": "CPU 90%+"},
            "startsAt": "2026-07-23T10:00:00Z",
            "endsAt": "0001-01-01T00:00:00Z",
        }
        event = normalize_alarm(payload, source="prometheus")

        assert event.alert_name == "HighCPUUsage"
        assert event.severity == AlarmSeverity.CRITICAL
        assert event.source == AlarmSource.PROMETHEUS
        assert event.service == "order-api"
        assert event.instance == "10.0.0.12:9100"
        assert event.resource_type == "host"
        assert event.resource_id == "10.0.0.12:9100"
        # 业务标签里不应残留 alertname / severity / service / instance
        assert event.labels == {"env": "prod"}
        assert event.summary == "CPU 90%+"
        # firing 状态下 endsAt 的零值不能被当成已恢复
        assert event.resolved_at is None
        assert event.is_firing is True

    def test_resolved_alert(self) -> None:
        payload = {
            "status": "resolved",
            "labels": {"alertname": "HighCPUUsage", "severity": "warning"},
            "annotations": {},
            "startsAt": "2026-07-23T10:00:00Z",
            "endsAt": "2026-07-23T10:30:00Z",
        }
        event = normalize_alarm(payload, source="prometheus")

        assert event.resolved_at == datetime.fromisoformat("2026-07-23T10:30:00+00:00")
        assert event.is_firing is False

    def test_webhook_batch(self) -> None:
        body = {
            "alerts": [
                {"status": "firing", "labels": {"alertname": "A", "severity": "critical"}},
                {"status": "firing", "labels": {"alertname": "B", "severity": "warning"}},
            ]
        }
        events = normalize_prometheus_webhook(body)
        assert [e.alert_name for e in events] == ["A", "B"]
        assert [e.severity for e in events] == [
            AlarmSeverity.CRITICAL,
            AlarmSeverity.WARNING,
        ]


class TestCrossSourceConsistency:
    """核心验收：两种格式的同一告警，归一化后关键字段一致。"""

    def test_manual_and_prometheus_converge(self) -> None:
        manual = normalize_alarm(
            {
                "alert_name": "HighCPUUsage",
                "severity": "critical",
                "service": "order-api",
                "instance": "10.0.0.12:9100",
            },
            source="manual",
        )
        prom = normalize_alarm(
            {
                "status": "firing",
                "labels": {
                    "alertname": "HighCPUUsage",
                    "severity": "critical",
                    "service": "order-api",
                    "instance": "10.0.0.12:9100",
                },
            },
            source="prometheus",
        )

        # 关键定位字段应当完全一致（来源不同不影响）
        assert manual.alert_name == prom.alert_name
        assert manual.severity == prom.severity
        assert manual.service == prom.service
        assert manual.instance == prom.instance
        assert isinstance(manual, AlarmEvent) and isinstance(prom, AlarmEvent)


class TestRetrievalQuery:
    def test_query_combines_fields(self) -> None:
        event = AlarmEvent(
            alert_name="HighCPUUsage",
            service="order-api",
            summary="CPU 使用率超过 90%",
        )
        query = event.retrieval_query()
        assert "HighCPUUsage" in query
        assert "order-api" in query
        assert "CPU 使用率超过 90%" in query

    def test_query_with_minimal_fields(self) -> None:
        event = AlarmEvent(alert_name="DiskFull")
        assert event.retrieval_query() == "DiskFull"


class TestUnknownSource:
    def test_unknown_source_raises(self) -> None:
        with pytest.raises(ValueError, match="未知的告警来源"):
            normalize_alarm({"alert_name": "X"}, source="splunk")

    def test_unsupported_but_valid_source_raises(self) -> None:
        # zabbix 是合法枚举但第一版未实现归一化函数
        with pytest.raises(ValueError, match="暂不支持"):
            normalize_alarm({"alert_name": "X"}, source=AlarmSource.ZABBIX)
