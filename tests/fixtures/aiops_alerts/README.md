# AIOps 告警样例集

用于 Day6 集成测试与 Day8 评测脚本的三条典型告警。

每条样例包含：
- `payload`：告警原始数据（可直接 POST 到 `/api/aiops/alerts/analyze`）
- `source`：告警来源（manual / prometheus）
- `expected_sop_keywords`：期望命中的 SOP 关键词（Day8 评测用）
- `expected_report_points`：期望首响报告里出现的排查要点（Day8 评测用）

| 文件 | 场景 | 来源 |
|------|------|------|
| `01_cpu_high.json` | CPU 持续高负载 | manual |
| `02_disk_full.json` | 磁盘空间不足 | manual |
| `03_service_unavailable.json` | 服务不可用（Prometheus 格式） | prometheus |

> 这些样例都可本地复现，不依赖真实生产系统（对齐 ADR-000）。
