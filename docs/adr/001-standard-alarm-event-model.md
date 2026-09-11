# ADR-001：标准化告警事件模型 AlarmEvent 与多来源归一化

- 状态：已接受
- 日期：2026-07-23
- 关联：[ADR-000](000-scope-first-response.md)、`docs/product-direction-aiops-first-response.md`

## 背景

告警首响分析的第一步，是让「一条告警」有一个**统一、稳定、可审计**的内部表示。
现状有两个问题：

1. 现有 `AlertInfo`（`alertname / severity / instance / duration / description`）
   是旧诊断接口的轻量结构，字段太少，表达不了来源、指标、标签、时间窗、原始 payload。
2. 真实告警来源格式各异：手工录入是一套字段名，Prometheus Alertmanager 的
   webhook 又是 `labels + annotations + startsAt/endsAt` 的嵌套结构，级别名称
   也不统一（critical / warn / disaster ...）。

如果每个下游节点（证据采集、SOP 检索、报告生成）各自去解析原始 payload，
就会到处是 `payload.get(...)` 的脏逻辑，且没有单一事实来源。

> 顺带修复：`app/models/aiops.py` 原本因非 UTF-8 保存，中文注释全部损坏成 `????`，
> 本次一并复原为 UTF-8。

## 决策

新增一个 **`AlarmEvent`** Pydantic 模型作为首响分析的**唯一输入契约**，
并提供 `normalize_alarm(payload, source)` 把不同来源的原始数据归一化成它。

关键设计：

- **级别归一化**：用 `_SEVERITY_ALIASES` 映射表把各来源级别名统一到
  `AlarmSeverity`（critical / warning / info / unknown），识别不了就 `unknown`，
  绝不丢弃告警。
- **来源分发表**：`_NORMALIZERS = {MANUAL: ..., PROMETHEUS: ...}`，
  新增来源只需登记一个函数，符合开闭原则（对扩展开放，对修改关闭）。
- **原始 payload 全量留档**：`raw_payload` 保留原始数据，满足审计与回放。
- **业务方法内聚**：`is_firing`（是否未恢复）、`retrieval_query()`（构造 SOP 检索语句）
  直接挂在模型上，让「告警怎么用」和「告警是什么」放在一起。
- **保留旧模型**：`AIOpsRequest / AlertInfo / DiagnosisResponse` 原样保留，
  不破坏现有 `/api/aiops`（遵循 ADR-000 的非破坏性原则）。

## 备选方案

1. **直接扩展旧 `AlertInfo`**：字段耦合在旧接口里，且它没有归一化概念，
   扩展后会变成一个谁都不敢动的「上帝结构」。放弃。
2. **每个来源各自建模型（PrometheusAlert / ZabbixAlert ...）**：下游要处理 N 种类型，
   违背单一事实来源。改为「N 种输入 → 1 种内部模型」。
3. **用 dict 到处传**：没有类型校验、没有 IDE 提示、字段名靠记忆，最容易出 bug。放弃。

## 后果

正面：

- 下游节点只认 `AlarmEvent` 一个结构，逻辑干净。
- 新增告警来源成本极低（加一个 normalizer + 登记）。
- Pydantic 提供类型校验和 JSON schema，接口文档自动生成。
- 15 条单测锁住归一化行为（两种来源、级别映射、时间零值、webhook 拆包）。

负面 / 代价：

- 多了一层归一化，简单场景看似「绕」了一下——但这是可维护性的必要投资。
- `_SEVERITY_ALIASES` 需要随真实来源持续补充（已预留 Zabbix 映射）。

## 验收

- 一条手工 JSON 告警和一条 Prometheus 风格告警，归一化后是同一个 `AlarmEvent`。
- `tests/unit/test_alarm_event.py` 全绿（14 用例）。
