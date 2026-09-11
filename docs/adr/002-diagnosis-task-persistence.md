# ADR-002：诊断任务化 —— 把「跑完即焚」升级为可追踪任务

- 状态：已接受
- 日期：2026-07-23
- 关联：[ADR-000](000-scope-first-response.md)、[ADR-001](001-standard-alarm-event-model.md)、产品方向文档 Phase 4

## 背景

现有 `/api/aiops` 诊断是**无状态**的：`PlanExecuteState` 只是一个内存 `TypedDict`
（`input / plan / past_steps / response`），随 LangGraph 图执行结束即销毁。

这带来三个问题：

1. **不可追踪**：一次诊断结束后，没有任何记录能回答「刚才那次诊断的输入、
   过程、结论是什么」。
2. **不可查询**：无法按 `alert_id` 找到历史诊断，无法列出「正在诊断中」的任务。
3. **不可审计**：产品方向文档明确要求「可审计的分析过程和报告产物」，
   内存态无法满足。

产品方向文档 Phase 4 已规划「分析任务化」，本 ADR 落地它。

## 决策

新增两张表，把一次诊断建模为**任务 + 事件时间线**：

- `aiops_diagnosis_tasks`：一次诊断任务。持有归一化后的 `alarm_event`（全量留档）、
  `status`（状态机）、`report`（Day4 的结构化报告）、`summary`。
- `aiops_diagnosis_events`：诊断过程中的事件流水，为 Day9 的 SSE 时间线打地基。

### 状态机

```
NEW ──▶ PLANNING ──▶ RETRIEVING ──▶ DIAGNOSING ──▶ DONE
          │             │              │
          └─────────────┴──────────────┴──────────▶ FAILED
```

状态跃迁在 **repository 层**用 `can_transition()` 强制校验，非法跳转抛
`ValueError`。这样「状态只能顺序前进」这条业务规则有唯一执行点，
不依赖调用方自觉（DRY + 单一职责）。

- 相同状态视为幂等允许（重试安全）。
- 任意状态都能转 `FAILED`（`mark_failed`），因为异常随时可能发生。
- `save_report` 只接受从 `DIAGNOSING → DONE`，保证「有报告才算完成」。

## 备选方案

| 方案 | 说明 | 为什么不选 |
|------|------|-----------|
| **复用 `chat_run_traces`** | 把诊断塞进现有 trace 表 | 语义不符：trace 是「一问一答的检索留痕」，诊断是「带状态机的长流程任务」，字段和生命周期都不同 |
| **只加一张 task 表，不要 event 表** | 事件塞进 task 的 JSON 字段 | Day9 的时间线需要按时间排序、独立查询事件；拆表更清晰，也符合参考项目的 `agent_executions` 分离思路 |
| **状态校验放 service 层** | repository 只做 CRUD | 状态规则会散落到多个调用点，难保证一致；放 repository 是唯一入口 |
| **状态用 String 而非 Enum** | 省一个枚举定义 | 丢失类型安全和 IDE 补全，且非法值无法在写入时拦截 |

## 后果

### 正面
- 每次诊断有唯一 `task_id`，可查状态、历史、事件时间线。
- 状态机集中防护，非法流转在最底层被拦截。
- `alarm_event` 与 `report` 全量留档，满足审计要求。
- 复用了 `chat_run_trace` / `protocol_ingestion` 的既有范式（`_utcnow_naive`、
  `StrEnum + values_callable`、Alembic 迁移命名），风格一致、学习成本低。

### 负面 / 代价
- 多了两张表和一个迁移，诊断链路要多写库（性能可忽略，诊断本就是秒级慢流程）。
- service 层（Day4+）需要改造现有 `diagnose()` 去驱动状态机，是后续工作量。

## 工程纪律（本次踩坑记录）

通过 WSL 网络路径（`//wsl.localhost/...`）写文件时，工具偶发**「报告成功但
实际未落盘」**。教训：**关键文件写完后，一律用 `wsl ... ls/wc` 复核真实落盘**，
不轻信返回提示。已按此纪律逐个验证本次所有新增文件。
