# 架构决策记录（ADR）

> ADR = Architecture Decision Record。每一篇记录一个「当时为什么这么选」的决策：
> 背景、可选项、选择、放弃了什么、代价。目的是让未来的自己（和面试官）
> 能看懂每个设计背后的取舍，而不只是「做了什么」。

## 为什么写 ADR

参考项目 [itops-agent-platform](https://github.com/qinshihu/itops-agent-platform)
用 36 篇 ADR 支撑起它的专业度。本项目在两周 AIOps 首响冲刺中同样坚持
「每个重要决策写一页」——成本极低，但它把「我做了 X」升级为
「我在 A 和 B 之间选了 A，因为……，代价是……」。

## 格式约定

采用简化版 MADR：

- 文件名：`NNN-kebab-case-title.md`（三位序号）
- 每篇包含：状态 / 背景 / 决策 / 备选方案 / 后果（正负）
- 状态取值：`提议中` → `已接受` → （可能）`已废弃 / 被 NNN 取代`

## 索引

| 编号 | 标题 | 状态 | 关联 |
|------|------|------|------|
| [000](000-scope-first-response.md) | 冻结两周范围：只做告警首响，不碰真实命令执行 | 已接受 | 产品方向文档 |
| [001](001-standard-alarm-event-model.md) | 标准化告警事件模型 AlarmEvent 与多来源归一化 | 已接受 | ADR-000 |
| [002](002-diagnosis-task-persistence.md) | 诊断任务化：状态机 + 事件留痕 + Alembic 迁移 | 已接受 | ADR-001 |
| [003](003-structured-first-response-report.md) | 结构化首响报告 schema + 解析降级兜底 | 已接受 | ADR-002 |
| [004](004-sop-retrieval-and-evidence-tracing.md) | ★SOP 检索接入首响 + 证据溯源（事实/推断区分） | 已接受 | ADR-003 |
| [005](005-alert-first-response-api.md) | 告警首响 API + 编排层：归一化→任务→报告端到端 | 已接受 | ADR-004 |
| [006](006-test-isolation-and-secret-hygiene.md) | 测试隔离 + 密钥卫生：顶层假 env 抢在 import 前 | 已接受 | ADR-005 |
| [007](007-aiops-first-response-eval.md) | AIOps 首响报告质量评估：SOP命中/要点覆盖/幻觉受控 | 已接受 | ADR-006 |
