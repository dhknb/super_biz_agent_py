# ADR-000：冻结两周范围 — 只做告警首响，不碰真实命令执行

- 状态：已接受
- 日期：2026-07-23
- 关联：[产品方向文档](../product-direction-aiops-first-response.md)、[itops-agent-platform](https://github.com/qinshihu/itops-agent-platform)

## 背景

当前 `super_biz_agent_py` 已具备 FastAPI + LangGraph（Plan-Execute-Replan）+
RAG（子查询 / rerank / 幻觉拦截）+ MCP 工具接入的底座。参考项目
itops-agent-platform（TypeScript/Express，785 stars，36 篇 ADR）已经做到
「LLM 驱动的 Zabbix/Prometheus 自动修复」，包含 SSH/Docker/K8s 真实命令执行、
命令注入防护、工具风险审计、多 Agent 协作。

我作为学习阶段的单人开发者，有两周时间。需要一个**能落地、能写进简历、
且不与那个成熟项目正面硬碰**的范围。

## 决策

两周内**只做「告警首响分析助手」**：给定一条告警事件，系统自动关联监控、
日志、SOP，产出一份带证据引用、区分「已验证事实 / 模型推断」的结构化首响报告。

明确的**边界**：

**做（In Scope）**
- 标准告警模型 `AlarmEvent` + 多来源归一化
- 诊断任务化：`task_id` + 状态机 + 落库留痕
- 结构化首响报告 schema + 解析失败降级兜底
- 用现有 RAG 检索 SOP，证据溯源、区分事实与推断（★核心差异点）
- 测试隔离 + 最小 RAG 评估脚本
- SSE 诊断时间线

**不做（Out of Scope，本次明确放弃）**
- ❌ 真实 SSH / Docker / K8s 命令执行
- ❌ 全自动修复、生产变更
- ❌ 多租户 / 细粒度 RBAC / CMDB / 拓扑
- ❌ 复杂多 Agent 编排框架

## 为什么不碰真实命令执行（关键取舍）

这是与参考项目最大的差异，也是最容易踩坑的地方，单独说明：

1. **安全底座不具备**：真实命令执行需要命令注入防护、白名单、沙箱、
   审计链路、回滚机制。参考项目为此写了专门的 ADR（命令过滤、SSH 注入修复、
   工具风险审计）。两周内做不出安全的执行层，做出来就是安全事故。
2. **价值定位不同**：我的项目最强的是 RAG 检索链路，参考项目最弱的恰是知识检索。
   应该放大自己的长板（把知识证据化地喂给诊断），而不是去补对方的长板（执行）。
3. **首响本就不需要执行**：一线值班的第一诉求是「先查什么、为什么查、有什么证据」，
   这是**只读**的分析，不需要改变系统状态。

因此工具调用一律限定为**只读查询**（监控、日志、知识检索）。未来若要做执行，
必须先补齐安全底座，另开 ADR。

## 备选方案

- **A：照搬参考项目做执行平台**。放弃 —— 追不上，且会做出一堆半成品。
- **B：只做通用 RAG 问答增强**。放弃 —— 没有把已有 AIOps 资产用起来，故事不完整。
- **C（选中）：RAG × AIOps 首响**。用自己的长板补 AIOps 的证据短板，形成差异化闭环。

## 后果

**正向**
- 范围可控，每天有可验收产出，两周能出可演示 MVP。
- 形成参考项目做不到的「知识证据化诊断」差异点，简历有独特卖点。
- 全程只读，无生产风险。

**负向 / 代价**
- 不具备「自动修复」这一最吸睛的能力，演示时需说明这是有意识的边界选择。
- 半固定链路会削弱一部分 Plan-Execute-Replan 的「自由规划」展示，
  但换来稳定性和可审计性（见后续 ADR）。

## 现有链路盘点（开工基线）

```
POST /api/aiops (session_id)
  -> aiops_service.diagnose()          # 硬编码一段要求输出 Markdown 的大 prompt
    -> execute(task)                   # 通用 Plan-Execute-Replan
      -> planner   (state.py: PlanExecuteState = TypedDict, 内存态, 不落库)
      -> executor  (调 MCP 工具 + retrieve_knowledge)
      -> replanner (continue / replan / respond)
  -> SSE: status / plan / step_complete / report / complete / error
```

盘点发现的三个改造点（后续 ADR 逐个处理）：
1. `app/models/aiops.py` 文件编码损坏（中文注释变 `????`）—— Day2 顺手修复。
2. 诊断无状态（`PlanExecuteState` 跑完即焚）—— Day3 引入任务表补齐。
3. `diagnose()` 强制 Markdown 输出、prompt 明写「不要 JSON」—— 与 Day4 结构化报告冲突，
   届时改造为「结构化 schema + 渲染成 Markdown」双轨。
