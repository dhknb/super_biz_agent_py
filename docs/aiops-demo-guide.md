# AIOps 告警首响分析 — 演示指南

> 两周冲刺产物：把一条告警变成一份**有证据、可追踪、可溯源**的结构化首响报告。
> 本指南让你用一条命令（或几个 API 调用）走完整条闭环，不依赖真实生产系统。

## 这是什么

面向一线值班工程师的**告警首响分析助手**。告警触发后，系统自动：

1. 归一化告警（支持手工 JSON / Prometheus Alertmanager 两种格式）
2. 落库成可追踪的诊断任务（状态机 + 事件时间线）
3. 用 RAG 检索匹配的运维 SOP（每条证据带来源溯源）
4. LLM 生成结构化首响报告（解析失败自动降级）
5. 报告严格区分「✅ 已验证事实」和「🤔 模型推断」

与通用聊天机器人的区别：**每条结论都能溯源到 SOP 或工具证据**，无证据时老实标注「待确认」而非编造根因。

## 快速体验（离线，不烧 token）

不连真实 LLM / Milvus，用假依赖跑通整条评估链路，产出质量基线：

```bash
wsl .venv/bin/python -m tests.eval.run_aiops_eval
```

预期输出：

```
=== AIOps 首响报告质量评估 ===
样本数:            3
SOP 命中率:        100%
报告要点覆盖率:    100%
幻觉受控率:        100%
```

> 这是「理想基线」（假 LLM + 完美命中），证明**评估管道本身正确**。
> 接真实 LLM / Milvus 后，这些数字会反映真实质量。

## 完整链路演示（需真实服务）

### 前置：启动依赖

```bash
# Milvus（向量库）+ Redis + PostgreSQL
docker compose -f vector-database.yml up -d

# 跑数据库迁移（建表）
wsl .venv/bin/alembic upgrade head

# 导入运维 SOP 到知识库
python -c "import requests, os, time; [requests.post('http://localhost:9900/api/upload', files={'file': open(f'aiops-docs/{f}', 'rb')}) or time.sleep(1) for f in os.listdir('aiops-docs') if f.endswith('.md')]"

# 启动主服务
python -m uvicorn app.main:app --host 0.0.0.0 --port 9900
```

### 演示 1：同步首响分析（CPU 告警）

```bash
curl -X POST "http://localhost:9900/api/aiops/alerts/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "manual",
    "payload": {
      "alert_name": "HighCPUUsage",
      "severity": "critical",
      "service": "order-api",
      "instance": "10.0.0.12:9100",
      "summary": "order-api CPU 使用率持续超过 90% 已 10 分钟"
    }
  }'
```

返回：`task_id` + 结构化报告 + Markdown 报告。

### 演示 2：流式首响分析（SSE 时间线）

```bash
curl -N -X POST "http://localhost:9900/api/aiops/alerts/analyze/stream" \
  -H "Content-Type: application/json" \
  -d '{"source": "manual", "payload": {"alert_name": "HighCPUUsage", "severity": "critical", "service": "order-api"}}'
```

实时推送阶段事件：`task_created → planning → retrieving → diagnosing → report`。

### 演示 3：Prometheus 格式告警

```bash
curl -X POST "http://localhost:9900/api/aiops/alerts/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "prometheus",
    "payload": {
      "status": "firing",
      "labels": {"alertname": "ServiceUnavailable", "severity": "critical", "service": "payment-api"},
      "annotations": {"summary": "payment-api 健康检查连续失败"},
      "startsAt": "2026-07-23T12:00:00Z"
    }
  }'
```

### 演示 4：查询诊断任务

```bash
# 列出所有诊断任务
curl "http://localhost:9900/api/aiops/tasks"

# 查单个任务详情（含事件时间线）
curl "http://localhost:9900/api/aiops/tasks/{task_id}"
```

## API 一览

| 功能 | 方法 | 路径 |
|------|------|------|
| 首响分析（同步） | POST | `/api/aiops/alerts/analyze` |
| 首响分析（SSE 流式） | POST | `/api/aiops/alerts/analyze/stream` |
| 诊断任务列表 | GET | `/api/aiops/tasks` |
| 诊断任务详情 | GET | `/api/aiops/tasks/{task_id}` |

## 架构（两周冲刺产物）

```
AlarmEvent（归一化：手工 / Prometheus）
  → AiopsDiagnosisTask（落库，状态机 new→planning→retrieving→diagnosing→done）
  → SopRetrievalService（RAG 检索 SOP，证据带来源溯源）
  → FirstResponseService（构造 prompt → LLM → 结构化报告 → 降级兜底）
  → FirstResponseReport（证据区分 ✅事实/🤔推断）
```

## 设计决策

所有关键决策记录在 [docs/adr/](adr/)：告警模型、任务化、结构化报告、SOP 证据溯源、
测试隔离、评估体系、流式时间线，共 9 篇 ADR。
