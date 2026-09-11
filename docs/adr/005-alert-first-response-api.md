# ADR-005：告警首响分析 API —— 端到端闭环编排

- 状态：已接受
- 日期：2026-07-23
- 关联：[[002-diagnosis-task-persistence]]、[[004-sop-retrieval-and-evidence-tracing]]

## 背景

Day1-5 已经造好了一条链路的所有零件：告警模型（Day2）、诊断任务化（Day3）、
结构化报告（Day4）、SOP 检索证据溯源（Day5）。但它们还只是**服务层的零件**，
没有一个真正能被外部调用的 HTTP 入口，也没有把「任务落库」和「首响分析」串起来。

Day6 要做的是**收口**：把零件组装成一条端到端、可调用、可追踪、可审计的链路，
并用三条样例告警 + 集成测试锁住它。

## 决策

### 1. 引入编排层 `AlertDiagnosisOrchestrator`

在 API 与服务之间加一个薄薄的编排层，负责「流程」：

```
payload + source
  → normalize_alarm            （Day2 归一化）
  → repo.create_task → NEW      （Day3 任务落库）
  → planning → retrieving → diagnosing（状态流转 + 事件留痕）
  → first_response_service.analyze   （Day4/5 SOP 检索 + 结构化报告）
  → repo.save_report → DONE
```

为什么单独一层，而不是把逻辑塞进 API：
- API 只负责收参数、序列化、转 HTTP 状态码，保持很薄，符合项目既有分层。
- 编排逻辑（状态怎么流转、失败怎么落 FAILED）可以脱离 HTTP 单独测试。
- 首响分析服务（`analyze`）内部已对所有异常降级，编排层只需兜底归一化错误
  与意外异常，保证任务状态一定落到 DONE 或 FAILED，不留悬空态。

### 2. 三个 API 接口（都很薄）

- `POST /api/aiops/alerts/analyze`：跑完整链路，返回任务 + 结构化报告 + markdown。
- `GET  /api/aiops/tasks`：列出任务，可按状态过滤。
- `GET  /api/aiops/tasks/{id}`：查详情，含报告与事件时间线（为 Day9 SSE 铺路）。

旧的 `POST /api/aiops`（SSE 演示入口）保留不动，符合 ADR-000 不破坏现有能力的约定。

### 3. 集成测试用真实内存 SQLite + 假首响服务

- 用内存 SQLite 覆盖 `get_db`，跑**真实 SQL**，验证任务确实落库、事件确实留痕。
- 用假 `_response_service` 替换编排器内部服务，不连 LLM / Milvus，不烧 token。

## 备选方案

- **不加编排层，逻辑写进 API**：API 会变胖、难测试，且状态流转逻辑与 HTTP 耦合。否决。
- **集成测试全程 mock 数据库**：那样测不出「任务是否真落库」，失去集成测试的意义。
  改用内存 SQLite 跑真实 SQL。

## 后果

### 正面
- 首响链路从「一堆服务层零件」变成「一个能 curl 的产品」。
- 每次告警都有唯一 task_id、状态机、事件时间线，可追踪可审计。
- 6 个集成测试锁住端到端行为，含三条真实样例告警。

### 负面 / 代价
- 编排层与 repository、first_response_service 有耦合，未来若换持久化方式需同步调整。
- 目前是同步调用（请求内跑完整条链路），大流量下应改异步任务（复用 Day3 的任务表已具备条件），列入后续。

## 踩坑记录

集成测试首次运行报 `SQLite objects created in a thread can only be used in
that same thread`：FastAPI TestClient 用独立线程处理请求，而内存 SQLite 连接
绑定创建线程。解法：`create_engine(..., connect_args={"check_same_thread": False},
poolclass=StaticPool)`，让所有线程共享同一个内存连接。
