# 06 · Day6 编排层与 API — 把零件串成能调的接口

> 该看的文件：
> - `app/services/alert_diagnosis_orchestrator.py`（编排层）
> - `app/api/aiops.py`（第 164 行往后，新增的三个接口）
> - `tests/integration/test_aiops_alerts_api.py`（集成测试）
> - `tests/fixtures/aiops_alerts/*.json`（三条样例告警）

---

## 1. 这天要解决什么问题

Day2-5 造好了一堆**零件**：

- Day2：`AlarmEvent`（告警长啥样）
- Day3：`AiopsDiagnosisRepository`（怎么存任务）
- Day4：`FirstResponseReport`（报告长啥样）
- Day5：`FirstResponseService`（怎么分析）

但这些零件**还串不起来**，外部也调不到。Day6 要做两件事：

1. **编排层**：把「归一化 → 建任务 → 状态流转 → 分析 → 存报告」串成一条线。
2. **API 层**：暴露成 HTTP 接口，外部能 POST 一条告警进来。

---

## 2. 为什么要单独一个「编排层」

你可能会问：为什么不直接在 API 里把这些零件串起来？

因为**分层职责**（回顾 Day1）：

- **API 层**只该管：收 HTTP 请求、校验参数、返回 JSON。它不该懂业务流程。
- **编排层（orchestrator）**管：业务流程的先后顺序（先建任务，再流转状态，最后存报告）。
- **repository 层**管：单条数据库操作。

如果把流程写进 API，API 就会变得又长又难测。抽出编排层后：
- API 只有几行（调用编排器 → 返回结果）。
- 编排逻辑可以脱离 HTTP 单独测试。

这个「编排层」在企业项目里也叫 **service 层**、**use-case 层** 或 **application 层**。

---

## 3. 编排层代码逐段讲

打开 `app/services/alert_diagnosis_orchestrator.py`。

### 3.1 类的骨架与依赖注入

```python
class AlertDiagnosisOrchestrator:
    def __init__(self, response_service: FirstResponseService | None = None):
        self._response_service = response_service or first_response_service
```

又是**依赖注入**（Day5 讲过）。编排器依赖「首响分析服务」，但不写死——
测试时可以塞一个假的进来，不真调 LLM。

### 3.2 `run()` 方法：完整跑一条链路

```python
async def run(
    self,
    db: Session,
    *,
    payload: dict[str, Any],
    source: AlarmSource | str = AlarmSource.MANUAL,
    session_id: str | None = None,
) -> tuple[AiopsDiagnosisTask, FirstResponseReport]:
```

- `db: Session`：数据库会话，从 API 层传进来（谁调用谁给数据库连接）。
- `payload`：告警原始数据。
- `source`：告警来源（决定用哪个归一化器）。
- 返回 `tuple[任务, 报告]`：既给你落库的任务对象，也给你报告对象。

**第一步：归一化**

```python
alarm: AlarmEvent = normalize_alarm(payload, source)
repo = AiopsDiagnosisRepository(db)
```

把原始 payload 变成标准 `AlarmEvent`（Day2 的能力）。归一化失败会抛
`ValueError`（比如未知来源），这个异常会一路抛到 API 层转成 HTTP 400。

**第二步：建任务**

```python
task = repo.create_task(
    alarm_event=alarm.model_dump(mode="json"),
    alert_name=alarm.alert_name,
    severity=alarm.severity.value,
    source=alarm.source.value,
    alert_id=str(payload.get("alert_id")) if payload.get("alert_id") else None,
    session_id=session_id,
)
```

- `alarm.model_dump(mode="json")`：把 Pydantic 模型转成**纯 JSON 可序列化的 dict**
  （`mode="json"` 会把 datetime、枚举都转成字符串），存进数据库的 JSON 字段。
- 任务创建后状态是 `NEW`（Day3）。

**第三步：状态流转 + 事件留痕（在 try 里）**

```python
try:
    repo.advance_status(task, DiagnosisTaskStatus.PLANNING)
    repo.add_event(task.id, phase="planning", message="构造告警上下文")

    repo.advance_status(task, DiagnosisTaskStatus.RETRIEVING)
    repo.add_event(task.id, phase="retrieving", message="检索 SOP 与采集证据")

    repo.advance_status(task, DiagnosisTaskStatus.DIAGNOSING)
    repo.add_event(task.id, phase="diagnosing", message="生成结构化首响报告")
```

每推进一个阶段：①更新任务状态（`advance_status`，Day3 的状态机会校验合法性）
②记一条事件（`add_event`，为 Day9 的时间线铺路）。

**第四步：真正分析**

```python
    report = await self._response_service.analyze(alarm)
```

调用 Day5 的首响服务。`await` 是因为它内部要调 LLM（异步 IO）。

**第五步：存报告，任务转 DONE**

```python
    repo.save_report(
        task,
        report=report.model_dump(mode="json"),
        summary=report.alert_summary,
    )
    repo.add_event(task.id, phase="done", message="首响报告已生成",
                   payload={"is_degraded": report.is_degraded})
    return task, report
```

**兜底：任何异常都让任务落到 FAILED**

```python
except Exception as exc:
    logger.exception(f"诊断任务失败: task_id={task.id}")
    repo.mark_failed(task, error_message=str(exc))
    repo.add_event(task.id, phase="failed", message=str(exc))
    raise
```

**为什么重要**：如果中途崩了却不 mark_failed，任务就永远卡在 `DIAGNOSING`
状态（悬空态），没人知道它其实已经死了。这里保证：**任务状态最终一定是
DONE 或 FAILED，不留中间态。**

> 注意 `first_response_service.analyze` 内部已经把 LLM/检索异常都降级了（Day5），
> 所以这里的 `except` 主要兜底「数据库操作失败」这类意外。双重保险。

---

## 4. API 层代码逐段讲

打开 `app/api/aiops.py`，看第 164 行往后新增的部分。

### 4.1 请求模型

```python
class AnalyzeAlertRequest(BaseModel):
    payload: dict[str, Any] = Field(description="告警原始数据（单条）")
    source: str = Field(default="manual", description="告警来源：manual / prometheus")
    session_id: Optional[str] = Field(default=None, description="可选会话 ID")
```

FastAPI 会自动把 HTTP 请求体的 JSON 解析进这个模型，并校验类型。
如果客户端传的 JSON 缺 `payload` 字段，FastAPI 自动返回 422 错误，
根本不会进到你的函数——**参数校验前置**，这是 FastAPI + Pydantic 的威力。

### 4.2 同步分析接口

```python
@router.post("/aiops/alerts/analyze", status_code=201)
async def analyze_alert(
    request: AnalyzeAlertRequest,
    db: Session = Depends(get_db),
):
    try:
        task, report = await alert_diagnosis_orchestrator.run(
            db,
            payload=request.payload,
            source=request.source,
            session_id=request.session_id,
        )
    except ValueError as exc:  # 未知来源等归一化错误
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "code": 201,
        "message": "success",
        "data": {
            "task_id": task.id,
            "status": task.status.value,
            "report": report.model_dump(mode="json"),
            "report_markdown": report.to_markdown(),
        },
    }
```

看这个接口有多**薄**——它只做三件事：

1. `Depends(get_db)`：**依赖注入**拿到数据库会话（FastAPI 的招牌功能，
   它会自动调用 `get_db()` 生成器，请求结束后自动关闭连接）。
2. 调编排器 `run()`。
3. 把结果包成 JSON 返回。同时返回 `report`（给程序用）和 `report_markdown`
   （给人看），很贴心。

`status_code=201`：HTTP 201 表示「已创建」（Created），比笼统的 200 更语义化——
因为这个接口确实在数据库里创建了一个新任务。

`except ValueError → HTTPException(400)`：把业务异常翻译成 HTTP 状态码。
归一化失败（比如传了未知来源）是「客户端的错」，所以是 4xx（400 Bad Request）。

### 4.3 查询接口

```python
@router.get("/aiops/tasks")
async def list_diagnosis_tasks(status: Optional[str] = None, limit: int = 50, ...):
    ...

@router.get("/aiops/tasks/{task_id}")
async def get_diagnosis_task(task_id: str, ...):
    task = repo.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="诊断任务不存在")
    ...
```

- 列表接口支持按 `status` 过滤、`limit` 限制数量。
- 详情接口找不到任务返回 404（这是「资源不存在」的标准状态码）。
- 详情里还带上了 `events`（事件时间线）——这些是 Day3 埋下的留痕。

**REST 风格小结**：

| 操作 | HTTP 方法 | 路径 | 状态码 |
|------|-----------|------|--------|
| 创建分析 | POST | `/aiops/alerts/analyze` | 201 |
| 列表 | GET | `/aiops/tasks` | 200 |
| 查详情 | GET | `/aiops/tasks/{id}` | 200 / 404 |

---

## 5. 集成测试：那个多线程坑

打开 `tests/integration/test_aiops_alerts_api.py`。

### 5.1 什么是集成测试

- **单元测试**：测一个函数/类（Day2-5 那些）。
- **集成测试**：测多个组件**串起来**能不能工作（API → 编排 → repository → DB）。

### 5.2 用真实内存数据库

```python
from sqlalchemy.pool import StaticPool

engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
```

这里踩了一个**经典的坑**，值得你记住：

- `TestClient`（FastAPI 的测试客户端）发请求时，**在独立线程里跑**。
- 而 SQLite 的内存数据库连接**默认绑定在创建它的线程**上，别的线程碰它会报错：
  `SQLite objects created in a thread can only be used in that same thread`。

**两个修复参数**：
- `check_same_thread=False`：允许跨线程使用连接。
- `poolclass=StaticPool`：让所有线程**共享同一个**内存连接（否则每个线程开新连接，
  内存数据库是「一个连接一个库」，数据就对不上了）。

> 这个坑面试能讲：「用 FastAPI TestClient 测内存 SQLite 时遇到多线程报错，
> 根因是 TestClient 独立线程 + SQLite 连接线程绑定，用 StaticPool + 
> check_same_thread=False 解决。」

### 5.3 依赖覆盖（override）

```python
def override_get_db():
    db = session_factory()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
```

FastAPI 允许**替换依赖**。生产环境 `get_db` 连的是 PostgreSQL，
测试时用 `override_get_db` 换成内存 SQLite。这样测试不碰真实数据库，
又能跑真实 SQL。这是 FastAPI 依赖注入设计的精髓。

### 5.4 参数化测试三条告警

```python
@pytest.mark.parametrize(
    "fixture_name",
    ["01_cpu_high.json", "02_disk_full.json", "03_service_unavailable.json"],
)
def test_analyze_alert_produces_report(integration_client, fixture_name):
    fixture = _load(fixture_name)
    response = integration_client.post(
        "/api/aiops/alerts/analyze",
        json={"payload": fixture["payload"], "source": fixture["source"]},
    )
    assert response.status_code == 201
    ...
```

`@pytest.mark.parametrize`：**同一个测试函数，用三组不同数据各跑一遍**。
不用写三个几乎一样的测试函数。这三条 fixture 分别是 CPU/磁盘/服务不可用，
其中第三条是 Prometheus 格式（测归一化的多来源能力）。

---

## 6. 三条样例告警 fixture

打开 `tests/fixtures/aiops_alerts/01_cpu_high.json`：

```json
{
  "source": "manual",
  "payload": {
    "alert_name": "HighCPUUsage",
    "severity": "critical",
    "service": "order-api",
    ...
  },
  "expected_sop_keywords": ["CPU", "top", "进程"],
  "expected_report_points": ["CPU", "进程", "日志"]
}
```

- `source` + `payload`：就是接口要的请求体。
- `expected_sop_keywords` / `expected_report_points`：**Day8 评估要用的期望值**
  （这条告警理应命中含「CPU/top/进程」的 SOP，报告理应覆盖这些要点）。

这就是「测试数据即评估集」的巧思——同一份 fixture，Day6 拿来测接口，
Day8 拿来评估质量。

---

## 7. 本篇小结

| 概念 | 一句话 |
|------|--------|
| 编排层 | 管业务流程顺序，让 API 保持薄 |
| `Depends(get_db)` | FastAPI 依赖注入，自动给数据库连接并善后 |
| HTTP 状态码 | 201 创建、400 客户端错、404 找不到 |
| 集成测试 | 测多组件串起来，用内存 SQLite + 依赖覆盖 |
| StaticPool 坑 | TestClient 多线程 + SQLite 线程绑定 |
| parametrize | 一个测试函数跑多组数据 |

下一篇 Day7，讲蕾姆打的最硬一仗：**API Key 泄漏的根治**。
