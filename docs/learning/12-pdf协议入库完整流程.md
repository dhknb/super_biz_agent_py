# PDF 协议入库完整流程

本文对应当前项目的实际实现，目标是解释一份协议 PDF 从上传到最终写入协议库的完整链路，并说明“上传后一直显示待处理”的原因与修复方式。

## 1. 功能目标

协议 PDF 通常包含协议名称、适用设备、检测项和阈值，例如：

```text
适用设备：变压器
绕组温度 >= 85℃
绝缘电阻 >= 500MΩ
```

系统不会直接把未经检查的抽取结果写进协议库，而是采用：

```text
上传 PDF
  → 保存原文件
  → 创建入库记录和后台任务
  → 异步抽取文本
  → 生成结构化草稿
  → 规则校验
  → 生成 dry-run 写入计划
  → 人工确认
  → 写入协议目录表
```

这条链路与 AIOps 首响分析独立；PDF 入库的结果是协议目录数据，AIOps/RAG 后续可以使用这些知识。

## 2. 涉及的文件

| 文件 | 职责 |
|---|---|
| `static/protocol_pdf.html` | PDF 入库页面结构 |
| `static/protocol_pdf.js` | 上传、列表、详情、确认和拒绝 |
| `app/api/protocol_pdf.py` | HTTP API |
| `app/core/task_queue.py` | Redis 连接和 RQ 队列定义 |
| `app/services/protocol_pdf_job_queue.py` | 把任务放入 RQ |
| `app/workers/protocol_pdf_worker.py` | 独立进程中的任务入口 |
| `app/services/protocol_pdf_ingestion_service.py` | PDF 抽取、结构化、校验、dry-run |
| `app/repositories/protocol_ingestion_repository.py` | 入库状态和协议表读写 |
| `app/models/protocol_ingestion.py` | 入库记录、任务和状态枚举 |
| `app/models/protocol_catalog.py` | 最终协议目录表 |
| `Makefile` | 本地 RQ Worker 启停命令 |

## 3. 前端上传

页面脚本是 `static/protocol_pdf.js`。

### 3.1 选择文件

```javascript
this.fileInput?.addEventListener("change", (event) =>
    this.handleFileSelect(event)
);
```

用户选择文件后，`handleFileSelect()` 做两步：

```javascript
this.validateFile(file);
await this.uploadFile(file);
```

`validateFile()` 只做前端体验层校验：扩展名必须是 `.pdf`，文件不能超过 30MB。后端仍会再次校验，不能把前端校验当成安全边界。

### 3.2 发送 multipart 上传

```javascript
const formData = new FormData();
formData.append("file", file);

const response = await fetch(`${this.apiBaseUrl}/upload`, {
    method: "POST",
    body: formData,
});
```

`FormData` 对应后端的 `UploadFile`。API 返回 HTTP `202`，表示“请求已接收，后台还在处理”，不是同步返回解析结果。

上传后前端会调用：

```javascript
await this.loadItems();
await this.loadDetail(payload.data.ingestion_id, false);
```

当前页面只有手动刷新按钮，没有持续 `setInterval` 轮询；因此后台状态变化后，需要点击“刷新列表”或重新打开页面才能看到最新状态。

## 4. 上传 API：保存文件并创建任务

文件：`app/api/protocol_pdf.py`。

### 4.1 文件校验和落盘

```python
@router.post("/upload", status_code=202)
async def upload_protocol_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    safe_filename = _sanitize_filename(file.filename)
    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 协议文件")

    content = await file.read()
    if len(content) > MAX_PDF_SIZE:
        raise HTTPException(status_code=400, detail="文件大小超过限制")
```

解释：

1. `UploadFile` 是 FastAPI 的文件上传对象。
2. `_sanitize_filename()` 清洗文件名，避免文件名携带目录跳转片段。
3. 后端重新校验扩展名和大小，防止绕过前端检查。
4. `await file.read()` 读取文件二进制内容。

保存目录由配置决定：

```python
PROTOCOL_UPLOAD_DIR = Path(config.upload_dir) / "protocol_pdfs"
```

如果同名文件已经存在，系统使用文件内容 SHA-256 的前 8 位生成新名字：

```python
if file_path.exists():
    stem = file_path.stem
    suffix = file_path.suffix
    file_path = PROTOCOL_UPLOAD_DIR / (
        f"{stem}_{hashlib.sha256(content).hexdigest()[:8]}{suffix}"
    )
```

随后写入磁盘：

```python
file_path.write_bytes(content)
```

数据库同时保存完整 `content_hash`，用于审计和内容识别。

### 4.2 创建业务记录与执行记录

```python
repo = ProtocolIngestionRepository(db)
ingestion = repo.create_ingestion(
    filename=file_path.name,
    original_filename=file.filename,
    file_path=str(file_path.resolve()),
    file_size=len(content),
    content_hash=hashlib.sha256(content).hexdigest(),
)
job = repo.create_job(ingestion.id)
```

这里有两个对象：

- `ingestion`：表示“这份 PDF 的完整业务生命周期”。
- `job`：表示“某一次后台执行尝试”。

拆成两个对象后，同一份 PDF 可以在失败后重新创建执行任务，而不必丢失原始入库记录。

### 4.3 入队并返回

```python
rq_job = enqueue_protocol_pdf_ingestion_job(job.id)
repo.bind_rq_job(job.id, rq_job.id)
```

入队成功后 API 返回：

```json
{
  "code": 202,
  "message": "accepted",
  "data": {
    "ingestion_id": "...",
    "job_id": "...",
    "rq_job_id": "...",
    "status": "pending"
  }
}
```

注意：此时 `pending` 只代表入库记录刚创建；真正处理在 RQ Worker 中异步发生。

## 5. Redis 和 RQ 队列

文件：`app/core/task_queue.py`。

```python
redis_conn = Redis.from_url(config.redis_url)
index_queue = Queue(config.rq_queue_name, connection=redis_conn)
protocol_pdf_queue = Queue("protocol_pdf_ingest", connection=redis_conn)
```

当前有两条业务队列：

| 队列 | 任务 |
|---|---|
| `knowledge_index` | 普通知识库文档索引 |
| `protocol_pdf_ingest` | 协议 PDF 抽取和结构化 |

Redis 在这里是任务消息存储；RQ（Redis Queue）负责把 Python 函数和参数序列化后放入 Redis，并由 Worker 取出执行。

文件：`app/services/protocol_pdf_job_queue.py`。

```python
def enqueue_protocol_pdf_ingestion_job(job_id: str) -> Job:
    return protocol_pdf_queue.enqueue(
        run_protocol_pdf_ingestion_job,
        job_id,
        job_timeout=600,
        result_ttl=86400,
        failure_ttl=86400,
    )
```

- `run_protocol_pdf_ingestion_job`：将来由 Worker 执行的函数。
- `job_id`：传给 Worker 的数据库任务 ID。
- `job_timeout=600`：单个任务最多执行 10 分钟。
- `result_ttl`、`failure_ttl`：RQ 结果和失败信息保留 1 天。

## 6. 本次修复的队列 Bug

原来的 `Makefile` 只启动：

```make
RQ_QUEUE = knowledge_index
.venv/bin/rq worker $(RQ_QUEUE) --url redis://localhost:6379/0
```

但 PDF 入队使用的是：

```python
Queue("protocol_pdf_ingest", connection=redis_conn)
```

结果是：

```text
PDF 任务 → protocol_pdf_ingest
Worker   → knowledge_index
```

Worker 不会消费 PDF 队列，入库记录就一直是 `pending`，RQ job 则一直是 `queued`。

修复后的 `Makefile`：

```make
RQ_QUEUES = knowledge_index protocol_pdf_ingest

nohup env PYTHONPATH=. .venv/bin/rq worker $(RQ_QUEUES) \
    --url redis://localhost:6379/0 > rq_worker.log 2>&1 &
```

一个 Worker 同时监听两条队列，原有知识库索引和 PDF 入库都能被消费。启动后日志应该出现类似：

```text
Listening on knowledge_index, protocol_pdf_ingest...
```

如果旧 Worker 仍在运行，需要先停止旧进程，再执行 `make start-worker`；否则旧进程仍只监听旧队列。

还要确认 Redis 正常：

```bash
redis-cli -u redis://localhost:6379/0 ping
```

预期：

```text
PONG
```

## 7. Worker：真正执行后台任务的地方

文件：`app/workers/protocol_pdf_worker.py`。

```python
def run_protocol_pdf_ingestion_job(job_id: str) -> None:
    db = SessionLocal()
    repo = ProtocolIngestionRepository(db)
    try:
        job = repo.get_job(job_id)
        if job is None:
            logger.warning(f"协议 PDF 入库任务不存在: {job_id}")
            return

        job.status = ProtocolIngestionJobStatus.RUNNING
        job.started_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()

        protocol_pdf_ingestion_service.process_ingestion(
            db,
            job.ingestion_id,
        )

        job.status = ProtocolIngestionJobStatus.SUCCEEDED
        job.finished_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()
    except Exception as exc:
        db.rollback()
        job = repo.get_job(job_id)
        if job is not None:
            job.status = ProtocolIngestionJobStatus.FAILED
            job.error_message = str(exc)
            job.finished_at = datetime.now(UTC).replace(tzinfo=None)
            ingestion = repo.get_ingestion(job.ingestion_id)
            if ingestion is not None:
                ingestion.status = ProtocolIngestionStatus.FAILED
                ingestion.error_message = str(exc)
            db.commit()
        raise
    finally:
        db.close()
```

逐段解释：

1. Worker 是独立进程，所以自己创建 `SessionLocal()`，不能复用 FastAPI 请求中的数据库会话。
2. 先把 job 标成 `running`，记录 `started_at`。
3. 调用核心服务 `process_ingestion()`。
4. 成功后把 job 标成 `succeeded`，记录 `finished_at`。
5. 异常时 `rollback()` 撤销未提交改动，再把 job 和 ingestion 都标为 `failed` 并保存错误信息。
6. `finally` 无论成功失败都关闭数据库连接。

## 8. 核心服务：抽取、结构化、校验

文件：`app/services/protocol_pdf_ingestion_service.py`。

### 8.1 主流程

```python
def process_ingestion(self, db: Session, ingestion_id: str):
    repo = ProtocolIngestionRepository(db)
    ingestion = repo.get_ingestion(ingestion_id)
    if ingestion is None:
        raise ValueError("protocol PDF ingestion not found")

    repo.set_status(ingestion, ProtocolIngestionStatus.EXTRACTING,
                    phase="extract_pdf")
    extracted = self.extractor.extract(ingestion.file_path)

    repo.set_status(ingestion, ProtocolIngestionStatus.STRUCTURING,
                    phase="structure_with_qwen_schema")
    structured = self.build_structured_draft(
        extracted.text,
        pages=extracted.pages,
        filename=ingestion.filename,
    )

    repo.set_status(ingestion, ProtocolIngestionStatus.VALIDATING,
                    phase="validate_schema_and_rules")
    validation = self.validate_structured_data(structured)
    dry_run_plan = self.build_dry_run_plan(structured, validation)

    return repo.save_dry_run(
        ingestion,
        extracted_text=extracted.text,
        structured_data=structured,
        validation_result=validation,
        dry_run_plan=dry_run_plan,
    )
```

状态变化是：

```text
pending → extracting → structuring → validating → awaiting_confirmation
```

`save_dry_run()` 将结果存入 JSON 字段，同时把状态改为 `awaiting_confirmation`。

### 8.2 PyMuPDF 抽取文本

```python
class PdfTextExtractor:
    def extract(self, file_path: str) -> ExtractedPdf:
        import fitz
        pages = []
        with fitz.open(file_path) as doc:
            for index, page in enumerate(doc, start=1):
                page_text = page.get_text("text") or ""
                pages.append({"page": index, "text": page_text})

        text = "\n\n".join(
            f"[page {page['page']}]\n{page['text']}"
            for page in pages
            if page["text"].strip()
        )
        return ExtractedPdf(text=text, pages=pages)
```

第三方库是 **PyMuPDF**，安装包名是 `pymupdf`，导入名是 `fitz`。它负责：

```text
打开 PDF → 逐页读取文本 → 保存页码和原文
```

当前实现是文本抽取，不包含 OCR。扫描图片型 PDF 没有文本层时，抽取结果可能为空。

### 8.3 生成结构化草稿

```python
def build_structured_draft(self, text, *, pages, filename):
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and not re.fullmatch(r"\[page\s+\d+\]", line.strip(), re.IGNORECASE)
    ]
    protocol_name = lines[0] if lines else Path(filename).stem
    devices = self._extract_devices(lines)
    detection_items = self._extract_detection_items(lines, pages)

    return {
        "schema_version": "protocol_pdf_v1",
        "protocol": {
            "name": protocol_name,
            "source_filename": filename,
        },
        "devices": devices,
        "detection_items": detection_items,
    }
```

当前阶段主要是 Python 文本清洗和正则抽取，并不是调用大模型。虽然状态阶段名保留了 `structure_with_qwen_schema`，但现有实现由 `build_structured_draft()` 完成。

设备抽取识别类似：

```text
设备：变压器
型号：ABC-100
适用装置：配电柜
```

阈值抽取使用正则：

```python
threshold_pattern = re.compile(
    r"(?P<name>[\w\u4e00-\u9fff（）()/-]{2,30}).{0,12}?"
    r"(?P<op>>=|<=|≤|≥|<|>|=)"
    r"\s*(?P<value>-?\d+(?:\.\d+)?)"
    r"\s*(?P<unit>[%℃ΩA-Za-z/]+)?"
)
```

它会将运算符、数值、单位拆成：

```json
{
  "measurement_point": "绕组温度",
  "threshold": {
    "operator": ">=",
    "value": 85,
    "unit": "℃",
    "raw": "绕组温度 >= 85℃"
  },
  "confidence": 0.55,
  "needs_review": true
}
```

`_find_source_page()` 会回到每页原文中寻找该行，保存阈值来自哪一页，便于人工复核。

### 8.4 规则校验

```python
def validate_structured_data(self, structured_data):
    errors = []
    warnings = []

    if not protocol.get("name"):
        errors.append(...)
    if not devices:
        warnings.append(...)
    if not detection_items:
        warnings.append(...)
```

规则分两类：

- `errors`：阻止确认写入，例如协议名称为空、测点为空。
- `warnings`：允许人工确认，但提醒人工复核，例如没识别出设备、检测项为空、测点重复。

返回值：

```json
{
  "valid": true,
  "errors": [],
  "warnings": [],
  "needs_review": false
}
```

### 8.5 dry-run 计划

`build_dry_run_plan()` 只生成“计划”，不执行数据库写入：

```json
{
  "state_order": [
    "protocol_saved",
    "equipment_mapped",
    "points_saved",
    "thresholds_saved"
  ],
  "can_confirm": true,
  "requires_human_review": false,
  "operations": [...],
  "operation_count": 4
}
```

这样前端可以展示“准备写哪些协议、设备、测点和阈值”，人工确认后才执行。

## 9. 数据库模型

### 9.1 入库过程表

`protocol_pdf_ingestions` 保存 PDF 的业务生命周期：

```text
id、filename、original_filename、file_path、file_size、content_hash
status、current_phase、extracted_text、structured_data
validation_result、dry_run_plan、state_trace、error_message
created_at、updated_at
```

`protocol_pdf_ingestion_jobs` 保存后台执行信息：

```text
id、ingestion_id、rq_job_id、status、started_at、finished_at、error_message
```

### 9.2 最终协议目录表

确认写入后，`ProtocolIngestionRepository.write_protocol_catalog()` 使用 upsert 写入：

1. `protocol_records`：协议主记录。
2. `protocol_equipment_mappings`：协议和设备/型号的映射。
3. `protocol_detection_points`：检测点和测点名称。
4. `protocol_threshold_rules`：测点的运算符、阈值、单位和来源。

upsert 的含义是：同一协议或同一测点已存在时更新，否则创建，降低重复上传造成重复数据的风险。

## 10. 人工确认与真正写库

详情页点击“确认写入”调用：

```http
POST /api/protocol-pdfs/{ingestion_id}/confirm
```

后端 `confirm_ingestion()` 会依次检查：

```python
if ingestion.status != ProtocolIngestionStatus.AWAITING_CONFIRMATION:
    raise ValueError(...)
if not ingestion.dry_run_plan:
    raise ValueError(...)
if validation.get("errors"):
    raise ValueError(...)
```

通过后：

```python
repo.set_status(ingestion, ProtocolIngestionStatus.WRITING,
                phase="protocol_saved")
state_trace = repo.write_protocol_catalog(ingestion)
return repo.confirm_with_state_trace(...)
```

`write_protocol_catalog()` 完成四组 upsert 后，状态变为：

```text
writing → completed
```

如果不希望写入，则调用：

```http
POST /api/protocol-pdfs/{ingestion_id}/reject
```

状态变为 `rejected`，并保存拒绝人和原因。

## 11. 前端状态显示

`static/protocol_pdf.js` 将后端枚举映射为中文：

```javascript
const labels = {
    pending: "待处理",
    extracting: "抽取中",
    structuring: "结构化中",
    validating: "校验中",
    awaiting_confirmation: "待确认",
    writing: "写入中",
    completed: "已完成",
    failed: "失败",
    rejected: "已拒绝",
};
```

页面把 `pending`、`extracting`、`structuring`、`validating`、`awaiting_confirmation`、`writing` 都计入“处理中”。

目前没有自动轮询，排查时必须点击“刷新列表”后再判断状态。

## 12. 故障排查清单

### 12.1 一直是 pending

按顺序检查：

```bash
# 1. Redis
redis-cli -u redis://localhost:6379/0 ping

# 2. Worker 进程和监听队列
ps -ef | grep '[r]q worker'
tail -f rq_worker.log

# 3. Worker 日志必须包含两个队列
# knowledge_index
# protocol_pdf_ingest
```

如果只看到：

```text
Listening on knowledge_index...
```

说明是旧 Worker，停止后重新启动：

```bash
make stop-worker
make start-worker
```

### 12.2 任务进入 extracting 后失败

检查：

```text
PyMuPDF 是否安装
PDF 文件是否存在
文件路径是否可被 Worker 进程访问
PDF 是否是扫描图片且没有文本层
```

依赖：

```toml
pymupdf>=1.24.0
```

### 12.3 任务进入 awaiting_confirmation

这是正常结果，不是卡住。它表示：

```text
PDF 已抽取
结构化草稿已生成
校验已完成
dry-run 计划已保存
正在等待人工确认
```

此时查看详情，确认没有 `validation_result.errors`，再点击“确认写入”。

## 13. 使用的第三方组件总结

| 组件 | 用途 |
|---|---|
| FastAPI | 上传、查询、确认、拒绝 API |
| PyMuPDF (`fitz`) | 从 PDF 逐页抽取文本 |
| Redis | RQ 的任务消息存储 |
| RQ | 异步后台任务队列和 Worker |
| SQLAlchemy | 数据库 ORM |
| PostgreSQL | 入库任务、协议目录和阈值规则持久化 |
| Pydantic | API 请求和结构化数据校验 |
| Loguru | Worker 和服务日志 |

当前 PDF 结构化阶段主要使用 Python 正则和规则校验，不是实时调用 Qwen。`structure_with_qwen_schema` 是阶段名，实际实现入口仍是 `build_structured_draft()`。

## 14. 一句话总结

PDF 功能不是“上传后立即写协议库”，而是：

```text
FastAPI 收文件 → Redis/RQ 排队 → Worker 用 PyMuPDF 抽文本
→ 正则生成草稿 → 规则校验 → dry-run
→ 人工确认 → SQLAlchemy 写 PostgreSQL 的四类协议表
```

本次 Bug 的根因是 Worker 只监听 `knowledge_index`，却没有监听 PDF 使用的 `protocol_pdf_ingest`；`Makefile` 已改为同时监听两个队列。
