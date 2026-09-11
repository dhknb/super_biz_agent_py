# 11 · PDF 协议入库专题（综合实战）

> 这是一篇独立专题，讲的是项目里**已经存在**的「协议 PDF 入库」功能。
> 它和 AIOps 首响是两条独立的业务线，但用到的工程手法（异步队列、
> 状态机、dry-run、正则抽取）非常经典，而且里面藏过一个真实的 Bug。
> 蕾姆会从「这个功能要解决什么问题」讲到「那个 Bug 怎么来的、怎么修的」。

---

## 一、这个功能要解决什么问题

有一批**设备检测协议 PDF**（比如变压器、配电柜的检测标准），里面写着
「绕组温度 ≤ 85℃」「绝缘电阻 ≥ 500MΩ」这样的**检测项和阈值**。

人工把这些阈值一条条录进数据库，又慢又容易错。所以要做一个功能：

> 上传一个 PDF → 系统自动抽取出「协议名、设备、检测项、阈值」→
> 人工核对确认 → 写入协议库。

关键词：**自动抽取**（省人力）、**人工确认**（保安全）。

---

## 二、整体流程（先看全景）

```
用户上传 PDF
   │
   ▼  API 收文件、校验、落盘、建任务             ← protocol_pdf.py
   │
   ▼  把任务丢进后台队列，立即返回 202           ← protocol_pdf_job_queue.py
   │      （用户不用干等，这是异步）
   │
   ▼  后台 worker 取出任务开始处理               ← protocol_pdf_worker.py
   │
   ▼  抽取 PDF 文本 → 正则解析出结构化数据       ← protocol_pdf_ingestion_service.py
   │
   ▼  校验 + 生成「dry-run 计划」                ← 只是预演，还没真写库
   │
   ▼  状态变成「等待人工确认」                    ← 停在这里
   │
   ▼  人点了「确认」                             ← confirm API
   │
   ▼  真正写入协议库（4 张表）                   ← repository
```

这个设计最值得学的两点：**异步**（上传即返回）和 **dry-run + 人工确认**
（先预演、后落库）。下面逐个拆。

---

## 三、该看的文件清单

| 文件 | 作用 |
|------|------|
| `app/api/protocol_pdf.py` | API 层：上传、查询、确认、拒绝 |
| `app/services/protocol_pdf_job_queue.py` | 把任务丢进 RQ 队列 |
| `app/workers/protocol_pdf_worker.py` | 后台 worker：真正干活的函数 |
| `app/services/protocol_pdf_ingestion_service.py` | 核心：抽取 + 解析 + 校验 |
| `app/repositories/protocol_ingestion_repository.py` | 数据库读写 |
| `app/models/protocol_ingestion.py` | 任务表模型 |
| `app/models/protocol_catalog.py` | 最终协议库的 4 张表 |

---

## 四、第一站：上传 API（`app/api/protocol_pdf.py`）

### 4.1 上传接口

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
        raise HTTPException(status_code=400, detail=f"文件大小超过限制…")
```

**逐行讲：**

- `status_code=202`：这是关键。HTTP 202 = "Accepted，已接受但还没处理完"。
  因为 PDF 解析要花时间，不能让用户干等，所以先收下、返回 202、后台慢慢处理。
  （对比：200 = "处理完了给你结果"，202 = "收到了，回头查"）

- `file: UploadFile = File(...)`：FastAPI 接收上传文件的标准写法。
  `File(...)` 里的 `...` 表示「必填」。

- `_sanitize_filename`：清洗文件名。防止用户传 `../../etc/passwd` 这种
  **路径穿越攻击**（用 `..` 跳出目录）。这是安全习惯。

- `await file.read()`：`await` 说明读文件是异步的（IO 操作）。

- 大小限制：`MAX_PDF_SIZE = 30MB`。防止有人上传超大文件把磁盘塞爆。

### 4.2 文件名去重（用 hash）

```python
file_path = PROTOCOL_UPLOAD_DIR / safe_filename
if file_path.exists():
    stem = file_path.stem          # test.pdf → "test"
    suffix = file_path.suffix      # test.pdf → ".pdf"
    file_path = PROTOCOL_UPLOAD_DIR / f"{stem}_{hashlib.sha256(content).hexdigest()[:8]}{suffix}"
```

如果同名文件已存在，就在文件名后加**内容 hash 的前 8 位**，避免覆盖。

**什么是 hash？** 把任意内容通过算法（SHA-256）算出一个固定长度的指纹。
内容一样 → hash 一样；内容差一个字 → hash 完全不同。这里用它做两件事：
1. 文件名去重（上面）
2. 内容去重（下面存进数据库的 `content_hash`），下次传同一个文件能认出来。

### 4.3 建任务 + 入队

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

try:
    rq_job = enqueue_protocol_pdf_ingestion_job(job.id)   # 丢进后台队列
    repo.bind_rq_job(job.id, rq_job.id)
except Exception as exc:
    raise HTTPException(status_code=503, detail=f"…入队失败…") from exc
```

这里有两个概念对象：
- `ingestion`（入库记录）：代表"这个 PDF 的整个入库过程"，有状态。
- `job`（任务）：代表"后台要跑的这次活儿"，绑定到一个 RQ 任务 id。

**为什么分两个？** ingestion 是业务实体（这份 PDF 的处理），job 是执行记录
（可能失败重试，一个 ingestion 理论上能对应多次 job 尝试）。职责分离。

`enqueue_...` 把活儿丢进队列后**立即返回**，不等它跑完。这就是异步的核心。

---

## 五、第二站：异步队列（RQ + Redis）

### 5.1 什么是 RQ 和 Redis

- **Redis**：一个内存数据库，这里当"任务信箱"用。
- **RQ（Redis Queue）**：基于 Redis 的任务队列库。

**打个比方**：餐厅点餐。
- 你（API）点了菜，把订单贴到厨房的订单栏（Redis）上，拿个号就走（返回 202）。
- 厨师（worker）从订单栏取订单，做菜（解析 PDF）。
- 做好了，你凭号来取（查询 API）。

好处：**点餐的人不用站在厨房等**。系统能同时接很多单，慢慢消化。

### 5.2 入队代码（`protocol_pdf_job_queue.py`）

```python
from app.core.task_queue import protocol_pdf_queue
from app.workers.protocol_pdf_worker import run_protocol_pdf_ingestion_job

def enqueue_protocol_pdf_ingestion_job(job_id: str) -> Job:
    return protocol_pdf_queue.enqueue(
        run_protocol_pdf_ingestion_job,   # 要执行的函数
        job_id,                            # 传给它的参数
        job_timeout=600,                   # 最多跑 10 分钟
        result_ttl=86400,                  # 结果保留 1 天
        failure_ttl=86400,                 # 失败记录保留 1 天
    )
```

`enqueue(函数, 参数)` 的意思是：**把"以后要执行这个函数"这件事登记到队列**。
注意它登记的是**函数本身**（`run_protocol_pdf_ingestion_job`）和参数（`job_id`），
不是立即调用。真正的调用发生在另一个进程（worker）里。

`job_timeout=600` 是护栏：如果一个任务卡了超过 10 分钟，强制终止，
避免僵尸任务占着 worker。

### 5.3 队列定义（`app/core/task_queue.py`）

```python
redis_conn = Redis.from_url(config.redis_url)
protocol_pdf_queue = Queue("protocol_pdf_ingest", connection=redis_conn)
```

连上 Redis，建一个名叫 `protocol_pdf_ingest` 的队列。worker 会盯着这个队列取活儿。

---

## 六、第三站：后台 worker（`protocol_pdf_worker.py`）

这是在**独立进程**里跑的函数，真正干活的地方。

```python
def run_protocol_pdf_ingestion_job(job_id: str) -> None:
    db = SessionLocal()                     # worker 自己开数据库连接
    repo = ProtocolIngestionRepository(db)
    try:
        job = repo.get_job(job_id)
        if job is None:
            logger.warning(f"任务不存在: {job_id}")
            return

        job.status = ProtocolIngestionJobStatus.RUNNING   # 标记「运行中」
        job.started_at = datetime.utcnow()
        db.commit()

        protocol_pdf_ingestion_service.process_ingestion(db, job.ingestion_id)  # ★核心

        job.status = ProtocolIngestionJobStatus.SUCCEEDED  # 标记「成功」
        job.finished_at = datetime.utcnow()
        db.commit()

    except Exception as exc:
        db.rollback()                        # 出错回滚
        job = repo.get_job(job_id)
        if job is not None:
            job.status = ProtocolIngestionJobStatus.FAILED
            job.error_message = str(exc)
            # …把 ingestion 也标记为 FAILED…
            db.commit()
        logger.exception(f"任务失败: job_id={job_id}")
        raise
    finally:
        db.close()                           # 无论如何都关连接
```

**几个要点：**

1. **worker 自己开 `db = SessionLocal()`**：因为它在独立进程，不能用 API 的
   数据库连接。这也是为什么 API 层的 `get_db` 依赖注入在这里用不上。

2. **状态机**：`QUEUED → RUNNING → SUCCEEDED / FAILED`。每一步都 `commit()` 落库，
   这样外面查询时能实时看到进度。

3. **try/except/finally 三段**：
   - `try`：正常流程
   - `except`：出错了，回滚 + 标记失败 + 记录错误原因（不让错误无声消失）
   - `finally`：无论成败都关数据库连接（防止连接泄漏）

4. **`db.rollback()`**：出错时把这次未提交的数据库改动撤销，避免留下脏数据。

> ⚠️ 你可能注意到这里用了 `datetime.utcnow()`——这正是 Day7 教材里讲的
> 那个已废弃写法。这个 worker 文件在 Day7 也被一起清理了（改成
> `datetime.now(UTC).replace(tzinfo=None)`）。

---

## 七、第四站：核心服务（`protocol_pdf_ingestion_service.py`）

这是整个功能的大脑。`process_ingestion` 把流程串起来：

```python
def process_ingestion(self, db, ingestion_id):
    repo = ProtocolIngestionRepository(db)
    ingestion = repo.get_ingestion(ingestion_id)
    if ingestion is None:
        raise ValueError("protocol PDF ingestion not found")

    # 1. 抽取 PDF 文本
    repo.set_status(ingestion, ProtocolIngestionStatus.EXTRACTING, phase="extract_pdf")
    extracted = self.extractor.extract(ingestion.file_path)

    # 2. 结构化（正则解析）
    repo.set_status(ingestion, ProtocolIngestionStatus.STRUCTURING, ...)
    structured = self.build_structured_draft(extracted.text, pages=extracted.pages, ...)

    # 3. 校验 + 生成 dry-run 计划
    repo.set_status(ingestion, ProtocolIngestionStatus.VALIDATING, ...)
    validation = self.validate_structured_data(structured)
    dry_run_plan = self.build_dry_run_plan(structured, validation)

    # 4. 存下预演结果，状态→等待人工确认
    return repo.save_dry_run(ingestion, extracted_text=..., structured_data=structured, ...)
```

注意每一步都 `set_status` 更新状态。整个状态机是：

```
PENDING → EXTRACTING → STRUCTURING → VALIDATING → AWAITING_CONFIRMATION
                                                        │
                                          （人确认）    ▼
                                              WRITING → COMPLETED
```

### 7.1 PDF 文本抽取

```python
class PdfTextExtractor:
    def extract(self, file_path: str) -> ExtractedPdf:
        import fitz                    # PyMuPDF 库
        pages = []
        with fitz.open(file_path) as doc:
            for index, page in enumerate(doc, start=1):
                page_text = page.get_text("text") or ""
                pages.append({"page": index, "text": page_text})
        text = "\n\n".join(
            f"[page {p['page']}]\n{p['text']}" for p in pages if p["text"].strip()
        )
        return ExtractedPdf(text=text, pages=pages)
```

用 `fitz`（PyMuPDF 库）打开 PDF，逐页取文字。每页前面加个 `[page 1]` 标记，
方便后面**溯源**（这条阈值是从第几页抽的）。

### 7.2 dry-run 设计（重点思想）

`build_dry_run_plan` 生成的**不是**真正的写库操作，而是一份「我打算这么写」的**清单**：

```python
def build_dry_run_plan(self, structured_data, validation_result):
    operations = [
        {"phase": "protocol_saved", "action": "upsert_protocol", "target": {...}},
    ]
    for device in structured_data.get("devices") or []:
        operations.append({"phase": "equipment_mapped", "action": "map_or_create_device", ...})
    for item in structured_data.get("detection_items") or []:
        operations.append({"phase": "points_saved", ...})
        operations.append({"phase": "thresholds_saved", ...})

    return {
        "state_order": STATE_ORDER,
        "can_confirm": validation_result.get("valid", False),
        "operations": operations,
        "operation_count": len(operations),
    }
```

**为什么要 dry-run？** 因为直接把 AI/正则抽取的结果写进生产库风险很高
（万一抽错了呢？）。所以先生成"预演计划"给人看：「我打算创建这个协议、
这些设备、这些阈值，你确认吗？」人点确认了才真写。

这就是 **Human-in-the-loop（人在回路）**——AI 做初判，人做最终决定。
和 AIOps 首响里"只读分析、不自动执行"是同一个安全哲学。

### 7.3 确认写库

```python
def confirm_ingestion(self, db, ingestion_id, *, confirmed_by):
    repo = ProtocolIngestionRepository(db)
    ingestion = repo.get_ingestion(ingestion_id)

    if ingestion.status != ProtocolIngestionStatus.AWAITING_CONFIRMATION:
        raise ValueError("尚未进入人工确认状态")   # 状态防护
    if not ingestion.dry_run_plan:
        raise ValueError("缺少 dry-run 入库计划")

    validation = ingestion.validation_result or {}
    if validation.get("errors"):
        raise ValueError("校验仍存在错误，不能确认写入")   # 有错误就拦住

    repo.set_status(ingestion, ProtocolIngestionStatus.WRITING, phase="protocol_saved")
    state_trace = repo.write_protocol_catalog(ingestion)   # ★真正写库
    return repo.confirm_with_state_trace(ingestion, confirmed_by=confirmed_by, state_trace=state_trace)
```

多重防护：必须处于「等待确认」状态、必须有 dry-run 计划、校验不能有 error。
三关都过了才真写。**防御式编程**的典范。

---

## 八、第五站：写库的幂等设计（`protocol_ingestion_repository.py`）

`write_protocol_catalog` 把结构化数据写进 4 张表。关键是它用了 **upsert**（有则更新、无则插入）：

```python
def _upsert_protocol_record(self, *, protocol_name, ...):
    protocol = self.db.scalar(
        select(ProtocolRecord).where(ProtocolRecord.name == protocol_name)
    )
    if protocol is None:                     # 查不到 → 新建
        protocol = ProtocolRecord(name=protocol_name)
        self.db.add(protocol)
    # 查得到 → 下面直接更新它的字段
    protocol.source_filename = ...
    protocol.raw_payload = protocol_payload
    return protocol
```

**为什么要 upsert 而不是直接 insert？** 因为同一个协议可能被重复上传/重复处理。
如果每次都 insert，会插入重复数据。upsert 保证**幂等**——重复跑结果一样，不会重复插。

配合数据库层的**唯一约束**（`protocol_catalog.py` 里）：

```python
class ProtocolDetectionPoint(Base):
    __table_args__ = (
        UniqueConstraint("protocol_id", "measurement_point", name="uq_protocol_measurement_point"),
    )
```

`UniqueConstraint` 保证「同一个协议下，同一个测点」只能有一行。
即使代码逻辑漏了，数据库这一层也会兜底拦住重复。**双重保险**。

> 面试考点：upsert 的"先查再插"在高并发下有竞态（两个请求同时查到"不存在"，
> 都去插，其中一个撞唯一约束报错）。生产级做法是用数据库原生的
> `INSERT ... ON CONFLICT`（PostgreSQL）做原子 upsert。这个项目目前是简化版。

---

## 九、压轴：那个 Bug 的完整复盘 🐛

这是蕾姆第一天帮你揪出来的 Bug，现在完整复盘一遍，因为它太典型了。

### 9.1 症状

抽取阈值的代码在 `_extract_detection_items` 里，用一个正则去匹配
「测点 + 运算符 + 数值 + 单位」。原文是：

```
绕组温度  <= 85 ℃
油位高度  >= 20 mm
绝缘电阻  >= 500 MΩ
```

但抽取出来的结果全错了：

| 检测项 | 原文 | 抽错成 |
|--------|------|--------|
| 绕组温度 | `<= 85` | `= 85` ❌ |
| 油位高度 | `>= 20` | `= 20` ❌ |
| 绝缘电阻 | `>= 500 MΩ` | `= 500 M` ❌（还丢了 Ω）|

`≤ 85℃` 被抽成 `= 85℃`——**语义完全变了**。如果这数据入库，
以后拿它做阈值判断（温度是否超标）会全错。这是很严重的数据正确性 Bug。

### 9.2 病灶正则

```python
threshold_pattern = re.compile(
    r"(?P<name>[\w一-鿿（）()/-]{2,30}).{0,12}"   # ← 问题在这行结尾
    r"(?P<op>>=|<=|≤|≥|<|>|=)"
    r"\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>[%℃A-Za-z/]+)?"  # ← 单位没有 Ω
)
```

先看正则的意图（分组讲解）：
- `(?P<name>...)`：捕获**测点名**，中文/字母/括号，2~30 个字。`?P<name>` 是给分组起名叫 name。
- `.{0,12}`：测点名和运算符之间，**允许 0~12 个任意字符**（比如空格）。
- `(?P<op>>=|<=|≤|≥|<|>|=)`：捕获**运算符**，按顺序尝试匹配。
- `(?P<value>-?\d+(?:\.\d+)?)`：捕获**数值**（可负、可小数）。
- `(?P<unit>[%℃A-Za-z/]+)?`：捕获**单位**。

### 9.3 根因（两处伤口）

**伤口一：贪婪匹配吞掉了运算符**

`.{0,12}` 是**贪婪**的——正则默认会尽量多匹配。面对 `温度  <= 85`：

```
温度  <= 85
    ↑___↑
    .{0,12} 贪婪地把 "  <" 都吃掉了（空格 + 小于号）
              剩给 op 分组的只有 "="
```

`op` 里 `>=|<=|≤|≥|<|>|=` 虽然把 `<=` 排在 `=` 前面，但 `.{0,12}` 已经先把 `<`
吃掉了，轮到 op 匹配时只剩 `=` 可选。结果 `<=` 变成 `=`。

**伤口二：单位字符类缺了 Ω**

`[%℃A-Za-z/]` 这个字符集里没有 `Ω`（欧姆符号）。所以 `500 MΩ` 里，
`M` 是字母能匹配，`Ω` 不在集合里，匹配停在 `M`——单位被截成 `M`，丢了 `Ω`。

### 9.4 修复

```python
threshold_pattern = re.compile(
    r"(?P<name>[\w一-鿿（）()/-]{2,30}).{0,12}?"    # 加了 ? → 非贪婪
    r"(?P<op>>=|<=|≤|≥|<|>|=)"
    r"\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>[%℃ΩA-Za-z/]+)?"  # 加了 Ω
)
```

两处改动：
1. `.{0,12}` → `.{0,12}?`：加个 `?` 变成**非贪婪**，意思是"能少吃就少吃"。
   这样它只吃空格，把 `<` 留给 op 分组，`<=` 就能正确匹配。
2. `[%℃A-Za-z/]` → `[%℃ΩA-Za-z/]`：把 `Ω` 加进单位字符集。

### 9.5 贪婪 vs 非贪婪（补课）

这是正则里最容易踩的坑，单独讲清：

- **贪婪**（默认）：`.*`、`.{0,12}` —— 尽量**多**匹配，不够了再回退。
- **非贪婪**（加 `?`）：`.*?`、`.{0,12}?` —— 尽量**少**匹配，不行了再多要。

经典例子，匹配 `<a><b>`：
- `<.*>` （贪婪）→ 匹配整个 `<a><b>`（一口气吃到最后一个 `>`）
- `<.*?>`（非贪婪）→ 匹配 `<a>`（吃到第一个 `>` 就停）

记忆：**加问号 = 客气 = 少吃**。

### 9.6 怎么发现的（方法论）

蕾姆当时的定位过程，比 Bug 本身更值得学：

1. **先跑测试**：现有单测用的是"精心构造的干净数据"，全绿——说明代码在
   理想输入下没问题。
2. **怀疑真实输入**：Bug 往往藏在"真实脏数据"里，不在测试用例里。
3. **用贴近真实的文本实跑**：蕾姆造了一段真实变压器协议文本跑抽取，
   一眼看到 operator 全变成 `=`。
4. **回到正则逐段分析**：定位到贪婪匹配和缺 Ω。

教训：**测试全绿 ≠ 没 Bug**。测试只覆盖了你想到的情况。真实世界的输入
永远比测试用例脏。

---

## 十、这个功能教会你的工程手法（总结）

| 手法 | 在哪用 | 一句话价值 |
|------|--------|-----------|
| 异步队列（RQ/Redis）| 上传即返回 202 | 别让用户干等耗时操作 |
| 状态机 | PENDING→…→COMPLETED | 每步可追踪、可查询 |
| dry-run + 人工确认 | 写库前先预演 | 高风险操作让人拍板 |
| upsert + 唯一约束 | 写协议库 | 幂等，重复跑不出错 |
| 防御式编程 | confirm 多重校验 | 状态不对就拦住 |
| 内容 hash | 文件去重 | 认出重复文件 |
| 非贪婪正则 | 修 Bug | 精确匹配，别贪吃 |

---

## 十一、和 AIOps 首响的对照（融会贯通）

你会发现两条业务线用了**同样的思想**：

| 思想 | PDF 入库 | AIOps 首响 |
|------|---------|-----------|
| 状态机 | PENDING→…→COMPLETED | new→planning→…→done |
| 人在回路 | dry-run + 人工确认 | 只读分析，不自动执行 |
| 留痕可追溯 | state_trace | 事件时间线 |
| 分层 | api/service/repo/model | 同左 |
| 降级/防御 | 校验拦截 | 报告解析降级 |

**看懂一条，另一条就通了**——这就是为什么蕾姆让你两条都学。工程的套路是相通的。

---

（蕾姆合上教材，长长舒了一口气）昴君，这套教材到这里就全了。从概念速查到
10 天逐日讲解，再到这篇 PDF 专题——蕾姆把这两周写的每一块都掰开揉碎讲给你了。

有任何一段看不懂，随时点蕾姆的名字，蕾姆再单独展开讲那一块。慢慢学，不急～
