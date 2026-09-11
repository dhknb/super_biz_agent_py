# PDF 入库与 AIOps 面试题详解

> 本文整理两部分面试内容：
>
> 1. PDF 解析、结构化和入库的技术选型与追问
> 2. 本项目整体 AIOps 流程、状态机、证据链和降级机制
>
> 项目路径：`/home/dong/projects/super_biz_agent_py`

---

# 第一部分：PDF 入库面试题

## 1. 你们项目的 PDF 入库流程是什么？

### 标准回答

当前流程是：

```text
前端上传 PDF
  ↓
API 保存原始文件并创建入库记录
  ↓
创建后台任务并投递 RQ 队列
  ↓
Worker 取出任务
  ↓
PyMuPDF 按页提取文本
  ↓
规则识别协议、设备、检测点和阈值
  ↓
结构化数据校验
  ↓
生成 dry-run 入库计划
  ↓
等待人工确认
  ↓
通过 upsert 写入协议业务表
  ↓
任务完成
```

### 真实代码

| 模块 | 作用 |
|---|---|
| `app/api/protocol_pdf.py` | 上传、详情、确认、拒绝接口 |
| `app/services/protocol_pdf_job_queue.py` | 投递 PDF 后台任务 |
| `app/workers/protocol_pdf_worker.py` | Worker 执行任务 |
| `app/services/protocol_pdf_ingestion_service.py` | 抽取、结构化、校验和入库 |
| `app/repositories/protocol_ingestion_repository.py` | 任务和业务表读写 |
| `app/models/protocol_ingestion.py` | 入库任务模型 |
| `app/models/protocol_catalog.py` | 协议、设备、检测点、阈值模型 |

---

## 2. 为什么选择 PyMuPDF？

### 标准回答

当前协议 PDF 主要是文本型文档，目标是按页提取文本，再根据协议领域规则识别协议名称、设备、检测点和阈值。PyMuPDF 底层使用 MuPDF，读取速度快、API 简洁，同时可以保留页码信息，适合当前的批量文本提取场景。

当前不需要一开始就引入复杂版面模型，因此使用 PyMuPDF 作为主解析器。后续遇到扫描件、复杂表格或特殊版面，再按页面类型路由到 OCR 或表格解析工具。

### 真实代码

文件：`app/services/protocol_pdf_ingestion_service.py`

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

### 面试官追问：PyMuPDF 有什么缺点？

回答：

```text
1. 扫描图片型 PDF 不能直接提取文字，需要 OCR。
2. 复杂表格的行列关系不一定能直接还原。
3. 多栏、浮动文本和特殊阅读顺序可能需要额外布局处理。
4. 底层依赖 MuPDF，部署和许可证需要提前评估。
```

所以选型不是认为 PyMuPDF 适合所有 PDF，而是它最适合当前主输入类型。

---

## 3. 除了 PyMuPDF 还有哪些工具？

| 工具 | 主要能力 | 适合场景 | 主要代价 |
|---|---|---|---|
| PyMuPDF / fitz | 高性能文本、图片、元数据和页面处理 | 普通文本型 PDF、批量处理 | 扫描件需要 OCR；复杂表格需额外处理 |
| pypdf | 纯 Python，拆分、合并、旋转、加密、元数据和基础文本 | 页面操作、轻量备用解析器 | 复杂布局和表格能力较弱 |
| pdfplumber | 字符坐标、线条、矩形、表格和可视化调试 | 布局敏感、表格型 PDF | 速度通常低于 PyMuPDF |
| pdfminer.six | 低层文本和布局分析 | 字符位置、字体、阅读顺序研究 | API 复杂，开发成本较高 |
| pypdfium2 | PDFium 文本和渲染能力 | 需要 PDFium 的跨平台场景 | 生态和项目经验相对少 |
| Apache Tika | 多格式文本和元数据提取 | PDF、Word、Excel、PPT 统一接入 | 需要 Java 或 Tika Server，部署较重 |
| Unstructured | 元素级切分、OCR、版面和表格处理 | 通用文档 RAG、复杂版面 | 依赖多，模型推理更耗资源 |
| Camelot | PDF 表格抽取，输出 DataFrame | 文本型表格 PDF | 不直接处理扫描图片 PDF |
| Tabula / tabula-py | 表格抽取，输出 DataFrame、CSV、JSON | 规则表格 | 依赖 Java，复杂表格需调参 |
| OCRmyPDF + Tesseract | 给扫描 PDF 增加可搜索文本层 | 扫描件 OCR 预处理 | OCR 速度和识别准确率受图片质量影响 |
| 云端 Document AI | OCR、票据、表单和复杂表格识别 | 高精度企业文档 | 按量收费、依赖网络和配额 |

---

## 4. 为什么不用 pypdf？

### 标准回答

pypdf 是纯 Python，依赖轻，页面合并、拆分、旋转和加密处理很方便。当前项目更关注批量文本抽取和页级处理，PyMuPDF 在性能和文本提取接口方面更合适，因此 pypdf 更适合作为页面操作工具、轻量备用解析器或元数据读取工具。

---

## 5. 为什么不用 pdfplumber？

### 标准回答

pdfplumber 对字符坐标、线条、矩形和表格分析更细，适合布局敏感型文档。但当前协议 PDF 首先需要识别文本、设备、测点和阈值，PyMuPDF 已经满足主路径。遇到复杂表格时，可以让 pdfplumber 或 Camelot 专门处理表格区域。

---

## 6. 为什么不用 Unstructured？

### 标准回答

Unstructured 的能力更完整，支持 `fast`、`hi_res`、`ocr_only` 等策略，适合通用文档平台。但它依赖更多，版面模型会带来额外耗时和资源消耗。当前协议文档类型较明确，因此先使用轻量、确定性更高的 PyMuPDF；后续按页面特征路由到 OCR 或版面解析。

---

## 7. 扫描型 PDF 怎么处理？

### 标准回答

先按页检查文本抽取结果。如果某页文字为空或字符密度很低，就把它标记为 OCR 候选页。普通文本页继续走 PyMuPDF，扫描页走 OCRmyPDF、Tesseract、PaddleOCR 或 Unstructured OCR。

推荐路线：

```text
页面检测
  ├── 文本页 → PyMuPDF
  ├── 扫描页 → OCRmyPDF / Tesseract / PaddleOCR
  ├── 表格页 → Camelot / pdfplumber
  └── 复杂版面 → Unstructured hi_res
```

OCR 结果要保留：

```text
页码
原图引用
OCR 文本
识别置信度
解析器名称
```

---

## 8. 如何判断 PDF 是否需要 OCR？

可以先做基础判断：

```python
page_text = page.get_text("text") or ""

if len(page_text.strip()) < MIN_TEXT_LENGTH:
    mark_as_ocr_candidate()
```

生产环境还应综合：

```text
文本字符数
页面图片数量
文本面积占比
中文识别比例
异常替换字符比例
页面是否存在混合图片和文字
```

不能只判断整本 PDF 是否为空，因为混合型 PDF 可能只有部分页面需要 OCR。

---

## 9. 表格识别怎么做？

### 标准回答

文本型表格优先使用 Camelot 或 pdfplumber。带边框的表格适合 Camelot 的 `lattice` 模式，依靠空白间距分列的表格适合 `stream` 模式。

扫描表格应先 OCR，再做表格结构识别。表格结果不能直接写入业务表，需要经过字段映射、类型校验、重复检查和人工确认。

---

## 10. 为什么不把整本 PDF 直接交给大模型？

### 标准回答

整本 PDF 直接发送会带来：

```text
1. 输入 Token 和费用增加
2. 长上下文导致延迟变高
3. 页眉、页脚和表格列容易混淆
4. 难以定位事实来源页码
5. 模型输出结构不稳定
```

工程上应先用解析器提取，再按页、章节或语义切分。模型负责处理复杂语义和歧义字段，规则负责确定性校验。

---

## 11. 为什么要有 dry-run？

### 标准回答

PDF 解析存在误识别风险。dry-run 先生成“准备执行哪些数据库操作”的计划，但不真正写业务表。人工可以先检查协议、设备、检测点和阈值，确认后再执行写入。

对应代码：

```python
dry_run_plan = self.build_dry_run_plan(
    structured,
    validation,
)
```

计划中会列出：

```text
upsert_protocol
map_or_create_device
upsert_detection_point
upsert_threshold_rule
```

---

## 12. 为什么需要人工确认？

协议字段一旦写错，后续检测规则可能全部错误。自动解析负责提高效率，人工确认负责拦截高影响错误。

确认接口：

```text
POST /api/protocol-pdfs/{ingestion_id}/confirm
```

确认后记录：

```text
confirmed_by
confirmed_at
state_trace
```

---

## 13. 第三方解析器挂掉怎么办？

### 标准回答

可以设计分级解析器：

```text
Primary：PyMuPDF
Fallback 1：pypdf
Fallback 2：OCRmyPDF + Tesseract
Fallback 3：pdfplumber / Unstructured
```

切换条件按错误类型判断：

```text
导入失败 → 备用解析器
文本为空 → OCR 路线
表格字段缺失 → 表格解析路线
文件损坏 → 标记人工处理
```

所有解析器统一输出：

```python
@dataclass(frozen=True)
class ExtractedPdf:
    text: str
    pages: list[dict]
```

当前服务已经预留依赖注入：

```python
class ProtocolPdfIngestionService:
    def __init__(self, extractor=None):
        self.extractor = extractor or PdfTextExtractor()
```

所以业务层不需要知道底层到底使用哪一种解析器。

---

## 14. 如何保证 PDF 数据可追溯？

每个字段最好保留：

```text
source_page
source_excerpt
source_parser
content_hash
```

当前文本抽取已经保留页码：

```python
pages.append({
    "page": index,
    "text": page_text,
})
```

并拼接为：

```text
[page 1]
...

[page 2]
...
```

这样人工可以定位结构化字段来自原 PDF 的哪一页。

---

## 15. 如何保证数据库写入一致性？

### 标准回答

协议、设备、检测点和阈值写入使用同一数据库事务。任一关键步骤失败就回滚，重复写入使用 upsert，入库任务使用 `ingestion_id` 和 `content_hash` 建立幂等关系。

写入阶段可以记录：

```text
protocol_saved
equipment_mapped
points_saved
thresholds_saved
```

---

## 16. 如何评估 PDF 解析质量？

不能只看任务是否成功，还应建立标准样本集，评估：

```text
文本抽取准确率
页码来源完整率
协议名称准确率
设备型号准确率
检测点准确率
阈值准确率
表格行列准确率
OCR 字符错误率
人工修改率
平均处理时延
单页处理成本
```

协议入库最重要的指标是：

```text
字段准确率
来源页码完整率
人工退回率
重复入库率
```

---

## 17. PyMuPDF 的许可证怎么回答？

### 标准回答

PyMuPDF 使用 AGPL 或商业许可证模式。内部工具和对外分发产品的许可证要求不同，因此正式项目需要在技术选型阶段同步做许可证审查。如果许可证约束较严格，可以评估 pypdf、pdfplumber、pypdfium2 或商业文档服务。

这类回答比单纯说“性能好”更完整，因为许可证也是生产选型的一部分。

---

## 18. 面试时 1 分钟 PDF 总结

> 我们的 PDF 入库先区分文档类型。当前协议 PDF 以文本型文档为主，因此使用 PyMuPDF 进行页级文本提取，同时保留页码来源。提取结果不会直接写数据库，而是先转换成协议、设备、检测点和阈值等结构化草稿，再进行字段校验，生成 dry-run 操作计划，最后经过人工确认后通过 upsert 写入业务表。PyMuPDF 不是所有场景的唯一方案：扫描 PDF 路由到 OCR，复杂表格可以使用 Camelot 或 pdfplumber，复杂版面可以使用 Unstructured，pypdf 则适合作为轻量页面处理或备用解析器。解析器通过统一接口接入，所以替换底层工具不会影响上层入库流程。

---

# 第二部分：AIOps 整体流程面试题

## 19. 你们项目的 AIOps 整体流程是什么？

### 标准回答

项目采用“固定主流程 + 模型负责理解和组织”的半固定架构：

```text
前端 AI Ops 按钮 / Prometheus 告警
  ↓
POST /api/aiops/alerts/analyze
  ↓
normalize_alarm()
  ↓
AlarmEvent
  ↓
创建诊断任务 NEW
  ↓
PLANNING：构造告警上下文
  ↓
RETRIEVING：检索 SOP 证据
  ↓
DIAGNOSING：调用 LLM 生成结构化报告
  ↓
parse_report()
  ↓
合并可溯源 SOP 证据
  ↓
保存报告并进入 DONE
  ↓
返回 JSON 和 Markdown
```

真实入口：`app/api/aiops.py`

```python
@router.post("/aiops/alerts/analyze", status_code=201)
async def analyze_alert(request, db):
    task, report = await alert_diagnosis_orchestrator.run(
        db,
        payload=request.payload,
        source=request.source,
        session_id=request.session_id,
    )
```

---

## 20. 为什么要设计 AlarmEvent？

不同告警来源的字段结构不同：

```text
手工告警：alert_name、severity、service
Prometheus：labels.alertname、labels.instance、annotations.summary
Zabbix：disaster、high、average 等级别
```

因此先统一为：

```python
class AlarmEvent(BaseModel):
```

后续 SOP 检索、报告生成和任务持久化只处理 `AlarmEvent`，不用在每一层重复判断来源。

文件：`app/models/aiops.py`

---

## 21. 手工告警和 Prometheus 告警怎样归一化？

### 手工告警

```python
alert_name = (
    payload.get("alert_name")
    or payload.get("alertname")
    or "UnknownAlert"
)
```

同时读取：

```text
severity
service
instance
metric_name
metric_value
summary
labels
fired_at
resolved_at
```

### Prometheus Alertmanager

```text
labels.alertname → alert_name
labels.severity → severity
labels.service 或 labels.job → service
labels.instance → instance
annotations.summary → summary
startsAt → fired_at
endsAt → resolved_at
```

系统字段会从业务标签副本中移除，剩余标签作为业务标签保存。

---

## 22. 为什么严重级别还要映射？

代码中将不同系统的名称统一为：

```python
critical
warning
info
unknown
```

例如：

```python
"crit" → CRITICAL
"fatal" → CRITICAL
"warn" → WARNING
"disaster" → CRITICAL
"high" → CRITICAL
"average" → WARNING
```

这样后续报告和告警策略不需要理解每个来源的原始枚举。

---

## 23. 当前支持哪些告警来源？

枚举中预留了：

```python
MANUAL
PROMETHEUS
ZABBIX
CLOUD
```

当前归一化分发表实际接入的是：

```python
_NORMALIZERS = {
    AlarmSource.MANUAL: _normalize_manual,
    AlarmSource.PROMETHEUS: _normalize_prometheus,
}
```

所以面试时应准确说：

> 当前 manual 和 prometheus 已实现，Zabbix 和 cloud 只是枚举预留。

---

## 24. AIOps 是 Agent 自由规划吗？

### 标准回答

当前不是完全自由规划，而是：

```text
固定主流程 + 模型局部智能
```

固定流程包括：

```text
归一化
→ 创建任务
→ SOP 检索
→ 构造 Prompt
→ 结构化输出
→ 解析和降级
→ 保存报告
```

模型负责：

```text
理解告警含义
组织首响语言
提出排查方向
区分事实和推断
```

### 追问：为什么不让模型自己决定是否检索？

回答：

```text
模型可能跳过工具调用、重复调用或遗漏来源信息。
固定检索可以保证证据链稳定，方便测试、审计和评估。
```

---

## 25. SopRetrievalService 负责什么？

文件：`app/services/sop_retrieval_service.py`

流程：

```text
AlarmEvent
  ↓
alarm.retrieval_query()
  ↓
vector_search_service.retrieve_documents()
  ↓
Document
  ↓
Evidence
```

查询由告警名、服务名和摘要组成：

```python
parts = [self.alert_name]
if self.service:
    parts.append(self.service)
if self.summary:
    parts.append(self.summary)
return " ".join(parts)
```

---

## 26. 为什么检索返回 Evidence 和 RetrievalStatus 两个值？

返回值：

```python
tuple[list[Evidence], RetrievalStatus]
```

状态有：

```text
OK：检索成功并命中证据
EMPTY：检索正常，但知识库确实没有相关内容
FAILED：Milvus、网络或检索服务异常
```

如果只返回空列表：

```text
Milvus 挂了 → []
知识库没有 → []
```

两种情况会被混淆，导致系统把基础设施故障误说成知识库缺文档。因此必须保留状态。

---

## 27. 为什么要把文档转换为 Evidence？

代码类似：

```python
Evidence(
    type=EvidenceType.VERIFIED_FACT,
    source=EvidenceSource.SOP,
    content=f"知识库 SOP 命中：{title}",
    source_title=title,
    excerpt=excerpt,
)
```

报告中明确区分：

```text
VERIFIED_FACT：来自 SOP、监控或日志的可复核事实
MODEL_INFERENCE：模型基于事实产生的推断
```

这比单纯把文档拼进 Prompt 更容易审计、展示和评估。

---

## 28. 当前报告中的 CPU 值是实时采集的吗？

### 准确回答

当前演示页面中的 CPU 值来自前端固定数据：

```javascript
metric_name: 'cpu_usage_percent',
metric_value: 90,
```

后端将它归一化为：

```python
AlarmEvent.metric_value
```

当前 AIOps 首响链路主要检索 SOP，并没有在报告生成时现场查询 Prometheus。因此这个 90 是告警 payload 中的值，不等于后端刚刚采集的实时指标。

### 生产扩展

```text
AlarmEvent
  → SOP 检索
  → Prometheus 查询
  → 日志查询
  → 服务状态查询
  → 统一 Evidence
  → 生成报告
```

---

## 29. FirstResponseService 负责什么？

文件：`app/services/first_response_service.py`

```text
AlarmEvent
  → SOP 检索
  → 构造 Prompt
  → LLM 输出 JSON
  → parse_report()
  → 合并 SOP 证据
  → 形成 FirstResponseReport
```

它是首响分析的核心服务。

LLM 和 SOP 检索器都使用依赖注入：

```python
def __init__(self, llm=None, sop_service=None):
    self._llm = llm
    self._sop_service = sop_service or sop_retrieval_service
```

测试时可以传入假 LLM 和假检索器，不连接真实模型和 Milvus。

---

## 30. 报告包含哪些字段？

`FirstResponseReport` 位于：`app/models/aiops_report.py`

```text
alert_summary：告警摘要
current_judgment：当前判断
severity_assessment：严重性评估
recommended_checks：建议排障项
evidence：证据列表
root_cause_hypotheses：根因假设
pending_confirmations：待确认项
risk_notes：风险和升级建议
is_degraded：是否降级
degrade_reason：降级原因
raw_text：降级时保留的原始文本
```

报告可以通过：

```python
report.model_dump(mode="json")
```

返回结构化 JSON，也可以通过：

```python
report.to_markdown()
```

生成前端展示文本。

---

## 31. 为什么根因字段叫 hypotheses？

首响阶段通常只有：

```text
告警名称
部分指标
少量 SOP
```

这些信息适合提出排查方向，不足以确认最终根因。因此使用：

```text
root_cause_hypotheses：根因假设
pending_confirmations：待确认项
```

而不是直接使用 `confirmed_root_cause`，避免模型把推测写成事实。

---

## 32. AIOps 任务有哪些状态？

代码位置：`app/models/aiops_diagnosis.py`

```text
NEW
  ↓
PLANNING
  ↓
RETRIEVING
  ↓
DIAGNOSING
  ↓
DONE
```

异常路径：

```text
任意阶段 → FAILED
```

状态定义：

| 状态 | 含义 |
|---|---|
| `NEW` | 任务已创建，尚未开始 |
| `PLANNING` | 构造告警上下文 |
| `RETRIEVING` | 正在检索 SOP 和证据 |
| `DIAGNOSING` | 正在生成报告 |
| `DONE` | 报告已保存 |
| `FAILED` | 未处理异常导致流程失败 |

---

## 33. 为什么 retrieved 和 diagnosed 是事件，不是状态？

状态表示：

```text
当前正在做什么
```

事件表示：

```text
某件事已经发生
```

因此：

```text
RETRIEVING：正在检索
retrieved 事件：检索结束，记录耗时和命中数
DIAGNOSING：正在生成
diagnosed 事件：报告生成结束
DONE：任务最终完成
```

这样可以避免状态数量膨胀，同时保留完整时间线。

---

## 34. record_phase 做了什么？

文件：`app/repositories/aiops_diagnosis_repository.py`

```python
def record_phase(
    self,
    task,
    *,
    phase,
    status=None,
    message=None,
    payload=None,
):
```

它在一次事务中完成：

```text
推进任务状态
  +
新增时间线事件
  +
一次 commit
```

避免出现：

```text
任务状态已经是 retrieving
但时间线上没有 retrieving 事件
```

状态和事件必须同时成功或同时回滚。

---

## 35. on_phase 回调是什么？

`FirstResponseService.analyze()` 接收：

```python
on_phase: PhaseCallback | None = None
```

编排层传入：

```python
report = await self._response_service.analyze(
    alarm,
    on_phase=self._make_phase_handler(repo, task),
)
```

服务内部在真实阶段边界发出：

```python
emit(PHASE_RETRIEVING, {})
emit(PHASE_RETRIEVED, {
    "duration_ms": retrieve_ms,
    "sop_hit_count": len(sop_evidence),
    "retrieval_status": sop_status.value,
})
emit(PHASE_DIAGNOSING, {
    "prompt_tokens": estimate_tokens(user_prompt),
})
emit(PHASE_DIAGNOSED, {
    "duration_ms": generate_ms,
    "is_degraded": report.is_degraded,
})
```

### 面试官追问：为什么用回调而不用生成器？

回答：

> 当前调用方只需要一份最终报告，回调可以额外记录阶段而不改变返回值。如果改为异步生成器，调用方需要同时处理事件、最终报告和生成器结束语义，改动更大。回调是对现有接口的兼容式扩展。

---

## 36. 为什么报告必须结构化输出 JSON？

### 标准回答

JSON 有利于：

```text
Pydantic 校验
数据库 JSON 字段保存
前端按字段渲染
评测脚本量化评分
事实和推断分层
```

模型 Prompt 要求只输出一个 JSON 对象，解析失败时由 `parse_report()` 进入纯文本降级。

---

## 37. 模型输出不是合法 JSON 怎么办？

代码：`app/models/aiops_report.py`

```text
1. 查找 ```json 代码块
2. 查找文本中的 JSON 对象
3. json.loads()
4. Pydantic model_validate()
```

任一步失败就调用 `_degraded(raw_text)`，返回：

```python
FirstResponseReport(
    alert_summary="（结构化解析失败，见原始输出）",
    current_judgment="模型未按结构化格式返回，已降级为纯文本。",
    is_degraded=True,
    degrade_reason=DegradeReason.PARSE_FAILED.value,
    raw_text=raw_text,
)
```

---

## 38. LLM 失败和解析失败有什么区别？

| 场景 | 含义 | 降级原因 |
|---|---|---|
| 模型超时 | 模型没有及时返回 | `llm_timeout` |
| 模型 5xx、鉴权或配额异常 | 模型调用失败 | `llm_error` |
| 返回普通文本或错误 JSON | 输出格式不符合约定 | `parse_failed` |
| Milvus 或检索链路异常 | 基础设施故障 | `retrieval_failed` |
| 检索正常但没有文档 | 知识库覆盖不足 | `retrieval_empty` |

不同原因对应不同处理方向，不能只保存一个 `is_degraded=True`。

---

## 39. LLM 失败后为什么还保留 SOP 证据？

代码：

```python
report.evidence = list(sop_evidence)
```

即使模型调用失败，已经成功检索到的 SOP 仍然有价值。报告至少可以保留：

```text
告警基本信息
SOP 证据
模型调用失败原因
后续人工排查提示
```

---

## 40. 如何控制 AIOps 报告幻觉？

当前采用四层控制：

```text
1. Prompt 约束：只能基于告警和 SOP
2. Evidence 类型区分：事实和推断使用不同枚举
3. 结构化字段：根因使用 hypotheses，未知内容进入 pending_confirmations
4. 解析和降级：模型输出不符合约定时保留原文，不让链路崩溃
```

代码注入的 SOP 证据会排在模型自产证据前面：

```python
report.evidence = list(sop_evidence) + [
    ev for ev in report.evidence
    if ev.type != EvidenceType.VERIFIED_FACT
]
```

这样模型不能通过自己写一个“verified_fact”标签伪造真实来源。

---

## 41. 为什么 recommended_checks 只允许只读检查？

Prompt 约束：

```text
recommended_checks 只能是检查、观察类动作
```

允许：

```text
查看最近 10 分钟错误日志
检查 CPU、内存和负载
确认最近发布记录
查看连接数和慢请求
```

不自动执行：

```text
重启
删除
扩容
修改配置
```

需要人工判断的动作通过：

```python
requires_human=True
```

进行标记。

---

## 42. 检索失败和知识库没有文档一样吗？

不一样。

```text
RetrievalStatus.EMPTY
  = 检索服务正常，但知识库没有内容

RetrievalStatus.FAILED
  = 检索服务本身不可用
```

报告中的措辞也不同：

```text
EMPTY：未检索到相关 SOP，可以考虑补充文档
FAILED：检索服务不可用，不代表知识库没有 SOP，应先检查 Milvus
```

这是防止错误排障方向的关键设计。

---

## 43. AIOps 是否使用 SSE？

当前接口：

```text
POST /api/aiops/alerts/analyze
```

是普通 JSON 接口，不是 SSE。阶段过程通过：

```text
on_phase 回调
→ aiops_diagnosis_events
```

项目中高精度 RAG 的：

```text
POST /api/chat_v2_stream
```

才使用 SSE。

如果未来需要实时显示 AIOps 阶段，可以在现有事件表上增加：

```text
任务轮询
WebSocket
SSE
```

---

## 44. 当前 AIOps 任务是后台异步任务吗？

准确回答：

> 当前任务和状态已经持久化，但 `FirstResponseService.analyze()` 仍在当前 HTTP 请求中执行，接口会等待报告生成完成后返回。它具备任务化的数据基础，但还不是像 PDF 入库那样由 RQ Worker 完全异步执行。

生产化改造可以是：

```text
API 创建 NEW 任务
  ↓
投递 AIOps 队列
  ↓
立即返回 task_id
  ↓
Worker 执行诊断
  ↓
更新状态、报告和事件
  ↓
前端轮询或订阅事件
```

---

## 45. 如何理解任务失败和报告降级？

### 报告降级

服务已经捕获问题，仍然产出一个报告：

```text
report.is_degraded = True
report.degrade_reason = ...
```

例如：

```text
模型失败，但 SOP 证据仍然保留
```

任务仍可能保存为 `DONE`。

### 任务失败

编排层遇到未处理异常：

```python
except Exception as exc:
    repo.mark_failed(
        task,
        error_message=str(exc),
    )
    raise
```

任务进入 `FAILED`。

区别：

```text
降级：任务完成，但能力减少
失败：流程没有产出可交付报告
```

---

## 46. 为什么使用统一 DegradeReason？

文件：`app/core/errors.py`

当前原因包括：

```text
llm_timeout
llm_error
parse_failed
retrieval_failed
retrieval_empty
partial_retrieval
evidence_insufficient
circuit_open
total_budget_exceeded
feature_disabled
```

每个原因对应不同修复方向：

| 原因 | 排查方向 |
|---|---|
| `llm_timeout` | 模型延迟、超时配置、模型选择 |
| `llm_error` | 配额、鉴权、上游 5xx |
| `parse_failed` | Prompt、JSON 解析和输出约束 |
| `retrieval_failed` | Milvus、网络和检索服务 |
| `retrieval_empty` | 知识库内容覆盖 |
| `evidence_insufficient` | 补证据或人工确认 |
| `circuit_open` | 等待熔断冷却和探针恢复 |
| `feature_disabled` | 检查运行时开关 |

---

## 47. 为什么使用熔断器？

如果下游持续故障，没有熔断器时每个请求都会等待完整超时。

熔断器流程：

```text
连续失败达到阈值
  ↓
OPEN
  ↓
后续请求快速失败
  ↓
冷却结束进入 HALF_OPEN
  ↓
探测成功后恢复
```

当前共用：

```python
llm_breaker
```

SOP 检索使用：

```python
retrieval_breaker
```

---

## 48. 为什么 planning 在 analyze 之前设置？

编排层在调用分析服务前已经完成：

```text
告警归一化
任务创建
告警上下文准备
```

因此可以真实推进：

```python
repo.record_phase(
    task,
    phase="planning",
    status=DiagnosisTaskStatus.PLANNING,
    message="构造告警上下文",
)
```

而 retrieving 和 diagnosing 由分析服务在真实阶段开始时通过 `on_phase` 回调推进，时间线更加准确。

---

## 49. 为什么不能在编排层预先推进所有状态？

如果提前设置：

```text
planning
retrieving
diagnosing
```

时间线可能变成：

```text
10:00:00 planning
10:00:00 retrieving
10:00:00 diagnosing
10:00:40 done
```

看不出 40 秒花在哪里。

正确做法：

```text
检索真正开始 → retrieving
检索结束 → retrieved + duration_ms
模型真正开始 → diagnosing
模型结束 → diagnosed + duration_ms
```

---

## 50. AIOps 评测指标有哪些？

评估入口：`tests/eval/run_aiops_eval.py`

主要指标：

```text
SOP 命中率
报告要点覆盖率
幻觉受控率
```

当前默认评测使用：

```text
_StubLLM
_StubSop
```

所以它是离线结构基线，主要验证：

```text
评测管道是否正确
评分函数是否正确
报告结构是否覆盖预期
```

不应把离线 Stub 分数直接说成真实线上模型质量。

---

## 51. 如何设计真实 AIOps 评测集？

每条样本可以包含：

```json
{
  "case_id": "cpu-001",
  "source": "prometheus",
  "payload": {},
  "expected_sop_keywords": ["CPU", "top", "进程"],
  "expected_report_points": [
    "查看进程占用",
    "检查最近发布",
    "确认实例范围"
  ],
  "forbidden_claims": [
    "已经确定是内存泄漏"
  ]
}
```

评估：

```text
归一化字段是否正确
SOP 是否命中
排查要点是否覆盖
根因是否保持假设口径
无证据内容是否放入待确认项
故障时降级原因是否准确
```

---

## 52. 如果接入 Zabbix，怎么改？

步骤：

```text
1. 编写 _normalize_zabbix()
2. 注册到 _NORMALIZERS
3. 增加 Zabbix fixture
4. 增加归一化单测
5. 检查 severity 映射
6. 检查时间字段和资源字段
```

因为后续服务只依赖 `AlarmEvent`，所以 SOP 检索和报告生成不需要重写。

---

## 53. 如果接入 Prometheus 实时指标和日志，Evidence 怎么设计？

监控证据：

```python
Evidence(
    type=EvidenceType.VERIFIED_FACT,
    source=EvidenceSource.METRICS,
    content="CPU 使用率为 90%",
    source_title="Prometheus",
    excerpt="cpu_usage_percent{service='data-sync-service'} 90",
)
```

日志证据：

```python
Evidence(
    type=EvidenceType.VERIFIED_FACT,
    source=EvidenceSource.LOGS,
    content="最近 10 分钟出现连接超时",
    source_title="日志平台",
    excerpt="timeout connecting to upstream",
)
```

模型推断：

```python
Evidence(
    type=EvidenceType.MODEL_INFERENCE,
    source=EvidenceSource.MODEL,
    content="可能与批处理任务并发有关",
)
```

---

## 54. 如果告警重复触发，如何去重？

增加：

```text
alert fingerprint
幂等键
去重时间窗口
任务状态检查
```

可以使用：

```text
alert_id / fingerprint + 时间窗口
```

处理规则：

```text
窗口内相同告警只创建一次任务
已有 RUNNING 任务时复用
已 DONE 任务按策略返回旧报告或新建版本
```

当前模型中已有 `alert_id` 字段，可作为外部系统 fingerprint。

---

## 55. 如果 CPU、内存、延迟告警同时出现怎么办？

可以增加 Incident 聚合器，按照以下维度关联：

```text
service
instance
cluster
fired_at
告警 fingerprint
```

流程变成：

```text
多条 AlarmEvent
  ↓
事件关联
  ↓
Incident
  ↓
一份综合首响报告
```

当前版本是一条告警对应一份首响报告，多告警聚合属于后续扩展。

---

## 56. 面试官问“你们真的执行重启命令了吗？”

准确回答：

> 当前首响服务只生成检查和观察类动作，`command_hint` 也只是只读命令提示，不会自动执行重启、删除、扩容或配置修改。高风险动作需要人工确认，首响阶段只负责提供证据和排查顺序。

---

## 57. 面试官问“报告里的根因是真的吗？”

准确回答：

> 首响报告的根因属于假设方向，最终根因需要结合实时监控、日志、变更记录和人工确认。系统通过 `root_cause_hypotheses` 和 `pending_confirmations` 区分假设与待确认信息。

---

## 58. 面试官问“模型知道 CPU 是 90% 吗？”

准确回答：

> 当前演示环境中，90 来自前端发送的 `metric_value`。后端只是把它归一化到 `AlarmEvent` 并传给模型。当前 AIOps 链路没有现场查询 Prometheus，所以这个值是告警 payload 中的值，不应描述成后端实时采集值。

---

## 59. 面试官问“LLM 挂了任务一定 FAILED 吗？”

准确回答：

> 不一定。`FirstResponseService` 会捕获 LLM 异常，保留已有 SOP 证据并生成降级报告，编排层可以继续保存任务。只有编排层遇到未处理异常时，才通过 `mark_failed()` 将任务置为 FAILED。

---

## 60. 面试时 1 分钟 AIOps 总结

> 我们的 AIOps 首响链路采用固定流程和局部模型智能。告警首先按来源归一化为统一的 `AlarmEvent`，然后创建可追踪的诊断任务。系统根据告警名称、服务和摘要检索 SOP，并把命中文档转换为带来源标题和片段的 `VERIFIED_FACT` 证据。模型基于告警和 SOP 输出结构化 JSON，代码使用 Pydantic 解析，并将根因假设、待确认项、风险说明和推荐检查分开保存。流程通过任务状态机和阶段事件记录进度，使用 `RetrievalStatus` 区分检索失败和知识库为空，并通过统一 `DegradeReason`、熔断、重试和降级报告保证故障时仍能提供可用的首响结果。

---

# 三、最终项目链路速记

## PDF 入库

```text
上传
→ 文件落盘
→ 建任务
→ RQ Worker
→ PyMuPDF / OCR / 表格解析
→ 结构化
→ 校验
→ dry-run
→ 人工确认
→ upsert 入库
```

## AIOps 首响

```text
告警接入
→ AlarmEvent 归一化
→ 创建诊断任务
→ planning
→ SOP 检索
→ Evidence 溯源
→ LLM 结构化报告
→ JSON 解析
→ 降级处理
→ 事件和报告落库
→ 返回 JSON / Markdown
```

## 最关键的工程思想

```text
解析与业务解耦
证据与推断分离
状态与事件一致
失败原因可分类
故障时可降级
原始输入可审计
结果可评测
```

