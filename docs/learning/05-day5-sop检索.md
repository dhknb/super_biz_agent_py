# 05 · Day5 — SOP 检索接入与证据溯源（★核心）

> 这是整个两周冲刺里**最值钱**的一天，也是你简历上最亮的一句话。
> 前面几天都是在打地基（模型、数据库、报告格式），今天要把项目**独有的 RAG 能力**
> 焊进告警诊断，让报告的每条结论都能「溯源」到知识库。

## 一、这天要解决什么问题

回顾一下前几天我们有了什么：
- Day2：`AlarmEvent`（标准化的告警）
- Day3：诊断任务能落库
- Day4：`FirstResponseReport`（结构化报告格式，里面有个 `Evidence` 证据类型）

但有个致命空缺：**报告里的内容从哪来？** 如果只是把告警丢给大模型让它自由发挥，
那模型只能靠「常识」瞎猜——它不知道你公司内部的排查流程（SOP）。

> SOP = Standard Operating Procedure，标准作业流程。
> 比如「CPU 高怎么排查」这种运维文档，就是一条 SOP。

这天要做的事：**告警来了，先去知识库里搜相关的 SOP，把搜到的内容作为「已验证事实」
喂给大模型，并且在报告里标明每条证据来自哪篇文档。**

这就是「证据溯源」——报告不再是模型的一面之词，而是「有据可查」。

### 为什么这是差异化的核心

你参考的那个 785-star 项目（itops-agent-platform）很强，但它是**执行型**平台
（能真的去重启服务、执行命令）。它最弱的恰恰是「知识检索」。

而你的项目本来就有一套完整的 RAG 检索链路（向量检索 + BM25 + rerank 重排）。
Day5 做的事，就是把你的长板（RAG）接到 AIOps 上——**这是那个项目做不到的**。
面试时这句话就是你的护城河：

> "我把 RAG 检索融进了 AIOps 首响，报告每条结论都能溯源到具体 SOP 文档，
> 并且严格区分'已验证事实'和'模型推断'。"

## 二、该看的文件

| 文件 | 作用 |
|------|------|
| `app/services/sop_retrieval_service.py` | SOP 检索服务：告警 → 可溯源证据 |
| `app/services/first_response_service.py` | 首响服务：把检索+大模型+报告串成闭环 |
| `app/tools/knowledge_tool.py` | 现有的 RAG 检索工具（复用，不改）|
| `app/services/vector_search_service.py` | 现有的混合检索实现（复用，不改）|

## 三、先理解「复用」——不重复造轮子

打开 `app/services/vector_search_service.py`，你会看到项目**早就有**一套检索：

```python
class VectorSearchService:
    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Document]:
        # Milvus 向量检索 + BM25 关键词检索 + RRF 融合排序
        ...
```

它接受一句查询文本，返回一批 `Document`（文档片段）。每个 `Document` 有：
- `doc.page_content`：文档的正文内容
- `doc.metadata`：元数据，比如 `_file_name`（来自哪个文件）、`h1/h2/h3`（标题层级）

**Day5 不重写这套检索**——它已经很成熟了。Day5 只做一件事：
把「告警」翻译成「查询」，再把返回的 `Document` 翻译成「带溯源的证据」。

这就是工程里的重要原则：**能复用就复用，只写增量的部分。**

## 四、逐段讲解 `sop_retrieval_service.py`

### 4.1 Protocol——「鸭子类型」的接口

```python
from typing import Any, Protocol

class DocumentLike(Protocol):
    """检索返回的文档最小接口（page_content + metadata）。"""
    page_content: str
    metadata: dict[str, Any]

class Retriever(Protocol):
    """检索器最小接口，便于依赖注入与测试替换。"""
    def retrieve_documents(self, query: str, top_k: int = 3) -> list[Any]: ...
```

**什么是 `Protocol`？**

Python 有句老话叫「鸭子类型」（duck typing）：
> 如果一个东西走起来像鸭子、叫起来像鸭子，那它就是鸭子。

`Protocol` 就是把这句话写成代码。它说：
> "我不管你是什么类，只要你有 `retrieve_documents` 这个方法，你就是一个 `Retriever`。"

**为什么要用它？** 看这个类的构造函数：

```python
class SopRetrievalService:
    def __init__(self, retriever: Retriever | None = None, top_k: int | None = None):
        self._retriever = retriever
        self._top_k = top_k if top_k is not None else config.rag_top_k
```

`retriever` 参数标注成 `Retriever`。这意味着：
- 生产环境：传真的 `vector_search_service`（会连 Milvus）
- 测试环境：传一个假的、只要有 `retrieve_documents` 方法就行

这就是**依赖注入**（下面细讲）。用 `Protocol` 而不是直接依赖 `langchain` 的 `Document` 类，
测试时就能传任意「长得像文档」的假对象，不必真的构造 langchain 对象。

### 4.2 惰性获取——避免 import 就连数据库

```python
    def _get_retriever(self) -> Retriever:
        if self._retriever is not None:
            return self._retriever
        # 默认使用项目已有的混合检索单例
        from app.services.vector_search_service import vector_search_service
        return vector_search_service
```

注意这个 `from ... import` 写在**方法内部**，不是文件顶部。这叫**惰性导入**。

**为什么？** 因为 `vector_search_service` 是个「模块级单例」——它在被 import 的那一刻
就会尝试连接 Milvus。如果写在文件顶部，那么任何人只要 `import sop_retrieval_service`
（比如测试收集时），就会触发连接 Milvus，很慢也很容易失败。

写在方法里，就只有真正调用检索时才会 import、才会连接。测试时因为传了假 retriever，
`if self._retriever is not None` 直接返回，永远不会走到这行 import。

> 这个细节，正是 Day7「测试隔离」的伏笔。好的设计是提前为测试考虑的。

### 4.3 核心方法：检索 SOP 证据

```python
    def retrieve_sop_evidence(self, alarm: AlarmEvent) -> list[Evidence]:
        query = alarm.retrieval_query()           # ① 告警 → 查询语句
        logger.info(f"SOP 检索: alert={alarm.alert_name}, query='{query}'")

        try:
            docs = self._get_retriever().retrieve_documents(query, top_k=self._top_k)  # ② 检索
        except Exception as exc:                  # ③ 检索失败不抛，降级为空
            logger.warning(f"SOP 检索失败，降级为无证据: {exc}")
            return []

        evidence: list[Evidence] = []
        for doc in docs or []:
            evidence.append(self._doc_to_evidence(doc))  # ④ 文档 → 证据

        logger.info(f"SOP 检索完成，命中 {len(evidence)} 条证据")
        return evidence
```

逐步看：

**① `alarm.retrieval_query()`** ——还记得 Day2 在 `AlarmEvent` 里写的这个方法吗？

```python
def retrieval_query(self) -> str:
    parts = [self.alert_name]
    if self.service:
        parts.append(self.service)
    if self.summary:
        parts.append(self.summary)
    return " ".join(parts)
```

它把「告警名 + 服务名 + 摘要」拼成一句自然语言，比如：
`"HighCPUUsage order-api CPU 使用率持续超过 90%"`。这句话拿去搜知识库，
就能找到最相关的 SOP。这就是 Day2 埋下、Day5 用上的伏笔。

**② 调用检索**——拿查询去 RAG 检索，返回一批文档。

**③ 降级为空**（关键的工程思维）——如果 Milvus 挂了、检索报错，
我们**不让整个诊断崩溃**，而是「降级」：返回空证据列表 `[]`。
诊断照样能继续（只是没有 SOP 证据而已）。

> 这是「优雅降级」（graceful degradation）的思想：局部失败不应该拖垮整体。
> 一个模块坏了，系统的其它部分还能提供部分价值。

**④ 文档转证据**——把每个 `Document` 转成一条 `Evidence`（下面细讲）。

### 4.4 文档 → 证据：证据溯源的落点

```python
    def _doc_to_evidence(self, doc: DocumentLike) -> Evidence:
        metadata = getattr(doc, "metadata", None) or {}
        content = getattr(doc, "page_content", "") or ""

        title = self._build_source_title(metadata)      # 拼来源标题
        excerpt = content.strip()[:_MAX_EXCERPT_CHARS]   # 截取片段（最多300字）

        return Evidence(
            type=EvidenceType.VERIFIED_FACT,   # ← 关键：标记为「已验证事实」
            source=EvidenceSource.SOP,         # ← 来源是知识库 SOP
            content=f"知识库 SOP 命中：{title}",
            source_title=title,                # ← 来自哪篇文档
            excerpt=excerpt,                   # ← 原文片段
        )
```

这是「证据溯源」的核心。看 `type=EvidenceType.VERIFIED_FACT`——
从知识库检索出来的内容，被明确标记为「**已验证事实**」，而不是模型瞎编的。

还带上了 `source_title`（哪篇文档）和 `excerpt`（原文片段），
这样报告里就能写「根据《CPU排查手册》，建议先用 top 定位进程」——**有据可查**。

`getattr(doc, "metadata", None) or {}` 是一种防御写法：
万一 doc 没有 metadata 属性，用 `None`，再 `or {}` 兜底成空字典，不会崩。

### 4.5 拼来源标题

```python
    @staticmethod
    def _build_source_title(metadata: dict[str, Any]) -> str:
        headers = [
            str(metadata[key])
            for key in ("h1", "h2", "h3")
            if metadata.get(key)
        ]
        if headers:
            file_name = metadata.get("_file_name")
            header_str = " > ".join(headers)
            return f"{file_name} · {header_str}" if file_name else header_str
        return str(metadata.get("_file_name") or "未知来源")
```

这个方法把元数据拼成人能读的来源标题：
- 优先用标题层级：`cpu_high_usage.md · CPU排查 > 定位进程`
- 没有标题就用文件名：`cpu_high_usage.md`
- 都没有就兜底：`未知来源`

`@staticmethod` 表示这是个「静态方法」——它不用访问 `self`（实例状态），
纯粹是个工具函数，放在类里只是为了归类。

## 五、什么是「依赖注入」（Dependency Injection）

这是 Day5 最重要的设计概念，也是面试高频词。

### 5.1 反面教材：硬编码依赖

假设我们这样写（**不好**）：

```python
class SopRetrievalService:
    def retrieve_sop_evidence(self, alarm):
        # 直接在方法里创建检索器
        from app.services.vector_search_service import vector_search_service
        docs = vector_search_service.retrieve_documents(...)  # 写死了！
```

问题：这个服务**永远只能用真的 `vector_search_service`**。
测试时想换成假的？做不到——它被写死在方法里了。测试就必须连真的 Milvus，
又慢又脆弱（还记得战报里那句"测试打印 API Key"的病吗？根源就是这种硬编码）。

### 5.2 正面做法：依赖注入

```python
class SopRetrievalService:
    def __init__(self, retriever: Retriever | None = None):
        self._retriever = retriever   # 依赖从外部"注入"进来
```

依赖（检索器）不是自己创建的，而是**从构造函数参数传进来**的。这就叫「注入」。

好处立竿见影：
- **生产**：`SopRetrievalService()`（不传，用默认的真检索器）
- **测试**：`SopRetrievalService(retriever=假检索器)`（传个假的）

看 Day5 的测试是怎么用的（`tests/unit/test_sop_retrieval_service.py`）：

```python
class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs
    def retrieve_documents(self, query: str, top_k: int = 3):
        return self._docs   # 直接返回预设的假文档，不连任何数据库

def test_retrieve_maps_docs_to_verified_sop_evidence():
    docs = [_doc("CPU 高排查步骤：先看 top", {"_file_name": "cpu_high_usage.md", "h1": "CPU 排查"})]
    retriever = _FakeRetriever(docs)
    service = SopRetrievalService(retriever=retriever)   # ← 注入假检索器

    evidence = service.retrieve_sop_evidence(_alarm())

    assert evidence[0].type == EvidenceType.VERIFIED_FACT
    assert "cpu_high_usage.md" in evidence[0].source_title
```

测试完全不碰 Milvus，一瞬间跑完，还能精确控制「检索返回什么」。
这就是依赖注入带来的**可测试性**。

> 一句话记住依赖注入：**「不要自己 new 依赖，让别人把依赖递给你。」**
> 这样你就能在测试时递一个假的进去。

## 六、逐段讲解 `first_response_service.py`——串成闭环

`SopRetrievalService` 只负责「检索」。真正把 `告警 → 检索 → 大模型 → 报告`
串起来的，是 `FirstResponseService`。

### 6.1 又是依赖注入

```python
class FirstResponseService:
    def __init__(
        self,
        llm: LLMLike | None = None,
        sop_service: SopRetrievalService | None = None,
    ):
        self._llm = llm
        self._sop_service = sop_service or sop_retrieval_service
```

这次注入了**两个**依赖：大模型 `llm` 和 SOP 检索服务 `sop_service`。
测试时两个都能换成假的——既不烧 token（不真的调用大模型），也不连 Milvus。

### 6.2 核心方法 `analyze`——五步闭环

```python
    async def analyze(self, alarm: AlarmEvent) -> FirstResponseReport:
        # 1. 固定步骤：检索 SOP 证据（已带溯源信息）
        sop_evidence = self._sop_service.retrieve_sop_evidence(alarm)

        # 2. 构造 prompt（告警上下文 + SOP 证据）
        user_prompt = self._build_user_prompt(alarm, sop_evidence)

        # 3. 调用 LLM
        try:
            llm = self._get_llm()
            response = await llm.ainvoke([
                ("system", _SYSTEM_PROMPT),
                ("user", user_prompt),
            ])
            raw_text = getattr(response, "content", None) or str(response)
        except Exception as exc:
            # LLM 挂了也不崩：降级报告，但保留已检索到的 SOP 证据
            logger.error(f"首响分析 LLM 调用失败: {exc}")
            report = FirstResponseReport(
                alert_summary=f"告警 {alarm.alert_name} 分析失败",
                current_judgment="LLM 调用异常，无法生成分析。",
                is_degraded=True,
                raw_text=f"LLM 调用异常: {exc}",
            )
            report.evidence = list(sop_evidence)
            return report

        # 4. 解析（失败自动降级，永不抛异常）—— 这是 Day4 写的 parse_report
        report = parse_report(raw_text)

        # 5. 合并 SOP 证据：代码注入的 SOP 证据是 VERIFIED_FACT，放最前面
        report.evidence = list(sop_evidence) + [
            ev for ev in report.evidence if ev.type != EvidenceType.VERIFIED_FACT
        ]
        return report
```

这段是全系统的「主动脉」，逐步理解：

**第1步 检索**：先拿告警去搜 SOP。注意这一步在**调用大模型之前**——
我们要把 SOP 作为「参考资料」喂给模型，而不是让模型空想。

**第2步 构造 prompt**：把告警信息和 SOP 片段组织成一段给模型的话（下面看）。

**第3步 调用大模型**：`await llm.ainvoke(...)` 异步调用。
注意这里的容错——如果大模型挂了（超时、限流），进入 `except`，
**不抛异常**，而是造一个「降级报告」。而且关键：**即便模型挂了，
之前检索到的 SOP 证据依然有价值**，所以 `report.evidence = list(sop_evidence)` 把它保留下来。

**第4步 解析**：`parse_report` 是 Day4 写的那个「永不抛异常」的解析器。
模型返回的文本可能是合法 JSON、可能是带 ```json 代码块的、也可能是一堆废话——
`parse_report` 都能处理，实在不行就降级为纯文本报告。

**第5步 合并证据**（精妙之处）：

```python
report.evidence = list(sop_evidence) + [
    ev for ev in report.evidence if ev.type != EvidenceType.VERIFIED_FACT
]
```

- `list(sop_evidence)`：我们代码检索到的 SOP 证据（可信，是真的 `VERIFIED_FACT`）
- 后半段：模型自己产出的证据里，**过滤掉**它自称的「已验证事实」

为什么过滤？因为**模型没资格自封「已验证事实」**——只有我们代码从知识库真检索到的
才算事实。模型自己说的最多算「推断」。这个列表推导式的意思是：
「SOP 证据（真事实）放前面，模型证据里只保留它标为『推断』的部分」。

这就守住了「事实与推断严格区分」的底线，防止模型「假装有依据」。

### 6.3 System Prompt——给模型立规矩

```python
_SYSTEM_PROMPT = dedent("""
    你是一线值班工程师的告警首响分析助手。给定一条告警和相关 SOP 片段，
    你要产出一份**结构化 JSON** 首响报告，帮助值班同学快速开始第一轮排查。

    严格要求：
    1. 只输出一个 JSON 对象，不要输出多余的解释文字...
    2. 所有结论必须基于给定信息。**没有证据支撑的判断，放进 pending_confirmations，
       不要写成确定根因。**
    3. recommended_checks 只能是「检查 / 观察」类动作...
       **禁止**给出重启、删除、扩容等会改变系统状态的高风险命令。
    4. 不要编造主机名、指标值、日志内容...
    JSON 结构：
    { ... }
""").strip()
```

这段 prompt 体现了几个 Day1 定的边界：
- 「结论必须基于给定信息，没证据的放待确认」→ 防幻觉
- 「只能给检查类动作，禁止重启/删除」→ 呼应 ADR-000「不碰真实命令执行」
- 「不要编造」→ 再次防幻觉

> `dedent` 是 Python 标准库函数，作用是把多行字符串前面的公共缩进去掉，
> 这样你能在代码里对齐着写、输出时又没有多余空格。

### 6.4 构造用户 prompt

```python
    def _build_user_prompt(self, alarm, sop_evidence) -> str:
        alarm_block = json.dumps({...告警字段...}, ensure_ascii=False, indent=2)

        if sop_evidence:
            sop_lines = [f"[SOP {i}] {ev.source_title}\n{ev.excerpt}"
                         for i, ev in enumerate(sop_evidence, 1)]
            sop_block = "\n\n".join(sop_lines)
        else:
            sop_block = "（未检索到相关 SOP，请基于告警本身分析，并把不确定项放进 pending_confirmations）"

        return dedent(f"""
            ## 告警事件
            {alarm_block}
            ## 相关 SOP 片段（已验证事实，可作为排查依据）
            {sop_block}
            请基于以上信息，输出结构化 JSON 首响报告。
        """).strip()
```

把告警和 SOP 组织成清晰的两段喂给模型。注意 `else` 分支——
没检索到 SOP 时，明确告诉模型「没有资料，不确定的放待确认」，再次防止瞎编。

`ensure_ascii=False` 让 JSON 里的中文正常显示（不转成 `\uXXXX`）。

## 七、这天的完整数据流

```
AlarmEvent（告警）
   │
   ▼  alarm.retrieval_query()  →  "HighCPUUsage order-api CPU超90%"
   │
   ▼  SopRetrievalService.retrieve_sop_evidence()
   │      → RAG 检索 → [Document, Document...]
   │      → 每个 Document 转成 Evidence(type=VERIFIED_FACT, source=SOP, 带溯源)
   │
   ▼  FirstResponseService.analyze()
   │      ① sop_evidence = 上面的证据
   │      ② 构造 prompt（告警 + SOP 片段）
   │      ③ 调用大模型（异步、容错）
   │      ④ parse_report 解析（Day4，可降级）
   │      ⑤ 合并证据：SOP事实优先，模型证据只留"推断"
   │
   ▼  FirstResponseReport（结构化报告，证据分事实/推断）
```

## 八、这天学到的核心概念

| 概念 | 一句话 | 在哪用 |
|------|--------|--------|
| **复用** | 能用现成的就别重写 | 直接用现有 `vector_search_service` |
| **Protocol** | 只要有某方法就算某类型（鸭子类型） | `Retriever` / `DocumentLike` |
| **依赖注入** | 不自己 new 依赖，让外部传进来 | `__init__(retriever=...)` |
| **惰性导入** | 用到时才 import，避免 import 就连库 | `_get_retriever` 内部 import |
| **优雅降级** | 局部失败不拖垮整体 | 检索失败返回 []、LLM 失败返回降级报告 |
| **证据溯源** | 每条结论标明来源，区分事实/推断 | `Evidence` 的 type + source_title |

## 九、面试怎么讲这天

> "我把项目已有的 RAG 混合检索接入了 AIOps 首响：告警来了先检索相关 SOP，
> 检索结果作为『已验证事实』喂给大模型，并在报告里溯源到具体文档。
> 关键设计有三点：一是用依赖注入 + Protocol 让检索器和大模型都可替换，
> 测试时不连真实服务；二是全链路优雅降级，检索挂了返回空证据、模型挂了返回降级报告，
> 诊断永不崩溃；三是严格区分事实与推断——只有代码检索到的算事实，
> 模型自称的『事实』会被过滤成推断，从根上防止幻觉。"

下一篇 [06-Day6-编排与API](06-day6-编排与api.md)，讲怎么把这套服务
变成真正能通过 HTTP 调用的接口。
