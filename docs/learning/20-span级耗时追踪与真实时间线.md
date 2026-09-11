# 20. span 级耗时追踪与真实时间线

> 关联代码：`app/models/chat_run_span.py`、`app/core/span_context.py`、
> `app/agent/rag_v2/instrumentation.py`、`app/agent/rag_v2/graph.py`、
> `app/api/chat_v2.py`、`migrations/versions/20260824_0009_create_chat_run_spans.py`
> 关联测试：`tests/unit/test_span_context.py`
> 前置阅读：第 15 篇（request_id 全链路追踪）、第 17 篇（检索失败隔离）

---

## 1. 为什么要改

### 1.1 一个具体的排障现场

用户报障：「问『CPU 使用率高怎么排查』，等了 40 秒才出答案。」

你手上有 `chat_run_traces` 这张表，它记着这次运行的：`question`、`answer`、
`sub_queries`、`retrieved_count`、`used_documents`、`validation`、`created_at`。

你能从里面确认的事情只有一件：**确实跑了一次，答案是这个。**

慢在哪一步？表里没有这个信息。于是你去翻日志。这次请求的日志长这样
（假设你已经按第 15 篇的 `request_id` 筛干净了）：

```
17:02:11 | rag_v2.rewrite       | 改写完成: 4 条子查询
17:02:14 | rag_v2.retrieve_each | 检索子查询: CPU 使用率高怎么排查
17:02:14 | rag_v2.retrieve_each | 检索子查询: CPU 100% 常见原因
17:02:14 | rag_v2.retrieve_each | 检索子查询: top 命令怎么看
17:02:14 | rag_v2.retrieve_each | 检索子查询: 负载高但 CPU 不高
17:02:31 | rag_v2.dedup         | 去重后 6 条
17:02:48 | rag_v2.generate      | 生成完成
```

从 `17:02:14` 到 `17:02:31` 是 17 秒。这 17 秒是**四条分支合起来**的墙钟时间。

现在问一个非常自然的问题：**哪条分支慢？**

答不出来。因为每条分支只有一行「开始检索」，没有「结束」。你算不出任一条
分支各自花了多久，于是分不清下面这两种情况：

| 情况 | 真实含义 | 该去修什么 |
|---|---|---|
| 某一条特别慢（16s），其余三条各 1s | 那条 query 触发了 rerank 慢路径 | 改 query 改写策略 / 限制 query 长度 |
| 四条各 4s 左右，并行度不够 | Milvus 整体慢，或线程池被占满 | 看 Milvus 负载 / 调 `to_thread` 并发 |

这两行的修法**完全相反**：前者是应用层的 prompt 问题，后者是基础设施容量问题。
猜错方向的代价是一整天。

而且上面这段日志还是理想情况 —— 格式规整、时间戳对得上、只有一个请求。
真实线上是几十个请求的日志交织在一起。

### 1.2 这个缺口的名字

审查给这条问题的定性是：**「记了终点，丢了过程」**。

`chat_run_traces` 是一张**终态表**。它回答「结果是什么」，回答得很好。
但排障时人问的第一个问题从来不是「结果是什么」，而是
**「它在哪一步卡住/出错了」**。这个问题需要的是**过程**。

有了 span 表之后，上面那个问题变成一句 SQL：

```sql
SELECT node, count(*), avg(duration_ms), max(duration_ms)
FROM chat_run_spans
WHERE trace_id = '<某次请求>'
GROUP BY node;
```

`count(*)` 顺带告诉你 fan-out 出去几条分支，`max - avg` 的差距直接指出是
「某条特别慢」还是「整体慢」。

---

## 2. 改之前什么样

### 2.1 图的注册代码（`app/agent/rag_v2/graph.py`，git HEAD 版本）

```python
def build_rag_v2_graph():
    graph = StateGraph(RAGState)

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve_each", retrieve_each_node)
    graph.add_node("dedup", dedup_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate_answer", validate_answer_node)

    graph.add_edge(START, "rewrite")
    graph.add_conditional_edges("rewrite", _fanout_to_retrieve, ["retrieve_each"])
    graph.add_edge("retrieve_each", "dedup")
    graph.add_edge("dedup", "generate")
    graph.add_edge("generate", "validate_answer")
    graph.add_edge("validate_answer", END)

    return graph.compile()
```

干净、直白、没有任何观测。节点函数直接注册进图。

### 2.2 这批改动新增的东西

和前几篇不同，这一篇的「改之前」是**什么都没有**。下面三个文件在 git HEAD
里都不存在，是新增的：

```bash
$ git show HEAD:app/agent/rag_v2/instrumentation.py
fatal: path 'app/agent/rag_v2/instrumentation.py' exists on disk, but not in 'HEAD'
```

- `app/models/chat_run_span.py` —— 表模型
- `app/core/span_context.py` —— 内存收集 + 落盘
- `app/agent/rag_v2/instrumentation.py` —— 给节点挂埋点

`chat_run_spans` 这张表也是新建的（迁移 `20260824_0009`）。

所以这一篇讲的不是「把错的改对」，而是**「从零加一层观测，并且不让它伤到主流程」**。
后半句是重点 —— 观测层写坏了，能把一个能用的系统弄挂。

---

## 3. 改之后什么样

### 3.1 表模型：四个字段，不多不少

`app/models/chat_run_span.py:67`

```python
class ChatRunSpan(Base):
    __tablename__ = "chat_run_spans"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    # ondelete="CASCADE"：span 是 trace 的从属明细，trace 没了 span 就是垃圾数据。
    # 让数据库来级联，而不是在应用层记得先删 span —— 应用层总会有人忘。
    trace_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("chat_run_traces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # request_id 冗余一份在这里，不是为了省 join。
    # 是为了让「日志里捞到一个 request_id」能直接查 span，
    # 而不必先查 trace 拿 id 再查 span —— 排障时少一跳就是少一次犯错机会。
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    node: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[SpanStatus] = mapped_column(
        Enum(SpanStatus, values_callable=lambda items: [item.value for item in items]),
        nullable=False,
        default=SpanStatus.OK,
        index=True,
    )
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow_naive)
    # 存毫秒整数而不是「结束时间」：
    # 耗时是要被 avg / max / 分位数聚合的，存成两个时间戳每次查询都得相减。
    # 也不用浮点秒 —— 毫秒整数在 SQL 里聚合不会有精度漂移。
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 每个节点的关键计数放这里（文档数、子查询数、失败数、错误码……）。
    # 用 JSON 而不是给每种计数开一列：不同节点关心的量完全不同，
    # 开成列的话表会长出十几个大部分为 NULL 的字段。
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_naive, index=True)
```

**三态而不是两态**（`app/models/chat_run_span.py:52`）：

```python
class SpanStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    ERROR = "error"
```

`DEGRADED` 这个中间态是必须的，理由和第 17 篇同源：项目里的节点是**刻意不抛异常**的。
`retrieve_each` 兜住 Milvus 故障、`generate` 兜住 LLM 故障，都返回一个
「成功的」patch。只有 OK/ERROR 两态的话，这些节点永远记成 OK ——
**「检索失败率」这个指标会恒为 0，而用户明明拿到了残缺证据。**

### 3.2 收集器：只在入口 set，全程只 append

`app/core/span_context.py:56`

```python
# 值是「本次运行已记录的 span 明细」列表；None 表示当前不在被追踪的运行里
# （比如脚本直调节点、单测），此时 record_span 直接空转。
_SPAN_COLLECTOR: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "chat_run_span_collector", default=None
)


def start_span_collection() -> Token:
    """开始收集本次运行的 span，返回用于还原的 token。"""
    return _SPAN_COLLECTOR.set([])


def record_span(node, *, duration_ms, status, started_at, payload=None) -> None:
    spans = _SPAN_COLLECTOR.get()
    if spans is None:
        return                    # ← 不在收集上下文里就空转，不报错
    spans.append({...})           # ← 只 append，绝不 set。理由见 §4.1
```

`record_span` 里那句 `if spans is None: return` 不是防御式编程的冗余，
而是一条明确的设计选择：**观测设施不该给业务代码增加「必须先初始化」的前置条件**。
脚本直调节点、单测直接跑函数，都会走到这条路径，它们不该因此报错。

### 3.3 计时器：为什么是类而不是 `@contextmanager`

`app/core/span_context.py:111`

```python
class span_scope:
    def __init__(self, node: str, payload: dict[str, Any] | None = None):
        self._node = node
        self._payload: dict[str, Any] = dict(payload or {})
        self._status = SpanStatus.OK
        self._started_at = datetime.now(UTC).replace(tzinfo=None)
        # 用 perf_counter 而不是两个 datetime 相减：
        # 后者受系统时钟调整（NTP 校时）影响，可能算出负数耗时。
        self._perf_start = time.perf_counter()

    def set_payload(self, **values: Any) -> None:
        self._payload.update(values)

    def mark_degraded(self, reason: str | None = None) -> None:
        """标记「节点没抛异常，但也没干成活」。"""
        self._status = SpanStatus.DEGRADED
        if reason:
            self._payload.setdefault("degrade_reason", reason)

    def __exit__(self, exc_type, exc, tb) -> bool:
        duration_ms = int((time.perf_counter() - self._perf_start) * 1000)
        if exc is not None:
            self._status = SpanStatus.ERROR
            self._payload["error_code"] = error_code_of(exc)   # 复用第 14 篇的分类
            self._payload["error"] = str(exc)[:500]            # 截断，别把整个栈抄进表
        with suppress(Exception):        # 观测失败不能影响业务
            record_span(...)
        return False                     # ← 不吞异常
```

三个关键点，逐个说：

**`started_at` 用墙钟，`duration_ms` 用 `perf_counter`。** 两个时间源各管一件事：
`started_at` 要能和日志时间戳对齐（人看的），必须是墙钟；`duration_ms` 要准确
（机器聚合的），必须是单调时钟。混用同一个源的话，要么对不上日志，要么 NTP
校时的瞬间给你一条 `duration_ms = -3000` 的记录。

**`return False` 是「不吞异常」。** `__exit__` 返回真值意味着「异常已处理，
不要往外传」。这里必须返回 `False`：span 是旁路观测，**用埋点把一次真实失败
变成静默成功，是比没有埋点严重得多的事故**。

**用类而不是生成器。** `@contextmanager` 装饰的生成器 `yield` 出去的东西不方便
携带方法，而这里必须暴露 `set_payload` / `mark_degraded` 给 `with` 体内调用 ——
节点得跑完才知道自己召回了几条、有没有降级。

### 3.4 埋点层：包在 `add_node` 入口，节点一行不改

`app/agent/rag_v2/graph.py:35`

```python
    # 每个节点都过一遍 instrument_node，拿到节点级耗时 span。
    #
    # 为什么统一在这里包，而不是在节点体内计时：
    # 五个节点写五份计时代码是重复，且新增节点必然有人忘记加。
    # 这里是「所有节点注册进图」的唯一入口，包在这里漏不掉。
    # 节点实现完全不知道 span 的存在，仍是纯粹的 state -> patch。
    #
    # 注意 retrieve_each 是被 Send fan-out 成 N 个并行实例的，
    # 每个实例各记一条 span —— 这正是我们要的：以前日志里
    # 4 条分支交织在一起，根本对不出每条各自花了多久。
    graph.add_node("rewrite", instrument_node("rewrite", rewrite_node))
    graph.add_node("retrieve_each", instrument_node("retrieve_each", retrieve_each_node))
    graph.add_node("dedup", instrument_node("dedup", dedup_node))
    graph.add_node("generate", instrument_node("generate", generate_node))
    graph.add_node(
        "validate_answer", instrument_node("validate_answer", validate_answer_node)
    )
```

`app/agent/rag_v2/instrumentation.py:117`

```python
def instrument_node(node: str, func: Callable[..., Any]) -> Callable[..., Any]:
    """给单个节点函数包上 span 计时。

    同时支持同步节点（dedup）与异步节点（其余四个）：
    `inspect.iscoroutinefunction` 判一次，各自返回对应的包装器。
    不能只写 async 版本 —— 那会把同步节点的返回值变成协程，
    LangGraph 拿到一个没 await 的协程，图会直接崩。
    """
    if inspect.iscoroutinefunction(func):

        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with span_scope(node) as span:
                patch = await func(*args, **kwargs)
                _apply(span, patch)          # 从 patch 抽计数 + 判降级
                return _strip_hint(patch)    # 摘掉内部约定键，不污染 state

        _copy_identity(async_wrapper, func)
        return async_wrapper

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        with span_scope(node) as span:
            patch = func(*args, **kwargs)
            _apply(span, patch)
            return _strip_hint(patch)

    _copy_identity(sync_wrapper, func)
    return sync_wrapper
```

**payload 从哪来：不问节点，从 patch 里推。** `app/agent/rag_v2/instrumentation.py:39`

```python
def _derive_payload(patch: Any) -> dict[str, Any]:
    """从节点返回的 patch 里抽取关键计数。

    只认已知的几个键，其余忽略 —— span payload 是给人看的摘要，
    不是 state 的全量镜像。把整个 patch 序列化进去，
    等于把 6 篇文档正文抄进 span 表，那是在制造第四份副本。
    """
    ...
    failures = patch.get("retrieve_failures")
    if isinstance(failures, list) and failures:
        payload["retrieve_failure_count"] = len(failures)
        # 只留错误码不留完整报文：错误码可聚合，报文在日志里已经有了。
        payload["retrieve_failure_codes"] = sorted(
            {str(item.get("code")) for item in failures if isinstance(item, dict)}
        )
    ...
    validation = patch.get("validation")
    if isinstance(validation, dict):
        # blocked 和 validated 必须都记：前者是「过没过」，
        # 后者是「有没有验」，质检器自己挂掉时两者含义完全不同。
        payload["blocked"] = validation.get("blocked")
        payload["validated"] = validation.get("validated")
```

节点的返回值（LangGraph 的 patch）**就是它干了什么的完整描述**，所以埋点层
不需要每个节点单独配置「我要记什么」，按通用规则抽即可。代价是记的东西是
通用的而非节点专属的 —— 接受这个代价，真需要专属指标时节点可以在 patch 里
多返回一个 `_span` 字段（`_SPAN_HINT_KEY`），按需扩展。

**降级判定：看 patch，不只看异常。** `app/agent/rag_v2/instrumentation.py:92`

```python
def _derive_degradation(patch: Any) -> tuple[bool, str | None]:
    """
    为什么要看 patch 而不只看有没有抛异常：
    项目里的节点是**刻意不抛**的 —— retrieve_each 兜住检索异常、
    generate 兜住 LLM 异常，都返回一个「成功的」patch。
    只按异常判定的话，这些节点永远是 OK，
    检索失败率、LLM 失败率两个指标恒为 0，而故障是真实发生了的。
    """
    reasons = patch.get("degrade_reasons")
    if reasons:
        return True, str(reasons[0]) if isinstance(reasons, list) and reasons else None
    if patch.get("retrieve_failures"):
        return True, None
    return False, None
```

**指标埋在这里，不在五个节点里。** `app/agent/rag_v2/instrumentation.py:149`

```python
def _apply(span: span_scope, patch: Any) -> None:
    """
    为什么 degrade_total 埋在这里,而不是在五个节点各自的降级分支里:

    这里是 rag_v2 全部降级判定的**唯一**汇聚点 —— `_derive_degradation`
    已经把「什么算降级」这条规则收敛完了。指标跟着它走,
    就自动获得两个性质:
      1. 新增节点自动被统计,不会有人忘记加 `count_degrade`(DRY + 开闭原则)。
      2. **指标与 span 永远同源**。如果各节点自己埋,迟早出现
         「span 表里记了降级、degrade_total 没涨」这种对不上的情况,
         而排障时两个数据源互相矛盾比没有数据更糟 ——
         你不知道该信哪个,只能两边都不信。
    """
    span.set_payload(**_derive_payload(patch))
    degraded, reason = _derive_degradation(patch)
    if degraded:
        span.mark_degraded(reason)
        count_degrade(reason)
```

最后那句「两个数据源互相矛盾比没有数据更糟」值得单独记住。它是整个可观测性
工程的一条基本纪律：**同一个事实只能有一处判定。**

### 3.5 落盘：跑完一次性写

`app/core/span_context.py:160`

```python
def flush_spans(db: Session, *, trace_id: str, spans=None) -> int:
    """把收集到的 span 一次性写库，返回写入条数。

    失败只记日志，绝不抛 —— 主流程（答案已经生成、trace 已经落库）
    不该因为写不进耗时明细而失败。
    """
    payload_spans = collected_spans() if spans is None else spans
    if not payload_spans:
        return 0

    request_id = get_request_id_or_none()      # ← 第 15 篇：DB 用 None 不用 "-"
    try:
        rows = [ChatRunSpan(trace_id=trace_id, request_id=request_id, ...) for item in payload_spans]
        db.add_all(rows)
        db.commit()
        return len(rows)
    except Exception as exc:
        logger.warning(f"span 落盘失败（不影响主流程）: trace_id={trace_id}: {exc}")
        with suppress(Exception):
            db.rollback()
        return 0
```

### 3.6 接口层接线：三个时机都不能错

`app/api/chat_v2.py:74`（非流式）

```python
    # span 收集必须在**进图之前**开启。
    #
    # 原因是 ContextVar 的拷贝语义：LangGraph fan-out 并行分支时会
    # copy_context()，子上下文拿到的是「同一个 list 对象」的引用。
    # 所以只有在父上下文（也就是这里）set 一次，各分支 append 的 span
    # 才能汇总到同一个篮子里。如果在图内部才 set，每个分支会各自
    # 建一个 list，父上下文一条都看不到。
    span_token = start_span_collection()
    try:
        ...
        trace = trace_repo.create_trace(...)
        # trace 落库之后才有 trace_id，span 才有地方挂 —— 这是「跑完一次性
        # flush」而不是「节点各自写库」的根本原因。
        flush_spans(db, trace_id=trace.id)
```

失败路径同样要 flush（`app/api/chat_v2.py:144`）：

```python
        # 失败路径的 span 比成功路径更值钱：它记着「跑到哪一步炸的、
        # 前面几步各花了多久」。这恰恰是排障第一个要问的问题，
        # 所以这里必须和成功路径一样 flush。
        flush_spans(db, trace_id=trace.id)
        raise
    finally:
        # 用 reset(token) 还原，而不是留着不管：
        # 事件循环里的 ContextVar 在同一个 task 内是复用的，
        # 不还原会让下一个请求继承上一个请求的 span 篮子。
        reset_span_collection(span_token)
```

流式路径的收集器位置**不一样**（`app/api/chat_v2.py:171`）：

```python
    async def event_generator():
        ...
        # 收集器必须在**生成器体内**开，不能开在外层 chat_v2_stream 里。
        #
        # 原因是执行时机：外层函数只是构造出 EventSourceResponse 就返回了，
        # 生成器体要等 sse-starlette 真正迭代它时才开始跑 —— 那时外层的栈帧
        # 早已退出，它 set 的 ContextVar 也已随之失效。
        # 开在这里，收集器的生命周期才和「图真正在跑」的那段时间对齐。
        span_token = start_span_collection()
```

以及 `finally` 里的注释点出了一个容易忽略的路径（`app/api/chat_v2.py:287`）：

```python
        finally:
            # 生成器也可能被客户端提前断开而不走完 —— 那时 GeneratorExit
            # 会在这里经过，收集器同样要还原，否则这个篮子会一直挂在
            # 当前上下文上，被后续复用同一 task 的请求继承。
            reset_span_collection(span_token)
```

### 3.7 迁移

`migrations/versions/20260824_0009_create_chat_run_spans.py`，`down_revision = "20260823_0008"`。
五个索引各有明确用途：

| 索引 | 查询场景 |
|---|---|
| `trace_id` | 看某次请求的完整时间线（最主要入口，且是外键） |
| `request_id` | 日志里捞到一个 id，直接查 span，不必先查 trace |
| `node` | 按节点做聚合（哪个节点平均最慢） |
| `status` | 只看失败/降级的 span |
| `created_at` | 按时间窗口做趋势统计 |

`downgrade()` 最后一行容易漏：

```python
    sa.Enum(name="spanstatus").drop(op.get_bind(), checkfirst=True)
```

PostgreSQL 的 Enum 是**独立的类型对象**，`drop_table` 不会带走它。不显式删除的话，
下一次 `upgrade` 会撞上 `type "spanstatus" already exists`。

---

## 4. 背后的工程原理

### 4.1 ContextVar 的拷贝语义 —— 本篇最重要的一条

这条如果没搞清楚，代码能跑、单测能过，**并发下静默丢数据**。

`asyncio.to_thread`、`asyncio.create_task`、LangGraph 内部 spawn 任务时，
都会调 `contextvars.copy_context()`。关键在于：

> `copy_context()` 拷贝的是**变量到对象的绑定关系**，不是对象本身。

也就是说，父上下文和子上下文里的 `_SPAN_COLLECTOR` **指向同一个 list 对象**。
由此得到两条方向相反的推论，必须都记住：

| 子上下文里做什么 | 父上下文能否看到 | 为什么 |
|---|---|---|
| `list.append(...)` | ✅ 看得见 | 同一个对象被就地修改了 |
| `ContextVar.set(...)` | ❌ 看不见 | 只改了子上下文自己的绑定 |

所以本模块的纪律是：**只在请求入口 `set` 一次，之后全程只 `append`。**
`record_span` 里绝不调 `.set()`。

违反这条会怎样？假设 `record_span` 写成了
「先取，取不到就 `set([])`，再 append」—— 单个协程里跑得好好的，
一旦 fan-out 成 4 条并行检索，每条分支各自 `set` 一个新 list，
append 进自己那个，父上下文里的篮子**一条都没有**。
这就是那种「本地单测全绿、上线才发现检索 span 永远是空」的 bug。

对应的测试是 `tests/unit/test_span_context.py:101` 的 `TestParallelBranches`，
它的类文档直接写着这句话。

### 4.2 为什么是「内存收集 + 一次 flush」，不是节点直接写库

节点直接写库有三个硬伤，任何一个都足以否掉这个方案：

1. **拿不到 `trace_id`。** `ChatRunTrace` 的主键是**跑完落库时**才生成的。
   节点执行期间那行记录还不存在，span 无处挂靠。
2. **节点拿不到 Session。** LangGraph 的节点签名是 `state -> patch`。
   要塞一个 db session 进去，就得让它穿过 state 或者改所有节点签名 ——
   而 state 是要被序列化的，往里塞 Session 是自找麻烦。
3. **写库次数爆炸。** 一次请求 8 个节点实例（含 4 条并行检索），
   各 commit 一次就是 8 次往返。

第 3 条这个项目里有前科：AIOps 时间线就是这么写的，每个阶段各 commit 一次。

改成内存收集之后，跑的时候只往 list 里 `append`（几乎零成本），
跑完拿到 `trace_id` 一次 `add_all` + 一次 `commit`。

### 4.3 观测设施的第一纪律：绝不影响主流程

`span_context.py` 里所有对外函数都不抛异常。`__exit__` 里的
`with suppress(Exception)`、`flush_spans` 的 except 分支、
`reset_span_collection` 的 suppress，全都是同一条原则的实例。

理由很朴素：**span 是可观测性设施，不是业务功能。用户要的是答案。**
「因为记不下耗时，所以这次回答失败了」是荒谬的因果。

这和第 19 篇的 `record_job_failure`「绝不 raise」是同一条原则，
和第 15 篇 `_patch_request_id` 用 `setdefault` 防 KeyError 也是同一条 ——
**辅助设施抛出自己的异常，会掩盖真正的问题**，那比没有这个设施更糟。

### 4.4 为什么不上 OpenTelemetry（YAGNI 的一个标准案例）

OTel 的完整方案要引 SDK、起 Collector、部署 Jaeger 或 Tempo，
换来的是**跨服务的分布式追踪**。

而当前需求是**单进程内**的节点耗时，四个字段（node / duration_ms / status /
payload）就够。

所以这里借用 OTel 的**概念**（span 是一段有始有终、可嵌套归属的工作），
但不引它的**实现**。

这不是「反对用成熟方案」，而是**判断成本收益**。判断方法：
把要引入的东西列出来（SDK 依赖、Collector 进程、后端存储、运维成本），
再把当前真正要回答的问题列出来（「哪个节点慢」）。如果后者能用前者的
十分之一成本解决，就先解决问题。

面试里这个点可以这样答：「我们借了 span 的模型，没引 OTel 的运行时。
表的四个字段能平移成 OTel 的 span 属性，真到了需要跨服务串联那天不算白做。」
**关键是要能说出迁移路径** —— 有迁移路径的简化叫 YAGNI，没有的叫技术债。

### 4.5 为什么没有 `parent_span_id`

OTel 的 span 有父子关系，用来还原调用树。这里刻意不做，理由在
`app/models/chat_run_span.py:28` 的模块文档里：

当前图是「rewrite → fan-out 检索 → dedup → generate → validate」，
是**一条扁平序列 + 一层扇出**。用 `node` 名字加 `started_at` 排序就能看清全貌。

加上 `parent_span_id` 就要在节点间传递 span 上下文，而节点是纯函数
（拿 state 返回 patch），传递链路会**污染所有节点签名**。代价大于收益。

判断方法：问自己「不加这个字段，我要回答的问题答不出来吗？」
当前要回答的是「哪个节点慢」，答得出来。等图变成多层嵌套子图时再说。

### 4.6 `duration_ms` 存整数毫秒，而不是存结束时间

三个理由，按重要性排：

1. **聚合友好。** 耗时是要被 `avg` / `max` / 分位数聚合的。
   存两个时间戳的话，每次查询都得先相减 —— 而且 PostgreSQL 的
   `interval` 类型做分位数比整数麻烦得多。
2. **不用浮点。** 毫秒整数在 SQL 里聚合不会有精度漂移。
   浮点秒的 `avg` 在大量行上会累积误差。
3. **语义直接。** `duration_ms > 5000` 一眼看懂，
   `ended_at - started_at > interval '5 seconds'` 要多想一秒。

### 4.7 `payload` 用 JSON，而不是给每种计数开一列

不同节点关心的量完全不同：`rewrite` 关心子查询数、`retrieve_each` 关心召回条数
和失败码、`validate_answer` 关心两个分数和是否拦截。

开成列的话，表会长出十几个大部分为 NULL 的字段，而且**每加一个节点专属指标
就要一次 DDL 迁移**。

JSON 的代价是「不能直接建索引做范围查询」。当前不需要 ——
需要索引的是 `node` / `status` / `trace_id`，这三个都是独立列。
真要按 payload 里某个键查，PostgreSQL 的 GIN 索引支持 JSONB，也有路可走。

### 4.8 `_copy_identity` 为什么不用 `functools.wraps`

`app/agent/rag_v2/instrumentation.py:185`

```python
def _copy_identity(wrapper, func) -> None:
    """保留原函数的名字与文档。

    不用 `functools.wraps`：它会连 `__wrapped__` 一起设上，
    而 LangGraph 在某些版本里会顺着 `__wrapped__` 去取原函数签名，
    从而绕过包装。只复制展示用的两个属性，够了也更安全。
    """
    wrapper.__name__ = getattr(func, "__name__", "node")
    wrapper.__doc__ = getattr(func, "__doc__", None)
```

对比第 19 篇的 `with_job_request_id` —— 那里**必须**用 `functools.wraps`，
因为 RQ 靠「模块路径 + 函数名」反序列化任务。

同一个问题（要不要保留原函数身份），两个相反的答案，取决于**下游框架怎么用这些属性**。
这也是为什么「照抄最佳实践」不总是对的：得知道那条实践在解决什么问题。

### 4.9 同步节点和异步节点必须分开包

```python
    if inspect.iscoroutinefunction(func):
        async def async_wrapper(...): ...
    else:
        def sync_wrapper(...): ...
```

只写 async 版本会怎样？`dedup_node` 是同步函数，被 async 包装器包住之后，
调用它返回的是一个**协程对象**。LangGraph 拿到一个没 await 的协程，
当成 patch 往 state 里合并 —— 图直接崩。

反过来只写 sync 版本，异步节点返回的协程不会被 await，同样崩。

---

## 5. 怎么验证

### 5.1 跑单测

```bash
cd /home/dong/projects/super_biz_agent_py
.venv/bin/python -m pytest tests/unit/test_span_context.py -v -p no:cacheprovider
```

四组契约对应四个测试类：

| 测试类 | 契约 |
|---|---|
| `TestSpanScope` | 三态记录（ok/degraded/error）+ 不吞异常 + 脱离上下文空转 |
| `TestParallelBranches` | **并行分支的 span 不丢**（本模块最容易写错的语义） |
| `TestFlushSpans` | 写库成功路径 + 失败绝不抛 + 显式传 spans |

### 5.2 看一条真实时间线

这段能直接跑，它模拟一次「4 条并行检索，其中 1 条失败」的运行，
然后打印出你排障时会看到的东西：

```bash
cd /home/dong/projects/super_biz_agent_py
.venv/bin/python - <<'PY'
import asyncio, time
from app.core.span_context import (
    collected_spans, span_scope, start_span_collection, reset_span_collection,
)

token = start_span_collection()

def branch(i):
    with span_scope("retrieve_each") as s:
        time.sleep(0.05 if i != 1 else 0.3)   # 第 1 条特别慢
        if i == 2:
            raise RuntimeError("milvus connection reset")
        s.set_payload(doc_count=2, query=f"sub-{i}")

async def main():
    with span_scope("rewrite") as s:
        await asyncio.sleep(0.02)
        s.set_payload(sub_query_count=4)
    await asyncio.gather(
        *[asyncio.to_thread(branch, i) for i in range(4)],
        return_exceptions=True,
    )

asyncio.run(main())

print(f"{'node':<16}{'status':<10}{'ms':>6}  payload")
for row in sorted(collected_spans(), key=lambda r: r['started_at']):
    st = row['status'].value if hasattr(row['status'], 'value') else row['status']
    print(f"{row['node']:<16}{st:<10}{row['duration_ms']:>6}  {row['payload']}")
reset_span_collection(token)
PY
```

关键是看输出里 4 条 `retrieve_each` 的 `ms` 列 —— 这正是改造前**算不出来**的那个数。

### 5.3 确认埋点没有漏节点

```bash
cd /home/dong/projects/super_biz_agent_py
# 图里注册的节点数，和过了 instrument_node 的节点数必须一致
grep -c 'graph.add_node' app/agent/rag_v2/graph.py
grep -c 'instrument_node(' app/agent/rag_v2/graph.py
```

第二个数会比第一个多 1（`import` 那行也会被 `instrument_node(` 匹配到吗？
不会 —— import 里没有括号）。实际两个数都应该是 5。

### 5.4 确认迁移能上能下

```bash
cd /home/dong/projects/super_biz_agent_py
.venv/bin/alembic heads          # 应该是 20260824_0009 (head)
.venv/bin/alembic history | head -5
```

---

## 6. 常见坑

### 6.1 在 `record_span` 里调 `.set()`

已经在 §4.1 说透了，这里只留一句结论：
**只在入口 set 一次，之后全程只 append。** 违反这条，并行分支的 span
静默丢失，而且单测能过。

### 6.2 SSE 路径把收集器开在生成器外面

```python
# ✗ 错的
async def chat_v2_stream(...):
    span_token = start_span_collection()      # ← 这里 set 的，等生成器跑起来就失效了
    async def event_generator(): ...
    return EventSourceResponse(event_generator())
```

外层函数只是**构造出** `EventSourceResponse` 就返回了。生成器体要等
sse-starlette 真正迭代它时才开始跑，那时外层栈帧早已退出。
结果是流式接口一条 span 都收不到，而非流式接口好好的 —— 排查起来相当费时间。

### 6.3 忘了 `reset(token)`

事件循环里的 ContextVar 在同一个 task 内是复用的。不还原，
下一个请求会继承上一个请求的 span 篮子 —— 两次请求的 span 混在一起，
`GROUP BY node` 出来的数字全是错的。

这和第 15 篇 `request_id` 不 reset 的后果是同一类：**串了比没有更糟**，
因为你会相信一个错的数字。

流式路径还多一条：客户端提前断开时 `GeneratorExit` 会经过 `finally`，
那里同样要 reset。

### 6.4 `flush_spans` 调在 `create_trace` 之前

`trace_id` 是外键。trace 还没落库就 flush，外键约束会让 commit 失败 ——
好在 `flush_spans` 不抛，只 warning + 返回 0。所以现象是**「span 表一直是空的，
但接口一切正常」**，没有任何报错指向问题所在。

对应测试 `test_write_failure_never_raises` 特意打开了 SQLite 的外键约束：

```python
# SQLite 默认不强制外键，显式打开才能触发约束失败
db.execute(text("PRAGMA foreign_keys=ON"))
```

这条 PRAGMA 值得单独记：**SQLite 默认不检查外键**，
所以「在 SQLite 上测过了」不代表 PostgreSQL 上不会炸。

### 6.5 PostgreSQL 的 Enum 类型不会被 `drop_table` 带走

`downgrade()` 里少了 `sa.Enum(name="spanstatus").drop(...)`，
下一次 `upgrade` 报 `type "spanstatus" already exists`。
这个错误信息离真正的原因（上次 downgrade 没清干净）有点远。

### 6.6 两个当前存在的遗留问题（如实记录）

**（1）`iter_span_payloads` 目前没有调用方。**

```bash
$ grep -rn 'iter_span_payloads' app/ tests/
app/core/span_context.py:199:def iter_span_payloads(...)     # 只有定义处
```

它是为「trace 详情接口返回 span 时间线」准备的，但 `app/api/chat_v2.py` 的
`_serialize_trace_detail` 还没用上它。也就是说 **span 现在只能用 SQL 查，
接口里看不到**。按 YAGNI 这个函数本该等到接口真要用时再写；
既然已经写了，就该尽快接上或者删掉 —— 留着的死代码会让下一个人以为接口已经支持了。

**（2）`migrations/env.py` 只导入了 4 个模型模块。**

```python
# migrations/env.py:8-11
from app.models import conversation           # noqa: F401
from app.models import knowledge_base         # noqa: F401
from app.models import protocol_catalog       # noqa: F401
from app.models import protocol_ingestion     # noqa: F401
```

`chat_run_span`、`chat_run_trace`、`aiops` 相关模型都没导入，而
`app/models/__init__.py` 是空的（只有一行 docstring），不会代为导入。

后果：`target_metadata = Base.metadata` 里**缺这几张表**。手写迁移不受影响
（现有的 0004、0006、0009 都是手写的），但一旦有人跑
`alembic revision --autogenerate`，Alembic 会认为这些表「在模型里不存在、
在数据库里存在」，于是生成 `op.drop_table('chat_run_spans')`。

**这是个会删表的坑**，而且触发条件是一个非常常规的操作。
修法很简单（在 `env.py` 补齐 import，或者让 `app/models/__init__.py` 统一导入），
但这属于本批次范围之外的改动，先记在这里。

---

## 7. 一句话总结

`chat_run_traces` 回答「结果是什么」，`chat_run_spans` 回答「过程发生了什么」。
排障时人问的第一个问题是后者，所以这一层不能省。

而加这一层的全部难点不在功能，在**纪律**：
ContextVar 只 set 一次全程 append（否则并发丢数据）、
所有对外函数不抛异常（否则观测层能把系统弄挂）、
降级判定只有一处实现（否则两个数据源互相矛盾）。

**观测设施写错的代价，是让你相信一个错的数字。** 那比没有观测更糟。
