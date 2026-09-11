# 16. LLM 超时与统一构造路径

> 关键词：单次超时、整体预算、构造收敛、`is None` vs `or`、部分结果
>
> 涉及文件：`app/core/llm_factory.py`、`app/config.py`、`app/agent/rag_v2/service.py`、`tests/unit/test_llm_factory.py`

---

## 1. 为什么要改

### 1.1 一个具体的故障场景

某天下午，DashScope 那边开始限流排队。它没有返回 429，也没有断开连接 ——
TCP 连接建立着，只是不再往回吐字节。

我们这边的情况：

```
用户 A 发起 chat_v2 请求
  → rewrite 调 LLM，挂住
用户 B 发起 chat_v2 请求
  → rewrite 调 LLM，挂住
...
用户 P 发起 chat_v2 请求
  → 连接池已满，在排队等一个可用连接
```

请求挂住 → SSE 连接挂住 → 连接池被逐渐占满 → 后续所有请求排队。
**服务没有崩溃，只是所有人都在等，而且永远等不到。**

健康检查接口 `/health` 还是 200（它不调 LLM），所以负载均衡认为这个实例是健康的，
继续往里送流量。整个服务被一个卡住的上游拖死，而监控上看不出任何异常 ——
没有 5xx，没有 OOM，只有请求数下降和延迟曲线冲出图表。

### 1.2 根因不是「忘了设超时」，而是「构造分散」

改造前项目里有三个各自 new 模型的地方：

```
app/core/llm_factory.py:40                   ChatOpenAI(...)
app/services/first_response_service.py:98    ChatQwen(...)
app/services/rag_agent_service.py:91         ChatQwen(...)
```

**三处都没有设置任何超时。**

这不是三次独立的疏忽，而是分散构造的必然结果。超时是一个**横切参数** ——
它跟具体业务无关，是每个 LLM 调用都需要的基础防护。
横切参数一旦分散到多个构造点，就没有任何一个地方能保证「全都设上了」。

更要命的是这种 bug 的发现方式：加超时的那天，你会改你正在看的那个文件，
另外两处继续裸奔。而线上真正挂住的，往往就是你没改的那一个。

对比一下当时项目里已有的超时防护：

| 位置 | 有超时吗 |
|---|---|
| Milvus 检索 | ✅ `milvus_timeout` |
| RQ 异步任务 | ✅ `job_timeout` |
| MCP 调用 | ✅ `retry_interceptor` |
| **LLM 调用** | ❌ **完全没有** |

最慢、最不可控、耗时波动最大的那条链路，反而是唯一没有防护的。

---

## 2. 改之前什么样

### 2.1 `app/core/llm_factory.py`（HEAD 版本）

```python
"""LLM 工厂类

使用 LangChain ChatOpenAI 通过 OpenAI 兼容模式调用阿里云 DashScope
...
"""

from langchain_openai import ChatOpenAI
from app.config import config
from loguru import logger


class LLMFactory:
    """LLM 工厂类 - 使用 OpenAI 兼容模式"""

    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @staticmethod
    def create_chat_model(
        model: str | None = None,
        temperature: float = 0.7,
        streaming: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> ChatOpenAI:
        model = model or config.dashscope_model
        base_url = base_url or LLMFactory.DASHSCOPE_BASE_URL
        api_key = api_key or config.dashscope_api_key

        extra_body = {}
        extra_body["stream"] = streaming

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            base_url=base_url,
            api_key=api_key,
            extra_body=extra_body if extra_body else None,
        )

        return llm


llm_factory = LLMFactory()
```

注意签名里**没有 `timeout`，也没有 `max_retries`**。
不传的后果比「用了个保守的默认值」严重得多，第 7.2 节有实测 ——
先记住结论：**底层 httpx 的读取超时是 `None`，无限期等待**。

而且这里也没有 `ChatQwen` 那条路径 ——
两个服务各自 `ChatQwen(...)`，完全绕过了这个工厂。

### 2.2 这里还藏着一个即将踩到的坑

看这三行：

```python
model = model or config.dashscope_model
base_url = base_url or LLMFactory.DASHSCOPE_BASE_URL
api_key = api_key or config.dashscope_api_key
```

对字符串参数，`or` 是没问题的（空字符串确实该当没传）。
但如果照着这个模式给 `timeout` 也写 `timeout or config.llm_timeout_seconds`，
就会引入一个经典 bug —— 下一节展开。

---

## 3. 改之后什么样

### 3.1 补上两个横切参数

`app/core/llm_factory.py:54-93`：

```python
    @staticmethod
    def create_chat_model(
        model: str | None = None,
        temperature: float = 0.7,
        streaming: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,        # ← 新增
        max_retries: int | None = None,      # ← 新增
    ) -> ChatOpenAI:
        """构造 ChatOpenAI 实例。

        Args:
            timeout: 单次调用超时（秒）。传 None 走 config.llm_timeout_seconds。
                注意不能用 `timeout or config.x` 这种写法 —— 那样 `timeout=0`
                会被静默替换成默认值。这里用 `is None` 判断，保留显式传 0 的语义。
            max_retries: 失败重试次数。同上，用 `is None` 判断。
        """
        model = model or config.dashscope_model
        base_url = base_url or LLMFactory.DASHSCOPE_BASE_URL
        api_key = api_key or config.dashscope_api_key
        # 关键：`is None` 而不是 `or`。见下面 3.2 节。
        timeout = config.llm_timeout_seconds if timeout is None else timeout
        max_retries = config.llm_max_retries if max_retries is None else max_retries

        extra_body = {}
        extra_body["stream"] = streaming

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,              # ← 注入
            max_retries=max_retries,      # ← 注入
            extra_body=extra_body if extra_body else None,
        )

        return llm
```

配置项在 `app/config.py`：

```python
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
```

### 3.2 `is None` 而不是 `or` —— 这一行值得单独讲

```python
# ❌ 错的写法
timeout = timeout or config.llm_timeout_seconds

# ✅ 对的写法
timeout = config.llm_timeout_seconds if timeout is None else timeout
```

`0` 在 Python 里是 falsy。用 `or` 的话：

```python
timeout = 0 or 60.0      # → 60.0
```

调用方明确写了 `timeout=0`（语义是「不等待」或「用底层默认」），
却被静默改成了等 60 秒。**没有报错，没有警告，行为完全相反。**

这类 bug 在测试里也很难发现，因为大多数用例根本不传 0。
所以我专门写了一条参数化用例把它钉住（`tests/unit/test_llm_factory.py:110-123`）：

```python
    @pytest.mark.parametrize("explicit", [0, 0.0])
    def test_zero_timeout_is_not_replaced_by_default(self, explicit: float) -> None:
        """显式传 0 不能被静默改成默认值。

        这条守的是 `timeout or config.x` 这个经典 bug：
        0 是 falsy，用 `or` 会让「显式要求不等待」变成「等 60 秒」。
        factory 里用的是 `is None` 判断，这个用例就是它的护栏。
        """
        model = LLMFactory.create_chat_model(
            model="test-model",
            api_key="test-key",
            timeout=explicit,
        )
        assert model.request_timeout == 0
```

一般化的规则：**当参数的合法取值包含任何 falsy 值（`0`、`0.0`、`False`、`""`、`[]`）时，
默认值判断必须用 `is None`。** `or` 只有在「所有 falsy 值都该被当作没传」时才是对的。

### 3.3 `ChatQwen` 那条路径：为什么保留而不是合并

`app/core/llm_factory.py:95-137`：

```python
    @staticmethod
    def create_qwen_model(
        model: str | None = None,
        temperature: float = 0.7,
        streaming: bool = False,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """构造 langchain_qwq 的 ChatQwen 实例。

        为什么不把 ChatQwen 的调用点直接改成 create_chat_model：
        ChatQwen 不只是「另一个 OpenAI 兼容客户端」，它对 Qwen 的
        推理内容（reasoning_content）等字段有专门处理，rag_agent_service
        依赖这些行为。强行换成 ChatOpenAI 会改变运行时语义 ——
        本次目标是补超时，不是换模型客户端（YAGNI，也避免夹带风险）。

        但**超时参数的默认值必须与 ChatOpenAI 路径完全一致**，
        所以两条路径共用同一套 config 默认值，这是收敛的实际意义所在。

        延迟 import langchain_qwq：它是较重的可选依赖，
        放在模块顶层会让所有 import app.core.llm_factory 的地方都被迫加载它。
        """
        from langchain_qwq import ChatQwen

        model = model or config.rag_model
        api_key = api_key or config.dashscope_api_key
        timeout = config.llm_timeout_seconds if timeout is None else timeout
        max_retries = config.llm_max_retries if max_retries is None else max_retries

        return ChatQwen(
            model=model,
            api_key=api_key,
            temperature=temperature,
            streaming=streaming,
            # ChatQwen 继承自 BaseChatOpenAI，其字段名为 request_timeout，
            # 但带有 alias "timeout"（已通过 model_fields 校验）。
            # 这里用 timeout=，与 ChatOpenAI 路径写法保持一致。
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
```

这里体现了一个容易被误解的点：**「收敛」收的是决策，不是代码行数。**

如果目标只是「减少重复代码」，那应该把 `ChatQwen` 全换成 `ChatOpenAI`，
删掉一整个方法。但那会改变运行时语义 —— `reasoning_content` 的处理方式变了，
`rag_agent_service` 依赖的行为就断了。本次目标是补超时，不该夹带这种风险（YAGNI）。

真正需要唯一化的是**「超时该是多少」这个决策**。
两条路径都读 `config.llm_timeout_seconds`，
所以改一个配置项就能同时影响全部 LLM 调用。这才是收敛的实际意义。

改造后所有 6 个构造点全部走工厂：

```
app/services/rag_agent_service.py:94         llm_factory.create_qwen_model(...)
app/services/first_response_service.py:153   llm_factory.create_qwen_model(temperature=0)
app/services/conversation_memory_service.py:193  llm_factory.create_chat_model(...)
app/agent/rag_v2/nodes.py:96                 llm_factory.create_chat_model(...)
app/agent/rag_v2/nodes.py:304                llm_factory.create_chat_model(temperature=0.3, ...)
app/agent/rag_v2/nodes.py:392                llm_factory.create_chat_model(temperature=0.0, ...)
```

六处，零个绕过。这是「唯一入口」能带来的性质：
以后再加一个横切参数（比如 `default_headers` 里塞 request_id），
改一处就全生效。

---

## 4. 单次超时 ≠ 整体预算

### 4.1 为什么单次超时不够

`timeout=60` 管的是**一次**调用。但 chat_v2 的一次请求是这样的：

```
用户提问
  ├─ rewrite     调 LLM  ← 最多 60s
  ├─ retrieve    并行检索 Milvus
  ├─ generate    调 LLM  ← 最多 60s
  └─ validate    调 LLM  ← 最多 60s
```

最坏情况 **3 × 60 = 180 秒**，还要加上检索耗时。
每一次调用都「没有超时」，但用户等了三分多钟。

这两个数字回答的是**不同的问题**：

| | 回答的问题 | 保护的对象 |
|---|---|---|
| `llm_timeout_seconds` | 这次调用最多挂多久 | 连接池、单个上游依赖 |
| `chat_total_budget_seconds` | 这个用户最多等多久 | 用户体验、SLA |

层级不同，缺一不可。只有单次超时，用户会等 3 倍；
只有总预算，一个卡住的连接会一直占着连接池直到预算耗尽 ——
而在此期间它无法被复用。

### 4.2 总预算的实现，以及一个关键选择

`app/agent/rag_v2/service.py:60-78`：

```python
        try:
            async with asyncio.timeout(config.chat_total_budget_seconds):
                async for state in self.graph.astream(inputs, stream_mode="values"):
                    if isinstance(state, dict):
                        last_state = state
        except asyncio.TimeoutError:
            # 不向上抛：预算耗尽是**可降级**的失败，我们有部分结果可以交付。
            # 抛异常会让用户既等满了预算又什么都拿不到，是最差的结果。
            timed_out = True
            logger.warning(
                f"[rag_v2:{session_id}] 总预算 {config.chat_total_budget_seconds}s 耗尽，"
                f"返回部分结果; 已有字段={sorted(last_state.keys())}"
            )

        result = _build_query_result(last_state)
        if timed_out:
            result["degrade_reason"] = DegradeReason.TOTAL_BUDGET_EXCEEDED.value
            result["answer"] = result["answer"] or _budget_exceeded_answer(last_state)
        return result
```

**为什么用 `astream` 累积状态，而不是 `ainvoke`？**

这是这一节最值得记住的设计。模块 docstring 里写得很清楚：

```
`asyncio.timeout` 触发时会取消里面的任务。如果用 `ainvoke`，
取消意味着**整个结果全丢**，我们只能告诉用户「超时了」，
而实际上改写和检索可能早已完成 —— 那些工作白做了，
排查时也看不到「到底卡在哪一步」。

`astream(stream_mode="values")` 每个超步结束都会吐出**完整快照**，
我们把最后一个快照留在手里。超时时就能回答两个关键问题：
    1. 已经拿到了什么（子查询、召回文档，可以返回给用户）
    2. 卡在了哪一步（最后一次快照缺哪个字段）
这才是「返回部分结果 + 降级说明」，而不是干巴巴一句超时。
```

用 `ainvoke` 的话，超时的用户体验是：等 90 秒，拿到一句「超时了」。
用 `astream` 累积快照的话：等 90 秒，拿到已检索的文档 + 一句说明卡在哪。
**代码复杂度只差三行，交付质量差一个量级。**

### 4.3 降级说明要说清「卡在哪一步」

`app/agent/rag_v2/service.py:155-172`：

```python
def _budget_exceeded_answer(state: Dict[str, Any]) -> str:
    """预算耗尽且还没生成答案时，给用户一句诚实的说明。

    刻意说清「卡在哪一步」而不是只说「超时」：
    检索到文档但没出答案 → 生成阶段慢，可能要调模型或加大预算；
    连文档都没有 → 检索阶段慢，该去看向量库。
    对用户是交代，对值班同学是线索。
    """
    doc_count = len(state.get("documents", []) or [])
    if doc_count:
        return (
            f"本次回答超出 {config.chat_total_budget_seconds} 秒预算，"
            f"已检索到 {doc_count} 条相关资料但未能完成答案生成，请重试或缩小问题范围。"
        )
    return (
        f"本次回答超出 {config.chat_total_budget_seconds} 秒预算，"
        "检索阶段未能完成，请稍后重试。"
    )
```

同一个 `TotalBudgetExceededError`，两句不同的话。
区别在于 `state` 里有没有 `documents` —— 而这个信息只有 `astream` 累积快照才拿得到。

### 4.4 流式路径也必须有预算，而且更危险

`app/agent/rag_v2/service.py:89-132`：

```python
        try:
            # 流式路径同样受总预算约束。没有它，SSE 连接会一直挂着 ——
            # 而流式场景更危险：客户端看到连接是「开着」的，会一直等下去，
            # 不像普通请求至少有客户端侧超时兜底。
            async with asyncio.timeout(config.chat_total_budget_seconds):
                async for update in self.graph.astream(inputs, stream_mode="updates"):
                    ...

            yield {"type": "complete"}

        except asyncio.TimeoutError:
            # SSE 已经返回了 HTTP 200，改不了状态码，只能用事件告知降级。
            # 关键是带上 degrade_reason：前端和日志都能区分
            # 「预算耗尽」和「LLM 报错」，两者的处置完全不同。
            logger.warning(...)
            yield {
                "type": "error",
                "data": f"本次回答超出 {config.chat_total_budget_seconds} 秒预算，已中断",
                "degrade_reason": DegradeReason.TOTAL_BUDGET_EXCEEDED.value,
            }
            # 不 raise：前面已经 yield 过若干有效事件，
            # 此时抛异常会让 SSE 连接以错误方式断开，客户端可能丢掉已收到的内容。
```

三个细节各自对应一个真实后果：

1. **流式更需要预算。** 普通 HTTP 请求，浏览器和 fetch 都有默认超时兜底。
   SSE 连接是「正常开着」的状态，客户端会一直等 —— 没有任何一层会救它。

2. **SSE 无法用状态码表达失败。** 响应头在第一个事件发出时就已经发送了，
   HTTP 200 已成事实。所以只能在事件流里用 `type: "error"` + `degrade_reason` 表达。
   这是第 18 篇「HTTP 语义」那条规则的一个例外，而例外的理由是协议限制，
   不是我们偷懒。

3. **`except TimeoutError` 里不 raise。** 前面已经吐过 `sub_queries`、`retrieved`
   这些有效事件了。此时抛异常会让连接以错误方式断开，客户端框架可能直接丢弃
   整个已接收缓冲。而下面那个 `except Exception` 是 raise 的 ——
   未知异常需要冒泡让上层记录，两者处置不同是有意的。

---

## 5. 背后的工程原理

### 5.1 超时是分布式系统的第一道防线

面试里被问到「你的系统怎么保证稳定性」，很多人会答限流、熔断、降级。
但这三个都建立在一个前提上：**失败要能被及时观察到。**

没有超时，一次调用的失败永远不会「发生」—— 它只是一直挂着。
于是：

- 熔断器永远不会计数（它统计的是失败，而挂住不算失败）
- 重试永远不会触发（没有异常抛出）
- 限流看到的是「请求数正常」（请求都在，只是不返回）

**超时是把「无限期等待」转换成「明确失败」的机制。**
只有明确的失败才能被计数、被分类、被降级。所以它必须是第一道防线，
在熔断器和重试之前。

这也解释了为什么改造顺序是：先补超时（第 16 篇）→ 再做错误分类（第 14 篇）
→ 最后加熔断器（第 21 篇）。后面两个都依赖前面那个。

### 5.2 资源耗尽是级联故障的传播路径

上游变慢，为什么会导致我们整个服务不可用？

关键在于**连接、线程、内存这些资源是有限且共享的**。
一个挂住的请求会持续占用：

```
一个挂住的 chat_v2 请求占用：
  ├─ 1 个 uvicorn worker 的协程槽位
  ├─ 1 个 httpx 连接池里的连接（到 DashScope）
  ├─ 若干内存（对话上下文、已召回文档）
  └─ 1 个数据库连接（如果它已经开了事务）
```

100 个并发挂住，连接池就空了。第 101 个请求即使是查健康状态，
只要它需要一个连接，也会排队。

**故障从一个依赖传播到整个服务，靠的是共享资源被耗尽。**
超时的作用是给每次占用设一个上界，让资源必然被归还。

工业界对这类问题的标准方案有三层，超时是最基础的那层：

| 层次 | 机制 | 作用 |
|---|---|---|
| 1 | **超时** | 保证资源必然归还 |
| 2 | **舱壁隔离**（bulkhead） | 不同依赖用不同的连接池，一个耗尽不影响另一个 |
| 3 | **熔断** | 已知上游挂了就别再占资源了 |

我们做了 1 和 3（熔断器按依赖分开，见第 21 篇），
2 的部分由「熔断器按下游依赖分别实例化」间接实现了 ——
`llm_breaker` 和 `retrieval_breaker` 是两个独立实例，一个开了不影响另一个。

### 5.3 超时值怎么定：60s 和 90s 是怎么来的

这是面试里的加分点 —— 说得出数字背后的推导过程，而不是「凭感觉」。

**单次超时 60s 的依据：**

- LLM 生成是流式的，长答案的 P99 首字延迟 + 生成时间在 20-40s 量级
- 设成 30s 会砍掉正常的长答案（false positive，误杀）
- 设成 120s 意味着一次卡住要占用连接 2 分钟（保护力度不足）
- 60s 留了 1.5 倍的余量，既能覆盖正常长尾，又不至于占用太久

**总预算 90s 的依据（这个更有意思）：**

朴素的算法是 3 次调用 × 60s = 180s。但 90s < 180s，**是刻意的**。

理由：三次调用**同时**跑到 60s 的概率极低。而用户能接受的等待上限
是一个独立的产品约束 —— 90 秒已经是极限了，超过就该给部分结果。

也就是说，总预算不是从单次超时推导出来的，
它是**从用户体验反推的**，然后单次超时在它之下取一个合理值。
两个数字的关系是：

```
单次超时 (60s)  <  总预算 (90s)  <  单次超时 × 调用次数 (180s)
```

左边的不等式保证「单次超时能在总预算内触发」——
如果单次超时 > 总预算，那么单次超时永远等不到触发就被总预算掐了，
形同虚设。右边的不等式则说明我们接受「串行调用可能被总预算提前中断」，
并为此准备了部分结果降级。

这也是为什么 `_budget_exceeded_answer` 必须存在：
我们主动选择了一个会被触发的预算，就必须为触发后的体验负责。

### 5.4 为什么 `max_retries=2` 而不是更多

`ChatOpenAI` 的 `max_retries` 是**客户端内部重试**，
发生在 `ainvoke` 内部，对我们完全透明。

设 2 的考虑：

- 重试对**瞬时故障**有效（网络抖动、偶发 5xx），这类故障重试 1-2 次基本能成
- 但重试会**乘上超时**：`max_retries=2` 最坏情况是 3 × 60 = 180s 才抛异常
- 而我们的总预算是 90s —— 也就是说重试到第二次时，总预算已经先掐了

这里有个容易忽略的交互：**客户端重试和总预算会互相干扰。**
`max_retries` 设太大是浪费的，因为总预算不会让它跑完。
设 2 是一个「够用且不会明显和总预算打架」的值。

如果要更严谨，应该按「单次超时 × (max_retries + 1) ≤ 总预算」来定，
也就是 timeout=60 时 max_retries 应该是 0。但我保留了 2，
因为**重试的价值主要在快速失败的场景**（连接被拒、立即返回 5xx），
那种情况下重试几乎不耗时间，而它能救回的失败是真实的。
这是一个有意识的权衡，不是疏忽。

### 5.5 工厂模式在这里的真实价值

工厂模式在教科书里的说法是「封装对象创建」，听起来像废话。
这个案例给了它一个具体的价值主张：

**工厂是横切关注点的注入点。**

一个 LLM 实例需要的参数分两类：

| 类型 | 例子 | 谁决定 |
|---|---|---|
| **业务参数** | `temperature`、`streaming` | 调用点（不同节点需求不同） |
| **横切参数** | `timeout`、`max_retries`、`base_url`、`api_key` | 全局策略 |

调用点应该只关心业务参数。横切参数由工厂统一注入 ——
调用点甚至不需要知道它们的存在。

看改造后的调用点有多干净（`app/agent/rag_v2/nodes.py:304`）：

```python
llm = llm_factory.create_chat_model(temperature=0.3, streaming=False)
```

它没提超时，但拿到了 60 秒超时和 2 次重试。
这才是「一处实现，全局生效」—— 而且新增调用点**自动**获得防护，
不需要写文档提醒别人「记得设超时」。

这也是**依赖倒置**的一个朴素实例：调用点依赖的是「给我一个能聊天的模型」
这个抽象，而不是「`ChatOpenAI` 这个具体类以及它的十几个参数」。
以后要换模型客户端，改工厂一处。

---

## 6. 怎么验证

### 6.1 跑测试

```bash
wsl.exe -d Ubuntu -- bash -lc 'cd /home/dong/projects/super_biz_agent_py && \
  .venv/bin/python -m pytest tests/unit/test_llm_factory.py -v --no-cov -p no:cacheprovider' 2>&1 | tr -d '\r'
```

11 个用例，重点看这四条：

- `test_timeout_defaults_to_config` —— 默认值从 config 读
- `test_explicit_timeout_takes_priority` —— 显式参数优先
- `test_zero_timeout_is_not_replaced_by_default[0]` / `[0.0]` —— `is None` 的护栏
- `test_timeout_is_always_set` —— 兜底保证：不传也一定有超时

### 6.2 亲手确认超时真的注入了

```bash
wsl.exe -d Ubuntu -- bash -lc 'cd /home/dong/projects/super_biz_agent_py && \
  .venv/bin/python -m pytest tests/unit/test_llm_factory.py::TestTimeoutAndRetries -v --no-cov -p no:cacheprovider' 2>&1 | tr -d '\r'
```

### 6.3 确认没有绕过工厂的构造点

这条是最有价值的检查 —— 它守的是「唯一入口」这个性质本身：

```bash
wsl.exe -d Ubuntu -- bash -lc 'cd /home/dong/projects/super_biz_agent_py && \
  grep -rn "ChatOpenAI(\|ChatQwen(" app/ --include=*.py | grep -v llm_factory.py' 2>&1 | tr -d '\r'
```

**期望输出：空。** 有任何一行输出，就意味着有一个绕过工厂的构造点，
它没有超时防护。这条命令值得加进 CI。

### 6.4 观察总预算的行为

```bash
wsl.exe -d Ubuntu -- bash -lc 'cd /home/dong/projects/super_biz_agent_py && \
  grep -n "chat_total_budget_seconds" app/config.py app/agent/rag_v2/service.py' 2>&1 | tr -d '\r'
```

想临时试探预算触发时的降级行为，把 `.env` 里的
`CHAT_TOTAL_BUDGET_SECONDS` 调到 `2`，然后发一个 chat_v2 请求 ——
应该拿到 `degrade_reason: "total_budget_exceeded"` 和一句说明卡在哪的答案，
而不是空响应。

---

## 7. 常见坑

### 7.1 `timeout or default` —— 已经讲过，但值得再说

这是本篇最容易被复制到别处的坑。任何「参数为空时取默认值」的地方，
先问一句：**这个参数的合法取值里有 falsy 值吗？**

| 参数类型 | 合法 falsy 值 | 该用 |
|---|---|---|
| `timeout: float` | `0`、`0.0` | `is None` |
| `max_retries: int` | `0` | `is None` |
| `verbose: bool` | `False` | `is None` |
| `items: list` | `[]` | `is None` |
| `name: str` | 通常空串就该当没传 | `or` 可以 |

### 7.2 以为「不传 timeout」会退化到某个合理的默认值

这条值得单独实测，因为直觉在这里是错的 —— 我最初在源码注释里写的是
「不传 timeout 时底层 httpx 使用其默认值」，实测发现比这**严重得多**。

先看 `openai` SDK 自己的默认值（`openai/_constants.py`）：

```
DEFAULT_TIMEOUT = Timeout(connect=5.0, read=600, write=600, pool=600)
```

读取超时 600 秒。长，但**是有限的**，卡住的连接终究会被放开。

问题在于这个默认值根本没机会生效。看 langchain 的实现
（`langchain_openai/chat_models/base.py`）：

```python
# 596-598 行：字段默认值是 None
request_timeout: float | tuple[float, float] | Any | None = Field(
    default=None, alias="timeout"
)

# 953-961 行：构造 SDK client 的参数
client_params: dict = {
    "organization": self.openai_organization,
    "base_url": self.openai_api_base,
    "timeout": self.request_timeout,     # ← 无条件传，值是 None
    ...
}
if self.max_retries is not None:         # ← 有条件传
    client_params["max_retries"] = self.max_retries
```

对比这两行是关键。`max_retries` 是**有条件**传的，不设置时这个键
根本不进 `client_params`，SDK 于是用自己的 `DEFAULT_MAX_RETRIES = 2`。
而 `timeout` 是**无条件**传的，不设置时等于显式传了一个 `None` 进去 ——
把 SDK 的 `DEFAULT_TIMEOUT` 顶掉了。

`openai` SDK 区分「没传」和「显式传 None」，用的是哨兵值 `NOT_GIVEN`：

```
不传(NOT_GIVEN) -> httpx=Timeout(connect=5.0, read=600, write=600, pool=600)
显式 None       -> httpx=Timeout(timeout=None)      ← 无限期
显式 60.0       -> httpx=Timeout(timeout=60.0)
```

langchain 走的是第二行。实测印证：

```
ChatOpenAI 默认 request_timeout  = None
root_client._client.timeout      = Timeout(timeout=None)   ← 完全无超时

ChatOpenAI 默认 max_retries      = None
root_client.max_retries          = 2                       ← SDK 默认生效了
```

自己跑一遍：

```bash
wsl.exe -d Ubuntu -- bash -lc 'cd /home/dong/projects/super_biz_agent_py && .venv/bin/python -c "
import openai
from langchain_openai import ChatOpenAI
print(\"SDK  DEFAULT_TIMEOUT     =\", openai._constants.DEFAULT_TIMEOUT)
print(\"SDK  DEFAULT_MAX_RETRIES =\", openai._constants.DEFAULT_MAX_RETRIES)
m = ChatOpenAI(model=\"x\", api_key=\"k\")
print(\"langchain 默认 timeout    =\", m.root_client._client.timeout)
print(\"langchain 默认 max_retries=\", m.root_client.max_retries)
"' 2>&1 | tr -d '\r'
```

所以「上游连上了但不返回数据」这个场景下，改造前的行为是
**永远等下去**，没有任何一层会把它放开。这就是第 1 节那个连接池被
占满的故障能成立的原因 —— 挂住的请求不会自己消失。

三条可迁移的教训：

1. **不要依赖三方库的默认超时。** 不只因为默认值可能不合适，
   更因为你依赖的其实是「A 库如何把参数转给 B 库」这个**中间层行为**，
   它不在任何一方的文档契约里，升级时可能悄悄变。
2. **看到 `Optional` 参数默认 `None`，不要假设它等于「用底层默认」。**
   要区分「不传」和「传 None」两种语义 —— 很多库（包括 `openai`）
   用 `NOT_GIVEN` 这类哨兵区分二者，而包装层往往把它们抹平了。
3. **验证方法是读实际生效的值，不是读文档。** 上面那条命令读的是
   `root_client._client.timeout`，也就是 httpx 真正会用的东西。
   下钻到最终消费者才叫验证过。

### 7.3 只加单次超时，以为万事大吉

单次超时只保护「一次调用」。任何**串行多次调用**的链路
都需要额外的整体预算。判断方法：数一下一个请求最多调几次外部依赖，
乘上单次超时，看这个数字用户能不能接受。

我们这里是 3 × 60 = 180s，显然不行，所以加了 90s 总预算。

### 7.4 总预算用 `ainvoke` 包，把部分结果丢干净

```python
# ❌ 超时 = 全部结果丢失
async with asyncio.timeout(budget):
    result = await graph.ainvoke(inputs)

# ✅ 超时 = 保留最后一个完整快照
last_state = {}
async with asyncio.timeout(budget):
    async for state in graph.astream(inputs, stream_mode="values"):
        last_state = state
```

差别只有两行代码，但决定了超时时用户是「拿到一句超时」
还是「拿到已检索的文档 + 说明」。

注意 `stream_mode` 的选择也有讲究：`"values"` 吐**完整快照**（适合累积状态），
`"updates"` 吐**增量补丁**（适合转发给前端）。
`query` 用前者，`query_stream` 用后者，各取所需。

### 7.5 `ChatQwen` 的 `timeout` 参数名陷阱

`ChatQwen` 继承自 `BaseChatOpenAI`，pydantic 字段名是 `request_timeout`，
但带了 alias `"timeout"`。所以：

```python
ChatQwen(timeout=60)            # ✅ 走 alias，能生效
ChatQwen(request_timeout=60)    # ✅ 走字段名，也能生效
model.request_timeout           # ← 读的时候用字段名
```

写入用 `timeout=`（与 `ChatOpenAI` 路径保持一致），
断言时用 `model.request_timeout`。这个不一致踩过一次，
所以源码里留了注释说明「已通过 model_fields 校验」。

### 7.6 mock config 时漏给新字段

`tests/unit/test_llm_factory.py:77-85` 有个注释记录了这个坑：

```python
    def test_model_falls_back_to_config_when_none(self) -> None:
        with patch("app.core.llm_factory.config") as mock_config:
            mock_config.dashscope_model = "qwen-plus"
            # 必须一并给出超时相关默认值：factory 现在会读它们，
            # MagicMock 属性会被 pydantic 拒绝（不是合法的 float/int）。
            mock_config.llm_timeout_seconds = 60.0
            mock_config.llm_max_retries = 2
```

`patch` 出来的 `MagicMock` 访问任意属性都会返回一个新的 `MagicMock`，
而 pydantic 校验 `float` 字段时会拒绝它，报一个跟超时八竿子打不着的
`ValidationError`。给 factory 加读取新 config 字段的代码时，
所有 mock 了 config 的既有测试都要补上对应字段。

这也是个信号：**`patch` 整个 config 对象是脆弱的做法。**
更健壮的是 `monkeypatch.setattr("app.core.llm_factory.config.llm_timeout_seconds", 45.0)`
—— 只替换需要的那一个字段，其余保持真实值。

---

## 8. 一句话总结

超时的价值不在于「让请求失败」，而在于**把无限期等待转换成明确失败**——
只有明确的失败才能被计数、分类、降级。而单次超时和整体预算回答的是
两个不同的问题（「这次调用最多挂多久」vs「这个用户最多等多久」），
必须都有。至于收敛到工厂，收的不是代码行数，是**「超时该是多少」这个决策**。
