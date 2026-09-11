# request_id 全链路追踪

> 上一篇建立了「失败的词汇表」。这一篇解决另一个正交的问题：
> **这一行日志，属于哪一次请求。**
>
> 两者合起来才能回答排障时的第一个问题 ——
> 「用户说他 10 点 03 分那次提问出错了，给我看那一次的完整链路」。

## 1. 为什么要改

### 1.1 一个捞不出来的日志

用户报障：「我刚才问了个问题，转了半天出来一句『暂时无法回答』。」

你打开日志，看到的是这样：

```text
2026-08-25 10:03:12 | INFO  | nodes.rewrite_node:118      | [rag_v2.rewrite] 原始问题: 报销流程是什么
2026-08-25 10:03:12 | INFO  | nodes.rewrite_node:118      | [rag_v2.rewrite] 原始问题: 年假怎么算
2026-08-25 10:03:13 | INFO  | nodes.retrieve_each_node:180| [rag_v2.retrieve_each] 子查询 3 条
2026-08-25 10:03:13 | WARNING| nodes.retrieve_each_node:196| [rag_v2.retrieve_each] 子查询 '差旅标准' 检索失败
2026-08-25 10:03:14 | INFO  | nodes.generate_node:305     | [rag_v2.generate] 开始生成
2026-08-25 10:03:14 | ERROR | nodes.generate_node:340     | [rag_v2.generate] LLM 调用失败
2026-08-25 10:03:14 | INFO  | nodes.generate_node:305     | [rag_v2.generate] 开始生成
```

现在回答这几个问题：

- 第 4 行那个检索失败，是「报销流程」那次请求的，还是「年假」那次的？
- 第 6 行 LLM 失败的，是哪一次？
- 用户说的那次到底是哪一条？

**答不出来。** 十个并发请求的日志在同一个文件里逐行交织，
每一行都只知道自己在哪个节点，不知道自己属于哪次请求。

有人会说「按时间窗口捞」。10:03:12 到 10:03:14 这两秒里有 7 行日志，
分属至少 3 次请求 —— 时间窗口切不开它们，因为它们本来就是同时发生的。

### 1.2 已有的 trace_id 为什么不够

项目里本来有 `ChatRunTrace` 表，一次对话会落一条记录，有 id。
但它救不了这个场景，原因有两个：

**第一，它诞生得太晚。** `ChatRunTrace` 是在整条链路**跑完之后**才落库的，
`trace_id` 在那一刻才存在。而请求执行期间产生的几十行日志，
当时根本没有 id 可用。

**第二，请求挂掉时它根本不存在。** 如果链路在中途崩了 ——
恰恰是最需要排障的情况 —— 落库那一步压根没执行，
这次请求在数据库里没有留下任何痕迹。日志里那几行报错就成了孤儿。

所以需要一个**在请求最开始就存在、并且贯穿全程**的 id。
这就是 request_id 与 trace_id 的分工：

| | 何时产生 | 存在于 | 失败时 |
|---|---|---|---|
| `request_id` | HTTP 入口，第一时间 | 日志、响应头、job meta | **仍然存在** |
| `trace_id` | 链路跑完，落库时 | 数据库 | 可能不存在 |

### 1.3 还有一段更黑的暗区

文件上传触发异步索引。API 侧返回 202，真正干活的是另一个进程里的 RQ worker。
索引失败了，你想查原因 —— 而 worker 的日志和 API 的日志之间
**没有任何关联字段**。你只能靠「大概那个时间上传的那个文件」去猜。

而失败恰恰绝大多数发生在 worker 侧（那里才有真正的 IO 和解析）。
也就是说，最需要追踪的那一段，是完全断开的。

## 2. 改之前什么样

改造前 `app/core/request_context.py`、`app/core/middleware.py`、
`app/core/job_context.py` 三个文件都不存在。日志配置是这样：

```python
# app/utils/logger.py（改造前，完整）
"""日志配置模块

使用 Loguru 配置应用日志
"""

import sys
from loguru import logger
from app.config import config


def setup_logger():
    logger.remove()

    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{module}</cyan>.<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level="DEBUG" if config.debug else "INFO",
        colorize=True,
        backtrace=True,
        diagnose=config.debug,
    )

    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="7 days",          # ← 后面会改
        compression="zip",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
               "{module}.{function}:{line} | {message}",   # ← 纯文本
    )
```

两个问题：

1. **格式串里没有任何请求标识** —— 就是第 1.1 节那个捞不出来的日志。
2. **文件输出是纯文本** —— 想接 Loki/ELK 必须先写正则去解析
   `时间 | 级别 | 模块.函数:行号 | 消息` 这个格式。格式一改解析就崩。

## 3. 改之后什么样

全链路七段，一段一段看：

```text
① HTTP 入口         RequestIdMiddleware        取/生成 id
② 协程隔离的存储     ContextVar                 全程可读，不用层层传参
③ 每一条日志         logger.configure(patcher)  业务代码零改动
④ 响应头            X-Request-ID               用户截图里就有 id
⑤ 落库              trace / diagnosis 两张表    日志与数据库能对上
⑥ 跨进程            job.meta                   API 侧 → worker 侧
⑦ worker 日志       with_job_request_id 装饰器  worker 那段不再是孤岛
```

### 3.1 第 ② 段：为什么是 ContextVar

先讲存储，因为它决定了其他六段的写法。

```python
# app/core/request_context.py:33
_request_id: ContextVar[str] = ContextVar("request_id", default=NO_REQUEST_ID)
```

**ContextVar 是协程隔离的。** asyncio 里多个请求在**同一个线程**上交替执行，
所以普通全局变量会被互相覆盖 —— 请求 A 设了 id，await 让出控制权，
请求 B 把它改成自己的，A 恢复执行时读到的是 B 的 id。
这比没有 id 更糟，因为它会**把两个请求的日志错误地关联在一起**。

ContextVar 的值绑定在「当前执行上下文」上，各协程各有一份。
而且 `asyncio.to_thread` 派生的线程会**复制**当前 context，
所以线程池里的代码也读得到 —— 这一点对项目里的同步阻塞调用很重要。

另一个不用「层层传参」的理由：那需要修改**每一个函数签名**。
从 API 到 service 到 node 到 repository，几十个函数多一个参数，
而且任何一个人漏传一层，下游就全断了。ContextVar 是隐式传递，
新增的代码自动就在链路里。

`default=NO_REQUEST_ID` 是必须给的：

```python
# app/core/request_context.py:29
NO_REQUEST_ID = "-"
```

脱离请求上下文的调用（启动脚本、worker 启动阶段、单元测试）
直接读也不会抛 `LookupError`。用 `"-"` 而不是空字符串，
是为了让日志列宽稳定，肉眼扫起来整齐。

### 3.2 一个容易忽略的细节：日志用 `"-"`，数据库用 `NULL`

```python
# app/core/request_context.py:50
def get_request_id_or_none() -> str | None:
    """读取当前 request_id，脱离请求上下文时返回 None。"""
    rid = _request_id.get()
    return None if rid == NO_REQUEST_ID else rid
```

为什么要有**两个**读函数？因为「没有值」在两个地方的正确表示不同：

- **日志用 `"-"`**：展示层需要固定列宽。
- **数据库用 `NULL`**：`NULL` 的语义是「没有这个值」，
  而 `"-"` 会被当成**一个真实取值**参与 `GROUP BY` 和关联查询 ——
  所有无上下文的记录会被聚成一个假的「请求 `-`」分组，污染统计。

这个函数存在的意义就是让这条判断只有一处实现（DRY）。
`chat_run_trace` 和 `aiops_diagnosis` 两个 repository 都要落 request_id，
各自写一遍 `None if rid == NO_REQUEST_ID else rid` 就是等着两边跑偏。

### 3.3 第 ① 段：中间件，以及为什么是纯 ASGI

```python
# app/core/middleware.py:47
class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # lifespan / websocket 等非 http 作用域直接放行：
        # 它们没有请求头，也不存在「一次请求」的概念。
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = _sanitize_incoming(incoming) or new_request_id()

        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            # 只有 response.start 这一个 message 携带响应头
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        token = set_request_id(request_id)
        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            reset_request_id(token)
```

**为什么写成纯 ASGI 类，而不是省事的 `@app.middleware("http")`？**

`@app.middleware("http")` 底层是 Starlette 的 `BaseHTTPMiddleware`，
它会把响应体包一层 anyio 内存流再转发。对普通 JSON 响应没影响，
但项目里有 SSE 端点（`app/api/chat_v2.py` 的 `EventSourceResponse`），
**额外的缓冲层会影响逐块下发的及时性和断连感知**。
纯 ASGI 中间件只是在 send 回调上加了一行 header，对流式响应零干扰。

**为什么 `finally` 里必须 reset？**

ASGI 服务器的 task 上下文可能被复用。不还原会让下一个请求
继承上一个的 id —— 又是那个「把两个请求错误关联在一起」的问题。

### 3.4 为什么要校验上游传入的 header（这是安全边界）

```python
# app/core/middleware.py:34
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def _sanitize_incoming(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip()
    if not _SAFE_REQUEST_ID.match(candidate):
        return None
    return candidate
```

复用上游传入的 id 是为了跨系统串联（网关已经有 id 时，两边的日志能对上）。
但这个值来自外部，必须校验。两个理由，第一个是安全问题：

**1. 日志伪造（log injection）。** 这个值会出现在**每一行日志**里。
如果放任 `\n` 通过，攻击者传这样一个 header：

```http
X-Request-ID: abc\n2026-08-25 10:00:00 | INFO | auth.login:42 | 管理员登录成功
```

日志文件里就凭空多出一整行**看起来完全合法**的记录。
在一个把日志当作审计证据的系统里，这足以掩盖真实的攻击痕迹，
或者栽赃给别人。这类漏洞有正式名字：CWE-117 Improper Output Neutralization for Logs。

**2. 存储与可读性。** 它还要落到数据库列里。超长值会撑爆字段、污染日志宽度。

**校验不通过时不报错，直接当作「没传」生成一个新的。**
这是刻意的选择：追踪 id 不该成为请求失败的理由。
为了一个畸形的 header 就返回 400，用户会遇到一个完全无法理解的错误，
而我们想要的只是一个能用的 id。

### 3.5 那个 `scope["state"]`：覆盖异常路径

```python
# app/core/middleware.py:86
scope.setdefault("state", {})["request_id"] = request_id
```

明明已经有 ContextVar 了，为什么还要往 scope 里写一份？
**这不是冗余，是为了覆盖异常路径。** 看 Starlette 的中间件栈：

```text
ServerErrorMiddleware        ← Exception 处理器在这里（最外层）
  └─ RequestIdMiddleware     ← 我们的 finally: reset_request_id 在这里
      └─ ExceptionMiddleware ← AppError 处理器在这里（最内层）
          └─ router
```

一个未分类异常从 router 冒泡到最外层时，会**先穿过** `RequestIdMiddleware`。
穿过意味着那个 `finally` 已经执行完了 —— ContextVar 已被 reset。
等到最外层的处理器想读 `get_request_id_or_none()`，拿到的是 `None`。

结果就是：**错误响应体里的 request_id 恒为 null。**
而出错恰恰是最需要它的时候。

scope 是一个贯穿整条调用链的**同一个 dict**，异常穿过中间件不会改变它，
所以它能安全地把 id 递到最外层。用 `scope["state"]` 而不是自造一个顶层键，
是因为 Starlette 已经把这个位置约定为「应用自定义数据」，
读侧可以直接 `request.state.request_id`。

对应的读侧（第 18 篇会细讲）：

```python
# app/core/exception_handlers.py:87 附近
# 先读 scope，再退回 ContextVar
```

### 3.6 第 ③ 段：一行 patcher 让数百处日志调用零改动

```python
# app/utils/logger.py:30
def _patch_request_id(record: dict) -> None:
    record["extra"].setdefault("rid", get_request_id() or NO_REQUEST_ID)


def setup_logger():
    logger.remove()
    # 注册 patcher：必须在 add() 之前，否则已注册的 handler 拿不到 rid。
    logger.configure(patcher=_patch_request_id)

    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<magenta>{extra[rid]}</magenta> | "          # ← 新增这一列
            "<cyan>{module}</cyan>.<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        ...
    )
```

**为什么用 patcher，而不是让每个调用点自己 `logger.bind(rid=...)`？**

项目里有数百处 `logger.info(...)`，还有 langchain / uvicorn 等第三方库的日志。
patcher 是唯一的「一处实现，全局生效」的切面（DRY）。
如果靠每个调用点自己 bind，第三方库的日志永远拿不到 rid，
而且任何新写的代码都可能忘记。

**两个坑，都踩过：**

坑一，关键字是 `patcher` 而不是 `patch`。loguru 里 `logger.patch()`
是一个返回新 logger 的**方法**，而 `configure(patcher=...)` 才是给全局装切面。
写错的话不报错，只是 rid 永远不生效。

坑二，用 `setdefault` 而不是直接赋值，有两个原因：

```python
record["extra"].setdefault("rid", ...)   # 不是 record["extra"]["rid"] = ...
```

- 保留调用方的显式意图 —— `logger.bind(rid="job-123")` 不该被覆盖。
- **防 KeyError**。格式串里写了 `{extra[rid]}`，一旦某条记录缺这个键，
  loguru 格式化时会抛 KeyError。而**日志系统自身抛异常是最糟的故障**：
  它会掩盖真正的错误。你在查一个 bug，结果日志系统先崩了。

顺手把文件输出改成 JSON：

```python
        retention="30 days",   # 原来是 7 days
        serialize=True,        # 输出 JSON 行
        format="{message}",
```

`retention` 从 7 天改成 30 天的理由：故障 case 常常几周后才被复盘追问，
7 天太短 —— 等真要查的时候日志已经被删了。压缩后单日体积很小。

`serialize=True` 让文件输出变成 JSON 行，`record.extra.rid`
直接就是可查询字段，不需要写正则去解析。控制台仍然是彩色文本给人看 ——
**同一份日志，两种消费者，两种格式**。

### 3.7 第 ④ 段：响应头，以及 CORS 的最后一公里

```python
# app/main.py:59
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    ...
    expose_headers=[REQUEST_ID_HEADER],   # ← 关键
)

# app/main.py:76
app.add_middleware(RequestIdMiddleware)
```

**`expose_headers` 不写，前端就拿不到。** CORS 默认只向 JS 暴露
6 个「安全响应头」（Cache-Control、Content-Type 等），
自定义头必须显式 expose。不写的话前端
`response.headers.get("X-Request-ID")` 拿到 `null` ——
链路追踪的最后一公里断在浏览器里，用户报障时给不出这个 id。

**中间件顺序：`RequestIdMiddleware` 在 CORS 之后添加。**

Starlette 的 `add_middleware` 是 `insert(0)`，构建时再 `reversed` ——
也就是**最后添加的位于最外层、最先执行**。放最外层有两个理由：

1. ContextVar 要在任何业务代码（**含异常处理器**）之前设好，
   否则那些日志拿不到 rid。
2. 响应头由它最后回写，连 CORS 预检和错误响应都会带上 `X-Request-ID`。

这个顺序反了不会报错，只会让「错误响应没有 rid」这个问题悄悄回来。

### 3.8 第 ⑥⑦ 段：跨进程，ContextVar 到这里就断了

```python
# app/core/job_context.py:37
JOB_META_REQUEST_ID = "request_id"


def enqueue_with_request_id(queue, func, *args, **kwargs) -> Job:
    """入队并把当前 request_id 带进 job.meta。"""
    meta: dict[str, Any] = dict(kwargs.pop("meta", None) or {})
    meta.setdefault(JOB_META_REQUEST_ID, get_request_id_or_none())
    return queue.enqueue(func, *args, meta=meta, **kwargs)
```

**ContextVar 只在同一个进程的同一个执行上下文里有效。**
入队时是 HTTP 请求进程持有 rid，而真正干活的是**另一台机器上的 worker 进程**。
两者之间唯一的通道是 Redis 里那条 job 记录。

所以必须显式「序列化进去、反序列化出来」。worker 侧：

```python
# app/core/job_context.py:76
def with_job_request_id(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with job_request_id_scope():
            return func(*args, **kwargs)

    return wrapper
```

用装饰器而不是在每个 worker 函数里手写 `with`：worker 函数体都是
「try 一大段 / except 落库失败状态」的结构，手写 with 会让整块代码
再缩进一层，diff 噪音大且容易漏。

`functools.wraps` 在这里**不是可选的美化**：RQ 通过
「模块路径 + 函数名」反序列化任务函数，不保留
`__name__` / `__qualname__` / `__module__` 的话，入队时写进 Redis 的
函数路径就是 `app.core.job_context.wrapper`，worker 反序列化直接失败。

（对比一下：第 20 篇的 `instrument_node` **刻意不用** `functools.wraps`，
因为 LangGraph 可能会跟着 `__wrapped__` 找到原函数，绕过埋点。
同一个工具，在两个场景下的结论相反 —— 判断依据是「谁会来读这些属性」。）

`JOB_META_REQUEST_ID` 定成常量而不是字面量：入队侧和 worker 侧
必须用同一个键，写错一个字母会**静默失效** —— 没有任何报错，
只是 rid 永远取不到。

`job_request_id_scope` 在取不到 job 或 meta 里没有 rid 时，
会**生成一个新的**而不是留空：后台任务哪怕是独立触发的，
也该有自己的追踪标识，否则它那一整段日志全落在 `"-"` 上，等于没有。

### 3.9 改完之后，第 1.1 节那段日志

```text
2026-08-25 10:03:12 | INFO   | a3f1c8d29b4e5f60 | nodes.rewrite_node:118      | [rag_v2.rewrite] 原始问题: 报销流程是什么
2026-08-25 10:03:12 | INFO   | 7b2e91a4c6d30f85 | nodes.rewrite_node:118      | [rag_v2.rewrite] 原始问题: 年假怎么算
2026-08-25 10:03:13 | INFO   | a3f1c8d29b4e5f60 | nodes.retrieve_each_node:180| [rag_v2.retrieve_each] 子查询 3 条
2026-08-25 10:03:13 | WARNING| a3f1c8d29b4e5f60 | nodes.retrieve_each_node:196| [rag_v2.retrieve_each] 子查询 '差旅标准' 检索失败[milvus_error]
2026-08-25 10:03:14 | INFO   | 7b2e91a4c6d30f85 | nodes.generate_node:305     | [rag_v2.generate] 开始生成
2026-08-25 10:03:14 | ERROR  | a3f1c8d29b4e5f60 | nodes.generate_node:340     | [rag_v2.generate] LLM 调用失败[llm_timeout]
2026-08-25 10:03:14 | INFO   | a3f1c8d29b4e5f60 | nodes.generate_node:305     | [rag_v2.generate] 开始生成
```

现在 `grep a3f1c8d29b4e5f60` 就是那一次请求的完整故事：
改写 → 检索（一条子查询挂了）→ 生成 → LLM 超时。
而用户手上有这个 id，因为它在响应头里。

注意方括号里那些错误码（`milvus_error`、`llm_timeout`）—— 那是第 14 篇的产物。
**两篇合起来才完整**：request_id 告诉你「哪一次」，
错误码告诉你「为什么」。少任何一个，日志都还是不可用。

## 4. 背后的工程原理

### 4.1 这就是分布式追踪的最小可用版本

工业界的标准方案是 OpenTelemetry，它的核心概念是：

```text
trace_id   一次完整调用的唯一标识（跨服务）
span_id    调用链上的一个环节
parent_id  span 之间的父子关系
baggage    随调用链传播的键值对
```

我们做的 `request_id` 就是其中的 `trace_id`，
传播机制（ContextVar + header + job meta）就是简化版的 context propagation。
第 20 篇会补上 span 那一层。

**为什么不直接上 OpenTelemetry？** 这是一个真实的取舍：

上 OTel 需要引入 SDK、配置 exporter、部署 collector 和后端（Jaeger/Tempo），
而收益要在有多个服务时才显现。本项目是**单体 + 一个 worker**，
需要跨越的进程边界只有一个。用一百行代码解决一个进程边界的问题，
比引入一整套基础设施更符合当下（YAGNI）。

但接口是兼容的：`X-Request-ID` 是事实标准 header，
将来上 OTel 时把 `set_request_id` 换成 OTel 的 context API，
业务代码一行不用改 —— 因为业务代码从来没有直接接触过 ContextVar。
**这就是把传播机制收在一个模块里的价值。**

### 4.2 隐式传递 vs 显式传参：一次真实的取舍

ContextVar 是**隐式**的。这有代价：读代码时你看不出 `logger.info(...)`
为什么会带上 rid，必须知道有一个 patcher 存在。这是「魔法」，
而魔法通常是要避免的。

那为什么这里还是选它？因为显式传参的代价更大：

```python
# 显式传参的样子
async def rewrite_node(state, *, request_id: str): ...
async def retrieve_each_node(state, *, request_id: str): ...
def search(query, *, request_id: str): ...
def embed(texts, *, request_id: str): ...
```

- 几十个函数签名要改，包括 LangGraph 节点（它的签名是框架约定的，改不了）。
- **任何一层漏传，下游全断**，而且不会报错，只是 rid 变成 None。
- 第三方库的日志永远拿不到（我们改不了 langchain 的签名）。

判断标准是：**这个值是不是「环境属性」而非「业务参数」。**
request_id 不参与任何业务逻辑，没有任何函数会根据它改变行为 ——
它只是「当前正在处理哪个请求」这个环境事实。环境属性适合隐式传递。
反过来，`user_id` 就该显式传，因为鉴权逻辑真的会读它。

### 4.3 为什么响应头这么重要

`expose_headers` 那三行代码看起来很小，但它接的是**人的环节**：

```text
用户看到报错 → 截图 → 截图里有 request_id → 你 grep 一下 → 3 秒定位
用户看到报错 → 截图 → 没有 id → 「大概几点？」→ 时间窗口捞 → 几十次请求里猜
```

可观测性的最后一公里往往在系统之外。一个只有工程师能查的追踪系统，
和一个用户能直接把 id 报给你的追踪系统，排障效率差一个量级。

### 4.4 面试怎么讲这一段

> 我做了 request_id 全链路透传。技术上不复杂，但有几个判断点值得说：
>
> 存储用 ContextVar 而不是显式传参，判断依据是「这是环境属性不是业务参数」——
> 没有任何函数会根据它改变行为。代价是隐式（读代码时看不出日志为什么带 rid），
> 收益是几十个函数签名不用改，而且第三方库的日志也能带上。
>
> 日志注入用 loguru 的 `configure(patcher=...)` 切面，一处实现全局生效，
> 数百处既有 logger 调用零改动。
>
> 三个坑值得讲：
>
> 一是**异常路径**。Starlette 的 ServerErrorMiddleware 在中间件栈最外层，
> 异常冒泡到那里时我的 finally 已经 reset 了 ContextVar，
> 错误响应里 rid 恒为 null —— 而出错恰恰最需要它。解法是同时往
> ASGI scope 里存一份，scope 是贯穿全链路的同一个 dict。
>
> 二是**外部传入的 header 必须校验**。这个值会进每一行日志，
> 放任换行符通过就是日志注入漏洞（CWE-117），攻击者能凭空造出一行假日志。
> 但校验失败不报 400，而是当作没传重新生成 —— 追踪 id 不该让请求失败。
>
> 三是**跨进程那一段 ContextVar 会断**。RQ worker 是另一个进程，
> 必须显式写进 job.meta 再读回来。而失败大多发生在 worker 侧，
> 所以这一段恰恰是最需要追踪的。

## 5. 怎么验证

### 5.1 单元测试

```bash
wsl.exe -d Ubuntu -- bash -lc 'cd /home/dong/projects/super_biz_agent_py && \
  .venv/bin/python -m pytest tests/unit/test_request_context.py -v --no-cov'
```

### 5.2 手动验证协程隔离（最关键的性质）

```bash
wsl.exe -d Ubuntu -- bash -lc 'cd /home/dong/projects/super_biz_agent_py && \
  .venv/bin/python -c "
import asyncio
from app.core.request_context import request_id_scope, get_request_id

async def worker(name, rid, delay):
    with request_id_scope(rid):
        await asyncio.sleep(delay)          # 交出控制权，让别的协程跑
        got = get_request_id()
        print(f\"{name}: 设的是 {rid}, 读到的是 {got}, {\"OK\" if got == rid else \"串了！\"}\")

async def main():
    await asyncio.gather(
        worker(\"请求A\", \"aaaa1111\", 0.03),
        worker(\"请求B\", \"bbbb2222\", 0.01),
        worker(\"请求C\", \"cccc3333\", 0.02),
    )
    print(\"退出作用域后:\", repr(get_request_id()))

asyncio.run(main())
"'
```

预期：三个协程各读到自己的 id，最后回到 `'-'`。
如果换成普通全局变量，B 会覆盖 A，A 读到 `bbbb2222`。

### 5.3 手动验证 header 校验（安全边界）

```bash
wsl.exe -d Ubuntu -- bash -lc 'cd /home/dong/projects/super_biz_agent_py && \
  .venv/bin/python -c "
from app.core.middleware import _sanitize_incoming

cases = [
    (\"gw-abc-123\", \"正常值，应复用\"),
    (\"a3f1c8d29b4e5f60\", \"我们自己生成的格式\"),
    (\"x\" * 200, \"超长，必须拒绝\"),
    (\"has space\", \"含空格，拒绝\"),
    (\"\", \"空值\"),
    (None, \"没传\"),
]
for value, desc in cases:
    got = _sanitize_incoming(value)
    verdict = \"复用\" if got else \"重新生成\"
    print(f\"{verdict:6} | {desc:24} | {value!r:.40}\")
"'
```

日志注入那一例要单独验 —— 直接把换行写进 shell 命令里会被 shell 自己吃掉，
根本传不进 Python。用 `chr(10)` 构造：

```bash
wsl.exe -d Ubuntu -- bash -lc 'cd /home/dong/projects/super_biz_agent_py && \
  .venv/bin/python -c "
from app.core.middleware import _sanitize_incoming

# 攻击载荷：一个换行，接一行看起来完全合法的伪造日志
evil = \"abc\" + chr(10) + \"2026-01-01 10:00:00 | INFO | auth.login:42 | 管理员登录成功\"
assert _sanitize_incoming(evil) is None, \"日志注入没有被拦住！\"
print(\"OK 带换行(LF)的 header 被拒绝，会重新生成一个干净的 id\")

assert _sanitize_incoming(\"abc\" + chr(13) + \"fake\") is None
print(\"OK 回车符(CR)同样被拒绝\")
"'
```

顺带说一句：跑第一个脚本时如果你用 `grep` 过滤输出，
那行伪造载荷很可能会**混进你的过滤结果**里 ——
因为它长得跟真日志一模一样。这就是这个漏洞的实际杀伤力。

### 5.4 端到端：响应头真的带回来了吗

```bash
# 起服务
wsl.exe -d Ubuntu -- bash -lc 'cd /home/dong/projects/super_biz_agent_py && \
  .venv/bin/python -m uvicorn app.main:app --port 8000'

# 另开一个终端：不传 id，看是否自动生成
curl -i http://localhost:8000/health 2>&1 | grep -i "x-request-id"

# 传一个自己的 id，看是否被复用
curl -i -H "X-Request-ID: my-trace-001" http://localhost:8000/health 2>&1 | grep -i "x-request-id"

# 传一个畸形的，看是否被换成新生成的（而不是报 400）
curl -i -H "X-Request-ID: bad value with spaces" http://localhost:8000/health 2>&1 | grep -i "x-request-id"
```

第二条应该回 `my-trace-001`，第三条应该回一个 16 位十六进制串。

### 5.5 端到端：日志能不能捞出一次请求

```bash
# 发一次请求，记下响应头里的 id
RID=$(curl -si -H "X-Request-ID: demo-trace-42" \
  -X POST http://localhost:8000/api/chat/v2 \
  -H "Content-Type: application/json" \
  -d '{"question":"报销流程是什么"}' | grep -i "^x-request-id" | tr -d '\r' | awk '{print $2}')

# 用它捞出这次请求的完整链路
grep "$RID" logs/app_$(date +%F).log | .venv/bin/python -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    print(f\"{r['record']['level']['name']:8} {r['record']['module']}.{r['record']['function']}:{r['record']['line']:<4} {r['record']['message']}\")
"
```

最后这个管道也顺便验证了 `serialize=True` 的收益：日志是 JSON，
可以直接用程序处理，不需要写正则。

## 6. 常见坑

**坑 1：忘了 `finally: reset_request_id(token)`。**

后果不是「rid 丢失」，而是**下一个请求继承上一个的 id**。
两次不相关的请求日志被关联在一起，比没有 id 更难排查 ——
因为你会以为看到的是一次请求的完整链路，而它实际上是两次拼起来的。

**坑 2：中间件顺序装反了。**

`add_middleware` 是 `insert(0)`，**最后添加的最先执行**。
这一点很反直觉，很容易写成「先加 RequestId 再加 CORS」。
装反了不报错，只是错误响应和 CORS 预检响应丢掉 rid。

**坑 3：以为 ContextVar 能穿过 `run_in_executor` 或新建的线程。**

`asyncio.to_thread` 会复制 context（可以），
但 `loop.run_in_executor(None, fn)` 和裸 `threading.Thread(target=fn)`
**不会** —— 新线程拿到的是空 context，rid 变成 `"-"`。
需要手动传：

```python
rid = get_request_id()
def wrapped():
    with request_id_scope(rid):
        return fn()
```

**坑 4：patcher 里用 `record["extra"]["rid"] = ...` 直接赋值。**

会覆盖调用方显式 `logger.bind(rid=...)` 的意图。用 `setdefault`。

**坑 5：格式串写了 `{extra[rid]}` 但 patcher 没注册（或注册在 `add()` 之后）。**

每条日志都会 KeyError。而日志系统崩掉会掩盖你正在排查的真正错误 ——
这是最难受的一类故障，因为你的排障工具本身坏了。
`logger.configure(patcher=...)` 必须在所有 `logger.add()` **之前**。

**坑 6：worker 侧忘了加 `@with_job_request_id`。**

不报错，只是那个任务的日志全是 `"-"`。而 worker 是失败最集中的地方。
新增 worker 函数时容易漏 —— 这一点没有自动保障，只能靠 review。

---

上一篇：[14-错误分类与降级原因体系](14-错误分类与降级原因体系.md)

下一篇：[16-LLM 超时与统一构造路径](16-LLM超时与统一构造路径.md) ——
有了词汇表和追踪 id，接下来处理第一个真正的失败源：
**没有超时的 LLM 调用**。
