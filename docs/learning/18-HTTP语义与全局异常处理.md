# 18 - HTTP 语义与全局异常处理

> 这一篇讲的是：**让 HTTP 状态码说真话**。
>
> 前四篇处理的是「系统内部对自己说谎」（把熔断说成模型故障、把检索故障说成知识库没有）。
> 这一篇处理的是「系统对外部世界说谎」—— 而外部世界里恰好住着所有的监控系统。
>
> 涉及文件：`app/core/exception_handlers.py`（新增）、`app/main.py`、
> `app/api/chat.py`、`app/api/chat_v2.py`

---

## 1. 为什么要改

### 1.1 一个具体的失败场景

凌晨三点，DashScope 区域性限流，所有 LLM 调用开始超时。

这个系统的实际状态是：**每一个请求都在失败**。用户点发送，转圈 90 秒，
得到一句「抱歉，处理失败」。

而监控大盘是这样的：

```
请求量        ████████████████  正常
错误率        0.00%             ← 绿的
5xx 计数      0                 ← 绿的
实例健康度    100%              ← 绿的
SLB 状态      全部 in-service   ← 绿的
Apdex         0.98              ← 绿的
```

没有任何告警触发。值班同学睡到天亮，早上被用户群里的截图叫醒。

**为什么会这样**，看改造前 `app/api/chat.py:46-56` 那段代码：

```python
except Exception as e:
    logger.error(f"chat endpoint error: {e}")
    return {
        "code": 500,          # ← 这个 500 写在 body 里
        "message": "error",
        "data": {
            "success": False,
            "answer": None,
            "errorMessage": str(e),
        },
    }
```

关键在于这是一个 `return`，不是 `raise`。

FastAPI 看到一个正常的返回值，于是发出：

```http
HTTP/1.1 200 OK
Content-Type: application/json

{"code": 500, "message": "error", "data": {...}}
```

**HTTP 状态行是 200。** body 里那个 `code: 500` 只是一个普通的 JSON 字段。

### 1.2 谁读 body，谁只读状态行

这是整篇的核心。`code: 500` 不是完全没用 —— 前端 JS 会读它，
`if (res.code !== 200) showError()` 这个分支是正常工作的。所以从产品视角看，
**功能是对的**：用户确实看到了错误提示。

问题在于，body 是给应用层读的，而下面这一整层基础设施**只看 HTTP 状态行**，
它们根本不解析你的 JSON：

| 组件 | 看什么 | 看到 200 的后果 |
|---|---|---|
| Nginx / 网关访问日志 | `$status` | 5xx 告警规则永不触发 |
| 负载均衡健康检查 | 状态码 | 挂掉的实例不会被摘除，继续接流量 |
| APM（SkyWalking / Sentry / OTel） | 状态码 | 错误率 0%，Apdex 满分，无异常事务 |
| Prometheus | `http_requests_total{status="5xx"}` | 该 series 恒为 0 |
| 云厂商 SLB 面板 | 状态码 | 一片绿 |
| CDN / API Gateway 熔断 | 状态码 | 不会对故障后端熔断 |

这些东西是**不能改的** —— 它们是标准，是整个行业的公共契约。
Nginx 不会为了你的项目去解析 body 里的 `code` 字段。

所以你有两个选择：说服全世界改成读你的 body，或者你自己说真话。

### 1.3 为什么这个项目里这件事格外讽刺

这是一个 **AIOps 项目**。它存在的意义就是帮别人发现故障、
分析告警、给出首响建议。

而它自己的故障，对所有标准监控系统隐身。

我们写的 SOP 里教值班同学「先看错误率曲线」，而我们自己的错误率曲线
在服务完全瘫痪时依然是 0%。

### 1.4 另一半问题：`HTTPException(500, detail=str(e))`

改造前 `chat.py` 里还有另一种写法（`chat.py:167-169`、`189-191`）：

```python
except Exception as e:
    logger.error(f"clear session endpoint error: {e}")
    raise HTTPException(status_code=500, detail=str(e))
```

这个至少状态码是真的。但它有另外三个毛病：

1. **恒为 500。** 上游超时也是 500，上游 502 也是 500。
   于是无法区分「我们的 bug」和「依赖挂了」—— 这两者的处置方向完全相反：
   前者要发版修代码，后者要找上游或者等它恢复。
2. **响应体形状不同。** `HTTPException` 产出 `{"detail": "..."}`，
   而另一批接口产出 `{"code":..., "message":..., "data":...}`。
   同一个 API 里两种错误结构，前端得写两套解析。
3. **散落各处。** 同一个仓库里两种失败写法并存，
   新写接口的人抄到哪个算哪个 —— 这正是「三套互相矛盾的降级策略」的来源。

---

## 2. 改之前什么样

### 2.1 `app/api/chat.py`（HEAD 版本）

```python
# app/api/chat.py:25-56（改造前）
        return {
            "code": 200,
            "message": "success",
            "data": {
                "success": True,
                "answer": answer,
                "errorMessage": None,
            },
        }

    except Exception as e:
        logger.error(f"chat endpoint error: {e}")
        return {
            "code": 500,
            "message": "error",
            "data": {
                "success": False,
                "answer": None,
                "errorMessage": str(e),
            },
        }
```

### 2.2 `app/api/chat_v2.py`（HEAD 版本）

```python
# app/api/chat_v2.py:45-55（改造前）
    except Exception as e:
        logger.error(f"chat v2 endpoint error: {e}")
        return {
            "code": 500,
            "message": "error",
            "data": {
                "success": False,
                "answer": None,
                "errorMessage": str(e),
            },
        }
```

注意这里连 `sub_queries` / `used_documents` / `validation` 都没有 ——
成功路径返回 7 个字段，失败路径返回 3 个。前端得判断字段存不存在。

### 2.3 全项目的失败写法清点

```
app/api/chat.py:46        return {"code": 500, ...}    → HTTP 200
app/api/chat.py:169       raise HTTPException(500)     → HTTP 500，形状是 {"detail":...}
app/api/chat.py:191       raise HTTPException(500)     → 同上
app/api/chat_v2.py:45     return {"code": 500, ...}    → HTTP 200
app/api/aiops.py:142      except → 塞进 stage 字段      → SSE，另一套
```

**三种写法，三种响应形状，两种状态码语义。** 这就是审查结论里
「三套互相矛盾的降级策略并存」在 HTTP 层的具体表现。

---

## 3. 改之后什么样

### 3.1 新增：`app/core/exception_handlers.py`

先看响应体构造。**关键约束是向后兼容**：JSON 结构一个字都不能变，
前端依赖 `code` / `data.success` / `data.errorMessage`。

```python
# app/core/exception_handlers.py:90-121
def build_error_body(
    exc: BaseException,
    *,
    http_status: int,
    request: Request | None = None,
) -> dict[str, Any]:
    """构造错误响应体。

    保持与改造前完全相同的三个键（`code` / `message` / `data`），
    `data` 内保留 `success` / `answer` / `errorMessage`。

    `code` 跟随 HTTP 状态码而不是恒为 500：前端判定成功的方式是
    `code === 200`，所以 500 变 504 不影响它的分支，但能让读日志的人
    从 body 就看出是超时而非普通失败。
    """
    request_id = _resolve_request_id(request)
    body: dict[str, Any] = {
        "code": http_status,          # ← 跟随真实状态码，不再恒为 500
        "message": "error",
        "data": {
            "success": False,
            "answer": None,
            # 保留自由文本给人看，但它不再是唯一的错误信息载体。
            "errorMessage": str(exc) or type(exc).__name__,
            # 以下三个是新增字段（加法，不破坏老前端）：
            # error_code 可聚合、degrade_reason 指明修复方向、request_id 用于串日志。
            "error_code": error_code_of(exc),
            "degrade_reason": degrade_reason_of(exc).value,
            "request_id": request_id,
        },
    }
    return body
```

逐点说明这几个字段的分工：

- **`errorMessage`** 是自由文本，给人看的。它**不能**用来做聚合 ——
  `"connect timeout to 10.0.0.5:19530"` 和 `"connect timeout to 10.0.0.6:19530"`
  是两条不同的字符串，指标会炸成两个 series（这正是第 22 篇讲的 label 基数问题）。
- **`error_code`** 是有限枚举（来自第 14 篇的 `error_code_of`），可以安全聚合。
- **`degrade_reason`** 指明**修复方向**：`llm_timeout` 去看模型服务，
  `retrieval_failed` 去看 Milvus，`circuit_open` 说明是我们自己熔断了。
- **`request_id`** 让用户报障时截个图，你就能直接去日志里捞整条链路（第 15 篇）。

### 3.2 两个处理器：分类过的 vs 没分类的

```python
# app/core/exception_handlers.py:148-158
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """处理我们自己抛出的 AppError —— 语义明确，状态码由异常类型决定。"""
    http_status = exc.http_status
    # AppError 是「预期内的失败」：已经分类过了，不需要 traceback 噪音。
    # 但 5xx 仍然按 error 记，4xx 按 warning —— 客户端参数错误不该污染错误率告警。
    log = logger.error if http_status >= 500 else logger.warning
    log(
        f"[{request.method} {request.url.path}] {exc.code} → HTTP {http_status}: "
        f"{exc.message} (retryable={exc.retryable})"
    )
    return _json_error_response(exc, http_status, request=request)
```

`AppError` 是我们主动抛的，第 14 篇已经给它标好了 `http_status`。
这里**不打 traceback** —— 一个 `RetrievalError` 的调用栈没有信息量，
我们早就知道它会从哪抛出来。打了只是噪音，还会把日志文件撑大。

`4xx 用 warning` 这条容易被忽略：客户端传了个非法参数不是我们的故障。
如果按 error 记，日志告警规则（通常是「error 级别每分钟超 N 条」）
会被一个乱试接口的爬虫触发。

```python
# app/core/exception_handlers.py:161-176
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底处理未被分类的异常。

    存在的意义：没有这个处理器，任何漏网异常都会走 Starlette 默认路径，
    返回一段 `Internal Server Error` 纯文本 —— 前端拿到的不是 JSON，
    解析直接炸，用户看到的是白屏而不是错误提示。

    这里仍然经过 `http_status_of`，所以裸 `asyncio.TimeoutError`
    也能正确变成 504 而不是笼统的 500。
    """
    http_status = http_status_of(exc)
    # 未分类异常需要完整堆栈：它代表我们没预料到的情况，是要修的 bug。
    logger.opt(exception=exc).error(
        f"[{request.method} {request.url.path}] 未处理异常 → HTTP {http_status}: {exc}"
    )
    return _json_error_response(exc, http_status, request=request)
```

这里和上面**恰好相反**：必须打完整堆栈。未分类异常代表「我们没预料到」，
那是个 bug，没有调用栈就修不了。

同时它仍然走 `http_status_of` —— 所以一个从第三方库里冒出来的裸
`asyncio.TimeoutError` 也能变成 504，而不是笼统的 500。
状态码判定逻辑只有一处实现（第 14 篇的 DRY）。

### 3.3 那个 ContextVar 陷阱

这是本篇最值得记住的一个细节。

```python
# app/core/exception_handlers.py:65-87
def _resolve_request_id(request: Request | None) -> str | None:
    """取当前请求的 request_id，优先读 ASGI scope，其次读 ContextVar。

    为什么不能只读 ContextVar —— 这是一个真实存在的坑：

    Starlette 把 `Exception` 处理器装在 **ServerErrorMiddleware**（最外层），
    而 `RequestIdMiddleware` 在它内侧，且带 `finally: reset_request_id(token)`。
    未分类异常冒泡到最外层时那个 finally 早已执行，ContextVar 已被还原 ——
    于是 `get_request_id_or_none()` 返回 None，错误响应体里的 request_id
    **永远是 null**。偏偏出错才是最需要这个 id 的时候。

    `AppError` 走的是最内层的 ExceptionMiddleware，ContextVar 还在，
    所以只有兜底处理器会踩到这个坑 —— 但两个处理器共用同一个取值函数，
    这条判断就只需要写一次（DRY）。

    scope 的 state 由中间件在入口写入，是一个贯穿整条链路的同一个 dict，
    异常穿越中间件不会影响它，因此能可靠地把 id 带到最外层。
    """
    if request is not None:
        rid = getattr(request.state, "request_id", None)
        if rid:
            return str(rid)
    return get_request_id_or_none()
```

画成图更清楚。Starlette 的中间件栈：

```
ServerErrorMiddleware          ← Exception 处理器挂在这
  └─ RequestIdMiddleware       ← set_request_id / finally: reset
      └─ ExceptionMiddleware   ← AppError 处理器挂在这
          └─ router → 你的接口
```

两条异常路径，命运完全不同：

```
AppError:
  接口 raise → ExceptionMiddleware 接住 → app_error_handler
                                          ↑ 还在 RequestIdMiddleware 内部
                                            ContextVar 有值  ✅

未分类 Exception:
  接口 raise → ExceptionMiddleware 不认识，继续上抛
            → 穿过 RequestIdMiddleware，触发 finally: reset_request_id ← 值没了
            → ServerErrorMiddleware 接住 → unhandled_exception_handler
                                          ↑ ContextVar 已被还原
                                            get_request_id() 返回 "-"  ❌
```

所以 `RequestIdMiddleware` 除了写 ContextVar，还往 ASGI scope 里存了一份
（见第 15 篇 `middleware.py:86`）。scope 是贯穿整条调用链的**同一个 dict**，
异常穿越中间件不会改变它。

**这个坑的隐蔽性在于**：它只在未分类异常时出现，而未分类异常在测试里往往
是用 `AppError` 模拟的 —— 于是测试全绿，线上却是「500 响应里 request_id 恒为 null」。
偏偏 500 才是最需要 request_id 的时候。

### 3.4 响应头也要自己写

```python
# app/core/exception_handlers.py:124-145
def _json_error_response(
    exc: BaseException,
    http_status: int,
    request: Request | None = None,
) -> JSONResponse:
    """生成错误响应，并把 request_id 也放进响应头。

    响应头本来由 RequestIdMiddleware 回写，这里再写一次不是冗余：
    未分类异常是在**它外层**的 ServerErrorMiddleware 里被兜住的，
    那时 `send_with_request_id` 这个包装已经不在调用链上了，
    响应头不会被补上。所以出错路径必须自己写。
    """
    headers = {}
    request_id = _resolve_request_id(request)
    if request_id:
        headers[REQUEST_ID_HEADER] = request_id

    return JSONResponse(
        status_code=http_status,
        content=build_error_body(exc, http_status=http_status, request=request),
        headers=headers or None,
    )
```

同一个根因的第二个表现。`RequestIdMiddleware` 是通过包装 `send` 回调来写响应头的，
而 `ServerErrorMiddleware` 在它外层 —— 那个包装已经不在调用链上。

### 3.5 刻意不接管 `HTTPException`

```python
# app/core/exception_handlers.py:179-190
def register_exception_handlers(app: FastAPI) -> None:
    """把处理器挂到应用上。

    只注册这两个，**不碰 HTTPException**：FastAPI 默认的 HTTPException
    处理器返回 `{"detail": "..."}`，现有接口（404 trace 不存在、400 未知告警来源）
    和前端都依赖这个形状，没有理由动它。

    这也是刻意保持窄的一个例子：全局处理器负责「没人管的异常」，
    不越权接管已经有明确语义的路径。
    """
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)  # type: ignore[arg-type]
```

这是 YAGNI 的一个具体应用。`HTTPException` 已经**做对了**：
`raise HTTPException(404, "trace not found")` 产出的就是 HTTP 404。
它唯一的问题是响应体形状与我们不同，而那个形状前端正在用。

统一它需要：改 `chat_v2.py:63` 的 404、改 `aiops.py` 的 400、
改前端两处解析、回归测试。收益是「形状一致」这个洁癖。

**不值得。** 这一批的目标是「让说谎的状态码说真话」，
`HTTPException` 从来没说谎。

### 3.6 接口侧的改动：`return` → `raise`

```python
# app/api/chat_v2.py:126-151（改造后）
    except Exception as e:
        # 先落 trace，再往上抛。
        #
        # 顺序很关键：trace 是排障的唯一线索，必须在异常离开这个函数之前写完。
        # 交给全局处理器再落库是做不到的 —— 那里拿不到 session_id / question，
        # 也没有这个请求的 db session。
        logger.error(f"chat v2 endpoint error: {e}")
        trace = trace_repo.create_trace(
            session_id=request.id,
            source="chat_v2",
            question=request.question,
            answer=None,
            sub_queries=[],
            retrieved_count=0,
            used_documents=[],
            validation=None,
            error_message=str(e),
        )
        # 失败路径的 span 比成功路径更值钱：它记着「跑到哪一步炸的、
        # 前面几步各花了多久」。这恰恰是排障第一个要问的问题，
        # 所以这里必须和成功路径一样 flush。
        flush_spans(db, trace_id=trace.id)
        # 改成 raise 而不是 return 200 + code:500。
        # 状态码由 app/core/exception_handlers.py 按异常类型裁决
        # （超时→504、上游不可用→502、其余→500），响应体结构不变。
        raise
```

两个要点：

**为什么不能把落库也交给全局处理器。** 全局处理器只有 `request` 和 `exc`，
拿不到 `session_id`、`question`，也没有这个请求的 db session（那是 FastAPI
依赖注入进来的）。所以接口层仍然要做自己的收尾，只把「状态码怎么定」交出去。

**为什么用裸 `raise` 而不是 `raise SomeError(...)`。** 裸 `raise` 保留原始异常
和完整 traceback。`http_status_of` 认识裸 `asyncio.TimeoutError`（→504），
包一层反而可能丢掉这个信息。

### 3.7 SSE：物理上改不了状态码

```python
# app/api/chat_v2.py:261-286
            # SSE 不能像普通接口那样 raise。
            #
            # 原因是物理性的：EventSourceResponse 一开始流式输出，响应头
            # （含 200 状态码）就已经发给客户端了。此刻再抛异常，全局处理器
            # 想写 504/502 也写不进去——状态行早就在网线上了。
            #
            # 所以流式接口的「说真话」只能落在**事件体**里:
            # 除了 data(给人看的消息)，额外带上 error_code / degrade_reason
            # / request_id，让前端能区分「超时」和「上游挂了」，
            # 也让排障的人拿着 request_id 直接去日志里捞全链路。
            #
            # 保留 type/data 两个老字段不动 —— 前端现有的
            # `if (msg.type === "error")` 分支照样能跑，新字段是加法。
            yield {
                "event": "message",
                "data": json.dumps(
                    {
                        "type": "error",
                        "data": str(e),
                        "error_code": error_code_of(e),
                        "degrade_reason": degrade_reason_of(e).value,
                        "request_id": get_request_id_or_none(),
                    },
                    ensure_ascii=False,
                ),
            }
```

这是一个**承认限制**的地方，值得单独说。

HTTP 响应是有顺序的：状态行 → 响应头 → body。SSE 一开始推第一个事件，
状态行就已经在网线上了。之后无论发生什么，那个 200 都改不了 ——
这不是设计取舍，是协议的物理性质。

所以流式接口的错误率**天然无法通过状态码监控**。补偿手段有两个：

1. 事件体里带全套结构化字段（`error_code` / `degrade_reason` / `request_id`），
   前端和日志侧仍然能区分故障类型。
2. `degrade_total` 指标（第 22 篇）—— 它不依赖 HTTP 状态码，
   所以流式路径的降级同样能被统计到。

这也解释了为什么**指标不能只靠 `http_requests_total{status="5xx"}`**：
SSE 那条路径永远不会出现在 5xx 里。

---

## 4. 背后的工程原理

### 4.1 HTTP 状态码是一份跨系统契约，不是装饰

这是最容易被忽略的一点。很多人把状态码当成「给前端的一个提示」，
反正前端读 body 就行。

但状态码是一份**契约**，签约方远不止前端：

```
你的应用
   ↓ HTTP 200/5xx
Nginx           → 记 $status，触发 5xx 告警
   ↓
负载均衡        → 判定实例健康，决定是否摘除
   ↓
APM Agent       → 算错误率、Apdex、异常事务
   ↓
Prometheus      → http_requests_total{status="5xx"}
   ↓
告警规则        → rate(5xx[5m]) > 0.05 → 呼叫值班
```

这条链上每一环都是**通用软件**，它们不知道你的 body 长什么样，
也不该知道。让它们工作的唯一方式是遵守契约。

**推论：任何「用 body 表达失败、状态码恒为 200」的设计，
等价于主动放弃整条基础设施监控链。**

面试里如果被问到「为什么不能用 200 + body 里的 code」，
答案不是「不符合 RESTful 规范」这种空话，而是上面这张图 ——
**具体列出哪些组件会因此失效，以及失效的后果是什么**。

### 4.2 5xx 和 4xx 的分界：责任在谁

选状态码时最实用的判据不是「哪个语义最贴切」，而是：

> **这个失败是谁的责任？谁该采取行动？**

| 类别 | 责任方 | 该谁行动 | 是否计入错误率 |
|---|---|---|---|
| 4xx | 客户端 | 调用方改请求 | **不该** |
| 5xx | 服务端 | 我们修 / 找上游 | **该** |

所以第 14 篇里 `InvalidRequestError` 是 400，`CircuitOpenError` 是 503：
熔断是我们自己的保护动作，责任在服务端一侧（下游挂了，我们主动拒绝），
不能因为「是客户端触发的这次请求」就算 4xx。

这条判据也解释了 §3.2 里的日志级别选择：4xx 用 warning，
因为它不代表我们的系统有问题，不该触发我们的告警。

### 4.3 502 / 503 / 504 的区分为什么值钱

都是 5xx，为什么不统一用 500？因为**处置动作不同**：

| 码 | 含义 | 值班动作 |
|---|---|---|
| 500 | 我们自己的代码炸了 | 看 traceback，发版修 bug |
| 502 | 上游返回了非法响应 | 找上游团队 / 看上游状态页 |
| 503 | 我们主动拒绝（熔断、过载） | 看下游依赖，等冷却或扩容 |
| 504 | 上游超时 | 看上游延迟，考虑调超时或降级 |

值班同学收到告警后第一个问题永远是「是我们的问题还是别人的问题」。
状态码就能回答的话，能省掉十几分钟的翻日志。

而如果全都是 500，这个信息就只存在于 body 和日志里 —— 而告警系统
（只看状态码）没法在通知里带上它。

### 4.4 横切关注点为什么要集中

「异常 → HTTP 状态码」是典型的**横切关注点**（cross-cutting concern）：
跟具体业务无关，但每个接口都需要。

散在各处的三个后果，`exception_handlers.py` 的模块 docstring 里列了：

```
1. 重复（违反 DRY），且迟早不一致 —— 有人写 500，有人写 502。
2. 新接口容易忘，忘了就又变成 200 撒谎。
3. 状态码映射规则改一次要动 N 个文件。
```

第 2 点最要命，因为它是**静默失败**。新同学写了个接口，
`except: return {"code": 500}`，代码评审时看着挺像样（跟旁边的接口一致），
测试也过（前端能显示错误）。这个接口就带着一个隐形的监控盲区上线了。

集中之后，业务代码只需要 `raise`。**做对事情的成本比做错更低** ——
这是横切关注点集中化的真正价值：不是省了几行代码，
而是让「忘记处理」这件事在物理上不可能发生。

对照第 14 篇：那边收敛的是「异常怎么分类」，这边收敛的是「分类结果怎么变成 HTTP 响应」。
两层各管一段，`http_status_of` 是它们的接缝。

### 4.5 向后兼容的三条纪律

改 API 最怕连带把前端搞崩。这次的兼容策略值得单独记：

**第一条：只做加法。**

```python
"errorMessage": str(exc),      # 老字段，保留
"error_code": ...,             # 新增
"degrade_reason": ...,         # 新增
"request_id": ...,             # 新增
```

老前端读不到新字段，但它不 care —— 它只读它认识的那几个。
反过来如果**改名**或**删字段**，老前端立刻炸。

**第二条：变状态码是安全的，前提是要确认过。**

这条得实际验证，不能想当然：

- `fetch()` 不会因为 5xx 抛异常，`res.json()` 照样能解析 body ✅
- `axios` **会**在 5xx 时走 `catch` 分支 ⚠️ ——
  但这个项目前端读的是 `code` 字段，`catch` 里同样拿得到 `error.response.data`
- `if (res.code !== 200)` 这个判断不受影响 ✅（`code` 现在是 504，仍然 ≠ 200）

**第三条：`code` 跟随状态码而不是恒为 500。**

这一点很微妙。`code: 500` → `code: 504` 对前端的 `!== 200` 判断毫无影响，
但对**读日志的人**有影响：他从 body 就能看出是超时，不用再去翻别处。

零成本的信息增量，就该拿。

### 4.6 承认限制比假装解决更诚实

SSE 那一段是本篇的一个反面注脚：**有些地方就是做不到**。

流式响应的状态码物理上无法在开流后修改。这时候有三种态度：

1. 假装没这个问题（不写注释，让后人以为已经处理了）—— 最糟
2. 硬凑一个方案（比如先缓冲整个响应再决定状态码）—— 那就不是流式了
3. **如实记录限制 + 提供补偿手段** —— 我们选这个

`chat_v2.py:261-270` 那段注释写清了「为什么改不了」和「所以我们改成在事件体里说」。
下一个读到这里的人不会浪费半小时试图给 SSE 加状态码。

这跟这四批的主题是一致的：**不说假话，包括不假装解决了做不到的事。**

---

## 5. 怎么验证

### 5.1 跑相关测试

```bash
wsl.exe -d Ubuntu -- bash -lc 'cd /home/dong/projects/super_biz_agent_py && .venv/bin/python -m pytest tests/integration/test_chat_api.py tests/integration/test_chat_v2_api.py -q --no-cov' 2>&1 | tr -d '\r'
```

### 5.2 关键用例

`tests/integration/test_chat_v2_api.py`：

- `test_chat_v2_error_returns_real_http_500` —— 断言 `status_code == 500`
  （改之前这里断言的是 `status_code == 200` + body 里 `code == 500`），
  并检查 `error_code == "runtime_error"`、`degrade_reason == "llm_error"`、
  以及 `response.headers["X-Request-ID"] == data["data"]["request_id"]`
- `test_chat_v2_timeout_returns_504` —— 超时映射到 504，`degrade_reason == "llm_timeout"`。
  **这是「状态码要区分故障类型」的护栏**：改回恒为 500 它就红
- `test_get_trace_returns_404_when_missing` —— `HTTPException` 路径没被我们接管，仍然是 404

`tests/integration/test_chat_api.py`：

- `test_chat_error_returns_real_http_500`
- `test_chat_missing_required_fields_returns_422` —— pydantic 校验失败仍是 422，我们没干扰它

### 5.3 那个特殊的 fixture

验证错误响应需要 `client_no_raise`，原因写在 `tests/conftest.py`：

```python
@pytest.fixture
def client_no_raise():
    """和 `client` 相同，但不把服务端异常抛回测试 —— 用于验证错误响应本身。

    Starlette 的 `ServerErrorMiddleware` 在调用完 `Exception` 处理器之后
    **总是 `raise exc`**（源码注释写的理由是「让服务器能记日志、让测试客户端
    可以选择在用例内抛出」）。而 `TestClient` 默认 `raise_server_exceptions=True`，
    于是 `client.post(...)` 会直接抛 RuntimeError，根本拿不到 response 对象 ——
    想断言状态码是 500 还是 504 就无从下手。
    """
    return TC(app, raise_server_exceptions=False)
```

保留默认 `client` 仍然抛异常是有意的：其他用例里冒出的意外异常应该响亮地失败，
而不是被悄悄吞成一个 500 响应。

### 5.4 手动看真实响应

起服务，打一个必然失败的请求（比如把 `DASHSCOPE_API_KEY` 设成无效值）：

```bash
curl -i -X POST http://localhost:8000/api/chat_v2 \
  -H 'Content-Type: application/json' \
  -d '{"id":"t1","question":"测试"}'
```

要看到的是：

```http
HTTP/1.1 500 Internal Server Error      ← 状态行说真话
X-Request-ID: a3f1c8d29b4e5f60          ← 出错路径也带 id

{"code": 500, "message": "error", "data": {
  "success": false,
  "answer": null,
  "errorMessage": "...",
  "error_code": "llm_error",             ← 可聚合
  "degrade_reason": "llm_error",         ← 指明修复方向
  "request_id": "a3f1c8d29b4e5f60"       ← 与响应头一致
}}
```

### 5.5 确认没有残留的 200 撒谎

```bash
wsl.exe -d Ubuntu -- bash -lc 'cd /home/dong/projects/super_biz_agent_py && grep -rn "\"code\": 500" app/ || echo "干净"' 2>&1 | tr -d '\r'
```

---

## 6. 文件清单

| 文件 | 改动 |
|---|---|
| `app/core/exception_handlers.py` | 新增。两个处理器 + 响应体构造 + request_id 双通道取值 |
| `app/main.py` | 调用 `register_exception_handlers(app)` |
| `app/api/chat.py` | `return {"code":500}` → `raise`；两处 `HTTPException(500)` → `raise` |
| `app/api/chat_v2.py` | 同上；SSE 路径改为在事件体里带结构化错误字段 |
| `tests/conftest.py` | 新增 `client_no_raise` fixture |
| `tests/integration/test_chat_api.py` | 状态码断言由 200 改为真实码 |
| `tests/integration/test_chat_v2_api.py` | 同上，新增 504 用例 |

---

## 7. 常见坑

### 7.1 `return` 一个「错误对象」而不是 `raise`

这是本篇的核心坑，值得再强调一次判别方法：

```python
# ❌ FastAPI 看到正常返回 → HTTP 200
return {"code": 500, "message": "error"}

# ✅ 异常冒泡到处理器 → 真实状态码
raise
```

**自查方法**：在项目里 grep 一遍。

```bash
grep -rn '"code": 5' app/
grep -rn 'return.*JSONResponse' app/     # 看有没有 status_code=200 配错误 body
```

任何在 `except` 块里出现的 `return`，都要问一句：这条路径的 HTTP 状态码是什么？

### 7.2 以为 `TestClient` 能直接测到错误响应

```python
# ❌ 这会抛 RuntimeError，拿不到 response
def test_error(client):
    response = client.post("/api/chat_v2", json={...})
    assert response.status_code == 500      # 到不了这一行
```

`ServerErrorMiddleware` 在调完处理器后**总是重新抛出**异常，
而 `TestClient` 默认 `raise_server_exceptions=True` 会把它传给测试。

必须用 `TestClient(app, raise_server_exceptions=False)`。

**不要因此把默认 fixture 也改成 False** —— 那样其他用例里的意外异常
会被悄悄吞成 500 响应，测试变成绿的但功能是坏的。

### 7.3 只读 ContextVar 取 request_id

见 §3.3。症状很有欺骗性：

- `AppError` 路径：request_id 正常 ✅
- 未分类异常路径：request_id 恒为 `null` ❌

而测试里模拟异常通常用自定义 `AppError` 子类，于是**测试全绿，
线上 500 响应里的 request_id 永远是 null**。

写测试时要专门用一个**裸 `RuntimeError`** 来覆盖兜底处理器那条路径。

### 7.4 顺手把 `HTTPException` 也统一了

诱惑很大：「都是错误响应，形状统一多好」。

但 `HTTPException` 从来没说谎，它产出的状态码是对的。统一它的收益只有
「形状一致」这个洁癖，成本是改前端 + 回归测试 + 夹带风险。

**判据**：这次改动要解决的问题是「状态码说谎」。
`HTTPException` 不在问题范围内，就不动它（YAGNI）。

### 7.5 忘了失败路径也要落 trace / flush span

```python
except Exception as e:
    raise          # ❌ 只是抛出去，什么都没记
```

失败路径的追踪数据**比成功路径更值钱** —— 它记着「跑到哪一步炸的、
前面几步各花了多久」。这是排障第一个要问的问题。

顺序也不能反：**先落库，再 raise**。异常一旦离开这个函数，
`session_id` / `question` / db session 就都拿不到了。

### 7.6 指望状态码覆盖所有失败

SSE 路径的状态码物理上无法反映失败（§3.7）。所以：

- 不要写「5xx 率 = 0 就说明系统健康」这样的告警规则
- 流式接口的健康度要靠 `degrade_total`（第 22 篇）来看

任何「响应头已发出」的场景都有这个限制：SSE、WebSocket、
chunked 传输中途失败、文件下载中途中断。

---

## 8. 与其他篇的关系

```
第 14 篇  错误分类
          ↓ 提供 http_status_of / error_code_of / degrade_reason_of
第 18 篇  本篇：把分类结果翻译成 HTTP 响应
          ↓ 产出真实的 4xx/5xx 状态码
第 22 篇  Prometheus 指标
          ← http_requests_total{status="5xx"} 消费这些状态码

第 15 篇  request_id
          ↓ 提供 scope + ContextVar 双通道
第 18 篇  本篇：在错误响应体和响应头里带上它
```

有个细节值得记住：`exception_handlers.py` 的模块 docstring 里
把 `http_requests_total{status="5xx"}` 列为状态码的消费者之一 ——
**但写这篇的时候那个消费者并不存在**。状态码说了真话，
却没有任何东西在听。直到第 22 篇接上 Prometheus，这条链才真正闭合。
