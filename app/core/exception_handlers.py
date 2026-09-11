"""全局异常处理器 —— 让 HTTP 状态码说真话。

## 为什么需要这一层

改之前，`app/api/chat_v2.py` 的失败路径是这样的：

    except Exception as e:
        return {"code": 500, "message": "error", "data": {...}}

FastAPI 看到一个正常 return，于是发出 **HTTP 200 OK**，body 里写着 `code: 500`。

问题在于，body 里的 `code` 只有前端 JS 会读。而下面这些东西**只看 HTTP 状态码**：

- Nginx / 网关的 `$status` 日志与 5xx 告警
- 负载均衡的健康判定与实例摘除
- APM（SkyWalking / OpenTelemetry / Sentry）的错误率、Apdex
- Prometheus 的 `http_requests_total{status="5xx"}`
- 云厂商 SLB 的监控面板

它们全都看到 200，于是**错误率永远是 0%**。模型全挂、Milvus 宕机、每个请求都在返回
「抱歉，处理失败」，监控大盘依然一片绿。对一个 AIOps 项目来说，这个讽刺有点大 ——
我们做的是帮别人发现故障的系统，自己的故障却对所有标准监控隐身。

## 为什么用全局处理器，而不是在每个接口里改 return

因为「异常 → HTTP 状态码」是一条**横切关注点**：它跟业务无关，但每个接口都需要。
散在各个 `except` 里写 `JSONResponse(status_code=...)` 会有三个后果：

1. 重复（违反 DRY），且迟早不一致 —— 有人写 500，有人写 502。
2. 新接口容易忘，忘了就又变成 200 撒谎。
3. 状态码映射规则改一次要动 N 个文件。

集中在这里之后，业务代码只需要 `raise`，语义由 `app/core/errors.py` 的
`http_status_of` 统一裁决 —— 分类逻辑仍然只有一处实现。

## 向后兼容约定

响应体的 JSON **结构保持不变**（前端依赖 `code` / `data.success` /
`data.errorMessage` 这三个字段）。这次变的只有两样：

- HTTP 状态码：200 → 真实的 4xx/5xx
- 新增字段：`data.error_code`、`data.degrade_reason`、`data.request_id`

新增字段是**加法**，老前端读不到它们也不会坏；HTTP 状态码变化对前端也是安全的 ——
`fetch` 不会因为 5xx 抛异常，而检查 `code !== 200` 的逻辑依然成立。
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.errors import (
    AppError,
    degrade_reason_of,
    error_code_of,
    http_status_of,
)
from app.core.request_context import REQUEST_ID_HEADER, get_request_id_or_none


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
        "code": http_status,
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


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底处理未被分类的异常。

    存在的意义：没有这个处理器，任何漏网异常都会走 Starlette 默认路径，
    返回一段 `Internal Server Error` 纯文本 —— 前端拿到的不是 JSON，
    解析直接炸，用户看到的是白屏而不是错误提示。

    这里仍然经过 `http_status_of`，所以裸 `asyncio.TimeoutError`
    也能正确变成 504 而不是笼统的 500。
    """
    http_status = (
        http_status_of(exc))
    # 未分类异常需要完整堆栈：它代表我们没预料到的情况，是要修的 bug。
    logger.opt(exception=exc).error(
        f"[{request.method} {request.url.path}] 未处理异常 → HTTP {http_status}: {exc}"
    )
    return _json_error_response(exc, http_status, request=request)


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
