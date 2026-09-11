"""HTTP 中间件：request_id 的入口与出口。

这是 request_id 全链路的第一站：
    **中间件** → ContextVar → 每条日志 → 落库 trace → RQ meta → worker 日志

为什么写成纯 ASGI 中间件，而不是 `@app.middleware("http")`：
`@app.middleware("http")` 底层是 Starlette 的 `BaseHTTPMiddleware`，它会把响应体
包一层 anyio 内存流再转发。对普通 JSON 响应没影响，但项目里有 SSE 端点
（`app/api/chat_v2.py` 的 `EventSourceResponse`），额外的缓冲层会影响
逐块下发的及时性和断连感知。纯 ASGI 中间件只是在 send 回调上加了一行 header，
对流式响应零干扰，开销也更低。
"""

from __future__ import annotations

import re

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.request_context import (
    REQUEST_ID_HEADER,
    new_request_id,
    reset_request_id,
    set_request_id,
)

# 允许的 request_id 形态：字母数字加少量分隔符，最长 64。
# 为什么必须校验外部传入值（这是安全边界，不是洁癖）：
# 1. **日志伪造**：这个值会出现在每一行日志里。如果放任 `\n` 通过，
#    攻击者传一个带换行的 header 就能凭空造出一整行假日志。
# 2. **存储与可读性**：它还要落到数据库列里，超长值会撑爆字段、污染日志宽度。
# 校验不通过时不报错，直接当作「没传」生成一个新的 —— 追踪 id 不该成为请求失败的理由。
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def _sanitize_incoming(value: str | None) -> str | None:
    """校验上游传入的 request_id，非法则返回 None（调用方会生成新的）。"""
    if not value:
        return None
    candidate = value.strip()
    if not _SAFE_REQUEST_ID.match(candidate):
        return None
    return candidate


class RequestIdMiddleware:
    """为每个 HTTP 请求绑定 request_id，并回写到响应头。

    三件事：
    1. 复用上游传入的 `X-Request-ID`（网关/前端已有 id 时跨系统串联），否则新生成。
    2. 写入 ContextVar —— 此后该请求内的所有日志自动带上它，业务代码零改动。
    3. 回写响应头 —— 用户报障时截图里就有 id，可以直接拿去查日志，
       不必再靠「大概几点几分」这种模糊条件捞。
    """

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

        # 除了 ContextVar，再把 id 写进 ASGI scope 的 state 里。
        #
        # 这不是冗余，是为了覆盖**异常路径**。Starlette 的中间件栈是：
        #     ServerErrorMiddleware      ← Exception 处理器在这里(最外层)
        #       └─ RequestIdMiddleware   ← 下面那个 finally 在这里
        #           └─ ExceptionMiddleware  ← AppError 处理器在这里(最内层)
        #               └─ router
        #
        # 未分类异常冒泡到最外层时，本中间件的 `finally: reset_request_id`
        # **已经执行完了**，处理器里再读 ContextVar 只能拿到 NO_REQUEST_ID。
        # 结果就是错误响应体里 request_id 恒为 null —— 而出错恰恰是最需要它的时候。
        #
        # scope 是一个贯穿整条调用链的同一个 dict，异常穿过中间件不会改变它，
        # 所以它能安全地把 id 递到最外层。用 scope["state"] 而不是自造顶层键，
        # 是因为 Starlette 已经把这个位置约定为「应用自定义数据」，
        # 读侧可以直接 `request.state.request_id`。
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
            # 必须 reset：ASGI 服务器的 task 上下文可能被复用，
            # 不还原会让下一个请求继承上一个的 id —— 那比没有 id 更糟，
            # 因为它会把两个请求的日志错误地关联在一起。
            reset_request_id(token)
