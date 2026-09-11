"""MCP 客户端管理 —— 链路一(Agent Tool Calling)的外部工具入口。

## 拦截器的分层与顺序

`_build_interceptor_chain` 用 `for interceptor in reversed(tool_interceptors)`
构建洋葱,所以**列表里第一个拦截器是最外层**(这一点在
langchain_mcp_adapters.interceptors 的 Protocol 文档里写作
"first is outermost")。本模块的层次是:

    circuit_breaker_interceptor   ← 最外层:熔断 + 指标 + fail-open
        retry_interceptor         ← 内层:瞬时故障重试
            实际的 MCP 调用

**熔断为什么必须在重试外面**:反过来的话,熔断打开的判断会被塞进
重试循环里跑三遍 —— 每次都立刻抛 CircuitOpenError,却仍然按指数退避
睡掉 1 + 2 秒。熔断器存在的意义就是「不要重复确认一个已知的坏消息」,
把它放到重试内层等于亲手取消这个意义。

## 熔断器该看见什么、不该看见什么

熔断器守护的是**MCP 进程的可用性**,不是工具的执行成功率。这条界线
决定了两种失败的处置完全不同:

- handler **抛异常**(连接被拒、握手失败、请求超时)→ 进程有问题,
  记熔断失败。
- handler 返回 `isError=True` → 进程活得很好,是工具自己执行报错
  (参数不对、查询的资源不存在)。这种只记 `tool_failures_total`,
  **不碰熔断器**。一个参数老是填错的工具把整个 MCP 服务熔断掉,
  会让同一进程上其他健康的工具一起不可用 —— 纯误伤。

## 为什么所有失败都 fail-open 成 isError=False

`_convert_call_tool_result` 遇到 `isError=True` 会 `raise ToolException`,
而 LangGraph 的 `_default_handle_tool_errors` 只放行 `ToolInvocationError`,
其余一律 `raise e` —— 于是一次工具失败会**中断整轮对话**,用户拿到 500。

但工具失败本来是可降级的:模型还能换个工具、还能如实告知用户
「日志服务暂时查不了」。所以这里统一返回 `isError=False` + 一段说明
真实原因的文本,让模型继续往下走。这跟 knowledge_tool 的处置、
以及 rag_v2 里质检器挂掉时的 fail-open 是同一个取向。

注意这不是「掩盖错误」:文本里带着真实错误码,指标和日志都如实记账。
被隐藏的只有「异常」这个**控制流形态**,而不是失败这个事实。
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from loguru import logger
from mcp.types import CallToolResult, TextContent

from app.config import config
from app.core.breakers import mcp_breaker
from app.core.errors import (
    MCPUnavailableError,
    ToolExecutionError,
    error_code_of,
    is_retryable,
)
from app.core.metrics import count_tool_failure, observe_tool_call

# 全局 MCP 客户端（延迟初始化）
_mcp_client: Optional[MultiServerMCPClient] = None

# handler 的类型别名 —— 拦截器签名里出现两次,写全了很占地方。
_Handler = Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]]

# 重试参数。为什么是模块常量而不是进 config:
# 这两个值跟「部署环境」无关,是拦截器的实现细节。config 里的每一项
# 都该是运维可能想调的东西,塞进去反而让配置文件变噪音(YAGNI)。
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0

# 给模型的降级说明。为什么要明确写「请如实告知」:
# 模型看到工具返回错误,默认行为常常是自己编一个答案或含糊过去。
# 把处置方式写进返回值,是在这个「只能通过一个字符串跟模型通信」
# 的接口上唯一能施加约束的地方(同 knowledge_tool 的 _UNAVAILABLE_HINT)。
_UNAVAILABLE_HINT = (
    "工具 {tool} 当前不可用（服务: {server}，错误码: {code}）。"
    "请如实告知用户该功能暂时无法使用，不要凭猜测编造结果。"
)

_TOOL_ERROR_HINT = (
    "工具 {tool} 执行失败（服务: {server}）：{detail}\n"
    "请根据错误信息判断是否换个参数重试，或如实告知用户。"
)


def _metric_label(request: MCPToolCallRequest) -> str:
    """指标里用的工具名。

    带上 server_name 前缀,是因为两个 MCP 服务可能暴露同名工具
    (比如各有一个 `query`),合到一个 label 上就分不清是谁在慢、谁在错。
    基数仍然有界:服务名来自配置,工具名来自服务器的工具表,
    两者都是代码/配置里写死的,不随请求变化。

    **绝对不能**把 request.args 放进 label —— 那是用户问题的一部分,无界。
    """
    return f"{request.server_name}.{request.name}"


async def retry_interceptor(
    request: MCPToolCallRequest,
    handler: _Handler,
    max_retries: int = _MAX_RETRIES,
    delay: float = _RETRY_BASE_DELAY,
) -> MCPToolCallResult:
    """MCP 工具调用重试拦截器 —— 只重试**值得重试**的失败。

    改造前这里对所有异常一律重试 3 次。后果是鉴权失败、参数非法这类
    确定性错误也要白等 1 + 2 = 3 秒,而结果注定一样:重试只是把失败
    延后暴露,还烧掉三倍资源。判据走 `is_retryable`,分类逻辑在
    app/core/errors.py 只有一处实现(DRY)。

    全部重试失败后**抛** MCPUnavailableError,而不是返回
    `CallToolResult(isError=True)`。两个理由:

    1. 外层的熔断器只能看见异常。返回一个值的话 `guard()` 会记成功,
       熔断器永远打不开 —— 重试挡住了它本该看见的所有失败。
    2. `isError=True` 并不像它看起来那样温和:adapter 会把它转成
       `raise ToolException`,反而中断整轮对话。真正的 fail-open
       在外层熔断拦截器里做,那里返回的是 `isError=False`。
    """
    last_error: BaseException | None = None

    for attempt in range(max_retries):
        try:
            logger.info(
                f"调用 MCP 工具: {request.name} "
                f"(服务器: {request.server_name}, 第 {attempt + 1}/{max_retries} 次尝试)"
            )
            result = await handler(request)
            logger.info(f"MCP 工具 {request.name} 调用成功")
            return result

        except asyncio.CancelledError:
            # 取消不重试、不改写,原样往上抛。
            # 来源通常是总预算耗尽 —— 此时重试是在花已经花光的钱,
            # 而且会让取消传播被推迟好几秒。
            raise

        except Exception as exc:
            last_error = exc

            if not is_retryable(exc):
                # 确定性失败:立刻抛,不睡也不重试。
                logger.warning(
                    f"MCP 工具 {request.name} 调用失败且不可重试"
                    f"[{error_code_of(exc)}]: {exc}"
                )
                raise

            logger.warning(
                f"MCP 工具 {request.name} 调用失败 "
                f"(第 {attempt + 1}/{max_retries} 次): {exc}"
            )

            if attempt < max_retries - 1:
                wait_time = delay * (2**attempt)  # 指数退避
                logger.info(f"等待 {wait_time:.1f} 秒后重试...")
                await asyncio.sleep(wait_time)

    raise MCPUnavailableError(
        f"工具 {request.name} 在 {max_retries} 次重试后仍然失败: {last_error}",
        cause=last_error,
        context={"tool": request.name, "server": request.server_name},
    )


async def circuit_breaker_interceptor(
    request: MCPToolCallRequest,
    handler: _Handler,
) -> MCPToolCallResult:
    """MCP 工具调用熔断拦截器 —— 每个服务一个熔断器,失败一律 fail-open。

    放在拦截器列表**第一个**(最外层),理由见模块文档。

    计时器在 `guard()` 内侧:熔断打开时 guard 直接抛,压根进不到
    `observe_tool_call`。反过来的话,熔断期成批的 0 秒样本会把
    `tool_call_duration_seconds` 的 P99 洗得非常好看 —— 而那恰好是
    故障最严重的时候(同 metrics.py 关于计时点位置的纪律)。
    """
    breaker = mcp_breaker(request.server_name)
    label = _metric_label(request)

    try:
        with breaker.guard():
            with observe_tool_call(label):
                result = await handler(request)
    except asyncio.CancelledError:
        # 原样抛。guard() 已经处理了「取消不计失败、但归还探针名额」,
        # 这里不需要也不应该把它转成友好返回值 —— 取消意味着调用方
        # 已经不要这个结果了,伪造一个成功返回反而会让取消传播断掉。
        raise
    except Exception as exc:
        # 到这里的异常有两种来源,处置相同、账目不同:
        # - CircuitOpenError:熔断打开,调用没发出去。observe_tool_call
        #   已经排除了它的失败计数,这里也不该重复记。
        # - 重试耗尽抛出的 MCPUnavailableError:真实失败,
        #   observe_tool_call 已经记过 tool_failures_total。
        # 两者都由 guard() 负责熔断计数,所以这里只做 fail-open 转换。
        code = error_code_of(exc)
        logger.error(
            f"MCP 工具不可用[{code}]: tool={request.name}, "
            f"server={request.server_name}, err={exc}"
        )
        return _text_result(
            _UNAVAILABLE_HINT.format(
                tool=request.name, server=request.server_name, code=code
            )
        )

    # 服务器活着,但工具自己报错了。
    #
    # 不记熔断失败:进程是健康的(它成功地返回了一个响应),把这种失败
    # 计进去会让一个参数老填错的工具把整个 MCP 服务熔断掉,同进程上
    # 其他健康的工具一起遭殃。但要记 tool_failures_total ——
    # observe_tool_call 看不见它(那是个返回值,不是异常),所以在这里补。
    if isinstance(result, CallToolResult) and result.isError:
        detail = _error_text(result)
        count_tool_failure(label, ToolExecutionError.code)
        logger.warning(
            f"MCP 工具执行报错: tool={request.name}, "
            f"server={request.server_name}, detail={detail}"
        )
        return _text_result(
            _TOOL_ERROR_HINT.format(
                tool=request.name, server=request.server_name, detail=detail
            )
        )

    return result


def _text_result(text: str) -> CallToolResult:
    """包一个纯文本的成功结果。

    `isError=False` 是刻意的 —— 理由见模块文档「为什么所有失败都
    fail-open 成 isError=False」。
    """
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=False)


def _error_text(result: CallToolResult) -> str:
    """从 isError 的结果里抽出人和模型都能读的错误文本。

    只取 TextContent。图片/音频这类 content block 对「这次为什么失败」
    没有信息量,拼进去只是噪音。
    """
    parts = [
        block.text
        for block in result.content
        if isinstance(block, TextContent) and block.text
    ]
    return "\n".join(parts) if parts else "（工具未提供错误详情）"


# 使用配置文件中定义的完整 MCP 服务器配置
DEFAULT_MCP_SERVERS = config.mcp_servers


async def get_mcp_client(
    servers: Optional[Dict[str, Dict[str, str]]] = None,
    tool_interceptors: Optional[List] = None,
    force_new: bool = False,
) -> MultiServerMCPClient:
    """
    获取或初始化 MCP 客户端（不带任何拦截器）

    这是一个单例模式，确保整个应用只有一个 MCP 客户端实例（除非 force_new=True）

    从 langchain-mcp-adapters 0.1.0 开始，MultiServerMCPClient 不再支持作为上下文管理器使用。
    直接创建实例即可使用。

    Args:
        servers: MCP 服务器配置，默认使用 DEFAULT_MCP_SERVERS
        tool_interceptors: 自定义工具拦截器列表
        force_new: 是否强制创建新实例（用于特殊场景，如需要不同配置）

    Returns:
        MultiServerMCPClient: MCP 客户端实例
    """
    global _mcp_client

    # 如果请求新实例，直接创建并返回（不缓存）
    if force_new:
        logger.info("创建新的 MCP 客户端实例（非单例）")
        return _create_mcp_client(servers or DEFAULT_MCP_SERVERS, tool_interceptors)

    # 单例模式：如果已存在，直接返回
    if _mcp_client is None:
        logger.info("初始化全局 MCP 客户端...")
        _mcp_client = _create_mcp_client(
            servers or DEFAULT_MCP_SERVERS, tool_interceptors
        )
        logger.info("全局 MCP 客户端初始化完成")

    return _mcp_client


async def get_mcp_client_with_retry(
    servers: Optional[Dict[str, Dict[str, str]]] = None,
    tool_interceptors: Optional[List] = None,
    force_new: bool = False,
) -> MultiServerMCPClient:
    """
    获取或初始化带**熔断 + 重试**的 MCP 客户端

    名字保留 `_with_retry` 是为了不破坏现有调用点;实际装的是
    「熔断在外、重试在内」这一整套防护(顺序的理由见模块文档)。

    Args:
        servers: MCP 服务器配置，默认使用 DEFAULT_MCP_SERVERS
        tool_interceptors: 自定义工具拦截器列表（会在内置拦截器之后添加，
            也就是位于更内层）
        force_new: 是否强制创建新实例（用于特殊场景，如需要不同配置）

    Returns:
        MultiServerMCPClient: 带熔断与重试的 MCP 客户端实例
    """
    interceptors: List = [circuit_breaker_interceptor, retry_interceptor]
    if tool_interceptors:
        interceptors.extend(tool_interceptors)

    return await get_mcp_client(
        servers=servers,
        tool_interceptors=interceptors,
        force_new=force_new,
    )


def _create_mcp_client(
    servers: Dict[str, Dict[str, str]],
    tool_interceptors: Optional[List] = None,
) -> MultiServerMCPClient:
    """
    创建 MCP 客户端实例

    Args:
        servers: MCP 服务器配置
        tool_interceptors: 工具拦截器列表

    Returns:
        MultiServerMCPClient: 未初始化的客户端实例
    """
    # MultiServerMCPClient 的第一个参数直接接收 servers 配置字典
    # 格式: {server_name: {"transport": "...", "url": "..."}}
    kwargs: Dict[str, Any] = {}

    if tool_interceptors:
        kwargs["tool_interceptors"] = tool_interceptors

    return MultiServerMCPClient(servers, **kwargs)  # type: ignore[arg-type]
