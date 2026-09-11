"""MCP 工具提供者 —— 「列举工具」这件事的唯一入口。

## 它解决的三个问题

改造前 `RagAgentService._initialize_agent` 直接 `await client.get_tools()`,
有三处缺陷,而且是叠加的:

**1. 永久降级(最严重)**

    except Exception:
        self.mcp_tools = []
    ...
    self._agent_initialized = True   # ← 失败路径也置 True

`rag_agent_service` 是模块级单例。MCP 在进程启动那一刻没起来,这个进程
就**永远**只剩本地工具 —— 除了重启没有任何恢复路径。这比「没有熔断器」
更糟:熔断器至少 30 秒后会自己放一个探针去看看下游好没好。

**2. 一个服务挂了拖垮全部**

`MultiServerMCPClient.get_tools()` 不传 server_name 时用
`asyncio.gather(*tasks)` 并行加载所有服务,**没有** `return_exceptions=True`。
于是 cls 连不上会让整个 gather 抛异常,健康的 monitor 的工具一起丢掉。
本模块改成逐个服务加载,失败的跳过、成功的留下。

**3. 列举完全没有防护**

拦截器(含熔断器)只包工具**调用**。`get_tools()` 是工具**列举**,
走的是 `load_mcp_tools` —— 压根不经过拦截器链。所以 MCP 进程没起来时,
每个请求都要在列举这一步白等一次完整的连接超时,然后才降级。
本模块给列举加了独立的冷却退避(config.mcp_tool_reload_cooldown_seconds)。

## 为什么是独立模块而不是塞进 mcp_client.py

mcp_client.py 管的是「客户端与拦截器」,是一层薄薄的连接管理。
本模块管的是「工具集合的生命周期」:什么时候该重载、失败了记多久、
哪些服务当前可用。两者的变化原因不同(前者跟着 adapter 的 API 变,
后者跟着我们的降级策略变),分开放才符合单一职责。

而且它要被**两条链路共用** —— 链路一的 agent 和 rag_v2 的
tool_facts 节点。放在任一条链路里都会让另一条产生反向依赖。

## 为什么不做后台定时刷新

那需要一个常驻任务,牵扯到应用生命周期钩子、优雅退出、
多 worker 下每个进程各刷一份。而「请求来了顺手检查一下冷却期过没过」
用惰性检查就够了 —— 代价是故障恢复后的第一个请求要多等一次列举,
换来的是零常驻状态(YAGNI)。这和熔断器 OPEN → HALF_OPEN
用惰性转换而不是定时器,是同一个取舍。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from loguru import logger

from app.agent.mcp_client import get_mcp_client_with_retry
from app.config import config
from app.core.errors import error_code_of
from app.core.metrics import count_mcp_tool_load_failure, observe_mcp_tool_load


class _ServerState:
    """单个 MCP 服务的工具缓存与冷却状态。

    为什么按服务存而不是存一个总的工具列表:cls 好了 monitor 还坏着
    是常态(两个独立进程),混在一起就只能整体重试,健康那个的工具
    会跟着坏的一起被丢掉又重载。
    """

    __slots__ = ("tools", "loaded", "last_failure_at", "last_error_code")

    def __init__(self) -> None:
        self.tools: List[BaseTool] = []
        # 是否**成功**加载过。跟 `tools` 非空不是一回事 ——
        # 一个服务可以合法地暴露零个工具,那也是加载成功。
        self.loaded: bool = False
        # 用 monotonic 而不是 time.time():墙钟会被 NTP 回拨,
        # 回拨之后冷却期可能变成几个小时(同 CircuitBreaker 的理由)。
        self.last_failure_at: float = 0.0
        self.last_error_code: str = ""


class MCPToolProvider:
    """按服务惰性加载并缓存 MCP 工具,失败后按冷却期重试。"""

    def __init__(
        self,
        servers: Optional[Dict[str, Dict[str, Any]]] = None,
        cooldown_seconds: Optional[float] = None,
    ) -> None:
        self._servers = servers if servers is not None else config.mcp_servers
        self._cooldown = (
            cooldown_seconds
            if cooldown_seconds is not None
            else config.mcp_tool_reload_cooldown_seconds
        )
        self._states: Dict[str, _ServerState] = {
            name: _ServerState() for name in self._servers
        }
        # 用 asyncio.Lock 而不是 threading.Lock:临界区里有 await
        # (真正的网络调用)。threading.Lock 在协程里持锁跨 await
        # 会阻塞整个事件循环 —— 这跟 CircuitBreaker 的选择相反,
        # 是因为那里的临界区只有几个整数赋值,不做 IO。
        self._lock = asyncio.Lock()

    # ── 对外 ────────────────────────────────────────────────

    async def get_tools(self) -> List[BaseTool]:
        """取当前可用的全部 MCP 工具。

        这个方法**不抛异常**:MCP 是增强能力,它整体不可用时链路应该
        带着本地工具继续跑。所有失败都落在日志和指标上。

        返回的可能是空列表(所有服务都不可用),调用方据此决定要不要
        在给模型的提示里说明「外部工具当前不可用」。
        """
        async with self._lock:
            pending = [name for name in self._servers if self._should_load(name)]

            if pending:
                # 逐个服务加载,一个失败不影响其他 —— 这正是不直接用
                # client.get_tools() 的理由(它的 gather 没有 return_exceptions)。
                #
                # 为什么这里可以并发:每个服务是独立的连接,互不影响。
                # 用 gather + return_exceptions=True,失败的那个拿到异常对象
                # 而不是让整组失败。
                results = await asyncio.gather(
                    *(self._load_server(name) for name in pending),
                    return_exceptions=True,
                )
                # 理论上 _load_server 自己吞了所有异常,不该有 Exception 漏出来。
                # 但真漏出来的话必须让它可见 —— 静默吞掉一个未预期的异常,
                # 就是下一次「为什么工具突然没了」查不出原因的起点。
                for name, outcome in zip(pending, results):
                    if isinstance(outcome, BaseException):
                        logger.error(
                            f"MCP 工具加载出现未预期异常: server={name}, err={outcome}"
                        )

            tools: List[BaseTool] = []
            for state in self._states.values():
                tools.extend(state.tools)
            return tools

    def available_servers(self) -> List[str]:
        """当前已成功加载工具的服务名,给日志和排障用。"""
        return [name for name, state in self._states.items() if state.loaded]

    def unavailable_servers(self) -> Dict[str, str]:
        """当前不可用的服务 → 最后一次的错误码。

        给调用方拼「哪些外部能力当前不可用」的说明用。返回错误码而不是
        原始异常文本:那些文本可能很长、还可能带内部地址。
        """
        return {
            name: state.last_error_code or "unknown"
            for name, state in self._states.items()
            if not state.loaded
        }

    def invalidate(self, server_name: Optional[str] = None) -> None:
        """丢弃缓存,让下一次 get_tools 重新加载。

        留这个口子是因为工具表**会变**:MCP 服务重新部署后可能多了
        一个工具。缓存永不失效的话,只有重启我们自己的进程才能看到。
        目前没有自动触发点(没人订阅 MCP 的变更通知),所以它是给
        运维接口和测试用的手动开关。
        """
        names = [server_name] if server_name else list(self._states)
        for name in names:
            state = self._states.get(name)
            if state is not None:
                state.tools = []
                state.loaded = False
                state.last_failure_at = 0.0

    # ── 内部 ────────────────────────────────────────────────

    def _should_load(self, name: str) -> bool:
        """这个服务现在该不该去加载一次。

        三种情况:
        - 已经加载成功 → 不用(工具表变了要靠 invalidate)
        - 从没试过 → 要
        - 试过但失败了 → 冷却期满才要
        """
        state = self._states[name]
        if state.loaded:
            return False
        if state.last_failure_at == 0.0:
            return True

        elapsed = time.monotonic() - state.last_failure_at
        if elapsed < self._cooldown:
            # 这条日志刻意用 debug:冷却期内每个请求都会走到这里,
            # info 级别会把日志刷满,而它表达的是「一切按计划进行」。
            logger.debug(
                f"MCP 工具加载仍在冷却期: server={name}, "
                f"剩余={self._cooldown - elapsed:.1f}s"
            )
            return False
        return True

    async def _load_server(self, name: str) -> None:
        """加载单个服务的工具。**不抛异常**,失败只记账并进入冷却。"""
        state = self._states[name]

        try:
            with observe_mcp_tool_load(name):
                client = await get_mcp_client_with_retry(servers=self._servers)
                tools = await client.get_tools(server_name=name)
        except asyncio.CancelledError:
            # 取消不进冷却:什么都没测出来,下一个请求该正常重试。
            # 进冷却的话,一次总预算耗尽会让这个服务白白静默 60 秒。
            raise
        except Exception as exc:
            code = error_code_of(exc)
            state.last_failure_at = time.monotonic()
            state.last_error_code = code
            count_mcp_tool_load_failure(name, code)
            logger.warning(
                f"MCP 工具加载失败[{code}]，该服务本轮不可用: "
                f"server={name}, 冷却={self._cooldown:.0f}s, err={exc}"
            )
            return

        state.tools = list(tools)
        state.loaded = True
        state.last_failure_at = 0.0
        state.last_error_code = ""
        tool_names = [getattr(tool, "name", str(tool)) for tool in state.tools]
        logger.info(
            f"MCP 工具加载成功: server={name}, 共 {len(state.tools)} 个"
            + (f" ({', '.join(tool_names)})" if tool_names else "")
        )


# 全局单例 —— 两条链路共用同一份工具缓存。
#
# 为什么共用:两条链路打的是同一个 MCP 进程。各自持一份缓存的话,
# 同一个服务的连接超时要各等一次,冷却期也各算一份 ——
# 跟熔断器「按下游依赖划分」是同一条判据。
mcp_tool_provider = MCPToolProvider()
