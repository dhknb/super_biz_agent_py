"""本项目实际用到的熔断器实例。

机制在 app/core/circuit_breaker.py,这个模块只放**策略**:
开几个熔断器、各自守护哪个下游、阈值取多少。

**为什么要单独一个模块**

circuit_breaker.py 是一个不认识本项目任何业务的通用机制。
它不该 import config,也不该知道「我们用的是 Milvus 和 DashScope」。
实例放这里,机制与策略就分开了(依赖倒置):
调阈值、加一个熔断器只动这一个文件,机制文件保持稳定。

**为什么按「下游依赖」划分,而不是按「调用点」划分**

熔断器的单位是**一个下游依赖**,不是一个函数。

- 只开一个:Milvus 挂了会把 LLM 调用一起切断 ——
  两个健康状况完全独立的依赖被绑在一起,纯误伤。
- 每个调用点开一个:rag 检索和 SOP 检索打的是同一个 Milvus,
  拆开之后各数各自的失败次数,同一个故障要失败 2N 次才熔断,
  灵敏度直接减半;而 SOP 检索 QPS 很低,可能永远攒不够 N 次
  —— 熔断器成了装饰品。

所以按下游分:检索类(Milvus + BM25)一个,LLM(模型服务)一个,
每个 MCP 服务各一个。
判断标准很简单:**会同时坏、也会同时好的调用,共用一个熔断器。**
"""

from __future__ import annotations

import threading

from app.config import config
from app.core.circuit_breaker import CircuitBreaker

# 守护向量检索链路:Milvus 连接、BM25 语料加载、混合检索。
# rag_v2 的检索节点与 AIOps 的 SOP 检索共用它 —— 它们打的是同一个库。
retrieval_breaker = CircuitBreaker(
    "retrieval",
    failure_threshold=config.circuit_breaker_failure_threshold,
    cooldown_seconds=config.circuit_breaker_cooldown_seconds,
)

# 守护 LLM 调用:答案生成、证据质检、AIOps 首响分析走的是同一个模型服务。
llm_breaker = CircuitBreaker(
    "llm",
    failure_threshold=config.circuit_breaker_failure_threshold,
    cooldown_seconds=config.circuit_breaker_cooldown_seconds,
)

# ── MCP:每个服务一个 ─────────────────────────────────────────

# 为什么 MCP 不像检索那样合成一个:
# cls 和 monitor 是**两个独立进程、两个端口**(见 config.mcp_servers)。
# 日志服务挂掉跟监控服务的健康状况没有任何关系,合用一个熔断器的话,
# cls 连续超时 5 次就会把 monitor 的查询一起切断 —— 上面那条判据
# (「会同时坏、也会同时好的调用共用一个」)在这里给出的答案是「不共用」。
#
# 为什么惰性建而不是在模块顶部把 config.mcp_servers 铺开:
# get_mcp_client() 的 servers 参数是可覆盖的,测试和将来的多租户
# 都可能传入配置里没有的服务名。惰性创建让「熔断器覆盖到哪些服务」
# 由实际调用决定,而不是由 import 时的一张静态表决定。
_mcp_breakers: dict[str, CircuitBreaker] = {}

# 保护上面这个字典。用 threading.Lock 的理由同 CircuitBreaker 自身:
# 调用点跨协程与线程池两种执行模型。临界区只有一次 dict 读写。
_mcp_breakers_lock = threading.Lock()


def mcp_breaker(server_name: str) -> CircuitBreaker:
    """取(或惰性创建)某个 MCP 服务的熔断器。

    名字统一加 `mcp_` 前缀:熔断器注册表是全局扁平的,
    服务名 `monitor` 直接当熔断器名的话,跟将来可能出现的
    非 MCP 的 `monitor` 依赖撞名,而 CircuitBreaker.__init__
    是「同名后者覆盖前者」—— 那种撞车不会报错,只会让两个
    依赖悄悄共享一份失败计数。
    """
    key = f"mcp_{server_name}"
    with _mcp_breakers_lock:
        breaker = _mcp_breakers.get(key)
        if breaker is None:
            breaker = CircuitBreaker(
                key,
                failure_threshold=config.circuit_breaker_failure_threshold,
                cooldown_seconds=config.circuit_breaker_cooldown_seconds,
            )
            _mcp_breakers[key] = breaker
        return breaker
