"""统一错误分类与降级原因体系。

为什么需要这一层：
项目里到处是 `except Exception as e` + `str(e)`，error_message 是自由文本。
后果有三个：
1. 无法聚合统计 —— 「最近一小时 LLM 失败多少次」只能 grep 日志。
2. 无法按类型选择重试策略 —— 网络抖动该重试，PDF 损坏重试一百次也没用。
3. 无法区分修复方向 —— `is_degraded=True` 既可能是「LLM 挂了」（要改代码/扩容），
   也可能是「证据不足」（要补文档），两者的处置方向完全相反。

所以这里只做两件事：
- `AppError` 体系：给异常挂上 `code`（可聚合）与 `retryable`（可决策重试）。
- `DegradeReason`：把降级原因收敛成有限枚举，让 trace/指标可以按原因下钻。

刻意保持窄：只定义当前链路真正会用到的类型（YAGNI），
不预先铺一整套「企业级异常树」。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class DegradeReason(StrEnum):
    """降级原因枚举 —— 回答「为什么这次没给出正常答案」。

    每个取值对应一个**不同的修复方向**，这是划分它们的唯一标准：

    - LLM_TIMEOUT / LLM_ERROR      → 模型侧问题：查配额、超时、扩容、切模型
    - PARSE_FAILED                 → prompt / 解析器问题：改 prompt 或解析兜底
    - RETRIEVAL_FAILED             → 基础设施问题：Milvus 挂了、网络不通
    - RETRIEVAL_EMPTY              → 知识库覆盖问题：需要补文档
    - PARTIAL_RETRIEVAL            → 部分分支失败：结果可用但证据不完整
    - EVIDENCE_INSUFFICIENT        → 质检拦截：证据覆盖/支撑不足
    - CIRCUIT_OPEN                 → 熔断器打开，主动跳过下游
    - TOTAL_BUDGET_EXCEEDED        → 整体预算耗尽，返回部分结果
    - FEATURE_DISABLED             → 运行时开关关掉了该环节：去看开关,别查依赖
    - TOOL_FAILED                  → 工具执行出错：去看那个工具的实现与入参
    - MCP_UNAVAILABLE              → MCP 服务进程不可用：去看端口、进程、网络

    注意 RETRIEVAL_FAILED 与 RETRIEVAL_EMPTY 必须分开：
    前者是「检索服务不可用」，后者是「知识库里确实没有」。
    混在一起会让报告写出「建议补充 SOP 文档」这种误导结论，
    而真实原因是向量库宕机 —— 值班同学会跑去写文档，问题一直不修。
    """

    LLM_TIMEOUT = "llm_timeout"
    LLM_ERROR = "llm_error"
    PARSE_FAILED = "parse_failed"
    RETRIEVAL_FAILED = "retrieval_failed"
    RETRIEVAL_EMPTY = "retrieval_empty"
    PARTIAL_RETRIEVAL = "partial_retrieval"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    CIRCUIT_OPEN = "circuit_open"
    TOTAL_BUDGET_EXCEEDED = "total_budget_exceeded"

    # 为什么它不能复用 LLM_ERROR / RETRIEVAL_FAILED:
    # 这个枚举的划分标准是**修复方向**。开关关着的时候,修复方向是
    # 「把开关打开」,而 LLM_ERROR 会把人送去查模型配额和鉴权,
    # RETRIEVAL_FAILED 会把人送去查 Milvus —— 那些地方一切正常,
    # 排障就此卡住。这跟 SOP 检索失败却说「未检索到相关 SOP」
    # 是同一类错误:用一个假原因掩盖真原因。
    FEATURE_DISABLED = "feature_disabled"

    # 工具执行失败:模型选了一个工具、工具自己抛了异常。
    # 为什么不复用 LLM_ERROR:模型侧一切正常 —— 它成功地做出了决策,
    # 是被调用的那段代码坏了。修复方向在工具实现或入参 schema 上,
    # 而 LLM_ERROR 会把人送去查模型配额。
    TOOL_FAILED = "tool_failed"

    # MCP 服务进程不可用:连不上、超时、握手失败。
    # 为什么不复用 TOOL_FAILED:工具代码本身没问题,是承载它的**进程**没了。
    # 修复方向是去起进程、查端口和网络,而不是读工具实现。
    # 这两个原因在指标上分开,才能回答「是我们的工具写错了,
    # 还是运维侧的服务掉了」——它们归属不同的人处理。
    MCP_UNAVAILABLE = "mcp_unavailable"


class RetrievalStatus(StrEnum):
    """检索结果状态 —— 让调用方能区分「失败」和「空」。

    只有三态，对应三种截然不同的下游行为：
    - OK      → 正常使用证据
    - EMPTY   → 知识库确实没有：可以说「未检索到相关文档」
    - FAILED  → 检索服务不可用：必须说「检索服务异常，本次未使用知识库证据」

    这个枚举存在的意义就是消灭「静默返回空列表」这种撒谎式降级。
    """

    OK = "ok"
    EMPTY = "empty"
    FAILED = "failed"


class AppError(Exception):
    """应用异常基类。

    三个属性构成了「可运维的异常」：
    - code:      稳定的机器可读标识，用于聚合统计与告警规则
    - retryable: 是否值得重试，Worker / 熔断器据此决策
    - http_status: 映射到 HTTP 语义，让网关与 APM 能看到真实错误率
    - degrade_reason: 若该异常触发降级，对应哪个降级原因

    子类只覆盖类属性，不重写 __init__ —— 保持构造方式统一（DRY）。
    """

    code: str = "app_error"
    retryable: bool = False
    http_status: int = 500
    degrade_reason: DegradeReason | None = None

    def __init__(
        self,
        message: str,
        *,
        cause: BaseException | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause
        self.context = context or {}

    def to_dict(self) -> dict[str, Any]:
        """结构化表示，用于日志、trace 落库与 API 响应。"""
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.degrade_reason is not None:
            payload["degrade_reason"] = self.degrade_reason.value
        if self.context:
            payload["context"] = self.context
        return payload

    def __str__(self) -> str:  # pragma: no cover - 直观表示
        return self.message


class RetrievalError(AppError):
    """检索链路失败：Milvus 不可用、BM25 语料加载异常等。

    retryable=True：向量库抖动是典型的瞬时故障，重试往往能成功。
    """

    code = "retrieval_error"
    retryable = True
    http_status = 502
    degrade_reason = DegradeReason.RETRIEVAL_FAILED


class LLMError(AppError):
    """LLM 调用失败（非超时）：鉴权失败、配额耗尽、上游 5xx 等。"""

    code = "llm_error"
    retryable = True
    http_status = 502
    degrade_reason = DegradeReason.LLM_ERROR


class LLMTimeoutError(LLMError):
    """LLM 调用超时。

    单独成类的理由：超时与一般错误的处置不同 ——
    超时要看是否该调大 timeout / 换更快的模型，而 4xx 类错误重试也没用。
    http_status 用 504，让网关能正确统计网关超时。
    """

    code = "llm_timeout"
    retryable = True
    http_status = 504
    degrade_reason = DegradeReason.LLM_TIMEOUT


class ParseError(AppError):
    """模型输出解析失败：期望 JSON 却拿到自然语言等。

    retryable=False：同样的 prompt 重试通常还是解析失败，
    真正的修复是改 prompt 或加解析兜底。
    """

    code = "parse_error"
    retryable = False
    http_status = 502
    degrade_reason = DegradeReason.PARSE_FAILED


class UpstreamUnavailableError(AppError):
    """依赖的上游服务不可用：MCP 服务、监控接口等。

    刻意**不设** degrade_reason:这个类涵盖的上游不止一种(MCP 进程、
    监控接口、将来别的 HTTP 依赖),给它挂一个 `mcp_unavailable`
    会让监控接口挂掉时也报「MCP 不可用」,值班同学跑去重启 MCP 进程 ——
    又是一次用假原因掩盖真原因。要带降级原因就用下面的具体子类。
    """

    code = "upstream_unavailable"
    retryable = True
    http_status = 502


class MCPUnavailableError(UpstreamUnavailableError):
    """MCP 服务进程不可用：连不上、握手失败、请求超时。

    retryable=True：进程重启后就好了，是典型的瞬时故障。
    """

    code = "mcp_unavailable"
    retryable = True
    http_status = 502
    degrade_reason = DegradeReason.MCP_UNAVAILABLE


class ToolExecutionError(AppError):
    """工具执行本身失败：工具函数抛了异常。

    retryable=False：工具的入参是模型给的，同样的入参重跑通常
    还是同样的结果。真正瞬时的那类失败（连接超时）应该由工具内部
    包成 RetrievalError / MCPUnavailableError 抛出，它们各自
    retryable=True —— 分类在抛出点做，比在这里一刀切更准。

    http_status 用 502 而不是 500：工具的绝大多数失败来自它包装的
    那个下游（Milvus、MCP、外部 HTTP），502「上游坏了」比 500
    「我们的代码坏了」更接近事实。
    """

    code = "tool_failed"
    retryable = False
    http_status = 502
    degrade_reason = DegradeReason.TOOL_FAILED


class CircuitOpenError(AppError):
    """熔断器处于打开状态，请求未真正发出。

    retryable=False：熔断期内立即重试毫无意义，
    要等冷却结束由 HALF_OPEN 探针决定。
    """

    code = "circuit_open"
    retryable = False
    http_status = 503
    degrade_reason = DegradeReason.CIRCUIT_OPEN


class TotalBudgetExceededError(AppError):
    """整体请求预算耗尽（多次串行 LLM 调用累计超时）。"""

    code = "total_budget_exceeded"
    retryable = False
    http_status = 504
    degrade_reason = DegradeReason.TOTAL_BUDGET_EXCEEDED


class InvalidRequestError(AppError):
    """请求参数非法：未知告警来源、缺必填字段等。

    retryable=False：客户端不改参数，重试一万次也是同样结果。
    """

    code = "invalid_request"
    retryable = False
    http_status = 400


# ── 分类工具 ──────────────────────────────────────────────────
# 这些函数是「错误分类逻辑的唯一实现处」（DRY）：
# Worker 决定是否重试、熔断器决定是否计数、异常处理器决定 HTTP 码，
# 都必须走这里，不允许各自 isinstance 一遍。

# 被视为瞬时故障的标准库/三方异常类型名（按名字匹配，避免强依赖三方包）
_RETRYABLE_EXCEPTION_NAMES = frozenset(
    {
        "TimeoutError",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionRefusedError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "MilvusException",
        "MilvusUnavailableException",
    }
)

# 明确不该重试的类型名：重试只是浪费资源
_NON_RETRYABLE_EXCEPTION_NAMES = frozenset(
    {
        "FileNotFoundError",
        "NotADirectoryError",
        "IsADirectoryError",
        "PermissionError",
        "UnicodeDecodeError",
        "ValueError",
        "TypeError",
        "KeyError",
        "AttributeError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "BadRequestError",
    }
)


def is_retryable(exc: BaseException) -> bool:
    """判断一个异常是否值得重试。

    优先级：
    1. AppError 自带 retryable，最权威（我们自己抛的，语义明确）
    2. 明确的不可重试类型（文件缺失、参数错误、鉴权失败）
    3. 已知的瞬时故障类型（超时、连接错误、限流、上游 5xx）
    4. 兜底 False —— 未知异常默认不重试，避免把「PDF 损坏」这类
       确定性失败重试三次，白烧三倍资源还延后失败暴露时间。
    """
    if isinstance(exc, AppError):
        return exc.retryable

    for klass in type(exc).__mro__:
        name = klass.__name__
        if name in _NON_RETRYABLE_EXCEPTION_NAMES:
            return False
        if name in _RETRYABLE_EXCEPTION_NAMES:
            return True
    return False


def error_code_of(exc: BaseException) -> str:
    """取异常的聚合用错误码。

    AppError 用自带 code；其他异常退回类名的下划线形式，
    保证指标标签始终是有限、可枚举的集合，而不是自由文本。
    """
    if isinstance(exc, AppError):
        return exc.code
    return _to_snake(type(exc).__name__)


def degrade_reason_of(exc: BaseException) -> DegradeReason:
    """把任意异常映射到降级原因。

    映射规则集中在这里，是「降级原因判定的唯一实现处」（DRY）：
    - AppError 自带 degrade_reason 优先
    - 超时类 → LLM_TIMEOUT（当前链路里超时几乎都来自 LLM 调用）
    - 其余兜底 LLM_ERROR
    """
    if isinstance(exc, AppError) and exc.degrade_reason is not None:
        return exc.degrade_reason

    import asyncio

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return DegradeReason.LLM_TIMEOUT
    if isinstance(exc, (ValueError, TypeError)):
        return DegradeReason.PARSE_FAILED
    return DegradeReason.LLM_ERROR


def http_status_of(exc: BaseException) -> int:
    """取异常对应的 HTTP 状态码。

    这条映射保证了「业务失败」能被网关、负载均衡和 APM 正确识别成失败，
    而不是像现在这样返回 HTTP 200 + body 里写 code:500 ——
    那会让错误率监控永远是 0%。
    """
    if isinstance(exc, AppError):
        return exc.http_status

    import asyncio

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return 504
    return 500


def to_app_error(exc: BaseException, *, context: dict[str, Any] | None = None) -> AppError:
    """把任意异常收敛为 AppError，供 HTTP 边界统一处理。

    用途：接口层 `except Exception` 之后不再自己拼响应体，而是
    `raise to_app_error(exc) from exc`，交给全局异常处理器统一映射
    HTTP 状态码与响应结构 —— 状态码映射逻辑只有一处（DRY）。

    **为什么不设 degrade_reason**

    降级的定义是「给出一个更差但仍可用的结果，并说明差在哪」。
    一个打到 HTTP 边界的未知异常不是降级，是彻底失败：没有结果可用。
    如果这里给它兜一个 `llm_error`，`degrade_total{reason="llm_error"}`
    就会混进各种真实的代码 bug（KeyError、AttributeError），
    指标失去意义 —— 看到 llm_error 上涨会跑去查模型配额，而实际是代码写错了。

    所以只有我们**主动抛出**的 AppError 子类才带 degrade_reason，
    它们的语义是明确的；未知异常只带 code，用于聚合统计。
    """
    if isinstance(exc, AppError):
        return exc

    wrapped = AppError(f"{type(exc).__name__}: {exc}", cause=exc, context=context)
    # 实例属性遮蔽类属性：不为每种未知异常都造一个子类（YAGNI），
    # 分类结果仍然全部来自上面那几个唯一实现处的函数。
    wrapped.code = error_code_of(exc)
    wrapped.retryable = is_retryable(exc)
    wrapped.http_status = http_status_of(exc)
    return wrapped


def wrap_llm_exception(exc: BaseException, *, context: dict[str, Any] | None = None) -> AppError:
    """把 LLM 调用抛出的任意异常收敛为一个**自带降级原因**的 AppError。

    调用点只写 `raise wrap_llm_exception(exc)`，
    分类判断不散落在每个节点里（DRY）。
    """
    # 已经能自我描述的异常原样放行,不要重新分类。
    #
    # 这里原本写的是 `isinstance(exc, LLMError)`,只放行 LLM 家族。
    # 后果是熔断器一接进 LLM 调用点就出问题:CircuitOpenError 继承的是
    # AppError 而不是 LLMError,于是走到最后一行被重包成 LLMError,
    # degrade_reason 从 circuit_open 变成 llm_error ——
    # 「因为熔断跳过了这次调用」被记成「模型调用失败了」。
    #
    # 值班同学看到 llm_error 上涨会去查模型配额和上游状态,
    # 而真实情况是我们自己的熔断器打开了,模型服务可能一直是好的。
    # 这跟 SOP 检索失败却说「未检索到相关 SOP」是同一类错误:
    # **用一个假原因掩盖真原因**。TotalBudgetExceededError 同理。
    #
    # 判据用 degrade_reason 是否为 None,而不是列一串 isinstance:
    # 「自带降级原因」正是「这个异常已经说清了自己为什么导致降级」的定义,
    # 以后新增任何带 degrade_reason 的子类都自动正确(开闭原则)。
    if isinstance(exc, AppError) and exc.degrade_reason is not None:
        return exc

    import asyncio

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return LLMTimeoutError(f"LLM 调用超时: {exc}", cause=exc, context=context)
    return LLMError(f"LLM 调用失败: {exc}", cause=exc, context=context)


def _to_snake(name: str) -> str:
    """CamelCase → snake_case，用于生成稳定的指标标签。"""
    out: list[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index > 0:
            out.append("_")
        out.append(char.lower())
    return "".join(out)
