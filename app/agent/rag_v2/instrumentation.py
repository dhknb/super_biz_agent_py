"""给 LangGraph 节点统一挂上 span 埋点。

## 为什么在这里包，而不是在每个节点里写计时代码

节点有五个（rewrite / retrieve_each / dedup / generate / validate_answer），
在每个节点体内写 `t0 = time.perf_counter()` 是五份重复代码，
而且新增节点时一定会有人忘记加（DRY）。

在 `graph.add_node` 的入口包一层，覆盖面是「所有注册进图的节点」，
漏不掉。节点本身的代码一行不用改 —— 它们仍然是纯粹的 `state -> patch`，
不知道 span 的存在。这也是 SOLID 里的 O：对扩展开放（加节点自动被埋点），
对修改关闭（不用改节点实现）。

## payload 里的计数从哪来

节点的返回值就是它干了什么的完整描述（LangGraph 的 patch）。
所以不需要每个节点单独告诉埋点层「我要记什么」——
从 patch 里按通用规则抽即可：有 documents 就记条数，
有 sub_queries 就记个数，有 retrieve_failures 就记失败数。

这样做的代价是「记的东西是通用的，不是每个节点最想记的」。
接受这个代价：真需要节点专属指标时，节点可以在 patch 里
多返回一个 `_span` 字段（下方 `_SPAN_HINT_KEY`），按需扩展。
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from app.core.metrics import count_degrade
from app.core.span_context import span_scope

# 节点可选地在 patch 里塞这个键，往 span payload 里补节点专属计数。
# 用下划线前缀 + 埋点层负责剔除，保证它不会污染图状态。
_SPAN_HINT_KEY = "_span"


def _derive_payload(patch: Any) -> dict[str, Any]:
    """从节点返回的 patch 里抽取关键计数。

    只认已知的几个键，其余忽略 —— span payload 是给人看的摘要，
    不是 state 的全量镜像。把整个 patch 序列化进去，
    等于把 6 篇文档正文抄进 span 表，那是在制造第四份副本。
    """
    if not isinstance(patch, dict):
        return {}

    payload: dict[str, Any] = {}

    documents = patch.get("documents")
    if isinstance(documents, list):
        payload["doc_count"] = len(documents)

    deduped = patch.get("deduped_documents")
    if isinstance(deduped, list):
        payload["deduped_count"] = len(deduped)

    sub_queries = patch.get("sub_queries")
    if isinstance(sub_queries, list):
        payload["sub_query_count"] = len(sub_queries)

    failures = patch.get("retrieve_failures")
    if isinstance(failures, list) and failures:
        payload["retrieve_failure_count"] = len(failures)
        # 只留错误码不留完整报文：错误码可聚合，报文在日志里已经有了。
        payload["retrieve_failure_codes"] = sorted(
            {str(item.get("code")) for item in failures if isinstance(item, dict)}
        )

    answer = patch.get("answer")
    if isinstance(answer, str):
        payload["answer_length"] = len(answer)

    validation = patch.get("validation")
    if isinstance(validation, dict):
        # 质检的三个关键判定：分数 + 是否拦截 + 是否真的校验过。
        # blocked 和 validated 必须都记：前者是「过没过」，
        # 后者是「有没有验」，质检器自己挂掉时两者含义完全不同。
        payload["coverage_score"] = validation.get("coverage_score")
        payload["groundedness_score"] = validation.get("groundedness_score")
        payload["blocked"] = validation.get("blocked")
        payload["validated"] = validation.get("validated")

    hint = patch.get(_SPAN_HINT_KEY)
    if isinstance(hint, dict):
        payload.update(hint)

    return payload


def _derive_degradation(patch: Any) -> tuple[bool, str | None]:
    """判断这次执行算不算降级，以及降级原因。

    返回 `(是否降级, 原因)`。原因可能为 None —— 确实降了但节点没说明原因时。

    为什么要看 patch 而不只看有没有抛异常：
    项目里的节点是**刻意不抛**的 —— retrieve_each 兜住检索异常、
    generate 兜住 LLM 异常，都返回一个「成功的」patch。
    只按异常判定的话，这些节点永远是 OK，
    检索失败率、LLM 失败率两个指标恒为 0，而故障是真实发生了的。
    """
    if not isinstance(patch, dict):
        return False, None

    reasons = patch.get("degrade_reasons")
    if reasons:
        first = str(reasons[0]) if isinstance(reasons, list) and reasons else None
        return True, first

    if patch.get("retrieve_failures"):
        return True, None

    return False, None


def instrument_node(
    node: str,
    func: Callable[..., Any],
) -> Callable[..., Any]:
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
                _apply(span, patch)
                return _strip_hint(patch)

        _copy_identity(async_wrapper, func)
        return async_wrapper

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        with span_scope(node) as span:
            patch = func(*args, **kwargs)
            _apply(span, patch)
            return _strip_hint(patch)

    _copy_identity(sync_wrapper, func)
    return sync_wrapper


def _apply(span: span_scope, patch: Any) -> None:
    """把从 patch 推导出的计数与状态写进 span,并顺手记一次降级指标。

    为什么 degrade_total 埋在这里,而不是在五个节点各自的降级分支里:

    这里是 rag_v2 全部降级判定的**唯一**汇聚点 —— `_derive_degradation`
    已经把「什么算降级」这条规则收敛完了。指标跟着它走,
    就自动获得两个性质:
      1. 新增节点自动被统计,不会有人忘记加 `count_degrade`(DRY + 开闭原则)。
      2. **指标与 span 永远同源**。如果各节点自己埋,迟早出现
         「span 表里记了降级、degrade_total 没涨」这种对不上的情况,
         而排障时两个数据源互相矛盾比没有数据更糟 ——
         你不知道该信哪个,只能两边都不信。

    reason 为 None 时 count_degrade 会归到 `unknown`,那是有意的:
    见 app/core/metrics.py 里的说明。
    """
    span.set_payload(**_derive_payload(patch))
    degraded, reason = _derive_degradation(patch)
    if degraded:
        span.mark_degraded(reason)
        count_degrade(reason)


def _strip_hint(patch: Any) -> Any:
    """把 `_span` 提示键从 patch 里摘掉，不让它进入图状态。

    RAGState 是 TypedDict(total=False)，多塞一个未声明的键不会报错，
    但会跟着 state 一路传下去，最终出现在 astream 的快照里 ——
    埋点的内部约定不该泄漏成对外可见的状态字段。
    """
    if isinstance(patch, dict) and _SPAN_HINT_KEY in patch:
        return {key: value for key, value in patch.items() if key != _SPAN_HINT_KEY}
    return patch


def _copy_identity(wrapper: Callable[..., Any], func: Callable[..., Any]) -> None:
    """保留原函数的名字与文档。

    不用 `functools.wraps`：它会连 `__wrapped__` 一起设上，
    而 LangGraph 在某些版本里会顺着 `__wrapped__` 去取原函数签名，
    从而绕过包装。只复制展示用的两个属性，够了也更安全。
    """
    wrapper.__name__ = getattr(func, "__name__", "node")
    wrapper.__doc__ = getattr(func, "__doc__", None)
