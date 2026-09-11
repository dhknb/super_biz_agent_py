"""request_id 上下文与 RQ 传递的单元测试。

覆盖三个契约：
1. ContextVar 的读写与还原语义（嵌套后能回到进入前的确切状态）
2. 落库用的 get_request_id_or_none 在无上下文时返回 None 而非占位符
3. RQ job.meta 的往返：入队写入、worker 侧读回
"""

from unittest.mock import MagicMock, patch

from app.core.job_context import (
    JOB_META_REQUEST_ID,
    enqueue_with_request_id,
    with_job_request_id,
)
from app.core.request_context import (
    NO_REQUEST_ID,
    get_request_id,
    get_request_id_or_none,
    new_request_id,
    request_id_scope,
    reset_request_id,
    set_request_id,
)


class TestRequestIdContextVar:
    def test_default_is_placeholder(self) -> None:
        """脱离请求上下文时读取不抛异常，返回占位符。"""
        assert get_request_id() == NO_REQUEST_ID

    def test_set_and_reset_restores_previous(self) -> None:
        token = set_request_id("abc123")
        assert get_request_id() == "abc123"
        reset_request_id(token)
        assert get_request_id() == NO_REQUEST_ID

    def test_nested_scope_restores_outer(self) -> None:
        """嵌套作用域退出后必须回到外层的值，而不是占位符。

        这条是 reset(token) 相对于「重新 set 回去」的关键差异。
        """
        with request_id_scope("outer"):
            assert get_request_id() == "outer"
            with request_id_scope("inner"):
                assert get_request_id() == "inner"
            assert get_request_id() == "outer"
        assert get_request_id() == NO_REQUEST_ID

    def test_scope_generates_id_when_none(self) -> None:
        """独立触发的后台任务也该有自己的 id，而不是落在占位符上。"""
        with request_id_scope(None) as rid:
            assert rid != NO_REQUEST_ID
            assert get_request_id() == rid

    def test_scope_restores_on_exception(self) -> None:
        try:
            with request_id_scope("boom"):
                raise RuntimeError("x")
        except RuntimeError:
            pass
        assert get_request_id() == NO_REQUEST_ID

    def test_new_request_id_is_short_and_unique(self) -> None:
        first, second = new_request_id(), new_request_id()
        assert first != second
        assert len(first) == 16


class TestGetRequestIdOrNone:
    def test_returns_none_outside_request(self) -> None:
        """落库存 NULL 而不是 "-"：占位符会被当成真实取值污染聚合统计。"""
        assert get_request_id_or_none() is None

    def test_returns_value_inside_scope(self) -> None:
        with request_id_scope("rid-1"):
            assert get_request_id_or_none() == "rid-1"


class TestJobMetaRoundTrip:
    def test_enqueue_writes_request_id_into_meta(self) -> None:
        queue = MagicMock()
        with request_id_scope("rid-enqueue"):
            enqueue_with_request_id(queue, print, "arg", job_timeout=60)

        _, kwargs = queue.enqueue.call_args
        assert kwargs["meta"][JOB_META_REQUEST_ID] == "rid-enqueue"
        # 其余参数必须原样透传，否则 job_timeout / retry 会被静默丢掉
        assert kwargs["job_timeout"] == 60

    def test_enqueue_preserves_caller_meta(self) -> None:
        queue = MagicMock()
        with request_id_scope("rid-2"):
            enqueue_with_request_id(queue, print, meta={"custom": "v"})

        _, kwargs = queue.enqueue.call_args
        assert kwargs["meta"]["custom"] == "v"
        assert kwargs["meta"][JOB_META_REQUEST_ID] == "rid-2"

    def test_worker_restores_request_id_from_meta(self) -> None:
        seen: list[str] = []

        @with_job_request_id
        def task() -> None:
            seen.append(get_request_id())

        job = MagicMock()
        job.meta = {JOB_META_REQUEST_ID: "rid-from-api"}
        with patch("app.core.job_context.get_current_job", return_value=job):
            task()

        assert seen == ["rid-from-api"]
        # 任务结束后必须还原，否则 worker 会把上一个 job 的 id 带给下一个
        assert get_request_id() == NO_REQUEST_ID

    def test_worker_generates_id_when_meta_missing(self) -> None:
        """直接以普通函数调用（测试/脚本）时不应报错，且仍有 id 可追踪。"""
        seen: list[str] = []

        @with_job_request_id
        def task() -> None:
            seen.append(get_request_id())

        with patch("app.core.job_context.get_current_job", return_value=None):
            task()

        assert seen and seen[0] != NO_REQUEST_ID

    def test_decorator_preserves_function_identity(self) -> None:
        """RQ 靠「模块路径 + 函数名」反序列化任务，包装后必须保住这些属性。"""

        @with_job_request_id
        def my_task() -> None: ...

        assert my_task.__name__ == "my_task"
        assert my_task.__module__ == __name__
