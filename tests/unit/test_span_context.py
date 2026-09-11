"""span 收集与落盘的单元测试。

覆盖四组契约：
1. `span_scope` 的三态记录（ok / degraded / error）与「不吞异常」
2. 脱离收集上下文时静默空转（脚本直调节点、单测不该报错）
3. **并行分支的 span 不丢** —— 本模块最容易写错的一条语义
4. `flush_spans` 写库成功路径，以及失败时绝不抛
"""

import asyncio
import time

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.core.span_context import (
    collected_spans,
    flush_spans,
    record_span,
    reset_span_collection,
    span_scope,
    start_span_collection,
)
from app.models import chat_run_span  # noqa: F401
from app.models import chat_run_trace  # noqa: F401
from app.models import conversation  # noqa: F401
from app.models.chat_run_span import ChatRunSpan, SpanStatus
from app.repositories.chat_run_trace_repository import ChatRunTraceRepository
from app.repositories.conversation_repository import ConversationRepository


@pytest.fixture
def collecting():
    """开一个 span 收集上下文，测试结束还原。"""
    token = start_span_collection()
    try:
        yield
    finally:
        reset_span_collection(token)


class TestSpanScope:
    def test_records_ok_span_with_duration(self, collecting) -> None:
        with span_scope("rewrite") as span:
            span.set_payload(sub_query_count=4)
            time.sleep(0.01)

        spans = collected_spans()
        assert len(spans) == 1
        assert spans[0]["node"] == "rewrite"
        assert spans[0]["status"] is SpanStatus.OK
        # 睡了 10ms，耗时必须是正数且量级对得上
        assert spans[0]["duration_ms"] >= 5
        assert spans[0]["payload"]["sub_query_count"] == 4

    def test_marks_degraded_with_reason(self, collecting) -> None:
        with span_scope("retrieve_each") as span:
            span.mark_degraded("retrieval_failed")

        span_row = collected_spans()[0]
        assert span_row["status"] is SpanStatus.DEGRADED
        assert span_row["payload"]["degrade_reason"] == "retrieval_failed"

    def test_records_error_and_reraises(self, collecting) -> None:
        """异常路径要记 ERROR + error_code，然后**原样放行异常**。

        span 是旁路观测，不能改变控制流 —— 吞掉异常等于用埋点
        把一次真实失败变成了静默成功。
        """
        with pytest.raises(ValueError, match="boom"):
            with span_scope("generate"):
                raise ValueError("boom")

        span_row = collected_spans()[0]
        assert span_row["status"] is SpanStatus.ERROR
        assert span_row["payload"]["error_code"] == "value_error"
        assert "boom" in span_row["payload"]["error"]

    def test_no_collection_context_is_noop(self) -> None:
        """脱离收集上下文时静默跳过，而不是抛错。

        脚本直调节点、单测直接跑函数都会走到这条路径。
        观测设施不该给业务代码增加「必须先初始化」的前置条件。
        """
        with span_scope("dedup") as span:
            span.set_payload(deduped_count=3)
        assert collected_spans() == []

    def test_record_span_outside_context_is_noop(self) -> None:
        record_span(
            "orphan",
            duration_ms=1,
            status=SpanStatus.OK,
            started_at=None,  # type: ignore[arg-type]
        )
        assert collected_spans() == []


class TestParallelBranches:
    """并行分支的 span 必须能被父上下文看到。

    这是整个模块能成立的前提。`copy_context()` 拷贝的是「变量→对象」
    的绑定关系，不是对象本身，所以子上下文里 `append` 父看得见、
    `set` 父看不见。写错成 set 的话，是那种「单测能过、并发才丢」的 bug。
    """

    def test_to_thread_branch_spans_reach_parent(self, collecting) -> None:
        def worker(name: str) -> None:
            with span_scope(name) as span:
                span.set_payload(doc_count=2)

        async def run() -> None:
            await asyncio.gather(
                asyncio.to_thread(worker, "retrieve_each"),
                asyncio.to_thread(worker, "retrieve_each"),
                asyncio.to_thread(worker, "retrieve_each"),
            )

        asyncio.run(run())

        spans = collected_spans()
        assert len(spans) == 3
        assert {row["node"] for row in spans} == {"retrieve_each"}

    def test_async_tasks_spans_reach_parent(self, collecting) -> None:
        async def worker(name: str) -> None:
            with span_scope(name):
                await asyncio.sleep(0)

        async def run() -> None:
            await asyncio.gather(worker("a"), worker("b"))

        asyncio.run(run())
        assert sorted(row["node"] for row in collected_spans()) == ["a", "b"]

    def test_nested_collection_restores_outer(self) -> None:
        """嵌套收集退出后要回到外层的篮子，而不是清空。"""
        outer = start_span_collection()
        with span_scope("outer-node"):
            pass

        inner = start_span_collection()
        with span_scope("inner-node"):
            pass
        assert [row["node"] for row in collected_spans()] == ["inner-node"]
        reset_span_collection(inner)

        assert [row["node"] for row in collected_spans()] == ["outer-node"]
        reset_span_collection(outer)


class TestFlushSpans:
    @staticmethod
    def _setup_db():
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return engine, sessionmaker(bind=engine, expire_on_commit=False)

    def test_writes_collected_spans_once(self, collecting) -> None:
        engine, SessionLocal = self._setup_db()
        try:
            with SessionLocal() as db:
                ConversationRepository(db).get_or_create_session("s-1")
                trace = ChatRunTraceRepository(db).create_trace(
                    session_id="s-1",
                    source="chat_v2",
                    question="q",
                    answer="a",
                    sub_queries=[],
                    retrieved_count=0,
                    used_documents=[],
                    validation=None,
                )

                with span_scope("rewrite"):
                    pass
                with span_scope("generate") as span:
                    span.mark_degraded("llm_error")

                written = flush_spans(db, trace_id=trace.id)
                assert written == 2

                rows = db.query(ChatRunSpan).order_by(ChatRunSpan.node).all()
                assert [row.node for row in rows] == ["generate", "rewrite"]
                assert rows[0].status is SpanStatus.DEGRADED
                assert rows[0].trace_id == trace.id
                assert rows[1].status is SpanStatus.OK
        finally:
            engine.dispose()

    def test_no_spans_writes_nothing(self, collecting) -> None:
        engine, SessionLocal = self._setup_db()
        try:
            with SessionLocal() as db:
                assert flush_spans(db, trace_id="whatever") == 0
        finally:
            engine.dispose()

    def test_write_failure_never_raises(self, collecting) -> None:
        """写不进 span 是坏消息，但让它顶掉一次已经成功的回答是更坏的消息。

        这里给一个不存在的 trace_id，外键约束会让 commit 失败。
        函数必须返回 0 而不是抛异常 —— 主流程此刻已经把答案生成完了。
        """
        engine, SessionLocal = self._setup_db()
        try:
            with SessionLocal() as db:
                # SQLite 默认不强制外键，显式打开才能触发约束失败
                db.execute(__import__("sqlalchemy").text("PRAGMA foreign_keys=ON"))
                with span_scope("rewrite"):
                    pass
                assert flush_spans(db, trace_id="does-not-exist") == 0
        finally:
            engine.dispose()

    def test_explicit_spans_argument_overrides_context(self, collecting) -> None:
        """允许显式传 spans：流式路径需要在还原收集器之前先取快照。"""
        engine, SessionLocal = self._setup_db()
        try:
            with SessionLocal() as db:
                ConversationRepository(db).get_or_create_session("s-2")
                trace = ChatRunTraceRepository(db).create_trace(
                    session_id="s-2",
                    source="chat_v2_stream",
                    question="q",
                    answer="a",
                    sub_queries=[],
                    retrieved_count=0,
                    used_documents=[],
                    validation=None,
                )
                with span_scope("rewrite"):
                    pass
                snapshot = collected_spans()

                reset_span_collection(start_span_collection())
                assert flush_spans(db, trace_id=trace.id, spans=snapshot) == 1
        finally:
            engine.dispose()
