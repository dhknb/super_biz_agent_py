"""Worker 失败处理的单元测试。

覆盖三个契约：
1. 重试裁决：可重试的失败保留额度，不可重试的失败就地否决剩余重试
2. 失败原因文案：带 error_code（可聚合）、带重试状态（不误导值班的人）
3. 落盘的「绝不外抛」：抢救过程中的任何异常都不能顶掉原始异常

第 3 条是本模块存在的理由。原来的写法在 `except` 里用**同一个 session**
再 commit 一次，而连接已断时这次 commit 也会抛 —— 于是 job.exc_info 里
留下的是 `PendingRollbackError` 之类的二次异常，真正的根因（比如 PDF 解析
失败）被彻底抹掉。偏偏基础设施故障是最常见的情形：它同时打断了正常工作
和抢救工作。
"""

from unittest.mock import MagicMock, patch

import pytest

from app.core.errors import LLMTimeoutError, ParseError, RetrievalError
from app.core.job_failure import (
    apply_retry_policy,
    default_job_retry,
    format_failure_message,
    record_job_failure,
)


def _fake_job(retries_left: int) -> MagicMock:
    """造一个只带 retries_left 的假 job。

    必须显式赋整数：MagicMock 的属性默认是 MagicMock，
    而 `retries_left > 0` 对 MagicMock 恒为真值，会让断言失去意义。
    """
    job = MagicMock()
    job.retries_left = retries_left
    return job


class TestApplyRetryPolicy:
    def test_retryable_error_keeps_remaining_retries(self) -> None:
        """向量库抖动是典型瞬时故障，额度必须留着。"""
        job = _fake_job(2)
        with patch("app.core.job_failure.get_current_job", return_value=job):
            will_retry = apply_retry_policy(RetrievalError("milvus 抖动"), job_id="j1")

        assert will_retry is True
        assert job.retries_left == 2

    def test_non_retryable_error_cancels_remaining_retries(self) -> None:
        """PDF 解析失败重试三次还是同样结果，白烧三倍资源还延后暴露失败。"""
        job = _fake_job(3)
        with patch("app.core.job_failure.get_current_job", return_value=job):
            will_retry = apply_retry_policy(ParseError("PDF 损坏"), job_id="j2")

        assert will_retry is False
        # 清零而不是留着：RQ 的 should_retry 判的是 `is not None and > 0`
        assert job.retries_left == 0

    def test_exhausted_retries_reports_no_retry(self) -> None:
        """额度用尽时即使异常可重试，也不能宣称「将自动重试」。"""
        job = _fake_job(0)
        with patch("app.core.job_failure.get_current_job", return_value=job):
            will_retry = apply_retry_policy(RetrievalError("还是抖"), job_id="j3")

        assert will_retry is False

    def test_without_rq_context_is_pure(self) -> None:
        """脱离 RQ 上下文（单测/脚本直调）时只返回判定，不产生副作用。"""
        with patch("app.core.job_failure.get_current_job", return_value=None):
            assert apply_retry_policy(RetrievalError("x"), job_id="j4") is True
            assert apply_retry_policy(ParseError("y"), job_id="j5") is False


class TestFormatFailureMessage:
    def test_message_carries_error_code(self) -> None:
        """error_code 是有限枚举，能做「这类失败出现过几次」的聚合。"""
        message = format_failure_message(ParseError("PDF 损坏"), will_retry=False)

        assert message.startswith("[parse_error]")
        assert "PDF 损坏" in message

    def test_message_states_pending_retry(self) -> None:
        """页面写着 FAILED、后台还排着重试时，必须说清楚。"""
        message = format_failure_message(LLMTimeoutError("超时"), will_retry=True)

        assert "RQ 将自动重试" in message

    def test_message_omits_retry_hint_when_final(self) -> None:
        message = format_failure_message(LLMTimeoutError("超时"), will_retry=False)

        assert "重试" not in message


class TestRecordJobFailure:
    def test_writes_through_independent_session(self) -> None:
        """落盘必须用新建的 session —— 原 session 的连接可能已经废了。"""
        session = MagicMock()
        seen: list[object] = []

        with patch("app.core.job_failure.SessionLocal", return_value=session):
            record_job_failure(lambda s: seen.append(s), job_id="j1")

        assert seen == [session]
        session.commit.assert_called_once()
        session.close.assert_called_once()

    def test_commit_failure_never_masks_original_exception(self) -> None:
        """本模块存在的理由：抢救失败不能变成新的异常往外抛。"""
        session = MagicMock()
        session.commit.side_effect = RuntimeError("connection already closed")

        with patch("app.core.job_failure.SessionLocal", return_value=session):
            # 不抛就是通过；抛了就说明二次异常会顶掉原始异常
            record_job_failure(lambda s: None, job_id="j2")

        session.rollback.assert_called_once()
        session.close.assert_called_once()

    def test_session_construction_failure_is_swallowed(self) -> None:
        """连接池耗尽时连 session 都建不出来，也只能记日志。"""
        with patch(
            "app.core.job_failure.SessionLocal",
            side_effect=RuntimeError("pool exhausted"),
        ):
            record_job_failure(lambda s: None, job_id="j3")

    def test_close_failure_never_escapes(self) -> None:
        """close 在连接已断时同样会抛，从 finally 里逃出去就又变成顶掉原始异常。"""
        session = MagicMock()
        session.close.side_effect = RuntimeError("socket is dead")

        with patch("app.core.job_failure.SessionLocal", return_value=session):
            record_job_failure(lambda s: None, job_id="j4")

    def test_write_status_failure_rolls_back_and_is_swallowed(self) -> None:
        """写状态本身失败（比如行已被删）也走同一条路径。"""
        session = MagicMock()

        def _boom(_: object) -> None:
            raise RuntimeError("row vanished")

        with patch("app.core.job_failure.SessionLocal", return_value=session):
            record_job_failure(_boom, job_id="j5")

        session.commit.assert_not_called()
        session.rollback.assert_called_once()
        session.close.assert_called_once()


class TestDefaultJobRetry:
    def test_returns_fresh_instance_each_call(self) -> None:
        """Retry 实例带可变状态，两个队列共享同一个对象是自找麻烦。"""
        first, second = default_job_retry(), default_job_retry()

        assert first is not second
        assert first.max == 3
        assert first.intervals == [10, 30, 60]
