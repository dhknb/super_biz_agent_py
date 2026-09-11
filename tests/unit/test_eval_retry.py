"""评测重试逻辑单元测试。

为什么这段必须有断言:它决定的是**报告可信度**,而不是功能对错。

写它的直接原因是同一个事故连吃两次 —— DashScope 嵌入接口一次瞬时
Connection error,让跑到一半(约 48/82)的整轮评测作废。更危险的不是
浪费时间,而是「失败样本被跳过」这种处理方式:分母跟着变小,82 条里
死掉 30 条也能算出一个漂亮的 MRR,报告看起来和干净的一模一样。

所以这里钉住三件事:
    1. 瞬时失败要重试,不能一碰就死
    2. 连续失败要记成**空召回并计数**,不能跳过、不能伪造
    3. 重试花掉的墙钟时间不能进耗时统计 —— s/case 是跨组对比列
"""

from __future__ import annotations

import pytest

from tests.eval.run_eval import _search_with_retry, run


class _FlakyRetriever:
    """前 fail_times 次抛异常,之后正常返回。"""

    def __init__(self, fail_times: int, exc: Exception | None = None) -> None:
        self._fail_times = fail_times
        self._exc = exc or RuntimeError("混合检索失败: 查询嵌入失败: Connection error.")
        self.call_count = 0

    def search(self, query: str, top_k: int):  # noqa: ANN201, ARG002
        self.call_count += 1
        if self.call_count <= self._fail_times:
            raise self._exc
        return ["cpu_high_usage"], ["c1"]


class _DeadRetriever:
    def __init__(self) -> None:
        self.call_count = 0

    def search(self, query: str, top_k: int):  # noqa: ANN201, ARG002
        self.call_count += 1
        raise RuntimeError("Connection error.")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """把退避 sleep 掉。

    真睡 2+4 秒会让这个文件跑成 6 秒起 —— 单测慢了就会被跳过,
    然后这些断言等于不存在。退避**时长**不是这里要测的东西。
    """
    monkeypatch.setattr("tests.eval.run_eval.time.sleep", lambda _s: None)


# ---------------------------------------------------------------------------
# _search_with_retry
# ---------------------------------------------------------------------------

def test_first_attempt_success_does_not_retry() -> None:
    """一次就成的样本必须只调一次 —— 否则整轮评测的接口调用量翻倍。"""
    retriever = _FlakyRetriever(fail_times=0)

    docs, chunks, attempts, failed = _search_with_retry(retriever, "q", top_k=5)

    assert docs == ["cpu_high_usage"]
    assert chunks == ["c1"]
    assert attempts == 1
    assert failed is False
    assert retriever.call_count == 1


def test_transient_failure_recovers_and_reports_attempt_count() -> None:
    """瞬时抖动要能救回来,且如实报告「这条重试过」。"""
    retriever = _FlakyRetriever(fail_times=2)

    docs, _, attempts, failed = _search_with_retry(retriever, "q", top_k=5)

    assert docs == ["cpu_high_usage"]
    assert attempts == 3
    assert failed is False


def test_persistent_failure_returns_empty_and_flags_failed() -> None:
    """真断网:记空召回 + failed=True,而不是抛出去炸掉整轮。

    返回空会把 MRR 拉低,这是**期望行为** —— 指标应该反映
    「这轮没拿到结果」,而不是把这条悄悄抹掉。
    """
    retriever = _DeadRetriever()

    docs, chunks, attempts, failed = _search_with_retry(
        retriever, "q", top_k=5, max_attempts=3
    )

    assert docs == []
    assert chunks == []
    assert attempts == 3
    assert failed is True
    assert retriever.call_count == 3


def test_max_attempts_is_respected() -> None:
    """max_attempts=1 等于关掉重试,不能偷偷多打接口。"""
    retriever = _DeadRetriever()

    _, _, attempts, failed = _search_with_retry(
        retriever, "q", top_k=5, max_attempts=1
    )

    assert attempts == 1
    assert failed is True
    assert retriever.call_count == 1


# ---------------------------------------------------------------------------
# run() 的元信息
# ---------------------------------------------------------------------------

_GOLDEN = [
    {"id": "case-1", "question": "CPU 高怎么办", "expected_doc_ids": ["cpu_high_usage"], "tags": ["cpu"]},
    {"id": "case-2", "question": "磁盘满了", "expected_doc_ids": ["disk_high_usage"], "tags": ["disk"]},
]


class _OneDeadRetriever:
    """第一条样本永久失败,第二条正常 —— 模拟评测中途网络抖了一下。"""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def search(self, query: str, top_k: int):  # noqa: ANN201, ARG002
        if query == "CPU 高怎么办":
            raise RuntimeError("Connection error.")
        return ["disk_high_usage"], []


def test_run_records_failed_cases_in_meta() -> None:
    """失败样本必须留在 meta 里,否则脏报告和干净报告长得一样。"""
    report, cases, meta = run(
        _OneDeadRetriever(), _GOLDEN, top_k=5, max_attempts=2
    )

    assert meta["failed_case_count"] == 1
    assert meta["failed_cases"] == ["case-1"]
    # 分母不能因为失败而变小:2 条就是 2 条
    assert report.total == 2
    assert len(cases) == 2
    # 失败那条如实记成空召回
    failed_case = next(c for c in cases if c.case_id == "case-1")
    assert failed_case.retrieved_doc_ids == []


def test_run_clean_pass_reports_zero_failures() -> None:
    """全绿时这两个字段也要存在 ——「本轮没有失败」本身是可信度的一部分。"""
    _, _, meta = run(_FlakyRetriever(fail_times=0), _GOLDEN, top_k=5)

    assert meta["failed_case_count"] == 0
    assert meta["failed_cases"] == []
    assert meta["retried_cases"] == []


def test_run_latency_excludes_retry_backoff() -> None:
    """重试的样本不进耗时统计。

    s/case 是跨组对比列(A 组 0.247 vs C 组 3.307 这种)。把退避 sleep
    算进去的话,一次网络抖动就能让某组看起来慢了几倍,而那与检索策略
    的成本毫无关系。
    """
    # 两条样本都重试一次成功 → 没有任何「一次成」的样本
    class _AllFlaky:
        def __init__(self) -> None:
            self._counts: dict[str, int] = {}

        def search(self, query: str, top_k: int):  # noqa: ANN201, ARG002
            self._counts[query] = self._counts.get(query, 0) + 1
            if self._counts[query] == 1:
                raise RuntimeError("Connection error.")
            return ["cpu_high_usage"], []

    _, _, meta = run(_AllFlaky(), _GOLDEN, top_k=5)

    assert sorted(meta["retried_cases"]) == ["case-1", "case-2"]
    assert meta["failed_case_count"] == 0
    # 没有干净样本可统计时,耗时应如实为 0,而不是把重试耗时顶上来
    assert meta["avg_seconds_per_case"] == 0.0
