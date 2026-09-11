"""SOP 检索服务单元测试。

全部用假检索器（依赖注入），不连 Milvus，覆盖：
- 正常检索 → 转成 VERIFIED_FACT / source=SOP 证据，status=OK
- 来源标题：h 标题优先、文件名兜底、未知来源
- 检索抛异常 → 空证据 + status=FAILED（主流程不崩，但**如实标记故障**）
- 空结果 → 空证据 + status=EMPTY
- excerpt 截断到上限

其中 FAILED 与 EMPTY 必须能被区分，是这一版最关键的行为：
两者证据都是空的，但一个是依赖故障、一个是真实业务结论，
处置方向相反（修向量库 vs 补文档）。
"""

from types import SimpleNamespace

from app.core.errors import RetrievalStatus
from app.models.aiops import AlarmEvent, AlarmSeverity
from app.models.aiops_report import EvidenceSource, EvidenceType
from app.services.sop_retrieval_service import SopRetrievalService, _MAX_EXCERPT_CHARS


def _doc(content: str, metadata: dict):
    """构造一个鸭子类型的假文档。"""
    return SimpleNamespace(page_content=content, metadata=metadata)


class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs
        self.last_query = None
        self.last_top_k = None

    def retrieve_documents(self, query: str, top_k: int = 3):
        self.last_query = query
        self.last_top_k = top_k
        return self._docs


class _BrokenRetriever:
    def retrieve_documents(self, query: str, top_k: int = 3):
        raise RuntimeError("Milvus 连接失败")


def _alarm() -> AlarmEvent:
    return AlarmEvent(
        alert_name="HighCPUUsage",
        severity=AlarmSeverity.CRITICAL,
        service="order-api",
        summary="CPU 持续超过 90%",
    )


def test_retrieve_maps_docs_to_verified_sop_evidence():
    docs = [
        _doc("CPU 高排查步骤：先看 top", {"_file_name": "cpu_high_usage.md", "h1": "CPU 排查"}),
    ]
    retriever = _FakeRetriever(docs)
    service = SopRetrievalService(retriever=retriever)

    evidence, status = service.retrieve_sop_evidence(_alarm())

    assert status is RetrievalStatus.OK
    assert len(evidence) == 1
    ev = evidence[0]
    assert ev.type == EvidenceType.VERIFIED_FACT
    assert ev.source == EvidenceSource.SOP
    assert "cpu_high_usage.md" in ev.source_title
    assert "CPU 排查" in ev.source_title
    assert ev.excerpt.startswith("CPU 高排查步骤")


def test_retrieval_query_passed_to_retriever():
    retriever = _FakeRetriever([])
    service = SopRetrievalService(retriever=retriever)
    service.retrieve_sop_evidence(_alarm())
    # retrieval_query 拼了告警名 + 服务 + 摘要
    assert "HighCPUUsage" in retriever.last_query
    assert "order-api" in retriever.last_query


def test_source_title_falls_back_to_filename():
    docs = [_doc("内容", {"_file_name": "disk.md"})]
    service = SopRetrievalService(retriever=_FakeRetriever(docs))
    evidence, _ = service.retrieve_sop_evidence(_alarm())
    assert evidence[0].source_title == "disk.md"


def test_source_title_unknown_when_no_metadata():
    docs = [_doc("内容", {})]
    service = SopRetrievalService(retriever=_FakeRetriever(docs))
    evidence, _ = service.retrieve_sop_evidence(_alarm())
    assert evidence[0].source_title == "未知来源"


def test_broken_retriever_reports_failed_not_empty():
    """检索服务挂了 → status=FAILED，而不是伪装成「没找到」。

    这个用例的旧版本叫 test_broken_retriever_degrades_to_empty，
    只断言 `evidence == []` —— 它把撒谎行为当成正确行为固化了下来：
    检索炸了和知识库里真没这篇，在返回值上完全一样。

    证据为空这点没变（确实没有证据可用），变的是**必须同时说清为什么空**。
    """
    service = SopRetrievalService(retriever=_BrokenRetriever())
    evidence, status = service.retrieve_sop_evidence(_alarm())
    assert evidence == []
    assert status is RetrievalStatus.FAILED


def test_empty_result_reports_empty_not_failed():
    """检索服务正常、知识库确实没有 → status=EMPTY。

    与上一个用例成对存在：两者的 evidence 都是 []，
    靠 status 区分。这一对是本次改动的核心，缺一个就测不出区别。
    """
    service = SopRetrievalService(retriever=_FakeRetriever([]))
    evidence, status = service.retrieve_sop_evidence(_alarm())
    assert evidence == []
    assert status is RetrievalStatus.EMPTY


def test_excerpt_is_truncated():
    long_content = "x" * (_MAX_EXCERPT_CHARS + 100)
    docs = [_doc(long_content, {"_file_name": "big.md"})]
    service = SopRetrievalService(retriever=_FakeRetriever(docs))
    evidence, _ = service.retrieve_sop_evidence(_alarm())
    assert len(evidence[0].excerpt) == _MAX_EXCERPT_CHARS


def test_custom_top_k_passed_through():
    retriever = _FakeRetriever([])
    service = SopRetrievalService(retriever=retriever, top_k=7)
    service.retrieve_sop_evidence(_alarm())
    assert retriever.last_top_k == 7
