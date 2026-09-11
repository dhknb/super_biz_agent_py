"""Multi-Query 评测检索器单元测试。

只测**不需要 Milvus 和 LLM** 的那部分:RRF 融合与 doc_id 推导。
检索与改写有各自的集成路径,不在这里重复。

为什么 RRF 值得单测:它是本次改动里唯一的**新算法**,而且错了不会报错 ——
只会安静地给出一份排序略有偏差的报告,然后所有基于报告的决策都建立在
这个偏差上。没有断言的算法等于没有算法。
"""

from __future__ import annotations

from langchain_core.documents import Document

from tests.eval.multi_query import (
    MultiQueryRetriever,
    doc_id_of,
    max_pool_to_docs,
    rrf_fuse,
)
# run_eval 的模块级 import 是轻的(argparse/json + 本目录两个模块);
# Milvus 与 DashScope 都在 LocalRerankedRetriever.__init__ 里才 import,
# 而下面用 __new__ 跳过了 __init__,所以这条 import 不会连任何外部服务。
from tests.eval.run_eval import DocIdFilter, LocalRerankedRetriever


# ---------------------------------------------------------------------------
# RRF 融合
# ---------------------------------------------------------------------------

def test_single_ranking_preserves_order() -> None:
    """单路输入时,RRF 不该改变原有名次。"""
    assert rrf_fuse([["a", "b", "c"]]) == ["a", "b", "c"]


def test_multi_hit_beats_single_top_hit() -> None:
    """这是整个 RRF 的立论点:多路共识 > 单路第一名。

    b 在两路里都是第 2 名,a 只在一路里是第 1 名。
    简单拼接去重会输出 [a, b];RRF 必须把 b 排到 a 前面 ——
    「两个不同角度都召回了它」是比「一个角度排第一」更强的证据。
    """
    fused = rrf_fuse([["a", "b"], ["c", "b"]])
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c"}


def test_duplicate_within_one_ranking_counts_once() -> None:
    """同一路里重复出现的 id 只投一票。

    一条子查询召回同一文档的 3 个 chunk,不等于 3 个子查询都认可它。
    不去重的话 a 会拿到 3 份加分,凭一路之力压过真正的多路共识项 b。
    """
    fused = rrf_fuse([["a", "a", "a"], ["b", "x"], ["b", "y"]])
    assert fused[0] == "b"


def test_empty_and_blank_ids_ignored() -> None:
    """空字符串不该占据一个名次 —— doc_id 推导失败时会产生空串。"""
    assert rrf_fuse([["", "a", ""], ["a"]]) == ["a"]


def test_empty_input_returns_empty() -> None:
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_deterministic_across_calls() -> None:
    """同一份输入必须永远得到同一份输出,否则评测不可复现。"""
    rankings = [["a", "b", "c"], ["c", "a"], ["b", "c"]]
    first = rrf_fuse(rankings)
    for _ in range(20):
        assert rrf_fuse(rankings) == first


def test_tie_broken_by_best_rank() -> None:
    """同分时,在某一路里排得更前的胜出。

    a 和 b 各只被一路召回一次,得分相同(1/(60+1) vs 1/(60+2) 不同,
    所以这里用同名次构造真正的平分)。
    """
    # a 在第一路第 1 名;b 在第二路第 1 名 —— 完全同分。
    # 平分时按首次出现顺序,a 先。
    assert rrf_fuse([["a"], ["b"]]) == ["a", "b"]


def test_k_controls_head_flattening() -> None:
    """k 越小,头部名次的权重差越大。

    k=0 时 rank1 的分是 1.0、rank2 是 0.5 —— 单路第一名足以压过双路共识。
    这正是 RRF 论文取 k=60 的理由,这个测试把那个理由钉住:
    如果有人把 RRF_K 改小,test_multi_hit_beats_single_top_hit 会开始失败。
    """
    rankings = [["a", "b"], ["c", "b"]]
    assert rrf_fuse(rankings, k=60)[0] == "b"   # 共识胜
    assert rrf_fuse(rankings, k=0)[0] == "a"    # 头部第一名胜


# ---------------------------------------------------------------------------
# doc_id 推导
# ---------------------------------------------------------------------------

def test_doc_id_strips_extension() -> None:
    doc = Document(page_content="x", metadata={"_file_name": "cpu_high_usage.md"})
    assert doc_id_of(doc) == "cpu_high_usage"


def test_doc_id_without_extension_kept_as_is() -> None:
    doc = Document(page_content="x", metadata={"_file_name": "cpu_high_usage"})
    assert doc_id_of(doc) == "cpu_high_usage"


def test_doc_id_keeps_only_last_extension() -> None:
    """rsplit 保证 `a.b.md` 只去掉 `.md`,不会把文件名切碎。"""
    doc = Document(page_content="x", metadata={"_file_name": "high.api.latency.md"})
    assert doc_id_of(doc) == "high.api.latency"


def test_doc_id_missing_metadata_returns_empty() -> None:
    """metadata 缺失时返回空串,由调用方过滤 —— 不猜、不编。"""
    assert doc_id_of(Document(page_content="x", metadata={})) == ""
    assert doc_id_of(Document(page_content="x")) == ""


# ---------------------------------------------------------------------------
# chunk 级得分 → 文档级名次(max-pool)
# ---------------------------------------------------------------------------

def test_max_pool_takes_document_best_chunk() -> None:
    """文档取其最高分 chunk,不是平均分。

    这是 chunk 级 rerank 的核心语义:一篇文档只要有**一段**极其对题,
    就该排在前面,不该被它自己其余几段样板文字拉低。
    """
    scored = [
        ("cpu_high_usage", 0.1),   # 样板段
        ("cpu_high_usage", 0.9),   # 标题+告警名段,对题
        ("slow_response", 0.5),
        ("slow_response", 0.5),
    ]
    # 用 mean 的话 cpu(0.5) 与 slow(0.5) 同分,max 才能让 cpu 胜出
    assert max_pool_to_docs(scored) == ["cpu_high_usage", "slow_response"]


def test_max_pool_deduplicates_documents() -> None:
    """同一篇文档命中多个 chunk,输出里只出现一次。"""
    scored = [("a", 0.3), ("a", 0.7), ("a", 0.5), ("b", 0.6)]
    assert max_pool_to_docs(scored) == ["a", "b"]


def test_max_pool_tie_broken_by_given_order() -> None:
    """同分时用传入的 RRF 名次兜底,而不是 dict 顺序。

    CrossEncoder 对样板文字会给出完全相同的分(本库实测 12 组文档开头
    一字不差),没有兜底顺序评测就不可复现。
    """
    scored = [("x", 0.5), ("y", 0.5), ("z", 0.5)]
    assert max_pool_to_docs(scored, tie_break_order=["z", "y", "x"]) == ["z", "y", "x"]
    assert max_pool_to_docs(scored, tie_break_order=["y", "x", "z"]) == ["y", "x", "z"]


def test_max_pool_unlisted_docs_go_last() -> None:
    """不在 tie_break_order 里的文档排在列内元素之后,且彼此按 doc_id 稳定排序。"""
    scored = [("known", 0.5), ("ghost", 0.5), ("another", 0.5)]
    out = max_pool_to_docs(scored, tie_break_order=["known"])
    assert out[0] == "known"
    assert out[1:] == ["another", "ghost"]


def test_max_pool_score_beats_tie_break_order() -> None:
    """得分优先级高于兜底顺序 —— 兜底只在同分时起作用。"""
    scored = [("low", 0.1), ("high", 0.9)]
    assert max_pool_to_docs(scored, tie_break_order=["low", "high"]) == ["high", "low"]


def test_max_pool_skips_empty_doc_ids() -> None:
    """空 doc_id 直接丢弃,不参与排序 —— 与 doc_id_of 返回空串的约定配套。"""
    assert max_pool_to_docs([("", 0.9), ("a", 0.1)]) == ["a"]


def test_max_pool_empty_input_returns_empty() -> None:
    assert max_pool_to_docs([]) == []


def test_max_pool_deterministic_across_calls() -> None:
    """同一份输入必须永远得到同一份输出,否则报告不可复现。"""
    scored = [("a", 0.5), ("b", 0.5), ("c", 0.5), ("d", 0.5)]
    first = max_pool_to_docs(scored, tie_break_order=["c", "a"])
    for _ in range(20):
        assert max_pool_to_docs(scored, tie_break_order=["c", "a"]) == first


# ---------------------------------------------------------------------------
# chunk 级 rerank 装配层
#
# 上面那批测的是纯函数 max_pool_to_docs,下面这批测的是把它接到
# CrossEncoder 上的那段胶水代码(_rerank_by_chunk)。
#
# 为什么必须单独测:纯函数全绿的情况下,胶水层写错一个变量名
# (实际发生过:zip 里引用了已删除的 pairs_meta)依然能通过编译,
# 要等真跑 82 条样本、加载完 2.2G 模型才炸。装配层有了假模型之后,
# 这类错误在毫秒级就能暴露。
#
# 用 __new__ 绕过 __init__:__init__ 会 import vector_search_service,
# 那会拖进 Milvus 连接和 DashScope 客户端。这里要测的只是排序装配,
# 不该依赖任何外部服务。
# ---------------------------------------------------------------------------

class _FakeCrossEncoder:
    """按 passage 正文查表打分的假模型,让期望值可以手写。"""

    def __init__(self, score_by_passage: dict[str, float]) -> None:
        self._scores = score_by_passage
        self.call_count = 0
        # 记下最后一次收到多少个 pair:去重是否生效只能从这里看出来,
        # 结果名次对重复打分是不敏感的(max-pool 会把重复分吸收掉)。
        self.last_pair_count: int | None = None

    def predict(self, pairs, show_progress_bar=False):  # noqa: ANN001, ARG002
        self.call_count += 1
        pairs = list(pairs)
        self.last_pair_count = len(pairs)
        return [self._scores.get(passage, 0.0) for _query, passage in pairs]


class _BoomCrossEncoder:
    def predict(self, pairs, show_progress_bar=False):  # noqa: ANN001, ARG002
        raise RuntimeError("模型炸了")


def _make_retriever(monkeypatch, model) -> MultiQueryRetriever:
    retriever = MultiQueryRetriever.__new__(MultiQueryRetriever)
    retriever._rerank_model = "fake-model"
    monkeypatch.setattr(
        "tests.eval.multi_query.load_cross_encoder", lambda _name: model
    )
    return retriever


def test_rerank_by_chunk_scores_are_pooled_onto_owner_docs(monkeypatch) -> None:
    """得分必须挂回**正确的**文档 —— 这条能抓住 zip 错位和变量名写错。

    构造:cpu 文档的第 2 个 chunk(标题段)得分最高,而它在 RRF
    chunk 名次里排最后。正确实现应让 cpu 排第一;若得分挂错文档,
    或 zip 的一侧引用了错误的变量,顺序就会不同。
    """
    model = _FakeCrossEncoder({
        "cpu 样板段": 0.10,
        "cpu 标题段 HighCPUUsage": 0.95,
        "slow 样板段": 0.50,
    })
    retriever = _make_retriever(monkeypatch, model)

    chunk_ids = ["c-slow-1", "c-cpu-1", "c-cpu-2"]
    chunk_passages = {
        "c-slow-1": ("slow_response", "slow 样板段"),
        "c-cpu-1": ("cpu_high_usage", "cpu 样板段"),
        "c-cpu-2": ("cpu_high_usage", "cpu 标题段 HighCPUUsage"),
    }

    out = retriever._rerank_by_chunk("CPU 高怎么办", chunk_ids, chunk_passages)

    assert out == ["cpu_high_usage", "slow_response"]
    assert model.call_count == 1


def test_rerank_by_chunk_falls_back_to_rrf_doc_order_on_model_failure(
    monkeypatch,
) -> None:
    """模型挂了要退回 RRF 的**文档级**名次,而不是空手而归。

    返回空列表会让报告显示 Hit@K=0,看起来像检索失败,
    实际只是精排不可用 —— 那是在用一个假的坏消息盖住真实情况。
    """
    retriever = _make_retriever(monkeypatch, _BoomCrossEncoder())

    chunk_ids = ["c-b-1", "c-a-1", "c-a-2", "c-b-2"]
    chunk_passages = {
        "c-b-1": ("doc_b", "b1"),
        "c-a-1": ("doc_a", "a1"),
        "c-a-2": ("doc_a", "a2"),
        "c-b-2": ("doc_b", "b2"),
    }

    out = retriever._rerank_by_chunk("q", chunk_ids, chunk_passages)

    # 按 chunk 名次里的首次出现压平,且每篇文档只留一次
    assert out == ["doc_b", "doc_a"]


def test_rerank_by_chunk_skips_chunk_ids_without_passage(monkeypatch) -> None:
    """chunk 名次里有、但正文表里没有的 id 直接跳过,不能崩也不能编。"""
    model = _FakeCrossEncoder({"a1": 0.2, "b1": 0.9})
    retriever = _make_retriever(monkeypatch, model)

    chunk_ids = ["ghost", "c-a-1", "missing", "c-b-1"]
    chunk_passages = {
        "c-a-1": ("doc_a", "a1"),
        "c-b-1": ("doc_b", "b1"),
    }

    out = retriever._rerank_by_chunk("q", chunk_ids, chunk_passages)

    assert out == ["doc_b", "doc_a"]


def test_rerank_by_chunk_single_candidate_skips_model(monkeypatch) -> None:
    """只有一个候选时没有可排的东西,不该白加载一次 2.2G 模型。"""
    model = _FakeCrossEncoder({"only": 0.5})
    retriever = _make_retriever(monkeypatch, model)

    out = retriever._rerank_by_chunk(
        "q", ["c-1"], {"c-1": ("doc_only", "only")}
    )

    assert out == ["doc_only"]
    assert model.call_count == 0


def test_rerank_by_chunk_empty_input_returns_empty(monkeypatch) -> None:
    retriever = _make_retriever(monkeypatch, _FakeCrossEncoder({}))
    assert retriever._rerank_by_chunk("q", [], {}) == []


# ---------------------------------------------------------------------------
# 单查询 + chunk 级 rerank(LocalRerankedRetriever)
#
# 与上面那批测的是同一个修复,但落在另一条代码路径上:multi-query 那条先
# RRF 融合再打分,这条直接拿粗排候选打分。两条路径各有一份 zip 装配,
# 各自都能静默错位,所以两边都要有断言。
#
# 这批测试的另一个作用是**钉住 doc 级口径不变**:B3 组(doc 级)的历史
# 数字是对照基准,如果重写 search 时顺手改了 doc 级的行为,F 组与 B3 的
# 差异就不再只来自「打分粒度」这一个变量,对照实验也就不成立了。
# ---------------------------------------------------------------------------

def _doc(file_name: str, chunk_id: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={"_file_name": file_name, "chunk_id": chunk_id},
    )


class _FakeSearchService:
    """替掉 vector_search_service,固定返回一份粗排候选。"""

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs
        self.last_top_k: int | None = None

    def retrieve_documents(self, query, top_k=None, vector_weight=None, bm25_weight=None):  # noqa: ANN001, ARG002
        self.last_top_k = top_k
        return self._docs


def _make_local_retriever(
    docs: list[Document],
    model,
    granularity: str = "chunk",
    doc_filter=None,
) -> LocalRerankedRetriever:
    # 同样用 __new__ 绕过 __init__(它会 import vector_search_service)。
    retriever = LocalRerankedRetriever.__new__(LocalRerankedRetriever)
    retriever._service = _FakeSearchService(docs)
    retriever._vector_weight = 0.7
    retriever._bm25_weight = 0.3
    retriever._candidate_k = 30
    retriever._filter = doc_filter
    retriever._rerank_granularity = granularity
    # 直接塞 _model:_get_model 见到非 None 就不会去碰 load_cross_encoder,
    # 于是整条测试不依赖 2.2G 权重也不依赖 monkeypatch。
    retriever._model = model
    return retriever


# 一份共用的候选:cpu 文档有两个命中 chunk,其中**第二个**(标题段)才是
# 真正能区分它和其他告警文档的正文 —— 而 doc 级口径永远只看第一个。
_CANDIDATE_DOCS = [
    _doc("slow_response.md", "s1", "slow 样板段"),
    _doc("postgres_connection_failed.md", "p1", "postgres 样板段"),
    _doc("cpu_high_usage.md", "c1", "cpu 样板段"),
    _doc("cpu_high_usage.md", "c2", "cpu 标题段 HighCPUUsage"),
]

_CANDIDATE_SCORES = {
    "slow 样板段": 0.50,
    "postgres 样板段": 0.40,
    "cpu 样板段": 0.10,
    "cpu 标题段 HighCPUUsage": 0.95,
}


def test_local_rerank_chunk_granularity_pools_onto_owner_docs() -> None:
    """chunk 级:cpu 靠它的第二个 chunk 拿到第一名。

    这条同时验证两件事 —— 得分挂回了正确的文档(zip 没错位),
    以及非首个 chunk 真的参与了打分。
    """
    model = _FakeCrossEncoder(_CANDIDATE_SCORES)
    retriever = _make_local_retriever(_CANDIDATE_DOCS, model)

    doc_ids, chunk_ids = retriever.search("CPU 高怎么办", top_k=2)

    assert doc_ids == ["cpu_high_usage", "slow_response"]
    # rerank 之后 chunk 名次与 doc 名次不同源,如实返回空
    assert chunk_ids == []
    assert model.call_count == 1


def test_local_rerank_doc_granularity_keeps_first_chunk_behaviour() -> None:
    """doc 级:同一份输入必须给出**旧**答案,否则 B3 的基准就被动过了。

    cpu 的标题段得分最高,但 doc 级口径看不到它 —— 只看 cpu 的第一个
    chunk(0.10),于是 cpu 掉出前二。两组结论相反,正是这次要测的差异。
    """
    model = _FakeCrossEncoder(_CANDIDATE_SCORES)
    retriever = _make_local_retriever(_CANDIDATE_DOCS, model, granularity="doc")

    doc_ids, _ = retriever.search("CPU 高怎么办", top_k=2)

    assert doc_ids == ["slow_response", "postgres_connection_failed"]


def test_local_rerank_falls_back_to_coarse_order_on_model_failure() -> None:
    """模型炸了要退回粗排顺序,而不是返回空导致报告显示 Hit@K=0。"""
    retriever = _make_local_retriever(_CANDIDATE_DOCS, _BoomCrossEncoder())

    doc_ids, _ = retriever.search("q", top_k=2)

    assert doc_ids == ["slow_response", "postgres_connection_failed"]


def test_local_rerank_skips_model_when_candidates_fit_top_k() -> None:
    """候选数不超过 top_k 时无可重排,不该白跑一次模型。"""
    model = _FakeCrossEncoder(_CANDIDATE_SCORES)
    retriever = _make_local_retriever(_CANDIDATE_DOCS, model)

    doc_ids, _ = retriever.search("q", top_k=5)

    assert doc_ids == [
        "slow_response",
        "postgres_connection_failed",
        "cpu_high_usage",
    ]
    assert model.call_count == 0


def test_local_rerank_filter_also_drops_that_docs_chunks() -> None:
    """被 filter 挡掉的文档,它的 chunk 不能偷偷溜进打分输入。

    过滤发生在 chunk 收集之前。写反了的话,被排除的文档虽然进不了
    doc_passages,它的 chunk 却仍会被打分,并通过 max-pool 把自己
    重新加回结果 —— filter 形同失效,而没有任何报错。
    """
    # noise 段故意给**全场最高分**。如果只给它 0 分,那么即使过滤写反了、
    # 它的 chunk 混进了打分输入,它也排不进前二 —— 这条测试就会因为
    # 「分低」而不是「被过滤」通过,等于什么都没测。
    model = _FakeCrossEncoder({**_CANDIDATE_SCORES, "noise 段": 0.99})
    doc_filter = DocIdFilter(mode="deny", names=["noise"])
    docs = _CANDIDATE_DOCS + [_doc("noise.md", "n1", "noise 段")]
    retriever = _make_local_retriever(docs, model, doc_filter=doc_filter)

    doc_ids, _ = retriever.search("q", top_k=2)

    assert "noise" not in doc_ids
    assert doc_ids[0] == "cpu_high_usage"


def test_local_rerank_dedups_repeated_chunk_ids() -> None:
    """同一个 chunk_id 出现两次只打一次分,免得它的文档被重复计票。"""
    model = _FakeCrossEncoder(_CANDIDATE_SCORES)
    docs = _CANDIDATE_DOCS + [_doc("cpu_high_usage.md", "c2", "cpu 标题段 HighCPUUsage")]
    retriever = _make_local_retriever(docs, model)

    retriever.search("q", top_k=2)

    assert model.last_pair_count == 4


def test_local_rerank_no_candidates_returns_empty() -> None:
    model = _FakeCrossEncoder({})
    retriever = _make_local_retriever([], model)

    assert retriever.search("q", top_k=5) == ([], [])
    assert model.call_count == 0
