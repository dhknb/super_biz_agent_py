"""Multi-Query 检索器 —— 把线上 rag_v2 的多查询召回搬进离线评测。

## 为什么需要这个文件

run_eval.py 里三个真实检索器(vector / rerank / local-rerank)都是
**拿原始 question 直接查一次**:

    docs = self._service.retrieve_documents(query, top_k=candidate_k, ...)

而线上 chat_v2 跑的是 `rewrite → Send fan-out → dedup`
(见 app/agent/rag_v2/graph.py:43-48)。

也就是说改造前的全部历史报告(baseline-real / rerank-v1 / local-bge-rerank-*)
测的都是**单查询召回**,线上真正在跑的多查询召回一次都没被量化过。
评测链路与线上链路不是同一条,指标再漂亮也不能用来判断线上好坏 ——
这是本模块要消灭的那个谎。

## 为什么复用 rewrite_node 而不是在这里重写一遍改写逻辑

改写的行为由三样东西决定:prompt、temperature、子查询个数。
如果这里抄一份,那么某天有人调了 NUM_SUB_QUERIES 或改了 prompt,
评测测的就不再是线上那条链路了 —— 而且**没有任何地方会报错**,
报告照样生成,只是从此开始说谎。直接 import 线上节点,
线上改了评测自动跟着改,这个偏差在物理上不可能发生(DRY)。

代价是评测需要 LLM 可用(改写要调一次模型)。这个代价是必须付的:
不调模型就测不出多查询,测不出多查询就等于没测线上。

## 为什么用 RRF 融合而不是简单拼接去重

多查询最有价值的信号是「**被多个子查询同时召回**」—— 这类文档几乎
必然真相关。简单拼接去重会把这个信号整个丢掉:先到先得,
一条只被 1 个子查询以第 1 名召回的文档,会排在
被 3 个子查询分别以第 2 名召回的文档前面。

RRF(Reciprocal Rank Fusion)给每个文档累加 `1 / (k + rank)`:

- 多路命中 → 多个加数 → 自然上浮,正是我们要的信号
- 只看名次不看分数 → 不要求各路分数可比。这一点很关键:
  向量距离和 BM25 分数本来就不在同一个量纲上,任何「把分数加起来」
  的融合都需要先归一化,而归一化又依赖分数分布假设。RRF 绕开了整个问题。

k=60 取自 RRF 原论文(Cormack et al. 2009),也是 Elasticsearch /
Milvus 的默认值。它的作用是压平头部差距:rank1 与 rank2 的差
(1/61 - 1/62 ≈ 0.00026) 远小于不加 k 时的 (1/1 - 1/2 = 0.5),
避免单路的第一名一票定音。

## 与线上 dedup_node 的差别(有意为之)

线上 dedup_node 走的是「按内容指纹去重 + 截断到 FINAL_TOP_K」,
没有跨子查询的名次融合 —— 因为线上把排序责任交给了下游 rerank。
这里做 RRF 是为了**在没有 rerank 的配置下也能公平地测出多查询本身的收益**:
如果只做去重拼接,测出来的就是「多查询 + 一个很差的排序器」,
分不清是多查询没用还是排序没用。

顺带一说,这个差异本身就是一条值得跟进的线索:如果评测显示
RRF 融合明显优于拼接去重,那线上 dedup_node 也该补上 RRF。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Sequence

from langchain_core.documents import Document

RRF_K = 60
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# 仓库内的本地权重目录(2.2G,含 model.safetensors)。
# 仓库里另有 .hf_cache/ 和 ~/.cache/huggingface/,但那两处只有 config.json 与 refs,
# 没有权重文件 —— 离线模式下按 hub id 加载必然失败。所以优先认这个目录。
LOCAL_RERANK_DIR = Path(__file__).resolve().parents[2] / "models" / "bge-reranker-v2-m3"

# rerank 时喂给 CrossEncoder 的正文长度,与 run_eval.py 既有 rerank 路径保持一致。
# 不一致的话两份报告就不可比 —— 截断长度直接影响 CrossEncoder 的判断。
#
# 保留 500 是为了让「文档级 rerank」这个旧口径可以原样复现(B/D 两组的
# 历史数字才有意义)。chunk 级 rerank 走下面那个更宽的值。
RERANK_PASSAGE_CHARS = 500

# chunk 级 rerank 的截断长度。
#
# 实测本库 chunk 正文 min=172 / p50=560 / max=1469 字,按 500 截会切掉 21/30 个
# 候选的后半段。bge-reranker-v2-m3 的窗口是 8192 token,2000 字远在窗口内,
# 所以这个截断实际上只是一道防御性上限,正常语料不会被切到。
RERANK_CHUNK_CHARS = 2000


# =============================================================================
# CrossEncoder 懒加载(进程内共享)
# =============================================================================

_cross_encoder_cache: dict[str, Any] = {}


def resolve_rerank_path(model_name: str = RERANK_MODEL_NAME) -> str:
    """把 hub id 解析成本地权重目录;本地没有才退回 hub id。

    判据是 model.safetensors 是否存在,而不是目录是否存在 —— 空壳目录
    (只有 config.json)恰恰是这次踩的坑,存在性检查会被它骗过去。
    """
    if model_name == RERANK_MODEL_NAME and (LOCAL_RERANK_DIR / "model.safetensors").is_file():
        return str(LOCAL_RERANK_DIR)
    return model_name


def load_cross_encoder(model_name: str = RERANK_MODEL_NAME) -> Any:
    """懒加载 CrossEncoder,并在进程内缓存。

    为什么要缓存在模块级而不是实例级:一次 `--compare` 式的实验里可能
    同时构造 local-rerank 和 multi-query-rerank 两个检索器,实例级缓存
    会把这个 2GB 出头的模型加载两遍。更重要的是,两者必须是**同一个模型**,
    否则报告之间不可比 —— 共享缓存让这件事由结构保证,而不是靠人记得。

    sentence_transformers 是重依赖,只在真正要 rerank 时才 import。
    """
    if model_name not in _cross_encoder_cache:
        from loguru import logger
        from sentence_transformers import CrossEncoder

        path = resolve_rerank_path(model_name)
        logger.info(f"加载本地 Rerank 模型: {path}")
        _cross_encoder_cache[model_name] = CrossEncoder(path)
    return _cross_encoder_cache[model_name]


# =============================================================================
# doc_id 推导
# =============================================================================

def doc_id_of(doc: Document) -> str:
    """从 chunk 的 metadata 推出文档级 id(去掉扩展名的文件名)。

    抽成函数是因为 run_eval.py 里这段表达式原本抄了三份
    (vector / rerank / local-rerank 各一份)。三份抄写在这里不只是丑 ——
    评测报告之间可比的前提是**所有检索器用同一套 doc_id 口径**,
    一旦某份抄写发生漂移,历史报告的对比会全部静默失效,
    而没有任何断言会失败。收敛到一处之后这种漂移不可能发生。
    """
    md = doc.metadata or {}
    file_name = md.get("_file_name") or ""
    return file_name.rsplit(".", 1)[0] if "." in file_name else file_name


# =============================================================================
# RRF 融合(纯函数,可单测)
# =============================================================================

def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[str]:
    """把多条有序 id 列表用 Reciprocal Rank Fusion 融合成一条。

    Args:
        rankings: 每个元素是一条按相关度降序排列的 id 列表(一个子查询的结果)
        k: RRF 平滑常数,越大越压平头部差距

    Returns:
        融合后按得分降序的 id 列表,**确定性**排序。

    确定性很重要:评测要能复现。得分相同时的比较键依次是
    `(-score, 最好名次, 首次出现顺序)` —— 前两个是语义上的合理偏好
    (同分则取在某一路里排得更前的),最后一个纯粹是为了消除
    dict 迭代顺序带来的不确定性,让同一份输入永远得到同一份输出。

    同一条 ranking 内部会先去重:一个子查询不该给同一个文档投两票,
    否则「一个子查询召回同一文档的 3 个 chunk」会被误当成
    「3 个子查询都认可这个文档」,而这两件事的证据强度完全不同。
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    order = 0

    for ranking in rankings:
        seen_here: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            if not item or item in seen_here:
                continue
            seen_here.add(item)

            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            if item not in best_rank or rank < best_rank[item]:
                best_rank[item] = rank
            if item not in first_seen:
                first_seen[item] = order
                order += 1

    return sorted(
        scores,
        key=lambda item: (-scores[item], best_rank[item], first_seen[item]),
    )


# =============================================================================
# chunk 级打分 → 文档级名次(max-pool)
# =============================================================================

def max_pool_to_docs(
    scored_chunks: Sequence[tuple[str, float]],
    tie_break_order: Sequence[str] | None = None,
) -> list[str]:
    """把 (doc_id, chunk 得分) 列表按「文档取其最高分 chunk」聚合成文档名次。

    Args:
        scored_chunks: 每个元素是 (doc_id, 该 chunk 的 CrossEncoder 得分)。
            同一个 doc_id 会出现多次 —— 一篇文档命中几个 chunk 就几次。
        tie_break_order: 同分时的兜底顺序(通常传 RRF 融合后的名次)。
            不在这个列表里的 doc_id 排在所有列内元素之后。

    Returns:
        按文档最高分降序的 doc_id 列表,确定性排序。

    ## 为什么是 max 而不是 mean

    一篇运维文档里,「告警名 + 触发条件」那一段和「排查步骤」那一段的
    相关度天差地别。用 mean 的话,一篇**恰好有一段极其对题**的文档会被
    它自己其余几段样板文字拉低,而一篇通篇不痛不痒的文档反而更占优 ——
    这与「用户要找的是能回答问题的那一段」正好相反。

    max 的语义是「这篇文档里最能回答问题的那一段有多好」,与检索目标一致。
    这也是 ColBERT / 多向量检索里 MaxSim 的同一个思路。

    ## 为什么需要 tie_break_order

    CrossEncoder 对样板文字会给出**完全相同**的得分(本库 12 组文档的
    开头 60 字一字不差)。同分时若靠 dict 迭代顺序决定名次,同一份输入
    会产出不同报告,评测就不可复现了。传入 RRF 名次做兜底,既确定
    又保留了「多路召回」这个先验。
    """
    order_index = {doc_id: i for i, doc_id in enumerate(tie_break_order or [])}
    fallback = len(order_index)

    best: dict[str, float] = {}
    for doc_id, score in scored_chunks:
        if not doc_id:
            continue
        # float() 是必需的:CrossEncoder.predict 返回 numpy.float32,
        # 直接比较能work,但落进报告 JSON 时不可序列化。
        value = float(score)
        if doc_id not in best or value > best[doc_id]:
            best[doc_id] = value

    return sorted(
        best,
        key=lambda doc_id: (-best[doc_id], order_index.get(doc_id, fallback), doc_id),
    )


# =============================================================================
# 同步桥:评测是同步 CLI,rewrite_node 是 async
# =============================================================================

def _run_sync(coro: Any) -> Any:
    """在无事件循环的同步上下文里跑一个协程。

    不用 `asyncio.get_event_loop().run_until_complete`:那个 API 在
    Python 3.12+ 已弃用,且在有循环运行时行为诡异。这里显式检测,
    有循环就直接报错 —— 与其静默走进未定义行为,不如让调用方知道用错了。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "MultiQueryRetriever 是同步接口,不能在已有事件循环中调用"
    )


# =============================================================================
# Multi-Query 检索器
# =============================================================================

class MultiQueryRetriever:
    """rewrite → 每个子查询各查一次 → RRF 融合 →(可选)rerank。

    与 run_eval.py 里单查询检索器的**唯一**差别就是前两步。
    per-query 的 candidate_k、权重、doc_id 口径全部保持一致,
    这样两份报告的差值才能归因到「多查询」本身,而不是别的变量。
    """

    def __init__(
        self,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
        doc_id_filter: Any | None = None,
        rrf_k: int = RRF_K,
        rerank: bool = False,
        rerank_model: str = RERANK_MODEL_NAME,
        rerank_granularity: str = "chunk",
    ) -> None:
        from app.services.vector_search_service import vector_search_service

        if rerank_granularity not in ("chunk", "doc"):
            raise ValueError(
                f"rerank_granularity 只能是 'chunk' 或 'doc',收到 {rerank_granularity!r}"
            )

        self._service = vector_search_service
        self._vector_weight = vector_weight
        self._bm25_weight = bm25_weight
        self._filter = doc_id_filter
        self._rrf_k = rrf_k
        self._rerank = rerank
        self._rerank_model = rerank_model
        # 'doc'  = 旧口径:每篇文档只取首个命中 chunk 的前 500 字送去打分。
        # 'chunk'= 新口径:每个 chunk 各自打分,再 max-pool 到文档级。
        #
        # 保留 'doc' 不是为了给人选,而是为了让 B/D 两组的历史数字可以原样
        # 复现 —— 否则「新口径更好」这句话就失去了对照,只能靠记忆断言。
        self._rerank_granularity = rerank_granularity

        # 供 run_eval 落进报告用:这次 search 实际用了哪些子查询。
        # 没有它就没法解释某条样本为什么变好或变坏 —— 报告里只剩一个数字,
        # 而数字不能告诉你改写是否合理。
        self.last_sub_queries: list[str] = []
        # 改写降级(LLM 挂了 → 退回单查询)的样本数。
        #
        # 这个计数**必须**存在:rewrite_node 的降级路径会安静地返回
        # [question],于是整场评测退化成单查询,而报告标题仍然写着
        # multi-query。那就是一份说谎的报告 —— 比没有报告更糟。
        self.rewrite_degraded = 0

    # -- 子查询 -------------------------------------------------------------

    def _sub_queries(self, query: str) -> list[str]:
        """调线上 rewrite_node 拿子查询;降级时如实计数。"""
        from app.agent.rag_v2.nodes import rewrite_node

        patch = _run_sync(rewrite_node({"question": query}))
        subs = [s for s in (patch.get("sub_queries") or []) if s and s.strip()]
        if not subs:
            subs = [query]

        if patch.get("degrade_reasons"):
            self.rewrite_degraded += 1

        return subs

    # -- 单条子查询的候选 ---------------------------------------------------

    def _candidates(
        self, query: str, candidate_k: int
    ) -> tuple[list[str], list[str], dict[str, str], dict[str, tuple[str, str]]]:
        """查一条子查询。

        Returns:
            (doc_id 名次表, chunk_id 名次表, doc_id→正文, chunk_id→(doc_id, 正文))

        最后那个 chunk 表是 chunk 级 rerank 的输入。它与 doc 级 passages
        并存而不是取代它 —— 两个 rerank 口径要能在同一份代码里都跑出来,
        否则就没法证明新口径比旧口径好。
        """
        docs = self._service.retrieve_documents(
            query,
            top_k=candidate_k,
            vector_weight=self._vector_weight,
            bm25_weight=self._bm25_weight,
        )

        doc_ids: list[str] = []
        chunk_ids: list[str] = []
        passages: dict[str, str] = {}
        chunk_passages: dict[str, tuple[str, str]] = {}

        for doc in docs:
            doc_id = doc_id_of(doc)
            if self._filter and not self._filter.allow(doc_id):
                continue

            chunk_id = (doc.metadata or {}).get("chunk_id")
            if chunk_id:
                chunk_key = str(chunk_id)
                chunk_ids.append(chunk_key)
                # chunk 正文按更宽的上限截断:实测 p50=560 字,按 doc 级那个
                # 500 的口径会切掉多数候选的后半段,而模型窗口远没到。
                if chunk_key not in chunk_passages and doc_id:
                    chunk_passages[chunk_key] = (
                        doc_id,
                        doc.page_content[:RERANK_CHUNK_CHARS]
                        if doc.page_content
                        else doc_id,
                    )

            if doc_id:
                doc_ids.append(doc_id)
                if doc_id not in passages:
                    passages[doc_id] = (
                        doc.page_content[:RERANK_PASSAGE_CHARS]
                        if doc.page_content
                        else doc_id
                    )

        return doc_ids, chunk_ids, passages, chunk_passages

    # -- 主入口 -------------------------------------------------------------

    def search(self, query: str, top_k: int) -> tuple[list[str], list[str]]:
        # 与 VectorServiceRetriever.search 同一个公式,不能自作聪明改深 ——
        # 候选深度变了,对比就不再是「单查询 vs 多查询」而是混了两个变量。
        candidate_k = max(top_k * 3, 10)

        sub_queries = self._sub_queries(query)
        self.last_sub_queries = list(sub_queries)

        doc_rankings: list[list[str]] = []
        chunk_rankings: list[list[str]] = []
        passages: dict[str, str] = {}
        chunk_passages: dict[str, tuple[str, str]] = {}

        for sub_query in sub_queries:
            doc_ids, chunk_ids, sub_passages, sub_chunks = self._candidates(
                sub_query, candidate_k
            )
            doc_rankings.append(doc_ids)
            chunk_rankings.append(chunk_ids)
            for doc_id, passage in sub_passages.items():
                passages.setdefault(doc_id, passage)
            for chunk_id, pair in sub_chunks.items():
                chunk_passages.setdefault(chunk_id, pair)

        fused_docs = rrf_fuse(doc_rankings, k=self._rrf_k)
        fused_chunks = rrf_fuse(chunk_rankings, k=self._rrf_k)

        if self._rerank:
            if self._rerank_granularity == "chunk":
                fused_docs = self._rerank_by_chunk(query, fused_chunks, chunk_passages)
            else:
                fused_docs = self._rerank_docs(query, fused_docs, passages)
            # rerank 只重排了 doc 级候选,chunk 级名次已经与之不一致了。
            # 返回旧的 chunk 名次会让报告里出现「doc 和 chunk 指标来自
            # 两套不同排序」的隐性矛盾 —— 与 run_eval.py 既有 rerank 路径
            # 一样,如实返回空,让 metrics 退化到 doc 级评测。
            fused_chunks = []

        return fused_docs[:top_k], fused_chunks[:top_k]

    def _rerank_by_chunk(
        self,
        query: str,
        chunk_ids: list[str],
        chunk_passages: dict[str, tuple[str, str]],
    ) -> list[str]:
        """chunk 级打分 → max-pool 成文档级名次;失败则如实降级为 RRF 名次。

        这是为了修掉 doc 级 rerank 的两处信息损失(见 max_pool_to_docs
        的注释):只看每篇文档的第一个命中 chunk、且按 500 字截断。
        """
        # 每个待打分 chunk 记住它属于哪篇文档 —— max-pool 时要把分挂回文档上。
        owner_docs: list[str] = []
        pairs: list[tuple[str, str]] = []
        for chunk_id in chunk_ids:
            entry = chunk_passages.get(chunk_id)
            if not entry:
                continue
            doc_id, passage = entry
            owner_docs.append(doc_id)
            # 同 _rerank_docs:用**原始问题**而不是子查询打分。
            pairs.append((query, passage))

        # 降级时要还原成 RRF 的**文档级**名次,而不是空手而归。
        # chunk 名次里同一篇文档会出现多次,这里按首次出现压平。
        def _rrf_doc_fallback() -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for chunk_id in chunk_ids:
                entry = chunk_passages.get(chunk_id)
                if not entry:
                    continue
                doc_id = entry[0]
                if doc_id not in seen:
                    seen.add(doc_id)
                    out.append(doc_id)
            return out

        if len(pairs) <= 1:
            return _rrf_doc_fallback()

        try:
            model = load_cross_encoder(self._rerank_model)
            scores = model.predict(pairs, show_progress_bar=False)
        except Exception:
            import traceback

            from loguru import logger

            logger.warning(f"Chunk 级 Rerank 失败,降级为 RRF 名次: {traceback.format_exc()}")
            return _rrf_doc_fallback()

        # max_pool_to_docs 只需要 (doc_id, 得分):chunk_id 到这一步已经用完了,
        # 它的作用是在上面把得分挂回正确的文档上。
        scored = [
            (doc_id, float(score)) for doc_id, score in zip(owner_docs, scores)
        ]
        # tie_break_order 传 RRF 的文档级名次:CrossEncoder 对样板文字会给出
        # 完全相同的分,没有兜底顺序的话同一份输入会产出不同报告。
        return max_pool_to_docs(scored, tie_break_order=_rrf_doc_fallback())

    def _rerank_docs(
        self, query: str, doc_ids: list[str], passages: dict[str, str]
    ) -> list[str]:
        """用 CrossEncoder 对融合后的候选重排;失败则如实降级为 RRF 名次。"""
        if len(doc_ids) <= 1:
            return doc_ids

        # 注意用**原始问题**而不是子查询做 rerank 的 query:
        # 子查询是为了扩大召回面而故意偏移过的,用它打分会把
        # 「与某个侧面最贴」的文档排到前面,而评测要问的是
        # 「与用户真正的问题最贴」。
        pairs = [(query, passages.get(doc_id, doc_id)) for doc_id in doc_ids]

        try:
            model = load_cross_encoder(self._rerank_model)
            scores = model.predict(pairs, show_progress_bar=False)
        except Exception:
            import traceback

            from loguru import logger

            logger.warning(f"Rerank 失败,降级为 RRF 名次: {traceback.format_exc()}")
            return doc_ids

        ranked = sorted(zip(doc_ids, scores), key=lambda pair: pair[1], reverse=True)
        return [doc_id for doc_id, _ in ranked]


def build_multi_query_retriever(
    vector_weight: float | None = None,
    bm25_weight: float | None = None,
    doc_id_filter: Any | None = None,
    rrf_k: int = RRF_K,
    rerank: bool = False,
    rerank_granularity: str = "chunk",
) -> MultiQueryRetriever:
    return MultiQueryRetriever(
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        doc_id_filter=doc_id_filter,
        rrf_k=rrf_k,
        rerank=rerank,
        rerank_granularity=rerank_granularity,
    )


__all__ = [
    "RERANK_CHUNK_CHARS",
    "RERANK_MODEL_NAME",
    "RERANK_PASSAGE_CHARS",
    "RRF_K",
    "MultiQueryRetriever",
    "build_multi_query_retriever",
    "doc_id_of",
    "load_cross_encoder",
    "max_pool_to_docs",
    "resolve_rerank_path",
    "rrf_fuse",
]
