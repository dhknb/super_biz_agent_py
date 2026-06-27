# 计划 2：检索链路企业化实施文档

## 1. 目标

当前项目已经实现了混合检索：

- Milvus 向量检索。
- BM25 关键词检索。
- `EnsembleRetriever` 融合排序。

这已经比简单向量检索更接近真实 RAG。但企业级 RAG 还需要做到：

- 检索结果可解释。
- 回答必须带引用。
- 有相似度阈值和拒答策略。
- 支持 query rewrite。
- 支持 reranker 精排。
- 支持检索评测。
- 每次回答能记录召回文档、耗时、模型成本。

本计划的目标是把当前“能搜到”升级为“可解释、可调优、可评测、可追踪”。

## 2. 当前项目现状

重点文件：

- `app/services/vector_search_service.py`：混合检索主逻辑。
- `app/tools/knowledge_tool.py`：Agent 使用的知识库工具。
- `app/services/rag_agent_service.py`：Agent 编排和流式输出。
- `app/api/chat.py`：对话接口。

当前问题：

- 检索结果没有稳定的真实分数，`SearchResult.score` 当前用排序名次模拟。
- 回答接口没有返回 sources。
- Agent 回答时不强制引用来源。
- 没有低置信度拒答。
- 没有 reranker。
- 没有检索评测集，调参靠感觉。

## 3. 目标检索架构

推荐链路：

```text
用户问题
  -> 查询规范化 query normalize
  -> 查询改写 query rewrite
  -> 多路召回
       - 向量召回
       - BM25 召回
       - 可选：标题/文件名召回
  -> 候选合并与去重
  -> reranker 精排
  -> 阈值过滤
  -> 构造 RAG 上下文
  -> LLM 基于上下文回答
  -> 返回 answer + sources + confidence
```

学习阶段可以分三步做：

1. 先让接口返回引用。
2. 再加阈值和拒答。
3. 最后加 reranker 和评测。

## 4. 定义检索结果模型

建议新建文件：`app/models/retrieval.py`

```python
from pydantic import BaseModel, Field


class RetrievedSource(BaseModel):
    document_id: str | None = Field(None, description="文档 ID")
    chunk_id: str | None = Field(None, description="分片 ID")
    filename: str | None = Field(None, description="文件名")
    source: str | None = Field(None, description="来源路径")
    content: str = Field(..., description="命中的分片内容")
    score: float = Field(..., description="归一化相关性分数")
    rank: int = Field(..., description="排序名次")
    retrieval_type: str = Field("hybrid", description="召回类型")


class RetrievalResult(BaseModel):
    query: str
    rewritten_query: str | None = None
    sources: list[RetrievedSource]
    confidence: float
    elapsed_ms: int
```

这个模型是后续 API、日志、评测的公共语言。企业系统里非常重要的一点是：内部对象要稳定。

## 5. 改造 `VectorSearchService`

当前 `retrieve_documents()` 只返回 `Document`。建议新增一个企业级方法，不破坏旧方法。

在 `app/services/vector_search_service.py` 中增加：

```python
import time

from app.models.retrieval import RetrievalResult, RetrievedSource


class VectorSearchService:
    MIN_CONFIDENCE = 0.35

    def retrieve_with_sources(self, query: str, top_k: int = 5) -> RetrievalResult:
        start = time.perf_counter()

        docs = self.retrieve_documents(query=query, top_k=top_k)
        sources: list[RetrievedSource] = []

        for rank, doc in enumerate(docs, start=1):
            metadata = doc.metadata or {}
            score = self._rank_to_score(rank)
            sources.append(
                RetrievedSource(
                    document_id=metadata.get("document_id"),
                    chunk_id=metadata.get("chunk_id"),
                    filename=metadata.get("_file_name"),
                    source=metadata.get("_source"),
                    content=doc.page_content,
                    score=score,
                    rank=rank,
                    retrieval_type="hybrid",
                )
            )

        confidence = sources[0].score if sources else 0.0
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        return RetrievalResult(
            query=query,
            sources=sources,
            confidence=confidence,
            elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def _rank_to_score(rank: int) -> float:
        # 当前 EnsembleRetriever 不暴露融合分数，先用稳定的 rank score。
        # 后续接 reranker 后替换为 reranker score。
        return 1.0 / rank
```

为什么先不删除旧方法：

- `knowledge_tool.py` 可能还依赖旧接口。
- 测试已有 `retrieve_documents()` 和 `search_similar_documents()`。
- 企业改造要尽量兼容已有调用方。

## 6. 新增 RAG Answer 模型

建议修改 `app/models/response.py`，增加：

```python
from app.models.retrieval import RetrievedSource


class RagAnswerResponse(BaseModel):
    success: bool
    answer: str | None = None
    sources: list[RetrievedSource] = Field(default_factory=list)
    confidence: float = 0.0
    errorMessage: str | None = None
```

如果担心影响前端，可以先不改原 `/api/chat`，新增 `/api/chat_rag`。

## 7. 做一个非 Agent 的可控 RAG 服务

当前 `RagAgentService` 是 Agent 形态，适合工具调用，但企业问答主链路建议先做一个确定性更强的 `RagAnswerService`。

新增文件：`app/services/rag_answer_service.py`

```python
from textwrap import dedent

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen

from app.config import config
from app.models.retrieval import RetrievalResult
from app.services.vector_search_service import vector_search_service


class RagAnswerService:
    def __init__(self) -> None:
        self.model = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0.2,
            streaming=False,
        )

    async def answer(self, question: str, top_k: int = 5) -> dict:
        retrieval = vector_search_service.retrieve_with_sources(question, top_k=top_k)

        if not retrieval.sources or retrieval.confidence < 0.35:
            return {
                "success": True,
                "answer": "我没有在知识库中找到足够可靠的依据，暂时不能确定回答。",
                "sources": retrieval.sources,
                "confidence": retrieval.confidence,
            }

        prompt = self._build_prompt(question, retrieval)
        response = await self.model.ainvoke(
            [
                SystemMessage(content=self._system_prompt()),
                HumanMessage(content=prompt),
            ]
        )

        return {
            "success": True,
            "answer": response.content,
            "sources": retrieval.sources,
            "confidence": retrieval.confidence,
        }

    def _system_prompt(self) -> str:
        return dedent("""
            你是企业知识库问答助手。
            你必须只基于给定的知识库片段回答问题。
            如果片段中没有答案，请明确说不知道。
            回答中要用 [来源1]、[来源2] 这样的格式标注依据。
            不要编造来源，不要使用片段之外的信息。
        """).strip()

    def _build_prompt(self, question: str, retrieval: RetrievalResult) -> str:
        context_blocks = []
        for source in retrieval.sources:
            context_blocks.append(
                f"[来源{source.rank}] 文件: {source.filename}\\n{source.content}"
            )

        context = "\\n\\n".join(context_blocks)

        return dedent(f"""
            用户问题：
            {question}

            知识库片段：
            {context}

            请基于知识库片段回答，并标注引用来源。
        """).strip()


rag_answer_service = RagAnswerService()
```

为什么推荐先做非 Agent RAG：

- 容易评测。
- 容易控制引用。
- 容易做低置信度拒答。
- Agent 更适合复杂工具协作，不一定适合作为知识库问答唯一入口。

## 8. 新增 API：`/api/chat_rag`

在 `app/api/chat.py` 中增加：

```python
from app.services.rag_answer_service import rag_answer_service


@router.post("/chat_rag")
async def chat_rag(request: ChatRequest):
    try:
        result = await rag_answer_service.answer(request.question, top_k=config.rag_top_k)
        return {
            "code": 200,
            "message": "success",
            "data": result,
        }
    except Exception as e:
        logger.error(f"RAG 问答失败: {e}")
        return {
            "code": 500,
            "message": "error",
            "data": {
                "success": False,
                "answer": None,
                "sources": [],
                "confidence": 0.0,
                "errorMessage": str(e),
            },
        }
```

注意要导入：

```python
from app.config import config
```

测试请求：

```bash
curl -X POST "http://localhost:9900/api/chat_rag" \
  -H "Content-Type: application/json" \
  -d '{"Id":"s1","Question":"CPU 使用率过高应该怎么排查？"}'
```

期望返回：

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "success": true,
    "answer": "... [来源1] ...",
    "sources": [
      {
        "filename": "cpu_high_usage.md",
        "score": 1.0,
        "rank": 1
      }
    ],
    "confidence": 1.0
  }
}
```

## 9. 改造 `knowledge_tool.py`

如果继续使用 Agent 工具，建议让工具也输出来源。

示例：

```python
from langchain_core.tools import tool

from app.services.vector_search_service import vector_search_service


@tool
def retrieve_knowledge(query: str) -> str:
    """从企业知识库检索相关信息，返回片段和来源。"""
    result = vector_search_service.retrieve_with_sources(query, top_k=3)
    if not result.sources:
        return "知识库中没有找到相关内容。"

    blocks = []
    for source in result.sources:
        blocks.append(
            f"[来源{source.rank}] 文件: {source.filename}\\n"
            f"相关度: {source.score:.2f}\\n"
            f"{source.content}"
        )
    return "\\n\\n".join(blocks)
```

然后修改 `RagAgentService._build_system_prompt()`：

```python
回答知识库问题时，必须优先调用 retrieve_knowledge。
如果工具返回来源，请在回答中标注 [来源1]、[来源2]。
如果知识库没有相关内容，请明确说明无法从知识库确认。
```

## 10. 增加 Query Rewrite

很多企业问题表达不稳定，例如：

- “机器负载高咋办”
- “CPU 爆了”
- “服务卡住”

这些都可能对应同一类知识。可以先做轻量 query rewrite。

新增文件：`app/services/query_rewrite_service.py`

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen

from app.config import config


class QueryRewriteService:
    def __init__(self) -> None:
        self.model = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0,
            streaming=False,
        )

    async def rewrite(self, query: str) -> str:
        response = await self.model.ainvoke(
            [
                SystemMessage(content="你负责把用户问题改写成适合企业知识库检索的短查询。只输出改写后的查询。"),
                HumanMessage(content=query),
            ]
        )
        rewritten = str(response.content).strip()
        return rewritten or query


query_rewrite_service = QueryRewriteService()
```

在 `RagAnswerService.answer()` 里改：

```python
rewritten_query = await query_rewrite_service.rewrite(question)
retrieval = vector_search_service.retrieve_with_sources(rewritten_query, top_k=top_k)
retrieval.rewritten_query = rewritten_query
```

学习建议：先不要默认启用。可以加配置：

```python
rag_query_rewrite_enabled: bool = False
```

## 11. 增加 Reranker

Reranker 是企业 RAG 很重要的一环。向量检索负责召回，reranker 负责精排。

初期可以定义抽象接口，先用假实现：

新增文件：`app/services/reranker_service.py`

```python
from app.models.retrieval import RetrievedSource


class BaseReranker:
    def rerank(self, query: str, sources: list[RetrievedSource]) -> list[RetrievedSource]:
        raise NotImplementedError


class NoopReranker(BaseReranker):
    def rerank(self, query: str, sources: list[RetrievedSource]) -> list[RetrievedSource]:
        return sources


class KeywordOverlapReranker(BaseReranker):
    def rerank(self, query: str, sources: list[RetrievedSource]) -> list[RetrievedSource]:
        query_chars = set(query.lower())

        def score(source: RetrievedSource) -> float:
            content_chars = set(source.content.lower())
            return len(query_chars & content_chars) / max(len(query_chars), 1)

        ranked = sorted(sources, key=score, reverse=True)
        for rank, source in enumerate(ranked, start=1):
            source.rank = rank
            source.score = max(source.score, score(source))
            source.retrieval_type = "hybrid+rerank"
        return ranked


reranker_service = KeywordOverlapReranker()
```

后续你可以替换成真实 reranker：

- DashScope text-rerank 模型。
- BGE reranker。
- Cohere rerank。
- 本地 cross-encoder。

在 `retrieve_with_sources()` 里加：

```python
sources = reranker_service.rerank(query, sources)
sources = sources[:top_k]
```

## 12. 低置信度拒答策略

在 `RagAnswerService.answer()` 中：

```python
if not retrieval.sources:
    return self._no_answer(retrieval, "知识库中没有找到相关内容。")

if retrieval.confidence < config.rag_min_confidence:
    return self._no_answer(retrieval, "检索结果相关性不足，无法可靠回答。")
```

配置：

```python
rag_min_confidence: float = 0.35
```

不要害怕拒答。企业 RAG 的质量不是“什么都答”，而是“该答时答得有依据，不该答时稳稳拒绝”。

## 13. 记录检索日志

建议新增结构化日志：

```python
logger.info(
    "rag_retrieval_done",
    query=question,
    rewritten_query=rewritten_query,
    top_k=top_k,
    source_count=len(retrieval.sources),
    confidence=retrieval.confidence,
    elapsed_ms=retrieval.elapsed_ms,
)
```

如果当前 loguru 配置不支持结构化字段，可以先这样：

```python
logger.info(
    f"RAG 检索完成 query={question}, sources={len(retrieval.sources)}, "
    f"confidence={retrieval.confidence:.2f}, elapsed_ms={retrieval.elapsed_ms}"
)
```

后续接 OpenTelemetry 时，这些字段会很有用。

## 14. 建立检索评测集

新增目录：

```text
evals/
  rag_cases.jsonl
  run_retrieval_eval.py
```

`evals/rag_cases.jsonl` 示例：

```json
{"question":"CPU 使用率过高应该怎么排查？","expected_files":["cpu_high_usage.md"]}
{"question":"磁盘空间快满了怎么处理？","expected_files":["disk_high_usage.md"]}
{"question":"服务 503 一般怎么定位？","expected_files":["service_unavailable.md"]}
```

`evals/run_retrieval_eval.py` 示例：

```python
import json

from app.services.vector_search_service import vector_search_service


def main() -> None:
    total = 0
    hit = 0

    with open("evals/rag_cases.jsonl", encoding="utf-8") as f:
        for line in f:
            case = json.loads(line)
            result = vector_search_service.retrieve_with_sources(case["question"], top_k=3)
            filenames = {source.filename for source in result.sources}
            expected = set(case["expected_files"])

            total += 1
            if filenames & expected:
                hit += 1
            else:
                print("MISS:", case["question"], "got=", filenames, "expected=", expected)

    print(f"Recall@3 = {hit / total:.2%} ({hit}/{total})")


if __name__ == "__main__":
    main()
```

运行：

```bash
python evals/run_retrieval_eval.py
```

第一阶段只看 Recall@3 就够了。等你熟了，再加 MRR、NDCG、答案忠实度。

## 15. 测试建议

新增单元测试：

```python
def test_retrieve_with_sources_returns_sources(mock_dependencies):
    from app.services.vector_search_service import vector_search_service

    result = vector_search_service.retrieve_with_sources("cpu high", top_k=2)

    assert result.query == "cpu high"
    assert len(result.sources) <= 2
    assert result.confidence >= 0
```

新增 API 测试：

```python
def test_chat_rag_returns_sources(client, mocker):
    mocker.patch(
        "app.api.chat.rag_answer_service.answer",
        return_value={
            "success": True,
            "answer": "测试回答 [来源1]",
            "sources": [],
            "confidence": 1.0,
        },
    )

    response = client.post(
        "/api/chat_rag",
        json={"Id": "s1", "Question": "hello"},
    )
    assert response.status_code == 200
    assert "sources" in response.json()["data"]
```

## 16. 验收标准

完成后检查：

- `/api/chat_rag` 返回 `answer`、`sources`、`confidence`。
- 回答中能看到 `[来源1]` 这类引用。
- 没有检索结果时不会编造答案。
- `VectorSearchService` 保留旧方法，新增企业级 sources 方法。
- 至少有 10 条检索评测 case。
- 调整 chunk 或 top_k 后，可以用评测脚本观察效果变化。

## 17. 学习重点

这一阶段你要掌握：

- 为什么企业 RAG 必须返回引用。
- 混合检索、rerank、阈值过滤分别解决什么问题。
- 为什么 Agent 不一定适合承担主问答链路。
- 如何构建最小可用检索评测集。
- 如何通过评测驱动 chunk、top_k、prompt、reranker 的调参。

