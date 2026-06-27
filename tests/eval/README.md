# RAG 离线评测

> 给 super_biz_agent_py 的检索 + 生成栈做"可复现、可对比、可上 CI"的离线评测。
> 是后续做 **混合检索 / Rerank / Coverage Critic** 优化的"指北针"——没指标的优化都是玄学。

---

## 文件结构

```
tests/eval/
├── __init__.py
├── README.md              ← 本文档
├── golden_set.jsonl       ← 评测集(每行一个 question + 期望 chunk_ids + 参考答案)
├── metrics.py             ← Hit@K / MRR / Faithfulness 等指标实现
├── run_eval.py            ← CLI 入口:跑评测 → 出报告
└── reports/               ← 每次跑的 JSON 报告(进 .gitignore)
    └── 2026-06-15T09-30.json
```

---

## 快速开始

```bash
# 1. 先把 golden_set 补到至少 30 条(目前只放了 5 条示例)
$EDITOR tests/eval/golden_set.jsonl

# 2. 跑评测(默认 baseline 配置)
uv run python -m tests.eval.run_eval

# 3. 跑某个对照实验(改完检索代码后)
uv run python -m tests.eval.run_eval --tag hybrid-rrf-v1

# 4. 看历史报告对比
uv run python -m tests.eval.run_eval --compare baseline hybrid-rrf-v1
```

---

## 评测集格式(`golden_set.jsonl`)

每行一个 JSON 对象:

```json
{
  "id": "q-001",
  "question": "CPU 使用率持续高于 90% 怎么处理?",
  "expected_chunk_ids": ["doc-cpu-high::0", "doc-cpu-high::2"],
  "expected_doc_ids": ["doc-cpu-high"],
  "reference_answer": "1) 通过 top 定位高 CPU 进程;2) 检查是否有死循环或异常服务;3) ...",
  "tags": ["aiops", "cpu", "easy"]
}
```

字段说明:

| 字段 | 必填 | 用途 |
|---|---|---|
| `id` | ✅ | 唯一 ID,便于跨次报告追踪某条样本 |
| `question` | ✅ | 用户口语化问题 |
| `expected_chunk_ids` | ⭕ | 命中的 chunk id(可选,有就能算 Hit@K) |
| `expected_doc_ids` | ✅ | 命中的文档 id(必填,Hit@K 的最低门槛) |
| `reference_answer` | ⭕ | 用于算 Faithfulness / Answer Relevance |
| `tags` | ⭕ | 分组统计,如按场景/难度切片看指标 |

---

## 指标说明

| 指标 | 算什么 | 怎么算 |
|---|---|---|
| **Hit@K** | Top-K 检索里有没有任一期望文档/chunk | 命中数 / 总样本数 |
| **MRR** | 期望项第一次出现的倒数排名平均 | mean(1/rank),没命中算 0 |
| **Recall@K** | Top-K 召回的期望项占比 | 命中期望数 / 期望总数 |
| **Faithfulness** | 回答有没有"出离"检索内容 | 用 ragas / LLM-as-judge |
| **Answer Relevance** | 回答跟问题的相关度 | 用 ragas / LLM-as-judge |

前 3 个是**检索质量**指标,后 2 个是**生成质量**指标。

**优化混合检索时,盯着 Hit@K / MRR;
做 Coverage Critic 时,盯着 Faithfulness。**

---

## 推荐工作流

1. **冻结一个 baseline**:当前单路向量召回的指标,作为基线
2. **每改一处检索/生成代码,跑一次评测**,带 `--tag` 区分版本
3. **CI 上跑评测**,Hit@K 退化 > 3% 自动失败
4. **bad case 自动落盘**:Hit 失败的样本进 `reports/bad_cases/`,人工 review

---

## 后续计划(对应 3 周 MVP 路线)

- [x] Week 1: baseline + Hit@K/MRR/Recall@K 跑通
- [x] Week 1: 接入真实向量检索器(`VectorServiceRetriever`)
- [x] Week 1: 跑出首份真实 baseline(`baseline-real-2026-06-22T01-34-04.json`)
- [ ] Week 1: 接入 Langfuse Trace,每次 eval 自动归档
- [ ] Week 2: 接入 ragas 算 Faithfulness/Answer Relevance
- [ ] Week 2: 调整 RRF 权重或 Rerank,目标 Hit@1 ≥ 0.6
- [ ] Week 3: 接入 Rerank,再出一组对比

---

## 📊 首份真实 baseline 观察(2026-06-22)

**当前检索栈**:向量 (0.7 权重) + BM25 (0.3 权重) RRF 融合,top_k=10

| 指标 | baseline-real | bm25-weak (0.85/0.15) | rerank-v1 |
|---|---|---|---|
| Hit@1 | **0.0** | 0.0 | 0.0 |
| Hit@3 | 1.0 | 1.0 | 1.0 |
| Hit@5 | 1.0 | 1.0 | 1.0 |
| MRR | 0.333 | 0.333 | 0.333 |

**根因观察**:看 `reports/baseline-real-*.json` 的 `cases` 字段,**所有 5 个查询的 Top-2 永远是 `day07` 和 `day05`**(用户最近上传的"学习总结"类文档,非 AIOps 专业文档)。猜测原因:
1. day07/day05 是杂文档,内容覆盖多关键词,被 BM25 和向量两路持续推高
2. chunk 切分时这两个文档产生了大量"通用关键词"chunk(目录/代码/概念)形成"通用召回器"
3. **实验结论**:调 RRF 权重无效果(因为两路各自都把它们排前面),Rerank 需 DashScope API 权限(当前 key 被 Denied)

**下一步实验建议**:
1. ~~调 `VECTOR_WEIGHT=0.85 / BM25_WEIGHT=0.15`~~ → **无效**,top-5 完全相同
2. ~~接 DashScope gte-rerank~~ → **403 AccessDenied**,key 无 rerank 权限
3. 🔥 **推荐**:对候选 doc 列表加一个"领域相关性 pre-filter"——假设 AIOps 文档名包含 `usage/cpu/memory/disk/service/slow` 等一系列特征词,把纯学习笔记类文档(day*/苍穹外卖)过滤或降权
4. 🔥 **备选**:用 `BAAI/bge-reranker-v2-m3` 做本地 Rerank,不走 DashScope
5. 扩充 golden_set 到 30+ 条,提高统计稳定性
