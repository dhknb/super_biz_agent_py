# ADR-007：AIOps 首响报告质量评估

- 状态：已接受
- 日期：2026-07-23
- 关联：[[006-test-isolation-and-secret-hygiene]]、[[004-sop-retrieval-and-evidence-tracing]]

## 背景

Day8 的目标是「让检索/报告好不好，从拍脑袋变成可量化」。

盘点现有资产时发现 `tests/eval/` 已经有一套**检索类**评估框架：
`golden_set.jsonl` + `metrics.py`（Hit@K / MRR / Recall@K）+ `run_eval.py`。
它度量的是「检索器有没有召回对的文档」，但**没有**覆盖 AIOps 首响
特有的三个质量维度：报告是否用上了 SOP、是否覆盖关键排查要点、
无证据时是否会编造根因。

## 决策

**不重复造轮子**，只补现有框架没覆盖的增量——新增 AIOps 首响专属评估：

1. `tests/eval/aiops_first_response_eval.py`：三个纯函数打分维度
   - `score_sop_hit`：报告证据里是否命中期望 SOP 关键词
   - `score_point_coverage`：报告全文覆盖多少比例的期望排查要点
   - `score_hallucination_controlled`：无 VERIFIED_FACT 证据却硬给根因
     且无待确认项 → 判定为「无据编造」，不受控
2. `tests/eval/run_aiops_eval.py`：一条命令跑出三个维度的基线百分比
3. 评估集复用 Day6 的 `tests/fixtures/aiops_alerts/*.json`
   （已带 `expected_sop_keywords` / `expected_report_points`）

评估口径与 `metrics.py` 保持一致：打分函数都是纯函数，
输入「报告 + 期望」，不耦合 LLM / 检索实现，易单测。

## 备选方案

- **直接接 RAGAS**：faithfulness / answer_relevancy 更权威，但要装重依赖、
  要真实 LLM 打分、跑得慢。当前阶段先用零依赖的自定义指标建立基线，
  RAGAS 留作后续（`metrics.py` 里已有 placeholder 接口位）。
- **只扩 golden_set.jsonl**：那套是 doc_id 级检索评测，表达不了
  「报告要点覆盖」「幻觉控制」这种生成侧维度，硬塞会污染检索评测语义。

## 后果

### 正面

- AIOps 首响有了可复现、可回归的质量基线；改 prompt / 检索后能对比数字。
- 三个维度都有单测（17 个），打分逻辑本身被锁住。
- 离线（假 LLM + 假 SOP）也能跑出「理想基线」，验证管道正确性，
  不依赖 Milvus / 真实 LLM。

### 负面 / 代价

- 离线基线是「理想值」（100%），不代表真实模型质量；真实数字要等
  接上 Milvus + 真实 LLM 才有意义。这一点在 CLI 文档里已注明。
- 关键词子串匹配比较粗；语义级匹配要等 RAGAS 或 embedding 相似度。
