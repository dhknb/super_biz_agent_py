# ADR-004：SOP 检索接入首响 + 证据溯源

- 状态：已接受
- 日期：2026-07-23
- 关联：[[000]]（范围）、[[001]]（AlarmEvent）、[[003]]（报告 schema）
- 产品依据：`docs/product-direction-aiops-first-response.md` §5.2 半固定链路、§9 Phase 3

## 背景

这是整个两周冲刺**最核心**的一天。项目要和参考项目
[itops-agent-platform](https://github.com/qinshihu/itops-agent-platform)
拉开差距，靠的不是执行能力（对方 SSH/Docker/K8s 真实执行 + 命令注入防护是护城河，
我们两周补不上，见 [[000]]），而是**本项目已有的 RAG 检索能力**——
对方是「执行型」平台，最弱的恰恰是知识检索。

所以本项目的差异化定位是：**把 RAG 焊进 AIOps 首响，让报告的每条证据可溯源。**

## 决策

新增两个服务，把告警和知识库 SOP 关联起来：

1. `SopRetrievalService`（`app/services/sop_retrieval_service.py`）
   - 输入 `AlarmEvent`，用 `event.retrieval_query()`（告警名 + 服务 + 摘要）
     去复用项目已有的混合检索 `vector_search_service.retrieve_documents`。
   - 把每个命中文档转成 `Evidence(type=VERIFIED_FACT, source=SOP)`，
     带 `source_title`（h 标题 / 文件名）和 `excerpt`（命中片段）——**这就是证据溯源**。
   - 检索器**依赖注入**，测试传假检索器，不连 Milvus。
   - 检索失败降级为空证据，绝不让主流程崩。

2. `FirstResponseService`（`app/services/first_response_service.py`）
   - 串起闭环：`AlarmEvent → SOP 检索 → 构造 prompt → LLM 结构化输出
     → parse_report（可降级）→ 合并 SOP 证据`。
   - **事实优先**：代码注入的 SOP 证据（VERIFIED_FACT）永远排在模型自产证据前面。
   - LLM 与 SOP 服务都依赖注入；LLM 异常时降级，但**仍保留 SOP 证据**。

### 「主流程固定、节点内部智能」

不同于旧 `aiops_service` 让模型自由规划每一步（不稳定），
首响链路的检索是**固定步骤**，模型只负责理解告警、组织语言、区分事实与推断。
这符合产品方向文档 §5.2 的半固定链路设计。

## 备选方案

1. **让模型自己调 `retrieve_knowledge` 工具检索**（现有 planner 的做法）
   - 放弃：模型可能不调、调了也不一定把来源信息带进最终报告，无法保证溯源。
2. **只把检索结果拼进 prompt，不转成结构化 Evidence**
   - 放弃：那样报告里就无法机器可读地区分「事实 / 推断」，评测脚本（Day8）也没法量化。

## 后果

### 正面
- 报告的 SOP 证据可溯源（来源标题 + 片段），可被值班同学复核。
- 「已验证事实 / 模型推断」在数据结构层面区分，不只是排版。
- 检索、LLM 全可注入，测试不连外部服务，快且稳。
- 这是简历上最有说服力的一段：把 RAG 从「聊天问答」延伸到「运维证据链」。

### 负面 / 代价
- 检索质量直接影响证据质量，需要 Day8 的评估脚本量化把关。
- SOP 证据目前只来自知识库；监控 / 日志证据（MCP 工具）留到后续。

## 验证

- `tests/unit/test_sop_retrieval_service.py`（8）+ `test_first_response_service.py`（6），全绿。
- 覆盖：证据溯源、事实优先、检索/LLM 异常降级、prompt 内容、无 SOP 兜底。
