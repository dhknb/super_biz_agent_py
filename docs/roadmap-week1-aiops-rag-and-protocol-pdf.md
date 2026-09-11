# AIOps Week 1 Roadmap

## 目标

在 1 周内，把当前项目收敛成一个可演示、可验证、可继续迭代的版本，重点完成两条主线：

1. `chat_v2` 高精度 RAG 闭环
2. 协议 PDF 入库闭环

本周不追求把所有设想一次性做完，而是优先补齐最关键的工程闭环，让系统具备：

- 可解释的高精度问答
- 可追踪的运行轨迹
- 可回放的 bad case 分析入口
- 可审阅的 PDF 协议入库流程

---

## 当前基础

### RAG 侧已有能力

- 基础 `chat` / `chat_stream` 接口
- `chat_v2` / `chat_v2_stream` 图编排问答链路
- 多查询改写
- 子查询并行检索
- 混合检索：Milvus + BM25
- 检索结果去重
- 基于证据生成答案
- `coverage_score` / `groundedness_score` 校验
- 对话持久化

### 协议 PDF 入库侧已有能力

- PDF 上传接口
- 异步任务入队与 worker 执行
- 文本抽取
- 结构化草稿生成
- 结构化校验
- dry-run plan 生成
- 人工确认 / 拒绝
- state trace 记录

---

## 本周核心交付

本周只盯住 4 个最重要的交付件：

1. `chat_v2` 的 `run_trace` 持久化
2. 前端 `High Precision` 模式展示 `sub_queries + validation`
3. `bad case` 回放入口
4. 协议 PDF 入库流程的可演示闭环

如果这 4 个点完成，项目就会从“有多个功能点”升级为“有完整业务链路的演示系统”。

---

## 范围边界

### 本周要做

- 强化 `chat_v2` 作为主演示链路
- 保存 `chat_v2` 中间运行轨迹
- 增强前端对高精度模式的解释性展示
- 提供 bad case 观察与回放能力
- 补齐协议 PDF 入库的展示与确认流程
- 完善 1 周内需要的文档与演示说明

### 本周先不做

- 完整三层记忆架构
- 通用长期记忆画像系统
- 复杂多 Agent 框架
- 全自动二次检索返工闭环
- 协议 PDF 正式写入全部业务表
- 大规模检索评测平台化建设

原因：这些内容跨度大、耦合高，不适合在 1 周内同时推进。

---

## 目标产出形态

1 周后，希望项目能达到以下状态：

### 问答链路

- 用户发起 `chat_v2` 请求
- 后端执行子查询改写、并行检索、去重、生成、校验
- 系统把运行轨迹保存下来
- 前端可以展示高精度过程信息
- 如果回答质量不足，可以被标记为 bad case
- bad case 可以查询和回放

### 协议入库链路

- 用户上传 PDF
- 系统异步抽取文本并生成结构化草稿
- 系统展示校验结果和 dry-run plan
- 用户可确认或拒绝
- 系统记录状态流转和操作痕迹

---

## 详细实施计划

## Day 1：收口目标与结构设计

### 目标

明确本周实现边界，先把数据结构和交付标准固定下来。

### 任务

#### 1. 明确 `chat_v2` 为主演示链路

- 后续所有新增可解释能力优先挂在 `chat_v2`
- `chat` 保留，但不作为本周重点

#### 2. 设计 `run_trace` 结构

建议定义统一结构，至少包含：

- `trace_id`
- `session_id`
- `question`
- `source`：`chat_v2` / `chat_v2_stream`
- `sub_queries`
- 每个子查询召回数量
- `used_documents`
- `answer`
- `validation`
- `is_bad_case`
- `created_at`

#### 3. 定义 bad case 判定规则

建议第一版直接按以下规则判定：

- `validation.blocked == true`
- 或 `coverage_score < coverage_min`
- 或 `groundedness_score < groundedness_min`
- 或最终回答为保守 fallback answer

#### 4. 确认 PDF 协议入库本周目标

本周目标不是“正式写业务表”，而是：

- 可上传
- 可抽取
- 可审阅
- 可确认/拒绝
- 可展示状态轨迹

### 验收标准

- 输出一份 `run_trace` 字段设计
- 输出一份 bad case 判定规则
- 明确本周不做项

---

## Day 2：实现 `chat_v2` 的 `run_trace` 持久化

### 目标

让 `chat_v2` 的执行过程不再只是接口即时返回，而是能被持久化保存和后续查询。

### 任务

#### 1. 新建运行轨迹存储

推荐单独建表，例如：

- `chat_run_traces`

字段建议：

- `id`
- `session_id`
- `question`
- `answer`
- `source`
- `sub_queries`
- `retrieved_count`
- `used_documents`
- `validation`
- `is_bad_case`
- `error_message`
- `created_at`

#### 2. 在 `chat_v2` 成功路径保存 trace

在 `chat_v2` 接口里保存：

- 用户问题
- 最终答案
- 检索使用情况
- 校验结果
- bad case 标记

#### 3. 在 `chat_v2_stream` 完成时保存 trace

流式接口也要在最终完成时落库，保证数据一致。

#### 4. 错误场景也记录

如果图执行失败，建议保留失败 trace，便于排查。

### 验收标准

- `chat_v2` 请求后数据库有 trace 记录
- `chat_v2_stream` 完成后数据库有 trace 记录
- trace 可以区分正常 / bad case / error

---

## Day 3：前端实现 `High Precision` 模式

### 目标

把 `chat_v2` 的中间结果显式展示出来，让系统从“黑盒回答”变成“可解释回答”。

### 任务

#### 1. 增加高精度模式入口

建议使用：

- 显式开关
- 或单独模式按钮

#### 2. 展示 `sub_queries`

展示用户问题被拆成了哪些子查询。

#### 3. 展示检索过程信息

最小展示版可以包含：

- 子查询数
- 每个子查询召回数
- 最终使用了哪些文档

#### 4. 展示 `validation`

建议展示：

- `coverage_score`
- `groundedness_score`
- `coverage_pass`
- `groundedness_pass`
- `unsupported_claims`
- `missing_aspects`
- `reason`

#### 5. 展示保守降级状态

当回答被阻断或保守降级时，前端应能明确告诉用户：

- 是因为什么触发
- 哪些点没覆盖
- 哪些说法缺少证据

### 验收标准

- 前端能开启 `High Precision`
- 用户能看到 `sub_queries + validation`
- bad case 在 UI 上有明显识别方式

---

## Day 4：实现 bad case 回放入口

### 目标

让 bad case 不只是被记录，还能被分析、复盘、用于后续优化。

### 任务

#### 1. 提供 bad case 列表接口

支持按以下维度筛选：

- 时间
- 会话
- 是否 blocked
- 分数低于阈值

#### 2. 提供 bad case 详情接口

返回：

- question
- answer
- sub_queries
- used_documents
- validation
- 错误信息

#### 3. 提供回放入口

回放不一定要重新真正执行第一版，可以先做“结果回放”：

- 看某次问答过程数据
- 查看最终被判为 bad case 的原因

#### 4. 预留“重试”能力

可以只先在接口层预留：

- re-run 某次问题
- 或人工复制问题重新测试

### 验收标准

- 可以列出 bad case
- 可以查看单个 bad case 全量过程
- 至少支持静态回放

---

## Day 5：补齐协议 PDF 入库展示闭环

### 目标

把已有的 PDF 协议入库能力，补成一个完整可演示流程。

### 当前已有能力

- 上传 PDF
- 创建 ingestion/job
- RQ 异步处理
- 文本抽取
- 结构化 draft
- 校验
- dry-run plan
- confirm/reject
- state trace

### 本日重点

#### 1. 补齐前端或详情展示

需要能展示：

- `structured_data`
- `validation_result`
- `dry_run_plan`
- `state_trace`
- `current_phase`
- `status`

#### 2. 明确状态机表现

建议展示完整状态流：

- `pending`
- `extracting`
- `structuring`
- `validating`
- `awaiting_confirmation`
- `writing`
- `completed`
- `failed`
- `rejected`

#### 3. 明确人工确认逻辑

用户应能看懂：

- 为什么可以确认
- 为什么被拒绝
- 是否仍存在 `errors` / `warnings`

#### 4. 梳理 dry-run plan 可视化

尤其是：

- `protocol_saved`
- `equipment_mapped`
- `points_saved`
- `thresholds_saved`

这些 phase 可以按步骤展示。

### 验收标准

- 用户可以上传 PDF
- 可以查看结构化结果和校验结果
- 可以确认或拒绝
- 可以看到完整状态轨迹

---

## Day 6：联调、验收、修正问题

### 目标

把两条主线串起来跑通，解决阻塞演示的问题。

### 任务

#### 1. 问答链路联调

验证：

- `chat_v2` 正常问答
- 高精度模式展示正常
- trace 能保存
- bad case 能被标记

#### 2. PDF 入库链路联调

验证：

- 上传成功
- worker 正常处理
- 结构化输出稳定
- 校验结果合理
- confirm/reject 正常更新

#### 3. 补测试或最小验证脚本

至少为新增功能补以下验证：

- trace 持久化
- bad case 判定
- PDF 状态流转

#### 4. 修复演示阻塞问题

优先级最高的是影响演示的 bug：

- 接口字段缺失
- 状态更新异常
- 前端展示错位
- 流式数据和持久化不一致

### 验收标准

- 两条主线都能完整走一遍
- 关键演示路径无明显阻塞

---

## Day 7：补文档、演示材料与后续计划

### 目标

把这一周成果沉淀成可以继续开发、可以讲述、可以复用的文档。

### 任务

#### 1. 更新产品方向文档

补充：

- 这周落地内容
- 为什么优先做这几项
- 目前能力边界

#### 2. 增加演示说明

建议写两条演示脚本：

##### 演示线 A：高精度运维问答

- 输入复杂运维问题
- 展示子查询拆解
- 展示检索使用情况
- 展示 validation
- 展示 bad case 标记或通过结果

##### 演示线 B：协议 PDF 入库

- 上传 PDF
- 查看结构化结果
- 查看校验和 dry-run
- 人工确认或拒绝
- 查看状态轨迹

#### 3. 输出下一阶段演进清单

建议列出：

- `validate -> retry` 自动返工闭环
- 运维场景排序增强
- 轻量排障状态摘要
- project context 打通
- 数据清理策略

### 验收标准

- 文档可直接指导下周继续开发
- 演示流程可直接照着跑

---

## 验收标准总表

本周完成的判定标准如下：

### RAG 侧

- `chat_v2` 有持久化 `run_trace`
- `chat_v2_stream` 完成时也保存 trace
- 可以识别并查询 bad case
- 前端可展示 `sub_queries + validation`

### PDF 侧

- PDF 上传可正常触发处理
- 可查看结构化结果
- 可查看校验结果和 dry-run
- 可 confirm/reject
- 可查看 state trace

### 工程侧

- 有本周 Roadmap 文档
- 有演示说明
- 有下一步演进列表

---

## 风险与注意事项

### 1. 不要同时大改两套问答链路

本周重点应放在 `chat_v2`，避免同时重构 `chat` 和 `chat_v2`。

### 2. 不要在 1 周内追求完整记忆架构

当前更重要的是高精度问答解释性和可追踪性，而不是复杂长期记忆。

### 3. PDF 入库先做“可审阅闭环”

不要强行在本周推进“真实写业务表”的全自动落地，风险高且范围大。

### 4. trace 结构要可扩展

后续很可能要接：

- 二次检索
- rerank 结果
- 工具执行轨迹
- prompt/version 信息

所以字段设计不要过于死板。

---

## 下一阶段建议

如果本周顺利完成，下阶段优先做：

1. `validate -> retry` 自动返工闭环
2. 运维场景化检索排序优化
3. 轻量排障状态摘要
4. project context 业务打通
5. 软删除数据清理策略

---

## 一句话总结

本周最值得做的，不是继续扩散需求，而是把现有能力收敛成两个可演示闭环：

- 一个是可解释、可追踪、可回放的 `chat_v2` 高精度运维问答链路
- 一个是可抽取、可审阅、可确认的协议 PDF 入库链路

只要这两条线打通，项目的工程完整度、演示价值和后续可迭代性都会明显提升。
