# Agent 岗位面经逐题回答（基于当前仓库）

> 原题来源：`docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md`
> 
> 处理原则：严格按原文件篇章和题目顺序保留重复题；每题先复述原题，再结合当前仓库回答；没有对应实现的题目明确写“项目中没有，不能硬套”；不补造原文件标记为未公开的内容。

> 原文件解析得到 **238 道实际面试题**。

## 字节面经-字节跳动AI Agent开发岗面经-03

> 来源：`https://www.nowcoder.com/discuss/922659809084571648`

## 面经 01

### 1. langchain和langgraph的区别和应用是什么

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:15`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 2. AI运营工具如何提效的，主要做了什么

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:16`

**回答：**

当前项目更准确的定位是企业智能对话和 AIOps 运维助手，而不是泛化的用户增长运营平台。它主要通过告警归一化、SOP 检索、结构化首响报告、MCP 日志/监控工具和阶段时间线，减少人工复制告警上下文、手动查文档、手动整理排查步骤的时间。

具体链路是：API 接收告警并归一化为 `AlarmEvent`，编排器创建诊断任务；`FirstResponseService` 检索 SOP、构造结构化 prompt、调用 LLM 生成首响报告，再把已验证 SOP 证据、降级原因和阶段耗时落库。SOP 检索失败、知识库为空和 LLM 超时分别保留不同语义，值班人员可以知道是补文档、修检索服务，还是处理模型依赖问题。MCP 工具不可用时，基础聊天仍可降级为本地工具。

因此“提效”可以落到三个可验证方向：首响时间缩短、排查步骤结构化、故障原因可追踪。当前仓库没有 DAU、转化率、营销漏斗或线上人工节省时长等运营指标，项目中没有对应实现，不能硬套；如果面试官追问数字，应只报告真实测量结果，不能把离线验证写成线上收益。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/api/aiops.py:50-75` — 告警分析 API。
- `/home/dong/projects/super_biz_agent_py/app/services/alert_diagnosis_orchestrator.py:71-131` — 归一化、任务创建、阶段流转和报告落库。
- `/home/dong/projects/super_biz_agent_py/app/services/first_response_service.py:190-287` — SOP 检索、结构化报告、LLM 降级和阶段耗时。
- `/home/dong/projects/super_biz_agent_py/app/services/sop_retrieval_service.py:108-183` — `RetrievalStatus` 与证据转换。
- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 工具调用和指数退避重试。

### 3. 字节用什么工具开发

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:17`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

## 面经 02

### 1. 为什么选择 AI 应用开发方向？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:24`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 2. 平时如何学习大模型及 Agent 领域的前沿技术？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:25`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 3. 你如何理解 Agent 系统？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:26`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 4. Tool 的设计原则是什么？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:27`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 5. Memory 有哪些类型？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:28`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 6. ReAct 和 Plan-Execute 架构分别适用于哪些场景？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:29`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

## 面经 08

### 1. 你现在大二，实习时间有多长，怎么平衡学校的事情？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:36`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 2. 说一下你这个agent项目怎么开发的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:37`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 3. 你项目rag是怎么存储和召回的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:38`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 4. 你项目的OCR是怎么实现的(多模态输入问题，把ASR的接入也说明了一下，其实还是大模型api)

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:39`

**回答：**

项目当前有 PDF/Markdown 文档入库和 LLM 处理链路，但没有 OCR、ASR 或会议转写服务的实现。可以通用回答：音视频先分片/解码，ASR 生成带时间戳文本，术语词典和后处理纠错，再由 LLM 做摘要；OCR 则是图像预处理、文字检测识别、版面结构恢复和多模态模型补充。识别质量应看 CER/WER、术语召回和摘要事实一致性。项目中没有对应实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 5. 你项目中uuid的实现和维度是什么？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:40`

**回答：**

当前仓库的真实链路可以概括为：FastAPI 接入层接收请求；服务层负责会话记忆、RAG/AIOps 编排和业务规则；核心层提供 LLM 工厂、错误分类、熔断、指标和数据库；Agent 层负责 LangGraph/LangChain 和 MCP；数据层保存会话、诊断任务、trace/span。RAG v2 的固定图是 rewrite → retrieve_each fan-out → dedup → generate → validate；AIOps 是 normalize → create task → SOP retrieve → LLM report → save report。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 6. 你使用的是多模块agent，为什么不使用多agent形式开发(尝试过muti-agent,协商开销太大,耗时不可以接受);

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:41`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

## 面经 09

### 1. 平台架构为什么设计成三层？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:48`

**回答：**

当前仓库的真实链路可以概括为：FastAPI 接入层接收请求；服务层负责会话记忆、RAG/AIOps 编排和业务规则；核心层提供 LLM 工厂、错误分类、熔断、指标和数据库；Agent 层负责 LangGraph/LangChain 和 MCP；数据层保存会话、诊断任务、trace/span。RAG v2 的固定图是 rewrite → retrieve_each fan-out → dedup → generate → validate；AIOps 是 normalize → create task → SOP retrieve → LLM report → save report。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 2. 意图识别简单问题拦截这块，有没有记忆机制？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:49`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 3. MCP 工具主要包含哪些？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:50`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 4. 项目里记忆模块具体怎么实现？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:51`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 5. 如何衡量agent效果好坏？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:52`

**回答：**

评估 Agent 不能只看最终文本。应拆成任务成功率、答案正确性/证据支撑、工具选择正确率、参数正确率、无效步骤数、循环率、恢复率、P95/P99 延迟、Token/成本和安全违规率。当前仓库已经有 RAG 评估脚本与 golden set、节点 span、chat run trace、validation 字段和 Prometheus 指标；但没有题目中某个外部 benchmark 或线上业务转化数据，因此只能说明现有离线评估和观测能力，不能编造线上提升数字。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/instrumentation.py:1-180` — 节点级 instrumentation。
- `/home/dong/projects/super_biz_agent_py/app/core/span_context.py:1-208` — span 收集。
- `/home/dong/projects/super_biz_agent_py/app/repositories/chat_run_trace_repository.py:1-112` — trace 落库。
- `/home/dong/projects/super_biz_agent_py/app/core/metrics.py:1-197` — 指标。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:71-156` — 成功/失败 trace 与 span flush。

### 6. 如果项目上线，可以用哪些指标评估效果？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:53`

**回答：**

评估 Agent 不能只看最终文本。应拆成任务成功率、答案正确性/证据支撑、工具选择正确率、参数正确率、无效步骤数、循环率、恢复率、P95/P99 延迟、Token/成本和安全违规率。当前仓库已经有 RAG 评估脚本与 golden set、节点 span、chat run trace、validation 字段和 Prometheus 指标；但没有题目中某个外部 benchmark 或线上业务转化数据，因此只能说明现有离线评估和观测能力，不能编造线上提升数字。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/instrumentation.py:1-180` — 节点级 instrumentation。
- `/home/dong/projects/super_biz_agent_py/app/core/span_context.py:1-208` — span 收集。
- `/home/dong/projects/super_biz_agent_py/app/repositories/chat_run_trace_repository.py:1-112` — trace 落库。
- `/home/dong/projects/super_biz_agent_py/app/core/metrics.py:1-197` — 指标。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:71-156` — 成功/失败 trace 与 span flush。

### 7. 这个项目几个人开发、耗时多久？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:54`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

## 面经 10

### 1. Q；当被问到“为什么选择字节做Agent方向”

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:61`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 2. ✅ 可以这样说：字节在Agent方向的投入和落地节奏一直走在前沿，从豆包的快速迭代到内部Harness框架的开放，能感受到他们不是在做Demo，是在真刀真枪搞生产级Agent？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:62`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 3. ✅ 可以尝试：我觉得字节在Agent这块最对的一件事是没把Agent做成‘大号聊天机器人’，而是真正把它当工程系统来搭？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:63`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 4. ✅ 可以参考：我习惯把职业发展分成技术和业务两条线？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:64`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

## 面经 11

### 1. 请先做一个简单的自我介绍。

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:71`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 2. 你上一段实习是在北京做算法工程师对吗？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:72`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 3. 以你参加的全栈挑战赛中多智能体导购系统为例，你提到iOS客户端部分是边做边学、借助AI辅助完成的。那么在遇到AI无法解决的编译构建或代码问题时，你通常是怎么解决的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:73`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 4. 你们系统的核心设计思路是"通过交互确认意图"，对吗？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:74`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 5. 你提到的Router在当前系统中是硬编码实现的，可以按需扩展。但如果用户意图不在Router预定义的范围内，系统如何处理？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:75`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 6. 请介绍你们对话系统的整体设计，包括上下文管理、服务端数据存储，以及从接口接收请求到Agent处理的全链路设计。

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:76`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

## 面经 12

### 1. 项目是网上找的还是什么？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:83`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 2. 项目中挑战最大的是什么？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:84`

**回答：**

当前仓库的真实链路可以概括为：FastAPI 接入层接收请求；服务层负责会话记忆、RAG/AIOps 编排和业务规则；核心层提供 LLM 工厂、错误分类、熔断、指标和数据库；Agent 层负责 LangGraph/LangChain 和 MCP；数据层保存会话、诊断任务、trace/span。RAG v2 的固定图是 rewrite → retrieve_each fan-out → dedup → generate → validate；AIOps 是 normalize → create task → SOP retrieve → LLM report → save report。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 3. 出现幻觉怎么处理？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:85`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 4. 提示词具体是怎么做？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:86`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 5. 还有其他提示词吗？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:87`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 6. Agent的短期长期记忆是怎么实现的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:88`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 7. 如果让你设计一个Agent要考虑哪些模块？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:89`

**回答：**

当前仓库的真实链路可以概括为：FastAPI 接入层接收请求；服务层负责会话记忆、RAG/AIOps 编排和业务规则；核心层提供 LLM 工厂、错误分类、熔断、指标和数据库；Agent 层负责 LangGraph/LangChain 和 MCP；数据层保存会话、诊断任务、trace/span。RAG v2 的固定图是 rewrite → retrieve_each fan-out → dedup → generate → validate；AIOps 是 normalize → create task → SOP retrieve → LLM report → save report。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

## 面经 13

### 1. 请做一下自我介绍

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:96`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 2. 为什么选择 LangGraph，而不是直接使用 LangChain AgentExecutor？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:97`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 3. 文档审校平台的 RAG 检索链路是怎么设计的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:98`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 4. RRF 为什么不需要直接比较 BM25 分数和向量相似度？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:99`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 5. 如何评估 RAG 的召回率和最终回答质量？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:100`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

## 面经 14

### 1. 请做一下自我介绍

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:107`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 2. 在工业设备故障诊断与知识协同平台中，检索结果为什么不能直接按向量相似度返回？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:108`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 3. 跨区域库存调拨平台如何避免“库存扣了但订单失败”或者“订单成功但库存没扣”？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:109`

**回答：**

这道题的通用回答应围绕事务边界、并发控制、幂等键、状态机和可观测性展开。当前仓库确实使用 SQLAlchemy/Alembic 持久化会话、诊断任务、trace/span，也有 Redis/RQ 依赖；但没有库存调拨、Kafka 消费者或 PostgreSQL MVCC 业务实现。因此可以解释通用原理，例如用本地事务记录业务状态和幂等键、用 outbox/重试/补偿处理跨系统一致性，但必须明确：项目中没有对应实现，不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/core/database.py:1-33`、`/home/dong/projects/super_biz_agent_py/app/repositories/conversation_repository.py:1-156` — 数据库会话与仓储。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260629_0004_create_chat_run_traces.py:1-57`、`/home/dong/projects/super_biz_agent_py/migrations/versions/20260824_0009_create_chat_run_spans.py:1-85` — trace/span 表。
- `/home/dong/projects/super_biz_agent_py/app/core/task_queue.py:1-11` — RQ 队列接线，全文仅 `redis_conn`、`index_queue`、`protocol_pdf_queue` 三个模块级对象，不含任何重试或失败分类逻辑。

### 4. 为什么不能把所有历史对话直接塞给大模型？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:110`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 5. Kafka 消费者处理库存事件时，如何做到“恰好一次”？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:111`

**回答：**

项目把失败分成“可降级结果”和“彻底失败”两类。单次 LLM timeout 由 LLM 工厂注入，整条 chat_v2 还由 `asyncio.timeout(chat_total_budget_seconds)` 控制总预算；RagV2Service 保存最后一个状态快照，预算耗尽时返回已完成的子查询/文档和降级说明。并行检索分支把异常收进 `retrieve_failures`，因此一个分支失败可以变成 partial result，而不是整张图 500。异常分类集中在 `AppError`、`to_app_error`、`wrap_llm_exception` 和 `DegradeReason`；原始异常保留为 cause，应用异常提供稳定 code、retryable、http_status。熔断器只统计真正发出的下游调用，打开后快速失败并进入降级。幂等和 checkpoint 应由任务状态、唯一 request/task id、已完成步骤和副作用记录共同保证；当前仓库已实现诊断任务、trace/span 和部分 Worker 状态落盘，但没有 Kafka exactly-once 或完整跨服务事务，不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/core/errors.py:25-387` — `DegradeReason`、`AppError`、`to_app_error`、`wrap_llm_exception`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:6-78` — 单次调用超时与总预算、部分结果。
- `/home/dong/projects/super_biz_agent_py/app/core/circuit_breaker.py:1-260` — 熔断状态机。
- `/home/dong/projects/super_biz_agent_py/app/core/exception_handlers.py:90-170` — HTTP 错误响应。
- `/home/dong/projects/super_biz_agent_py/app/core/job_failure.py:68-128` — `default_job_retry`、`apply_retry_policy`：按 `AppError.retryable` 决定是否把 `retries_left` 清零。
- `/home/dong/projects/super_biz_agent_py/app/core/job_failure.py:146-183` — `record_job_failure`：独立 Session 写回失败态，自身永不抛异常。
- `/home/dong/projects/super_biz_agent_py/app/workers/index_worker.py:105-147` — except 块的固定顺序：先定重试策略、再拼消息、rollback、换 Session 写回、最后裸 `raise` 交还 RQ。
- `/home/dong/projects/super_biz_agent_py/app/core/task_queue.py:1-11` — 仅队列接线，不承担失败语义。

### 6. Agent 调用工具时，如何避免模型越权调用、参数幻觉和循环执行？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:112`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 7. PostgreSQL 的 MVCC 为什么会产生死元组？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:113`

**回答：**

这道题的通用回答应围绕事务边界、并发控制、幂等键、状态机和可观测性展开。当前仓库确实使用 SQLAlchemy/Alembic 持久化会话、诊断任务、trace/span，也有 Redis/RQ 依赖；但没有库存调拨、Kafka 消费者或 PostgreSQL MVCC 业务实现。因此可以解释通用原理，例如用本地事务记录业务状态和幂等键、用 outbox/重试/补偿处理跨系统一致性，但必须明确：项目中没有对应实现，不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/core/database.py:1-33`、`/home/dong/projects/super_biz_agent_py/app/repositories/conversation_repository.py:1-156` — 数据库会话与仓储。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260629_0004_create_chat_run_traces.py:1-57`、`/home/dong/projects/super_biz_agent_py/migrations/versions/20260824_0009_create_chat_run_spans.py:1-85` — trace/span 表。
- `/home/dong/projects/super_biz_agent_py/app/core/task_queue.py:1-11` — RQ 队列接线，全文仅 `redis_conn`、`index_queue`、`protocol_pdf_queue` 三个模块级对象，不含任何重试或失败分类逻辑。

## 面经 18（未公开/未补造）

> 原文件只保留了篇章标题和“面试时”残片，没有公开题目。按原文件边界保留，不自行补题。

## 字节面经-字节跳动AI Agent开发岗面经-02

> 来源：`https://www.nowcoder.com/discuss/922659486983036928`

## 面经 01

### 1. 二面：自我介绍 -> 手撕 -> 更换、添加条件手撕 -> 哈希表的了解 -> 场景设计（让说用什么数据结构）

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:130`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 2. 总结：首先声明我投的是AI应用开发，给我调剂到了云架构后端开发？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:131`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 3. 1 主要拷打AI项目 问各种实现 做了什么贡献

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:132`

**回答：**

当前仓库的真实链路可以概括为：FastAPI 接入层接收请求；服务层负责会话记忆、RAG/AIOps 编排和业务规则；核心层提供 LLM 工厂、错误分类、熔断、指标和数据库；Agent 层负责 LangGraph/LangChain 和 MCP；数据层保存会话、诊断任务、trace/span。RAG v2 的固定图是 rewrite → retrieve_each fan-out → dedup → generate → validate；AIOps 是 normalize → create task → SOP retrieve → LLM report → save report。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 4. 最近一年你的agent学习路线是什么，接触过哪些比较好的agent产品？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:133`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 5. 你用过trae、claude code、codex，这几个agent之间的优劣势？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:134`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 6. 你如果去评价一个agent的能力，不考虑模型，你怎么去评价？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:135`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 7. 在一些场景下可能会对模型的选型做出限制。针对不同的模型，去优化一个agent在全流程的表现，你的方法论是什么？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:136`

**回答：**

当前仓库的真实链路可以概括为：FastAPI 接入层接收请求；服务层负责会话记忆、RAG/AIOps 编排和业务规则；核心层提供 LLM 工厂、错误分类、熔断、指标和数据库；Agent 层负责 LangGraph/LangChain 和 MCP；数据层保存会话、诊断任务、trace/span。RAG v2 的固定图是 rewrite → retrieve_each fan-out → dedup → generate → validate；AIOps 是 normalize → create task → SOP retrieve → LLM report → save report。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 8. 真实的业务场景下，一类agent的需求通常是收敛的，你需要对agent在特定场景上做优化，你有什么优化的思路吗？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:137`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 9. skill的目录结构是什么样的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:138`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 10. claude code检索代码的方式？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:139`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

## 面经 02

### 1. 用的什么基础模型？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:146`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 2. 介绍一下 RAG Agent 项目

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:147`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 3. 评测集的整体构建思路？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:148`

**回答：**

评估 Agent 不能只看最终文本。应拆成任务成功率、答案正确性/证据支撑、工具选择正确率、参数正确率、无效步骤数、循环率、恢复率、P95/P99 延迟、Token/成本和安全违规率。当前仓库已经有 RAG 评估脚本与 golden set、节点 span、chat run trace、validation 字段和 Prometheus 指标；但没有题目中某个外部 benchmark 或线上业务转化数据，因此只能说明现有离线评估和观测能力，不能编造线上提升数字。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/instrumentation.py:1-180` — 节点级 instrumentation。
- `/home/dong/projects/super_biz_agent_py/app/core/span_context.py:1-208` — span 收集。
- `/home/dong/projects/super_biz_agent_py/app/repositories/chat_run_trace_repository.py:1-112` — trace 落库。
- `/home/dong/projects/super_biz_agent_py/app/core/metrics.py:1-197` — 指标。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:71-156` — 成功/失败 trace 与 span flush。

### 4. 做评测观测哪些指标？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:149`

**回答：**

评估 Agent 不能只看最终文本。应拆成任务成功率、答案正确性/证据支撑、工具选择正确率、参数正确率、无效步骤数、循环率、恢复率、P95/P99 延迟、Token/成本和安全违规率。当前仓库已经有 RAG 评估脚本与 golden set、节点 span、chat run trace、validation 字段和 Prometheus 指标；但没有题目中某个外部 benchmark 或线上业务转化数据，因此只能说明现有离线评估和观测能力，不能编造线上提升数字。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/instrumentation.py:1-180` — 节点级 instrumentation。
- `/home/dong/projects/super_biz_agent_py/app/core/span_context.py:1-208` — span 收集。
- `/home/dong/projects/super_biz_agent_py/app/repositories/chat_run_trace_repository.py:1-112` — trace 落库。
- `/home/dong/projects/super_biz_agent_py/app/core/metrics.py:1-197` — 指标。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:71-156` — 成功/失败 trace 与 span flush。

### 5. 有没有研究过行业通用/开源的评测基准？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:150`

**回答：**

评估 Agent 不能只看最终文本。应拆成任务成功率、答案正确性/证据支撑、工具选择正确率、参数正确率、无效步骤数、循环率、恢复率、P95/P99 延迟、Token/成本和安全违规率。当前仓库已经有 RAG 评估脚本与 golden set、节点 span、chat run trace、validation 字段和 Prometheus 指标；但没有题目中某个外部 benchmark 或线上业务转化数据，因此只能说明现有离线评估和观测能力，不能编造线上提升数字。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/instrumentation.py:1-180` — 节点级 instrumentation。
- `/home/dong/projects/super_biz_agent_py/app/core/span_context.py:1-208` — span 收集。
- `/home/dong/projects/super_biz_agent_py/app/repositories/chat_run_trace_repository.py:1-112` — trace 落库。
- `/home/dong/projects/super_biz_agent_py/app/core/metrics.py:1-197` — 指标。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:71-156` — 成功/失败 trace 与 span flush。

### 6. 假设你入职后，我让你做"评测集生成 + 更新维护"的行业调研，你怎么开展工作？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:151`

**回答：**

评估 Agent 不能只看最终文本。应拆成任务成功率、答案正确性/证据支撑、工具选择正确率、参数正确率、无效步骤数、循环率、恢复率、P95/P99 延迟、Token/成本和安全违规率。当前仓库已经有 RAG 评估脚本与 golden set、节点 span、chat run trace、validation 字段和 Prometheus 指标；但没有题目中某个外部 benchmark 或线上业务转化数据，因此只能说明现有离线评估和观测能力，不能编造线上提升数字。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/instrumentation.py:1-180` — 节点级 instrumentation。
- `/home/dong/projects/super_biz_agent_py/app/core/span_context.py:1-208` — span 收集。
- `/home/dong/projects/super_biz_agent_py/app/repositories/chat_run_trace_repository.py:1-112` — trace 落库。
- `/home/dong/projects/super_biz_agent_py/app/core/metrics.py:1-197` — 指标。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:71-156` — 成功/失败 trace 与 span flush。

### 7. 追问；线上 log 是海量的，怎么转化成有限的线下评测集？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:152`

**回答：**

评估 Agent 不能只看最终文本。应拆成任务成功率、答案正确性/证据支撑、工具选择正确率、参数正确率、无效步骤数、循环率、恢复率、P95/P99 延迟、Token/成本和安全违规率。当前仓库已经有 RAG 评估脚本与 golden set、节点 span、chat run trace、validation 字段和 Prometheus 指标；但没有题目中某个外部 benchmark 或线上业务转化数据，因此只能说明现有离线评估和观测能力，不能编造线上提升数字。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/instrumentation.py:1-180` — 节点级 instrumentation。
- `/home/dong/projects/super_biz_agent_py/app/core/span_context.py:1-208` — span 收集。
- `/home/dong/projects/super_biz_agent_py/app/repositories/chat_run_trace_repository.py:1-112` — trace 落库。
- `/home/dong/projects/super_biz_agent_py/app/core/metrics.py:1-197` — 指标。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:71-156` — 成功/失败 trace 与 span flush。

## 面经 03

### 1. 在实习中做的自动化怎么做的

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:159`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 2. 手撕：合并两个有序数组。

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:160`

**回答：**

这是算法题，项目中没有对应算法实现，不能硬套。回答时先明确不变量、边界和复杂度：双指针合并有序数组为 O(m+n)；合并区间先按左端点排序后线性扫描为 O(n log n)；LRU 用哈希表加双向链表使 get/put 平均 O(1)；Top-K 用容量 K 的小根堆为 O(n log K)；树形题用父子映射或 BFS。面试中应补充空输入、重复值、溢出和并发要求。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 3. 项目介绍(utbench是如何去衡量模型生成效果的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:161`

**回答：**

评估 Agent 不能只看最终文本。应拆成任务成功率、答案正确性/证据支撑、工具选择正确率、参数正确率、无效步骤数、循环率、恢复率、P95/P99 延迟、Token/成本和安全违规率。当前仓库已经有 RAG 评估脚本与 golden set、节点 span、chat run trace、validation 字段和 Prometheus 指标；但没有题目中某个外部 benchmark 或线上业务转化数据，因此只能说明现有离线评估和观测能力，不能编造线上提升数字。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/instrumentation.py:1-180` — 节点级 instrumentation。
- `/home/dong/projects/super_biz_agent_py/app/core/span_context.py:1-208` — span 收集。
- `/home/dong/projects/super_biz_agent_py/app/repositories/chat_run_trace_repository.py:1-112` — trace 落库。
- `/home/dong/projects/super_biz_agent_py/app/core/metrics.py:1-197` — 指标。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:71-156` — 成功/失败 trace 与 span flush。

### 4. 实习经历(sql多表查询和慢查询问题，如何去解决的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:162`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 5. agent是怎么调用工具的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:163`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 6. agent还有其他调用方式吗？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:164`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 7. agent怎么去判断是否需要调用工具，MCP协议解决了什么问题？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:165`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 8. agent的上下文窗口满了怎么办，如何去进行压缩，有哪几种方式？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:166`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

## 面经 04

### 1. 绩点？有论文么？有实习么？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:173`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 2. 整个链路运转的流程？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:174`

**回答：**

当前仓库的真实链路可以概括为：FastAPI 接入层接收请求；服务层负责会话记忆、RAG/AIOps 编排和业务规则；核心层提供 LLM 工厂、错误分类、熔断、指标和数据库；Agent 层负责 LangGraph/LangChain 和 MCP；数据层保存会话、诊断任务、trace/span。RAG v2 的固定图是 rewrite → retrieve_each fan-out → dedup → generate → validate；AIOps 是 normalize → create task → SOP retrieve → LLM report → save report。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 3. skill分层体系是怎么做，为什么这么设计？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:175`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 4. 用户输入怎么和相关skill匹配？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:176`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 5. 有skill沉淀机制么？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:177`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 6. 长短期记忆怎么设计的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:178`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

## 面经 05

### 1. Claude code与codex他们各自有什么特别？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:185`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 2. 输入到模型的prompt由哪些部分组成？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:186`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 3. 你做的coding agent和Claude code与codex区别在哪？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:187`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 4. 你提到了上下文压缩，是怎么做的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:188`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 5. 为什么要采用三层压缩策略，每一层压缩的内容一致么？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:189`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 6. 如果模型因为压缩过度，效果不理想，你是怎么发现的，怎么处理这种异常？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:190`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 7. 如果让agent需要对原来的任务上进一步修改，比如新加一个功能，你这个系统是怎么做的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:191`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

## 面经 06

### 1. 介绍一下两个项目，你觉得你这两个项目哪个更有收获？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:198`

**回答：**

当前仓库的真实链路可以概括为：FastAPI 接入层接收请求；服务层负责会话记忆、RAG/AIOps 编排和业务规则；核心层提供 LLM 工厂、错误分类、熔断、指标和数据库；Agent 层负责 LangGraph/LangChain 和 MCP；数据层保存会话、诊断任务、trace/span。RAG v2 的固定图是 rewrite → retrieve_each fan-out → dedup → generate → validate；AIOps 是 normalize → create task → SOP retrieve → LLM report → save report。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 2. url到页面显示？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:199`

**回答：**

项目后端提供 FastAPI API 和 SSE 流式接口，前端流式状态有测试，但仓库没有完整 React Hook 实现、浏览器从 URL 到渲染的前端源码或懒加载框架代码。React Hook 题可以回答规则：Hook 必须在组件顶层按稳定顺序调用，条件分支会让链表位置错位；跨域要由服务端 CORS 中间件和凭据策略控制；URL 到页面通常经过 DNS、TCP/TLS、HTTP、HTML 解析、资源加载、脚本执行和渲染。以上是通用原理，不能说成当前项目已实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:159-260` — SSE 流式协议。
- `/home/dong/projects/super_biz_agent_py/tests/frontend/chat_v2_stream_state.test.js:1-35` — 前端流式状态测试。
- 项目后端仓库中没有 React Hook 链表、浏览器渲染管线或完整前端工程实现。

### 3. 跨域和解决方案？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:200`

**回答：**

项目后端提供 FastAPI API 和 SSE 流式接口，前端流式状态有测试，但仓库没有完整 React Hook 实现、浏览器从 URL 到渲染的前端源码或懒加载框架代码。React Hook 题可以回答规则：Hook 必须在组件顶层按稳定顺序调用，条件分支会让链表位置错位；跨域要由服务端 CORS 中间件和凭据策略控制；URL 到页面通常经过 DNS、TCP/TLS、HTTP、HTML 解析、资源加载、脚本执行和渲染。以上是通用原理，不能说成当前项目已实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:159-260` — SSE 流式协议。
- `/home/dong/projects/super_biz_agent_py/tests/frontend/chat_v2_stream_state.test.js:1-35` — 前端流式状态测试。
- 项目后端仓库中没有 React Hook 链表、浏览器渲染管线或完整前端工程实现。

### 4. 第二个项目的性能优化问了一下细节，确实做的很少，问了懒加载的原理？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:201`

**回答：**

项目后端提供 FastAPI API 和 SSE 流式接口，前端流式状态有测试，但仓库没有完整 React Hook 实现、浏览器从 URL 到渲染的前端源码或懒加载框架代码。React Hook 题可以回答规则：Hook 必须在组件顶层按稳定顺序调用，条件分支会让链表位置错位；跨域要由服务端 CORS 中间件和凭据策略控制；URL 到页面通常经过 DNS、TCP/TLS、HTTP、HTML 解析、资源加载、脚本执行和渲染。以上是通用原理，不能说成当前项目已实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:159-260` — SSE 流式协议。
- `/home/dong/projects/super_biz_agent_py/tests/frontend/chat_v2_stream_state.test.js:1-35` — 前端流式状态测试。
- 项目后端仓库中没有 React Hook 链表、浏览器渲染管线或完整前端工程实现。

### 5. 问我学没学过编译原理？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:202`

**回答：**

这道题属于 Java/系统基础，当前仓库是 Python 项目，项目中没有 Java 线程池、JVM 类加载、mmap/writev 或编译器实现，不能硬套。通用回答应区分 CPU 密集与 I/O 密集：进程提供隔离，线程共享地址空间，协程在用户态调度；线程池要有有界队列、拒绝策略、取消/超时/幂等语义和指标；JVM 类加载通常经历加载、验证、准备、解析、初始化。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

## 面经 07

### 1. 先问要不要写代码？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:209`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 2. 再问是单 Agent 还是多 Agent？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:210`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 3. 最后问绑不绑定单一模型厂商？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:211`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 4. 我自己做过 ×× 项目，当时因为 ×× 原因选了 ××，踩过 ×× 坑，后来怎么解决的——"

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:212`

**回答：**

当前仓库的真实链路可以概括为：FastAPI 接入层接收请求；服务层负责会话记忆、RAG/AIOps 编排和业务规则；核心层提供 LLM 工厂、错误分类、熔断、指标和数据库；Agent 层负责 LangGraph/LangChain 和 MCP；数据层保存会话、诊断任务、trace/span。RAG v2 的固定图是 rewrite → retrieve_each fan-out → dedup → generate → validate；AIOps 是 normalize → create task → SOP retrieve → LLM report → save report。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

## 面经 08

### 1. React hook使用时要注意什么？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:219`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 2. 为什么不能放到条件语句中？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:220`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 3. 能不能手写一些hook的链表结构实现来体现它不能放到条件语句中？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:221`

**回答：**

这是算法题，项目中没有对应算法实现，不能硬套。回答时先明确不变量、边界和复杂度：双指针合并有序数组为 O(m+n)；合并区间先按左端点排序后线性扫描为 O(n log n)；LRU 用哈希表加双向链表使 get/put 平均 O(1)；Top-K 用容量 K 的小根堆为 O(n log K)；树形题用父子映射或 BFS。面试中应补充空输入、重复值、溢出和并发要求。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 4. JS的数据类型？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:222`

**回答：**

项目后端提供 FastAPI API 和 SSE 流式接口，前端流式状态有测试，但仓库没有完整 React Hook 实现、浏览器从 URL 到渲染的前端源码或懒加载框架代码。React Hook 题可以回答规则：Hook 必须在组件顶层按稳定顺序调用，条件分支会让链表位置错位；跨域要由服务端 CORS 中间件和凭据策略控制；URL 到页面通常经过 DNS、TCP/TLS、HTTP、HTML 解析、资源加载、脚本执行和渲染。以上是通用原理，不能说成当前项目已实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:159-260` — SSE 流式协议。
- `/home/dong/projects/super_biz_agent_py/tests/frontend/chat_v2_stream_state.test.js:1-35` — 前端流式状态测试。
- 项目后端仓库中没有 React Hook 链表、浏览器渲染管线或完整前端工程实现。

### 5. 数据类型判断的方法？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:223`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

## 面试问题汇总

### 1. 简单讲讲你的Agent项目？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:242`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 2. 短期记忆的具体实现方式是什么？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:260`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 3. 什么叫“快到上限了”？对话是怎么逐步叠加的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:274`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 4. 如果对话轮次过多，你怎么去做优化？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:292`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 5. 什么时候去触发这个总结动作？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:306`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 6. 是每一轮对话都要去做总结吗？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:320`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 7. 假设已进行 10 轮并做了总结，第 11 轮开始时，总结怎么处理？是重算前 11 轮还是叠加？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:332`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 8. 如果前 10 轮都变成了总结，那之前的原始上下文就不需要了吗？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:348`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 9. 长期记忆可以按需检索召回，具体在什么情况下需要检索？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:362`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 10. 每轮对话都要注入记忆吗？长短期记忆是同时注入吗？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:376`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 11. 如何减少工具过多带来的 Token 消耗？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:390`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 12. 你的 RAG 是用什么技术实现的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:404`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 13. 如何判断向量的相似度？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:418`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 14. 除了余弦相似度，还了解其他相似度算法吗？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:432`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 15. 听不懂，说人话？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:448`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 16. 余弦相似度与欧几里得距离在工程应用中的具体区别是什么？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:458`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 17. 项目中一共使用了几个模型？分别是什么？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:476`

**回答：**

当前仓库的真实链路可以概括为：FastAPI 接入层接收请求；服务层负责会话记忆、RAG/AIOps 编排和业务规则；核心层提供 LLM 工厂、错误分类、熔断、指标和数据库；Agent 层负责 LangGraph/LangChain 和 MCP；数据层保存会话、诊断任务、trace/span。RAG v2 的固定图是 rewrite → retrieve_each fan-out → dedup → generate → validate；AIOps 是 normalize → create task → SOP retrieve → LLM report → save report。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 18. 如何控制模型的幻觉问题？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:488`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 19. 如何观察模型召回了哪些块？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:502`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 20. 简历别的项目随便问了问主要工作

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:514`

**回答：**

当前仓库的真实链路可以概括为：FastAPI 接入层接收请求；服务层负责会话记忆、RAG/AIOps 编排和业务规则；核心层提供 LLM 工厂、错误分类、熔断、指标和数据库；Agent 层负责 LangGraph/LangChain 和 MCP；数据层保存会话、诊断任务、trace/span。RAG v2 的固定图是 rewrite → retrieve_each fan-out → dedup → generate → validate；AIOps 是 normalize → create task → SOP retrieve → LLM report → save report。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 21. 算法题：Leetcode 143. 重排链表

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:526`

**回答：**

这是算法题，项目中没有对应算法实现，不能硬套。回答时先明确不变量、边界和复杂度：双指针合并有序数组为 O(m+n)；合并区间先按左端点排序后线性扫描为 O(n log n)；LRU 用哈希表加双向链表使 get/put 平均 O(1)；Top-K 用容量 K 的小根堆为 O(n log K)；树形题用父子映射或 BFS。面试中应补充空输入、重复值、溢出和并发要求。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 22. 你平时调试代码怎么调试的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:579`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 23. 不对吧，最简单的方式不是打印一下吗？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:591`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 24. 最近了解的 AI 内容？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:601`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 25. 讲一下 Claude Code 架构

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:613`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 26. 建议下去再补一点 Harness 的知识

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:632`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 27. 反问业务

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:644`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

## 一面

### 1. 简单介绍一下自己的背景，以及选择AI Agent方向的原因。

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:681`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 2. 一个完整的Agent系统通常包含哪些核心模块？相比传统LLM Chatbot，它最大的区别是什么？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:691`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 3. 在Agent项目中，常见的规划（Planning）、记忆（Memory）、工具调用（Tool Use）和执行模块分别承担什么职责？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:712`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 4. 意图识别模块通常有哪些实现方式？规则匹配、小模型分类和LLM分类分别适用于哪些业务场景

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:723`

**回答：**

当前仓库的真实链路可以概括为：FastAPI 接入层接收请求；服务层负责会话记忆、RAG/AIOps 编排和业务规则；核心层提供 LLM 工厂、错误分类、熔断、指标和数据库；Agent 层负责 LangGraph/LangChain 和 MCP；数据层保存会话、诊断任务、trace/span。RAG v2 的固定图是 rewrite → retrieve_each fan-out → dedup → generate → validate；AIOps 是 normalize → create task → SOP retrieve → LLM report → save report。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 5. 如果使用大模型完成意图分类，如何选择Zero-shot、Few-shot方案？当标注数据较少时，如何提升分类稳定性？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:735`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 6. RAG系统从文档进入到最终生成答案的完整流程是什么？离线知识库构建和在线检索阶段分别包含哪些步骤？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:755`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 7. 文档切片有哪些常见策略？RecursiveCharacterTextSplitter的实现逻辑是什么？针对中文文档处理需要注意哪些问题？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:773`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 8. 如果RAG系统出现召回效果差的问题，你会如何定位？会优先检查Embedding、Chunk策略、Query Rewrite、Hybrid Search还是Rerank？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:783`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 9. Embedding模型如何选择？不同Embedding模型会对检索效果产生哪些影响？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:800`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 10. LangChain框架主要有哪些核心组件？相比传统Chain模式，LCEL带来了哪些改进？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:816`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 11. Function Calling和Tool Calling的执行流程是什么？模型是如何判断需要调用工具，以及生成对应参数的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:832`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 12. 如果Agent接入大量工具，如何避免工具描述过长导致Prompt膨胀？有哪些优化方式？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:850`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 13. Prompt一般如何设计和组织？System Prompt、Few-shot示例以及CoT通常分别承担什么作用？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:867`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 14. 如何降低大模型输出幻觉？除了Prompt约束之外，还有哪些工程优化方案？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:879`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 15. Agent执行任务时，如果出现重复调用工具、无法结束或者任务循环的问题，应该如何设计保护机制？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:895`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 16. Agent中的Memory通常如何实现？短期记忆和长期记忆分别解决什么问题？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:909`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 17. 如何设计一个记忆检索流程？历史信息应该如何筛选、存储和召回？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:919`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 18. 手撕代码：合并重叠区间（LeetCode 56），并分析算法复杂度

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:934`

**回答：**

这是算法题，项目中没有对应算法实现，不能硬套。回答时先明确不变量、边界和复杂度：双指针合并有序数组为 O(m+n)；合并区间先按左端点排序后线性扫描为 O(n log n)；LRU 用哈希表加双向链表使 get/put 平均 O(1)；Top-K 用容量 K 的小根堆为 O(n log K)；树形题用父子映射或 BFS。面试中应补充空输入、重复值、溢出和并发要求。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

## 一面

### 1. 自我介绍一下。

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:981`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 2. 介绍一下你这个___实习关注指标是什么？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:996`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 13. 说一下你这个项目Agent用在哪里。

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1025`

**回答：**

当前仓库的真实链路可以概括为：FastAPI 接入层接收请求；服务层负责会话记忆、RAG/AIOps 编排和业务规则；核心层提供 LLM 工厂、错误分类、熔断、指标和数据库；Agent 层负责 LangGraph/LangChain 和 MCP；数据层保存会话、诊断任务、trace/span。RAG v2 的固定图是 rewrite → retrieve_each fan-out → dedup → generate → validate；AIOps 是 normalize → create task → SOP retrieve → LLM report → save report。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 14. OpenManus 是用的什么框架？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1037`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 15. 讲一下你怎么理解Harness？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1056`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 16. 多 Agent 并发有什么性能问题？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1075`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 17. 你平时怎么控制 Agent 并发量？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1090`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 18. 讲一下长时记忆和短时记忆的底层实现方式？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1111`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 19. 讲一下 ReAct 的流程？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1125`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 20. Planner、Executor、Critic 分别怎么做的。

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1146`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 21. Critic什么意思？为什么不用ReAct？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1164`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 22. 上下文怎么做的？你是拆分的嘛？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1176`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 25. 讲一下Redis的数据结构。

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1197`

**回答：**

这道题的通用回答应围绕事务边界、并发控制、幂等键、状态机和可观测性展开。当前仓库确实使用 SQLAlchemy/Alembic 持久化会话、诊断任务、trace/span，也有 Redis/RQ 依赖；但没有库存调拨、Kafka 消费者或 PostgreSQL MVCC 业务实现。因此可以解释通用原理，例如用本地事务记录业务状态和幂等键、用 outbox/重试/补偿处理跨系统一致性，但必须明确：项目中没有对应实现，不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/core/database.py:1-33`、`/home/dong/projects/super_biz_agent_py/app/repositories/conversation_repository.py:1-156` — 数据库会话与仓储。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260629_0004_create_chat_run_traces.py:1-57`、`/home/dong/projects/super_biz_agent_py/migrations/versions/20260824_0009_create_chat_run_spans.py:1-85` — trace/span 表。
- `/home/dong/projects/super_biz_agent_py/app/core/task_queue.py:1-11` — RQ 队列接线，全文仅 `redis_conn`、`index_queue`、`protocol_pdf_queue` 三个模块级对象，不含任何重试或失败分类逻辑。

### 26. 每种的底层机理了解嘛？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1207`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 27. 讲一下java的类加载机制

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1225`

**回答：**

这道题属于 Java/系统基础，当前仓库是 Python 项目，项目中没有 Java 线程池、JVM 类加载、mmap/writev 或编译器实现，不能硬套。通用回答应区分 CPU 密集与 I/O 密集：进程提供隔离，线程共享地址空间，协程在用户态调度；线程池要有有界队列、拒绝策略、取消/超时/幂等语义和指标；JVM 类加载通常经历加载、验证、准备、解析、初始化。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 28. 场景题:  我们想要通过Agent实现对于广告投放的自我迭代，能够自我判别出哪种广告投放方式最优并执行，讲一下你的看法。

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1241`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

## 拼多多 AI Agent 岗两轮技术面经

### 1. 自我介绍，重点讲一下实习经历和 AI Agent 相关项目。

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1284`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 2. 介绍会议转写项目的完整流程，你在其中主要负责哪些模块？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1286`

**回答：**

项目当前有 PDF/Markdown 文档入库和 LLM 处理链路，但没有 OCR、ASR 或会议转写服务的实现。可以通用回答：音视频先分片/解码，ASR 生成带时间戳文本，术语词典和后处理纠错，再由 LLM 做摘要；OCR 则是图像预处理、文字检测识别、版面结构恢复和多模态模型补充。识别质量应看 CER/WER、术语召回和摘要事实一致性。项目中没有对应实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 3. 音视频流进入系统后，经过哪些步骤生成转写文本和会议纪要？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1288`

**回答：**

项目当前有 PDF/Markdown 文档入库和 LLM 处理链路，但没有 OCR、ASR 或会议转写服务的实现。可以通用回答：音视频先分片/解码，ASR 生成带时间戳文本，术语词典和后处理纠错，再由 LLM 做摘要；OCR 则是图像预处理、文字检测识别、版面结构恢复和多模态模型补充。识别质量应看 CER/WER、术语召回和摘要事实一致性。项目中没有对应实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 4. 为什么选择阿里云 ASR？对比过其他云服务或开源语音识别模型吗？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1290`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 5. ASR 的识别效果怎么评估？遇到专有名词、同音字和错别字时怎么处理？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1292`

**回答：**

评估 Agent 不能只看最终文本。应拆成任务成功率、答案正确性/证据支撑、工具选择正确率、参数正确率、无效步骤数、循环率、恢复率、P95/P99 延迟、Token/成本和安全违规率。当前仓库已经有 RAG 评估脚本与 golden set、节点 span、chat run trace、validation 字段和 Prometheus 指标；但没有题目中某个外部 benchmark 或线上业务转化数据，因此只能说明现有离线评估和观测能力，不能编造线上提升数字。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/instrumentation.py:1-180` — 节点级 instrumentation。
- `/home/dong/projects/super_biz_agent_py/app/core/span_context.py:1-208` — span 收集。
- `/home/dong/projects/super_biz_agent_py/app/repositories/chat_run_trace_repository.py:1-112` — trace 落库。
- `/home/dong/projects/super_biz_agent_py/app/core/metrics.py:1-197` — 指标。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:71-156` — 成功/失败 trace 与 span flush。

### 6. 视频通话、语音和转写文本涉及用户隐私，你们如何保证传输、存储和访问安全？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1294`

**回答：**

项目当前有 PDF/Markdown 文档入库和 LLM 处理链路，但没有 OCR、ASR 或会议转写服务的实现。可以通用回答：音视频先分片/解码，ASR 生成带时间戳文本，术语词典和后处理纠错，再由 LLM 做摘要；OCR 则是图像预处理、文字检测识别、版面结构恢复和多模态模型补充。识别质量应看 CER/WER、术语召回和摘要事实一致性。项目中没有对应实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 7. 系统采用中心化还是去中心化架构？当时为什么这样设计？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1296`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 8. 介绍一下项目中的 RAG 流程，离线建库和在线检索分别做了什么？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1298`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 9. 文档如何切块？Chunk 大小和重叠窗口是根据什么确定的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1300`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 10. 为什么使用当前的切块策略？是否通过 Recall[@K、MRR 等指标做过实验？](/users/1)

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1302`

**回答：**

评估 Agent 不能只看最终文本。应拆成任务成功率、答案正确性/证据支撑、工具选择正确率、参数正确率、无效步骤数、循环率、恢复率、P95/P99 延迟、Token/成本和安全违规率。当前仓库已经有 RAG 评估脚本与 golden set、节点 span、chat run trace、validation 字段和 Prometheus 指标；但没有题目中某个外部 benchmark 或线上业务转化数据，因此只能说明现有离线评估和观测能力，不能编造线上提升数字。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/instrumentation.py:1-180` — 节点级 instrumentation。
- `/home/dong/projects/super_biz_agent_py/app/core/span_context.py:1-208` — span 收集。
- `/home/dong/projects/super_biz_agent_py/app/repositories/chat_run_trace_repository.py:1-112` — trace 落库。
- `/home/dong/projects/super_biz_agent_py/app/core/metrics.py:1-197` — 指标。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:71-156` — 成功/失败 trace 与 span flush。

### 11. 短期记忆和长期记忆分别怎么设计？历史对话在什么情况下会被摘要或召回？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1304`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 12. Agent 的工具调用成功率是怎么定义的？项目最初的数据是多少，后来如何提升？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1306`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 13. 工具调用超时、返回脏数据或重复执行时，系统如何重试和兜底？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1308`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 14. 进程、线程和协程有什么区别？它们分别适合处理什么任务？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1310`

**回答：**

这道题属于 Java/系统基础，当前仓库是 Python 项目，项目中没有 Java 线程池、JVM 类加载、mmap/writev 或编译器实现，不能硬套。通用回答应区分 CPU 密集与 I/O 密集：进程提供隔离，线程共享地址空间，协程在用户态调度；线程池要有有界队列、拒绝策略、取消/超时/幂等语义和指标；JVM 类加载通常经历加载、验证、准备、解析、初始化。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 15. 已经有进程了，为什么还需要线程？进程创建和上下文切换为什么更重？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1312`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 16. 并发和并行有什么区别？多个任务看起来同时运行时，CPU 是否真的在并行执行？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1314`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 17. 举一个适合使用进程隔离的场景，再举一个适合使用多线程的场景。

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1316`

**回答：**

这道题属于 Java/系统基础，当前仓库是 Python 项目，项目中没有 Java 线程池、JVM 类加载、mmap/writev 或编译器实现，不能硬套。通用回答应区分 CPU 密集与 I/O 密集：进程提供隔离，线程共享地址空间，协程在用户态调度；线程池要有有界队列、拒绝策略、取消/超时/幂等语义和指标；JVM 类加载通常经历加载、验证、准备、解析、初始化。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 18. 手撕：实现 LRU Cache，要求 `get` 和 `put` 的时间复杂度都是 `O`。

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1318`

**回答：**

这是算法题，项目中没有对应算法实现，不能硬套。回答时先明确不变量、边界和复杂度：双指针合并有序数组为 O(m+n)；合并区间先按左端点排序后线性扫描为 O(n log n)；LRU 用哈希表加双向链表使 get/put 平均 O(1)；Top-K 用容量 K 的小根堆为 O(n log K)；树形题用父子映射或 BFS。面试中应补充空输入、重复值、溢出和并发要求。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 1. 自我介绍，并选择一个最熟悉的 Agent 项目做深入介绍。

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1322`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 2. 你怎么看当前 Agent 技术的发展？现阶段适合落地在哪些场景？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1324`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 3. 相比传统工作流，Agent 有哪些优势？生产环境中又存在哪些工程问题？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1326`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 4. Token 是如何划分的？为什么 `strawberry` 在部分模型中会被拆成多个 Token？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1328`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 5. 为什么大模型以 Token，而不是字符或完整单词作为基本处理单位？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1330`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 6. 讲一下 Transformer 和 Self-Attention，Q、K、V 分别有什么作用，具体如何计算？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1332`

**回答：**

Transformer 的核心思想是：让序列中每个位置都能根据任务需要，动态地关注其他位置，而不是像 RNN 一样按时间步串行传递状态。它由多层 Transformer Block 堆叠而成。一个典型 Block 包含多头注意力、前馈网络（MLP）、残差连接和 LayerNorm；大模型通常使用 Decoder-only 结构，也就是每个 Token 只能关注自己及左侧的历史 Token。

给定输入表示矩阵 `X ∈ R^(n×d_model)`，先通过三组可学习的线性变换得到：

```text
Q = XW_Q    K = XW_K    V = XW_V
Attention(Q, K, V) = softmax((QK^T / sqrt(d_k)) + Mask) V
```

- `Q（Query）` 是当前位置“想找什么信息”的查询。
- `K（Key）` 是每个候选位置“我能被怎样匹配”的索引。
- `V（Value）` 是匹配成功后真正要聚合的内容。
- `QK^T` 给出任意两个位置的相关性；除以 `sqrt(d_k)` 是为了抑制维度变大带来的点积方差，避免 softmax 过早饱和；softmax 后得到注意力权重，再对 `V` 做加权求和。
- Decoder 的 `causal mask` 会把未来位置设为不可见，保证训练时第 `t` 个 Token 不会偷看 `t+1` 之后的答案。

多头注意力并不是重复计算，而是把表示投影到多个子空间，让不同头分别学习不同关系，例如指代、语法依赖、关键词匹配或远距离依赖。各头输出拼接后再经过一次输出投影。注意力层负责 Token 间的信息交互，MLP 则在每个 Token 位置上做非线性特征变换；残差连接保证深层训练的梯度通路，LayerNorm 稳定激活分布。位置信息不是注意力天然具备的，必须通过绝对位置编码或 RoPE 一类相对位置机制注入。

面试中我会补充两个工程点。第一，标准全注意力的预填充（prefill）阶段时间和显存访问量随序列长度近似是 `O(n^2)`，长上下文会明显抬高延迟和成本。第二，自回归生成时会缓存历史 Token 的 `K/V`，即 KV Cache；新增一个 Token 时只计算新 Token 的 `Q/K/V` 并读取历史缓存，避免每步重复计算整个前缀。KV Cache 降低了解码阶段的重复计算，但它本身会随上下文长度、层数和头数线性增长，常成为并发场景的显存瓶颈。

可以用一句话收束：Transformer 用注意力完成全局信息路由，用多层 MLP 做特征变换；Q 决定“找什么”，K 决定“和谁匹配”，V 是“匹配后取回什么”。

**项目证据：**

- 这是大模型通用原理，当前仓库调用模型服务而不训练或实现 Transformer 内核，面试时不要把该原理表述为项目自研模型能力。
- `/home/dong/projects/super_biz_agent_py/app/core/llm_factory.py` — 项目中的模型接入层；可作为“应用如何使用模型”的证据，但不是 Transformer 训练实现。

### 7. 什么是 Lost in the Middle？长上下文中的重要信息为什么容易被模型忽略？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1334`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 8. 上下文窗口越大越好吗？它对显存、延迟、成本和模型效果有什么影响？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1336`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 9. 对话超过上下文限制后怎么处理？滑动窗口、摘要压缩和向量召回怎么选择？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1338`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 10. 简单介绍 SFT 的训练流程，并比较 PPO、DPO 和 GRPO 的区别。

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1340`

**回答：**

大模型微调的目的不是重新学习世界知识，而是在保留基座模型通用能力的前提下，让它更稳定地遵循某个领域的指令、输出格式、工具调用协议或任务偏好。实际流程通常是：明确目标和验收集，准备并清洗数据，选择微调方法，训练和监控，再通过离线评测与线上灰度验证闭环。

SFT（Supervised Fine-Tuning）是最基础的一步。数据一般是 `instruction + input -> target answer`，聊天模型则组织成多轮 messages，并只对 assistant 回复计算交叉熵损失。关键不在于“数据越多越好”，而在于任务定义清楚、输入输出一致、答案正确、去重且覆盖边界场景。训练前要按用户、文档或任务簇切分 train/validation/test，避免同一问题的改写版本同时出现在训练和测试中造成数据泄漏。

全参数微调会更新全部权重，表达能力强，但显存、训练成本和灾难性遗忘风险都更高。资源受限时更常用 PEFT：LoRA 冻结原权重 `W`，只训练低秩增量 `ΔW = BA`，推理时等效为 `W + BA`；`r` 是低秩维度，越大容量越强但参数和显存也越高，`alpha/r` 控制增量缩放，dropout 用于正则化。QLoRA 则把基座模型以 4-bit 量化方式加载，同时训练 LoRA adapter，进一步降低显存；它适合多数指令微调场景，但量化和训练配置仍需以验证集效果和吞吐来确认。

SFT 后如果目标是“更符合人类偏好、推理质量或业务评分规则”，可以做偏好优化或强化学习：

- `PPO`：先训练奖励模型，再让策略模型最大化奖励；通过 KL 约束限制模型偏离参考模型，并用 clipped objective 限制单步更新幅度。它表达力强、可接入可验证的环境奖励，但训练链路包含奖励模型、采样和优势估计，成本高且容易受 reward hacking、超参数和训练不稳定影响。
- `DPO`：使用同一 prompt 下的偏好对 `(chosen, rejected)`，直接优化模型相对参考模型更偏向 chosen 的概率，不需要显式训练奖励模型，也不需要在线 PPO roll-out。它实现和训练更简单、稳定性通常更好；前提是偏好对质量足够高，且它主要学习静态偏好，无法天然利用多步环境交互回报。
- `GRPO`：对同一 prompt 采样一组回答，用组内相对奖励构造优势，通常不再单独训练 value/critic 网络；尤其适合答案可由规则、单元测试或判题器验证的任务，例如数学、代码和结构化输出。它仍需要在线采样和奖励计算，训练成本高于 DPO，奖励函数设计依然是核心。

选择上，我会先用高质量 SFT 建立正确格式和基础任务能力；有高质量偏好对、希望低复杂度迭代时选 DPO；有可验证的结果奖励、需要优化多步决策或推理时再考虑 PPO/GRPO，其中 GRPO 对可批量验证的任务更有吸引力。不能只报 loss：离线至少看任务成功率、格式/工具调用合法率、拒答与安全边界、领域基准、幻觉率、延迟和单次成本；上线还要做灰度、对照组、人工抽检和可回滚版本管理。

面试里的诚实表述是：当前项目没有训练基座模型或 LoRA adapter，它主要通过 RAG、提示词、工具编排、会话记忆和检索评测来提升效果。因此这部分应回答为通用训练知识，再说明在本项目中会优先通过 RAG 补充易变知识；只有当问题稳定、高频、格式一致且 RAG/Prompt 仍无法解决时，才会考虑收集数据做 SFT 或 LoRA。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — 通过固定图编排检索和生成，而非训练模型参数。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 知识更新通过文档切分、向量化和索引进入 RAG 链路。
- `/home/dong/projects/super_biz_agent_py/tests/eval/README.md` — 项目已有离线检索评测基线；模型微调若落地，也应以独立验证集和可复现评测报告验收。

### 11. 为什么项目采用多 Agent？如果改成单 Agent，会遇到什么问题？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1342`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 12. 多个子 Agent 如何分工、分发任务和合并全局状态？为什么没有采用去中心化协作？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1344`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 13. 两个 Agent 给出冲突结论时如何仲裁？什么情况下需要人工介入？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1346`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 14. 了解哪些进程和内存监控工具？线上服务出现 CPU 飙高或内存持续增长时怎么排查？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1348`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 15. `mmap` 是什么？它解决了什么问题，适合哪些文件或进程通信场景？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1350`

**回答：**

这道题属于 Java/系统基础，当前仓库是 Python 项目，项目中没有 Java 线程池、JVM 类加载、mmap/writev 或编译器实现，不能硬套。通用回答应区分 CPU 密集与 I/O 密集：进程提供隔离，线程共享地址空间，协程在用户态调度；线程池要有有界队列、拒绝策略、取消/超时/幂等语义和指标；JVM 类加载通常经历加载、验证、准备、解析、初始化。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 16. `writev` 是什么？它如何通过聚合多个缓冲区减少系统调用开销？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1352`

**回答：**

这道题属于 Java/系统基础，当前仓库是 Python 项目，项目中没有 Java 线程池、JVM 类加载、mmap/writev 或编译器实现，不能硬套。通用回答应区分 CPU 密集与 I/O 密集：进程提供隔离，线程共享地址空间，协程在用户态调度；线程池要有有界队列、拒绝策略、取消/超时/幂等语义和指标；JVM 类加载通常经历加载、验证、准备、解析、初始化。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 17. 海量数据中找最大的 K 个数有哪些方案？为什么通常使用容量为 K 的小根堆，具体过程和复杂度是什么？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1354`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 18. 手撕：将扁平 JSON 列表恢复成树形结构，或者实现二叉树的锯齿形层序遍历。

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1356`

**回答：**

项目后端提供 FastAPI API 和 SSE 流式接口，前端流式状态有测试，但仓库没有完整 React Hook 实现、浏览器从 URL 到渲染的前端源码或懒加载框架代码。React Hook 题可以回答规则：Hook 必须在组件顶层按稳定顺序调用，条件分支会让链表位置错位；跨域要由服务端 CORS 中间件和凭据策略控制；URL 到页面通常经过 DNS、TCP/TLS、HTTP、HTML 解析、资源加载、脚本执行和渲染。以上是通用原理，不能说成当前项目已实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:159-260` — SSE 流式协议。
- `/home/dong/projects/super_biz_agent_py/tests/frontend/chat_v2_stream_state.test.js:1-35` — 前端流式状态测试。
- 项目后端仓库中没有 React Hook 链表、浏览器渲染管线或完整前端工程实现。

## 拼多多 - AI Agent 开发（提前批一面）

### 1. 介绍你的 Agent 代码生成链路全流程，长流程下如何解决上下文超限问题？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1367`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 2. 长流程任务的“断点恢复”能力你是怎么做的？服务重启后如何加载未完成状态？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1368`

**回答：**

项目把失败分成“可降级结果”和“彻底失败”两类。单次 LLM timeout 由 LLM 工厂注入，整条 chat_v2 还由 `asyncio.timeout(chat_total_budget_seconds)` 控制总预算；RagV2Service 保存最后一个状态快照，预算耗尽时返回已完成的子查询/文档和降级说明。并行检索分支把异常收进 `retrieve_failures`，因此一个分支失败可以变成 partial result，而不是整张图 500。异常分类集中在 `AppError`、`to_app_error`、`wrap_llm_exception` 和 `DegradeReason`；原始异常保留为 cause，应用异常提供稳定 code、retryable、http_status。熔断器只统计真正发出的下游调用，打开后快速失败并进入降级。幂等和 checkpoint 应由任务状态、唯一 request/task id、已完成步骤和副作用记录共同保证；当前仓库已实现诊断任务、trace/span 和部分 Worker 状态落盘，但没有 Kafka exactly-once 或完整跨服务事务，不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/core/errors.py:25-387` — `DegradeReason`、`AppError`、`to_app_error`、`wrap_llm_exception`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:6-78` — 单次调用超时与总预算、部分结果。
- `/home/dong/projects/super_biz_agent_py/app/core/circuit_breaker.py:1-260` — 熔断状态机。
- `/home/dong/projects/super_biz_agent_py/app/core/exception_handlers.py:90-170` — HTTP 错误响应。
- `/home/dong/projects/super_biz_agent_py/app/core/job_failure.py:68-128` — `default_job_retry`、`apply_retry_policy`：按 `AppError.retryable` 决定是否把 `retries_left` 清零。
- `/home/dong/projects/super_biz_agent_py/app/core/job_failure.py:146-183` — `record_job_failure`：独立 Session 写回失败态，自身永不抛异常。
- `/home/dong/projects/super_biz_agent_py/app/workers/index_worker.py:105-147` — except 块的固定顺序：先定重试策略、再拼消息、rollback、换 Session 写回、最后裸 `raise` 交还 RQ。
- `/home/dong/projects/super_biz_agent_py/app/core/task_queue.py:1-11` — 仅队列接线，不承担失败语义。

### 3. 多 Agent 运行机制是怎样的？如何防止并发

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1369`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

## 一面

### 1. 简单介绍一下自己的经历，以及为什么选择AI Agent方向？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1388`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 2. 平时开发过程中主要使用哪些AI Coding工具？比如Claude Code这类工具，通常有哪些使用习惯？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1403`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 3. 如何理解Spec Coding和Harness？你认为Harness为什么能够提升Agent完成复杂任务的能力？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1419`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 4. 如果让你从零开始设计一个Agent系统，你认为需要包含哪些关键模块？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1437`

**回答：**

我会把 Agent 看成“模型决策 + 工具执行 + 状态管理 + 验证治理”的系统，而不是一次 LLM 调用。当前仓库同时存在两条真实路径：`RagAgentService` 用 LangChain 的 `create_agent`、工具和 `MemorySaver`；`rag_v2` 用 LangGraph `StateGraph` 显式编排 rewrite、并行 retrieve、dedup、generate、validate。生产上选择 ReAct 还是 Plan-Execute，要看任务是否需要显式计划、可恢复步骤和人工审批；简单问答用固定图或单 Agent，长链任务用状态机更容易控制最大步数、超时、重试、checkpoint 和终止条件。项目没有完整多 Agent 协商系统，不能把多 Agent 经验硬套成已实现功能。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 5. Agent在执行任务时，如果调用外部工具或API出现超时、异常等情况，一般应该如何设计容错机制？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1460`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 6. 是否考虑过让大模型参与错误分析和自动恢复？这种方案有什么优缺点？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1478`

**回答：**

项目把失败分成“可降级结果”和“彻底失败”两类。单次 LLM timeout 由 LLM 工厂注入，整条 chat_v2 还由 `asyncio.timeout(chat_total_budget_seconds)` 控制总预算；RagV2Service 保存最后一个状态快照，预算耗尽时返回已完成的子查询/文档和降级说明。并行检索分支把异常收进 `retrieve_failures`，因此一个分支失败可以变成 partial result，而不是整张图 500。异常分类集中在 `AppError`、`to_app_error`、`wrap_llm_exception` 和 `DegradeReason`；原始异常保留为 cause，应用异常提供稳定 code、retryable、http_status。熔断器只统计真正发出的下游调用，打开后快速失败并进入降级。幂等和 checkpoint 应由任务状态、唯一 request/task id、已完成步骤和副作用记录共同保证；当前仓库已实现诊断任务、trace/span 和部分 Worker 状态落盘，但没有 Kafka exactly-once 或完整跨服务事务，不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/core/errors.py:25-387` — `DegradeReason`、`AppError`、`to_app_error`、`wrap_llm_exception`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:6-78` — 单次调用超时与总预算、部分结果。
- `/home/dong/projects/super_biz_agent_py/app/core/circuit_breaker.py:1-260` — 熔断状态机。
- `/home/dong/projects/super_biz_agent_py/app/core/exception_handlers.py:90-170` — HTTP 错误响应。
- `/home/dong/projects/super_biz_agent_py/app/core/job_failure.py:68-128` — `default_job_retry`、`apply_retry_policy`：按 `AppError.retryable` 决定是否把 `retries_left` 清零。
- `/home/dong/projects/super_biz_agent_py/app/core/job_failure.py:146-183` — `record_job_failure`：独立 Session 写回失败态，自身永不抛异常。
- `/home/dong/projects/super_biz_agent_py/app/workers/index_worker.py:105-147` — except 块的固定顺序：先定重试策略、再拼消息、rollback、换 Session 写回、最后裸 `raise` 交还 RQ。
- `/home/dong/projects/super_biz_agent_py/app/core/task_queue.py:1-11` — 仅队列接线，不承担失败语义。

### 7. Agent中的短期记忆和长期记忆分别有什么作用？通常会如何实现？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1498`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 8. 在Agent开发过程中，上下文为什么需要进行压缩？压缩后可能导致信息缺失，应该如何解决？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1510`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 9. Prompt优化除了添加规则约束之外，还有哪些常见的方法？你在实际使用中有哪些经验？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1531`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 10. 当模型出现幻觉问题时，一般有哪些解决思路？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1547`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 11. 如果Token消耗突然增加，你会如何定位原因并进行优化？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1563`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 12. 你有没有设计过Agent Skill？一个Skill从设计到上线通常需要考虑哪些因素？如何判断效果好坏？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1584`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 13. 介绍一下之前做过的相关项目，这个项目是完全自主开发，还是基于已有开源方案进行改造？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1602`

**回答：**

当前仓库的真实链路可以概括为：FastAPI 接入层接收请求；服务层负责会话记忆、RAG/AIOps 编排和业务规则；核心层提供 LLM 工厂、错误分类、熔断、指标和数据库；Agent 层负责 LangGraph/LangChain 和 MCP；数据层保存会话、诊断任务、trace/span。RAG v2 的固定图是 rewrite → retrieve_each fan-out → dedup → generate → validate；AIOps 是 normalize → create task → SOP retrieve → LLM report → save report。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 14. 在项目过程中遇到过哪些比较困难的问题？你主要负责哪些部分？最后取得了什么结果？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1618`

**回答：**

当前仓库的真实链路可以概括为：FastAPI 接入层接收请求；服务层负责会话记忆、RAG/AIOps 编排和业务规则；核心层提供 LLM 工厂、错误分类、熔断、指标和数据库；Agent 层负责 LangGraph/LangChain 和 MCP；数据层保存会话、诊断任务、trace/span。RAG v2 的固定图是 rewrite → retrieve_each fan-out → dedup → generate → validate；AIOps 是 normalize → create task → SOP retrieve → LLM report → save report。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 15. Java线程池的核心参数有哪些？线程提交任务后的执行流程是怎样的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1630`

**回答：**

工具调用不能只依赖模型“自觉”：应用先注册当前允许的工具和 schema，再校验调用名、类型、权限、资源状态、deadline、预算和副作用；工具返回结构化结果或结构化错误，模型再根据 Observation 决定下一步。当前仓库的 MCP 客户端提供 CLS/Monitor 等外部工具接入，并用拦截器做指数退避重试；RagAgentService 合并本地工具和 MCP 工具，MCP 不可用时降级为本地工具。当前仓库没有完整 Skill 目录协议或统一 policy engine，不能硬套已经实现。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/mcp_client.py:18-183` — MCP 客户端、指数退避重试、单例初始化。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:100-137` — 本地工具和 MCP 工具合并。
- `/home/dong/projects/super_biz_agent_py/app/tools/knowledge_tool.py:1-80` — 知识库工具。
- `/home/dong/projects/super_biz_agent_py/app/tools/time_tool.py:10-32` — 时间工具。

### 16. 如果让你重新设计一个线程池，你会重点考虑哪些问题？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1648`

**回答：**

这道题属于 Java/系统基础，当前仓库是 Python 项目，项目中没有 Java 线程池、JVM 类加载、mmap/writev 或编译器实现，不能硬套。通用回答应区分 CPU 密集与 I/O 密集：进程提供隔离，线程共享地址空间，协程在用户态调度；线程池要有有界队列、拒绝策略、取消/超时/幂等语义和指标；JVM 类加载通常经历加载、验证、准备、解析、初始化。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 17. JVM加载一个class文件的完整过程是什么？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1664`

**回答：**

这道题属于 Java/系统基础，当前仓库是 Python 项目，项目中没有 Java 线程池、JVM 类加载、mmap/writev 或编译器实现，不能硬套。通用回答应区分 CPU 密集与 I/O 密集：进程提供隔离，线程共享地址空间，协程在用户态调度；线程池要有有界队列、拒绝策略、取消/超时/幂等语义和指标；JVM 类加载通常经历加载、验证、准备、解析、初始化。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 18. MySQL中的undo log、redo log和binlog分别解决什么问题？它们之间有什么区别？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1680`

**回答：**

这道题的通用回答应围绕事务边界、并发控制、幂等键、状态机和可观测性展开。当前仓库确实使用 SQLAlchemy/Alembic 持久化会话、诊断任务、trace/span，也有 Redis/RQ 依赖；但没有库存调拨、Kafka 消费者或 PostgreSQL MVCC 业务实现。因此可以解释通用原理，例如用本地事务记录业务状态和幂等键、用 outbox/重试/补偿处理跨系统一致性，但必须明确：项目中没有对应实现，不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/core/database.py:1-33`、`/home/dong/projects/super_biz_agent_py/app/repositories/conversation_repository.py:1-156` — 数据库会话与仓储。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260629_0004_create_chat_run_traces.py:1-57`、`/home/dong/projects/super_biz_agent_py/migrations/versions/20260824_0009_create_chat_run_spans.py:1-85` — trace/span 表。
- `/home/dong/projects/super_biz_agent_py/app/core/task_queue.py:1-11` — RQ 队列接线，全文仅 `redis_conn`、`index_queue`、`protocol_pdf_queue` 三个模块级对象，不含任何重试或失败分类逻辑。

### 19. 什么是MySQL两阶段提交？为什么需要这个机制？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1692`

**回答：**

这道题的通用回答应围绕事务边界、并发控制、幂等键、状态机和可观测性展开。当前仓库确实使用 SQLAlchemy/Alembic 持久化会话、诊断任务、trace/span，也有 Redis/RQ 依赖；但没有库存调拨、Kafka 消费者或 PostgreSQL MVCC 业务实现。因此可以解释通用原理，例如用本地事务记录业务状态和幂等键、用 outbox/重试/补偿处理跨系统一致性，但必须明确：项目中没有对应实现，不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/core/database.py:1-33`、`/home/dong/projects/super_biz_agent_py/app/repositories/conversation_repository.py:1-156` — 数据库会话与仓储。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260629_0004_create_chat_run_traces.py:1-57`、`/home/dong/projects/super_biz_agent_py/migrations/versions/20260824_0009_create_chat_run_spans.py:1-85` — trace/span 表。
- `/home/dong/projects/super_biz_agent_py/app/core/task_queue.py:1-11` — RQ 队列接线，全文仅 `redis_conn`、`index_queue`、`protocol_pdf_queue` 三个模块级对象，不含任何重试或失败分类逻辑。

### 20. 算法题：合并两个有序数组

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1710`

**回答：**

这是算法题，项目中没有对应算法实现，不能硬套。回答时先明确不变量、边界和复杂度：双指针合并有序数组为 O(m+n)；合并区间先按左端点排序后线性扫描为 O(n log n)；LRU 用哈希表加双向链表使 get/put 平均 O(1)；Top-K 用容量 K 的小根堆为 O(n log K)；树形题用父子映射或 BFS。面试中应补充空输入、重复值、溢出和并发要求。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 21. 反问环节

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1738`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

## 面试问题汇总

### 1. 介绍一下自己的项目，整体架构和技术栈是什么？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1774`

**回答：**

当前仓库的真实链路可以概括为：FastAPI 接入层接收请求；服务层负责会话记忆、RAG/AIOps 编排和业务规则；核心层提供 LLM 工厂、错误分类、熔断、指标和数据库；Agent 层负责 LangGraph/LangChain 和 MCP；数据层保存会话、诊断任务、trace/span。RAG v2 的固定图是 rewrite → retrieve_each fan-out → dedup → generate → validate；AIOps 是 normalize → create task → SOP retrieve → LLM report → save report。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 2. 项目中 Agent 的完整流程是怎样的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1786`

**回答：**

当前仓库的真实链路可以概括为：FastAPI 接入层接收请求；服务层负责会话记忆、RAG/AIOps 编排和业务规则；核心层提供 LLM 工厂、错误分类、熔断、指标和数据库；Agent 层负责 LangGraph/LangChain 和 MCP；数据层保存会话、诊断任务、trace/span。RAG v2 的固定图是 rewrite → retrieve_each fan-out → dedup → generate → validate；AIOps 是 normalize → create task → SOP retrieve → LLM report → save report。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 3. 项目过程中针对 Agent 做过哪些优化？具体怎么调优？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1798`

**回答：**

当前仓库的真实链路可以概括为：FastAPI 接入层接收请求；服务层负责会话记忆、RAG/AIOps 编排和业务规则；核心层提供 LLM 工厂、错误分类、熔断、指标和数据库；Agent 层负责 LangGraph/LangChain 和 MCP；数据层保存会话、诊断任务、trace/span。RAG v2 的固定图是 rewrite → retrieve_each fan-out → dedup → generate → validate；AIOps 是 normalize → create task → SOP retrieve → LLM report → save report。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 4. Agent 优化有哪些常见手段？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1816`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 5. 基于项目设计一个场景，如果遇到类似问题应该怎么解决？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1831`

**回答：**

当前仓库的真实链路可以概括为：FastAPI 接入层接收请求；服务层负责会话记忆、RAG/AIOps 编排和业务规则；核心层提供 LLM 工厂、错误分类、熔断、指标和数据库；Agent 层负责 LangGraph/LangChain 和 MCP；数据层保存会话、诊断任务、trace/span。RAG v2 的固定图是 rewrite → retrieve_each fan-out → dedup → generate → validate；AIOps 是 normalize → create task → SOP retrieve → LLM report → save report。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/graph.py:17-53` — `_fanout_to_retrieve`、`build_rag_v2_graph`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/state.py:10-34` — `RAGState`。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:77-137` — `RagAgentService`、`create_agent`、`MemorySaver`。
- `/home/dong/projects/super_biz_agent_py/pyproject.toml:12-32` — LangChain、LangGraph、MCP 依赖。

### 6. 语义检索是如何实现的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1847`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 7. 向量数据库在项目中具体承担什么作用？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1863`

**回答：**

当前项目的 RAG 链路是：文档切分和向量化后写入 Milvus；在线请求先由 rewrite 节点生成互补子查询，再用 LangGraph `Send` 并行检索，fan-in 后按文档 id 或内容指纹去重，最后交给生成和答案校验节点。检索失败、检索为空和部分分支失败被区分为不同状态，不能把“服务挂了”说成“知识库没有文档”。如果题目问 RRF、BM25、Cross-Encoder 或特定行业 RAG，通用回答可以说明它们分别解决候选融合和排序问题，但当前仓库的 `rag_v2` 代码证据主要是向量检索、并行分支、去重和验证；项目中没有实现的算法不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:91-211` — Query Rewrite、并行检索和分支失败收集。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/nodes.py:214-390` — 去重、生成、答案校验。
- `/home/dong/projects/super_biz_agent_py/app/services/vector_search_service.py:1-228` — 向量检索服务。
- `/home/dong/projects/super_biz_agent_py/app/services/document_splitter_service.py:1-177` — 文档切分。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:47-179` — 总预算、部分结果和流式状态。

### 8. langchain、langgraph 这些框架为什么选择？底层流程了解吗？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1873`

**回答：**

这类问题涉及个人履历或外部团队事实，不能把仓库代码冒充成我的个人经历。基于当前项目可以诚实回答：我参与/维护的是一个 Python + FastAPI 的企业智能对话与 AIOps 项目，重点工作集中在 RAG、LangGraph 编排、MCP 工具接入、会话记忆、统一错误治理和可观测性。回答时应把“我亲自负责的模块、实际改动、验证方式和结果”替换为真实经历；没有真实数字就明确说项目仍处于离线验证或灰度阶段。对字节内部工具、个人使用过的 IDE/Agent 产品，项目中没有实现，不能硬套。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 9. 模型输出结果如何控制规则？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1883`

**回答：**

基于当前仓库能确认的回答是：系统以 FastAPI 为入口，围绕 LangChain/LangGraph、Milvus、MCP、SQLAlchemy/Alembic 和 RQ 构建企业智能对话与 AIOps 能力。题目若涉及仓库没有的业务、个人经历或外部平台，项目中没有对应实现，不能硬套；应明确区分“项目已实现”“项目设计文档讨论过”和“通用方案”。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

### 10. 记忆模块是怎么设计的？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1895`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 11. 多轮、多会话场景下 memory 如何处理？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1909`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 12. 如果系统出现异常，整体的容错和异常处理机制怎么设计？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1919`

**回答：**

项目把失败分成“可降级结果”和“彻底失败”两类。单次 LLM timeout 由 LLM 工厂注入，整条 chat_v2 还由 `asyncio.timeout(chat_total_budget_seconds)` 控制总预算；RagV2Service 保存最后一个状态快照，预算耗尽时返回已完成的子查询/文档和降级说明。并行检索分支把异常收进 `retrieve_failures`，因此一个分支失败可以变成 partial result，而不是整张图 500。异常分类集中在 `AppError`、`to_app_error`、`wrap_llm_exception` 和 `DegradeReason`；原始异常保留为 cause，应用异常提供稳定 code、retryable、http_status。熔断器只统计真正发出的下游调用，打开后快速失败并进入降级。幂等和 checkpoint 应由任务状态、唯一 request/task id、已完成步骤和副作用记录共同保证；当前仓库已实现诊断任务、trace/span 和部分 Worker 状态落盘，但没有 Kafka exactly-once 或完整跨服务事务，不能硬套。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/core/errors.py:25-387` — `DegradeReason`、`AppError`、`to_app_error`、`wrap_llm_exception`。
- `/home/dong/projects/super_biz_agent_py/app/agent/rag_v2/service.py:6-78` — 单次调用超时与总预算、部分结果。
- `/home/dong/projects/super_biz_agent_py/app/core/circuit_breaker.py:1-260` — 熔断状态机。
- `/home/dong/projects/super_biz_agent_py/app/core/exception_handlers.py:90-170` — HTTP 错误响应。
- `/home/dong/projects/super_biz_agent_py/app/core/job_failure.py:68-128` — `default_job_retry`、`apply_retry_policy`：按 `AppError.retryable` 决定是否把 `retries_left` 清零。
- `/home/dong/projects/super_biz_agent_py/app/core/job_failure.py:146-183` — `record_job_failure`：独立 Session 写回失败态，自身永不抛异常。
- `/home/dong/projects/super_biz_agent_py/app/workers/index_worker.py:105-147` — except 块的固定顺序：先定重试策略、再拼消息、rollback、换 Session 写回、最后裸 `raise` 交还 RQ。
- `/home/dong/projects/super_biz_agent_py/app/core/task_queue.py:1-11` — 仅队列接线，不承担失败语义。

### 13. 再给一个上下文处理相关的场景题，如何优化 context 管理？

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1931`

**回答：**

项目采用“当前请求上下文 + 有界会话记忆 + 按需知识证据”的思路，而不是把全部历史消息直接塞给模型。旧 Agent 有消息裁剪逻辑；chat_v2 在进入图前通过会话记忆服务构造上下文；rag_v2 的 prompt 明确区分会话记忆和知识库证据，禁止把历史指标写成实时事实。超过预算时应优先保留系统规则、当前目标、待办、关键证据和最近交互，把已完成步骤压成带来源的摘要，原文留在数据库以便回放或按需召回。

**项目证据：**

- `/home/dong/projects/super_biz_agent_py/app/services/conversation_memory_service.py:1-292` — 会话记忆、摘要和 Token 预算。
- `/home/dong/projects/super_biz_agent_py/app/services/rag_agent_service.py:32-74` — 消息裁剪。
- `/home/dong/projects/super_biz_agent_py/app/api/chat_v2.py:28-35,71-109` — 记忆加载与会话落库。
- `/home/dong/projects/super_biz_agent_py/migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 记忆快照表。

### 14. 手撕算法：Greedy（贪心相关）

**原文位置：** `/home/dong/projects/super_biz_agent_py/docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:1945`

**回答：**

这是算法题，项目中没有对应算法实现，不能硬套。回答时先明确不变量、边界和复杂度：双指针合并有序数组为 O(m+n)；合并区间先按左端点排序后线性扫描为 O(n log n)；LRU 用哈希表加双向链表使 get/put 平均 O(1)；Top-K 用容量 K 的小根堆为 O(n log K)；树形题用父子映射或 BFS。面试中应补充空输入、重复值、溢出和并发要求。

**项目证据：**

- 项目中没有对应实现，不能硬套。
- 可引用的仓库范围是 `/home/dong/projects/super_biz_agent_py/README.md:1-22` 与 `/home/dong/projects/super_biz_agent_py/pyproject.toml:1-50`，只能说明当前项目的 Python/FastAPI/LangChain/LangGraph/Milvus/MCP 技术栈。

---

**生成校验：** 已按原文顺序生成 238 道题。
