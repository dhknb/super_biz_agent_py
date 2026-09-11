# Agent 面经逐题回答 v2

> 原题来源：[[agent_interviews_2026-06-06_to_2026-09-06]]
>
> **与 v1 的区别**：v1（`_answers.md`）238 道题只有 15 段独立正文，99% 的题落在 14 个复用模板里——关键词匹配填充，连「React hook 使用要注意什么」这道前端题都被套上了 Agent 架构模板。v2 每题独立作答，不复用段落。
>
> **写作规则**
> 1. 按原文件篇章与题目顺序推进，重复题分别作答，不合并
> 2. 每题：题目 → 基于本仓库的回答 → 项目证据（`路径:行号`）
> 3. 仓库里没有的，写明「项目中没有，不能硬套」，并说清为什么不能拿相邻模块顶替
> 4. 假设性内容标 `[假设]`
> 5. 原文件标记未公开的部分不补造
>
> **本仓库真实技术栈**（原题文件里的描述与此不符，答题以代码为准）
>
> | 项 | 原题文件声称 | 本仓库实际 | 证据 |
> |---|---|---|---|
> | Embedding | BGE-large-zh-v1.5 | DashScope `text-embedding-v4`，1024 维 | `app/services/vector_embedding_service.py:171-176` |
> | 向量索引 | IVF_FLAT + L2 | HNSW + COSINE，`M=16, efConstruction=200` | `app/core/milvus_client.py:192-208` |
> | 重排 | BGE reranker，Top-20→Top-5 | **无 reranker**，`RETRIEVE_TOP_K=4`/`FINAL_TOP_K=6` | `app/agent/rag_v2/nodes.py:28-33` |
> | AIOps 架构 | Plan-Execute-Replan | **已删除**，现为固定编排管线 | `app/agent/aiops/` 全部 deleted |

---

## 一、字节跳动 AI Agent 开发岗-03

> 来源：`https://www.nowcoder.com/discuss/922659809084571648`
> 本篇共 10 份面经、51 道题。面经 18 原文为空，按未公开处理。

### 面经 01 · 2026-08-17

#### Q1 · langchain 和 langgraph 的区别和应用是什么

**回答**

一句话的区别：**LangChain 给你零件和一条直路，LangGraph 给你一张可以带环、带分支、能存档的状态机。** 前者的控制流藏在框架里，后者的控制流是你自己写出来的。

再说细一点。LangChain 的核心资产是两样东西：一套组件抽象（模型、Prompt 模板、输出解析器、检索器、工具、记忆），和一套把组件串起来的表达式语言 LCEL。LangGraph 的核心资产只有一样：一个图执行引擎。所以它们不是竞品，是**上下层**——LangGraph 负责编排，LangChain 的组件是被编排的对象。

判断该用哪个，我只看一个问题：**这次的控制流，是"一条直线走完"，还是"要根据中间结果决定下一步、可能要绕回去"？**

- 直线：检索 → 拼 Prompt → 调模型 → 解析输出。LCEL 一个 `|` 串完，别上 LangGraph，纯属加复杂度。
- 要绕回去、要并行扇出、要在中途停下等人确认、要断点续跑、要"某一路挂了但整体还得出结果"——这些 LCEL 表达不了，必须 LangGraph。

**知识展开一：LCEL 为什么表达不了 Agent 循环**

LCEL（LangChain Expression Language）是 LangChain 的管道语法，长这样：

```python
chain = prompt | model | parser
result = chain.invoke({"question": "..."})
```

能被 `|` 串起来的东西统称 **Runnable**，它们共享同一套接口：`invoke`（单条同步）、`ainvoke`（单条异步）、`stream`（流式）、`batch`（批量）。这套统一接口是 LCEL 最大的价值——你写完 `prompt | model | parser`，自动就获得了流式、批量、异步三种调用方式，不用自己实现。

但 LCEL 拼出来的结构是 **DAG（有向无环图）**。注意"无环"这三个字：数据从左往右流一遍就结束，没有任何语法能表达"回到上一步再来一次"。

而 Agent 的本质就是一个环：

```
模型看状态 → 决定调哪个工具 → 执行工具 → 结果塞回状态 → 模型再看状态 → ...
```

这个循环要跑几轮，**写代码的时候不知道**，取决于模型当时怎么想。LCEL 的 `|` 是编译期就定死的固定管道，装不下运行期才决定的循环。这就是为什么 LangChain 早期要单独造一个 `AgentExecutor`——那玩意儿内部就是一个手写的 `while` 循环，跳出了 LCEL 体系。

LangGraph 的做法是把"下一步去哪"变成图上的一等公民：`add_conditional_edges(源节点, 路由函数, 可能的目标)`。路由函数在**运行时**读当前状态返回目标节点名，所以 `agent → tools → agent` 这种环是天然合法的。

**知识展开二：Pregel / BSP 执行模型，以及"super-step 失败"到底是什么**

这一层是 LangGraph 最容易被跳过、但面试一追问就露馅的地方。

LangGraph 的执行引擎抄的是 Google Pregel 论文的思路，学名 **BSP（Bulk Synchronous Parallel，整体同步并行）**。跑一次图不是"一个节点一个节点顺序执行"，而是分成一轮一轮的 **super-step（超级步）**，每一轮干三件事：

1. **执行**：把这一轮所有被激活的节点**并发**跑掉（异步节点走事件循环，同步节点走线程池）
2. **合并**：等这一轮全部跑完，收集每个节点返回的 patch（增量补丁），用 reducer 统一合并进共享状态
3. **调度**：根据新状态和图上的边，算出下一轮该激活谁；没人可激活就结束

关键在第 2 步的"**等这一轮全部跑完**"——这是一道**同步屏障（barrier）**。屏障的直接后果就是：

> **同一个 super-step 里任何一个节点抛出异常，这一轮就整体失败，已经跑完的兄弟节点的成果一起丢掉。**

不是"失败的那个分支返回 None，其他照常"，是**整批作废**。因为状态合并是一次原子操作，引擎不会给你一个"合并了 3/4 个 patch"的中间态——那样状态就不一致了。

这条语义是所有"并行检索容错"设计的起点。想让一路失败不拖垮全局，唯一的办法是**别让异常逃出节点**：节点内部 try/except 兜住，把失败转换成一条**正常返回值里的失败记录**。异常一旦被转成数据，引擎眼里这个节点就是成功的，屏障不会触发，其他分支的成果全部保留。

这套东西有个通用名字叫**错误具体化（reify errors）**：把控制流层面的"异常"，降级成数据层面的"字段"。写业务系统时到处都用得上，不只是 LangGraph。

**知识展开三：reducer，以及并行写同一字段为什么会报错**

LangGraph 的状态一般用 `TypedDict` 声明。默认合并规则是**覆盖**（last-write-wins）：

```python
class State(TypedDict):
    answer: str          # 后写的覆盖先写的
```

覆盖在串行链路上没问题，但并行分支同时写同一个字段就出事了——两个分支各返回一个 `documents`，引擎不知道该留哪个，直接抛 `InvalidUpdateError`。

解法是给字段挂一个 **reducer**，用 `Annotated` 声明"这个字段该怎么合并"：

```python
from typing import Annotated
import operator

class State(TypedDict):
    documents: Annotated[list[Document], operator.add]   # 追加合并
```

`operator.add` 对 list 就是 `+`，所以 N 个分支各返回一段 list，合并结果是全部拼起来。reducer 可以是任意二元函数，你也能自己写（比如"取分数最高的那个"、"字典深度合并"）。

一个容易被忽略的推论：**reducer 是"是不是并行节点"的判据**。看到 `Annotated[..., operator.add]` 就知道这个字段是 fan-in（多路汇入）来的；没挂 reducer 的字段，如果哪天被并行节点写了，会立刻炸出来——这其实是个好性质，它让"同一个节点被意外跑了两次"这种 bug 早暴露，而不是静默地多出一份数据。

**知识展开四：Send，动态 fan-out**

普通的边是**静态**的：写代码时就定死了 A 之后去 B。但"把用户问题拆成 3 个子查询，每个子查询并行检索一次"这种需求，分支数是**运行时**才知道的（模型今天拆出 3 个、明天拆出 5 个）。

`Send` 就是干这个的：

```python
def fanout(state) -> list[Send]:
    return [Send("retrieve", {"query": q}) for q in state["sub_queries"]]

graph.add_conditional_edges("rewrite", fanout, ["retrieve"])
```

`Send(节点名, 载荷)` 的语义是"创建一个该节点的实例，并且**只**把这份载荷喂给它"——注意每个实例看到的输入是自己的 payload，不是完整的全局状态。N 个 `Send` 就是 N 个并发实例，全部落在同一个 super-step 里。

这就是 **map-reduce** 模式：`Send` 是 map，下游第一个汇聚节点是 reduce。汇聚节点有个特殊地位——**它是整条链路上唯一能看到全部分支结果的位置**，所以全局去重、全局重排、全局截断这类操作只能放在那里，放在 map 阶段每个分支只看得见自己那一份。

**知识展开五：checkpointer，以及它买来的三个能力**

`graph.compile()` 可以传一个 `checkpointer`。传了之后，**每个 super-step 结束时**引擎会把当前状态快照存一份，用 `thread_id` 做隔离键（一个 thread_id 就是一条会话）。

存档不是为了好看，它换来三样在生产里很硬的能力：

1. **多轮记忆**。下一轮请求带同一个 `thread_id` 进来，引擎自动把历史状态加载回来。不传 checkpointer 的图是**无状态**的，每次调用都从零开始——这时候多轮对话的上下文得你自己在图外面拼。
2. **人工介入（HITL）**。可以在指定节点前 `interrupt`，图停在那儿，状态已落盘；人在界面上确认或改一改，再 `resume` 继续。没有存档就没法"停下来等人"，因为进程一返回，内存里的状态就没了。
3. **断点续跑与时间旅行**。跑挂了可以从最后一个成功的 super-step 重来，不用从头；也可以回放到任意历史检查点重新分叉（debug 神器）。

实现上分内存和持久两类：`MemorySaver` 存进程内存（重启即失、多副本部署不共享，只适合单机和开发），`SqliteSaver` / `PostgresSaver` 存数据库（可跨重启、可多副本共享）。

反过来说，**不传 checkpointer 是一个可以理直气壮的选择**：如果这张图是"一个请求进来跑完就出结果"的单轮流水线，本身不需要中途暂停，那存档就是纯开销——每步一次写库，还得管过期清理。上下文记忆可以在进入图之前拼好、当作输入字段传进去，比让引擎存全量状态轻得多。

**知识展开六：一个容易答错的点——`create_agent` 其实是 LangGraph**

面试里如果把这题答成"LangChain 是老的、LangGraph 是新的替代品"，就有点危险了，因为现在 LangChain 里那个一行建 Agent 的 `create_agent`（前身 `langgraph.prebuilt.create_react_agent`），**底层就是一张预先搭好的 LangGraph 图**。

所以更准确的表述是：

> 不是"LangChain vs LangGraph"，是"**用别人搭好的图** vs **自己搭图**"。

用预建图的代价不是性能，是**扩展点只有框架留的那几个**。想改上下文怎么裁、想在工具调用前加审计、想让某一路失败不拖垮全局——你只能在框架给的中间件钩子里做手脚，钩子没覆盖到的地方就无从下手。自己搭图则是每条边都在你手里，代价是所有边界条件都得自己想清楚。

**结合项目**

这个仓库里两条路径同时活着，正好是实物对照。

预建图那条在 `app/services/rag_agent_service.py:133-137`，一句 `create_agent(self.model, tools=all_tools, checkpointer=self.checkpointer)` 就把整个 ReAct 循环包完了，工具选择、调用、观察、再决策全在框架内部。它的代价体现在 `app/services/rag_agent_service.py:37-74` 的 `trim_messages_middleware`：想控制上下文，只能在框架留的中间件钩子上做手脚——保住首条 System 消息加最近 6 条，靠 `RemoveMessage(id=REMOVE_ALL_MESSAGES)` 全删再重放。这是在框架缝隙里干活，不是在设计流程。

自己搭图那条在 `app/agent/rag_v2/graph.py:22-51`，五个节点的边是手写的：`rewrite → retrieve_each（并行）→ dedup → generate → validate_answer`。差别不在能不能跑通，在于**故障边界能不能自己划**。

`app/agent/rag_v2/graph.py:17-19` 的 `_fanout_to_retrieve` 就是上面讲的 `Send` 扇出。而 super-step 屏障那条语义带来的坑，`app/agent/rag_v2/nodes.py:161-169` 的注释里记着——改造前只要 Milvus 抖一下，哪怕另外 3 个分支已经召回了充足证据，用户拿到的仍然是 500。

所以 `retrieve_each_node` 在 `app/agent/rag_v2/nodes.py:189-205` 把异常全兜住，返回一条 `retrieve_failures` 明细而不是抛出去，也就是前面说的"错误具体化"。对应的字段声明在 `app/agent/rag_v2/state.py:27`：`Annotated[List[Dict], operator.add]`，和 `documents` 用同一套 fan-in 语义。语义从此变成"一个分支失败 = 证据少一份"，而不是"整个请求失败"。

这张图 `compile()` 时**没传 checkpointer**（`graph.py:51`），理由就是知识展开五最后那段：它是单轮流水线，会话记忆在进图之前就拼成 `conversation_context` 字段传进来了（`state.py:15`）。

**项目证据**

- `app/services/rag_agent_service.py:133-137` — `create_agent` + `MemorySaver`
- `app/services/rag_agent_service.py:37-74` — `trim_messages_middleware`，保 System + 最近 6 条
- `app/agent/rag_v2/graph.py:17-19` — `_fanout_to_retrieve`，`Send` 扇出
- `app/agent/rag_v2/graph.py:22-51` — `build_rag_v2_graph`，显式节点与边
- `app/agent/rag_v2/nodes.py:161-169` — super-step 失败语义的注释
- `app/agent/rag_v2/nodes.py:189-205` — 分支异常兜底，返回失败明细

#### Q2 · AI运营工具如何提效的，主要做了什么

**回答**

这类"你做的东西怎么提效"的问题，面试官想听的不是功能清单，是**你有没有把工作量化过**。答"我做了个自动化工具很方便"等于没答。答题结构应该是：原来的人工动线是什么 → 哪几步被代码接走了 → 用什么指标证明它真的省了事 → 哪些地方反而没省（这一条最能加分，因为它说明你测过）。

**知识展开：AI 提效类项目该报什么指标**

这块先讲通用方法，因为它是可迁移的。AI 提效项目的指标一般分四层，从硬到软：

1. **时延类**：首次响应时间（从请求进来到用户看到第一个有效输出）、端到端完成时间。这是最硬的，因为可直接测。要注意报的是**分位数**而不是平均值——平均值会被少数超时样本拖歪，而用户体感取决于 P90/P99。
2. **人工动作数**：原来要点几次、切几个系统、复制粘贴几回，现在要几次。这个指标土但极有说服力，因为它可以数出来。
3. **一次通过率**：模型给的结果，人不用改就能直接用的比例。这条最能暴露真实价值——如果每次都要人大改，时延省下来的时间全在返工里还回去了。
4. **质量类**：准确率、覆盖率、幻觉率。这层最软，因为需要标注集才能测，而标注集本身就是成本。

还有一个反向指标很多人不报，但恰恰是成熟度的标志：**降级率**。系统在多少比例的请求里没能给出完整结果、退化成了保守输出。不报这个数的"提效"是不可信的，因为你不知道那 100% 的自动化里有多少是空壳。

**知识展开：为什么"撒谎式降级"是提效项目最大的隐性成本**

这个概念值得单独讲，它是我在这个项目里踩过的坑，也是所有 RAG / Agent 系统的通病。

假设你的检索服务挂了。最省事的写法是 `except: return []`——返回空列表，链路继续跑，用户拿到一个"没有找到相关资料"的答复。表面上很优雅：没报错、没 500、有输出。

问题是这个输出**在撒谎**。"检索服务挂了"和"知识库里确实没有这条"是两件完全不同的事，但它们产生了一模一样的用户可见结果。后果是：

- 值班人员看到"没找到 SOP"，会去补文档——补完还是没用，因为服务还是挂的
- 监控上一切正常，因为没有异常抛出，成功率 100%
- 真正的故障可以持续几天没人发现

修法就是**三态而非两态**：不要用"有结果 / 空结果"表达，要用"成功 / 失败 / 确实为空"三个状态显式返回，让调用方能区分。这在工程上叫**让失败可见**，它和"提效"是一体的——一个会撒谎的自动化系统，长期看是负提效。

**结合项目**

这个项目不是运营工具，是 AIOps 首响助手（告警进来，自动给值班人员一份结构化排查报告）。按上面的框架讲：

**原人工动线四步**：从告警平台复制上下文 → 去 wiki 翻对应 SOP → 按 SOP 逐条查日志和监控 → 把结论手写成工单。

**现在这四步串成一条代码路径**：`app/services/alert_diagnosis_orchestrator.py:71-131` 里是归一化告警 → 建诊断任务 → 记录 `PLANNING` 阶段 → 调 `analyze` → 存报告 → 标 `DONE`。

**指标侧只有时延是真测的**：`app/services/first_response_service.py:70-76` 的 `_elapsed_ms` 用 `perf_counter` 而不是两个 `datetime` 相减——后者受 NTP 校时影响，机器对一次时间就可能算出负数耗时，那个数会静默污染整条耗时曲线。

**降级率这条是真做了的**：`app/core/metrics.py:157-161` 的 `degrade_total` 按原因分标签计数，`app/agent/rag_v2/instrumentation.py:149-171` 是全项目降级判定的唯一汇聚点，所以新增节点会自动被统计，不会有人忘记埋点。

**"撒谎式降级"在这个项目里是被显式修掉的**：`app/services/sop_retrieval_service.py` 返回的是 `(evidence, RetrievalStatus)` 二元组而不是裸列表，`app/core/errors.py:65-78` 的 `RetrievalStatus` 就是上面说的三态。`app/services/first_response_service.py:319-348` 的 `_apply_retrieval_status` 只在状态为 `FAILED` 时改写报告，并把警告写进 `risk_notes` 让人看见——区别就是值班人员会去修服务，而不是去补文档。

**没测的部分要承认**：一次通过率、人工动作数的前后对比、准确率，这些需要真实值班数据和标注集，项目中没有。DAU、转化率这类运营指标更是不沾边，不能硬套。

**项目证据**

- `app/api/aiops.py:56-88` — `analyze_alert` 入口
- `app/services/alert_diagnosis_orchestrator.py:71-131` — `run`，任务创建与阶段流转
- `app/services/alert_diagnosis_orchestrator.py:50-53` — `_PHASE_STATUS`，只有 RETRIEVING/DIAGNOSING 是真状态迁移
- `app/services/first_response_service.py:155-294` — `analyze`，SOP 检索 + 结构化报告
- `app/services/first_response_service.py:319-348` — `_apply_retrieval_status`
- `app/services/first_response_service.py:70-76` — `_elapsed_ms`
- `app/services/sop_retrieval_service.py:68-167` — `retrieve_sop_evidence`，返回 `(evidence, RetrievalStatus)`

#### Q3 · 字节用什么工具开发

**回答**

这题问的是外部团队事实，仓库代码答不了，也不该拿代码冒充经历。

能诚实说的是自己这边的工具链：Python 3 + FastAPI + uv（`uv.lock` 在仓库里）、SQLAlchemy 2.x DeclarativeBase、Alembic 迁移（`migrations/versions/` 下 9 个版本，从 `20260612_0001` 到 `20260824_0009`）、Milvus 向量库、Redis + RQ 异步队列、Prometheus 指标、pytest + pre-commit。

字节内部的 Harness 框架、豆包相关工具链，项目中没有，不能硬套。

**项目证据**

- 项目中没有对应实现，不能硬套
- 可如实引用的范围：`pyproject.toml`、`uv.lock`、`migrations/versions/`（9 个迁移版本）、`.pre-commit-config.yaml`、`Makefile`

---

### 面经 02 · 2026-08-10

#### Q1 · 为什么选择 AI 应用开发方向？

**回答**

这题是个人动机，仓库代码不能替我回答。但选择的理由可以用这个仓库里最花时间的那部分来说明——不是模型调用，是模型不可靠时系统怎么办。

`app/core/errors.py` 397 行里几乎没有一行是「调 LLM」，全是在回答一件事：模型超时、返回垃圾、下游挂掉的时候，这个系统对外说什么。`DegradeReason`（`app/core/errors.py:25-62`）那个枚举的划分标准是**修复方向**——`RETRIEVAL_FAILED` 和 `RETRIEVAL_EMPTY` 必须分开，因为一个是去修服务，一个是去补文档。

AI 应用开发吸引我的正是这一层：传统后端的不确定性来自网络和并发，是有限的；LLM 把不确定性搬进了业务逻辑内部。用确定性的工程手段兜住一个不确定的核心，这件事比调 prompt 有意思得多。

**项目证据**

- `app/core/errors.py:25-62` — `DegradeReason`，划分标准是修复方向
- `app/core/errors.py:65-78` — `RetrievalStatus`，三态消灭静默空返回
- `app/core/errors.py:81-124` — `AppError` 基类

#### Q2 · 平时如何学习大模型及 Agent 领域的前沿技术？

**回答**

这题问习惯，但仓库里有两处能证明学习方式不是「读博客」而是「读源码」。

第一处，`app/core/llm_factory.py` 的模块文档记录了一次真实排查：langchain-openai 会无条件把 `request_timeout`（默认 `None`）塞进 OpenAI SDK client，覆盖掉 SDK 自己的 `DEFAULT_TIMEOUT = Timeout(connect=5.0, read=600, ...)`，于是 httpx 拿到 `Timeout(timeout=None)`——无限等待。而同一处的 `max_retries` 用的是 `if ... is not None` 守卫，所以 SDK 默认重试反而正常。这个结论只能从源码得到，任何文档都不会写「我们这里有个不对称的参数传递」。

第二处，`app/core/job_failure.py:77-128` 的 `apply_retry_policy` 注释里引了 rq 2.9 的源码行为：`_job_stack.push(self)` 意味着 `get_current_job()` 拿到的是同一个对象，而 `handle_job_failure` 读的是内存里的 `retries_left` 且不做 `job.refresh()`——所以在异常处理器里把 `retries_left` 改成 0 是有效的。这也是读源码才能确定的事。

方法论就一句：文档告诉我「设计意图」，源码告诉我「实际行为」，两者不一致的地方才是线上事故的来源。

**项目证据**

- `app/core/llm_factory.py:1-60` — 模块文档，langchain-openai 覆盖 SDK 超时的根因
- `app/core/llm_factory.py:84` — `timeout = config.llm_timeout_seconds if timeout is None else timeout`，刻意用 `is None` 而非 `or`，让显式传入的 `0` 存活
- `app/core/job_failure.py:77-128` — `apply_retry_policy`，引 rq 2.9 源码行为

#### Q3 · 你如何理解 Agent 系统？

**回答**

我先给一句定义，再拆开讲，最后说我自己项目里怎么落的。

**定义：Agent 是一个「让语言模型参与控制流决策」的系统。**

这句话的重心在「控制流」三个字。普通程序里，下一步执行什么是程序员在写代码时就定死的——`if` 走哪个分支、循环转几圈，全在源码里。Agent 系统把这个决定权的一部分交给了模型：下一步调哪个工具、要不要再查一次、什么时候算完成，由模型在运行时判断。

所以判断一个系统是不是 Agent，不看它有没有用大模型，看**控制流有没有一部分是模型决定的**。一个调 GPT 做文本分类的接口，模型只是在生成内容，控制流完全固定，那不是 Agent，是一个带 AI 能力的普通服务。

---

**一个完整的 Agent 系统有哪几层**

我习惯按「出问题时该找谁」来分层，一共四层，前两层是能力，后两层决定它能不能上生产：

**第一层：决策层（模型）。** 负责「下一步干什么」。输入是当前上下文，输出是一个动作意向——调工具、或者直接回答。这一层的本质特征是**它会失败，而且失败方式很脏**：超时、返回不合法的 JSON、幻觉出一个不存在的工具名、陷进「反复调同一个工具」的循环。传统后端的失败是二值的（成功/异常），模型的失败是连续的（它给你返回了东西，但那东西是错的），这是最难处理的地方。

**第二层：执行层（工具）。** 负责「真的把事做了」。查数据库、调 API、读文件、执行代码。这一层的失败是传统后端那种失败——网络超时、下游 5xx、参数校验不过，处理手段是成熟的（重试、超时、熔断）。

**第三层：状态层。** 负责「记住发生过什么」。这一层容易被低估。它至少要装三种东西：

- *当前这一轮的工作状态* ——已经调了哪些工具、拿到了什么结果
- *跨轮次的会话记忆* ——用户前面说过什么，得记住，否则每轮都在重新自我介绍
- *失败痕迹* ——第 2 个检索分支挂了这件事必须留在状态里。如果只打日志不进状态，下游节点就不知道「这次的证据是不完整的」，会拿着一半的材料当全部材料用

**第四层：治理层。** 负责「出问题之后对外说什么」。这层的产物是三样东西：一个降级原因（degrade reason，用来说明"这次结果为什么不完美"）、一套可观测数据（指标 + 链路）、一个对用户可解释的输出。

四层里，很多人只做前两层就上线，然后在生产上被打回来。原因很简单：前两层决定 demo 能不能跑通，后两层决定故障时能不能定位、以及用户会不会被静默地骗。

---

**「静默失败」为什么是 Agent 系统最大的敌人**

这是我想重点讲的一点，因为它是 Agent 系统区别于普通后端的地方。

普通后端服务失败了，它返回 500，调用方立刻知道出事了。Agent 系统失败了，它**返回一段通顺的话**。检索一条证据都没查到，模型照样能凭参数化知识编出一段听起来很专业的回答；工具调用超时了，模型会说"根据我的了解……"然后继续。用户看到的是一个正常的答案，无从判断这次的可信度和上次差了多少。

这意味着：**在 Agent 系统里，"失败"必须被显式建模成一等公民**，不能指望异常机制帮你兜住。异常只能捕获"程序崩了"，捕获不了"程序好好地给出了一个不该给的答案"。

具体做法就是给系统加一套降级原因枚举。我项目里的做法是划分标准取"修复方向"——检索失败（`RETRIEVAL_FAILED`）和检索到零条（`RETRIEVAL_EMPTY`）必须是两个不同的原因，因为前者是去修服务，后者是去补文档。如果合并成一个 `RETRIEVAL_ERROR`，值班的人看到告警还得先翻日志才知道该找运维还是找内容运营。

---

**我项目里的落法**

我的主链路是五个节点：改写 → 并行检索 → 去重 → 生成 → 质检。数一下模型出现的位置：改写调一次、生成调一次、质检调一次，检索和去重一次都不调（检索是向量库 + BM25，去重是内容 md5 指纹）。

这个比例是我刻意做的：**模型只在真的需要语义判断的地方出现，其余全部是确定性代码。** 理由是模型每多出现一次，就多一个不确定的失败点、多一份 token 成本、多几秒延迟。能用规则做的事就不要问模型——去重用 md5 就够了，问模型"这两段文字是不是重复的"既慢又不稳定。

失败痕迹这一层，我在图状态里放了三个用 reducer 合并的列表字段（`documents`、`retrieve_failures`、`degrade_reasons`）。reducer 的意思是：并行的几个分支各自返回一小段，框架自动把它们拼起来，而不是后写的覆盖先写的。有了 `retrieve_failures`，"这次少了一份证据"就是一个下游可读的事实，不是一行日志。

治理层有一处细节值得单独说，因为它是我踩过的坑：三次模型调用外面各套了两层保护——熔断器和耗时计时器，**顺序是计时器在熔断器内侧**。这个顺序不能反。熔断器打开的时候，它第一行就抛异常，模型调用根本没发出去。如果计时器在外侧，就会记下一个约 0.1 毫秒的耗时样本，熔断期间几百个"根本没打出去的调用"全以接近 0 秒计入耗时直方图——**P99 反而变好看了**。下游全挂了，监控面板上显示一切正常，这是最糟糕的一种可观测性事故：数据是错的，但它自洽，你不会怀疑它。

**项目证据**

- `app/agent/rag_v2/nodes.py:91-155,264-331,334-469` — 三处 LLM 调用节点
- `app/agent/rag_v2/nodes.py:158-211,214-261` — 两处纯确定性节点（检索、去重）
- `app/agent/rag_v2/nodes.py:127-133` — 计时器必须在 guard 内侧的原因
- `app/agent/rag_v2/state.py:17,27,34` — 三个 reducer 字段
- `app/core/breakers.py:35-46` — `retrieval_breaker` 与 `llm_breaker`
- `app/core/errors.py:25-62` — `DegradeReason`，按修复方向划分

#### Q4 · Tool 的设计原则是什么？

**回答**

先讲清工具在 Agent 里的位置，再说原则。

模型本身只会做一件事：根据上文生成下文。它不能查数据库、不能读文件、不能知道现在几点。工具（Tool）就是把这些能力包成模型可以"调用"的形式。

所谓调用其实是一次约定俗成的接力：我把工具清单（名字、说明、参数格式）连同用户问题一起发给模型；模型判断需要用某个工具，就返回一段结构化的调用请求（工具名 + 参数），**它自己不执行**；我的代码解析这段请求、真的去执行、把结果拼回对话历史；再发给模型让它继续。所以工具设计的本质是**设计一份给模型看的接口文档**——读者是模型，不是人。这决定了下面所有原则。

---

**一、描述要写「什么时候该用我」，而不只是「我是什么」**

模型选工具的唯一依据就是你写的那段描述。写"获取时间的工具"，模型不知道用户问"今天星期几"算不算；写"当用户询问『现在几点』『今天星期几』『今天日期』等时间相关问题时，使用此工具"，模型就有了明确的路由信号。

这是给模型的**触发条件**，不是给人的功能说明。同理，参数说明里要写清格式和取值范围——模型幻觉出一个非法参数值，很多时候是因为你没告诉它合法值有哪些。

反面案例是工具描述之间语义重叠。两个工具的描述都说"查询服务的日志"，模型会随机选一个，而且你没法调试——因为选择过程发生在模型内部。工具集设计时，**描述必须互相排斥**，这比每条描述本身写得好更重要。

---

**二、工具内部失败要 `return` 一段错误文本，不要 `raise`**

这条是我认为最反直觉、也最重要的一条。

普通的 Python 函数出错就抛异常，让上层处理。但工具的调用方是**模型**，异常抛出去会直接终止整个 Agent 循环——用户拿到 500。

而如果 `return "检索知识时发生错误：连接超时"`，这段文本会作为工具结果进入对话历史，模型看到它，然后模型可以自己决定怎么办：换个关键词重试、改用另一个工具、或者老实告诉用户"知识库暂时不可用"。

判据就一句：**这个失败是模型能处理的信息，还是系统级的故障。** 检索超时、查不到结果、参数不合法——模型能处理，`return`。数据库连接池耗尽、配置缺失——模型处理不了，让它抛。

我项目里的两个工具都走 `return`，MCP 远程工具那边也是同一个选择：重试耗尽之后返回一个 `isError=True` 的结果对象，而不是抛异常。

---

**三、返回值要同时喂给模型和喂给系统**

模型只需要一段能读的文本。但系统需要原始物料——比如"这次答案引用了哪几篇文档、各自的 ID 和来源"。

如果工具只返回格式化后的字符串，这些结构化信息就永久丢失了：你没法做引用标注、没法做证据回溯、没法统计哪篇文档被引用得最多。

所以工具的返回值最好是双通道的：一路给模型看的摘要文本，一路给系统留的结构化物料。LangChain 里对应的是 `@tool(response_format="content_and_artifact")`，返回一个 `(content, artifact)` 二元组——`content` 进对话历史给模型，`artifact` 不进对话历史，由框架旁路交给我的代码。

这个设计还有一个附带好处：**artifact 不占 token**。文档全文如果塞进对话历史，几轮之后上下文就爆了；只有摘要进历史，全文走旁路。

---

**四、粒度：一个工具做一件可独立完成的事**

太粗的工具（`manage_database(action, params)` 什么都能干）让模型必须理解一大堆内部分支，出错率高，而且描述根本写不清。太细的工具（把一次查询拆成"连接""构造SQL""执行""关闭"四个）让模型必须自己编排顺序，那些顺序本来是确定的，交给模型纯属自找麻烦。

判据是：**一个工具调用应该对应用户能理解的一个动作。** "查一下 CPU 使用率"是一个动作，"建立监控平台连接"不是。

顺带一条：工具数量也要克制。工具清单是每轮都要塞进上下文的，几十个工具的描述能吃掉几千 token，还会显著降低模型的选择准确率。超过十来个就该考虑分组或者两级路由（先让模型选类别，再在类别内选具体工具）。

---

**五、幂等与副作用要在描述里明说**

模型会重试，框架也会重试。一个"发送邮件"的工具被重试三次就是三封邮件。

所以有副作用的工具要么自己做幂等（接一个 idempotency key，重复调用返回同一个结果），要么在描述里写明"此操作不可撤销，执行前需用户确认"，配合人工确认环节。只读工具没这个问题，可以放心重试。

---

**六、系统提示词里不要重复列工具清单**

这是个具体的坑。框架已经会把工具信息自动传给模型（走的是 API 的 `tools` 参数），你在系统提示词里再列一遍，就有了**两份真相**。工具增删的时候一定会漏改一处，然后模型看到一个已经删掉的工具名，或者对同一个工具有两份互相矛盾的说明。

**项目证据**

- `app/tools/time_tool.py:12-21` — 描述写触发条件而非功能说明
- `app/tools/knowledge_tool.py:41-43` — 失败 `return` 而非 `raise`
- `app/agent/mcp_client.py:71-74` — MCP 重试耗尽返回 `isError=True`，同一选择
- `app/tools/knowledge_tool.py:13` — `response_format="content_and_artifact"` 双通道
- `app/core/used_documents.py:34,150` — artifact 的下游消费方（引用归一化）
- `app/services/rag_agent_service.py:146-155` — 不在提示词里重复工具清单
- 工具级幂等键、两级工具路由：项目中没有，不能硬套

#### Q5 · Memory 有哪些类型？

**回答**

这题容易答成背名词（短期记忆、长期记忆、情景记忆、语义记忆……）。我按**生命周期**和**存储介质**来分，这两个维度决定了工程上怎么做。

---

**先说为什么需要 Memory**

大模型的 API 是**无状态**的。每次调用你必须把全部上下文重新发一遍，模型自己什么都不记得。所谓"模型记住了我们前面的对话"，实际是你的代码把历史消息重新拼进了这次请求。

于是问题变成：历史越来越长，而上下文窗口有上限、token 要花钱、内容太长模型的注意力还会被稀释。Memory 系统就是在解决**"该记什么、该忘什么、忘掉的东西怎么压缩保留"**。

---

**第一类：请求内工作状态（生命周期 = 一次请求）**

一次请求内部流转的中间产物：改写出的子查询、召回的文档、哪个分支失败了、生成的答案、质检结果。请求结束就没用了，不落库（落库的是最终结果和链路追踪，不是这些中间态）。

严格说它不算"记忆"，是并发状态管理。但它有一个 Memory 系统共通的核心难题：**并行分支的结果怎么合并**。

我项目里三个字段用了 reducer（`documents`、`retrieve_failures`、`degrade_reasons`）。reducer 的意思是给这个字段声明一个合并函数（这里是列表相加），框架在多个并行分支都往同一个字段写时，用这个函数合并而不是后者覆盖前者。没有 reducer 的话，三个并行检索分支各返回 4 篇文档，最后只剩最后一个分支的 4 篇——而且这个 bug 在单分支测试里根本测不出来。

---

**第二类：会话记忆（生命周期 = 一个会话，可跨请求）**

这是狭义上的"对话记忆"。核心矛盾是**完整性 vs 预算**：全量历史最准确但会撑爆上下文，只留最近几条则更早的约定会丢。

主流有四种做法，各有取舍：

1. **全量塞进去。** 最简单，短对话够用，长对话必爆。
2. **滑动窗口（只留最近 N 轮）。** 实现简单、成本恒定，代价是超出窗口的信息彻底消失——用户第 1 轮说"我用的是 Python 3.9"，到第 20 轮模型给出 3.12 的语法。
3. **滚动摘要（Rolling Summary）。** 把较早的对话压成一段摘要，摘要 + 最近几轮原文一起发。这是工程上的主流选择，成本可控且早期信息不完全丢失。
4. **向量检索式记忆。** 把每轮对话存进向量库，按当前问题的相似度召回相关历史。适合超长会话或跨会话，代价是引入检索误差——召回错了，模型会基于不相关的历史作答。

我项目里用的是第 3 种，实现上有几个点值得说：

- **压缩是按 token 预算触发的，不是按轮数。** 超过阈值（我配的是 2400 token）才把较早部分压成摘要。按轮数触发的问题是一轮可能是"嗯"也可能是三千字的报错日志。
- **摘要用固定小标题**——【明确事实】【任务进展】【约束与偏好】【待确认项】。固定结构不是为了好看，是为了**摘要可以增量追加**：下次压缩时，模型能把新内容并进对应的小标题下，而不是把整段重写一遍（重写会导致早期信息在每次重写中逐渐流失，几轮之后面目全非）。
- **需要一个游标记录"摘要覆盖到哪条消息为止"。** 我这里是 `summarized_through_message_id`。没有这个游标，就会把已经摘要过的内容再摘要一遍，摘要里同一件事出现三遍。

这里有个坑我踩过：**如果游标指向的那条消息被删了（软删），要能优雅退化。** 否则代码按 ID 找不到位置，会退化成"从头开始算 pending"，把整段历史重新摘要一遍。

---

**第三类：Checkpoint（生命周期 = 取决于存储后端）**

Checkpoint 是框架层的机制，存的是**整张图的执行状态快照**，用途和会话记忆不一样：

- 会话记忆解决的是"模型需要知道前面聊了什么"
- Checkpoint 解决的是"这次执行中断了，能不能从断点恢复"，以及"人工审批之后能不能接着跑"

Checkpoint 的存储后端决定了它的能力边界。**内存版**（`MemorySaver`）进程重启即失，只适合开发调试和单机会话延续；**数据库版**（Postgres/Redis Saver）才能支撑真正的断点续跑、多副本部署、人工介入后恢复。

我项目里 `create_agent` 那条路用了 `MemorySaver`，是进程内的，重启就没了。而主链路 `rag_v2` 那张图**编译时根本没传 checkpointer**——因为它是一条 90 秒内跑完的短链路，没有中断恢复的需求，加 checkpointer 只是白付一次序列化开销。这个决定我要在面试里主动说清楚，因为"用了 LangGraph"容易被默认为"有 checkpoint"。

---

**第四类：长期记忆 / 跨会话记忆**

用户的稳定偏好（"我习惯用 TypeScript"）、历史结论沉淀、多次会话之间的连续性。典型做法是单独一张表或一个向量库，按用户 ID 索引，每轮结束时判断"这轮有没有值得长期记住的事实"，有则抽取入库；下次会话开始时按当前话题召回。

难点不在存，在**判断什么值得记，以及记错了怎么改**。用户改主意了（"我现在换 Go 了"），旧记忆要能被覆盖而不是并存——两条矛盾的偏好同时召回，模型的行为就不可预测了。

这一块我**项目中没有**，不能硬套。我的记忆是会话级的，会话结束就不再使用。

---

**一句话总结取舍**

生命周期越长的记忆，价值越高、出错的代价也越大。请求内状态错了这次请求失败；会话记忆错了这个会话变傻；长期记忆错了，用户之后每一次对话都被一条错误的"事实"污染，而且很难发现。所以我的顺序是先把短的做扎实，再往长的走。

**项目证据**

- `app/agent/rag_v2/state.py:10-34` — `RAGState`，请求内状态与三个 reducer 字段
- `app/services/conversation_memory_service.py:93-131` — `build_context`，滚动摘要主流程
- `app/services/conversation_memory_service.py:133-144` — `_messages_after_snapshot`，软删游标防御
- `app/services/conversation_memory_service.py:28-35` — 摘要固定四小标题
- `app/config.py:133-136` — 记忆的四个 token 预算参数
- `app/services/rag_agent_service.py:107,308-373` — `MemorySaver`，进程内 checkpoint
- `app/agent/rag_v2/graph.py:51` — `graph.compile()` 未传 checkpointer
- `migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 快照表
- 跨会话长期记忆、偏好抽取与覆盖：项目中没有，不能硬套

#### Q6 · ReAct 和 Plan-Execute 架构分别适用于哪些场景？

**回答**

先把两个架构讲清楚，再说判据，最后给我自己的取舍记录。

---

**ReAct：Reasoning + Acting 交替**

名字来自 2022 年那篇论文，核心是让模型在"思考"和"行动"之间循环：

```
Thought:  我需要知道用户说的服务当前 CPU 多少
Action:   query_cpu_metrics(service="order-api")
Observation: {"cpu_percent": 93.2, ...}
Thought:  CPU 93% 偏高，我需要看是不是有慢查询
Action:   search_logs(service="order-api", keyword="slow")
Observation: ...
Thought:  证据够了，可以回答
Answer:   ...
```

关键特征是**没有预先计划**。每一步做什么，都是看完上一步的观察结果才决定的。循环的终止也由模型自己判断——它认为够了就输出答案。

这带来两个后果。好的一面：极强的适应性，中途发现路走错了可以立刻换方向。坏的一面：**步数不可预测，成本不可预测**。每一轮都是一次完整的模型调用，上下文还在不断变长（历史观察结果全都累积在里面），所以第 10 轮的单次成本远高于第 1 轮。而且它可能陷进循环——反复调同一个工具、或者在两个工具之间来回跳。所以工程上必须给它加一个硬性的最大步数上限。

---

**Plan-and-Execute：先规划，再执行**

分成两个角色（有时三个）：

- **Planner**：一次性把任务拆成有序的步骤清单
- **Executor**：逐步执行，每一步可以用小模型甚至纯代码
- **Replanner**（可选）：执行到某一步发现计划不对了，重新规划剩余部分

```
Plan:  1. 查 order-api 的 CPU 和内存
       2. 查最近 10 分钟的错误日志
       3. 查最近有没有发布
       4. 综合以上给出根因假设
Execute 1 → Execute 2 → Execute 3 → Execute 4 → Answer
```

好处是**成本和步数在开始前就基本确定**，而且执行阶段不需要大模型全程参与——Planner 用最强的模型调一次，Executor 用便宜的模型跑 N 次，整体成本比 ReAct 低不少。计划本身还是一个可以给人看、甚至可以让人改的中间产物，这在需要人工审批的场景里是硬需求。

坏处是**计划是在信息最少的时刻做的**。规划的时候还没查任何数据，凭什么知道要查哪三样？现实里经常第 1 步查完就发现整个计划的前提错了。Replanner 就是补这个洞的，但每次重规划都是一次额外的大模型调用，补多了成本又回去了。

---

**判据：任务的步骤是否收敛**

这是我用的唯一判据，比"任务复杂不复杂"好用得多。

**步骤收敛**——同类任务的处理步骤基本固定，做十次有八次是同一套流程。这时候连 Plan-Execute 都不需要，直接写成固定的工作流（DAG）。让模型现场规划一套本来就固定的流程，收益是零，代价是多一次模型调用、多一个失败点、步骤边界变得不可预测。

**步骤不收敛但可提前枚举**——每次的步骤不同，但看到任务描述就能大致列出要做什么。这是 Plan-Execute 的主场，典型是研究类任务、多文档汇总、数据分析报告。

**步骤不收敛且无法提前枚举**——下一步严格依赖上一步的结果，比如调试一个未知的 bug、在陌生网站上完成一个操作。只有 ReAct 能干这个活。

还有一个次要判据是**步数量级**。三五步以内 ReAct 完全够用，规划的开销不划算；十几步往上，ReAct 的上下文累积和跑偏风险都会显著上升，值得先规划。

---

**我项目里的取舍记录**

我这个仓库正好两条路都有过，而且有一条是被我删掉的，所以能给出真实的取舍而不是纸面对比。

ReAct 那条还活着，用 `create_agent` 实现——框架内部就是 Thought-Action-Observation 循环。它适合的场景很清楚：工具少（本地两个 + MCP 七个）、步数短、下一步真的取决于上一步。用户问"现在几点"就调时间工具，问业务问题就调检索，这个判断确实没必要预先规划。

Plan-Execute 那条我写过完整的三件套——`planner.py`、`executor.py`、`replanner.py`，用在 AIOps 告警首响上。**后来全删了**，换成一条固定编排管线：归一化告警 → 建诊断任务 → 检索 SOP → 生成诊断 → 存报告 → 完成。

删的理由就是上面那条判据：**告警首响的步骤是收敛的。** 收到一条 CPU 高告警，要做的事十次里有十次一样——找到对应的 SOP、按 SOP 查监控和日志、给出初步结论。让模型现场规划"先查日志还是先查监控"，收益为零。

而代价是实打实的三条：多一次模型调用（延迟 + token）、多一个可能返回垃圾计划的失败点、**阶段边界变得不可预测**。最后这条是决定性的——我需要把诊断过程的时间线落库给值班人员看，哪个阶段是真的状态迁移、哪个只是记一条事件，必须是确定的。Planner 每次生成的步骤名都不一样，这套时间线就没法建模了。

这里我要明确一句：**现在这条管线不是 Plan-Execute，是固定工作流。** Planner/Executor/Replanner 已经从代码库删除了，我不会拿现在的编排器冒充 Plan-Execute 经验。

---

**补充：还有一种混合形态**

实践中越来越常见的是**外层固定工作流 + 某个节点内部跑 ReAct**。整体流程是确定的（保证阶段可观测、预算可控），只在真正需要自由探索的那一两个节点上放开让模型循环，并给那个节点单独设步数上限和时间预算。

我的两条路径其实就差一步就是这个形态——如果把 AIOps 管线里"查证据"那一步换成一个带步数上限的 ReAct 子循环，就是标准的混合架构。这是我下一步想做的，目前**还没做**，标 `[假设]`。

**项目证据**

- `app/services/rag_agent_service.py:133-137` — `create_agent`，ReAct 循环
- `app/agent/aiops/planner.py`、`executor.py`、`replanner.py`、`state.py`、`utils.py` — 全部 deleted
- `app/services/aiops_service.py` — deleted
- `app/services/alert_diagnosis_orchestrator.py:71-131` — 现行固定编排管线
- `app/services/alert_diagnosis_orchestrator.py:50-53` — `_PHASE_STATUS`，区分真状态迁移与事件记录
- ReAct 步数上限（`recursion_limit` / `max_iterations`）：项目中没有，不能硬套
- 混合架构（工作流内嵌 ReAct 子循环）：`[假设]`，尚未实现

---

### 面经 08 · 2026-08-03

#### Q1 · 你现在大二，实习时间有多长，怎么平衡学校的事情？

**回答**

个人情况题，仓库答不了，也不能编。

唯一能诚实说的是：`git log` 里的提交节奏和 `docs/` 下的日计划文档（`docs/aiops-itops-convergence-daily-plan.md`、`docs/roadmap-week1-aiops-rag-and-protocol-pdf.md`）能反映实际投入方式——按周切目标、按天切任务。回答这题应该用真实的时间安排，不是拿代码充数。

**项目证据**

- 项目中没有对应实现，不能硬套

#### Q2 · 说一下你这个agent项目怎么开发的？

**回答**

按可靠性倒推着开发的，不是按功能正推。

第一版能跑通：FastAPI 收请求，`rag_v2` 五个节点串起来，Milvus 出证据，LLM 出答案。然后开始一层层补——每一层都是因为发现了一个具体的谎。

**第一个谎：HTTP 200 + body 里 `code: 500`。** 端点用 `return {"code": 500, ...}` 报错，于是 nginx 的 `$status` 是 200、LB 健康检查通过、Prometheus 的 `http_requests_total{status="5xx"}` 恒为 0——所有错误率监控读到的都是 0%。修法是 `app/api/chat_v2.py:151` 改成裸 `raise`，状态码交给 `app/core/exception_handlers.py:180-191` 注册的全局处理器决定。

**第二个谎：并行分支挂一个整张图崩。** `app/agent/rag_v2/nodes.py:161-169` 记着这个坑。修法是分支内兜异常，失败明细进 `retrieve_failures`。

**第三个谎：检索服务挂了返回空列表。** 报告读起来像「知识库没这条」。修法是 `RetrievalStatus` 三态（`app/core/errors.py:65-78`）。

**第四个谎：熔断期间 P99 变好看。** `app/agent/rag_v2/nodes.py:127-133`。修法是计时器移进 guard 内侧。

**第五个谎：worker 失败原因被二次异常掩盖。** 原来在 `except` 里 `commit()`，连接已死会抛新异常，`job.exc_info` 里存的是这个新异常，真正原因丢了，任务永远停在 RUNNING。修法是 `app/core/job_failure.py:146-183` 的 `record_job_failure` 用独立 `SessionLocal()`，且永不抛。

所以开发顺序是：先让它能跑，然后逐个消灭「系统在撒谎」的地方。

**项目证据**

- `app/api/chat_v2.py:151` — 裸 `raise`
- `app/core/exception_handlers.py:180-191` — `register_exception_handlers`
- `app/core/exception_handlers.py:1-60` — 模块文档，列举只读 HTTP 状态码的消费方
- `app/agent/rag_v2/nodes.py:161-169,189-205` — 并行分支兜底
- `app/core/errors.py:65-78` — `RetrievalStatus`
- `app/agent/rag_v2/nodes.py:127-133` — 计时器位置
- `app/core/job_failure.py:146-183` — `record_job_failure`，独立 session

#### Q3 · 你项目rag是怎么存储和召回的？

**回答**

存储和召回分两条独立路径，先说存储。

**存储。** `app/core/milvus_client.py:149-190` 的 `_create_collection` 建四个字段：`id` VARCHAR 主键、`vector` FLOAT_VECTOR、`content` VARCHAR（上限 8000）、`metadata` JSON。索引在 `app/core/milvus_client.py:192-208`：**HNSW + COSINE**，`M=16, efConstruction=200`。这里跟原题文件描述不一样——不是 IVF_FLAT + L2，git log 里有一次专门的提交 `IVF_FLAT+L2 → HNSW+COSINE`。

向量化走 `app/services/vector_embedding_service.py`，DashScope `text-embedding-v4`，1024 维（`app/services/vector_embedding_service.py:171-176`）。这里有个服务端硬限制：`MAX_BATCH_SIZE = 10`（第 22 行）。而且 DashScope **不保证返回顺序**，所以 `embed_documents`（72-135）必须按 `item.index` 重排（110-112），最后还有一个数量相等断言（127-130）。这个断言不是洁癖——顺序错了会让向量和文本静默错配，检索结果看起来「有点不准」，很难查。

`app/core/milvus_client.py:111-123` 还有一段：检测到维度不匹配时自动 drop 并重建 collection。换 embedding 模型时避免维度不一致的脏数据。

**召回。** `app/services/vector_search_service.py:52-107` 的 `retrieve_documents` 是混合检索：Milvus 向量 + BM25，用 `EnsembleRetriever` 做 RRF 融合（第 96 行），权重 0.7 / 0.3（42-43 行）。候选数是 `candidate_k = max(top_k * 3, top_k, 10)`（第 84 行）——先多召回再融合。

BM25 语料为空时退化为纯向量（92-94 行）。BM25 检索器有缓存，用 `collection.num_entities` 做失效判据（`app/services/vector_search_service.py:140-163`）。分词在 203-224 行，中文逐字 + ASCII 按词缓冲。

在线链路上面还有一层：`app/agent/rag_v2/nodes.py:91-155` 先把问题改写成多个互补子查询，`app/agent/rag_v2/graph.py:17-19` 用 `Send` 并行检索，`app/agent/rag_v2/nodes.py:214-261` 按 md5 内容指纹去重并截到 `FINAL_TOP_K=6`。

**没有 reranker。** `.hf_cache/hub/` 下有 `BAAI/bge-reranker-v2-m3`，最近一次提交信息里也提到「rerank重排召回测试」，但读过的代码路径里没有任何一处调用它。原题文件说的 Top-20→Top-5 两阶段重排，当前代码里不存在，不能硬套。

**项目证据**

- `app/core/milvus_client.py:149-190` — collection schema
- `app/core/milvus_client.py:192-208` — HNSW + COSINE，`M=16, efConstruction=200`
- `app/core/milvus_client.py:111-123` — 维度不匹配自动重建
- `app/services/vector_embedding_service.py:22` — `MAX_BATCH_SIZE = 10`
- `app/services/vector_embedding_service.py:110-112,127-130` — 按 index 重排 + 数量断言
- `app/services/vector_search_service.py:42-44` — 权重 0.7/0.3
- `app/services/vector_search_service.py:84,96` — `candidate_k`、`EnsembleRetriever`
- `app/services/vector_search_service.py:92-94` — BM25 空语料回退
- `app/services/vector_search_service.py:140-163` — BM25 缓存与失效
- `app/services/vector_search_service.py:203-224` — `_tokenize_for_bm25`
- `app/agent/rag_v2/nodes.py:28-33` — `RETRIEVE_TOP_K=4`、`FINAL_TOP_K=6`

#### Q4 · 你项目的OCR是怎么实现的（多模态输入问题，把ASR的接入也说明了一下，其实还是大模型api）

**回答**

项目中没有，不能硬套。

这个仓库没有任何 OCR 或 ASR 代码路径，多模态入口一个都没有。唯一沾「文档解析」的是 `app/services/protocol_pdf_ingestion_service.py`，但它处理的是**文本层已存在**的 PDF，走文本抽取而不是图像识别。把它说成 OCR 是偷换概念。

能诚实说的是相邻环节：`app/services/document_splitter_service.py` 的切分策略、`app/services/metadata_enricher.py` 的 chunk 摘要（`app/services/metadata_enricher.py:5` 写着 `chunk_summary` 取首句不超过 120 字，用于 rerank 或前端预览）。这些是「文档进知识库之后」的事。

OCR/ASR 那一整段——图像预处理、识别置信度、专有名词纠错、时间戳对齐、说话人分离——在这个项目里没有对应实现，不能拿 PDF 文本抽取顶替。

**项目证据**

- 项目中没有 OCR / ASR 实现，不能硬套
- 相邻但不同的模块：`app/services/protocol_pdf_ingestion_service.py`（文本层 PDF 抽取）、`app/services/document_splitter_service.py`、`app/services/metadata_enricher.py:5`

#### Q5 · 你项目中uuid的实现和维度是什么？

**回答**

仓库里有两处 uuid，用途和取舍都不同。

**第一处，request_id。** `app/core/request_context.py:36-42` 的 `new_request_id()` 用 `uuid.uuid4().hex[:16]`。截断到 16 位是有理由的——注释里写着：足够避免碰撞，又比完整 uuid 短得多，而日志每一行都要带它，短一半就是可观的可读性收益。

request_id 这条链路值得多说两句，因为它有两个不显然的设计。

一是**同一个值有两种缺省表示**。`app/core/request_context.py:45-47` 的 `get_request_id()` 没值时返回 `"-"`（给日志，固定列宽好扫），`get_request_id_or_none()`（50-64）返回 `None`（给数据库，NULL 的语义是「没有这个值」，而 `"-"` 会被当成真实取值参与 `GROUP BY`，把所有无上下文的记录聚成一个假的「请求 -」分组）。这个判断只有一处实现，因为两个 repository 都要落 request_id，各写一遍就是等着跑偏。

二是**外部传入的值必须校验**。`app/core/middleware.py:34` 的 `_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")`。这是安全边界不是洁癖：这个值会出现在每一行日志里，放任 `\n` 通过，攻击者传一个带换行的 header 就能凭空造出一整行假日志。校验不通过时不报错，直接当没传——追踪 id 不该成为请求失败的理由。

**第二处，MCP 相关的 uuid**：项目里没有自己生成，走的是框架。

至于「维度」——如果问的是向量维度，那是 1024（`app/services/vector_embedding_service.py:171-176`，`app/core/milvus_client.py:49` 的 `VECTOR_DIM`）。如果问的是 uuid 的位数，是 uuid4 的 128 位取 hex 前 16 字符即 64 位。

**项目证据**

- `app/core/request_context.py:36-42` — `new_request_id`，`uuid4().hex[:16]`
- `app/core/request_context.py:45-47,50-64` — 两种缺省表示及其理由
- `app/core/middleware.py:34,37-44` — `_SAFE_REQUEST_ID`、`_sanitize_incoming`
- `app/core/middleware.py:86` — 写入 `scope["state"]`
- `app/core/milvus_client.py:49` — `VECTOR_DIM = 1024`

#### Q6 · 你使用的是多模块agent，为什么不使用多agent形式开发（尝试过multi-agent，协商开销太大，耗时不可以接受）

**回答**

这个仓库的情况和题干描述的一致，但理由要说准——不是「协商开销大」这么笼统。

现状是多模块单图：`rag_v2` 五个节点（`app/agent/rag_v2/graph.py:22-51`）在一张 `StateGraph` 里，共享一个 `RAGState`。唯一的并发是 `Send` 扇出的检索分支（`app/agent/rag_v2/graph.py:17-19`），而它们是**同质**的——同一个函数的 N 个实例跑不同 query，之间不通信。

不做多 Agent 的具体理由有三条，都能在代码里指出来。

**一、状态合并已经用 reducer 解决了，不需要协商。** `app/agent/rag_v2/state.py:17,27,34` 三个 `Annotated[List, operator.add]`。分支之间不需要商量谁的结果更好，`dedup_node` 在 fan-in 之后统一看全局（`app/agent/rag_v2/nodes.py:214-261`）——它是第一个有全局视野的节点，所以降级判定放在这里：有失败有文档 = `PARTIAL_RETRIEVAL`，有失败无文档 = `RETRIEVAL_FAILED`，无失败无文档 = `RETRIEVAL_EMPTY`。多 Agent 协商解决的是「谁说得对」，而这里的问题是「证据够不够」，用规则就能判。

**二、总预算无法跨 Agent 收口。** `app/agent/rag_v2/service.py:61` 一句 `async with asyncio.timeout(config.chat_total_budget_seconds)` 罩住整张图，90 秒（`app/config.py:81`）。超时后 65 行捕获 `TimeoutError` 并且**不重抛**，用 `astream` 一路留着的 `last_state` 拼部分结果。多 Agent 各自持有 LLM 客户端和重试策略时，这个「一个地方管总时间」的性质就没了。

**三、熔断器的划分标准会失效。** `app/core/breakers.py` 的模块文档写着划分规则：**会同时坏、也会同时好的调用，共用一个熔断器**。所以是按下游分（`retrieval_breaker` 管 Milvus，`llm_breaker` 管模型网关），不是按调用点分。多 Agent 各建一个熔断器的话，同一个下游挂了要被 N 个熔断器各自独立地失败 5 次才跳闸。

所以答案是：**在这个业务里没有需要协商的分歧**。多 Agent 的成本（Token 随分支增长、尾延迟放大、共享状态竞争、取消传播）都是真的，但更根本的是收益为零。

需要说明：这个仓库里没有 multi-agent 的实现，也没有 A/B 对比的实测数据。「尝试过 multi-agent、协商开销太大」这个结论如果要说，必须是真实做过的实验，不能拿这里的代码当证据。

**项目证据**

- `app/agent/rag_v2/graph.py:17-19,22-51` — 单图 + 同质扇出
- `app/agent/rag_v2/state.py:17,27,34` — reducer 合并
- `app/agent/rag_v2/nodes.py:214-261` — `dedup_node` 全局视野与三态降级判定
- `app/agent/rag_v2/service.py:61,65` — 总预算 + 不重抛
- `app/config.py:81` — `chat_total_budget_seconds = 90.0`
- `app/core/breakers.py:1-46` — 熔断器划分规则
- 项目中没有 multi-agent 实现，相关实测数据不能硬套

---

### 面经 09 · 2026-07-31

#### Q1 · 平台架构为什么设计成三层？

**回答**

这题有个陷阱：大部分人会背「controller-service-dao，为了解耦」。这个答案不算错，但它没有回答"为什么是三层而不是两层或五层"，面试官追问一句"那你为什么不把 service 和 dao 合并"就答不上了。所以我先把分层的判据讲清楚，再说我项目里是怎么落的。

---

**先讲概念：分层的本质是「变化的传播被挡住」**

软件分层不是为了代码整齐，是为了让**一类变化只影响一层**。判断某两块代码该不该分层，问一个问题就够了：**它们变化的原因是不是同一个？**

这就是 SOLID 里的 S（单一职责）的原始定义 —— Robert Martin 的原话不是"一个类只做一件事"，而是"一个模块应该只有一个**引起它变化的原因**"。两个理由完全一样的东西，强行分层只是增加转发代码；两个理由不同的东西混在一起，改任何一个都会波及另一个。

放到 Web 后端上，三层各自的"变化原因"是：

| 层 | 变化原因 | 典型变更 |
|---|---|---|
| 接入层 | **协议**变了 | HTTP 换成 gRPC、加一个 WebSocket 入口、状态码约定调整 |
| 业务层 | **规则**变了 | 降级策略改了、加一个审批环节、算法换了 |
| 数据层 | **存储**变了 | MySQL 换 PG、加一层缓存、表结构迁移 |

这三个变化在真实项目里是**独立发生**的 —— 换协议不该动业务规则，换数据库不该动接口契约。这才是分层成立的理由。反过来，如果你的 service 层每个方法都只是把参数原样转给 dao，那这两层的变化原因其实是同一个，分层只是在制造样板代码，该合。

**为什么不是更多层。** 每加一层，就多一次数据结构转换（DTO → Entity → PO），多一处可能漏字段的地方。层数的上限由"你能列出几个**真正独立**的变化原因"决定。常见的第四层是「领域层」（DDD 的 domain），它成立的前提是业务规则复杂到需要独立建模；如果业务就是增删改查，硬加一层领域模型只会让每次改字段都要动四个文件。

---

**再讲 Agent/AI 应用的分层跟传统后端有什么不同**

这一点是这道题在 AI 岗上的真正考点。传统三层的假设是：**每一层的失败都是异常**，往上抛，接入层统一转成 500。

AI 应用把这个假设打破了，因为多了一类东西 —— **「成功了，但结果不可靠」**。检索返回了 0 条文档，这不是异常；模型返回了一段话但没有依据支撑，这也不是异常。它们在类型系统里都是"正常返回"，可是业务含义完全不同。

于是分层的职责要多一条：**降级决策必须发生在有业务语义的那一层**。

为什么必须在业务层？看这三层各自看到什么：

- 数据层看到的是「查询返回了空列表」—— 它不知道空列表意味着什么
- 接入层看到的是「业务层返回了一个对象」—— 它不知道这个对象里的答案是否有依据
- 只有业务层同时知道「用户问的是什么」「检索到了什么」「工具查到了什么」，才能判断这次该正常回答、该降级回答、还是该拒答

这就是为什么 AI 应用的业务层通常比传统 CRUD 的 service 层厚得多 —— 它承载的不只是流程编排，还有**不确定性的收敛**。

---

**我项目里的三层，以及每层是被什么逼出来的**

**接入层（`app/api/`）—— 被「状态码要说真话」逼出来的。**

这个仓库里有一段真实的修复记录：原来端点是 `return {"code": 500, "message": "..."}`，HTTP 状态码给的是 **200**。代码看起来没问题，日志里也有错误，但整个监控体系是瞎的 —— 因为下面这些消费方**只读 HTTP 状态码**，读不懂响应体里的 `code` 字段：

- nginx 的 `$status`（访问日志、错误率统计）
- 负载均衡的健康检查（决定要不要把这个实例摘掉）
- APM 的错误采样（决定哪些请求值得抓 trace）
- Prometheus 的 `http_requests_total{status="5xx"}`
- 云厂商 SLB 面板上那条错误率曲线

结果是：服务在大面积报错，而所有面板上的错误率是 **0%**。这类故障最难查，因为监控本身在骗你。

所以接入层的职责被定死了：**把内部错误语义翻译成协议语义**。`http_status_of`（`app/core/errors.py:311-325`）做映射，全局处理器（`app/core/exception_handlers.py:148-158`）统一落地。业务层从头到尾不需要知道 HTTP 是什么 —— 它抛 `RetrievalError`，翻成 503 是接入层的事。

**业务层（`app/services/`、`app/agent/`）—— 被「两种空结果必须分开」逼出来的。**

`generate_node` 在拿到 0 篇文档时，必须区分两种情况：

1. 检索服务挂了（Milvus 连不上 / 熔断器打开）→ 应该告诉用户「检索服务暂时不可用」，运维该去修服务
2. 检索正常，知识库里确实没有 → 应该告诉用户「知识库里没有相关内容」，运营该去补文档

这两句话给用户的体验差别不大，但**给排障的人的指向完全相反**。而这个区分只有业务层做得到 —— 数据层只看到空列表，接入层只看到一个正常返回的对象。

我用一个三态枚举 `RetrievalStatus`（`app/core/errors.py:65-78`）把它固化下来：`OK` / `EMPTY` / `FAILED`。为什么必须是三态而不是"返回空列表表示没有、抛异常表示失败"？因为**并行检索的部分失败**是第四种情况 —— 3 个子查询里 1 个挂了、2 个成功，这既不是 OK 也不是 FAILED。这种情况我用 `PARTIAL_RETRIEVAL` 这个降级原因单独标出来，答案照发但标注"证据可能不全"。

**数据层（`app/repositories/`、`app/models/`）—— 被「失败写回必须独立于业务事务」逼出来的。**

这条最反直觉，也是我踩过最深的坑。

场景：异步任务跑文档索引，跑到一半 Milvus 抛异常。我要在 `except` 块里把任务状态从 RUNNING 改成 FAILED、把文档状态从 INDEXING 改成 FAILED，好让前端知道这活废了。

自然的写法是复用传进来的那个 session：

```python
except Exception as exc:
    job.status = JobStatus.FAILED      # 用业务 session
    db.commit()                        # ← 这里会二次抛异常
    raise
```

问题在于：**业务 session 上的连接此刻可能已经是坏的**。异常可能就是数据库连接断了引起的，或者事务已经进入 aborted 状态（PostgreSQL 里一个语句失败后，同一事务里的后续语句全部拒绝执行）。这时 `db.commit()` 会抛第二个异常，而第二个异常会**覆盖掉第一个** —— 真正的失败原因从 `job.exc_info` 里被挤掉了。

结果：任务永远停在 RUNNING，文档永远停在 INDEXING，前端转圈，而日志里只有一句莫名其妙的 "connection already closed"。

正确做法是**开一个全新的 session** 专门写失败：

```python
def record_job_failure(job_id: str, exc: BaseException) -> None:
    db = SessionLocal()          # 全新连接，不复用业务 session
    try:
        ...                      # 写 FAILED 状态
        db.commit()
    except Exception as e:
        logger.warning(...)      # 写不进也只记日志，绝不再抛
    finally:
        db.close()
```

代码在 `app/core/job_failure.py:146-183`。这里还有一条纪律：**这个函数本身绝不抛异常**。它是善后设施，不是业务功能 —— "因为记不下失败原因，所以失败得更彻底了"是荒谬的因果。

---

**所以我的答案是：三层的边界不是按「controller-service-dao」的习惯划的，是按故障传播路径划的。**

每一层要回答一个不同的问题：
- 接入层：这个失败对外**说成什么**（协议语义）
- 业务层：这个失败**算不算失败**（降级决策）
- 数据层：这个失败**怎么留痕**（独立于业务事务）

如果面试官追问"那你怎么防止业务层泄漏到接入层"，我的答案是看依赖方向 —— `app/api/` 可以 import `app/services/`，反过来不行；业务层的异常类型定义在 `app/core/errors.py` 而不是 `app/api/` 里。这是依赖倒置（SOLID 的 D）在包结构上的体现：底层不认识上层。

**项目证据**

- `app/core/exception_handlers.py:1-60` — 只读状态码的消费方清单（那段模块文档就是这次修复的记录）
- `app/core/exception_handlers.py:148-158,161-177,180-191` — 三个处理器与注册顺序
- `app/core/errors.py:311-325` — `http_status_of`，错误语义 → 协议语义
- `app/core/errors.py:65-78` — `RetrievalStatus` 三态
- `app/agent/rag_v2/nodes.py:264-331` — `generate_node` 区分两种无证据
- `app/core/job_failure.py:146-183` — `record_job_failure`，独立 session
- `app/core/database.py:20-33` — `SessionLocal`、`get_db`
- 领域层（DDD domain）：项目中没有，业务复杂度没到那一步

#### Q2 · 意图识别简单问题拦截这块，有没有记忆机制？

**回答**

分两半答，因为这两半在这个仓库里一半有一半没有。

**意图识别拦截：项目中没有，不能硬套。** 没有意图分类器，也没有「简单问题直接回、复杂问题走 RAG」的分流。`app/agent/rag_v2/graph.py:43-48` 的边是固定的，每个请求都走完 rewrite → retrieve → dedup → generate → validate 五步。唯一的条件边是 `app/agent/rag_v2/graph.py:17-19` 的 `_fanout_to_retrieve`，那是并行扇出不是意图路由。LangChain 那条路径（`app/services/rag_agent_service.py:133-137`）由模型自己决定调不调工具，也不是显式的意图识别。

**记忆机制：有，而且它确实参与「理解当前问题指的是什么」。** `app/agent/rag_v2/nodes.py:101-115` 把会话上下文包进 `<conversation_memory>` 标签塞给 rewrite 节点，并且明确要求「利用记忆消解代词和省略的对象」。所以用户说「那个服务的 CPU 呢」，rewrite 能靠记忆把「那个服务」还原成具体名字。

这里有一处刻意的约束值得说：`app/agent/rag_v2/nodes.py:87-88` 的 `MEMORY_POLICY` 写着——会话记忆只用于理解用户已明确说明的对象、任务延续和约束，它不是知识库证据，更不是实时监控、日志或工具查询结果，**不得将历史指标写成当前实时值**。这条约束针对的是一类具体的幻觉：用户十分钟前问过 CPU 是 92%，现在问「现在怎么样」，模型很容易把记忆里那个数字当成当前值报出来。

**项目证据**

- 意图识别拦截：项目中没有，不能硬套
- `app/agent/rag_v2/graph.py:17-19,43-48` — 固定边 + 扇出条件边
- `app/agent/rag_v2/nodes.py:101-115` — 记忆注入 rewrite
- `app/agent/rag_v2/nodes.py:87-88` — `MEMORY_POLICY`
- `app/api/chat_v2.py:28-34` — `_load_conversation_memory`，记忆失败降级为空而不阻断聊天

#### Q3 · MCP 工具主要包含哪些？

**回答**

两个自建 server，七个工具，都是 `streamable-http` 传输。

**CLS 日志服务**（`mcp_servers/cls_server.py`，`FastMCP("CLS")` 在第 20 行，跑在 8003 端口 `/mcp` 路径，第 470 行），五个工具：

| 工具 | 行号 | 作用 |
|---|---|---|
| `get_current_timestamp` | 106 | 取当前时间戳 |
| `get_region_code_by_name` | 138 | 地域名 → 地域码 |
| `get_topic_info_by_name` | 171 | 日志主题名 → 主题信息 |
| `search_topic_by_service_name` | 214 | 服务名 → 日志主题 |
| `search_log` | 348 | 日志检索 |

**Monitor 监控服务**（`mcp_servers/monitor_server.py`，`FastMCP("Monitor")` 在第 27 行，8004 端口，第 435 行），两个工具：`query_cpu_metrics`（126）、`query_memory_metrics`（279）。

这套工具的形状值得说一句：前四个 CLS 工具全是**查 ID**，只有 `search_log` 是真正取数据。原因是日志检索需要地域码和主题 ID，而模型手里只有「order-api 这个服务」这种人类说法。把 ID 解析拆成独立工具，比让模型猜 ID 靠谱得多——猜错了 `search_log` 会返回空，看起来像「没有日志」。

客户端在 `app/agent/mcp_client.py`。三个点：

一、`get_mcp_client`（84-127）是单例，`_mcp_client` 模块级缓存。注释里记着一个版本行为：langchain-mcp-adapters 0.1.0 起 `MultiServerMCPClient` 不再支持作为上下文管理器使用，直接创建实例即可，不需要 `__aenter__()`。

二、`retry_interceptor`（18-74）是指数退避，`wait_time = delay * (2 ** attempt)`（第 64 行），默认三次。

三、**重试耗尽后返回而不是抛**（71-74）：`CallToolResult(content=[TextContent(...)], isError=True)`。这是刻意的——工具失败是模型能处理的信息（换个参数、告诉用户不可用），抛出去会把整个 agent 循环炸掉。

配置在 `app/config.py:143-160`，`mcp_servers` 写成 `@property` 而不是模块常量，这样改 URL 只需要改环境变量。

MCP 服务不可用时的降级在 `app/services/rag_agent_service.py:120-128`：捕获异常、记 warning、`self.mcp_tools = []`，基础聊天和知识库问答照常。注释写着「MCP 服务是增强工具，不应阻塞基础聊天」。

**项目证据**

- `mcp_servers/cls_server.py:20,104,136,169,212,346,470` — CLS server 与五个工具
- `mcp_servers/monitor_server.py:27,124,277,435` — Monitor server 与两个工具
- `app/agent/mcp_client.py:18-74` — `retry_interceptor`，指数退避
- `app/agent/mcp_client.py:71-74` — 耗尽后 return `isError=True`
- `app/agent/mcp_client.py:84-127` — 单例，含版本行为注释
- `app/agent/mcp_client.py:130-158` — `get_mcp_client_with_retry`，拦截器排在最前
- `app/config.py:143-160` — MCP 配置与 `mcp_servers` property
- `app/services/rag_agent_service.py:120-128` — MCP 不可用时降级

#### Q4 · 项目里记忆模块具体怎么实现？

**回答**

`app/services/conversation_memory_service.py` 292 行，核心是 `build_context`（93-131）。实现方式是**滚动摘要 + 有界最近窗口 + 游标**。

流程：读活跃消息 → 取上次快照 → 用快照的 `summarized_through_message_id` 算出之后的 `pending_messages` → 保留最近几轮原文（`conversation_memory_recent_turns = 3`，`app/config.py:136`）→ 剩下的如果超过 `compact_threshold_tokens`（2400，`app/config.py:135`）就压成摘要 → 存新快照并更新游标。

四个细节是这个实现真正的内容。

**一、游标必须防软删。** `_messages_after_snapshot`（133-144）在游标指向的消息被软删时返回 `[]`。否则会把已经摘要过的内容重新摘一遍，摘要越滚越长还越滚越重复。

**二、摘要 prompt 用固定小标题。** `_SUMMARY_SYSTEM_PROMPT`（28-35）规定输出【明确事实】【任务进展】【约束与偏好】【待确认项】四段。固定结构的目的是让下一轮摘要能在同样的骨架上增量更新，而不是每次重新组织一遍——自由格式的摘要滚三轮就会丢掉早期的约束条件。

**三、Token 估算要认中文。** `estimate_tokens`（48-54）对 CJK 字符按一字一 token 算，其余按四字符一 token。用通用的 `len/4` 会把中文严重低估，预算形同虚设。

**四、摘要失败不阻断对话。** `_summarize`（163-186）失败时降级为只用最近窗口。记忆是增强，不是前置依赖。这条在 `app/api/chat_v2.py:28-34` 的 `_load_conversation_memory` 里又兜了一层：整个记忆加载失败也只是返回空上下文。

渲染在 `_render_context`（207-247）：先给头部文字预留 token，然后硬上限截断。`_take_tail_to_budget`（249-272）保尾不保头——最近的对话优先。

四个预算参数在 `app/config.py:133-136`：注入上限 1800、摘要上限 700、压缩阈值 2400、最近轮数 3。

快照落库的表结构在 `migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35`。**原始消息全量保留在数据库里**，摘要只是投影——需要回放或按需召回时原文还在。

**项目证据**

- `app/services/conversation_memory_service.py:93-131` — `build_context`
- `app/services/conversation_memory_service.py:133-144` — `_messages_after_snapshot`，软删防御
- `app/services/conversation_memory_service.py:28-35` — 固定小标题
- `app/services/conversation_memory_service.py:48-54` — `estimate_tokens`，CJK 感知
- `app/services/conversation_memory_service.py:146-158` — `_recent_message_count`，按用户轮次计
- `app/services/conversation_memory_service.py:163-186` — `_summarize`，失败降级
- `app/services/conversation_memory_service.py:207-247,249-272` — 渲染与保尾截断
- `app/services/conversation_memory_service.py:38-45` — `ConversationMemoryContext` dataclass
- `app/config.py:133-136` — 四个预算参数
- `app/api/chat_v2.py:28-34` — 外层再兜一次
- `migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 快照表

#### Q5 · 如何衡量agent效果好坏？

**回答**

这题我先把「衡量」这个词拆开，因为它其实混着三个互相答不了的问题。想清楚这一层，答案就自然分层了。

---

**第一步：三个问题，三套指标**

| 问题 | 什么时候问 | 用什么答 | 特征 |
|---|---|---|---|
| **这次为什么错了** | 有人报障之后 | 日志 + trace | 有个体信息，能定位到具体一次 |
| **整体现在什么形状** | 值班盯盘 | 指标（metrics） | 只有聚合，没有个体 |
| **改动之后变好还是变坏** | 上线前 | 离线评测集 | 可复现，能对比两个版本 |

这三个**不能互相替代**，这是最关键的一点，我详细说说为什么。

日志答不了第二个问题。你没法从日志里问出"过去一小时降级率是多少"——那需要遍历几百万行做聚合，等你算完故障已经过去了。

指标答不了第一个问题。指标里没有任何个体信息（下面会讲为什么**不能**有），所以"张三那次查询为什么没给出 SOP"这个问题，指标一个字也答不上来。

评测集答不了前两个。它跑的是固定用例，真实流量里的长尾问题它根本没见过。

所以值班的实际动线是：**指标发现异常 → 日志/trace 定位个体 → 修完之后用评测集验证没有回退**。缺任何一环，链路就断在那里。只有指标就知道"有问题"但不知道在哪；只有日志就得等人报障才会去看；只有评测集就不知道线上到底怎么样。

---

**第二步：指标的四种类型，选错了会得出反向结论**

Prometheus 有四种指标类型，Agent 系统里主要用三种。这块必须讲清楚，因为选型错误会让你的监控**说谎**。

**Counter（计数器）** —— 只增不减的累计量。适合"发生了多少次"：降级次数、失败次数、工具调用次数。查询时用 `rate()` 算增长率。

**Gauge（仪表）** —— 可增可减的**当前值**。适合"现在是什么状态"：熔断器开着没有、队列里堆了多少任务、连接池用了几个。

**Histogram（直方图）** —— 把观测值分桶计数。适合耗时、体积这种要看分布的量。

**Summary（摘要）** —— 也算分位数，但**在客户端算**。这是最容易踩的坑。

耗时为什么必须用 Histogram，不能用 Gauge 或 Summary，三个理由：

- **Gauge 只记最后一次。** 平均值和 P99 都算不出来。故障的时候你想知道"是所有请求都慢了，还是只有 1% 的请求慢得离谱"，Gauge 一个字也答不了——这两种情况的排查方向完全不同。
- **Summary 在客户端算分位数，多实例部署时无法聚合。** 这条要理解透：假设你有 3 台机器，各自算出 P99 是 2s、3s、10s。这三个数字**求平均没有任何数学意义**，因为你不知道每台机器处理了多少请求、分布长什么样。真实的全局 P99 可能是 8s，也可能是 3s，从这三个数推不出来。
- **Histogram 存的是分桶计数，可以跨实例相加再算分位数。** 三台机器的"落在 0.5s 桶里的请求数"直接相加是有意义的，加完再算分位数就是全局的真值。

Histogram 的代价是**要预先定死桶边界**，而这个边界选错了指标就废了。我项目里的桶是这样（`metrics.py:84`）：

```
0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0
```

选型依据有两条。**密在 1~10s** 是因为那是正常区间，要能看出 P50/P90 的移动——性能退化最早就体现在这里。**上限一直铺到 120s** 是因为单次 LLM 超时配的是 60s、总预算 90s，桶必须比超时值大；否则所有超时样本全挤进 `+Inf` 那个桶，而"慢到什么程度"恰好是排障最需要的信息。

工具调用我另开了一套桶（`metrics.py:100`），`0.05` 起步：

```
0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0
```

理由是直接复用 LLM 那套不行——它最小刻度 0.5s，而向量检索、MCP 查日志正常在几十毫秒到几秒，九成样本会挤进第一个桶，P50 的移动完全看不出来。而"工具开始变慢"恰好是链路出问题时**最早**出现的症状。

---

**第三步：label 基数纪律 —— 这是指标最容易出的生产事故**

这一条我单独讲，因为它是那种"加一行代码，一周后把监控系统打爆"的坑。

机制是这样：Prometheus 里每一个 **label 值的组合**都会生成一条**独立的时间序列**，每条序列常驻内存。所以 label 的取值集合必须**有界且很小**。

我项目里所有 label 都来自枚举或写死的字符串：

- `node` → 五个节点名，有界
- `reason` → `DegradeReason` 枚举，十来个值，有界
- `code` → 错误码，来自异常类名转换，有界
- `tool` → 工具名，代码里写死的，有界

**绝对不能**做 label 的：`request_id`、`trace_id`、用户问题原文、文件名、告警实例 IP。这些是无界的，加进去就是内存泄漏。这类信息属于**日志**，不属于指标。

同一个坑还有个隐蔽版本：**HTTP 指标必须按路由模板聚合**。如果你手写中间件、用 `request.url.path` 当 label，拿到的是展开后的真实路径——`/api/aiops/trace/abc-123`、`/api/aiops/trace/def-456`……每个 trace_id 一条序列，几万次查询就把 Prometheus 打爆。必须用 `/api/aiops/trace/{trace_id}` 这个**模板**。

这是我引 `prometheus-fastapi-instrumentator` 这个库、而不是自己写中间件的唯一实质理由——它会从 Starlette 的 route 对象里取模板。对比一下：熔断器我就没引 pybreaker，自己写了六十行，因为那里没有这种坑。**引库的判据是"这个库帮我避开了什么我自己想不到的坑"，不是"省了多少行代码"。**

---

**第四步：Agent 特有的质量指标（不只是系统健康）**

上面那套是任何后端服务都该有的。Agent 系统还要额外量三件事：

**一、降级率，且必须按原因拆开。** 这是我认为 Agent 可观测性里最重要的一个指标，理由在下一题（Q6）会展开——核心是 `reason` 这个 label 的划分标准应该是**修复方向**，不是错误类型。

**二、检索质量。** 命中率（Hit@K，前 K 条里有没有一条是对的）、召回率（Recall@K，该找到的找到了几成）、MRR（Mean Reciprocal Rank，第一条正确结果排在第几位的倒数——排第 1 得 1 分，排第 3 得 0.33 分）。这三个是**离线**指标，需要标注好的 golden set。我项目里 `tests/eval/metrics.py:75/81/90` 就是这三个，跑在两个标注集上（18 条基础 + 83 条扩展）。

**三、生成质量。** 分两个维度，业界通常叫：

- **Faithfulness（忠实度/事实性）** —— 答案里的每个断言，能不能在给的证据里找到支撑。这是**幻觉的直接度量**。
- **Answer Relevance（答案相关性）** —— 答案有没有真的回答用户的问题。一个完全忠实但答偏了的回答，这一项就该低分。

这两个跟前面的检索指标有个本质区别：**没有唯一正确答案可以字符串比对**，所以要么人工标注，要么用另一个模型判分（LLM-as-Judge）。

我项目在这里是**半成品，得说实话**：`tests/eval/metrics.py:174-185` 两个函数 `faithfulness_placeholder` 和 `answer_relevance_placeholder` 直接抛 `NotImplementedError`，注释是 `TODO(week 2): 接入 ragas`。所以**离线**的生成质量量化我还没做完。

但**线上**这一层我做了 —— `validate_answer_node` 是个跑在主链路上的 LLM 判分器，输出 `coverage_score`（覆盖度，对应 Answer Relevance）和 `groundedness_score`（有据度，对应 Faithfulness），阈值 0.60 / 0.75，不过就换成保守回答。两者的差别是：线上这个允许有噪声、不需要复现；离线那个必须确定性可复现，才能用来对比两个版本。这也是为什么不能拿线上那个当离线评测用。

---

**第五步：介于线上和离线之间的一层 —— 坏例回放**

这一层我觉得比前面两层都实用，但很多人不做。

做法：每次请求把完整执行链路落库——问题、改写出的子查询、召回了哪些文档、用了哪几篇、答案、判分结果、降级原因、每个节点的耗时，然后**自动标记坏例**（判分不过、有降级、耗时超阈值的），提供一个接口列出来。

价值在于它填了一个真实的空白：线上指标告诉你"降级率涨了 3%"，但你**没有样本可看**。有了坏例列表，你可以直接调出那 3% 里的具体请求，一个个看是怎么错的，然后把它们**加进离线评测集**——这样评测集是从真实流量长出来的，不是拍脑袋编的。

我项目里这套是 `chat_run_traces` 表（trace 级）加 `chat_run_spans` 表（节点级），接口是 `chat_v2.py:37` 列坏例、`:55` 查单条 trace。`is_bad_case` 是个建了索引的布尔字段，就是为了这个查询。

---

**总结成一句面试能说的话**

衡量 Agent 分三层：**系统健康**（延迟分布、失败率、熔断状态，用 Prometheus 指标，注意 label 基数）、**答案质量**（检索侧 Hit/Recall/MRR 可以精确算，生成侧 Faithfulness/Relevance 要靠标注或模型判分）、**可回放的个案**（trace + span 落库 + 坏例自动标记，用来喂养评测集）。三层缺一层，就会出现"知道有问题但找不到"或者"改完了不知道有没有变好"。

我项目里第一层和第三层是完整的，第二层的检索侧完整、生成侧的离线量化还是 placeholder——这一点我不掩饰，因为掩饰了追问一句就穿。

**项目证据**

- `app/core/metrics.py:1-67` — 模块文档：指标与日志为什么不能互相替代、label 基数纪律、为什么用 Histogram
- `app/core/metrics.py:84` — LLM 耗时桶边界，铺到 120s 的理由
- `app/core/metrics.py:100` — 工具耗时另开一套桶，0.05s 起步
- `app/core/metrics.py:86,102,113,132,148,157,163,173` — 八个指标定义
- `app/core/metrics.py:198-215` — `observe_llm_call`，try/finally 保证超时样本也记
- `app/core/metrics.py:254-293` — `observe_tool_call`，计时与失败计数合成一个（记账点唯一）
- `app/core/metrics.py:296-304` — `count_degrade`，reason 为 None 归 `unknown` 而不丢弃
- `app/core/metrics.py:312-329` — 抓取时刷新熔断器状态（因为 OPEN→HALF_OPEN 是懒转换）
- `app/models/chat_run_trace.py:22-54` — trace 表结构，含 `is_bad_case` 索引字段
- `app/models/chat_run_span.py:52,67` — span 表与状态枚举
- `app/api/chat_v2.py:37,55` — 坏例列表与 trace 详情接口
- `tests/eval/metrics.py:75,81,90,96,120` — Hit@K / Recall@K / MRR / 单例计算 / 聚合
- `tests/eval/metrics.py:174-185` — Faithfulness 与 Answer Relevance 仍是 `NotImplementedError` 占位
- `app/agent/rag_v2/nodes.py:36-37` — 线上判分阈值 0.60 / 0.75
- 工具调用成功率的**业务语义**（区分"工具报错"和"工具返回了但没用上"）：目前只埋了前者，后者没有

#### Q6 · 如果项目上线，可以用哪些指标评估效果？

**回答**

上一题问的是"怎么衡量质量"，这题问的是"上线之后运维看什么"。这两件事经常被混在一起答，但它们的**消费者不同**——质量指标给算法/产品看，决定要不要改 prompt、补文档；上线指标给值班的人看，决定半夜三点被叫起来之后先去动哪个服务。我先把这个领域的通用框架讲清楚，再说我项目里具体埋了什么。

---

**一、先建立三个基础概念：SLI、SLO、SLA**

这三个词经常被混用，但层次是清楚的：

- **SLI（Service Level Indicator，服务等级指标）**：一个**能测量的数**。比如"过去 5 分钟内 HTTP 5xx 的比例"、"P99 响应耗时"。SLI 必须是可以从监控系统里直接查出来的具体数值。
- **SLO（Service Level Objective，服务等级目标）**：给 SLI 定的**内部目标**。比如"5xx 比例 < 0.1%"、"P99 < 3 秒"。SLO 是团队自己承诺的线，达不到就要停下来修，而不是继续加功能。
- **SLA（Service Level Agreement，服务等级协议）**：写进**合同**给客户的承诺，通常比 SLO 宽松（SLO 是内部预警线，SLA 是对外底线），违约要赔钱。

为什么要分这三层：SLO 严于 SLA，是为了**在违约之前就发现问题**。如果内部目标和对外承诺一样紧，那么内部报警响起的那一刻，合同已经违约了。

还有一个配套概念叫 **Error Budget（错误预算）**：如果 SLO 是"可用性 99.9%"，那么一个月允许不可用的时间就是 43 分钟。这 43 分钟是**可以花的预算**——用来做发布、演练、灰度。预算没花完，说明团队可以更激进地发版；预算超支，说明必须冻结发布先修稳定性。这个框架把"稳定性"和"迭代速度"从对立关系变成了一个可量化的权衡。

---

**二、通用的指标框架：Golden Signals / RED / USE**

业界有三套常用的指标清单，各自适用的对象不同，选错了就会漏掉关键信号：

**Google SRE 的四个黄金信号（Golden Signals）**，适用于**用户可见的服务**：

| 信号 | 含义 | 典型 SLI |
|---|---|---|
| Latency（延迟） | 请求耗时。**成功请求和失败请求要分开算** | P50/P95/P99 耗时 |
| Traffic（流量） | 请求量 | QPS、并发数 |
| Errors（错误） | 失败率 | 5xx 比例、业务错误码比例 |
| Saturation（饱和度） | 资源用到几成 | 队列深度、连接池占用率、内存水位 |

延迟必须区分成功和失败，这一条最容易漏。快速失败（连接被拒，1 毫秒返回）会把平均延迟拉低，看起来"服务变快了"，而真相是大部分请求根本没被处理。

**RED**（Rate / Errors / Duration），适用于**请求驱动的服务**，是黄金信号的简化版，少了 Saturation。

**USE**（Utilization / Saturation / Errors），适用于**资源**——CPU、内存、磁盘、网络。Utilization 是"忙的时间占比"，Saturation 是"排队的量"。这两个的区别很重要：CPU 利用率 100% 不一定有问题（可能刚好跑满），但如果同时有很长的运行队列，说明请求在排队等 CPU，那才是饱和。

**为什么这三套不能互相替代**：RED 看不到资源耗尽（连接池满了但请求还在慢慢成功，RED 三个指标都正常），USE 看不到用户体验（CPU 很闲但因为下游超时导致用户全在等）。真实系统要两边都埋。

---

**三、LLM Agent 系统需要额外埋什么**

上面那三套是通用后端的框架，直接搬到 Agent 系统上会漏掉这一类系统特有的失败模式。Agent 特有的至少这几类：

**1. 成本指标。** 传统后端不需要监控"这次请求花了多少钱"，Agent 需要。要埋 token 消耗（区分输入/输出，因为输出通常贵 3-4 倍）、按模型分、按用户/租户分。没有这个指标，一次 prompt 改动导致上下文膨胀三倍，账单月底才发现。

**2. 降级率，且必须按原因拆开。** Agent 系统的典型特征是**大量失败不表现为 HTTP 错误**——检索挂了但答案照样生成（只是没依据）、工具调不通但模型编了个结果、质检器挂了但答案放行了。这些在 HTTP 层面全是 200。如果只看 5xx，这个系统在崩溃的过程中会一直显示"健康"。

**3. 质量指标的线上代理量。** 离线评测跑的是固定集，线上没有标准答案。所以要埋**能反映质量的间接量**：拒答率、"我不知道"的比例、答案长度分布（突然变短通常意味着上下文被裁太狠）、引用文档数为 0 的比例、用户重问率、点赞点踩比。

**4. 步数与循环。** ReAct 类系统要埋每次请求的工具调用次数分布。这个分布的**右尾**是关键——如果 P99 是 12 步而配置上限是 15，说明有一批请求在接近失控。

**5. 缓存命中率。** 有 prompt 缓存或语义缓存的话，命中率直接决定成本和延迟。

---

**四、指标分层的原则：按"看到之后去动谁"来分**

这是我在项目里实际用的判据，也是我觉得这道题最该讲的一点。

指标不是越多越好，指标多到没人看就等于没有。所以我分层的标准不是"技术栈的层次"，而是**值班人员看到这个指标之后，下一步该去动哪个系统**。同一层内的指标指向同一个处置动作，不同层之间处置动作不同。

按这个原则，我这个项目上线要看六层：

**第一层：请求是否成功。** `http_requests_total{status="5xx"}`。

这一层有个前提，很多人不知道它会失效：**状态码必须说真话**。我项目里这个前提是**修出来的**——原来端点写的是 `return {"code": 500, "message": "..."}`，HTTP 层面返回的是 **200**。于是 nginx 的 `$status`、负载均衡的健康检查、APM 的错误率、Prometheus 的 `5xx` 计数，**全部显示 0% 错误率**，而系统实际上在报错。

修法是端点里直接 `raise`（`app/api/chat_v2.py:151`），交给全局异常处理器统一翻译成状态码（`app/core/exception_handlers.py:180-191`）。这件事的教训不是"要用 raise"，而是：**任何依赖某个信号的监控，都要先验证那个信号本身是不是真的**。我后来把这条当成一个检查项——每加一个告警规则，先人为制造一次故障，确认告警真的会响。

**第二层：降级率按原因拆开。** `degrade_total{reason=...}`。

这一层是 Agent 系统区别于传统后端的地方，也是我这套设计里最花心思的部分。关键不在"埋了一个降级计数器"，在于 **`reason` 这个 label 的取值是怎么划分的**。

我的划分依据是**修复方向**——两个失败如果处置动作不同，就必须是两个不同的 reason；如果处置动作相同，合并成一个：

| reason | 修复方向 | 该找谁 |
|---|---|---|
| `llm_timeout` / `llm_error` | 模型网关、配额、限流 | 找算法平台 |
| `retrieval_failed` | 检索服务本身挂了 | 找运维查 Milvus |
| `retrieval_empty` | 服务是好的，**知识库里就是没有** | **找业务补文档** |
| `partial_retrieval` | 部分分支失败，证据不全 | 看是否要提召回或加重试 |
| `circuit_open` | 我方熔断器跳了，是我主动不发请求 | 先看下游为什么连续失败 |
| `total_budget_exceeded` | 整体链路太慢 | 看是哪个节点吃掉了预算 |
| `feature_disabled` | 是我自己关的开关 | **什么都不用做** |

`retrieval_failed` 和 `retrieval_empty` 必须分开，这是整张表里最典型的一对。两者在代码里都表现为"没拿到文档"，但一个是去修服务，一个是去补文档。如果合并成一个 `retrieval_error`，值班人员看到它上涨，第一反应会是去查服务，而真相可能是运营删了一批文档——排查方向从一开始就错了。

`feature_disabled` 单独占一个值，也是有意的。它表示"这次降级是我自己配置开关关掉的功能导致的"，处置动作是**不处置**。如果不把它单列，它会混进真实故障的计数里，制造一个永远存在的背景噪声，久而久之团队就会对这个指标脱敏。

**这张表能成立，靠的是一个具体的 bug 修复**，值得单独讲，因为它说明了这类"翻译层"的典型陷阱：

我有个函数 `wrap_llm_exception`（`app/core/errors.py:358-387`），职责是把各种底层异常统一包装成带 `degrade_reason` 的应用异常。它原来的判断条件是 `isinstance(exc, LLMError)`——如果已经是 LLM 类异常就直接返回，否则重新包装。

问题是 `CircuitOpenError`（熔断器打开）继承的是 `AppError`，**不是** `LLMError`。所以它走进了"重新包装"分支，`degrade_reason` 从 `circuit_open` 被覆写成了 `llm_error`。

后果：熔断器跳了（说明下游已经连续失败到被我方判定为不可用），而监控上显示的是"LLM 调用错误"。值班人员被指向"去查模型配额和网关"，实际该做的是"看下游为什么挂了，以及熔断器什么时候该恢复"。**指标不仅没帮上忙，还把人引向了错误的方向**——这比没有指标更糟。

修法是把判断条件从"是不是某个类型"改成"**有没有已经确定的降级原因**"：`exc.degrade_reason is not None`（第 380 行）。这个改动的思路是——**已经有明确归因的异常，不要再被上层重新归因**。类型检查会随着继承关系变化而失效，而"有没有归因"这个语义判断不会。

**第三层：检索失败按错误码拆开。** `retrieval_failures_total{code=...}`。

这一层和第二层的关系值得说清，因为看起来像重复：第二层的 `reason` 是**业务语义**（要不要补文档），第三层的 `code` 是**技术故障类型**（连接超时 / 认证失败 / 熔断打开）。同一个 `retrieval_failed` 可以对应好几个 code，而处置细节不同。

埋点位置我放在了节点内部（`app/agent/rag_v2/nodes.py:200`），没放在统一的埋点层。这是个有意的例外——统一埋点层能看到的只是"这个节点降级了"这个笼统事实，拿不到具体错误码，而 code 恰恰是区分"Milvus 连不上"和"熔断器打开"的那个 label。**能统一的地方要统一（DRY），但统一层拿不到的信息，宁可在具体位置多写一行。**

**第四层：熔断器状态。** `circuit_breaker_state{name=...}`。

用 Gauge（当前值）而不是 Counter（累计值），因为熔断状态是"现在开着还是关着"，Counter 只增不减，表达不了"又闭合了"。

取值按**严重程度递增**编号：closed=0、half_open=1、open=2。这样告警规则可以直接写成 `max_over_time(circuit_breaker_state[1h]) >= 2`，不需要在表达式里做字符串匹配。**给枚举值编号时按严重度排序，是为了让阈值比较有意义**——如果随便编号（closed=1、open=0），任何基于大小的告警规则都写不出来。

**第五层：耗时分布，按节点拆。** `llm_call_duration_seconds{node=...}`。

作用是回答"慢在哪一步"。整体 P99 变差是个结果，不是原因；按节点拆开才知道是 rewrite 慢了还是 validate 慢了。

**第六层：可回放的完整链路。** trace + span 落库。

前五层都是**聚合**，回答"整体什么形状"。但值班的动线是"指标发现异常 → 定位到具体某次请求 → 看它到底发生了什么"，最后这一步聚合指标答不了。所以每次请求的执行链路要能被完整调出来：走了几个节点、各自耗时、召回了几条、质检分多少、哪一步降级了。

---

**五、还有两类必须埋但容易被忘的**

**1. 成本。** 我项目里**没有**埋 token 消耗和费用，这是个真实的缺口。当前只有耗时直方图，回答不了"这个月花了多少钱"、"哪类查询最贵"。补法很明确：在 LLM 调用的统一包装处拿 `response.usage_metadata`，按 `(model, node, 输入/输出)` 三个 label 记 Counter。之所以现在没做，是因为项目里只有一个模型、还没有多租户，成本归因暂时没有消费者——但上线前必须补上，标 `[假设]`。

**2. 业务效果。** 上面全是系统指标。真正决定这个项目值不值得继续投入的是业务指标：值班人员的**首响时间**是否真的缩短了、报告的采纳率、有多少次是人工推翻了系统结论。这些数据需要产品侧配合埋点（用户行为、工单流转），不在应用代码范围内。面试里如果只答系统指标，会显得是纯后端视角；这一层要主动提，即使自己没做过。

---

**六、一个安全问题必须主动说**

`/metrics` 端点默认是**无鉴权**的（`app/config.py:125` 的 `enable_metrics_endpoint`）。它吐出的东西信息量很大：系统有哪些节点、各节点耗时、降级率、熔断器状态、有哪些模型和工具。这些是内部架构形状，不该对公网可见。

我的处置是在配置注释里写明**生产环境必须在网关层限制来源 IP**，应用里这个开关只是最后一道保险。没有在应用内加 Basic Auth，理由是那会让抓取端配置复杂化（密码要发给 Prometheus 并跟着轮转），而"限制谁能访问"这件事在网关/ACL 层做既标准又彻底。

这个取舍在面试里值得主动讲——不是因为它有多聪明，而是因为**主动指出自己系统的安全缺口，比等面试官问出来要好**。

**项目证据**

- `app/api/chat_v2.py:151` — 裸 `raise`，让状态码说真话
- `app/core/exception_handlers.py:180-191` — 全局处理器统一翻译
- `app/core/errors.py:25-62` — `DegradeReason`，划分依据是修复方向
- `app/core/errors.py:358-387` — `wrap_llm_exception`，含 `circuit_open` 被误翻成 `llm_error` 的记录
- `app/core/errors.py:380` — 改判 `exc.degrade_reason is not None`
- `app/core/errors.py:280-288` — `error_code_of`
- `app/agent/rag_v2/nodes.py:200` — `count_retrieval_failure` 埋在节点内的理由
- `app/core/metrics.py:157-167` — `degrade_total`、`retrieval_failures_total`
- `app/core/metrics.py:173-186` — `circuit_breaker_state` 与按严重度编号
- `app/config.py:125` — `enable_metrics_endpoint` 与无鉴权说明
- token/费用指标、业务效果指标（首响时间、采纳率、人工推翻率）：项目中没有，补法标 `[假设]`

#### Q7 · 这个项目几个人开发、耗时多久？

**回答**

个人事实题，仓库里能查到的只有客观痕迹：`git log` 的提交历史、`docs/aiops-itops-convergence-daily-plan.md` 和 `docs/roadmap-week1-aiops-rag-and-protocol-pdf.md` 这两份计划文档、`migrations/versions/` 下从 `20260612_0001` 到 `20260824_0009` 九个迁移版本的时间跨度。

回答这题应该报真实的人数和周期，不能拿代码量推算。

**项目证据**

- 项目中没有对应实现，不能硬套
- 可如实引用：`git log`、`docs/aiops-itops-convergence-daily-plan.md`、`docs/roadmap-week1-aiops-rag-and-protocol-pdf.md`、`migrations/versions/`（9 个版本，2026-06-12 至 2026-08-24）

---

### 面经 10 · 2026-07-24

> 原帖这四条是作者给出的「可以这样说」参考话术，不是面试官的提问。按原顺序保留，但答案只能落在自己能验证的事实上。

#### Q1 · 当被问到"为什么选择字节做Agent方向"

**回答**

外部团队事实加个人动机，仓库答不了。

能诚实说的是自己在这个项目里认定的方向：Agent 工程的难点不在模型调用，在模型不可靠时系统对外说什么。`app/core/errors.py` 397 行几乎没有一行在调 LLM，全在回答这个问题。谈方向选择时用这个当论据比夸对方公司实在。

**项目证据**

- 项目中没有对应实现，不能硬套

#### Q2 · 字节在Agent方向的投入和落地节奏一直走在前沿，从豆包的快速迭代到内部Harness框架的开放，能感受到他们不是在做Demo，是在真刀真枪搞生产级Agent？

**回答**

这是话术模板，对方公司的内部情况我无法核实，照念是有风险的——面试官追问一句「你觉得 Harness 解决了什么问题」就露底。

能把这句话变成真话的方式是换成自己的判据：什么算「生产级」而不是 Demo。这个仓库里的判据是五条，每条对应一个被修掉的谎：

1. HTTP 状态码不撒谎（`app/api/chat_v2.py:151`、`app/core/exception_handlers.py:180-191`）
2. 并行分支挂一个不等于整张图崩（`app/agent/rag_v2/nodes.py:189-205`）
3. 检索失败和知识库为空是两件事（`app/core/errors.py:65-78`）
4. 熔断期间耗时指标不会变好看（`app/agent/rag_v2/nodes.py:127-136`）
5. worker 失败原因不被二次异常掩盖（`app/core/job_failure.py:146-183`）

Demo 和生产级的差别就在这五条上，跟框架是谁家的无关。

**项目证据**

- `app/api/chat_v2.py:151`、`app/core/exception_handlers.py:180-191`
- `app/agent/rag_v2/nodes.py:189-205`
- `app/core/errors.py:65-78`
- `app/agent/rag_v2/nodes.py:127-136`
- `app/core/job_failure.py:146-183`

#### Q3 · 我觉得字节在Agent这块最对的一件事是没把Agent做成'大号聊天机器人'，而是真正把它当工程系统来搭？

**回答**

同样是话术。但「当工程系统来搭」这句话可以落到具体的东西上——在这个仓库里，聊天机器人和工程系统的分界线是**有没有 span**。

`app/core/span_context.py` 208 行做的事：请求进来时在 `app/api/chat_v2.py:81` 调 `start_span_collection()`，每个节点执行时由 `app/agent/rag_v2/instrumentation.py:117-146` 的 `instrument_node` 记一条 span，最后在 `app/api/chat_v2.py:109` trace 落库拿到 id 之后 `flush_spans(db, trace_id=trace.id)`。

这里有一个不显然的约束，是 ContextVar 的语义决定的。`app/core/span_context.py:85-108` 的 `record_span` **只 append，绝不 `.set()`**。原因写在模块文档里：`copy_context()` 复制的是绑定关系而不是对象，所以子上下文里 `append` 到同一个 list 对象上，父上下文看得见；但子上下文里 `.set()` 一个新 list，父上下文看不见。并行分支的 span 如果用 `.set()` 就会全部丢掉。

配套的是「入口 set 一次，各处 append」——`start_span_collection` 必须在进图之前调（`app/api/chat_v2.py:81`），流式那条更微妙，得在生成器**函数体内**调（`app/api/chat_v2.py:177`），因为 sse-starlette 真正迭代它的时候外层栈帧已经没了。

聊天机器人不需要这些。工程系统需要，因为出问题时要能回答「这一次到底发生了什么」。

**项目证据**

- `app/core/span_context.py:56,61-67,85-108` — collector、启动、append-only
- `app/core/span_context.py:1-55` — 模块文档，ContextVar 复制语义
- `app/core/span_context.py:111-157` — `span_scope`，`__exit__` 记 ERROR 且返回 `False` 不吞异常
- `app/core/span_context.py:160-196` — `flush_spans`，`add_all` + 单次 commit，永不抛
- `app/agent/rag_v2/instrumentation.py:117-146` — `instrument_node`
- `app/api/chat_v2.py:81,109,156` — 非流式：入口启动、trace 后 flush、finally 重置
- `app/api/chat_v2.py:177` — 流式：生成器体内启动

#### Q4 · 我习惯把职业发展分成技术和业务两条线？

**回答**

个人规划题，仓库不能替我回答。

能提供论据的地方是：这个仓库里技术决策几乎每一条都是被业务语义逼出来的，两条线在实操里分不开。举一个——`DegradeReason` 为什么必须把 `RETRIEVAL_FAILED` 和 `RETRIEVAL_EMPTY` 分开（`app/core/errors.py:25-62`）？技术上它们都是「没拿到文档」，一个枚举值够了。分开的唯一理由是业务上修复方向不同：一个去修服务，一个去补文档。

**项目证据**

- 项目中没有对应实现，不能硬套
- 可作为论据：`app/core/errors.py:25-62`（`DegradeReason` 按修复方向划分）

---

### 面经 11 · 2026-07-15

#### Q1 · 请先做一个简单的自我介绍

**回答**

个人履历题，仓库不能代答。

能从仓库里取的是「说什么」的素材：这个项目是 Python + FastAPI 的企业智能对话与 AIOps 助手，`app/` 76 个文件 11246 行，`tests/` 8275 行。自己投入最多的三块是 LangGraph 编排（`app/agent/rag_v2/`）、统一错误治理（`app/core/errors.py` 397 行）、可观测性（`app/core/metrics.py` + `app/core/span_context.py`）。

自我介绍里不该说的：没上线就不能讲线上收益，没做过多 Agent 就不能讲协商开销。

**项目证据**

- 项目中没有对应实现，不能硬套
- 可如实引用的规模：`app/`（76 文件 11246 行）、`tests/`（8275 行）

#### Q2 · 你上一段实习是在北京做算法工程师对吗？

**回答**

事实确认题，仓库无关。答「是」或「不是」，然后主动补一句这段经历和岗位的连接点。

**项目证据**

- 项目中没有对应实现，不能硬套

#### Q3 · 以你参加的全栈挑战赛中多智能体导购系统为例，你提到iOS客户端部分是边做边学、借助AI辅助完成的。那么在遇到AI无法解决的编译构建或代码问题时，你通常是怎么解决的？

**回答**

原帖的项目背景（全栈挑战赛、iOS 客户端）我没有，不能冒充。

但「AI 解不了的问题怎么办」在这个仓库里有真实案例，而且是最能说明方法的那类——**症状和病因不在同一层**。

`app/core/llm_factory.py` 的模块文档记着一个：LLM 调用会无限期挂住。表面看是超时没生效，但配置里 `llm_timeout_seconds` 明明是 60.0（`app/config.py:70`）。查下去才发现根因在 langchain-openai：它把 `request_timeout`（默认 `None`）**无条件**透传进 OpenAI SDK 的 client，覆盖掉 SDK 自己的 `DEFAULT_TIMEOUT = Timeout(connect=5.0, read=600, ...)`，于是 httpx 拿到 `Timeout(timeout=None)` = 永不超时。对照之下 `max_retries` 用的是 `if ... is not None` 守卫，所以 SDK 默认重试反而正常工作。

这种问题问 AI 问不出来，因为它不在自己的代码里。解法是读依赖库源码，确认参数传递路径，然后把构造收口到一处（`app/core/llm_factory.py:63-102`），并且刻意写成 `timeout = config.llm_timeout_seconds if timeout is None else timeout`（`app/core/llm_factory.py:84`）——用 `is None` 不用 `or`，这样显式传 `0` 也能存活。

方法概括起来是三步：先让症状可复现，再往依赖栈下一层读源码定位真实传递路径，最后把修复收口成唯一实现防止再犯。另一个例子是 `app/core/milvus_client.py:18-41` 的 `_patch_pymilvus_milvus_client_orm_alias`——langchain_milvus 会创建 `cm-{id}` 形式的 alias，而 ORM 的 `Collection(using=...)` 解析不了，只能强制 `_using="default"`。

**项目证据**

- `app/core/llm_factory.py:1-62` — 模块文档，langchain-openai 覆盖 SDK 超时的根因
- `app/core/llm_factory.py:63-102` — `create_chat_model`，构造唯一收口
- `app/core/llm_factory.py:84` — `is None` 而非 `or`
- `app/config.py:70,75` — `llm_timeout_seconds=60.0`、`llm_max_retries=2`
- `app/core/milvus_client.py:18-41` — pymilvus alias 补丁

#### Q4 · 你们系统的核心设计思路是"通过交互确认意图"，对吗？

**回答**

原帖那个导购系统的设计我不能替它回答。这个仓库**没有**交互式意图确认——不会反问用户「您是指 A 还是 B」，`app/agent/rag_v2/graph.py:43-48` 的边是固定的单向链路，没有回到用户的分支。

有一个功能上相邻但语义不同的东西，得说清区别：`app/services/first_response_service.py:101-130` 的系统提示词强制模型把不确定的判断写进 `pending_confirmations` 字段，而不是当成结论输出。这是**把不确定性显式标注给人看**，不是回过头找用户澄清。值班人员看到这个字段就知道哪几条需要自己核实。

真要做交互式意图确认，缺的是这些：一个能在等待用户输入时挂起并持久化的 checkpoint（`MemorySaver` 是内存的，进程重启就没了，`app/services/rag_agent_service.py:107`）、一个澄清轮次上限、以及超时未回复的默认路径。**项目中没有，不能硬套。**

**项目证据**

- `app/agent/rag_v2/graph.py:43-48` — 固定单向边，无澄清回路
- `app/services/first_response_service.py:101-130` — `pending_confirmations`，标注不确定性而非反问
- `app/services/rag_agent_service.py:107` — `MemorySaver()`，进程内 checkpoint
- 交互式意图确认：项目中没有，不能硬套

#### Q5 · 你提到的Router在当前系统中是硬编码实现的，可以按需扩展。但如果用户意图不在Router预定义的范围内，系统如何处理？

**回答**

这个仓库没有意图 Router，但「预定义范围外怎么办」这个问题它以另一种形式回答了，而且答案是可验证的。

`app/agent/rag_v2/graph.py:43-48` 的路由是静态的：`rewrite → retrieve_each（fan-out）→ dedup → generate → validate → END`，唯一的条件边是 `_fanout_to_retrieve`（`app/agent/rag_v2/graph.py:17-19`），它只决定扇出几路，不决定走哪条路。所以不存在「意图落到预定义之外」这个状态——**任何问题都走同一条链路**。

范围外的情况被推迟到了检索环节，由检索结果的三态来表达（`app/core/errors.py:65-78` 的 `RetrievalStatus`）：

- `OK` — 有证据
- `EMPTY` — 检索成功但知识库里没有
- `FAILED` — 检索本身失败

`app/agent/rag_v2/nodes.py:264-331` 的 `generate_node` 在无文档时会分两种话术回答，一种是「检索服务不可用」，一种是「知识库中没有」。这两句对用户的意义完全不同，混成一句就是撒谎。

如果要加 Router，兜底分支必须遵守同一条原则：**不能装作能处理**。落到实现上是三件事——一个显式的 `unknown_intent` 分支而不是 fallthrough 到某个默认技能、一条对应的 `DegradeReason`（`app/core/errors.py:25-62` 的枚举按修复方向划分，「意图无法识别」的修复方向是补 Router 规则，跟检索失败不是一类）、以及回给用户的话里明确说「这个问题超出我能处理的范围」。**Router 本身项目中没有，以上是设计方案，不是已实现功能。**

**项目证据**

- `app/agent/rag_v2/graph.py:17-19,43-48` — 静态边 + 唯一条件边（只管扇出）
- `app/core/errors.py:65-78` — `RetrievalStatus` 三态
- `app/agent/rag_v2/nodes.py:264-331` — 无文档时区分「服务不可用」和「知识库没有」
- `app/core/errors.py:25-62` — `DegradeReason` 按修复方向划分
- 意图 Router：项目中没有，不能硬套

#### Q6 · 请介绍你们对话系统的整体设计，包括上下文管理、服务端数据存储，以及从接口接收请求到Agent处理的全链路设计

**回答**

按请求实际经过的顺序讲，这条链路每一站都能指到行号。

**入口**。`app/core/middleware.py:47-101` 的 `RequestIdMiddleware`，纯 ASGI 中间件而不是 `@app.middleware("http")`——因为后者底层是 `BaseHTTPMiddleware`，会把响应体包一层 anyio 内存流，对 `app/api/chat_v2.py` 的 SSE 端点有影响。它做三件事：复用或生成 `X-Request-ID`、写入 ContextVar（`app/core/request_context.py:33`）、回写响应头。

入口这里有两处不显然的设计：

一是外部传入的 request_id 必须校验（`app/core/middleware.py:34` 的 `_SAFE_REQUEST_ID`）。这个值会出现在每一行日志里，放任 `\n` 通过，攻击者传一个带换行的 header 就能凭空造出一整行假日志。校验不过不报错，当没传处理——追踪 id 不该成为请求失败的理由。

二是 id 除了写 ContextVar 还要写进 `scope["state"]`（`app/core/middleware.py:86`）。这不是冗余，是为了覆盖异常路径。Starlette 的中间件栈是 `ServerErrorMiddleware → RequestIdMiddleware → ExceptionMiddleware → router`，未分类异常冒泡到最外层时，`RequestIdMiddleware` 的 `finally: reset_request_id` 已经执行完了，处理器再读 ContextVar 只能拿到 `NO_REQUEST_ID`。结果是错误响应体里 request_id 恒为 null——而出错恰恰是最需要它的时候。所以 `app/core/exception_handlers.py:65-87` 的 `_resolve_request_id` 优先读 `request.state.request_id`。

**上下文管理**。`app/api/chat_v2.py:28-34` 的 `_load_conversation_memory` 调用记忆服务，失败降级为空上下文，绝不阻塞聊天。记忆服务本体在 `app/services/conversation_memory_service.py:93-131`，是滚动摘要加有限最近窗口：读活跃消息、取快照、算出快照游标之后的 `pending_messages`、保留最近若干轮、超过 `compact_threshold_tokens`（2400，`app/config.py:135`）就把较早的压成摘要、写回新快照并记 `summarized_through_message_id`。

游标这里有个细节：`app/services/conversation_memory_service.py:133-144` 的 `_messages_after_snapshot` 在游标消息被软删除时返回 `[]`，而不是把整段历史当成未摘要重新处理。

**图执行**。`app/api/chat_v2.py:81` 先 `start_span_collection()`（必须在进图之前），然后进 `app/agent/rag_v2/service.py:47-78`。总预算包在这里：`async with asyncio.timeout(config.chat_total_budget_seconds)`（90 秒，`app/config.py:81`），用 `astream(stream_mode="values")` 而不是 `ainvoke`，逐步保留 `last_state`。超时时捕获 `asyncio.TimeoutError` 但**不重新抛出**，而是拿 `last_state` 拼一个部分结果，并标 `TOTAL_BUDGET_EXCEEDED`。

**存储**。`app/api/chat_v2.py:109` trace 落库拿到 id 之后 `flush_spans(db, trace_id=trace.id)`——顺序不能反，span 需要 trace_id 做外键。异常路径在 `app/api/chat_v2.py:126` 先写 trace 再 `raise`（裸 raise，`app/api/chat_v2.py:151`），HTTP 状态码交给全局处理器决定。`finally` 里 `reset_span_collection(span_token)`（`app/api/chat_v2.py:156`）。

**流式那条的差别**。collector 要在生成器**函数体内**启动（`app/api/chat_v2.py:177`），因为 sse-starlette 真正迭代它的时候外层栈帧已经没了。失败时不能 raise——200 状态行早就发出去了——只能 yield 一个 error 事件，带上 `error_code` / `degrade_reason` / `request_id`。

**异步任务**。request_id 通过 RQ 的 `job.meta` 传到 worker，worker 侧用 `app/core/request_context.py:81-97` 的 `request_id_scope` 重新绑定，于是 worker 日志和原请求日志能串起来。

**项目证据**

- `app/core/middleware.py:47-101` — `RequestIdMiddleware`，纯 ASGI
- `app/core/middleware.py:34,37-44` — `_SAFE_REQUEST_ID`，日志伪造防护
- `app/core/middleware.py:70-86` — `scope["state"]`，兜住异常路径
- `app/core/request_context.py:33,45-64,81-97` — ContextVar、`get_request_id_or_none`、`request_id_scope`
- `app/core/exception_handlers.py:65-87` — `_resolve_request_id` 优先读 `request.state`
- `app/api/chat_v2.py:28-34` — 记忆加载失败降级
- `app/services/conversation_memory_service.py:93-131,133-144` — 滚动摘要 + 游标
- `app/config.py:81,133-136` — 总预算 90s、记忆预算 1800/700/2400/3 轮
- `app/agent/rag_v2/service.py:47-78` — `asyncio.timeout` + `astream` + `last_state`
- `app/api/chat_v2.py:81,109,126,151,156` — 非流式全链路
- `app/api/chat_v2.py:177` — 流式：生成器体内启动 collector

---

### 面经 12 · 2026-07-14

#### Q1 · 项目是网上找的还是什么？

**回答**

这题在问真实性。回答不该辩解，该给不可能抄来的证据。

这个仓库能给的是**修复痕迹**。教程项目里不会有这些东西：

`app/core/errors.py:358-387` 的 `wrap_llm_exception` 里有一段注释记着一个真实 bug——它原先判断的是 `isinstance(exc, LLMError)`，于是 `CircuitOpenError`（它是 `AppError` 但不是 `LLMError`）被重新包装，`degrade_reason` 从 `circuit_open` 翻成了 `llm_error`。后果是值班的人跑去查模型配额，而真实原因是我们自己的熔断器打开了。修法是改判 `degrade_reason is not None`（`app/core/errors.py:380`）。

`app/agent/rag_v2/nodes.py:127-136` 的注释记着另一个：计时器原先在 `breaker.guard()` 外侧，熔断打开时 `guard()` 第一行就抛，`ainvoke` 根本没执行，但计时器照样记下一个 ~0.0001 秒的样本。于是熔断期间几百个「没打出去的调用」全以 0 秒计入直方图，**P99 反而变好看**——下游挂了，耗时指标显示一切正常。

`app/core/job_failure.py:146-183` 记着第三个：worker 原先在 `except` 里直接 `commit()`，连接已经死了的时候这一步会抛二次异常，把 `job.exc_info` 里真正的原因盖掉，同时 job 卡在 RUNNING、document 卡在 INDEXING 永远不动。修法是用独立的 `SessionLocal()` 写回，且该函数永不抛。

`git log` 里也有：`IVF_FLAT+L2 → HNSW+COSINE` 是一次索引类型迁移。

这些都不是抄得到的，因为它们是踩过之后才知道的。

**项目证据**

- `app/core/errors.py:358-387,380` — `wrap_llm_exception` 的 `circuit_open` 被覆盖 bug
- `app/agent/rag_v2/nodes.py:127-136` — 计时器在熔断器外侧导致 P99 变好看
- `app/core/job_failure.py:146-183` — `except` 内 commit 掩盖真实原因
- `app/core/llm_factory.py:1-62` — langchain-openai 覆盖 SDK 超时
- `git log` — `IVF_FLAT+L2 → HNSW+COSINE`

#### Q2 · 项目中挑战最大的是什么？

**回答**

最难的不是让链路跑通，是让它在坏掉的时候**说实话**。

具体到一件事：判断「这次请求算不算失败」这件事本身没有单一答案，需要一套贯穿全栈的语义。展开是四层。

第一层，**异常要分类**。`app/core/errors.py:81-124` 的 `AppError` 带三个类属性：`code`、`retryable`、`http_status`。子类各自覆写——`RetrievalError`（502，retryable，`app/core/errors.py:127`）、`LLMTimeoutError`（504，`app/core/errors.py:148`）、`ParseError`（502，**不可重试**，`app/core/errors.py:162`）、`CircuitOpenError`（503，`app/core/errors.py:183`）。`ParseError` 不可重试这条是刻意的：模型输出格式错了，重试同一个 prompt 大概率还是错。

第二层，**原始异常要能转成应用异常**。`app/core/errors.py:328-355` 的 `to_app_error` 做这件事，但它**刻意不设 `degrade_reason`**。原因写在 docstring 里：一个未知异常走到 HTTP 边界，那不是降级，是彻底失败。给它打上 `llm_error` 会让 `degrade_total{reason="llm_error"}` 这个指标混进 KeyError、AttributeError 这类自己的代码 bug，指标就废了。

第三层，**降级原因按修复方向分类**。`app/core/errors.py:25-62` 的 `DegradeReason` 有十个值，划分标准只有一条：**修复方向是否相同**。所以 `RETRIEVAL_FAILED` 和 `RETRIEVAL_EMPTY` 必须分开——技术上都是「没文档」，但一个去修服务，一个去补文档。

第四层，**状态码不能撒谎**。改造前端点是 `return {"code": 500, ...}`，HTTP 状态是 200。后果是 nginx 的 `$status`、LB 健康检查、APM、Prometheus 的 `http_requests_total{status="5xx"}` 全部读到 0% 错误率（这几个消费者列在 `app/core/exception_handlers.py` 的模块文档里）。改法是端点 `raise`，由 `app/core/exception_handlers.py:180-191` 注册的全局处理器决定状态码。

四层都做完，才有「这次答案是在证据不完整的情况下生成的」这个事实不丢失的保证。

**项目证据**

- `app/core/errors.py:81-124` — `AppError`，`code`/`retryable`/`http_status`
- `app/core/errors.py:127,139,148,162,175,183,196,205` — 八个子类及其状态码
- `app/core/errors.py:328-355` — `to_app_error` 及其不设 `degrade_reason` 的理由
- `app/core/errors.py:25-62` — `DegradeReason`，按修复方向划分
- `app/core/exception_handlers.py:1-64` — 模块文档，只读状态码的消费者清单
- `app/core/exception_handlers.py:180-191` — `register_exception_handlers`

#### Q3 · 出现幻觉怎么处理？

**回答**

先把"幻觉"这个词拆开，因为它在工程上其实是**四种不同的病**，混在一起谈就没法治。

---

**一、四类幻觉，成因和治法都不同**

**第一类：参数知识幻觉（Parametric Hallucination）。** 模型没有任何证据，纯靠预训练时记住的东西编。典型表现是编造 API 名、编造函数签名、编造一个不存在的配置项。这类幻觉在 RAG 里最好治——把证据喂给它，并且明确要求"只用给定证据回答"。

**第二类：证据外推（Unfaithful Generation）。** 有证据，但答案里有一部分证据支撑不了。比如证据说"CPU 使用率超过 80% 需要关注"，模型答成"CPU 超过 80% 会导致服务崩溃"——后半句是它自己加的。这类最难治，因为答案看起来有依据，而且大部分内容确实有依据。

**第三类：证据张冠李戴（Attribution Error）。** 检索回来 5 篇文档，模型把 A 文档的结论安到 B 文档的场景上。这在多文档 RAG 里非常常见，尤其是文档之间结构相似的时候（比如多份不同服务的 SOP，格式一模一样）。

**第四类：时效性幻觉（Staleness Hallucination）。** 这类在业务系统里最危险，也最容易被忽略。用户十分钟前问过"order-api 的 CPU 怎么样"，模型答了"92%"。现在用户问"现在呢"，模型从对话历史里翻出那个 92% 报出来——它没编数字，数字是真的，但那是十分钟前的值。答案在字面上有据，在语义上是假的。

四类的治法完全不同：

| 类型 | 根因 | 治法 |
|---|---|---|
| 参数知识幻觉 | 没证据 | 给证据 + 强约束"只用证据" |
| 证据外推 | 生成超出证据边界 | 事后校验 groundedness |
| 张冠李戴 | 多证据混淆 | 证据编号 + 要求标引用来源 |
| 时效性幻觉 | 把历史当现在 | 显式区分"记忆"与"实时数据" |

---

**二、防幻觉的三道闸门（业界通用分层）**

**闸门一：检索侧——让证据够用。** 这一层的逻辑是"幻觉的一半是检索失败导致的"。如果检索本来就没召回到正确文档，模型再老实也答不对。常见手段：查询改写（把口语问题变成多个检索友好的子查询）、混合检索（向量召回语义相近的，BM25 召回关键词精确匹配的）、重排（把最相关的排到前面，因为模型对上下文开头结尾更敏感）。

**闸门二：生成侧——约束 + 引用。** 系统提示词明确"仅依据给定资料回答，资料不足时说明不足"；给每段证据编号，要求答案里标 `[1]`、`[2]`；把证据的完整性状态告诉模型（"以下证据可能不完整"）。最后这一条很多人忽略，但它对第二类幻觉有直接效果——模型知道自己手上的东西是缺的，就不会硬撑着给一个完整答案。

**闸门三：事后校验——量化并拦截。** 这是唯一能真正"发现"幻觉的一层。核心是两个正交的分数：

- **Groundedness / Faithfulness（有据性）**：答案里的每个断言，能不能在证据里找到支撑？这个分数低 = 模型在编。
- **Coverage / Relevance（覆盖度）**：证据够不够回答这个问题？这个分数低 = 检索没做好，不是模型的错。

两个分数必须分开，因为它们指向不同的修复方向。**Groundedness 低是模型问题，Coverage 低是检索问题。** 混成一个"质量分"，看到分数低你不知道该改 prompt 还是补文档。

---

**三、我项目里的实现**

我的实现是三道闸门都有，重点在第三道。

`validate_answer_node`（`app/agent/rag_v2/nodes.py:569-718`）是一个独立的校验节点，跑在生成之后。校验提示词（`:79-104`）要求模型输出严格 JSON，包含 `coverage_score`、`groundedness_score`、两个 pass 布尔、`unsupported_claims`（哪些断言没有依据）、`missing_aspects`（问题的哪些方面没被证据覆盖）。判分调用用 `temperature=0.0, streaming=False`（`:679`）——判分要可复现，不需要创造性。

阈值是 `MIN_GROUNDEDNESS_SCORE = 0.75` / `MIN_COVERAGE_SCORE = 0.60`（`:36-37`）。两个数字不一样是有意的：有据性的要求必须比覆盖度高，因为"编造"比"回答不全"严重得多。低于阈值就 `blocked`，换成 `_build_fallback_answer`（`:882`）的保守回答，并标降级原因 `EVIDENCE_INSUFFICIENT`。

第一道闸门在 `rewrite_node`：一个问题拆成 3 个互补子查询（`NUM_SUB_QUERIES = 3`，`:32`），并行检索后去重取 Top 6。第二道闸门在 `generate_node`（`:500-521`）：证据不完整时把 `partial_note` 拼进 prompt，明确告诉模型"你手上的证据是缺的"。

第四类幻觉我单独治了一道，因为 AIOps 场景里它最致命。`MEMORY_POLICY`（`:106`）作为独立文本块拼进用户消息，明文写着：会话记忆只用于理解用户已明确说明的对象和任务延续，**它不是知识库证据，更不是实时监控、日志或工具查询结果，不得将历史指标写成当前实时值**。这条约束就是针对"十分钟前 92%"那个场景写的。

---

**四、最花时间的其实是"校验器自己挂了怎么办"**

这部分是我这套实现里最有工程含量的地方，也是面试里最容易被追问的地方。校验器有三种失败方式，我给了**两种相反的策略**：

**情况一：校验器被开关关掉**（`:648`）。fail-open 放行，降级原因标 `FEATURE_DISABLED`。注意这里刻意**不用** `LLM_ERROR`——是我自己关的，不是模型坏了，混在一起会让值班的人去查模型配额。

**情况二：校验器调用失败**（超时、熔断、网关挂了，`:695-718`）。也 fail-open，但用 `_unvalidated_validation`（`:848-880`），它设的是 `blocked=False` 且 `validated=False`。这两个字段的组合是整段设计的核心：**放行了，但明确记录这次没校验过。**

**情况三：校验器返回了，但输出解析不出来**（`:810-814`）。fail-**closed**，`_default_validation` 设 `blocked=True`。

方向相反，理由是这样的：

> 校验器**服务挂了**就拦答案，等于让一个辅助组件的故障升级成整个服务不可用——用户的问题本来能答，却因为质检排队而失败，这个因果关系是荒谬的。
>
> 但校验器**返回了垃圾**却放行，等于假装校验通过了——这是在撒谎，而且下游完全看不出来。

前者是可用性权衡，后者是诚实性底线。所以前者放行、后者拦。

`validated` 这个字段存在的唯一理由就是让下游能区分"校验通过了"和"根本没校验"。只有成功走完校验的路径才设 `validated=True`。有了它，读侧可以统计"有多少答案是没过质检就发出去的"——这个数字如果开始上涨，说明质检器在悄悄失效，而答案质量看起来一切正常。

---

**五、我没做的部分（诚实说）**

- **离线 Faithfulness 量化**：`tests/eval/metrics.py:174-185` 里 `faithfulness_placeholder` 和 `answer_relevance_placeholder` 都是 `NotImplementedError` 占位，计划接 ragas。所以我只有线上单次判分，没有一批用例上的分布统计。
- **异构 judge**：`config.py:46` 和 `:155` 都是 `qwen-max`，生成和判分同模型，存在自评偏见。偏见的**方向**我知道（倾向于认为自己有据），**幅度**我没测过。
- **引用标注到句级**：现在只能标"用了哪几篇文档"，不能标"这句话来自第 3 篇第 2 段"。

**项目证据**

- `app/agent/rag_v2/nodes.py:569-718` — `validate_answer_node` 全体
- `app/agent/rag_v2/nodes.py:79-104` — 校验提示词与输出字段
- `app/agent/rag_v2/nodes.py:36-37` — 两个阈值，0.75 / 0.60
- `app/agent/rag_v2/nodes.py:679` — 判分 `temperature=0.0`
- `app/agent/rag_v2/nodes.py:648` — 开关关闭，`FEATURE_DISABLED` 而非 `LLM_ERROR`
- `app/agent/rag_v2/nodes.py:695-718`、`:848-880` — 调用失败 fail-open + `_unvalidated_validation`
- `app/agent/rag_v2/nodes.py:810-814`、`:830` — 解析失败 fail-closed + `_default_validation`
- `app/agent/rag_v2/nodes.py:882` — `_build_fallback_answer`
- `app/agent/rag_v2/nodes.py:106` — `MEMORY_POLICY`，治时效性幻觉
- `app/agent/rag_v2/nodes.py:32`、`:500-521` — 子查询数量、`partial_note`
- `app/config.py:122` — `enable_answer_validation`
- `tests/eval/metrics.py:174-185` — 离线 faithfulness 未实现
- 句级引用、异构 judge、Faithfulness 离线量化：项目中没有

#### Q4 · 提示词具体是怎么做？

**回答**

这个仓库的提示词分散在四处，每一处解决的问题不同。

`app/agent/rag_v2/nodes.py:36` 的 `REWRITE_SYSTEM_PROMPT`，带 `{n}` 占位符，由 `NUM_SUB_QUERIES = 3`（`app/agent/rag_v2/nodes.py:28`）填充。要求生成互补而非近义的子查询。

`app/agent/rag_v2/nodes.py:50` 的 `GENERATE_SYSTEM_PROMPT`，约束只用给定证据回答。

`app/agent/rag_v2/nodes.py:60` 的 `VALIDATION_SYSTEM_PROMPT`，要求输出两个 0-1 分数。配套的 `_parse_validation_result`（`app/agent/rag_v2/nodes.py:494`）和 `_coerce_score`（`app/agent/rag_v2/nodes.py:526`）负责容错解析——不能假定模型一定输出合法 JSON。

`app/services/first_response_service.py:101-130` 的 `_SYSTEM_PROMPT` 是里面最有讲究的一个，因为它面对的是运维场景。两条硬约束：**禁止输出任何状态变更命令**（不能让模型建议 `kill -9` 或 `systemctl restart`），以及**把不支持的判断强制写进 `pending_confirmations`** 而不是当成结论。

还有一个不在提示词里但同等重要的：`app/agent/rag_v2/nodes.py:87` 的 `MEMORY_POLICY` 作为独立文本块拼进用户消息（`app/agent/rag_v2/nodes.py:110`），明确会话记忆不是知识库证据、更不是实时监控数据，不得把历史指标写成当前值。

提示词工程里真正花时间的部分不是措辞，是**证据的格式和预算**。`app/agent/rag_v2/nodes.py:613` 的 `_format_context` 决定证据怎么呈现给模型，`app/agent/rag_v2/nodes.py:631-637` 的 `_estimate_tokens` 和 `_truncate_to_token_budget` 保证不超 `rag_document_context_token_budget`（2800，`app/config.py:130`）。token 估算对中文做了特殊处理（`app/services/conversation_memory_service.py:48-54`，CJK 按 1 token/字，其余除 4）——按 `len/4` 一把算会严重低估中文。

**项目证据**

- `app/agent/rag_v2/nodes.py:28,36` — `NUM_SUB_QUERIES` + `REWRITE_SYSTEM_PROMPT`
- `app/agent/rag_v2/nodes.py:50` — `GENERATE_SYSTEM_PROMPT`
- `app/agent/rag_v2/nodes.py:60,494,526` — 校验提示词 + 容错解析
- `app/agent/rag_v2/nodes.py:87,110` — `MEMORY_POLICY` 及其注入位置
- `app/services/first_response_service.py:101-130` — 运维场景硬约束
- `app/agent/rag_v2/nodes.py:613,631,637` — 证据格式化与 token 预算
- `app/services/conversation_memory_service.py:48-54` — CJK 感知的 token 估算
- `app/config.py:130` — `rag_document_context_token_budget=2800`

#### Q5 · 还有其他提示词吗？

**回答**

有，而且这一批的性质和上一题不同——上一题那些是「让模型做对事」，这一批是「让模型在信息不全时不要编」。

`app/services/first_response_service.py:350-416` 的 `_build_user_prompt` 里，**零证据的情况有三种不同措辞**。这不是啰嗦，是三种情况对模型的暗示完全不同：SOP 检索服务失败、检索成功但知识库没有对应文档、功能开关关掉了根本没检索。如果统一写成「没有找到相关文档」，模型会倾向于用自己的参数知识补，而值班人员看到报告也无法判断该去修服务还是补文档。

`app/services/conversation_memory_service.py:28-35` 的 `_SUMMARY_SYSTEM_PROMPT` 用固定小标题——【明确事实】【任务进展】【约束与偏好】【待确认项】。固定结构的作用是让摘要可被下一轮稳定解析，而不是每次生成不同格式的散文。

`app/services/rag_agent_service.py:146-174` 的 `_build_system_prompt` 是旧 Agent 路径的，注释里记了一条：不需要在提示词里列举工具，LangChain 会自动把工具 schema 传给模型。手写一份工具清单进提示词，一改工具就会不同步。

`app/agent/rag_v2/nodes.py:264-331` 的 `partial_note` 是动态拼的，只在证据不完整时出现，告诉模型「你手上的证据是缺的」。

**项目证据**

- `app/services/first_response_service.py:350-416` — 三种零证据措辞
- `app/services/conversation_memory_service.py:28-35` — 固定小标题的摘要提示词
- `app/services/rag_agent_service.py:146-174` — 不列举工具的理由
- `app/agent/rag_v2/nodes.py:264-331` — `partial_note` 动态注入

#### Q6 · Agent的短期长期记忆是怎么实现的？

**回答**

这个仓库有两套记忆，一套是玩具一套是能用的，差别正好说明这题。

**旧 Agent 路径**：`app/services/rag_agent_service.py:107` 的 `MemorySaver()`，LangGraph 的进程内 checkpointer，按 `thread_id`（即 session_id）隔离。配套 `app/services/rag_agent_service.py:37-74` 的 `trim_messages_middleware`：超过 7 条时保住首条 System 加最近 6 条，实现方式是 `RemoveMessage(id=REMOVE_ALL_MESSAGES)` 全删再重放。问题很明显——进程重启全丢，而且裁掉的历史就是真的没了。

**chat_v2 路径**：`app/services/conversation_memory_service.py`，292 行，滚动摘要加有界最近窗口。

核心在 `build_context`（`app/services/conversation_memory_service.py:93-131`），流程是：读活跃消息 → 取上次快照 → 算出快照游标之后的 `pending_messages` → 保留最近若干轮（`conversation_memory_recent_turns = 3`，`app/config.py:136`）→ 剩下的如果超过 `compact_threshold_tokens`（2400）就压成摘要 → 写回新快照并记 `summarized_through_message_id`。

游标（`summarized_through_message_id`）是这套设计的关键。有它才叫**增量**摘要——每次只摘要「上次摘要之后新增的那些」，而不是每轮把全部历史重新摘一遍。后者的成本随对话长度线性增长，而且每次摘要结果都会漂移。

三个细节值得讲：

`app/services/conversation_memory_service.py:133-144` 的 `_messages_after_snapshot`，游标指向的消息被软删除时返回 `[]`，而不是把整段历史当成未摘要重新处理。

`app/services/conversation_memory_service.py:163-186` 的 `_summarize`，失败时降级为纯最近窗口，**绝不阻塞聊天**。摘要是优化不是必需品。

`app/services/conversation_memory_service.py:207-247` 的 `_render_context`，先扣掉 header 本身的 token 成本再算预算，最后硬截总长度。以及 `app/config.py:133-135` 三个预算是分开的：上下文总预算 1800、摘要预算 700、压缩阈值 2400——摘要自己也有上限，不然它会吃掉留给最近消息的空间。

**原始消息始终在数据库里**（`migrations/versions/20260624_0002_create_conversation_context_tables.py`），摘要只是投影。这意味着摘要丢了信息可以回读原文重建，而 `MemorySaver` 那条路径裁掉就是永久丢失。

要说「长期记忆」的话得诚实：这个仓库的记忆都是**会话内**的，没有跨会话的用户画像、偏好沉淀或经验库。`app/config.py:133-136` 那几个预算全是单会话内的。跨会话长期记忆**项目中没有，不能硬套**。

**项目证据**

- `app/services/rag_agent_service.py:107,37-74` — `MemorySaver` + 消息裁剪
- `app/services/conversation_memory_service.py:93-131` — `build_context` 主流程
- `app/services/conversation_memory_service.py:38-45` — `ConversationMemoryContext` dataclass
- `app/services/conversation_memory_service.py:133-144` — 游标消息被软删的处理
- `app/services/conversation_memory_service.py:146-158,163-186` — 轮次计数、摘要失败降级
- `app/services/conversation_memory_service.py:207-247,249-272` — 渲染预算、尾部截取
- `app/config.py:133-136` — 1800 / 700 / 2400 / 3 轮
- `migrations/versions/20260823_0007_create_conversation_memory_snapshots.py:1-35` — 快照表
- 跨会话长期记忆：项目中没有，不能硬套

#### Q7 · 如果让你设计一个Agent要考虑哪些模块？

**回答**

不按教科书的「感知-规划-记忆-行动」四件套答，按这个仓库实际存在的分层答，因为每一层都是被具体故障逼出来的。

**编排层**。`app/agent/rag_v2/graph.py:22-51`，五个节点显式连边。状态定义在 `app/agent/rag_v2/state.py:10-34`，用 `TypedDict(total=False)`，并行分支的合并靠 `Annotated[List[...], operator.add]` reducer——`documents`（`app/agent/rag_v2/state.py:17`）、`retrieve_failures`（`:27`）、`degrade_reasons`（`:34`）三个字段都是这样，fan-in 时自动拼接而不是互相覆盖。

**预算层**。两层，不能只有一层。单次调用超时在 `app/core/llm_factory.py:84`（60 秒）；整个请求的总预算在 `app/agent/rag_v2/service.py:61`（90 秒）。只有单次超时的话，三次重试加上多个节点，总耗时可以远超用户忍耐；只有总预算的话，一个卡死的调用会把预算全吃掉。

**降级层**。`app/core/errors.py:25-62` 的十个 `DegradeReason`，`app/core/errors.py:65-78` 的三态 `RetrievalStatus`。每个节点失败时返回降级原因而不是抛异常，由 `app/agent/rag_v2/nodes.py:214-261` 的 `dedup_node` 汇总判定——它是第一个有全局视野的 fan-in 节点，所以由它区分 `PARTIAL_RETRIEVAL`（有失败也有文档）、`RETRIEVAL_FAILED`（有失败无文档）、`RETRIEVAL_EMPTY`（无失败无文档）。

**熔断层**。`app/core/circuit_breaker.py`，275 行。按下游而不是按调用点划分（`app/core/breakers.py` 只有两个：`retrieval_breaker` 和 `llm_breaker`），依据写在模块文档里——「会同时坏、也会同时好的调用，共用一个熔断器」。

**验证层**。`app/agent/rag_v2/nodes.py:334-469`，且 fail-open / fail-closed 分得清楚（见 Q3）。

**可观测层**。三样东西，各司其职：`app/core/metrics.py` 的 Prometheus 指标（Histogram 而不是 Gauge，label 基数受控）、`app/core/span_context.py` 的 span（append-only，覆盖并行分支）、`app/core/request_context.py` 的 request_id（全链路，含 worker）。

**异常边界层**。`app/core/exception_handlers.py:180-191`，端点 `raise`，状态码由全局处理器决定。

**异步任务层**。`app/core/job_failure.py`，重试策略（`app/core/job_failure.py:68-74`，3 次，间隔 10/30/60 秒）加不可重试判定（`app/core/job_failure.py:77-128`，把 `retries_left` 归零）加失败写回（`app/core/job_failure.py:146-183`，独立 session，永不抛）。

这套分层里没有的东西也得说清楚：没有规划器（`app/agent/aiops/planner.py` 等文件已从仓库删除）、没有多 Agent、没有跨会话记忆、没有人工审批环节。**这几块项目中没有，不能硬套。**

**项目证据**

- `app/agent/rag_v2/graph.py:22-51`、`app/agent/rag_v2/state.py:10-34,17,27,34` — 编排与 reducer
- `app/core/llm_factory.py:84`、`app/agent/rag_v2/service.py:61` — 两层预算
- `app/core/errors.py:25-62,65-78`、`app/agent/rag_v2/nodes.py:214-261` — 降级语义与汇总判定
- `app/core/circuit_breaker.py:90-110,135-153,204-242`、`app/core/breakers.py:1-46` — 熔断与划分依据
- `app/agent/rag_v2/nodes.py:334-469` — 验证层
- `app/core/metrics.py:83-123,135-152`、`app/core/span_context.py:85-108`、`app/core/request_context.py:33` — 可观测三件套
- `app/core/exception_handlers.py:180-191` — 异常边界
- `app/core/job_failure.py:68-74,77-128,146-183` — 异步任务失败治理
- 规划器 / 多 Agent / 跨会话记忆 / 人工审批：项目中没有，不能硬套

### 面经 13 · 2026-07-10

#### Q1 · 请做一下自我介绍

**回答**

自述题，没有对应代码实现。给一版每句话都能被追问到文件的版本，这样面试官往下挖的时候不会断。

我做的是一个 RAG + Agent 的后端服务，Python / FastAPI / LangGraph / Milvus / PostgreSQL / Redis-RQ。核心链路是五个节点的显式图：查询改写 → 子查询并行检索 → 去重合并 → 生成 → 答案验证（`app/agent/rag_v2/graph.py:22-51`）。我在这个项目里花时间最多的不是把链路跑通，而是把失败路径讲清楚：

- 超时分两层，单次 LLM 调用 60 秒（`app/core/llm_factory.py:84`），整个请求总预算 90 秒（`app/agent/rag_v2/service.py:61`）；总预算耗尽时不抛异常，用 `astream` 攒下的最后一个状态快照交付部分结果（`app/agent/rag_v2/service.py:61-73`）。
- 降级原因是一个十项枚举（`app/core/errors.py:25-62`），划分依据只有一条：**修复方向不同才拆成两个**。检索状态是三态而不是布尔（`app/core/errors.py:65-78`），专门治「检索失败了却返回空列表」这种会撒谎的降级。
- 熔断器按下游划分而不是按调用点，全项目只有两个（`app/core/breakers.py`），用 `threading.Lock` 而不是 `asyncio.Lock`，因为同步检索是在 `to_thread` 里跑的。
- HTTP 状态码不撒谎：端点直接 `raise`，状态码由全局异常处理器决定（`app/api/chat_v2.py:151`、`app/core/exception_handlers.py:180-191`）。

需要说明的是这个项目**没有**规划器、没有多 Agent 协作、没有跨会话长期记忆、没有人工审批环节——这几块项目中没有，不能硬套。

**项目证据**

- `app/agent/rag_v2/graph.py:22-51` — 五节点显式图
- `app/core/llm_factory.py:84`、`app/agent/rag_v2/service.py:61-73` — 两层预算与部分结果
- `app/core/errors.py:25-62,65-78` — 降级原因与三态检索状态
- `app/core/breakers.py:1-46`、`app/core/circuit_breaker.py:90-110` — 熔断划分与锁选型
- `app/api/chat_v2.py:151`、`app/core/exception_handlers.py:180-191` — 异常边界

#### Q2 · 为什么选择 LangGraph，而不是直接使用 LangChain AgentExecutor？

**回答**

先把这三个东西是什么讲清楚，因为它们是同一条演进线上的三代产品，答这题如果说不出"上一代差在哪"，就只是在背特性表。

---

**第一代：`AgentExecutor`（2023 年，现已废弃）**

它的本体是一个 `while True` 循环，伪代码大概这样：

```python
intermediate_steps = []
while True:
    # 1. 把「原始问题 + 到目前为止的所有观察」拼成 prompt
    output = llm.invoke(format_prompt(question, intermediate_steps))
    # 2. 从模型输出里解析出「调哪个工具、传什么参数」
    action = output_parser.parse(output)
    # 3. 如果模型说「我答完了」，退出
    if isinstance(action, AgentFinish):
        return action.return_values
    # 4. 否则执行工具，把结果 append 到观察列表，回到第 1 步
    observation = tools[action.tool].run(action.tool_input)
    intermediate_steps.append((action, observation))
```

这就是 ReAct 循环的直接实现。它在 2023 年是巨大的进步——在它之前，用 LLM 调工具要自己写 prompt 模板、自己解析输出、自己管循环。

**它的四个结构性缺陷**（注意：都不是"功能少"，而是"架构上做不到"）：

1. **状态只有一个 `intermediate_steps` 列表。** 它是一条线性的 `[(action, observation), ...]`。你没法定义"这个字段来自三个并行分支，需要合并"这种语义——列表就是列表，append 就是 append。

2. **控制流被藏在框架内部。** `while True` 在库的代码里，你插不进任何东西。想在"第 3 步之后、第 4 步之前"加一道检查，唯一的办法是继承 `AgentExecutor` 重写方法——那已经是在改框架而不是用框架了。

3. **只能串行。** 循环体一次只走一个 action。即使模型一次性输出了四个可以并行的工具调用（现代模型的 parallel tool calls），执行仍然是一个接一个。

4. **失败就是全失败。** 循环中间任何一步抛异常，整个 `invoke()` 抛出去。前面已经做完的三步工作全部丢弃——因为那些结果只存在于函数栈上的局部变量里，异常一抛栈就没了。

---

**第二代：`create_agent`（2024 年至今，LangChain 的现行推荐）**

它是用 LangGraph 重写的 `AgentExecutor`。对外接口几乎一样（给模型、给工具、给 checkpointer），但内部是一张预置的 LangGraph 图：`agent 节点 ⇄ tools 节点` 两个节点循环。

它修掉了第一代的两个问题：可以并行执行同一轮里的多个工具调用；有 checkpointer 之后状态是持久的，能断点续跑。但**没有修掉前两个**——状态形状仍然是固定的（本质还是 `messages` 一条列表），控制流仍然是"模型决定下一步"这一种模式。

这不是它做得不好，是它的**定位**：它要覆盖"任意工具、任意顺序"这个通用场景，就不能让你自定义节点和边，否则它就不是一个开箱即用的 Agent 了。

---

**第三代：直接用 `StateGraph` 手写图（LangGraph 的原生用法）**

这里你自己定义三样东西：状态的形状（哪些字段、每个字段怎么合并）、有哪些节点、节点之间怎么连边。`create_agent` 是这套原语的一个特例——它就是"两个节点 + 一条条件边"拼出来的。

---

**所以这题的真实判据不是"哪个框架更强"，而是一句话：下一步该由谁决定。**

- 下一步由**模型**决定（用户可能问时间、可能问业务、可能要查日志，顺序完全取决于对话内容）→ 用 `create_agent`。这时你**画不出**一张图，因为图要求你预先知道所有路径。
- 下一步由**代码**决定（业务流程是设计出来的：先改写问题，再检索，再去重，再生成，再质检）→ 用 `StateGraph`。这时用 `create_agent` 是浪费——你要为一件确定的事付一次 LLM 调用的钱和时间，还要承担模型选错工具的风险。

---

**我这个项目两条路都在跑，正好是这个判据的实物对照。**

`create_agent` 那条在 `app/services/rag_agent_service.py:133-137`，一行 `create_agent(self.model, tools=all_tools, checkpointer=self.checkpointer)`。它必须走这条路，因为它的**工具集是运行时才知道的**——MCP 工具在 `:123` 动态拉取，拉失败就降级成"只有本地两个工具"（`:115-139`）。工具集不确定，调用顺序更不确定，画不出图。

`StateGraph` 那条在 `app/agent/rag_v2/graph.py`，五个节点的边全是我自己连的。选它的四个理由，每一个都对应上面那些"结构上做不到"：

**一、要并行 fan-out，而且是动态数量的。** `_fanout_to_retrieve`（`graph.py:17-19`）返回一个 `list[Send]`：

```python
def _fanout_to_retrieve(state: RAGState) -> list[Send]:
    sub_queries = state.get("sub_queries", []) or []
    return [Send("retrieve_each", {"query": q}) for q in sub_queries]
```

`Send` 是 LangGraph 的原语，意思是"把这个 payload 送给那个节点，作为一个独立实例执行"。返回三个 `Send` 就是三个 `retrieve_each` 并行跑。数量取决于 rewrite 节点拆出了几个子查询，**运行时才知道**——这是普通的静态边表达不了的（静态边要求你写图的时候就知道有几条分支）。

**二、状态字段需要自定义合并语义。** 三个并行分支各自返回 `{"documents": [doc1, doc2]}`，LangGraph 怎么知道该合并还是该覆盖？靠状态定义里的 reducer：

```python
documents: Annotated[List[Document], operator.add]
```

`Annotated[类型, reducer]` 是 LangGraph 读取的约定——`operator.add` 就是 Python 的 `+`，对列表是拼接。没有这个标注，三个分支写同一个字段就是**互相覆盖**，最后只剩一个分支的结果（而且是哪个分支取决于调度顺序，是个随机 bug）。

`state.py` 里有三个字段带 reducer：`documents`（`:17`）、`retrieve_failures`（`:27`）、`degrade_reasons`（`:48`）。第二个尤其关键，下一条讲。

**三、一个分支失败不能拖垮整张图。** LangGraph 的执行单位叫 super-step——同一批并行的节点算一步，**任一个抛异常，整步失败，整张图崩**。三个分支里两个成功一个失败，用户拿到 500，成功那两份证据一起丢掉。

所以 `retrieve_each_node`（`nodes.py:189-205`）把异常全兜住，返回一条失败明细进 `retrieve_failures` 而不是抛出去。语义从"一个分支炸 = 整个请求失败"变成"一个分支炸 = 证据少一份"。这个改造在 `create_agent` 里没有位置可以做——你碰不到它的节点实现。

**四、超时要能交付部分结果。** `service.py:62` 用的是 `astream(stream_mode="values")` 而不是 `ainvoke`：

```python
async with asyncio.timeout(config.chat_total_budget_seconds):
    async for snapshot in graph.astream(initial, stream_mode="values"):
        last_state = snapshot          # 每一步都留一份完整快照
```

`stream_mode="values"` 每完成一个节点就 yield 一次**完整状态快照**。所以 90 秒预算用尽时，`last_state` 里已经有了 sub_queries 和 documents——可以告诉用户"检索到了这些文档，但生成超时了"。`ainvoke` 超时就是彻底空手：所有中间结果都在协程栈上，取消即消失。

**五、每个节点要能统一埋点。** `graph.py:35-41` 每个节点注册时都过一层 `instrument_node`。这是唯一入口，包在这里漏不掉，而节点自己完全不知道 span 存在。尤其是被 `Send` 展开的 N 个并行实例，每个各记一条 span——不然日志里三条分支交织在一起，对不出各自花了多久。

---

**一句话收口**：`AgentExecutor` 已废弃，讨论它只有历史意义；真正的选择是 `create_agent`（模型驱动，工具集不确定）对 `StateGraph`（代码驱动，流程确定且需要并行/降级/部分结果）。两者不是替代关系，我这个项目里就是并存的。

**项目证据**

- `app/agent/rag_v2/graph.py:17-19` — `_fanout_to_retrieve`，动态数量的 `Send`
- `app/agent/rag_v2/graph.py:22-51,35-41,44` — 显式图、`instrument_node`、`add_conditional_edges`
- `app/agent/rag_v2/state.py:17,27,48` — 三个 `Annotated[..., operator.add]` reducer 字段
- `app/agent/rag_v2/nodes.py:189-205` — 分支异常兜底，避免 super-step 全崩
- `app/agent/rag_v2/service.py:62-73` — `astream(stream_mode="values")` + 部分结果
- `app/services/rag_agent_service.py:115-139,123,133-137` — `create_agent` 路径与运行时工具集
- `AgentExecutor` 这个类：项目中没有用（用的是它的后继 `create_agent`）

#### Q3 · 文档审校平台的 RAG 检索链路是怎么设计的？

**回答**

原帖的「文档审校平台」是那位面试者自己的项目，不是本仓库。按同类问题回答本仓库真实的检索链路。

**入库侧**：上传后先算 `sha256`（`app/api/file.py:81`），用 `content_hash` 比对做去重（`:95-99`）——同一份文件重复提交不产生重复副作用。然后切分（`app/services/document_splitter_service.py`，`chunk_max_size=800`、`chunk_overlap=100`，见 `app/config.py`），向量化走 DashScope `text-embedding-v4`，1024 维，注意服务端有 batch 上限 10，所以批处理必须切段。索引是 Milvus **HNSW + COSINE**（git log 有一条 `IVF_FLAT+L2 → HNSW+COSINE` 的提交）。索引任务丢进 RQ 异步跑（`app/workers/index_worker.py`），分片级别再算一次 `chunk_hash`（`:62-68`）。

**检索侧**：五个节点。

1. `rewrite_node` 把原始问题拆成多个子查询。
2. `app/agent/rag_v2/graph.py:17-19` 按子查询数量 `Send` 出 N 个并行 `retrieve_each`。
3. 每一路走 `app/services/vector_search_service.py:52-107` 的 `retrieve_documents`：先取宽——`candidate_k = max(top_k * 3, top_k, 10)`（`:84`）；向量检索器和 BM25 检索器各出一份候选，用 `EnsembleRetriever` 融合（`:96-99`），权重向量 0.7 / BM25 0.3（类常量在 `:42-43`）；最后截断到 `top_k`（`:101`）。
4. `dedup_node`（`app/agent/rag_v2/nodes.py:214-261`）fan-in 去重，并且判定三态检索状态——它是第一个有全局视野的节点，所以由它区分「部分失败」「全失败」「知识库为空」。
5. `generate_node` 按 token 预算截断文档上下文（`app/agent/rag_v2/nodes.py:617-637`，预算 2800，`app/config.py:130`），然后 `validate_answer_node`（`app/agent/rag_v2/nodes.py:334-469`）做证据校验和幻觉拦截。

**降级点**：BM25 语料为空时降级为纯向量检索（`app/services/vector_search_service.py:92-94`）。BM25 检索器带缓存，按 Milvus 的 `num_entities` 变化判断是否重建（`:140-163`），语料是从 Milvus 用 `query_iterator` 分批全量读出来的（`:165-201`，batch 1000）。中英混排的分词是自己写的简单规则（`:203-224`）：ASCII 连续字母数字聚成一个 token，CJK 逐字切。

**这条链路没有的东西**：没有在线 rerank——仓库里只有 `.hf_cache/hub/models--BAAI--bge-reranker-v2-m3/` 的模型文件、三处提到未来接 rerank 的注释（`app/services/metadata_enricher.py:5`、`app/core/used_documents.py:34,150`）、以及评测脚本里的 `--retriever local-rerank` 选项（`tests/eval/run_eval.py`）。**在线路径上没有 reranker，不能硬套。** 也没有查询意图分类、没有召回后的 LLM 重排。

**项目证据**

- `app/api/file.py:81,95-99`、`app/workers/index_worker.py:62-68` — 入库去重与分片哈希
- `app/services/vector_search_service.py:42-43,52-107,84,92-99,101,140-163,165-201,203-224` — 混合检索全貌
- `app/agent/rag_v2/graph.py:17-19,43-48`、`app/agent/rag_v2/nodes.py:214-261,334-469,617-637` — 五节点链路
- `app/config.py:130` — 文档上下文 token 预算 2800
- 在线 reranker、查询意图分类、LLM 重排：项目中没有，不能硬套

#### Q4 · RRF 为什么不需要直接比较 BM25 分数和向量相似度？

**回答**

一句话：**RRF 只吃名次，不吃分数**。分数不可比是前提，名次天然可比是解法。

下面把这件事从头讲一遍，因为「为什么不可比」比「RRF 怎么算」更值得搞清楚。

---

**先说这两个分数分别是什么东西**

**BM25** 是关键词检索的经典打分函数，全称 Best Match 25，是 TF-IDF 的改进版。它的直觉是：一个词在这篇文档里出现得多（TF 词频高）、同时在整个语料里很罕见（IDF 逆文档频率高），那这个词就很能说明这篇文档相关。公式大致长这样（省略常数项）：

```
score(D, Q) = Σ  IDF(qᵢ) · (f(qᵢ,D) · (k₁+1)) / (f(qᵢ,D) + k₁·(1 - b + b·|D|/avgdl))
              qᵢ∈Q
```

关键是看它的性质，而不是记公式：

- **没有上界。** 分数是各个查询词的贡献**求和**，查询里词越多、命中越多，分数越大。一个 3 个词的查询可能打出 8 分，一个 15 个词的长查询能打出 40 分。
- **量纲随语料变化。** IDF 里有一项 `log(N / df)`，N 是语料总文档数。**同一篇文档、同一个查询，在语料从 1 万篇涨到 10 万篇之后，BM25 分数会变。** 这是理解本题的关键。
- **依赖文档长度归一化。** 公式里 `|D|/avgdl` 是当前文档长度除以平均长度，所以 avgdl 一变，所有分数都跟着变。

**向量相似度（COSINE）** 是另一套东西。文档和查询各被 embedding 模型编码成一个高维向量（本项目是 DashScope `text-embedding-v4`，1024 维），余弦相似度算的是两个向量夹角的余弦值：

```
cos(A, B) = (A · B) / (|A| · |B|)
```

它的性质正好相反：

- **有界。** 严格落在 `[-1, 1]`。文本 embedding 实践中通常落在 `[0, 1]` 偏上区间，语义相近的一对文本常常是 0.7~0.9。
- **量纲不随语料变化。** 只跟这一对向量有关，跟库里有多少文档完全无关。
- **分布很挤。** 这一点很多人忽略：好的匹配 0.85、平庸的匹配 0.78、无关的 0.65 —— 区分度都压在小数点后两位。

---

**所以「加权求和」为什么是错的**

假设你想按向量 0.7、BM25 0.3 的比例融合，直接写：

```
final = 0.7 × cosine + 0.3 × bm25          # 错
      = 0.7 × 0.83   + 0.3 × 12.7
      = 0.581        + 3.81
```

BM25 那一项贡献了 3.81，向量那项只有 0.58。**你写的权重是 7:3，实际生效的是大约 1:6.6，而且方向反了。** 你以为在做「以向量为主、关键词为辅」，实际做出来的是「几乎纯 BM25」。更糟的是这个偏差不稳定 —— 短查询时 BM25 打 5 分，长查询打 40 分，融合比例每次请求都不一样。

**那归一化行不行？** 这是最常见的下一步想法，但它有三种做法，三种都有坑：

| 归一化方式 | 做法 | 为什么不行 |
|---|---|---|
| Min-Max | `(x - min) / (max - min)` | min/max 从**当前这批候选**里取，所以同一篇文档在不同查询下归一化后的分数不同，无法跨查询比较；且一个离群高分会把其余全部压到接近 0 |
| Z-score | `(x - μ) / σ` | 需要知道分数分布的均值和标准差。BM25 的分布随语料规模、平均文档长度变化，**每次入库都会漂移**，昨天标定好的参数今天就不准 |
| 固定除数 | `bm25 / 20` 这种 | 纯拍脑袋。语料一扩，20 就不对了，而且没有任何原则能告诉你该除多少 |

归一化的根本困难在这：**它要求你知道分数的分布，而 BM25 的分布是语料的函数，语料是会变的。** 一个每次入库都要重新标定的融合参数，工程上等于不可用。

---

**RRF 是怎么绕开的**

RRF 全称 Reciprocal Rank Fusion（倒数排名融合），2009 年 Cormack 等人提出。它的做法极简 —— **把分数扔掉，只留名次**：

```
RRF_score(d) = Σ  weightᵢ / (k + rankᵢ(d))
               i∈检索器
```

- `rankᵢ(d)`：文档 d 在第 i 个检索器结果里排第几名（从 1 开始）
- `k`：一个平滑常数，论文里取 60，多数实现沿用这个默认值
- `weightᵢ`：给第 i 个检索器的权重

**为什么这就解决了问题？** 因为「第 1 名」这个概念在两个检索器里含义完全相同。BM25 的第 1 名 `rank=1`，向量的第 1 名也是 `rank=1`。名次是**序数**，不带量纲，不随语料规模漂移。于是你写的权重 0.7/0.3 才真的按 7:3 生效。

**k 是干什么的？** 它控制头部名次之间的差距有多大。看一下带 k 和不带 k 的对比：

| 名次 | `1/rank`（k=0） | `1/(60+rank)`（k=60） |
|---|---|---|
| 1 | 1.000 | 0.0164 |
| 2 | 0.500 | 0.0161 |
| 3 | 0.333 | 0.0159 |
| 10 | 0.100 | 0.0143 |
| 50 | 0.020 | 0.0091 |

`k=0` 时第 1 名的权重是第 2 名的 2 倍、第 3 名的 3 倍 —— 头部差距被拉得极大，等于「谁家的第一名说话最响」，另一个检索器就算把这篇文档排到第 2 也很难翻盘。`k=60` 把这个差距压平了：第 1 名和第 3 名只差 3%，于是**「两个检索器都觉得还不错」会胜过「一个检索器觉得极好、另一个没排进来」**。

这正是混合检索想要的行为 —— 我们要的是**共识**，不是单边冠军。k 越大越平权，越小越偏向头部。60 是个经验值，实践中很少需要动。

**RRF 还有一个常被忽略的好处：它不要求两个检索器返回等长的列表，也不要求它们对同一篇文档都有分数。** 某篇文档只在 BM25 结果里出现、向量结果里没有，那它就只累加 BM25 那一项 —— 天然处理缺失，不需要填默认值。加权求和就没这么好办，你得给缺失的那一侧编一个分数。

---

**本项目里的实现**

`app/services/vector_search_service.py:96-98`：

```python
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[vw, bw],          # 0.7 / 0.3
)
```

LangChain 的 `EnsembleRetriever` 内部就是 RRF。权重常量在 `:43` —— `BM25_WEIGHT = 0.3`，向量权重是 `1 - 0.3`。

为什么给向量更高的权重：本项目的知识库是运维 SOP 和协议文档，用户提问偏自然语言描述（「服务响应变慢怎么排查」），语义匹配比关键词匹配更管用。BM25 那 0.3 的存在价值是兜住**语义模型的盲区** —— 具体的错误码、服务名、指标名这类专有名词，embedding 容易把 `ERR_5023` 和 `ERR_5032` 编码到相近位置，而 BM25 是精确匹配，不会搞混。

**两条检索路径的分工，一句话概括：向量负责「意思相近」，BM25 负责「字面相同」，RRF 负责在两者之间取共识。**

---

**代价：分数信息真的丢了**

RRF 的代价是**融合后的分数不再有可解释的绝对含义**。0.0164 这个数只说明「它是共识第一名」，不说明「它有多相关」。这个代价在本项目里留下了两处痕迹，处理方式**故意不同**：

**第一处，给调用方排序用 —— 拿名次倒数凑。** `app/services/vector_search_service.py:129` 的注释写明 `EnsembleRetriever` 不暴露融合分数，所以 `search_similar_documents` 只能用 `score=1.0 / index`（`:130`）。这是够用的：调用方只需要一个单调递减的序，不关心数值本身。

**第二处，落库做长期分析 —— 如实存 `None`。** `app/core/used_documents.py:9-15` 走了相反的路，理由写得很直接：「一个看起来像相关性分数、实际是排名倒数的值，比没有分数更危险」。

同一个代价两种处理，判据是**这个值给谁看**。给代码当排序依据，名次倒数没问题；给人做长期数据分析，一个伪装成相关性的排名倒数会导致错误结论 —— 半年后有人拿这一列算「平均相关性下降了」，而那根本不是相关性。宁可为空。保留这个键是为了将来接上 rerank 时不必改表结构。

---

**什么时候 RRF 不够用**

RRF 是**无监督**融合 —— 它不学习，只按名次投票。它的天花板在于：它只知道「谁排前面」，不知道「排前面的这几个哪个才真答了问题」。想再进一步就得上 **rerank（重排）**：用一个 CrossEncoder 把「查询 + 文档」拼成一对送进模型，直接输出相关性分数。CrossEncoder 能看到两者的交互（而向量检索是把查询和文档**分别**编码，交互信息在编码时就丢了），所以精度明显更高，代价是必须逐对推理，慢得多 —— 所以只能用在 RRF 已经筛出的小候选集上。

我在这个项目里试过接 BGE reranker，**离线测评结论是负向的**，所以没上线。原因不在 rerank 本身，而在本项目的候选集太小（`RETRIEVE_TOP_K = 4`，`FINAL_TOP_K = 6`）—— 6 条里重排，排列空间本来就窄，RRF 已经把对的排上来了，重排只能制造抖动。**rerank 的收益需要一个足够宽的候选集（通常 Top-50 到 Top-100）才能体现。** 这也是为什么标准做法是「粗排取宽 → 精排收窄」，我这里粗排就没取宽，精排自然无用。

**项目证据**

- `app/services/vector_search_service.py:96-98` — `EnsembleRetriever` + 权重
- `app/services/vector_search_service.py:43` — `BM25_WEIGHT = 0.3`
- `app/services/vector_search_service.py:129-130` — 融合分数不可得，用 `1.0/index` 代替
- `app/core/used_documents.py:9-15` — 落库如实存 `None` 的理由
- `app/core/milvus_client.py:198-200` — `metric_type="COSINE"`，向量侧的度量
- `app/agent/rag_v2/nodes.py:33-34` — `RETRIEVE_TOP_K = 4` / `FINAL_TOP_K = 6`，候选集偏窄
- CrossEncoder 精排：试过并回退，离线测评负向，当前**未上线**

#### Q5 · 如何评估 RAG 的召回率和最终回答质量？

**回答**

这题必须分两半答，因为本项目**一半有实现，一半是占位**。

**召回侧：有，且能跑 A/B。** `tests/eval/metrics.py`：

- `hit_at_k`（`:69-72`）——前 K 条里有没有命中，看的是「有没有」
- `recall_at_k`（`:75-81`）——命中了几分之几，看的是「够不够」
- `reciprocal_rank`（`:84-87`）——命中在第几名，看的是「排得靠不靠前」
- `compute_case`（`:90-107`）——优先按 chunk 级 id 匹配，拿不到就退化到 doc 级
- `aggregate` / `_aggregate`（`:114-119` / `:122-161`）——`by_tag` 只在外层算，避免递归无限套

评测入口 `tests/eval/run_eval.py`（506 行）支持 `--retriever vector|local-rerank` 做召回方案 A/B、`--tag` 分桶、`--filter domain`、`--only-names`、`--top-k`、`--compare`。`VectorServiceRetriever` 里 `candidate_k = max(top_k * 3, 10)`（`:98`），和生产代码的取宽策略对齐。另外有个 `CheatRetriever`（`:65-76`）直接返回标准答案——这是给指标本身做自测的，确认指标算对了再去测检索器。

**回答质量侧：没有。** `faithfulness_placeholder`（`tests/eval/metrics.py:168-174`）和 `answer_relevance_placeholder`（`:177-179`）都是 `raise NotImplementedError`，docstring 里写着 TODO 接 ragas。**Faithfulness / Answer Relevance 项目中没有，不能硬套。**

但线上确实有一个更粗、且真在跑的东西，别和评测混为一谈：`validate_answer_node`（`app/agent/rag_v2/nodes.py:334-469`）做证据校验和幻觉拦截，并且区分 fail-open 和 fail-closed——验证器自己挂了要放行（不能因为质检坏了就不发货），验证器判定答案没有证据支撑才拦截。这是**运行时闸门**，不是量化指标：它只管这一次要不要放出去，不告诉你整体质量是多少。离线量化和在线拦截是两件事，一个缺了另一个补不上。

**项目证据**

- `tests/eval/metrics.py:69-72,75-81,84-87,90-107,114-119,122-161` — 召回三指标与聚合
- `tests/eval/run_eval.py:41-52,59-62,65-76,83-98` — 评测入口、Retriever 协议、CheatRetriever
- `tests/eval/metrics.py:168-174,177-179` — Faithfulness / Answer Relevance 均为 `NotImplementedError`
- `app/agent/rag_v2/nodes.py:334-469` — 在线证据校验（不是评测指标）

### 面经 14 · 2026-07-10

#### Q1 · 请做一下自我介绍

**回答**

（本篇与面经 13 Q1 同题，按要求不合并。面经 13 那版按**分层**讲，这版换成按**决策**讲——面试官问第二遍往往是想听点别的。）

我做的是一个 RAG + Agent 后端。技术栈说完只要十秒，我更想讲三个当时纠结过、现在能说清依据的决策。

**第一个：什么时候 return，什么时候 raise。** 这件事在项目里出现了三次，三次结论不同。`app/api/chat_v2.py:151` 是 bare `raise`——HTTP 层必须让状态码说真话，之前是 HTTP 200 配一个 `code: 500` 的响应体，监控看到的成功率是假的。`app/tools/knowledge_tool.py:13-43` 是 `return` 错误字符串——工具抛异常会把整个 agent 循环炸掉，返回错误文本能让模型看到失败原因、自己换个查法。`app/agent/mcp_client.py:68-74` 重试耗尽后 `return CallToolResult(isError=True)`——同上，模型是这个失败的处理者。判据是**谁能处理这个失败**。

**第二个：降级原因按什么划分。** `app/core/errors.py:25-62` 十个 `DegradeReason`，划分依据只有一条——**修复方向不同才拆**。「检索超时」和「检索连接失败」修的是不同的东西，所以是两个；「向量检索超时」和「BM25 超时」修的是同一件事，所以不拆。配套的 `RetrievalStatus`（`:65-78`）是三态而不是布尔，因为「检索失败」和「知识库里确实没有」这两件事，返回值都是空列表，但一个要报警一个不用。

**第三个：熔断器用 `threading.Lock` 而不是 `asyncio.Lock`。** 因为同步的检索调用是在 `to_thread` 里跑的（`app/core/circuit_breaker.py`），`asyncio.Lock` 跨不了线程边界。同理时间基准用 `time.monotonic` 而不是 `time.time`，系统时钟被 NTP 回调时不能让熔断器提前进 HALF_OPEN。

**项目证据**

- `app/api/chat_v2.py:151`、`app/tools/knowledge_tool.py:13-43`、`app/agent/mcp_client.py:68-74` — return vs raise 三处不同结论
- `app/core/errors.py:25-62,65-78` — 降级原因划分依据与三态状态
- `app/core/circuit_breaker.py:90-110,135-153` — 锁选型与单调时钟

#### Q2 · 在工业设备故障诊断与知识协同平台中，检索结果为什么不能直接按向量相似度返回？

**回答**

原帖的「工业设备故障诊断与知识协同平台」是那位面试者的项目，不是本仓库。按同类问题答本仓库的真实链路。

四个原因，每个都能落到代码：

**一、向量相似度对专有名词失效。** 用户问「HNSW 的 efSearch 怎么调」，embedding 会把 `HNSW` 映射到「向量索引」这个语义邻域，于是 IVF、HNSW、ScaNN 的文档相似度都很接近——可用户要的是精确命中那个词。这就是为什么 `app/services/vector_search_service.py:96-99` 里混了 0.3 权重的 BM25：BM25 认字面。

**二、单一召回源不可靠，而且当前的降级是不对称的。** `:92-94` 处理了 BM25 语料为空的情况——降级为纯向量检索，日志留一条 warning。但反过来向量检索挂了**没有**对应的降级为纯 BM25 的路径，`:105-107` 直接 `raise RuntimeError`。这个不对称是实现现状，如实说：向量是主路，BM25 是补充，主路挂了这条链路就是失败，交给上层的三态判定和熔断去处理。

**三、相似度高不等于能回答。** top-3 可能都在讲同一个主题、都没有那个具体答案。所以链路后面还有 `validate_answer_node`（`app/agent/rag_v2/nodes.py:334-469`）做证据校验——检索的职责是「找到可能相关的」，判断「够不够回答」是另一个节点的事。

**四、candidate_k 必须和 top_k 分开。** `:84` 是 `candidate_k = max(top_k * 3, top_k, 10)`。如果两路都只取 `top_k=3`，融合后交集一去重，结果数会被打薄，甚至不足 3 条。先取宽（至少 10）、再融合、最后截断到 `top_k`（`:101`）。

还有一个更根本的：**在这条链路里「按向量相似度返回」连表达都不成立。** `EnsembleRetriever.invoke()` 出来的顺序是 RRF 名次序，不是相似度序，而且它压根不暴露融合分数（`:129` 的注释）。想按相似度排，得先绕过融合层。

**项目证据**

- `app/services/vector_search_service.py:84,92-99,101,105-107,129` — 混合权重、不对称降级、取宽策略、分数不可得
- `app/agent/rag_v2/nodes.py:334-469` — 证据校验是独立职责

#### Q3 · 跨区域库存调拨平台如何避免「库存扣了但订单失败」或者「订单成功但库存没扣」？

**回答**

**项目中没有，不能硬套。** 本仓库没有库存、没有订单、没有跨区域部署、没有分布式事务，也没有 TCC / Saga / 本地消息表 / 事务消息中任何一种实现。

**不能拿什么冒充**：`app/core/database.py` 的 `SessionLocal`（`autoflush=False, autocommit=False, expire_on_commit=False`）是单库单机的 SQLAlchemy session，它解决的是「一个请求里什么时候 flush、什么时候 commit」这种单库事务边界问题。跨服务、跨库的一致性是完全不同的问题域——单库事务再怎么讲也变不出二阶段提交，拿它答这题是偷换概念。

**有关系、但要划清边界的是幂等。** 分布式一致性方案的最后一环几乎都要靠幂等键兜底，这一环本仓库有真实实现：

- `app/api/file.py:81` 对上传内容算 `sha256`，`:95-99` 用 `content_hash` 比对，同一份内容重复提交不产生第二份索引。
- `app/workers/index_worker.py:62-68` 对每个切片单独算 `chunk_hash`，所以任务被 RQ 重试时，已经入库的分片不会重复写入。
- `app/core/used_documents.py:52-53,81` 的 `slim_used_documents` 是显式设计成幂等的，因为流式路径的数据可能被反复加工，幂等让调用方不必判断「这个到底瘦没瘦」。

这些能支撑「重试必然发生，所以消费侧必须幂等」这个论点，是真实的实例。但它们只覆盖**单库单表**的重复写入，不涉及跨服务的原子性。

**项目证据**

- `app/api/file.py:81,95-99`、`app/workers/index_worker.py:62-68`、`app/core/used_documents.py:52-53,81` — 幂等键的真实实现
- `app/core/database.py:1-33` — 单库事务边界，不能当分布式事务讲
- 库存 / 订单 / 跨区域 / TCC / Saga / 本地消息表 / 事务消息：项目中没有，不能硬套

#### Q4 · 为什么不能把所有历史对话直接塞给大模型？

**回答**

这题看着简单，但答"窗口装不下"只答到了最表层。真正的原因有四条，而且**它们的性质完全不同**——一条是硬报错，一条是花钱，一条是效果悄悄变差，一条是会给出错误答案。第三条和第四条最要紧，因为它们不报错，你不会发现。

---

**先说清"上下文窗口"到底是什么限制**

Transformer 的 self-attention 要算每个 token 和其他所有 token 的关系，是一个 n×n 的注意力矩阵。序列长度翻倍，计算量变四倍——这是 **O(n²)** 复杂度。另外推理时要缓存每一层每个 token 的 Key 和 Value 向量（**KV cache**），这部分是 O(n) 的显存占用，长上下文时它会成为显存瓶颈。

所以"窗口 128K"不是一个随便设的数字，是训练时的位置编码范围加上工程上能承受的显存和延迟综合出来的上限。超出这个范围有两种表现：要么直接报错（API 拒绝请求），要么模型在训练时没见过这么远的位置，注意力行为退化（外推失效）。

**原因一：硬上限——功能问题。**

超了直接 400 错误。这条最容易理解也最容易防：算 token 数，超了就裁。它的危险在于**触发时机不可控**——一个正常跑了两周的会话，某天用户贴了一大段日志进来，突然就炸了。所以预算不能只在"感觉快满了"的时候算，得每轮都算。

**原因二：成本和延迟——按 token 计费，还影响首字延迟。**

输入 token 是要计费的，而且历史对话是**每轮都重传一遍**。第 10 轮的时候，前 9 轮的内容已经被传了 9 次。成本随轮数是 O(n²) 增长，不是线性。

延迟这边要分清两段：
- **prefill（预填充）**：把输入的所有 token 过一遍，算出 KV cache。这一段的耗时和输入长度成正比，它决定**首 token 延迟（TTFT）**。
- **decode（解码）**：逐 token 生成输出。这一段和输入长度关系不大，主要看输出多长。

所以塞长历史，最直接的体感是"半天不出第一个字"。用户对首字延迟极度敏感——等 3 秒才开始吐字，比吐字慢但立刻开始要难受得多。

**原因三：Lost in the Middle——注意力会稀释，而且不报错。**

这是 2023 年斯坦福那篇 *Lost in the Middle: How Language Models Use Long Contexts* 的发现：把关键信息放在长上下文的**开头或结尾**，模型能用上；放在**中间**，准确率显著下降，有时甚至不如干脆不给这段上下文。

形状是个 U 型曲线——两端高，中间低。原因大致是训练数据里重要信息通常出现在开头（题设）和结尾（结论），模型学到了这个先验；加上位置编码在长距离上的衰减。

这条的可怕之处是**它不报错**。你塞了 50 轮历史，模型看起来正常回答，但它其实压根没用上第 20 轮那句关键约束。你查日志发现"上下文里明明有啊"，然而模型确实没读到。所以塞进去 ≠ 用得上，这是长上下文最反直觉的一点。

**原因四：过期事实冲突——会给出错误答案。**

第 3 轮用户说"部署在华东"，第 30 轮说"已经迁到华北了"。全塞进去，模型可能采信前者——尤其当前者的表述更明确、后者是顺带提的。

这条比第三条更严重：第三条是漏掉信息，第四条是**用错信息**。而且用户会觉得"我明明说过了"，信任直接崩掉。

在 AIOps 场景这条会变得特别危险：十分钟前 CPU 是 92%，现在用户问"现在怎么样"，模型很容易把历史数值当当前值报出来。这不是"记错了"，是**把过期数据当实时数据**，值班的人会照着一个假数字去处置。

---

**那怎么裁？裁剪策略有一个谱系**

从便宜到贵排：

| 策略 | 做法 | 成本 | 丢什么 |
|---|---|---|---|
| **滑动窗口** | 只保留最近 N 条 | 零 | 早期内容永久丢失 |
| **Token 预算截断** | 按 token 算，从头砍 | 零 | 同上，但预算更准 |
| **摘要压缩** | 早期内容压成摘要 | 一次 LLM 调用 | 细节丢失，主线保留 |
| **滚动摘要** | 增量压缩，只摘新增部分 | 一次 LLM 调用（量小） | 同上，但成本不随轮数涨 |
| **向量召回** | 历史存向量库，按当前问题召回相关片段 | 检索 + 存储 | 不丢，但可能召不回 |
| **分层记忆** | 短期原文 + 中期摘要 + 长期向量库 | 全都要 | 最少，最复杂 |

选哪个的判据是**这个会话有多长、历史信息的价值有多高**：

- 客服会话通常 5-10 轮就结束，滑动窗口够用，上摘要是浪费。
- 长期助手类产品（用户可能聊几个月），必须有摘要甚至向量召回，不然"你上次说过"这种能力根本做不出来。
- 需要审计的场景（金融、医疗、运维），**原文必须全量落库**，摘要只是给模型看的投影——这一条和裁剪不冲突，是两件事。

---

**我项目里两套实现，取舍不同，正好对着讲**

**实现一：按条数裁，粗但零成本。**

`app/services/rag_agent_service.py:37-74` 的 `trim_messages_middleware`。逻辑很直白：`len(messages) <= 7` 不裁；超过则保留 `messages[0]` 加最后 6 条。

两个细节值得说：

保留首条是刻意的——首条通常是 system prompt 或用户的首个提问，丢了会失去话题锚点。这也正好利用了 Lost in the Middle 的 U 型特性：**首尾是模型注意力最强的位置**，那就把最重要的两样东西放在这两头。

奇数情况下取 7 条而不是 6 条，是为了保证 human/ai 成对。裁成"半轮"（只剩 AI 回复没有对应的用户提问）会让模型困惑，某些模型的 API 甚至直接拒绝这种消息序列。

实现方式是 `{"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_messages]}`——全删再重放。这是 LangGraph 消息状态的标准改写手法，因为 `messages` 字段的 reducer 是 append 语义，你没法"就地删除"，只能先清空再写入。

代价：看条数不看 token。一条 5000 字的消息和一条"好的"算一样，预算完全失控。被裁掉的内容永久丢失，没有任何回收途径。

**实现二：滚动摘要 + 有界窗口 + 游标，细但多花一次 LLM 调用。**

`app/services/conversation_memory_service.py`。核心是 `build_context`（`:93-131`）：

读活跃消息 → 取上次快照 → 用快照的 `summarized_through_message_id` 算出之后的增量 → 保留最近 N 轮原文 → 剩下的如果超过压缩阈值就压成摘要 → 写回新快照并更新游标。

**游标是这套设计的关键**，它让摘要变成"增量"而不是"每轮重做"。没有游标的话，每轮都要把全部历史重新摘一遍——成本随轮数线性增长，而且每次摘要结果会漂移（同一段内容摘两次，措辞和取舍都会不一样，滚几轮就面目全非）。有了游标，每次只摘"上次摘完之后新增的那几条"，成本恒定。

三个我踩过的细节：

**Token 估算必须认中文。** `estimate_tokens`（`:48-54`）对 CJK 字符按一字一 token 算，其余按四字符一 token。用通用的 `len/4` 会把中文严重低估——中文一个字通常就是 1-2 个 token，按 4 个字符算一个 token 等于低估 4 倍，预算形同虚设。这里选择**宁可高估**：估低了是预算穿透（真炸），估高了只是少塞一点（无害）。

**游标要防软删。** `_messages_after_snapshot`（`:133-144`）在游标指向的消息被软删除时返回 `[]`，而不是把整段历史当成"未摘要"重新处理。否则会把已经摘过的内容再摘一遍，摘要越滚越长还越滚越重复。

**摘要失败不能阻断对话。** `_summarize`（`:163-186`）失败时降级为纯最近窗口。记忆是增强，不是前置依赖——"因为摘要服务挂了所以你不能聊天"是荒谬的因果。这条在 `app/api/chat_v2.py:28-34` 的 `_load_conversation_memory` 里又兜了一层。

**预算分配的顺序不能反。** `app/config.py:133-136` 是四个独立参数：注入上限 1800、摘要上限 700、压缩阈值 2400、最近轮数 3。`_render_context`（`:207-247`）里摘要和最近消息**共享** 1800 这个总预算——先扣掉摘要头和摘要本身的开销，剩下的给最近消息。

顺序不能反：如果先装最近消息，它可能把摘要挤掉，而摘要压缩了几十轮内容，单位信息密度比任何单条消息都高得多。挤掉摘要保留几条闲聊，是把最值钱的东西扔了。

`_take_tail_to_budget`（`:249-272`）保尾不保头——最近的对话优先。这和 `trim_messages_middleware` 保首条的做法看似矛盾，其实不冲突：那边的首条是 system prompt（锚点），这边被截断的是普通对话消息（越早越不重要）。

**原文全量留在 PostgreSQL。** `migrations/versions/20260823_0007_create_conversation_memory_snapshots.py` 建的是快照表，原始消息在另一张表里完整保留。摘要只是投影——摘丢了信息可以回读原文重建，而 `MemorySaver` 那条路径裁掉就是真的没了。

---

**两套并存，判据是什么**

`create_agent` 那条路径的会话通常短，工具调用顺序不确定，按条数裁够用，不值得多付一次 LLM 调用。

`chat_v2` 那条要跨请求、要审计、会话可能很长，值得付摘要成本。

一句话：**会话越长、历史价值越高、越需要审计，越要往谱系右边走。**

需要说清楚的缺口：我这里**没有向量召回式的历史记忆**，也没有跨会话的长期记忆。所有记忆都是会话内的。要做"用户三个月前提过的偏好"这种能力，得把历史消息向量化并按需召回，这块**项目里没有**。

**项目证据**

- `app/services/rag_agent_service.py:37-74` — 按条数裁剪，保首条 + 成对保证 + `REMOVE_ALL_MESSAGES` 重放
- `app/services/conversation_memory_service.py:93-131` — `build_context` 主流程与游标
- `app/services/conversation_memory_service.py:48-54` — CJK 感知的 token 估算，宁可高估
- `app/services/conversation_memory_service.py:133-144` — 游标防软删
- `app/services/conversation_memory_service.py:163-186` — 摘要失败降级
- `app/services/conversation_memory_service.py:207-247,249-272` — 预算分配顺序与保尾截断
- `app/services/conversation_memory_service.py:28-35` — 摘要 prompt 明禁把历史数值写成实时值
- `app/config.py:133-136` — 1800 / 700 / 2400 / 3 轮
- `app/api/chat_v2.py:28-34` — 外层再兜一次
- `migrations/versions/20260823_0007_create_conversation_memory_snapshots.py` — 快照表，原文另存
- 向量召回式历史记忆、跨会话长期记忆：项目里没有

#### Q5 · Kafka 消费者处理库存事件时，如何做到「恰好一次」？

**回答**

先说结论：**「恰好一次投递」在分布式系统里做不到，能做到的是「恰好一次处理效果」。** 这两个说法差一个字，但是完全不同的命题——面试时把这个区分说清楚，比背 Kafka 配置项有用得多。

---

**一、三种投递语义，以及为什么 exactly-once delivery 不可能**

任何跨网络的消息传递都面临同一个问题：**发送方无法区分「消息没送到」和「消息送到了但确认丢了」**。

假设消费者处理完一条消息，正要提交 offset 的时候进程崩了。重启之后它读到的还是旧 offset，于是**同一条消息被处理第二次**。反过来，如果它先提交 offset 再处理，崩溃就意味着**这条消息永远不会被处理**。

这就逼出三种语义：

| 语义 | 做法 | 后果 |
|---|---|---|
| **at-most-once**（至多一次） | 先提交 offset，再处理业务 | 崩溃时消息丢失。适合日志采集这类丢一条无所谓的场景 |
| **at-least-once**（至少一次） | 先处理业务，再提交 offset | 崩溃时消息重复。**这是工业界默认选择** |
| **exactly-once**（恰好一次） | 让「处理业务」和「提交 offset」变成一个原子操作 | 需要额外机制，成本高 |

为什么 exactly-once **投递**不可能：这是「两将军问题」的变体。要保证「恰好送到一次」，发送方必须知道接收方到底收没收到；而告知这件事本身要走网络，网络本身又可能丢。理论上需要无限次确认才能确定，实践中做不到。

所以真正可行的路是：**允许重复投递，但让重复处理不产生额外效果**。也就是 at-least-once + 幂等 = exactly-once processing。**这句话是这整道题的核心。**

---

**二、Kafka 自己提供了什么**

Kafka 在 0.11 版本（2017）之后确实提供了 exactly-once 的能力，但它保护的范围要看清楚。

**第一层：幂等 Producer（`enable.idempotence=true`）。**

解决的是**生产端重复写入**。原理是给每个 Producer 分配一个 PID（Producer ID），每条消息带一个单调递增的 sequence number，Broker 端按 `(PID, 分区)` 维度记住最后一个 sequence。如果收到的 sequence 是已经见过的，直接丢弃并返回成功。

所以 Producer 因为超时重发同一条消息，Broker 不会写第二份。这一层是**免费的**（现代 Kafka 默认开启），只保护单个 Producer 会话内、单个分区内的重复。

**第二层：事务 Producer（`transactional.id`）。**

解决的是**跨分区的原子写**。开启后可以这么写：

```
producer.initTransactions()
producer.beginTransaction()
producer.send(记录A到分区1)
producer.send(记录B到分区2)
producer.sendOffsetsToTransaction(消费位点)   ← 关键在这一行
producer.commitTransaction()
```

关键是 `sendOffsetsToTransaction`——它把**消费位点的提交也纳入这个事务**。于是「读取输入 → 处理 → 写出结果 → 提交位点」四件事变成一个原子单元，这就是 Kafka 所谓的 **read-process-write** 模式，也是 Kafka Streams 的 `processing.guarantee=exactly_once_v2` 底层做的事。

实现机制值得讲一句：Broker 端有个 Transaction Coordinator，事务状态存在内部 topic `__transaction_state` 里。提交时，Coordinator 往每个涉及的分区写一条 **control message**（事务标记 COMMIT/ABORT）。

**第三层：消费端 `isolation.level=read_committed`。**

配合上面那层用。设成 `read_committed` 之后，消费者**只能看到已提交事务的消息**——未提交和已回滚的消息对它不可见。默认值是 `read_uncommitted`，什么都能读到，那事务就白做了。

代价：消费者要等事务提交才能读到消息，所以**端到端延迟变大**。而且事务 Producer 有额外开销（每次事务两次 Coordinator 往返），吞吐会掉。

---

**三、致命的限制：Kafka 的 exactly-once 只在 Kafka 内部成立**

这是最容易被追问、也最容易答错的地方。

上面那套 read-process-write 之所以能原子，是因为**输出目标也是 Kafka**——位点和业务数据写在同一个系统里，可以共用一个事务协调器。

而库存这个场景，输出目标是**数据库**。「扣库存」这个动作发生在 MySQL 或 PostgreSQL 里，Kafka 的事务管不到它。于是又回到了老问题：

```
consumer.poll()           → 拿到「扣 100 件库存」事件
db.update(库存 -= 100)     → 数据库提交成功
consumer.commitSync()     ← 这一步之前崩了
```

重启后 offset 没动，同一个事件再来一次，库存被扣两次。**Kafka 的事务对这个场景完全无效**，因为跨了两个存储系统。

所以库存这题的正确答案不在 Kafka 配置里，在**消费端幂等**上。

---

**四、库存场景的三种落地方案**

**方案一：唯一约束 + 去重表（最常用）。**

给每个事件一个全局唯一的业务 ID（不是 Kafka 的 offset，而是上游生成的 `event_id` 或者 `order_id + 操作类型`），然后：

```sql
BEGIN;
  INSERT INTO processed_events(event_id) VALUES (?);  -- 唯一索引，重复会报错
  UPDATE inventory SET qty = qty - 100 WHERE sku = ? AND qty >= 100;
COMMIT;
```

`processed_events` 上有唯一索引。第二次处理同一个事件时，`INSERT` 违反唯一约束，整个事务回滚，库存不会被扣第二次。捕获这个约束冲突异常，直接把它当成「已处理过」，提交 offset 即可。

关键点：**去重记录和业务写入必须在同一个数据库事务里**。分成两步的话，「写了去重记录但业务没执行」或者「业务执行了但去重没记上」都会出问题。

去重表要有清理策略（按时间分区、定期删除超过保留期的记录），否则它会无限增长。

**方案二：状态机 + 条件更新（业务语义幂等）。**

不用额外的去重表，而是让业务写入本身带条件：

```sql
UPDATE orders SET status = 'DEDUCTED', qty = 100
WHERE order_id = ? AND status = 'PENDING';   -- 只有 PENDING 才能改
```

返回受影响行数为 0 就说明已经处理过了。这个方案更轻，但要求业务实体本身有状态字段，而且状态转换是单向的。

**方案三：Transactional Outbox（解决反向问题）。**

上面两个方案解决的是「消费重复」。反过来还有个「本地事务成功但消息没发出去」的问题——也就是原题里的另一半：「订单成功但库存没扣」。

做法是：业务写入和「待发消息」写在同一个本地事务里，写进一张 `outbox` 表。然后由一个独立的 relay 进程（或者 CDC 工具比如 Debezium 读 binlog）把 outbox 里的记录发到 Kafka。

```sql
BEGIN;
  INSERT INTO orders(...);              -- 业务
  INSERT INTO outbox(event_payload);    -- 待发消息，同一个事务
COMMIT;
-- relay 进程异步读 outbox 发 Kafka，发成功后标记或删除
```

这样「业务成功」和「消息一定会发出去」被本地事务绑在一起了。relay 可能重发（所以消费端还是要幂等），但不会漏发。

**这三个方案在实际系统里通常是组合使用的**：Outbox 保证不漏发，去重表保证不重复消费，两端都兜住。

---

**五、还有一个容易忽略的重复来源：Rebalance**

即使代码写对了，consumer group 的 **rebalance** 也会造成重复。

场景：消费者 A 正在处理一批消息，处理到一半时因为 `max.poll.interval.ms` 超时被判定为死亡，分区被重新分配给消费者 B。A 其实还活着，它处理完了、去提交 offset，但它已经不是这个分区的所有者了（`CommitFailedException`）。而 B 从旧 offset 开始，重复处理了这批消息。

对应的处理：
- 调大 `max.poll.interval.ms`，或者减小 `max.poll.records` 让单批处理更快
- 用 `ConsumerRebalanceListener` 在分区被回收前提交 offset
- 根本上还是靠幂等兜住——因为 rebalance 是无法完全避免的

---

**六、本项目的情况（诚实划界）**

**Kafka 项目里没有。** 全仓库唯一出现 `Kafka` 字样的地方是 `uploads/新建_文本文档_(2).txt:1221`，那是一份被上传进知识库的素材文档，是**数据不是代码**。

本项目的异步是 **Redis + RQ**（`app/core/task_queue.py` 全文 11 行，只有 `redis_conn`、`index_queue`、`protocol_pdf_queue` 三个模块级对象）。RQ 是任务队列不是事件流：没有 offset、没有 consumer group、没有分区、没有事务性 producer。**拿它讲 Kafka exactly-once 是概念偷换，我不会这么答。**

但上面第一节那个核心论点——「at-least-once + 幂等消费」——这条路本项目两端都有真实实现，可以拿来说明我理解这个模式：

**「至少一次」这一端。** `app/core/job_failure.py:68-74` 的 `Retry(3, [10, 30, 60])` 是**无条件**重试，因为 RQ 的 `Retry` 不看异常类型。所以 `:77-128` 加了一层判定，遇到不可重试的错误（比如文件格式根本不对）就把 `retries_left` 归零，避免对着一个永远不会成功的任务重试三次。重试存在，就意味着同一个任务体可能被执行多次——这就是 at-least-once 的处境。

**「幂等消费」这一端。** 两处都用的是内容哈希做幂等键，对应上面方案一的思路：
- `app/api/file.py:81` 对上传内容算 `sha256`，`:95-99` 用 `content_hash` 比对，同一份内容重复提交不产生第二份索引
- `app/workers/index_worker.py:62-68` 对每个**切片**单独算 `chunk_hash`，所以任务被 RQ 重试时，已经入库的分片不会重复写入

粒度选择值得说一句：如果只在文件级做幂等，一个任务在写第 50 个切片时崩了，重试时前 49 个会重复写入。切片级哈希让重试可以从任意中断点安全继续——这是**幂等键的粒度要和写入的粒度对齐**。

**还有一处相关的设计。** `app/core/job_failure.py:146-183` 的失败写回用**独立的短生命周期 session**，并且承诺永不抛异常。理由是主流程的 session 在异常发生后可能已经不可用（连接已断），而「记录失败原因」这件事本身再抛一次异常，会把真正的失败原因从 `job.exc_info` 里挤掉，任务永远停在 RUNNING、文档永远停在 INDEXING。

**明确没有的**：Kafka、事务 Producer、`read_committed`、consumer group rebalance 处理、去重表、Outbox 表、CDC。这些**项目中没有，不能硬套**。

**项目证据**

- `app/core/task_queue.py:1-11` — RQ 接线全文，无重试无分类
- `app/core/job_failure.py:68-74` — `Retry(3, [10, 30, 60])`，无条件重试
- `app/core/job_failure.py:77-128` — 不可重试判定，把 `retries_left` 归零
- `app/core/job_failure.py:146-183` — 独立 session 写回失败，永不抛
- `app/api/file.py:81,95-99` — 文件级 `content_hash` 幂等
- `app/workers/index_worker.py:62-68` — 切片级 `chunk_hash` 幂等
- Kafka / offset / consumer group / 事务 producer / read_committed / 去重表 / Outbox：项目中没有，不能硬套

#### Q6 · Agent 调用工具时，如何避免模型越权调用、参数幻觉和循环执行？

**回答**

这三件事听起来是一类问题（都跟工具调用有关），其实是三个完全不同的风险，防线也在三个不同的层。我先把每件事到底是什么讲清楚，再说各自的防法。

---

**先讲清楚：工具调用这件事在技术上是怎么发生的**

理解这三个风险的前提，是知道「模型调用工具」这句话的实际机制 —— 因为很多人以为是模型自己去发的请求，那就完全防不住了。

真实流程是：

1. 我把工具的 **JSON Schema** 连同用户问题一起发给模型。Schema 里有工具名、参数名、参数类型、描述。
2. 模型**不执行任何东西**，它只是生成一段文本，内容是「我想调用 `search_log`，参数是 `{"topic_id": "xxx", "query": "error"}`」。这段文本按 OpenAI 的协议格式化成 `tool_calls` 字段。
3. **我的代码**收到这段文本，解析出工具名和参数，然后**由我去执行**那个函数。
4. 执行结果作为一条 `ToolMessage` 塞回消息列表，再发给模型。

关键在第 3 步：**执行权在我手里，不在模型手里。** 模型只能「请求」，不能「执行」。这个事实决定了三件事里有两件是有硬防线的（因为我可以在第 3 步拦），一件没有硬防线（因为它发生在循环层面，不在单次调用里）。

---

**一、越权调用：模型请求了它不该碰的东西**

**风险是什么。** 有两种形态，容易混：

- **工具级越权**：模型请求了一个它本不该用的工具。比如客服 Agent 的工具集里有个 `refund_order`（退款），模型被用户一句「帮我把这单退了」说动就调了，而这个用户其实无权发起退款。
- **数据级越权**：工具本身没问题，但参数越界。比如 `query_order(order_id)` 是个合法工具，模型传了别人的订单号 —— A 用户问「我的订单怎么样」，模型从上下文里翻出了 B 用户的单号。这种更隐蔽，因为工具调用完全合法。

**为什么不能靠 prompt 防。** 在系统提示词里写「不要调用退款工具」这类约束是**软约束** —— 它跟用户输入在同一个通道里，用户可以用后面的话把它盖掉（这就是 prompt injection 的本质，见后面那类题）。软约束能降低概率，不能构成防线。真正的防线必须在**代码层**，也就是上面第 3 步。

**分层防法，从外到内四层：**

**第 1 层 · 工具集裁剪（能力边界）。** 最有效也最简单的一条：**根本不给它危险工具**。这不是妥协，是主动的设计决策 —— 如果一个只读的 Agent 就能满足需求，那就别给它写操作。我项目里的工具集是彻底只读的：本地两个（`retrieve_knowledge` 查知识库、`get_current_time` 取时间），MCP 侧七个（五个日志查询、两个指标查询）。没有写、没有删、没有变更。所以「越权造成损失」在当前形态下**天然受限** —— 最坏情况是查到了不该查的日志，不会是删了不该删的数据。

这条要诚实说：这是**能力边界**带来的安全，不是**权限校验**带来的安全。面试官如果追问「那你加了写操作怎么办」，答不出来就露怯了。所以下面三层要能说清楚。

**第 2 层 · 按身份动态装配工具集。** 不要给所有用户同一份工具表。构造 Agent 时就根据当前用户的角色过滤：普通用户拿到查询工具，客服拿到查询 + 改地址，主管才拿到退款。这样越权在**模型看不到那个工具**的层面就被排除了 —— 它连请求都请求不出来，因为 schema 里没有。

这比「模型请求了再拦」更好，因为拦截会产生一次失败的工具调用，模型看到失败会重试、会换措辞、会绕，反而消耗预算。看不见就不会想。

**第 3 层 · 执行前的服务端鉴权（真正的硬防线）。** 这是不能省的一层，因为第 2 层只防了工具级，防不住数据级。做法是：在第 3 步执行之前，把「当前用户身份」和「工具参数」一起交给一个鉴权函数。

关键设计点是**身份不能从模型参数里来**。工具签名里绝对不能有 `user_id` 这种参数 —— 一旦有，模型就能填任意值，等于把鉴权的输入交给了被鉴权的对象。身份必须来自**带外通道**：从 HTTP 请求的 session 里取，通过 `ContextVar` 或依赖注入传到工具执行处。模型永远不知道也不能影响这个值。

我项目里有一个现成的带外通道可以直接用：`app/core/request_context.py` 用 `ContextVar` 存 `request_id`，全链路可见（包括异步 worker），而且模型完全接触不到。如果要加工具级鉴权，用户身份就该走这条同样的路。这条通道存在是真的，用它做鉴权**还没做**。

**第 4 层 · 高危操作的人工确认（HITL）。** 有些操作不该由任何自动判断放行 —— 退款、删除、批量变更、对外发消息。这类操作的正确形态是：Agent 只**准备**操作（把参数填好、把影响范围算出来），然后停下，等人点确认。

这个我项目里有真实实现，虽然不在工具链路上：协议 PDF 入库流程有 `AWAITING_CONFIRMATION` 状态（`app/models/protocol_ingestion.py:24`），解析完不直接入库，要等 `/confirm` 或 `/reject` 端点被调用（`app/api/protocol_pdf.py:122-137` / `:141`）。AIOps 报告那边也有类似的：`requires_human` 字段（`app/models/aiops_report.py:97`）和 `pending_confirmations`（`:121-122`），把模型没有证据支撑的判断强制写成「待人工确认项」而不是结论。

所以「HITL 该怎么做」我有实物可以讲，只是它长在别的链路上，不是工具调用上。

---

**二、参数幻觉：请求的工具对，参数是编的**

**风险是什么。** 模型不知道某个 ID，但它必须填一个值才能构成合法的工具调用，于是它**编一个看起来合理的**。典型形态：

- 编 ID：`topic_id: "topic-order-api-prod"` —— 格式完美，数据库里不存在。
- 编时间：算错时间戳，或者用了自己训练时的「今天」。
- 编枚举值：`region: "beijing"` 而实际枚举是 `ap-beijing`。
- 编字段名：查询语句里用了不存在的字段。

**为什么这个比越权更难防。** 越权可以在执行前判断「这个人能不能做这件事」，是个布尔判断。参数幻觉要判断的是「这个值是不是真的」，而这需要去查真实数据 —— 也就是说，**校验成本跟执行成本差不多**。

**Schema 能挡什么、挡不住什么。** JSON Schema 是第一道，但它只管**结构**：参数名对不对、类型对不对、必填项有没有、枚举值在不在范围里。`@tool` 装饰器会从 Python 类型注解自动生成这个 schema，类型不匹配框架会直接挡回去，不进我的函数。

挡不住的是**语义**：`topic_id` 声明为 `str`，模型编的那个字符串类型完全合法。所以 schema 之后必须有第二道。

**真正有效的三个手段：**

**手段一 · 把 ID 解析拆成独立工具（这条最有效）。** 不要让模型直接填 ID，让它先用**人类可读的名字**去查真 ID。

我项目的 CLS 日志服务就是这么设计的，五个工具里有三个是纯查 ID 的：`get_region_code_by_name`（地域名 → 地域码）、`get_topic_info_by_name`（主题名 → 主题信息）、`search_topic_by_service_name`（服务名 → 日志主题）。只有 `search_log` 是真正取数据的。再加一个 `get_current_timestamp` 避免模型自己算时间。

这个设计的道理是：**模型手里只有「order-api 这个服务」这种人类说法，它不可能知道真 ID。** 与其让它猜，不如给它一条从名字查到 ID 的路。而且这个查询工具的失败是**明确的** —— 查不到就返回「没有这个服务」，模型知道自己错了；如果让它直接猜 ID 去查日志，返回的是空结果，模型会以为「这个服务没有日志」，然后给用户一个错误的结论。

**这两种失败的区别很重要**：前者是「你的输入不对」，后者是「数据不存在」。混淆它们会让幻觉从参数层传导到结论层。

**手段二 · 工具内做存在性校验，失败信息要可操作。** 工具执行前查一下这个 ID 存不存在，不存在就返回一段**能指导模型下一步**的错误文本。不要返回「参数错误」，要返回「topic_id 'xxx' 不存在，可用的有 A/B/C，或先调 get_topic_info_by_name 查询」。

差别在于：前者模型只能重试（大概率再编一个），后者模型知道该怎么改。

**手段三 · 收窄参数空间。** 能用枚举就不用自由字符串，能用相对时间（`last_1h`）就不用绝对时间戳，能给默认值就别让模型填。每一个可以由代码决定的参数，都不该交给模型 —— 模型填的每个字段都是一次幻觉机会。

---

**三、循环执行：单步都对，整体不收敛**

**风险是什么。** ReAct 循环的终止条件是「模型认为不需要再调工具了」—— 这是个**模型判断**，不是代码判断。所以它可能不终止。常见形态：

- **同工具重复调用**：模型调 `search_log` 没查到想要的，换个措辞再调，再换再调，措辞在变但本质是同一个查询。
- **两工具互相触发**：A 的结果让模型想调 B，B 的结果让模型又想调 A。
- **参数微调死循环**：时间范围从 1h 改到 2h 到 3h，每次都觉得「再放宽一点就有了」。

**为什么这三件事里它最危险。** 因为它**每一步都是合法的**。越权和参数幻觉都是「某一次调用有问题」，可以在那一次拦下来。循环执行的每一次调用都通过了鉴权、通过了 schema、参数也是真的，问题只存在于**整体轨迹**上。任何单点检查都看不见它。

而且它的代价是复利式的：每一轮都要把完整消息历史重新发给模型，所以第 10 轮的输入 token 远大于第 1 轮。二十轮下来，成本和延迟都不是线性增长。

**四层防线，从粗到细：**

**第 1 层 · 步数硬上限。** LangGraph 的 `recursion_limit`、其他框架的 `max_iterations`。这是最粗但最可靠的一层 —— 无论模型在想什么，超过 N 步就强制停。设成多少取决于任务：单轮问答 5-10 步够了，复杂排查可能要 20-30。

**这一层我项目里没有。** 全仓库 grep 不到 `recursion_limit`、`max_iterations`、`max_steps` 任何一个。`create_agent` 走的是框架默认值（LangGraph 默认 25），我从来没有显式设置过，也就是说**没人验证过这个值对我的场景合不合适**。这是当前实现的真实缺口，我不掩饰。

**第 2 层 · 时间预算。** 整个请求一个总时限，到点强制收尾。这一层我在另一条路径上做了：`rag_v2` 那条显式图有 `chat_total_budget_seconds = 90.0`，用 `asyncio.timeout` 包住整个图的执行（`app/agent/rag_v2/service.py`），超时后靠 `astream(stream_mode="values")` 一路存下来的最后一个状态快照构造**部分结果**返回，而不是全丢。

但要说清楚它的覆盖范围：**这个预算只守 rag_v2 那条图，不覆盖 `create_agent` 那条路径。** 后者上面只有单次 LLM 调用的 60 秒超时（`app/core/llm_factory.py:84`）—— 一个反复调工具的循环，每次调用都在 60 秒内，可以在不触发任何超时的情况下跑很久。这是第 1 层缺失带来的直接后果。

**第 3 层 · 重复检测。** 比步数上限更聪明的一层：记录已经发生过的 `(工具名, 参数指纹)`，重复出现就直接返回「你刚才用同样的参数调过这个工具，结果是 XXX，请换个思路或者告诉用户查不到」。

参数指纹的做法是把参数字典规范化（排序键、去空白）后算 hash。要注意**不能做得太严** —— 分页查询本来就要用不同的 offset 反复调同一个工具，那是正常的。所以通常只对「完全相同」的调用做拦截，或者对「相似度极高」的调用做计数，超过 2-3 次才拦。

**第 4 层 · 无进展检测。** 最细的一层：判断的不是「调用有没有重复」，而是「状态有没有推进」。做法是给循环定义一个进度量 —— 比如「已确认的事实条数」、「已收集的证据数」—— 连续 N 步这个量没变，就判定无进展并终止。

这一层实现成本高，需要业务侧定义什么叫「进展」，一般只在长链任务里做。

**第 2、3、4 层我都没有实现重复检测和无进展检测**，只有时间预算，且只覆盖一条路径。

---

**最后说一个和这三件事都相关的、我项目里真做过的决策：工具失败要 return 不要 raise。**

`app/tools/knowledge_tool.py` 捕获所有异常后 `return f"检索知识时发生错误: {str(e)}", []`，不抛。MCP 侧同样：`retry_interceptor` 指数退避重试（`wait_time = delay * (2 ** attempt)`，三次）耗尽后返回 `CallToolResult(..., isError=True)`，也不抛。

理由是**谁能处理这个失败**。工具失败对模型来说是可处理的信息 —— 它可以换参数、换工具、或者告诉用户查不到。抛出去的话，异常会穿透整个 agent 循环，用户拿到 500，而这次对话里之前已经查到的东西全部作废。

反过来，HTTP 层是 bare `raise`（`app/api/chat_v2.py:151`），因为那一层的消费者是 nginx、负载均衡、Prometheus，它们只读状态码 —— 那里必须让异常穿透到全局处理器，才能给出正确的 5xx。

同一个项目里两个相反的选择，判据是一致的：**看谁是这个失败的处理者。**

**项目证据**

- `app/services/rag_agent_service.py:131,133-137` — 工具白名单与 create_agent
- `app/tools/knowledge_tool.py:13,13-43`、`app/tools/time_tool.py:1-32` — 本地工具与 return-not-raise
- `mcp_servers/cls_server.py:104-106,136-138,169-171,212-214,346-348`、`mcp_servers/monitor_server.py:124-126,277-279` — 七个只读 MCP 工具，含三个「名称→ID」防幻觉工具
- `app/agent/mcp_client.py:18-74,64,68-74` — 指数退避 + 返回 isError
- 工具级鉴权 / 人工审批 / `recursion_limit` 或迭代上限：项目中没有，不能硬套

#### Q7 · PostgreSQL 的 MVCC 为什么会产生死元组？

**回答**

这题要从「数据库怎么让读写不互相等」讲起，死元组是那个设计的必然副产物，不是 bug。

---

**一、先搞清 MVCC 要解决什么问题**

**MVCC = Multi-Version Concurrency Control，多版本并发控制。**

设想没有 MVCC 的世界。事务 A 在读第 100 行，事务 B 要改第 100 行。为了不让 A 读到一个改了一半的中间状态，只能加锁：要么 B 等 A 读完，要么 A 等 B 写完。这就是**读写互斥**。

后果很难接受。一个跑五分钟的报表查询，会把它扫过的所有行都锁住五分钟，期间所有写入全部堵死。这在 OLTP 系统里等于不可用。

MVCC 的思路是：**别让它们争同一份数据，给它们各自一份。**

B 要改第 100 行时，不覆盖原来那份，而是**另写一份新的**。原来那份留着。于是 A 继续读它开始时看到的那个版本，B 写它的新版本，两边谁也不用等谁。

这就是那句常被背诵的「读不阻塞写，写不阻塞读」——它不是一句口号，是「同一行同时存在多个版本」这个物理事实带来的直接结果。

---

**二、快照：谁该看到哪个版本**

有多个版本之后，新问题来了：一个事务该看哪个版本？

PostgreSQL 的答案是**快照（snapshot）**。每个事务（或每条语句，取决于隔离级别）开始时拿一个快照，快照本质上记录了「此刻哪些事务已经提交、哪些还在跑」。

每一行的每个版本都带两个隐藏字段：

| 字段 | 含义 |
|---|---|
| `xmin` | 创建这个版本的事务 ID |
| `xmax` | 删除/覆盖这个版本的事务 ID（还没被删就是 0） |

可见性判断简化说就是：**`xmin` 对我可见，且 `xmax` 对我不可见** —— 这个版本我能看到。

举个具体的：

```
事务 100：INSERT 一行 → 版本 V1，xmin=100, xmax=0
事务 200：UPDATE 这一行 → V1 的 xmax 改成 200
                          新写 V2，xmin=200, xmax=0
```

现在有个事务 150（在 100 之后、200 之前开始的）来读：
- V1：xmin=100 已提交可见，xmax=200 对我不可见（200 在我之后）→ **我看到 V1**
- V2：xmin=200 对我不可见 → 跳过

而事务 300 来读：
- V1：xmax=200 已提交可见 → 这个版本已被删，跳过
- V2：xmin=200 可见，xmax=0 → **我看到 V2**

两个事务读同一行，看到不同内容，谁都没等谁。这就是 MVCC 在工作。

---

**三、关键点：PostgreSQL 的 UPDATE 不是原地修改**

这是这题的核心，也是最反直觉的地方。

`UPDATE users SET name='x' WHERE id=1` 在 PostgreSQL 里做的事是：

1. 在数据页里**追加**一条新版本（新的物理位置）
2. 把旧版本的 `xmax` 标成当前事务 ID
3. 旧版本**继续物理存在**在页里，一个字节都没删

`DELETE` 更明显 —— 它只做第 2 步。数据完全没动，只是标了个「谁删的」。

所以 PostgreSQL 里 UPDATE 和 DELETE 都是**只追加、不覆盖**的。表文件只会长大，不会因为 UPDATE 或 DELETE 自己变小。

> **对比 MySQL/InnoDB**：InnoDB 是**原地更新 + undo log**。它在原位置改数据，把旧值写进 undo log（单独的存储区域）。需要读旧版本时，顺着 `roll_pointer` 去 undo log 里回溯重建。
>
> 两种设计的差异后果很实际：PostgreSQL 的旧版本混在主表里，所以需要 VACUUM 清理主表；InnoDB 的旧版本在 undo log 里，所以长事务导致的是 **undo log 膨胀**而不是主表膨胀。PostgreSQL 的回滚极快（旧版本本来就在原地，什么都不用做）；InnoDB 的回滚要按 undo 反着执行一遍。没有哪个更好，是不同取舍。

---

**四、死元组是什么**

现在可以给定义了。

一个旧版本，在**所有活跃快照都不再可能需要它**之后，就对任何事务都不可见了。但它还占着数据页的空间。

这个「谁都看不见、但还占着地方」的行版本，就是**死元组（dead tuple）**。

注意它和「被删除的行」不是一回事。刚 DELETE 掉的行，对更早的事务可能还可见 —— 那时它还不是死元组，是**还有人需要的旧版本**。只有当最老的活跃事务也已经不需要它了，它才升级为死元组。

判断的那条线叫 **xmin horizon**（也叫 OldestXmin）：当前所有活跃事务里最老的那个事务 ID。比这条线更早、且已被标记删除的版本，才可以回收。

---

**五、后果：膨胀**

死元组积累起来会造成两种膨胀。

**表膨胀（table bloat）。** 数据页数量增长，但有效行数没变。顺序扫描要读更多页 —— 假设一张表被反复 UPDATE，逻辑上 10 万行，物理上可能占了 50 万行的空间，全表扫描就要多读 5 倍的页。

**索引膨胀（index bloat）。** 索引条目仍然指向那些已死的 heap 位置。索引扫描找到条目后还得去 heap 做一次可见性检查（这就是 PostgreSQL 里 index-only scan 需要 visibility map 的原因）。死元组多了，这些检查全是白做的。

膨胀的隐蔽之处在于：**它不报错，只是慢**。而且慢得很均匀，不像死锁那样有明显信号。

---

**六、清理：VACUUM 与 VACUUM FULL**

| | 做什么 | 锁 | 文件是否缩小 |
|---|---|---|---|
| `VACUUM` | 把死元组占的空间标记为可复用 | 不阻塞读写 | **不缩小** |
| `VACUUM FULL` | 重写整个表和索引 | `ACCESS EXCLUSIVE`（全锁） | 缩小 |

关键差别：普通 `VACUUM` 之后文件大小不变，但那些空间可以被后续 INSERT/UPDATE 填进去。也就是说它把膨胀**止住**了，但没有**回收**。

`VACUUM FULL` 才真正把磁盘还给操作系统，代价是全表排他锁 —— 生产环境基本不能随便做（大表可能锁几十分钟）。真要缩容一般用 `pg_repack` 这类工具，它用触发器+影子表的办法在不长时间锁表的前提下完成重写。

**autovacuum** 是后台自动跑普通 VACUUM 的守护进程。默认阈值是「死元组数超过 表行数×0.2 + 50」才触发，对大表来说这个比例太宽松了 —— 一张一亿行的表要积累两千万死元组才触发一次，那时膨胀已经很严重。所以大表通常要单独调 `autovacuum_vacuum_scale_factor` 到 0.01 甚至更低。

---

**七、长事务是最大的放大器（这是实际事故的主因）**

前面说过，回收的界线是 xmin horizon。

一个开着不提交的事务会把 horizon **钉死在它开始的那一刻**。在它结束之前，那一刻之后产生的所有旧版本**一个都不能回收**，因为理论上这个老事务还可能需要看它们。

所以典型的膨胀事故不是「写入量太大」，而是：

- 某个连接 `BEGIN` 了之后忘了 `COMMIT`（应用逻辑 bug，或者连接池配置问题）
- 一个跑了几小时的分析查询
- 一个 idle in transaction 状态的连接（最阴险的一种 —— 它看起来是空闲的）

期间 autovacuum 会正常运行，也会正常报告「我清理了 0 个元组」—— 它不是没跑，是**不允许**清理。

监控上要看的是 `pg_stat_activity` 里的 `state = 'idle in transaction'` 和 `xact_start` 的年龄，以及 `pg_stat_user_tables.n_dead_tup`。

---

**八、本项目的相关性（诚实划界）**

项目确实用 PostgreSQL（`app/config.py:55-56`，`postgresql+psycopg`，端口 55432），但**VACUUM 调优、膨胀监控、长事务告警一个都没有** —— 这些是运维层面的东西，项目里没做，不能硬套。

有一处间接相关、但当初不是为这个设计的：`app/core/job_failure.py:146-183` 的失败写回刻意用一个**独立的短生命周期 session**，不复用主流程那个。

当初的理由跟 MVCC 无关 —— 是因为主 session 在异常发生后可能已经不可用，在 `except` 里用它 commit 会抛二次异常，把真正的失败原因掩盖掉。

但它顺带避免了一个坏模式：假如失败写回复用那个跨越整个任务执行期的 session，那就是一个长事务 —— 正好是钉住 horizon 的典型写法。同理 `app/core/database.py` 的 `get_db` 写成生成器，请求结束就关，不让 session 跨请求存活。

这两个决定都是为了别的目的做的，恰好也是 MVCC 友好的写法。面试里可以这么说，但别包装成「我为了避免表膨胀所以这样设计」—— 那是事后叙事。

**项目证据**

- `app/config.py:55-56` — PostgreSQL + psycopg，端口 55432
- `app/core/job_failure.py:146-183` — 独立短生命周期 session（初衷是异常安全，顺带避免长事务）
- `app/core/database.py:1-33` — `get_db` 生成器，请求级 session 生命周期
- VACUUM / autovacuum 调优、膨胀监控、长事务告警、`pg_repack`：项目中没有，不能硬套

---

## 二、字节跳动 AI Agent 开发岗-02

来源：https://www.nowcoder.com/discuss/922659486983036928

### 面经 01 · 2026-07-06

#### Q1 · 二面：自我介绍 → 手撕 → 更换/添加条件手撕 → 哈希表的了解 → 场景设计（让说用什么数据结构）

**回答**

这是一条流程记录而不是单一问题，逐段说。

自我介绍部分我会讲这个仓库：一个 oncall 运维方向的 RAG Agent 服务，FastAPI + LangGraph，入口在 `app/main.py:51`，两条问答链路（`app/api/chat.py` 走 LangChain `create_agent`，`app/api/chat_v2.py` 走显式 LangGraph 图）。我的重点不在"接通了大模型"，而在失败路径：降级语义、熔断、总预算、部分结果。

手撕和"更换条件再手撕"这两段**项目中没有对应实现，不能硬套**——仓库里没有算法题代码，`tests/` 下只有评测和单测。硬把 `_tokenize_for_bm25`（`app/services/vector_search_service.py:203-224`）说成"手撕算法"是偷换概念，那是个字符分类循环，不是数据结构题。

哈希表这段仓库里有真实使用，而且用在了"为什么选哈希"的关键位置上：
- 分片去重用 SHA-256 摘要，`app/workers/index_worker.py:62` 拿 `hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()` 当 chunk 主键；文件级去重在 `app/api/file.py:81`，`app/api/file.py:99` 用 `existing.content_hash == content_hash` 判断"同一份文件重复上传"。这里选散列而不是全文比较的理由是把 O(n) 的内容比较压成定长键比较，代价是要接受碰撞概率——SHA-256 下这个概率工程上可忽略。
- rag_v2 的 `dedup` 节点同样按内容指纹去重（`app/agent/rag_v2/nodes.py:214-261`），因为 3+1 条子查询并行检索必然召回重叠文档。

场景设计"用什么数据结构"这段可以接到真代码上：`app/agent/rag_v2/state.py:17,27,34` 三个字段选了 `Annotated[List[...], operator.add]` 而不是 dict 或 set。选 list + reducer 的原因是 LangGraph 的 fan-in 需要一个可结合的合并操作，list 拼接天然满足；换成 dict 会在并行分支写同一个 key 时互相覆盖，换成 set 会丢掉召回顺序，而顺序在 `dedup` 裁剪 `FINAL_TOP_K`（`app/agent/rag_v2/nodes.py:31`，值 6）时是有意义的。

**项目证据**

- `app/workers/index_worker.py:62,68`、`app/api/file.py:81,95,99,132` — 散列去重的两处真实用法
- `app/agent/rag_v2/nodes.py:214-261,31` — 指纹去重与 Top-K 裁剪
- `app/agent/rag_v2/state.py:17,27,34` — 为什么是 list + `operator.add` 而不是 dict/set
- 手撕算法题：项目中没有，不能硬套

#### Q2 · 总结：首先声明我投的是 AI 应用开发，给我调剂到了云架构后端开发？

**回答**

这是面经作者的吐槽而不是技术题，但值得当成一个真实问题回答：被调剂到后端岗时，手上这个项目能不能撑住后端方向的拷打。

答案是能，而且不需要改口径。这个仓库超过一半的代码量根本不是"调模型"，是后端工程：
- 关系库层：`app/core/database.py:1-33`（`DeclarativeBase` + `pool_pre_ping=True` + `expire_on_commit=False`）、`app/models/` 下七个 ORM 模型、`app/repositories/` 三个仓储。库是 PostgreSQL（`app/config.py:55-56`）。
- 异步任务层：Redis + RQ，队列接线在 `app/core/task_queue.py:1-11`，失败治理在 `app/core/job_failure.py`（重试策略 68-74、不可重试判定 77-128、失败写回 146-183）。
- HTTP 层：全局异常处理器 `app/core/exception_handlers.py:180-191`、纯 ASGI 中间件 `app/core/middleware.py:1-101`、CORS 与中间件顺序 `app/main.py:58-76`。
- 可观测层：Prometheus 指标 `app/core/metrics.py`、request_id 全链路 `app/core/request_context.py:33`。

所以如果被转到后端，我会把叙述重心从"RAG 效果"换成"这套服务怎么在依赖不稳的情况下不撒谎"，代码是同一份。

**项目证据**

- `app/core/database.py:1-33`、`app/config.py:55-56` — 关系库层
- `app/core/task_queue.py:1-11`、`app/core/job_failure.py:68-74,77-128,146-183` — 异步任务层
- `app/core/exception_handlers.py:180-191`、`app/core/middleware.py:1-101`、`app/main.py:58-76` — HTTP 层
- 云架构（K8s / 服务网格 / 多可用区）：项目中没有，不能硬套

#### Q3 · 一面主要拷打 AI 项目，问各种实现，做了什么贡献

**回答**

"做了什么贡献"要落到能指出行号的改动上，否则就是空话。这个仓库里我能指名的几处：

1. **把 HTTP 状态码改成不撒谎**。改之前端点内部 catch 住异常、返回 HTTP 200 而 body 里写 `code: 500`——监控看到的是全绿。现在 `app/api/chat_v2.py:151` 是裸 `raise`，状态码交给 `app/core/exception_handlers.py:180-191` 的全局处理器决定。
2. **把"检索失败"和"知识库为空"拆成两个状态**。`app/core/errors.py:65-78` 的 `RetrievalStatus` 是三态（OK / EMPTY / FAILED）。之前检索异常被 catch 成空列表返回，生成节点拿到空上下文照样编答案，这是最难查的一类线上问题。
3. **降级原因按"修复方向"分类**。`app/core/errors.py:25-62` 十个 `DegradeReason`，划分依据不是错误来源而是"看到这个原因该去修哪里"。
4. **两层超时预算**。单次调用 60 秒在 `app/core/llm_factory.py:84`，整个请求 90 秒在 `app/agent/rag_v2/service.py:61`。后者用 `asyncio.timeout` 包住 `astream`，超时时不抛异常而是交付 `last_state` 里的部分结果（`app/agent/rag_v2/service.py:56-72`）。
5. **熔断按下游划分**。`app/core/breakers.py` 只有两个实例，理由写在模块文档里——会同时坏也会同时好的调用共用一个熔断器。
6. **计时器放在熔断判定之内**。`app/agent/rag_v2/nodes.py:127-136`，否则熔断打开期间会记录大量约 0 秒的样本，P99 反而变好看。

**项目证据**

- `app/api/chat_v2.py:151`、`app/core/exception_handlers.py:180-191`
- `app/core/errors.py:25-62,65-78`
- `app/core/llm_factory.py:84`、`app/agent/rag_v2/service.py:56-72,61`
- `app/core/breakers.py:1-46`、`app/agent/rag_v2/nodes.py:127-136`

#### Q4 · 最近一年你的 Agent 学习路线是什么，接触过哪些比较好的 Agent 产品？

**回答**

学习路线这部分是个人经历，**项目中没有对应实现**，但仓库的 git 历史恰好能作为路线的客观证据，这比口述可信：

`git log` 里三个提交对应三个阶段——`ac108cd` "IVF_FLAT+L2 → HNSW+COSINE"（索引与度量的选型阶段）、`e5497a8` "建立数据库，记录系统里有哪些文档，新增 RQ 异步队列，redis，hash 去重"（从脚本走向服务的阶段）、`4466801` "rerank 重排召回测试，子查询，证据校验与幻觉拦截，对话进行数据库存储"（从"能答"走向"答得住"的阶段）。

这条线的方向是：先解决召回质量，再解决工程可靠性，最后解决"答案可信度"。仓库里最新一层就是证据校验与幻觉拦截（`app/agent/rag_v2/nodes.py:334-469`）。

"接触过哪些 Agent 产品"属于个人使用经历，仓库里唯一的客观痕迹是 MCP 生态的接入（`app/agent/mcp_client.py:1-183`，两个自建 server 在 `mcp_servers/cls_server.py`、`mcp_servers/monitor_server.py`）。除此之外的产品评价**不能从这个仓库推出来**，面试时应作为个人经验陈述，不要伪装成项目产出。

**项目证据**

- git log `ac108cd` / `e5497a8` / `4466801` — 三阶段路线的客观记录
- `app/agent/rag_v2/nodes.py:334-469` — 最新一层：证据校验
- `app/agent/mcp_client.py:1-183`、`mcp_servers/cls_server.py:20,470`、`mcp_servers/monitor_server.py:27,435` — MCP 生态接入
- 学习路线叙述与产品横向评价：项目中没有，不能硬套

#### Q5 · 你用过 trae、Claude Code、codex，这几个 Agent 之间的优劣势？

**回答**

**项目中没有对应实现，不能硬套。** 这个仓库是一个服务端 RAG Agent，不是 coding agent；没有代码库索引、没有文件编辑工具、没有 shell 执行工具、没有 diff 应用逻辑。把 `app/tools/` 下的两个工具（`knowledge_tool.py` 的知识检索、`time_tool.py` 的取时间）说成"和 Claude Code 一类"是明显的冒充。

能从仓库出发讲的只有一个可迁移的判据：工具层的错误返回方式。`app/tools/knowledge_tool.py:13-43` 捕获异常后 **return** 一个错误字符串给模型，而不是抛出；`app/agent/mcp_client.py:64-74` 重试耗尽后 **return** `CallToolResult(isError=True)`。coding agent 的差异很大一部分就落在这个选择上——工具失败是让模型看见并自己纠正，还是直接中断整轮。这一点可以类比，但产品对比本身不属于这个项目。

**项目证据**

- `app/tools/knowledge_tool.py:13-43`、`app/agent/mcp_client.py:64-74` — 唯一可类比的点：工具错误 return 而非 raise
- coding agent 能力（代码索引 / 文件编辑 / shell 执行 / diff 应用）：项目中没有，不能硬套

#### Q6 · 你如果去评价一个 Agent 的能力，不考虑模型，你怎么去评价？

**回答**

"不考虑模型"是这题的关键约束——它把评价目标锁在 Harness 上。我按仓库里真实存在的可测项来分。

**检索层**。`tests/eval/` 下是能跑的评测：`tests/eval/metrics.py:69-72` 的 `hit_at_k`、`:75-81` 的 `recall_at_k`、`:84-87` 的 `reciprocal_rank`，`compute_case`（`:90-107`）优先用 chunk 级 id，拿不到时降级到 doc 级。金标集在 `tests/eval/golden_set.jsonl`（18 条）和 `tests/eval/golden_set_expanded.jsonl`（83 条）。`tests/eval/run_eval.py:456` 的 `--retriever` 可以切换检索实现，`:461-462` 可以覆盖向量与 BM25 权重——这就是不换模型也能对比 Harness 的手段。

**降级诚实度**。这是我认为比召回指标更能区分生产级和 Demo 的一项：同一个失败，系统是报告失败还是伪装成"没找到"。`app/core/errors.py:65-78` 的三态 `RetrievalStatus` 就是为此存在的，`dedup_node`（`app/agent/rag_v2/nodes.py:214-261`）负责在 fan-in 时做出区分。

**部分结果交付率**。总预算耗尽时能交付多少。`app/agent/rag_v2/service.py:56-72` 用 `astream(stream_mode="values")` 持续记录 `last_state`，超时后拿它构造结果而不抛异常。

**幻觉拦截率**。`app/agent/rag_v2/nodes.py:334-469`，阈值 `MIN_GROUNDEDNESS_SCORE=0.75`（`:33`）、`MIN_COVERAGE_SCORE=0.60`（`:34`）。

**生成质量**这一项要说实话：`tests/eval/metrics.py:168-174` 的 `faithfulness_placeholder` 和 `:177-179` 的 `answer_relevance_placeholder` 都是 `raise NotImplementedError`。**这两项项目中没有实现，不能硬套。**

**项目证据**

- `tests/eval/metrics.py:69-72,75-81,84-87,90-107` — 已实现的检索指标
- `tests/eval/golden_set.jsonl`（18 条）、`tests/eval/golden_set_expanded.jsonl`（83 条）
- `tests/eval/run_eval.py:456,461-462` — 不换模型对比 Harness 的开关
- `app/core/errors.py:65-78`、`app/agent/rag_v2/nodes.py:214-261` — 降级诚实度
- `app/agent/rag_v2/service.py:56-72` — 部分结果
- `app/agent/rag_v2/nodes.py:33-34,334-469` — 幻觉拦截阈值
- Faithfulness / Answer Relevance：项目中没有（占位函数直接抛 `NotImplementedError`），不能硬套

#### Q7 · 场景会限制模型选型。针对不同的模型，去优化一个 Agent 在全流程的表现，你的方法论是什么？

**回答**

方法论的第一条是：把"模型可换"做成代码事实，而不是口号。仓库里的做法是所有 LLM 实例只从一个工厂出（`app/core/llm_factory.py`），换模型是改 `app/config.py:46`（`dashscope_model="qwen-max"`）或 `:129`（`rag_model="qwen-max"`），不需要动节点代码。

工厂里有一处细节直接关系到"换模型不出事"：`app/core/llm_factory.py:70-85` 用 `timeout = config.llm_timeout_seconds if timeout is None else timeout`，注释明确写了不能用 `timeout or config.x`——那样显式传 `timeout=0` 会被静默替换成默认值。换到一个更慢的模型时，调用方需要能显式覆盖超时，这个语义必须保住。

第二条是把模型能力差异吸收到 Harness 里，而不是靠 prompt 硬顶：
- 弱模型改写子查询会失败或输出脏格式。`app/agent/rag_v2/nodes.py:472` 的 `_parse_sub_queries` 带 fallback，解析不出来就退回原问题（`:143` 的 `"sub_queries": [question]`），并且 `:150-152` 强制把原问题插回列表首位——即使改写全废，也保证至少有一条正确查询。
- 弱模型更容易编。所以证据校验（`:334-469`）不是可选项，阈值 0.75 / 0.60 是对模型的外部约束。
- 弱模型指令跟随差，输出格式不稳。生成节点的上下文预算 `rag_document_context_token_budget=2800`（`app/config.py:130`）配合 `_truncate_to_token_budget`（`app/agent/rag_v2/nodes.py:637`）保证输入侧可控。

第三条是有对比手段。`tests/eval/run_eval.py:459` 的 `--compare A B` 就是为"换模型/换参数后是变好还是变坏"准备的。

**项目证据**

- `app/core/llm_factory.py:70-85,84` — 单一构造入口与显式 0 的语义保留
- `app/config.py:46,129,130` — 模型与预算配置点
- `app/agent/rag_v2/nodes.py:143,150-152,472` — 改写失败的兜底
- `app/agent/rag_v2/nodes.py:334-469,617,637` — 证据校验与输入预算
- `tests/eval/run_eval.py:459` — `--compare` 对比
- 多模型路由 / 按难度选模型：项目中没有，不能硬套

#### Q8 · 真实业务场景下一类 Agent 的需求通常是收敛的，你需要对 Agent 在特定场景上做优化，有什么优化思路？

**回答**

"需求收敛"这个前提在这个仓库里是成立的——场景就是 oncall 运维问答，知识库是 `aiops-docs/` 下六篇 SOP（cpu / disk / memory / api 延迟 / 服务不可用 / 慢响应）。收敛带来的优化空间是实打实的：

**分词可以按语料定制**。`app/services/vector_search_service.py:203-224` 的 `_tokenize_for_bm25` 是自己写的：ASCII 字母数字加 `_`、`-` 连成一个 token（保住 `drop_caches`、`free -h` 这类运维命令不被切碎），中文按单字切。通用分词器会把 `drop_caches` 切成两段，运维场景下这正是最关键的检索词。

**权重可以按语料调**。向量 0.7 / BM25 0.3（`app/services/vector_search_service.py:42-43`）是给这批语料定的，且留了运行时覆盖参数（`:56-57,75-76`），注释写明"评测 / A/B 时可临时覆盖，生产代码调用不需要传"。

**子查询数量可以按场景收紧**。`app/agent/rag_v2/nodes.py:29` 的 `NUM_SUB_QUERIES=3`，加原问题共 4 条并行（`:152`）。运维问题通常有"现象—原因—处置"三个角度，3 是按这个定的，不是随手填的。

**候选集可以按库大小定**。`app/services/vector_search_service.py:84` 的 `candidate_k = max(top_k * 3, top_k, 10)`——库小的时候固定下限 10 比按倍数放大更稳。

**金标集直接用场景语料**。`tests/eval/golden_set.jsonl` 里问题是口语化的（"线上机器 CPU 一直九十多下不来，我先看哪里?"），tags 带 `colloquial`，`expected_doc_ids` 指向 `cpu_high_usage`。收敛场景下金标集能做得很贴真实用户说话方式，这是通用 benchmark 给不了的。

**项目证据**

- `aiops-docs/` 六篇 SOP — 收敛的语料范围
- `app/services/vector_search_service.py:203-224` — 按运维语料定制的分词
- `app/services/vector_search_service.py:42-43,56-57,75-76,84` — 权重与候选集
- `app/agent/rag_v2/nodes.py:29,152` — 子查询数量
- `tests/eval/golden_set.jsonl` — 口语化金标集

#### Q9 · skill 的目录结构是什么样的？

**回答**

**项目中没有，不能硬套。** 这个仓库没有 skill 机制：没有 skill 目录、没有 skill 注册表、没有 skill 加载器、没有按需装载的分层。`app/tools/` 下只有两个用 `@tool` 装饰的普通函数（`knowledge_tool.py:13`、`time_tool.py`），加 MCP 侧五个 CLS 工具和两个 Monitor 工具，一共七个工具全部在 `_initialize_agent`（`app/services/rag_agent_service.py:131`）里一次性合并进 `create_agent`。这是"固定工具集"，不是 skill。

差别不能含糊：skill 的要点在于渐进式披露（先只给名字和描述，命中后才加载正文），而这里 `all_tools = self.tools + self.mcp_tools` 是全量注入。硬把工具列表说成 skill 目录，面试官追问"那你怎么做到不把所有 skill 正文都塞进上下文"就直接露底了。

**项目证据**

- `app/tools/knowledge_tool.py:13`、`app/tools/time_tool.py` — 本地两个工具
- `mcp_servers/cls_server.py:104,136,169,212,346`、`mcp_servers/monitor_server.py:124,277` — MCP 侧五 + 二个工具
- `app/services/rag_agent_service.py:131,133-137` — 全量注入，非按需加载
- skill 目录结构 / 注册 / 渐进披露：项目中没有，不能硬套

#### Q10 · Claude Code 检索代码的方式？

**回答**

**项目中没有，不能硬套。** 仓库里没有任何代码检索能力：没有 AST 解析、没有符号索引、没有 ripgrep 类工具封装、没有仓库文件遍历工具。检索链路只面向文档语料（Milvus 向量 + BM25），入口是 `app/services/vector_search_service.py:52-107`。

唯一沾边的对比点值得说清楚，因为它解释了为什么代码检索一般不走向量：这个仓库的混合检索里 BM25 占 0.3 权重（`app/services/vector_search_service.py:43`），而它的分词函数（`:203-224`）专门保住了 `drop_caches` 这种带下划线的标识符不被切碎。代码检索场景下精确标识符匹配的重要性远高于语义相似——所以那类工具以字面/正则搜索为主。这个推论可以讲，但仓库里没有实现，不能算项目经验。

**项目证据**

- `app/services/vector_search_service.py:43,52-107,203-224` — 面向文档而非代码的检索链路
- 代码索引 / AST / 符号表 / 仓库级搜索工具：项目中没有，不能硬套

### 面经 02 · 2026-07-03

#### Q1 · 用的什么基础模型？

**回答**

三处分开说，因为它们确实不是同一个配置项。

对话与生成用 DashScope 的 `qwen-max`：`app/config.py:46` 是 `dashscope_model: str = "qwen-max"`，rag_v2 链路另有 `app/config.py:129` 的 `rag_model: str = "qwen-max"`，后面跟着注释"使用快速响应模型，不带扩展思考"。两个配置项而不是一个，是为了让 rag_v2 能独立换模型而不影响 chat 链路。

Embedding 用 `text-embedding-v4`：`app/config.py:47`，注释写明"v4 支持多种维度（默认 1024）"。Milvus 侧维度就是 1024。

接入方式是 OpenAI 兼容协议：`app/core/llm_factory.py` 里构造的是 `ChatOpenAI`，`base_url` 指向 DashScope 兼容端点（`LLMFactory.DASHSCOPE_BASE_URL`）。这一点在面试里值得主动说，因为它意味着换供应商只改 base_url 和 model 名。

需要纠正一处常见误记：仓库里**没有**使用 BGE 作为 embedding 模型。`.hf_cache/hub/models--BAAI--bge-reranker-v2-m3/` 这个缓存目录存在，但它是 reranker 而非 embedding，且不在任何线上代码路径里。

**项目证据**

- `app/config.py:46,47,129` — 对话模型、embedding 模型、rag_v2 模型
- `app/core/llm_factory.py:70-95` — `ChatOpenAI` + DashScope 兼容端点
- BGE embedding：项目中没有使用，不能硬套

#### Q2 · 介绍一下 RAG Agent 项目

**回答**

一句话定位：面向 oncall 运维的知识问答服务，两条链路并存，重点在失败路径的正确性而不是"接通大模型"。

**链路一（chat）**走 LangChain：`app/services/rag_agent_service.py:133-137` 的 `create_agent(self.model, tools=all_tools, checkpointer=self.checkpointer)`，checkpointer 是 `MemorySaver`，工具由模型自己决定调不调。这条路适合多轮工具编排，但执行顺序不可控。

**链路二（chat_v2）**走显式 LangGraph：`app/agent/rag_v2/graph.py:22-51`，五个节点连成 rewrite → retrieve_each（Send fan-out）→ dedup → generate → validate_answer。每个节点都被 `instrument_node` 包一层（`:35-41`），注释解释了为什么统一包而不在节点体内计时——五份计时代码是重复，且新增节点必然有人忘记加。

**检索**是混合检索：Milvus 向量（HNSW + COSINE，1024 维）加 BM25，用 `EnsembleRetriever` 做 RRF 融合（`app/services/vector_search_service.py:96-99`），权重 0.7 / 0.3（`:42-43`）。BM25 语料从 Milvus 反读（`:165-201`），并按 `collection.num_entities` 变化决定是否重建（`:143-147`）——这是个缓存失效判据，文档数没变就复用。

**可靠性**是这个项目真正花时间的地方：三态检索状态（`app/core/errors.py:65-78`）、十个按修复方向划分的降级原因（`:25-62`）、两层超时（`app/core/llm_factory.py:84` 与 `app/agent/rag_v2/service.py:61`）、超时后交付部分结果（`app/agent/rag_v2/service.py:56-72`）、两个按下游划分的熔断器（`app/core/breakers.py`）、证据校验与幻觉拦截（`app/agent/rag_v2/nodes.py:334-469`）。

**异步侧**是 Redis + RQ：文档入库和协议 PDF 解析走 worker（`app/workers/index_worker.py`、`app/workers/protocol_pdf_worker.py`），失败治理在 `app/core/job_failure.py`。

**项目证据**

- `app/services/rag_agent_service.py:133-137`、`app/agent/rag_v2/graph.py:22-51,35-41` — 两条链路
- `app/services/vector_search_service.py:42-43,96-99,143-147,165-201` — 混合检索与语料缓存
- `app/core/errors.py:25-62,65-78`、`app/agent/rag_v2/service.py:56-72,61`、`app/core/llm_factory.py:84`
- `app/agent/rag_v2/nodes.py:334-469`、`app/core/breakers.py:1-46`
- `app/workers/index_worker.py`、`app/core/job_failure.py:68-74,77-128,146-183`

#### Q3 · 评测集的整体构建思路？

**回答**

仓库里是两份 JSONL，一份小一份大：`tests/eval/golden_set.jsonl` 18 条，`tests/eval/golden_set_expanded.jsonl` 83 条。分两份的用途不同——小的用于改动后的快速回归，大的用于出结论。

单条的字段结构（取 `golden_set.jsonl` 第一条）：

```json
{"id": "q-001", "question": "线上机器 CPU 一直九十多下不来,我先看哪里?",
 "expected_chunk_ids": [], "expected_doc_ids": ["cpu_high_usage"],
 "reference_answer": "通过 top/htop 定位高 CPU 进程...",
 "tags": ["aiops", "cpu", "easy", "colloquial"]}
```

几个设计点：

**chunk 级和 doc 级两套期望 id 并存**。`expected_chunk_ids` 常为空、`expected_doc_ids` 一定有值。原因在 `tests/eval/metrics.py:90-107` 的 `compute_case`：它优先用 chunk 级 id，拿不到就降级到 doc 级。这样切分策略改变（`chunk_max_size` / `chunk_overlap` 调整）后，chunk id 全部失效，但 doc 级期望依然可用，评测集不用重标。

**问题按真实用户口吻写**。tags 里有 `colloquial`，问题是"CPU 一直九十多下不来"而不是"如何排查 CPU 使用率过高"。用规范问法建的评测集会系统性高估召回。

**tags 是多维的**：领域（`aiops`）、子类（`cpu` / `disk` / `memory`）、难度（`easy` / `medium`）、语体（`colloquial`）。`tests/eval/metrics.py:122-161` 的 `_aggregate` 支持按 tag 分组出分，`run_eval.py:464` 的 `--filter` 和 `:449` 的 `--tag` 配套。分组出分是为了发现"整体 Recall 还行但 memory 类特别差"这种被平均值掩盖的问题。

**`reference_answer` 存了但目前用不上**——生成类指标是占位（见下一题）。存着是为了将来接上时不用重标。

**项目证据**

- `tests/eval/golden_set.jsonl`（18 条）、`tests/eval/golden_set_expanded.jsonl`（83 条）
- `tests/eval/metrics.py:90-107` — chunk 级优先、doc 级降级
- `tests/eval/metrics.py:122-161`、`tests/eval/run_eval.py:449,464` — 按 tag 分组
- 评测集自动生成 / 定期更新流程：项目中没有，不能硬套

#### Q4 · 做评测观测哪些指标？

**回答**

已实现的是检索侧三个，都在 `tests/eval/metrics.py`：

- `hit_at_k`（`:69-72`）：Top-K 里有没有命中，0/1。回答"用户第一屏能不能看到对的"。
- `recall_at_k`（`:75-81`）：Top-K 覆盖了多少期望文档，比例。回答"证据齐不齐"——这个对 RAG 比 hit 更重要，因为生成需要完整证据，只命中一篇可能不够。
- `reciprocal_rank`（`:84-87`）：命中位置的倒数，聚合成 MRR。回答"对的排在多前"，直接影响 `FINAL_TOP_K=6` 裁剪后还剩不剩。

聚合在 `aggregate`（`:114-119`）和 `_aggregate`（`:122-161`），后者里 `by_tag` 只在外层算——注释点明了这是为了避免无限递归。输出走 `run_eval.py:411-413` 按 tag 打印。

生成侧必须说实话：`faithfulness_placeholder`（`:168-174`）和 `answer_relevance_placeholder`（`:177-179`）都是 `raise NotImplementedError("... 尚未接入,见 README week 2 计划")`。**Faithfulness 和 Answer Relevance 项目中没有实现，不能硬套。**

不过线上侧有一个和 faithfulness 目标接近但机制不同的东西，值得区分清楚：`validate_answer_node`（`app/agent/rag_v2/nodes.py:334-469`）算 groundedness 和 coverage 两个分，阈值 0.75 / 0.60（`:33-34`）。它是**请求内的拦截器**而不是离线评测指标——不出报表，只决定这次要不要降级。

**项目证据**

- `tests/eval/metrics.py:69-72,75-81,84-87,114-119,122-161` — 已实现指标与聚合
- `tests/eval/run_eval.py:411-413` — 按 tag 输出
- `tests/eval/metrics.py:168-174,177-179` — Faithfulness / Answer Relevance：项目中没有
- `app/agent/rag_v2/nodes.py:33-34,334-469` — 线上拦截器（非离线指标）

#### Q5 · 有没有研究过行业通用/开源的评测基准？

**回答**

仓库里**没有接入任何开源 benchmark**，`tests/eval/` 下全部是自建金标集，这一点不掩饰。能指出的只有一处方向性痕迹：`tests/eval/metrics.py:168-174` 的注释写"TODO(week 2): 接入 ragas.metrics.faithfulness"，并写了实现思路——用 LLM 判断 `generated_answer` 里每个 claim 是否都能由 `retrieved_chunks` 支撑，返回支撑 claim 数除以总 claim 数。所以 ragas 是计划中的依赖，不是已用的依赖。

为什么当前是自建而不是直接上通用基准，理由能站得住：这个场景收敛在 oncall 运维（`aiops-docs/` 六篇 SOP），通用基准的语料和问法都不匹配，跑出来的分对"我这版改动是好是坏"没有指示作用。`tests/eval/run_eval.py:459` 的 `--compare A B` 才是当前真正依赖的判据——相对对比，不是绝对分数。

**项目证据**

- `tests/eval/metrics.py:168-174` — ragas 是 TODO，含实现思路
- `tests/eval/run_eval.py:459` — 当前依赖相对对比
- 开源 benchmark 接入 / 榜单复现：项目中没有，不能硬套

#### Q6 · 假设你入职后，我让你做"评测集生成 + 更新维护"的行业调研，你怎么开展工作？

**回答**

这是假设性工作规划题，仓库里没有对应实现，我按 `[假设]` 标注，但把每一步锚定到现有代码，说明改造点在哪。

**[假设] 第一步：先把现状的空缺量化。** 现在的空缺是明确的——生成侧两个指标是占位（`tests/eval/metrics.py:168-179`），金标集 18 + 83 条全部手工标注，没有自动生成，也没有更新机制。这三条就是调研要解决的问题清单。

**[假设] 第二步：生成方案选型的关键约束来自现有字段结构。** `golden_set.jsonl` 的 `expected_chunk_ids` / `expected_doc_ids` 双轨设计（配合 `tests/eval/metrics.py:90-107` 的降级逻辑）决定了：自动生成必须能产出 doc 级标注，chunk 级可选。否则每次调 `chunk_max_size`（`app/config.py`，当前 800）就得重标一遍，维护成本吃掉自动化收益。

**[假设] 第三步：更新机制挂在真实数据上。** 仓库已经有落库的问答 trace（`app/repositories/chat_run_trace_repository.py`），`used_documents` 会瘦身后入库（`app/core/used_documents.py:30-38` 的 `slim_used_documents`，且是幂等的）。这是线上转线下的现成入口，不用新建管道。

**[假设] 第四步：验收标准要定成"能区分好坏"而不是"分高"。** 判据是新评测集在 `--compare A B`（`tests/eval/run_eval.py:459`）下能否复现已知的好坏关系——比如纯向量 vs 混合检索（`--retriever` 开关，`:456`）应该分出差距。区分不出来的评测集，分再漂亮也没用。

**项目证据**

- `tests/eval/metrics.py:90-107,168-179` — 现状空缺与字段约束
- `tests/eval/golden_set.jsonl`、`golden_set_expanded.jsonl` — 手工标注的现状
- `app/repositories/chat_run_trace_repository.py`、`app/core/used_documents.py:30-38` — 线上转线下的现成入口
- `tests/eval/run_eval.py:456,459` — 验收手段
- 评测集自动生成 / 维护流程：项目中没有，以上为 `[假设]` 方案

#### Q7 · 追问：线上 log 是海量的，怎么转化成有限的线下评测集？

**回答**

仓库里已经有这条路的**前半段**，后半段是 `[假设]`，分清楚说。

**已有的部分：落库时就已经瘦身。** `app/core/used_documents.py` 的 `slim_used_documents`（`:30-38`）把每条 used_document 压成摘要格式，`EXCERPT_MAX_CHARS = 200`（`:23`），注释写明理由——"200 字符够认出「这是哪一段」，又不至于变成第二份全文"。这个函数是**幂等**的（`:52-53,81`），因为流式路径的 `used_documents` 来自 SSE payload 可能被重复加工，幂等省掉调用方"这个到底瘦没瘦"的判断。

瘦身的切点选在写库前而不是 `_serialize_docs`（`app/agent/rag_v2/service.py:175`），理由写在 `app/core/used_documents.py` 模块文档里：`_serialize_docs` 的产物同时喂给 SSE 事件、HTTP 响应体和 trace 落库，前两处是前端契约，动了就是破坏性变更，只有落库这一路该瘦身。

还有一处诚实处理值得提：`score` 字段如实存 `None`。因为 `EnsembleRetriever.invoke()` 不暴露 RRF 融合分数（`app/services/vector_search_service.py:129` 的注释），模块文档里明确写了"不用名次伪造一个分数——一个看起来像相关性分数、实际是排名倒数的值，比没有分数更危险"。这一点直接关系到用 log 建评测集：如果当初存了伪造分数，现在筛选样本时就会按假分数排序。

**[假设] 缺的后半段是采样策略。** 现在没有从 trace 中筛样本的代码。我会按三类筛而不是随机采：降级过的请求（`degrade_reasons` 非空，`app/agent/rag_v2/state.py:34`）、检索为空的请求（`RetrievalStatus.EMPTY`，`app/core/errors.py:65-78`）、被证据校验拦下的请求（`app/agent/rag_v2/nodes.py:334-469`）。这三类是系统已经自己标出来的"可疑样本"，信息量远高于随机采样中的成功请求。random sample 会得到一个和线上分布一致但几乎全是简单问题的集合。

去重也得做：线上同一个问题会被问很多次。仓库里已有的 SHA-256 去重模式（`app/workers/index_worker.py:62`）可以直接复用到问题文本上。

**项目证据**

- `app/core/used_documents.py:23,30-38,52-53,81` — 已实现的瘦身与幂等
- `app/core/used_documents.py` 模块文档、`app/agent/rag_v2/service.py:175` — 切点选择理由
- `app/services/vector_search_service.py:129` — score 存 None 而非伪造
- `app/repositories/chat_run_trace_repository.py:36` — 落库入口
- `app/agent/rag_v2/state.py:34`、`app/core/errors.py:65-78`、`app/agent/rag_v2/nodes.py:334-469` — 可疑样本的现成标记
- 采样策略 / 自动筛选代码：项目中没有，以上为 `[假设]` 方案

### 面经 03 · 2026-06-23

#### Q1 · 在实习中做的自动化怎么做的

**回答**

实习经历相关的代码不在这个仓库里，**项目中没有，不能硬套**。

不过如果把问题落到"这个仓库里的自动化是什么"，有两处实打实的：

一是 `Makefile` 的一键链路。`make init` 串起 Docker 起容器 → 等服务健康 → 上传文档，`Makefile:24-28` 的 `.PHONY` 列表里有 40 多个目标。里面有个细节值得说：`Makefile:12-15` 的 `RQ_QUEUES = knowledge_index protocol_pdf_ingest`，上面那两行注释记着为什么必须写两个队列名——"如果这里只监听 knowledge_index，PDF 任务会一直停留在 pending/queued"。这是踩过的坑，不是抄的模板。

二是 `.pre-commit-config.yaml` 的提交前拦截。`detect-private-key`（`:17`）比格式化重要得多：格式不对是审美问题，私钥进了 git 历史是安全事故，而且 git 历史里的东西删不干净。`mixed-line-ending --fix=lf`（`:18-19`）也是必要的，因为我的开发机是 Windows + WSL 混合环境。

这两样都算"自动化"，但都是本地开发流程自动化，不是"实习里做的业务自动化"。要是面试官问的是后者，我只能说没有。

**项目证据**

- `Makefile:7-15,24-28` — 一键链路与队列名的坑
- `.pre-commit-config.yaml:6-19` — 提交前拦截
- 实习期的自动化产出：项目中没有，不能硬套

#### Q2 · 手撕：合并两个有序数组

**回答**

算法题，和仓库代码无关。直接给解法。

关键点是**从后往前填**。如果从前往后，写 `nums1[0]` 会覆盖掉还没读的 `nums1` 元素，就得额外开数组;从后往前写，目标位置永远在两个已读指针的右侧，天然不会覆盖。

```python
def merge(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
    """把 nums2 的前 n 个元素并入 nums1 的前 m 个元素，原地完成。

    nums1 长度为 m + n，后 n 位是预留空位。
    """
    i, j, k = m - 1, n - 1, m + n - 1
    while j >= 0:                      # nums2 没走完才需要继续
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
```

循环条件写 `j >= 0` 而不是 `i >= 0 or j >= 0`：`nums2` 走完时，`nums1` 剩下的元素已经在正确位置上了，不用再动。这能省掉一轮无意义的自我赋值。

时间 O(m+n)，空间 O(1)。

**项目证据**

- 项目中没有对应实现，这是纯算法题

#### Q3 · 项目介绍：utbench 是如何去衡量模型生成效果的

**回答**

utbench 是原帖面试者自己的项目，**项目中没有，不能硬套**。我这边没有单测生成的 benchmark。

我有的是 RAG 检索评测 harness，衡量的东西完全不同——衡量"检索有没有把该找到的文档找回来"，不衡量"生成的代码能不能跑"。这两件事的评测方法论差别很大：检索有确定的 ground truth（哪个文档是对的），代码生成的正确性得靠执行来判定。

我的这套在 `tests/eval/`：金标集 `golden_set.jsonl` 18 条、`golden_set_expanded.jsonl` 83 条，每条形如

```json
{"id": "q-001", "question": "线上机器 CPU 一直九十多下不来,我先看哪里?",
 "expected_chunk_ids": [], "expected_doc_ids": ["cpu_high_usage"],
 "reference_answer": "...", "tags": ["aiops", "cpu", "easy", "colloquial"]}
```

指标在 `tests/eval/metrics.py`：`hit_at_k:69-72`、`recall_at_k:75-81`、`reciprocal_rank:84-87`。

需要如实说的是：生成质量的指标**没有实现**。`faithfulness_placeholder:168-174` 和 `answer_relevance_placeholder:177-179` 两个函数体都是 `raise NotImplementedError`，注释里写着 "TODO(week 2): 接入 ragas"。所以要是面试官问"你怎么衡量生成效果"，我的答案是目前只衡量检索，生成侧只有线上的证据校验（`app/agent/rag_v2/nodes.py:334-469`）在拦，没有离线打分。

**项目证据**

- `tests/eval/golden_set.jsonl`（18 条）、`tests/eval/golden_set_expanded.jsonl`（83 条）
- `tests/eval/metrics.py:69-72,75-81,84-87` — 已实现的检索指标
- `tests/eval/metrics.py:168-174,177-179` — 生成指标是 `NotImplementedError` 占位
- utbench / 单测生成评测：项目中没有，不能硬套

#### Q4 · 实习经历：SQL 多表查询和慢查询问题，如何去解决的

**回答**

实习期的慢查询治理**项目中没有，不能硬套**。这个仓库的数据库用得很浅，我不假装做过索引优化。

能说的是仓库里确实存在的部分。库是 PostgreSQL，走 psycopg 驱动（`app/config.py:55-56`，端口 55432）。索引是在模型上显式声明的，不是靠 ORM 猜：`app/models/chat_run_trace.py` 里 `request_id:30`、`source:37`、`is_bad_case:46`、`created_at:54` 都带 `index=True`；`app/models/chat_run_span.py` 的 `request_id:82`、`node:83` 同理。

这些索引不是随手加的，是按查询模式加的。`app/repositories/chat_run_trace_repository.py:75` 的查询是 `order_by(created_at.desc()).limit(limit)`——排障时按时间倒序翻最近的 trace，`created_at` 上没索引这条查询会全表扫加排序。`is_bad_case` 加索引是因为要单独筛坏样本。

至于"多表查询"：这个仓库里几乎没有 JOIN。我搜过，`app/repositories/` 下没有 `.join(`。数据访问是按聚合根分开读的，trace 和 span 是两次查询而不是一次 JOIN。这是个可以被质疑的设计，但现在的量级下 N+1 还没成为问题。

慢查询这块完全空白：没有配 `log_min_duration_statement`，没有 `EXPLAIN ANALYZE` 的分析记录，没有 pg_stat_statements。**这几样项目中没有。**

**项目证据**

- `app/config.py:55-56` — PostgreSQL + psycopg
- `app/models/chat_run_trace.py:30,37,46,54`、`app/models/chat_run_span.py:82,83` — 显式索引
- `app/repositories/chat_run_trace_repository.py:75` — 索引对应的查询模式
- 多表 JOIN：`app/repositories/` 下不存在
- 慢查询日志 / EXPLAIN 分析 / pg_stat_statements：项目中没有，不能硬套

#### Q5 · agent 是怎么调用工具的

**回答**

这题看起来简单，但真正答透要说清一件事：**大模型本身没有执行任何代码的能力。** 它只能输出文本。所谓「调用工具」，全程是一场约定好格式的对话，真正执行代码的是你写的那段循环。

我先把完整机制拆开讲，再说我项目里的实现。

---

**一、Function Calling 的本质是「结构化输出 + 外部执行」**

从 OpenAI 2023 年 6 月引入 function calling 开始，这套机制的形状就固定了。一次完整的工具调用是**两轮 HTTP 请求**：

**第一轮**，你把工具清单和用户问题一起发给模型：

```json
{
  "messages": [
    {"role": "user", "content": "北京现在几点"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_time",
        "description": "获取当前时间。当用户询问现在几点、今天日期、星期几时使用",
        "parameters": {
          "type": "object",
          "properties": {
            "timezone": {"type": "string", "description": "时区，如 Asia/Shanghai"}
          },
          "required": []
        }
      }
    }
  ]
}
```

模型不返回自然语言，而是返回一个**结构化的调用意图**：

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_current_time",
        "arguments": "{\"timezone\": \"Asia/Shanghai\"}"
      }
    }
  ]
}
```

注意 `arguments` 是**字符串**不是对象 —— 这是 OpenAI 协议的历史设计，你得自己 `json.loads`，而且模型偶尔会输出不合法的 JSON，所以这里必须有容错。

**你的代码**拿到这个意图，查工具表找到 `get_current_time` 这个函数，用解析出的参数调它，拿到返回值。模型完全不参与这一步。

**第二轮**，你把执行结果作为一条新消息追加回去：

```json
{
  "messages": [
    {"role": "user", "content": "北京现在几点"},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_abc123", "content": "2026-09-08 15:42:31 CST"}
  ],
  "tools": [...]
}
```

`tool_call_id` 必须和第一轮的 `id` 对上 —— 因为模型可能一次要求调用多个工具（parallel tool calls），结果回传时要能对应。模型这次看到了结果，才生成给用户的自然语言回答。

**关键推论：模型看到的「工具」就是那段 JSON Schema。** 它不知道函数体长什么样，不知道背后是 HTTP 请求还是本地计算。所以 `description` 字段不是文档，是**给模型的路由信号** —— 模型靠它判断「什么时候该想起这个工具」。这也是为什么工具描述该写「当用户询问 X 时使用」而不是「本函数返回一个时间戳」。

---

**二、模型是怎么学会输出这种格式的**

这不是 prompt 工程能做到的，是**训练出来的能力**。模型在后训练阶段见过大量「工具清单 + 用户问题 → tool_calls」的样本，学会了在合适的时候切换到这种输出模式。

具体实现上，主流做法是给模型加特殊 token。比如某些模型的模板里，工具调用被包在 `<tool_call>` 和 `</tool_call>` 之间；推理框架检测到这对 token，就把中间内容解析成结构化的 `tool_calls` 字段返回。所以「模型返回 JSON」这个说法有点简化了 —— 模型返回的仍然是 token 序列，只是推理层帮你解析了。

这解释了两件工程上会遇到的事：

- **同一段代码换个模型可能就不工作了。** 没经过 tool calling 训练的模型（或者训练得不好的）会直接用自然语言回答「我需要调用 get_current_time 工具」，而不是输出结构化字段。
- **约束解码（constrained decoding）能让它更可靠。** vLLM、llama.cpp 这些框架支持在生成时用 JSON Schema 做语法约束，强制每一步只能采样出符合 schema 的 token。这样不合法 JSON 在物理上无法产生。这是比「在 prompt 里求它输出合法 JSON」更根本的解法。

---

**三、Agent 循环 = 把这两轮变成 while**

单次工具调用是两轮。Agent 的能力来自**把这个过程放进循环**：

```python
messages = [{"role": "user", "content": question}]
while True:
    response = llm.invoke(messages, tools=tool_schemas)
    messages.append(response)

    if not response.tool_calls:
        return response.content        # 模型不再要工具，说明它准备回答了

    for call in response.tool_calls:
        result = execute(call.name, json.loads(call.arguments))
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": str(result),
        })
    # 循环回去，模型看到结果决定下一步
```

这二十行就是 ReAct 循环的骨架，也是 `AgentExecutor` / `create_agent` 内部在做的事。退出条件只有一个：**模型这一轮没要求调工具**。

而这个退出条件是不可靠的 —— 模型可能一直要求调工具，所以生产实现必须额外加：步数上限、总时间预算、状态指纹去重（同样的工具+参数重复出现就是循环了）。这三样缺一个都可能出现「一个请求跑十分钟」。

---

**四、我项目里的实现**

我这边工具装配在 `app/services/rag_agent_service.py:115-139`，来源是两处合并：

- **本地工具**在 `__init__` 里装好 —— `app/tools/knowledge_tool.py:13-43` 的 `retrieve_knowledge`、`app/tools/time_tool.py` 的 `get_current_time`。`@tool` 装饰器负责从函数签名和 docstring 自动生成上面那份 JSON Schema，我不手写。
- **MCP 工具**是运行时拉的（`:122-123`，`await mcp_client.get_tools()`），跨进程 HTTP 取回对方的工具清单。

两者在 `:131` 合并成 `all_tools`，交给 `create_agent(self.model, tools=all_tools, checkpointer=self.checkpointer)`（`:133-137`）。上面那个 while 循环由框架实现，我没自己写。

两个设计选择值得说：

**一、system prompt 里不列举工具清单**（`:146-155` 的注释）。因为工具 schema 已经通过 `tools` 参数单独传了，prompt 里再抄一遍就是**两份真相**。加了工具忘记改 prompt 时，模型会同时看到一份过期清单和一份正确 schema，行为不可预测。所以 `:158-176` 的 prompt 只写工作原则不写工具名。

**二、MCP 加载失败降级为仅本地工具**（`:118-127`）。整段包 try/except，失败时 `self.mcp_tools = []` 并打日志。判据是 MCP 提供的是运维增强能力（查日志、查监控），而知识库问答不依赖它 —— 让一个可选依赖挂掉整个聊天服务不划算。

还有一处和 Q5 直接相关的：**工具内部失败要 `return` 不要 `raise`。** `app/tools/knowledge_tool.py:41-43` 捕获异常后返回错误文本。原因看上面那个循环就明白了 —— 工具抛异常会让 `execute()` 崩掉，整个 while 循环终止，用户什么都拿不到。而返回一段错误文本，模型下一轮会看到「检索失败了」，它可以换个查法、或者告诉用户暂时查不到。**判据是「谁能处理这个失败」**：模型能处理就 return 给它，只有 HTTP 层要靠状态码说真话时才 raise。

**项目证据**

- `app/services/rag_agent_service.py:115-139` — 工具装配与 `create_agent`
- `app/services/rag_agent_service.py:118-127` — MCP 加载失败降级
- `app/services/rag_agent_service.py:146-176` — prompt 不列举工具的理由
- `app/tools/knowledge_tool.py:13-43` — `@tool` 生成 schema；`:41-43` return 而非 raise
- `app/tools/time_tool.py:1-32` — 描述写触发条件
- `app/agent/mcp_client.py:130-158` — 带重试的 MCP 客户端
- 约束解码 / 步数上限 / 状态指纹去重：项目中没有，属于该补的部分

#### Q6 · agent 还有其他调用方式吗

**回答**

有，而且不止一种。这题我按技术演进顺序讲五种，因为它们不是并列的备选项，而是一条"格式越来越可靠、表达能力越来越强"的演进线。理解了这条线，就知道现在为什么默认用 Function Calling，以及什么场景该跳出它。

**方式一：纯 Prompt + 文本解析（ReAct 时代，2022-2023）**

在 Function Calling 出现之前，唯一的办法是把工具清单写进 system prompt，再规定一套输出格式，让模型按格式生成文本，应用层用字符串解析把它抠出来。ReAct 论文里那套格式就是典型：

```
你可以使用以下工具：
- search(query: str): 搜索网页
- calculator(expr: str): 计算数学表达式

请按以下格式回复：
Thought: <你的思考>
Action: <工具名>
Action Input: <参数>
```

模型输出 `Action: search` / `Action Input: 2026年GDP`，应用层正则匹配 `Action:` 后面那一段，查表找到函数，把参数传进去，执行完再把结果拼成 `Observation: ...` 追加到 prompt 里，重新调一次模型。这就是 LangChain 早期 `AgentExecutor` 里 `output_parser` 的活。

这套方式的问题非常具体，不是"不优雅"而是"经常坏"：

- **格式脆弱。** 模型会写成 `Action: search(query="xxx")`，会漏掉 `Action Input:`，会在 `Thought` 里提前把答案说了导致解析器抓错位置。LangChain 里那个 `OutputParserException` 就是为这个存在的，还得配一个"解析失败就把错误信息喂回去让模型重试"的兜底。
- **参数只能是字符串。** 格式里没有类型信息，`Action Input: 5` 到底是数字 5 还是字符串 "5"，要靠应用层猜。嵌套对象基本没法表达。
- **无法并行。** 一次输出只能表达一个 Action，要调三个工具就得来回三轮，延迟是三倍。
- **prompt 成本高。** 工具清单每一轮都要重复送一遍，工具多了光清单就吃掉几千 token。

**方式二：Function Calling / Tool Use（当前主流，2023-06 之后）**

OpenAI 在 2023 年 6 月给 API 加了 `functions` 参数（后来改名 `tools`），这件事的关键变化不是"语法更好看"，而是**工具描述从 prompt 里挪到了请求的独立字段**，并且模型在训练阶段就针对这个字段做了对齐。

请求长这样（这是 Q5 讲过的，这里强调它和方式一的结构差别）：

```json
{
  "messages": [{"role": "user", "content": "查一下昨天的错误日志"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "search_logs",
      "description": "查询指定时间范围的日志",
      "parameters": {"type": "object", "properties": {...}, "required": [...]}
    }
  }]
}
```

响应里 `tool_calls` 是结构化字段，不需要解析文本。带来的三个实质改进：

- **类型信息完整。** JSON Schema 能表达 integer / enum / 嵌套 object / 数组，参数类型是协议保证的。
- **可以并行。** `tool_calls` 是数组，模型一次可以返回多个调用，应用层并发执行完把多条 `ToolMessage` 一起送回。这是 2023-11 之后 OpenAI 和 Anthropic 都支持的能力，对"同时查 CPU 和内存"这种场景延迟能砍一半。
- **格式由解码层兜底。** 主流厂商在生成 arguments 时会做约束解码，保证吐出来的至少是合法 JSON，不会出现半个括号。

需要注意的是：**它保证格式合法，不保证语义正确。** 模型完全可以给你一个 JSON 完全合法但业务上不存在的 `user_id`。这是 Q5 和后面安全题反复提到的那条线。

**方式三：结构化输出 / 约束解码（Structured Output）**

这个和方式二容易混，区别在意图：Function Calling 是"从多个工具里挑一个并给参数"，Structured Output 是"我只要你按这个 schema 吐一个对象，没有挑选动作"。

技术手段是 grammar-constrained decoding —— 在每一步解码时，根据 schema 算出"下一个 token 合法集合"，把不合法的 token 的 logits 直接压到负无穷。所以它不是"生成完再校验"，而是**从物理上让模型没法生成非法格式**。OpenAI 的 `response_format: {"type": "json_schema", "strict": true}`、vLLM 的 outlines/xgrammar 后端都是这个原理。

适用场景是"动作唯一但参数复杂"：意图分类要吐 `{intent, confidence, entities[]}`、信息抽取要吐一个固定结构的表单、评分要吐 `{score, reason}`。用 Function Calling 干这个也能跑，但语义上是错位的 —— 你并不是在调用工具。

这也是 LLM-as-Judge 的落地方式：让评委模型按固定 schema 吐分数和理由，而不是让它自由写作文再解析。

**方式四：Code-as-Action（CodeAct）**

让模型直接写代码，工具是代码里可调用的函数，应用层把代码丢进沙箱执行，把 stdout 和异常回传。HuggingFace 的 smolagents 和 CodeAct 论文走的都是这条路。

```python
# 模型的输出本身就是这段代码
cpu = get_cpu_usage(host="web-01", minutes=30)
if max(cpu) > 90:
    logs = search_logs(host="web-01", keyword="OOM")
    print(logs[:5])
else:
    print("CPU 正常，无需查日志")
```

它的优势非常实在：**一次输出就能表达循环、条件、中间变量和组合调用**。同样的任务用 Function Calling 得来回四五轮（查 CPU → 看结果 → 决定要不要查日志 → 查日志 → 汇总），CodeAct 一轮就完事。CodeAct 论文实测在多步任务上成功率更高、平均轮次更少，直觉上也说得通 —— 代码本来就是人类表达"多步带条件的动作序列"最自然的语言，而模型在代码上的训练数据密度远高于任何自定义 DSL。

代价是安全模型完全变了。Function Calling 的攻击面是"参数值不对"，可以用 schema + 白名单挡住；CodeAct 的攻击面是"任意代码执行"，必须上真沙箱（gVisor / Firecracker / 至少是受限的子进程 + seccomp），还要限制 import、限制网络、限制 CPU 时间。这个基建成本不低，所以生产系统里用得比 Function Calling 少。

**方式五：编排层直接调用（Workflow，模型不参与决策）**

代码写死"什么时候调什么"，模型只负责它擅长的那部分（改写、生成、判断），完全不碰控制流。这不是"低级方案"，恰恰是收敛场景下更正确的方案 —— 你已经知道流程该怎么走，就没有理由把它交给一个有概率犯错的组件。

**我这边落了哪几种**

方式二和方式五同时存在，可以直接对比，这也是我理解这条演进线的实际来源：

- **方式二（模型决定）**：`app/services/rag_agent_service.py:133-137` 的 `create_agent`。控制权在模型手里，它可以调 0 次也可以调 5 次，顺序也由它定。
- **方式五（编排层决定）**：`app/agent/rag_v2/graph.py:43-48` 把五个节点的连边写死 —— `START → rewrite → retrieve_each → dedup → generate → validate_answer → END`。检索一定会发生、一定在生成之前、一定只发生一次。模型在这条链路里被调三次（改写、生成、校验），但**从不决定要不要检索**。

中间还有个半自主形态值得说：`retrieve_each` 是被 `Send` fan-out 的（`app/agent/rag_v2/graph.py:17-19`），并行分支数由 `rewrite` 节点吐出的子查询个数决定。所以"调几次"是模型输出决定的，"调什么工具"仍然写死。这在设计上是有意的 —— 把模型的自由度限制在"数量"这个安全维度上，不让它碰"调用哪个"这个危险维度。

方式一我只在读 LangChain 早期源码时见过，没有在生产代码里手写过文本解析。方式三和方式四**项目里没有**：`validate_answer` 那个校验节点是让模型吐 JSON 然后自己 `json.loads`（`app/agent/rag_v2/nodes.py:810-814` 有解析失败的兜底），本质上是方式一的退化版而不是真正的约束解码 —— 如果重做，这里应该换成 structured output，能直接省掉那段 fallback。CodeAct 完全没有，也没有沙箱基建。

**怎么选**

判断标准就一句：**问题收敛就交给编排层，不收敛才交给模型。**

RAG 问答是收敛的 —— 用户问知识库里的东西，必须检索，让模型"判断"要不要检索只是白送一个它偶尔判断错的机会，收益为零。运维排障是不收敛的 —— 可能要查日志、可能要查 CPU、可能两个都要、可能查完日志才知道要查 CPU，写死顺序一定会漏掉真实场景，所以那条走 Function Calling。

多步且步骤间有明显的条件依赖、工具调用次数动辄十几次的场景（比如数据分析、自动化运维脚本），才值得考虑 CodeAct 那套沙箱成本。

**项目证据**

- `app/services/rag_agent_service.py:133-137` — 方式二，模型自主决定
- `app/agent/rag_v2/graph.py:43-48` — 方式五，显式连边，检索不可跳过
- `app/agent/rag_v2/graph.py:17-19` — `Send` fan-out，分支数由上一步决定，工具仍写死
- `app/agent/rag_v2/nodes.py:91,264,334` — 三个 LLM 调用点，均不做工具决策
- `app/agent/rag_v2/nodes.py:810-814` — 手写 JSON 解析的失败兜底，正是缺少约束解码的证据
- 结构化输出 / 约束解码、CodeAct 与沙箱：项目中没有

#### Q7 · agent 怎么去判断是否需要调用工具，MCP 协议解决了什么问题

**回答**

这是两个独立的问题串在一起问的，容易答混。前一个问"决策发生在哪、由谁做"，后一个问"协议解决了什么工程问题"。分开答。

##### 一、"要不要调工具"这个判断到底发生在哪

先纠正一个很常见的误解：**没有一个独立的分类器在做这个判断。**

很多人以为流程是「先跑一个二分类判断需不需要工具 → 需要就走工具分支 → 不需要就直接生成」。不是。模型只做了一次前向传播，在这一次里同时完成了"判断"和"生成"。

具体机制：请求发出去时，工具的 JSON Schema 是作为一个独立字段 `tools` 传给 API 的（不是拼进 system prompt）。模型解码时，下一个 token 有两种走向——继续生成自然语言，或者生成工具调用结构。这两条路在概率空间里是竞争关系。所谓"判断要不要调工具"，本质上就是**这两类 token 谁的概率更高**。

所以它没有阈值可调，也没有中间置信度可以拿出来看（多数 API 不返回 logprobs，返回了也难以映射成"调工具的置信度"）。这点在面试里说清楚很占优势，因为它直接决定了后面所有可控性讨论的边界。

**模型是怎么获得这个能力的。** 靠训练。tool-use 数据在 SFT 阶段被大量喂进去，格式大致是「问题 + 工具列表 → 期望的工具调用」和「问题 + 工具列表 → 不该调工具，直接回答」两类样本混合。后一类同样重要，是教它**什么时候该忍住**。缺了后一类，模型会变成"给了工具就想用"，这是早期开源模型很典型的毛病。

**三个真正能影响判断的杠杆。**

第一个是**工具描述的质量**，也是唯一在应用层完全可控、且效果最大的。模型认识这个工具的全部信息就是名字、描述、参数说明。描述写"查询数据"，模型不知道查什么、什么场景该用；写"根据自然语言问题检索企业内部知识库（产品手册、运维文档、协议规范），返回最相关的文档片段。当用户询问公司内部资料、业务规范或历史文档时使用"，模型才有判断依据。

描述里要写清三件事：**干什么**、**什么时候该用**、**什么时候不该用**。第三件最容易漏。工具一多之后，模型的主要错误不是"该调不调"，而是"调错了一个功能相近的"，把边界写进描述是最直接的解法。

第二个是 **`tool_choice` 参数**。这是 API 级别的开关，能把决策权从模型手里拿回来：

- `auto`（默认）—— 模型自己决定，可以不调
- `none` —— 本轮禁止调工具，强制走文本生成
- `required` / `any` —— 本轮必须调至少一个，模型只能选调哪个
- 指定具体工具名 —— 必须调这一个

工程上很有用。比如做纯 RAG 问答接口，检索必须发生，那就第一轮用 `tool_choice` 指定检索工具、之后切回 `auto`，既保证检索一定发生，又保留后续多轮工具调用的空间。很多人不知道这个参数存在，遇到"模型偶尔不检索"就去改 prompt，改到最后还是不稳定。

第三个是**动态裁剪工具集**。模型判断准确率随工具数量下降，超过二三十个之后错误率明显上升（描述占的 token 也线性膨胀）。做法是调模型前先加一层轻量筛选——规则匹配，或者用工具描述的 embedding 和用户问题做相似度检索，只把 top-k 塞进 `tools`。这层可以做得很糙，因为它只需保证"该有的在候选集里"，不需要精确到选中哪一个。

**判断失败的三种典型形态和各自修法。**

| 现象 | 根因 | 修法 |
|---|---|---|
| 该调不调（模型硬答，答出幻觉） | 描述太抽象；或系统提示里有"尽量简洁作答"之类反向引导 | 补描述的触发条件；关键路径用 `tool_choice` 强制 |
| 不该调乱调（简单问候也去查库） | 描述太宽泛；缺"不该用"的边界；训练侧缺负样本 | 描述里写明排除场景；system prompt 补"闲聊和常识问题直接回答" |
| 调错工具（功能相近的选混） | 多个描述语义重叠 | 描述里互相指认边界；或合并成一个工具用参数区分 |

**再补一层：应用层要不要自己做判断。** 可以，成熟系统基本都做，通常是前置一个意图分类——模型或规则先把请求分成「知识问答 / 数据查询 / 闲聊 / 操作类」，不同意图走不同工具集和不同编排。好处是决策可解释、可测试、可埋点；坏处是意图体系一旦定死，边界外的请求没地方去，维护成本会慢慢上来。

我这边**没有做**这层。`rag_agent_service` 那条完全交给模型判断；`rag_v2` 那条根本不判断——检索写死在图里无条件执行（`app/agent/rag_v2/graph.py:44-45`），相当于用编排把这个判断消掉了。唯一影响模型判断的手段就是 docstring：`app/tools/knowledge_tool.py:13-43`。**意图分类器、工具路由层、工具选择的置信度阈值，这三样项目中没有，不能硬套。**

##### 二、MCP 解决了什么问题

**核心矛盾是 M×N 的集成爆炸。**

MCP 之前，"让 AI 应用用上外部能力"是点对点做的。假设有 M 个 AI 应用（Claude Desktop、Cursor、你自己的 Agent、某个客服系统）和 N 个外部系统（GitHub、数据库、内部 CRM、日志平台），每个应用要用上每个系统就得写一份对接——总共 M×N 份。而这 M×N 份里绝大部分逻辑是重复的：怎么描述能力、怎么传参、怎么回结果。

MCP 在中间插一层标准协议：外部系统实现一次 MCP Server，AI 应用实现一次 MCP Client，集成量从 M×N 降到 M+N。这个思路不新鲜——LSP（Language Server Protocol）用同样的办法解决了「M 个编辑器 × N 种语言」，MCP 官方文档直接类比过 LSP，也常被说成"AI 领域的 USB-C"。

**协议的三个原语。** 这是容易只答出一个的地方。MCP Server 可以暴露三类东西，语义完全不同：

- **Tools（工具）**—— 模型可以主动调用的函数，有副作用，控制权在模型（model-controlled）。发邮件、查数据库、重启服务。
- **Resources（资源）**—— 可被读取的数据，用 URI 标识，控制权在应用（application-controlled）。文件内容、数据库表结构、某个 API 的返回快照。它不是"调用"而是"读取"，由宿主应用决定要不要塞进上下文。
- **Prompts（提示模板）**—— 预置的提示词模板，控制权在用户（user-controlled）。典型场景是 Server 端提供"帮我 review 这段代码"这类模板，用户在 UI 上主动选。

区分它们的意义在于**谁来决定这个东西进不进上下文**：Tools 是模型决定，Resources 是应用决定，Prompts 是用户决定。混淆这三层会导致设计问题——比如把一份大文档做成 Tool，模型就得"决定"去读它，而它本该是 Resource，由应用直接注入。

**传输层。** 两类：

- **stdio** —— Server 作为子进程被拉起，通过标准输入输出通信。适合本地工具，Claude Desktop 那些配置就是这种。零网络开销，进程生命周期跟着宿主。
- **HTTP 系** —— 早期是 HTTP + SSE 双端点（一个 POST 发请求、一个 SSE 收响应），后来收敛成 **Streamable HTTP** 单端点，兼容无状态部署。适合远程部署、多客户端共享。

我这边用 streamable-http，两个 Server 各自独立起进程监听端口（`mcp_servers/cls_server.py:470`、`mcp_servers/monitor_server.py:435`，8003 和 8004）。选它而不选 stdio 的原因很实际：这两个 Server 依赖腾讯云 SDK，跟主应用的依赖树不干净，独立进程能把冲突隔开。

**能力协商。** 连接建立时有一次 `initialize` 握手，双方交换协议版本和各自支持的能力（Server 说我有 tools 和 resources 但没有 prompts，Client 说我支持 sampling），之后才开始正常通信。意义是让协议能向前演进——新增能力时老客户端不会因为看不懂而崩掉。

**MCP 和 Function Calling 是什么关系。** 这个追问频率很高，且很容易答成"两个竞品"。它们不在一层：

- **Function Calling 是模型和应用之间的接口** —— 解决"模型怎么表达它想调一个工具"，产出是 `tool_calls` 结构。
- **MCP 是应用和工具提供方之间的接口** —— 解决"工具怎么被发现、schema 怎么传递、调用怎么路由"。

一次完整调用是两者串起来的：MCP Client 从 Server 拉到工具 schema → 应用把 schema 放进 Function Calling 的 `tools` 字段 → 模型返回 `tool_calls` → 应用把它翻译成 MCP 的 `tools/call` 请求 → Server 执行 → 结果包成 `ToolMessage` 回去。**没有 MCP 也能做 Function Calling**（工具直接写成本地函数就行，这就是 `knowledge_tool.py` 的做法）；**没有 Function Calling 的模型也能用 MCP Server**（应用层自己决定调什么）。

**落到我的代码上，MCP 带来三个具体收益。**

第一，**故障隔离**。两个 MCP Server 是独立进程，它们挂了主应用不挂——降级路径是 `app/services/rag_agent_service.py:124-127` 那段 try/except，MCP 工具拉不到就只带本地工具继续跑。如果这些工具是 in-process 函数，腾讯云 SDK 的一个依赖冲突就足以让整个 API 起不来。这个收益严格说是进程边界给的，不是协议给的，但协议让进程边界变得廉价。

第二，**一份 schema 服务多个客户端**。工具用 `@mcp.tool()` 声明一次（`cls_server.py:104,136,169,212,346`、`monitor_server.py:124,277`，共 7 个），schema 由 FastMCP 从函数签名和 docstring 自动生成。谁来连都是这一份，不存在"Agent 里的 schema 和文档里的 schema 不一致"这种漂移。

第三，**配置化的服务发现**。`app/agent/mcp_client.py:81` 的 `DEFAULT_MCP_SERVERS = config.mcp_servers`，地址在配置里（`mcp_cls_url`、`mcp_monitor_url`）。换环境改配置不改代码。

**MCP 明确不解决的事**（主动说出来比被追问出来强）：

- **认证授权** —— 规范里有针对 HTTP 传输的 OAuth 2.1 授权框架，但"哪个用户能调哪个工具"这种业务级权限协议不管。我这边两个 Server 直接裸奔在 `127.0.0.1`，零鉴权，靠网络边界兜着，**这是已知欠账**。
- **调用配额和限流** —— 协议层没有，得自己在 Client 侧做。
- **参数的语义校验** —— JSON Schema 只能保证类型对，保证不了"这个 IP 是你有权查的 IP"。
- **工具版本管理** —— Server 改了参数含义，Client 感知不到。
- **供应链安全** —— 这是 MCP 生态目前最现实的风险。装一个第三方 Server 等于在自己机器上跑别人的代码，而且它的工具描述会直接进模型上下文，构成一条提示注入路径。

**项目证据**

- `app/tools/knowledge_tool.py:13-43` — docstring 即工具描述，是唯一影响模型判断的手段
- `app/agent/rag_v2/graph.py:44-45` — 用编排消掉"要不要检索"的判断
- `mcp_servers/cls_server.py:104,136,169,212,346` + `:470`、`mcp_servers/monitor_server.py:124,277` + `:435` — 7 个工具，两个独立进程，streamable-http
- `app/agent/mcp_client.py:81`、`app/config.py`（`mcp_servers`）— 配置化服务发现
- `app/services/rag_agent_service.py:124-127` — 进程隔离带来的降级能力
- 意图分类 / 工具路由 / 工具权限 / 调用配额 / MCP 鉴权：项目中没有，不能硬套

#### Q8 · agent 的上下文窗口满了怎么办，如何去进行压缩，有哪几种方式

**回答**

这题我分四步答：窗口为什么会满、满了会发生什么、有哪四类手段、每类的代价是什么。最后拿我项目里的两套实现当例子。

---

**第一步：先说清楚"上下文窗口"到底是什么**

上下文窗口是模型单次前向计算能接受的 token 总量上限。注意是**总量**，输入加输出一起算。比如一个 128K 窗口的模型，你塞了 120K 输入，那它最多只能再生成 8K 输出。很多人以为 128K 是"输入上限"，实际上输出也占同一个池子，这个误解在生产里会变成"为什么模型答到一半就断了"。

这个上限不是软性的。它由位置编码的设计范围和训练时见过的最长序列决定，超了不是"效果变差"而是**直接报错**——OpenAI 返回 `context_length_exceeded`，其他家类似。所以这不是优化题，是可用性题：处理不好，请求会失败。

**第二步：Agent 为什么特别容易撑满**

单轮问答很难撑满 128K，但 Agent 会。原因是 Agent 的上下文是**单调累积**的：

```
第 1 轮：system + user                          →   500 token
第 2 轮：system + user + assistant + user       →  1200 token
第 3 轮：... + tool_call + tool_result + ...    →  4000 token   ← 工具返回是大头
第 4 轮：... + 另一个 tool_result + ...          → 12000 token
...
第 12 轮                                        → 130000 token  ← 炸了
```

三个放大器：

1. **工具返回不可控。** 我这边最典型的是日志查询。`mcp_servers/cls_server.py` 里的检索工具，一次查询返回几百条日志是常态，一条日志一两百字，一次调用就是几万 token。用户看不到这些原始返回，但它们全在上下文里。
2. **ReAct 的每一步都留痕。** 思考、工具调用、工具结果，三段都要保留在历史里——因为下一步的推理依赖前面的结果。跑 10 步的任务，上下文里就有 10 组这样的三段。
3. **多轮对话叠加多轮工具调用。** 单轮任务 12000 token，用户接着问第二个问题，前一轮的所有工具痕迹还在。

所以 Agent 的上下文管理不是"锦上添花"，是**必须做的基础设施**。这也是它和普通聊天机器人在工程上最大的差别之一。

**第三步:满了会发生什么——三种失败模式，别只想到报错**

- **硬失败**：请求直接被 API 拒绝，`context_length_exceeded`。这个最好处理，因为它明显。
- **软失败（更危险）**：没超限，但有效信息被淹没。这就是 Lost in the Middle 现象——模型对上下文首尾的注意力显著高于中间，塞进去的东西越多，中间那段的实际影响力越低。表现是"我明明把答案放进去了，模型却说不知道"。
- **成本失败**：输入 token 是要计费的，而且注意力的计算是 O(n²)。上下文从 4K 涨到 40K，成本涨 10 倍，延迟涨得更多（首 token 延迟主要由 prefill 阶段决定，prefill 就是在算这个 n²）。一个不做上下文管理的 Agent，跑到第 10 轮时单次调用的钱可能是第 1 轮的 30 倍。

**第四步：四类手段（这是答题的骨架）**

业界的做法归纳起来就四类，我按"信息损失程度"从大到小排：

| 类别 | 做法 | 信息损失 | 计算成本 | 适用场景 |
|---|---|---|---|---|
| **Trim（截断）** | 直接丢掉一部分消息 | 大，且不可恢复 | 几乎为零 | 老历史确实不重要；或作为兜底 |
| **Summarize（摘要）** | 用模型把旧历史压成一段短文 | 中，有损但保留要点 | 一次额外 LLM 调用 | 多轮对话，需要记住"之前谈了什么" |
| **Retrieve（检索式召回）** | 历史全存起来，每轮只按当前问题取回相关片段 | 小，但可能取错 | 一次检索（毫秒级） | 历史很长且话题跳跃 |
| **Externalize（外化）** | 大对象存外部，上下文里只放句柄+摘要，模型需要时用工具去读 | 几乎无损 | 需要额外的工具和存储 | 大文件、长日志、代码库 |

逐个说清楚——

**Trim。** 最简单的一类，但里面有两个容易答错的细节。

第一个细节：**按什么单位截**。按条数截是错的，或者说是不完备的。一条消息可能 5 个字也可能 5000 字，"保留最近 7 条"完全挡不住一条超长工具返回。正确做法是按 token 预算截，从最新往回累加，加到快超预算就停。

第二个细节：**哪些消息不能截**。至少三类：
- system message 必须留。它定义了模型的行为边界和输出格式，丢了模型的行为直接变样。
- 不能切开 `tool_call` / `tool_result` 配对。OpenAI 的 API 会直接报错——一个 assistant 消息声明了 `tool_calls`，后面必须紧跟对应 id 的 `tool` 消息。只留一半是非法请求。
- 尽量不切开 user/assistant 配对。只剩 assistant 回复而没有对应问题，模型看到的是一段没头没尾的话，容易误判语境。

LangChain 的 `trim_messages` 就是干这个的，它支持 `strategy="last"`、`token_counter`、`include_system=True`、`allow_partial=False`，这几个参数正好对应上面的细节。

**Summarize。** 核心是"用一次 LLM 调用换一大段 token"。有三个工程要点，前两个是我踩过的：

要点一：**滚动摘要，不要每轮全量重算**。假设第 11 轮时前 10 轮要压缩，做法应该是 `新摘要 = LLM(旧摘要 + 第 10 轮新增)`，而不是 `新摘要 = LLM(第 1 到 10 轮全文)`。后者的成本随轮次线性增长，跑到 50 轮时单次摘要调用比原始请求还贵。代价是误差会累积——摘要的摘要的摘要，细节会一层层丢。所以通常配一个"每 N 轮从原始历史重算一次"来纠偏。

要点二：**摘要必须有固定结构，不能让模型自由发挥**。这个坑很实在：模型天然倾向于总结"发生了什么"（叙事），而会漏掉"不能做什么"（约束）。用户在第 2 轮说过"我们用的是 PostgreSQL 不是 MySQL"，自由格式摘要十次里有三次会漏掉，然后第 12 轮模型就开始给 MySQL 的语法。解法是在 prompt 里强制固定字段，缺失项也要显式写"无"。

要点三：**摘要里的时效信息要降级**。历史里的"CPU 使用率 92%"是三轮前的观测值，摘要如果写成"CPU 使用率 92%"，模型会当成当前状态。必须写成"曾观测到 CPU 92%（历史值）"。这一条不写进 prompt，摘要就是个定时炸弹。

**Retrieve。** 把对话历史当成一个小型知识库，每轮用当前问题去检索最相关的 k 条历史片段。

好处是它天然处理话题跳跃——用户在第 30 轮突然回头问第 3 轮的事，摘要方案早就把细节压掉了，检索方案能精确捞回来。

代价有两个，都要说：一是**检索可能取错**，取错的后果比摘要漏掉更糟，因为它给了模型一段看似相关实则无关的历史；二是**破坏了时序**，取回来的是散落的片段，模型看不到"先后顺序"，对"我改主意了"这类修正意图的识别会变差。

**Externalize。** 这是四类里最优雅的，也是 Claude Code / Cursor 这类编程 Agent 的主要手段。

思路是：大对象不进上下文。一个 5000 行的文件，不塞全文，只在上下文里放"文件路径 + 结构摘要 + 可以用 read_file(path, offset, limit) 去读"。模型需要哪一段就自己去取哪一段。

它把"上下文管理"从被动压缩变成了**模型主动的信息获取**。信息几乎无损，因为原文一直在，只是不常驻。代价是要设计一整套工具（读、搜、列目录），而且模型可能"忘了去读"——它得先意识到自己需要那段信息。

**四类之外还有一个正交的优化：KV cache 友好性。**

这个点在面试里加分，因为它是"压缩"这件事的反直觉一面。主流推理服务对**前缀完全相同**的请求会复用 KV cache，命中的部分不用重新 prefill，首 token 延迟和成本都大幅下降（各家的折扣不同，量级上是显著的）。

关键在"前缀完全相同"。这意味着：

- 你在上下文**开头**做任何改动，后面全部缓存失效。
- 所以压缩应该**优先动尾部或中段，保持头部稳定**。
- 所以 system prompt 里别塞当前时间戳这种每次都变的东西——一个 `当前时间：2026-09-08 14:23:07` 就能让整个前缀的缓存 100% 失效。要放就放在靠后的位置。
- 所以摘要一旦更新，从摘要往后的缓存全掉。这是"频繁重算摘要"的隐性成本，比 LLM 调用费本身更容易被忽略。

**第五步：实际方案怎么组合**

生产上不会只用一类，我的组合是分层兜底：

```
稳定头部（system + 固定指令）      ← 永不改动，保 KV cache
    ↓
滚动摘要（旧历史压缩）             ← 超过阈值才更新
    ↓
最近 N 轮原文                     ← 按 token 预算保留
    ↓
[兜底] 硬截断                     ← 上面三层算完还超预算时
```

外化那一层是正交的：大对象从一开始就不进这个流水线。

---

**我项目里的两套实现（当例子看）**

**例子一：条数截断——一个反面教材。** `app/services/rag_agent_service.py:37-74` 的 `trim_messages_middleware`。规则是 `len(messages) <= 7` 不动，超过就保留 `messages[0]` 加最近 6 条（偶数）或 7 条（奇数），返回 `[RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_messages]`。

它做对了两件事：保 `messages[0]`（system message），以及 `:59` 的奇偶判断——那是为了不切开 user/assistant 配对。

但它按条数不按 token，正好是上面说的第一个错。7 条的实际占用可以差三个数量级，一条超长日志返回就能撑爆。这条链路正好是接了日志工具的那条，所以这个缺陷是真实存在的风险，不是理论问题。要修就是换成 `trim_messages` 加 token counter。

**例子二：token 预算 + 滚动摘要——这套是认真做的。** `app/services/conversation_memory_service.py`。

- token 估算 `:48-54`，中文按字数、其他按字符除 4，注释写着"以中文更保守的规则估算"。这里的取舍是：中文一个字往往就是一个 token 甚至更多，统一按 len/4 会严重低估，算出来"预算安全"实际已经超了。宁可保守。
- 单条成本 `:57-58` 是 `estimate_tokens(content) + 4`，那 4 是角色标记的固定开销。
- 主流程 `build_context:93-131`：读历史 → 找出快照之后的新消息（`:105-108`）→ 留出最近 N 轮（`:109-110`）→ 剩下的超阈值就摘要（`:112-116`）→ 存快照（`:118-122`）。**快照机制就是上面说的"滚动摘要"**，避免每轮全量重算。
- 摘要 prompt `_SUMMARY_SYSTEM_PROMPT:28-35` 强制四个固定标题：`【明确事实】【任务进展】【约束与偏好】【待确认项】`，缺失写"无"。这正是要点二那个坑的解法，`【约束与偏好】` 这个字段就是专门用来接"用户说过不能做什么"的。
- `:30` 还有一句："不要把模型猜测、SOP 内容或历史数值写成实时事实"。这是要点三，防止把三轮前的 CPU 读数固化成当前状态。
- 预算分配 `_render_context:207-224`：摘要预算 `min(summary_budget, context_budget - header_tokens)`，剩下给最近消息。**摘要先被切**，因为它本身就是可压缩的产物；最近消息不该为了塞摘要被砍。

默认值在 `app/config.py`：`conversation_memory_context_token_budget=1800`（`:133`）、`conversation_memory_summary_token_budget=700`（`:134`）。

这套里我认为最值得说的设计是：**原始消息全量存 PostgreSQL 作审计，只有喂模型的那份被压缩**（模块文档第 3 行写了这个分工）。压缩必然有损，但损失只发生在模型视图，不发生在数据。这样出了问题能回溯原始对话，也随时能换一套压缩策略重放。

**我没实现的两类，说清楚边界。**

检索式召回（对话历史向量化后按需取回）**项目中没有**。

外化也**没有**。现在检索到的文档是直接截断的——`app/agent/rag_v2/nodes.py:617-637` 的 `_truncate_to_token_budget`，预算 2800（`app/config.py:130`）。截断和外化的区别很关键：截断是被切掉的内容这轮就没了；外化是内容还在，模型可以再去取。要改成外化，得给模型一个 `read_document(doc_id, offset)` 工具，这块没做。

**项目证据**

- `app/services/rag_agent_service.py:37-74` — 条数截断，及其 token 盲区
- `app/services/conversation_memory_service.py:1-5` — 全量存库 / 压缩视图的分工
- `app/services/conversation_memory_service.py:48-54,57-58` — 中文保守 token 估算
- `app/services/conversation_memory_service.py:93-131` — 主流程，含快照式滚动摘要
- `app/services/conversation_memory_service.py:28-35` — 固定字段摘要 prompt
- `app/services/conversation_memory_service.py:207-224` — 预算分配顺序
- `app/config.py:130,133,134` — 三个预算默认值
- `app/agent/rag_v2/nodes.py:617-637` — 文档侧截断（不是外化）
- 检索式历史召回 / 外化 + 按需取回：项目中没有，不能硬套

### 面经 04 · 2026-06-23

#### Q1 · 绩点？有论文么？有实习么？

**回答**

个人背景问题，和仓库代码无关，**项目中没有对应实现**。这类题按实际情况答即可，不编造。

#### Q2 · 整个链路运转的流程

**回答**

以 `chat_v2` 这条为例，从 HTTP 进来到响应出去。

1. **中间件层**。`app/main.py:71-77` 把 `RequestIdMiddleware` 加在 CORS 之后——Starlette 的 `add_middleware` 是 `insert(0)` 再 reversed，所以**最后添加的在最外层**。放最外层有两个理由，注释里写了：ContextVar 要在任何业务代码（含异常处理器）之前设好，否则那些日志拿不到 rid；响应头由它最后回写，连 CORS 预检和错误响应都会带上 `X-Request-ID`。

2. **端点层**。`app/api/chat_v2.py`，建 trace 记录、设 span 篮子。

3. **服务层**。`app/agent/rag_v2/service.py:61` 用 `asyncio.timeout(config.chat_total_budget_seconds)`（90 秒，`app/config.py:81`）包住整个图，并且用 `graph.astream(inputs, stream_mode="values")` 而不是 `ainvoke`——因为 `astream` 每个 super-step 都会吐出完整状态快照，超时被取消时 `last_state` 里还握着已完成部分的结果。用 `ainvoke` 的话超时就是纯粹的异常，什么都拿不到。

4. **图层**。`app/agent/rag_v2/graph.py:43-48`：`rewrite` 改写出子查询 → `_fanout_to_retrieve:17-19` 用 `Send` 把每条子查询分发成一个 `retrieve_each` 实例并行检索 → `dedup` 按内容指纹去重裁到 `FINAL_TOP_K` → `generate` 生成 → `validate_answer` 校验证据覆盖。

5. **检索层**。`app/services/vector_search_service.py:52-107`：Milvus 向量检索 + BM25 关键词检索，`EnsembleRetriever` 做 RRF 融合，权重 0.7/0.3（`:42-43`）。候选集是 `candidate_k = max(top_k * 3, top_k, 10)`（`:84`）——先多召回再截断，直接取 top_k 会让融合无从发挥。

6. **异常出口**。任何一步炸了，`app/api/chat_v2.py:151` 是裸 `raise`，状态码交给 `app/core/exception_handlers.py:180-191` 按异常类型裁决（超时→504、上游不可用→502、其余→500）。炸之前 `:147` 会 `flush_spans`——失败路径的 span 比成功路径值钱，它记着跑到哪一步炸的。

7. **收尾**。`:153-158` 的 `finally` 里 `reset_span_collection(span_token)`，用 reset(token) 而不是留着不管，因为事件循环里的 ContextVar 在同一个 task 内复用，不还原下个请求会继承上个请求的 span 篮子。

**项目证据**

- `app/main.py:58-77` — 中间件顺序与理由
- `app/api/chat_v2.py:147,151,153-158` — flush / raise / reset
- `app/agent/rag_v2/service.py:56-75` — 总预算 + astream 保部分结果
- `app/agent/rag_v2/graph.py:17-19,43-48` — 图结构与 fan-out
- `app/services/vector_search_service.py:42-43,52-107,84` — 混合检索与候选集
- `app/core/exception_handlers.py:180-191` — 状态码裁决

#### Q3 · skill 分层体系是怎么做，为什么这么设计

**回答**

先把 skill 这个词说清楚，因为它和工具、prompt、RAG 文档很容易混，而面试官问"为什么这么设计"就是在考这个边界。

**Skill 是什么**

Skill 是一份**按需加载的过程性知识**。物理形态通常就是一个文件夹：里面一个 `SKILL.md`，开头 frontmatter 写 `name` 和 `description`，正文写这件事该怎么做；旁边可以放脚本、参考表、示例文件。Anthropic 2025 年提的 Agent Skills 规范就是这个形状，Claude Code 的 `.claude/skills/` 也是。

它要解决的问题很具体：**模型有通用能力，但不知道你们公司是怎么做事的**。模型知道怎么排查 CPU 打满，但不知道你们的机器要先从 CMDB 查归属、不知道你们的告警静默要走哪个平台、不知道三年前踩过"重启前必须先摘流量"的坑。这些东西写进 skill。

**Skill 和另外三个东西的区别**

这是最值得记住的一张对比，因为它直接决定了分层的必要性。

| | 加载时机 | 形态 | 模型知道它存在吗 | 典型体积 |
|---|---|---|---|---|
| **Tool** | 每次请求都带 schema | 函数签名 + 执行逻辑 | 知道（schema 在上下文） | 几十 token |
| **System Prompt** | 每次请求全量常驻 | 纯文本指令 | 知道 | 几百到几千 token |
| **RAG 文档** | 被动召回，靠相似度撞上 | 切片后的语料 | **不知道** | 库可以无限大 |
| **Skill** | 按需加载，有明确触发 | 说明文本 + 可选附件 | 知道（只知道名字和描述） | 描述几十，正文几千 |

关键差别有两条。

第一，**Tool 是"能做什么"，Skill 是"该怎么做"**。工具是单次函数调用：schema 固定、无状态、执行结果确定。Skill 是多步流程：包含判断条件、顺序、异常处理、经验教训。一个 skill 通常会编排好几个 tool。把 skill 说成"复杂一点的 tool"是答错了——它们不在同一个层次上。

第二，**Skill 比 System Prompt 省 token，比 RAG 文档可靠**。都写进 system prompt 的话，一百个流程每次请求都吃几十万 token；扔进 RAG 库的话，模型根本不知道有这个流程存在，只能靠用户提问和文档内容的向量相似度碰巧撞上——用户问"服务器好卡"，未必能召回标题叫《CPU 使用率高处理规范》的文档。Skill 拿到的是两者的中间态：**名字和描述常驻（所以模型知道有这个能力），正文按需加载（所以不占常驻预算）**。

**为什么要分层：渐进式披露**

这是分层唯一真正的理由——**上下文是稀缺资源**。标准做法是三级：

- **Level 1 常驻**：只有 `name` + `description`。一个 skill 几十 token。模型看到的是一份能力清单。
- **Level 2 触发后加载**：模型判断这次要用某个 skill，才把 `SKILL.md` 正文读进来。几百到几千 token。
- **Level 3 用到才读**：正文里引用的附件——脚本、字段对照表、长示例。可以任意大，因为大多数时候根本不加载。

算一下就知道为什么必须分层。100 个 skill，每个描述 50 token，常驻 5000 token，完全可以接受。同样 100 个 skill 全文加载，按每份 3000 token 算是 30 万 token，直接爆窗口，而且其中 99 份和当前问题无关。**分层不是为了架构好看，是让 skill 数量能 scale 的唯一办法。**

打个比方：这就是一本书的目录、章节、附录。你查一个 API 不会先把整本书背下来，你先看目录定位到章节，章节里让你去附录查参数表你才翻附录。

**另一个正交的分层维度：抽象粒度**

上面那层是"加载多少"，还有一层是"抽象到什么程度"。常见三层：

- **原子层**：单个工具调用。`query_cpu_usage`、`search_logs`。几乎不变。
- **场景层**：一类问题的固定流程。"CPU 打满排障"——先看是用户态还是系统态、再看是单核还是全核、再定位进程。跟着业务改，变更最频繁。
- **领域层**：跨场景的方法论。"任何性能问题先看资源饱和度再看队列长度"。最抽象也最稳定。

分开放的依据是**变更频率和复用范围不同**。原子层被所有场景复用、极少改；场景层只在特定告警类型下触发、经常改。混在一起的后果是改一个场景要动到公共部分，回归面积失控。这和后端分层的道理是一样的——按变更传播路径切，不是按名词切。

**项目现状：没有，不能硬套**

我的仓库里没有 skill 这个概念——没有注册表、没有目录约定、没有加载器、没有分层结构。

有两个东西容易被拿来冒充，得说清为什么不行。

一是 `app/tools/`（2 个工具）和 MCP 的 7 个工具。按上面那张表，它们是 Tool 不是 Skill：单次调用、schema 固定、无状态、没有流程和经验。

二是 `aiops-docs/` 下的 6 篇 SOP（`cpu_high_usage.md`、`disk_high_usage.md`、`memory_high_usage.md`、`high_api_latency.md`、`service_unavailable.md`、`slow_response.md`）。这几篇**内容上确实是过程性知识**，写的就是"这类告警怎么排查"。但它们在系统里的角色是 RAG 语料——被切片、向量化、靠相似度召回。模型不知道它们存在，没有触发条件，没有分层加载。按上面那张表，它们落在"RAG 文档"那一列，不是 Skill 那一列。这个区别恰好是本题的核心，说成 skill 体系就是把答案讲反了。

`[假设]` 如果要在这个仓库里做，改造成本最低的路径是：给这 6 篇 SOP 各加一段 frontmatter（`name` + `description` + 触发条件），描述汇总成一份清单常驻上下文，正文改成命中后才读。这样它们就从"被动语料"变成了 Level 1/Level 2 的 skill。但这是设想，代码里没有。

**项目证据**

- `app/tools/`（2 个工具）、`mcp_servers/`（7 个工具）— 是 Tool 不是 Skill
- `aiops-docs/*.md`（6 篇）— 内容是过程性知识，但角色是 RAG 语料，无触发无分层
- skill 注册 / 分层 / 渐进式加载：项目中没有，不能硬套

#### Q4 · 用户输入怎么和相关 skill 匹配

**回答**

这题本质是**路由**：一个自然语言输入，怎么落到 N 个候选单元里正确的那一个（或几个）。skill 匹配、意图识别、工具选择，底层是同一个问题，答法可以通用。

**先明确一件反直觉的事：主流做法是不做匹配**

Anthropic 的 Agent Skills 设计里，skill 的 `description` 直接进系统提示词，**由模型在生成时自己决定要不要加载**。没有独立的匹配器，没有分类模型，没有向量召回。

为什么？因为**模型的注意力机制本身就是一个匹配器**。用户问题和所有 skill 描述在同一个上下文里，模型做 next-token 预测时，注意力自然会在语义相关的描述上加权。你另外写一个匹配器，是在模型已经能做的事情上再套一层，而且那一层的准确率大概率不如模型——它看不到对话历史，也理解不了"接着上面那个问题"这种指代。

这一点值得在面试里明确说出来，因为它区分了"跟过最新实践"和"套用了老一代 NLP 那套 intent classification 的思路"。

**四种做法，以及各自什么时候才该上**

**做法一：模型自主选择（默认）**

所有 skill 描述常驻，模型自己挑。

- 成本：N 个描述的 token，每次请求都付。
- 上限：描述总量到几千 token 就开始有明显干扰，实践上几十个 skill 是舒适区。
- 失效模式：**描述写得烂**。这是唯一的调优抓手。描述必须写"什么时候用我"而不是"我是什么"。反例：`处理服务器性能相关问题`——太泛，用户问网络问题它也会被选中。正例：`当用户反馈 CPU 使用率超过 80%、或监控告警显示 cpu.usage 指标异常时使用；不适用于内存和磁盘问题`。**写清楚边界和排除项，比写清楚功能更重要**，因为路由错误几乎都是边界模糊导致的。

**做法二：向量召回 + 模型二选**

skill 描述向量化建索引，用户问题先召回 Top-K，只把这 K 个的描述给模型。

- 什么时候需要：skill 数量到几百上千，描述全放进上下文已经不现实。
- 好处：召回环节**可测**。可以建一批 (query, 正确 skill) 标注，跑 Hit@K、Recall@K、MRR，用数字驱动优化。模型自主选择那条路是黑盒，只能靠 case 分析。
- 失效模式：**召回天花板**。Top-K 没召回到正确的那个，后面模型再聪明也选不出来。所以 K 不能太小，而且要专门看漏召回的 case——通常是用户用了业务黑话，和描述里的书面表达对不上。这时候要么给 skill 加同义词别名，要么先做一次 query 改写。

**做法三：规则前置**

正则、关键词、结构化字段直接命中。

- 什么时候用：**输入本身就是结构化的**。告警系统推过来的 JSON 里有 `alert_type: cpu_high`，那就直接映射，不要给模型判断的机会。
- 好处：零延迟、零成本、100% 确定。
- 原则：**能确定化的决策不要交给概率模型**。这句话值得单独记住。很多人一上来就想用 LLM 做路由，但如果上游已经给了明确的类型字段，用 LLM 反而是引入了不确定性。
- 失效模式：覆盖不全，自然语言输入基本没法穷举规则。所以规则只做兜底和快速通道，不做主路径。

**做法四：小模型分类器**

微调一个 BERT 级别的分类头。

- 什么时候用：意图集合**封闭且稳定**，有标注数据，QPS 高到 LLM 调用成本受不了。
- 好处：几毫秒出结果，成本几乎为零，准确率在封闭集上可以做得比 LLM 高。
- 失效模式：**加一个新 skill 就要重新训练**。这是致命的运维成本。Agent 场景下 skill 是持续新增的，所以这条路在 Agent 里用得越来越少，更多留在传统对话系统里。

**选型的判断依据**

我的顺序是：**先看输入是否结构化**（是就上规则）→ **再看 skill 数量**（几十个以内直接给模型，上百个加向量召回）→ **最后看 QPS 和成本**（高到扛不住才考虑小模型）。

多数团队的正确答案是做法一，而不是做法二三四。上更复杂的方案要有触发条件，不然就是过度设计。

**一个容易被追问的点：能不能匹配到多个**

能，而且经常需要。"CPU 打满导致接口超时"同时命中性能排障和接口延迟两个 skill。这时候有两种处理：全部加载（简单，但 token 翻倍，而且两份流程可能给出冲突指令），或者让模型选主 skill、其余作为参考。我倾向前者加一句约束——在系统提示词里写明"多个流程冲突时以哪一份为准"，把仲裁规则显式化，而不是让模型自己纠结。

**项目现状：没有，不能硬套**

没有 skill，自然没有 skill 匹配。也没有意图分类器、没有路由节点、没有基于 embedding 的能力召回。

有一件事表面上像，必须说清区别：`app/agent/rag_v2/nodes.py:91-155` 的 `rewrite_node` 会把用户问题改写成多个子查询，各自去检索。这是**语料匹配**——找的是文档片段，命中之后作为证据拼进 prompt。而 skill 匹配找的是**可执行单元**，命中之后要改变后续的执行路径。两者的输出去向完全不同：子查询进的是向量库（`app/services/vector_search_service.py`），不是任何 skill 的入口。

另外那两条链路都没有路由：`rag_v2` 的边是写死的（`app/agent/rag_v2/graph.py:43-48`），`rag_agent_service` 那条把选择权完全交给了模型（走的是上面的做法一，但对象是 tool 不是 skill）。

`[假设]` 如果要在这个仓库里做，我会走做法二，因为向量检索的基础设施已经有了——`vector_search_service` 直接复用，把 6 篇 SOP 的触发描述单独建一个小 collection，召回质量能用 `tests/eval/metrics.py` 里现成的 `hit_at_k` / `reciprocal_rank` 来量化。规则前置那一层也值得加，因为 AIOps 场景的输入往往来自告警系统，本身带 `alert_type`。但这是设想，代码里没有。

**项目证据**

- `app/agent/rag_v2/nodes.py:91-155` — 子查询改写，是语料匹配不是 skill 匹配
- `app/agent/rag_v2/graph.py:43-48` — 写死的边，无路由
- `tests/eval/metrics.py` — 现成的 Hit@K / MRR，可复用于评召回
- 意图分类 / skill 路由 / skill 召回：项目中没有，不能硬套

#### Q5 · 有 skill 沉淀机制么

**回答**

先说清楚这道题在问什么。skill 沉淀（也叫 skill acquisition、skill library、经验积累）问的是：**Agent 干完一件事之后，下次遇到同类任务能不能少走弯路。** 如果不能，那它每次都是从零开始推理 —— 同样的任务重复消耗同样的 token，同样的坑重复踩。

这是 Agent 和普通 LLM 调用最本质的区别之一。人类工程师第一次配 nginx 反向代理要查半天文档，第二次十分钟搞定，第三次直接从上次的配置文件改。Agent 默认没有这个能力：每次对话结束，上下文清空，学到的东西全部蒸发。

##### 为什么需要沉淀：三笔实际的损失

**第一笔是成本。** 假设一个排障 Agent 处理"CPU 打满"，每次都要：读监控 → 判断是哪类打满 → 查进程 → 查日志 → 定位到具体服务。这个推理链每次都花 8 轮工具调用、两万 token。如果第一次做完就把"CPU 打满的标准排查顺序"沉淀下来，后面同类问题可以直接按流程走，三轮结束。

**第二笔是稳定性。** LLM 的输出是采样出来的，同一个问题问十次，推理路径可能有十种。没有沉淀就意味着**每次都在重新赌一次**。有沉淀之后，稳定的部分被固化，只有真正需要判断的部分交给模型 —— 方差大幅下降。

**第三笔是失败的重复。** 更要命的是错误。Agent 第一次调某个 API 时用错了参数格式，报错、重试、试对了。如果没沉淀，第二次它还会用错。用户会觉得"这个东西怎么一直犯同一个错"。

##### 行业里几种真实的沉淀路径

**路径一：Voyager 式的代码技能库。** NVIDIA 2023 年的 Voyager 是这条路的代表作，场景是 Minecraft。它的做法很直接：让模型写 JavaScript 代码去操作游戏，代码跑通了（能真的挖到钻石、能真的造出工具），就把这段代码连同一句自然语言描述一起存进向量库。下次遇到新任务，先用任务描述去检索技能库，把相关的旧代码作为示例塞进 prompt，模型在此基础上改写。

这条路的关键洞察是：**可执行的代码是最好的沉淀载体。** 因为它自带验证机制 —— 跑得通就是对的，跑不通就不存。相比之下，沉淀成自然语言描述的"经验"没法自动验证真假。

**路径二:Reflexion 式的失败反思。** 2023 年的 Reflexion 走的是另一个方向：不沉淀成功路径，沉淀**失败教训**。Agent 尝试失败后，先让模型写一段自我反思（"我刚才为什么错了"），把这段反思存进 episodic memory，下次尝试时把反思放进上下文。

它比沉淀成功路径更划算的原因是：成功的路径往往有很多种，沉淀哪一条都可以；但失败的原因通常是收敛的 —— 就那几个坑。一条"这个 API 的时间参数必须是毫秒时间戳，我上次传了秒被拒了"的教训，价值比十条成功路径都高。

**路径三:Generative Agents 式的分层反思。** 斯坦福那个"小镇"论文里的机制更抽象一层：Agent 定期回顾最近的记忆流，问自己"从这些事情里能得出什么更高层的结论"，把结论写回记忆。这样记忆不只是事件的堆积，而是逐渐形成"抽象观点"。这条路沉淀的不是操作步骤，而是**判断依据**。

**路径四：人工沉淀（Claude Code 的 skill / CLAUDE.md 就是这条）。** 不追求自动化，直接让人把经验写成结构化文件放进仓库。看起来最笨，但在真实生产里往往最有效 —— 因为它跳过了整个自动沉淀里最难的环节（判断什么值得沉淀、判断沉淀得对不对），把这个判断交给人。代价是不能规模化，人不写就没有。

**路径五：case-based reasoning（老派做法）。** 这其实是 90 年代专家系统时代的思路：把历史案例连同解法一起存起来，新问题来了找最相似的旧案例，改一改用。LLM 时代它复活了，因为"找相似"和"改一改"这两步现在都好做了。

##### 自动沉淀里真正难的四件事

我认为面试时能把这四个难点说清楚，比背出五条路径更有价值。

**第一，什么值得沉淀。** 不是所有执行历史都该进技能库。判断维度大致是三个乘起来：**出现频率**（一年一次的任务沉淀了也用不上）× **单次成本**（本来三秒就完的事不值得沉淀）× **过程稳定性**（如果每次的正确做法都不一样，沉淀下来的就是噪声）。三者都高才值得。

**第二，抽象粒度。** 这是最难的。沉淀得太具体，比如把"查 xxx-service 在 2026-09-01 的错误日志"整条存下来，只有那一个 case 能复用；沉淀得太抽象，比如"遇到问题要先看日志"，等于没说。合适的粒度通常是**把参数抽出来变成模板** —— 流程固定，服务名/时间窗/关键词变成占位符。Voyager 用代码函数来表达这个粒度，就是因为函数天然就是"固定逻辑 + 可变参数"。

**第三，验证。** 这是我认为最容易被忽略但最危险的一环。**沉淀错的东西比不沉淀更糟。** 因为它会被反复检索出来，反复误导后续决策 —— 也就是记忆污染（memory poisoning）。所以每条沉淀最好带着来源和置信度，最好能自动验证（代码能跑 / 断言能过），至少要能被后续的失败反向标记为不可信。

**第四，淘汰。** 技能会过期。API 改版了、内部系统下线了、业务规则变了，旧 skill 就从有用变成有害。所以需要有生命周期：记录最近命中时间、记录使用后的成功率，长期不命中的降权，命中但导致失败的删掉。没有淘汰机制的技能库，跑一年之后大概一半是垃圾。

##### 我这边的情况

**项目里没有 skill 沉淀机制。** 有沉淀的**原料**，但没有沉淀的**动作** —— 这个区分很重要，我不打算把落库说成沉淀。

原料是两张表。`ChatRunTrace` 记着每次请求的问题、答案、`used_documents`、以及一个 `is_bad_case` 标记（`app/models/chat_run_trace.py:46`）；`ChatRunSpan` 记着每个节点的耗时和降级原因（`app/models/chat_run_span.py:83` 的 `node` 字段）。数据在库里，但读取侧只有 `app/repositories/chat_run_trace_repository.py:75` 那个按时间倒序的查询，是给人看的，没有任何代码去做归纳。

对照上面那个"识别 → 抽象 → 验证 → 注册"的闭环，我这里四个环节一个都没有。

`[假设]` 如果要在这个仓库上做，我会从 `is_bad_case` 那批入手 —— 也就是走 Reflexion 那条路而不是 Voyager 那条路。理由有两个：一是我的场景（知识库问答 + 运维排障）里失败原因比成功路径更收敛，二是失败案例本来就被标出来了，起点是现成的。具体形态大概是：把 bad case 按失败类型聚类，人工确认后写成 SOP 文档，进现有的向量库 —— 相当于用检索代替注册，复用已有的基础设施。但这是设想，代码里没有。

**项目证据**

- `app/models/chat_run_trace.py:46`、`app/models/chat_run_span.py:83` — 沉淀的原料
- `app/repositories/chat_run_trace_repository.py:75` — 只有按时间倒序的读取，无归纳逻辑
- 模式识别 / 抽象 / 验证 / 注册：项目中没有，不能硬套

#### Q6 · 长短期记忆怎么设计的

**回答**

这道题几乎每场 Agent 面试都会问，而且很容易答得太浅 —— 答成"短期是这轮对话，长期是存数据库"就止步了。下面按我认为完整的层次讲。

##### 先破一个常见误解：LLM 本身没有记忆

大模型的推理是**无状态**的。每次调用，它看到的就是你这一次传进去的那串 token，除此之外什么都不知道。它不"记得"上一轮说了什么 —— 之所以看起来记得，是因为客户端每次都把整段历史重新发了一遍。

所以"Agent 的记忆"从头到尾都是**工程问题，不是模型问题**。所谓设计记忆，本质就在回答一件事：

> 这一次调用，我该把哪些信息拼进这个有限的窗口里？

短期记忆和长期记忆的区别，不在于"存多久"，而在于**信息是怎么进入上下文的**：短期记忆是默认全带上（在窗口预算内），长期记忆是按需检索出来才带上。这是最关键的一条区分，比"一个存内存一个存数据库"准确得多。

##### 短期记忆（Short-term / Working Memory）

短期记忆管的是**当前任务的连续性**：用户上一句说的"那个"指什么、刚才那个工具返回了什么、这个任务已经做到哪一步。

它的物理载体就是消息列表，难点全在容量管理上。当历史超过预算时，四种手段：

| 手段 | 做法 | 代价 |
|---|---|---|
| 滑动窗口 / 截断 | 只保留最近 N 条或 N token | 丢掉的信息永久消失，且可能丢的是关键约束 |
| 滚动摘要 | 旧消息压成摘要，摘要 + 最近消息一起送 | 要多花一次 LLM 调用；摘要有损且不可逆 |
| 分层缓存 | 最近原文 + 中期摘要 + 远期主题标签 | 实现复杂，三层的一致性要维护 |
| 向量召回 | 历史消息入库，按当前问题检索相关片段 | 有召回率天花板，可能漏掉真正相关的那条 |

生产系统一般是**滑动窗口 + 滚动摘要**的组合，因为这两者成本可控且行为可预测。

有三个细节是区分"做过"和"看过"的：

**第一，触发条件应该按 token 而不是按轮数。** 按轮数触发在"三轮长对话"和"三轮短对话"上表现完全不同 —— 前者可能已经两万 token 了，后者可能才三百。

**第二，摘要必须约束格式。** 让模型自由总结，它倾向于总结"发生了什么"（叙事），而系统性地漏掉"不能做什么"（约束）。用户三轮前说过"我们用的是 MySQL 5.7 不要给我 8.0 的语法"，自由摘要很可能把这句丢掉，然后模型继续给 8.0 的答案。所以摘要 prompt 要强制字段。

**第三，摘要要增量做。** 每次重压全部历史，成本随对话长度线性增长，而且每次重压结果都不一样（同一段历史压两次内容有出入），会造成上下文抖动。正确做法是记住"压到哪条了"，下次只压新增部分，摘要滚动叠加。

##### 长期记忆（Long-term Memory）

长期记忆管的是**跨会话的持久信息**。用户上周说过他负责的是订单服务，今天新开一个会话，Agent 应该还知道。

学术上按内容类型分三种，这个分类值得记住，因为它直接对应三种不同的存储和写入策略：

- **语义记忆（Semantic）** —— 事实。"用户负责订单服务"、"生产库是 PG 14"。适合结构化存储或 KV，写入判定相对明确。
- **情景记忆（Episodic）** —— 经历过的事。"上周三处理过一次同类告警，当时是连接池打满"。适合向量库，按相似度召回。
- **程序记忆（Procedural）** —— 怎么做。"这个服务重启要先摘流量"。适合文档或代码形态，也就是上一题说的 skill。

长期记忆真正的难点**不在存储，在两个判定**：

**写入判定：什么值得记。** 全记等于不记 —— 检索时噪声太多，相关的那条召不出来。用户随口说的"今天好累"不该记，"我们不用 Redis 集群模式"该记。这个判定通常要么靠规则（只记特定类型的陈述），要么靠一次额外的 LLM 抽取调用（成本换质量）。

**冲突处理：新旧矛盾怎么办。** 用户三个月前说负责订单服务，现在说负责支付服务。是覆盖、是并存、还是带时间戳让检索侧自己判断？直接覆盖会丢历史，并存会让模型看到矛盾信息然后随机选一个。比较稳的做法是带时间戳并存 + 检索时按时间衰减加权，让新的自然胜出，但旧的还查得到。

这两个判定处理不好，就是**记忆污染（memory poisoning）** —— 错的或过期的信息被反复召回，反复误导决策。而且它比没有记忆更难排查，因为你不知道是哪条旧记忆在捣鬼。

还有一个工程上的点：长期记忆的注入不该每轮都做。它不是每次都相关，无条件注入既浪费预算又稀释注意力。合理的做法是有触发条件 —— 比如新会话开始时注入用户画像，或者当前问题和某条记忆相似度超阈值时才拉进来。

##### 我这边的情况：短期是真的，长期没有

**短期记忆：`app/services/conversation_memory_service.py`，这套按上面的标准算做得比较认真。**

核心设计是把"存什么"和"喂什么"分开。模块文档头三行写明：完整原始消息全量存 PostgreSQL 作审计，模型上下文只用摘要加未压缩消息的尾部。这个分工的价值是**让压缩变成纯粹的视图问题** —— 压坏了可以重新压，原始数据一条没丢。这也是我认为这套设计里最值得讲的一点。

结构是三段：

- **滚动摘要**。旧消息压成固定四字段（`:28-35` 的 `【明确事实】【任务进展】【约束与偏好】【待确认项】`，缺失项写"无"），存进 `ConversationMemorySnapshot`，带 `summarized_through_message_id` 标记压到哪条了（`:107,121`）。下次只压新增的 —— 就是上面说的增量摘要。
- **最近窗口**。`_recent_message_count:146` 按 `recent_turns` 配置（默认 3 轮）保留尾部完整消息。
- **预算控制**。`_render_context:207-224`，总预算 1800 token（`app/config.py:133`），摘要最多 700（`:134`）。切的顺序是摘要先切、最近消息后切 —— 因为摘要本来就是可压缩的，而最近消息不该为了塞摘要被砍。

触发条件按 token 不按轮数：`:112` 判断 `_token_count(compact_candidates) >= self._compact_threshold_tokens`。

摘要 prompt 里还有一句我自己踩坑加上的（`:30`）："不要把模型猜测、SOP 内容或历史数值写成实时事实"。起因是摘要把三轮前的 CPU 读数固化成了当前状态，后面模型基于一个过期数值继续推理。

**长期记忆：项目中没有。** 没有跨会话的用户画像、没有偏好持久化、没有向量化的经验库。

有一个容易被误认成长期记忆的东西要说清楚：`app/services/rag_agent_service.py` 里的 `MemorySaver` checkpointer 是**进程内**的，重启即失效。它解决的是"同一进程内多轮对话的状态传递"，属于短期记忆的载体，不是长期记忆。会话数据确实进了 PostgreSQL，但读取侧只按 `session_id` 查（`app/repositories/conversation_repository.py:102`），跨会话没有任何聚合。

差距正好落在上面说的那两个判定上：跨会话写入判定我完全没有 —— 什么信息值得记到"用户"这个维度而不只是"这次会话"，这个逻辑一行都没写。存储层面倒不难加（PG 里加张表，或者复用现有的 Milvus），难的是那个判定。

**项目证据**

- `app/services/conversation_memory_service.py:1-5` — 存 / 喂分离
- `app/services/conversation_memory_service.py:28-35` — 固定四字段摘要 prompt
- `app/services/conversation_memory_service.py:30` — 禁止把历史数值写成实时事实
- `app/services/conversation_memory_service.py:93-131,107,121` — 增量摘要与快照标记
- `app/services/conversation_memory_service.py:112` — 按 token 而非轮数触发
- `app/services/conversation_memory_service.py:146,207-224` — 最近窗口与预算切分顺序
- `app/config.py:133,134` — 预算默认值
- `app/repositories/conversation_repository.py:102` — 只按 session 读，无跨会话聚合
- 跨会话长期记忆 / 用户画像 / 写入判定 / 冲突处理：项目中没有，不能硬套

### 面经 05 · 2026-06-17

#### Q1 · Claude Code 与 Codex 他们各自有什么特别

**回答**

产品对比题，和仓库代码无关，**项目中没有对应实现**。这类题只能凭使用经验答，我不从代码里编证据。

如实说的话：这两个工具我用过，但要我给出可验证的架构差异对比，我给不出——那需要看它们的内部实现。能说的只是使用层面的观察，而观察容易带上个人偏好，作为面试答案分量不足。

#### Q2 · 输入到模型的 prompt 由哪些部分组成

**回答**

我这边两条链路的 prompt 组成不一样,分开说。

**`rag_agent_service` 那条**（`app/services/rag_agent_service.py:176-236`）：
- system prompt（`:158-176`），只写工作原则，**不列工具清单**——框架已经单独传 schema，抄一遍就是两份真相
- 工具 schema，由 `@tool` 装饰器从签名和 docstring 生成，框架自动注入
- 历史消息，经过 `trim_messages_middleware:37-74` 截断
- 当前用户输入

**`rag_v2` 那条**（`app/agent/rag_v2/nodes.py:264-330` 的 `generate_node`）：
- system prompt（`:50` 的 `GENERATE_SYSTEM_PROMPT`）
- 会话记忆，包在 `<conversation_memory>` 标签里（`:287-288`）
- 检索到的文档上下文，每篇按 token 预算截断（`_truncate_to_token_budget:637`，总预算 2800，`app/config.py:130`）
- 当前问题

用 XML 标签包记忆而不是直接拼文本，是为了让模型能区分"这是历史结论"和"这是检索到的资料"。两者混在一起时，模型会把三轮前的推测当成资料里的事实。

这条链路上没有工具 schema——`rag_v2` 不做 tool calling，检索是编排层执行的。

**项目证据**

- `app/services/rag_agent_service.py:146-176,37-74` — 原则型 system prompt + 历史截断
- `app/agent/rag_v2/nodes.py:50,264-330,287-288,637` — 分段组装与 XML 标签
- `app/config.py:130` — 文档上下文预算

#### Q3 · 你做的 coding agent 和 Claude Code 与 Codex 区别在哪

**回答**

**项目中没有，不能硬套。** 我没有做过 coding agent。这个仓库是 RAG 问答加运维排障，不生成代码、不改文件、不执行命令。

不能拿现有东西冒充的理由：coding agent 的核心难点在于文件系统操作的可回滚、多文件改动的一致性、以及执行结果的验证闭环。我的 7 个 MCP 工具全是只读查询（查日志、查 CPU、查内存），没有一个会修改状态。这是完全不同的问题域，硬类比只会暴露我没做过。

#### Q4 · 你提到了上下文压缩，是怎么做的

**回答**

见本篇面经 03 的 Q8——我这里有两套，条数截断（`app/services/rag_agent_service.py:37-74`）和 token 预算加滚动摘要（`app/services/conversation_memory_service.py`）。这题问的是同一件事，答案在那题里已经展开。

补充一个那题没细说的点：两套压缩共存本身是个问题。`rag_agent_service` 走截断，`rag_v2` 走摘要，同一个用户在两个端点上的记忆行为不一致。这是演进过程的残留——截断是先写的，摘要是后来认真做的。如果收敛，应该统一到摘要那套，因为它有 token 感知；但 `rag_agent_service` 用的是 LangChain 的 `create_agent`，接入自定义记忆需要改中间件签名，还没做。

**项目证据**

- `app/services/rag_agent_service.py:37-74` — 截断那套
- `app/services/conversation_memory_service.py:93-131` — 摘要那套
- 两套统一：尚未完成

#### Q5 · 为什么要采用三层压缩策略，每一层压缩的内容一致么

**回答**

这题问的是"为什么要分层"，而不是"分了几层"。先把分层的依据讲清楚，再说具体分法。

**为什么不能用一套压缩打天下**

新手最容易犯的错，是把上下文当成一个同质的大字符串：满了就从头砍、或者整体丢给模型让它总结一遍。这么做一定会出事，因为**上下文里不同来源的内容，可压缩性差了一个量级**。

判断某段内容能不能压、该怎么压，看两个属性：

**信息密度**。同样一千个 token，五轮寒暄式对话里真正有用的可能只有一句"用户关心的是 A 服务的磁盘问题"；而一段 SOP 里写着 `du -sh /* | sort -rh | head -20`，每一个字符都是有效载荷。前者压缩比可以做到 20:1 而几乎无损，后者压缩比 2:1 就已经废了。

**结构冗余度**。对话历史里有大量的礼貌用语、重复确认、被推翻的中间结论 —— 这些是天然冗余，摘要能干净地滤掉。而一份 JSON 格式的 API 返回，字段名和值之间没有冗余，任何"总结"都是在丢字段。

这两个属性组合起来，就得出一条规则：**高密度、低冗余的内容只能截断或整块淘汰，不能摘要；低密度、高冗余的内容才适合摘要。**

再加一个很多人忽略的属性：**可再生性**。检索到的文档是可以重新检索回来的（同样的 query 再查一次就有），所以丢了不心疼；而用户在第三轮说的那句"注意我们的数据库是 PG 不是 MySQL"，一旦从上下文里消失就再也回不来（除非专门存下来）。可再生的内容优先牺牲，不可再生的内容优先保护 —— 这一条决定了预算抢占时谁让位。

**一个通用的分层方案**

按内容来源切，一个成熟的 Agent 上下文通常能分出五类，每类的处理方式都不一样：

| 层 | 内容 | 特点 | 处理方式 |
|---|---|---|---|
| L0 系统指令 | System prompt、角色定义、输出格式约束 | 密度极高、绝不可再生 | **永不压缩**，占用视为固定成本 |
| L1 任务状态 | 当前目标、已确认的约束、待办项 | 密度高、不可再生 | 结构化存储，只增不删，超限时人工裁剪 |
| L2 对话历史 | 多轮 user/assistant 消息 | 密度低、冗余高 | **滚动摘要**，压缩比可到 10:1 |
| L3 工具返回 | API 响应、日志片段、命令输出 | 密度中等、可再生 | 单条截断 + 只保留最近 N 条 |
| L4 检索文档 | RAG 召回的知识片段 | 密度高、可再生 | **按 token 预算截断整篇**，宁缺毋残 |

L4 那条"宁缺毋残"值得单独说：如果预算只够放两篇半，正确做法是放两篇完整的，而不是三篇各砍掉尾巴。因为半篇文档会让模型看到一个被截断的步骤列表 —— 它不知道后面还有第 5、6 步，会当作"总共 4 步"来回答，这比少给一篇更危险。少给一篇模型只是知识不足，给半篇是给了错误信息。

L0 和 L1 为什么绝不能压，还有一个隐蔽的理由：它们在 prompt 的最前面。压缩它们会改变前缀，导致 **KV cache 全部失效**（后面 Q8 那题详细讲了这个机制）。压 L2 只会让 L2 之后的部分重算，压 L0 会让整个请求重算。

**我这个项目的实际情况**

先纠正前提：**我这里是两层，不是三层。** 原帖面试者的三层是他自己的方案，我不假装有第三层。

对照上面的表格，我实际做了 L2 和 L4：

**L2 · 滚动摘要**（`app/services/conversation_memory_service.py:166-186`）。压的是已完成的旧对话，输出四个固定字段。有损且不可逆 —— 摘要之后原文不再进上下文（虽然仍在 PostgreSQL 里）。

**L4 · 文档截断**（`app/agent/rag_v2/nodes.py:617-637` 的 `_truncate_to_token_budget`）。压的是本轮检索到的文档，按 token 预算硬切。

两层压的内容完全不一致，这正是它们必须分开的原因：把五轮对话压成"用户想查 A 服务的 CPU 问题，已确认不是业务进程"，信息几乎没丢；但 SOP 里的 `du -sh` 一摘要就变成"用磁盘查看命令"，直接废掉。所以文档只能截断不能摘要。

预算也是分开算的：对话记忆 1800 token（`app/config.py:133`），文档上下文 2800（`:130`）。分开而不是共用一个总池，是因为它们抢的不该是同一块空间 —— 真到了要抢的时候我选择保文档，因为这是 RAG 系统，答案必须落在证据上，宁可让模型忘了三轮前聊过什么。

**我没做的那几层**，如实说：L0 的 system prompt 是写死的字符串，没有做过压缩或分级注入；L1 的任务状态完全没有实体化（下面 Q7 那题会讲这个缺口）；L3 的工具返回没有单独的截断策略，MCP 工具返回什么就原样进上下文 —— 这是个真实的隐患，某个日志查询返回一万行的话，现在没有任何东西挡它。**这几层项目中没有。**

**追问预案**

面试官很可能追问"那你怎么决定 1800 和 2800 这两个数"。诚实的答案是：这是拍的，按模型上下文窗口倒推留出安全余量，没有做过参数扫描实验。要做的话正确方式是固定评测集，扫描不同预算组合下的答案质量，找到质量随预算增长的饱和点 —— 但这需要 faithfulness 指标先落地（`tests/eval/metrics.py:168-174` 还是 `NotImplementedError`）。

**项目证据**

- `app/services/conversation_memory_service.py:166-186` — L2 摘要层
- `app/agent/rag_v2/nodes.py:617-637` — L4 截断层
- `app/config.py:130,133` — 两份独立预算，及其"保文档"的取舍
- L0 分级注入 / L1 任务状态 / L3 工具返回截断：项目中没有，不能硬套

#### Q6 · 如果模型因为压缩过度效果不理想，你是怎么发现的，怎么处理这种异常

**回答**

这题问的是**质量归因**，是压缩相关问题里最难的一道。难点不在"怎么压"，而在压完之后你其实不知道自己丢了什么 —— 因为你手里只有压缩后的版本，而"本来应该答出什么"这个参照系已经没了。

**先说清楚"压缩过度"到底长什么样**

压缩过度不是一种故障，是四种，表现完全不同：

**第一种，事实丢失。** 用户三轮前说过"我们用的是 MySQL 5.7"，摘要里没保留这条，第五轮问"这个语法支持吗"，模型按 8.0 答了。特征是答案**看起来很自信、很具体，但基于错误前提**。这种最危险，因为它没有任何异常信号 —— 模型不知道自己缺信息，它只是在缺失的地方填了个默认值。

**第二种，约束丢失。** 用户说过"不要用第三方库"，摘要总结成了"用户在做 Python 项目"，约束没了，模型开始推荐 pandas。这种之所以高频，是因为摘要模型天然倾向于总结"发生了什么"（叙事）而不是"不能做什么"（约束）—— 叙事在训练数据里的占比压倒性地高。

**第三种，指代断裂。** 摘要把具体对象抽象掉了。原文是"检查 order-service 的 CPU"，摘要成"排查了服务性能问题"，下一轮用户说"那再看看它的内存"，模型不知道"它"是谁。特征是模型会**反问**或者**换一个对象**继续答。

**第四种，答案空泛化。** 这种最隐蔽 —— 信息还在，但细节的密度降了。原来能答出"改 innodb_buffer_pool_size 到物理内存的 70%"，压缩后只能答"调整缓冲池大小"。答案不算错，也不算幻觉，就是没用了。

区分这四种很重要，因为**它们的检测手段完全不同**，混在一起谈就会得出"加个 faithfulness 校验就行了"这种错误结论。

**发现机制要分三层，缺一层就只能靠人肉**

**第一层：可观测字段。** 最基础的，是让每次请求都能回答"这次压了没有、压了多少"。至少要记：本次上下文的实际 token 数、是否使用了摘要、摘要覆盖了多少条原文、保留了几条未压缩消息、摘要本身占多少 token。

这一层单独看没有意义 —— 知道"用了摘要"并不能说明质量差。它的价值在于**给后面两层提供分组维度**。没有这一层，你连"压缩组"和"未压缩组"都分不出来，任何对比都做不了。

我这边这一层是有的：`app/services/conversation_memory_service.py:38-45` 的 `ConversationMemoryContext` 带 `estimated_tokens`、`used_summary`、`recent_message_count` 三个字段。但**没有任何代码去消费它们** —— 字段落在返回值里，没进指标、没进 trace 表、没有告警。这是典型的"埋了点但没接消费方"，等于白埋。

**第二层：质量信号。** 在答案上直接算一个可以监控的数。常见的三个：

- **证据支撑度（groundedness / faithfulness）**：答案里的每个论断能不能在给定证据里找到出处。做法是把答案拆成原子论断，逐条去证据里核对。这一层能抓住**第一种**（事实丢失导致的编造）。
- **答案相关性（answer relevance）**：答案是否真的回答了问题。反向生成 —— 拿答案去反推它在回答什么问题，再和原问题比语义相似度。这一层能抓住**第四种**（空泛化），因为空泛的答案反推出来的问题也是空泛的。
- **约束符合性**：把用户提过的硬约束抽出来做成检查项，每次答案都过一遍。这一层是抓**第二种**（约束丢失）的唯一手段，而且它必须**独立于摘要** —— 如果约束本身存在摘要里，摘要丢了它，检查项也就丢了。所以约束应该单独存一份结构化的，不走摘要通道。

我这边只有第一个：`app/agent/rag_v2/nodes.py:39-40` 的 `MIN_GROUNDEDNESS_SCORE = 0.75` 和 `MIN_COVERAGE_SCORE = 0.60`，在 `validate_answer_node` 里算。但要说清一件事 —— **它检测的是"答案没落在证据上"，不是"压缩过度"**。这两件事有交集但不等价：压缩过度完全可能表现为答案空泛却仍然 grounded（因为空泛的话通常都能在证据里找到影子），那样它检测不到。所以它只是个近似信号，不是直接检测。

第二个和第三个我没有：`tests/eval/metrics.py:174-185` 里 `faithfulness_placeholder` 和 `answer_relevance_placeholder` 两个函数都是直接 `raise NotImplementedError`。**这两项项目中没有。**

**第三层：归因判定。** 前两层给了"这次质量差"，第三层要回答"是不是压缩造成的"。这一步不能省，因为答案质量差有至少五个来源：检索没召回、模型本身能力不够、prompt 写得不好、用户问题本身有歧义、上下文压缩丢了信息。不做归因就调压缩参数，是在拿一个可能无关的旋钮试运气。

归因的标准做法是**分组对比**：把第一层的 `used_summary` 当分组键，看两组的第二层指标分布有没有显著差异。如果压缩组的 groundedness 中位数明显低于未压缩组，且两组的检索命中率相当（排除掉检索这个混淆因素），那就能归因到压缩。

更强的做法是**回放对照**：拿线上真实请求，同一个问题分别用"压缩上下文"和"完整上下文"各跑一次，直接比答案。这是唯一能测出**第四种**（空泛化）的方法，因为它有参照系 —— 你能看到完整上下文本来能答出什么。代价是双倍 token 成本，所以一般只对采样的一小批做，或者只对已经标记为坏样本的那批做。

这一层我完全没有。既没有分组对比的代码，也没有回放机制。**项目中没有。**

**处理机制：从便宜到贵四档**

发现之后怎么办，按成本排序：

**第一档，调参数。** 把压缩阈值放宽、把保留的未压缩轮数加大、把摘要预算提高。最便宜，但是全局的 —— 所有请求都变贵了，只为了救那一小部分压坏的。适合"整体偏紧"这种系统性问题。

我这边阈值是配置项（`app/config.py:133-134` 的 `conversation_memory_context_token_budget=1800` 和 `conversation_memory_summary_token_budget=700`），能调，但那是运维手段，不是自动处理。

**第二档，改摘要 prompt。** 如果丢的是特定类型的信息（比如老是丢约束），最有效的办法不是给更多预算，是**改摘要的输出结构**，强制它必须有那个字段。这是我这边唯一算做到位的一环：`conversation_memory_service.py:28-35` 的摘要 prompt 强制四个固定标题 ——`【明确事实】【任务进展】【约束与偏好】【待确认项】`，缺失项必须写"无"。

强制字段的作用是把"模型愿不愿意保留约束"变成"模型必须给约束留个位置"。空着比没有这一栏好 —— 空着是可见的缺失，没这一栏是不可见的缺失。prompt 里还有一句"不要把模型猜测、SOP 内容或历史数值写成实时事实"（`:30`），这条防的是摘要把三轮前的 CPU 读数固化成当前状态，属于第一种失效的针对性补丁。

**第三档，运行时回退。** 检测到质量不达标，就用更完整的上下文重跑一次。相当于把"压缩"变成一个可以撤销的决定。代价是延迟翻倍加成本翻倍，而且要小心别形成死循环（回退后还是不达标怎么办 —— 必须设最多回退一次）。

**这个我没有实现。** `validate_answer_node` 检测到低分之后只是打降级标记，不会重跑。

**第四档，改架构。** 如果压缩反复出问题，说明"把历史压进上下文"这个思路本身在这个场景不合适，该换成外部化 —— 完整历史留在库里，上下文只放句柄和索引，模型需要哪段自己去取。这是根治，但要新建取回工具和存储层。

**关键的设计原则**

有一条我这边做对了，值得单独说：**压缩只发生在"喂给模型的视图"上，不发生在数据上。**

`conversation_memory_service.py:1-5` 的模块文档明确了这个分工 —— 完整原始消息全量存 PostgreSQL 作审计，只有构造模型上下文时才压。这条保证了压缩的**可逆性**：压坏了可以改策略重压，可以回放对照，可以事后审计到底丢了哪句。

如果压缩是破坏性的（压完就把原文删了），上面第三层的所有归因手段都做不了 —— 你连参照系都没有。很多人在设计压缩时会顺手把原文丢掉省存储，这是把一个可恢复的问题变成不可恢复的。存储比 token 便宜得多，这笔账不用算。

**如果被追问"那你现在怎么知道压坏了"**

老实答：靠人看出来。没有自动检测，没有告警，没有归因。

`[假设]` 最小可行的补法是：把 `used_summary` 这个字段写进 `ChatRunTrace` 表，然后按它分组统计已有的 `coverage_score` 和 `groundedness_score` 分布。这个不需要新建任何基础设施 —— 两边的数据都已经在了，缺的只是把它们关联起来的那几行代码和一个查询。但代码确实没写。

**项目证据**

- `app/services/conversation_memory_service.py:38-45` — 第一层可观测字段（有埋点，无消费方）
- `app/agent/rag_v2/nodes.py:39-40,334-469` — 第二层的 groundedness/coverage（近似信号，非直接检测压缩）
- `app/services/conversation_memory_service.py:28-35` — 强制四字段摘要，第二档处理手段
- `app/services/conversation_memory_service.py:30` — 禁止把历史数值写成实时事实
- `app/services/conversation_memory_service.py:1-5` — 压缩只作用于模型视图，保证可逆
- `app/config.py:133,134` — 第一档可调参数
- `tests/eval/metrics.py:174-185` — answer relevance / faithfulness 均未实现
- 约束符合性检查 / 分组归因 / 回放对照 / 运行时回退：项目中没有，不能硬套

#### Q7 · 如果 agent 需要对原来的任务进一步修改，比如新加一个功能，你这个系统是怎么做的

**回答**

**项目中没有，不能硬套。** 我的系统是单轮请求-响应模型，没有"任务"这个持久化实体，因此也没有"修改已有任务"的能力。

具体缺什么：没有任务状态机、没有任务的中间产物存储、没有基于已有产物做增量修改的入口。每次 `chat_v2` 请求都是独立的，图跑完就结束（`app/agent/rag_v2/graph.py:48` 到 `END`）。

不能拿来冒充的东西有两个，得说明白：

一是 `MemorySaver` checkpointer（`app/services/rag_agent_service.py:133-137`）。它按 `thread_id` 存对话状态，看起来像"能接着上次继续"，但它存的是消息列表，不是任务产物；而且是进程内的，重启就没了。这是对话续接，不是任务修改。

二是 `AiopsDiagnosisTask` 的状态机（`app/models/aiops_diagnosis.py:73` 有 `can_transition_to`，注释写着"相同状态视为幂等允许"）。这个确实是任务状态机，但它管的是诊断流程的推进（pending → running → done），不支持"回到已完成的诊断上加一个新的排查项"。方向是单向的。

`[假设]` 要支持增量修改，最小改动是把每一步的产物按 `task_id + step_id` 落库，并允许从任意 step 分叉重跑。LangGraph 的 checkpoint 机制原生支持从某个 checkpoint 恢复，理论上接得上——但我现在的 graph 是 `compile()` 不带 checkpointer 的（`app/agent/rag_v2/graph.py:50`），连状态都没存。

**项目证据**

- `app/agent/rag_v2/graph.py:48,50` — 跑完即 END，compile 不带 checkpointer
- `app/services/rag_agent_service.py:133-137` — MemorySaver 是对话续接，非任务修改
- `app/models/aiops_diagnosis.py:73` — 单向状态机，不支持分叉
- 任务实体 / 产物存储 / 增量修改入口：项目中没有，不能硬套

### 面经 06 · 2026-06-13

> 面试岗位：AI Agent 开发岗

#### Q1 · 介绍一下两个项目，你觉得你这两个项目哪个更有收获？

**回答**

这个仓库里其实是两条并存的链路，可以当成"两个项目"来讲，它们的架构取向完全相反。

第一条是 `chat_v2` 的显式编排链路。`app/agent/rag_v2/graph.py:22-51`，五个节点手工连边：rewrite → retrieve_each（`Send` fan-out）→ dedup → generate → validate_answer。状态用 `TypedDict(total=False)`（`app/agent/rag_v2/state.py:10-34`），并行分支的合并靠 `Annotated[List[...], operator.add]` reducer。

第二条是 `chat` 的框架托管链路。`app/services/rag_agent_service.py:133-137`，一句 `create_agent(self.model, tools=all_tools, checkpointer=self.checkpointer)`，工具选择、循环终止、消息拼装全交给框架。

更有收获的是第一条，理由很具体：第二条链路出问题时我基本没有介入点。举个真实差别——第一条链路里检索失败是"某个分支返回降级原因、由 `dedup_node` 汇总判定"（`app/agent/rag_v2/nodes.py:214-261`），我能区分"三条子查询失败一条"和"全部失败"；第二条链路里 `retrieve_knowledge` 工具失败时是 `return f"检索知识时发生错误: {str(e)}", []`（`app/tools/knowledge_tool.py:13-43`），一个错误字符串直接进模型上下文，模型可能把它当成检索结果的一部分来编答案，而我在服务层看不到"这次检索失败了"这个事实。

第二条链路的收获在另一个地方：它让我知道了框架默认值不能信。`app/core/llm_factory.py` 的模块文档写着改造前三处各自 `new` 模型（`llm_factory.py:40`、`first_response_service.py:98`、`rag_agent_service.py:91`）全都没设超时——不是"用了某个默认值"，是真的无限期挂住。这个坑是在第二条链路上踩到的。

**项目证据**

- `app/agent/rag_v2/graph.py:22-51`、`app/agent/rag_v2/state.py:10-34` — 显式编排链路
- `app/services/rag_agent_service.py:133-137` — 框架托管链路
- `app/agent/rag_v2/nodes.py:214-261` — 显式链路的失败汇总能力
- `app/tools/knowledge_tool.py:13-43` — 托管链路里失败被降级成字符串
- `app/core/llm_factory.py` 模块文档 — 三处分散构造均无超时

#### Q2 · url 到页面显示？

**回答**

标准链路是：URL 解析 → DNS 查询（浏览器缓存 / 系统 hosts / 本地 DNS / 递归查询）→ TCP 三次握手 → TLS 握手（HTTPS）→ 发送 HTTP 请求 → 服务端处理并响应 → 浏览器解析 HTML 构建 DOM → 遇到外部资源发起子请求 → 构建 CSSOM → 合成渲染树 → 布局（Layout）→ 绘制（Paint）→ 合成（Composite）。

这条链路在这个仓库里能对上真实代码的部分，是"服务端处理"到"浏览器执行 JS"这一段：

访问 `/` 时命中 `app/main.py:148` 的 `FileResponse(index_path)`，返回 `static/index.html`。静态资源由 `app/main.py:141` 的 `app.mount("/static", StaticFiles(directory=static_dir), name="static")` 提供。

浏览器解析 HTML 时，`static/index.html:161-163` 有三个不带 `defer`/`async` 的 `<script>`：

```html
<script src="/static/chat_history_state.js"></script>
<script src="/static/chat_v2_stream_state.js"></script>
<script src="/static/app.js"></script>
```

这三个是**同步阻塞**加载的，顺序有意义——`app.js` 里的 `SuperBizAgentApp` 依赖前两个暴露的全局对象（`chat_history_state.js:5` 的 `root.ChatHistoryState = factory()`）。放在 `</body>` 前而不是 `<head>` 里，是为了让 DOM 先构建完，`constructor` 里的 `initializeElements()` 才拿得到节点。

第 9 和 12 行还有两个 CDN 脚本（`marked`、`highlight.js`），这两个是外域资源，`app.js:24` 和 `:35` 用 `typeof marked !== 'undefined'` 做了存在性检查——CDN 挂了页面不至于白屏。

渲染之后的交互走 SSE：`app/api/chat_v2.py:293` 返回 `EventSourceResponse`，前端在 `app.js:1219` 逐条解析 `sseMessage.type`。这一段是"页面显示"之后的增量更新，不属于首屏链路。

**项目证据**

- `app/main.py:141,148` — 静态挂载与首页响应
- `static/index.html:161-163` — 三个同步 script 及其顺序依赖
- `static/chat_history_state.js:1-7` — UMD 包装，暴露全局对象
- `static/app.js:24,35,66,82` — CDN 资源的存在性检查
- `app/api/chat_v2.py:293`、`static/app.js:1219` — 首屏之后的 SSE 增量

#### Q3 · 跨域和解决方案？

**回答**

跨域是浏览器同源策略的产物：协议、域名、端口三者任一不同即为跨源，浏览器会拦截 JS 读取响应的行为（请求通常已经发出去了，拦的是读）。

这个仓库确实会跨域，原因很具体：前端页面由 FastAPI 在 9900 端口提供，但 `static/app.js:3` 写死了 `this.apiBaseUrl = 'http://localhost:9900/api'`。同源部署时不跨域，一旦前端被单独起在别的端口（比如本地 `python -m http.server 8080` 调试），就是跨源请求。

解决方案在 `app/main.py:58-69`：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[REQUEST_ID_HEADER],
)
```

有三个点值得单独说：

第一，`allow_origins` 从配置读（`app/config.py:31-40`，带 `NoDecode` 和 `_split_cors_origins` 验证器做逗号分隔解析），注释写着"生产环境必须显式设置"。这里不能用 `["*"]`：`allow_credentials=True` 和 `allow_origins=["*"]` 组合起来浏览器会直接拒绝——带凭据的请求不允许通配源。

第二，`expose_headers=[REQUEST_ID_HEADER]` 是这段配置里最容易漏的一行。`app/main.py:65-67` 的注释说明了原因：CORS 默认只暴露 6 个安全响应头，不显式 expose 的话前端 `response.headers.get("X-Request-ID")` 拿到 `null`，用户报障时给不出这个 id，链路追踪的最后一公里断在浏览器里。

第三，中间件顺序。`app/main.py:71-77` 的注释写了：Starlette 的 `add_middleware` 是 `insert(0)`，构建时再 `reversed`，也就是**最后添加的位于最外层**。`RequestIdMiddleware` 在 CORS 之后添加，所以它在最外层，连 CORS 预检（OPTIONS）和错误响应都会带上 `X-Request-ID`。

其他方案（反向代理同源、JSONP、postMessage）这个项目没用。生产部署下最干净的是网关层同源转发，那样连 CORS 中间件都不需要——但项目中没有网关配置，**这块项目中没有，不能硬套。**

**项目证据**

- `app/main.py:58-69` — CORS 中间件配置
- `app/main.py:65-67` — expose_headers 的必要性说明
- `app/main.py:71-77` — 中间件顺序与 insert(0)/reversed 语义
- `app/config.py:31-40` — cors_origins 配置与解析验证器
- `static/app.js:3` — 前端写死的 apiBaseUrl
- 反向代理 / 网关同源转发：项目中没有，不能硬套

#### Q4 · 第二个项目的性能优化问了一下细节，确实做的很少，问了懒加载的原理？

**回答**

先说清边界：**前端懒加载这个项目中没有，不能硬套。** `static/` 下是原生 JS（`app.js` 1759 行，一个 `SuperBizAgentApp` class），没有构建工具、没有模块打包、没有路由分割，三个 `<script>` 是同步全量加载的（`static/index.html:161-163`）。图片懒加载、`IntersectionObserver`、`loading="lazy"`、动态 `import()` 一个都没有。

懒加载的原理本身分两类：

**资源级懒加载**。核心是把"资源何时加载"从"文档解析时"推迟到"真正需要时"。原生方案是 `<img loading="lazy">`，浏览器根据元素与视口的距离自行决定；手工方案是 `IntersectionObserver` 观察占位元素，回调里把 `data-src` 赋给 `src` 触发真实请求。关键点是**必须预留尺寸**，否则图片加载后撑开布局造成 CLS（累积布局偏移）。

**代码级懒加载**。动态 `import()` 返回 Promise，打包器据此切出独立 chunk，运行到该语句时才发请求。路由懒加载就是这个的应用。代价是首次进入该路由有一次网络往返，所以通常配 prefetch 预热。

两类的共同本质是**用一次延迟换首屏体积**，判断标准是"这个资源在首屏是否会被用到"——判断错了就是负优化。

顺带说一个容易被拿来充数但**不是同一回事**的东西：这个项目里有 Python 的延迟 import，比如 `app/core/llm_factory.py:125` 的注释——"延迟 import langchain_qwq：它是较重的可选依赖，放在模块顶层会让所有 import app.core.llm_factory 的地方都被迫加载它"，还有 `app/services/sop_retrieval_service.py:56`（避免 import 期触发单例连接 Milvus）、`app/services/first_response_service.py:147`（避免 import 期需要 API key）。这三处解决的是**模块加载时机和副作用**问题，不是浏览器端"按视口触发资源请求"，机制和场景都不同，不能当成前端懒加载的证据。

**项目证据**

- `static/index.html:161-163` — 三个同步 script，无 defer/async/动态 import
- `static/app.js` 共 1759 行单文件，无模块分割
- 图片懒加载 / IntersectionObserver / 路由懒加载 / 构建工具：项目中没有，不能硬套
- `app/core/llm_factory.py:125`、`app/services/sop_retrieval_service.py:56`、`app/services/first_response_service.py:147` — Python 延迟 import，机制不同，仅作对比说明

#### Q5 · 问我学没学过编译原理？

**回答**

**编译原理相关的实现这个项目中没有，不能硬套。** 仓库里没有编译器、解释器、DSL、语法分析器，也没有 AST 相关代码。

课程内容能讲的部分：词法分析（字符流 → token 流，正则 / DFA）、语法分析（token 流 → 语法树，递归下降 / LL(1) / LR(1)）、语义分析（符号表、类型检查）、中间代码生成（三地址码、SSA）、优化（常量折叠、死代码消除、循环不变量提升）、目标代码生成与寄存器分配。

有一处**看起来像但不是**的代码，我主动区分一下，避免让人误会：`app/agent/rag_v2/nodes.py:472-492` 的 `_parse_sub_queries`，以及 `:495` 起的 `_parse_validation_result`。它们从 LLM 输出里抽 JSON：

```python
fenced = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
if fenced:
    text = fenced.group(1)
else:
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        text = match.group(0)
try:
    data = json.loads(text)
    ...
except json.JSONDecodeError:
    logger.warning(f"[rag_v2.rewrite] 解析子查询失败, raw={raw!r}")
return [fallback]
```

这是**容错提取**，不是解析：正则先剥掉 markdown 围栏，真正的解析交给 `json.loads`，失败就回退到原问题（`return [fallback]`）。它没有文法、没有 token 流、没有语法树。硬把它讲成"我实现了词法分析"，追问一句"你的文法是什么"就崩了。

要说和编译原理沾边的思路，只有一点是真的：**分层容错**——先做宽松的结构定位（正则找边界），再做严格的语义校验（`json.loads`），失败有明确回退。这是 error recovery 的思路，但用到的技术是正则加标准库，跟编译原理的关系仅止于"思想上有共同点"。

**项目证据**

- 编译器 / 解释器 / DSL / AST / 语法分析器：项目中没有，不能硬套
- `app/agent/rag_v2/nodes.py:472-492,495+` — 正则加 `json.loads` 的容错提取，明确不是语法分析

### 面经 07 · 2026-06-08

> 面试岗位：AI Agent 开发岗

#### Q1 · 先问要不要写代码？

**回答**

要写，而且这个仓库里能拿出来讲的部分基本都是写代码时才会遇到的问题，不是设计文档层面的问题。

举三个只有动手才会碰到的例子：

一是超时。`app/core/llm_factory.py` 的模块文档记了一次实测：langchain-openai 在 `chat_models/base.py` 里把 `request_timeout`（默认 `None`）**无条件**塞进 openai SDK 的 client 参数，于是 SDK 自带的 `DEFAULT_TIMEOUT = Timeout(connect=5.0, read=600, ...)` 被覆盖成 `None`，底层 httpx 拿到 `Timeout(timeout=None)`。同一段代码里 `max_retries` 用的是 `if ... is not None` 的条件写法，所以 SDK 默认重试能生效——两个参数待遇不同。这种事只能读源码加实测得出，看文档看不出来。

二是 `timeout=0` 的语义。`app/core/llm_factory.py:66-80` 的 docstring 明确写了不能用 `timeout or config.x`，因为那样 `timeout=0` 会被静默替换成默认值，所以用 `is None` 判断。这是写单测时才会想到的边界。

三是 ContextVar 的作用域。`app/core/middleware.py:70-86` 的注释记录了：`ServerErrorMiddleware` 在 `RequestIdMiddleware` 外层，`finally: reset_request_id` 已经跑过了，所以最外层错误处理器读 ContextVar 拿到的是空值——错误响应体里的 `request_id` 恰好在最需要它的时候是 `null`。修法是把 id 同时写进 ASGI `scope["state"]`（`app/core/middleware.py:86`）。这个 bug 不写代码不会遇到。

**项目证据**

- `app/core/llm_factory.py` 模块文档 — request_timeout 被覆盖的实测记录
- `app/core/llm_factory.py:66-80` — `is None` 而非 `or` 的边界处理
- `app/core/middleware.py:70-86` — 中间件层级导致的 ContextVar 失效及修法

#### Q2 · 再问是单 Agent 还是多 Agent？

**回答**

单 Agent。**多 Agent 这个项目中没有，不能硬套。**

准确说是一个 Agent、两条实现路径：

`chat_v2` 是显式编排的固定流程（`app/agent/rag_v2/graph.py:22-51`），五个节点连成一条链，只有一个 fan-out（`_fanout_to_retrieve`，`graph.py:17-19`），fan-out 出来的 N 个 `retrieve_each` 实例是同一个节点函数的并行副本，不是不同角色的 Agent。它们没有独立的目标、没有相互通信、不做决策——每个实例只做一件事：拿一条子查询跑一次混合检索。

`chat` 是 `create_agent` 托管的单个 ReAct 循环（`app/services/rag_agent_service.py:133-137`），工具集是本地两个（`retrieve_knowledge`、`get_current_time`）加 MCP 加载到的（`rag_agent_service.py:123,131`），共 7 个 MCP 工具（CLS 5 个 + Monitor 2 个）。一个模型、一个消息历史、一个循环。

需要明确否认的是：`app/agent/aiops/` 下的 `planner.py`、`executor.py`、`replanner.py`、`state.py`、`utils.py` 以及 `app/services/aiops_service.py` **已从仓库删除**（git status 显示为 `D`）。如果按文件名猜，会以为这里有 Plan-Execute-Replan 的多角色架构，实际上没有。规划器、角色分工、Agent 间消息传递、共享黑板——一个都没有。

要往多 Agent 走，`Send` 机制是现成的地基：`graph.py:19` 的 `Send("retrieve_each", {"query": q})` 已经在做动态 fan-out，把目标节点换成不同角色就是多 Agent 的雏形。但这是 `[假设]`，不是现状。

**项目证据**

- `app/agent/rag_v2/graph.py:17-19,22-51` — 单链路，唯一 fan-out 是同质副本
- `app/services/rag_agent_service.py:123,131,133-137` — 单个 ReAct 循环
- `mcp_servers/cls_server.py`、`mcp_servers/monitor_server.py` — 7 个工具，非 7 个 Agent
- `app/agent/aiops/{planner,executor,replanner,state,utils}.py`、`app/services/aiops_service.py` — 已删除
- 多 Agent / 角色分工 / Agent 间通信：项目中没有，不能硬套

#### Q3 · 最后问绑不绑定单一模型厂商？

**回答**

诚实回答是：**架构上留了切换口，但实际绑在阿里云 DashScope 上，而且有一条路径是 Qwen 专用的。**

留的口在 `app/core/llm_factory.py`。它走 OpenAI 兼容模式，模块文档列了四个可切换目标（DashScope、OpenAI、Azure OpenAI、其他兼容服务），`base_url` 是参数（`llm_factory.py:64-82`），默认值 `DASHSCOPE_BASE_URL` 在 `:61`。换厂商理论上只改 `base_url` 加 `api_key`。

绑定在三个地方，从轻到重：

第一层，模型名。`app/config.py:46` 的 `dashscope_model: str = "qwen-max"`、`:129` 的 `rag_model: str = "qwen-max"`。这层最轻，改配置即可。

第二层，`ChatQwen` 专用路径。`app/core/llm_factory.py:105-148` 的 `create_qwen_model` 用的是 `langchain_qwq.ChatQwen`，不是 `ChatOpenAI`。docstring 写了理由：ChatQwen 对 Qwen 的推理内容（`reasoning_content`）等字段有专门处理，`rag_agent_service` 依赖这些行为，强行换成 ChatOpenAI 会改变运行时语义。所以这条路径换厂商就是重写。注释里还有一句关键的：ChatQwen 继承自 `BaseChatOpenAI`，字段名是 `request_timeout`——**但超时参数的默认值必须与 ChatOpenAI 路径完全一致**，两条路径共用同一套 config 默认值。这是收敛的实际意义。

第三层也是最重的，Embedding。`app/config.py:47` 的 `dashscope_embedding_model: str = "text-embedding-v4"`，1024 维。换 embedding 厂商不是改配置的事——维度变了 Milvus collection 要重建、全部文档要重新向量化、BM25 语料要重载（`app/services/vector_search_service.py:165-201` 从 Milvus 读全量分片建语料）。这是真正的锁定点。而且 DashScope 的 embedding 有服务端硬限制：单次批量上限 10 条，这个数字已经编进了索引流程。

所以我的说法是：对话模型换厂商是天的成本，embedding 换厂商是周的成本，而后者才是选型时该重点考虑的。

**项目证据**

- `app/core/llm_factory.py` 模块文档、`:61,64-82` — OpenAI 兼容模式与可切换的 base_url
- `app/core/llm_factory.py:105-148` — ChatQwen 专用路径及其不可替换理由
- `app/config.py:46,129` — 对话模型名（轻绑定）
- `app/config.py:47` — embedding 模型与 1024 维（重绑定）
- `app/services/vector_search_service.py:165-201` — 换 embedding 需重载的 BM25 语料

#### Q4 · 原帖第 4 条：「我自己做过 ×× 项目，当时因为 ×× 原因选了 ××，踩过 ×× 坑，后来怎么解决的」

**回答**

原帖这一条不是面试官的提问，是原作者给出的**答题模板**，按原样保留、不改写成问题。这里用仓库里的真实事件把模板填满：

**我做过 RAG Agent 项目**，`chat_v2` 这条链路一次请求要串行调用 LLM 三次（rewrite → generate → validate）。

**当时因为"每个服务各自构造自己需要的模型"看起来更内聚，选了分散构造**。`app/core/llm_factory.py` 模块文档里留了改造前的三个位置：`llm_factory.py:40`、`first_response_service.py:98`、`rag_agent_service.py:91`。

**踩的坑是这三处全都没设超时**。这不是巧合，是分散构造的必然结果——超时这类横切参数一旦分散，就没有任何一个地方能保证"全都设上了"。而且不传 `timeout` 时这条链路是**完全没有超时**，不是用了某个默认值：langchain-openai 无条件把 `request_timeout=None` 塞给 openai SDK，覆盖掉 SDK 自带的 `DEFAULT_TIMEOUT`，httpx 最终拿到 `Timeout(timeout=None)`。后果是一旦上游 TCP 连接建立后不再返回数据（限流排队、网关半开连接），调用无限期挂住 → SSE 挂住 → 连接池被占满 → 后续所有请求排队。当时全项目只有三处有超时：`milvus_timeout`、RQ `job_timeout`、MCP `retry_interceptor`，最慢最不可控的 LLM 链路反而完全没防护。

**后来的解决分两步。**

一是收敛构造入口。`app/core/llm_factory.py` 成为唯一入口，`create_chat_model`（`:64`）和 `create_qwen_model`（`:105`）共用同一套 config 默认值（`app/config.py:70` 的 `llm_timeout_seconds = 60.0`、`:75` 的 `llm_max_retries = 2`）。这里有个细节：不能写 `timeout or config.x`，那样 `timeout=0` 会被静默替换；用 `is None` 判断保留显式传 0 的语义（`llm_factory.py:66-80` 的 docstring）。

二是加总预算。单次超时管不住串行三次——最坏 3×60 秒再加检索耗时。所以在 graph 外层套 `asyncio.timeout(config.chat_total_budget_seconds)`（`app/agent/rag_v2/service.py:61`，90 秒，`app/config.py:81`）。这里的关键设计是超时**不向上抛**：

```python
except asyncio.TimeoutError:
    # 不向上抛：预算耗尽是**可降级**的失败，我们有部分结果可以交付。
    # 抛异常会让用户既等满了预算又什么都拿不到，是最差的结果。
    timed_out = True
```

能这么做的前提是用了 `astream(stream_mode="values")` 而不是 `ainvoke`，每个 super-step 的完整状态快照都存进 `last_state`（`service.py:62-64`），被取消时手里还握着部分结果。

**教训一句话**：横切关注点分散实现，等价于没实现——因为你无法证明它处处都在。

**项目证据**

- `app/core/llm_factory.py` 模块文档 — 三处分散构造与无超时的因果分析
- `app/core/llm_factory.py:64,66-80,105` — 收敛后的唯一入口与 `is None` 边界
- `app/config.py:70,75,81` — 超时 / 重试 / 总预算的唯一默认值
- `app/agent/rag_v2/service.py:61-70` — 总预算与"不向上抛"的降级决策
- `app/agent/rag_v2/service.py:62-64` — astream 保留部分结果的前提

### 面经 08 · 2026-06-08

> 面试岗位：AI Agent 开发岗

#### Q1 · React hook 使用时要注意什么？

**回答**

先说边界：**React 这个项目中没有，不能硬套。** `static/` 下是原生 JS，`app.js` 是一个 1759 行的 `SuperBizAgentApp` class，grep `react|useState|useEffect|jsx` 零命中，也没有 `package.json` 管理前端依赖。下面纯按知识回答，不假借项目代码充证据。

注意事项按重要性排：

**只在顶层调用，不在条件 / 循环 / 嵌套函数里调用。** 这是 Rules of Hooks 的第一条，原因见下一题。

**依赖数组必须完整。** `useEffect`/`useMemo`/`useCallback` 的依赖漏了会拿到闭包里的旧值（stale closure）。典型症状是 `setInterval` 回调里读到的永远是首次渲染的 state。修法是补全依赖，或用 `useRef` 存可变值，或用 `setState(prev => ...)` 的函数式更新避开对当前值的依赖。

**清理副作用。** `useEffect` 返回的清理函数要取消订阅、清定时器、abort 请求。React 18 的 StrictMode 在开发环境会故意 mount → unmount → 再 mount，专门用来暴露没清理的副作用。

**不要用 useMemo/useCallback 做"预防性优化"。** 两者本身有比较依赖和占内存的成本，只有在下游是 `memo` 组件、或计算确实昂贵时才划算。

**useState 的更新是异步批处理的。** 连续两次 `setCount(count + 1)` 只加一，因为两次读的是同一个 `count` 闭包值。要连加得用 `setCount(c => c + 1)`。

**自定义 hook 的命名必须以 use 开头。** 这不只是约定——ESLint 的 `react-hooks` 插件靠这个前缀识别哪些函数内部允许调用 hook。

**项目证据**

- React / hooks / JSX / package.json：项目中没有，不能硬套
- `static/app.js` — 1759 行原生 JS class，与 React 无关

#### Q2 · 为什么不能放到条件语句中？

**回答**

**同上，React 相关实现项目中没有，不能硬套。** 按知识回答。

根本原因是 React 靠**调用顺序**而不是名字来识别 hook。每个函数组件对应一个 fiber 节点，fiber 上挂着一条 hook 单向链表。首次渲染时每次调用 `useState`/`useEffect` 都往链表尾部追加一个 hook 对象；更新渲染时不再创建，而是按同样的顺序沿链表往下走，第 N 次调用取第 N 个节点。

所以 hook 和它的状态之间没有任何显式标识做绑定——绑定关系**就是**调用次序。放进条件语句里，某次渲染跳过了一次调用，后续所有 hook 的下标就整体前移一位，第 2 个 hook 会去读第 1 个 hook 的状态。

具体错法：

```jsx
function Bad({ flag }) {
  if (flag) {
    const [a, setA] = useState(0);   // flag 为 false 时这次调用消失
  }
  const [b, setB] = useState('');     // flag 从 true 变 false 时，b 会读到 a 的 memoizedState
  useEffect(() => {}, []);            // 依赖比较也会拿错对象
}
```

`flag: true → false` 时，React 发现本次渲染的 hook 调用次数少于链表长度，会抛 `Rendered fewer hooks than expected`；反向变化则是把数字状态赋给了期望字符串的变量——这种不报错的情况更危险。

正确写法是把条件移到 hook 内部，让调用本身无条件发生：

```jsx
const [a, setA] = useState(0);
useEffect(() => {
  if (!flag) return;   // 条件在 effect 内部
  doSomething();
}, [flag]);
```

顺带说：这个设计是有代价换来的。用链表加顺序而不是用 key 显式绑定，好处是 hook 不需要命名、可以在自定义 hook 里自由组合、运行时开销极小（一次指针移动）；代价就是这条"必须顺序稳定"的硬约束，只能靠 ESLint 静态检查兜住。

**项目证据**

- React fiber / hook 链表：项目中没有，不能硬套

#### Q3 · 能不能手写一些 hook 的链表结构实现来体现它不能放到条件语句中？

**回答**

**React 项目中没有，不能硬套。** 下面是我按 fiber hook 链表机制手写的最小复现，不是仓库代码。

```javascript
// 最小 hook 链表实现：只保留「顺序即绑定」这一个核心机制
let currentFiber = null;   // 当前渲染的组件
let workInProgressHook = null;  // 游标：指向本次渲染正在处理的 hook

function mountWorkInProgressHook() {
  const hook = { memoizedState: null, next: null };
  if (workInProgressHook === null) {
    currentFiber.memoizedState = hook;   // 链表头
  } else {
    workInProgressHook.next = hook;      // 追加到尾部
  }
  workInProgressHook = hook;
  return hook;
}

function updateWorkInProgressHook() {
  // 更新渲染：不创建，沿链表往下走
  const nextHook = workInProgressHook === null
    ? currentFiber.memoizedState        // 第一个 hook
    : workInProgressHook.next;          // 第 N 个 hook

  if (nextHook === null) {
    // 本次调用次数超过了上次 —— React 真实报错就在这个位置
    throw new Error('Rendered more hooks than during the previous render.');
  }
  workInProgressHook = nextHook;
  return nextHook;
}

function useState(initial) {
  const isMount = currentFiber.memoizedState === null || !currentFiber.mounted;
  const hook = isMount ? mountWorkInProgressHook() : updateWorkInProgressHook();

  if (isMount) {
    hook.memoizedState = initial;
  }
  const setState = (next) => {
    hook.memoizedState = typeof next === 'function' ? next(hook.memoizedState) : next;
  };
  // 注意返回的是 hook.memoizedState —— 取哪个 hook 完全由调用顺序决定
  return [hook.memoizedState, setState];
}

function render(fiber, Component, props) {
  currentFiber = fiber;
  workInProgressHook = null;   // 每次渲染游标归零，从链表头重新走
  const out = Component(props);
  fiber.mounted = true;
  return out;
}
```

用它复现错位：

```javascript
function Component({ flag }) {
  if (flag) {
    var [a] = useState('A');     // 条件调用
  }
  const [b] = useState('B');
  console.log({ a, b });
}

const fiber = { memoizedState: null, mounted: false };

render(fiber, Component, { flag: true });
// 链表：[hook1='A'] -> [hook2='B']
// 输出 { a: 'A', b: 'B' }

render(fiber, Component, { flag: false });
// 本次只调用了一次 useState，游标从头走 -> 拿到 hook1
// 输出 { a: undefined, b: 'A' }   ← b 读到了 a 的状态
```

第二次渲染 `b` 拿到 `'A'`，而 `hook2` 存的 `'B'` 成了孤儿。**没有抛错**——这是最危险的情况，状态错位静默发生。

反向复现会抛错：

```javascript
const f2 = { memoizedState: null, mounted: false };
render(f2, Component, { flag: false });  // 建立 1 个 hook 的链表
render(f2, Component, { flag: true });   // 要取第 2 个，next 是 null
// Error: Rendered more hooks than during the previous render.
```

这段代码说明的就是那句话：hook 与状态之间没有名字级别的绑定，`workInProgressHook` 这个游标是唯一的对应关系，游标每次渲染从头开始按调用次数前进，少一次调用就整体错位一格。

**项目证据**

- React fiber / hook 链表 / useState 实现：项目中没有，以上为手写复现代码，非仓库代码

#### Q4 · JS 的数据类型？

**回答**

**这题是 JS 语言知识，项目中有原生 JS 代码但不构成"实现"，不拿它当证据。**

7 种原始类型加 1 种引用类型：

原始类型：`undefined`、`null`、`boolean`、`number`、`string`、`symbol`（ES6）、`bigint`（ES2020）。

引用类型：`object`——数组、函数、Date、RegExp、Map、Set、Promise 全部归在这一类下。

几个必须说清的点：

`typeof null === 'object'`，这是 JS 最早期的实现遗留：类型标签用低位比特存，对象是 `000`，而 `null` 是全零指针，恰好被判成对象。这个 bug 因为修了会破坏太多存量代码，被永久保留。

`number` 是 IEEE 754 双精度，所以 `0.1 + 0.2 !== 0.3`，安全整数范围是 ±(2^53 - 1)，超出要用 `bigint`。`bigint` 和 `number` 不能直接混合运算（`1n + 1` 抛 TypeError），但 `==` 比较可以（`1n == 1` 为 true）。

`symbol` 的值唯一且不可枚举于 `for...in` / `Object.keys`，常用于加不冲突的属性键和内置协议（`Symbol.iterator`、`Symbol.asyncIterator`）。

`undefined` 和 `null` 的区别是语义：`undefined` 是"未赋值"（声明未初始化、函数无返回值、访问不存在的属性），`null` 是"显式的空"。所以 `??` 和 `?.` 同时把两者当空值处理，但 `typeof` 区别对待。

**项目证据**

- JS 语言规范知识，无需项目实现支撑
- `static/app.js`、`static/chat_history_state.js` — 项目中的原生 JS，但不作为本题证据

#### Q5 · 数据类型判断的方法？

**回答**

四种方法，各有适用边界：

**`typeof`**。快，但只能区分原始类型，且有两个坑：`typeof null === 'object'`，以及所有引用类型（数组、Date、正则）都返回 `'object'`。唯一能可靠判断 `function` 的方法。

**`instanceof`**。看原型链上有没有目标构造函数的 `prototype`。跨 iframe / 跨 realm 失效（每个 realm 有自己的 `Array.prototype`），且对原始值无效（`'a' instanceof String` 为 false）。

**`Object.prototype.toString.call()`**。最通用：

```javascript
Object.prototype.toString.call([])        // '[object Array]'
Object.prototype.toString.call(null)      // '[object Null]'
Object.prototype.toString.call(undefined) // '[object Undefined]'
Object.prototype.toString.call(new Date())// '[object Date]'
```

原理是读内部的 `Symbol.toStringTag`。缺点是自定义类都返回 `'[object Object]'`，且可以被 `Symbol.toStringTag` 伪造。

**专用判断方法**。`Array.isArray()` 判数组（跨 realm 安全，优于 `instanceof`）、`Number.isNaN()` 判 NaN（`isNaN('a')` 会先转换再判，返回 true，是错的）、`Number.isInteger()`、`Object.is()` 区分 `+0`/`-0` 和 `NaN`。

这个项目里前端代码用的就是"专用方法优先"这条：`static/app.js` 有 8 处 `Array.isArray(...)`（`:557,868,869,953,968,970` 等），处理后端返回的 `data`、`sub_queries`、`used_documents` 字段——用 `Array.isArray` 而不是 `typeof x === 'object'`，因为后者对 `null` 和对象都返回 true，拿到 `null` 时 `.map` 会直接抛错。同一文件 `:971` 判 `validation` 用的是 `payload.validation && typeof payload.validation === 'object'`，先做真值判断挡住 `null` 再 `typeof`——这是必须两步的原因。

`:24,35,66,82` 的 `typeof marked !== 'undefined'` 是另一个正确用法：判断一个**可能根本没声明**的全局变量，只有 `typeof` 不会抛 ReferenceError，换成 `marked !== undefined` 在 CDN 加载失败时会直接报错。

**项目证据**

- `static/app.js:557,868,869,953,968,970` — `Array.isArray` 而非 typeof 判数组
- `static/app.js:971` — 真值判断 + typeof 两步挡住 null
- `static/app.js:24,35,66,82` — `typeof x !== 'undefined'` 判未声明全局变量

---

## 三、月之暗面 AI Agent 开发岗一面

> 来源：https://www.nowcoder.com/discuss/922643334898647040 ｜ 面试时间：2026 年 7 月 21 日
>
> 这一篇原帖自带长回答。**原帖第 12、17 题自报的技术栈与本仓库实际不符**（原帖写 BGE-large-zh-v1.5 + IVF_FLAT + BGE reranker + BERT 路由 + Qwen-72B）。按要求逐题以本仓库真实代码回答，凡与原帖冲突处按代码纠正并说明。

### 面经 01 · 2026-07-21

#### Q1 · 简单讲讲你的 Agent 项目？

**回答**

我讲两条链路，因为它们的编排方式完全不同，混着讲会说不清。

第一条是 `chat_v2` 的多查询 RAG，用 LangGraph 显式编排五个节点：`rewrite → retrieve_each → dedup → generate → validate_answer`（`app/agent/rag_v2/graph.py:43-48`）。`rewrite` 把原问题改写成 3 个互补子查询（`NUM_SUB_QUERIES = 3`，`app/agent/rag_v2/nodes.py:28`），再用 `Send` fan-out 成并行分支（`app/agent/rag_v2/graph.py:17-19`），每条分支各跑一次混合检索。`dedup` 按内容指纹合并去重裁到 6 条（`FINAL_TOP_K = 6`，`nodes.py:30`），`generate` 生成答案，`validate_answer` 校验证据覆盖率和 groundedness。

第二条是 `chat` 的工具型 Agent，走 LangChain `create_agent` + `MemorySaver`（`app/services/rag_agent_service.py:133-137`），由模型自己决定要不要调工具。工具一共 9 个：本地 2 个（`retrieve_knowledge`、`get_current_time`），MCP 7 个（CLS 日志 5 个 + 监控 2 个，`mcp_servers/cls_server.py`、`mcp_servers/monitor_server.py`）。

关键是说清为什么这两条不合并。`chat_v2` 的五个节点是**固定顺序**的，不需要模型决策，所以用显式图——每一步耗时、每一步降级都能单独归因。`chat` 的工具选择本质是动态的，用图硬编排反而不如让模型自己判断。

我的贡献集中在可靠性层，不在链路本身：把三处各自 new 模型收敛到 `llm_factory`（`app/core/llm_factory.py`，模块文档里记着改造前三处全都没设超时）、把 HTTP 200 + body code:500 改成真状态码（`app/api/chat_v2.py:151` 的裸 `raise` + `app/core/exception_handlers.py:180-191` 裁决）、把"检索失败静默返回空列表"改成三态（`app/core/errors.py:65-78`）。

**这里要主动纠正一个常见说法**：ReAct 循环本身不保证长链可靠。`chat_v2` 根本不是 ReAct——它是固定五步的 DAG，没有 Thought/Action 循环。硬说成 ReAct 是往简历上贴名词。

**项目证据**

- `app/agent/rag_v2/graph.py:17-19,43-48` — Send fan-out 与五节点连边
- `app/agent/rag_v2/nodes.py:28,30,91,158,214,264,334` — 常量与五个节点入口
- `app/services/rag_agent_service.py:131-137` — create_agent + MemorySaver
- `mcp_servers/cls_server.py:104,136,169,212,346`、`mcp_servers/monitor_server.py:124,277` — 7 个 MCP 工具
- `app/core/llm_factory.py` 模块文档、`app/api/chat_v2.py:151`、`app/core/errors.py:65-78` — 我做的可靠性改造
- ReAct 循环 / 规划器 / 人工审批点：项目中没有，不能硬套

#### Q2 · 短期记忆的具体实现方式是什么？

**回答**

项目里有两套短期记忆，实现完全不同，得分开说。

`chat` 走 `trim_messages_middleware`（`app/services/rag_agent_service.py:37-74`）。逻辑很粗：消息数 ≤ 7 就不动；超了就保留 `messages[0]` 加最后 6 条（奇数时 7 条），然后返回 `{"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_messages]}`。这是纯滑窗，没有摘要，被砍掉的内容就是丢了。

`chat_v2` 走 `ConversationMemoryService`（`app/services/conversation_memory_service.py:61`），是"滚动摘要 + 有限最近窗口"。模块文档第 3-4 行写得很清楚：**完整原始消息只存 PostgreSQL 作审计，模型上下文只用压缩摘要加未压缩尾部。**

具体三层。第一层是 token 估算：`estimate_tokens`（`:48-54`）对中文按字符数、非中文按 `ceil(n/4)` 算，故意保守，宁可高估也不让预算穿。第二层是尾部窗口：`_recent_message_count`（`:146-158`）按**用户轮次**而不是消息条数倒着数，一轮 = 一条用户消息加它后面的助手消息，默认 3 轮（`app/config.py:136`）。第三层是摘要：超阈值时把尾部之前的消息压成固定字段的摘要。

摘要不是自由复述，prompt 里写死了四个标题（`app/services/conversation_memory_service.py:28-35`）：`【明确事实】【任务进展】【约束与偏好】【待确认项】`，缺失项写"无"。固定字段就是为了防摘要漏约束——自由复述最容易丢掉的正是"用户说过不要用 X"这类否定信息。

预算三档：上下文 1800 token、摘要 700 token、压缩阈值 2400 token（`app/config.py:133-135`）。

**项目证据**

- `app/services/rag_agent_service.py:37-74` — chat 的纯滑窗，无摘要
- `app/services/conversation_memory_service.py:1-5` — 原文存库、上下文用压缩的分工
- `app/services/conversation_memory_service.py:48-54` — 中文保守的 token 估算
- `app/services/conversation_memory_service.py:146-158` — 按用户轮次而非条数取尾部
- `app/services/conversation_memory_service.py:28-35` — 摘要四个固定字段
- `app/config.py:133-136` — 1800 / 700 / 2400 / 3 轮
- KV Cache / Workflow state / checkpoint 存储：项目中没有，不能硬套

#### Q3 · 什么叫"快到上限了"？对话是怎么逐步叠加的？

**回答**

项目里"快到上限"的判定条件是一行代码：`app/services/conversation_memory_service.py:112`

```python
if self._token_count(compact_candidates) >= self._compact_threshold_tokens:
```

注意判的是 `compact_candidates` 而不是全部消息。`compact_candidates` 是"尾部窗口之外的待压缩部分"（`:110`），也就是说尾部 3 轮永远不参与阈值判断——它们无论多长都要保留，判的只是"可以被压掉的那部分累积了多少"。

阈值 2400 token（`app/config.py:135`），上下文预算 1800（`:133`）。阈值高于预算是故意的：先攒够一批再压，比每次超一点就压一次要省调用。

**这题得实话说：项目里没有完整的预算分解。** 面试标准答案会说预算等于 system + tools + history + retrieval + query + 输出预留，逐项相加再判。仓库里只有两处独立的预算，互相不知道对方存在——会话记忆预算 1800（`app/config.py:133`）和文档上下文预算 2800（`app/config.py:130`，用在 `app/agent/rag_v2/nodes.py:617`）。system prompt、工具 schema、输出预留这三项**完全没有计入任何预算**。所以严格说，项目做的是"两个独立子预算各自不超"，不是"总预算不超"。**统一 Token Budgeter 项目中没有，不能硬套。**

叠加方式上，`chat_v2` 每轮是重新组装的，不是无脑追加。`build_context`（`:93-131`）每次都从库里读 `list_active_messages`，按游标 `summarized_through_message_id` 切出未摘要部分，再拼摘要加尾部。所以历史增长体现在库里，注入模型的部分是有界的。

**项目证据**

- `app/services/conversation_memory_service.py:110,112` — 阈值只判待压缩部分，尾部不参与
- `app/config.py:133,135` — 预算 1800、阈值 2400，阈值故意高于预算
- `app/config.py:130`、`app/agent/rag_v2/nodes.py:617` — 另一个互不知情的文档预算
- `app/services/conversation_memory_service.py:93-131` — 每轮重新组装而非追加
- 统一 Token Budgeter / system 与 tools schema 计入预算 / 输出预留：项目中没有，不能硬套

#### Q4 · 如果对话轮次过多，你怎么去做优化？

**回答**

项目里做了三件事，按优先级从高到低。

**保尾部**。`_recent_message_count`（`app/services/conversation_memory_service.py:146-158`）按用户轮次倒数 3 轮，这部分原文不动。为什么按轮次不按条数：按条数取 6 条可能正好把一条用户消息和它的助手回复切开，剩一个孤零零的回答，模型看不出这是在回什么。

**压中段**。尾部之前的消息累积超 2400 token 才压（`:112`），压完把游标推到 `compact_candidates[-1].id`（`:121`），下次从游标之后再算。

**存原文**。压缩不删数据，原始消息一直在 PostgreSQL 里（模块文档 `:1-5`）。

还有一处细节值得说，因为它是"宁可少给也不给错"的取舍。`_messages_after_snapshot`（`:133-144`）在游标对应的消息被软删除时，返回的是空列表而不是全部消息，注释写着"保守地不注入旧原文，避免重复历史"。找不到游标意味着系统不知道哪些已经压过了，此时注入全部会让模型看到重复内容——重复历史比缺一段历史更有害，因为模型会把同一件事当成发生了两次。

**项目里没有的**：按相关性挑选历史（现在是纯时间序）、失败轨迹单独保留、外部存储加事件 ID 回读、高风险决策留审批记录。**这四项项目中没有，不能硬套。**

**项目证据**

- `app/services/conversation_memory_service.py:146-158` — 按轮次保尾部的理由
- `app/services/conversation_memory_service.py:112,121` — 阈值触发与游标推进
- `app/services/conversation_memory_service.py:1-5` — 原文留库
- `app/services/conversation_memory_service.py:133-144` — 游标失效时保守返回空
- 相关性选择 / 失败轨迹保留 / 事件 ID 回读 / 审批记录：项目中没有，不能硬套

#### Q5 · 什么时候去触发这个总结动作？

**回答**

项目里的触发条件只有一个，纯预算驱动：待压缩部分的 token 数 ≥ 2400（`app/services/conversation_memory_service.py:112`，`app/config.py:135`）。触发点在 `build_context` 里，也就是每次要用记忆之前检查一次（`:93-131`）——不是后台定时，是懒触发。

没有软水位硬水位，没有异步预生成，没有语义触发。**子任务完成、目标切换、工具返回过大、连续失败重规划、转人工审批这些语义触发点，项目中一个都没有，不能硬套。**

并发这块得说实话。摘要写回是 `repo.save_memory_snapshot`（`:118-122`），**没有版本号，没有 compare-and-swap**。同一个 session 并发两个请求同时越过阈值，两个都会算摘要、都会写回，后写的覆盖先写的。这不是设计成"最后写入胜出"，是根本没考虑。

不过实际影响有限，原因在游标机制而不在并发控制：摘要连同 `summarized_through_message_id` 一起写（`:121`），两个并发请求的 `compact_candidates` 高度重叠，所以游标位置接近，覆盖掉的那份和留下的那份内容差不多。这是运气，不是正确性保证。**要真正解决，得给 snapshot 加版本号做 CAS——项目中没有。**

有一点做对了：摘要失败不阻断主流程。`_summarize` 的 except 分支（`:183-186`）注释写着"摘要是节省 Token 的优化，不可阻断主聊天"，返回空串，退回纯尾部窗口。这是 fail-open，用对了地方——摘要挂了最坏结果是上下文短一点，不是聊不了。

**项目证据**

- `app/services/conversation_memory_service.py:112`、`app/config.py:135` — 唯一的触发条件
- `app/services/conversation_memory_service.py:93-131` — 懒触发而非后台定时
- `app/services/conversation_memory_service.py:118-122` — 写回无版本号无 CAS
- `app/services/conversation_memory_service.py:183-186` — 摘要失败 fail-open
- 软/硬水位 / 异步预生成 / 语义触发 / CAS 并发控制 / 摘要一致性检查：项目中没有，不能硬套

#### Q6 · 是每一轮对话都要去做总结吗？

**回答**

不是。项目里绝大多数轮次不触发摘要，只有待压缩部分累积过 2400 token 才触发（`app/services/conversation_memory_service.py:112`）。

阈值设成 2400 而上下文预算只有 1800（`app/config.py:133,135`），这个大小关系是刻意的。如果阈值设成等于或小于预算，那么每次预算刚满就压一次，压完又慢慢涨满，很快又压——变成高频摘要。设成 2400 意味着攒够一批再压，一次压掉的量足够大，摊下来调用次数少得多。

每轮都摘要的代价不只是多一次 LLM 调用。更麻烦的是"摘要的摘要"：`_summarize` 的输入包含上一版摘要（`:172` 的 `existing`）加新消息，每轮压一次意味着同一条信息被反复有损压缩，限定条件和否定信息会一层层掉。固定四字段（`:28-35`）能缓解，但压的次数越多，掉的概率越大。

项目里没有的：确定性解析器优先（现在不管什么内容都交给 LLM 压）、周期性从原文全量重建（现在只有增量）、按单位收益评估摘要质量。**这三项项目中没有，不能硬套。**

**项目证据**

- `app/services/conversation_memory_service.py:112` — 阈值触发，非每轮
- `app/config.py:133,135` — 阈值 2400 > 预算 1800 的刻意设计
- `app/services/conversation_memory_service.py:172` — 摘要输入含上一版，存在累积有损
- `app/services/conversation_memory_service.py:28-35` — 固定字段缓解信息丢失
- 确定性解析器 / 周期性全量重建 / 摘要收益评估：项目中没有，不能硬套

#### Q7 · 假设已进行 10 轮并做了总结，第 11 轮开始时，总结怎么处理？是重算前 11 轮还是叠加？

**回答**

项目里是**叠加，不重算**，靠游标实现。

第 11 轮进来时，`build_context` 先读 snapshot 拿到 `summarized_through_message_id`（`app/services/conversation_memory_service.py:104-108`），再用 `_messages_after_snapshot`（`:133-144`）找到游标对应的那条消息，切出它之后的所有消息作为 `pending_messages`。已经被摘要覆盖的前 10 轮原文不再进上下文，注入的是"摘要 + 游标之后的消息"（`:128-131`）。

叠加体现在 `_summarize` 的 prompt 上（`:167-176`）：

```python
prompt = (
    f"已有滚动摘要：\n{existing or '（无）'}\n\n"
    f"需要合并的新消息：\n{source}\n\n"
    "请合并为新的滚动摘要。"
)
```

输入是旧摘要加增量消息，不是前 11 轮原文。所以单次摘要成本与总轮数无关，只与增量大小有关。

复杂度上，因为从不重算，累计处理量是线性的而非二次。但代价是**误差只能累积不能修正**：第 3 轮摘要里丢的一个约束，第 11 轮无法找回，因为第 11 轮看不到第 3 轮原文。

**项目里没有周期性全量重建**，这是这套实现最明显的缺口。原文一直在库里（模块文档 `:1-5`），技术上重建是可行的——`list_active_messages`（`:100`）能取到全部消息，把游标清空就会从头压。但没有任何代码这么做，也没有触发条件。同样**没有 version、没有覆盖区间记录、没有生成模型/prompt 版本号**，所以摘要质量下降了也没法发现，更没法对比新旧版本。**这几项项目中没有，不能硬套。**

**项目证据**

- `app/services/conversation_memory_service.py:104-108,133-144` — 游标定位，切出增量
- `app/services/conversation_memory_service.py:167-176` — 旧摘要 + 增量的叠加 prompt
- `app/services/conversation_memory_service.py:128-131` — 注入摘要加游标后消息
- `app/services/conversation_memory_service.py:100`、模块文档 `:1-5` — 原文可取，重建技术可行
- 周期性全量重建 / version / 覆盖区间 / prompt 版本：项目中没有，不能硬套

#### Q8 · 如果前 10 轮都变成了总结，那之前的原始上下文就不需要了吗？

**回答**

不是不需要，是不再每轮发给模型。项目里这两件事分得很清楚，模块文档第 3-4 行就是这个意思：完整原始消息只保存在 PostgreSQL 作为审计记录，模型上下文只用压缩摘要和未压缩尾部。

原文的存法是软删除而不是物理删除。`list_active_messages`（`app/services/conversation_memory_service.py:100`）名字里的 active 说明有非 active 的行留着。这带来一个副作用，前面 Q4 提过：游标指向的消息被软删除后，`_messages_after_snapshot` 找不到它，保守返回空列表（`:142-144`）。

原文留着在项目里的实际用途，有一个是真的：`ChatRunTrace` 落库（`app/repositories/chat_run_trace_repository.py`）记着每次请求的 `sub_queries`、`used_documents`、`validation`，`is_bad_case` 字段（`app/models/chat_run_trace.py:46`）带索引，可以捞出标记过的坏样本回放。

**但隐私和合规这块项目里是空的。** 没有加密、没有脱敏、没有 TTL、没有租户隔离、没有用户删除入口。`ConversationMessage.content` 是裸文本存 PostgreSQL。落库时唯一做的减法是 `slim_used_documents`（`app/core/used_documents.py:30-38`）把文档正文截到 200 字符（`EXCERPT_MAX_CHARS = 200`，`:23`），但那是为了不存第二份全文，不是为了脱敏。**加密 / 脱敏 / TTL / 租户隔离 / 删除权：项目中没有，不能硬套。**

还有一点，摘要里**没有原事件引用**。四个固定字段（`:28-35`）都是自然语言，不带 message id。所以"模型核对某个关键约束时精确读取原片段"这条在项目里做不到——摘要和原文之间没有可导航的链接。**这项项目中没有。**

**项目证据**

- `app/services/conversation_memory_service.py:1-5` — 原文存库、上下文用压缩的明确分工
- `app/services/conversation_memory_service.py:100,142-144` — 软删除与游标失效的保守处理
- `app/models/chat_run_trace.py:46`、`app/repositories/chat_run_trace_repository.py` — bad case 可回放
- `app/core/used_documents.py:23,30-38` — 截断是为了不存第二份全文，非脱敏
- 加密 / 脱敏 / TTL / 租户隔离 / 删除权 / 摘要含原事件引用：项目中没有，不能硬套

#### Q9 · 长期记忆可以按需检索召回，具体在什么情况下需要检索？

**回答**

**项目中没有长期记忆，不能硬套。**

得说清楚缺的是什么，因为项目里有两个东西容易被拿来冒充。

一个是 `ConversationMemoryService`（`app/services/conversation_memory_service.py:61`）。它确实叫 memory，也确实持久化到 PostgreSQL，但作用域是**单个 session**：`build_context(db, session_id)`（`:93-97`）的所有查询都带 `session_id`，换一个 session 什么都读不到。跨会话复用是长期记忆的定义性特征，这里没有。

另一个是知识库检索（`app/services/vector_search_service.py`）。它是跨会话的，但存的是**文档**，不是"用户说过什么、用户偏好什么、上次决定了什么"。把 RAG 当长期记忆是概念混淆——RAG 检索的是外部知识，长期记忆检索的是关于用户的事实。

所以缺的具体是：用户画像存储、跨会话事实抽取、记忆写入时机判定、memory type 分类、作用域过滤、冲突检测。一个都没有。

`[假设]` 如果要接，触发条件我会分两级。规则级先捞显式历史指代——"上次""昨天""还是用之前那个"这类词加上实体名；这层用正则就够，成本几乎为零。模型级判断隐式依赖，比如用户直接问"那个配置改好了吗"却没说是哪个。高风险操作要无条件查授权状态，不管用户提没提历史。

冲突处理是这套东西最容易做错的地方，标准答案往往说"长期更稳定所以优先"，这是反的。当前用户明确表达 > 业务主库实时状态 > 历史偏好 > 模型推断出的记忆。模型自己推断的记忆权重最低，它没有被用户确认过。

**项目证据**

- `app/services/conversation_memory_service.py:61,93-97` — 作用域限于单 session，不是长期记忆
- `app/services/vector_search_service.py` — 检索文档而非用户事实，不能冒充
- 长期记忆存储 / 跨会话事实抽取 / 记忆路由 / 冲突检测：项目中没有，不能硬套
- 以上触发条件与冲突优先级为 `[假设]` 方案

#### Q10 · 每轮对话都要注入记忆吗？长短期记忆是同时注入吗？

**回答**

短期记忆每轮都注入，但注入的是有界投影而不是全部；长期记忆**项目中没有，不能硬套**，所以"同时注入"这个问题在项目里不存在。

短期记忆的注入方式看 `generate_node`（`app/agent/rag_v2/nodes.py:264`）。会话上下文以 `<conversation_memory>` 标签包裹进 prompt（`:287-288`）：

```python
f"<conversation_memory>\n{conversation_context}\n</conversation_memory>\n\n"
if conversation_context
```

三个用到会话上下文的节点都做了同样的标签包裹：`rewrite_node`（`:102`）、`generate_node`（`:287`）、`validate_answer_node`（`:380`）。用标签而不是直接拼进正文，是为了让模型能区分"这是历史记忆"和"这是当前检索到的证据"——两者混在一起，模型会把历史里提过的数值当成刚查到的事实。

`if conversation_context` 这个条件说明空上下文不注入，不会塞一个空标签进去。

组装顺序在 `generate_node` 里是固定的：system prompt（`:306`）→ 会话记忆 → 检索到的文档 → 当前问题。检索文档有独立预算 2800 token（`app/config.py:130`，`nodes.py:617`），会话记忆有独立预算 1800（`app/config.py:133`）。两个预算互不知情，这是 Q3 说过的缺口。

**没有的**：记忆的 Top-K 或相关性阈值（会话记忆是全量注入尾部窗口，不做筛选）、长短期冲突裁决、记忆来源与时间标注。**这三项项目中没有。**

**项目证据**

- `app/agent/rag_v2/nodes.py:102,287-288,380` — 三个节点统一用 `<conversation_memory>` 标签隔离
- `app/agent/rag_v2/nodes.py:287-288` — 空上下文不注入
- `app/agent/rag_v2/nodes.py:306,617`、`app/config.py:130,133` — 组装顺序与两个独立预算
- 长期记忆 / 记忆 Top-K / 冲突裁决 / 来源时间标注：项目中没有，不能硬套

#### Q11 · 如何减少工具过多带来的 Token 消耗？

**回答**

**项目里没有做这件事，不能硬套。** 工具是全量绑定的。

`_initialize_agent`（`app/services/rag_agent_service.py:115-139`）里：

```python
all_tools = self.tools + self.mcp_tools
self.agent = create_agent(self.model, tools=all_tools, checkpointer=self.checkpointer)
```

9 个工具（本地 2 + MCP 7）的完整 schema 每轮都发给模型。没有目录层、没有路由、没有按需展开。

不过项目规模下这不是当前的瓶颈，说清楚比假装做了优化更好。9 个工具的 schema 大概几百到一千 token 量级，相对 1800 的会话预算和 2800 的文档预算，占比不算主要矛盾。progressive disclosure 是几十上百个工具时才必要的机制，现在做是过度设计（YAGNI）。

项目里倒是有一个和"工具返回体过大"相关的真实处理，方向对得上但不是同一件事。`slim_used_documents`（`app/core/used_documents.py:30-38`）把文档正文截到 200 字符再落库，模块文档解释了为什么切点在写库前而不是在序列化时——`_serialize_docs` 的产物同时喂给 SSE 事件、HTTP 响应体和 trace 落库，前两者是前端契约不能动，只有落库这一路该瘦身。这是"限制存储体积"，不是"限制进模型的体积"。

`[假设]` 真要做，第一步不是压 schema 而是分组：CLS 的 5 个日志工具和 Monitor 的 2 个指标工具本来就是两个领域，按 MCP server 天然分组，先路由到领域再加载该领域的 schema。这比逐字段精简 description 收益大得多，因为省掉的是整组而不是几个词。

**项目证据**

- `app/services/rag_agent_service.py:115-139` — 全量绑定，无路由无分层
- `mcp_servers/cls_server.py`、`mcp_servers/monitor_server.py` — 7 个 MCP 工具天然分两域
- `app/core/used_documents.py:30-38` 及模块文档 — 限制存储体积（方向相关但非同一问题）
- progressive disclosure / 工具目录 / 工具检索 / schema 精简 / Prompt caching：项目中没有，不能硬套

#### Q12 · 你的 RAG 是用什么技术实现的？

**回答**

**这题必须先纠正原帖。** 原帖写的是 BGE-large-zh-v1.5 + IVF_FLAT + BGE reranker 精排 Top-20→Top-5。本仓库实际代码是另一套：

| 原帖口径 | 仓库实际 | 证据 |
|---|---|---|
| BGE-large-zh-v1.5 | DashScope `text-embedding-v4` | `app/config.py:47` |
| IVF_FLAT | HNSW | `app/core/milvus_client.py:199` |
| （未提 metric） | COSINE | `app/core/milvus_client.py:198` |
| BGE reranker Top-20→Top-5 | **无 reranker** | 见下 |

维度 1024（`app/core/milvus_client.py:49` 的 `VECTOR_DIM`，用在 `:162`）。git log 里有一条提交 `IVF_FLAT+L2 → HNSW+COSINE`，说明 IVF_FLAT 是改造前的状态，原帖写的是旧口径。

**reranker 这条要说得更细，因为仓库里有容易误判的痕迹。** `.hf_cache/hub/models--BAAI--bge-reranker-v2-m3/` 目录确实存在，`tests/eval/run_eval.py:456` 也确实有 `--retriever local-rerank` 这个评测选项。但**线上任何代码路径都不经过 reranker**：`retrieve_knowledge`（`app/tools/knowledge_tool.py:13-43`）和 `retrieve_each_node`（`app/agent/rag_v2/nodes.py:158`）都是直接调 `vector_search_service.retrieve_documents`，返回即用。另外三处提到 rerank 的地方（`app/services/metadata_enricher.py:5`、`app/core/used_documents.py:34,150`）都是注释，写的是"将来接上 rerank 时"。所以准确说法是：**下载过模型、评测脚本里能跑，生产链路没接。**

实际的检索是混合检索加 RRF。`VectorSearchService.retrieve_documents`（`app/services/vector_search_service.py:52-107`）：向量检索走 Milvus，关键词检索走 `BM25Retriever`，两路交给 `EnsembleRetriever` 融合（`:96-99`），权重 0.7 / 0.3（`:42-43`）。候选数 `candidate_k = max(top_k * 3, top_k, 10)`（`:84`）——先多召回再截，不是直接取 top_k。

BM25 语料是从 Milvus 反读出来的（`_load_documents_from_milvus`，`:165-201`），用 `query_iterator` 分批拉，批大小 1000（`:44`）。缓存按 `collection.num_entities` 判失效（`:145`），文档数变了才重建。分词器是自己写的（`_tokenize_for_bm25`，`:203-224`）：ASCII 字母数字连成词，中文逐字切，其余作分隔。中文逐字而非分词，是因为不想引入 jieba 这类依赖，逐字对 BM25 在中文上够用。

一个降级细节：BM25 语料为空时不报错，退成纯向量检索（`:92-94`），日志 warning。这是 fail-open，因为空语料通常意味着知识库刚建还没索引完，此时纯向量仍能用。

**评测**：`tests/eval/metrics.py` 实现了 Hit@K（`:69-72`）、Recall@K（`:75-81`）、MRR（`:84-87`）。**Faithfulness 和 Answer Relevance 是 `NotImplementedError` 占位（`:168-174`、`:177-179`），没有实现，不能说做了生成层评测。** golden set 18 条（`tests/eval/golden_set.jsonl`）加扩展集 83 条。

**项目证据**

- `app/config.py:47` — text-embedding-v4，非 BGE
- `app/core/milvus_client.py:49,162,198,199` — 1024 维、COSINE、HNSW，非 IVF_FLAT
- `app/services/vector_search_service.py:42-44,52-107,84,92-94,96-99,145,165-201,203-224` — 混合检索全链路
- `app/tools/knowledge_tool.py:13-43`、`app/agent/rag_v2/nodes.py:158` — 线上路径不过 reranker
- `app/services/metadata_enricher.py:5`、`app/core/used_documents.py:34,150` — rerank 仅存在于注释
- `tests/eval/metrics.py:69-72,75-81,84-87` — 检索层三指标已实现
- `tests/eval/metrics.py:168-174,177-179` — 生成层两指标未实现
- BGE embedding / IVF_FLAT / 生产 reranker：项目中没有，原帖口径有误

#### Q13 · 如何判断向量的相似度？

**回答**

项目里用余弦相似度，配置在一行：`app/core/milvus_client.py:198`

```python
"metric_type": "COSINE",  # 余弦相似度，文本向量检索的最佳实践
```

配合 HNSW 索引（`:199`），维度 1024（`:49`）。

选 COSINE 而不是 L2 的实际原因，git log 有记录：那条 `IVF_FLAT+L2 → HNSW+COSINE` 提交同时改了索引类型和 metric。改 metric 的动机是文本 embedding 的模长通常不携带语义信息，L2 会让长文档和短文档因为模长差异被拉开距离，而实际上它们可能说的是同一件事。

**但项目里有一个必须说的缺口：没有在任何地方做显式归一化。** 代码里搜不到 normalize 相关操作，`app/services/vector_embedding_service.py:173` 直接把 DashScope 返回的向量交给 Milvus。这依赖 `text-embedding-v4` 自己返回归一化向量——这是 DashScope 侧的行为，仓库里没有断言也没有校验。如果哪天换了不归一化的模型，COSINE 仍然能算（Milvus 内部会除模长），但如果同时有人改成 IP，结果就会静默错掉。**归一化校验项目中没有。**

阈值这块也是空的。项目里**不按分数卡阈值**，`retrieve_documents`（`app/services/vector_search_service.py:101`）是 `ensemble_retriever.invoke(query)[:top_k]`，无条件取前 K 个。所以"检索到的文档相关性太低"这种情况在检索层不会被拦下，只能靠后面的 `validate_answer_node`（`app/agent/rag_v2/nodes.py:334`）用 coverage 和 groundedness 分数兜（阈值 0.60 / 0.75，`:32-33`）。这个设计有道理：向量分数的绝对值没有跨领域可比性，0.8 在一个语料上算高在另一个上算低，卡阈值容易误杀；交给 LLM 判"证据够不够"更贴近真实需求。

另外走了 `EnsembleRetriever` 之后**拿不到分数**。`app/services/vector_search_service.py:129-130` 的注释说明了这点，`SearchResult.score` 填的是 `1.0 / index`，也就是排名倒数而非真实相似度。`app/core/used_documents.py` 模块文档进一步解释为什么落库时 score 存 `None` 而不存这个名次值：一个看起来像相关性分数、实际是排名倒数的值，比没有分数更危险。

**项目证据**

- `app/core/milvus_client.py:49,198,199` — COSINE + HNSW + 1024 维
- `app/services/vector_embedding_service.py:173` — 直接使用模型返回向量，无归一化校验
- `app/services/vector_search_service.py:101` — 不卡分数阈值，无条件取 top_k
- `app/agent/rag_v2/nodes.py:32-33,334` — 相关性判断交给证据校验层
- `app/services/vector_search_service.py:129-130`、`app/core/used_documents.py` 模块文档 — 融合后无真实分数，不伪造
- 归一化断言 / 检索分数阈值校准：项目中没有，不能硬套

#### Q14 · 除了余弦相似度，还了解其他相似度算法吗？

**回答**

项目里实际用到两种打分机制，一种是向量距离，一种不是。

向量侧只有 COSINE（`app/core/milvus_client.py:198`）。Milvus 支持的 IP、L2 在这个仓库里没有用到——**IP / L2 / 曼哈顿距离项目中没有配置过，不能硬套。**

非向量侧是 BM25（`app/services/vector_search_service.py:155-158` 的 `BM25Retriever.from_documents`）。BM25 不是向量距离，是基于词频和逆文档频率的稀疏检索打分。项目里同时用两者，正是因为它们互补：向量能处理"重置密码"和"忘记密码找回"这种换词表达，BM25 能处理型号、数字、专有名词这类必须精确命中的内容。设备型号 `X-2000` 在向量空间里和 `X-3000` 几乎重合，BM25 能分开。

融合方式是 RRF（`app/services/vector_search_service.py:95` 的注释 `#RRF混合排序`，实现在 `EnsembleRetriever`），权重 0.7 / 0.3（`:42-43`）。RRF 的好处正是它**不需要比较两路分数的绝对值**——余弦分数和 BM25 分数量纲完全不同，一个在 [-1,1]，一个是无上界的正数，直接加权求和是没有意义的。RRF 只用排名。

Cross-encoder 这条：模型下载了（`.hf_cache/hub/models--BAAI--bge-reranker-v2-m3/`），评测脚本能跑（`tests/eval/run_eval.py:456` 的 `--retriever local-rerank`），**但生产链路没接，不能说项目用了 Cross-encoder。**

Jaccard 项目里没有。分词器 `_tokenize_for_bm25`（`:203-224`）产出的是 token 列表，理论上能算集合重合度，但没有任何代码这么做。

**项目证据**

- `app/core/milvus_client.py:198` — 向量侧仅 COSINE
- `app/services/vector_search_service.py:155-158,203-224` — BM25 稀疏检索与自写分词
- `app/services/vector_search_service.py:42-43,95` — RRF 融合与权重，不比较绝对分数
- `tests/eval/run_eval.py:456` — Cross-encoder 仅在评测选项中
- IP / L2 / 曼哈顿 / Jaccard / 生产 Cross-encoder：项目中没有，不能硬套

#### Q15 · 听不懂，说人话？

**回答**

把每段文本想象成高维空间里的一支箭。Embedding 模型负责把文字变成箭的方向，余弦相似度只看两支箭指向是否一致，不管箭有多长。「怎么重置登录密码」和「忘记密码怎么找回」用词不同，但训练好的模型会把它们指向差不多的方向；「密码」和「今天天气」会指向两个不相干的方向。检索就是拿用户这一问当作箭，去库里找方向最接近的几支。

这个比喻到此为止，不能当成保证。方向接近不接近，完全由 Embedding 模型训练时见过什么决定，余弦公式本身不懂语义。我们这个库用的是 DashScope `text-embedding-v4`，1024 维，通用语料训出来的——它没专门学过 AIOps 运维术语，所以「CPU 打满」和「负载高」能不能对上，靠猜没用，只能拿评测集跑。项目里这件事是落在 `tests/eval/` 的：Hit@K、Recall@K、MRR 三个指标是真实现的，跑一遍就知道哪些问法召不回来。

所以我会这样收口：比喻用来解释机制，评测用来证明有效。项目里也正因为纯向量不够，才在上面叠了 BM25 关键词路，两路 RRF 融合——比喻里的「方向」解决不了「文档里就写着这个词」这类精确匹配。

**项目证据**
- `app/services/vector_embedding_service.py:27-28` — `model="text-embedding-v4"`, `dimensions=1024`，默认参数
- `app/services/vector_embedding_service.py:173-174` — 模块级单例实例化，`dimensions=1024`
- `app/core/milvus_client.py:198` — `"metric_type": "COSINE"`，注释写着「文本向量检索的最佳实践」
- `app/services/vector_search_service.py:42-43,95-99` — 向量 0.7 / BM25 0.3，`EnsembleRetriever` 做 RRF 融合，补的就是纯向量补不了的精确词
- `tests/eval/metrics.py` — Hit@K / Recall@K / MRR 已实现
- `tests/eval/golden_set_expanded.jsonl` — 83 条口语化提问，`tags` 里带 `colloquial`，就是用来验这类「说法不一样」的场景
- 领域微调 Embedding：项目中没有，不能硬套

#### Q16 · 余弦相似度与欧几里得距离在工程应用中的具体区别是什么？

**回答**

先说结论：只要向量是单位长度的，这两个排序完全一样，选哪个是工程问题不是数学问题；一旦不归一化，两者就不等价。

数学上，对 `‖x‖ = ‖y‖ = 1`：

```
‖x − y‖² = ‖x‖² + ‖y‖² − 2·xᵀy = 2 − 2·cos(x, y)
```

余弦从大到小、点积从大到小、欧氏距离从小到大，三种排法给出同一个名次表。这时候该看的是向量库索引支持哪种 metric、哪种实现更快、以及历史配置是什么。

不归一化就不一样了。余弦把模长除掉，只留方向，适合「只关心说的是不是一件事」；欧氏和点积会把模长带进来。如果模型训练时故意让模长携带置信度或热度信号，强行归一化反而是丢信息。

两个常见误区要纠正。一是别假定所有 Embedding API 都返回单位向量，得查模型文档或在代码里验；二是别以为欧氏「要开方所以慢」——排序时直接比平方距离就行，开方是单调的，ANN 库也都做过优化。

回到这个项目，我得把边界说清楚。库里建索引写死 `COSINE`，维度 1024；但代码里**没有**任何显式归一化步骤——`grep normalize` 命中的只有 `text_cleaner` 的换行/空白规整和 `vector_index_service` 的路径规整，跟向量无关。也就是说我们是直接信任 DashScope 返回向量的性质，既没自己做 L2 归一化，也没跑过 COSINE / L2 / IP 的对比评测。真实情况是「选了 COSINE 并且它能用」，不是「比过之后选的 COSINE」。要补的话，`tests/eval/run_eval.py` 已经有 `--compare` 和权重参数，加一个 metric 维度的对照跑是最小改动。

**项目证据**
- `app/core/milvus_client.py:198-199` — `metric_type=COSINE`, `index_type=HNSW`
- `app/core/milvus_client.py:49` — `VECTOR_DIM: int = 1024`；`:162` — 建 schema 时 `dim=self.VECTOR_DIM`
- `app/services/vector_embedding_service.py:28,174` — `dimensions=1024`，入库与查询共用同一个实例，维度天然一致
- `app/services/text_cleaner.py:78,90` — `_normalize_newlines` / `_normalize_whitespace`，是文本清洗不是向量归一化
- `app/services/vector_index_service.py:155-156` — `normalized_path`，是路径规整
- `tests/eval/run_eval.py:459,461-462` — `--compare` / `--vector-weight` / `--bm25-weight`，已有对照跑的骨架
- 显式 L2 归一化、metric 对比评测、模长携带信号：项目中没有，不能硬套

#### Q17 · 项目中一共使用了几个模型？分别是什么？

**回答**

两个。生成侧 `qwen-max`，向量侧 DashScope `text-embedding-v4`（1024 维），都走 DashScope 兼容 OpenAI 的接口。

原帖这一题的答案报了四类模型——BGE-large-zh-v1.5、BGE reranker、微调 BERT-base 做意图路由、Qwen-72B 或 Claude 3.5 主模型。我这个仓库里这四样一个都没有，我不能照抄。逐条说：

- **BGE-large-zh-v1.5**：没有。Embedding 是 DashScope API 调的 `text-embedding-v4`，不是本地 BGE。
- **BGE reranker**：没有在任何线上路径上。`.hf_cache/hub/models--BAAI--bge-reranker-v2-m3/` 这个权重目录确实躺在仓库里，`tests/eval/run_eval.py` 也有 `--retriever local-rerank` 这个评测选项，代码里还有三处「以后接 rerank」的注释——但检索主链路 `vector_search_service.retrieve_documents` 从头到尾没有精排这一步。下过权重、留过评测入口，不等于线上有 rerank，这两件事我要分开讲。
- **微调 BERT-base 意图路由**：没有。没有路由层，也没有分类模型。
- **Qwen-72B / Claude 3.5**：没有。就 `qwen-max` 一个，`config.py` 里 `dashscope_model` 和 `rag_model` 都是它。

构造入口有两个，容易被误读成两个模型：`llm_factory.create_chat_model` 走 `ChatOpenAI` 打 DashScope 兼容端点，`create_qwen_model` 延迟导入 `langchain_qwq.ChatQwen`（留着是为了它对 `reasoning_content` 的处理）。两条路的模型名都从同一份 config 读，落地还是 `qwen-max`。

所以按真实情况，这题的价值不在数量，在于承认分层选型没做：路由、精排这两层该有的地方现在是空的，代价是每个请求都直接吃大模型的延迟和成本，也没有 Cross-encoder 兜召回质量。这是我能说清楚的缺口，比报四个模型名更经得起追问。

**项目证据**
- `app/config.py:46` — `dashscope_model: str = "qwen-max"`
- `app/config.py:129` — `rag_model: str = "qwen-max"`，注释「使用快速响应模型，不带扩展思考」
- `app/config.py:47` — `dashscope_embedding_model: str = "text-embedding-v4"`
- `app/config.py:45` — `dashscope_api_base`，兼容 OpenAI 模式端点
- `app/services/vector_embedding_service.py:12,104-106,154-156` — `DashScopeEmbeddings`，`embed_documents` / `embed_query` 都传同一个 `model` + `dimensions`
- `app/core/llm_factory.py:64` — `create_chat_model`；`:105` — `create_qwen_model`，延迟导入 `ChatQwen`
- `tests/eval/run_eval.py:456` — `--retriever` 默认 `cheat`，`local-rerank` 只是评测选项
- BGE-large-zh-v1.5 / 线上 reranker / BERT 意图路由 / Qwen-72B / Claude 3.5：项目中没有，不能硬套

#### Q18 · 如何控制模型的幻觉问题？

**回答**

项目里是三层，从「敢不敢答」到「答完有没有人查」。

第一层，没证据就不硬答。`validate_answer_node` 进来先看 `deduped_documents`：空的话直接出保守回答，并且把 `needs_second_retrieval` 置 True，不让模型拿零证据编。答案本身为空也走同一条路。这两个 guard 是纯结构检查，不花 LLM 调用。

第二层，证据质量。检索是向量 + BM25 两路 RRF 融合，`candidate_k = max(top_k*3, top_k, 10)` 先拉宽再收窄；rewrite 节点把一问拆成 3 条子查询并行召回，再 dedup。进 Prompt 的上下文有 token 预算截断，`GENERATE_SYSTEM_PROMPT` 明确要求只用给定证据。

第三层，答完由独立一次 LLM 调用做证据校验，温度 0，输出 `groundedness_score` 和 `coverage_score`，门槛分别是 0.75 和 0.60。不过关就换成 `_build_fallback_answer` 的保守说法。两个分数和 `is_bad_case` 标记都落 `chat_run_traces` 表，能事后捞出来看。

有两个设计取舍值得单说。一是校验器的失败方向选了 fail-open：`enable_answer_validation=false` 或者熔断打开时，答案照常发出，只是标 `_unvalidated_validation` 并记 `DegradeReason.FEATURE_DISABLED`。为什么不 fail-closed？因为 fail-closed 会让每个答案都变成「当前证据还不够支持直接下结论」，用户以为是自己问题缺证据，实际是我们把质检关了——关掉一道校验不该让答案消失。二是降级原因刻意选 `FEATURE_DISABLED` 而不是 `LLM_ERROR`，后者会把值班同学送去查模型配额和鉴权，而那边一切正常，排障就卡在这。

缺口我也讲清楚：写操作的「计划→预览→审批→执行→回读」这套项目里没有，因为当前没有任何会改动外部系统的工具；`tests/eval/metrics.py` 里 `faithfulness_placeholder` 和 `answer_relevance_placeholder` 都还是 `NotImplementedError`，所以幻觉率目前只有线上 `is_bad_case` 的统计，没有离线基准。降温度这种手段项目里用了（校验器 temperature=0.0），但它只压采样随机性，压不住参数知识错误，我不会把它当主要手段讲。

**项目证据**
- `app/agent/rag_v2/nodes.py:334` — `validate_answer_node`
- `app/agent/rag_v2/nodes.py:32-33` — `MIN_GROUNDEDNESS_SCORE = 0.75`, `MIN_COVERAGE_SCORE = 0.60`
- `app/agent/rag_v2/nodes.py:540,558,592` — `_default_validation` / `_unvalidated_validation` / `_build_fallback_answer`
- `app/agent/rag_v2/nodes.py:28-30` — `NUM_SUB_QUERIES = 3`, `RETRIEVE_TOP_K = 4`, `FINAL_TOP_K = 6`
- `app/agent/rag_v2/nodes.py:50` — `GENERATE_SYSTEM_PROMPT`；`:613,631,637` — `_format_context` / `_estimate_tokens` / `_truncate_to_token_budget`
- `app/services/vector_search_service.py:84,95-99` — `candidate_k` 放宽 + RRF 融合
- `app/models/chat_run_trace.py:43-46` — `validation` / `coverage_score` / `groundedness_score` / `is_bad_case` 落库
- `tests/eval/metrics.py` — `faithfulness_placeholder`、`answer_relevance_placeholder` 均 `raise NotImplementedError`
- 写操作审批链、Human-in-the-loop、对抗样本集：项目中没有，不能硬套

#### Q19 · 如何观察模型召回了哪些块？

**回答**

项目里这件事是靠 span 落盘做的，不是靠翻日志。

每个节点在图注册处就被 `instrument_node` 包了一层，跑完写一条 span：节点名、状态、开始时间、耗时毫秒、payload。落到 `chat_run_spans` 表，`trace_id` 串起一次请求，`request_id` 也带着，跟 HTTP 层的 X-Request-ID 对得上。检索节点的 payload 里带召回条数，所以「四条子查询各自召回了几条、各花了多久」是能直接查出来的。

这里有个细节是并行分支的关键：`record_span` 只 `append` 不 `set`。`retrieve_each` 是 `Send` 扇出成 N 个并行实例的，如果用 `.set()` 写 ContextVar，`copy_context()` 之后各分支改的是自己那份，主流程收不到；`append` 改的是同一个 list 对象，所以能穿回来。计时用 `perf_counter` 而不是两个 datetime 相减——后者遇到 NTP 校时能算出负数耗时。埋点放在注册处而不是每个节点内部，是因为五个节点各写一遍计时代码是重复，而且新加节点一定会忘。

具体召回了哪些块，走 `used_documents`：`chat_run_traces.used_documents` 是 JSON 列，API 响应里也带这个字段。这里做了瘦身——每条只留 200 字符的 `excerpt`，并且用 `excerpt_only: True` 如实标注这是截断过的，不假装是全文。`score` 拿不到真值时存 `None` 而不是编一个。裁剪点刻意放在写库之前，理由是全文进库既占空间又把知识库原文复制了一份到 trace 表里。

有两条边界。`flush_spans` 失败只记 warning 不抛——答案已经生成、trace 已经落库，不该因为写不进耗时明细而让请求失败。这是有意的旁路语义。另外 `span_scope` 遇到异常记 ERROR 状态和 error_code，然后原样放行异常，不改控制流。

缺的部分：没有检索审计面板 UI，只能查库；span payload 里没有 Embedding 版本号和索引版本号字段，模型换代之后老 trace 没法归因；`excerpt` 的 200 字符截断在事实上起到了限制敏感原文外泄的作用，但项目里**没有**按租户授权、没有保存期限策略，这不是脱敏设计，只是恰好截短了，我不会把它讲成隐私方案。

**项目证据**
- `app/agent/rag_v2/instrumentation.py:117` — `instrument_node`，注册处统一埋点；`:131,140` — `span_scope` 包住同步/异步两条路
- `app/agent/rag_v2/instrumentation.py:36,39,92,149` — `_SPAN_HINT_KEY`、`_derive_payload`、`_derive_degradation`、`_apply`
- `app/core/span_context.py:85-110` — `record_span`，注释写明「只 append，不 set —— 这是并行分支下不丢数据的前提」
- `app/core/span_context.py:112+` — `span_scope` 类，`perf_counter` 计时，注释说明为何不用 datetime 相减
- `app/core/span_context.py:160+` — `flush_spans`，失败只 warning + rollback，返回 0
- `app/core/span_context.py:76` — `reset_span_collection`
- `app/models/chat_run_span.py:67-99` — `chat_run_spans` 表：`trace_id` / `request_id` / `node` / `status` / `started_at` / `duration_ms` / `payload`，node 和 created_at 都建了索引
- `app/models/chat_run_trace.py:40-42` — `sub_queries` / `retrieved_count` / `used_documents` 落库
- `app/core/used_documents.py:42,46` — `EXCERPT_MAX_CHARS = 200`，`_SLIM_MARKER = "excerpt"` 及选它的理由
- `app/core/used_documents.py:104-134` — `_normalize_one`，`excerpt_only: True` 如实标注截断；`:146` — `_pick_score`
- `app/agent/rag_v2/graph.py:44` — `_fanout_to_retrieve` 用 `Send` 扇出，每个实例各记一条 span
- `app/api/chat_v2.py:189,204,224,254` — `used_documents` 透传到响应
- 审计面板 UI、Embedding/索引版本字段、按租户授权与日志保存期限：项目中没有，不能硬套

#### Q20 · 简历别的项目随便问了问主要工作

**回答**

这题看着随意，实际在验简历真不真。我的答法是 60 到 90 秒说清五件事：业务问题是什么、我负责到哪、关键技术决策为什么这么选、结果怎么衡量、有过一次什么失败以及从里面改了什么。

STAR 可以用，但技术岗必须把 Action 说到具体位置。不能说「负责模型优化」，要说改了哪个模块、处理的是哪类数据、为什么没选替代方案、离线和线上指标各自怎么变、上线风险怎么控。团队成果和个人贡献要分开，说「我们把延迟降了一半」和「我把检索这一段从 3s 压到 800ms」是两回事。

如果那个项目跟 Agent 岗不直接相关，我会主动搭迁移关系而不是硬包装：后端项目练的是接口、并发和可观测性，搜索项目练的是召回和排序，数据项目练的是治理和评测。这些怎么迁到 Agent 系统上，比把所有经历都说成大模型项目可信。

没有量化指标的时候我讲可验证的交付物——上线的模块、覆盖的场景、测试集规模、省掉的人工步骤，不编百分比。这个仓库本身就有能拿出来的东西：`docs/learning/` 下 27 篇专题文档，从 Day1 边界与架构一路到熔断器与运行时降级开关，每篇都对着真实代码写；`tests/eval/` 有 83 条 golden set 和三个已实现的检索指标。面试官往下挖的时候，我能落到具体文件、具体行、具体的 bad case。

**项目证据**
- `docs/learning/01-day1-边界与架构.md` 到 `docs/learning/27-异步任务与数据层面试题.md` — 27 篇专题文档，可验证交付物
- `docs/learning/14-错误分类与降级原因体系.md`、`21-熔断器与运行时降级开关.md`、`16-LLM超时与统一构造路径.md` — 三个可深挖的技术决策记录
- `tests/eval/golden_set_expanded.jsonl` — 83 条评测样本
- `tests/eval/metrics.py` — Hit@K / Recall@K / MRR
- 简历上另一个项目的代码：不在本仓库内，无法引行号，只讲答题方法

#### Q21 · 算法题：Leetcode 143. 重排链表

**回答**

把 `L0→L1→…→Ln` 重排成 `L0→Ln→L1→Ln-1→…`，原地做，三步：快慢指针找前半段尾节点、反转后半段、两段交替合并。

```python
def reorder_list(head: ListNode | None) -> None:
    if head is None or head.next is None:
        return

    # 1. 找中点。循环条件用 fast.next / fast.next.next，
    #    保证偶数长度时 slow 落在前半段最后一个，前半段不短于后半段。
    slow = fast = head
    while fast.next is not None and fast.next.next is not None:
        slow = slow.next
        fast = fast.next.next

    # 2. 断链。这一步必须做，否则第 3 步合并会成环。
    second = slow.next
    slow.next = None

    # 3. 反转后半段
    prev = None
    while second is not None:
        nxt = second.next
        second.next = prev
        prev = second
        second = nxt

    # 4. 交替合并。先把两个 next 都存下来再改指针，否则会丢链。
    first, second = head, prev
    while second is not None:
        next_first, next_second = first.next, second.next
        first.next = second
        second.next = next_first
        first, second = next_first, next_second
```

时间复杂度 O(n)：找中点、反转、合并各扫一遍，3n 仍是线性。空间复杂度 O(1)：全靠指针腾挪，没有额外容器。用数组装节点再双指针也是 O(n) 时间，但空间退化成 O(n)，而且正好绕开了这题想考的断链、反转、交错合并三件事，面试里给这个解法基本等于没答。

两个坑我会主动说。第一个是 `slow.next = None` 不能省，不断链的话合并到最后会把后半段的尾节点指回前半段，形成环，`while second is not None` 就永远不结束。第二个是合并循环里必须先把 `next_first` 和 `next_second` 两个后继都存下来再改指针，改完再取就已经取到新链上去了。边界至少测空链表、单节点、两节点、奇数长度、偶数长度五种。

**项目证据**
- 链表数据结构与算法代码：项目中没有，不能硬套。这是纯算法题，仓库是 FastAPI + LangGraph 的 RAG 服务，全库没有 `ListNode`、没有手写链表、没有指针反转类代码，最近的只有 Python 内置 `list` 的切片操作（如 `app/services/conversation_memory_service.py` 里的 `pending_messages[:-recent_count]`），那是语言容器，跟链表指针操作不是一回事，不能拿来冒充

#### Q22 · 你平时调试代码怎么调试的？

**回答**

我的顺序是稳定复现 → 缩小范围 → 加观测 → 验假设 → 修完补回归测试。不同层次用不同工具，不是一上来就上重武器，也不是全程只靠 print。

算法题和单文件脚本，最快的就是构造最小输入 + 打印关键状态，跟手推的结果对。业务代码就换成结构化日志加测试：这个仓库的日志系统在 `app/utils/logger.py`，`_patch_request_id`（30 行）给每条日志补 `rid` 字段，用 `setdefault` 而不是直接赋值——一是保留调用方 `logger.bind(rid="job-123")` 的显式意图，二是防 KeyError，因为格式串里写了 `{extra[rid]}`，某条记录缺这个键 loguru 就会抛，而日志系统自己抛异常是最糟的故障，它会盖掉真正的错误。控制台带颜色给人读，文件走 JSON 行按天轮转给机器查（`setup_logger`，42 行）。

链路问题靠 span。`app/core/span_context.py:85` 的 `span_scope` 给一段代码计时，退出时记一条 span，异常路径记 ERROR 带 error_code 然后**原样放行异常**——span 是旁路观测，不改控制流。计时用 `time.perf_counter` 而不是两个 datetime 相减，后者受 NTP 校时影响能算出负数耗时。这些 span 通过 `flush_spans`（160 行）落到 `chat_run_spans` 表，写失败只记 warning 不抛，因为答案已经生成了，不该因为写不进耗时明细而整个请求失败。排障时查 `trace_id` 就能看到每个节点各花了多久、哪个降级了。

跑测试和检查走 Makefile：`make test`（627 行）、`make test-quick`（631）、`make lint`（606）、`make type-check`（617）、`make check-all`（635）。pytest 配了 `asyncio_mode = "auto"`（`pyproject.toml:142`），异步测试不用每个都挂装饰器。

AI 能帮读堆栈、生成用例、提排查假设，但它拿不到我的运行环境和真实数据，结论得我自己用日志和测试验。

**项目证据**
- `app/utils/logger.py:30` — `_patch_request_id`，setdefault 防 KeyError 的两条理由写在 docstring 里
- `app/utils/logger.py:42` — `setup_logger`，控制台给人读 / JSON 行给机器查
- `app/core/span_context.py:85` — `span_scope`，perf_counter 计时、异常原样放行
- `app/core/span_context.py:160` — `flush_spans`，写失败只 warning
- `app/agent/rag_v2/instrumentation.py:117` — `instrument_node`，在注册处统一挂观测
- `pyproject.toml:128-142` — pytest 配置，`asyncio_mode = "auto"`、`testpaths = ["tests"]`
- `Makefile:606,617,622,627,631,635,652` — lint / type-check / security / test / coverage 入口
- `tests/` — 35 个测试文件
- `pdb` / `breakpoint()` / IDE 断点配置：项目中没有，不能硬套。全库 grep 不到任何 `pdb`、`breakpoint()` 或 `.vscode/launch.json`，这个仓库的调试手段实际是日志 + span + 测试三件套，断点调试只是我个人习惯，代码里没有痕迹

#### Q23 · 不对吧，最简单的方式不是打印一下吗？

**回答**

对，算法题里最简单有效的就是打印。上一题的重排链表，我实际会在找完中点、反转完后半段、每次合并后各打一次链表，错在哪一阶段立刻就看出来了。

但打印要围着假设设计，不是把所有变量全倒出来。带标签的输出才有用，`phase=reverse curr=3 next=4` 这种，扫一眼就知道当前在哪一步、指针指哪。全打的话日志越多越难看，等于没打。

这个思路在服务代码里的延伸就是结构化日志，本质是同一件事——把关键状态标记出来。仓库里 `app/agent/rag_v2/nodes.py` 每个节点入口都有 `logger.info(f"[rag_v2.validate] 开始校验答案, context={len(docs)}")` 这种，前缀标节点、后面带关键数量；`app/services/vector_search_service.py:92-94` 在 BM25 语料为空时打 `"BM25 语料为空，降级为纯向量检索"`，出问题时一看日志就知道走的是混合检索还是降级路径，不用猜。这就是加了标签、带上下文的 print。

打印的边界也得说清楚。并发时序问题，print 本身会改变时序，多线程下输出还会交织；性能问题，print 的 IO 开销可能把要测的东西盖掉；生产环境，正文可能带用户隐私和企业敏感信息，不能随便打。这三种情况要换 span、指标、trace。仓库里之所以做 `span_scope` 而不是在每个节点写两行 print，就是因为 `retrieve_each` 会 fan-out 成多个并行实例，print 出来的四条分支交织在一起，根本对不出每条各自花了多久。

这句追问考的不是工具高低，是会不会从最低成本的观测手段开始。

**项目证据**
- `app/agent/rag_v2/nodes.py:334` 起 — `validate_answer_node` 入口 `logger.info` 带节点前缀和证据条数
- `app/services/vector_search_service.py:92-94` — 降级路径的显式日志，一眼区分混合检索与纯向量
- `app/agent/rag_v2/graph.py:25-34` — 注释写明并行分支下逐节点计时的动机（四条分支交织，对不出各自耗时）
- `app/core/span_context.py:85-100` — 并行场景下只 append 不 set，这是不丢数据的前提
- `print()` 调试语句：项目中没有，不能硬套。全库业务代码里没有留下的 print，都是 loguru

#### Q24 · 最近了解的 AI 内容？

**回答**

我最近在看的是编码 Agent 的重心从「模型能不能写对代码」转到「Harness 能不能稳定交付」。决定这类系统可用性的，越来越不是单次补全的质量，而是它能不能读到仓库上下文、检索到对的代码、规划出修改、跑测试、处理失败、守住权限、留下可审计的变更记录。同一个模型放进不同 Harness，任务成功率能差很多。

这个判断跟我自己项目里踩的坑对得上。模型换成更强的不解决的问题，靠工程层解决了：LLM 调用没有统一超时，我把三处分散构造收拢到 `app/core/llm_factory.py`，顺手处理了 langchain-openai 那个 `request_timeout` 默认 `None` 会无条件盖掉 OpenAI SDK 默认超时的坑；单次超时不等于总预算，所以 `app/agent/rag_v2/service.py:50-75` 又加了一层 `asyncio.timeout(config.chat_total_budget_seconds)` 包住整个 graph，预算耗尽时不往上抛，因为已经有部分结果可以交付，抛出去等于让用户等满预算还什么都拿不到；工具调用失败要退避重试，`app/agent/mcp_client.py:18-73` 的 `retry_interceptor` 做指数退避，重试全失败后 **return** `CallToolResult(isError=True)` 而不是 raise，让模型看到工具报错继续决策，而不是整条链断掉。这三件事没有一件是模型能力问题。

一个还没解决的问题是评测。轨迹级评测——工具选对了没、失败后恢复得好不好——比单点指标难做太多。我这边检索侧的 Hit@K / Recall@K / MRR 已经实现了（`tests/eval/metrics.py`），但 faithfulness 和 answer relevance 还是 `NotImplementedError` 占位，工具选择和恢复能力的评测更是完全没有。所以我讲这个方向的时候不会说自己做完了，只说看到了缺口在哪。

**项目证据**
- `app/core/llm_factory.py` 模块 docstring — 三处分散构造缺超时的记录、langchain-openai `request_timeout` 覆盖 SDK 默认值的坑
- `app/agent/rag_v2/service.py:50-75` — 总预算层，超时不抛、交付部分结果
- `app/agent/mcp_client.py:18-73` — 工具重试拦截器，指数退避（65 行），失败 return `isError=True`（71-73）而非 raise
- `tests/eval/metrics.py` — Hit@K / Recall@K / MRR 已实现；`faithfulness_placeholder` / `answer_relevance_placeholder` 抛 `NotImplementedError`
- 轨迹级评测（工具选择正确率、失败恢复率）：项目中没有，不能硬套。`tests/eval/` 只评检索，没有任何针对工具调用序列的评测数据或指标

#### Q25 · 讲一下 Claude Code 架构

**回答**

先说清边界：我能讲的是从公开文档和产品行为能观察到的架构，不能声称看过全部源码，所以不会给出具体工具数量、代码行数或者「绝对无共享状态」这类内部结论。

从可观察的部分看，它更适合理解成一个编码 Agent Harness，而不是模型外面套了个终端。主线是迭代式 agent loop：读用户目标和项目上下文 → 模型决定下一步工具调用 → 权限层审核 → 工具执行 → 结果作为 observation 回到模型 → 直到完成、失败或耗尽预算。按模块拆六块：Context 层整合用户指令、项目规则、相关代码、当前修改和压缩后的历史；Agent loop 让模型基于当前状态决定检索、读写、执行命令、测试或派子 Agent；Tool 层用明确的输入输出契约暴露能力；Permission/Sandbox 约束高风险命令、文件范围和外部访问，支持人工确认；State 层保存进度和修改状态，支持压缩、checkpoint、回退；扩展层用项目规则、Hooks、MCP、子 Agent 接上仓库规范和企业系统。

拿我的项目对一遍，这六块只有一半有对应物，我不会说自己实现了一个 Harness。Tool 层有：`app/tools/knowledge_tool.py:13-14` 用 `@tool(response_format="content_and_artifact")` 声明检索工具，`app/tools/time_tool.py:10-11` 是时间工具，外部工具走 MCP，`app/agent/mcp_client.py` 连 `mcp_servers/cls_server.py`（日志检索）和 `mcp_servers/monitor_server.py`（指标查询），`app/services/rag_agent_service.py:131` 把本地工具和 MCP 工具拼成 `all_tools` 交给 `create_agent`（133 行），MCP 加载失败时 `self.mcp_tools = []`（128 行）降级成只有本地工具，不让整个 Agent 起不来。Context 压缩有：`app/services/conversation_memory_service.py` 做滚动摘要，完整原文只留在 PostgreSQL 做审计，模型上下文只吃摘要加未压缩的尾部。State 有一点：`app/services/rag_agent_service.py:107` 的 `MemorySaver`，但那是进程内的对话续接，不是可恢复的任务 checkpoint。

缺的三块得明说。Permission/Sandbox 完全没有，工具调用没有任何审批、白名单或沙箱隔离，`mcp_servers/` 下的工具是只读查询所以暂时没出事，但这是场景侥幸不是设计。子 Agent 派发没有，`app/agent/rag_v2/graph.py` 是固定五节点的 DAG，跑完即 END。失败回退/回滚没有，`compile()` 不带 checkpointer，中断了只能重跑。

**项目证据**
- `app/tools/knowledge_tool.py:13-14`、`app/tools/time_tool.py:10-11` — 本地工具声明
- `app/agent/mcp_client.py:84,130,150,161` — MCP client 与拦截器注册
- `mcp_servers/cls_server.py:104,136,169,212,346`、`mcp_servers/monitor_server.py:124,277` — 外部工具实现，均为只读查询
- `app/services/rag_agent_service.py:101,104,123-128,131,133,142-143` — 工具注册、MCP 加载失败降级为空列表、`create_agent` 装配
- `app/services/rag_agent_service.py:107` — `MemorySaver`，进程内对话续接
- `app/services/conversation_memory_service.py` 模块 docstring — 原文留库审计、上下文只吃摘要 + 尾部
- Permission / Sandbox / 人工确认：项目中没有，不能硬套。全库 grep 不到 permission、sandbox、approval 相关代码，工具调用是直通的
- 子 Agent 派发 / 任务回滚 / 持久化 checkpoint：项目中没有，不能硬套。`app/agent/rag_v2/graph.py:48,50` 是固定 DAG + 无 checkpointer 编译
- Claude Code 内部源码：不可引用，只按公开文档与产品行为作接口层分析

#### Q26 · 建议下去再补一点 Harness 的知识

**回答**

这条反馈我接受，也顺着说一下我理解的 Harness 边界。它不是某个框架，也不等于放三个 markdown 文件在仓库根目录，而是围着模型的整套运行与治理环境，决定 Agent 怎么感知、怎么行动、受什么约束、失败怎么恢复、怎么被评估。规则文件只是 Context/Policy 的一种载体。

按组件对着我的项目盘一遍，有的和没的分得很清楚。

有的：上下文压缩，`conversation_memory_service` 的滚动摘要，CJK 加权估 token、快照游标、摘要失败退回最近窗口而不阻断主聊天。工具 schema 管理，`@tool` 装饰器加 MCP 协议。超时重试，`llm_factory` 的单次超时加 `service.py` 的总预算，MCP 侧的指数退避重试。熔断，`validate_answer_node` 里熔断器守在质检 LLM 调用外面，`CircuitOpenError` 被同一个 except 接住走 fail-open。日志 tracing，request_id 全链路加 span 落库。评测器，`tests/eval/` 的 golden set 加三个检索指标。

没有的：执行沙箱、权限审批、模型路由、人工接管、工具检索（工具全量塞进 prompt，没做按需检索）、可恢复的状态机 checkpoint、轨迹级评测。这七项里前四项是安全相关，后三项是规模相关，我的场景现在还撑得住，但都不是做完了。

要补的话我会围着真实问题学，而不是背配置文件名：工具多到几十个之后怎么做检索而不是全塞 prompt；命令怎么最小权限执行；长任务怎么 checkpoint 才能真的恢复而不是只存了个快照；测试失败怎么回滚已经改掉的文件；怎么用轨迹数据评测工具选择和错误恢复。这些机制想明白了，比记住某几个文件名有跨周期价值。

**项目证据**
- `app/services/conversation_memory_service.py:48-54,93-131,163-186` — 上下文压缩、快照游标、摘要失败不阻断
- `app/core/llm_factory.py` + `app/agent/rag_v2/service.py:50-75` — 双层超时
- `app/agent/mcp_client.py:18-73` — 工具调用重试
- `app/agent/rag_v2/nodes.py:334` 起 — 熔断器 + fail-open 质检
- `app/core/span_context.py`、`app/utils/logger.py:30` — tracing 与 request_id 注入
- `tests/eval/golden_set_expanded.jsonl`、`tests/eval/run_eval.py` — 评测闭环
- 执行沙箱 / 权限审批 / 人工接管 / 模型路由：项目中没有，不能硬套
- 工具按需检索：项目中没有，不能硬套。`app/services/rag_agent_service.py:131` 是把 `all_tools` 全量交给 `create_agent`，没有任何按 query 筛工具的逻辑
- 轨迹级评测：项目中没有，不能硬套

#### Q27 · 反问业务

**回答**

反问要问出岗位真正解决什么问题、错了成本多大、交付边界在哪，而不是问「团队做什么」。如果方向是安全风险类 AI，我会问五个。

第一，当前主场景是内容审核、违规识别、风险研判，还是给内部分析人员做 Agent 辅助？模型输出是建议，还是会直接触发处置动作？这决定了自动化权限和人工复核的强度。我自己项目里工具全是只读的日志和指标查询（`mcp_servers/cls_server.py`、`monitor_server.py`），所以没有审批层也没出事；如果这边的 Agent 会写、会封号、会拦流量，权限设计就是第一优先级，我想先知道它落在哪一档。

第二，团队看哪些指标？安全场景不能只看总体准确率，误杀率、漏放率、严重风险召回、校准度、人工复核成本得分开看，不同错误的业务代价完全不对称。我这边对应的是把检索状态做成三态而不是「静默返回空列表」，因为空列表和检索失败在下游是两种完全不同的处理，混在一起就是在骗自己。

第三，数据和评测怎么建？有没有持续更新的对抗样本、灰产变体、线上回放集？LLM Judge 是靠规则、专家标注还是真实处置结果校准？我这边 golden set 只有 83 条、faithfulness 还是占位实现，所以对这个问题特别有体感，想知道成熟团队是怎么把评测集养起来的。

第四，工作比例怎么分？Agent 工程、模型训练、RAG/知识建设、业务规则、客户交付各占多少？项目里的共性能力能不能沉淀成平台，还是长期做一次性需求。

第五，高风险动作前有哪些权限、审计和 human-in-the-loop 机制？误判之后怎么回放、归因、回滚？

这五个问题既是我判断岗位的依据，也是在说明我理解安全 Agent 的核心不是会调工具，是每个判断有证据、每个动作有权限、每次错误能追溯。

**项目证据**
- `mcp_servers/cls_server.py`、`mcp_servers/monitor_server.py` — 现有工具全为只读查询，这是「暂时不需要审批层」的实际原因
- `app/models/rag_state.py` / `app/agent/rag_v2/state.py` 中的 `RetrievalStatus` 三态 — 空结果与失败分开，对应「不同错误代价不对称」
- `tests/eval/golden_set_expanded.jsonl` — 83 条样本，规模不足是真实短板
- `tests/eval/metrics.py` — faithfulness / answer relevance 仍为 `NotImplementedError`
- 权限审批 / human-in-the-loop / 误判回放归因：项目中没有，不能硬套。这也是我把它列成反问的原因——自己没做过，想知道对方怎么做

---

## 四、哔哩哔哩 AI 应用岗 Agent 开发一面

来源：【三年面试五年模拟】2026-08-18_哔哩哔哩_AI应用岗Agent开发一面面经（含完整答案）
原题位置：`docs/learning/agent_interviews_2026-06-06_to_2026-09-06.md:665-964`，共 18 题，按原帖顺序。

### 面经 01 · 2026-08-18

#### Q1 · 简单介绍一下自己的背景，以及选择 AI Agent 方向的原因。

**回答**

我会用 60 到 90 秒走完「背景 → 证据 → 为什么转 Agent → 和岗位怎么对上」这条线，不报课程名也不报模型名。

背景说清学历阶段、主技术栈、最相关的那个项目。证据部分讲我在这个仓库里具体负责什么：一条多路召回的 RAG 链路，`app/agent/rag_v2/` 下面五个节点——改写、并行检索、去重、生成、证据校验，用 LangGraph 的 `StateGraph` 编排；以及一整套错误治理，`app/core/errors.py` 里 8 个应用异常类加 `to_app_error` / `wrap_llm_exception` 两条转换路径，`app/core/circuit_breaker.py` 的三态熔断器，`app/core/metrics.py` 的 Prometheus 埋点。

为什么选 Agent 方向，我不说「它是风口」。真实理由是：这个方向把我原来会的东西和模型能力接上了。我在这个项目里花时间最多的地方根本不是调 Prompt，是「一个并行检索分支挂了要不要让整张图崩」（`retrieve_each_node` 兜住异常，`dedup_node` 汇总判定降级原因）、「总预算耗尽要不要把已经生成的部分吐给用户」（`asyncio.timeout` 包 `astream`，超时不抛、交部分结果）、「HTTP 状态码要不要说真话」（改成 bare `raise`，由全局异常处理器按异常类型裁决 504/502/500）。这些都是后端工程问题，只是承载物换成了模型调用。

结尾我会说：我选 AI Agent，是因为它跟我已有的工程能力是连续的——状态、并发、超时、幂等、可观测性这些我能接着用，再补上模型调用、RAG、工具编排和评测，就能把模型能力变成可交付的流程。项目里的数字我只报能查到的：`tests/eval/golden_set_expanded.jsonl` 83 条评测样本、`docs/learning/` 27 篇专题文档，没有线上流量数据我就明说是离线评测。

**项目证据**
- `app/agent/rag_v2/graph.py:36-52` — 五节点 `StateGraph`，`add_conditional_edges` 做 fan-out，`compile()` 不带 checkpointer
- `app/core/errors.py:81-220` — `AppError` 及 8 个子类；`328` `to_app_error`；`358` `wrap_llm_exception`
- `app/core/circuit_breaker.py:74-266` — `CircuitState` 三态与 `CircuitBreaker.guard`
- `app/agent/rag_v2/nodes.py:158-212` — `retrieve_each_node` 兜住异常，单分支失败不炸整图
- `app/agent/rag_v2/service.py:50-75` — `asyncio.timeout` + `astream`，超时交部分结果
- `tests/eval/golden_set_expanded.jsonl` — 83 条离线评测样本
- 线上流量 / QPS / 用户量：项目中没有，不能硬套，只讲离线评测口径

#### Q2 · 一个完整的 Agent 系统通常包含哪些核心模块？相比传统 LLM Chatbot，它最大的区别是什么？

**回答**

我按「这个仓库有哪些、缺哪些」来答，不背八层架构图。

有的部分：**上下文构建层**是 `conversation_memory_service.build_context`，滚动摘要加最近窗口，按 Token 预算裁；**编排层**是 `rag_v2/graph.py` 的 `StateGraph`，改写 → 并行检索 → 去重 → 生成 → 校验；**工具层**是两个本地 `@tool` 加 7 个 MCP 工具（`cls_server` 5 个、`monitor_server` 2 个），MCP 侧有 `retry_interceptor` 做指数退避；**知识层**是 Milvus HNSW+COSINE 向量库加 BM25，`EnsembleRetriever` 做 RRF 融合；**验证层**是 `validate_answer_node`，`MIN_GROUNDEDNESS_SCORE = 0.75` / `MIN_COVERAGE_SCORE = 0.60` 两条线卡幻觉；**治理层**是熔断、降级原因枚举、总预算；**可观测层**是 span 落库（`chat_run_spans` 表）加 Prometheus 指标。

缺的部分我直接说：**任务与状态层**没有。图跑完即 `END`，`compile()` 不带 checkpointer，没有任务实体表、没有步骤状态机、没有 checkpoint 恢复。`rag_agent_service` 那条路挂了 `MemorySaver`，但那是对话续接用的内存 checkpointer，不是任务状态持久化。**规划层**也没有——五个节点是编译期写死的固定边，不是模型决定下一步做什么。所以严格讲这个项目是「带验证和降级的 RAG 流水线」，不是自主 Agent，这点我不会含糊。

跟传统 Chatbot 最大的区别，我不说「多了工具」。区别在闭环：Chatbot 是输入 → 一次生成 → 输出，模型说完就结束；Agent 是模型给下一步候选、运行时负责权限和执行、外部环境返回真实 Observation、状态更新后再决定。关键是**终止条件由外部验收，不是模型说完成就完成**。这个项目在验证环节体现了这一点——`validate_answer_node` 是独立的 LLM 调用，拿证据核答案，不通过就换成 `_build_fallback_answer` 的保守回答；但在「多步执行」这一层它确实还是单趟流水线。

**项目证据**
- `app/services/conversation_memory_service.py:93-131` — `build_context`，摘要 + 最近窗口 + Token 预算
- `app/agent/rag_v2/graph.py:36-52` — 编排层，五节点固定边
- `app/tools/knowledge_tool.py:13-14`、`app/tools/time_tool.py:10-11` — 两个本地工具
- `mcp_servers/cls_server.py:104,136,169,212,346`、`mcp_servers/monitor_server.py:124,277` — 7 个 MCP 工具
- `app/agent/mcp_client.py:18-73` — `retry_interceptor` 指数退避
- `app/core/milvus_client.py:198-199` — HNSW + COSINE；`app/services/vector_search_service.py:96-99` — `EnsembleRetriever` RRF
- `app/agent/rag_v2/nodes.py:32-33,334` — 两条质检阈值线与 `validate_answer_node`
- `app/core/span_context.py:160` `flush_spans`、`app/models/chat_run_span.py:67` — span 落库
- 任务实体表 / 步骤状态机 / checkpoint 恢复 / 模型自主规划：项目中没有，不能硬套。`rag_agent_service.py:107` 的 `MemorySaver` 是对话续接，不是任务状态持久化，不能拿它冒充

#### Q3 · 在 Agent 项目中，常见的规划（Planning）、记忆（Memory）、工具调用（Tool Use）和执行模块分别承担什么职责？

**回答**

四个模块里这个项目真正实现了两个半，我按实现程度分开讲。

**Memory** 是实现最完整的。职责是两件事：让当前会话连得上，让跨轮信息不无限膨胀。`conversation_memory_service` 的做法是完整原文只进 PostgreSQL 做审计，喂给模型的只有压缩摘要加未压缩尾部。Token 估算按中日韩字符单独计数、其余按 4 字符 1 token（`estimate_tokens`），每条消息再加 4 的固定开销（`_message_token_cost`）。摘要有游标快照，只压缩游标之后、尾部窗口之前的那一段，不重算全部历史。摘要失败不阻断聊天，`_summarize` 的 except 直接返回空串退回最近窗口——这是我认为记忆模块最该有的性质：它是省 Token 的优化，不是主链路依赖。

**Tool Use** 有，但边界要说清。职责是把模型意图映射成结构化调用，并负责发现、Schema、超时、失败语义。项目里 `retrieve_knowledge` 用 `response_format="content_and_artifact"`，让工具同时回文本和 Document 对象；MCP 侧 `retry_interceptor` 做三次指数退避，全失败后**返回** `CallToolResult(isError=True)` 而不是抛异常——这是我在这个项目里第二处刻意的「return 而非 raise」，理由是工具失败是模型可以据此改参数重试的 Observation，抛异常会让整个 Agent Loop 断掉。缺的是权限、幂等键、副作用确认，因为现有 7 个工具全是只读查询，没有写操作。

**Execution** 只有半个。`instrument_node` 在注册时统一包一层计时和 span 记录，节点跑完把降级原因、召回条数写进 payload，这是执行层该做的「记录 Observation」。但驱动状态机、重试、重规划、人工接管这些没有，因为图是固定边，跑完即 END。

**Planning** 没有。项目中没有，不能硬套——五个节点在 `build_rag_v2_graph` 里写死，`_fanout_to_retrieve` 只是把改写出的子查询扇出，不是模型在决定下一步。`rewrite_node` 拆 3 条子查询看着像规划，实际是固定的一次性查询扩写，`NUM_SUB_QUERIES = 3` 是常量。删掉的 `app/agent/aiops/planner.py` / `replanner.py` 曾经是 Plan-Execute-Replan 结构，但已从仓库移除，不能拿来讲。

**项目证据**
- `app/services/conversation_memory_service.py:48-54` `estimate_tokens`；`57-58` `_message_token_cost`；`93-131` `build_context`；`133-145` `_messages_after_snapshot`；`163-186` `_summarize` 失败退窗口
- `app/tools/knowledge_tool.py:13` — `response_format="content_and_artifact"`
- `app/agent/mcp_client.py:18-73` — 重试拦截器；`71-73` 全失败后 return `isError=True` 而非 raise
- `app/agent/rag_v2/instrumentation.py:117` `instrument_node`；`92` `_derive_degradation`；`149` `_apply`
- `app/agent/rag_v2/graph.py:29-52` — 固定边，无动态规划
- `app/agent/rag_v2/nodes.py:28` — `NUM_SUB_QUERIES = 3` 是常量，不是模型决策
- Planning / 重规划 / 工具权限 / 幂等键：项目中没有，不能硬套。`app/agent/aiops/planner.py`、`replanner.py` 已从仓库删除，不作为证据

#### Q4 · 意图识别模块通常有哪些实现方式？规则匹配、小模型分类和 LLM 分类分别适用于哪些业务场景

**回答**

意图识别这一层，**项目中没有，不能硬套**。仓库里 grep `intent` / `router` 只能匹配到 FastAPI 的 `APIRouter`（`app/api/chat.py:18`、`aiops.py:16`、`protocol_pdf.py:21`），那是 HTTP 路由，不是意图路由。前端选模式（`static/app.js` 的 `currentMode`）走的是不同 API 端点，是用户手选而非模型分类。所以下面讲的是我会怎么做，以及为什么这个项目现在不需要。

三种实现按不确定性分层：**规则匹配**用关键词、正则、有限状态机，延迟低、可审计、改了能立刻验证，适合边界清晰且错了代价大的意图——退款、注销、转人工这类。缺点是对表达变体不鲁棒，规则堆到几百条之后维护成本会翻上来。**小模型分类**用句向量加线性分类器或轻量 Transformer，适合意图集合稳定、有标注数据、吞吐和成本敏感的线上主路径；关键是必须输出置信度并配拒识类，不能把 argmax 当绝对正确。**LLM 分类**适合类别变动快、语义复杂、要同时抽槽位或需要解释的场景，代价是延迟、成本和格式稳定性。

工程上是分层路由：规则先接高置信高风险，小模型覆盖大多数，低置信和长尾兜给 LLM，最后仍由权限和业务状态校验动作。比方案时看 macro-F1、各类召回、拒识准确率、校准误差、P95 和错误意图的业务损失，不是只看总准确率。

这个项目不需要意图路由的真实原因是入口已经把意图分掉了：`/chat` 是通用对话（`create_agent` 带工具自己决定要不要查知识库），`/chat/v2` 是明确的知识问答（进 RAG 图），`/aiops/alerts/analyze` 是告警诊断，`/protocol-pdfs/upload` 是文档入库。四个端点各自意图确定，硬加一层模型路由只会引入新的错误源。如果要合成一个统一入口，我会先上规则加端点映射，而不是先训分类器。

**项目证据**
- `app/api/chat.py:18,21,67`、`app/api/chat_v2.py`、`app/api/aiops.py:16,50`、`app/api/protocol_pdf.py:21,36` — 四个入口按意图分开，端点即路由
- `app/services/rag_agent_service.py:101,133` — `create_agent` 带 `tools`，要不要查知识库由模型在工具选择时决定，不是独立意图分类
- 意图识别模块 / 规则引擎 / 分类小模型 / 拒识与置信度阈值：项目中没有，不能硬套。`APIRouter` 是 HTTP 路由，与意图路由不是一回事，不能冒充

#### Q5 · 如果使用大模型完成意图分类，如何选择 Zero-shot、Few-shot 方案？当标注数据较少时，如何提升分类稳定性？

**回答**

意图分类这个具体任务项目里没有，但「让 LLM 输出结构化判定并稳定解析」这件事项目里做了两处，可以拿来答稳定性部分。

先说选型。Zero-shot 只给任务定义、类别说明、输出 Schema，适合类别定义清晰、表达变化大、要快速冷启动。Few-shot 加少量示例，适合类别边界易混、输出格式有特殊约定、模型不熟领域术语。示例不能只堆正常样本，要覆盖相邻类别、拒识、缺槽位和对抗输入。选哪个用验证集定：固定模型、温度、Prompt、Schema，比 macro-F1、混淆矩阵、拒识率、解析成功率、延迟和成本。类别多的时候先粗粒度域路由再细分，减少一次 Prompt 里的类别竞争。

标注少时的稳定化手段，项目里已经落地的有三条。第一，**温度压到 0**：`validate_answer_node` 里 `create_chat_model(temperature=0.0, streaming=False)`，判定类任务不需要采样多样性。第二，**输出结构化 + 服务端强校验**：`VALIDATION_SYSTEM_PROMPT` 要求模型返回 JSON，`_parse_validation_result` 负责解析，`_coerce_score` 把分数强制到合法区间、`_coerce_str_list` 把字段强制成字符串列表，模型返回 `"0.8分"` 这种脏值不会污染下游。第三，**解析失败有兜底而非崩掉**：`_parse_sub_queries` 先剥 ```json 围栏，再退化到裸 `[...]` 正则，最后 JSONDecodeError 就返回 `[fallback]`（原始 query）。这三条是我认为「少样本下让 LLM 分类可用」的工程底座——不是靠 Prompt 写得更长，是靠输出契约 + 强制转换 + 降级路径。

项目里没做的：主动学习挑低置信样本、多次采样一致性投票、同义改写数据增强、监督微调或蒸馏。这些我能讲原理但没有代码可引，标 `[假设]`。特别要说明的是**降温不等于正确**，它只减少随机性，修不了标签定义模糊——真正的稳定来自标签边界、示例质量、结构化约束和拒识机制。

**项目证据**
- `app/agent/rag_v2/nodes.py:60` `VALIDATION_SYSTEM_PROMPT` — 要求 JSON 输出的结构化契约
- `app/agent/rag_v2/nodes.py:494` `_parse_validation_result`；`526` `_coerce_score`；`534` `_coerce_str_list` — 服务端强制类型转换
- `app/agent/rag_v2/nodes.py:472` `_parse_sub_queries` — 剥围栏 → 裸正则 → JSONDecodeError 兜底返回原 query
- `app/agent/rag_v2/nodes.py:334` `validate_answer_node` 内 `temperature=0.0` — 判定类任务不采样
- `app/agent/rag_v2/nodes.py:31` `REWRITE_TEMPERATURE = 0.3` — 生成型任务留一点多样性，与判定任务区分
- 意图分类任务本身 / Few-shot 示例集 / 主动学习 / 一致性投票 / 微调蒸馏：项目中没有，不能硬套

#### Q6 · RAG 系统从文档进入到最终生成答案的完整流程是什么？离线知识库构建和在线检索阶段分别包含哪些步骤？

**回答**

这个项目两条链路都有真代码，我按仓库实际跑的顺序讲。

**离线**：入口是 `vector_index_service.index_directory`（扫目录）或 `index_single_file`（单文件）。先按扩展名分流到 `document_splitter_service.split_document`——Markdown 走 `split_markdown`（先按 `#` / `##` 标题切，`strip_headers=False` 把标题留在正文里，再用 `RecursiveCharacterTextSplitter` 二次切），其他走 `split_text`。切完过 `_merge_small_chunks` 把过小的片合回去，避免碎片污染召回。然后 `vector_store_manager.add_documents` 调 `DashScopeEmbeddings.embed_documents` 编码——注意这里有个服务端限制，DashScope `text-embedding-v4` 单次最多 10 条，所以 `MAX_BATCH_SIZE = 10`，内部自己分批循环。向量进 Milvus，HNSW 索引、COSINE 度量、1024 维。重建索引前先 `delete_by_source(normalized_path)` 删旧的，路径统一 `as_posix()`，这是幂等重入的关键。

PDF 协议文档另有一条更长的离线链路：上传 → 202 异步 → RQ 队列 → worker 解析 → 人工确认（`/confirm`）或驳回（`/reject`）才真正入库，这条链路上有哈希去重和状态机。

**在线**：`/chat/v2` 进 `rag_v2_graph`。`rewrite_node` 把原问题扩写成 3 条子查询（`NUM_SUB_QUERIES = 3`，温度 0.3）；`_fanout_to_retrieve` 用 `Send` 把 3 条并行扇出到 `retrieve_each_node`，每条 `RETRIEVE_TOP_K = 4`，内部走 `retrieve_documents` 做向量 + BM25 的 RRF 融合，`candidate_k = max(top_k*3, top_k, 10)` 先取宽候选；`dedup_node` 按 metadata id 或内容 MD5 去重，截到 `FINAL_TOP_K = 6`，同时汇总各分支失败情况判定降级原因；`generate_node` 拼上下文生成；`validate_answer_node` 独立调一次 LLM 做证据校验，groundedness < 0.75 或 coverage < 0.60 就拦下换保守回答。整趟由 `asyncio.timeout(90s)` 包着，超时不抛、交部分结果。

要说清的差别：教科书流程里的 rerank 这一步项目里**没有**在线路径，只在 `tests/eval/run_eval.py` 里作为 `--retriever local-rerank` 评测选项存在。

**项目证据**
- `app/services/vector_index_service.py:67` `index_directory`；`131` `index_single_file`；`155-159` `as_posix` + `delete_by_source` + 切分
- `app/services/document_splitter_service.py:21-30` MarkdownHeaderTextSplitter（h1/h2，`strip_headers=False`）；`32-38` RecursiveCharacterTextSplitter；`45` `split_markdown`；`83` `split_text`；`119` `split_document`；`135` `_merge_small_chunks`
- `app/services/vector_embedding_service.py:22` `MAX_BATCH_SIZE = 10`；`76,90-91` 内部分批；`173-174` 模块级实例
- `app/core/milvus_client.py:49,198-199` — 1024 维、COSINE、HNSW
- `app/api/protocol_pdf.py:36,122,139` — PDF 异步入库 + 确认/驳回
- `app/agent/rag_v2/nodes.py:28-30,91,158,214,264,334` — 在线五节点与 top-k 常量
- `app/services/vector_search_service.py:84,95-99` — 宽候选 + RRF 融合
- `app/agent/rag_v2/service.py:50-75` — 90s 总预算 + 部分结果
- 在线 rerank：项目中没有，不能硬套；只在 `tests/eval/run_eval.py:456` 作为评测选项存在

#### Q7 · 文档切片有哪些常见策略？RecursiveCharacterTextSplitter 的实现逻辑是什么？针对中文文档处理需要注意哪些问题？

**回答**

常见策略我按「保不保结构」分两类：定长滑窗（简单、可控，但会把句子和表格切断）、结构感知切分（按标题、段落、列表、表格边界切，保住语义单元）、语义切分（按句向量相似度找断点，成本高）、父子切分（小块用于检索、大块用于生成）。这个项目用的是**结构感知 + 定长兜底**的两级方案。

`RecursiveCharacterTextSplitter` 的逻辑是按分隔符优先级递归降级：默认先试 `\n\n`（段落），切完的片如果还超 `chunk_size`，就对这片用下一级 `\n`（行）再切，还超就用空格，最后退到按字符硬切。每一级都尽量在"更大的语义边界"上下刀，只有实在切不下来才降级。切完相邻片按 `chunk_overlap` 留重叠，避免答案正好落在边界上被劈开。项目里的配置是 `chunk_max_size = 800`、`chunk_overlap = 100`，但注意二次切分器传的是 `chunk_size=self.chunk_size * 2`，也就是 1600——注释写的原因是「加倍 chunk_size，减少分片数」，因为一级标题切完的块已经有语义完整性，再按 800 切会过碎。

中文的坑我按项目里实际踩到的说。第一，**默认分隔符表里没有中文标点**。`。！？；` 都不在默认列表里，纯中文长段落会一路降级到按字符硬切，正好切在词中间。第二，**`length_function=len` 数的是字符不是 token**。中文 1 字约 1 token，英文 1 token 约 4 字符，同样 800 的 `chunk_size` 中文实际是 ~800 token、英文是 ~200 token，预算算法要按语言调——项目里 `_estimate_tokens` 和 `conversation_memory_service.estimate_tokens` 都做了 CJK 单独计数，就是为了这个。第三，**分词影响的是 BM25 不是切片**。`_tokenize_for_bm25` 对中文逐字切、ASCII 字母数字才缓冲成词，这是因为没引 jieba；逐字切让 BM25 退化成字级匹配，短查询召回会偏，这是已知取舍。第四，`strip_headers=False` 在中文技术文档里尤其重要——标题往往是唯一出现完整术语的地方，剥掉之后块内可能只剩代词。

**项目证据**
- `app/services/document_splitter_service.py:7` — 两个 splitter 的导入
- `app/services/document_splitter_service.py:21-30` — `headers_to_split_on=[("#","h1"),("##","h2")]`，注释「不再按三级标题分割，避免过度碎片化」，`strip_headers=False`
- `app/services/document_splitter_service.py:32-38` — `chunk_size=self.chunk_size * 2`、`length_function=len`、`is_separator_regex=False`
- `app/config.py:139-140` — `chunk_max_size = 800`、`chunk_overlap = 100`
- `app/services/document_splitter_service.py:135-163` `_merge_small_chunks`；`160` 小块合并的尺寸条件
- `app/services/vector_search_service.py:203-224` `_tokenize_for_bm25` — 中文逐字、ASCII 缓冲成词，未引 jieba
- `app/agent/rag_v2/nodes.py:631` `_estimate_tokens`、`app/services/conversation_memory_service.py:48-54` — CJK 单独计数
- 语义切分 / 父子切分 / jieba 分词：项目中没有，不能硬套

#### Q8 · 如果 RAG 系统出现召回效果差的问题，你会如何定位？会优先检查 Embedding、Chunk 策略、Query Rewrite、Hybrid Search 还是 Rerank？

**回答**

我不会先动 rerank。定位顺序是先把「召回差」拆成三个互斥的问题：证据**在不在**候选集里、**排**得对不对、进了上下文之后**用**没用上。这三层的修法完全不同，混在一起查会永远归因成「模型幻觉」。

第一步查候选集 Recall，用 `tests/eval/run_eval.py` 跑 golden set。这个 CLI 支持 `--top-k`、`--retriever`、`--vector-weight` / `--bm25-weight`、`--compare`、`--filter`、`--only-names`，能固定其他变量只动一个参数对比。指标是 `tests/eval/metrics.py` 里的 Hit@K / Recall@K / MRR——**Recall@K 是这一步唯一该看的指标**，因为它回答的正是「证据进候选了吗」。如果 Recall@K 已经很高但线上答不对，问题就不在召回侧，动 Embedding 和 Chunk 都是白费。

第二步按代价从低到高排查召回侧。**先查数据和入库**：文档解析了吗、`delete_by_source` 之后重建过吗、metadata 里 id 在不在（`dedup_node` 拿不到 id 会退化用内容 MD5 去重，两个近似块可能都留下挤掉别的证据）。**再查 Chunk**：`chunk_max_size = 800`、二次切分实际是 1600，证据是不是被劈成两半、标题有没有丢（`strip_headers=False` 保住了，但要确认走的是 `split_markdown` 而不是 `split_text`）。**再查 Query Rewrite**：`rewrite_node` 拆 3 条子查询，`_parse_sub_queries` 解析失败会退回原 query，这是个静默降级——日志里 `[rag_v2.rewrite]` 那条能看出实际扩写成了什么，改写把实体或否定条件丢了就是单点故障。**再查 Hybrid**：术语、编号、产品名这类精确匹配靠 BM25 补，项目权重是 `VECTOR_WEIGHT = 0.7` / `BM25_WEIGHT = 0.3`，`--vector-weight` / `--bm25-weight` 可以直接跑对比；还要确认 BM25 语料非空，`_get_bm25_retriever` 拿不到语料会静默降级成纯向量，日志有「BM25 语料为空，降级为纯向量检索」。

Rerank 放最后，而且只在候选集里**已经有**正确证据时才有用。候选里没有的文档，换任何 reranker 都补不回来。这个项目在线路径本来也没有 rerank，只有评测里的 `--retriever local-rerank` 选项。

**项目证据**
- `tests/eval/run_eval.py:449,455,456,459,461,462,464,467` — `--tag` / `--top-k` / `--retriever` / `--compare` / `--vector-weight` / `--bm25-weight` / `--filter` / `--only-names`
- `tests/eval/metrics.py` — Hit@K / Recall@K / MRR 已实现
- `app/agent/rag_v2/nodes.py:214-262` `dedup_node` — 无 id 时退化用内容 MD5
- `app/agent/rag_v2/nodes.py:472` `_parse_sub_queries` — 解析失败静默退回原 query
- `app/services/vector_search_service.py:42-43` 权重；`84` `candidate_k`；`92-94` BM25 空语料降级并打日志
- `app/services/vector_index_service.py:155-156` — 重建前 `delete_by_source`
- 在线 rerank：项目中没有，不能硬套；`tests/eval/run_eval.py:456` 的 `local-rerank` 仅评测可选

#### Q9 · Embedding 模型如何选择？不同 Embedding 模型会对检索效果产生哪些影响？

**回答**

这个项目的实际选择是 DashScope `text-embedding-v4`，1024 维，走 OpenAI 兼容接口。选它的三个真实理由：中文表现够用、跟主 LLM（`qwen-max`）同一家不用管两套鉴权、不用自己部署 GPU 推理。代价是数据要出域、单次批量上限只有 10 条、维度和归一化行为得看服务方文档而不是自己控制。

选型该看的维度：目标语言和领域、输入长度上限、query 和 document 是否需要不同指令、部署形态（API 还是本地）、维度与存储成本、吞吐和 P95、许可、以及数据能不能出域。数据敏感的场景，本地部署这一条可以直接压倒榜单上几个点的差距。

不同模型影响三个层面。**语义召回**：近义表达、上下位概念能不能聚近；过度语义化会把「用词相近但事实不同」的文档召回来。**细节区分**：版本号、错误码、类名、数字这类精确信息向量模型天生不擅长，得靠 BM25 补——这正是项目里 `EnsembleRetriever` 存在的理由，0.7/0.3 的权重就是在给精确匹配留 3 成话语权。**系统资源**：维度直接决定索引大小和内存，`VECTOR_DIM = 1024` 和 Milvus schema 里的 `dim` 是绑死的，换模型换维度必须重建整个 collection，不能混用旧索引。

有两个坑要说清。第一，**query 和 document 必须用同一个模型和同一套归一化**，`DashScopeEmbeddings` 里 `embed_documents` 和 `embed_query` 共用同一份 `self.model` / `self.dimensions` 就是为了这个。第二，**批量上限是服务端硬限制不是调优项**，`MAX_BATCH_SIZE = 10`，构造时 `batch_size` 超出范围直接抛，内部按批循环——这个数字改大了不是变快，是报错。

至于换模型怎么验：构一份和线上分布一致的 query-证据集，比 Recall@K、MRR，同时记 P95、吞吐、成本和迁移方案。项目里 `tests/eval/` 有这个框架，但目前 golden set 只有 83 条，规模不够支撑「换 Embedding 模型」这种量级的决策，这是真实短板。

**项目证据**
- `app/config.py:47` `dashscope_embedding_model = "text-embedding-v4"`；`45` OpenAI 兼容 base；`46,129` `qwen-max` 同厂
- `app/services/vector_embedding_service.py:12,27-28` — `DashScopeEmbeddings`，默认 v4 / 1024 维；`22,44-46` `MAX_BATCH_SIZE = 10` 及越界抛错；`72,137` `embed_documents` / `embed_query` 共用同一 model
- `app/core/milvus_client.py:49,162` — `VECTOR_DIM = 1024` 与 schema `dim` 绑定，换维度需重建 collection
- `app/services/vector_search_service.py:42-43` — 0.7/0.3 给精确匹配留权重
- `tests/eval/golden_set_expanded.jsonl` — 83 条，规模不足以支撑换模型决策
- 本地 Embedding 部署 / 多模型 A/B / 向量分布迁移方案：项目中没有，不能硬套

#### Q10 · LangChain 框架主要有哪些核心组件？相比传统 Chain 模式，LCEL 带来了哪些改进？

**回答**

先说我这个仓库真实用到了哪些：模型接口用 `langchain_openai.ChatOpenAI`，统一在 `llm_factory` 里构造；消息类型用 `SystemMessage` / `HumanMessage`；工具用 `@tool` 装饰器；Embedding 是继承 `langchain_core.embeddings.Embeddings` 自己实现的 `DashScopeEmbeddings`；向量库用 `langchain_milvus.Milvus`；切分用 `langchain_text_splitters` 的 `MarkdownHeaderTextSplitter` 加 `RecursiveCharacterTextSplitter`；检索融合用 `EnsembleRetriever`；Agent 那条路径用 `langchain.agents.create_agent` 配 `langgraph.checkpoint.memory.MemorySaver`。

LCEL 我一处都没用。grep `Runnable` 和管道组合，仓库里是空的。这不是漏了，是取舍：编排在 LangGraph 手里，`build_rag_v2_graph` 里五个节点用五条 `add_edge` 显式连起来，唯一的非线性是 `add_conditional_edges` 那个 fan-out。如果换成 LCEL 的 `prompt | llm | parser` 表达式，三次 LLM 调用会被藏进一个链式表达里，而我恰恰需要每个节点单独出一条 span——`instrument_node` 是包在注册那一层的，`graph.add_node("rewrite", instrument_node("rewrite", rewrite_node))`，节点边界就是计时边界。LCEL 会把这个边界糊掉。

要我讲 LCEL 相比手写 Chain 的改进，我只能讲机制不能讲经验：统一的 `invoke` / `batch` / `stream` 接口，可组合的 Runnable 契约，并行分支和 fallback 有统一写法，callback 和 tracing 挂载点统一。但它不等于可靠性。超时、总预算、幂等、熔断、状态落盘这些东西 LCEL 一个都不管——我这边 `chat_total_budget_seconds` 是在 `asyncio.timeout` 里强制的，熔断是 `retrieval_breaker.guard()` 包在 `to_thread` 外面的，这两处都不是框架给的。循环和分支多、状态要持久化的流程，LangGraph 或自研状态机更合适，这也是我当初的选择。

**项目证据**
- `app/core/llm_factory.py:64,105` — `create_chat_model` / `create_qwen_model`，唯一的模型构造入口
- `app/services/vector_embedding_service.py:12` — `class DashScopeEmbeddings(Embeddings)`，实现 LangChain 的 Embeddings 契约
- `app/services/document_splitter_service.py:7,20,32` — 两个 text splitter 组合使用
- `app/services/vector_search_service.py:96-99` — `EnsembleRetriever` 做 RRF 融合
- `app/services/rag_agent_service.py:9,16,107,133` — `create_agent` + `MemorySaver`
- `app/agent/rag_v2/graph.py:38-48` — 五条 `add_edge` + 一条 `add_conditional_edges`，编排显式写在这里
- `app/agent/rag_v2/graph.py:31-36` — `instrument_node` 包在注册层，节点边界即 span 边界
- LCEL / `Runnable` / 管道组合：项目中没有，不能硬套。grep 无一处命中，编排职责整体交给了 LangGraph

#### Q11 · Function Calling 和 Tool Calling 的执行流程是什么？模型是如何判断需要调用工具，以及生成对应参数的？

**回答**

我这里有两条真实的工具调用路径，流程不一样，讲清楚区别比背通用流程有用。

第一条是 `create_agent` 托管的。工具列表在 `self.tools = [retrieve_knowledge, get_current_time]`，加上运行时从 MCP 拉到的 `self.mcp_tools`，合成 `all_tools` 传进 `create_agent`。工具的 schema 不是我手写的——`@tool` 装饰器从函数签名和 docstring 自动生成：`retrieve_knowledge(query: str)` 的类型注解变成 JSON Schema 的 `{"query": {"type": "string"}}`，docstring 第一行「从知识库中检索相关信息来回答问题」和下面那句「当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具」就是模型看到的工具描述和触发条件。所以我写 docstring 的时候是在写 prompt，不是在写注释，这点很容易被忽略。之后模型输出 `tool_calls`，LangChain runtime 解析、执行、把结果作为 tool message 写回上下文，再让模型决定继续还是收尾。

第二条是 MCP。工具定义在远端 server 上，`mcp_servers/cls_server.py` 有 5 个 `@mcp.tool()`，`monitor_server.py` 有 2 个，客户端 `await mcp_client.get_tools()` 动态拉取。这条路上我加了一层 `retry_interceptor`，包在 handler 外面，指数退避重试三次。

关于「模型怎么判断要不要调工具」，我不会说成模型在推理。它是在训练分布和当前上下文条件下预测「输出 tool_call」还是「输出文本」的概率，选得对不对取决于工具描述质量、候选集大小和当前状态。参数合法也不等于业务正确——`query: str` 只保证是字符串，保证不了这个 query 检索得到东西。所以 `retrieve_knowledge` 内部对空结果有显式处理，返回「没有找到相关信息。」和空列表，而不是让上层拿到 None 去崩。

权限校验、幂等键、副作用确认这三样我没有。原因是现有 7 个 MCP 工具全是只读查询——查 CPU、查内存、查日志、查 topic，没有一个会改变外部状态。没有写操作就没有幂等需求，硬加一层反而是过度设计。但这也意味着一旦接入写类工具，这三样必须先补上，不能直接放进 `all_tools`。

**项目证据**
- `app/tools/knowledge_tool.py:13-14` — `@tool(response_format="content_and_artifact")`，schema 从签名生成
- `app/tools/knowledge_tool.py:16-23` — docstring 就是模型看到的工具描述与触发条件
- `app/tools/knowledge_tool.py:33-36` — 空结果显式返回提示文本 + 空列表，不返回 None
- `app/tools/time_tool.py:10-11` — `@tool` 装饰的 `get_current_time(timezone: str = "Asia/Shanghai")`，默认值也进 schema
- `app/services/rag_agent_service.py:101,123-135` — 本地工具 + MCP 工具合并后传入 `create_agent`
- `app/agent/mcp_client.py:18-73` — `retry_interceptor` 包在 handler 外，指数退避三次
- `mcp_servers/cls_server.py:104,136,169,212,346` — 5 个只读查询工具
- `mcp_servers/monitor_server.py:124,277` — 2 个只读指标查询工具
- 参数级权限校验 / 幂等键 / 副作用确认：项目中没有，不能硬套。现有工具全只读，没有写操作可保护

#### Q12 · 如果 Agent 接入大量工具，如何避免工具描述过长导致 Prompt 膨胀？有哪些优化方式？

**回答**

我先说实话：我这个项目还没遇到这个问题。工具总数是 9 个——2 个本地（`retrieve_knowledge`、`get_current_time`）加 7 个 MCP，descriptions 加起来撑不起「膨胀」这个词。所以渐进披露、工具目录检索、按角色分配最小权限集，这些我都没实现，项目中没有，不能硬套。

但有一件相关的事我确实做了，而且方向恰好相反：先膨胀的不是工具描述，是工具结果。检索一次返回 4 条 chunk，三条子查询并行就是 12 条，去重后取 6 条进 prompt，每条正文可能上千字。所以我在两处做了裁剪。一处是进 prompt 之前，`_truncate_to_token_budget` 按 token 预算截断 context；另一处是落库之前，`slim_used_documents` 把每条证据的正文压成 200 字摘要，并且用 `excerpt_only: True` 如实标注「这是残缺的」。第二处的关键是幂等——`_slim_one` 开头判 `if _SLIM_MARKER in row: return row`，已经瘦身过的原样放行，所以调用方传全文还是传瘦身格式都行，重复调用不会把摘要再截一次。

要我讲怎么优化工具描述，我只能讲机制不能讲战绩：先用路由收窄候选集，工具目录分两阶段披露（先给名字和一句能力摘要，选定后再加载完整 schema），把相似工具合并成统一参数模型，把稳定的工具文档放到外部检索库按需取。但这些都要配指标验证——工具选择准确率、参数解析成功率、任务成功率、工具描述占的 token 比例，四个一起看。只看 prompt 变短是不够的，把工具藏起来当然能省 token，代价是模型选不到该选的那个。

**项目证据**
- `app/services/rag_agent_service.py:101,131` — 本地 2 个 + MCP 动态加载，`all_tools` 总数 9
- `app/agent/rag_v2/nodes.py:637` — `_truncate_to_token_budget`，进 prompt 前按预算截断
- `app/agent/rag_v2/nodes.py:29-30` — `RETRIEVE_TOP_K = 4` / `FINAL_TOP_K = 6`，候选量在源头限死
- `app/core/used_documents.py:42,46,79-80` — `EXCERPT_MAX_CHARS = 200`，`_SLIM_MARKER` 判定实现幂等
- `app/core/used_documents.py:111-113` — `excerpt_only: True` 标注数据残缺，不悄悄提供半截内容
- 工具渐进披露 / 工具目录检索 / 按角色分配工具集 / 工具描述缓存：项目中没有，不能硬套。9 个工具没到需要治理的规模

#### Q13 · Prompt 一般如何设计和组织？System Prompt、Few-shot 示例以及 CoT 通常分别承担什么作用？

**回答**

我的组织方式是按职责拆成独立常量，放在各自模块顶部，不共用一个全局大 prompt。现在有六个：`REWRITE_SYSTEM_PROMPT` 管子查询改写，`GENERATE_SYSTEM_PROMPT` 管基于证据生成，`VALIDATION_SYSTEM_PROMPT` 管证据校验和幻觉拦截，`MEMORY_POLICY` 管会话记忆的使用边界，`_SUMMARY_SYSTEM_PROMPT` 管滚动摘要，`first_response_service` 里的 `_SYSTEM_PROMPT` 管首响。拆开的好处是改一个不影响另一个——调质检严格度不会顺手改坏改写。

System prompt 承担的是任务契约：角色、目标、输出格式、不可违反的边界。举个具体的，`MEMORY_POLICY` 那段是专门为了防一类错误写的——会话记忆和知识库证据混在一个上下文里，模型会把「用户上轮说过的话」当成知识库里的事实来引用。所以我在质检的 user prompt 里显式写「仅将 `<evidence>` 中的内容作为知识库可验证证据；会话记忆只能帮助判断对象指代」，并且用 `<question>` / `<answer>` / `<evidence>` / `<conversation_memory>` 四个标签把来源物理隔开。这不是 prompt 技巧，是在划信任边界。

`_SUMMARY_SYSTEM_PROMPT` 是另一种写法：固定四个小标题【明确事实】【任务进展】【约束与偏好】【待确认项】，并且要求「缺失项写"无"」。要求写「无」而不是允许省略，是因为下游要靠标题定位内容，省略会让格式不稳定。

Few-shot 我没用。grep few-shot 和示例块，仓库里是空的。项目中没有，不能硬套。我用的是另一条路子稳格式：`REWRITE_SYSTEM_PROMPT` 要求输出 JSON 数组，然后在 `_parse_sub_queries` 里做三层兜底——先剥 ```json 围栏，再用裸 `[...]` 正则捞，最后 `JSONDecodeError` 就退回 `[fallback]` 用原问题检索。示例能提高格式命中率但不能保证，解析侧的降级是保证。两者不冲突，我只是先做了后者。

显式 CoT 也没有。质检节点用 `temperature=0.0`，要的是稳定的结构化判定分数，不是推理过程。真要说有什么替代，是让模型输出可验证的中间结构——`groundedness_score`、`coverage_score`、`unsupported_claims`，这些能被 `_coerce_score` 和 `_coerce_str_list` 校验，比一段自然语言推理好查。

**项目证据**
- `app/agent/rag_v2/nodes.py:36,50,60,87` — 四个 prompt 常量各管一段职责
- `app/agent/rag_v2/nodes.py:363-372` — `<question>` / `<answer>` / `<evidence>` / `<conversation_memory>` 四标签隔离来源
- `app/agent/rag_v2/nodes.py:371-372` — 显式声明只有 evidence 算可验证证据，记忆仅供指代
- `app/services/conversation_memory_service.py:28-35` — 固定四标题 + 「缺失项写"无"」
- `app/services/first_response_service.py:101` — 首响独立 `_SYSTEM_PROMPT`
- `app/agent/rag_v2/nodes.py:472-492` — `_parse_sub_queries` 三层降级兜格式
- `app/agent/rag_v2/nodes.py:526,534` — `_coerce_score` / `_coerce_str_list` 校验模型输出的中间结构
- Few-shot 示例 / 显式 CoT 要求 / prompt 版本化与回归集：项目中没有，不能硬套

#### Q14 · 如何降低大模型输出幻觉？除了 Prompt 约束之外，还有哪些工程优化方案？

**回答**

我这里有一层真实的拦截，不是 prompt 里加一句「不要编造」。`validate_answer_node` 是图上第五个节点，答案生成完之后必过这一关：把 question、answer、evidence 三份材料交给一个 `temperature=0.0` 的独立 LLM 调用，让它输出两个分数——`groundedness_score` 判答案有多少是证据支持的，`coverage_score` 判证据覆盖了问题的多少。阈值是 `MIN_GROUNDEDNESS_SCORE = 0.75` 和 `MIN_COVERAGE_SCORE = 0.60`。过不了就用 `_build_fallback_answer` 换成保守回答，而不是把原答案发出去。

节点入口有两个不花钱的结构性 guard，顺序是刻意的。答案为空直接判不通过；证据一条都没有就直接降级成保守回答并标 `needs_second_retrieval=True`。这两个 guard 放在开关判断之前——因为它们不依赖那个可能被关掉的 LLM 调用，该继续生效。

质检本身也会挂，所以走的是 fail-open：熔断打开抛 `CircuitOpenError`，或者解析失败，都被同一个 except 接住，答案照发但打上降级标记。这里有个细节我踩过——关掉质检开关的时候不能用 `_default_validation(blocked=True)`，那会把每个答案都换成「当前证据还不够支持直接下结论」，用户以为是自己的问题缺证据，实际是我们把质检关了。关掉一道校验不该让答案消失，所以走 `_unvalidated_validation`，答案原样返回，只在 `degrade_reasons` 里记 `FEATURE_DISABLED`。降级原因也不能用 `LLM_ERROR`，那会把值班同学送去查模型配额和鉴权，而那边一切正常，排障就卡在这儿。

再往上一层，检索侧的三态区分也是防幻觉的一部分。`RETRIEVAL_EMPTY` 是知识库真没有，`RETRIEVAL_FAILED` 是 Milvus 挂了，`PARTIAL_RETRIEVAL` 是部分分支失败但仍有证据。分开的意义在于：全失败的时候答案是在零证据上生成的，这个事实必须传到最后，不能被「返回了空列表」这种沉默降级吃掉。判定放在 `dedup_node` 而不是 `retrieve_each_node`，因为单个分支看不到全局——它只知道自己失败了，判断不了是全都失败还是只有它失败。

真实短板要说清楚：`faithfulness_placeholder` 和 `answer_relevance_placeholder` 都还是 `NotImplementedError`。也就是说线上有拦截，但没有离线指标能证明拦截率和误拦率是多少。golden set 只有 83 条，规模也不够支撑阈值调优。这两个数字（0.75 / 0.60）现在是拍的，不是测出来的。

**项目证据**
- `app/agent/rag_v2/nodes.py:334` — `validate_answer_node`，独立质检节点
- `app/agent/rag_v2/nodes.py:32-33` — `MIN_GROUNDEDNESS_SCORE = 0.75`、`MIN_COVERAGE_SCORE = 0.60`
- `app/agent/rag_v2/nodes.py:341-356` — 答案为空 / 无证据两个结构性 guard，先于开关判断
- `app/agent/rag_v2/nodes.py:358-380` — 开关关闭走 `_unvalidated_validation` + `FEATURE_DISABLED`，答案不消失
- `app/agent/rag_v2/nodes.py:592` — `_build_fallback_answer` 生成保守回答
- `app/agent/rag_v2/nodes.py:60` — `VALIDATION_SYSTEM_PROMPT` 定义质检契约
- `app/agent/rag_v2/nodes.py:227-259` — `dedup_node` 在 fan-in 后判定三种检索降级原因
- `app/core/errors.py:65` — `RetrievalStatus` 三态，杀掉「静默返回空列表」
- `app/core/metrics.py:94,100` — `degrade_total` / `retrieval_failures_total`，降级可计量
- `tests/eval/metrics.py` — `faithfulness_placeholder` / `answer_relevance_placeholder` 仍是 `NotImplementedError`
- 引用精确率 / 拒答准确率 / Prompt Injection 防护 / 敏感信息泄漏率：项目中没有，不能硬套

#### Q15 · Agent 执行任务时，如果出现重复调用工具、无法结束或者任务循环的问题，应该如何设计保护机制？

**回答**

这题我得先把我这个系统的结构说清楚，否则答案会假。`rag_v2` 那张图是 DAG，不可能循环——`START → rewrite → (fan-out) retrieve_each → dedup → generate → validate_answer → END`，五条 `add_edge` 全是单向，唯一的非线性是 `_fanout_to_retrieve` 那个 `Send` 列表，而它的分支数是 `NUM_SUB_QUERIES = 3` 定死的。跑完即 `END`，没有回边，所以我没有最大步数限制，也不需要。这条路上不存在「无法结束」这个失效模式，硬说我做了循环保护是编的。

真正需要保护的是另外三处，这三处是真的。

单次调用超时：`llm_timeout_seconds = 60.0`，统一在 `llm_factory` 里注入。这里有个坑值得讲——langchain-openai 的 `request_timeout` 默认 `None`，会无条件覆盖 OpenAI SDK 的 `DEFAULT_TIMEOUT`（read=600），所以不显式传就等于把超时放到十分钟。三个构造点各写一遍是重复，也一定会漏，所以收进工厂。默认值的写法也有讲究：`timeout = config.llm_timeout_seconds if timeout is None else timeout`，不能写 `timeout or config.x`——后者会把显式传进来的 `timeout=0` 悄悄换成 60。

总预算：`chat_total_budget_seconds = 90.0`，在 `service.py` 用 `async with asyncio.timeout(...)` 包住整个 `astream`。单次 60 秒不等于总预算，`chat_v2` 一条链上串行调三次 LLM，最坏 180 秒，用户早走了。超时之后我不往上抛——用 `stream_mode="values"` 逐步把中间状态存进 `last_state`，预算耗尽就交付已有的部分结果。抛异常会让用户既等满了预算又什么都拿不到，是最差的结果。

下游重试：MCP 那条路上 `retry_interceptor` 重试三次，`wait_time = delay * (2 ** attempt)` 指数退避。三次都失败之后它 **return** `CallToolResult(content=[...], isError=True)` 而不是 raise——这是刻意的，工具失败是模型可以处理的 observation，抛出去会打断整个 agent loop。再上一层是熔断器，`retrieval_breaker.guard()` 包在 `to_thread` 外面而不是塞进线程里，这样熔断打开时连线程池 worker 都不必占用，省下的正是最紧张的那份资源。

现在说真实缺口，因为这题正好戳在上面。`create_agent` 那条路径是有循环的（ReAct loop），它继承 LangGraph 的默认 `recursion_limit`，但我 grep 全仓库没有一处显式设置过 `recursion_limit`。也就是说这条路上的步数上限是框架默认值，不是我定的，我也没测过它到底在第几步停。动作指纹去重、连续相同调用检测、单工具重试次数上限、按 deadline 强制中止——这些在 `create_agent` 那条路上全都没有。项目中没有，不能硬套。这是我会主动在面试里说的缺口，因为 DAG 那条路的安全不能拿来给 ReAct 那条路背书。

**项目证据**
- `app/agent/rag_v2/graph.py:41-48` — 五条单向 `add_edge`，DAG 结构上无环
- `app/agent/rag_v2/graph.py:18-21` — `_fanout_to_retrieve` 分支数由子查询数决定
- `app/agent/rag_v2/nodes.py:28` — `NUM_SUB_QUERIES = 3`，fan-out 宽度定死
- `app/config.py:70` — `llm_timeout_seconds = 60.0`
- `app/core/llm_factory.py` 模块 docstring — 记录三处构造点缺 timeout 与 `request_timeout` 覆盖 SDK 默认值的坑
- `app/config.py:81` — `chat_total_budget_seconds = 90.0`
- `app/agent/rag_v2/service.py:50-58` — `asyncio.timeout` 包住 `astream`，超时不抛、交付部分结果
- `app/agent/mcp_client.py:60-63` — `delay * (2 ** attempt)` 指数退避
- `app/agent/mcp_client.py:67-73` — 重试耗尽后 **return** `isError=True` 而非 raise
- `app/agent/rag_v2/nodes.py:181-186` — 熔断器 `guard()` 包在 `to_thread` 外，拒绝发生在占用 worker 之前
- `app/core/circuit_breaker.py:80,135,205` — `CircuitBreaker` / `allow` / `guard`
- `recursion_limit` 显式设置 / 动作指纹去重 / 最大步数 / 单工具重试上限 / deadline 强制中止：项目中没有，不能硬套。`create_agent` 那条 ReAct 路径目前只有框架默认值兜着

#### Q16 · Agent 中的 Memory 通常如何实现？短期记忆和长期记忆分别解决什么问题？

**回答**

短期记忆我这边是真实现了的，讲得动。它解决的是「输入 token 不能随对话轮数无限增长，但上下文又不能断」这个矛盾。做法是三段：全量原文只进 PostgreSQL 当审计记录，模型上下文只用「压缩摘要 + 未压缩消息的尾部」。

token 估算是自己写的，`estimate_tokens` 里 `cjk_chars + math.ceil(other_chars / 4)`——中文按字算，其他按四字符一 token 算，因为主流 tokenizer 对中日韩基本是一字一 token，直接除 4 会低估一半多。每条消息再加 4，算的是 role 和分隔符的固定开销。

`build_context` 的流程：读活跃消息，取快照游标，游标之后的算 `pending_messages`，从尾部留出 `recent_count` 条不动，剩下的 `compact_candidates` 只有在 token 数超过 `_compact_threshold_tokens` 时才去摘要。所以不是每轮都摘要，是攒够了才摘。摘要产出用 `_render_context` 拼成带【会话滚动摘要】头的块注入。

两个边界值得讲。一是快照游标那条消息被软删除的时候，`_messages_after_snapshot` 返回 `[]`——保守地不注入旧原文，避免重复历史。二是摘要 LLM 调用失败的时候不阻断主聊天，except 里直接返回 `""`，退回最近消息窗口。摘要是省 token 的优化，不是聊天的前提条件，让它挂掉就整个聊不了是本末倒置。

另一条路径是 `create_agent` 配 `MemorySaver`，那是 LangGraph 的 checkpointer，按 thread_id 存对话状态，`get_session_history` 从里面读、`clear_session` 从里面删。它和上面那套是两套并存的机制，服务的场景不同——一套给 rag_v2 的显式编排，一套给 agent loop 的对话续接。

长期记忆——跨会话的偏好、事实、经验，带作用域和置信度，按需检索召回——项目中没有，不能硬套。这里要特别澄清一点，PostgreSQL 里存了全量对话原文，但那是审计不是长期记忆：没有提取、没有去重、没有作用域标记、没有检索入口，它只是能查出来「这个 session 说过什么」，不能回答「这个用户偏好什么」。把它说成长期记忆是偷换概念。

**项目证据**
- `app/services/conversation_memory_service.py` 模块 docstring — 全量原文进 PG 做审计，模型上下文只用摘要 + 尾部
- `app/services/conversation_memory_service.py:48-54` — `estimate_tokens`，CJK 按字、其他除 4
- `app/services/conversation_memory_service.py:57-58` — `_message_token_cost` 每条加 4
- `app/services/conversation_memory_service.py:93-131` — `build_context`，快照游标 + 尾部保留 + 阈值触发
- `app/services/conversation_memory_service.py:133-145` — 游标消息被软删时返回 `[]`，不重复注入旧原文
- `app/services/conversation_memory_service.py:163-186` — 摘要失败返回 `""`，退回最近窗口，不阻断主聊天
- `app/services/conversation_memory_service.py:207,213` — `_render_context` + 【会话滚动摘要】头
- `app/services/rag_agent_service.py:107,310,377` — `MemorySaver` 及其读取/清空入口
- `app/models/conversation.py` — 全量消息表，审计用途
- 长期记忆（跨会话事实/偏好/经验 + 作用域 + 置信度 + 检索召回）：项目中没有，不能硬套。PG 里的对话原文是审计记录，没有提取、去重、作用域和检索入口，不能当长期记忆用

#### Q17 · 如何设计一个记忆检索流程？历史信息应该如何筛选、存储和召回？

**回答**

这题我得先划清界限：记忆**检索**流程，项目中没有，不能硬套。我现在的机制是无条件注入——`build_context` 拼出来的摘要块，每一轮都进 prompt，不做相关性判断，不做按需召回。没有查询构建、没有多路召回、没有权限过滤、没有重排。所以我不能拿它冒充检索流程。

差在哪儿我说得出来，因为这是我下一步要补的。一是写入门控：现在是「攒够 token 就整段摘要」，没有候选事实提取，没有判断这条信息是否稳定、是否可复用、是否敏感，寒暄和一次性状态和用户明确说的约束混在一起进摘要。二是分型存储：现在只有一张消息表加一个摘要快照，稳定偏好、事件流、语义经验没有分开存，也就没法按类型走不同召回。三是作用域：摘要绑在 session 上，跨 session 不共享也不隔离——不共享是功能缺失，但正因为不共享，跨用户串线这个风险目前也不存在。四是权限过滤要在召回时做而不是召回后做，这一点我现在连召回都没有，无从谈起。

有一处形似的东西可以讲，但要标清楚它不是检索。`build_context` 里的快照游标机制解决的是「哪些历史已经被压缩过、哪些还是原文」这个边界问题：游标之前的已进摘要，之后的是待处理。游标那条消息被软删除时返回 `[]` 保守回退，避免把已经压缩过的历史当成新原文再注入一遍。这是状态管理的正确性，跟按相关性召回是两件事。

真要我设计，顺序是「写入门控 → 分型存储 → 查询构建 → 多路召回 → 重排注入 → 反馈更新」。关键的几个点：召回前先做权限过滤而不是召回后；注入时标记成记忆而非当前事实，并带来源和更新时间；用户当轮明确指令优先于历史偏好，业务主库实时事实优先于摘要，模型推断不能覆盖明确事实；纠正产生新版本而不是静默覆盖审计记录。评价看记忆 precision/recall、错误记忆引用率、重复询问减少量、删除可达性。但这些是设计意图，不是我跑过的东西，我会在面试里明确这么标。

**项目证据**
- `app/services/conversation_memory_service.py:93-131` — 当前是无条件注入，无相关性判断、无按需召回
- `app/services/conversation_memory_service.py:133-145` — 快照游标 + 软删保守回退，是状态边界管理，不是检索
- `app/services/conversation_memory_service.py:28-35` — 摘要 prompt 有四个固定小标题，但没有事实级提取与去重
- `app/models/conversation.py` — 单表存全量消息，无分型存储
- 写入门控 / 分型存储 / 查询构建 / 多路召回 / 权限前置过滤 / 重排注入 / 反馈更新：项目中没有，不能硬套。整条记忆检索链路尚未实现

#### Q18 · 手撕代码：合并重叠区间（LeetCode 56），并分析算法复杂度

**回答**

按左端点升序排序，然后一遍扫。维护结果里最后一个区间 `last`：当前区间左端点大于 `last` 右端点就是不重叠，直接追加；否则把 `last` 的右端点更新成两者较大的那个。端点相等按可合并处理，`[1,4]` 和 `[4,5]` 并成 `[1,5]`。

```python
from typing import List


def merge(intervals: List[List[int]]) -> List[List[int]]:
    if not intervals:
        return []

    intervals.sort(key=lambda item: item[0])
    merged: List[List[int]] = [intervals[0][:]]

    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])

    return merged
```

排序 `O(n log n)`，扫一遍 `O(n)`，总体 `O(n log n)`。结果之外的额外空间 `O(1)`——注意 `merged[0]` 用了 `intervals[0][:]` 切片拷贝，避免原地改写调用方的第一个区间；如果要求完全不碰输入，先复制再排序，额外空间 `O(n)`。

算法成立靠一个不变量：扫到当前位置时，`merged` 已经是此前所有区间的最小不重叠表示。排序保证了后面的区间左端点不会更小，所以只需要跟最后一个比，不用回头。边界至少测空数组、单区间、完全包含（`[1,10]` 吃掉 `[2,3]`）、链式重叠（`[1,3][2,5][4,8]` 并成一个）、端点相接、负数端点。

仓库里没有算法题代码，`tests/` 下没有链表、区间这类结构，所以我不能假装有对应实现。但同一个不变量思路在 `dedup_node` 里有工程版：`seen` 集合加有序遍历，命中过的 key 直接 `continue`，结果集 `unique_docs` 只追加不回改，最后 `[:FINAL_TOP_K]` 截断。key 的取法有个细节——先取 `doc.metadata["id"]`，没有就用 `hashlib.md5(content)` 兜，因为历史分片的 metadata 里可能没 id，缺 key 会让去重整个失效。跟合并区间一样，正确性来自「遍历时结果集始终是已处理部分的正确表示」这个不变量，而不是来自多写几层判断。

**项目证据**
- `app/agent/rag_v2/nodes.py:214-233` — `dedup_node`，`seen` 集合 + 有序遍历 + 结果集只追加的同类不变量
- `app/agent/rag_v2/nodes.py:221-223` — key 取 `metadata["id"]`，缺失时 fallback 到 content 的 md5
- `app/agent/rag_v2/nodes.py:230` — `unique_docs[:FINAL_TOP_K]` 尾部截断
- 算法题实现代码：项目中没有，不能硬套。`tests/` 下无链表、区间、图这类数据结构练习，上面的代码是现场手写

---

## 五、小红书 AI Agent 开发一面

来源：牛客【三年面试五年模拟】2026-08-12_小红书_AI Agent开发一面面经（含完整答案）

原帖题号为 1、2、3-12（折叠成一条）、13-22、25-28、反问。**23、24 两题原帖缺号，公开页面没有内容，按边界处理不自行补题。** 第 3-12 题原帖本身就是一条折叠描述（"全都是实习相关问题"），不是十道独立题，所以按一题回答"项目深挖怎么组织"，不拆成十问。

### 面经 01 · 2026-08-12

#### Q1 · 自我介绍一下。

**回答**

自我介绍是给后面的追问铺路，不是把简历念一遍。我会控制在 60 到 90 秒，走"定位 → 一段代表工作 → 能力主线 → 岗位落点"。

定位一句话：后端出身，现在做 Agent 应用工程，主栈 Python + FastAPI + PostgreSQL + Milvus。

代表工作我只讲一个，讲深：这个仓库是一套企业知识问答 + AIOps 告警诊断服务。我负责的是可靠性和可观测性这条线——把一个"抖一下就整体 500"的多路检索 RAG，改造成单分支失败只降级、总预算耗尽仍交付部分结果、每个节点的耗时和降级原因都能落库回查的系统。具体到机制：`retrieve_each_node` 把异常兜在分支内不向上抛，因为 LangGraph 的语义是任一并行分支抛异常整个 super-step 失败；`dedup_node` 作为 fan-in 后第一个节点判定降级原因，因为单分支看不到"是全挂了还是只有我挂了"；`service.py` 用 `astream` 而不是 `ainvoke`，超时后 `last_state` 里还留着已经跑完的部分。

能力主线是连续的，不是临时转方向：后端练的并发、连接池、幂等、熔断、指标，在 Agent 系统里全都要用——只是被调用的下游从 MySQL 换成了模型和向量库，而模型的失败率比数据库高一个量级，所以这套东西更需要而不是更不需要。

落到岗位：我对 Agent 的理解不是给模型套一层 Prompt，是把模型的概率决策约束在真实系统能接受的边界里。内容社区和广告场景尤其需要这个——模型可以扩大候选空间，但预算、频控、合规必须是确定性代码说了算。

**项目证据**
- `app/agent/rag_v2/nodes.py:158-176` — `retrieve_each_node` 的 docstring 写明 LangGraph 并行分支异常语义与兜异常的理由
- `app/agent/rag_v2/nodes.py:234-262` — `dedup_node` 在 fan-in 后判定 `PARTIAL_RETRIEVAL` / `RETRIEVAL_FAILED` / `RETRIEVAL_EMPTY`
- `app/agent/rag_v2/service.py:50-75` — `asyncio.timeout` + `astream(stream_mode="values")`，超时保留 `last_state`
- `app/core/circuit_breaker.py:80-266`、`app/core/metrics.py:85-119` — 熔断器与指标
- 实习经历本身不在本仓库，无法引行号

#### Q2 · 介绍一下你这个___实习关注指标是什么？

**回答**

题目里的空白要按真实实习补，我这里按本仓库的口径答，因为能引到代码和评测脚本的才是我能兑现的部分。

我不会只报一个结果指标，会给指标树加口径。这个项目分四层：

第一层检索质量。`tests/eval/metrics.py` 里实现了 Hit@K、Recall@K、MRR 三个，跑在 `golden_set_expanded.jsonl` 的 83 条样本上。口径必须说清：`expected_doc_ids` 是文档级，`expected_chunk_ids` 大部分样本是空数组，所以我只能声称文档级命中，块级命中没有标注就不报。

第二层答案质量。`validate_answer_node` 输出 `groundedness_score` 和 `coverage_score`，阈值 0.75 / 0.60，低于阈值标 `is_bad_case` 落进 `chat_run_traces`。这里要坦白一个限制：打分是 LLM 自评，不是人工标注，所以它能筛 bad case 供人复查，不能当成事实正确率的最终口径。`faithfulness_placeholder` 和 `answer_relevance_placeholder` 至今是 `NotImplementedError`，这块没做完就不能报数。

第三层可靠性。降级率按原因分维度看：`degrade_total` 这个 Counter 带 `reason` label，`RETRIEVAL_FAILED` 涨要修 Milvus，`RETRIEVAL_EMPTY` 涨要补文档，混在一起看就没有可执行动作。`retrieval_failures_total` 带 `code` label 区分"连不上"和"熔断打开"。

第四层成本与延迟。`llm_call_duration_seconds` 是 Histogram，桶边界 `(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0)`——选 Histogram 不选 Summary 是因为桶计数可以跨实例相加，Summary 的分位数不能。span 级耗时落 `chat_run_spans`，能看到单次请求里每个节点各花了多久。

最后是我的影响链路，也是最该说实话的部分：我改的是可靠性层，中间指标是降级率和 P95，但**这个项目没有线上流量，没有 A/B**，所以我只能给故障注入下的对比和离线评测结果，不给业务收益数字。

**项目证据**
- `tests/eval/metrics.py` — Hit@K / Recall@K / MRR 已实现；`faithfulness_placeholder`、`answer_relevance_placeholder` 为 `NotImplementedError`
- `tests/eval/golden_set_expanded.jsonl` — 83 条，`expected_chunk_ids` 多为空，限定了可声称的口径
- `app/agent/rag_v2/nodes.py:32-33` — `MIN_GROUNDEDNESS_SCORE = 0.75`、`MIN_COVERAGE_SCORE = 0.60`
- `app/models/chat_run_trace.py:44-46` — `coverage_score` / `groundedness_score` / `is_bad_case` 落库字段
- `app/core/metrics.py:83,85` — `_LLM_BUCKETS` 与 `llm_call_duration_seconds`
- `app/core/metrics.py:94,100` — `degrade_total{reason}`、`retrieval_failures_total{code}`
- `app/models/chat_run_span.py:83,94` — span 的 `node` 与 `duration_ms`
- 线上 A/B、业务转化指标：项目中没有，不能硬套

#### Q3-12 · 全都是实习相关问题，涉及 Agent 就狂追问，问了问数字指标，还有业务上下游，代码框架。

**回答**

原帖把第 3 到 12 题折叠成了这一条描述，没有列出十道独立题目，所以我按"项目深挖怎么组织"回答，不补造十问。

我的做法是提前给同一个项目画一张"可追问地图"，六层，每层都必须能落到文件和行号，不能停在名词上。

**业务上下游。** 输入两个入口：知识问答从 `/api/chat_v2` 进，AIOps 从 `/api/aiops/alerts/analyze` 进；PDF 协议入库从 `/protocol-pdfs/upload` 进，返回 202 后转异步。输出给谁：问答直接回前端 SSE，诊断结果落 `aiops_diagnosis` 表。失败影响谁：问答降级用户看得见（答案带"证据不完整"），入库失败只影响后台任务，但 `record_job_failure` 要把原因写回数据库，否则用户在列表页看到的永远是"处理中"。

**系统架构。** 接入层 FastAPI + 三个 middleware（`RequestIdMiddleware` 最后添加所以最外层）；状态层 PostgreSQL + SQLAlchemy；编排层 LangGraph `StateGraph`；知识层 Milvus + BM25 混合检索；工具层 MCP（两个 server，7 个工具）；异步层 Redis + RQ；观测层 loguru + Prometheus + span 落库。

**Agent 决策边界。** 这一层最容易被追问穿。模型决策的只有三处：改写子查询（`rewrite_node`）、生成答案（`generate_node`）、质检打分（`validate_answer_node`）。其余全是确定性代码：fan-out 数量固定 3、`top_k` 固定、去重规则固定、降级原因由 `if/elif` 判定、HTTP 状态码由 `exception_handlers.py` 按异常类型裁决。说清"哪里不用模型"比说"哪里用了模型"更能证明我想过边界。

**代码边界。** 我负责的包：`app/core/`（errors、circuit_breaker、metrics、span_context、llm_factory）、`app/agent/rag_v2/`、`app/workers/`。用了哪些开源件、为什么改默认实现：`langchain-openai` 的 `request_timeout` 默认 `None` 会**无条件覆盖** OpenAI SDK 的 `DEFAULT_TIMEOUT`，等于没有超时，所以必须走 `llm_factory` 统一注入；RQ 的 `Retry(max=N)` 是**无条件**重试，不区分异常类型，所以 `apply_retry_policy` 要按异常把 `retries_left` 清零。

**数字指标。** 每个数字都要知道公式和来源，见上一题。

**失败案例。** 最难的一个：并行检索里一个分支抛异常，整张图崩，用户拿 500，而另外两个分支已经召回了足够证据。定位靠的是 span 表——改造前四条分支的日志交织在一起对不出各自耗时，这才是先做 span 再做熔断的顺序原因。

回答追问我用"先结论、再机制、后证据"三段式。不是想到什么说什么，也不把团队架构说成个人贡献。

**项目证据**
- `app/api/chat_v2.py`、`app/api/aiops.py:50`、`app/api/protocol_pdf.py:36` — 三个业务入口
- `app/main.py:58-71` — middleware 层序，`add_middleware` 最后添加 = 最外层
- `app/agent/rag_v2/nodes.py:91,264,334` — 模型决策仅此三处
- `app/agent/rag_v2/nodes.py:28-30` — `NUM_SUB_QUERIES=3`、`RETRIEVE_TOP_K=4`、`FINAL_TOP_K=6` 全为常量
- `app/core/llm_factory.py` 模块 docstring — `request_timeout` 覆盖 SDK 默认超时的记录
- `app/core/job_failure.py:84-128` — `apply_retry_policy` 按异常类型清零 `retries_left`
- `app/core/span_context.py:85,160`、`app/models/chat_run_span.py` — span 记录与落库
- `mcp_servers/cls_server.py`（5 个 tool）、`mcp_servers/monitor_server.py`（2 个 tool）
- 实习项目的上下游：不在本仓库，无法引行号

#### Q13 · 说一下你这个项目 Agent 用在哪里。

**回答**

先说结论：这个仓库里**只有一处是真 Agent**，其余是 Workflow。这个区分我必须先讲，否则后面全是虚的。

真 Agent 是 `app/services/rag_agent_service.py`：用 LangChain 的 `create_agent` 构造，工具集是 `[retrieve_knowledge, get_current_time]` 加上运行时从 MCP 加载的 7 个工具，配 `MemorySaver` 做 checkpointer。模型自己决定调不调工具、调哪个、调几次，这是动态决策，是 Agent。

`app/agent/rag_v2/` 不是 Agent，是固定 DAG：`rewrite → retrieve_each(×3) → dedup → generate → validate_answer → END`。边是 `add_edge` 写死的，只有 fan-out 那条用了 `add_conditional_edges`，但它返回的也只是"按子查询数量发 N 个 Send"，不是"模型决定下一步去哪"。模型在里面只负责三件事：改写、生成、打分。它决定不了流程走向。

为什么这么切：问答链路的步骤是确定的——必须先改写才能多路检索，必须先检索才有证据，必须先有答案才能质检。这种情况用 Agent 只会引入不确定性和额外的模型调用，没有收益。反过来 AIOps 诊断要看告警内容决定查 CPU 还是查日志还是查内存，那才需要模型选工具。

哪里坚决不用模型：HTTP 状态码由 `exception_handlers.py` 按异常类型裁决；降级原因由 `dedup_node` 的 `if/elif` 判定；重试值不值得由 `apply_retry_policy` 按异常类型裁决；熔断开不开由 `CircuitBreaker` 按失败计数裁决。这些都是"错一次代价明确"的判断，交给概率模型不划算。

还有一处要说实话：`app/agent/aiops/` 下的 planner、executor、replanner 已经从工作区删掉了（见 git status），所以**Plan-Execute-Replan 这条路径目前不存在**，我不能拿它当卖点。

**项目证据**
- `app/services/rag_agent_service.py:101,107,131,133-135` — `create_agent` + `MemorySaver` + 工具集组装，唯一的真 Agent
- `app/agent/rag_v2/graph.py:36-52` — 五个 `add_edge` 写死的固定 DAG
- `app/agent/rag_v2/graph.py:20-23,44` — `_fanout_to_retrieve` 按子查询数量发 Send，不是模型选路
- `app/agent/rag_v2/nodes.py:91,264,334` — 模型只做改写、生成、打分
- `app/core/exception_handlers.py`、`app/core/job_failure.py:77`、`app/core/circuit_breaker.py:172` — 三处刻意不交给模型的裁决
- `app/agent/aiops/planner.py`、`executor.py`、`replanner.py` — 已删除（git status: `D`），Plan-Execute-Replan：项目中没有，不能硬套

#### Q14 · OpenManus 是用的什么框架？

**回答**

按公开仓库代码，OpenManus **不是 LangChain/LangGraph 的套壳**，是一套 Python 自研的轻量 Agent/Flow 框架。

核心继承链 `BaseAgent → ReActAgent → ToolCallAgent → Manus`：`BaseAgent` 管运行循环、状态和消息；`ReActAgent` 抽出 `think()` / `act()` 两个抽象方法；`ToolCallAgent` 用 OpenAI 兼容的 tool calling 接口选工具、执行、把 Observation 写回 Memory；`Manus` 组装 Python 执行、浏览器、文本编辑、询问用户、终止和 MCP 工具，并设最大步数与 Observation 长度上限。

多 Agent / 长任务在独立的 `flow` 层：`PlanningFlow` 用 PlanningTool 建步骤、维护步骤状态，再按步骤类型选 Executor。官方 README 至今把 `run_flow.py` 标为不稳定的多 Agent 版本，所以不能说成生产级分布式编排。

依赖层面：Pydantic 做数据模型和配置，OpenAI SDK 做模型调用，Tenacity 做重试，Browser-Use/Playwright 做浏览器，MCP SDK 接外部工具，另有 FastAPI、Docker、Crawl4AI。团队成员来自 MetaGPT 社区，但"作者来自 MetaGPT"不等于"运行时基于 MetaGPT"——判断框架看继承链和依赖，不看作者背景。

跟本仓库对照有一处值得说：OpenManus 的 `ToolCallAgent` 是自己写 tool calling 循环，本仓库是用 LangChain 的 `create_agent` 直接拿现成的循环。自研的好处是循环里每一步都能插自己的检查（最大步数、Observation 截断），代价是得自己维护；用 prebuilt 的好处是省事，代价是想加步数上限就得绕。这也是本仓库目前**没有最大步数保护**的直接原因——`create_agent` 的循环我没接管。

**项目证据**
- `app/services/rag_agent_service.py:9,133-135` — 用 `from langchain.agents import create_agent`，循环由框架提供，未自研
- `app/agent/mcp_client.py:18-73` — 本仓库的 MCP 接入方式，用 interceptor 包重试
- OpenManus 源码：不在本仓库，以上是对公开仓库的阅读结论，不引本地行号
- 自研 Agent 循环、最大步数上限、Observation 截断：项目中没有，不能硬套

#### Q15 · 讲一下你怎么理解 Harness？

**回答**

Harness 是模型外面那一整套运行和治理环境：决定模型看见什么、能调什么、动作怎么执行、状态存在哪、失败怎么恢复、结果谁来验。模型给概率性的下一步决策，Harness 把这个决策约束在真实系统能接受的边界内。同一个模型放进不同 Harness，任务完成率和成本能差出量级——这不是模型变强了，是长任务被外化成了可见状态、真实工具结果被反馈回去了、错误被限制在可恢复的步骤里了。

对着本仓库逐个模块说，有的有有的没有，我分开讲。

**有的部分。** Context Builder 有：`conversation_memory_service.build_context` 按 Token 预算把"滚动摘要 + 最近窗口"投影进 Prompt，`_format_context` + `_truncate_to_token_budget` 管证据侧预算。Tool Runtime 部分有：MCP 的 `retry_interceptor` 管重试和超时，工具描述靠 `@tool` 的 docstring。State/Checkpoint 部分有：`MemorySaver` 存对话，PostgreSQL 存 trace 和 span，但**任务级 checkpoint 没有**——`rag_v2_graph` 的 `compile()` 不带 checkpointer，跑完即 END。Reliability 有：熔断、降级、重试、部分结果，这块是我做的重点。Trace 有：`chat_run_traces` + `chat_run_spans` 两张表加 Prometheus 指标。

**没有的部分，得说清。** 沙箱没有——`grep` 全仓无 sandbox/permission 相关实现，现有 MCP 工具全是只读查询，所以还没被逼到必须做隔离。权限层没有。人工审批没有。Eval Harness 只有半套：`tests/eval/run_eval.py` 能跑检索指标和对比实验，但答案质量的 faithfulness / answer relevance 是 `NotImplementedError`，回归门禁也没有接进 CI。

区分两个语境有用：Runtime Harness 管线上运行和控制，Eval Harness 管构造任务、回放轨迹、评分和版本比较。两者可以共享 trace、沙箱和验证器，但不是一个东西。本仓库的 Runtime 侧比 Eval 侧完整得多，这是我下一步要补的方向。

**项目证据**
- `app/services/conversation_memory_service.py:93-131` — `build_context`，Context Builder 的对话侧
- `app/agent/rag_v2/nodes.py:613,631,637` — `_format_context` / `_estimate_tokens` / `_truncate_to_token_budget`，证据侧 Token 预算
- `app/agent/mcp_client.py:18-73,150` — Tool Runtime 的重试与 interceptor 注册
- `app/services/rag_agent_service.py:107` — `MemorySaver`，对话级 checkpoint
- `app/agent/rag_v2/graph.py:52` — `compile()` 不带 checkpointer，任务级 checkpoint 缺失
- `app/core/circuit_breaker.py`、`app/core/errors.py:25-220` — Reliability 层
- `app/models/chat_run_trace.py`、`app/models/chat_run_span.py`、`app/core/metrics.py` — Trace/Metrics 层
- `tests/eval/run_eval.py:449-467` — Eval Harness 的 CLI 参数
- 沙箱 / 权限策略 / 人工审批 / CI 回归门禁：项目中没有，不能硬套

#### Q16 · 多 Agent 并发有什么性能问题？

**回答**

先给结论：本仓库**没有多 Agent**，只有单图内的并行分支（fan-out 3 条子查询检索）。所以我把通用问题讲清，再说哪几条我在这条并行链路上真的踩到了。

通用的六类问题：

一是模型网关瓶颈。fan-out 会快速吃掉 RPM/TPM 触发 429，而且每个子分支都重复携带 system prompt 和工具 schema，Token 近似随分支数线性涨。

二是尾延迟放大。汇总节点等最慢的分支，分支越多撞上一个慢调用或一次重试的概率越高，P99 可能反而变差。

三是下游资源争用。数据库连接池、Redis、向量库、第三方 API 都有上限，上游无限并发只是把等待推到下游队列里。

四是共享状态竞争。多个分支同时改计划、Memory 或业务记录会丢更新、重复副作用。

五是重复探索和上下文膨胀。子分支可能检索同一份资料，汇总时全塞进上下文。

六是协调故障。互相等待、取消传播失败、部分成功没人处理。

我在这条 fan-out 链路上真踩到的是第二和第六类，而且是同一个 bug 的两面：改造前任一分支抛异常，LangGraph 的语义是整个 super-step 失败、整张图崩——三个分支里两个已经召回了充足证据，用户拿到的还是 500。这是"部分成功没人处理"最直白的形态。修法是在 `retrieve_each_node` 里把异常兜住，返回 `{"documents": [], "retrieve_failures": [...]}`，让"一个分支失败 = 证据少一份"而不是"整个请求失败"。

第二类我是靠 span 表才看清的。改造前四条分支的日志交织在一起，根本对不出每条各自花了多久，也就判断不出尾延迟是被谁拖的。`instrument_node` 在注册时统一包装，fan-out 出去的 N 个实例各记一条 span，这才有了按分支的耗时明细。

第三类我用熔断器处理了一个具体场景：Milvus 挂掉时，如果熔断包在 `to_thread` 里面，拒绝发生在已经拿到线程池 worker 之后，省下的只有网络往返；包在外面，连 worker 都不占用——600 个请求各占一个 worker 白等 60 秒，这才是最紧张的那个资源。

第一、四、五类在这个仓库里没有真实压力：分支数固定 3，不共享可变状态（靠 `Annotated[List, operator.add]` reducer 归并，各分支只 append 自己的），检索侧也没有模型调用。

**项目证据**
- `app/agent/rag_v2/graph.py:20-23,44` — `Send` fan-out，分支数 = 子查询数
- `app/agent/rag_v2/nodes.py:28` — `NUM_SUB_QUERIES = 3`，fan-out 上限固定
- `app/agent/rag_v2/nodes.py:158-176` — docstring 记录"任一分支抛异常 → 整个 super-step 失败 → 整张图崩"及修法
- `app/agent/rag_v2/nodes.py:196-212` — 失败分支返回 `retrieve_failures` 而非抛出
- `app/agent/rag_v2/nodes.py:175-186` — 熔断包在 `to_thread` 外面，保住线程池 worker
- `app/agent/rag_v2/instrumentation.py:117,131` — `instrument_node` 让每个 fan-out 实例各记一条 span
- `app/agent/rag_v2/state.py` — `Annotated[List[...], operator.add]` reducer 做无竞争 fan-in
- 多 Agent、Supervisor、子 Agent 交接：项目中没有，不能硬套

#### Q17 · 你平时怎么控制 Agent 并发量？

**回答**

这题我得先承认一个缺口：本仓库**没有并发闸门**。`grep` 全仓没有 `Semaphore`、没有限流器、没有 token bucket、没有对模型调用的并发上限。有的只是三样间接约束，我按实际情况说。

**已有的三样。** 第一是 fan-out 上限固定：`NUM_SUB_QUERIES = 3`，所以单个请求最多 3 路并行检索，不会因为模型改写出 20 个子查询就打爆下游。这是"用常量替代限流"，够用但不是限流。第二是总预算：`chat_total_budget_seconds = 90.0`，`asyncio.timeout` 包住整个 astream，超了就带部分结果返回——这限制的是单请求占用时长，间接限制了在途请求堆积。第三是熔断器：下游连续失败后直接拒绝，这是过载时的最后一道，但它是"坏了才拒"，不是"忙了就排队"。

**异步侧有真队列，但不是给 Agent 的。** Redis + RQ 那套（`index_queue`、`protocol_pdf_queue`）管的是文档索引和 PDF 入库，是后台任务，跟在线问答的并发不是一回事。这里不能混着说。

**如果要补，我会怎么补。** 分层并发预算，不是设一个全局线程数：

入口层做 admission control，队列满就明确拒绝或降级，不无限堆积。运行时按 `tenant → user → task → model/tool` 分层设上限：模型调用受 RPM/TPM 和成本预算约束，向量库按连接池约束，第三方 API 按对方限额约束。实现上 `Semaphore` 限同时执行数，token bucket 限速率，bulkhead 给不同租户和工具隔离资源，有界队列做 backpressure。

并发值要靠压测和线上反馈校准：看 429、工具错误率、队列等待和 P95 动态降上限，稳定后缓慢探测上调，加滞回避免振荡。

本质上并发控制不是为了"同时跑更多"，是让每个成功任务的资源消耗可预测。这个仓库目前是靠"分支数写死 + 总预算 + 熔断"兜住的，在单机低流量下没出问题，但它不是并发控制，我不会把它说成并发控制。

**项目证据**
- `app/agent/rag_v2/nodes.py:28` — `NUM_SUB_QUERIES = 3`，用常量限住 fan-out
- `app/config.py:81` — `chat_total_budget_seconds = 90.0`
- `app/agent/rag_v2/service.py:51-58` — `asyncio.timeout` 包住整个流式执行
- `app/core/circuit_breaker.py:135-193` — `allow` / `record_failure`，过载时拒绝
- `app/core/task_queue.py:9-11` — RQ 队列，仅用于索引与 PDF 入库，非在线问答
- Semaphore / 限流器 / token bucket / admission control / bulkhead / 分租户配额：项目中没有，不能硬套

#### Q18 · 讲一下长时记忆和短时记忆的底层实现方式？

**回答**

短时记忆撑当前任务的连续性，长时记忆撑跨会话复用。两者底层都不是"把聊天记录塞进向量库"。

**本仓库的短时记忆是真做了的，讲细一点。** `conversation_memory_service` 是一套滚动摘要：完整原文只进 PostgreSQL 当审计记录，喂给模型的上下文只有"压缩摘要 + 未压缩消息的尾部"。关键设计有四处。

一是 Token 估算得贴合中文。`estimate_tokens` 的公式是 `cjk_chars + ceil(other_chars / 4)`——CJK 一字约一 token，ASCII 约四字符一 token。用 `len(text)/4` 那种通用估法在中文上会低估三到四倍，预算就形同虚设。`_message_token_cost` 再加 4，是消息结构本身的开销。

二是快照游标。摘要不是每轮重算，是记住"上次压缩到哪条消息"，之后的消息才是 `pending_messages`。这样第 11 轮只需要在旧摘要上叠加新增部分，不用把 11 轮重跑一遍。游标指向的消息被软删除时，`_messages_after_snapshot` 返回空列表——保守地不注入旧原文，避免重复历史。

三是触发条件是 Token 数不是轮数。`compact_candidates` 是 `pending_messages` 去掉尾部保留窗口的部分，只有它的 token 数超过阈值才触发压缩。按轮数触发会在"十轮短寒暄"和"两轮长文档"之间失效。

四是摘要失败不能阻断聊天。`_summarize` 的 except 分支返回空字符串，退回最近消息窗口——摘要是省 Token 的优化，不是主链路。

摘要 prompt 用固定四个小标题【明确事实】【任务进展】【约束与偏好】【待确认项】，并要求缺失项写"无"，是为了让输出可解析、长度可控。

**长时记忆本仓库没有，得说清界限。** 有的是知识库检索（Milvus + BM25 混合），但那是 RAG，不是记忆——它没有用户/租户作用域，没有来源和置信度字段，没有有效期，没有冲突更新，也没有删除可达性。把 RAG 说成长期记忆是最常见的偷换。

如果要做，我会按类型分库：稳定偏好和身份事实进关系表，历史事件进 append-only log，自然语言经验进向量 + 全文混合索引，实体关系复杂才上图库。每条至少带 `tenant/user/project` 作用域、来源、时间、置信度、有效期、权限和原始事件引用。生命周期是候选提取 → 敏感性/重要性过滤 → 去重结构化 → 写入 → 按需检索 → 权限/时效/相关性重排 → 注入 → 冲突更新或遗忘。优先级顺序是：当前用户明确表达 > 业务主库实时状态 > 历史摘要 > 模型推断。

还有一点常被说错：KV Cache 是推理加速缓存，不是 Agent Memory，两者生命周期和用途都不一样。

**项目证据**
- `app/services/conversation_memory_service.py:48-54` — `estimate_tokens`，`cjk_chars + ceil(other/4)`
- `app/services/conversation_memory_service.py:57-58` — `_message_token_cost` 加 4 的结构开销
- `app/services/conversation_memory_service.py:93-131` — `build_context`：快照游标 → `pending_messages` → `compact_candidates` → 按 Token 阈值触发
- `app/services/conversation_memory_service.py:133-145` — `_messages_after_snapshot`，游标消息被软删时返回空
- `app/services/conversation_memory_service.py:163-186` — `_summarize` 失败返回 `""`，退回最近窗口
- `app/services/conversation_memory_service.py:28-35` — 摘要 prompt 的四个固定小标题
- `app/services/rag_agent_service.py:107,310,377` — `MemorySaver` 与历史读取/清空
- `app/services/vector_search_service.py:52-107` — 混合检索，属 RAG 非记忆
- 长期记忆表、作用域字段、置信度/有效期、冲突更新、删除可达性：项目中没有，不能硬套

#### Q19 · 讲一下 ReAct 的流程？

**回答**

ReAct 是 Thought → Action → Observation 的循环：模型先写一段推理说明下一步要干什么，再输出一个结构化的动作（工具名 + 参数），运行时执行后把真实结果作为 Observation 写回上下文，模型据此决定继续调工具还是给最终答案。关键在于 Observation 来自外部世界而不是模型自己编的，所以每一轮的决策都被真实结果校正一次。

它的好处是 grounding：模型不必只靠参数知识猜，可以去查。代价是局部贪心（每步只看下一步，容易在错误方向上走很远）、工具循环（同一个调用反复发）、Observation 累积挤占上下文，以及每步串行等待带来的延迟。

这个仓库里两条路径的选择正好说明了什么时候该用、什么时候不该用。

`rag_agent_service.py` 那条是 tool-calling loop，本质上就是 ReAct 的现代形态——只是 Thought 不再是自由文本，而是由 `create_agent` 内部的消息序列承载。工具三类：`retrieve_knowledge`（知识检索）、`get_current_time`（时间）、加上启动时从 MCP 拉回来的日志和监控工具。模型确实在运行时决定调哪个，因为"用户问的是文档知识还是当前 CPU 指标"这件事没法在编码期定死。

`rag_v2` 那条**刻意不是** ReAct。它是固定 DAG：rewrite → fan-out 到 N 个 retrieve_each → dedup → generate → validate_answer → END。为什么不用循环？因为 RAG 问答的步骤本来就是确定的——先改写、再检索、再生成、再校验，顺序不依赖运行时观察。让模型每一步都重新决定"接下来干什么"，只会引入不必要的不确定性和额外的 LLM 调用。固定 DAG 换来的是：耗时可预测、每个节点可以单独埋点、失败位置一眼可见。

我要老实说的是缺口：这套 tool-calling loop 没有最大步数上限，也没有重复动作检测。`grep -rn "recursion_limit"` 整个仓库为空，`create_agent` 用的是框架默认值。防线只有一层——`chat_total_budget_seconds = 90.0` 的总预算。这挡得住"卡死"，挡不住"在 90 秒里把同一个工具调 20 次"。生产上应该补动作指纹和步数上限，这条我没做。

**项目证据**
- `app/services/rag_agent_service.py:133-137` — `create_agent(model, tools=all_tools, checkpointer=self.checkpointer)`，tool-calling loop
- `app/services/rag_agent_service.py:101,123-131` — 本地工具 + MCP 工具合并成 `all_tools`
- `app/tools/knowledge_tool.py:13-25`、`app/tools/time_tool.py:10-11` — 两个本地工具及其给模型看的描述
- `app/agent/rag_v2/graph.py:41-49` — 固定 DAG，无循环边
- `app/config.py:81` — `chat_total_budget_seconds = 90.0`，唯一的兜底
- 最大步数上限、重复动作指纹、`recursion_limit` 显式设置：项目中没有，不能硬套。全仓库 grep `recursion_limit` 为空

#### Q20 · Planner、Executor、Critic 分别怎么做的。

**回答**

先说三者的职责分工。Planner 把目标拆成有依赖的步骤，每步写清输入、期望产物、可用工具和完成条件，计划要外置并版本化，失败时做局部重规划而不是整份重写。Executor 只执行当前 ready 的那一步，内部可以用 ReAct 选工具，受权限、超时、幂等约束，不能自己改业务硬规则。Critic 按 rubric、测试、业务状态判定结果合格与否，输出 pass / revise / replan / escalate。

对着这个仓库，我必须分三种情况说，因为三者的落实程度完全不同。

**Critic 是真的有，而且是完整的一层。** `validate_answer_node` 拿问题、答案、证据三样东西送给 LLM，要求它输出 groundedness（答案有多少被证据支持）和 coverage（证据覆盖了问题的多少）两个分数，阈值 0.75 和 0.60。低于阈值就把答案换成保守回答。这一层还带 fail-open：`enable_answer_validation` 关掉时走 `_unvalidated_validation`，降级原因记 `FEATURE_DISABLED` 而不是 `LLM_ERROR`——后者会把值班同学送去查模型配额，而那边一切正常。

**Executor 有对应物但不是通用执行器。** `retrieve_each_node` 是被 `Send` fan-out 出来的 N 个并行实例，每个跑一条子查询。它的关键设计是任何异常都不向上抛，只记一条失败明细进 `retrieve_failures`——因为 LangGraph 的语义是任一并行分支抛异常整个 super-step 就失败，改造前只要 Milvus 抖一下，另外三个分支已经召回的证据会被整批丢掉。

**Planner 现在不在代码里。** `rewrite_node` 把一个问题拆成 3 条子查询，形式上像规划，但它拆的是**检索角度**不是**任务步骤**——3 条子查询之间没有依赖关系，可以完全并行，不存在"第 2 步要等第 1 步的产物"。真正的任务规划器曾经有过：`app/agent/aiops/planner.py`、`executor.py`、`replanner.py` 这三个文件在 git status 里是 deleted 状态，那是一套 Plan-Execute-Replan 的 AIOps 诊断流程，现在已经从仓库里移除了。所以严格讲，任务级 Planner：项目中没有，不能硬套。

还有一个缺口要讲：Critic 只判一次，不打回重做。判不过就降级成保守回答，没有 revise 环，也没有"最大修订次数"这个概念——因为没有环就不需要这个上限。这是有意的取舍（省一轮 LLM 调用、耗时可预测），但确实不是完整的 Plan-Execute-Critic 闭环。

**项目证据**
- `app/agent/rag_v2/nodes.py:334-` — `validate_answer_node`，真正的 Critic 层
- `app/agent/rag_v2/nodes.py:32-33` — `MIN_GROUNDEDNESS_SCORE = 0.75`、`MIN_COVERAGE_SCORE = 0.60`
- `app/agent/rag_v2/nodes.py:60` — `VALIDATION_SYSTEM_PROMPT`，证据校验与幻觉拦截的判分标尺
- `app/agent/rag_v2/nodes.py:558` — `_unvalidated_validation`，开关关闭时的 fail-open 分支
- `app/agent/rag_v2/nodes.py:158-212` — `retrieve_each_node`，Executor 对应物，异常不上抛
- `app/agent/rag_v2/nodes.py:91-157` — `rewrite_node`，拆检索角度不拆任务步骤
- `app/agent/rag_v2/nodes.py:28` — `NUM_SUB_QUERIES = 3`
- 任务级 Planner / Replanner：项目中没有，不能硬套。`app/agent/aiops/planner.py`、`executor.py`、`replanner.py` 在 git status 里是 deleted，已从仓库移除
- Critic 的 revise 回环、最大修订次数、escalate 到人工：项目中没有，不能硬套

#### Q21 · Critic 什么意思？为什么不用 ReAct？

**回答**

先纠正一下题目里的隐含前提：Critic 和 ReAct 不在同一个维度上，所以不构成"用哪个"的二选一。ReAct 描述的是执行过程中 Thought/Action/Observation 的局部循环，Critic 描述的是质量反馈机制。常见组合恰恰是两者并存——Executor 在步骤内部用 ReAct 调工具，Critic 在步骤边界验收产物。

Critic 本身是评估器：判定当前产物是否满足 rubric、事实和业务约束，给出修订或停止信号。它可以是规则、单测、奖励模型、LLM Judge 或人工，不天然等于一个独立大模型。能用确定性检查的就别用 LLM——schema 校验、数据库状态、单元测试都比 LLM Judge 稳。

这个仓库的选择正好能回答"为什么这条路不用 ReAct"。`rag_v2` 是固定 DAG + 末端 validator，理由是 RAG 问答的步骤序列不依赖运行时观察：无论用户问什么，都是改写、检索、生成、校验这四步，顺序固定。让模型每步重新决定"接下来干什么"，多花 LLM 调用、耗时不可预测、失败位置难定位，换来的灵活性在这个场景里用不上。

而同一个仓库里 `rag_agent_service.py` 那条路**保留了** tool-calling loop，因为那边的步骤真的不确定——用户问"现在几点"要调时间工具，问"CPU 怎么样"要调 MCP 监控工具，问文档内容要调知识检索，这个分支判断没法编码期定死。

所以真正的答法是：按任务的确定性选控制模式，不按名词新旧选。步骤确定 → Workflow + Validator；步骤依赖外部观察 → ReAct。两条路径在同一个仓库里共存，正好证明这不是非此即彼。

Critic 也不是越多越好。LLM Critic 有偏好偏差、位置偏差和自洽偏差，自己评自己尤其容易放过错误。我这套用的是独立调用（`temperature=0.0`、`streaming=False`）而不是让生成模型自评，但同一个 `qwen-max` 既生成又判分，自洽偏差是真实存在的，我没法说它已经解决了。

**项目证据**
- `app/agent/rag_v2/graph.py:41-49` — 固定 DAG，刻意不做 ReAct 循环
- `app/agent/rag_v2/nodes.py:334-` — 末端 validator 承担 Critic 职责
- `app/agent/rag_v2/nodes.py` 内 `validate_answer_node` 中 `llm_factory.create_chat_model(temperature=0.0, streaming=False)` — 判分用独立调用、零温度
- `app/services/rag_agent_service.py:133-137` — 另一条路径保留 tool-calling loop，因为步骤不确定
- `app/config.py:46,129` — 生成与判分同为 `qwen-max`，自洽偏差未解决
- 盲评、位置随机化、标尺校准、人工抽检：项目中没有，不能硬套

#### Q22 · 上下文怎么做的？你是拆分的嘛？

**回答**

拆了。但要先说清一件事：模型最终看到的仍是一段连续 Token，所谓"拆分"是工程侧按来源、生命周期、权限和优先级分区管理，再由组装层为当前这一步拼出最小充分上下文。

这个仓库拆成三块，每块有独立的 Token 预算，而且**分区之间用标签隔开**，这一点比预算分配更重要。

**第一块是会话记忆。** `_render_context` 拼上下文时，摘要区带 `【会话滚动摘要】` 头，最近消息区单独一段。注意 `summary_header` 本身要算进预算（`estimate_tokens(summary_header + summary)`）——标题不是白送的。摘要区的内容由固定四个小标题组织：【明确事实】【任务进展】【约束与偏好】【待确认项】，缺失项写"无"。固定小标题不是为了好看，是为了让下一轮摘要能在同样的结构上叠加，而不是每次都重新自由发挥。

**第二块是检索证据。** `_format_context` 逐条扣预算：每加一条文档，先算 prefix 占多少，剩下的额度给正文，正文超了就 `_truncate_to_token_budget` 截断。这样不会出现"前两条塞满、后四条一条都进不去"。

**第三块是分区声明本身。** `MEMORY_POLICY` 这段文本被同时注入到 generate 和 validate 两个节点的 prompt 里，明确写"会话记忆只用于理解用户已明确说明的对象、任务延续和约束"。validate 节点还额外加一句"仅将 `<evidence>` 中的内容作为知识库可验证证据；会话记忆只能帮助判断对象指代"。这是为了防一类具体错误：模型拿会话里用户随口说的话当证据，然后 groundedness 判分时自己给自己放行。分区不只是切开，还要告诉模型每个分区能干什么。

标签用的是 XML 风格：`<question>`、`<answer>`、`<evidence>`、`<conversation_memory>`。选闭合标签而不是 markdown 标题，是因为证据正文里可能本身就带 `##`，闭合标签的边界更明确。

Token 估算这里有个中文特化：`estimate_tokens` 把 CJK 字符按 1 token 算，其余按 4 字符 1 token 算，然后每条消息再加 4 的结构开销。不是精确 tokenizer，但对中文对话的偏差比通用的"字符数除 4"小得多——后者会把中文对话的 Token 数低估三到四倍，预算控制直接失效。

**项目证据**
- `app/services/conversation_memory_service.py:207-230` — `_render_context`，摘要区 + 最近消息区分段拼装
- `app/services/conversation_memory_service.py:213,217,220` — `summary_header` 与它自身的 Token 成本计入预算
- `app/services/conversation_memory_service.py:28-35` — 摘要四个固定小标题
- `app/services/conversation_memory_service.py:249-273` — `_take_tail_to_budget`，最近消息按预算从尾部取
- `app/services/conversation_memory_service.py:48-58` — `estimate_tokens` 的 CJK 特化 + 每条消息 +4
- `app/agent/rag_v2/nodes.py:613-628` — `_format_context` 逐条扣预算
- `app/agent/rag_v2/nodes.py:637-` — `_truncate_to_token_budget` 单条截断
- `app/agent/rag_v2/nodes.py:87` — `MEMORY_POLICY` 分区声明
- `app/agent/rag_v2/nodes.py:110,300,388` — 同一段 policy 注入 rewrite / generate / validate 三处
- 工具 schema 分区与渐进披露、多 Agent 上下文隔离、按权限过滤召回：项目中没有，不能硬套

#### Q23-24 · 原帖缺号

原帖题号从 22 直接跳到 25，第 23、24 题在公开页面上没有内容。按"不自行补题"的边界处理，这两题留空，不猜测也不用其他题填充。

#### Q25 · 讲一下 Redis 的数据结构。

**回答**

Redis 对外暴露的是数据类型，对内会按元素数量和长度选择不同编码，这两层要分开讲。

核心类型六个。String 存二进制安全的字符串，也用来做计数器。Hash 是字段到值的映射，适合对象的字段级更新——改一个字段不用把整个对象读出来再写回。List 是有序可重复序列，两端插入删除快。Set 是无序唯一集合，支持交并差。Sorted Set 每个成员带一个 score，支持按 score 范围查询和排名，是排行榜和延时队列的常用底座。Stream 是带单调 ID 的追加日志，配 Consumer Group 提供 pending / ack 语义，这是它和 List 做队列的本质区别——List 弹出即丢失，Stream 有未确认列表可以重投。

另外还有 Bitmap / Bitfield（底层是 String，把 offset 映射到 bit）、HyperLogLog（固定小内存估算基数，结果有统计误差不是精确值）、GEO（把经纬度编码成 score 存进 Sorted Set）。Redis Stack 的 JSON、Search、TimeSeries 是模块能力，不是核心内置类型，这条边界要划清。

选型看访问模式：字段级更新用 Hash，唯一集合用 Set，排序范围用 ZSet，需要可靠消费和重投用 Stream。生产上还要管 key 大小、BigKey / HotKey、过期策略、内存淘汰、持久化和集群 slot 分布——数据结构选对不等于系统就可靠了。

**项目证据**

这个仓库里 Redis 只做一件事：RQ 的消息 broker。`task_queue.py` 从 `redis_url` 建连接，挂两个 Queue（`knowledge_index` 和 `protocol_pdf_ingest`）。上面讲的 Hash / ZSet / Stream / Bitmap / HyperLogLog / GEO 一个都没用到——队列的数据结构由 RQ 封装，我的代码没有直接碰。

- `app/core/task_queue.py:9-11` — `Redis.from_url(config.redis_url)` + 两个 `Queue`，全部 Redis 用法
- `app/config.py:58-59` — `redis_url`、`rq_queue_name = "knowledge_index"`
- `app/services/index_job_queue.py:11-28` — 入队，job 参数走 RQ 而非手写 Redis 命令
- `app/api/file.py:145,215` — 入队失败返回 503，把"Redis/RQ 不可用"这个事实透给调用方
- Hash / ZSet / Stream / Bitmap / HyperLogLog / GEO、缓存穿透击穿雪崩、分布式锁、集群 slot：项目中没有，不能硬套。Redis 在这里是 RQ 的实现细节，我没有直接使用这些结构，答案是通用知识不是项目经验

#### Q26 · 每种的底层机理了解嘛？

**回答**

了解，但要先加一句版本边界：Redis 的内部编码一直在演进，阈值也可配置，所以不能把某个版本的实现当成永恒答案。现场可以用 `OBJECT ENCODING key` 看当前对象实际编码。

String 用 SDS。它记录 len 和 alloc，所以取长度是 O(1) 不用扫到 `\0`，也支持预分配减少 realloc。按值和长度分三种编码：能转整数的用 int，短字符串用 embstr（SDS 和 redisObject 连续分配，一次内存分配），长的用 raw。

Hash 小对象用 listpack 紧凑顺序存储——元素少的时候顺序扫比哈希表快且省内存；超过元素数或单值长度阈值转 hashtable，查找摊还 O(1) 但内存开销上来了。

List 现在是 quicklist：双向链表串起多个 listpack 节点。这是在"纯链表指针开销大"和"纯连续数组插入要搬数据"之间折中。

Set 全整数且规模小用 intset（有序紧凑数组，二分查找），否则 hashtable。

Sorted Set 小集合用 listpack，大集合是 skiplist + dict 双结构：dict 做 member → score 的 O(1) 查找，skiplist 做按 score 排序和范围查询，插入删除期望 O(log n)。两个结构共享成员对象，不是存两份。

Stream 的条目按 ID 组织在 radix tree 里，叶子节点用 listpack 紧凑存。Consumer Group 维护 last-delivered-id 和 PEL（Pending Entries List），这是 ack 和重投的基础。

还有两个机制值得单独提：dict 的渐进式 rehash——扩容时保留两张表，每次操作搬一部分，避免一次性 rehash 造成长时间阻塞；listpack 替代旧 ziplist 的动因是后者存在级联更新问题（改一个元素的长度可能引发后续所有元素的 prevlen 字段连锁变长）。

**项目证据**

这题在我的仓库里没有对应实现，我必须说清楚。Redis 只作 RQ broker（`app/core/task_queue.py:9-11`），编码层完全被 RQ 和 redis-py 封装，我的代码没有一处直接发 Redis 命令，也没有观察过 `OBJECT ENCODING`。上面这些是通用知识，不是从这个项目里得到的经验，不能硬套。

要说仓库里有什么可以类比的"编码随规模切换"设计，只有一处：`_get_bm25_retriever` 按 `collection.num_entities` 做缓存 key，实体数变了就重建 BM25 索引。这和 Redis 的编码转换在"按规模换实现"这一点上形似，但机制完全不同，我只把它当类比不当同类。

- `app/core/task_queue.py:9-11` — Redis 用法的全部范围
- `app/services/vector_search_service.py:140-163` — `_get_bm25_retriever`，按规模重建的类比（仅类比）
- SDS / listpack / quicklist / intset / skiplist / radix tree / 渐进式 rehash：项目中没有，不能硬套

#### Q27 · 讲一下 Java 的类加载机制

**回答**

生命周期通常描述为加载 → 链接（验证、准备、解析）→ 初始化 → 使用 → 卸载。解析可以在规范允许的范围内延迟，所以不要把所有 JVM 实现说成完全相同的同步顺序。

加载阶段，类加载器按二进制名找到字节流，JVM 解析 class 文件，在方法区 / 元空间建立运行时表示，并创建对应的 `java.lang.Class` 对象。

验证检查文件格式、元数据、字节码和符号引用，保证类型和控制流满足 JVM 约束。这一步是安全边界，不能省。

准备为静态字段分配存储并设置零值。注意区别：带 `ConstantValue` 属性的编译期常量在这一步就赋值了，普通静态赋值语句还没执行。

解析把运行时常量池里的符号引用换成直接引用，可以惰性发生。

初始化执行 `<clinit>`，也就是静态字段赋值和静态代码块按源码顺序合并后的类初始化方法。JVM 保证同一个类只初始化一次并且加锁同步，初始化子类前先初始化父类。

触发初始化的主动使用包括 `new`、调用静态方法、访问非编译期常量的静态字段、反射、初始化子类。只引用编译期常量或者创建该类型的数组，通常不触发。

两个容易漏的点。第一，类的身份是"二进制名 + 定义它的类加载器"共同决定的——同一个 class 文件被两个加载器加载出来是两个不同的类，互相转型会抛 ClassCastException。第二，双亲委派先向父加载器请求，避免核心类被重复定义；但插件、容器和模块系统可能故意用子优先或隔离加载器打破这个顺序。类卸载要求定义它的 ClassLoader 和所有类对象都不可达，实际回收时机由 JVM 决定。

**项目证据**

项目中没有，不能硬套。这是纯 Python 仓库，`find . -name "*.java"` 结果为空，没有 JVM、没有类加载器、没有字节码验证。上面是通用知识。

唯一在"延迟加载"这个概念上形似的是 Python 的函数内 import：`llm_factory.py:125` 把 `from langchain_qwq import ChatQwen` 放在函数体里，因为那是个重量级可选依赖，模块级 import 会让所有不用 Qwen 的路径也付启动开销；`sop_retrieval_service.py:56` 和 `first_response_service.py:147` 同理。但 Python 的模块 import 是"执行一遍模块顶层代码并缓存到 `sys.modules`"，没有验证、准备、解析这些阶段，也没有双亲委派——机制完全不同，我只当类比不当等同。

- `app/core/llm_factory.py:125` — 函数内 import 重量级可选依赖
- `app/services/sop_retrieval_service.py:56`、`app/services/first_response_service.py:147` — 同类延迟 import
- Java 代码 / JVM / 类加载器：项目中没有，不能硬套。`find . -name "*.java"` 为空

#### Q28 · 场景题：通过 Agent 实现广告投放的自我迭代，自我判别哪种投放方式最优并执行

**回答**

我不会让 Agent 拿自己的自然语言评价直接改生产投放。正确的结构是三分：**Agent 提候选，实验和因果评估判优，策略引擎决定能不能执行**。理由很直接——广告涉及真实资金、延迟转化、用户干扰和合规，而 LLM 的"我觉得这个策略更好"不是一个可以对账的证据。

第一步定目标和硬约束。主目标可能是增量转化、利润或长期价值；CPA、ROAS 只能在统一归因窗口和成本口径下比。护栏包括预算上限、频控、品牌安全、用户体验、公平性、素材合规。多目标要做约束优化，不能把所有指标加权成一个 LLM 打的总分——权重一旦由模型自己解释，就没人能复核了。

第二步建可审计的数据和实验层。记录广告、受众、素材、出价、曝光、点击、转化、成本、时间和策略版本。候选策略先过规则校验，再离线回放，再小流量随机对照。核心是看增量而不是相关：历史上高转化的人群本来就会转化，把它算成策略功劳是最常见的自欺。延迟转化、样本比例失衡、跨广告干扰、归因污染都要进实验设计。

第三步才是在线分配。稳定期用 A/B；需要在探索和利用之间动态分配时用 UCB、Thompson Sampling 或带约束的 Contextual Bandit，但必须有最小样本量、置信/后验判定标准、单步最大预算变动、以及不可探索人群（比如已经明确表示不感兴趣的用户）。离线 RL 只有在行为策略日志完整、覆盖度够、反事实评估可信的前提下才值得上。

第四步是执行控制面。Agent 只输出结构化策略提案，Policy Engine 检查预算、频控、权限和实验状态，高风险变更走人工审批，上线用小流量 canary，实时盯 spend / CPA / ROAS / 投诉 / 分布漂移，kill switch 触发就回滚到已验证策略。Memory 只沉淀带实验 ID、样本量、置信度和适用范围的经验——不能把 Critic 的主观结论直接写成"最佳策略"，那等于把一次噪声固化成长期偏见。

**项目证据**

在线实验平台、bandit、因果推断、Policy Engine、canary、kill switch：项目中没有，不能硬套。这是纯离线的 RAG 系统，没有 A/B 框架、没有流量分配、没有真实资金动作，`grep -rn "bandit\|thompson\|ab_test"` 为空。

不过"离线回放比较两组参数"这一层有雏形，可以拿来说明我理解这套东西该长什么样。`tests/eval/run_eval.py` 支持 `--compare` 跑两组配置对比，`--vector-weight` / `--bm25-weight` 直接调混合检索的权重，`--tag` 给每次运行打标签，`--filter` / `--only-names` 缩小样本范围。指标层 `Hit@K`、`Recall@K`、`MRR` 已实现，`faithfulness_placeholder` 和 `answer_relevance_placeholder` 还是 `NotImplementedError`。样本 83 条，这个规模只够看趋势不够下结论——真要做参数决策，这个置信度我自己不敢用。

差距要说清：我这套是**离线固定集回放**，没有随机对照、没有流量分配、没有增量效果估计。它能回答"换个权重在这 83 条上召回是涨还是跌"，回答不了"这个改动能带来多少增量转化"。这两件事的区别正是这道题的核心。

- `tests/eval/run_eval.py:459` — `--compare`，两组配置对比
- `tests/eval/run_eval.py:461-462` — `--vector-weight` / `--bm25-weight`，参数可调
- `tests/eval/run_eval.py:449,464,467` — `--tag` / `--filter` / `--only-names`
- `tests/eval/metrics.py` — `Hit@K` / `Recall@K` / `MRR` 已实现
- `tests/eval/metrics.py` — `faithfulness_placeholder`、`answer_relevance_placeholder` 仍是 `NotImplementedError`
- `tests/eval/golden_set_expanded.jsonl` — 83 条，规模不足以支撑参数决策
- `app/services/vector_search_service.py:42-43` — `VECTOR_WEIGHT = 0.7` / `BM25_WEIGHT = 0.3`，被评测脚本对比的那两个参数
- 在线 A/B、bandit、增量效果估计、Policy Engine、canary、kill switch、归因窗口：项目中没有，不能硬套

#### Q29 · 反问：什么时候二面，有什么可以提高的地方。

**回答**

可以问，但要把流程问题和能力反馈分开，措辞也再职业一点。

流程可以这样问："想了解一下后续大致有几轮、预计推进节奏如何，我也方便安排时间。"别把"什么时候二面"问成默认自己已经过了——语气上的这点差别，面试官听得出来。

能力反馈可以问："结合今天的交流，您觉得我在 Agent 工程、后端基础还是项目表达上，哪一部分最需要继续补强？"如果对方给了具体建议，可以追问团队实际工作中这项能力怎么用，但不要当场争辩。

再补一个业务问题会更有价值："这个岗位当前最核心的 Agent 场景和成功标准是什么？团队更看任务成功率、业务增量，还是研发效率？"这个问题帮我判断岗位，也顺带说明我关心的是能不能交付而不只是面试技巧。

反问控制在两三个，按对方回答顺着追，别把准备好的清单念完。

**项目证据**
- 这题问的是面试沟通，不涉及仓库代码，无需引证据。上面提到的三类能力（Agent 工程 / 后端基础 / 项目表达）在本文件前 156 题里都有对应的真实代码位置，反问时如果对方指出短板，我能立刻定位到具体文件讨论

---

## 六、拼多多 AI Agent 岗两轮技术面

来源：https://www.nowcoder.com/discuss/918542384726540288

### 面经 01 · 一面 · 项目深挖、RAG 与工程基础

#### Q1 · 自我介绍，重点讲一下实习经历和 AI Agent 相关项目

**回答**

自我介绍要为后面的追问铺路，不是把简历念一遍。我会用 60 到 90 秒讲四件事：现在的阶段和主技术栈、最相关的一个项目、这个项目里我负责到哪、为什么这些经验对 Agent 岗有用。

具体到这个仓库，我会说它是一个企业知识库 + AIOps 诊断的 Agent 服务，我负责的是可靠性和检索这两条线。可靠性线上做了四件事：把 LLM 构造统一到一个工厂，解决三处构造点全都漏 timeout 的问题；把单次调用超时和整请求总预算分成两层，超预算不抛异常而是交付部分结果；给检索和 LLM 各挂一个熔断器；把 HTTP 状态码从"一律 200 + body 里写 code:500"改成按异常类型裁决。检索线上做了混合检索、子查询并行 fan-out、证据校验和幻觉拦截。

结尾我会落到岗位匹配上：我对 Agent 的理解不是在模型外面套 Prompt，而是把状态、预算、失败语义、可观测性这些后端工程的东西补到模型调用链上。这个仓库里能拿出来的每一处改动都有对应的问题记录和代码位置，追问到哪一层我都能定位。

**项目证据**
- `app/core/llm_factory.py:1-60` — 模块 docstring 记录了三处构造点漏 timeout 的原始问题
- `app/config.py:70,81` — `llm_timeout_seconds = 60.0` 与 `chat_total_budget_seconds = 90.0` 两层预算
- `app/agent/rag_v2/service.py:50-75` — 超预算交付部分结果而非抛异常
- `app/core/circuit_breaker.py:80-268` — 熔断器实现
- `app/api/chat_v2.py` 失败路径 — 裸 `raise` 交给全局处理器裁决状态码
- `app/agent/rag_v2/graph.py:38-52` — 五节点编排与 `Send` fan-out
- 实习经历：不在本仓库内，无法引行号

#### Q2 · 介绍会议转写项目的完整流程，你在其中主要负责哪些模块？

**回答**

项目中没有，不能硬套。这个仓库里没有任何会议转写相关代码：没有音视频接入、没有 ASR 调用、没有说话人分离、没有会议纪要生成。`app/services/` 下十六个服务全是文本侧的（切分、清洗、向量化、检索、摘要、PDF 入库），`mcp_servers/` 下两个 MCP server 是日志查询和监控指标查询。

我也不能拿 PDF 协议入库那条链去冒充它。表面上两者都是"非结构化输入进来、结构化产物出去"，但会议转写的难点集中在流式音频分片、实时性、说话人归属和 ASR 错字纠正上，PDF 入库一个都不涉及——它处理的是已经落盘的静态文件，走的是异步队列而不是实时流。硬把它说成会议转写，面试官往下问一句"音频分片边界怎么切"就露了。

如果面试官问的是"你有没有做过类似形态的链路"，我会明确改口径回答：做过一条七态状态机的异步入库链路，PDF 上传后先算 sha256 去重，然后 extracting → structuring → validating → awaiting_confirmation，人工确认后才 writing → completed，任何一步失败落 failed。这条链路的可迁移经验是状态外置、幂等键、人工确认闸门，但它不是会议转写。

**项目证据**
- 会议转写 / ASR / 音视频接入 / 说话人分离 / 会议纪要：项目中没有，不能硬套
- `app/models/protocol_ingestion.py:19-28` — 形态相近但业务不同的九态入库状态机，只能作为"类似链路"提及
- `app/api/protocol_pdf.py:68` — `content_hash=hashlib.sha256(content).hexdigest()` 去重
- `app/models/protocol_ingestion.py:24` — `AWAITING_CONFIRMATION`，人工确认闸门
- `mcp_servers/cls_server.py`、`mcp_servers/monitor_server.py` — 现有全部外部工具，均为文本/指标查询

#### Q3 · 音视频流进入系统后，经过哪些步骤生成转写文本和会议纪要？

**回答**

项目中没有，不能硬套。仓库里没有流式音视频处理管线。

我可以讲清楚这类系统的通常分层，但要标明是通用认知而不是我做过的：接入层收流并做分片（通常按静音检测或固定窗口加重叠切）；ASR 层逐片识别，输出带时间戳的 token 序列；后处理层做标点恢复、数字规整、专有名词替换和说话人归属；聚合层把片段拼成带说话人和时间轴的完整转写；纪要层再在转写文本上做摘要、待办抽取和决议提取。

分片重叠这一点跟仓库里的文档切分有形式上的共性——都要靠重叠避免切在语义中间。仓库里 `chunk_overlap = 100`，理由和音频分片重叠是一样的：切点落在句子中间会让两侧都失去可理解性。但这只是同一个思路在不同介质上的应用，不能因此说我做过音频链路。

**项目证据**
- 音视频接入 / 分片 / ASR / 标点恢复 / 说话人归属 / 纪要生成：项目中没有，不能硬套
- `app/config.py:140` — `chunk_overlap = 100`，重叠避免切断语义，与音频分片重叠同思路不同介质
- `app/services/document_splitter_service.py:32-38` — 二次分割器的重叠配置

#### Q4 · 为什么选择阿里云 ASR？对比过其他云服务或开源语音识别模型吗？

**回答**

项目中没有，不能硬套。仓库里没有接任何 ASR 服务。

不过"为什么选这家云服务、对比过什么"这个问法，我在这个仓库里有真实的对应决策可以讲：模型侧选的是阿里云 DashScope，`qwen-max` 做生成、`text-embedding-v4` 做向量化，走的是 OpenAI 兼容模式的 endpoint。选它的原因很实际——兼容模式让我可以直接用 `langchain_openai` 的 `ChatOpenAI`，不必为一家厂商写适配层；国内访问延迟稳定；Embedding 和 LLM 同一家，鉴权和配额只管一处。

代价我也清楚，而且踩到过一个具体的坑：`langchain-openai` 里 `request_timeout` 默认 `None`，会无条件覆盖 OpenAI SDK 自带的 `DEFAULT_TIMEOUT`，也就是说不显式传 timeout 的话，读超时会变成 600 秒。这个坑跟选哪家云无关，是兼容层本身的行为，我把它写进了 `llm_factory.py` 的 docstring 里，并且在工厂里统一兜住。

真实的短板是：我没有做过多家云服务的横向对比实验。DashScope 是项目起步时定的，没有留下 A/B 记录。如果面试官追问对比数据，我只能说没有，不能编。

**项目证据**
- `app/config.py:45-47` — DashScope endpoint 与两个模型名
- `app/config.py:46,129` — `qwen-max` 是唯一的 LLM
- `app/services/vector_embedding_service.py:27-28,173-174` — `text-embedding-v4`，1024 维
- `app/core/llm_factory.py` docstring — `request_timeout` 覆盖 SDK `DEFAULT_TIMEOUT` 的兼容层坑
- `app/services/vector_embedding_service.py:22,44-46` — `MAX_BATCH_SIZE = 10`，DashScope 服务端限制
- ASR 选型 / 多云横向对比实验：项目中没有，不能硬套

#### Q5 · ASR 的识别效果怎么评估？遇到专有名词、同音字和错别字时怎么处理？

**回答**

项目中没有，不能硬套。没有 ASR 就没有 WER/CER 评估，也没有热词表和纠错词典。

能真答的是评估方法论这一层，因为仓库里有一套检索评测框架，思路可以对照。ASR 评估的核心是 WER 和 CER，本质是编辑距离归一化；专有名词处理通常靠热词表加权、自定义语言模型和后处理替换；同音字要靠上下文语言模型或领域词典消歧。

仓库这边的对应物是 `tests/eval/`：83 条 golden set，每条带 `question`、`expected_doc_ids`、`reference_answer` 和 `tags`，指标实现了 Hit@K、Recall@K、MRR。`run_eval.py` 支持 `--compare` 做两组配置对比、`--vector-weight`/`--bm25-weight` 调混合权重、`--filter` 按标签筛。这套东西回答"怎么衡量效果、怎么做对比实验"是够的。

短板要说清两处：一是 faithfulness 和 answer relevance 两个指标还是 `NotImplementedError` 占位，也就是生成质量没有自动化评估；二是 83 条样本规模偏小，跑出来的差异未必显著。这两条我不打算美化。

**项目证据**
- ASR 评估 / WER / CER / 热词表 / 同音字纠错：项目中没有，不能硬套
- `tests/eval/golden_set_expanded.jsonl` — 83 条评测样本，含 `expected_doc_ids` 与 `tags`
- `tests/eval/metrics.py` — Hit@K / Recall@K / MRR 已实现
- `tests/eval/metrics.py` — `faithfulness_placeholder`、`answer_relevance_placeholder` 仍为 `NotImplementedError`
- `tests/eval/run_eval.py:449,455,456,459,461-462,464,467` — `--tag`/`--top-k`/`--retriever`/`--compare`/权重/`--filter`/`--only-names`

#### Q6 · 视频通话、语音和转写文本涉及用户隐私，你们如何保证传输、存储和访问安全？

**回答**

音视频那部分项目中没有，不能硬套。但访问安全这个问题我必须如实说仓库的现状：**没有任何鉴权层**。我 grep 过 `Security`、`OAuth2`、`HTTPBearer`、`verify_token`、`current_user`、`Authorization`，`app/` 下一处都没有。所有接口——上传文件、触发索引、聊天、清空会话、查诊断任务——都是裸接口，任何能访问到端口的人都能调。这是个真实的安全缺口，不是"还没做的优化项"。

现在只有三件跟安全沾边的事是真的做了。一是密钥不落日志：`vector_embedding_service.py` 里有 `_mask_api_key`，只打印前 8 后 4 位，其余用省略号；API Key 本身从环境变量读，`config.py:44` 的默认值是空串而不是硬编码。二是 CORS 有白名单，`cors_origins` 从配置读并做逗号分隔解析，不是 `allow_origins=["*"]`。三是 span 落盘时对证据正文做了瘦身，`used_documents` 只存 200 字摘要而不是全文，`EXCERPT_MAX_CHARS = 200`，并且用 `excerpt_only: True` 如实标注这是残缺数据——这条虽然初衷是省存储，但客观上减少了正文在 trace 表里的暴露面。

如果按题目要求设计，我会补三层：传输层强制 TLS 并对音视频用一次性签名 URL；存储层做静态加密和按租户分桶，转写文本按保留期自动删除；访问层做 OIDC 鉴权加 RBAC，检索时把权限过滤下推到向量库的 `expr` 里而不是召回后再过滤——后者会让越权内容先进到进程内存。这三层现在一层都没有，我不会说成已实现。

**项目证据**
- 鉴权 / 授权 / RBAC / TLS 强制 / 静态加密 / 数据保留期：项目中没有，不能硬套。`app/` 下无任何 `Security`/`OAuth2`/`HTTPBearer`/`verify_token`/`Authorization` 相关代码
- `app/services/vector_embedding_service.py:58-70` — `_mask_api_key`，密钥日志掩码
- `app/config.py:44` — API Key 默认空串，从环境变量加载，不硬编码
- `app/main.py:58-69` — CORS 白名单来自配置，非 `*`
- `app/config.py:31-38` — `cors_origins` 及其 `field_validator` 解析
- `app/core/used_documents.py:42,112-113` — 证据正文 200 字瘦身 + `excerpt_only` 标注
- `app/services/vector_search_service.py:52-107` — 检索无权限过滤参数，权限下推未实现

#### Q7 · 系统采用中心化还是去中心化架构？当时为什么这样设计？

**回答**

这个仓库是中心化的，而且是刻意的。一个 FastAPI 进程承担全部编排，RQ worker 是另一组进程但只做重活（索引、PDF 入库），不参与决策；两者之间唯一的通道是 Redis 里的那条 job 记录，`job_context.py:9` 的注释就是这么写的。没有服务发现、没有节点间协商、没有分布式共识。

选中心化的理由有三个具体的。第一，编排本身是有序的五步：rewrite → 并行 retrieve → dedup → generate → validate，`graph.py:44-51` 就是这五条边。这种拓扑没有需要协商的地方，做成去中心化只会把 super-step 的边界拆散，反而丢掉"一处 fan-in 能看到全部分支结果"这个性质——`dedup_node` 判定降级原因恰恰依赖它，单个分支只知道自己失败了，判断不了是全挂还是只有自己挂。第二，熔断器用的是进程内 `threading.Lock` 加内存计数，`circuit_breaker.py:31` 明确写了 Redis 共享状态这类特性"本项目一个都用不上"——单进程下不需要跨节点同步熔断状态。第三，总预算 `asyncio.timeout` 只在单进程内有意义，跨节点要做分布式 deadline 传播，复杂度翻几倍。

代价我清楚：API 进程是单点，横向扩容后熔断器状态各算各的，会出现一半实例已熔断、另一半还在打下游的情况。这个问题目前没解，规模没到。

**项目证据**
- `app/core/job_context.py:9` — "两者之间唯一的通道是 Redis 里的那条 job 记录"
- `app/core/task_queue.py:9-11` — 单 Redis 连接，两个队列，无服务发现
- `app/agent/rag_v2/graph.py:44-51` — 五节点线性拓扑 + 一处 fan-out
- `app/agent/rag_v2/nodes.py:236-240` — 降级判定放 fan-in 之后，依赖中心化的全局视图
- `app/core/circuit_breaker.py:31` — Redis 共享状态等特性"本项目一个都用不上"
- `app/core/circuit_breaker.py:84` — `threading.Lock` 跨两种执行模型
- `app/agent/rag_v2/service.py:56` — `asyncio.timeout` 进程内总预算
- 服务发现 / 节点协商 / 分布式共识 / 跨实例熔断状态同步：项目中没有，不能硬套

#### Q8 · 介绍一下项目中的 RAG 流程，离线建库和在线检索分别做了什么？

**回答**

离线建库分五步，都在仓库里。文件上传后先算 sha256 存进 `content_hash` 做去重；然后 `text_cleaner` 做换行和空白归一；接着切分——`MarkdownHeaderTextSplitter` 只按 h1/h2 切且 `strip_headers=False` 保留标题在正文里，超长的再用 `RecursiveCharacterTextSplitter` 二次切，过小的用 `_merge_small_chunks` 合并；然后 `DashScopeEmbeddings` 批量向量化，因为 DashScope 单次上限 10 条，内部按 `batch_size` 分批；最后写 Milvus，HNSW 索引 + COSINE 度量，1024 维。整条链走 RQ 异步队列，状态落 PostgreSQL。

在线检索是五节点 LangGraph。`rewrite_node` 把原问题拆成 3 条子查询；`_fanout_to_retrieve` 用 `Send` 把 3 条并行 fan-out 到 `retrieve_each_node` 的 3 个实例，每条各取 top 4，每个实例内部先过熔断器再 `asyncio.to_thread` 调同步检索；`dedup_node` 在 fan-in 后按 `metadata["id"]` 去重、截到 6 条、并判定降级原因；`generate_node` 生成；`validate_answer_node` 做证据校验，`groundedness` 低于 0.75 或 `coverage` 低于 0.60 就拦。

单条检索内部是混合的：Milvus 向量检索加 BM25 关键词检索，用 `EnsembleRetriever` 做 RRF 融合，权重 0.7 / 0.3，候选数 `max(top_k*3, top_k, 10)`。BM25 语料为空时降级为纯向量，会打一行 warning。

**项目证据**
- `app/api/protocol_pdf.py:68` — sha256 去重
- `app/services/text_cleaner.py:57-90` — 换行/空白归一
- `app/services/document_splitter_service.py:20-42` — 两级切分器配置，`strip_headers=False`
- `app/services/document_splitter_service.py:135-160` — `_merge_small_chunks`
- `app/services/vector_embedding_service.py:22,76,90-91` — 单次上限 10，内部分批
- `app/core/milvus_client.py:49,198-199` — 1024 维，COSINE + HNSW
- `app/services/index_job_queue.py:11-28` — RQ 异步入队，30m 超时，7 天 TTL
- `app/agent/rag_v2/nodes.py:28-33` — `NUM_SUB_QUERIES=3`、`RETRIEVE_TOP_K=4`、`FINAL_TOP_K=6`、两个校验阈值
- `app/agent/rag_v2/graph.py:19-22,44-51` — `Send` fan-out 与五节点边
- `app/agent/rag_v2/nodes.py:175-188` — 熔断器包在 `to_thread` 外侧
- `app/services/vector_search_service.py:42-43,84,92-99` — 权重、候选数、RRF 融合、BM25 空降级

#### Q9 · 文档如何切块？Chunk 大小和重叠窗口是根据什么确定的？

**回答**

切块是两级的。第一级按 Markdown 标题切，只切 h1 和 h2，注释里写明"不再按三级标题分割，避免过度碎片化"，并且 `strip_headers=False` 把标题留在正文里——这一点很关键，标题往往是这个块唯一的主题标识，剥掉之后向量化出来的东西容易失去指向。第二级对超长块用 `RecursiveCharacterTextSplitter` 兜，注意它的 `chunk_size` 是 `self.chunk_size * 2` 也就是 1600 而不是 800，注释说明是"加倍 chunk_size，减少分片数"。反方向还有一个 `_merge_small_chunks`，把过小的块并进相邻块。

参数是 `chunk_max_size = 800`、`chunk_overlap = 100`，配比 12.5%。

现在说实话的部分：这两个数字**不是实验调出来的**。它们是按知识库文档形态定的经验值——`aiops-docs/` 下那六篇运维文档是标准的二级标题结构，一节正文大致就在几百字量级，800 刚好装得下一节而不跨节；100 的重叠是为了让切点落在句子中间时两侧都还能读懂。仓库里没有留下 800 与 512、1024 的对比记录。

**项目证据**
- `app/services/document_splitter_service.py:20-29` — 只按 h1/h2 切，`strip_headers=False`
- `app/services/document_splitter_service.py:32-38` — 二次分割器 `chunk_size * 2`
- `app/services/document_splitter_service.py:135-160` — `_merge_small_chunks` 反向合并
- `app/config.py:139-140` — `chunk_max_size = 800`、`chunk_overlap = 100`
- `aiops-docs/cpu_high_usage.md` 等六篇 — 参数所依据的实际文档形态
- 切块参数的对比实验记录：项目中没有，不能硬套。800/100 是经验值，无 A/B 数据

#### Q10 · 为什么使用当前的切块策略？是否通过 Recall@K、MRR 等指标做过实验？

**回答**

这题得拆成两半答，因为两半的答案不一样。

**指标框架是真的。** `tests/eval/metrics.py` 是我自己写的，开头就注明"仅实现零依赖的检索类指标"——刻意不引 RAGAS 那套，因为 RAGAS 的 faithfulness 要额外调模型打分，跑一轮评测的成本和不确定性都上去了，而 Hit@K / Recall@K / MRR 这三个纯粹是集合运算，不依赖任何外部服务，任何时候跑结果都一样。

具体实现：`_first_hit_rank` 找第一个命中的排名，`hit_at_k` 判断前 K 里有没有命中，`recall_at_k` 算前 K 覆盖了多少期望文档，`reciprocal_rank` 取首命中排名的倒数。`compute_case` 默认在 `ks=(1, 3, 5, 10)` 四个档位上同时算，`aggregate` 做跨用例聚合。评测集是 `tests/eval/golden_set_expanded.jsonl`，83 条样本，`run_eval.py` 是入口。

**但切块策略本身没有跑过对比实验。** 这是要如实说的。评测框架建起来之后，我用它验的是检索链路的效果——混合检索的权重配比、`candidate_k` 取多大、子查询数量的影响。切块参数 800/100 是在评测框架之前就定下来的经验值，后面也没有回头拿 512 / 1024 各建一次库再跑一遍 Recall@K 对比。真要做这个实验，成本在于每换一次切块参数就要整库重建、重新 embedding，而不是评测本身。

还有一处要坦白：`metrics.py:174` 和 `:183` 有两个函数叫 `faithfulness_placeholder` 和 `answer_relevance_placeholder`，名字里的 placeholder 就是字面意思——占位，还没实现。生成质量这一侧我最终是用运行时的 `validate_answer_node` 兜的（覆盖度 + 忠实度双阈值），而不是离线评测。

所以如果面试官问"你的切块策略有数据支撑吗"，诚实的回答是：指标能力有，这条实验没做。

**项目证据**
- `tests/eval/metrics.py:3` — "仅实现零依赖的检索类指标(Hit@K / MRR / Recall@K)"
- `tests/eval/metrics.py:66,75,81,90` — `_first_hit_rank` / `hit_at_k` / `recall_at_k` / `reciprocal_rank`
- `tests/eval/metrics.py:96,120,128` — `compute_case`（默认 ks=(1,3,5,10)）、`aggregate` / `_aggregate`
- `tests/eval/golden_set_expanded.jsonl` — 83 条评测样本
- `tests/eval/run_eval.py` — 评测入口
- `tests/eval/metrics.py:174,183` — `faithfulness_placeholder` / `answer_relevance_placeholder`，未实现
- `app/agent/rag_v2/nodes.py:32-33` — 生成质量改用运行时双阈值校验
- 切块参数的 Recall@K / MRR 对比实验：项目中没有，不能硬套。框架有，这条实验没跑

#### Q11 · 短期记忆和长期记忆分别怎么设计？历史对话在什么情况下会被摘要或召回？

**回答**

先划清边界：项目里做的是**会话内的分层记忆**，短期是最近消息原文，"长期"是同一会话的滚动摘要。跨会话的用户画像、长期偏好库、记忆的向量化召回——这些项目中没有，不能硬套。

`ConversationMemoryService` 的核心设计是一句话：完整原文只进 PostgreSQL 当审计记录，喂给模型的上下文永远只有"压缩摘要 + 未压缩消息的尾部"，目的是让输入 Token 不随对话轮数无限增长。

**Token 估算先说，因为它是整个预算体系的地基。** `estimate_tokens` 用的是 CJK 加权：中日韩字符按 1 个 Token 算，其余字符按 `ceil(n/4)` 算。这个规则对中文是刻意保守的——真实分词里一个汉字常常不到 1 个 Token，但宁可高估，因为低估的后果是超上下文直接报错，高估的后果只是少注入一点历史。每条消息再额外 `+4`（`_message_token_cost`），算的是 role 前缀和分隔符的开销。

**四个预算参数**（`config.py:133-136`）：
- `context_token_budget = 1800` — 注入总上限
- `summary_token_budget = 700` — 摘要单独的上限
- `compact_threshold_tokens = 2400` — 触发压缩的阈值
- `recent_turns = 3` — 保留最近 3 轮用户消息

**摘要什么时候触发。** `build_context` 的流程：先取活跃消息，再取记忆快照 `get_memory_snapshot`，用 `_messages_after_snapshot` 按游标切出"还没被摘要过"的部分，然后 `_recent_message_count` 从尾部倒数 3 个用户轮次留作原文，剩下的中间段就是压缩候选。**只有当候选段的 Token 数 ≥ 2400 才触发摘要**——不是按轮数，是按 Token 量，因为三句短对话和三段长贴代码的成本差一个数量级。

摘要成功后写快照，`summarized_through_message_id` 记住压缩到哪一条，下次直接从游标之后接着算。这是增量的，不会每轮都把全部历史重摘一遍。

**两个我觉得值得讲的细节：**

第一，`_messages_after_snapshot:143-144` 有个反直觉的分支：如果游标指向的那条消息已经被软删除、在活跃列表里找不到了，函数返回**空列表**而不是整个列表。注释写的是"保守地不注入旧原文，避免重复历史"。理由是：摘要已经覆盖了游标之前的内容，此时如果退回去注入全量原文，模型会同时看到摘要版和原文版同一段历史，等于自己制造矛盾输入。宁可这一轮少点上下文。

第二，`_summarize` 的 except 分支只 `logger.warning` 然后 `return ""`，注释写明"摘要是节省 Token 的优化，不可阻断主聊天；失败时退回最近消息窗口"。摘要模型挂了不影响聊天能进行——退化后就是纯滑动窗口，效果变差但功能在。这和 `validate_answer_node` 的 fail-open 是同一条原则：优化性组件失败不能升级成主链路故障。

**渲染阶段的预算控制**是三层收口：`_render_context` 先给摘要留 700 且要扣掉标题本身的 Token，再把剩余额度给最近消息，最后对拼好的整段文本 `_truncate_text` 兜一次硬上限。注释解释了为什么要兜第三次——"角色前缀、分隔符等是实际 Prompt 的一部分"，只算消息内容会漏掉这些。`_take_tail_to_budget` 里还有个边界：如果单条消息本身就超预算，不是丢掉它，而是裁剪内容后仍然保留，因为最后一条用户消息通常就是当前问题。

**至于"召回"**——问题里说的向量召回历史对话，项目中没有。历史进上下文的路径只有两条：摘要，或者最近窗口原文。没有把历史消息也 embedding 进 Milvus 再按相关性捞回来的机制。

**项目证据**
- `app/services/conversation_memory_service.py:1-5` — 原文只进 PG 审计，上下文只用摘要 + 尾部
- `app/services/conversation_memory_service.py:27,48-54` — `_CJK_RE` 与 CJK 加权 `estimate_tokens`
- `app/services/conversation_memory_service.py:57-58` — 每条消息 `+4` 的结构开销
- `app/services/conversation_memory_service.py:93-131` — `build_context` 主流程
- `app/services/conversation_memory_service.py:112` — 按 Token 量而非轮数触发压缩
- `app/services/conversation_memory_service.py:118-126` — 写快照 + 游标推进（增量摘要）
- `app/services/conversation_memory_service.py:133-144` — 游标失效时返回空列表，避免摘要与原文重复
- `app/services/conversation_memory_service.py:146-158` — `_recent_message_count` 按用户轮次倒数
- `app/services/conversation_memory_service.py:183-186` — 摘要失败降级为最近窗口，不抛
- `app/services/conversation_memory_service.py:207-247` — `_render_context` 三层预算收口
- `app/services/conversation_memory_service.py:249-272` — 单条超预算时裁剪保留而非丢弃
- `app/services/conversation_memory_service.py:28-35` — 摘要 Prompt 的四个固定标题
- `app/config.py:133-136` — 1800 / 700 / 2400 / 3 四个预算参数
- 跨会话长期记忆、用户画像库、历史对话的向量化召回：项目中没有，不能硬套

#### Q12 · Agent 的工具调用成功率是怎么定义的？项目最初的数据是多少，后来如何提升？

**回答**

**这题我没有数据可以给，项目中没有工具调用成功率这个指标，不能硬套。** 先把这句放在最前面，因为这题最容易顺口编一个"从 70% 提到 95%"。

我把指标面翻了一遍，`app/core/metrics.py` 里注册的一共四个：

- `llm_call_duration_seconds` — Histogram，按 `node` 分标签的模型调用耗时
- `degrade_total` — Counter，按 `reason` 分标签的降级次数
- `retrieval_failures_total` — Counter，按 `code` 分标签的检索失败次数
- `circuit_breaker_state` — Gauge，按 `name` 分标签的熔断器状态

没有 `tool_call_total`，没有按工具名分标签的成功/失败计数。`tool_call` 这个词在代码里只出现在两处，都不是指标：一是 `app/api/chat.py:94,98` 的 SSE 事件类型，把工具调用事件推给前端；二是 `app/services/rag_agent_service.py:224-225` 从 `last_message.tool_calls` 取名字写日志。日志不是指标——它能事后 grep，但没法算比率、出不了看板、也不能配告警。

如果面试官接着问"那你怎么会知道工具在出问题"，能诚实答的是间接信号：主链路唯一的检索路径失败会进 `retrieval_failures_total`（按 error code 分），降级会进 `degrade_total`（按 `DegradeReason` 分），熔断器打开会把 `circuit_breaker_state` 的 Gauge 拉到对应值。这三个能覆盖"检索这件事成功没成功"，但它们的口径是 RAG 节点，不是工具调用层——`retrieve_knowledge` 这个 `@tool` 走的是 `rag_agent_service` 那条 `create_agent` 路径，它的成败并不进上面任何一个 Counter。

补一句我确实想过的设计，但要标成假设：**[假设]** 如果要加，我会定义成 `tool_call_total{tool, outcome}`，outcome 分 `success` / `error` / `timeout` / `empty` 四态而不是二态——因为 `retrieve_knowledge` 返回"没有找到相关信息。"和抛异常是完全不同的两件事，前者是正常的空结果，混进失败率里会把指标做脏。这个思路和 `RetrievalStatus` 的三态设计是一致的，只是没落到 Prometheus 上。

**项目证据**
- `app/core/metrics.py:85,94,100,110` — 全部四个指标，无工具维度
- `app/core/metrics.py:56-61` — 为什么耗时用 Histogram 而非 Gauge/Summary 的取舍记录
- `app/core/metrics.py:108-113` — 熔断状态用 Gauge 而非 Counter（"Counter 只能增，表达不了又闭合了"）
- `app/api/chat.py:94,98` — `tool_call` 仅作为 SSE 事件类型
- `app/services/rag_agent_service.py:224-225` — 工具名仅写日志
- `app/agent/rag_v2/state.py` — `RetrievalStatus` 三态，是我想复用到工具指标上的口径
- 工具调用成功率指标、历史基线数据、提升前后的对比：项目中没有，不能硬套

#### Q13 · 工具调用超时、返回脏数据或重复执行时，系统如何重试和兜底？

**回答**

三种故障在项目里的处理机制不一样，分开讲。

**超时 —— 分层防御，不是单点。**

最外层是整请求总预算：`asyncio.timeout(config.chat_total_budget_seconds)`，90 秒。里层是单次模型调用的 `timeout`，60 秒。两层的意义不同：单次 timeout 防的是一次调用挂死，总预算防的是"每次都没超时但累加起来超了"——三个子查询 + 生成 + 校验，每个都 50 秒不超单点阈值，加起来早就把用户等跑了。

这里有个我踩过的坑值得讲：参数默认值不能写 `timeout or config.x`。因为 `0` 是 falsy，显式传 `timeout=0`（意思是"不等"）会被 `or` 静默替换成配置里的 60 秒，行为和调用方的意图完全相反。要写 `timeout if timeout is not None else config.x`。

MCP 那一侧是 `retry_interceptor`：最多 3 次，间隔 `delay * 2**attempt` 指数退避。

**脏数据 —— 靠"永不抛异常 + 三态返回"兜。**

`retrieve_knowledge` 这个工具的实现里，空结果返回 `("没有找到相关信息。", [])`，异常返回 `(f"检索知识时发生错误: {str(e)}", [])`，**两条路径都是 return，都不 raise**。原因是工具异常会直接打断 agent 的循环，而模型其实有能力处理"这次没查到"这个信息——把失败当数据交给它，比让整个请求崩掉更有用。

`retry_interceptor` 是同一个思路的更明显版本：重试全部失败后，它 `return CallToolResult(..., isError=True)` 而不是把最后那个异常抛出去。`isError=True` 让调用方知道出事了，但控制流没断。

RAG 主链路上对应的是 `RetrievalStatus` 三态（OK / EMPTY / FAILED）。这里必须三态不能两态：检索到 0 条和检索报错，对下游是不同的处理——前者应该老实说"知识库里没有"，后者应该带上降级原因让值班的人知道去查什么。合成两态会把这两件事的区别抹掉。

生成结果这一层还有 `validate_answer_node` 做覆盖度 + 忠实度双阈值校验，拦的是"检索没脏但模型编了"这种情况。

**重复执行 —— 幂等性，三处不同做法。**

入库侧用内容哈希：`content_hash` 是 sha256，同一份文档重复提交直接命中已有记录，不会建第二遍索引。这是最干净的幂等——键由内容本身决定，不依赖调用方传对 ID。

状态机侧：`ProtocolIngestionStatus` 九个状态，相同状态之间的转移当幂等处理，重放一次不会把流程推错。

数据裁剪侧：`slim_used_documents` 用 `_SLIM_MARKER` 标记已裁剪过的记录，重复调用不会二次裁剪。

RQ 的重试是 `Retry(max, interval)`，注意它是**无条件**的——只要 job 抛异常就重试，不管这个异常该不该重试。所以判定层单独做了处理：不可重试的错误直接 `job.retries_left = 0`，把额度清零。另外 retry 必须在**入队时**声明，因为 RQ 把重试额度存进 job 记录，worker 只能减不能补。

**缺的部分要说清楚：** 工具级的幂等键（同一个 tool call 带唯一 ID、结果缓存复用）项目中没有，不能硬套；分布式锁防同一任务被两个 worker 同时领取也没有。RQ 本身的队列语义保证了一个 job 只会被一个 worker 取走，但如果 worker 执行到一半进程被杀，job 会停在 RUNNING 且没有回收机制——这一点 `job_failure.py:30` 的注释自己承认了："失败状态没写进去，job 行永远停在 RUNNING"。

**项目证据**
- `app/config.py` — `chat_total_budget_seconds = 90.0` 总预算
- `app/api/chat_v2.py` — `asyncio.timeout` 包住整个请求
- `app/core/llm_factory.py` — 单次调用 60s，`timeout if timeout is not None else ...` 的正确写法
- `app/agent/mcp_client.py:18-73` — `retry_interceptor`，3 次指数退避后 `return CallToolResult(isError=True)`
- `app/tools/knowledge_tool.py:13-44` — 空结果与异常都 return，不 raise
- `app/agent/rag_v2/state.py` — `RetrievalStatus` 三态
- `app/agent/rag_v2/nodes.py:32-33` — `MIN_GROUNDEDNESS_SCORE` / `MIN_COVERAGE_SCORE` 双阈值
- `app/models/document.py` — `content_hash`（sha256）内容级幂等
- `app/models/protocol_ingestion.py:19-35` — 九状态机，同状态转移幂等
- `app/agent/rag_v2/nodes.py` — `slim_used_documents` 的 `_SLIM_MARKER` 幂等标记
- `app/core/job_failure.py:68-74,84,121` — `default_job_retry`、无条件重试的说明、`retries_left = 0`
- `app/services/index_job_queue.py:11-28` — retry 必须入队时声明的原因
- `app/core/job_failure.py:30` — 卡在 RUNNING 无回收的自认
- 工具级幂等键、结果缓存复用、分布式锁：项目中没有，不能硬套

#### Q14 · 进程、线程和协程有什么区别？它们分别适合处理什么任务？

**回答**

这三层在我这个项目里刚好各有一个真实落点，我按代码讲会更准确。

**进程 —— 独立地址空间，故障域边界。** 这个服务跑起来是两个进程：`uvicorn app.main:app` 起 API，`rq worker knowledge_index protocol_pdf_ingest` 起后台索引 worker，两条命令在 Makefile 里是分开的。它们不共享任何内存，靠 Redis 传 job。选进程的理由不是性能，是**隔离**：建索引的 job 声明了 `job_timeout="30m"`，读整份 PDF、切块、批量 embedding，是最有可能吃爆内存的一段。它单独一个进程，被 OOM killer 杀掉时 API 进程毫发无损，用户那边只是这一个索引任务失败，聊天接口照常。如果把索引塞进 API 进程里跑，一次 OOM 就是整个服务下线。

**线程 —— 共享地址空间，用来兜住阻塞调用。** `pymilvus` 是同步 SDK，一次 `search` 就是一次阻塞的网络往返。在 asyncio 里直接调它会把事件循环整个卡住——那一刻所有其他请求的协程都停转。所以检索走 `asyncio.to_thread`，把阻塞调用挪到线程池，事件循环继续跑别的。线程共享内存这件事在这里既是便利也是负担：便利在 contextvars 会被复制过去，`request_context.py:14` 的注释写了"`asyncio.to_thread` 派生的线程会复制当前 context，所以线程池里也读得到"，所以 request_id 和 span 收集器在线程里仍然有效；负担在共享状态得自己加锁，熔断器用的是 `threading.Lock` 而不是 `asyncio.Lock`，`circuit_breaker.py:84` 说明了原因——"调用点跨两种执行模型 —— 检索走 `asyncio.to_thread`（真线程）"，asyncio.Lock 在真线程里是不安全的。

**协程 —— 单线程内的协作式切换，用来编排。** LangGraph 的每个节点都是 `async def`，三路子查询的 fan-out 用 `Send` 发出去，框架并发调度这三个协程。协程的切换成本几乎为零：没有系统调用，没有内核态往返，就是一次函数栈的保存与恢复。代价是**它不抢占**——一个协程里写了 CPU 密集的循环，或者不小心调了同步阻塞函数，整个事件循环就停在那里，没有任何机制能把它踢走。这也正是为什么 Milvus 调用必须显式 `to_thread`，而不能指望协程自己让出。

适用场景就是从这三个特性倒推出来的：进程适合需要故障隔离或真正吃 CPU 的任务；线程适合被同步阻塞 API 困住、又不想改成异步的场景；协程适合大量 I/O 等待 + 需要密集编排的场景，比如这里的图调度。

**项目证据**
- `Makefile:326` — `uvicorn app.main:app` API 进程
- `Makefile:340` — `rq worker $(RQ_QUEUES)` 独立 worker 进程
- `app/services/index_job_queue.py:11-28` — `job_timeout="30m"`，索引任务的资源画像
- `app/agent/rag_v2/nodes.py:186` — `await asyncio.to_thread(...)` 把同步 Milvus 调用挪进线程
- `app/core/request_context.py:14` — contextvars 跨 `to_thread` 复制
- `app/core/span_context.py:20` — 同上，span 收集器在线程与 LangGraph spawn 任务里都可见
- `app/core/circuit_breaker.py:84` — 为什么用 `threading.Lock` 而不是 `asyncio.Lock`
- `app/agent/rag_v2/graph.py:19-22` — `Send` fan-out 三个协程
- `multiprocessing` / 进程池并行计算：项目中没有，不能硬套。进程边界只有"API 与 worker 分离"这一处，不是用来做并行加速的

#### Q15 · 已经有进程了，为什么还需要线程？进程创建和上下文切换为什么更重？

**回答**

因为这两件事的成本不在一个量级，而且共享数据的方式完全不同。

**创建成本。** fork 一个进程，内核要建新的 task_struct、复制页表、复制文件描述符表、复制信号处理设置。现代内核用 COW 让物理页先不复制，但页表本身必须复制，进程地址空间越大这一步越贵。创建线程只需要新的内核调度实体加一个栈，页表直接共享——同一个 mm_struct。数量级上通常是几百微秒对几微秒。

**上下文切换成本。** 线程之间切换，切的是寄存器、栈指针、程序计数器，页表不动。进程之间切换要换 CR3（页表基址寄存器），这会导致 TLB 大面积失效——即使有 PCID 缓解，切回来之后一段时间内的内存访问都要重新走页表翻译，这部分间接成本比切换本身的直接指令数更贵。CPU cache 也是同理，两个进程的工作集不同，切过去等于把对方的 cache 全冲掉。

**共享数据的方式。** 这是我这个项目里最直接的证据。熔断器的状态必须被三路并行检索共同看到、共同更新——一路失败要计进同一个失败计数，达到阈值要让另外两路也读到 OPEN。用线程，这个状态就是一个普通的 Python 对象加一把 `threading.Lock`，几行代码。如果换成进程，同样的语义要靠共享内存或者 Redis 实现，得处理序列化、失效、竞态。`circuit_breaker.py:31` 的注释正好写了反面：熔断器实现里明确说"（Redis 共享状态、事件监听器、排除异常列表）本项目一个都用不上"——因为它只需要在**同一个进程内**跨线程共享，进程内共享是免费的。

反过来说，线程的这个"免费"是有代价的，就是隔离性归零。一个线程踩坏内存或者 OOM，整个进程一起走。所以这个项目的分工是：需要隔离的（索引任务）用进程，需要共享状态又只是等 I/O 的（三路检索）用线程。两者不是替代关系，是按"要隔离还是要共享"选的。

**项目证据**
- `app/core/circuit_breaker.py:31` — Redis 共享状态"本项目一个都用不上"，因为只需进程内共享
- `app/core/circuit_breaker.py:84` — 跨 `to_thread` 用 `threading.Lock` 的原因
- `app/agent/rag_v2/nodes.py:175-188` — 三路检索共享同一个熔断器实例
- `app/core/request_context.py:14` — 线程继承 contextvars，进程做不到这件事
- `Makefile:326,340` — 需要隔离的那部分才用进程边界
- 进程创建耗时、TLB miss 率的实测数据：项目中没有，不能硬套。上面的量级判断来自通用原理，我没在这个仓库里做过 benchmark

#### Q16 · 并发和并行有什么区别？多个任务看起来同时运行时，CPU 是否真的在并行执行？

**回答**

并发是**任务的组织方式**——多个任务在时间上重叠推进；并行是**执行的物理事实**——同一时刻多个执行单元真的在算。并发不要求多核，并行要求。

我项目里三路子查询检索这段代码，正好能把这个区别说透，而且答案跟直觉不太一样。

代码是 `Send` fan-out 三个协程，每个协程内部 `await asyncio.to_thread(...)` 调 Milvus。那么这三路是并发还是并行？

分两段看。**等待网络返回的那一段是真并行**——三个线程各自阻塞在 socket 上，Milvus 服务端确实在同时处理三个查询，这一刻挂钟时间被三路共享，总耗时接近单路而不是三倍。**执行 Python 字节码的那一段不并行**——GIL 保证同一时刻只有一个线程在解释器里跑字节码。所以序列化请求、解析返回结果这些 CPU 工作，三个线程是轮着来的。

`to_thread` 在这里能起作用，完全是因为瓶颈在网络 I/O 而不在 CPU。阻塞的系统调用会释放 GIL，这才是三路能重叠的真正原因。如果我把三路检索换成三段纯计算，`to_thread` 一点用都没有，甚至更慢——GIL 争抢加上线程切换开销。

再往上一层，`uvicorn` 进程和 `rq worker` 进程是**真并行**，两个独立解释器，各自一把 GIL，多核上确实同时跑。所以这个项目里真正的并行只发生在进程之间。

所以回到题目问的"CPU 是否真的在并行执行"：在这个项目的检索路径上，答案是 CPU 并没有并行，并行的是等待。这不影响效果，因为要省的挂钟时间本来就花在等待上。

**项目证据**
- `app/agent/rag_v2/graph.py:19-22` — `Send` 分发三路，并发的组织形式
- `app/agent/rag_v2/nodes.py:186` — `asyncio.to_thread` 把阻塞 I/O 挪出事件循环，重叠发生在等待期
- `app/agent/rag_v2/nodes.py:28` — `NUM_SUB_QUERIES = 3`，并发度是写死的常量
- `Makefile:326,340` — 两个进程，项目里唯一的真并行
- `app/agent/rag_v2/nodes.py:175-186` — 熔断器在 `to_thread` 外侧，熔断时不占线程池 worker
- CPU 密集任务的并行化（进程池 / 多核计算）：项目中没有，不能硬套。整条链路都是 I/O 密集，没有需要吃多核的计算段

#### Q17 · 举一个适合使用进程隔离的场景，再举一个适合使用多线程的场景。

**回答**

两个都用项目里真实存在的。

**适合进程隔离：文档索引任务。** 这是 `rq worker` 进程干的活——读整份 PDF、切块、调 embedding、写 Milvus。选进程隔离有三个具体理由，都不是性能：

第一，资源画像不可控。入队时声明 `job_timeout="30m"`，说明这个任务本身预期就可能跑很久；一份大 PDF 全文读进内存再切块，峰值内存跟文档大小成正比，是这个服务里最容易 OOM 的一段。它单独一个进程，被 OOM killer 挑中时 API 进程不受影响。

第二，失败要可观测、可重放。job 状态写在数据库里，`failure_ttl` 和 `result_ttl` 都是 7 天，失败的 job 记录留着能查能重跑。进程崩了就是 job 失败，边界干净。

第三，部署上能独立伸缩。索引积压了就多起几个 worker 进程，API 的实例数不用动。

**适合多线程：Milvus 检索调用。** `pymilvus` 是同步 SDK，一次 search 是一次阻塞往返。放线程池里的理由很单纯——不能让它卡住事件循环，同时又要跟熔断器共享状态。这两个要求合起来正好只有线程能满足：共享内存让熔断器就是个普通对象加一把 `threading.Lock`，不用 Redis 不用序列化；挪出事件循环让其他请求的协程继续跑。

这里有个我自己踩过的细节值得说：熔断器要包在 `to_thread` **外面**，不能塞进线程里。`nodes.py:175` 的注释写了原因——熔断器已经打开的时候，本来是要立刻拒绝的，如果判断逻辑在线程内部，那就得先申请到一个线程池 worker、再进去发现"哦熔断了"、然后返回。故障期间请求量往往还更大，这等于让被拒绝的请求白占线程池名额，把线程池挤爆。判断放外面，熔断时连线程都不申请。

**项目证据**
- `Makefile:340` — worker 独立进程启动
- `app/services/index_job_queue.py:11-28` — `job_timeout="30m"`、`failure_ttl` / `result_ttl` 各 7 天
- `app/workers/index_worker.py` — worker 侧入口，job 状态落库
- `app/agent/rag_v2/nodes.py:186` — `to_thread` 包住同步 Milvus 调用
- `app/agent/rag_v2/nodes.py:175-186` — 熔断器在线程外侧，避免占用线程池 worker
- `app/core/circuit_breaker.py:84` — 跨执行模型必须用 `threading.Lock`
- 沙箱执行不可信代码的进程隔离（seccomp / 容器 / namespace）：项目中没有，不能硬套。这个项目不执行用户提供的代码，没有这个需求
- 进程池并行加速：项目中没有，不能硬套

#### Q18 · 手撕：实现 LRU Cache，要求 `get` 和 `put` 的时间复杂度都是 `O(1)`。

**回答**

哈希表 + 双向链表。哈希表负责 O(1) 定位节点，双向链表负责 O(1) 维护访问顺序。

```python
class Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity 必须为正整数")
        self.capacity = capacity
        self.table: dict[object, Node] = {}
        # 哨兵头尾节点：省掉所有 "是否为空表/是否首尾节点" 的分支判断
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node: Node) -> None:
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        node = self.table.get(key)
        if node is None:
            return -1
        self._remove(node)
        self._add_to_front(node)
        return node.value

    def put(self, key, value) -> None:
        node = self.table.get(key)
        if node is not None:
            node.value = value
            self._remove(node)
            self._add_to_front(node)
            return
        if len(self.table) >= self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.table[lru.key]      # 必须用节点里存的 key 反查，这就是节点要存 key 的原因
        node = Node(key, value)
        self.table[key] = node
        self._add_to_front(node)
```

几个容易被追问的点：

**为什么必须是双向链表？** 淘汰时要删尾节点，命中时要把中间某个节点摘出来。单链表删除节点需要前驱指针，只能从头遍历，退化成 O(n)。双向链表的 `node.prev` 让摘除是纯指针操作。

**节点为什么要存 key？** 淘汰时手里只有链表尾节点，但要从哈希表里删掉对应条目，必须知道它的 key。只存 value 的话这一步就得反向扫哈希表，O(n)。

**哨兵节点的作用。** 头尾各放一个不存数据的节点，链表永远非空，`_remove` 和 `_add_to_front` 里不需要任何 `if node is self.head` 之类的边界判断。这不是为了性能，是为了让代码没有分支、不容易写错。

**Python 里的偷懒写法**，面试可以提一句：`collections.OrderedDict` 的 `move_to_end` 和 `popitem(last=False)` 都是 O(1)，内部就是哈希表加双向链表，所以 `OrderedDict` 版本大概十行；`functools.lru_cache` 更是直接可用。但手撕题要的是链表操作本身。

**项目证据**
- LRU Cache 实现：项目中没有，不能硬套。仓库里没有 `functools.lru_cache`，也没有任何手写的淘汰式缓存
- `app/services/vector_search_service.py:140-163` — 项目里唯一的缓存是 BM25 检索器缓存，但它**不是 LRU**：按 `collection.num_entities` 当版本号判定失效，命中就直接返回，不命中就全量重建整个检索器。这里不做逐条淘汰是有原因的——语料一旦变化，BM25 的 IDF 是全局统计量，旧检索器里每一条的打分都失效了，淘汰单条没有意义
- 上面这段说明也是我在真实项目里对"什么时候该用 LRU"的判断：LRU 的前提是条目之间相互独立、可以单独淘汰。BM25 检索器不满足这个前提，所以用了版本号整体失效

### 二面 · Agent 设计、模型原理与系统能力

#### Q1 · 自我介绍，并选择一个最熟悉的 Agent 项目做深入介绍。

**回答**

自我介绍部分略过（同一面 Q1）。这里直接讲项目，我选 RAG v2 这条链路，因为它是我从零写到有可观测数据的那一条。

一句话定位：这是一个运维知识库问答服务，输入一个运维问题，输出带证据来源的处置建议，中间做了子查询扩展、混合检索、去重、生成、答案校验五步。

我想强调的不是功能，是**这个项目里有两条并存的实现路径**，而这恰好是我最想聊的部分：

第一条是 `app/services/rag_agent_service.py`，用 LangChain 的 `create_agent` 配 `MemorySaver`，把检索包成一个 tool 交给模型，让模型自己决定要不要调、调几次。这是标准的 Agent 循环写法，几十行就能跑起来。

第二条是 `app/agent/rag_v2/`，我把流程拆成显式的五节点 `StateGraph`：`rewrite` → `retrieve_each`（`Send` fan-out 成 3 个并行分支）→ `dedup` → `generate` → `validate_answer`，边是固定的，模型只在 `rewrite`、`generate`、`validate_answer` 三个位置被调用，检索路径本身不交给模型决策。

为什么两条都留着：第一条的问题是不可控——模型可能一次都不检索就直接答，也可能连着检索五次，我在日志里既看不出它为什么这么选，也没法给它设预算。第二条虽然写得多，但每一步的耗时、每一步的降级原因、每次用了哪几个文档块，都能落到 `chat_run_spans` 表里查。线上主链路走的是第二条。

这个取舍是这个项目对我来说最有价值的部分：**Agent 的自主性和工程的可观测性是有冲突的，我选了后者，并且知道自己放弃了什么。**

**项目证据**
- `app/services/rag_agent_service.py:9,16,107,133-136` — `create_agent` + `MemorySaver`，Agent 循环路径
- `app/agent/rag_v2/graph.py:22-51` — 五节点显式图，`_fanout_to_retrieve` 用 `Send` 扇出
- `app/agent/rag_v2/nodes.py:28-33` — `NUM_SUB_QUERIES=3`、`RETRIEVE_TOP_K=4`、`FINAL_TOP_K=6` 与两个校验阈值
- `app/models/chat_run_span.py:67-99` — 节点级 span 落库字段
- `app/agent/rag_v2/graph.py:26-33` — 统一在注册处包 `instrument_node` 的理由（漏不掉、节点不感知）

#### Q2 · 你怎么看当前 Agent 技术的发展？现阶段适合落地在哪些场景？

**回答**

我只讲我自己写代码撞出来的判断，不讲行业趋势。

**现在能落地的，是"有明确知识边界 + 有人工兜底"的场景。** 我这个项目就是典型：运维告警来了，去知识库里找 SOP，生成一份首响报告。它能落地的原因不是模型强，是**这件事允许出错**——报告是给值班同学看的参考，不是自动执行的命令。值班同学看一眼觉得不对就自己判断，代价接近零。

**现在不好落地的，是"要模型自己拍板做动作"的场景。** 我在项目里能感受到这个边界在哪：我的 `validate_answer` 节点做的就是让模型校验模型，检查生成的答案有没有超出检索到的证据。这一层做出来之后，`MIN_GROUNDEDNESS_SCORE` 和 `MIN_COVERAGE_SCORE` 两个阈值卡下去，确实拦住了一批无依据的判断。但它只能拦"说了没证据的话"，拦不住"证据本身选错了"。**校验器和被校验者是同一个模型，共享同一套错误倾向。** 这就是为什么我不敢把这条链路接到任何真实执行动作上。

**所以我的判断是：Agent 现在的主要价值是压缩人的信息检索成本，不是替代人的决策。** 场景上优先选：内部工具、知识密集但低风险、输出给人二次确认的。

反过来说说我看到的最大工程障碍——不是模型能力，是**没有可靠的失败语义**。传统服务失败会抛异常，你能捕获、能重试、能熔断。模型"答错了"不抛异常，它返回一个格式完全正确、语气非常自信的错误答案。我项目里为这件事专门做了三态检索状态和降级原因枚举，就是想把"看起来成功但实际没用"这件事变成一个可以被代码判断的状态。

**项目证据**
- `app/agent/rag_v2/nodes.py:32-33` — `MIN_GROUNDEDNESS_SCORE`、`MIN_COVERAGE_SCORE` 双阈值
- `app/agent/rag_v2/nodes.py` — `validate_answer_node`，模型校验模型
- `app/agent/rag_v2/state.py` — `RetrievalStatus` 三态（OK / EMPTY / FAILED），把"空结果"和"失败"分开
- `app/agent/rag_v2/state.py` — `DegradeReason` 枚举，按修复方向分区
- `app/services/first_response_service.py:155` — `analyze` 内部对所有异常降级，输出给人看的报告而非自动动作
- 自动执行动作、审批流、human-in-the-loop 闸门：项目中没有，不能硬套。这正是我上面说"不敢接"的那部分

#### Q3 · 相比传统工作流，Agent 有哪些优势？生产环境中又存在哪些工程问题？

**回答**

这题我能给一手对比，因为**这两种写法在我这个仓库里同时存在，跑的是同一个业务**。

`rag_agent_service.py` 是 Agent 写法：`create_agent(model, tools, checkpointer)`，检索是一个 tool，模型自己决定调不调。`rag_v2/graph.py` 是工作流写法：五个节点固定连边，检索必然发生、必然扇出三路。

**Agent 的优势，我实际感受到的有两条。**

一是**开发成本低到不成比例**。Agent 那条路径核心就是 `create_agent` 一次调用，加上工具函数本身。工作流那条我写了 `graph.py`、`nodes.py`、`state.py`、`instrumentation.py` 四个文件。功能上前者还多一项：多轮工具调用天然支持，模型觉得第一次检索不够可以再来一次，我一行代码都不用写。

二是**输入分布变化时不用改代码**。工作流路径里 `NUM_SUB_QUERIES = 3` 是我写死的常量，不管问题简单还是复杂都扇出三路。简单问题上这是浪费，复杂问题上这可能不够。Agent 路径没有这个问题，模型自己判断。

**生产环境的工程问题，我按踩到的顺序说。**

**第一，没有预算控制。** Agent 循环里模型可以反复调工具，我在 `rag_agent_service.py` 里没有任何步数上限——grep 过整个仓库，`recursion_limit` 一处都没有。这意味着最坏情况下一个请求的耗时和 Token 消耗都是不可预测的。工作流路径反过来是完全可预测的：3 个子查询 × 每路 top 4，融合后截到 6 条，文档上下文预算 2800 Token 硬截断，总预算 90 秒。我知道一个请求最多花多少。

**第二，没有分支级的失败语义。** 这个在工作流路径里我处理得很细：`retrieve_each_node` 的 docstring 写明了为什么任何异常都不向上抛——LangGraph 的语义是"任一并行分支抛异常 → 整个 super-step 失败 → 整张图崩"，兜住之后语义才变成"一个分支失败 = 证据少一份"。然后 `dedup` 作为 fan-in 后第一个节点，是唯一能看到全部分支结果的位置，在那里判定是 `PARTIAL_RETRIEVAL` 还是 `RETRIEVAL_FAILED`。Agent 路径里工具失败就是工具失败，模型看到一段错误文本，它怎么反应我控制不了。

**第三，观测粒度对不上。** 这是我最终选工作流的直接原因。工作流路径每个节点一条 span 落 `chat_run_spans`，`retrieve_each` 被扇出成 3 个实例就记 3 条，我能对出每条分支各花了多久。Agent 路径我只能拿到整体耗时和一串 `tool_calls` 名字，中间到底为什么慢，日志里对不出来。

**第四，熔断挂不上去。** 工作流里我把熔断器包在 `asyncio.to_thread` **外面**——理由是熔断已经打开时不该再占用一个线程池 worker 去执行必定失败的调用。这个位置只有在我自己控制调用点的时候才挂得上。Agent 路径里工具调用由框架发起，我没有那个插入点。

**结论：** Agent 适合原型和探索期，工作流适合上线。我的做法是保留 Agent 路径做对照，主链路走工作流，把模型的自主性收缩到三个明确的节点内部。

**项目证据**
- `app/services/rag_agent_service.py:133-136` — Agent 路径的全部编排代码
- `app/agent/rag_v2/graph.py:44-51` — 工作流路径的固定连边
- `app/agent/rag_v2/nodes.py:28-30` — 写死的 `NUM_SUB_QUERIES=3` / `RETRIEVE_TOP_K=4` / `FINAL_TOP_K=6`
- `app/config.py:81,130` — `chat_total_budget_seconds=90.0`、`rag_document_context_token_budget=2800`
- `app/agent/rag_v2/nodes.py` — `retrieve_each_node` docstring 里的 super-step 崩图说明
- `app/agent/rag_v2/nodes.py:214-233` — `dedup_node` 作为唯一 fan-in 观测点做三态判定
- `app/agent/rag_v2/nodes.py:175-188` — 熔断器包在 `to_thread` 外侧及其理由
- `app/models/chat_run_span.py:83,94` — 按 `node` 索引的 span 耗时表
- Agent 路径的步数上限：项目中没有，不能硬套。全仓库无 `recursion_limit`，这是上面第一条问题的直接证据

#### Q4 · Token 是如何划分的？为什么 `strawberry` 在部分模型中会被拆成多个 Token？

**回答**

先把结论说清楚：**我的项目里没有真正的分词器**，只有一个用来做预算控制的估算函数。所以这题我分两半答，理论归理论，代码归代码，不混。

**理论部分。** 主流模型用的是 BPE 一类的子词切分。训练时从字符级别开始，统计语料里相邻符号对的共现频率，把最高频的那一对合并成一个新符号，反复迭代到词表满（比如 10 万）。结果是：高频词整体成为一个 Token，低频词被拆成若干个高频子词片段。

`strawberry` 被拆开的原因就在这里——它作为一个整体在英文语料里的频率不足以让它在词表里独占一个条目，于是按已有的合并规则切成类似 `str` + `aw` + `berry` 的几段。切法完全取决于那个特定词表的合并历史，不同模型切法不同。

这件事的实际后果是**模型看不到字母**。它拿到的是三个子词 ID，"strawberry 里有几个 r"这个问题需要的信息在切分阶段就已经丢了。这不是推理能力不足，是输入表示的问题。同理，反转字符串、数字逐位运算这类任务模型都不擅长，根源相同。

**代码部分。** 我项目里的 `_estimate_tokens` 是这么写的：

```python
def _estimate_tokens(text: str) -> int:
    """中文按字符、其他内容按约四字符一 Token 的保守估算。"""
    cjk = sum("㐀" <= char <= "鿿" or "豈" <= char <= "﫿" for char in text)
    return cjk + max(0, (len(text) - cjk + 3) // 4)
```

中文一字一 Token，其他内容四字符一 Token，向上取整。这**不是分词**，是一个刻意偏保守的上界估算，用途只有一个：在 `_format_context` 里按预算装文档时不要超。它宁可高估也不能低估——低估会让实际请求超出模型上下文，高估只是少装一点文档。

`conversation_memory_service.py:48` 里有一份同规则的实现，用 `_CJK_RE` 正则做 CJK 统计，服务于会话记忆的预算。两处规则一致但代码独立，这是我知道的一处重复。

**项目证据**
- `app/agent/rag_v2/nodes.py:631-634` — `_estimate_tokens`，中文 1、其他 1/4 的保守估算
- `app/services/conversation_memory_service.py:27,48-54` — `_CJK_RE` 与 `estimate_tokens`，同规则的第二份实现
- `app/agent/rag_v2/nodes.py:613-628` — `_format_context` 里估算值的唯一用途：按预算装文档
- `app/agent/rag_v2/nodes.py:637-650` — `_truncate_to_token_budget`，CJK 记 1.0、其他记 0.25 的逐字裁剪
- 真正的 BPE 分词器：项目中没有，不能硬套。仓库里没有 `tiktoken`、没有 `transformers` 的 tokenizer、没有任何词表文件。Token 计费由 DashScope 服务端完成，本地只做预算估算

#### Q5 · 为什么大模型以 Token，而不是字符或完整单词作为基本处理单位？

**回答**

这是在两个极端之间取平衡的结果，两头都不可行。

**按完整单词不行，因为词表会爆而且永远不全。** 英文光是常用词加变形就是几十万量级，加上专有名词、代码标识符、拼写错误，是开放集合。词表大小直接决定输出层的参数量和每一步 softmax 的计算量——词表翻倍，输出层参数翻倍。更要命的是任何词表都会遇到没见过的词，OOV 只能映射成 `<unk>`，信息直接丢失。中文更没法按词，分词本身就有歧义。

**按字符不行，因为序列会爆。** 词表确实小了（几千个 Unicode 字符就够），但同一段文本的序列长度会拉长好几倍。Self-Attention 的复杂度是序列长度的平方，长度乘 4 意味着注意力计算量乘 16。而且单个字符携带的语义几乎为零，模型得花很多层去把字符重新组装成有意义的单元。

**子词是中间解。** 词表可控（几万到十几万），高频词整体成 Token 保住了语义密度，低频词拆成子词片段保证了零 OOV——任何字符串都能被表示，最坏情况退化到字符级。

这个平衡的**代价**就是上一题说的：字符级信息在切分后不可见。所以模型数字母、反转字符串这类任务表现差。这不是缺陷，是设计取舍的必然结果。

从我做工程的角度补一句实际影响：**Token 是计费和限长的单位，所以它直接决定了我的预算设计。** 我项目里所有预算都是 Token 单位——文档上下文 2800、会话记忆 1800、摘要 700、压缩触发阈值 2400。如果基本单位是字符，这些数字会完全不同；如果是单词，中文根本没法定这个预算。我的估算函数对中文按一字一 Token，对其他按四字符一 Token，就是因为 CJK 在主流词表里基本是一字一 Token，而英文一个 Token 平均覆盖三到四个字符——**这个经验比例本身就是子词切分的直接产物**。

**项目证据**
- `app/config.py:130,133-136` — 四个 Token 单位的预算常量：文档 2800、会话 1800、摘要 700、压缩阈值 2400
- `app/agent/rag_v2/nodes.py:631-634` — 中文 1 : 英文 1/4 的经验比例，来源就是子词切分的实际表现
- `app/services/conversation_memory_service.py:57-58` — `_message_token_cost` 每条消息额外 +4，为角色前缀等结构开销留量
- 词表、分词器训练、BPE 合并规则：项目中没有，不能硬套

#### Q6 · 讲一下 Transformer 和 Self-Attention，Q、K、V 分别有什么作用，具体如何计算？

**回答**

**项目中没有 Transformer 实现，不能硬套。** 仓库里没有任何模型结构代码，模型全部走 DashScope API 调用。这题我按理论答，最后只讲一点它对我工程决策的实际影响。

**Self-Attention 的计算。** 输入序列每个位置的表示 $x_i$，先用三个独立的线性变换投影出三个向量：

$$Q = XW^Q,\quad K = XW^K,\quad V = XW^V$$

然后：

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**三者的分工，用一个检索的类比最容易说清楚**（这个类比对我来说不是比喻，是我天天在写的东西）：

- **Q（Query）是"我在找什么"** —— 当前位置发出的查询意图
- **K（Key）是"我能被什么找到"** —— 每个位置的可被匹配特征
- **V（Value）是"找到我之后拿走什么"** —— 每个位置实际提供的内容

$QK^T$ 算的是每个位置的查询和所有位置的键的相似度，得到一个 $n \times n$ 的注意力分数矩阵。softmax 归一化成权重后，对 V 做加权求和——**每个位置的输出是全序列 V 的加权平均，权重由 Q·K 决定**。

**为什么除 $\sqrt{d_k}$。** 如果 Q、K 的分量近似独立、方差为 1，那么 $q \cdot k$ 的方差是 $d_k$。$d_k$ 取 64 或 128 时点积的量级会很大，softmax 落进饱和区，梯度接近 0。除以 $\sqrt{d_k}$ 把方差拉回 1。

**为什么 K 和 V 要分开。** 这是最容易被追问的点。"用来匹配的特征"和"匹配上之后提供的信息"不需要是同一个东西。分开之后模型可以学到"我用 A 特征被找到，但我提供 B 内容"。如果强行让 K = V，表达能力会受限。

**多头。** 把 $d_{model}$ 拆成 $h$ 份，每份独立做上面一套，最后拼接再过一个线性层。目的是让不同的头关注不同类型的关系——有的头学句法依赖，有的头学指代。单头只能给出一套注意力分布。

**这件事对我工程的实际影响，只有一条但很重要：$QK^T$ 是 $n \times n$，复杂度和显存都是序列长度的平方。** 这就是为什么我的项目里所有上下文都是硬预算而不是"能塞就塞"：`rag_document_context_token_budget = 2800`，`_format_context` 装到超预算就 `break` 停止，不是截断最后一条而是干脆不装。上下文长度不是线性成本，多塞一倍的代价远大于一倍。

**项目证据**
- Transformer / Self-Attention / 多头注意力实现：项目中没有，不能硬套。仓库无任何模型结构代码，无 `torch.nn` 定义的层
- `app/core/llm_factory.py` — 模型全部走 DashScope API，本地只做客户端配置
- `app/config.py:130` — `rag_document_context_token_budget = 2800`，注意力平方复杂度在工程侧的唯一落点
- `app/agent/rag_v2/nodes.py:618-623` — `available <= 0` 时直接 `break`，不再往下装文档
- `app/services/vector_search_service.py:42-43` — 项目里真实的"Q/K 匹配"是向量检索的余弦相似度，和注意力是同一个数学动作但不是同一层东西，我不把它们混为一谈

#### Q7 · 什么是 Lost in the Middle？长上下文中的重要信息为什么容易被模型忽略？

**回答**

**现象**：把同一份关键信息分别放在长上下文的开头、中间、末尾，模型的回答准确率呈 U 型——开头和结尾好，中间明显掉。上下文越长，中间那段塌得越厉害。原始论文测的是多文档问答，把答案所在的文档放在第 10 位（共 20 篇）时，准确率甚至低于只给模型 20 篇里的少数几篇。

**原因我理解有三层。**

一是**位置编码的训练分布**。长距离位置组合在训练数据里出现得少，模型对"第 8000 个 Token 和第 200 个 Token 的关系"这种远距离依赖学得不充分。

二是**注意力被稀释**。softmax 是全序列归一化的，序列越长，单个位置能分到的权重越薄。中间位置既没有开头的"起始锚点"效应，也没有结尾的"就近效应"，是最容易被摊薄的。

三是**训练目标的偏置**。next-token prediction 天然让模型更依赖临近上文，而 instruction tuning 的样本里指令通常在开头，这两股力量共同强化了首尾。

**我项目里针对这件事做了什么。**

最直接的一条：**不喂长上下文**。这听起来像回避，但它是真实的设计决定。`FINAL_TOP_K = 6`，融合去重后只保留 6 条送进生成；文档上下文预算 2800 Token 硬顶。我不做"上下文越大越好"的假设，也就基本不进入 Lost in the Middle 的高危区间。

第二条是**顺序有意义并且被利用**。`dedup_node` 里 `seen` 集合按有序遍历去重，保留的是 RRF 融合后的排序，最后 `[:FINAL_TOP_K]` 截断——所以进 `_format_context` 的 6 条是**按相关性降序**的，最相关的排第 1 位、进开头。这不是巧合，是我留着的：既然首位效应存在，就让最该被看到的排最前。

第三条是**给每条加来源标注**。`_format_context` 里每条前缀是 `[{idx}] (源子查询: {src})`。编号让答案能引用具体某条，子查询来源让我在校验阶段能对出这条证据是回答哪个子问题的。带标注的块比裸文本更容易被模型定位——这一点我没有实验数据，是从 `validate_answer` 的表现上感觉到的。

**没做的部分**：重要信息重复放在首尾、按位置重排（把次相关的塞中间）、长上下文的位置敏感性测试，项目中都没有，不能硬套。

**项目证据**
- `app/agent/rag_v2/nodes.py:30` — `FINAL_TOP_K = 6`，从源头限制上下文长度
- `app/config.py:130` — `rag_document_context_token_budget = 2800` 硬预算
- `app/agent/rag_v2/nodes.py:214-233` — `dedup_node` 有序去重保留 RRF 排序
- `app/agent/rag_v2/nodes.py:230` — `[:FINAL_TOP_K]` 截断，最相关的落在开头
- `app/agent/rag_v2/nodes.py:619-620` — `[{idx}] (源子查询: {src})` 前缀，编号 + 来源标注
- `app/services/vector_search_service.py:92-99` — RRF 融合决定顺序，顺序进而决定位置
- 首尾重复放置、按位置重排、位置敏感性实验：项目中没有，不能硬套

#### Q8 · 上下文窗口越大越好吗？它对显存、延迟、成本和模型效果有什么影响？

**回答**

不是。四个维度分别说，最后说我项目里怎么定的。

**显存**：训练/预填充阶段注意力矩阵是 $n^2$，推理阶段 KV Cache 是线性但常数很大——每层每个头都要缓存 K 和 V，长上下文下 KV Cache 往往比模型权重本身还占显存。这决定了服务端的并发上限：单请求上下文翻倍，同一张卡能并发的请求数大致减半。

**延迟**：要分两段看，这是我在项目里实际观察到的区别。**预填充**（处理输入）是可并行的，但计算量随长度平方增长；**解码**（逐 Token 输出）每一步都要和全部历史 KV 做注意力，所以上下文越长，每个输出 Token 都更慢。后者对流式响应的体感影响更大——首 Token 慢是预填充，之后一直卡是解码变慢。

**成本**：输入 Token 直接计费。而且这里有个容易被忽略的点：**多轮对话下历史会被反复重发**。第 10 轮请求携带前 9 轮的内容，这些 Token 每一轮都重新计费一次。所以上下文管理不做，成本是随轮数超线性增长的。这正是我做滚动摘要的直接动机。

**效果**：这是最反直觉的一维——**上下文变长，效果可能变差**。就是上一题的 Lost in the Middle。喂 20 篇文档不如喂 5 篇精准的，因为无关内容会稀释注意力，也会给模型提供更多可以拿来"编"的素材。我在 `validate_answer` 里见过这个：证据块给多了，模型更容易把两个不相关文档里的内容缝到一起。

**所以我的项目是怎么定的。** 我把总预算拆成三块，各自有硬顶，加起来受一个总限制：

- 文档上下文 `rag_document_context_token_budget = 2800`
- 会话记忆 `conversation_memory_context_token_budget = 1800`
- 其中摘要部分不超过 `conversation_memory_summary_token_budget = 700`

拆开的理由写在 `_format_context` 的 docstring 里：**"避免知识库内容挤占会话记忆和问题空间"**。如果只有一个总预算，检索文档会把会话历史全部挤掉——因为文档块通常又长又多，抢预算天然占优。分块预算保证每一类信息都有下限。

`_render_context` 里还有一处细节值得说：它预留了 `【会话滚动摘要】` 和 `【最近会话原文】` 这两个标题本身的 Token，注释是"确保最终渲染文本（含标签）也不会突破总预算"，最后再 `_truncate_text` 兜一次硬上限。**预算控制要算到渲染后的最终文本，不是算到内容为止。**

**项目证据**
- `app/config.py:130` — 文档上下文 2800
- `app/config.py:133-134` — 会话记忆 1800，其中摘要不超过 700
- `app/agent/rag_v2/nodes.py:613-614` — docstring 明写分预算理由："避免知识库内容挤占会话记忆和问题空间"
- `app/agent/rag_v2/nodes.py:618-623` — 逐条扣减预算，`available <= 0` 就停
- `app/services/conversation_memory_service.py:213-225` — 预留标题 Token 的预算计算
- `app/services/conversation_memory_service.py:240-241` — 渲染后再兜一次硬上限
- `app/config.py:81` — `chat_total_budget_seconds = 90.0`，延迟侧的对应约束
- KV Cache 占用测量、上下文长度与准确率的实测曲线：项目中没有，不能硬套。显存与延迟数据在 DashScope 服务端，我这边观测不到

#### Q9 · 对话超过上下文限制后怎么处理？滑动窗口、摘要压缩和向量召回怎么选择？

**回答**

三种我项目里都有，而且**是组合用的，不是选一个**。先说各自的位置，再说为什么这么组合。

**滑动窗口** —— `_recent_message_count`，保留最近 `conversation_memory_recent_turns = 3` 轮。注意它是**按轮次而不是按条数**切的，实现是从尾往前扫，数到第 3 条 user 消息就停：

```python
for index in range(len(messages) - 1, -1, -1):
    if messages[index].role.value == "user":
        user_messages_seen += 1
        if user_messages_seen >= self._recent_turns:
            start_index = index
            break
```

按条数切会把"用户问 + 助手答"这一对拆开，留下一个孤零零的助手回复，上下文反而更难读。

**摘要压缩** —— 超过 `conversation_memory_compact_threshold_tokens = 2400` 才触发，只压缩窗口之外的旧消息（`pending_messages[:-recent_count]`），压出来的摘要存进 snapshot，带一个游标 `summarized_through_message_id` 记录压到哪条为止。下次只处理游标之后的新消息，**已压缩的部分不重复压**。

摘要的 prompt 是限定格式的四个标题：【明确事实】【任务进展】【约束与偏好】【待确认项】。而且明确写了"不要把模型猜测、SOP 内容或历史数值写成实时事实；发生冲突时以较新的用户消息为准"——摘要最大的风险是把猜测固化成事实，然后一路传下去，所以这条约束必须写进 prompt。

**向量召回** —— 就是 RAG 主链路本身，但它检索的是知识库文档，**不是历史对话**。这个区别要说清楚：我没有做"把历史对话向量化后按需召回"，`rag_v2` 检索的目标是 Milvus 里的运维文档。

**为什么这么组合。** 三者管的是不同性质的信息：

- 最近几轮的**原文**必须保留，因为指代、追问、"刚才那个"全靠它，摘要会把这些细节磨掉
- 更早的历史只需要**结论**，滚动摘要够了
- 知识库内容量级上根本装不进上下文，**只能按需检索**

**降级路径**，这是我觉得这块设计里最实用的一处：摘要调的是 LLM，会失败。失败时的处理是——

```python
except Exception as exc:
    # 摘要是节省 Token 的优化，不可阻断主聊天；失败时退回最近消息窗口。
    logger.warning(f"会话记忆摘要失败，降级为最近消息窗口: {exc}")
    return ""
```

返回空串，`build_context` 那边 `if summary:` 判空，直接跳过保存 snapshot，游标不动，上下文只带最近窗口。**摘要是优化，不是必需路径**，它挂了对话照样能进行，只是这一轮的旧历史丢了。下一轮阈值还在，会再试一次。

还有一处边界处理值得提：`_messages_after_snapshot` 里，如果游标指向的消息已被软删除，找不到匹配，返回的是**空列表**而不是全部消息：

```python
# 游标对应的消息已被软删除时，保守地不注入旧原文，避免重复历史。
return []
```

保守方向选的是"少注入"而不是"多注入"——多注入会导致已被摘要过的历史又以原文形式出现一遍，模型会看到重复内容。

**项目证据**
- `app/services/conversation_memory_service.py:146-158` — `_recent_message_count`，按轮次而非条数的滑动窗口
- `app/config.py:136` — `conversation_memory_recent_turns = 3`
- `app/config.py:135` — `conversation_memory_compact_threshold_tokens = 2400` 触发阈值
- `app/services/conversation_memory_service.py:104-126` — snapshot 游标机制，已压缩部分不重复压
- `app/services/conversation_memory_service.py:28-35` — 摘要 prompt 的四段格式与"不要把猜测写成事实"约束
- `app/services/conversation_memory_service.py:183-186` — 摘要失败降级为最近窗口
- `app/services/conversation_memory_service.py:143-144` — 游标失效时保守返回空列表
- `app/services/conversation_memory_service.py:249-272` — `_take_tail_to_budget`，单条超预算时仍保留并裁剪
- `app/services/conversation_memory_service.py:1-5` — 模块 docstring：原文留 PostgreSQL 做审计，模型上下文只用摘要 + 尾部
- 历史对话的向量化召回：项目中没有，不能硬套。向量检索的目标是知识库文档，不是对话历史

#### Q10 · 简单介绍 SFT 的训练流程，并比较 PPO、DPO 和 GRPO 的区别。

**回答**

先说清楚边界：**训练侧项目中没有，不能硬套。** 这个仓库是纯推理服务，模型全部走 DashScope 的 HTTP API（`llm_factory` 创建 OpenAI 兼容客户端指向 `dashscope.aliyuncs.com`），没有任何 loss 定义、optimizer、训练循环。唯一在本地加载过权重的地方是 `tests/eval/_test_rerank.py` 的 reranker 实验，那也只是 forward 推理。所以下面是理论回答，不是我在这个项目里做过的事。

**SFT 的流程。** 数据是 (prompt, response) 对，做的还是 next-token 预测的交叉熵，关键差别在 **loss mask**：prompt 部分的 token 不算 loss，只在 response 上回传。这一点新手常错，把 prompt 也算进 loss 会让模型学着去生成用户的提问。工程上还有 packing——把多条短样本拼进一个序列填满上下文以提高显存利用率，但要用 attention mask 隔断，否则样本之间会互相看到。全参微调 vs LoRA 的选择基本由显存决定，LoRA 只训低秩增量矩阵，冻结主干。

数据这一层比算法重要得多：几千条高质量样本通常胜过几十万条爬来的噪声数据，因为 SFT 是在教格式和风格，噪声会被忠实地学进去。

**PPO。** 需要四个模型同时在显存里：policy（在训的）、ref（冻结的初始模型，算 KL 惩罚防止跑偏）、reward model（先用偏好对训出来的打分器）、value/critic（估计状态价值做优势的 baseline）。流程是两段：先用 pairwise ranking loss 训 RM，再用 RM 的打分跑 PPO。目标函数里的 clip 项限制单步策略更新的幅度，避免一次更新把策略推到 RM 打分虚高但实际崩坏的区域。问题就是四个模型的显存和一堆敏感超参。

**DPO。** 核心洞察是数学上的：PPO 那个带 KL 约束的目标有闭式最优解，从这个解可以把 reward 反解成 policy 与 ref 的对数比，代进 Bradley-Terry 偏好模型，得到一个只含 policy 和 ref 的 loss。于是 RM 和 critic 都不需要了，只剩两个模型，训练退化成一个分类问题——让 chosen 的相对对数概率高于 rejected。代价是**离线**：只能在给定的偏好对上学，不能像 PPO 那样采样探索，对偏好数据覆盖不到的行为没有约束。

**GRPO。** DeepSeek 那套的做法，砍掉的是 value model。对同一个 prompt 采样一组 G 个回答，用**组内 reward 的均值和标准差做归一化**，直接当优势用，替代 critic 的估计。省一个模型，而且它天然适合有可验证答案的任务——数学题、代码题的 reward 可以是规则判定（答案对不对、测试过不过），连 RM 都不用学。

**三者的递进关系一句话概括**：PPO 四模型、在线、最灵活也最重；DPO 两模型、离线、最轻但不能探索；GRPO 去掉 critic、在线、reward 可以是规则，适合可验证任务。

**项目证据**
- SFT / RLHF 训练代码：项目中没有，不能硬套。仓库是纯推理服务
- `app/core/llm_factory.py` — 只创建指向 DashScope 的 OpenAI 兼容客户端
- `app/config.py:46,129` — `qwen-max` 是唯一的对话模型，通过 API 调用
- `tests/eval/_test_rerank.py` — 唯一本地加载权重的地方，只做 forward
- `app/agent/rag_v2/nodes.py` — `validate_answer_node` 用 LLM 给答案打分，这是推理期校验，**不是**训练期的 reward model，别混

#### Q11 · 为什么项目采用多 Agent？如果改成单 Agent，会遇到什么问题？

**回答**

这题我必须先纠正前提：**这个项目不是多 Agent，项目中没有多 Agent，不能硬套。** 仓库里没有 supervisor、没有 sub-agent、没有 handoff 机制，grep 这几个概念一条都不返回。`app/agent/aiops/` 这个目录现在只剩 `__pycache__`，源文件已经删掉了。

实际存在的是三条**单** Agent / 无 Agent 的路径：

一是 `rag_agent_service.py`，用 `create_agent` 配 `MemorySaver`，一个 ReAct Agent 挂一个工具（`retrieve_knowledge`），循环由框架自动跑。

二是 `rag_v2`，手写的五节点 `StateGraph`：rewrite → retrieve_each → dedup → generate → validate_answer，固定 DAG，**根本没有 Agent 循环**，模型不决定下一步走哪里，边是写死的。

三是 `alert_diagnosis_orchestrator`，158 行的线性编排：归一化告警 → 建诊断任务 → planning → retrieving → diagnosing → 存报告 → DONE。它的 docstring 自己写明"本编排层负责流程"，是一条流水线，不是一群 Agent。

**有个容易被当成多 Agent 的地方要说清楚。** `_fanout_to_retrieve` 用 `Send` 把 retrieve_each 分裂成 N 个并行实例，N = 子查询数 = 3。这是**同一个节点的三个实例带不同 query**，不是三个有各自角色、各自工具、各自 prompt 的 Agent。分不清这两者，面试里会被追问穿。

**下面是设计判断，标注为推理不是项目事实。** 什么时候值得上多 Agent：工具数量涨到十几个以上时，单 Agent 的工具选择准确率会掉——所有工具描述都塞在一个 prompt 里，模型要在语义相近的工具间做区分；角色需要不同的模型或温度（比如规划要稳所以 temperature=0，创意生成要发散）；需要独立的失败域，一个子任务崩了不能拖垮全局。

改成单 Agent 会遇到的问题，按严重程度排：prompt 膨胀导致工具选择退化；上下文污染，前一个子任务的中间结果留在历史里干扰后一个；失败粒度粗，一次崩溃整个任务重来。反过来，多 Agent 的代价是状态合并、消息传递、循环检测都得自己做，本项目工具只有一个，这些代价完全不值得付——所以是单 Agent，这个选择本身是对的。

**项目证据**
- 多 Agent 编排（supervisor / sub-agent / handoff）：项目中没有，不能硬套
- `app/agent/aiops/` — 目录只剩 `__pycache__`，源文件已删除
- `app/services/rag_agent_service.py:133-136` — `create_agent` + `MemorySaver`，单 Agent 单工具
- `app/agent/rag_v2/graph.py:26-51` — 五节点固定 DAG，边写死，无 Agent 循环
- `app/agent/rag_v2/graph.py:19-22` — `_fanout_to_retrieve`，同节点三实例，不是三个 Agent
- `app/agent/rag_v2/nodes.py:28` — `NUM_SUB_QUERIES = 3`，扇出宽度固定
- `app/services/alert_diagnosis_orchestrator.py:1-25` — docstring 明写是线性流程编排
- `app/tools/knowledge_tool.py:13-14` — 全项目只有这一个 LangChain 工具

#### Q12 · 多个子 Agent 如何分工、分发任务和合并全局状态？为什么没有采用去中心化协作？

**回答**

子 Agent 分工：**项目中没有，不能硬套。** 但"任务分发 + 全局状态合并"这套机制在**节点级别**有完整的真实实现，而且踩过的坑正是这题的核心，我按真实代码讲。

**分发。** `_fanout_to_retrieve` 是一个 conditional edge 的目标函数，返回一个 `Send` 列表：

```python
def _fanout_to_retrieve(state: RAGState) -> list[Send]:
    sub_queries = state.get("sub_queries", []) or []
    return [Send("retrieve_each", {"query": q}) for q in sub_queries]
```

每个 `Send` 携带自己的 payload，框架据此把 retrieve_each 实例化 N 份并发跑。注意这里 `or []` 的兜底——rewrite 失败返回空列表时扇出宽度是 0，图不会崩，只是没有分支。

**合并全局状态，这是最容易写错的地方。** state 里凡是被并行分支写的字段，都必须带 reducer：

```python
Annotated[List[Document], operator.add]
```

每个分支返回一个**单元素 list**，框架用 `operator.add` 把它们拼起来。如果分支直接 `return {"docs": docs}` 去 set 同一个 key，LangGraph 会抛 `InvalidUpdateError`——它无法决定该采信哪个分支。这个约束逼出了一条通用纪律：**并行分支只能 append，不能 set。** 同样的纪律在 span 记录上也体现了，`record_span` 的注释写得很直白："注意这里只 append，不 set —— 这是并行分支下不丢数据的前提。"

**全局视角在哪。** `dedup_node` 是 fan-in 之后的第一个节点，也是**唯一能看到全部分支结果**的位置。所以三态判定必须放在这里：部分分支失败 → `PARTIAL_RETRIEVAL`，全失败 → `RETRIEVAL_FAILED`，全成功但空 → `RETRIEVAL_EMPTY`。节点代码里注释明确写了这个理由。单个 retrieve_each 分支只知道自己成没成，不知道兄弟分支的情况，判不了"部分成功"。

**为什么不去中心化。** 就是上面那条：降级判定需要一个能看到全局的位置。去中心化协作下，三个分支互相通信，没有任何一方掌握"三个里坏了两个"这个事实，也就没人能做出"证据不足，改成追问而不是硬答"的决定。中心化的成本是那个中心节点成为单点，收益是可判定性——对一个要对答案质量负责的 RAG 系统，可判定性更值钱。

另外一个现实理由：去中心化要处理消息路由、终止条件、活锁，这些在只有三个同质分支的场景下纯粹是自找麻烦。

**项目证据**
- 多子 Agent 的分工与协商：项目中没有，不能硬套
- `app/agent/rag_v2/graph.py:19-22` — `Send` 分发，含 `or []` 空兜底
- `app/agent/rag_v2/graph.py:45` — `add_conditional_edges("rewrite", _fanout_to_retrieve, ["retrieve_each"])`
- `app/agent/rag_v2/state.py` — `Annotated[List[...], operator.add]` reducer，并行写入的唯一合法方式
- `app/agent/rag_v2/nodes.py:214-233` — `dedup_node`，fan-in 后第一个节点，三态判定的唯一可行位置
- `app/core/span_context.py` — `record_span` 只 append 不 set，注释说明这是并行不丢数据的前提
- `app/agent/rag_v2/nodes.py:36-48` — 子查询生成的 prompt，分发内容的来源
- 去中心化 Agent 协作、消息总线、协商协议：项目中没有，不能硬套

#### Q13 · 两个 Agent 给出冲突结论时如何仲裁？什么情况下需要人工介入？

**回答**

Agent 之间的仲裁：**项目中没有，不能硬套**——没有两个 Agent，也就没有 Agent 间冲突。但"第二个模型审第一个模型的输出、冲突时谁说了算"这件事，项目里有真实实现，是双角色冲突的最小形态。

**真实的仲裁结构。** `generate_node` 产出答案，`validate_answer_node` 另起一次 LLM 调用给这个答案打两个分：groundedness（有没有编造证据里没有的内容）和 coverage（有没有覆盖问题的关键点）。任一项低于阈值就拦下。

冲突时谁赢：**校验方赢**，但赢的方式很重要——不是让 validator 去改写答案，而是把答案降级成"证据不足 + 明确指出缺什么 + 让用户补充"。这是我特意选的：让第二个模型改写第一个模型的输出，等于把两次幻觉叠在一起，谁都不知道最终那句话的依据在哪。只做"拦 + 说明为什么拦"，责任链是清楚的。

**fail-open 的取舍。** 校验开关关掉时走 fail-open，答案照样返回，但降级原因记的是 `FEATURE_DISABLED` 而不是 `LLM_ERROR`。这个区分不是洁癖——`LLM_ERROR` 会把值班同学送去查模型配额和鉴权，而实际原因只是有人把开关关了。降级原因的枚举是按**修复方向**分区的，不是按"哪里出错了"分区。

**人工介入，项目里真的有一处。** 协议 PDF 入库的状态机里有 `AWAITING_CONFIRMATION`：PDF 抽取 → 结构化 → 校验之后，流程**停下来等人确认**，人确认了才进 `WRITING` 真正写库，否决就是 `REJECTED`。这是仓库里唯一真正的 human-in-the-loop，而且它选的介入点是对的——写库是不可逆操作，不可逆动作前面卡一道人工，比事后回滚便宜。

对比之下，RAG 那条链路上 validator 判低分只是降级返回，**没有任何通道把这个冲突升级给人**。是否该有：如果同一类问题反复被判低分，说明知识库缺内容，这个信号值得推给运营。目前只落在 `is_bad_case` 字段里等人主动查，没有推送。

**项目证据**
- Agent 间投票 / 多数表决 / 置信度加权仲裁：项目中没有，不能硬套
- `app/agent/rag_v2/nodes.py` — `validate_answer_node`，第二次 LLM 调用审第一次的输出
- `app/agent/rag_v2/nodes.py:32-33` — `MIN_GROUNDEDNESS_SCORE`、`MIN_COVERAGE_SCORE` 双阈值
- `app/agent/rag_v2/nodes.py:595-610` — 拦下后组装"仍缺少 / 已拦截未证实判断"的追问文案，不改写答案
- `app/agent/rag_v2/nodes.py` — `enable_answer_validation=false` 时 fail-open，用 `FEATURE_DISABLED` 而非 `LLM_ERROR`
- `app/models/protocol_ingestion.py:19-35` — `AWAITING_CONFIRMATION` / `REJECTED`，项目里唯一的人工确认卡点
- `app/models/chat_run_trace.py:46` — `is_bad_case` 建了索引，但没有主动推送通道
- 把冲突主动升级给人的告警链路：项目中没有，不能硬套

#### Q14 · 了解哪些进程和内存监控工具？线上服务出现 CPU 飙高或内存持续增长时怎么排查？

**回答**

**项目里真实有的监控。** `app/core/metrics.py` 用 prometheus_client 暴露四个指标，走 `/metrics` 端点：LLM 调用耗时是 Histogram（按 node 打标签），降级次数和检索失败次数是 Counter（分别按 reason、code 打标签），熔断器状态是 Gauge。

选型的两个判断写在文件注释里，值得一说。耗时用 Histogram 不用 Gauge：Gauge 只留"最后一次"，故障时平均值和 P99 都算不出来，最没用；Histogram 存的是分桶计数，可以跨实例相加再算分位数。熔断器状态用 Gauge 不用 Counter：它是一个**当前值**，Counter 只能增，表达不了"又闭合了"。还有一个坑是 `refresh_circuit_breaker_metrics` 解决的——如果只在 `record_success` 里更新 Gauge，一个打开后再没有流量的熔断器会永远停在旧值，所以要在采集时主动刷一遍。

**没有的部分说清楚：** psutil 项目里没有，进程级 CPU/内存采集不能硬套。`mcp_servers/monitor_server.py` 看名字像监控，实际是**mock 数据生成器**——`base_cpu = 10.0` 加 `random.uniform(-2, 2)` 造出时序数据给 AIOps 链路当测试输入，不采集任何真实进程指标。把它当监控实现来讲，面试官一翻代码就穿了。

**下面是排查方法论，标注为通用知识不是本项目实践。**

CPU 飙高：先 `top -H -p <pid>` 找到真正吃 CPU 的**线程**而不是进程，`printf %x <tid>` 转十六进制，再去对线程栈。Python 侧首选 py-spy——不用改代码、不用重启、attach 上去直接出火焰图，这是它比 cProfile 强的地方，线上没有第二次机会重现的问题只能靠 attach 型工具。JVM 侧是 jstack 对 nid。

内存持续增长：先分清两种情况。RSS 涨但 Python 堆不涨，多半是内存碎片或 glibc arena 没归还，这类换 jemalloc 或调 `MALLOC_ARENA_MAX` 往往就压住了；堆真的涨才是对象泄漏，用 `tracemalloc` 取两次快照做 `compare_to` 定位分配点，再用 objgraph 看引用链找出谁还攥着它不放。

**本项目最可能被误判成泄漏的点**，这是可以基于代码验证的推断：BM25 检索器缓存把**全量语料常驻在内存里**，`_get_bm25_retriever` 按 `collection.num_entities` 判失效，语料越大常驻越大，而且没有上限。它不是泄漏——是设计上就常驻——但监控图上看起来和泄漏一模一样：随着入库文档增加而阶梯式上涨，且永不回落。真要治，得给它加容量上限或者改成外部 BM25 服务。

**项目证据**
- `app/core/metrics.py:85-113` — 四个指标：LLM 耗时 Histogram、降级 Counter、检索失败 Counter、熔断状态 Gauge
- `app/core/metrics.py:56-61` — 为什么用 Histogram 而不是 Gauge/Summary 的完整论证
- `app/core/metrics.py:108-113` — 为什么熔断状态用 Gauge：Counter 表达不了"又闭合了"
- `app/core/metrics.py:171-188` — `refresh_circuit_breaker_metrics`，避免无流量熔断器指标僵死
- `app/core/metrics.py:191` — `render_metrics`，`/metrics` 端点的数据来源
- psutil / 进程级资源采集：项目中没有，不能硬套
- `mcp_servers/monitor_server.py:207-229,355-381` — `base_cpu` / `base_memory` 加 `random.uniform` 造数据，是 mock 不是采集
- `app/services/vector_search_service.py:140-163` — BM25 全量语料常驻，无容量上限，会被误读成内存泄漏

#### Q15 · `mmap` 是什么？它解决了什么问题，适合哪些文件或进程通信场景？

**回答**

**项目中没有，不能硬套。** 仓库里 grep 不到 mmap，一处都没有。下面是理论回答。

mmap 把文件（或匿名内存）映射进进程的虚拟地址空间，之后读写文件退化成普通的访存操作，真正的磁盘 I/O 由缺页中断触发，内核按页调入。

**它省的是什么。** 常规 `read` 路径是：磁盘 → page cache → 用户缓冲区，中间有一次内核态到用户态的拷贝。mmap 让用户空间的虚拟页直接指向 page cache 的物理页，这次拷贝没有了。注意省掉的是**一次拷贝**，不是全部——数据从磁盘进 page cache 那一步依然存在。把 mmap 说成"零拷贝"是不准确的，`sendfile` 那种才是全程不进用户态。

**适合的场景。** 大文件随机读，典型是索引文件：只访问其中几页，其余页永远不用调进来，虚拟地址空间可以远大于物理内存。多进程共享只读数据，`MAP_SHARED` 下多个进程映射同一文件共用同一份 page cache，N 个进程的常驻内存不是 N 倍。进程间通信，匿名 `MAP_SHARED` 加锁就是一块共享内存，比管道少一次拷贝。

**不适合的场景，这部分更能体现是否真懂。** 顺序流式读——内核对 `read` 的预读做得很好，mmap 反而要一次次缺页中断，反而慢。小文件——建立映射本身有 mmap 系统调用加页表操作的开销，收益盖不住。需要精确控制落盘时机的写入——脏页什么时候回写由内核决定，要保证必须 `msync`，而且 mmap 写入时磁盘满或 I/O 错误的表现是 `SIGBUS` 信号而不是返回错误码，错误处理比 `write` 难写得多。

**和本项目的关系。** 向量库确实是 mmap 的经典用户，Milvus 和 FAISS 都支持 mmap 索引，目的就是让索引不必全量进内存。但本项目用的是 Milvus **独立服务**，索引在 Milvus 自己的进程里，Python 侧通过 pymilvus 走网络调用，看不到也控制不了那一层的内存策略。硬说"我们用 mmap 优化了索引"是不老实的。

**项目证据**
- mmap 使用：项目中没有，不能硬套。全仓库无 mmap 调用
- `app/core/milvus_client.py:198-199` — 索引是 HNSW + COSINE，建在 Milvus 服务端，Python 侧只发建索引请求
- `app/services/vector_search_service.py` — 检索走 pymilvus 网络调用，进程内不持有索引文件

#### Q16 · `writev` 是什么？它如何通过聚合多个缓冲区减少系统调用开销？

**回答**

**项目中没有，不能硬套。** 仓库里没有 writev，也没有任何 iovec 相关代码。

writev 是 scatter-gather I/O：接受一个 `iovec` 数组（每项是一个 {地址, 长度} 对），一次系统调用把这些**不连续的**缓冲区按顺序写出去。

**省的到底是什么，这里最容易答偏。** 不是"拷贝更快"——数据该拷多少还是多少。省的是**系统调用次数**：N 个缓冲区从 N 次 `write` 压成 1 次 writev，省下 N-1 次用户态/内核态切换。在现代 CPU 上一次系统调用大约几百纳秒到一微秒，单看不多，但在高频小包场景下是主要开销。

另一个常被忽略的收益是**避免用户态的预拼接**。协议编码时 header 和 body 通常在两块内存里，不用 writev 的话要么两次 write，要么先 `memcpy` 拼成一块再写——后者多了一次内存拷贝和一次分配。writev 让你既不多调用也不多拷贝。

**原子性。** 单次 writev 对普通文件是原子的（不会和其他写交织）；对管道，总长度不超过 `PIPE_BUF` 时原子。这是它在多写者场景下比多次 write 更安全的原因。

**和本项目的关系，我的判断是这里不该用。** SSE 流式输出是一个 chunk 一次写，看起来符合"多次小写"的模式。但瓶颈根本不在系统调用——LLM 生成一个 token 要几十毫秒，系统调用那一微秒完全被淹没。更关键的是，为了凑批把多个 chunk 攒起来一次写，会直接破坏流式的首字延迟，那是这个接口最重要的体验指标。**用 writev 的判据是"系统调用次数已经成为瓶颈"，本项目不满足这个前提，用了是负优化。** 这个判断本身比背出 writev 的签名更能说明问题。

**项目证据**
- writev / iovec：项目中没有，不能硬套
- `app/api/chat_v2.py` — SSE 逐 chunk yield，瓶颈在 LLM 生成速度而非系统调用
- `app/core/metrics.py:85-91` — `llm_call_duration_seconds` 记的是几十到几千毫秒量级，和微秒级系统调用不同数量级

#### Q17 · 海量数据中找最大的 K 个数有哪些方案？为什么通常使用容量为 K 的小根堆，具体过程和复杂度是什么？

**回答**

**三种方案先摆开。** 全排序取前 K，O(n log n) 时间，而且要求全部数据能进内存——海量场景直接出局。快速选择（BFPRT / `nth_element`），平均 O(n)，但需要随机访问全部数据、会打乱原数组、最坏 O(n²)，同样要全量在内存。容量 K 的小根堆，O(n log K) 时间、O(K) 空间，**只需单遍顺序扫描**，数据可以是流式的、可以来自磁盘、可以分布在多台机器上。

**为什么是小根堆而不是大根堆**——这是这题的考点。堆里维护的是"当前最大的 K 个"，小根堆的堆顶是这 K 个里**最小**的，也就是入选门槛。新元素来了只要和堆顶比一次：不大于门槛直接丢，大于就替换堆顶再下沉。门槛是 O(1) 拿到的。换成大根堆，堆顶是最大值，想知道门槛得扫整个堆 O(K)，每个元素都这么来一遍就废了。

**过程。** 前 K 个元素直接建堆，`heapify` 是 O(K) 不是 O(K log K)。之后每个元素与堆顶比较，需要替换时用 `heapreplace`（先换堆顶再下沉，一次调整）而不是 `heappop` + `heappush`（两次调整）。扫完堆里就是 Top-K，要有序输出再排一次 O(K log K)。

复杂度 O(K + (n-K) log K)，n 远大于 K 时约等于 O(n log K)，K 是常数时近似线性。

**海量与分布式。** 每台机器算出局部 Top-K，再把这些局部结果归并求全局 Top-K——**Top-K 是可合并的**，这是它能分而治之的根本原因。对比一下：中位数不可合并，各分片的中位数归并起来不等于全局中位数，得用别的办法。能不能分布式，取决于这个性质，不取决于算法本身多聪明。

**真实对照：本项目的 Top-K 没有用堆，用的是全排序后切片。** 检索侧 `vector_retriever.invoke(query)[:top_k]` 和 `ensemble_retriever.invoke(query)[:top_k]`，去重侧 `unique_docs[:FINAL_TOP_K]`。

**这个选择是对的，理由是 n 和 K 同阶。** 候选数 `candidate_k = max(top_k * 3, top_k, 10)`，也就是十几条，`FINAL_TOP_K = 6`。n=十几、K=6 的时候，O(n log K) 和 O(n log n) 没有实质差别，而排序切片的可读性明显更好。更根本的是，**真正的 Top-K 发生在 Milvus 里**——HNSW 索引返回的本来就是按距离排好序的前 k 个，Python 侧那个切片只是二次截断。用不用堆的判据是 n/K 的比值和数据能否全量进内存，不是"堆听起来更高级"。

**项目证据**
- `heapq` / 小根堆实现：项目中没有，不能硬套。全仓库无 heapq 导入
- `app/services/vector_search_service.py:94` — `vector_retriever.invoke(query)[:top_k]`，全排序切片
- `app/services/vector_search_service.py:101` — `ensemble_retriever.invoke(query)[:top_k]`，RRF 融合后切片
- `app/agent/rag_v2/nodes.py:231` — `unique_docs[:FINAL_TOP_K]`，去重后截断
- `app/agent/rag_v2/nodes.py:30` — `FINAL_TOP_K = 6`，K 很小
- `app/services/vector_search_service.py` — `candidate_k = max(top_k * 3, top_k, 10)`，n 只有十几，与 K 同阶
- `app/core/milvus_client.py:198-199` — HNSW 索引，真正的 Top-K 排序在 Milvus 服务端完成

#### Q18 · 手撕：将扁平 JSON 列表恢复成树形结构，或者实现二叉树的锯齿形层序遍历。

**回答**

两题都写。

**扁平列表转树。**

```python
def build_tree(records):
    """records: [{"id":..., "parent_id":..., ...}]，parent_id 为 None 表示根。"""
    index = {}
    for r in records:
        index[r["id"]] = {**r, "children": []}   # 第一遍：只建索引，不挂父子

    roots, orphans = [], []
    for r in records:                            # 第二遍：挂关系
        node = index[r["id"]]
        pid = r.get("parent_id")
        if pid is None:
            roots.append(node)
        elif pid in index:
            index[pid]["children"].append(node)
        else:
            orphans.append(node)                 # parent 不存在，收集而不是静默丢
    return roots, orphans
```

时间 O(n)，空间 O(n)。三个容易被追问的点：

**为什么必须两遍。** parent 记录可能出现在 child 后面。一遍边读边挂的话，遇到还没建出来的 parent 就只能跳过或者临时占位，逻辑立刻变复杂。两遍是最简写法，代价只是多一次 O(n) 遍历。

**孤儿节点不能静默丢。** `parent_id` 指向一个不存在的 id，通常意味着上游数据有问题。直接丢掉会让 bug 无声无息——树建出来了，但少了一整棵子树，而且没人知道。收集到 `orphans` 返回，调用方可以决定是报错还是降级。

**环怎么办。** 上面的代码遇到 A→B→A 这种环，两个节点都不会进 `roots`，也不会进 `orphans`，结果是它们从返回值里彻底消失。要检测就在挂完之后统计 `roots` 和 `orphans` 里可达的节点总数，少于 n 就说明有环。

**二叉树锯齿形层序。**

```python
from collections import deque

def zigzag(root):
    if not root:
        return []
    result, queue, left_to_right = [], deque([root]), True
    while queue:
        level = deque()
        for _ in range(len(queue)):              # 先取长度，锁定本层节点数
            node = queue.popleft()
            if left_to_right:
                level.append(node.val)
            else:
                level.appendleft(node.val)       # 反向层直接头插
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(list(level))
        left_to_right = not left_to_right
    return result
```

两个点：`for _ in range(len(queue))` 必须先把长度取出来，循环体里还在往 queue 里加节点，直接用 `while queue` 会把下一层混进来。层内用 `deque` 的 `appendleft` 而不是 `list.insert(0, x)`——后者每次都要搬移整个列表，O(n)，层宽大的时候这一处就把复杂度从 O(n) 拖成 O(n²)。

**真实对照。** 项目里有一个扁平转树的近亲结构但**没有建树**：`MarkdownHeaderTextSplitter` 产出的 chunk 列表就是扁平记录带父级标识——每个 chunk 的 metadata 里有 `h1`、`h2`，本质上就是 `parent_id`。没有还原成树，因为下游是向量化检索，需要的是**独立可检索的平坦块**，建树对这个目标没有任何帮助（YAGNI）。

什么时候会需要建树：如果要做"命中二级标题的块时把它所在一级标题的上下文一起带上"这种父级扩展，就必须有父子关系。这是项目结构上已经具备、但目前用不上的能力——`strip_headers=False` 把标题留在了正文里，某种意义上是用"每块自带标题文本"这种更廉价的方式，替代了建树。

**项目证据**
- 扁平转树 / 锯齿层序的算法实现：项目中没有，不能硬套。仓库无树结构操作代码
- `app/services/document_splitter_service.py:20-29` — 只按 h1/h2 切，chunk metadata 带层级标识，是"扁平记录 + 父级"的形态
- `app/services/document_splitter_service.py:22` — `strip_headers=False`，标题留在正文里，用自带标题替代建树
- `app/services/document_splitter_service.py:135-160` — `_merge_small_chunks` 是在扁平列表上做相邻合并，不涉及层级
- 父级上下文扩展（命中子标题时带上父标题内容）：项目中没有，不能硬套。结构上具备，功能未做

### 提前批一面 · 代码生成链路与断点恢复

> 来源：https://www.nowcoder.com/discuss/925163213870612480
> 原帖为双机位面试，编号原题只公开到第 3 题且第 3 题句子未完。按边界处理：**不补题**，第 3 题只答已公开的那半句。

#### Q1 · 介绍你的 Agent 代码生成链路全流程，长流程下如何解决上下文超限问题？

**回答**

前半问必须先划清界限：**代码生成链路项目中没有，不能硬套。** 这个仓库是 RAG 知识问答加 AIOps 告警首响诊断，没有任何代码生成、代码补全、AST 操作或编译反馈回路。我不能拿 RAG 的检索链路冒充代码生成链路——两者对上下文的需求根本不同：代码生成要的是跨文件符号定义、类型签名、调用点，是**结构化依赖**；RAG 要的是语义相近的文本片段，是**平坦相似度**。把后者说成前者，面试官问一句"你怎么解析符号依赖"就会露底。

后半问是我真做过的，而且做法比"截断"具体得多。

上下文超限在我这里不是一个"超了再处理"的兜底动作，而是**在拼 Prompt 之前就分好预算**。三个数字：知识库文档 2800 Token、会话记忆 1800 Token、其中摘要部分最多 700 Token。这三块加起来是一个固定天花板，不管对话进行到第几轮、检索回来多少篇文档，送进模型的上下文总量都不会漂。

关键在于**预算是在组装时逐项扣减的，不是事后砍**：

```python
for idx, doc in enumerate(docs, start=1):
    prefix = f"[{idx}] (源子查询: {src})\n"
    available = budget - used_tokens - _estimate_tokens(prefix)
    if available <= 0:
        break
    content = _truncate_to_token_budget(doc.page_content, available)
```

注意 `available` 把 `prefix` 的开销也算进去了。`[1] (源子查询: xxx)` 这种标注本身要占 Token，如果只按正文长度算预算，最后拼出来的字符串一定超。这类"标签也是上下文的一部分"的细节，是真写过才会踩到的。

会话记忆那侧同样处理了标题开销：

```python
summary_cost = estimate_tokens(summary_header + summary) if summary else 0
# 预留最近消息标题，确保最终渲染文本（含标签）也不会突破总预算。
remaining_budget = max(0, self._context_token_budget - summary_cost - estimate_tokens(recent_header))
```

Token 估算用的是中文加权规则：CJK 字符按 1 算，其他字符按 4 字符 1 Token 算。这是保守方向的偏差——宁可高估一点提前截断，也不要低估导致真正调用时被服务端拒。

**原帖那句追问"1M 也不够怎么办"，我的答法是：**

这个追问的陷阱在于它诱导你继续在"怎么装更多"上加码。正确方向是承认**上下文不是存储层**。1M 装不下就说明设计错了——不该往上下文里塞的东西塞进去了。真正的做法是三层分工：

- 全量原文进数据库，只做审计和回溯，永远不进模型。这一条在模块 docstring 里写得很直接："完整原始消息只保存在 PostgreSQL，作为审计记录；模型上下文仅使用压缩摘要和未压缩消息的尾部"
- 语义可检索的部分进向量库，按需召回而不是全量注入
- 上下文里只放**当前这一步推理必需的**：滚动摘要加最近 3 轮原文

补一句成本视角：即使模型支持 1M，把它填满也是不理性的。注意力是 O(n²)，Prompt 计费是按 Token 线性算的，而 Lost in the Middle 意味着填到中段的内容模型大概率读不到——花钱买了一段模型不看的上下文。

**项目证据**
- Agent 代码生成链路：项目中没有，不能硬套。无代码生成、无 AST 解析、无编译反馈回路
- `app/config.py:130` — `rag_document_context_token_budget = 2800`
- `app/config.py:133-134` — 会话上下文 1800、摘要 700
- `app/agent/rag_v2/nodes.py:613-628` — `_format_context` 逐项扣减预算，`prefix` 开销计入 `available`
- `app/agent/rag_v2/nodes.py:631-634` — `_estimate_tokens`，中文按字符、其他按 4 字符 1 Token 的保守估算
- `app/agent/rag_v2/nodes.py:637-650` — `_truncate_to_token_budget` 按字符成本累加截断
- `app/services/conversation_memory_service.py:215-225` — 标题开销预留
- `app/services/conversation_memory_service.py:1-5` — 原文留 PostgreSQL 做审计，不进模型上下文
- `app/config.py:81` — `chat_total_budget_seconds = 90.0`，时间侧的同类预算约束

#### Q2 · 长流程任务的"断点恢复"能力你是怎么做的？服务重启后如何加载未完成状态？

**回答**

这一题我按真实情况分成三块讲，因为项目里这三块的成熟度差得很远，混在一起说就变成吹了。

**RAG 图这一层：没有断点恢复，而且是有意不做的。**

`build_rag_v2_graph` 结尾是裸的 `graph.compile()`，没传 `checkpointer`。这意味着一次问答的中间状态全在内存里，进程挂了这次请求就没了。原因是这条链路的总预算是 90 秒——一次问答从改写到校验跑完就是十几秒量级，为一个十几秒的请求做状态持久化，写 checkpoint 的开销可能比重跑一遍还大。重试的语义也更干净：用户重新问一次，得到的是基于当前知识库的全新回答，而不是从半小时前的中间状态接着跑。

**Chat 这一层：有 checkpointer，但它是内存的。**

`rag_agent_service.py` 走的是 `create_agent` 加 `MemorySaver`，`self.checkpointer = MemorySaver()`。这条路径能按 `session_id` 读历史，也能删 thread，但 `MemorySaver` 顾名思义存在进程内存里——**服务重启后全部丢失**。这就是为什么后来专门做了 `ConversationMemoryService` 把消息落 PostgreSQL：真正跨重启的会话连续性靠数据库，不靠 checkpointer。这两套东西并存是演进的痕迹，不是设计的冗余。

**异步任务这一层：状态持久化做了，恢复没做完。这是真实的缺口。**

入库任务的状态是落库的：九个状态从 PENDING 到 COMPLETED/FAILED/REJECTED，worker 开工时把行改成 RUNNING。所以"未完成状态"在数据库里是**查得到的**——一条停在 RUNNING 的记录就是一个没跑完的任务。

但服务重启后没有任何东西去查它。`app/main.py` 的 lifespan 只做了一件事：连 Milvus。没有扫 RUNNING、没有 reaper、没有租约超时判定。

代码注释自己承认了这个洞：

```python
# 失败状态没写进去，job 行永远停在 RUNNING
```

所以诚实的回答是：**状态可恢复，但恢复动作没实现。** 一个 worker 执行到一半被 kill，那条记录会永久停在 RUNNING——不会被重试，也不会被标失败，就一直挂着。

**如果要补，我会怎么做（这部分是 `[假设]`，项目里没有）：**

`[假设]` 给 job 行加 `heartbeat_at`，worker 执行期间定期刷新。启动时扫 `status = RUNNING AND heartbeat_at < now() - 租约时长` 的行——心跳停了就说明持有它的 worker 已经死了，可以安全地重新入队或标失败。用心跳而不是单纯用 `started_at` 做判定，是因为长任务本来就可能跑很久，`started_at` 太老不代表它死了。

`[假设]` 恢复的粒度要看幂等性。入库任务已经有 `content_hash` 做内容级幂等，整体重跑是安全的，所以最简单的恢复就是重新入队。如果要做**阶段级**恢复（比如 PDF 抽取完了但结构化没做完，只重跑结构化），就需要把每个阶段的中间产物也落库——现在的九状态机记录了"到哪一步了"，但没有存"这一步产出了什么"，所以阶段级恢复目前做不到。

**项目证据**
- `app/agent/rag_v2/graph.py:52` — `graph.compile()`，无 `checkpointer` 参数（全仓 grep `checkpointer=` 在 rag_v2 下无结果）
- `app/services/rag_agent_service.py:107` — `self.checkpointer = MemorySaver()`，进程内存，重启即丢
- `app/services/rag_agent_service.py:133-136` — `create_agent(..., checkpointer=self.checkpointer)`
- `app/services/rag_agent_service.py:321-325,386-387` — checkpointer 的读取与 `delete_thread`
- `app/models/protocol_ingestion.py:19-35` — 九状态机，"未完成"在库里查得到
- `app/workers/index_worker.py:46`、`app/workers/protocol_pdf_worker.py:35` — worker 开工置 RUNNING
- `app/core/job_failure.py:30` — 注释自认"失败状态没写进去，job 行永远停在 RUNNING"
- `app/main.py:26-49` — lifespan 只连 Milvus，**不扫 RUNNING、无 reaper、无租约判定**
- `app/models/document.py` — `content_hash`，整体重跑安全的前提
- 心跳字段、租约超时回收、阶段级中间产物持久化：项目中没有，不能硬套。上面标 `[假设]` 的部分是设计意图而非已实现代码

#### Q3 · 多 Agent 运行机制是怎样的？如何防止并发

> **原帖到这里句子未完**（原文止于"如何防止并发"）。按公开边界处理，不猜测后半句是"并发冲突""并发失控"还是"并发写同一状态"，只回答已公开的部分。

**回答**

前半问要直说：**多 Agent 运行机制项目中没有，不能硬套。** 全仓 grep `supervisor`、`sub_agent`、`handoff`、`agent_executor` 全部无结果，`app/agent/aiops/` 目录下只剩 `__pycache__`，源文件已删。项目里是两条**单 Agent** 路径：一条 `create_agent` 的自动 ReAct 循环，一条手写的五节点 LangGraph 图。五个节点是流水线上的阶段，不是五个会互相协商的 Agent——它们没有独立的目标、没有独立的工具集，也不会互相发消息。把节点说成 Agent 是概念注水。

已公开的"如何防止并发"这半句，我按项目里真实存在的并发控制答。

**并发的来源只有一处，而且是有界的。** `_fanout_to_retrieve` 按 `sub_queries` 数量发 `Send`，而 `NUM_SUB_QUERIES = 3` 是常量。所以并行度恒等于 3，不随用户输入变化——用户问一个多复杂的问题，也只会 fan out 三条检索分支。这是最朴素但最有效的并发控制：**上界写死在常量里，不给它膨胀的机会。**

**并行分支的状态合并靠 reducer，不靠锁。** `RAGState` 用 `Annotated[List[...], operator.add]`，三条分支各自返回自己那一份，LangGraph 在 fan-in 时按 reducer 合并。所有分支都只 append 不 set，所以不存在两条分支互相覆盖的问题——这是从数据结构层面消除了竞争，而不是用锁去保护竞争。

同样的原则在 span 记录上也贯彻了，注释里点明了原因：

```python
# 注意这里只 append，不 set —— 这是并行分支下不丢数据的前提。
```

**跨执行模型的共享状态才用锁。** 熔断器是三条分支共享的，它的计数和状态跃迁必须互斥。用的是 `threading.Lock` 而不是 `asyncio.Lock`，原因写在注释里：调用点跨两种执行模型——检索走 `asyncio.to_thread` 是真线程，其他地方是协程。`asyncio.Lock` 只在单个事件循环内有效，跨线程就失灵了。

熔断器还有一个位置上的讲究：它包在 `to_thread` **外面**而不是塞进线程里。已经跳闸的时候直接返回，不占用线程池的 worker——否则熔断的意义就打了一半折扣，明知会失败还先占个线程。

**并行分支的异常隔离。** `retrieve_each_node` 内部把所有异常都吃掉，不向上抛。这是 LangGraph 的语义决定的：任一并行分支抛异常会让整个 super-step 失败，整张图崩。兜住之后语义变成"一个分支失败 = 证据少一份"，另外两条分支的结果照样能用。这也是一种并发保护——防止一个分支的失败通过并发机制放大成全局失败。

**缺的部分要说清楚：** 没有 `Semaphore` 之类的全局并发限流器，也没有请求级的排队。当前的并发上界是"单进程内 fan-out 恒定 3"加上 uvicorn 自身的连接处理能力，没有一个显式的"同时最多处理 N 个请求"的闸门。高并发下真正会先顶不住的是 DashScope 的 API 配额，而这一层现在只有熔断器在兜，没有主动限流。

**项目证据**
- 多 Agent 运行机制：项目中没有，不能硬套。`supervisor` / `sub_agent` / `handoff` / `agent_executor` 全仓 grep 无结果，`app/agent/aiops/` 只剩 `__pycache__`
- `app/agent/rag_v2/graph.py:19-22` — `_fanout_to_retrieve`，按 `sub_queries` 发 `Send`
- `app/agent/rag_v2/nodes.py:28` — `NUM_SUB_QUERIES = 3`，并行度上界写死在常量里
- `app/agent/rag_v2/state.py` — `Annotated[List[...], operator.add]` reducer，只 append 不 set
- `app/core/span_context.py` — `record_span` 注释："只 append，不 set —— 这是并行分支下不丢数据的前提"
- `app/core/circuit_breaker.py:84` — 用 `threading.Lock` 的原因：调用点跨 `to_thread`（真线程）与协程两种执行模型
- `app/agent/rag_v2/nodes.py:175-188` — 熔断器包在 `to_thread` 外侧，跳闸时不占线程池 worker
- `app/agent/rag_v2/nodes.py` — `retrieve_each_node` 吞掉所有异常，防止单分支失败放大成整图崩
- 全局并发限流（`Semaphore` / 请求排队）：项目中没有，不能硬套。上界只有"fan-out 恒定 3"加 uvicorn 自身能力

#### 原帖叙述中提到但未编号的追问

> 说明：以下四条出自原帖**正文叙述**（"手撕算法考了 LRU，他追问……后半场全是数据库和缓存八股，问得极其细致"），不在原帖"面试题（≥20道）"的编号列表里。原帖只公开了追问的**题面**，没有公开面试者的回答内容。这里如实记录题面并作答，标明它们的来源层级与编号原题不同，不做编号扩充。

**追问 A · LRU 如果要求支持泛型和过期时间呢？**

泛型在 Python 里靠 `Generic[K, V]` 声明，`table: dict[K, Node[K, V]]`，运行时不做检查，只是把类型意图交给 mypy——真正的价值在静态检查而非运行时安全。

过期时间是更有意思的一半，因为它和 LRU 的淘汰逻辑是**两套独立的失效维度**：LRU 按访问顺序淘汰，TTL 按绝对时间失效。节点上加 `expire_at`，`get` 命中后先判时间，过期就当 miss 处理并顺手删掉——这叫惰性删除，好处是零后台开销，代价是过期但没人访问的条目会一直占内存。要治这个就得加定期清扫，但清扫本身要遍历，又得权衡频率。

我在项目里做的那个缓存恰好走的是**第三条路**，值得对照：BM25 检索器缓存既不按 LRU 也不按 TTL，而是按**版本号**失效——拿 `collection.num_entities` 当版本，变了就整体重建。选版本号而不选 TTL 是因为 TTL 在这里两头不讨好：设短了语料没变也白重建，设长了语料变了还在用旧的。版本号是精确的因果关系，语料变才失效，不变就永远有效。

**追问 B · B+ 树叶子节点的存储空间大小？**

叶子节点大小等于一个数据页，InnoDB 默认 16KB。这个数字不是随便定的——它对齐的是磁盘 I/O 的最小有效单位，一次随机读的成本主要在寻道，读 4KB 和读 16KB 的时间差远小于两次寻道的差。

由此能推出一件更有用的事：一个 16KB 的叶子页能装多少行，直接决定树高。假设一行 1KB，一页装 16 行；非叶子节点存的是键加指针，假设键 8 字节加指针 6 字节，一页能存约 1170 个索引项。三层 B+ 树就是 1170 × 1170 × 16 ≈ 2000 万行。这就是"三层 B+ 树够撑千万级表"的算法——面试官问存储空间大小，考的往往是你能不能顺着推到树高和 I/O 次数。

**追问 C · 索引存在硬盘还是内存？**

存在硬盘，InnoDB 里索引和数据在同一个 `.ibd` 文件里——聚簇索引的叶子节点**就是**数据行本身，这是 InnoDB 和 MyISAM 的关键差异。

但查询时用到的部分会被加载进 Buffer Pool 缓存。所以准确的说法是：**持久化在硬盘，热数据缓存在内存**。上面那个三层树的例子里，根节点和第二层几乎必然常驻 Buffer Pool，实际磁盘 I/O 通常只有最后一层的一次——这才是 B+ 树在真实系统里快的原因，不是因为树矮，而是因为矮到上层能全部缓存住。

**追问 D · 关于这四条的诚实说明**

数据库这几条我答的是原理，**项目里没有对应的深度实践，不能硬套。** 仓库用 SQLAlchemy 声明式建模，索引是 `index=True` 加 `UniqueConstraint`，交给 ORM 生成，没有手写过 DDL、没有做过 `EXPLAIN` 调优、没有分库分表、也没有 alembic 迁移目录。

真实做过的只有两件事，都是设计层面而非调优层面：一是给查询路径上的字段加了索引，`request_id`、`node`、`is_bad_case`、`created_at` 这些都是排查故障时真会用来过滤的列；二是用 `UniqueConstraint` 做业务级唯一约束，比如 `(protocol_id, equipment_name)` 组合唯一——这是把幂等性下沉到数据库层，比在应用层查一遍再插更可靠。

**项目证据**
- `app/services/vector_search_service.py:140-163` — 版本号失效的缓存，对照 LRU 与 TTL 两种策略
- `app/models/chat_run_span.py:77,82,83,88,99` — `trace_id`/`request_id`/`node`/`started_at`/`created_at` 索引
- `app/models/chat_run_trace.py:30,35,37,46,51,54` — 含 `is_bad_case` 索引，为的是能直接筛坏案例
- `app/models/protocol_catalog.py:55-56,81-82,112-113` — 三处 `UniqueConstraint`，把幂等下沉到数据库层
- `app/models/knowledge_base.py:94` — `rq_job_id` 索引，用于按 job 反查记录
- 手写 DDL、`EXPLAIN` 调优、分库分表、alembic 迁移：项目中没有，不能硬套。无 `alembic/versions/` 目录
- B+ 树页大小、Buffer Pool 命中率的实测数据：项目中没有，不能硬套。以上是原理推导，非本项目实测

## 七、影石创新 AI Agent 一面

> 来源：https://www.nowcoder.com/discuss/919595418265059328
>
> 说明：原帖自带一份完整答案（WeThinkIn 出品）。下面**不转述原帖答案**，而是按本仓库真实代码重写。原帖答案偏通用方法论，本篇的目标是把每一条方法论落到项目里能指出行号的位置，指不出来的直接写"项目中没有"。

### 一面 · Agent 设计、容错与 Java 后端基础

#### Q1 · 简单介绍一下自己的经历，以及为什么选择 AI Agent 方向？

**回答**

我讲项目而不讲履历模板。这个仓库里有两条并存的 Agent 链路，它们的差别就是我回答"为什么选这个方向"的全部理由。

第一条是 `rag_agent_service.py`，用 LangChain 的 `create_agent` 配 `MemorySaver`，模型自己决定要不要调 `retrieve_knowledge`，循环由框架托管。写起来很快，一百多行就能跑通。

第二条是 `rag_v2`，一张手写的五节点 `StateGraph`：`rewrite` → `retrieve_each`（`Send` fan-out 成 3 个并行分支）→ `dedup` → `generate` → `validate_answer`。没有 checkpointer，没有模型自由决策的回路，每一步的输入输出都是我定的。

为什么要有第二条？因为第一条在出问题的时候我什么都看不见也管不了。模型调了几次工具、每次检索花了多久、检索空了之后它是承认不知道还是自己编——这些在 `create_agent` 的循环里都是黑盒。改成显式图之后，`instrument_node` 统一给五个节点包上计时，每个 `retrieve_each` 并行实例各记一条 span；`RetrievalStatus` 把检索结果分成 OK / EMPTY / FAILED 三态而不是"有没有返回"两态；`validate_answer` 用双阈值拦截没有证据支撑的结论。

所以我选 Agent 方向的理由很具体：模型是概率的，但线上服务要求可解释、可恢复、可归因。这两件事之间的落差就是工程量，而这部分工程量恰好是后端能力能直接迁移的——超时预算、熔断、幂等、状态机、可观测性，换个场景照样是这些。不是因为这个词新。

**项目证据**
- `app/services/rag_agent_service.py:9,133,136` — `create_agent` + `MemorySaver`，框架托管循环
- `app/agent/rag_v2/graph.py:19-22,44-51` — 手写五节点图，`Send` fan-out，`compile()` 不带 checkpointer
- `app/agent/rag_v2/graph.py:26-40` — 注释写明为什么统一在注册处包 `instrument_node`：五个节点写五份计时代码是重复，且新增节点必然有人忘记加
- `app/agent/rag_v2/state.py` — `RetrievalStatus` 三态、`DegradeReason` 按修复方向分区
- `app/agent/rag_v2/nodes.py:32-33` — `MIN_GROUNDEDNESS_SCORE` / `MIN_COVERAGE_SCORE` 双阈值
- 我的完整履历：不在仓库里，此处不编

#### Q2 · 平时开发过程中主要使用哪些 AI Coding 工具？比如 Claude Code 这类工具，通常有哪些使用习惯？

**回答**

工具本身不是重点，我说的是这个仓库里能看出来的习惯。

**先给验收命令，再让它动手。** `Makefile` 里所有常用动作都是固定目标：起服务、起 worker、停进程、跑测试。我不会让 Agent 自己拼 `uvicorn` 命令行，因为端口、`PYTHONPATH`、日志重定向每次拼都可能不一样，而一旦不一样，"它说跑起来了"就没有意义。`pyproject.toml` 里 `asyncio_mode = "auto"` 也是同一个道理——异步测试不需要每个文件手写 `@pytest.mark.asyncio`，少一处能写错的地方。

**注释写"为什么"而不是"是什么"。** 这是我对生成代码最硬的一条要求，仓库里到处是这种痕迹：`circuit_breaker.py:31` 直接写"（Redis 共享状态、事件监听器、排除异常列表）本项目一个都用不上"，说明的是为什么不引三方库；`job_failure.py:68` 写"定义成函数而不是模块级常量：`Retry` 实例携带可变状态"；`span_context.py` 写"用 perf_counter 而不是两个 datetime 相减：后者受系统时钟调整（NTP 校时）影响，可能算出负数耗时"。这些是六个月后我自己回来看还需要的信息，`# 创建熔断器` 那种注释不是。

**改动前先读。** `nodes.py` 里 `retrieve_each_node` 那段"任何异常都不向上抛"的注释，是先搞清楚 LangGraph 的 super-step 语义（任一并行分支抛异常 → 整个 super-step 失败 → 整张图崩）之后才写的兜底，不是先写 try/except 再补理由。

**最后看 diff 和失败日志，不看"完成了"。** 仓库里 35 个测试文件，`.pre-commit-config.yaml` 管格式和静态检查。

**项目证据**
- `Makefile:326,340,368,439,492,497` — 固定的起停与开发目标，验收命令不靠现场拼
- `pyproject.toml:128,142` — `[tool.pytest.ini_options]`、`asyncio_mode = "auto"`
- `.pre-commit-config.yaml` — 提交前的格式与静态检查
- `app/core/circuit_breaker.py:31` — 注释解释为什么不引三方熔断库
- `app/core/job_failure.py:68` — 注释解释为什么 `Retry` 用函数而非模块级常量
- `app/core/span_context.py` — 注释解释为什么用 `perf_counter` 而不是 datetime 相减
- `tests/` — 35 个测试文件
- Spec 文件、Agent 权限白名单配置、CI 门禁：项目中没有，不能硬套。仓库无 `.github/workflows/`

#### Q3 · 如何理解 Spec Coding 和 Harness？你认为 Harness 为什么能够提升 Agent 完成复杂任务的能力？

**回答**

**Spec Coding** 是把模糊意图先沉淀成可验证的规格。这个仓库里 spec 的实际载体是 ADR：`docs/adr/002-diagnosis-task-persistence.md` 定诊断任务怎么持久化，`docs/adr/004-sop-retrieval-and-evidence-tracing.md` 定 SOP 检索与证据溯源怎么做。`alert_diagnosis_orchestrator.py` 的模块 docstring 直接把这两个 ADR 链回去，并且画出了完整链路：`payload + source → normalize_alarm → 创建诊断任务 NEW → planning → retrieving → diagnosing → analyze → save_report → DONE`。这就是先有规格后有实现的痕迹——不是写完代码补文档。

更能说明问题的是 spec 里被显式否决的东西。`_PHASE_STATUS` 那张表只映射了两个阶段，注释解释得很清楚："只有「开始做某件事」才是新状态；「某件事做完了」不是……强行给它们发明状态，会让状态机从 5 个状态膨胀到 7 个，而多出来的两个没有任何代码需要分支判断（YAGNI）。"Spec 的价值一半在于写明不做什么。

**Harness** 是模型之外的运行时。本项目里它由这几块拼成，都能指到具体位置：

- 上下文组装与预算：`_format_context` 在 2800 token 预算内组装文档，超了就截断而不是整块丢；会话记忆另有 1800 的独立预算
- 编排与状态：`graph.py` 的五节点图，`RAGState` 用 `TypedDict(total=False)` 加 `operator.add` reducer 做 fan-in 合并
- 超时分层：单次 LLM 调用 60s，整个请求 `asyncio.timeout(90.0)` 总预算
- 失败隔离：熔断器包在 `to_thread` 外侧，分支异常在节点内兜住不外抛
- 观测：`instrument_node` + `span_scope` + `flush_spans`，落 `chat_run_spans` 和 `chat_run_traces` 两张表
- 判分：`validate_answer_node` 的 groundedness / coverage 双阈值

为什么 Harness 能提升复杂任务成功率？因为它把"一次生成"改成了"每一步都有外部证据的状态迁移"。模型每次只在当前状态上决定下一步，而下一步的结果由真实工具返回、由确定性代码校验、失败了有兜底路径。同一个 `qwen-max`，放进第一条链路（`create_agent` 自由循环）和放进 `rag_v2`，可归因程度完全不同。

**缺的部分要说清楚：** 执行沙箱、工具权限模型、checkpoint 恢复、人工接管、成本预算这五项**项目中没有，不能硬套**。`compile()` 不带 checkpointer，图崩了从头再来；`job_failure.py:30` 自己承认"失败状态没写进去，job 行永远停在 RUNNING"。所以本项目的 Harness 只覆盖了"上下文、编排、超时、隔离、观测、判分"这六项。

**项目证据**
- `docs/adr/002-diagnosis-task-persistence.md`、`docs/adr/004-sop-retrieval-and-evidence-tracing.md` — spec 的实际载体
- `app/services/alert_diagnosis_orchestrator.py:1-25` — 模块 docstring 里的完整链路与 ADR 反链
- `app/services/alert_diagnosis_orchestrator.py:42-56` — `_PHASE_STATUS` 只映射两个阶段的 YAGNI 论证
- `app/agent/rag_v2/nodes.py:613-628` — `_format_context` 的 2800 预算与逐条截断
- `app/config.py:81,130,133-135` — 总预算 90s、文档预算 2800、会话预算 1800/摘要 700/压缩阈值 2400
- `app/agent/rag_v2/graph.py:44-51` — 五节点边
- `app/agent/rag_v2/nodes.py:175-188` — 熔断器在 `to_thread` 外侧
- `app/agent/rag_v2/instrumentation.py`、`app/core/span_context.py` — 节点级 span
- `app/models/chat_run_span.py:67-99`、`app/models/chat_run_trace.py:22-54` — 观测落库
- 沙箱、工具权限、checkpoint、人工接管、成本预算：项目中没有，不能硬套

#### Q4 · 如果让你从零开始设计一个 Agent 系统，你认为需要包含哪些关键模块？

**回答**

先说前置判断：路径确定、风险低的流程别用 Agent。这个仓库两条链路正好是反例和正例——AIOps 首响那条是编排层写死的状态机（`normalize → 建任务 → retrieving → diagnosing → save_report`），因为它的步骤是固定的；`rag_v2` 那条也不是自由 Agent loop，是固定五节点，因为 RAG 问答的路径本来就确定。真正需要模型自由决策的只有第一条 `create_agent` 链路。

分层来说，八层里本项目有五层半：

**1. 接入与身份层——完全没有。** 这是最大的缺口，必须说在前面。全仓 grep `Security|OAuth2|HTTPBearer|api_key_header|verify_token|current_user|Authorization` 结果为空。没有鉴权、没有租户、没有配额。唯一相关的是 `config.py:31-38` 的 `cors_origins`，那不是鉴权。

**2. 任务与状态层——有，但只在异步任务侧完整。** `ProtocolIngestionStatus` 九个状态外置在 PostgreSQL；`DiagnosisTaskStatus` 五个状态带事件留痕。但对话链路的状态是 `RAGState` 这个进程内 `TypedDict`，请求结束就没了，没有 checkpoint。

**3. Context Builder——有。** `ConversationMemoryService.build_context` 做滚动摘要加尾部窗口，`_format_context` 组装检索文档，两者各有独立 token 预算。

**4. 模型与编排层——有一半。** 有图编排、有结构化输出、有 `temperature=0`。**没有意图路由**（全仓只有 FastAPI 的 `APIRouter`，不是意图分类器），**没有最大步数限制**（grep `recursion_limit` 为空），**没有重规划**。

**5. 工具层——有基础，缺治理。** `@tool(response_format="content_and_artifact")` 声明 schema，MCP 侧有 `retry_interceptor` 做 3 次指数退避。但没有权限、没有速率限制、没有沙箱、没有副作用确认。结果裁剪有（`_format_context` 截断）。

**6. 知识与记忆层——有。** Milvus 向量检索加 BM25 混合（RRF 融合，权重 0.7/0.3），滚动摘要做长期记忆，原文留 PostgreSQL。缺的是长期记忆的写入门控和冲突更新——摘要是覆盖式的，没有"这条事实和三轮前矛盾了怎么办"的处理。

**7. 可靠性与安全层——可靠性有，安全没有。** 校验器、熔断、降级、三态状态都在。回滚、人工审批、Prompt Injection 防护、密钥隔离——只有一个 `_mask_api_key` 保证 key 不进日志，其余都没有。

**8. 可观测与评测层——有。** span/trace 落库，四个 Prometheus 指标，`tests/eval/` 下 83 条 golden set 加 Hit@K / Recall@K / MRR。缺回归门禁。

一句话总结我的设计原则：每个概率决策之后要有外部证据，每个副作用之前要有确定性约束，每个长任务要有可恢复状态。本项目前两条做到了，第三条没做到。

**项目证据**
- 鉴权/租户/配额：项目中没有，不能硬套。auth 相关符号 grep 全空
- `app/config.py:31-38` — `cors_origins` 与 `_split_cors_origins`，唯一的接入层配置
- `app/models/protocol_ingestion.py:19-35` — 九状态机外置
- `app/models/aiops_diagnosis.py` — 诊断任务状态与事件表
- `app/services/conversation_memory_service.py:93-131` — `build_context`
- `app/agent/rag_v2/nodes.py:613-628` — 文档上下文组装
- `app/tools/knowledge_tool.py:13-14` — `@tool` 声明与 schema
- `app/agent/mcp_client.py:18-73` — `retry_interceptor` 指数退避
- `app/services/vector_search_service.py:42-43,92-101` — 混合检索权重与 RRF
- `app/core/metrics.py:85-113` — 四个指标
- `tests/eval/golden_set_expanded.jsonl`、`tests/eval/metrics.py:66-96` — 离线评测
- `app/services/vector_embedding_service.py:58-70` — `_mask_api_key`
- 意图路由、最大步数、重规划、沙箱、权限、速率限制、回滚、人工审批、注入防护、回归门禁：项目中没有，不能硬套

#### Q5 · Agent 在执行任务时，如果调用外部工具或 API 出现超时、异常等情况，一般应该如何设计容错机制？

**回答**

第一步不是重试，是**分类**。这个项目里分类是落在类型系统上的，不是靠 if-else 猜。

`AppError` 的子类各自带三个字段：`code`、`retryable`、`http_status`，另外可选 `degrade_reason`。`retryable` 是个布尔字段而不是调用方现场判断——因为"这个错该不该重试"是错误本身的属性，写在抛出点最准，让每个调用方各自判断必然会不一致。`to_app_error` 负责把原始异常（`ConnectionError`、`TimeoutError`、Milvus 的各种异常）翻译成应用异常，`wrap_llm_exception` 专门处理 LLM 侧的。翻译这一层的意义是：上层只认 `AppError`，不需要知道下面是 Milvus 还是 DashScope 还是 Redis 在报错。

分类之后的四条处置路径：

**瞬时故障 → 有上限的指数退避。** MCP 侧 `retry_interceptor` 最多 3 次，`delay * (2 ** attempt)`。关键在它失败之后的动作：**不抛，`return CallToolResult(isError=True)`**。理由是工具调用失败是模型需要知道的信息，抛出去就变成了模型看不到的框架异常，而模型看到 `isError=True` 加错误文本之后，至少能告诉用户"这个工具现在不可用"。

**永久故障 → 不重试。** RQ 的 `Retry(max=N)` 是**无条件**的，只要 job 抛异常就重试，不管该不该。所以判定层单独处理：不可重试的错误直接 `job.retries_left = 0` 把额度清零。这是我在这个项目里踩过的最实际的一个坑——参数错误重试三次只是把同一个错误记三遍。

**连续失败 → 熔断。** CLOSED → OPEN → HALF_OPEN 三态，用 `threading.Lock` 而不是 `asyncio.Lock`，因为调用点跨两种执行模型（检索走 `asyncio.to_thread` 是真线程）。熔断器包在 `to_thread` **外面**：如果放里面，熔断已经打开了还要先占一个线程池 worker 才能被拒绝，那就白占。

**全局兜底 → 总预算。** 单次 LLM 调用 60s，整个请求 `asyncio.timeout(90.0)`。90 不是 60 的整数倍是故意的——留出重写、检索、校验的时间，不让单次调用把总预算吃光。这里有个写法上的细节：`timeout if timeout is not None else config.x`，不能写 `timeout or config.x`，后者会把显式传进来的 0 悄悄替换成默认值。

**partial result 优于全失败。** `retrieve_each_node` 里任何异常都不向上抛，因为 LangGraph 的语义是"任一并行分支抛异常 → 整个 super-step 失败 → 整张图崩"。兜住之后语义变成"一个分支失败 = 证据少一份"，`dedup_node` 作为 fan-in 后第一个能看到全部分支结果的节点，据此判定 PARTIAL_RETRIEVAL / RETRIEVAL_FAILED / RETRIEVAL_EMPTY。

**缺的部分：** 写操作的幂等键、结果不确定时先查状态再决定是否重放、`Retry-After` 的尊重、jitter、备用工具切换、故障注入测试——**项目中没有，不能硬套**。幂等只做到了内容哈希和状态机同态转移这两层。

**项目证据**
- `app/core/errors.py` — `AppError` 子类携带 `code` / `retryable` / `http_status` / `degrade_reason`
- `app/core/errors.py` — `to_app_error` 与 `wrap_llm_exception` 两条翻译入口
- `app/agent/mcp_client.py:18-73` — 3 次指数退避后 `return CallToolResult(isError=True)`
- `app/core/job_failure.py:84,121` — RQ 无条件重试的说明与 `retries_left = 0`
- `app/core/circuit_breaker.py:84` — `threading.Lock` 的跨执行模型理由
- `app/agent/rag_v2/nodes.py:175-188` — 熔断器在 `to_thread` 外侧及其 worker 成本理由
- `app/core/llm_factory.py` — 单次 60s 与 `timeout if timeout is not None else` 的正确写法
- `app/config.py:81` — `chat_total_budget_seconds = 90.0`
- `app/agent/rag_v2/nodes.py` — `retrieve_each_node` 不外抛的 super-step 论证
- `app/agent/rag_v2/nodes.py:214-233` — `dedup_node` fan-in 判定三态
- 幂等键、状态查询后重放、`Retry-After`、jitter、备用工具、故障注入：项目中没有，不能硬套

#### Q6 · 是否考虑过让大模型参与错误分析和自动恢复？这种方案有什么优缺点？

**回答**

考虑过，而且项目里有一个真实的半成品，正好能说明这条路的边界在哪。

**已经做的部分：模型参与分析。** `validate_answer_node` 让模型评判刚生成的答案——打两个分（groundedness 有没有证据支撑、coverage 有没有覆盖问题），并列出 `missing` 和 `unsupported`。这就是"模型做错误分析"：它读的是自己的输出加检索到的证据，判断哪里没根据。分类完之后的动作是确定性的：低于 `MIN_GROUNDEDNESS_SCORE` 或 `MIN_COVERAGE_SCORE` 就走拼接逻辑，把"仍缺少 X"、"已拦截未证实判断 Y"接到回答后面，最多各列三条，最后固定一句"补齐这些信息后，再继续判断根因和处置步骤"。

注意这里的分工：**模型输出诊断，代码决定动作**。模型不能说"我觉得应该重新检索一遍"然后就真的重新检索。

**没有做的部分：模型驱动恢复。** 自动重规划、模型选择恢复动作并执行——**项目中没有，不能硬套**。图是固定五节点单向边，`validate_answer` 后面直接 `END`，没有回到 `rewrite` 的边。

为什么停在这里？三个具体顾虑：

**一是循环。** 如果允许 `validate_answer` 判定不合格后跳回 `rewrite`，就需要一个步数上限，而全仓 grep `recursion_limit` 为空——也就是说现在加这条边，就是加一个没有刹车的回路。总预算 90s 只能保证它不会跑太久，不能保证它不会在 90s 内空转三轮然后超时返回什么都没有。

**二是模型可能误诊。** `validate_answer` 本身也是一次 LLM 调用，它也会错。已经出现过的情况是：检索到的证据其实够，但校验模型判低分，于是给用户的答案后面挂了一句"仍缺少 XX"——这比不校验更让人困惑。

**三是失败模式的选择。** 这一点我做了明确取舍：校验功能关掉时（`enable_answer_validation=false`）走 **fail-open**，答案照常返回，降级原因记 `FEATURE_DISABLED` 而不是 `LLM_ERROR`。这个区分是有代价换来的——如果都记成 `LLM_ERROR`，值班同学看到告警会先去查模型配额和鉴权，而实际上是有人手动关了开关。降级原因必须按"修复方向"分类，不是按"发生在哪一层"分类。

**优缺点总结。** 优点是模型能处理没有错误码的长尾语义问题——"这段回答听起来像编的"是规则写不出来的判断。缺点是它自己也会错、会形成循环、会增加一次调用的延迟和成本，而且如果给它执行权限，故障半径会从"一个坏答案"扩大到"一串错误动作"。所以我的边界是：错误码分类、重试次数、deadline、熔断、幂等由代码控制；模型只输出符合 schema 的诊断。

**项目证据**
- `app/agent/rag_v2/nodes.py` — `validate_answer_node` 完整逻辑
- `app/agent/rag_v2/nodes.py:32-33` — `MIN_GROUNDEDNESS_SCORE` / `MIN_COVERAGE_SCORE`
- `app/agent/rag_v2/nodes.py:595-610` — `missing` / `unsupported` 各取前三条的拼接与固定收尾语
- `app/agent/rag_v2/nodes.py:87` — 校验 prompt
- `app/agent/rag_v2/nodes.py` — `enable_answer_validation=false` 时 fail-open 且记 `FEATURE_DISABLED` 的理由
- `app/agent/rag_v2/state.py` — `DegradeReason` 按修复方向分区
- `app/agent/rag_v2/graph.py:44-51` — `validate_answer` 直接连 `END`，无回边
- `app/config.py:81` — 90s 总预算是目前唯一的空转兜底
- 自动重规划、模型驱动的恢复执行、步数上限：项目中没有，不能硬套。`recursion_limit` grep 全空

#### Q7 · Agent 中的短期记忆和长期记忆分别有什么作用？通常会如何实现？

**回答**

先说作用的区别，再说这个项目怎么实现的。

**短期记忆解决的是"这一轮听不懂上一轮"。** 用户说"那它的阈值是多少"，"它"指什么必须靠上一轮。这类信息要求**原文保真**——代词消解不能靠摘要，摘要里写"讨论了 CPU 告警"没法还原"它"是哪个服务。

**长期记忆解决的是"聊到第三十轮时前面的都塞不进去了"。** 这类信息可以有损，需要的是稳定事实：对象名称、环境、约束、已完成事项、待确认项。

本项目的实现是**滚动摘要 + 有限最近窗口**，两者在一个统一预算下拼装。

短期就是尾部窗口，但**按轮次切不按条数切**：`_recent_message_count` 从后往前数用户消息，数到 `conversation_memory_recent_turns = 3` 为止。为什么不按条数——一轮是"一条用户消息加它后面的助手回复"，按条数截可能把用户问题留下、助手回答切掉，剩下半轮比没有更糟。

长期是滚动摘要，触发条件是**待压缩部分的 token 数**超过 `conversation_memory_compact_threshold_tokens = 2400`，不是轮数。`snapshot` 里存 `summarized_through_message_id` 当游标，已经压过的部分不会再压第二遍，新摘要是"旧摘要 + 新消息"合并出来的增量结果。

摘要 prompt 里有一条我认为最重要的约束：**"不要把模型猜测、SOP 内容或历史数值写成实时事实；发生冲突时以较新的用户消息为准。"** 因为摘要一旦把猜测写成事实，后面每一轮都会读到这个假事实，错误会自我强化——这是滚动摘要最危险的失效模式。输出格式固定四段：【明确事实】【任务进展】【约束与偏好】【待确认项】，缺的写"无"。固定格式是为了让下一次合并有稳定的结构可依。

三个工程细节值得单独说：

**原文永远留在 PostgreSQL。** 模块 docstring 写明"完整原始消息只保存在 PostgreSQL，作为审计记录；模型上下文仅使用压缩摘要和未压缩消息的尾部"。摘要是给模型省 token 的，不是删数据。

**摘要失败降级而不是报错。** `_summarize` 整个包在 try 里，异常时返回空串并记 warning，注释写"摘要是节省 Token 的优化，不可阻断主聊天；失败时退回最近消息窗口"。

**游标失效时保守返回空。** 如果 `summarized_through_message_id` 对应的消息被软删了，`_messages_after_snapshot` 返回空列表而不是全部消息——注释解释了方向选择："保守地不注入旧原文，避免重复历史"。多注入会让已经被摘要过的内容又以原文出现一遍，模型会看到重复。

**项目中没有的部分：** 跨会话的用户画像、按需向量召回历史对话、记忆的过期与删除策略——**项目中没有，不能硬套**。向量检索的目标是知识库文档，不是对话历史。

**项目证据**
- `app/services/conversation_memory_service.py:1-5` — 模块 docstring：原文留 PG 做审计，上下文只用摘要 + 尾部
- `app/services/conversation_memory_service.py:146-158` — `_recent_message_count` 按轮次而非条数
- `app/config.py:136` — `conversation_memory_recent_turns = 3`
- `app/config.py:135` — `conversation_memory_compact_threshold_tokens = 2400`
- `app/services/conversation_memory_service.py:104-126` — snapshot 游标与增量合并
- `app/services/conversation_memory_service.py:28-35` — 摘要 prompt 的四段格式与"不要把猜测写成事实"
- `app/services/conversation_memory_service.py:183-186` — 摘要失败降级为最近窗口
- `app/services/conversation_memory_service.py:143-144` — 游标失效时保守返回空列表
- `app/services/conversation_memory_service.py:48-54` — `estimate_tokens` 中文按字、其他按四字符一 token
- `app/models/conversation.py` — 原文与 snapshot 的表结构
- 跨会话画像、对话历史向量召回、记忆过期删除：项目中没有，不能硬套

#### Q8 · 在 Agent 开发过程中，上下文为什么需要进行压缩？压缩后可能导致信息缺失，应该如何解决？

**回答**

压缩的必要性在我的项目里是一道算术题，不是一个理念。三块上下文各有硬预算：知识库文档 `rag_document_context_token_budget = 2800`、会话记忆 `conversation_memory_context_token_budget = 1800`、其中摘要自身再切出 `summary_token_budget = 700`。这三个数字加起来是有意留出余量的——上面还要放 system prompt、当前问题和输出空间。

不压缩会发生什么，也是具体的：会话历史随轮数线性增长，第 20 轮的原文很容易就超过 1800。一旦超了，**挤掉的不是最旧的历史，而是排在后面的东西**——在我的 `_format_context` 实现里，文档是按顺序装的，装不下就 `break`，也就是说超预算时被丢掉的是排序最靠后的检索文档。会话历史挤爆预算的代价，是证据变少。这就是"压缩上下文"和"保住检索质量"在同一个预算池里的直接竞争关系。

**信息缺失的对抗手段，我这里有四个，从便宜到贵：**

第一，**原文不删**。模块 docstring 明确写"完整原始消息只保存在 PostgreSQL，作为审计记录"。压缩只发生在"喂给模型的那一份"，数据库里那份是完整的。所以任何时候要回查，历史都还在。这是最重要的一条——压缩造成的"缺失"只是上下文里缺失，不是数据缺失。

第二，**尾部原文永远保留**。`_recent_turns = 3`，最近三轮用户消息及其后的助手消息是原文，不进摘要。理由是模型对最近一轮的指代、修正、追问最敏感，"上一句说的那个"如果被摘要成第三人称描述，指代关系就断了。

第三，**摘要 prompt 约束保留什么**。四段结构【明确事实】【任务进展】【约束与偏好】【待确认项】不是随便定的——这四类恰好是"后面还会被用到"的信息，而闲聊、已经完成且无后续影响的动作不在其中。这是一种有方向的取舍：宁可丢掉细节，不能丢掉约束和待确认项。

第四，**截断留标记**。`_truncate_text` 裁完会补一个 `…`：

```python
return "".join(chars).rstrip() + "…"
```

一个省略号很便宜，但它让模型知道"这里被切过"，而不是把半句话当完整信息处理。同一个做法在 `_truncate_to_token_budget` 里也有。

还有一个反直觉的细节值得讲：`_take_tail_to_budget` 里，如果**单条消息自己就超预算**，代码不是丢掉它，而是裁剪后保留：

```python
if not selected and cost > token_budget:
    # 单条消息超预算时仍保留尾部，内容裁剪由渲染阶段处理。
    clipped = ConversationMessage(..., content=self._truncate_text(...))
```

因为丢掉的话，最近一轮就完全消失了，模型会答一个跟当前问题无关的东西。裁一半总比没有好。

**项目中没有的部分：** 压缩前后的信息保留率评测、摘要质量的人工抽检、被压缩内容的按需回捞（发现摘要里缺了某个细节时回数据库取原文补进上下文）——**项目中没有，不能硬套**。目前是单向压缩，压完就不回头。

**项目证据**
- `app/config.py:130,133,134` — 三块预算：文档 2800、会话 1800、摘要 700
- `app/agent/rag_v2/nodes.py:613-628` — `_format_context` 装不下就 `break`，被丢的是排序靠后的文档
- `app/services/conversation_memory_service.py:1-5` — 原文完整留 PostgreSQL
- `app/config.py:136` — `recent_turns = 3`，尾部原文不进摘要
- `app/services/conversation_memory_service.py:28-35` — 四段摘要格式，定义"保留什么"
- `app/services/conversation_memory_service.py:274-289` — `_truncate_text` 补 `…` 标记
- `app/agent/rag_v2/nodes.py:637-650` — `_truncate_to_token_budget` 同样补标记
- `app/services/conversation_memory_service.py:260-269` — 单条超预算时裁剪保留而非丢弃
- `app/services/conversation_memory_service.py:215-225` — 预留标题 token，确保含标签的最终文本也不破预算
- 信息保留率评测、摘要质量抽检、压缩内容按需回捞：项目中没有，不能硬套

#### Q9 · Prompt 优化除了添加规则约束之外，还有哪些常见的方法？你在实际使用中有哪些经验？

**回答**

我先说项目里真实做过的四类，再说没做的。

**一、把自由文本输出改成结构化输出。** 这是我认为收益最大的一类，因为它把"模型有没有听话"从主观判断变成可解析的断言。子查询生成的 prompt 要求"每行一个查询，不要编号"，然后代码按行切；答案校验的 prompt 要求输出 JSON 带 `coverage_score`、`groundedness_score` 和缺失项列表，代码直接读字段跟阈值 `MIN_GROUNDEDNESS_SCORE` / `MIN_COVERAGE_SCORE` 比。约束从 prompt 里的一句话变成了一个数值比较。

**二、在 prompt 里划定证据边界。** 生成节点的 prompt 明确要求只依据给定文档回答，检索为空时不要编造。这一条单独看是"规则约束"，但它跟第一类是配套的——光说不许编造没用，得有个校验器去查是否真的没编。prompt 划边界，校验器抓违规，两者缺一不可。

**三、给输入加来源标记。** `_format_context` 组装文档时每条前面加 `[{idx}] (源子查询: {src})`。这不是给人看的，是让模型知道这段证据是回答哪个子问题的——三个子查询并行召回，混在一起丢给模型的话，模型分不清哪段对应哪个角度。

**四、明确禁止某类错误，而不只是要求正确。** 摘要 prompt 里那句"不要把模型猜测、SOP 内容或历史数值写成实时事实；发生冲突时以较新的用户消息为准"，是我在这个项目里写过的最有效的一句 prompt。正面要求"请准确总结"没有约束力，反面点名"不要把 SOP 内容写成实时事实"才拦得住具体的失效模式。写 prompt 的经验就是这个：**禁止具体的错误行为，比要求抽象的正确行为有效**。

**五、缺失项写"无"。** 同一个 prompt 里要求"缺失项写'无'"，是为了让输出格式恒定。不写这一条，模型遇到没内容的段落会直接省略标题，下一轮合并时结构就对不上了。

**没做的部分要说清楚：** Few-shot 示例、CoT 引导、prompt 版本管理与 A/B 对比、自动化 prompt 优化（如 DSPy 那类）——**项目中没有，不能硬套**。仓库里所有 prompt 都是模块级常量，改一次就是一次代码提交，没有版本号、没有灰度、没有效果对比记录。这是个真实缺口：现在如果改了 prompt 让效果变差，我没有数据能证明是这次改动导致的。

**项目证据**
- `app/agent/rag_v2/nodes.py:36-48` — 子查询 prompt，"每行一个，不要编号"的结构化约束
- `app/agent/rag_v2/nodes.py:87+` — 答案校验 prompt，要求 JSON 输出评分与缺失项
- `app/agent/rag_v2/nodes.py:32-33` — `MIN_GROUNDEDNESS_SCORE` / `MIN_COVERAGE_SCORE`，把 prompt 约束变成数值断言
- `app/agent/rag_v2/nodes.py:50-59` — 生成 prompt，划定"只依据给定文档"的证据边界
- `app/agent/rag_v2/nodes.py:613-628` — `[{idx}] (源子查询: {src})` 来源标记
- `app/services/conversation_memory_service.py:28-35` — 摘要 prompt：禁止具体错误行为 + 缺失写"无"
- Few-shot、CoT、prompt 版本管理与 A/B、自动化 prompt 优化：项目中没有，不能硬套。prompt 全是模块级常量，无版本无灰度

#### Q10 · 当模型出现幻觉问题时，一般有哪些解决思路？

**回答**

这题在前面几篇里出现过，这里按影石这轮的语境重答一遍，重点放在"分层"而不是罗列手段。

我的项目里防幻觉是**三层**，每层拦的是不同的东西。

**第一层，喂对证据（输入侧）。** 幻觉最常见的成因不是模型爱编，是给的证据不够。所以先解决召回：一个问题拆三个子查询并行检索，向量 + BM25 双路 RRF 融合，候选 `candidate_k = max(top_k*3, top_k, 10)` 再截断。这一层的目标是让答案有据可依。

**第二层，约束生成（生成侧）。** prompt 明确要求只依据给定文档，检索为空时不要编造。文档带 `[idx]` 来源标记。这一层是"提醒",约束力有限，所以必须有第三层。

**第三层，事后校验（输出侧）。** 这是唯一有强制力的一层。`validate_answer_node` 让模型对照证据给两个分数：`groundedness_score`（答案有多少是证据支撑的）和 `coverage_score`（问题被覆盖了多少），任一低于阈值就走 `_build_insufficient_answer`，把答案换成"仍缺少 X；已拦截未证实判断 Y"的说明。**注意这里是替换答案，不是加个警告** —— 加警告用户会忽略，替换掉才拦得住。

三层之外还有一条更基础的：`RetrievalStatus` 三态（OK / EMPTY / FAILED）。检索为空和检索失败是两回事，前者是知识库真没有，后者是系统故障。区分开的意义在于——检索失败时给用户"我这里出问题了"，检索为空时给"知识库里没有这个内容"，两种情况下都不会让模型在零证据上自由发挥。

校验器本身的失效也要处理。`enable_answer_validation=false` 时走 fail-open（放行答案），降级原因记 `FEATURE_DISABLED` 而不是 `LLM_ERROR`。这个区分是为值班同学做的——记成 `LLM_ERROR` 会让人去查模型配额和鉴权，而真实原因只是开关关着。

**项目中没有的部分：** 引用级别的溯源（答案每句话标注来自哪个 chunk 的哪一段）、幻觉率的离线评测集、生成后的事实核查工具调用（去查外部权威源）——**项目中没有，不能硬套**。`tests/eval/metrics.py` 里 `faithfulness_placeholder` 和 `answer_relevance_placeholder` 两个函数是占位的，没有真实实现。

**项目证据**
- `app/agent/rag_v2/nodes.py:28-30` — `NUM_SUB_QUERIES=3`、`RETRIEVE_TOP_K=4`、`FINAL_TOP_K=6`
- `app/services/vector_search_service.py:42-43,84,92-101` — 向量 0.7 / BM25 0.3，RRF 融合，候选放大
- `app/agent/rag_v2/nodes.py:50-59` — 生成 prompt 的证据边界约束
- `app/agent/rag_v2/nodes.py:87+,32-33` — 校验 prompt 与双阈值
- `app/agent/rag_v2/nodes.py:595-610` — `_build_insufficient_answer`，替换答案而非附加警告
- `app/agent/rag_v2/state.py` — `RetrievalStatus` 三态，空与失败分开
- `app/agent/rag_v2/nodes.py` — `validate_answer_node` 的 fail-open 与 `FEATURE_DISABLED`
- `tests/eval/metrics.py:174,183` — `faithfulness_placeholder` / `answer_relevance_placeholder` 是占位函数，未实现
- 句级引用溯源、幻觉率离线评测集、外部事实核查工具：项目中没有，不能硬套

#### Q11 · 如果 Token 消耗突然增加，你会如何定位原因并进行优化？

**回答**

先说实话：**我的项目里没有 token 维度的埋点**。`app/core/metrics.py` 里只有四个指标——`llm_call_duration_seconds`（Histogram）、`degrade_total`、`retrieval_failures_total`（Counter）、`circuit_breaker_state`（Gauge），没有一个记 token。所以"token 突然涨了"这件事，我的系统**现在发现不了**。这是真实缺口，先说清楚再谈方案。

**但我能定位得比较快，因为 token 的三个来源都是有硬预算的，可以逐一排除：**

会话记忆上限 1800，知识库文档上限 2800，摘要上限 700。这三块都在代码里写死。所以如果 token 涨了，先看它是不是从这三块之外来的——比如用户输入本身变长了（我没有对用户输入做长度限制，这是另一个缺口），或者子查询数从 3 变成了别的值（`NUM_SUB_QUERIES` 是常量，改了就是代码变更，git 能查）。

**已经有的旁路观测能帮上忙。** `ChatRunSpan` 记每个节点的 `duration_ms` 和 `payload`，`ChatRunTrace` 记 `used_documents`。token 数没有，但**文档条数有**——如果 `used_documents` 里的条数或内容长度异常，那基本就是检索侧的问题。加上 `llm_call_duration_seconds` 按 `node` 分标签，哪个节点变慢通常也就是哪个节点的输入变长了，延迟能当 token 的间接指标用。

**要真正做对，缺的这一块得补上：** 在 `metrics.py` 里加 `llm_tokens_total` Counter，按 `node` 和 `direction`（prompt / completion）打标签。DashScope 的响应里有 usage 字段，`llm_factory` 那一层能拿到。加完之后就能做到"按节点看 token 分布"，这是定位的前提。**这部分是我认为该做但没做的，不是已实现的功能。**

**优化手段，项目里已经有的：**

`_format_context` 按预算装文档，装不下 `break`，且每条前面的 `[idx] (源子查询: ...)` 前缀也算进预算——这个细节很容易漏，前缀累积起来不是小数目。`_truncate_to_token_budget` 逐字符按权重扣（中文 1.0，其他 0.25），保证硬上限。会话侧 `_take_tail_to_budget` 从尾往前装，先保最近的。

`estimate_tokens` 是**中文按字符计、其他按四字符一 token**的保守估算。保守是故意的：宁可高估导致少装一点内容，也不要低估导致真实调用超限。低估的后果是 API 报错，高估的后果只是少放一段文档。

**项目证据**
- `app/core/metrics.py:85-113` — 只有四个指标，**没有 token 指标**
- `app/config.py:130,133,134` — 三块硬预算：2800 / 1800 / 700
- `app/agent/rag_v2/nodes.py:28` — `NUM_SUB_QUERIES = 3` 是常量，变更可从 git 追溯
- `app/models/chat_run_span.py:83,94,98` — span 记 `node` / `duration_ms` / `payload`
- `app/models/chat_run_trace.py:42` — trace 记 `used_documents`，可看文档条数与长度
- `app/core/metrics.py:85-91` — `llm_call_duration_seconds` 按 node 分标签，延迟作为 token 的间接信号
- `app/agent/rag_v2/nodes.py:613-628` — `_format_context` 把前缀也计入预算
- `app/agent/rag_v2/nodes.py:631-650` — 逐字符加权估算与硬截断
- `app/services/conversation_memory_service.py:48-54` — `estimate_tokens` 保守估算，宁高估不低估
- token 埋点、token 告警、用户输入长度限制、按会话的 token 成本归因：项目中没有，不能硬套

#### Q12 · 你有没有设计过 Agent Skill？一个 Skill 从设计到上线通常需要考虑哪些因素？如何判断效果好坏？

**回答**

**Skill 这个机制项目中没有，不能硬套。** 仓库里 grep `Skill` / `skill` 在 `app/` 下零命中，没有 skill 注册表、没有 skill 描述文件、没有按需加载机制。

我做过的最接近的东西是**工具**，两套：LangChain 的 `@tool` 装饰器注册的 `retrieve_knowledge`，和 MCP 协议下的两个 server（`monitor_server.py` 435 行、`cls_server.py` 470 行）。这跟 Skill 的差别我说清楚——Skill 通常指一个可插拔的能力包，含描述、触发条件、执行逻辑和示例，能被 Agent 按需发现和加载；我这里的工具是**启动时全量注册**的固定集合，没有发现和加载环节。

**基于这两套工具，我能讲的真实经验：**

**一、描述就是接口。** `retrieve_knowledge` 的 docstring 是"从知识库中检索相关信息来回答问题 / 当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。"——模型看到的就是这段字。第二句"当...时使用"是关键，它给的是**触发条件**，不是功能描述。只写功能不写触发条件，模型不知道什么时候该调。

**二、返回契约要显式声明。** `@tool(response_format="content_and_artifact")` 让工具返回 `(str, List[Document])` 二元组：字符串给模型看，Document 列表进 artifact 供后续引用。这个设计的价值是模型上下文里只有精简文本，完整文档不占 token 但仍可追溯。

**三、工具永不抛异常。** 空结果 return `("没有找到相关信息。", [])`，异常 return `(f"检索知识时发生错误: {str(e)}", [])`。理由是抛出去会中断整个 agent 循环，而 return 一个说明让模型能决定下一步——换个查询、告诉用户、还是继续。**这是我认为工具设计里最重要的一条**：工具的失败应该是模型能看见并处理的信息，不是控制流中断。

**四、MCP 侧的重试放在拦截器里。** `retry_interceptor` 三次指数退避，全失败后 `return CallToolResult(isError=True)` 而不是 raise，跟上面同一个原则。放在拦截器而不是每个工具里，是为了所有工具统一行为，新增工具不会漏掉重试。

**效果判断，我这里能做到和做不到的：**

能做到的是**工具的输入输出可追溯**——`ChatRunTrace.used_documents` 记了这次用了哪些文档，`ChatRunSpan` 按节点记耗时。做不到的是工具级的成功率、调用次数、参数正确率统计——**项目中没有，不能硬套**，`metrics.py` 里没有工具维度的指标，`tool_call` 只在 SSE 事件类型和日志里出现过。

如果要补，我会先加 `tool_call_total{tool, outcome}` Counter，把 outcome 分成 success / empty / error 三态——注意 empty 要单独算，"检索到 0 条"既不是成功也不是失败，混在任一边都会让指标失真。

**项目证据**
- Skill 机制：项目中没有，不能硬套。`app/` 下 grep `skill` 零命中，无注册表、无描述文件、无按需加载
- `app/tools/knowledge_tool.py:13` — `@tool(response_format="content_and_artifact")` 声明返回契约
- `app/tools/knowledge_tool.py:14-25` — docstring 含触发条件，是模型实际看到的描述
- `app/tools/knowledge_tool.py:30-44` — 空结果与异常都 return，永不 raise
- `app/agent/mcp_client.py:18-73` — `retry_interceptor` 三次指数退避后 `return CallToolResult(isError=True)`
- `app/agent/mcp_client.py:150` — `interceptors = [retry_interceptor]`，统一拦截而非每工具实现
- `mcp_servers/monitor_server.py`、`mcp_servers/cls_server.py` — 两个 MCP server，共 905 行
- `app/models/chat_run_trace.py:42` — `used_documents` 提供事后可追溯性
- 工具成功率 / 调用次数 / 参数正确率指标：项目中没有，不能硬套。`metrics.py` 无工具维度指标

#### Q13 · 介绍一下之前做过的相关项目，这个项目是完全自主开发，还是基于已有开源方案进行改造？

**回答**

自主开发的业务系统，但**编排和检索都建在开源框架上**，这个界线我说清楚。

**用的开源件：** LangGraph 负责图式编排（`StateGraph` + `Send` fan-out），LangChain 提供 `create_agent`、`@tool`、`EnsembleRetriever`、`BM25Retriever`、`MarkdownHeaderTextSplitter`、`RecursiveCharacterTextSplitter`，Milvus 做向量库，FastAPI 做 HTTP 层，SQLAlchemy + PostgreSQL 做持久化，RQ + Redis 做异步队列，Prometheus client 做指标，loguru 做日志，fastmcp 做 MCP server。

**自己写的部分，这些是我认为有工程价值的：**

**RAG v2 的五节点图**是自己设计的：rewrite → (Send fan-out N 路) retrieve_each → dedup → generate → validate_answer。用手写图而不是 `create_agent` 的自动循环，是因为我要在每个环节插自己的东西——校验、降级、span 记录，agent 的黑盒循环插不进去。项目里两条路径都留着：`rag_agent_service.py` 是 `create_agent` + `MemorySaver` 的版本，`rag_v2` 是手写图的版本，正好是一组对照。

**熔断器**是自己写的，注释里明确说了不用现成库的理由：现成库的"Redis 共享状态、事件监听器、排除异常列表本项目一个都用不上"，而我需要的 `threading.Lock` 跨 `to_thread` 边界这件事，反而是现成库不保证的。

**span 观测层**自己写：`record_span` 只 append 不 set（并行分支下不丢数据的前提）、用 `perf_counter` 而不是两个 datetime 相减（后者受 NTP 校时影响可能算出负数）、`flush_spans` 失败只记日志绝不抛。这三条都是踩过或想清楚了才这么写的。

**会话记忆的滚动摘要**自己写：CJK 加权的 token 估算、snapshot 游标避免重复压缩、摘要失败降级到最近窗口。LangChain 有 `ConversationSummaryMemory`，但它的 token 估算对中文不准，且失败行为不是我想要的。

**应用异常体系**自己写：`AppError` 子类带 `code` / `retryable` / `http_status` / `degrade_reason`，`to_app_error` 和 `wrap_llm_exception` 做原始异常到应用异常的转换。这一层框架不会提供，因为"哪个异常该重试、降级原因怎么归类"是业务判断。

**改造过的地方：** `EnsembleRetriever` 的权重定成 0.7 / 0.3，候选数放大到 `max(top_k*3, top_k, 10)` 再截断；BM25 加了中文分词 `preprocess_func` 和按 `num_entities` 当版本号的缓存失效；Markdown 切分只切 h1/h2 且 `strip_headers=False`。都是在开源件的接缝上做的调整。

**项目证据**
- `app/agent/rag_v2/graph.py:19-22,44-51` — 自己设计的五节点图与 `Send` fan-out
- `app/services/rag_agent_service.py:9,16,107,133-136` — `create_agent` + `MemorySaver` 的对照路径
- `app/core/circuit_breaker.py:31` — 不用现成库的明确理由
- `app/core/circuit_breaker.py:84` — `threading.Lock` 跨两种执行模型的原因
- `app/core/span_context.py` — `record_span` 只 append、`perf_counter`、`flush_spans` 不抛
- `app/services/conversation_memory_service.py:48-54,104-126,183-186` — 自写滚动摘要三个关键点
- `app/services/vector_search_service.py:42-43,84,140-163` — 对 `EnsembleRetriever` / BM25 的改造
- `app/services/document_splitter_service.py:20-29` — 切分策略调整
- `pyproject.toml` — 开源依赖清单

#### Q14 · 在项目过程中遇到过哪些比较困难的问题？你主要负责哪些部分？最后取得了什么结果？

**回答**

挑三个真正卡住过的，都能指到代码。

**一、并行分支里一个失败，整张图崩。**

`Send` fan-out 出 N 个 `retrieve_each` 实例，LangGraph 的语义是"任一并行分支抛异常 → 整个 super-step 失败 → 整张图崩"。也就是说三路检索里一路 Milvus 抖动，用户拿到的是 500，而不是两份证据的答案。

解法是在节点内部兜住所有异常，把语义从"一个分支失败 = 全崩"改成"一个分支失败 = 证据少一份"。然后在 fan-in 的 `dedup_node` 里判定整体状态：全失败 → `RETRIEVAL_FAILED`，部分失败 → `PARTIAL_RETRIEVAL`，全成功但没结果 → `RETRIEVAL_EMPTY`。`dedup` 是 fan-in 后的第一个节点，**是唯一能看到全部分支结果的位置**，判定只能放这里。

结果：单路检索故障从"请求失败"降级成"答案标注证据不全"。

**二、熔断器放在哪一侧。**

检索走 `asyncio.to_thread`，最初我把熔断判断写在线程函数里。问题是熔断打开时，这次调用**仍然占用了一个线程池 worker** 才发现"哦不该调"。线程池就那么大，故障期间每个请求都白占一个 worker，反而加剧拥塞。

改成熔断判断在 `to_thread` **外面**——熔断打开时连线程都不派。代码里留了注释说明这一点。同时熔断器内部用 `threading.Lock` 而不是 `asyncio.Lock`，因为调用点跨两种执行模型（真线程 + 协程），`asyncio.Lock` 在真线程里不成立。

**三、耗时统计出现负数。**

最初用两个 `datetime.utcnow()` 相减算节点耗时，偶发负值。原因是 NTP 校时会把系统时钟往回拨。改成 `time.perf_counter()`——单调时钟，不受校时影响。这个 bug 的隐蔽性在于它不报错，只是数据偶尔离谱，如果不看原始数据根本发现不了。

顺带做对的两个决定：`record_span` **只 append 不 set**，因为并行分支各自写 span，用 set 会互相覆盖；`flush_spans` 失败只记日志绝不抛，注释写"主流程不该因为写不进耗时明细而失败"。

**我负责的部分：** RAG v2 整条链路（图、五个节点、状态定义）、异常体系（`AppError` / `to_app_error` / `DegradeReason`）、熔断器、span 观测层、会话记忆服务、RQ 异步入库链路。

**结果，能说的和不能说的分开：**

能说的是**行为层面的确定改善**：并行分支故障不再导致整请求失败；熔断打开时不再消耗线程池 worker；耗时数据不再出现负值；答案不达标时会被替换成缺口说明而不是直接返回；节点级耗时可以按 `node` 拆开看，以前日志里三条分支交织根本对不出来。

不能说的是量化指标：**没有故障前后的成功率对比数据、没有 P99 延迟的前后对比、没有线上流量**——项目中没有，不能硬套。`tests/eval/` 下有 83 条 golden set 和 Hit@K / Recall@K / MRR 的实现，是离线检索指标，不是端到端的可靠性指标。我在面试里会明确说这是离线验证的工程项目，不是生产系统。

**项目证据**
- `app/agent/rag_v2/nodes.py` — `retrieve_each_node` docstring：super-step 失败语义与兜住后的新语义
- `app/agent/rag_v2/nodes.py:214-233` — `dedup_node` 的三态判定，fan-in 唯一可见点
- `app/agent/rag_v2/state.py` — `RetrievalStatus` / `PARTIAL_RETRIEVAL` / `RETRIEVAL_FAILED` / `RETRIEVAL_EMPTY`
- `app/agent/rag_v2/nodes.py:175-188` — 熔断器包在 `to_thread` 外侧及其原因注释
- `app/core/circuit_breaker.py:84` — `threading.Lock` 跨两种执行模型
- `app/core/span_context.py` — `perf_counter` 替代 datetime 相减的注释
- `app/core/span_context.py` — `record_span` 只 append 不 set
- `app/core/span_context.py:160` — `flush_spans` 失败只记日志
- `app/core/errors.py` — `AppError` 体系与 `to_app_error`
- `tests/eval/golden_set_expanded.jsonl`（83 条）、`tests/eval/metrics.py` — 离线检索指标
- 端到端成功率 / P99 延迟的前后对比、线上流量数据：项目中没有，不能硬套

#### Q15 · Java 线程池的核心参数有哪些？线程提交任务后的执行流程是怎样的？

**回答**

先把边界说清楚：**这个项目里没有 Java 线程池，不能硬套**。仓库是纯 Python，没有任何 `.java` 文件，也没有 `ThreadPoolExecutor` 的显式构造。下面先答八股本身，再说项目里真正对应的池化对象是什么。

`ThreadPoolExecutor` 的七个参数：`corePoolSize`（核心线程数，即使空闲也不回收）、`maximumPoolSize`（最大线程数）、`keepAliveTime` + `unit`（非核心线程的空闲存活时间）、`workQueue`（任务队列）、`threadFactory`（线程创建工厂，用来命名线程，排查问题时很关键）、`handler`（拒绝策略）。

提交任务后的流程，顺序不能记错：

1. 当前线程数 < `corePoolSize` → **直接建新线程**执行，不进队列。
2. 已达 `corePoolSize` → **任务入 `workQueue`**。
3. 队列满了 → 才继续建线程，直到 `maximumPoolSize`。
4. 队列满且线程数已达 `maximumPoolSize` → 触发 `handler` 拒绝策略。

最容易答错的是第 2 和第 3 步的顺序——是**先塞队列，队列满了才扩线程**，不是先把线程扩到 max。这个顺序直接导致一个常见陷阱：如果用无界队列（`LinkedBlockingQueue` 不传容量），第 3 步永远不会发生，`maximumPoolSize` 就成了废参数，线程数永远停在 core，任务无限堆积到 OOM。

四种拒绝策略：`AbortPolicy`（默认，抛 `RejectedExecutionException`）、`CallerRunsPolicy`（提交者自己跑，天然形成背压）、`DiscardPolicy`（静默丢弃，最危险）、`DiscardOldestPolicy`（丢队头最老的）。生产上我更倾向 `CallerRunsPolicy` 或自定义策略加告警——静默丢弃在事故里查都查不出来。

**项目里真正对应的池化对象是 SQLAlchemy 连接池。** `create_engine(config.database_url, pool_pre_ping=True)`，只显式配了一个参数：`pool_pre_ping=True`。这个参数的作用和线程池的存活检测是同类问题——连接从池里取出来时先发一个轻量探测（`SELECT 1`），确认连接还活着再交给业务。不加这个参数的典型故障是：PostgreSQL 或中间的连接跟踪表把空闲连接断开了，但池子不知道，业务拿到一个死连接，第一次查询直接报 `connection already closed`。这是"池化资源必须处理后端已失效"的通用问题，Java 的 HikariCP 用 `connectionTestQuery` / `validationTimeout` 解决同一件事。

`pool_size` 和 `max_overflow` 用的是 SQLAlchemy 默认值（5 + 10），没有显式调过——因为这个项目没有真实并发压力，调了也没有依据。这一点我会直接说，不会编一个"压测调到 20"的数字。

另外项目里唯一的线程使用是 `asyncio.to_thread`，它用的是 asyncio 的默认线程池执行器，参数也没动过。

**项目证据**
- Java 线程池：项目中没有，不能硬套。仓库无 `.java` 文件，也无显式 `ThreadPoolExecutor` 构造
- `app/core/database.py:15-18` — `create_engine` + `pool_pre_ping=True`，项目里唯一显式配置的池化参数
- `app/core/database.py:20-25` — `sessionmaker(autoflush=False, autocommit=False, expire_on_commit=False)`
- `app/core/database.py:28-33` — `get_db` 用 `try/finally` 保证连接归还，这是池化资源最基本的纪律
- `app/agent/rag_v2/nodes.py:186` — `asyncio.to_thread`，走 asyncio 默认执行器，未自定义参数
- `pool_size` / `max_overflow` 的压测调优记录：项目中没有，不能硬套，用的是默认值

#### Q16 · 如果让你重新设计一个线程池，你会重点考虑哪些问题？

**回答**

同样先标边界：**我没有重新实现过线程池，项目中没有，不能硬套**。但这个问题实际上问的是"池化资源的设计要点"，而这一点我在项目里做熔断器和 RQ 队列时踩过真实的坑，可以答得有依据。

如果真要设计，我会按下面几点排优先级：

**队列必须有界。** 这是第一优先级，理由在上一题说过——无界队列让 `maximumPoolSize` 失效，把一个"拒绝服务"的快速失败变成"内存耗尽"的慢性死亡。有界 + 明确的拒绝策略，比无界 + 祈祷要好。

**拒绝必须可观测。** 拒绝策略绝不能静默。项目里 `metrics.py` 有一条相关的取舍可以直接搬过来：熔断状态用的是 `Gauge` 而不是 `Counter`，注释写"熔断状态是一个**当前值**（现在开着还是关着），不是一个累计量。Counter 只能增，表达不了「又闭合了」"。线程池同理——队列深度、活跃线程数是 Gauge，拒绝次数是 Counter，两类要分清，混用会导致故障时算不出想要的东西。

**线程要有名字。** `threadFactory` 的唯一价值就是命名。项目里有个等价的教训：span 的 `node` 标签让"三条并行分支交织在日志里根本对不出各自耗时"变得可分辨。线程池不给线程命名，jstack 出来全是 `pool-1-thread-7`，压根不知道是哪个业务的池子。

**任务的异常不能吞。** `submit` 返回的 `Future` 如果没人 `get()`，任务里抛的异常就静默消失了。这和项目里 `flush_spans` 的取舍正好是**反向**的一对：`flush_spans` 故意吞异常，因为它是旁路观测，"主流程不该因为写不进耗时明细而失败"；但业务任务的异常必须暴露。判断标准是这件事失败了要不要有人管——旁路可以吞，主路不能吞。

**隔离不同性质的任务。** 快任务和慢任务混一个池，慢任务会把池占满，快任务全部排队。项目里对应的决定是熔断器包在 `to_thread` **外侧**而不是里侧——注释写明理由是熔断打开时不该再占用一个线程池 worker。同一个思路：宝贵的池资源不应该被注定失败的调用消耗掉。

**优雅关闭。** `shutdown()` 拒绝新任务但跑完已有的，`shutdownNow()` 中断正在跑的。项目里 RQ 的对应问题暴露得很清楚：worker 执行到一半被杀，job 会停在 RUNNING 且没有回收机制——`job_failure.py:30` 的注释自己承认"失败状态没写进去，job 行永远停在 RUNNING"。这就是关闭路径没设计好的直接后果，我会把它当成设计线程池时必须先想清楚的事，而不是事后补。

**项目证据**
- 自研线程池：项目中没有，不能硬套
- `app/core/metrics.py:108-113` — Gauge vs Counter 的取舍，"Counter 表达不了又闭合了"
- `app/core/metrics.py:56-61` — Histogram 记耗时的理由，"Gauge 只记最后一次，故障时最没用"
- `app/core/span_context.py:160` — `flush_spans` 故意吞异常（旁路可吞）
- `app/agent/rag_v2/nodes.py:175-188` — 熔断器在 `to_thread` 外侧，不让失败调用占用 worker
- `app/core/job_failure.py:30` — 关闭路径没设计好的真实后果：卡在 RUNNING 无回收
- `app/services/index_job_queue.py:11-28` — `job_timeout="30m"`，有界超时是队列纪律的一部分

#### Q17 · JVM 加载一个 class 文件的完整过程是什么？

**回答**

**项目中没有 JVM，不能硬套。** 仓库是纯 Python，没有 `.java` 文件，没有字节码类加载器。这道题我按八股答，然后说一个 Python 侧真实遇到过的同类问题。

class 加载分五个阶段，`加载 → 验证 → 准备 → 解析 → 初始化`，中间三个合称"连接"：

1. **加载（Loading）：** 通过全限定名拿到二进制字节流，转成方法区的运行时数据结构，在堆里生成一个 `Class` 对象作为访问入口。字节流来源不限于文件，可以是网络、动态代理生成、加密后解密的。
2. **验证（Verification）：** 文件格式、元数据、字节码、符号引用四步校验。目的是确保字节流不会危害 JVM 自身——手写字节码或被篡改的 class 会在这里被拦下。
3. **准备（Preparation）：** 为**类变量**（static）在方法区分配内存并设**零值**。注意 `static int a = 5` 在这一步是 0，不是 5；赋 5 是初始化阶段的事。但 `static final int a = 5` 是编译期常量，会在准备阶段直接给 5。这是最常考的细节。
4. **解析（Resolution）：** 常量池里的符号引用替换成直接引用。可以延迟到真正使用前，所以不一定紧跟准备。
5. **初始化（Initialization）：** 执行 `<clinit>()`，即 static 变量的显式赋值和 static 代码块，按源码顺序合并。JVM 保证 `<clinit>` 的线程安全——这正是"静态内部类单例"能成立的原因。

双亲委派：`Bootstrap → Extension/Platform → Application → 自定义`。加载请求先往上传，父加载器加载不了才自己动手。目的是保证核心类不被替换——自己写一个 `java.lang.String` 也加载不进来。打破委派的典型场景是 SPI（`Thread.setContextClassLoader`）和热部署容器。

**Python 侧的同类问题。** Python 的模块加载没有这五个阶段，但"加载时机"这件事在项目里真实咬过一次。`conversation_memory_service.py` 里的 `_get_summarizer` 是**函数内 import**：

```python
def _get_summarizer(self) -> SummaryModel:
    if self._summarizer is not None:
        return self._summarizer
    from app.core.llm_factory import llm_factory
    ...
```

这个 import 放在函数里而不是模块顶部，是为了避免模块级循环导入——`llm_factory` 与记忆服务互相引用时，顶部 import 会在模块初始化阶段就成环。这和 JVM 的解析阶段"可以延迟到真正使用前"是同一个思路：把符号绑定推迟到实际调用的那一刻，环就解开了。差别是 JVM 的延迟解析是规范允许的行为，Python 这里是我手动做的规避。

另一个真实细节：`app/core/job_failure.py:68` 把 `Retry` 定义成**函数返回**而不是模块级常量，注释写明理由是"`Retry` 实例携带可变状态"。模块级常量在 Python 里只初始化一次，多个 job 共享同一个实例就会互相污染——这对应 JVM 里 static 变量只在 `<clinit>` 执行一次的语义。

**项目证据**
- JVM 类加载：项目中没有，不能硬套。仓库无 `.java` 文件，无字节码加载逻辑
- `app/services/conversation_memory_service.py:188-197` — `_get_summarizer` 函数内 import，规避模块级循环导入
- `app/core/job_failure.py:68-74` — `default_job_retry` 定义成函数而非模块级常量，注释解释"`Retry` 实例携带可变状态"
- `app/agent/rag_v2/graph.py` 末尾 — `rag_v2_graph = build_rag_v2_graph()` 模块级实例化，导入即建图，这是 Python 模块只初始化一次的正向利用

#### Q18 · MySQL 中的 undo log、redo log 和 binlog 分别解决什么问题？它们之间有什么区别？

**回答**

**这个项目用的是 PostgreSQL，不是 MySQL，undo/redo/binlog 这三样在项目中没有，不能硬套。** 下面按原理答，最后说 PostgreSQL 的对应机制和项目里真正落到的点。

三者的定位完全不同，容易混是因为都叫"log"：

**undo log —— 解决原子性和 MVCC。** InnoDB 引擎层，逻辑日志，记的是"反向操作"：insert 记 delete，update 记旧值。两个用途：事务回滚时按 undo 反着执行；以及 MVCC 里，读到一行的当前版本不可见时，顺着 `roll_pointer` 沿 undo 链往回找可见版本。所以 undo 不能在事务提交时立刻删——可能还有更早的快照读需要它，得等所有可能引用它的事务都结束（purge 线程做这件事）。长事务导致 undo 膨胀就是这个原因。

**redo log —— 解决持久性和崩溃恢复。** InnoDB 引擎层，物理日志，记的是"某个页做了什么改动"。核心是 WAL：改动先顺序写 redo 再慢慢刷脏页，把随机写换成顺序写。固定大小、循环覆盖，靠 `checkpoint` 推进；宕机重启后从 checkpoint 往后重放 redo，把 Buffer Pool 里没刷盘的改动补回来。`innodb_flush_log_at_trx_commit` 控制刷盘时机，设 1 才真正保证提交即持久。

**binlog —— 解决归档和复制。** MySQL Server 层（不是引擎层，这是关键区别），逻辑日志，记的是逻辑上的数据变更（`ROW` 格式记行前后镜像，`STATEMENT` 记 SQL）。追加写不覆盖，用于主从复制和按时间点恢复。

三点核心区别：**层次**（redo/undo 在引擎层，binlog 在 Server 层）、**内容**（redo 物理页改动，undo 逻辑反向操作，binlog 逻辑变更）、**写入方式**（redo 循环覆盖，binlog 追加保留）。

**PostgreSQL 的对应。** PG 没有独立的 undo log——它用**堆内多版本**：update 直接在页内写一个新版本的元组，旧版本原地留着，靠 `xmin`/`xmax` 判可见性。所以 PG 不需要 undo 链，代价是死元组要靠 `VACUUM` 回收，表膨胀是 PG 的典型运维问题（对应 MySQL 的 undo 膨胀）。redo 的角色由 WAL 承担，机制思路一致。binlog 的角色由逻辑复制槽（logical replication slot）承担。

**项目里真正落到的点只有一处，但是真的。** `get_db` 用 `try/finally` 保证连接归还，`sessionmaker` 配了 `autocommit=False`，也就是每个请求显式管理事务边界。span 落库那段代码有一个和事务语义直接相关的细节：`flush_spans` 写失败时会 `db.rollback()`，并且 rollback 本身包在 `suppress` 里。这个顺序不是随手写的——如果一个 session 里某条语句失败，PostgreSQL 会把整个事务标记为 aborted，之后任何语句都报 `current transaction is aborted`。不 rollback 就继续用这个 session，后面所有操作都会连带失败。这就是"事务原子性在应用层的直接后果"，也是我唯一能拿真实代码谈的部分。

**项目证据**
- MySQL undo log / redo log / binlog：项目中没有，不能硬套。项目用 PostgreSQL
- `app/config.py:55-56` — `postgresql+psycopg://...`，确认是 PG 不是 MySQL
- `app/core/database.py:20-25` — `autocommit=False`，显式事务边界
- `app/core/database.py:28-33` — `get_db` 的 `try/finally` 连接归还
- `app/core/span_context.py:160+` — `flush_spans` 失败时 `db.rollback()` 且 rollback 本身 `suppress`，处理的正是事务 aborted 后 session 不可用的问题
- `innodb_flush_log_at_trx_commit` 等参数的实际调优、主从复制搭建：项目中没有，不能硬套

#### Q19 · 什么是 MySQL 两阶段提交？为什么需要这个机制？

**回答**

**项目中没有，不能硬套** —— 没有 MySQL，也没有跨资源的分布式事务。按原理答，然后说项目里存在的**同构问题**：两个存储系统之间的一致性缺口，这个是真实的。

MySQL 内部的两阶段提交解决的是 **redo log 和 binlog 这两份日志之间的一致性**。它们分属引擎层和 Server 层，是两次独立的写入。如果不做协调，单独提交任一份都会出问题：

- 只写了 redo 就宕机 → 重启后 redo 重放，数据在，但 binlog 没这条记录 → 从库和按 binlog 恢复的备份**丢了这条数据**，主从不一致。
- 只写了 binlog 就宕机 → 主库重启后这条改动没了（redo 里没有），但从库按 binlog 回放**多了一条**，同样不一致。

所以流程被拆成：`prepare` 阶段写 redo 并标记 prepare 状态 → 写 binlog → `commit` 阶段把 redo 标记为 commit。崩溃恢复的判定规则很简洁：redo 里是 commit 状态就直接提交；是 prepare 状态就去 binlog 里找对应的完整记录，**找到就提交，找不到就回滚**。binlog 的完整性成了唯一裁判，两份日志因此对齐。

关键点是：两阶段提交不是为了单库的原子性（那是 undo 的事），而是为了**两个独立持久化组件之间的一致性**。这个抽象一提取出来，就和项目里的真实问题对上了。

**项目里的同构问题：PostgreSQL 与 Milvus 之间没有两阶段提交。** 入库链路要写两个存储：文档元数据和 chunk 记录进 PG，向量进 Milvus。这两个写入没有任何跨资源事务保护。失败组合的后果是真实存在的：

- PG 写成功、Milvus 写失败 → 有元数据没向量，检索时这份文档等于不存在。
- Milvus 写成功、PG 记录状态没更新 → 有向量没有对应的管理记录，成了孤儿向量。

这个缺口我没有装作解决了。仓库里能看到的间接证据是 `tests/eval/` 下留着 `_check_orphans.py`、`_diag_orphan_vec.py`、`_diag_missing_vec.py` 三个诊断脚本——它们的存在本身就说明这两类不一致**真的发生过**，需要人工去查。正确的工程做法是 outbox 模式或者对账任务，项目里都没有做。

现有的部分缓解只有两处，且都不解决根本问题：`content_hash`（sha256）让重复入库不会建第二遍索引，这是幂等而不是一致性；三处 `UniqueConstraint` 把唯一性下沉到 PG，防的是重复写而不是跨库失配。

**项目证据**
- MySQL 两阶段提交：项目中没有，不能硬套
- `tests/eval/_check_orphans.py`、`_diag_orphan_vec.py`、`_diag_missing_vec.py` — 三个诊断脚本的存在，说明 PG/Milvus 不一致真实发生过
- `app/workers/index_worker.py` — 先写 PG 状态再写 Milvus，两次独立写入无跨资源事务
- `app/models/document.py` — `content_hash` sha256 内容级幂等（幂等 ≠ 一致性）
- `app/models/protocol_catalog.py:55-56,81-82,112-113` — 三处 `UniqueConstraint`，唯一性下沉到 PG
- outbox 模式、定时对账任务、跨资源两阶段提交：项目中没有，不能硬套。这是我知道的欠账

#### Q20 · 算法题：合并两个有序数组

**回答**

题目按 LeetCode 88 的形态：`nums1` 长度 `m + n`，后 `n` 位是空位，把 `nums2` 的 `n` 个数并进去，要求原地。

唯一的关键是**从后往前填**。从前往后会覆盖 `nums1` 里还没处理的元素，必须额外开数组；从后往前写的是尾部空位，天然安全。

```python
def merge(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
    i, j, k = m - 1, n - 1, m + n - 1
    while j >= 0:                       # 只需 j 作为循环条件
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
```

时间 O(m+n)，空间 O(1)。

两个容易写错的点：循环条件只需要 `j >= 0`——如果 `nums2` 先耗尽，`nums1` 剩下的部分已经在正确位置上了，不用管；反过来如果 `nums1` 先耗尽，`nums2` 剩下的还必须继续填，所以不能用 `i >= 0` 当条件。判断里 `i >= 0` 的短路检查必须在前，否则 `nums1[-1]` 会读到数组末尾（Python 的负索引不报错，这个 bug 特别隐蔽）。

**项目里的真实对照。** 归并的核心——多路有序输入合并成一路——在检索链路里有对应物，但**没有用归并实现**：`EnsembleRetriever` 的 RRF 融合要把向量检索和 BM25 两路结果合成一路。它没有做归并，因为两路的分数**不在同一个量纲**上：余弦相似度是 [-1,1]，BM25 是无上界的正数，直接比大小没有意义。RRF 的做法是只用**排名**不用分数，按 `1/(k+rank)` 求和重排。这正好是归并前提不成立时的替代方案——归并要求两路可直接比较，RRF 处理的是不可比较的情况。

`dedup_node` 里那段更接近归并的形态：多个并行分支的结果合并成一路，用 `seen` 集合去重后 `[:FINAL_TOP_K]` 截断。同样不是归并，因为分支间无序。

**项目证据**
- 合并有序数组的实现：项目中没有，不能硬套
- `app/services/vector_search_service.py:42-43` — `VECTOR_WEIGHT = 0.7` / `BM25_WEIGHT = 0.3`
- `app/services/vector_search_service.py:92-101` — `EnsembleRetriever` RRF 融合，用排名不用分数的原因
- `app/agent/rag_v2/nodes.py:214-233` — `dedup_node` 多分支合并 + `seen` 去重 + `[:FINAL_TOP_K]` 截断

#### Q21 · 反问环节

**回答**

我会问三个，都是基于这次面试里聊到的内容往下追，不是背好的清单。

**第一个，关于评测闭环。** 今天聊了不少可靠性设计，但我自己项目最大的短板是**只有离线检索指标，没有端到端的成功率数据**——83 条 golden set 能算 Hit@K / Recall@K / MRR，但答案质量的验证只有 groundedness / coverage 两个 LLM 打的分，没有人工标注基线。我想知道团队线上的 bad case 是怎么转成评测样本的？是人工标注还是有自动化的采样规则？我在项目里为这件事留了字段——`ChatRunTrace` 有 `is_bad_case` 且加了索引，就是打算按这个筛样本，但筛出来之后的标注流程我没做过。

**第二个，关于跨存储一致性。** 刚才那道两阶段提交的题让我想到自己项目里没解决的问题：PG 和 Milvus 之间没有跨资源事务，写一半失败会留下孤儿向量或者缺向量的元数据，我现在只有三个诊断脚本靠人工查。想问团队在向量库和关系库之间是怎么做对账的？是 outbox、定时对账，还是接受不一致然后靠重建兜底？

**第三个，关于 Harness 的边界。** 今天问到 Spec Coding 和 Harness，我的项目在这条线上是不完整的——图 `compile()` 没传 checkpointer，长任务断点恢复、人工接管、权限沙箱这三块都是空的。我想知道在你们的实际场景里，这三块的优先级是怎么排的？是先做 checkpoint 保证长任务能续，还是先做权限层保证副作用可控？

不问的东西也说一下：薪资、加班这些留给 HR；官网能查到的业务介绍不问。听完回答我会挑一条接着追，比单方向念问题更能确认双方是不是在同一个理解上。

**项目证据**
- `app/models/chat_run_trace.py:46` — `is_bad_case` 带索引，为 bad case 筛选留的字段
- `tests/eval/golden_set_expanded.jsonl`（83 条）、`tests/eval/metrics.py` — 现有的离线指标能力边界
- `app/agent/rag_v2/nodes.py:32-33` — `MIN_GROUNDEDNESS_SCORE` / `MIN_COVERAGE_SCORE`，LLM 自评而非人工标注
- `tests/eval/_check_orphans.py`、`_diag_orphan_vec.py`、`_diag_missing_vec.py` — 靠人工查的一致性诊断
- `app/agent/rag_v2/graph.py` — `compile()` 未传 checkpointer
- 人工标注流程、bad case 回流管线、跨存储对账、权限沙箱、human-in-the-loop：项目中没有，不能硬套。这也是我把它们列成反问的原因

---

## 八、阿里巴巴 AI Agent 开发岗暑期实习

来源：https://www.nowcoder.com/discuss/918182032297979904
原帖形态：单轮面试，14 题，含一道贪心手撕。原帖自带 Rocky 的通用答案，下面是按本仓库真实代码重写的版本。

### 面经 · 2026-08-02

#### Q1 · 介绍一下自己的项目，整体架构和技术栈是什么？

**回答**

一个运维知识问答服务，核心是把 `aiops-docs/` 下的运维 SOP 文档做成可检索的知识库，用户提问时检索证据再生成答案，答案要经过一道校验才返回。

分层说明，每层都是仓库里真实存在的目录：

**接入层** `app/api/`，六个路由文件：`chat.py`（v1 单 Agent）、`chat_v2.py`（v2 显式图）、`aiops.py`（告警诊断）、`protocol_pdf.py`（协议 PDF 入库）、`health.py`。这一层很薄——收参数、调 service、序列化返回。

**编排层** `app/agent/rag_v2/`，四个文件分工明确：`graph.py` 建图，`nodes.py` 五个节点实现，`state.py` 状态定义，`instrumentation.py` 节点级埋点包装。

**服务层** `app/services/`，向量检索、切片、嵌入、会话记忆、首响分析、入库队列。

**数据层** `app/models/` + `app/repositories/`，SQLAlchemy 模型加仓储。

**基础设施** `app/core/`：LLM 工厂、Milvus 客户端、熔断器、异常体系、span 上下文、请求上下文、Prometheus 指标、任务队列。

技术栈跟职责绑着说：

- Python 3.13 + FastAPI + uvicorn，异步接入
- LangGraph `StateGraph` 做图式编排，`Send` 做 fan-out
- LangChain `create_agent` 走 v1 那条对照路径
- Milvus 做向量检索，HNSW + COSINE，1024 维
- PostgreSQL 存会话原文、文档元数据、诊断任务、span 明细
- Redis + RQ 做异步入库队列
- DashScope `text-embedding-v4` 嵌入 + `qwen-max` 生成
- Prometheus 客户端暴露四个指标
- loguru 结构化日志，带 `request_id`

一次请求的状态变化：请求进来先取 `request_id` 放进 contextvar，读会话记忆得到有界上下文，进图 → `rewrite` 拆三条子查询 → `Send` 扇出三路 `retrieve_each` 各自检索并写 span → `dedup` fan-in 合并去重并判定三态 → `generate` 组装上下文生成答案 → `validate_answer` 双阈值校验，不达标就替换成缺口说明 → 出图后 `flush_spans` 把节点耗时落库。

关键取舍讲三个：v1 用 `create_agent` 自动循环，v2 手写五节点显式图，保留两条是因为 v1 让模型自己决定要不要检索，v2 强制每次都检索；Milvus 索引从 IVF_FLAT+L2 改成 HNSW+COSINE；检索从纯向量改成向量 + BM25 的 RRF 融合，权重 0.7/0.3。

我负责的模块：RAG v2 整条链路、异常体系、熔断器、span 观测层、会话记忆服务、RQ 入库链路。

**实话部分：** 这是离线验证的工程项目，**没有线上流量、没有真实用户、没有租户体系**——项目中没有，不能硬套。`tests/eval/` 下有 83 条 golden set 和 Hit@K / Recall@K / MRR，是离线检索指标。原帖答案里提到的 OpenTelemetry、消息队列做长任务异步化里的"长任务"只有文档入库这一类，不是 Agent 任务本身。

**项目证据**
- `app/api/` — 六个路由文件，接入层
- `app/agent/rag_v2/graph.py` — `build_rag_v2_graph`，五节点 + `Send` fan-out
- `app/agent/rag_v2/nodes.py:28-33` — `NUM_SUB_QUERIES=3`、`RETRIEVE_TOP_K=4`、`FINAL_TOP_K=6`、两个校验阈值
- `app/core/milvus_client.py:49,198-199` — 1024 维、COSINE + HNSW
- `app/config.py:47` — `text-embedding-v4`
- `app/config.py:46,129` — `qwen-max`
- `app/config.py:55-56` — PostgreSQL 连接串
- `app/core/task_queue.py` — Redis + RQ 两个队列
- `app/core/metrics.py:85-113` — 四个 Prometheus 指标
- `app/services/rag_agent_service.py:133-136` — v1 的 `create_agent` + `MemorySaver`
- `git log` — `ac108cd IVF_FLAT+L2 → HNSW+COSINE`，索引切换的真实提交
- 线上流量、真实用户、租户体系、OpenTelemetry：项目中没有，不能硬套

#### Q2 · 项目中 Agent 的完整流程是怎样的？

**回答**

两条路径的流程完全不同，得分开讲。

**v1：模型自己决定要不要用工具。** `rag_agent_service.py:133` 用 `create_agent(model, tools, checkpointer=MemorySaver())`。流程是 LangChain 内部的 ReAct 循环：模型看到 `retrieve_knowledge` 的工具描述，自己判断要不要调；调了就把结果塞回消息列表再问一遍；不调就直接答。工具描述就是 `knowledge_tool.py:14` 那个函数的 docstring——"从知识库中检索相关信息来回答问题 / 当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。"模型的判断依据只有这几句话。

这条路径的问题：模型可能判断"这题不用查"，然后凭参数记忆答运维问题。对知识问答场景来说这是错的——用户来问就是要文档里的说法。

**v2：不给模型选择权。** 五个节点固定顺序，`START → rewrite → (Send fan-out) retrieve_each ×3 → dedup → generate → validate_answer → END`。

`rewrite` 把原问题拆成三条子查询，LLM 输出 JSON 数组，`_parse_sub_queries` 尽力抠出来，抠不到就 `return [fallback]`——用原问题当唯一子查询，不让流程断。

`_fanout_to_retrieve` 读 `state["sub_queries"]` 返回 `[Send("retrieve_each", {"query": q}) for q in sub_queries]`。三条子查询就是三个并行节点实例。

`retrieve_each` 每个实例独立检索。这里有整条链路最要紧的一条纪律：**任何异常都不向上抛**。docstring 写明原因——LangGraph 的语义是"任一并行分支抛异常 → 整个 super-step 失败 → 整张图崩"。兜住之后语义变成"一个分支失败 = 证据少一份"。

`dedup` 是 fan-in 之后的第一个节点，也是唯一能看到全部分支结果的位置。按 `md5(page_content)` 去重，有序遍历保留首次出现的顺序，最后 `[:FINAL_TOP_K]` 截到 6 条。同时判定三态：全失败 → `RETRIEVAL_FAILED`，部分失败 → `PARTIAL_RETRIEVAL`，全成功但没结果 → `RETRIEVAL_EMPTY`。

`generate` 用 `_format_context` 在 `rag_document_context_token_budget = 2800` 的预算内组装文档，每条带 `[n] (源子查询: xxx)` 前缀，超预算就截断。

`validate_answer` 让 LLM 对答案自评 coverage 和 groundedness 两个分，双阈值都是 0.6。不达标就把答案替换成缺口说明——列出"仍缺少"和"已拦截未证实判断"，而不是把不可靠答案返回给用户。

**项目证据**
- `app/services/rag_agent_service.py:133-136` — v1 的 `create_agent`
- `app/tools/knowledge_tool.py:14` — 模型唯一的判断依据就是这个 docstring
- `app/agent/rag_v2/graph.py:19-22` — `_fanout_to_retrieve` 返回 `Send` 列表
- `app/agent/rag_v2/graph.py:44-51` — 五条边的固定顺序
- `app/agent/rag_v2/nodes.py:472-492` — `_parse_sub_queries`，失败回落到原问题
- `app/agent/rag_v2/nodes.py` — `retrieve_each_node` docstring：super-step 失败语义
- `app/agent/rag_v2/nodes.py:214-233` — `dedup_node` 去重与三态判定
- `app/agent/rag_v2/nodes.py:613-628` — `_format_context` 预算内组装
- `app/config.py:130` — `rag_document_context_token_budget = 2800`
- `app/agent/rag_v2/nodes.py:32-33` — 双阈值 0.6
- `app/agent/rag_v2/nodes.py:595-610` — 缺口说明的拼装

#### Q3 · 项目过程中针对 Agent 做过哪些优化？具体怎么调优？

**回答**

四处，都有提交记录或可验证的对照。

**一、Milvus 索引从 IVF_FLAT + L2 换成 HNSW + COSINE。** 提交 `ac108cd`。换的理由是两件事：IVF_FLAT 要调 `nlist`/`nprobe`，小规模语料上召回不稳；HNSW 在这个量级上不需要调参就有稳定召回。度量从 L2 换成 COSINE 是语义上更对——嵌入向量比的是方向不是长度。这里有个可以顺带说的点：对 L2 归一化的向量，`‖x−y‖² = 2 − 2cos(x,y)`，所以余弦、点积、欧氏在单位向量上排序完全一致，换度量本身不改变排名，改的是不用再依赖"向量已归一化"这个隐含前提。

**二、检索从纯向量改成向量 + BM25 的 RRF 融合。** `EnsembleRetriever`，权重 `VECTOR_WEIGHT = 0.7` / `BM25_WEIGHT = 0.3`。这一处是有测评结论的：多查询 + RRF 是正向的。加 BM25 的动机是运维文档里有大量专有名词和错误码，纯向量对这类精确 token 的召回不如字面匹配。

**三、CrossEncoder 精排试过，是负向的，所以没进主链路。** `tests/eval/_test_rerank.py` 留着实验代码。这一条我觉得比前两条更值得讲——它说明我做优化是看数据而不是看"业界都在用"。reranker 权重现在还躺在 `models/` 下，但主链路里没有任何调用点。

**四、并行分支的失败语义。** 最初 `retrieve_each` 里的异常会直接把整张图带崩，因为 LangGraph 的 super-step 语义是"任一分支异常则整步失败"。改成兜住所有异常、在返回值里带 `status`，语义从"一个分支失败 = 整请求失败"变成"一个分支失败 = 证据少一份"。这不是性能优化，是可用性优化，但影响面比前三条都大。

顺带做的两个小调优：切片不再按三级标题分割（"避免过度碎片化"），二次分割器的 `chunk_size` 加倍到 1600（"减少分片数"），反方向加了 `_merge_small_chunks` 把过小的块并进相邻块。

**没做成的部分：** Prompt 的 A/B 对照、嵌入模型的横向对比、`FINAL_TOP_K` 从 6 调到其他值的效果对比——**项目中没有，不能硬套**。

**项目证据**
- `git log` — `ac108cd IVF_FLAT+L2 → HNSW+COSINE`
- `app/core/milvus_client.py:198-199` — HNSW + COSINE 的当前定义
- `app/services/vector_search_service.py:42-43` — 0.7 / 0.3 权重
- `app/services/vector_search_service.py:92-101` — `EnsembleRetriever` RRF 融合与 BM25 空时降级
- `tests/eval/_test_rerank.py` — CrossEncoder 实验代码，未进主链路
- `models/` — reranker 权重存在但无调用点
- `app/agent/rag_v2/nodes.py` — `retrieve_each_node` 的异常兜底与 docstring 原因说明
- `app/services/document_splitter_service.py:20-38` — 只切 h1/h2、二次分割器加倍
- `app/services/document_splitter_service.py:135-160` — `_merge_small_chunks`
- Prompt A/B、嵌入模型横向对比、`FINAL_TOP_K` 调参对比：项目中没有，不能硬套

#### Q4 · Agent 优化有哪些常见手段？

**回答**

按"改哪一层"分，我按自己动过和没动过分开说。

**检索层，动过的：** 查询改写（拆子查询扇出）、混合检索（向量 + BM25 RRF）、索引与度量选择（HNSW + COSINE）、去重（内容 md5）、候选数放大再截断（`candidate_k = max(top_k*3, top_k, 10)` 先取宽再切窄）。

**检索层，试过但否掉的：** CrossEncoder 精排。测评是负向的，代码留在 `tests/eval/_test_rerank.py`。

**上下文层，动过的：** 文档上下文预算 2800、会话记忆预算 1800、摘要预算 700，三个预算独立，互不挤占。这个隔离是有意的——早期把它们混在一起算，结果检索文档一多就把会话记忆挤没了，模型看不到用户前面说过什么。

**生成层，动过的：** 答案校验双阈值，不达标替换成缺口说明。

**可靠性层，动过的：** 单次调用 60s 超时 + 整请求 90s 总预算的两层防线、熔断器、并行分支异常兜底、三态检索状态、RQ 重试与不可重试错误清零额度。

**观测层，动过的：** 节点级 span、四个 Prometheus 指标、`request_id` 贯穿日志。

**没动过的，面试要明说：** 模型微调（SFT / DPO）、Prompt 自动优化、工具的动态筛选、多 Agent 分工、语义缓存、步数上限与循环检测——**项目中没有，不能硬套**。

这里我想补一句判断：这些手段的收益顺序在我这个项目里不是常见排序。收益最大的是**并行分支异常兜底**——它把"整请求崩"变成"证据少一份"；第二是**三个 token 预算隔离**；反倒是大家最常提的 rerank 在我这儿是负向的。所以优化手段没有普适排序，得看自己系统的瓶颈在哪。

**项目证据**
- `app/agent/rag_v2/nodes.py:28` — `NUM_SUB_QUERIES = 3` 查询改写
- `app/services/vector_search_service.py:42-43,84,92-101` — 混合检索、候选放大、RRF
- `app/core/milvus_client.py:198-199` — 索引与度量
- `app/agent/rag_v2/nodes.py:220-231` — md5 去重
- `app/config.py:130,133,134` — 三个独立 token 预算
- `app/agent/rag_v2/nodes.py:32-33` — 双阈值校验
- `app/config.py:81` — `chat_total_budget_seconds = 90.0`
- `app/core/circuit_breaker.py` — 熔断器
- `app/core/job_failure.py:68-74,121` — RQ 重试与额度清零
- `app/core/span_context.py` — 节点级 span
- `tests/eval/_test_rerank.py` — 被数据否掉的手段
- 微调、Prompt 自动优化、工具动态筛选、多 Agent、语义缓存、步数上限：项目中没有，不能硬套

#### Q5 · 基于项目设计一个场景，如果遇到类似问题应该怎么解决？

**回答**

原帖没写具体场景，我按面试里最可能被指定的一个来答：**"用户问了一个知识库里根本没有的问题，系统怎么表现？"** 这个场景我在项目里真实处理过，链路上有四道关卡。

**第一道，检索层不装作有结果。** `retrieve_each` 检索到空时返回的是空列表加 `status=EMPTY`，不是随便返回相似度最高的几条。这一点很关键——向量检索永远会返回 top_k 条，哪怕相似度低到没有意义，所以"有结果"不等于"有相关结果"。

**第二道，`dedup` 判定三态。** 全成功但合并后为空 → `RETRIEVAL_EMPTY`。这个状态会一路传到生成节点。

**第三道，生成节点看到空证据时的行为。** 上下文里没有文档，prompt 里的约束会让模型说"知识库中没有相关信息"而不是自己编。但这一层靠 prompt，不可靠。

**第四道，校验节点兜底。** 这是真正起作用的一层。`validate_answer` 让 LLM 对答案自评 groundedness——"答案里的每一句是否都能在给定证据里找到支撑"。证据为空时任何有内容的答案都拿不到 groundedness 分，于是走替换分支：答案被换成缺口说明，列出"仍缺少：..."和"已拦截未证实判断：..."，结尾是"补齐这些信息后，再继续判断根因和处置步骤。"

用户最终看到的是"我缺哪些信息"，不是一段编造的运维建议。对运维场景来说这个方向选得对——按错误的处置步骤操作生产环境，代价比"没答上来"高得多。

**顺带说这个设计的代价。** 校验节点是一次额外的 LLM 调用，延迟大概翻倍。所以有开关 `enable_answer_validation`，关掉时走 fail-open——直接放行答案，但降级原因记成 `FEATURE_DISABLED` 而不是 `LLM_ERROR`。这个区分是特意做的：后者会把值班同学送去查模型配额和鉴权，而实际上只是开关关着。

**项目证据**
- `app/agent/rag_v2/nodes.py` — `retrieve_each_node` 空结果返回 `status=EMPTY`
- `app/agent/rag_v2/state.py` — `RetrievalStatus` 三态与 `RETRIEVAL_EMPTY`
- `app/agent/rag_v2/nodes.py:214-233` — `dedup_node` 三态判定
- `app/agent/rag_v2/nodes.py:60-86` — 生成 prompt 的约束
- `app/agent/rag_v2/nodes.py:87+` — 校验 prompt 的 groundedness 定义
- `app/agent/rag_v2/nodes.py:32-33` — 双阈值
- `app/agent/rag_v2/nodes.py:595-610` — 缺口说明拼装
- `app/agent/rag_v2/nodes.py` — `validate_answer_node` 的 fail-open 与 `FEATURE_DISABLED`
- `app/config.py` — `enable_answer_validation` 开关

#### Q6 · 语义检索是如何实现的？

**回答**

从"一句话进来"到"六条文档出去"，中间有六步。

**一、拆子查询。** 原问题先过 `rewrite` 节点，LLM 拆成三条不同角度的子查询。这一步的作用是覆盖单条 query 覆盖不到的表述——用户问"服务挂了怎么办"，拆出来可能是"服务不可用排查步骤"、"进程异常退出原因"、"健康检查失败处理"。

**二、每条子查询独立嵌入。** DashScope `text-embedding-v4`，1024 维。注意这里有个服务端限制：单次最多 10 条，所以 `vector_embedding_service.py` 内部按 `MAX_BATCH_SIZE = 10` 分批，构造时还做了校验，传入超过 10 直接报错而不是等调用时才失败。

**三、Milvus ANN 检索。** HNSW 索引，COSINE 度量。HNSW 是分层可导航小世界图——上层稀疏图快速定位大致区域，逐层下降到底层稠密图精确搜索。查的不是全部向量，是图上的一条路径，所以是近似最近邻而不是精确最近邻。

**四、BM25 字面检索并行做。** `_get_bm25_retriever` 从 Milvus 全量拉文档建 BM25 索引，中文用 `_tokenize_for_bm25` 分词。这个检索器有缓存，缓存键是 `collection.num_entities`——语料条数变了就整体重建。为什么不做逐条更新：BM25 的 IDF 是全局统计量，加一条文档理论上会改变所有词的 IDF，局部更新做不对。

**五、RRF 融合。** `EnsembleRetriever` 权重 0.7 / 0.3。倒数排名融合的好处是不需要两路分数可比——向量的余弦相似度和 BM25 的打分量纲完全不同，直接加权求和是错的，RRF 只用排名位次。候选数是 `candidate_k = max(top_k*3, top_k, 10)`，先取宽再截窄，给融合留出重排空间。

**六、去重截断。** 三条子查询的结果汇到 `dedup`，按 `md5(page_content)` 去重，有序遍历保首次出现，截到 `FINAL_TOP_K = 6`。

一个降级细节：BM25 建不起来（语料为空、分词失败）时直接退回纯向量检索 `vector_retriever.invoke(query)[:top_k]`，不让混合检索的故障影响基础能力。

**项目证据**
- `app/agent/rag_v2/nodes.py:36-49` — 子查询改写 prompt
- `app/services/vector_embedding_service.py:22,27-28,44-46` — `MAX_BATCH_SIZE = 10`、`text-embedding-v4`、1024 维、构造时校验
- `app/services/vector_embedding_service.py:76,90-91` — 内部分批调用
- `app/core/milvus_client.py:198-199` — HNSW + COSINE
- `app/services/vector_search_service.py:140-163` — `_get_bm25_retriever` 与 `num_entities` 版本号缓存
- `app/services/vector_search_service.py:42-43` — 0.7 / 0.3
- `app/services/vector_search_service.py:84` — `candidate_k` 计算
- `app/services/vector_search_service.py:92-101` — RRF 融合与 BM25 空降级
- `app/agent/rag_v2/nodes.py:220-231` — md5 去重
- `app/agent/rag_v2/nodes.py:30` — `FINAL_TOP_K = 6`

#### Q7 · 向量数据库在项目中具体承担什么作用？

**回答**

Milvus 在这个项目里承担两件事，其中第二件是意外的。

**第一件，ANN 检索。** 存 1024 维嵌入向量，建 HNSW 索引，按 COSINE 度量查 top_k。集合定义在 `milvus_client.py`，维度 1024 写死在 `:49`，索引参数在 `:198-199`。

**第二件，BM25 的语料来源。** 这一点常被忽略但在我这儿是真的：`_get_bm25_retriever` 里的文档不是从 PostgreSQL 拉的，是从 Milvus 全量拉的。因为 Milvus 里存了 `page_content` 字段，本身就是完整的切片正文。这带来一个副作用——Milvus 事实上是切片正文的一份权威副本，PostgreSQL 存的是文档级元数据（文件名、`content_hash`、入库状态），不是切片正文。

这个分工有个已知代价，我在测评脚本里专门查过：两边可能不一致。`tests/eval/_check_orphans.py`、`_diag_orphan_vec.py`、`_diag_missing_vec.py` 三个诊断脚本就是为这个写的——查有向量没元数据的孤儿、有元数据没向量的缺失。**跨存储对账没有自动化，靠人工跑脚本**——项目中没有，不能硬套。

**BM25 缓存的版本号也依赖 Milvus。** `collection.num_entities` 当版本号,这是 Milvus 提供的一个便宜的一致性信号——不用自己维护版本表。代价是它只能感知条数变化，如果条数不变但内容被替换，缓存不会失效。这个场景在当前入库链路里不会出现（入库只增不改），但如果以后加了"重新索引同一份文档"的功能，这个假设就破了。

**项目证据**
- `app/core/milvus_client.py:49` — 1024 维
- `app/core/milvus_client.py:198-199` — HNSW + COSINE
- `app/services/vector_search_service.py:140-163` — BM25 语料从 Milvus 全量拉取、`num_entities` 当版本号
- `app/models/document.py` — PostgreSQL 存文档级元数据与 `content_hash`
- `tests/eval/_check_orphans.py`、`_diag_orphan_vec.py`、`_diag_missing_vec.py` — 跨存储一致性诊断脚本
- `app/main.py:26-49` — lifespan 里只连 Milvus
- 自动化跨存储对账、内容变更时的缓存失效：项目中没有，不能硬套

#### Q8 · langchain、langgraph 这些框架为什么选择？底层流程了解吗？

**回答**

这个项目里两个框架都在用，而且用在**不同的地方**，这个分界本身就是答案。

**LangChain 用在"组件"层面，不用在"编排"层面。** 具体用到的是：`langchain_core` 的 `Document`、`HumanMessage` / `SystemMessage` 这些消息类型；`@tool` 装饰器（`knowledge_tool.py:13`）；`EnsembleRetriever` 做 RRF 融合；`BM25Retriever`；`MarkdownHeaderTextSplitter` / `RecursiveCharacterTextSplitter`；`Embeddings` 基类（`vector_embedding_service.py:12` 的 `DashScopeEmbeddings` 继承它）。选它的理由很实际：这些是已经被大量项目验证过的接口约定，自己重写一遍 `Document` 和 retriever 协议没有任何收益。

**编排层面我没有用 LangChain 的 Chain / LCEL。** 仓库里搜不到 `Runnable`、搜不到 `|` 管道组合——**项目中没有 LCEL，不能硬套**。原因是 LCEL 的强项是线性和简单分支的组合，而我需要的是 `Send` 动态 fan-out 加带 reducer 的状态合并，这在 LCEL 里表达得很别扭。

**LangGraph 用在编排层。** `rag_v2/graph.py` 是全手写的 `StateGraph`：五个节点、一次条件 fan-out、四条普通边。选它的三个具体理由：

第一，`Send` 能做**运行时决定数量**的 fan-out。`_fanout_to_retrieve` 读 `state["sub_queries"]` 返回 `[Send("retrieve_each", {"query": q}) for q in sub_queries]`，几个子查询就几个并行分支。这不是编译期固定的分支数。

第二，状态合并靠 **reducer** 而不是手写归并。`RAGState` 是 `TypedDict(total=False)`，需要 fan-in 累加的字段标成 `Annotated[List[...], operator.add]`。N 个分支各返回一个单元素列表，框架自动 `+` 起来。如果自己写，就要处理并发写同一个 key 的竞争。

第三，`astream(stream_mode="values")` 能拿到**每个 super-step 之后的完整状态快照**。这是我保住 partial result 的机制：`service.py:71` 那句 `f"返回部分结果; 已有字段={sorted(last_state.keys())}"` 就是靠持续记录最后一个快照实现的——图中途崩了，手里还有崩之前那一步的完整状态。

**底层流程，说我实际踩到的语义。** LangGraph 的执行单位是 super-step（Pregel 模型的一轮）：一轮里所有被激活的节点并行跑完，再统一把 patch 合并进状态，然后进下一轮。这个模型有一条硬语义我吃过教训：**同一个 super-step 里任一分支抛异常，整个 super-step 失败，整张图崩**。所以 `retrieve_each_node` 的 docstring 明确写了它任何异常都不向上抛，兜住之后语义才变成"一个分支失败 = 证据少一份"。

`compile()` 我没传 checkpointer（`graph.py` 末尾），所以图状态是纯内存的、单请求生命周期。这是有意的：RAG 单轮问答不需要跨请求恢复。但代价要说清——**进程重启后图状态全丢，没有断点恢复**，这是 `rag_v2` 明确不具备的能力。有趣的对照是另一条路径 `rag_agent_service.py` 用了 `create_agent` + `MemorySaver`，那条路径有 checkpointer，`get_session_history` 靠 `checkpointer.get(config)` 读回历史。两条路径的取舍不同。

**项目证据**
- `app/agent/rag_v2/graph.py:19-22` — `_fanout_to_retrieve` 的 `Send` 列表
- `app/agent/rag_v2/graph.py:44-51` — 五节点注册与四条边
- `app/agent/rag_v2/graph.py` 末尾 — `compile()` 未传 checkpointer
- `app/agent/rag_v2/state.py` — `TypedDict(total=False)` + `Annotated[..., operator.add]` reducer
- `app/agent/rag_v2/service.py:71` — 部分结果日志，`stream_mode="values"` 快照的用途
- `app/agent/rag_v2/nodes.py` — `retrieve_each_node` docstring 里的 super-step 失败语义
- `app/tools/knowledge_tool.py:13` — LangChain `@tool` 装饰器
- `app/services/vector_search_service.py` — `EnsembleRetriever` / `BM25Retriever`
- `app/services/document_splitter_service.py:20-38` — 两个 LangChain splitter
- `app/services/vector_embedding_service.py:12` — 继承 LangChain `Embeddings`
- `app/services/rag_agent_service.py:133-136,325` — 另一条路径的 `create_agent` + `MemorySaver`
- LCEL / `Runnable` 管道组合：项目中没有，不能硬套
- `rag_v2` 的图级断点恢复：项目中没有，不能硬套。`compile()` 无 checkpointer 是有意的取舍

#### Q9 · 模型输出结果如何控制规则？

**回答**

这题我有三层真东西可以讲，而且第三层是很多人会忽略的。

**第一层，Prompt 里把格式写死。** 四个 prompt 常量在 `nodes.py:36/50/60/87`。校验器的 prompt 直接给出 JSON 字段清单和取值范围，摘要 prompt 给的是固定四段标题【明确事实】【任务进展】【约束与偏好】【待确认项】，并要求缺失项写"无"。要求"缺的写无"而不是"缺的省略"，是因为字段恒定存在才能让下一次合并有稳定结构可依。

**第二层，解析层假设模型一定会不听话。** 这是项目里最实在的部分——**我没有用 `with_structured_output`**，仓库里搜不到，也没有用 function calling 强制 schema。用的是手写的"尽力抠 JSON + 失败降级"，一共三处：

`_extract_json_block`（`aiops_report.py:193`）的 docstring 列了它兼容的三种情况：整段就是 JSON、被 ```json 代码块包裹、文本里嵌了一个 `{...}`。策略是先试 fenced 正则，再退化到"第一个 `{` 到最后一个 `}` 的最大区间"。

`_parse_sub_queries`（`nodes.py:472`）同构，但抠的是 `[...]` 数组，失败时 `return [fallback]`——退回原始查询，等于把多查询降级成单查询，链路不中断。

`_parse_validation_result`（`nodes.py:496`）抠 `{...}`，并且对每个字段都过一遍强转：`_coerce_score` 收紧分数、`bool(data.get(..., False))` 给缺失字段兜默认、`_coerce_str_list` 保证列表元素都是字符串。模型给了字符串 `"0.8"` 而不是数字 `0.8` 也能吃下去。

**第三层，解析失败的方向选择。** `parse_report` 的 docstring 直接写了纪律："这是「AI 输出必须可降级」纪律的落点：无论模型返回什么，调用方永远拿到一个可用的 `FirstResponseReport`，不会抛异常。" 四道防线依次是：抠不出 JSON → `_degraded`；`json.loads` 抛 → `_degraded`；解析出来不是 dict → `_degraded`；Pydantic `model_validate` 失败 → `_degraded`，注释写"结构不完整也不崩：能塞的字段尽量塞，其余降级"。

方向为什么选降级而不是报错：这里的调用方是告警首响链路，值班同学要的是"哪怕只有纯文本也先看到内容"，而不是一个 500。反过来在 `validate_answer_node` 那个位置，方向就不同了——那里校验不通过会**替换答案**为缺口说明，因为那个场景下"给出没有证据支撑的答案"比"告诉用户缺什么"更糟。同一个"控制输出"的诉求，两个位置的处置方向相反，取决于下游要什么。

**第四层是事后校验，也算规则控制。** `validate_answer_node` 用 LLM 给答案打 `coverage_score` / `groundedness_score`，双阈值都是 0.6（`nodes.py:32-33`），不达标就替换答案并记降级原因。这是"生成之后再审"，和前三层的"生成时约束"是互补的。

**项目中没有的：** `with_structured_output`、function calling 的 JSON Schema 强约束、Guardrails 之类的输出校验框架、正则白名单过滤敏感内容——**项目中没有，不能硬套**。

**项目证据**
- `app/agent/rag_v2/nodes.py:36,50,60,87` — 四个 prompt 常量，格式写死在 prompt 里
- `app/services/conversation_memory_service.py:28-35` — 摘要 prompt 的固定四段与"缺失写无"
- `app/models/aiops_report.py:193-215` — `_extract_json_block` 兼容三种输出形态
- `app/models/aiops_report.py:225-245` — `parse_report` 四道降级防线与「AI 输出必须可降级」docstring
- `app/agent/rag_v2/nodes.py:472-492` — `_parse_sub_queries`，失败 `return [fallback]` 降级为单查询
- `app/agent/rag_v2/nodes.py:496-520` — `_parse_validation_result` 的逐字段强转
- `app/agent/rag_v2/nodes.py:32-33` — `MIN_COVERAGE_SCORE` / `MIN_GROUNDEDNESS_SCORE` 双阈值 0.6
- `with_structured_output`、function calling schema 强约束、Guardrails、敏感词正则：项目中没有，不能硬套

#### Q10 · 记忆模块是怎么设计的？

**回答**

`ConversationMemoryService` 是一个完整实现，我按"存什么、给模型什么、什么时候压"三段讲。

**存什么：原文全留，留在 PostgreSQL。** 模块 docstring 第一句就是设计意图："完整原始消息只保存在 PostgreSQL，作为审计记录；模型上下文仅使用压缩摘要和未压缩消息的尾部，避免输入 Token 随对话轮数无限增长。" 摘要是为了省 token，不是为了删数据——这两件事必须分开。

**给模型什么：固定预算的两段文本。** `_render_context` 组装出【会话滚动摘要】+【最近会话原文】两段，总预算 `conversation_memory_context_token_budget = 1800`，其中摘要单独限 700。这里有个细节值得说：预算计算把**标题本身的 token 也算进去了**——`summary_budget` 减掉了 `estimate_tokens(summary_header)`，`remaining_budget` 又减掉了 `estimate_tokens(recent_header)`，注释写"预留最近消息标题，确保最终渲染文本（含标签）也不会突破总预算"。最后 `text = self._truncate_text(text, self._context_token_budget)` 还兜一次硬上限。三重保险是因为角色前缀、分隔符这些"结构性字符"也是真实 token，很容易在估算里漏掉。

**什么时候压：按 token 不按轮数。** `build_context` 的流程是：读全部活跃消息 → 读 snapshot 拿游标 → 算出游标之后的 `pending_messages` → 按轮次留尾部（`recent_turns = 3`）→ 剩下的算 token，超过 `compact_threshold_tokens = 2400` 就触发摘要。

按 token 而不按轮数，是因为轮数和上下文压力不是线性关系——三轮长对话可能比十轮"好的/收到"更占预算。

**尾部窗口按轮次而不按条数。** `_recent_message_count` 从后往前数用户消息，数到第 3 条用户消息就停，返回从那里到末尾的**全部**消息。这样切点永远落在用户消息前面，不会出现"只留了助手回复但问题被切掉"的断裂上下文。

**增量而非重算。** snapshot 存 `summarized_through_message_id` 做游标，新摘要的输入是"旧摘要 + 游标之后的新消息"，已压过的部分不再压第二遍。这既省 token 也保证摘要是收敛的。

**三处降级方向，每处都有注释说明选择理由：**

摘要 LLM 调用失败 → 返回空串，退回纯最近窗口，注释："摘要是节省 Token 的优化，不可阻断主聊天"。

游标指向的消息被软删了 → `_messages_after_snapshot` 返回**空列表**而不是全部消息，注释："保守地不注入旧原文，避免重复历史"。方向选"少注入"，因为多注入会让已被摘要的内容以原文再出现一遍。

单条消息就超预算 → 不丢弃，构造一个内容被裁剪的 `ConversationMessage` 副本保留，注释："单条消息超预算时仍保留尾部"。因为最新那条通常就是当前问题，丢了等于没上下文。

**Token 估算是自己写的，中文加权。** `estimate_tokens` 里中文字符算 1，其他按 4 字符 1 token 向上取整。用真 tokenizer 会引入依赖和额外耗时，而这里只需要"保守不低估"——低估会突破真实窗口，高估只是少注入一点。方向选对了就够用。

**项目中没有的：** 跨会话的用户长期画像、把历史对话向量化后按需召回、记忆的过期与主动删除策略——**项目中没有，不能硬套**。向量检索的目标是知识库文档，不是对话历史。

**项目证据**
- `app/services/conversation_memory_service.py:1-5` — 模块 docstring 的设计意图
- `app/services/conversation_memory_service.py:93-131` — `build_context` 主流程
- `app/services/conversation_memory_service.py:146-158` — `_recent_message_count` 按轮次数
- `app/services/conversation_memory_service.py:207-247` — `_render_context` 的三重预算控制
- `app/services/conversation_memory_service.py:249-272` — `_take_tail_to_budget` 单条超预算仍保留
- `app/services/conversation_memory_service.py:48-54` — `estimate_tokens` 中文加权
- `app/services/conversation_memory_service.py:104-126` — snapshot 游标与增量摘要
- `app/services/conversation_memory_service.py:143-144` — 游标失效返回空列表
- `app/services/conversation_memory_service.py:183-186` — 摘要失败降级
- `app/services/conversation_memory_service.py:28-35` — 摘要 prompt 的四段格式
- `app/config.py:133-136` — 四个预算参数：1800 / 700 / 2400 / 3
- 跨会话画像、对话历史向量召回、记忆过期删除：项目中没有，不能硬套

#### Q11 · 多轮、多会话场景下 memory 如何处理？

**回答**

先把"多轮"和"多会话"分开，因为项目里这两件事的成熟度完全不同。

**多轮，是做完的。** 隔离键是 `session_id`。`ConversationRepository.list_active_messages(session_id)` 按会话拉消息，`get_memory_snapshot(session_id)` / `save_memory_snapshot(session_id, ...)` 也都以 `session_id` 为键。也就是说每个会话有**自己独立的一份滚动摘要和游标**，不共享。数据库层面 `conversation.py` 里 `session_id` 相关字段带索引（`:53`、`:76`、`:107`、`:112`），`created_at` 也有索引（`:116`）——按会话取消息并按时间排序是最热的查询路径。

另一条路径 `rag_agent_service.py` 的隔离方式不同：它用 LangGraph `MemorySaver`，`session_id` 当 `thread_id`，`get_session_history` 靠 `checkpointer.get(config)` 按 thread 读回。两条路径的隔离粒度一致（都是会话级），但存储介质不同：一个落 PostgreSQL，一个在进程内存。

**多会话，这里要说实话，有明确缺口。**

`MemorySaver` 是**纯内存**的，进程重启后所有 thread 的 checkpoint 全丢。它也没有容量上限——会话越多内存单调增长，没有 LRU 淘汰，没有 TTL。这在 `rag_agent_service.py` 那条路径上是真实存在的问题。**内存 checkpointer 的持久化与淘汰：项目中没有，不能硬套。**

`ConversationMemoryService` 这条路径不受重启影响（数据在 PostgreSQL），但也有两个缺口：**没有跨会话的用户级记忆**——同一个用户开两个会话，第二个会话读不到第一个会话里说过的偏好；**没有会话数据的过期与清理**——消息表只增不删，只有软删标记。

**并发同一会话，也是缺口。** 如果同一个 `session_id` 有两个请求同时进来，两边都会读到同一个 snapshot、各自算出要压缩的区间、各自调 LLM 摘要、各自 `save_memory_snapshot`。后写的覆盖先写的，中间那次摘要的 LLM 调用白花了。**会话级并发锁或乐观版本号：项目中没有，不能硬套。** 当前链路是单用户调试场景，这个竞态没有暴露，但它是真实存在的。

**如果要补，我会怎么做（这部分是设想，标清楚）。** `[假设]` 把 `MemorySaver` 换成持久化 checkpointer；给 snapshot 表加 `version` 字段做乐观锁，`save` 时带上读到的版本号，冲突则重试；用户级记忆单独一张表，按 `user_id` 存稳定偏好，写入要过门控（不是每句话都进长期记忆）。这些都不在当前代码里。

**项目证据**
- `app/repositories/conversation_repository.py` — `list_active_messages` / `get_memory_snapshot` / `save_memory_snapshot` 均以 `session_id` 为键
- `app/models/conversation.py:53,76,107,112,116` — `session_id` 与 `created_at` 索引
- `app/services/conversation_memory_service.py:99-104` — 按会话读消息与 snapshot
- `app/services/rag_agent_service.py:107,133-136` — `MemorySaver` 作为另一条路径的会话存储
- `app/services/rag_agent_service.py:310-325` — `checkpointer.get(config)` 按 thread 读历史
- `app/services/rag_agent_service.py:377-387` — `delete_thread` 清会话
- `MemorySaver` 的持久化与容量淘汰：项目中没有，不能硬套。进程重启即全丢，无 TTL 无 LRU
- 跨会话用户级记忆、会话数据过期清理：项目中没有，不能硬套
- 同一会话的并发写保护（锁或乐观版本号）：项目中没有，不能硬套。后写覆盖先写是真实存在的竞态

#### Q12 · 如果系统出现异常，整体的容错和异常处理机制怎么设计？

**回答**

这题是我项目里花时间最多的部分，分五层讲，每层都有代码。

**第一层，异常分类。** `AppError` 体系里每个子类携带四个属性：`code`（机器可读的错误码）、`retryable`（能不能重试）、`http_status`（映射到什么 HTTP 状态）、`degrade_reason`（降级原因）。分类不是为了好看，是因为**不同类型的处置方式不同**——`retryable=False` 的错误重试一百次也没用，反而浪费预算。

`to_app_error` 负责把原始异常转成应用异常：已经是 `AppError` 的原样返回，其他的按类型映射。`wrap_llm_exception` 是 LLM 场景的专用版本。这一层的意义是**边界收口**：底层可能抛 Milvus 的连接错误、DashScope 的 HTTP 错误、asyncio 的超时，上层只需要认识 `AppError`。

**第二层，超时的双层预算。** 单次 LLM 调用 60s（`llm_factory.py`），整个请求 90s（`chat_total_budget_seconds = 90.0`，`chat_v2.py` 用 `asyncio.timeout` 包住）。两层都要有：只有单次超时，三次调用串起来能跑 180s；只有总预算，一次卡死的调用会吃掉全部额度。

这里有个我踩过的坑值得单独讲：`llm_factory` 里写的是 `timeout if timeout is not None else config.x`，**不是** `timeout or config.x`。后者在传入 0 时会被判为 falsy 而静默替换成默认值——调用方明确要求"不等待"，结果等了 60 秒。`or` 在处理"0 和 None 语义不同"的参数时是错的。

**第三层，重试与熔断。** 重试在两个地方：MCP 工具调用有 `retry_interceptor`，3 次指数退避（`delay * 2**attempt`）；RQ 任务有 `Retry(max, interval)`。注意 RQ 的重试是**无条件**的（`job_failure.py:84` 的注释写明了），只要抛异常就重试，所以判定层对不可重试错误直接 `job.retries_left = 0` 清零额度。另外 retry 必须**入队时**声明，因为 RQ 把额度存进 job 记录，worker 只能减不能补。

熔断器 CLOSED → OPEN → HALF_OPEN 三态，用 `threading.Lock` 而不是 `asyncio.Lock`，因为调用点跨两种执行模型（`circuit_breaker.py:84` 有注释）。放置位置是个关键决定：**熔断器包在 `to_thread` 外面而不是里面**（`nodes.py:175-188`），因为如果放里面，熔断打开时还要先占一个线程池 worker 才能发现"哦我不该调用"——白占一个 worker。

**第四层，失败的方向选择——这是最需要判断力的部分。**

并行检索分支：`retrieve_each_node` **任何异常都不向上抛**。因为 LangGraph 的语义是"任一并行分支抛异常 → 整个 super-step 失败 → 整张图崩"。兜住之后语义变成"一个分支失败 = 证据少一份"。`dedup_node` 是 fan-in 后第一个节点、唯一能看到全部分支结果的位置，在那里判定 `PARTIAL_RETRIEVAL` / `RETRIEVAL_FAILED` / `RETRIEVAL_EMPTY` 三态。

工具层：`retrieve_knowledge` 空结果 `return ("没有找到相关信息。", [])`，异常 `return (f"检索知识时发生错误: {str(e)}", [])`，**都是 return 不是 raise**。因为抛出去 agent loop 就断了，返回则让模型看到"检索失败了"这个事实、自己决定下一步。MCP 的 `retry_interceptor` 同理，重试耗尽后 `return CallToolResult(isError=True)`。

答案校验器：**fail-open**。`enable_answer_validation=false` 时放行，降级原因用 `FEATURE_DISABLED` 而**不是** `LLM_ERROR`——后者会把值班同学送去查模型配额和鉴权，方向全错。错误码的准确性直接影响排查路径。

首响报告解析：四道降级防线，最终一定给出一个可用的 `FirstResponseReport`，docstring 写明"无论模型返回什么，调用方永远拿到一个可用的报告，不会抛异常"。

**第五层，可观测性。** span 记录节点级耗时（`record_span` 只 append 不 set，因为并行分支各写各的，set 会互相覆盖）；`perf_counter` 而不是 datetime 相减（NTP 校时会算出负耗时）；`flush_spans` 失败只记日志绝不抛（"主流程不该因为写不进耗时明细而失败"）；Prometheus 四个指标，耗时用 Histogram（可跨实例相加算分位数），熔断状态用 Gauge（Counter 表达不了"又闭合了"）。

`refresh_circuit_breaker_metrics` 有个细节：不在 `record_success` 里更新 Gauge，而是单独刷——因为一个打开后再没流量的熔断器永远不会触发那个回调，Gauge 会停在过期值。

**明确的缺口：** 长任务的断点恢复（`rag_v2` 的 `compile()` 无 checkpointer，job 卡在 RUNNING 无回收机制，`job_failure.py:30` 自己承认了）、工具级幂等键、分布式锁、故障注入测试、人工接管流程——**项目中没有，不能硬套**。

**项目证据**
- `app/core/errors.py` — `AppError` 四属性、`to_app_error`、`wrap_llm_exception`
- `app/agent/rag_v2/state.py` — `DegradeReason` 按修复方向划分、`RetrievalStatus` 三态
- `app/config.py:81` — `chat_total_budget_seconds = 90.0`
- `app/api/chat_v2.py` — `asyncio.timeout` 包整个请求
- `app/core/llm_factory.py` — 60s 单次超时、`timeout if timeout is not None else` 的正确写法
- `app/agent/mcp_client.py:18-73` — 指数退避 3 次后 `return CallToolResult(isError=True)`
- `app/core/job_failure.py:68-74,84,121` — `default_job_retry`、无条件重试说明、`retries_left = 0`
- `app/services/index_job_queue.py:11-28` — retry 必须入队时声明
- `app/core/circuit_breaker.py:84` — `threading.Lock` 跨两种执行模型
- `app/agent/rag_v2/nodes.py:175-188` — 熔断器包在 `to_thread` 外侧及原因
- `app/agent/rag_v2/nodes.py` — `retrieve_each_node` 不抛异常的 docstring
- `app/agent/rag_v2/nodes.py:214-233` — `dedup_node` 三态判定
- `app/tools/knowledge_tool.py:13-44` — 空结果与异常都 return
- `app/agent/rag_v2/nodes.py` — `validate_answer_node` fail-open 用 `FEATURE_DISABLED`
- `app/models/aiops_report.py:225-245` — 四道降级防线
- `app/core/span_context.py` — 只 append 不 set、`perf_counter`、`flush_spans` 不抛
- `app/core/metrics.py:56-61,85-113,171-188` — Histogram vs Gauge 的选择、熔断 Gauge 单独刷
- `app/core/job_failure.py:30` — 卡在 RUNNING 无回收的自认
- 断点恢复、工具级幂等键、分布式锁、故障注入、人工接管：项目中没有，不能硬套

#### Q13 · 再给一个上下文处理相关的场景题，如何优化 context 管理？

**回答**

项目里 context 是**分预算**管理的，不是一个大池子。这是我认为最值得讲的设计。

**三块预算相互隔离：**

知识库文档 2800 token（`rag_document_context_token_budget`），会话记忆 1800（其中摘要单独限 700）。为什么要拆开而不是给一个总数：因为它们**会互相挤压**。`_format_context` 的 docstring 直接写了动机："在固定预算内组装检索文档，避免知识库内容挤占会话记忆和问题空间。" 如果只有一个总预算，检索到六篇长文档就能把会话历史全挤掉，模型会突然"忘记"用户前两轮说过什么——而且这种失效是间歇性的，取决于当次检索命中了多长的块，极难排查。

**逐条扣减而不是最后截断。** `_format_context` 的循环里，每条文档先算 `available = budget - used_tokens - _estimate_tokens(prefix)`，`available <= 0` 就 `break`，否则把内容裁到 `available` 再拼。注意 `prefix` 也参与扣减——`f"[{idx}] (源子查询: {src})\n"` 这个标注本身是真实 token。如果只在最后统一截断，会出现"最后一篇被切在半句话中间"的情况；逐条扣减保证每篇都是完整可读的，只有最后一篇可能被裁短。

**会话记忆那侧同样是三重控制。** 摘要标题算进预算、最近消息标题预留、最后整段再兜一次硬上限（`_render_context`）。注释写"角色前缀、分隔符等是实际 Prompt 的一部分，最后再兜底一次确保硬上限"。这些结构性字符最容易在估算里被漏掉。

**Token 估算自己写，方向是保守。** `_estimate_tokens` 中文按 1、其他按 4 字符 1 token 向上取整。用真 tokenizer 会引入依赖和每次调用的额外耗时，而这里只需要"不低估"——低估会突破真实窗口导致 API 报错，高估只是少注入一点内容。这是个明确的方向选择，不是偷懒。

**裁剪按字符成本而非字符数。** `_truncate_to_token_budget` 里中文 cost 1.0、非中文 0.25，逐字符累加到超预算才停。这样中英混排的文本裁出来的长度是按 token 算的，不是按字符算的。

**上游还有一层减量：** `slim_used_documents` 把落库的 `used_documents` 裁剪掉大字段，用 `_SLIM_MARKER` 标记已裁剪保证幂等。这处不是为了模型上下文，是为了不让 trace 表被文档正文撑爆。

**如果这道题问的是"上下文还是超了怎么办"，我的顺序是：** 先减检索条数（`FINAL_TOP_K = 6` 往下调，代价是召回覆盖降低）→ 再压会话摘要（700 往下调，代价是历史信息损失）→ 最后才是砍最近窗口（3 轮往下调，这个代价最大，直接影响多轮连贯性）。顺序依据是"损失的信息离当前问题有多远"。

**项目中没有的：** 按重要性给文档动态排序后再分配预算、上下文的 KV cache 复用、prompt 压缩模型（如 LLMLingua）、把超长文档二次摘要后注入——**项目中没有，不能硬套**。

**项目证据**
- `app/config.py:130` — `rag_document_context_token_budget = 2800`
- `app/config.py:133-135` — 会话记忆 1800 / 摘要 700 / 压缩阈值 2400
- `app/agent/rag_v2/nodes.py:613-628` — `_format_context` 的 docstring 动机与逐条扣减
- `app/agent/rag_v2/nodes.py:620-621` — `prefix` 参与预算扣减
- `app/agent/rag_v2/nodes.py:631-634` — `_estimate_tokens` 中文加权
- `app/agent/rag_v2/nodes.py:637-650` — `_truncate_to_token_budget` 按字符成本裁剪
- `app/services/conversation_memory_service.py:207-247` — `_render_context` 三重预算控制
- `app/services/conversation_memory_service.py:215-225` — 标题 token 预留的注释
- `app/agent/rag_v2/nodes.py:30` — `FINAL_TOP_K = 6`，第一顺位可调项
- `app/agent/rag_v2/nodes.py` — `slim_used_documents` 与 `_SLIM_MARKER` 幂等标记
- 重要性排序分配预算、KV cache 复用、prompt 压缩模型、超长文档二次摘要：项目中没有，不能硬套

#### Q14 · 手撕算法：Greedy（贪心相关）

**回答**

原帖只写了"Greedy（贪心相关）"，没有给出具体题目——按边界处理，**不自行补题**。这里给贪心的判定框架和一道最常考的实现。

**贪心能用的前提是两条，缺一不可：**

**贪心选择性质**——每一步取局部最优，最终能得到全局最优。这条要能证明，通常用交换论证：假设存在一个最优解不含我的贪心选择，把它换成贪心选择后解不会变差，所以贪心选择一定在某个最优解里。

**最优子结构**——做完当前选择后，剩下的问题是同类型的子问题，且原问题最优解包含子问题最优解。

判断题目能不能贪心，我的实际做法是先问"能不能构造反例"。想不出反例再去证。区分贪心和 DP 的关键：贪心当下决定不回头，DP 要枚举所有子问题的组合。如果一个局部选择会影响后面可选的范围、而且需要比较多种走法，那就是 DP。

**跳跃游戏 II（LeetCode 45）** 是贪心里最能体现"边界思维"的一道，给实现：

```python
def jump(nums: list[int]) -> int:
    steps = 0
    current_end = 0      # 当前这一步能到的最远边界
    farthest = 0         # 下一步能到的最远位置

    for i in range(len(nums) - 1):   # 注意不遍历最后一个
        farthest = max(farthest, i + nums[i])
        if i == current_end:         # 走到当前边界，必须再跳一步
            steps += 1
            current_end = farthest

    return steps
```

三个容易写错的点：`range(len(nums) - 1)` 不能到最后一个元素，否则走到终点还会多加一步；`current_end` 和 `farthest` 必须是两个变量，前者是"这一步的势力范围"、后者是"下一步的势力范围"，混用会算错；只在 `i == current_end` 时才 `steps += 1`，因为在边界内的所有位置都属于同一步。复杂度 O(n) 时间、O(1) 空间。

贪心的正确性在于：每一步都把"下一步能到的最远处"取到最大，所以步数一定最少。反证——如果存在更少步数的方案，那它某一步跳得比 `farthest` 更远，但 `farthest` 已经是所有可达位置里的最大值，矛盾。

**项目里的贪心，是真的有一处。** `_take_tail_to_budget` 从最新消息往前扫，能装下就装、装不下就停——这是标准的贪心装载。它成立的原因不是"数学上最优"，而是**业务上有明确偏序**：越新的消息越重要。所以这里不需要背包 DP，贪心就是对的。

这里有一处专门破坏纯贪心的处理：如果第一条（最新那条）自己就超预算，按纯贪心该直接停、一条不装。但代码选择保留并裁剪内容——注释写"单条消息超预算时仍保留尾部"。因为最新那条通常就是当前问题，丢了等于没上下文。**业务约束覆盖了算法的形式最优**，这是我在真实代码里对贪心的实际处理方式。

`_format_context` 也是同构的贪心装载：按检索排序逐条装，装不下 `break`。同样依赖偏序假设——检索结果已经按相关性排好，靠前的更重要。

**项目证据**
- 具体贪心题目：原帖未给出，按"未公开"边界处理，不自行补题
- 跳跃游戏 II 的实现代码：项目中没有，不能硬套。仓库无算法题实现
- `app/services/conversation_memory_service.py:249-272` — `_take_tail_to_budget`，按"越新越重要"偏序的贪心装载
- `app/services/conversation_memory_service.py:260-269` — 单条超预算仍保留并裁剪，业务约束覆盖形式最优
- `app/agent/rag_v2/nodes.py:613-628` — `_format_context` 同构的贪心装载，依赖相关性排序偏序

---

## 九、外部题库（非面经）

> ⚠️ **这一节不是面经。** 以下题目来自两篇公开博客整理的 Agent 面试题库，**没有对应的真实面试场景、公司、时间和追问链**，与前八节性质不同，因此单独成节，不与真实面经混排。
>
> 来源：
> - 题库 A（18 题）：`https://www.cnblogs.com/zxsoul/p/22641333`
> - 题库 B（44 题 + 5 道系统设计）：`https://www.cnblogs.com/itech/p/20111938`
>
> 题库题目普遍比面经题更"标准化"，容易诱导背模板。所以这一节的答法与前八节一致：**每题仍然先回到本仓库的真实代码**，仓库里没有的照旧写明「项目中没有，不能硬套」。凡是只能讲通用原理的，明确标注这是原理陈述而非项目经验——面试里这两者的可信度差距很大。

### 题库 A · Agent 基础与架构（18 题）

> 来源：`https://www.cnblogs.com/zxsoul/p/22641333`
> 原文另有四处标注为"挖坑"的待补充问题（如何写好系统提示词、如何写好工具描述、思维链如何决定推理能力、记忆框架的了解），原文未给答案，按未公开处理，不自行补题。

#### A1 · 你对 AI Agent 的理解

**回答**

我更愿意用自己项目里的一个具体区分来回答，而不是给定义：**我这个仓库里有两条路径，一条是 Agent，一条不是，而我最后主用的是不是 Agent 的那条。**

`rag_agent_service.py` 走 LangChain 的 `create_agent`，配 `MemorySaver`，模型自己决定要不要调 `retrieve_knowledge` 工具、调几次、什么时候停。这是标准的 Agent：控制流由模型输出决定。

`rag_v2` 是我后来写的，**不是 Agent**。`graph.py` 里五个节点的边全是 `add_edge` 静态声明的：`rewrite → retrieve_each → dedup → generate → validate_answer → END`。唯一的动态成分是 `_fanout_to_retrieve` 用 `Send` 把子查询扇出成 N 个并行分支，但 N 来自上一个节点的输出条数，不是模型的路由决策。整张图跑完的路径是固定的，模型在里面只负责"改写查询""生成答案""判分"这三件事，不负责"下一步去哪"。

所以我理解的 Agent 核心是**控制流的归属**：模型决定下一个动作的是 Agent，代码决定的是 Workflow。这个区分有工程后果——`rag_v2` 的每一次请求耗时和 LLM 调用次数都是可预估的（3 次固定：改写 1 次、生成 1 次、校验 1 次），`create_agent` 那条路不是，模型多调一轮工具就多一轮延迟。

我选 Workflow 是因为这个场景**不需要** Agent：知识库问答的步骤是确定的，让模型来决定"要不要检索"只会引入不确定性，收益是零。这也是我对 Agent 的另一半理解——它是一个有代价的选择，不是默认选择。

**项目证据**
- `app/agent/rag_v2/graph.py:44-51` — 五条静态边，控制流由代码决定
- `app/agent/rag_v2/graph.py:19-22` — `_fanout_to_retrieve`，N 由上游输出条数决定而非模型路由
- `app/services/rag_agent_service.py:133-136` — `create_agent` + `MemorySaver`，模型决定工具调用时机
- `app/agent/rag_v2/nodes.py:28` — `NUM_SUB_QUERIES = 3`，扇出宽度写死
- 模型驱动的动态路由、ReAct 循环、自主终止判断：项目中没有，不能硬套。`rag_v2` 是固定管线

#### A2 · Agent 的优势

**回答**

我只能说我**实际观察到**的优势，以及我为什么在自己项目里放弃了它。

`create_agent` 那条路的真实优势是**省代码**。工具注册、循环、消息累积、checkpoint 全由框架处理，`rag_agent_service.py:133-136` 四行就有了一个能用的 Agent，`MemorySaver` 顺带给了会话历史的读写和清理（`:321-325` 读、`:386-387` 删）。同样的能力我在 `rag_v2` 里手写了五个节点、一个 state 定义、一套 span 观测，几百行。

第二个优势是**对未知路径的适应**。如果用户的问题有时需要检索、有时不需要（比如"今天几号"这种，仓库里有 `time_tool.py`），Agent 能自己判断；Workflow 要么全走检索，要么我得自己写一个意图分类节点——而**意图分类在这个仓库里是不存在的**（我 grep 过，只有 FastAPI 的 `APIRouter`，没有任何语义路由），所以 `rag_v2` 事实上对所有问题都走一遍检索。这是我为了确定性付出的代价。

反过来说劣势，也是我切走的原因：Agent 的循环次数不可控。`rag_v2` 一次请求恰好 3 次 LLM 调用，能算出总预算；`create_agent` 那条路我没法在入口处判断这次会花多久，而 `chat_total_budget_seconds = 90.0` 这个总预算是要卡死的。加上并行分支的失败语义我需要自己定义（一个分支挂了算部分成功还是整体失败），框架给的默认行为不是我要的——这个在 `retrieve_each_node` 的 docstring 里写得很清楚。

**项目证据**
- `app/services/rag_agent_service.py:133-136` — 四行拿到一个 Agent
- `app/services/rag_agent_service.py:321-325,386-387` — `MemorySaver` 顺带给的历史读取与清理
- `app/tools/time_tool.py` — 存在一个不需要检索的工具，理论上适合 Agent 自主路由
- `app/config.py:81` — `chat_total_budget_seconds = 90.0`，需要可预估的调用次数
- `app/agent/rag_v2/nodes.py` — `retrieve_each_node` docstring：并行分支失败语义需要自己定义
- 意图分类 / 语义路由：项目中没有，不能硬套。`rag_v2` 对所有问题都走检索

#### A3 · Agent 的基本架构有哪几部分组成

**回答**

不按教科书的分层讲，按我仓库里**真实存在的东西**讲，缺的部分单独列出来。

**编排层**：`app/agent/rag_v2/graph.py`，`StateGraph` 加五个节点。所有节点注册都过一遍 `instrument_node`，这是为了拿节点级耗时——注释里写了为什么统一在这里包而不是节点体内计时："五个节点写五份计时代码是重复，且新增节点必然有人忘记加"。

**状态层**：`app/agent/rag_v2/state.py`，`TypedDict(total=False)`，扇入字段用 `Annotated[List[...], operator.add]` 让并行分支的结果自动合并。这是 LangGraph 的 reducer 机制，也是整个并行设计的地基。

**模型层**：`app/core/llm_factory.py`，统一创建 chat model，单次调用超时 60s。全仓库只有一个 LLM：`qwen-max`（`config.py:46`）。

**工具层**：`app/tools/`，两个工具——`knowledge_tool.py` 的 `retrieve_knowledge` 和 `time_tool.py`。工具的错误处理方式很明确：空结果和异常都 `return` 字符串，不 raise。

**知识层**：Milvus（1024 维、HNSW、COSINE）+ BM25 混合检索，`vector_search_service.py`，RRF 融合权重 0.7 / 0.3。

**记忆层**：`conversation_memory_service.py`，滚动摘要 + 最近三轮窗口，原文留 PostgreSQL 审计。

**可靠性层**：`app/core/errors.py` 的 `AppError` 体系、`circuit_breaker.py` 的熔断器、`RetrievalStatus` 三态、`validate_answer_node` 的答案校验。

**可观测层**：`span_context.py` + `chat_run_span.py` / `chat_run_trace.py` 两张表，`metrics.py` 四个 Prometheus 指标。

**缺的部分，逐项说清：**
- **接入与身份层：项目中没有，不能硬套。** 我 grep 过 `Security|OAuth2|HTTPBearer|verify_token|current_user|Authorization`，整个 `app/` 返回空。没有鉴权、没有租户、没有配额。唯一沾边的是 `config.py:31-38` 的 CORS 配置。
- **意图路由：项目中没有，不能硬套。**
- **工具沙箱与权限：项目中没有，不能硬套。** 两个工具都是只读的，所以还没遇到这个需求，但这是运气不是设计。
- **Checkpoint：`rag_v2` 的 `compile()` 没传 checkpointer**，只有 `create_agent` 那条路有 `MemorySaver`（进程内内存，重启即失）。

**项目证据**
- `app/agent/rag_v2/graph.py:24-42` — 五节点注册与 `instrument_node` 统一包裹的原因注释
- `app/agent/rag_v2/state.py` — `TypedDict(total=False)` + `operator.add` reducer
- `app/core/llm_factory.py` — 统一模型创建，单次 60s 超时
- `app/config.py:46` — `qwen-max`，全仓库唯一 LLM
- `app/tools/knowledge_tool.py:13-44`、`app/tools/time_tool.py` — 两个工具，错误一律 return
- `app/services/vector_search_service.py:42-43` — RRF 权重 0.7 / 0.3
- `app/services/conversation_memory_service.py:1-5` — 记忆层分工
- `app/core/errors.py`、`app/core/circuit_breaker.py` — 可靠性层
- `app/core/span_context.py`、`app/core/metrics.py:85-113` — 可观测层四个指标
- 鉴权 / 租户 / 配额 / 意图路由 / 工具沙箱 / `rag_v2` 的 checkpoint：项目中没有，不能硬套

#### A4 · 工作流是什么

**回答**

工作流是**控制流写死在代码里**的编排形态。我仓库里的 `rag_v2` 就是一个纯工作流，可以直接当例子讲。

`graph.py:44-51` 五条边：

```python
graph.add_edge(START, "rewrite")
graph.add_conditional_edges("rewrite", _fanout_to_retrieve, ["retrieve_each"])
graph.add_edge("retrieve_each", "dedup")
graph.add_edge("dedup", "generate")
graph.add_edge("generate", "validate_answer")
graph.add_edge("validate_answer", END)
```

注意第二条是 `add_conditional_edges`，看起来像分支，实际不是——`_fanout_to_retrieve` 返回的是 `[Send("retrieve_each", {"query": q}) for q in sub_queries]`，它做的是**扇出**不是**选路**：所有子查询都去同一个节点，只是并行 N 份。真正的条件分支（根据状态决定去 A 还是去 B）这张图里一处都没有。

工作流的收益在我这儿很具体：

**成本可预估。** 一次请求固定 3 次 LLM 调用。`chat_total_budget_seconds = 90.0` 这个总预算能定得下来，是因为我知道最坏情况有几次调用。

**每一步都能装观测。** 五个节点都过 `instrument_node`，每个节点一条 span，`retrieve_each` 被扇出成 3 份就记 3 条。注释里写了这解决的实际痛点："以前日志里 4 条分支交织在一起，根本对不出每条各自花了多久。"如果控制流由模型决定，节点序列每次都不同，这种按 `node` 维度聚合的分析就做不了。

**故障点可穷举。** 我能把每个节点的失败行为逐一定义：`retrieve_each` 兜住所有异常返回空、`dedup` 判定三态、`generate` 失败走降级、`validate_answer` 在特性关闭时 fail-open。Agent 循环里节点会重复出现，"第二次调用检索失败"和"第一次失败"要不要区别对待，这个状态空间大得多。

**代价：不灵活。** 所有问题都走一遍完整检索，包括那些明显不需要检索的。因为没有意图分类层（**项目中没有，不能硬套**），我没有便宜的方式绕过。

**项目证据**
- `app/agent/rag_v2/graph.py:44-51` — 五条静态边全文
- `app/agent/rag_v2/graph.py:19-22` — `_fanout_to_retrieve` 是扇出不是选路
- `app/agent/rag_v2/graph.py:24-42` — `instrument_node` 与"分支交织对不出耗时"的痛点注释
- `app/config.py:81` — 固定调用次数才使总预算可定
- `app/agent/rag_v2/nodes.py` — 四个节点各自的失败行为定义
- 真正的条件分支（按状态选路）、意图分类：项目中没有，不能硬套

#### A5 · AI 的读书和数据分析

**回答**

这道题在原博客里就是一个没展开的条目，标题也含混。我按字面理解为"用 LLM 做长文档阅读理解与数据分析"这类应用形态，先说仓库里对应到什么，再说不对应的部分。

**文档阅读这一半，仓库里有真东西。** 整条链路是：PDF/Markdown 入库 → 切块 → 向量化 → 检索 → 带证据生成 → 校验。`protocol_pdf_ingestion_service.py` 处理 PDF，`document_splitter_service.py` 两级切块（按 h1/h2 标题切，超长块用 `RecursiveCharacterTextSplitter` 兜），入库走 RQ 异步队列。生成时 `_format_context` 把检索到的块按 `rag_document_context_token_budget = 2800` 装进 prompt，每块带 `[i] (源子查询: ...)` 前缀——这个前缀是为了让答案能溯源到具体哪一块。

**数据分析这一半，项目中没有，不能硬套。** 没有表格理解、没有 SQL 生成、没有代码解释器、没有 pandas/numpy 的数据处理链路，也没有图表生成。`mcp_servers/monitor_server.py` 看着像监控数据分析，但读了就知道它是**mock 数据生成器**——用 `random.uniform` 造时序值（`:224`、`:373`），不是真实采集，更没有分析逻辑。拿它冒充数据分析能力是不诚实的。

有一处沾边但要说清边界：`aiops_report.py` 的 `parse_report` 把 LLM 输出解析成结构化报告，这是"让模型产出结构化数据"，不是"让模型分析数据"。方向正好相反。

**项目证据**
- `app/services/protocol_pdf_ingestion_service.py` — PDF 入库链路
- `app/services/document_splitter_service.py:20-38` — 两级切块
- `app/services/index_job_queue.py:11-28` — RQ 异步入库
- `app/agent/rag_v2/nodes.py:613-628` — `_format_context` 带来源前缀的证据装载
- `app/config.py:130` — `rag_document_context_token_budget = 2800`
- `mcp_servers/monitor_server.py:224,373` — `random.uniform` 造数据，不是真实采集
- `app/models/aiops_report.py:225-247` — 结构化输出解析，方向与"分析数据"相反
- 表格理解 / Text2SQL / 代码解释器 / 图表生成 / 真实数据采集：项目中没有，不能硬套

#### A6 · 了解哪些其他的 Agent 设计范式？Agent 和 Workflow 的区别是什么？

**回答**

**区别，一句话：控制流谁定。** 模型定是 Agent，代码定是 Workflow。判据不是"用没用 LLM"（两者都用），也不是"有没有工具"（Workflow 也能调工具），而是"下一步执行什么"这个决策由谁做出。

我仓库里两条路正好各占一边，前面几题已经讲过，这里补范式部分。

**我实际用过的范式，只有两个：**

**固定管线（Pipeline）** —— `rag_v2`。五节点静态边，扇出宽度写死 3。适合步骤确定的任务，代价是不灵活。

**工具增强的单 Agent** —— `rag_agent_service.py` 的 `create_agent`。模型自主决定工具调用，`MemorySaver` 存 checkpoint。

**我知道但项目里没有的范式，逐个标注：**

- **ReAct（Thought-Action-Observation 循环）：项目中没有，不能硬套。** `create_agent` 内部的循环形态接近 ReAct，但我没有自己实现过，也没有对 Thought 做任何处理或约束。
- **Plan-and-Execute：项目里曾经有，现在已删除。** `app/agent/aiops/` 下原本有 `planner.py` / `executor.py` / `replanner.py`，现在整个目录只剩 `__pycache__`，git 状态是全部 deleted。现存的 `alert_diagnosis_orchestrator.py` 是**固定编排**（normalize → 建任务 → planning/retrieving/diagnosing 状态流转 → 分析 → 存报告），阶段名里有 "planning" 但那是状态标签，不是模型规划。这个我必须说清，否则就是拿删掉的代码充数。
- **Reflection / 自我批评循环：项目中没有，不能硬套。** `validate_answer_node` 会给答案打覆盖度和依据性两个分，不达标时替换成缺口说明——但它**不会退回重做**。`needs_second_retrieval` 这个字段在校验结果里解析出来了，图里没有任何边消费它。这是一个"识别了问题但没有修复回路"的状态，讲的时候不能说成 Reflection。
- **多 Agent（Supervisor / 去中心化协作 / handoff）：项目中没有，不能硬套。** grep `supervisor|sub_agent|handoff` 全空。
- **Tree of Thoughts / 多路径搜索：项目中没有，不能硬套。**

**项目证据**
- `app/agent/rag_v2/graph.py:44-51` — 固定管线范式
- `app/services/rag_agent_service.py:133-136` — 工具增强单 Agent 范式
- `app/agent/aiops/` — Plan-Execute-Replan 三个文件已 deleted，仅剩 `__pycache__`
- `app/services/alert_diagnosis_orchestrator.py:50-63` — 阶段名是状态标签，`_PHASE_STATUS` 的注释解释了为什么只有"开始做"才是新状态
- `app/agent/rag_v2/nodes.py:505-518` — `needs_second_retrieval` 被解析出来
- `app/agent/rag_v2/graph.py:44-51` — 没有任何边消费 `needs_second_retrieval`，即无重做回路
- ReAct 自实现 / Reflection 回路 / 多 Agent / ToT：项目中没有，不能硬套

#### A7 · CoT 推理链

**回答**

**先把项目边界说清：仓库里没有任何 CoT 脚手架，项目中没有，不能硬套。** 我 grep 过 few-shot 示例、"让我们一步步思考"这类引导、思维链解析——四个 prompt（`nodes.py:36`、`:50`、`:60`、`:87`）里一处都没有。所以下面讲的是原理和我的判断，不是项目经验，这一点我在面试里会主动说明。

**原理部分。** CoT 的作用是把答案的生成过程从"一步映射"变成"多步展开"。Transformer 是逐 token 自回归的，每个 token 的计算深度是固定的，需要多步推导的问题（多位数乘法、多跳推理）在单步内算不完；把中间步骤显式写进输出，后续 token 就能读到前面的中间结果，等价于把有限深度的计算摊成了序列长度上的多次计算。所以 CoT 的收益本质来自"用序列长度换计算深度"。

**它不是免费的。** 输出变长意味着延迟和成本线性上升，而且中间步骤一旦有错，后面会顺着错的走——错误会被放大而不是被纠正。小模型上 CoT 常常反而更差，因为它生成的中间步骤本身不可靠。

**我为什么在项目里没用它。** `rag_v2` 的四个 prompt 都是**要求结构化输出**的：改写节点要 JSON 数组，校验节点要 JSON 对象带七个字段，AIOps 的首响报告要固定 schema。这类任务要的是格式稳定，而 CoT 的自由文本推导恰好和格式约束打架——我更需要能可靠 `json.loads` 的输出，而不是一段漂亮的推理。

真要说项目里有什么和 CoT 同类的东西，是**把推理拆到多个节点之间**而不是塞进一次生成里：改写节点负责"这个问题该从哪几个角度查"（拆成 3 个子查询），生成节点负责"根据证据回答"，校验节点负责"这个回答站不站得住"。这是用图结构做的显式分步，每一步的输入输出我都能看到、能落 span、能单独测。代价是三次调用而不是一次。

**项目证据**
- `app/agent/rag_v2/nodes.py:36,50,60,87` — 四个 prompt，无 few-shot、无 CoT 引导
- `app/agent/rag_v2/nodes.py:472-492` — 改写节点要求 JSON 数组输出
- `app/agent/rag_v2/nodes.py:494-520` — 校验节点要求 JSON 对象带七字段
- `app/models/aiops_report.py:193-222` — 首响报告的固定 schema 解析
- `app/agent/rag_v2/nodes.py:28` — 拆 3 个子查询，把分步落在图结构上
- CoT prompt / few-shot 示例 / 思维链解析：项目中没有，不能硬套。以上原理陈述非项目经验

#### A8 · 复杂任务拆分如何做，效果如何提升

**回答**

仓库里有一个**真的在跑的任务拆分**，就是 `rewrite_node`：把用户的一个问题拆成 3 个子查询，各自独立检索，再扇入合并。这是我唯一能拿真实代码讲的拆分。

**怎么拆。** `NUM_SUB_QUERIES = 3` 写死。prompt 要求模型输出 JSON 数组，解析走 `_parse_sub_queries`——它做三层兜底：先找 ```json 代码块，再找裸的 `[...]`，`json.loads` 失败就 `return [fallback]`，fallback 是原始 query。所以最坏情况下拆分失败，退化成"用原问题查一次"，链路不会断。

**为什么拆有效果。** 单次向量检索受查询表述影响很大，同一个意图换个说法召回结果能差不少。拆成 3 个角度各查 4 条（`RETRIEVE_TOP_K = 4`），去重后取 6 条（`FINAL_TOP_K = 6`），本质是**用多样化的查询降低单次表述的方差**。这一点我有离线数据支撑：`tests/eval/multi_query.py` 是专门测这个的脚本，`tests/eval/golden_set_expanded.jsonl` 83 条样本，指标是 Hit@K / Recall@K / MRR（`tests/eval/metrics.py:75,81,90`）。结论是多查询 RRF 正向。

**拆分怎么扇入。** 三个分支的结果靠 state 里的 `Annotated[List[...], operator.add]` reducer 自动合并，`dedup_node` 是扇入后第一个节点，也是**唯一能看到全部分支结果**的位置——所以三态判定（全失败 / 部分失败 / 全空）只能放在它里面。

**失败处理是这个设计里最关键的一处。** `retrieve_each_node` 兜住所有异常，一个分支挂了返回空而不是抛。docstring 里写了原因：LangGraph 的语义是"任一并行分支抛异常 → 整个 super-step 失败 → 整张图崩"，兜住之后语义变成"一个分支失败 = 证据少一份"。拆分如果不配这个兜底，拆得越多整体成功率越低——3 个分支各有 p 的失败率，整体失败率是 1-(1-p)³，拆分反而是负收益。

**没做的部分：** 依赖关系的任务拆分（子任务 B 需要 A 的结果）、动态决定拆几个、拆分失败后的重新拆分——**项目中没有，不能硬套**。这里的 3 个子查询是**完全独立、可并行、无依赖**的，属于最简单的一类拆分。

**项目证据**
- `app/agent/rag_v2/nodes.py:28-30` — `NUM_SUB_QUERIES=3`、`RETRIEVE_TOP_K=4`、`FINAL_TOP_K=6`
- `app/agent/rag_v2/nodes.py:472-492` — `_parse_sub_queries` 三层兜底，失败退回原 query
- `app/agent/rag_v2/graph.py:19-22` — `Send` 扇出
- `app/agent/rag_v2/state.py` — `operator.add` reducer 扇入合并
- `app/agent/rag_v2/nodes.py:214-233` — `dedup_node` 三态判定，扇入唯一可见点
- `app/agent/rag_v2/nodes.py` — `retrieve_each_node` docstring：super-step 失败语义
- `tests/eval/multi_query.py`、`golden_set_expanded.jsonl`（83 条）、`metrics.py:75,81,90` — 离线验证
- 带依赖的任务拆分 / 动态拆分宽度 / 重新拆分：项目中没有，不能硬套

#### A9 · Agent 的记忆机制有哪些，实际开发过程中如何设计记忆模块

**回答**

仓库里的记忆是**两层，不是三层**，这一点我要先说准，因为标准答案通常讲"短期 + 长期 + 工作记忆"三层。

**短期：最近 N 轮原文。** `_recent_message_count` 按**用户轮次**取尾部，不是按消息条数——`conversation_memory_recent_turns = 3`。为什么按轮次：一轮里用户一条、助手一条，按条数取 6 条可能切在半轮中间，只留下助手的回答没有对应的提问，模型会看到一段没有上文的答案。

**长期：滚动摘要。** 触发条件是**待压缩部分的 token 数**超过 `conversation_memory_compact_threshold_tokens = 2400`，不是轮数。`snapshot` 里存 `summarized_through_message_id` 当游标，已压缩的部分不重复压，新摘要 = 旧摘要 + 新消息合并。

**设计上我认为最重要的三个决定：**

**第一，原文永远不删。** 模块 docstring 明确写了"完整原始消息只保存在 PostgreSQL，作为审计记录；模型上下文仅使用压缩摘要和未压缩消息的尾部"。摘要是为了省 token，不是为了省存储。任何时候要复盘"模型当时到底看到了什么"，原文都在。

**第二，摘要 prompt 里有一条硬约束**："不要把模型猜测、SOP 内容或历史数值写成实时事实；发生冲突时以较新的用户消息为准。"这条是我认为滚动摘要最危险的失效模式——摘要一旦把猜测写成事实，之后每一轮都会读到这个假事实，错误会自我强化，而且原文里根本找不到它是哪来的。输出固定四段：【明确事实】【任务进展】【约束与偏好】【待确认项】，缺的写"无"，固定格式是为了让下一次合并有稳定结构可依。

**第三，摘要失败必须降级而不是报错。** `_summarize` 整个包在 try 里，异常返回空串，注释写"摘要是节省 Token 的优化，不可阻断主聊天；失败时退回最近消息窗口"。记忆压缩是优化项，优化项失败不能拖垮主流程。

**预算是三层嵌套的**，这是实现里最容易写错的地方：总预算 1800、摘要占 700、剩下给最近消息。`_render_context` 里连"【会话滚动摘要】"这个标题本身的 token 都算进去了，注释解释了原因："预留最近消息标题，确保最终渲染文本（含标签）也不会突破总预算。"最后还有一次兜底截断。

**缺的部分：** 跨会话的用户画像、按需向量召回历史对话、记忆的过期与删除策略、记忆冲突的显式检测——**项目中没有，不能硬套**。向量检索的目标是知识库文档，不是对话历史，拿它冒充"长期记忆召回"是偷换概念。

**项目证据**
- `app/services/conversation_memory_service.py:146-158` — `_recent_message_count` 按轮次
- `app/config.py:133-136` — 四个预算参数
- `app/services/conversation_memory_service.py:104-126` — snapshot 游标与增量合并
- `app/services/conversation_memory_service.py:28-35` — 摘要 prompt 四段格式与"不要把猜测写成事实"
- `app/services/conversation_memory_service.py:183-186` — 摘要失败降级
- `app/services/conversation_memory_service.py:207-247` — 三层预算嵌套与标题 token 预留
- `app/services/conversation_memory_service.py:1-5` — 原文留 PG 审计
- `app/services/conversation_memory_service.py:48-54` — CJK 加权的 token 估算
- 跨会话画像 / 对话历史向量召回 / 记忆过期删除 / 冲突检测：项目中没有，不能硬套

#### A10 · 上下文不够怎么处理

**回答**

本仓库把"上下文不够"拆成两个不同的问题，处理方式完全不同。

**第一个是装不下（预算超了）。** 这是确定性问题，用静态预算切分解决。`app/config.py:156-160` 声明了三个独立预算：知识库文档 2800 token、会话记忆 1800 token、记忆里的摘要部分 700 token。三者不共享池子，各自独立裁剪。这么切的理由是：如果只设一个总预算让三方去抢，检索到一篇长文档就能把会话记忆整个挤掉，用户下一句"那它呢"就失去了指代对象——而这类失败在日志里表现为"模型答非所问"，极难归因。分开设预算是用一点空间效率换可归因性。

装载策略是**带偏序的贪心**，不是均匀截断。会话记忆走 `conversation_memory_service.py:249-272` 的 `_take_tail_to_budget`，从最新消息往前装，装不下就停——偏序是"越新越重要"。文档走 `nodes.py:613-628` 的 `_format_context`，按检索相关性顺序装——偏序是"越相关越重要"。两处结构同构，偏序来源不同。

有一个边界值得单独说：`conversation_memory_service.py:260-269`，单条消息本身就超预算时，不丢弃，而是保留并裁剪。形式上这违反了预算约束，但业务上必须这样——用户刚粘进来的一大段报错就是当前问题本身，丢了它整轮对话就没有主题了。这里是业务约束覆盖形式最优的一处，注释里写明了。

**第二个是历史太长（累积溢出）。** 这是增长性问题，用滚动摘要解决。`conversation_memory_service.py:104-126` 维护一个 snapshot 游标 `summarized_through_message_id`：游标之前的消息已被压成摘要，游标之后的保留原文，两段拼接成上下文。新消息只增量追加，不重新摘要全量历史——否则每轮对话的摘要成本随历史线性增长。

摘要失败时不阻断主流程。`conversation_memory_service.py:183-186` 把整个摘要调用包在 try/except 里，失败返回空串，上下文退化成"只有最近消息窗口"。注释写的是"摘要是节省 Token 的优化，不可阻断主聊天"——这是分清了优化和功能：优化失败只该损失优化收益。

最近窗口按**轮次**而不是条数算，`conversation_memory_service.py:146-158` 的 `_recent_message_count` 从尾部往前数 user 消息。按条数算会在有工具调用的轮次里把一问一答劈成两半，只留下工具结果没有原始提问。

**缺的部分：** 上下文不够时**主动向用户追问**、按需从历史里向量召回相关片段、多轮压缩后的摘要再压缩（摘要的摘要）——**项目中没有，不能硬套**。

**项目证据**
- `app/config.py:156-160` — 三个独立 token 预算，2800 / 1800 / 700
- `app/services/conversation_memory_service.py:249-272` — `_take_tail_to_budget`，按新旧偏序的尾部贪心装载
- `app/services/conversation_memory_service.py:260-269` — 单条超预算保留并裁剪，业务约束覆盖形式最优
- `app/services/conversation_memory_service.py:274-289` — `_truncate_text` 单条裁剪
- `app/agent/rag_v2/nodes.py:613-628` — `_format_context` 按相关性偏序装载
- `app/agent/rag_v2/nodes.py:630-650` — `_estimate_tokens` 保守估算
- `app/services/conversation_memory_service.py:104-126` — snapshot 游标与增量摘要
- `app/services/conversation_memory_service.py:183-186` — 摘要失败降级为最近窗口
- `app/services/conversation_memory_service.py:146-158` — 按 user 轮次而非消息条数取窗口
- `app/services/conversation_memory_service.py:207-247` — `_render_context` 先扣标题成本再分配正文
- 主动追问 / 历史片段按需向量召回 / 摘要的二次压缩：项目中没有，不能硬套

#### A11 · 知识图谱

**回答**

**项目中没有，不能硬套。** 全仓没有实体抽取、没有关系抽取、没有图存储（无 Neo4j / NebulaGraph / networkx 依赖）、没有多跳推理、没有 Cypher 或 Gremlin 查询。检索层只有两条腿：Milvus 向量检索和 BM25 关键词检索，靠 `EnsembleRetriever` 做 RRF 融合。

有两样东西看起来像图，但都不是，不能拿来冒充：

**一是 Markdown 的标题层级。** `document_splitter_service.py` 用 `MarkdownHeaderTextSplitter` 切分时，会把 h1/h2 标题写进 chunk 的 metadata。这确实是结构化信息，但它是**单文档内的树**，不是图：没有跨文档的实体对齐（两篇文档里的"订单服务"是两个孤立的字符串，不是同一个节点），没有边的类型（h1 到 h2 只有"包含"一种关系），也没有任何基于它的多跳查询——它只被当作展示用的来源标注和排序时的辅助字段。把树叫成图，把"有 metadata"叫成"有知识图谱"，是两次偷换。

**二是 PostgreSQL 里的外键。** `app/models/knowledge_base.py` 有知识库到文档的关联。这是关系型建模的规范化，服务的是数据一致性，不是语义推理。用外键做多跳等于手写递归 JOIN，仓库里没有这样的查询。

**如果要在本仓库补图谱，真实的切入点在哪：** AIOps 那条线最有价值。`aiops-docs/` 下的六篇 SOP（cpu_high_usage / memory_high_usage / disk_high_usage / high_api_latency / slow_response / service_unavailable）之间存在真实的因果关系——磁盘满会导致服务不可用，CPU 高会导致响应慢。现在这些关系只以自然语言写在文档正文里，模型每次都要重新从文本里读出来。若抽成显式的因果边，就能支持"从告警指标出发，两跳内找到所有可能根因"这类查询，而向量检索做不到——它只能召回"文字上像"的文档，召回不了"因果上相关但用词完全不同"的文档。`[假设]` 这是设计判断，仓库里没有对应实现。

**项目证据**
- 知识图谱 / 实体抽取 / 关系抽取 / 图数据库 / 多跳推理：项目中没有，不能硬套
- `app/services/document_splitter_service.py` — `MarkdownHeaderTextSplitter` 只产出单文档内的标题层级 metadata，无跨文档实体对齐
- `app/services/hybrid_retriever_service.py` — 检索只有向量 + BM25 两路，RRF 融合，无图检索
- `app/models/knowledge_base.py` — 外键是关系型规范化，非语义图
- `aiops-docs/*.md` — 六篇 SOP 间的因果关系目前只存在于自然语言正文中
- 图谱切入点分析：`[假设]`，仓库无实现

#### A12 · 长期记忆的重复、冲突、碎片化如何维护

**回答**

三个问题在本仓库的落实程度差别很大，逐个说。

**冲突——有明确规则，但靠模型执行。** `conversation_memory_service.py:28-35` 的摘要 system prompt 里写了两条硬规则："不要把模型猜测、SOP 内容或历史数值写成实时事实"和"发生冲突时以较新的用户消息为准"。第一条防的是记忆污染：SOP 文档里写着"正常 CPU 使用率应低于 60%"，模型很容易把它摘成"当前 CPU 使用率 60%"，一旦写进摘要，后续几十轮都会把这个虚构数值当作既有事实。第二条给出了时序优先的冲突消解序。

但要说清楚它的性质：这是**声明式约束，不是机制**。没有代码去检测摘要里是否真的出现了冲突，也没有校验模型是否遵守了规则。摘要一旦生成就直接进上下文。它的可靠性等于模型的指令遵循能力，不等于系统保证。

**重复——记忆层没有去重。** 仓库里的 `content_hash` 有三处（`knowledge_base.py:42`、`knowledge_base.py:77`、`protocol_ingestion.py:46`），全部服务于**文档入库**去重：同一份文件重复上传时不重复切分和向量化。这跟记忆去重是两回事，不能拿来充数。记忆侧的实际情况是：`_messages_after_snapshot` 取游标后的原文直接拼接，语义重复的多轮对话会原样进上下文；摘要里如果同一件事被反复提及，也没有任何合并逻辑。

**碎片化——结构上被规避了一部分。** 摘要不是自由文本，`conversation_memory_service.py:28-35` 强制四段固定标题：【明确事实】【任务进展】【约束与偏好】【待确认项】，缺失项写"无"。固定 schema 让同类信息落在同一段里，这在一定程度上限制了碎片化——新的约束会追加到【约束与偏好】而不是散落各处。但这只是**格式约束**，段内条目之间仍没有合并、没有去重、没有优先级。

**有一处设计值得单独说，因为它是保守取舍：** `conversation_memory_service.py:133-144` 的 `_messages_after_snapshot`，当游标指向的消息已被软删除时，直接 `return []`，不注入任何旧原文。注释写的是"游标对应的消息已被软删除时，保守地不注入旧原文，避免重复历史"。这里宁可少给上下文也不冒重复的风险——判断依据是"上下文少一点模型会说不知道，上下文重复会让模型把同一件事当成发生了两次"，后者更难发现。

**缺的部分：** 记忆条目级的去重、冲突的显式检测与告警、碎片合并、记忆的重要性评分与过期淘汰、跨会话的长期记忆——**项目中没有，不能硬套**。严格说，本仓库只有"单会话内的滚动压缩记忆"，把它叫"长期记忆"本身就不准确。

**项目证据**
- `app/services/conversation_memory_service.py:28-35` — 摘要 prompt 的两条硬规则：不写猜测、冲突以较新用户消息为准；四段固定标题
- `app/services/conversation_memory_service.py:133-144` — 游标消息被软删时返回空列表，宁少给不重复
- `app/services/conversation_memory_service.py:104-126` — snapshot 游标与增量合并
- `app/models/knowledge_base.py:42` / `app/models/knowledge_base.py:77` / `app/models/protocol_ingestion.py:46` — `content_hash` 服务于文档入库去重，不是记忆去重
- 记忆条目去重 / 冲突显式检测 / 碎片合并 / 重要性评分 / 过期淘汰 / 跨会话长期记忆：项目中没有，不能硬套

#### A13 · 反思机制

**回答**

这题我要给一个跟通常答法相反的结论：**本仓库有"事后校验"，但没有"反思循环"，两者的区别恰好在有没有回边上。**

`nodes.py:569` 的 `validate_answer_node` 做的是证据校验：把问题、答案、证据三段送给一个 temperature=0 的模型，判断答案是否被证据支持，输出 `coverage_score`、`blocked` 和 `needs_second_retrieval`。看到 `needs_second_retrieval` 这个字段，很自然会以为存在"校验不通过 → 重新检索 → 重新生成"的闭环。

实际上没有。两个证据：

第一，全仓 grep `needs_second_retrieval` 得到十处引用（`nodes.py:92, 103, 517, 556, 579, 586, 747, 757, 775, 781, 813`），**全部是写入方或声明方**——prompt 模板里的字段说明、`_default_validation` 的参数、解析函数里的字段提取。没有任何一处读取它并据此改变控制流。它被算出来、被写进 state、被返回给前端，然后就结束了。

第二，`graph.py:48` 是 `graph.add_edge("validate_answer", END)`。整张图的边是一条直线：START → rewrite → (Send 扇出) retrieve_each → dedup → generate → validate_answer → END。`add_conditional_edges` 只用了一次，在 `rewrite` 之后做子查询扇出，不是用来做校验回环的。没有回边，就没有循环，"反思"这个词就不成立。

**校验不通过时实际发生什么：** 走降级而不是重试。`nodes.py:583-588` 没有证据时，把答案换成 `_build_fallback_answer` 的保守回答；`nodes.py:576-581` 答案为空时标记未通过。也就是说，本仓库选择的是**一次性判定 + 降级输出**，不是**多轮自我修正**。

**这个选择有它的道理，也有它的代价。** 道理是：反思循环的成本是乘法级的，每轮反思要多一次校验调用加一次生成调用，而 `config.py:81` 的全局预算 `chat_total_budget_seconds = 90.0` 是硬上限，两轮反思就可能吃掉整个预算，最后既没答案也超时。代价是：`needs_second_retrieval` 这个字段目前是**死字段**，它给读代码的人（包括面试官）一个存在闭环的错误印象。诚实的说法是"这是为未来的重试留的接口，当前没有消费方"。

**另一处更接近"反思"的真实设计，在校验器自身的失效处理上。** `nodes.py:601-611`：当运行时开关 `enable_answer_validation` 关掉时，走 fail-open——保留答案，只标记 `DegradeReason.FEATURE_DISABLED`。注释解释了为什么不用 `_default_validation(blocked=True)`：那会把每个答案都换成"当前证据还不够支持直接下结论"，用户以为是自己的问题缺证据，实际是我们把质检关了。**关掉一道校验不该让答案消失。** 降级原因用 FEATURE_DISABLED 而不是 LLM_ERROR，也是为了不把值班同学送去查模型配额——那边一切正常，排障会卡住。这是"对自己的判断能力做判断"，比 LLM 自我批评更接近工程意义上的反思。

**缺的部分：** 反思循环、重试上限、Reflexion 式的经验累积、把校验结论写回记忆以避免重犯——**项目中没有，不能硬套**。

**项目证据**
- `app/agent/rag_v2/graph.py:43-48` — 五节点线性边，`validate_answer → END`，无回边
- `app/agent/rag_v2/graph.py:44` — 唯一的 `add_conditional_edges` 用于子查询扇出，不是校验回环
- `app/agent/rag_v2/nodes.py:569-650` — `validate_answer_node` 一次性证据校验
- `app/agent/rag_v2/nodes.py:92, 103, 517, 556, 579, 586, 747, 757, 775, 781, 813` — `needs_second_retrieval` 十处全为写入/声明，无消费方
- `app/agent/rag_v2/nodes.py:576-588` — 校验不通过走降级输出，不走重试
- `app/agent/rag_v2/nodes.py:601-611` — 校验器被关闭时 fail-open，降级原因选 FEATURE_DISABLED 而非 LLM_ERROR
- `app/config.py:81` — `chat_total_budget_seconds = 90.0`，反思循环成本受此硬约束
- 反思循环 / 重试上限 / Reflexion 经验累积：项目中没有，不能硬套

#### A14 · 什么是 Multi-Agent

**回答**

**项目中没有 Multi-Agent，不能硬套。** 全仓 grep 无 supervisor、无 sub_agent、无 handoff、无 subgraph、无 Agent 间消息传递。`app/agent/aiops/` 目录下现在只剩 `__pycache__`，原来的 planner / executor / replanner / state 已在本轮重构中删除（见 git status 的 D 标记）。

仓库里有两条**单 Agent** 路径，它们的差别值得说，因为这正是"什么时候还不需要多 Agent"的真实素材：

**路径一，托管式循环。** `rag_agent_service.py:173` 用 `create_agent` 拼装，配 `MemorySaver` 做 checkpointer（`:115`、`:176`）。工具调用循环由框架驱动，模型自己决定调几次工具、什么时候停。适合开放式问答——步骤数不可预知。它还带一个指纹缓存（`:145`），配置指纹不变就复用同一个 agent 实例，不每个请求都重建。

**路径二，手写图。** `rag_v2/graph.py` 五个节点固定顺序：改写、扇出检索、去重、生成、校验。没有模型驱动的循环，控制流完全由代码决定。适合流程已知的场景——换来的是每一步都能插桩、能单独降级、能精确归因。

**为什么这两条都不是多 Agent：** 判据不在于有几个节点或几个并行分支，而在于**有没有多个独立的决策主体**。`rag_v2` 用 `Send` 做了子查询扇出（`graph.py:44`），跑起来是 N 个并行分支——但每个分支只是执行同一个检索函数，没有自己的目标、没有自己的工具集、没有决定"我要不要做"的权力。并行 ≠ 多 Agent。同理，`create_agent` 的多轮工具调用是一个主体的多次行动，不是多个主体的协作。

**多 Agent 真正引入的东西是什么：** 独立的决策权带来了状态一致性问题（两个 Agent 同时改一份 state 谁赢）、终止判定问题（谁来决定整体任务结束）、以及归因问题（最终答案错了是哪个 Agent 的锅）。本仓库现在这三个问题都不存在——因为只有一个决策主体。这不是能力缺失，是需求还没到。

**项目证据**
- Multi-Agent / supervisor / sub_agent / handoff / subgraph / Agent 间消息传递：项目中没有，不能硬套
- `app/agent/aiops/` — 目录下仅剩 `__pycache__`，planner/executor/replanner/state 已删除
- `app/services/rag_agent_service.py:10` — `from langchain.agents import create_agent`
- `app/services/rag_agent_service.py:115, 173, 176` — `MemorySaver` checkpointer 与 agent 装配
- `app/services/rag_agent_service.py:145` — 配置指纹相同则复用 agent 实例
- `app/agent/rag_v2/graph.py:43-50` — 五节点手写图，`compile()` 无 checkpointer
- `app/agent/rag_v2/graph.py:44` — `Send` 扇出是并行分支，非独立决策主体

#### A15 · 多 Agent 的三种协作方式

**回答**

原帖指的三种通常是**中心化编排（Supervisor）**、**去中心化协作（Swarm / 点对点 handoff）**、**层级式（Hierarchical，Supervisor 嵌套 Supervisor）**。

**项目中没有，不能硬套。** 三种都没有实现——没有 supervisor 节点、没有 handoff 工具、没有嵌套子图。

有一个东西容易被误认成中心化编排，必须先排除：`app/services/alert_diagnosis_orchestrator.py`。名字里有 orchestrator，实际是**线性流水线**，不是 Agent 编排。它的 `_PHASE_STATUS`（`:50`）和 `_PHASE_MESSAGES`（`:57`）是固定的阶段映射，`AlertDiagnosisOrchestrator`（`:65`）按顺序推进阶段并更新状态。整个类里没有任何"选择下一个执行者"的决策——阶段顺序写死在代码里。它编排的是**步骤**，不是 **Agent**。头部 docstring 里记录了一个 YAGNI 决策：没有为 `retrieved` / `diagnosed` 这些中间态发明独立状态，因为当时没有分支需求。

**如果要在本仓库落其中一种，哪种最合适：** 中心化。理由来自现有代码里的一个真实约束——`rag_v2` 的图 `compile()` 时**没有传 checkpointer**（`graph.py:50`）。去中心化 handoff 的前提是每个 Agent 能独立读写共享状态并在中断后恢复，没有持久化 checkpointer 就没有这个前提。而中心化编排可以把状态收在 supervisor 一处，对持久化的要求最低。层级式则要求先有稳定的单层 supervisor，跳过去做嵌套是自找归因困难。`[假设]` 这是设计判断，仓库无实现。

**还有一个成本判据值得说：** 中心化的代价是 supervisor 每次路由都要一次模型调用，N 步任务就是 N 次额外调用。本仓库 `config.py:81` 的 `chat_total_budget_seconds = 90.0` 是全局硬上限，单次 LLM 调用超时 60s——两次 supervisor 路由加上真正的执行调用就会顶到预算上限。所以在本仓库现有的预算结构下，即便要做中心化，也得先把路由改成规则判定而不是模型判定。

**项目证据**
- Supervisor / Swarm / Hierarchical 三种协作方式：项目中没有，不能硬套
- `app/services/alert_diagnosis_orchestrator.py:50` — `_PHASE_STATUS` 固定阶段映射
- `app/services/alert_diagnosis_orchestrator.py:57` — `_PHASE_MESSAGES` 固定文案
- `app/services/alert_diagnosis_orchestrator.py:65` — `AlertDiagnosisOrchestrator` 按写死顺序推进，编排步骤非 Agent
- `app/services/alert_diagnosis_orchestrator.py:1-40` — 头部 docstring 记录不为中间态发明状态的 YAGNI 决策
- `app/agent/rag_v2/graph.py:50` — `compile()` 无 checkpointer，缺去中心化协作的持久化前提
- `app/config.py:81` — 90s 全局预算约束路由调用次数
- 选型判断与成本分析：`[假设]`

#### A16 · 两种设计模式

**回答**

原帖上下文里指的是 **Supervisor 模式**和 **Swarm（去中心化）模式**这两种多 Agent 组织形态。

**项目中没有，不能硬套。** 承 A15，两种都无实现。

不过题目问"设计模式"，本仓库有两个**真实的、已落地的**架构模式选择，跟这题问的不是同一层，但可以说清边界后作答：

**模式一：托管循环 vs 手写图。** 这是同一个仓库里并存的两种 Agent 实现范式，前面 A14 已展开。选择依据是"步骤数是否可预知"：不可预知用 `create_agent` 让模型自己决定循环几次（`rag_agent_service.py:173`）；可预知用手写图换取每步可插桩、可单独降级（`rag_v2/graph.py:43-48`）。

**模式二：扇出-扇入（Scatter-Gather）。** `graph.py:44` 用 `Send` 把改写后的多个子查询扇出成并行分支，`graph.py:45` 全部汇聚到 `dedup`。这个模式的关键设计在扇入侧而不是扇出侧：`nodes.py:469-479` 的注释解释了为什么降级判定必须放在 `dedup` 而不是 `retrieve_each`——单个分支只知道"我失败了"，判断不了"是全都失败了，还是只有我失败"，而这两者的处置方向完全不同（前者是检索不可用，后者是证据不完整）。`dedup` 是 fan-in 之后的第一个节点，是唯一能看到全部分支结果的位置。

扇出-扇入在 LangGraph 上有一个必须知道的陷阱：**super-step 语义**。同一个 super-step 内任一并行分支抛异常，整个 super-step 失败，整张图崩。所以 `retrieve_each` 内部必须自己吞掉异常并把失败记进 `retrieve_failures`，让 `dedup` 去做全局判定——不能让异常穿出分支。这不是防御式编程的偏好问题，是框架语义的硬要求。

**项目证据**
- Supervisor / Swarm 两种多 Agent 模式：项目中没有，不能硬套
- `app/services/rag_agent_service.py:173` — 托管循环范式
- `app/agent/rag_v2/graph.py:43-48` — 手写图范式
- `app/agent/rag_v2/graph.py:44-45` — `Send` 扇出与 `dedup` 扇入
- `app/agent/rag_v2/nodes.py:449-467` — `dedup_node` 按 metadata id 或内容 md5 去重，截到 `FINAL_TOP_K`
- `app/agent/rag_v2/nodes.py:469-479` — 降级判定放在 fan-in 位置的完整理由，区分 PARTIAL_RETRIEVAL 与全失败

#### A17 · 单 Agent 与多 Agent 如何决策选型

**回答**

这题本仓库能给出真实答案，因为它实际做过这个选型，而且是**从多节点编排退回单路径**的方向。

**已发生的事实：** `app/agent/aiops/` 原来有 planner / executor / replanner / state 四个模块，是典型的"规划-执行-重规划"多阶段结构。本轮重构把它们全部删除（git status 里四个 D 标记），改成 `app/services/alert_diagnosis_orchestrator.py` 的线性流水线，158 行。头部 docstring 记录了这个决定的理由，核心是 YAGNI：replanner 存在的意义是"计划错了要改计划"，但告警诊断这个场景的步骤是固定的——拉指标、查 SOP、出报告，没有需要重规划的分支。为一个不存在的分支维护三个模块的状态传递，成本大于收益。

**从这次退回里提炼出的选型判据，按优先级：**

**第一判据是控制流是否真的会分叉。** 不是"任务是否复杂"，而是"是否存在需要在运行时决定走哪条路的点"。AIOps 诊断复杂，但不分叉，所以线性就够。这个判据最容易被误用——复杂度高的任务人们本能地想上多 Agent，但复杂度和分叉是两回事。

**第二判据是归因成本。** 单 Agent 出错，日志是一条线，`instrument_node` 在 `graph.py` 一处包住全部五个节点（注释写"这里是「所有节点注册进图」的唯一入口，包在这里漏不掉"），span 是扁平的。多 Agent 出错要先定位是哪个 Agent 的哪一步，再定位是不是交接时丢了信息。本仓库的可观测性只有四个 Prometheus 指标（`metrics.py:85, 94, 100, 110`），没有 per-Agent 维度，上多 Agent 就是在没有归因工具的情况下引入归因难题。

**第三判据是预算。** `config.py:81` 全局 90s，单次 LLM 调用超时 60s。多 Agent 的每一次交接通常伴随一次模型调用，两次交接就顶到预算。在这个预算结构下多 Agent 不是"更强"，是"更容易超时"。

**什么情况下我会选多 Agent：** 需要不同的**工具权限边界**时。比如诊断 Agent 只能读监控、执行 Agent 才能重启服务——这时用一个 Agent 加权限判断，等于把安全边界交给模型的指令遵循能力；拆成两个 Agent，边界是代码级的。本仓库现在没有写操作工具（`monitor_server.py` 全是只读的模拟指标），所以这个理由还不成立。`[假设]`

**项目证据**
- `app/agent/aiops/` — planner / executor / replanner / state 四模块已删除（git status D）
- `app/services/alert_diagnosis_orchestrator.py:1-40` — 头部 docstring 记录退回线性流水线的 YAGNI 理由
- `app/services/alert_diagnosis_orchestrator.py:50, 57, 65` — 固定阶段映射与线性推进，全文 158 行
- `app/agent/rag_v2/graph.py:30-42` — `instrument_node` 在唯一注册入口统一包裹五节点
- `app/core/metrics.py:85, 94, 100, 110` — 仅四个指标，无 per-Agent 维度
- `app/config.py:81` — `chat_total_budget_seconds = 90.0`
- `mcp_servers/monitor_server.py:124, 277` — 工具全为只读模拟指标，无写操作，权限边界理由尚不成立
- 多 Agent 选型的权限边界判据：`[假设]`

#### A18 · Agent 协作与动态切换机制

**回答**

**Agent 间的动态切换（handoff）项目中没有，不能硬套。** 没有 handoff 工具、没有 Agent 注册表、没有路由决策节点。

有三处**运行时切换**的真实实现，但切的都不是 Agent，得先把差别说清楚再谈：

**一、按开关切换执行路径。** `nodes.py:601-611`，`enable_answer_validation` 关掉时，`validate_answer_node` 走 fail-open 分支：保留答案、标记 `DegradeReason.FEATURE_DISABLED`、跳过模型调用。这是同一个节点内的两条路径，靠配置选，不涉及执行主体变化。注释里还记了一个易错点：这个开关判断放在两个 guard（答案为空、没有证据）之后，因为那两个是不花钱的结构性检查，不依赖被关掉的 LLM 调用，该继续生效。

**二、按配置指纹重建实例。** `rag_agent_service.py:145` 的注释写"指纹相同就直接复用,不会每个请求都重建一次 create_agent"。配置变了（模型、工具集、prompt）才重建。这是实例生命周期管理，不是运行时路由——重建后旧实例就不存在了，不是两个 Agent 并存互相切换。`:127` 的注释还提到临界区包含 await，说明这里有并发保护。

**三、降级路径切换。** `dedup_node`（`nodes.py:473-479`）根据分支失败情况选择 `PARTIAL_RETRIEVAL` 还是全失败降级，后续节点行为随之不同。这是数据驱动的分支，粒度是节点行为，不是 Agent 身份。

**这三处和真正的 Agent handoff 差在哪：** 差在**切换后的决策权归属**。上面三处切换后，做决定的还是同一段代码；handoff 切换后，做决定的是另一个有自己目标和工具集的主体。前者切的是"怎么做"，后者切的是"谁来做"。把配置开关说成动态切换机制，是把参数化冒充成架构。

**在本仓库落 handoff 缺什么：** 最硬的一块是持久化。`graph.py:50` 的 `compile()` 没有 checkpointer，进程重启后图的执行状态全丢。handoff 的语义是"A 把任务和上下文交给 B"，如果交接过程中崩了，既不知道 A 做完了没有，也不知道 B 收到了没有——`app/services/job_failure.py:30` 已经承认了一个同类问题：worker 崩溃时任务会卡在 RUNNING 状态，`main.py:26-49` 启动时也没有做恢复扫描。在这个基础上加 handoff，等于把一个已知的状态不一致问题乘以 Agent 数量。**先补 checkpointer 和启动恢复，再谈协作。** `[假设]`

**项目证据**
- Agent handoff / Agent 注册表 / 路由决策节点：项目中没有，不能硬套
- `app/agent/rag_v2/nodes.py:601-611` — 开关切换执行路径，fail-open，开关判断置于两个 guard 之后的理由
- `app/services/rag_agent_service.py:145` — 配置指纹相同则复用实例
- `app/services/rag_agent_service.py:127` — 临界区包含 await，并发保护
- `app/agent/rag_v2/nodes.py:473-479` — 数据驱动的降级路径选择
- `app/agent/rag_v2/graph.py:50` — `compile()` 无 checkpointer，缺 handoff 的持久化前提
- `app/services/job_failure.py:30` — worker 崩溃时任务卡在 RUNNING 的已知问题
- `app/main.py:26-49` — 启动无恢复扫描
- 落地路径判断：`[假设]`

#### 题库 A 原帖标记为"挖坑"未作答的四题

按写作规则第五条处理：原帖作者自己标注为未作答，这里不代作者补答案，只记录题面与本仓库的可用素材位置，供后续自行展开。

- **如何写好系统提示词** — 原帖未答。本仓库可用素材：`app/agent/rag_v2/nodes.py:80-105` 的校验器 prompt（含字段说明与判定规则）、`app/services/conversation_memory_service.py:28-35` 的摘要 prompt（含两条防污染硬规则与四段固定输出格式）、`nodes.py:619-625` 的 `MEMORY_POLICY` 与"仅将 evidence 作为可验证证据"的边界声明。
- **如何写好工具描述** — 原帖未答。本仓库可用素材：`app/tools/knowledge_tool.py:53` 的 `@tool(response_format="content_and_artifact")`、`app/tools/time_tool.py`、`mcp_servers/monitor_server.py:124` 与 `:277` 的两个 `@mcp.tool()` 定义。
- **思维链如何决定推理能力** — 原帖未答。本仓库无 CoT 实现，见 A7。
- **对记忆框架的了解**（Mem0 / Zep 等） — 原帖未答。本仓库未引入任何记忆框架，记忆层是手写的，见 A9。

### 题库 B-1 · 推理框架：CoT / ReAct / ToT（11 题）

> 来源：`https://www.cnblogs.com/itech/p/20111938`（全 44 题 + 5 道系统设计，按原文五个分组拆成 B-1 至 B-5）
> 这一组题原理成分最重，而本仓库**没有任何一处显式的 CoT / ReAct / ToT 实现**。所以下面每题都先说清仓库里到底有什么、缺什么，再谈原理，并把原理陈述和项目经验分开标注。

#### B1 · 为什么 Chain-of-Thought 能提升推理能力

**回答**

本仓库没有 CoT 实现，但有一处**结构上同源**的设计，我用它来回答这题会比背原理可信。

`VALIDATION_SYSTEM_PROMPT` 要求质检器输出的不是一个 `pass` 布尔值，而是八个字段：`coverage_score`、`groundedness_score`、`coverage_pass`、`groundedness_pass`、`needs_second_retrieval`、`unsupported_claims`、`missing_aspects`、`reason`。判定原则还写了六条，明确"哪些断言在证据里找不到依据就列进 `unsupported_claims`""问题的哪个维度没被覆盖就列进 `missing_aspects`"。

这等于强制模型**先把中间判断显式写出来，再给结论**——和 CoT 让模型先写推理步骤再给答案是同一个机制：把压在单个 token 里的判断摊开成多步，每一步都可以被下一步引用。

但这里有一层 CoT 讨论里常被忽略的收益，而它才是我真正加这些字段的原因：**中间字段是给人看的**。线上出现"答案被拦了"的时候，`unsupported_claims` 直接告诉我模型认为哪句话没证据，`missing_aspects` 告诉我该补什么文档。一个布尔值给不了这个，我只能去猜。所以在工程里，中间步骤显式化的价值有一半在可观测性上，不全在准确率上。

必须说清的边界：**我没有做过 A/B 对比**，说不出"输出中间字段比只输出布尔值准确率高多少"。这是设计判断，不是实验结论。项目里唯一跑过的量化对比是检索侧的（多查询 RRF 正向、CrossEncoder 精排负向），生成与判定侧没有量化过。

**项目证据**
- `app/agent/rag_v2/nodes.py:79-104` — `VALIDATION_SYSTEM_PROMPT`，八字段 + 六条判定原则
- `app/agent/rag_v2/nodes.py:99-100` — `coverage_score` / `groundedness_score` 的语义定义
- `app/agent/rag_v2/nodes.py:798-806` — `_parse_validation_result` 逐字段类型收敛
- `tests/eval/metrics.py:66-118` — 检索侧有量化指标（Hit@K / Recall@K / MRR）
- CoT 实现、推理链长度控制、中间步骤准确率对比实验：项目中没有，不能硬套

#### B2 · Zero-shot CoT 和 Few-shot CoT 有什么区别

**回答**

CoT 项目里没有，但 **zero-shot 与 few-shot 的取舍**在本仓库是真实做过的选择：**全部 prompt 都是 zero-shot，一处示例都没塞。**

四个系统 prompt（`RAG_SYSTEM_PROMPT`、`VALIDATION_SYSTEM_PROMPT`、`TOOL_FACTS_SYSTEM_PROMPT`、摘要 prompt）走的都是同一条路：给严格的输出格式规范 + 编号的判定原则，不给示例。

原因是 token 预算，而且这个账很好算。`validate` 节点的 prompt 里已经装了：问题 + 候选答案 + 证据片段（`rag_document_context_token_budget = 2800`）+ 会话记忆（`conversation_memory_context_token_budget = 1800`）。再塞两三个 few-shot 示例，示例本身就得带上"假的证据片段"才有意义，一个示例几百 token，直接从真实证据的预算里扣。**few-shot 在 RAG 场景里的成本不是"多几百 token"，是"少几段真证据"**，这是它和纯推理场景最大的区别。

代价是实打实的：不给示例，格式遵守率就不是 100%。所以 `_parse_validation_result` 必须写成四层降级——先试 ```` ```json ```` 围栏正则，再试裸 `{...}` 正则，再 `json.loads`，最后 `JSONDecodeError` 就 `logger.warning` + 落 `_default_validation(reason="质检器输出不可解析，已按未通过处理。")`。`_parse_sub_queries` 是同一套结构，失败落 `[fallback]`。

这两段代码就是"选了 zero-shot"要付的账。反过来说，如果我不写这套降级，就不该选 zero-shot。

**项目证据**
- `app/agent/rag_v2/nodes.py:79-104` — zero-shot：只有格式规范和判定原则，无示例
- `app/agent/rag_v2/nodes.py:106-108` — `MEMORY_POLICY`，同样是规则式约束
- `app/config.py:156` — `rag_document_context_token_budget: int = 2800`
- `app/config.py:159` — `conversation_memory_context_token_budget: int = 1800`
- `app/agent/rag_v2/nodes.py:784-814` — `_parse_validation_result` 四层降级
- `app/agent/rag_v2/nodes.py:762-782` — `_parse_sub_queries` 同构降级
- few-shot 示例库、示例选择策略、示例数量与效果的对比：项目中没有，不能硬套

#### B3 · CoT 有哪些局限

**回答**

按原理说，CoT 的局限通常列三条：错误会沿链条传播、增加 token 与延迟、小模型上可能反而变差。前两条在本仓库有真实对应物，我只讲这两条。

**第一条，错误沿链条传播——项目里的对应物比 CoT 更严重。** 单轮 CoT 的错误随这一轮结束就消失了，而会话摘要的错误会被**反复注入到之后每一轮**。摘要 prompt 里那两条硬规则就是为这个写的：`不要把模型猜测、SOP 内容或历史数值写成实时事实；发生冲突时以较新的用户消息为准`。`MEMORY_POLICY` 在 generate 和 validate 两个节点里又重申一次：`它不是知识库证据，更不是实时监控、日志或工具查询结果；不得将历史指标写成当前实时值`。

同一条约束写在两个地方不是冗余——摘要 prompt 管"写进摘要时别写错"，`MEMORY_POLICY` 管"读到摘要时别当成实时数据"。写入侧和读取侧都得设防，因为一条错误事实一旦落进摘要，后面每一轮都会读到它。

**第二条，token 与延迟——在有总预算的系统里这是硬约束。** `rag_v2` 一次请求固定三次串行 LLM 调用（rewrite / generate / validate），整个请求被 `asyncio.timeout(chat_total_budget_seconds = 90.0)` 包着。如果让质检器输出长推理链，它是第三次调用，前两次已经花掉的时间它拿不回来，最容易撞预算。所以 `VALIDATION_SYSTEM_PROMPT` 最后一条明确写 `不要输出额外解释，不要使用 markdown`——这句话就是在**主动压制** CoT 式的展开。

也就是说，本仓库对 CoT 的态度不是"没用上"，是在判定环节**显式禁止**了它，理由是延迟预算。

**项目证据**
- `app/services/conversation_memory_service.py:28-35` — 摘要 prompt 的两条防污染硬规则
- `app/agent/rag_v2/nodes.py:106-108` — `MEMORY_POLICY`，读取侧重申
- `app/agent/rag_v2/nodes.py:104` — `不要输出额外解释，不要使用 markdown`
- `app/config.py:81` — `chat_total_budget_seconds: float = 90.0`
- `app/agent/rag_v2/graph.py:44-48` — 三次串行 LLM 调用的固定路径
- CoT 实现、链条长度与错误率的关系测量、小模型上的 CoT 退化验证：项目中没有，不能硬套

#### B4 · 什么是 Auto-CoT

**回答**

**项目中没有，不能硬套。** 仓库里没有任何示例聚类、自动选样、示例池的代码。全仓无 few-shot 示例（见 B2），也就谈不上自动构造示例。

需要主动排除一个容易被拿来冒充的模块：`app/services/metadata_enricher.py` 会给每个 `Document.metadata` 注入 `doc_tags` 和 `chunk_summary`，看起来像"自动生成的中间产物"。但它的用途是**给检索加过滤维度和摘要**，产物进的是 Milvus 的字段，不是进 prompt 的示例位。Auto-CoT 的核心是"聚类问题空间、每簇选一个代表、自动生成它的推理链"，这三步在本仓库一步都没有。拿 `metadata_enricher` 说成 Auto-CoT 是偷换概念，面试里一追问"那你的示例是怎么选进 prompt 的"就露底。

原理层面我能说的只有一句：Auto-CoT 解决的是 few-shot CoT 里"人工写示例贵且有偏"的问题，靠对问题做聚类采样来保证示例覆盖面。这是原理陈述，不是项目经验。

**项目证据**
- `app/services/metadata_enricher.py:37` — 产物是 `doc_tags` + `chunk_summary`，进检索字段而非 prompt 示例位
- 示例聚类、自动选样、示例池、推理链自动生成：项目中没有，不能硬套

#### B5 · ReAct 和纯 CoT 的核心区别

**回答**

本仓库的两条链路正好是这个对比的实物，虽然一条是 ReAct 型、另一条连 CoT 都不是。

`rag_agent_service.py` 用 `create_agent` + `MemorySaver`，模型自己决定要不要调 `retrieve_knowledge`、调几次、什么时候停——ReAct 型，控制流在模型手里。`rag_v2` 是五节点固定 DAG，`rewrite → retrieve_each → dedup → generate → validate_answer → END` 全是 `add_edge` 静态声明的，模型只负责三件具体工作，不负责路由。

核心区别我实际感受到的是**外部信号的有无**：ReAct 每一步 Observation 都是工具返回的真实数据，模型推错了下一步会被数据打回来；纯 CoT 全程在模型内部，错了没有任何外部纠正信号。

而这条区别在工程上有一个直接后果，就是 `knowledge_tool.py` 里为什么要为"没找到"和"检索不可用"写两句不同的话。`_EMPTY_HINT = "知识库中没有与该问题相关的内容。"` 是真实业务结论；`_UNAVAILABLE_HINT` 写的是 `检索服务当前不可用（错误码: {code}）…请如实告知用户知识库暂时无法访问，不要凭记忆作答。`

如果 Milvus 挂了却回一句"没有找到相关信息"，模型会据此得出"知识库里没有这个内容"——**Observation 被污染了，ReAct 就退化成了 CoT**：模型以为自己拿到了外部事实，实际拿到的是一句假话，而它没有任何办法察觉。这是我理解的两者最要紧的差别：ReAct 的可靠性完全建立在 Observation 的诚实度上，Observation 一撒谎，ReAct 反而比 CoT 更危险，因为它带着"我查过了"的虚假置信。

**项目证据**
- `app/services/rag_agent_service.py:173-176` — `create_agent(..., checkpointer=self.checkpointer)`，ReAct 型控制流
- `app/agent/rag_v2/graph.py:43-48` — 五条静态边，非 ReAct
- `app/tools/knowledge_tool.py:44-50` — 两句措辞的区分
- `app/tools/knowledge_tool.py:97` — 不可用时返回 `_UNAVAILABLE_HINT.format(code=code)`
- `app/tools/knowledge_tool.py:101` — 真实空结果返回 `_EMPTY_HINT`
- `app/tools/knowledge_tool.py:1-30` — 文件头 docstring 记录了"不许用假原因掩盖真原因"的纪律
- 手写 ReAct 循环、Thought/Action/Observation 三段的显式解析：项目中没有，不能硬套

#### B6 · 如何手写一个 Thought-Action-Observation 循环

**回答**

**项目中没有手写的 ReAct 循环，不能硬套。** `create_agent` 内部确实有这个循环，但那是 LangChain 框架实现的，我只是传了 `tools`、`model`、`checkpointer` 三个参数，循环本身不是我写的代码，我不能把它算成自己的实现经验。

不过这题值得答的是：本仓库有几个件，如果真要手写这个循环，它们是必需的，而其中一半是我在别处踩过坑才知道要加的。

**已经有的、可以直接搬过去的：**

工具侧返回值绝不抛异常。`retrieve_knowledge` 把 `AppError` 转成 `_UNAVAILABLE_HINT` 字符串返回（`knowledge_tool.py:97`），MCP 侧的 `retry_interceptor` 更彻底，失败时返回 `CallToolResult(isError=True)` 而不是 raise。理由是 Observation 是循环的输入，工具一抛异常，循环就断在半路，模型连"这一步失败了"这个信息都拿不到。

熔断器包在调用外侧。`with retrieval_breaker.guard():` 在 `with observe_tool_call(TOOL_NAME):` 外面（`knowledge_tool.py:78-79`）。手写循环时这个顺序更重要——循环可能连着调同一个工具好几轮，下游挂了的话，熔断器能让第二轮开始直接快速失败，而不是每轮都白等一次完整超时。

单次调用超时和总预算分开。`llm_factory.create_chat_model` 管单次（`config.llm_timeout_seconds`），`service.py` 的 `asyncio.timeout` 管整个请求（90s）。`llm_factory.py:43` 的注释写了原因：`timeout 管的是单次调用；三次串行调用最坏情况是 3×timeout`。ReAct 的轮数不固定，这个乘数就是不确定的，总预算是唯一的兜底。

**[假设] 还需要补的：** 步数上限（见 B7，本仓库确实没有）、重复动作检测（同一个 `(tool_name, args)` 连续出现就中止）、以及每轮 Observation 的长度裁剪——这一条本仓库在别处有同构实现可以照搬，`_take_tail_to_budget` 那套"按预算装载 + 单条超预算就裁剪而不是丢弃"的逻辑（`conversation_memory_service.py:249-272`），因为 Observation 累积起来撑爆上下文是必然会发生的，不是可能。

**项目证据**
- `app/services/rag_agent_service.py:173-176` — 循环由 `create_agent` 提供，非手写
- `app/tools/knowledge_tool.py:97` — 工具失败返回字符串而非抛异常
- `app/agent/mcp_client.py:106` — `retry_interceptor`，失败返回 `CallToolResult(isError=True)`
- `app/tools/knowledge_tool.py:78-79` — 熔断在外、计时在内的嵌套顺序
- `app/core/llm_factory.py:84-85` — 单次超时 `is None` 判断而非 `or`
- `app/core/llm_factory.py:43-46` — 单次超时与总预算的关系说明
- `app/services/conversation_memory_service.py:249-272` — 可复用的预算装载逻辑
- 手写循环、步数上限、重复动作检测、Observation 裁剪：项目中没有，`[假设]`

#### B7 · 如何设置最大步数，如何判定死循环

**回答**

这题我按真实缺口答，因为本仓库的状态比标准答案有信息量。

**结论先说：本仓库没有任何步数上限。** 全仓 grep `recursion_limit`、`max_iterations` 零命中。`create_agent` 那条链路走的是 LangGraph 的默认递归上限，我没有显式配过，也就是说这个值不是我选的。

但缺口的分布不均匀，这一点很重要：

**`rag_v2` 结构上不可能死循环。** 它是 DAG，边全是静态的，`validate_answer → END` 是终边，没有任何一条回边。所以这条链路不需要步数上限，需要的是别的东西——而它有：`asyncio.timeout(90.0)` 总预算，加上固定三次 LLM 调用，最坏耗时可以事前算出来。

**真正暴露在死循环风险下的是 `create_agent` 那条**，而恰恰是它没配上限。

还有一个更细的点，也是我觉得这题真正该讲的：**总预算和步数上限防的不是同一件事，不能互相替代。**

90s 超时能保证请求不会永远挂着，但它防不了"在 90 秒内空转 15 轮工具调用"。那 15 轮每一轮都真实发生了——真的打了 Milvus、真的调了模型、真的花了钱、真的写了日志——最后超时把整个请求丢掉，用户什么也没拿到，而账单已经产生了。总预算是**时间**的闸门，步数上限是**动作次数**的闸门，缺后者的系统会在预算内烧钱。

`[假设]` 要补的话我会加两层，而不只是一个计数器：硬上限（`recursion_limit`，防结构性失控）+ 重复动作检测（同一个 `(tool_name, args)` 连续命中两次就中止）。只有计数器的话，模型反复用同一个参数查同一个工具，会一直查到上限才停，中间每一轮都是纯浪费；重复检测能在第二轮就切断。

**项目证据**
- 全仓 grep `recursion_limit` / `max_iterations` — 零命中，无步数上限
- `app/agent/rag_v2/graph.py:43-48` — DAG 结构，`validate_answer → END` 无回边
- `app/config.py:81` — `chat_total_budget_seconds: float = 90.0`，时间闸门
- `app/services/rag_agent_service.py:173-176` — 未传 `recursion_limit`
- 步数上限、重复动作检测、单请求 LLM 调用次数上限：项目中没有，不能硬套；补法为 `[假设]`

#### B8 · ReAct 和 Function Calling 是什么关系

**回答**

我在项目里的理解是分层的：**Function Calling 是"模型怎么表达它要调工具"的协议层，ReAct 是"什么时候调、调完了怎么接着走"的控制层。** 前者是一次消息的格式，后者是多次消息之间的循环。可以有 Function Calling 而没有 ReAct（调一次就结束），反过来不行。

本仓库最能说明这个分层的是 `@tool(response_format="content_and_artifact")` 这个选择。

它让 `retrieve_knowledge` 的返回值分成两条通道：`content` 是给模型看的格式化文本，`artifact` 是 `List[Document]` 原始文档，**不进模型上下文**。

选它的理由正好落在两层的分界上：ReAct 循环只需要 `content`——那是 Observation，模型拿它决定下一步。而 API 响应需要原始 `Document` 做引用溯源，前端要显示"这段答案来自哪个文件哪一段"。

如果只有一条返回通道，两个需求会互相挤：把完整文档序列化进 `content`，模型上下文被塞爆，而且 ReAct 多调几轮就直接超预算；只回摘要文本，前端拿不到溯源数据，`search_results` 那个 SSE 事件就没内容可发。

`content_and_artifact` 是协议层提供的机制，用来同时喂饱控制层（Observation）和应用层（溯源）。这也说明两层不是简单的上下包含关系——协议层的设计得同时考虑"循环要什么"和"循环之外的系统要什么"。

**项目证据**
- `app/tools/knowledge_tool.py:53-54` — `@tool(response_format="content_and_artifact")`，返回 `Tuple[str, List[Document]]`
- `app/tools/knowledge_tool.py:97` / `:101` — `content` 通道的两种措辞
- `app/api/chat.py:103-106` — `search_results` SSE 事件消费 artifact 侧数据
- `app/api/chat.py:95-98` — `tool_call` SSE 事件
- `app/agent/mcp_client.py:336` / `app/config.py:191-201` — MCP 工具经 `MultiServerMCPClient` 注册，同样走 Function Calling 协议
- ReAct 控制层的自研实现：项目中没有，不能硬套（框架提供，见 B6）

#### B9 · Tree-of-Thought 和 CoT 的区别

**回答**

**ToT 项目中没有，不能硬套。** 仓库里没有任何多分支候选生成、评分、回溯的实现。

但这题有一个本仓库的真实状态值得讲，它比原理对比更有意思：**我生成了回溯信号，却没有接上回溯路径。**

`validate_answer` 节点输出的 `validation` 字典里有一个字段 `needs_second_retrieval`，语义就是 ToT 里的回溯判定——"当前这条路的证据不足，应该退回去重新检索"。`VALIDATION_SYSTEM_PROMPT` 第 5 条明确要求模型填它：`当 coverage_score 明显不足时，needs_second_retrieval=true`。代码里五处会设置它：`_default_validation` 按参数传入、答案为空时传 `not docs`、无证据时传 `True`、解析失败时传 `False`、正常解析时从模型输出取。

然后全仓 grep 这个字段，**十处引用全是写入方，没有一处读取方**。`graph.py:48` 是 `add_edge("validate_answer", END)`，没有条件边、没有回到 `retrieve_each` 的路径。

所以真实情况是：判定逻辑完整实现了、prompt 里教了模型怎么判、字段一路收敛到返回值里，**但树被剪成了单路径**。信号发出去，没人接。

我认为这个状态本身就是对"ToT 和 CoT 区别"最实在的回答：区别不在于"能不能判断这条路走不通"——单路径系统也能判断，我这里就判断了；区别在于**判断之后有没有第二条路可走**。ToT 贵的地方是维护多个候选状态和回溯路径，不是评分。我把便宜的那半做了，贵的那半没做。

诚实说这算半个技术债。不接的原因是接上就意味着图里出现回边，而回边一出现，B7 那个"没有步数上限"的问题立刻从潜在风险变成现实风险——`rag_v2` 会从"结构上不可能死循环"变成"可能死循环"。所以这两件事得一起做，不能只做一件。

**项目证据**
- `app/agent/rag_v2/nodes.py:103` — prompt 第 5 条要求填 `needs_second_retrieval`
- `app/agent/rag_v2/nodes.py:92` — 输出 JSON 模板中的该字段
- `app/agent/rag_v2/nodes.py:517` / `:556` / `:579` / `:586` — 四处写入
- `app/agent/rag_v2/nodes.py:747` / `:757` / `:781` / `:813` — 解析与默认值中的写入
- `app/agent/rag_v2/graph.py:48` — `add_edge("validate_answer", END)`，无回边、无消费方
- 多候选生成、候选评分排序、回溯执行：项目中没有，不能硬套

#### B10 · ToT 里怎么选 BFS 还是 DFS

**回答**

**ToT 项目中没有，不能硬套**，搜索策略的选择更谈不上。

能对上的只有一点：本仓库唯一的"宽度扩展"是 `_fanout_to_retrieve` 用 `Send` 把 3 个子查询扇成 3 个并行分支，`retrieve_each` 跑完立刻 fan-in 到 `dedup`。用 ToT 的话说是**一层 BFS，深度恒为 1**，之后不再展开。

值得说的是我为什么没往深走，因为原因不是"没需求"，是 `Send` 扇出的一个真实约束：**同一次扇出的分支属于同一个 super-step，任一分支抛异常，整个 super-step 失败，整张图崩。**

所以 `retrieve_each` 内部必须把所有异常吞掉，把失败写进 `retrieve_failures` 让下游判定。而"谁来判定"这件事也不能随便放——`dedup_node` 里的注释写清了：单个分支只知道"我失败了"，判断不了"是全都失败还是只有我失败"，而这两者处置方向完全不同（全失败是 `retrieval_failed`，部分失败是 `PARTIAL_RETRIEVAL`）。`dedup` 是 fan-in 之后第一个节点，是唯一能看到全部分支结果的位置。

把这个约束往深处推演：树深两层就是"扇出的分支里再扇出"，每一层都得自己吞异常、自己判断局部失败还是全局失败，而**判定所需的全局视野只在每一层的 fan-in 点存在**。三层树就要三个 fan-in 判定点，每个判定点的语义还不一样（某个子树全废 vs 整棵树全废）。这个复杂度是指数级涨的，不是线性的。

所以我的选择是：宽度只做一层，深度不做。这不是搜索策略的取舍，是"并行错误处理的复杂度上限"的取舍。真要答 BFS/DFS 的原理选择，那是原理陈述，我没有实践支撑。

**项目证据**
- `app/agent/rag_v2/graph.py:19-22` — `_fanout_to_retrieve` 返回 `[Send("retrieve_each", ...)]`
- `app/agent/rag_v2/graph.py:44-45` — 扇出后立即 fan-in 到 `dedup`，深度 1
- `app/agent/rag_v2/nodes.py:469-472` — 为什么降级判定必须放在 fan-in 点的注释
- `app/agent/rag_v2/nodes.py:473-479` — `retrieve_failures` 与 `PARTIAL_RETRIEVAL` 的区分
- `app/agent/rag_v2/nodes.py:28` — `NUM_SUB_QUERIES = 3`，扇出宽度写死
- BFS / DFS 搜索、多层树展开、剪枝策略：项目中没有，不能硬套

#### B11 · 什么场景该用 ToT，成本怎么控制

**回答**

场景判断我只能给原理性的回答（ToT 项目里没有），但**成本控制**这半在本仓库有完整的真实落点，而且正好能说明 ToT 为什么在我这个场景里装不进去。

本仓库现在的成本护栏有三层，而且三层是互相咬合的：

一是**调用次数固定**。`rag_v2` 每次请求恰好三次 LLM 调用，因为图是静态 DAG。这条让成本可以事前算，不是事后统计。

二是**时间总预算**。`asyncio.timeout(chat_total_budget_seconds = 90.0)` 包住整个请求，`llm_timeout_seconds` 管单次。两层分开是因为 `llm_factory.py:43` 那条：单次超时乘以调用次数才是最坏耗时，而只有调用次数固定时这个乘法才有意义。

三是**token 三层预算**。文档上下文 2800、会话记忆 1800、摘要 700，`_render_context` 还会先扣掉标题开销再分配剩余预算，`_take_tail_to_budget` 按"越新越重要"贪心装载。

ToT 会同时破坏第一层和第二层：候选数 × 深度决定调用次数，而这个乘积依赖模型的展开决策，事前算不出来；调用次数一不固定，第二层的"单次超时 × 次数"就失去意义，只剩 90s 硬截断这一道，而硬截断的后果是**钱花了、结果丢了**（同 B7）。

所以对我这个场景的结论很直接：知识库问答的答案质量瓶颈在检索召回，不在推理深度。量化数据支持这个判断——检索侧的多查询 RRF 是正向的，说明"多找几次"有效；而生成侧我从来没观察到"模型想得不够深"导致的错误，观察到的是"证据不够"导致的错误，`unsupported_claims` 和 `missing_aspects` 记的都是这类。给证据不足的问题加推理深度，是在错的地方花钱。

`[假设]` 如果真要引入 ToT，我会先补三样，缺一样都不能上：候选数与深度的硬上限（否则第一层护栏直接失效）、按 token 计的显式成本闸门（现在只有时间闸门，见 B7）、以及分支中止后的部分结果回收——现在 90s 超时是整个请求丢掉，ToT 下丢掉的是已经算完的多个候选，浪费成倍放大。

**项目证据**
- `app/agent/rag_v2/graph.py:43-48` — 静态 DAG，调用次数固定为 3
- `app/config.py:81` — `chat_total_budget_seconds: float = 90.0`
- `app/core/llm_factory.py:43-46` — 单次超时与总耗时的乘法关系
- `app/core/llm_factory.py:84-85` — `is None` 判断保证 `timeout=0` 不被吞掉
- `app/config.py:156` / `:159` / `:160` — 三层 token 预算
- `app/services/conversation_memory_service.py:207-247` — `_render_context` 预留标题开销后再分配
- `app/services/conversation_memory_service.py:249-272` — `_take_tail_to_budget` 贪心装载
- `app/agent/rag_v2/nodes.py:79-104` — `unsupported_claims` / `missing_aspects` 记录的是证据问题
- ToT 场景实践、候选数与深度上限、按 token 的成本闸门、分支部分结果回收：项目中没有，不能硬套；补法为 `[假设]`

### 题库 B-2 · Agent 架构：记忆、规划、工具（11 题）

> 来源：`https://www.cnblogs.com/itech/p/20111938`（原文编号 12-22）

#### B12 · 如何设计一个包含短期记忆和长期记忆的系统

**题目**：如何设计一个包含短期记忆和长期记忆的 Agent 记忆系统？

**回答**

我这个仓库里只有**短期记忆**是真的，长期记忆没有。先说真的那部分怎么设计的。

短期记忆的载体是 `ConversationMemoryService`。它的核心设计决定是**存储和注入分离**：完整原始消息只写进 PostgreSQL 的 `conversation_messages`，作为审计记录；喂给模型的上下文只包含"压缩摘要 + 未压缩消息的尾部"。这条纪律写在文件头的 docstring 里（`conversation_memory_service.py:1-5`）。分离的理由是这两件事的约束不同——审计要求不丢、不改、可追溯，模型上下文要求短。用同一份数据同时满足两个约束会同时做坏两件事。

具体结构是三层预算嵌套（`config.py:156/159/160`）：知识库文档 2800 token、会话记忆总量 1800 token、其中摘要部分最多 700 token。`_render_context`（`:207-247`）分配时会**先扣掉标题的开销再算剩余额度**——`<conversation_summary>` 这类标签本身也占 token，不预留就会超。

摘要的推进用游标而不是重算：`summarized_through_message_id` 记住"摘要覆盖到哪条消息为止"，`_messages_after_snapshot`（`:133-144`）只取游标之后的增量。这里有个刻意的保守决定：游标对应的消息被软删除时**不注入旧原文**，直接 `return []`，宁可少给上下文也不重复历史。

最近窗口按**轮次**而不是**条数**算（`_recent_message_count`，`:146-158`），从后往前数 user 消息。按条数会把一问一答切一半，模型看到一个没有答案的提问，比看不到更糟。

**长期记忆项目中没有，不能硬套。** 跨会话的用户画像、按需向量召回历史对话、记忆的重要性评分与衰减、过期删除策略——全都没有。这里要特别声明一件事：仓库里**有** Milvus 向量检索，但它的检索目标是知识库文档，不是对话历史。拿它冒充"长期记忆的向量召回"是偷换概念，两者的数据源、写入时机、生命周期都不一样。

`[假设]` 要补长期记忆，我会从**明确事实**这一段开始——摘要 prompt 已经把输出切成【明确事实】【任务进展】【约束与偏好】【待确认项】四段（`:28-35`），其中"约束与偏好"天然是跨会话该保留的，"任务进展"天然是不该跨会话的。这个切分已经隐含了长短期的边界，落地时不用重新设计分类。

**项目证据**
- `app/services/conversation_memory_service.py:1-5` — 存储与注入分离的纪律
- `app/services/conversation_memory_service.py:38-45` — `ConversationMemoryContext` 四字段
- `app/services/conversation_memory_service.py:93` — `build_context` 入口
- `app/services/conversation_memory_service.py:104-126` — 游标与增量合并
- `app/services/conversation_memory_service.py:133-144` — 软删除时保守返回 `[]`
- `app/services/conversation_memory_service.py:146-158` — 按轮次算最近窗口
- `app/services/conversation_memory_service.py:207-247` — `_render_context` 预留标题开销
- `app/services/conversation_memory_service.py:28-35` — 摘要输出的四段格式
- `app/config.py:156` / `:159` / `:160` — 三层 token 预算
- `app/models/conversation.py:125` — 摘要表只存注入用内容，原文另存
- 长期记忆、用户画像、对话历史向量召回、重要性评分、过期删除：项目中没有，不能硬套

#### B13 · 上下文窗口溢出时怎么处理

**题目**：当上下文窗口即将溢出时，Agent 应该如何处理？

**回答**

我这里不是"即将溢出时处理"，而是**从来不让它接近溢出**。这个区别是设计上的：溢出检测是事后补救，需要一个"当前用了多少"的实时读数和一条紧急裁剪路径；预算分配是事前约束，每一段内容在拼进 prompt 之前就已经被限制住了。我选后者，因为前者要求我能准确知道模型侧的真实 token 数，而我只有估算。

三层预算（`config.py:156/159/160`）加起来是 2800 + 1800 = 4600 token 的上限，其中会话记忆内部再切出 700 给摘要。剩下的问题文本和 system prompt 是有界的。所以总量在拼装前就封住了。

估算函数是手写的，按 CJK 加权（`estimate_tokens`，`:48-54`）：中日韩字符按 1 个 token 算，其他字符按 4 个字符 1 token。这是保守估算，不是精确计数——它会略微高估中文，而高估在这个方向上是安全的。

裁剪策略在 `_take_tail_to_budget`（`:249-272`），从最新往旧装，装不下就停。**关键的一个决定在 `:260-269`**：单条消息本身就超预算时，不丢弃，而是截断后保留。理由是丢掉最新那条用户消息等于让模型回答一个它看不见的问题，而截断至少保住了问题的开头。形式上的最优装载在这里让位给业务约束。

检索侧同构：`_format_context`（`nodes.py:903-949`）按同样的贪心装载组织文档，依赖的偏序是相关性排序而不是时间。

摘要失败时的处置也算这个题的一部分：`_summarize`（`:163-186`）整段包在 try/except 里，失败返回空串，**退回最近消息窗口**。注释写得很直接——"摘要是节省 Token 的优化，不可阻断主聊天"。这意味着溢出压力最大的时候（摘要服务挂了）反而是走原文的时候，所以最近窗口本身的预算必须独立够用，不能依赖摘要成功。

**项目中没有的部分**：运行时的实时 token 用量读数、溢出告警、模型侧真实 token 数的回读（DashScope 返回的 usage 我没有采集）。所以我说不出"实际有没有触发过溢出"。

**项目证据**
- `app/config.py:156` / `:159` / `:160` — 事前预算而非事后检测
- `app/services/conversation_memory_service.py:48-54` — CJK 加权保守估算
- `app/services/conversation_memory_service.py:249-272` — 尾部贪心装载
- `app/services/conversation_memory_service.py:260-269` — 单条超预算截断而非丢弃
- `app/services/conversation_memory_service.py:274-289` — `_truncate_text`
- `app/services/conversation_memory_service.py:163-186` — 摘要失败退回原文窗口
- `app/agent/rag_v2/nodes.py:903-949` — `_format_context` 同构装载
- `app/agent/rag_v2/nodes.py:950` — `_estimate_tokens` 检索侧的同类估算
- 实时 token 用量读数、溢出告警、模型侧 usage 采集：项目中没有，不能硬套

#### B14 · 什么是记忆污染，怎么防止

**题目**：什么是记忆污染（Memory Poisoning）？如何防止？

**回答**

这题我有真实的防护代码，而且防的正是最容易被忽略的那一类污染：**模型自己的猜测被写进摘要，下一轮当成事实读回来。**

摘要 prompt 里有两条硬规则（`conversation_memory_service.py:28-35`）：

> 不要把模型猜测、SOP 内容或历史数值写成实时事实；发生冲突时以较新的用户消息为准。

三个被点名的污染源各有各的机制：

- **模型猜测**——上一轮模型说"可能是内存泄漏"，如果摘要写成"问题是内存泄漏"，下一轮它会基于这个假前提继续推理，而且再也不会回头质疑。
- **SOP 内容**——SOP 是知识库里的处置流程文档，写的是"应该怎么做"。混进记忆会变成"已经做了什么"。
- **历史数值**——上一轮查到 CPU 78%，写进摘要后下一轮问"现在多少"，模型会答 78%。时间维度被抹掉了。

冲突处理是"以较新的用户消息为准"，这条给了模型一个明确的优先级，而不是让它自己权衡。

第二道防线在注入侧。`MEMORY_POLICY`（`nodes.py:106-108`）会拼进 generate 和 validate 两个节点的 prompt：

> 会话记忆只用于理解用户已明确说明的对象、任务延续和约束。它不是知识库证据，更不是实时监控、日志或工具查询结果；不得将历史指标写成当前实时值。

这是**同一条约束在两个位置各说一遍**——写入时约束摘要器，读取时约束使用者。单点约束在这里不够，因为摘要是 LLM 生成的，它不保证遵守规则；注入侧再说一遍，等于假定上游可能失守。

validate 节点还有第三层（`nodes.py:684` 附近的 user_prompt 构造）：明确写"仅将 `<evidence>` 中的内容作为知识库可验证证据；会话记忆只能帮助判断对象指代"。质检器如果把记忆当证据，会给一个证据不足的答案打高分——污染就从记忆传染到了质检结论。

同源的纪律在 AIOps 侧也有：`aiops_report.py:69` 规定模型推断只能标记为 `model`，真实 SOP 证据由检索代码注入而非模型自述。`first_response_service.py:17` 和 `:272` 是这条的落点——SOP 证据在代码里合并成 `VERIFIED_FACT`，模型没有权限给自己的输出盖这个章。

**项目中没有的部分**：污染的检测与回滚（发现摘要写错了没有修正路径）、对用户输入的注入检测、多轮之间摘要一致性的自动校验。都是**项目中没有，不能硬套**。

**项目证据**
- `app/services/conversation_memory_service.py:28-35` — 摘要 prompt 的两条防污染硬规则
- `app/agent/rag_v2/nodes.py:106-108` — `MEMORY_POLICY` 注入侧约束
- `app/agent/rag_v2/nodes.py:619-625` — validate 节点的证据边界声明
- `app/models/aiops_report.py:69` — 模型推断只能标 `model`
- `app/services/first_response_service.py:17` — 模型推断与已验证事实始终区分
- `app/services/first_response_service.py:272` — SOP 证据由代码注入为 `VERIFIED_FACT`
- 污染检测与回滚、输入注入检测、摘要一致性自动校验：项目中没有，不能硬套

#### B15 · 向量检索、关键词检索、混合检索怎么取舍

**题目**：向量检索、关键词检索、混合检索各自的适用场景和取舍是什么？

**回答**

这题我有实测数据，而且结论和"混合检索一定更好"的通行说法有出入，所以我更愿意讲实际发生的事。

现在跑的是混合检索：Milvus 向量 + BM25 关键词，用 LangChain 的 `EnsembleRetriever` 做 RRF 融合，权重 0.7 / 0.3（`vector_search_service.py:43`，`:96-98`）。向量权重更高是因为我的知识库是运维 SOP 文档，用户提问用的词和文档里的词经常对不上（"服务起不来" vs "service_unavailable"），纯 BM25 在这种场景下召回很差。BM25 留 0.3 是为了保住精确匹配的部分——错误码、接口路径、指标名这些 token 是不能靠语义近似的，`CPU_HIGH_USAGE` 和 `MEMORY_HIGH_USAGE` 在向量空间里很近，在 BM25 里完全不同。

BM25 这边有个工程上不太显然的问题：它需要**全量语料在内存里**。`_load_documents_from_milvus`（`:165`）按 1000 一批把 Milvus 里所有分片读出来建索引。这带来两个后果：一是启动成本，二是**缓存失效判定**。我用 `collection.num_entities` 作版本号（`:140-162`），数量变了就重建。这是个粗判据——改一条文档内容而总数不变时它检测不到。我选它是因为它是 Milvus 免费给的，精确的内容版本号需要我自己维护一套东西。

降级路径：BM25 语料为空时退回纯向量检索（`:93`），日志记 warning。注意这是**成功**路径而不是失败——所以熔断器不该看见它（这条纪律写在 `knowledge_tool.py:78` 附近的注释里，guard 放在调用点而不是检索服务内部，就是为了不让内部的局部降级触发熔断）。

`EnsembleRetriever` 有个限制：**它不暴露 RRF 的融合分数**。所以 `search_similar_documents`（`:109-139`）里的相关性分数是我用排序名次算的（`:129`），不是真的相似度。这一点在讲"我们有相关性分数"时必须说清楚，否则是在报一个假指标。

实测结论（记在 `tests/eval/`）：**多查询 + RRF 是正向的，CrossEncoder 精排是负向的。** 前者说明"从多个角度多找几次"对我这个知识库有效；后者我下掉了，重排模型在我的语料上把对的往下压。这是我对这题最实在的取舍观点——检索策略的好坏是**语料相关**的，不能照搬结论。

**项目证据**
- `app/services/vector_search_service.py:43` — `BM25_WEIGHT = 0.3`
- `app/services/vector_search_service.py:96-98` — `EnsembleRetriever` RRF 融合，权重 0.7/0.3
- `app/services/vector_search_service.py:93` — 语料为空降级纯向量
- `app/services/vector_search_service.py:140-162` — `num_entities` 作缓存版本号
- `app/services/vector_search_service.py:165-171` — 全量语料按 1000 批读出
- `app/services/vector_search_service.py:129` — RRF 分数不可得，用名次替代
- `app/services/vector_search_service.py:204-215` — `_tokenize_for_bm25` 中英混合分词
- `app/tools/knowledge_tool.py:78` — guard 放调用点，内部局部降级不触发熔断
- `tests/eval/golden_set_expanded.jsonl` — 82 条评测集，多查询 RRF 正向 / CrossEncoder 负向的依据

#### B16 · Generative Agents 里的 Reflection 机制

**题目**：Generative Agents 论文里的 Reflection 机制是什么？如何实现？

**回答**

**项目中没有 Reflection，不能硬套。** 论文里的 Reflection 是：Agent 定期回看 memory stream，按重要性挑出若干条，让模型归纳出更高层的判断，写回记忆作为新的一条。三个必要成分——重要性评分、周期性触发、归纳结果写回记忆——我一个都没有。

但我有一个**长得像反思、实际不是**的东西，讲清楚这个区别比强行套论文有用：`validate_answer_node`（`nodes.py:569`）是个批判器，它拿问题、答案、证据让模型打两个分（coverage / groundedness），列出 `unsupported_claims` 和 `missing_aspects`。这是自我批判的形式。

它不是 Reflection，因为**没有回路**。`graph.py:48` 是 `graph.add_edge("validate_answer", END)`——判完就结束。反思的定义性特征是结论要影响后续行为，这里不影响。

最能说明问题的是 `needs_second_retrieval` 这个字段。质检 prompt 明确要求它（`nodes.py:103`："当 coverage_score 明显不足时，needs_second_retrieval=true"），解析时正确取出（`:798-805`），几个降级分支也各自给了值（`:517`、`:556`、`:579`、`:586`）。我全仓搜了一遍这个字段的引用，**十处全是写入方，没有一个消费方**。图上没有任何一条边读它。

所以现在的行为是：质检器判断出"证据不够，应该再检索一次"，这个判断被完整地算出来、结构化、记进 validation 字典，然后**没有任何人看**。它最终只出现在返回给前端的 validation 里。

这是半成品，我不打算说成设计。真实的处置是：coverage 或 groundedness 不达标时（阈值 0.60 / 0.75，`nodes.py:36-37`）走 `_build_fallback_answer` 换成保守回答，并记 `EVIDENCE_INSUFFICIENT` 降级原因（`:722-745`）。**降级而不是重试**——这是个有意识的取舍，二次检索要花掉总预算里的一大块，而 90s 预算已经很紧。但取舍的结果是那个字段成了死代码。

`[假设]` 要补成真反思，最小改动是把 `add_edge("validate_answer", END)` 换成条件边，读 `needs_second_retrieval` 决定回 `retrieve_each` 还是去 `END`，同时必须加一个"只允许回退一次"的计数器（否则质检器持续判不通过就是死循环，而现在**没有** `recursion_limit`，见 B7）。

**项目证据**
- `app/agent/rag_v2/nodes.py:569` — `validate_answer_node`，批判器而非反思
- `app/agent/rag_v2/graph.py:48` — `validate_answer → END`，无回边
- `app/agent/rag_v2/nodes.py:103` — prompt 要求 `needs_second_retrieval`
- `app/agent/rag_v2/nodes.py:798-805` — 解析并保留该字段
- `app/agent/rag_v2/nodes.py:517` / `:556` / `:579` / `:586` — 各降级分支都给了值
- 全仓 `needs_second_retrieval` 十处引用均为写入方，无消费方 — 死字段
- `app/agent/rag_v2/nodes.py:36-37` — `MIN_COVERAGE_SCORE = 0.60` / `MIN_GROUNDEDNESS_SCORE = 0.75`
- `app/agent/rag_v2/nodes.py:722-745` — 不达标走降级，不走重试
- 重要性评分、周期性反思触发、归纳写回记忆、反思闭环：项目中没有，不能硬套；补法为 `[假设]`

#### B17 · Planning 能力的实现路径有哪些

**题目**：Agent 的 Planning 能力有哪些实现路径？

**回答**

我这个仓库对这题有一段特殊的经历：**曾经有过 plan-execute-replan，后来我把它删了。** 讲这个比讲实现路径清单有价值。

原来 `app/agent/aiops/` 下有 `planner.py`、`executor.py`、`replanner.py`、`state.py`，是标准的规划-执行-重规划三段结构。现在这个目录下只剩 `__pycache__`，四个文件都已删除。替代它的是 `alert_diagnosis_orchestrator.py`（158 行）——一条**线性管线**，阶段固定，`_PHASE_STATUS`（`:50`）和 `_PHASE_MESSAGES`（`:57`）两张表把每个阶段映射到状态和文案。

删的理由写在这个文件的头部 docstring 里，核心是 YAGNI：告警诊断的步骤是确定的（取告警 → 检索 SOP → 生成首响报告），让模型来规划这三步不会规划出第四种排法，只会引入不确定性和额外的 LLM 调用。docstring 里还专门记了一个**没做**的决定：不为 `retrieved` / `diagnosed` 这类中间态发明新的状态枚举，因为管线是线性的，中间态没有外部观察者需要区分它们。

现在仓库里还剩的"规划"只有一种，而且我不认为它算 Planning：`rewrite_node`（`nodes.py:129`）把用户问题改写成 `NUM_SUB_QUERIES = 3` 个子查询。这是**查询分解**，不是任务规划——三个子查询之间没有依赖关系、没有顺序、没有中间结果传递，它们是三个平行的检索输入，扇出后各跑一次混合检索。真正的 Planning 要能表达"第二步依赖第一步的输出"，`Send` 扇出表达不了这个。

而且分解宽度是**写死的 3**，不是模型决定的。问题简单还是复杂，都拆 3 个。这是刻意的：宽度可变意味着耗时和成本可变，而它们都要计入 90s 总预算。

**项目中没有的部分**：任务级 DAG、步骤间依赖、动态重规划、计划的持久化与恢复。ReAct 式的边走边想也没有——见 B5、B6。

**项目证据**
- `app/agent/aiops/` — `planner.py` / `executor.py` / `replanner.py` / `state.py` 均已删除，目录下只剩 `__pycache__`
- `app/services/alert_diagnosis_orchestrator.py:1-40` — 线性管线的设计说明与 YAGNI 决定
- `app/services/alert_diagnosis_orchestrator.py:50` — `_PHASE_STATUS`
- `app/services/alert_diagnosis_orchestrator.py:57` — `_PHASE_MESSAGES`
- `app/services/alert_diagnosis_orchestrator.py:65` — `AlertDiagnosisOrchestrator`
- `app/agent/rag_v2/nodes.py:129` — `rewrite_node` 做查询分解，非任务规划
- `app/agent/rag_v2/nodes.py:32` — `NUM_SUB_QUERIES = 3`，宽度写死
- `app/agent/rag_v2/graph.py:17-19` — `Send` 扇出，分支间无依赖
- 任务 DAG、步骤依赖、动态重规划、计划持久化：项目中没有，不能硬套

#### B18 · 如何做子目标分解，规划失败了怎么办

**题目**：如何做子目标分解？如果规划失败了怎么办？

**回答**

分解那半见 B17——我只有查询分解，没有子目标分解，两者的区别是有没有依赖和顺序。这题我重点答后半句，因为"分解失败怎么办"我有真实代码。

`rewrite_node` 的失败处置在 `_parse_sub_queries`（`nodes.py:762-782`）。它的降级是三级的：

1. 先试 ```` ```json ``` ```` 代码块正则；
2. 不中就试裸 `[...]` 方括号正则；
3. 都不中或 `json.loads` 抛 `JSONDecodeError`，记 warning，**返回 `[fallback]`**——也就是把原始问题本身当成唯一的子查询。

这个 fallback 的选择是关键：不是抛异常，也不是返回空列表。返回空列表会让 `_fanout_to_retrieve`（`graph.py:17-19`）扇出 0 个分支，`retrieve_each` 一次都不执行，`dedup` 拿到空文档，最后走成"没有证据"的降级答案——用户看到的是"证据不足"，而真实原因是查询改写的 JSON 解析失败。**用假原因掩盖真原因**，这是仓库里反复强调要避免的事。

返回原始问题则退化成"单查询检索"，也就是最朴素的 RAG。功能降级了（少了多角度召回），但链路是通的，答案质量只降不断。

同一个模式在质检侧也有：`_parse_validation_result`（`:784-815`）解析失败时走 `_default_validation`，reason 明确写"质检器输出不可解析，已按未通过处理"——**按未通过处理**是保守方向，宁可多降级一次也不放过一个可能有幻觉的答案。注意这里 `needs_second_retrieval=False`，因为解析失败说明质检器本身有问题，再检索一次也解决不了。

AIOps 侧的 `parse_report`（`aiops_report.py:193-245`）是同一条纪律的第三个落点，docstring 写得最直白："这是「AI 输出必须可降级」纪律的落点：无论模型返回什么，调用方永远拿到一个可用的 `FirstResponseReport`，不会抛异常。"

三处手写解析器，都是"正则抠 JSON → `json.loads` → 类型强转 → 失败记 warning 返回默认值"。我没用 `with_structured_output`，因为它在解析失败时的行为是抛异常，而我要的是降级。

**项目中没有的部分**：重规划（改写失败后换个 prompt 再试一次）、分解质量的评估、失败率的指标采集。都是**项目中没有，不能硬套**——`metrics.py` 里只有四个指标，没有解析失败计数。

**项目证据**
- `app/agent/rag_v2/nodes.py:762-782` — `_parse_sub_queries` 三级降级，兜底返回原问题
- `app/agent/rag_v2/graph.py:17-19` — 返回空列表会扇出 0 个分支
- `app/agent/rag_v2/nodes.py:784-815` — `_parse_validation_result` 失败按未通过处理
- `app/agent/rag_v2/nodes.py:830-847` — `_default_validation`
- `app/agent/rag_v2/nodes.py:816-829` — `_coerce_score` 类型强转兜底
- `app/models/aiops_report.py:193-245` — `parse_report` 的「AI 输出必须可降级」纪律
- 全仓无 `with_structured_output` — 三处解析器均为手写
- 重规划、分解质量评估、解析失败指标：项目中没有，不能硬套

#### B19 · Human-in-the-Loop 应该在什么时候介入

**题目**：Human-in-the-Loop 机制应该在什么时机介入？

**回答**

这题我有真的 HITL，而且介入时机的判据很明确：**写入外部系统之前。**

落点是协议 PDF 入库链路。`ProtocolIngestionStatus` 里有一个 `AWAITING_CONFIRMATION` 状态（`protocol_ingestion.py:24`），PDF 解析完成后任务停在这个状态，不自动入库。人工看过之后走两个接口之一：`POST /{ingestion_id}/confirm`（`protocol_pdf.py:122-136`）或 `POST /{ingestion_id}/reject`（`:140`）。确认时记 `confirmed_by` 和 `confirmed_at`（`protocol_ingestion.py:64-65`），仓储层的方法叫 `confirm_with_state_trace`（`protocol_ingestion_repository.py:111`）——名字里的 state_trace 说明状态流转本身要留痕，不只是改个字段。

为什么这条链路要卡人工、而知识库上传不卡：入库产生的是 `protocol_catalog` 的目录记录（`protocol_catalog.py:1`），会被后续检索当成权威数据用。PDF 解析是有错误率的，错误的目录记录进去之后，污染的是所有下游查询，而且很难发现。知识库文档上传的失败影响面小得多——检索不到就是检索不到。**介入时机的判据不是"操作危险"，而是"错了之后能不能被发现"。**

第二处是 AIOps 报告里的两个字段：`ActionCheck.requires_human`（`aiops_report.py:97`）和 `FirstResponseReport.pending_confirmations`（`:121-122`）。渲染时带"（需人工确认）"后缀（`:159`），待确认项单独成段（`:179-181`）。这是**声明式**的 HITL——报告告诉值班同学哪些动作不能自动执行，但仓库里没有任何代码去阻塞这些动作，因为 AIOps 侧本来就只生成报告、不执行操作。

要说清楚这个边界：`requires_human` 是给人看的标记，不是给系统看的闸门。把它说成"我们实现了危险操作的人工审批"是夸大。

**项目中没有的部分**：LangGraph 的 `interrupt` / `NodeInterrupt` 一处都没有，问答链路上没有任何人工介入点。这不是遗漏——问答是只读的，没有需要审批的副作用。真正的缺口是：`rag_v2` 的图 `compile()` 时**不带 checkpointer**（`graph.py:50`），而 `interrupt` 依赖 checkpointer 保存中断时的状态。所以就算将来要在问答链路加 HITL，前置条件是先把 checkpointer 补上。

**项目证据**
- `app/models/protocol_ingestion.py:24` — `AWAITING_CONFIRMATION` 状态
- `app/repositories/protocol_ingestion_repository.py:105` — 置为 `awaiting_confirmation`
- `app/repositories/protocol_ingestion_repository.py:111-121` — `confirm_with_state_trace`，记 `confirmed_by` / `confirmed_at`
- `app/api/protocol_pdf.py:122-136` — 确认入库接口
- `app/api/protocol_pdf.py:140` — 驳回接口
- `app/api/protocol_pdf.py:28` — `ConfirmProtocolPdfRequest.confirmed_by`
- `app/models/protocol_catalog.py:1` — 确认后才产生的目录记录
- `app/models/aiops_report.py:97` — `requires_human`，声明式标记而非闸门
- `app/models/aiops_report.py:121-122` / `:159` / `:179-181` — `pending_confirmations` 与渲染
- `app/agent/rag_v2/graph.py:50` — `compile()` 无 checkpointer，`interrupt` 的前置条件缺失
- 问答链路的人工介入、`interrupt` / `NodeInterrupt`、危险操作的执行期阻塞：项目中没有，不能硬套

#### B20 · 工具调用失败时如何优雅降级

**题目**：工具调用失败时，Agent 应该如何优雅降级？

**回答**

这题是我这个仓库花心思最多的地方之一，纪律可以总结成一句：**不许用假原因掩盖真原因。**

`knowledge_tool.py` 是最完整的例子。它有两句话给模型，措辞刻意区分开：

```python
_UNAVAILABLE_HINT = (
    "检索服务当前不可用（错误码: {code}），本次未能查询知识库。"
    "请如实告知用户知识库暂时无法访问，不要凭记忆作答。"
)
_EMPTY_HINT = "知识库中没有与该问题相关的内容。"
```

区别的必要性写在文件头 docstring 里：检索服务挂了却回"没有找到相关信息"，模型会据此回答"知识库中没有相关内容"——**一个彻底的谎**，而且用户和值班同学都无法从答案里看出真实原因。返回值会被模型当作事实读进上下文，所以这个字符串是唯一能施加约束的接口。`_UNAVAILABLE_HINT` 里明确写"请如实告知用户"，是因为模型看到工具报错的默认行为常常是自己编一个答案。

三层防护的顺序也有讲究（`knowledge_tool.py:78-79`）：

```python
with retrieval_breaker.guard():      # 熔断在外
    with observe_tool_call(TOOL_NAME):  # 计时在内
```

反过来会让熔断打开期间每次请求都往耗时直方图塞一个 ≈0 秒的样本——Milvus 挂得越久，P99 看起来越好，因为分母全是熔断器的快速拒绝。

MCP 侧是另一套。`retry_interceptor`（`mcp_client.py:106-140`）只重试**值得重试**的失败，判据走 `is_retryable`。改造前是无条件重试 3 次，后果是鉴权失败、参数非法这类确定性错误也要白等 1 + 2 = 3 秒，而结果注定一样。重试全失败后**抛** `MCPUnavailableError`，而不是返回 `CallToolResult(isError=True)`——两者的区别是异常能被上层的错误码体系归类，`isError=True` 只是个布尔值。

`circuit_breaker_interceptor`（`:174`）在重试之外单独一层，`_mcp_breakers_lock`（`breakers.py:67`）保护每个 server 各自的熔断器。

节点级的降级在 `tool_facts_node`（`nodes.py:252`）的 docstring 里：任何异常都不向上抛。理由是它跑在并行分支上，抛异常会让整个 super-step 失败，把检索侧已经做完的工作一起丢掉。"工具事实是**增强**，不是回答的前提。"

最后一层在 `generate_node`（`:503-512`）：只有 `ok=True` 的工具事实算证据。失败条目的 content 是"查询失败（xxx）"这类说明文字，它能帮模型如实告知用户，但不能充当"我们有证据"——否则一个 MCP 全挂的请求会因为"有 3 条 tool_facts"绕过无证据兜底。这个区分很容易漏。

**项目证据**
- `app/tools/knowledge_tool.py:44-50` — `_UNAVAILABLE_HINT` 与 `_EMPTY_HINT` 措辞区分
- `app/tools/knowledge_tool.py:1-30` — 「不许用假原因掩盖真原因」的完整论述
- `app/tools/knowledge_tool.py:78-79` — 熔断在外、计时在内的嵌套顺序
- `app/tools/knowledge_tool.py:97` / `:101` — 两条返回路径
- `app/agent/mcp_client.py:106-140` — `retry_interceptor` 按 `is_retryable` 分类
- `app/agent/mcp_client.py:174` — `circuit_breaker_interceptor`
- `app/core/breakers.py:38` / `:45` / `:67` — 检索、LLM、MCP 三组熔断器
- `app/agent/rag_v2/nodes.py:252-270` — `tool_facts_node` 不向上抛异常的理由
- `app/agent/rag_v2/nodes.py:503-512` — 只有 `ok=True` 才算证据
- `app/services/sop_retrieval_service.py:96-167` — `RetrievalStatus` 三态，`FAILED` 不伪装成 `EMPTY`

#### B21 · 多个工具并行调用和依赖关系怎么处理

**题目**：多个工具需要并行调用、或者存在依赖关系时，如何处理？

**回答**

这题我必须先说清楚一件事，否则后面讲的都是假的：**并行工具调用的代码我写了，但它没有接进图里，现在是死代码。**

`tool_facts_node` 定义在 `nodes.py:252`，`_run_tool_calls` 在 `:341`，`asyncio.gather` 并发在 `:360`，上限 `MAX_TOOL_CALLS_PER_TURN = 4` 在 `:45`。文件头的注释（`:1-9`）把它列为六个节点之一，写着"tool_facts → 单轮工具选择，查实时事实（与检索并行，不占串行预算）"。配置里有 `enable_tool_facts`（`config.py:122`）和 `tool_facts_budget_seconds`（`:135`），state 里有 `tool_facts` 字段（`state.py:42`），`generate_node` 也已经改好了读取侧（`nodes.py:503-512`）。

但 `graph.py:35-41` 只 `add_node` 了五个节点——`rewrite`、`retrieve_each`、`dedup`、`generate`、`validate_answer`。没有 `tool_facts`。我全仓搜过 `tool_facts_node`，只有 `nodes.py:252` 的定义那一行，**没有任何调用点**。所以这个节点从来没有执行过。

这是接线漏了，不是设计。承认它比把注释当成事实讲更重要——面试里说"我们有一个与检索并行的工具节点"，追问一句"图上哪条边指向它"就露了。

已经写好的那部分设计我可以讲，因为代码是完整的：

**并行的判据**——同一轮里模型一次性给出全部调用，它们之间没有依赖，所以 `asyncio.gather` 并发。docstring 写了理由：串行会让耗时变成两次 MCP 往返之和，而本节点的耗时直接计入总预算。

**为什么是"单轮"而不是 agent 循环**（`:252-264`）——这是最有价值的一段。循环轮数由模型决定，也就是预算什么时候被吃光由模型决定，而超预算会取消**所有**分支，包括那条已经召回了完整证据的检索链。一轮换来的是可预测的耗时上界。

**超出上限的处置**（`:363-372`）——丢弃的调用**记成失败条目**而不是静默截断。理由：模型请求了 6 个工具却只有 4 个结果，它会以为剩下两个"没查到数据"，进而在答案里说"未发现异常"。

**依赖关系我没有处理，因为单轮结构下不存在依赖。** 真正需要"第二个工具的参数来自第一个工具的结果"时，单轮做不到，必须有循环——而循环的代价就是上面那段说的预算不可控。这是个真实的取舍，不是能力缺失。

**项目证据**
- `app/agent/rag_v2/nodes.py:252` — `tool_facts_node` 定义
- `app/agent/rag_v2/graph.py:35-41` — 只注册五个节点，无 `tool_facts`
- 全仓 `tool_facts_node` 仅一处定义、零调用点 — 未接线的死代码
- `app/agent/rag_v2/nodes.py:1-9` — 文件头注释把它列为节点之一（与实际接线不一致）
- `app/agent/rag_v2/nodes.py:341-360` — `_run_tool_calls` 用 `asyncio.gather` 并发
- `app/agent/rag_v2/nodes.py:40-45` — `MAX_TOOL_CALLS_PER_TURN = 4` 及取值理由
- `app/agent/rag_v2/nodes.py:363-372` — 超出上限记成失败条目
- `app/agent/rag_v2/nodes.py:252-264` — 单轮而非 agent 循环的预算论证
- `app/agent/rag_v2/nodes.py:381-434` — `_run_one_tool_call`，同步工具走 `ainvoke` 自动转线程
- `app/agent/rag_v2/nodes.py:46-52` — `MAX_TOOL_CONTENT_CHARS = 1200` 防止工具输出挤占证据
- `app/config.py:122` / `:135` — `enable_tool_facts` / `tool_facts_budget_seconds`
- `app/agent/rag_v2/state.py:42` — state 里的 `tool_facts` 字段
- 工具间依赖、多轮工具循环：项目中没有，不能硬套；单轮结构下依赖不可表达

#### B22 · Computer Use Agent 的主要挑战

**题目**：Computer Use Agent（操作电脑的 Agent）面临哪些主要挑战？

**回答**

**项目中没有，不能硬套。** 仓库里没有任何浏览器自动化、屏幕截图、GUI 元素定位、键鼠操作的代码。没有 Playwright、Selenium、pyautogui 一类依赖。我也没有 `psutil`——连读本机指标都没有，`mcp_servers/monitor_server.py` 的 CPU 和内存值是 `random.uniform` 合成的 mock（`:207` `base_cpu = 10.0`，`:355` `base_memory = 30.0`）。

不硬套，但可以说清楚我的项目在哪些地方**碰到了同类问题的弱化版本**，这些是真的：

**动作不可逆**。Computer Use 最大的风险是点错一个按钮无法撤销。我这里对应的是 PDF 入库——解析结果写进 `protocol_catalog` 之后污染所有下游检索，所以卡了人工确认（见 B19）。判据是"错了之后能不能被发现"，Computer Use 的动作大多不满足这一条，这就是它必须默认 HITL 的原因。

**观察不可靠**。GUI Agent 靠截图理解状态，截图会因为渲染时机拿到中间态。我这里的弱化版是工具返回值——`_UNAVAILABLE_HINT` 和 `_EMPTY_HINT` 要区分开（`knowledge_tool.py:44-50`），本质就是"观察结果必须能表达'我没看清'和'确实没有'的区别"。Computer Use 里这个区别更难，截图不会告诉你它是不是加载完了。

**步数无界**。这是我最有共鸣的一条。Computer Use 的一个任务动辄几十步，而我这里连 `recursion_limit` 都没有设（全仓无引用，见 B7），唯一的闸门是 `chat_total_budget_seconds = 90.0` 的时间预算（`config.py:81`）。时间闸门在 Computer Use 上是不够的——它的单步很快，90 秒能点几百次，够造成实际损害了。所以那个场景必须有独立的步数上限和动作白名单，而这两样我都没写过。

**没有回滚点**。`rag_v2` 的图 `compile()` 不带 checkpointer（`graph.py:50`），失败就是整个请求重来。问答只读，重来的代价只是延迟。Computer Use 有副作用，重来会重复执行已经做过的操作——所以那个场景的 checkpoint 不是优化而是正确性前提。

这四条我都只有弱化版本的经验，说成 Computer Use 的实践是虚构。

**项目证据**
- 全仓无 Playwright / Selenium / pyautogui / 截图 / GUI 定位代码
- 全仓无 `psutil` — 连本机指标读取都没有
- `mcp_servers/monitor_server.py:207` / `:355` — CPU / 内存为 `random.uniform` 合成的 mock
- `app/models/protocol_ingestion.py:24` — 不可逆操作卡人工确认的真实对照
- `app/tools/knowledge_tool.py:44-50` — 「观察不可靠」的弱化版对照
- 全仓无 `recursion_limit` — 步数无界
- `app/config.py:81` — 唯一闸门是 90s 时间预算
- `app/agent/rag_v2/graph.py:50` — 无 checkpointer，无回滚点
- Computer Use 的实际实践、动作白名单、步数上限、GUI 状态判定：项目中没有，不能硬套

### 题库 B-3 · 工具与协议：Function Calling / MCP / A2A（10 题）

> 来源：`https://www.cnblogs.com/itech/p/20111938`

#### B23 · 工具的 JSON Schema 怎么定义

**回答**

我这个仓库里**没有一处手写 JSON Schema**，全部靠装饰器从 Python 签名和 docstring 反射生成。这不是偷懒，是一个有代价的选择，我说清楚代价。

本地工具走 LangChain 的 `@tool`。`retrieve_knowledge` 的签名是 `(query: str) -> Tuple[str, List[Document]]`，schema 里 `query` 的类型来自类型注解，描述来自 docstring 的 `Args:` 段。返回值这里有个额外声明：`response_format="content_and_artifact"`，意思是返回的二元组前一项进模型上下文、后一项作为 artifact 旁路给调用方——**schema 描述的是给模型看的那一半，另一半模型看不到**。这个区分很实用：格式化后的文本给模型读，原始 `Document` 列表给前端渲染引用来源，不用为了前端把结构化数据塞进模型上下文。

MCP 侧走 FastMCP 的 `@mcp.tool()`，同样从签名生成。`query_cpu_metrics` 的签名是 `(service_name: str, start_time: Optional[str] = None, end_time: Optional[str] = None, interval: str = "1m")`——`Optional[str] = None` 生成的是可选参数，`interval: str = "1m"` 生成的是带默认值的必填项。这里能看出反射生成的**真实局限**：`start_time` 的类型是 `str`，schema 只能说"这是字符串"，说不了"必须是 `YYYY-MM-DD HH:MM:SS` 格式"。约束表达不了，只能退到 docstring 里用自然语言写——`monitor_server.py:138-140` 就是这么写的：格式一行、示例一行。

所以我的实际做法是**约束分两层**：能用类型系统表达的（必填/可选/枚举/嵌套）交给签名，表达不了的（格式、取值范围、语义前提）写进 docstring。前者是硬约束，模型给错了框架会拒；后者是软约束，模型可以无视，所以调用侧必须自己校验。

`mcp_servers` 里没有做参数校验——传进来的时间字符串直接用。这是真实的缺口，不是设计。

**项目证据**
- `app/tools/knowledge_tool.py:53` — `@tool(response_format="content_and_artifact")`
- `app/tools/knowledge_tool.py:54-63` — docstring 的 `Args:` / `Returns:` 段即 schema 描述来源
- `app/tools/time_tool.py` — `@tool`，`timezone: str = "Asia/Shanghai"` 默认值进 schema
- `mcp_servers/monitor_server.py:124-131` — `@mcp.tool()` 与四参数签名，`Optional[str] = None` 生成可选参数
- `mcp_servers/monitor_server.py:138-140` — 格式约束只能落在 docstring 的自然语言里
- `mcp_servers/cls_server.py:212-218` — `search_topic_by_service_name` 的 `fuzzy: bool = True`
- 手写 JSON Schema、参数格式校验、枚举约束：项目中没有，不能硬套

#### B24 · 工具描述怎么优化

**回答**

我在这块有一条**别人通常不写**的经验：工具描述要优化的不只是入口的 docstring，还有**出口的返回文案**。因为工具与模型之间的全部通信就是一个字符串，模型对工具的理解一半来自描述、一半来自它实际读到的返回值。

入口侧的优化是常规的：`query_cpu_metrics` 的 docstring 里每个参数都带格式和示例值，因为反射生成的 schema 表达不了格式（见 B23）；`retrieve_knowledge` 的第一行写的是**触发条件**而不是功能——"当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具"，描述的是模型该在什么情况下选它，而不是它内部干什么。模型做的是选择题，它需要判据不需要实现说明。

出口侧是我真正花心思的地方。`knowledge_tool.py` 里两个返回文案是分开写的常量：

- `_EMPTY_HINT = "知识库中没有与该问题相关的内容。"` —— 这是一个**真实的业务结论**。
- `_UNAVAILABLE_HINT` 里写的是"检索服务当前不可用（错误码: {code}），本次未能查询知识库。请如实告知用户知识库暂时无法访问，不要凭记忆作答。"

为什么必须分开：如果检索服务挂了却返回"没有找到相关信息"，模型会据此回答"知识库中没有相关内容"——这是一句彻底的谎，而且用户和值班同学都无法从答案里看出真实原因。文件头的注释把这条写成了纪律：不许用假原因掩盖真原因。

更进一步，`_UNAVAILABLE_HINT` 里带了一句**行为指令**："请如实告知用户……不要凭记忆作答"。这是因为模型看到工具报错时的默认行为常常是自己编一个答案或者含糊过去。在这个只能通过字符串通信的接口上，把处置方式写进返回值是唯一能施加约束的地方。错误码 `{code}` 也一起带出去，是为了让值班同学从答案里就能拿到排障线索。

还有个细节：工具名抽成了常量 `TOOL_NAME = "retrieve_knowledge"`，同时用作 Prometheus 指标 label 和日志前缀。这样"指标里的名字"和"注册给模型的名字"不会各自漂移——改名时一处改完全部同步。

**缺的部分**：描述的 A/B 对比、按调用成功率反推描述质量——**项目中没有，不能硬套**。`metrics.py` 只有四个指标，没有工具级成功率（见拼多多 Q12），所以我没有量化依据说某个描述改动带来了多少提升，只能说改完之后"检索挂了模型不再编答案"这个行为是稳定复现的。

**项目证据**
- `app/tools/knowledge_tool.py:54-56` — docstring 首行写触发条件而非功能
- `app/tools/knowledge_tool.py:44-50` — `_UNAVAILABLE_HINT` 与 `_EMPTY_HINT` 分开定义
- `app/tools/knowledge_tool.py:1-30` — 文件头纪律："不许用假原因掩盖真原因"
- `app/tools/knowledge_tool.py:37` — `TOOL_NAME` 常量，指标 label 与注册名同源
- `app/tools/knowledge_tool.py:97` / `:101` — 两条路径分别返回不同文案
- `mcp_servers/monitor_server.py:134-140` — 参数描述带格式与示例
- 描述的 A/B 测试、工具级调用成功率指标：项目中没有，不能硬套

#### B25 · 并行 Function Calling 和失败重试怎么做

**回答**

我有一处真实的并行工具调用实现，但要先说清楚它的状态：`tool_facts_node` 代码完整、注释详尽，**但全仓没有任何调用点，`graph.py` 也没注册它**——写了没接线。所以下面讲的是代码里的设计，不是线上跑过的经验，这个边界我不含糊。

并行部分在 `_run_tool_calls`：模型在单轮里一次性给出全部工具调用，调用之间没有依赖，所以走 `asyncio.gather` 并发。串行的话耗时会变成多次 MCP 往返之和，而这个节点的耗时直接计入 90s 总预算。

上限是 `MAX_TOOL_CALLS_PER_TURN = 4`。为什么要有上限：工具调用参数是模型给的，它可以在一轮里请求 20 个调用，每个都是一次 MCP 往返，并发打满会同时压垮 MCP 进程和总预算。取 4 是因为够覆盖"查监控 + 查日志"这类组合，又不至于让单个节点变成压测客户端。

超出上限的处理是我认为最值得讲的一点：**丢弃但如实记录成失败条目**，而不是静默截断。理由是模型请求了 6 个工具却只收到 4 个结果时，它会以为剩下两个"没查到数据"，进而在答案里说"未发现异常"——那又是用假原因掩盖真原因。所以被丢弃的调用会补一条 `content = "未执行：单轮工具调用数超过上限 4"` 的记录。

重试不在这一层。`_run_one_tool_call` 的契约是**不抛异常**，失败就返回一条 `ok=False` 的记录——因为它跑在与检索并行的分支上，抛异常会让整个 super-step 失败，把检索侧已经做完的工作一起丢掉。重试发生在更下面的 MCP 拦截器 `retry_interceptor`，而且只重试**值得重试**的失败：改造前这里对所有异常一律重试 3 次，结果鉴权失败、参数非法这类确定性错误也要白等 1+2=3 秒，而结果注定一样——重试只是把失败延后暴露，还烧掉三倍资源。判据走 `is_retryable`，分类逻辑在 `app/core/errors.py` 只有一处实现。

下游怎么区分"工具失败"和"工具说没事"：`generate_node` 里 `usable_facts` 只取 `ok=True` 的条目。这样一个 MCP 全挂的请求不会因为"有 3 条 tool_facts"就绕过无证据兜底。失败条目的文案仍然进 prompt，让模型能如实告知用户，但不算证据。

**项目证据**
- `app/agent/rag_v2/nodes.py:341-372` — `_run_tool_calls`，`asyncio.gather` 并发
- `app/agent/rag_v2/nodes.py:45` — `MAX_TOOL_CALLS_PER_TURN = 4` 与取值理由注释（`:39-44`）
- `app/agent/rag_v2/nodes.py:364-372` — 超限调用丢弃并记成失败条目
- `app/agent/rag_v2/nodes.py:381` — `_run_one_tool_call`，契约是不抛异常
- `app/agent/rag_v2/nodes.py:252-270` — 节点 docstring 说明为何"单轮"而非 agent 循环
- `app/agent/mcp_client.py:106-120` — `retry_interceptor` 只重试可重试失败
- `app/agent/rag_v2/nodes.py:509` — `usable_facts` 只取 `ok=True`
- `app/agent/rag_v2/nodes.py:52` — `MAX_TOOL_CONTENT_CHARS = 1200`，单个结果注入上限
- **未接线**：`tool_facts_node` 全仓无调用点，`app/agent/rag_v2/graph.py:35-41` 只注册五个节点

#### B26 · MCP 解决了什么问题

**回答**

我的答案基于两个真实跑起来的 MCP server，而且我更想讲**它带来的新问题**，因为那部分是我实际写代码解决的。

解决的问题很直接：工具实现从 Agent 进程里搬了出去。`cls_server.py`（日志查询，4 个工具）和 `monitor_server.py`（CPU / 内存指标，2 个工具）是两个独立进程，各自 `mcp.run(transport="streamable-http", host="127.0.0.1", port=8003/8004)`。Agent 侧只在 `config.py` 里声明 `{server_name: {"transport": ..., "url": ...}}`，`MultiServerMCPClient` 把它们的工具拉进来变成 LangChain `BaseTool`。加一个工具不用改 Agent 代码，也不用把日志 SDK 的依赖装进主进程。

新问题是**多了一个会独立失败的进程边界**，这才是我花时间的地方：

第一，工具**列举**不经过拦截器链。`load_mcp_tools` 是直接调的，不走 `tool_interceptors`，所以 MCP 进程没起来时，列举会每次都去连、每次都超时。为此 `mcp_tool_provider` 给列举加了独立的冷却退避：失败后 60 秒内不再尝试，直接用上次的结果或空列表。这跟工具**调用**侧的熔断器是两套机制，因为失败模式不同——调用失败是"这个工具用不了"，列举失败是"这个 server 整个不可见"。

第二，调用侧接了拦截器链：`retry_interceptor` 负责选择性重试，`circuit_breaker_interceptor` 负责按 server 熔断。这两个是 MCP 特有的需求——本地工具调用不会有网络抖动，也不需要熔断。

第三，可见性要能查。`available_servers` 和 `unavailable_servers` 把"哪些 server 现在能用、哪些不能用及原因"暴露出来，否则模型少了几个工具，排查时只能猜。

**缺的部分**：MCP server 的动态发现（现在是 `config.py` 里写死两个 URL）、版本协商、server 侧的资源隔离——**项目中没有，不能硬套**。

**项目证据**
- `mcp_servers/cls_server.py:470` / `mcp_servers/monitor_server.py:435` — 两个独立进程，`streamable-http`
- `app/config.py:169-172` — 两个 server 的 transport 与 URL
- `app/config.py:191-201` — `mcp_servers` property 组装配置
- `app/agent/mcp_client.py:336-357` — `_create_mcp_client`，注入 `tool_interceptors`
- `app/agent/mcp_tool_provider.py:29-31` — `load_mcp_tools` 不经过拦截器链的说明
- `app/agent/mcp_tool_provider.py:184-207` — `_should_load` 冷却退避判定
- `app/config.py:188` — `mcp_tool_reload_cooldown_seconds: float = 60.0`
- `app/agent/mcp_tool_provider.py:150-164` — `available_servers` / `unavailable_servers`
- 动态发现、版本协商、server 侧资源隔离：项目中没有，不能硬套

#### B27 · MCP 的三个原语是什么

**回答**

标准答案是 Tools、Resources、Prompts。我只用了**第一个**，另外两个**项目中没有，不能硬套**——全仓 grep 不到任何 `resources` / `prompts` 原语的注册代码，两个 server 里只有 `@mcp.tool()`。

所以我只能讲 Tools 这一个原语的实际形态，以及为什么另两个在我这个场景没用上：

Tools 是**模型主动调用、有副作用语义、结果不可缓存**的。`cls_server.py` 的 4 个工具（`get_current_timestamp`、`get_region_code_by_name`、`get_topic_info_by_name`、`search_topic_by_service_name`）和 `monitor_server.py` 的 2 个（`query_cpu_metrics`、`query_memory_metrics`）都是查询实时数据，天然属于 Tools。

Resources 的定位是**由客户端读取的上下文数据**，寻址靠 URI，模型不主动调。我这个场景里对应的东西是知识库文档——但我没走 MCP，走的是自己的 Milvus + `retrieve_knowledge` 工具。原因是知识库需要向量检索和混合排序，Resources 的 URI 寻址表达不了"按语义相似度取前 K 条"这种访问模式。这是一个**主动的取舍**，不是没想到。

Prompts 的定位是服务端提供可复用的 prompt 模板。我的 prompt 全部硬编码在 `nodes.py` 里——`REWRITE_SYSTEM_PROMPT`、`GENERATE_SYSTEM_PROMPT`、`VALIDATION_SYSTEM_PROMPT`、`MEMORY_POLICY`、`TOOL_FACTS_POLICY` 五个常量。放在代码里的好处是跟着版本一起走，改 prompt 和改判定逻辑在同一个 commit 里；坏处是不能热更新。这个场景下我认为前者更重要，因为 prompt 里的字段名和 `_parse_validation_result` 的解析代码是强耦合的，分开部署会漂移。

**项目证据**
- `mcp_servers/cls_server.py:104` / `:136` / `:169` / `:212` — 四个 `@mcp.tool()`
- `mcp_servers/monitor_server.py:124` / `:277` — 两个 `@mcp.tool()`
- `app/tools/knowledge_tool.py:53` — 知识库走自己的工具而非 MCP Resources
- `app/agent/rag_v2/nodes.py:55` / `:69` / `:79` / `:106` / `:110` — 五个 prompt 常量硬编码
- `app/agent/rag_v2/nodes.py:784-815` — `_parse_validation_result` 与 prompt 字段强耦合
- MCP Resources、MCP Prompts：项目中没有，不能硬套

#### B28 · MCP Server 的安全问题怎么考虑

**回答**

诚实回答：我这两个 MCP server **完全没有鉴权**，任何能访问端口的进程都能调用全部工具。grep 不到任何 token 校验、签名验证、调用方白名单。

唯一起作用的控制是网络层的：`mcp.run(..., host="127.0.0.1", port=8003)` 绑在回环地址上，所以只有本机进程能连。但我要说清楚——**这是部署方式的副产品，不是安全设计**。它挡住的是外网，挡不住同一台机器上的任何进程，而且一旦为了跨机部署把 host 改成 `0.0.0.0`，这层保护立刻消失，代码里没有任何东西会提醒你补鉴权。

真正降低风险的是**工具本身的性质**：6 个工具全是只读查询（查日志、查指标、查主题映射），没有一个能写数据、删数据或执行命令。所以最坏后果是信息泄露，不是数据损坏。这是运气，不是设计——如果哪天加一个"重启服务"的工具，当前架构下它会和只读工具享受完全相同的零门槛。

如果要补，我会按这个顺序（都是 `[假设]`）：

第一步是**工具分级**。给每个工具标只读 / 可变更，可变更的走 `requires_human` 那条已经存在的模式——`aiops_report.py:97` 已经有这个字段，`protocol_ingestion.py:24` 的 `AWAITING_CONFIRMATION` 状态机是现成的人工确认落点。先把不可逆操作卡住，比先做鉴权收益大。

第二步是**调用方身份**。streamable-http 是标准 HTTP，加一个 header token 校验的成本很低，能把"同机任意进程"收窄到"持有 token 的进程"。

第三步是**参数校验**。现在时间字符串直接用，没有格式或范围检查，`search_log` 这类工具理论上能被构造出超大时间范围的查询把 server 拖死。这一条其实比鉴权更急——它在当前的可信环境里就能被误用触发，不需要恶意方。

**项目证据**
- `mcp_servers/cls_server.py:470` / `mcp_servers/monitor_server.py:435` — 绑 `127.0.0.1`，无鉴权参数
- 两个 server 文件内 grep 无 token / auth / verify / apikey 相关代码
- `mcp_servers/monitor_server.py:30-42` — `log_tool_call` 只做日志，不做校验
- `mcp_servers/monitor_server.py:207` / `:355` — 返回的是 mock 数据，泄露风险实际为零
- `app/models/aiops_report.py:97` — `requires_human` 字段，工具分级的现成落点
- `app/models/protocol_ingestion.py:24` — `AWAITING_CONFIRMATION`，人工确认状态机
- 鉴权、调用方白名单、限流、参数校验：项目中没有，不能硬套；补法为 `[假设]`

#### B29 · MCP 的远程连接和 OAuth 怎么处理

**回答**

远程连接的**传输层**我用的是真的，鉴权部分**项目中没有，不能硬套**。

传输走 `streamable-http`，两处都是：server 侧 `mcp.run(transport="streamable-http", host="127.0.0.1", port=8003, path="/mcp")`，客户端侧 `config.py` 里配 `mcp_cls_transport = "streamable-http"` 和 `mcp_cls_url = "http://localhost:8003/mcp"`。选它而不是 stdio 是有具体理由的：stdio 要求客户端**负责拉起并持有 server 子进程**，而我这里 uvicorn 主进程和 RQ worker 是两个独立进程，都要用同一批 MCP 工具——stdio 下会变成两份 server 实例，日志和连接池各自一套。HTTP 让 server 变成独立部署单元，谁都能连同一个。

这也意味着**跨机部署只需要改 URL**，代码不用动。但代价在 B28 说过：改成跨机的那一刻，`127.0.0.1` 这层唯一保护就没了，而代码里没有任何东西会拦住这个改动。

OAuth 相关的东西全仓零引用：没有 token 获取、没有刷新、没有 `Authorization` header 注入。整个仓库连**用户级鉴权都没有**——FastAPI 侧 grep 不到 `Security` / `HTTPBearer` / `api_key_header` / `current_user` 任何一个。所以不存在"MCP 这块没做鉴权而别处做了"的情况，是整体缺失。

`[假设]` 如果要补 OAuth，我会关注一个在这个架构下必然踩的点：token 有效期与 `mcp_tool_provider` 的 60 秒冷却退避会互相干扰。token 过期导致列举失败后，冷却期内不再重试——也就是说刷新完 token 也要等最多 60 秒工具才恢复。`invalidate()` 已经预留了手动清缓存的入口，鉴权失败应该显式调它跳过冷却，而不是等冷却自然过期。这类"两个正确机制叠在一起产生错误行为"的坑，是我在 `retry_interceptor` 那次改造里踩过同类的（对确定性错误重试三次），所以会先想到。

**项目证据**
- `mcp_servers/cls_server.py:470` / `mcp_servers/monitor_server.py:435` — `transport="streamable-http"`
- `app/config.py:169-172` — 客户端侧 transport 与 URL 配置
- `app/agent/mcp_client.py:351-357` — `MultiServerMCPClient` 按配置建连
- `app/agent/mcp_tool_provider.py:166-183` — `invalidate()`，手动跳过冷却的入口
- `app/config.py:188` — 60 秒冷却，与 token 过期的干扰点
- `app/agent/mcp_client.py:106-120` — 选择性重试的同类教训
- OAuth、token 刷新、`Authorization` 注入、用户级鉴权：项目中没有，不能硬套；补法为 `[假设]`

#### B30 · A2A 和 MCP 有什么区别

**回答**

A2A **项目中没有，不能硬套**——全仓 grep 不到 A2A、Agent Card、agent-to-agent 任何相关代码。我只有 MCP 一侧的实践，另一侧只能讲我对协议定位的理解，并明确这是原理陈述而非项目经验。

我能讲实的部分是 MCP 在我这里的**实际定位**：它连接的是"一个 Agent"和"一批工具"，关系是**不对等**的。`monitor_server.py` 里的 `query_cpu_metrics` 不会反过来调用我的 Agent，也不持有任务状态、不做决策——它是纯函数式的数据源，给定参数返回结果。客户端侧 `MultiServerMCPClient` 把它们的工具拉过来变成 `BaseTool`，进模型的 tool schema 列表。整条链路上只有一个决策主体。

A2A 要解决的是不同性质的问题：两边都是有决策能力的 Agent，需要交换的不只是"调用参数和返回值"，还有任务状态、能力声明、协商过程。这带来 MCP 不需要处理的东西——比如双方对任务进度的理解可能不一致，比如需要知道对方能干什么才能决定要不要委派。

对照我这个仓库能说明为什么用不上 A2A：仓库里**根本没有第二个 Agent**。两条路径（`create_agent` 那条和 `rag_v2` 那条）是同一个进程里的两个入口，不是两个能互相委派的主体；`app/agent/aiops/` 目录现在只剩 `__pycache__`，之前的 planner / executor / replanner 已经删掉了。没有多 Agent，就没有 Agent 间通信的需求——上多 Agent 协作协议之前得先有协作，这个顺序不能反。

**项目证据**
- 全仓无 A2A / AgentCard / agent-to-agent 相关代码
- `app/agent/mcp_client.py:336-357` — MCP 客户端只做工具拉取，不交换任务状态
- `mcp_servers/monitor_server.py:124-131` — 工具是纯函数式数据源，无决策、无状态
- `app/services/rag_agent_service.py:173-176` / `app/agent/rag_v2/graph.py:22-51` — 两条单 Agent 路径，同进程两个入口
- `app/agent/aiops/` — 仅剩 `__pycache__`，planner / executor / replanner 已删除
- A2A 协议实践、Agent 间任务状态交换：项目中没有，不能硬套

#### B31 · Agent Card 是什么

**回答**

**项目中没有，不能硬套**。Agent Card 是 A2A 协议里 Agent 自我描述能力的清单（身份、技能、可接受的输入输出、端点），我这个仓库既没有 A2A 也没有第二个 Agent，不存在需要向外声明能力的场景。

我不拿别的东西冒充它。可能被拿来充数的两样，我说明为什么不是：

一是 MCP 的工具列举。`mcp_tool_provider.get_tools()` 拿到的是**工具**的 schema 列表，不是 Agent 的能力声明——每个 `@mcp.tool()` 描述的是"这个函数怎么调"，粒度是函数级、无状态、无身份。Agent Card 描述的是"这个 Agent 能承接什么任务"，粒度是任务级、有身份、隐含决策能力。把工具 schema 说成 Agent Card 是偷换粒度。

二是 `available_servers` / `unavailable_servers`。这两个返回的是 MCP server 的**健康状态**，用途是排障时知道模型为什么少了几个工具，不是对外的能力声明——它甚至不出进程，只给日志和内部调用方看。

`[假设]` 如果这个仓库真要对外声明能力，我会先解决一个现实问题：**能力声明的可信度**。仓库里已经有一个反面教材——`tool_facts_node` 代码完整、注释里写清了它做什么，但全仓没有调用点、`graph.py` 也没注册它（见 B21、B25）。如果照代码自动生成一张 Card，它会声明一个"能查实时监控事实"的能力，而这个能力在线上根本不会被触发。所以 Card 必须从**实际接线**生成而不是从代码定义生成——比如从 `graph.py` 注册的节点和真正被引用的工具反推，而不是扫描装饰器。这个坑在没有多 Agent 的时候就已经存在了，只是没人对外声明所以没暴露。

**项目证据**
- 全仓无 A2A / AgentCard 相关代码
- `app/agent/mcp_tool_provider.py:113-149` — `get_tools` 返回工具 schema，粒度为函数级
- `app/agent/mcp_tool_provider.py:150-164` — `available_servers` 是健康状态，不出进程
- `app/agent/rag_v2/nodes.py:252` — `tool_facts_node` 定义完整但无调用点
- `app/agent/rag_v2/graph.py:35-41` — 只注册五个节点，未含 `tool_facts`
- Agent Card、能力声明、对外服务发现：项目中没有，不能硬套；补法为 `[假设]`

#### B32 · 怎么做跨框架的互操作

**回答**

这题我有一个**真实且不太起眼**的实例：同一批 MCP 工具同时被两条不同的执行路径消费，而这两条路径的工具调用模型完全不同。

MCP 侧的工具经 `MultiServerMCPClient` 拉过来后，变成 LangChain 的 `BaseTool`。这一步之后：

- `rag_agent_service.py` 那条路走 `create_agent`，把工具 `bind_tools` 给模型，模型自己决定调几次、什么时候停，循环由框架驱动。
- `tool_facts_node` 那条路是手写的单轮调用，模型只给一次 tool_calls，代码用 `asyncio.gather` 并发执行完就结束，循环由我控制。

两条路消费的是同一个 `BaseTool` 对象。这说明**互操作的关键是抽象层的位置**——`BaseTool` 这一层只规定"怎么调用一个工具"，不规定"谁决定何时调用"，所以框架驱动的循环和手写的单轮都能用它。如果 MCP 客户端返回的是绑死在 `create_agent` 上的东西，第二条路就得重新实现一遍。

还有一个更具体的互操作细节，是我踩过才知道的：`_run_one_tool_call` 里统一用 `await tool.ainvoke(args)`，而工具来源是混的——MCP 工具是异步的，本地 `get_current_time` 是同步的。能统一写成 `ainvoke` 是因为 `BaseTool.ainvoke` 对同步工具会自动转线程执行。这就是抽象层在替我抹平差异：调用侧不需要知道每个工具的同异步性质，也不需要维护一张"哪个工具要 await"的表。反过来说，如果哪天引入一个不遵守 `BaseTool` 契约的工具来源，这行代码就得改成分支判断——互操作的成本会立刻从零变成 O(工具来源数)。

**缺的部分**：跨框架的 Agent 互操作（LangGraph 的 Agent 与其他框架的 Agent 互相委派）——**项目中没有，不能硬套**。我这里互操作的粒度是**工具**，不是 Agent；仓库里只有一个决策主体，见 B30。

**项目证据**
- `app/agent/mcp_client.py:336-357` — MCP 工具统一转为 LangChain `BaseTool`
- `app/services/rag_agent_service.py:173-176` — `create_agent` 消费同一批工具，框架驱动循环
- `app/agent/rag_v2/nodes.py:293-321` — 手写单轮调用消费同一批工具
- `app/agent/rag_v2/nodes.py:381-390` — `ainvoke` 统一同步与异步工具的注释说明
- `app/agent/rag_v2/nodes.py:414` — `content = await tool.ainvoke(args)`，无同异步分支
- `app/tools/time_tool.py` — 同步本地工具，与异步 MCP 工具混用
- 跨框架 Agent 互操作、Agent 级委派：项目中没有，不能硬套

### 题库 B-4 · 编码实现（3 题）

> 来源：`https://www.cnblogs.com/itech/p/20111938`

#### B33 · 手写一个带死循环防护的 ReAct 循环

**回答**

ReAct 循环本身**项目中没有，不能硬套**——B6、B7 已经说明：`rag_v2` 是静态 DAG，`create_agent` 那条路的循环是框架内部的，我没写过循环体，也没配 `recursion_limit`（全仓零引用）。所以这题我不能拿代码顶，只能讲我在**非循环**场景下真实写过的那几种防护，以及把它们搬到循环里各自能防住什么。

我手上有四种真实的防护，它们防的是不同的东西：

**第一种，硬计数上限。** `MAX_TOOL_CALLS_PER_TURN = 4`（`nodes.py:45`）。这个能直接搬进 ReAct 循环当步数上限，但要注意它防的是**单轮内的宽度**，不是**轮数的深度**。ReAck 循环里真正会失控的是深度——模型每轮只调一个工具，看起来很老实，但可以调五十轮。宽度上限对这种一点用没有。

这个常量的注释（`nodes.py:40-44`）写的理由是"工具调用参数是模型给的，它可以在一轮里请求 20 个调用"——**参数来自模型输出，所以必须有代码侧的上界**。这条推理对深度完全成立：轮数也来自模型决定，所以也必须有代码侧的上界。区别只是我当时只需要防宽度。

**第二种，超出部分如实记录而不是静默截断。** `_run_tool_calls`（`nodes.py:341-375`）里超限的调用被丢弃，但会往结果里塞一条 `"未执行：单轮工具调用数超过上限 4"`。注释说的理由是"模型请求了 6 个工具却只有 4 个结果，它会以为剩下两个「没查到数据」，进而在答案里说「未发现异常」"。

这条搬到 ReAct 循环里更重要。步数上限触发时如果只是 `break` 出循环，模型（或者下游的总结步骤）看到的是一个**看起来正常结束**的轨迹，它会基于不完整的观察下结论。必须显式告诉它"你被截断了，结论不可靠"。

**第三种，总时间预算。** `chat_total_budget_seconds = 90.0`（`config.py:81`）包在 `asyncio.timeout` 里（`service.py:61`、`:93`）。这是**唯一与模型行为无关**的闸门——步数上限可以被"每步都很快"绕过，时间预算不能。ReAct 循环里我会两个都要：步数防"逻辑绕圈"，时间防"每步都慢"。

**第四种，也是我认为最有价值的一条：超时后仍然交付部分结果。** `service.py:62` 用 `astream(stream_mode="values")` 而不是 `ainvoke`，理由写在模块注释 `service.py:15-27`：`asyncio.timeout` 触发会取消里面的任务，用 `ainvoke` 意味着整个结果全丢，而"改写和检索可能早已完成，那些工作白做了，排查时也看不到到底卡在哪一步"。`values` 模式每个 super-step 吐一个完整快照，超时时手里还有最后一个快照，于是能回答两件事：已经拿到了什么、卡在哪一步。

搬到 ReAct 循环就是：**每轮的 Thought-Action-Observation 都要落到一个循环外可见的容器里**，而不是只活在循环体的局部变量里。截断时把这个容器交出去。我这条是真做了的，只是我的"步"是 DAG 节点不是 ReAct 轮次。

**我没有的、而 ReAct 循环真正需要的是第五种：语义层面的循环检测。** 上面四条全是"计数到了就停""时间到了就停"，防不住"模型用三个不同措辞反复查同一个东西"这种在预算内的空转。真要做，判据得是 `(工具名, 规范化后的参数)` 的重复检测——这个我仓库里没有任何近似实现，`dedup_node`（`nodes.py:449-467`）的去重是对**检索结果文档**按内容指纹去重，不是对**动作**去重，拿它冒充动作级循环检测是偷换概念。

`[假设]` 如果让我现在写，防护会是这四层的组合：`step < MAX_STEPS` 的硬计数、`asyncio.timeout` 的总预算、`(tool, args)` 指纹的连续重复检测（连续 2 次相同就中止而不是等计数用满）、以及截断时把已有轨迹和明确的"被截断"标记一起返回。前两层我有真实实现可搬，第三层是新写的，第四层的模式我有（`_budget_exceeded_answer`，`service.py:155-172`）。

**项目证据**
- ReAct 循环实现：项目中没有，不能硬套。`graph.py:43-48` 是静态 DAG
- 全仓无 `recursion_limit` — 框架侧步数上限也没配
- `app/agent/rag_v2/nodes.py:45` — `MAX_TOOL_CALLS_PER_TURN = 4`，宽度上限
- `app/agent/rag_v2/nodes.py:40-44` — "参数来自模型，必须有代码侧上界"的推理
- `app/agent/rag_v2/nodes.py:341-375` — 超限调用如实记录而非静默截断
- `app/config.py:81` — `chat_total_budget_seconds = 90.0`
- `app/agent/rag_v2/service.py:61` / `:93` — 两条路径都套 `asyncio.timeout`
- `app/agent/rag_v2/service.py:15-27` — `values` 快照换取部分结果的完整推理
- `app/agent/rag_v2/service.py:62` — `astream(stream_mode="values")`
- `app/agent/rag_v2/service.py:155-172` — `_budget_exceeded_answer`，截断时的交付
- `app/agent/rag_v2/nodes.py:449-467` — `dedup_node` 是文档去重，不是动作去重
- 动作级循环检测、`(tool, args)` 指纹去重：项目中没有，不能硬套；补法为 `[假设]`

#### B34 · Chunking 策略怎么选

**回答**

这题我有完整的真实实现，`document_splitter_service.py` 全文 177 行就干这一件事，而且是**三阶段**而不是常见的一刀切。

**第一阶段按标题分。** `MarkdownHeaderTextSplitter`（`:22-30`），切 h1/h2。这是结构分块，切出来的片段边界和文档的逻辑边界重合，标题进 metadata。我的语料是 SOP 运维文档（`aiops-docs/*.md`），天然有标题层级，不用这个纯属浪费已有结构。

**第二阶段按大小再分。** `RecursiveCharacterTextSplitter`（`:32-38`），关键是 `chunk_size=self.chunk_size * 2`，也就是 `800 * 2 = 1600`，注释写的理由是"加倍 chunk_size，减少分片数"。为什么第一阶段不够：一个 h2 小节可能有几千字，超了 embedding 的有效长度，也超了检索粒度——一个 chunk 里塞太多主题会稀释向量。

**第三阶段合并太小的片。** `_merge_small_chunks(docs, min_size=300)`（`:135-176`）。这一阶段是前两阶段的**必然后果**：按标题切会产出"只有一个标题加两行字"的片段，那种片段单独进向量库是噪声——它的向量几乎全由标题词决定，会在各种不相关的查询下命中。

这一阶段有个细节值得说，因为它是我读代码时才注意到的：合并的判据是

```python
elif doc_size < min_size and len(current_doc.page_content) < self.chunk_size * 2:
    current_doc.page_content += "\n\n" + doc.page_content
```

判的是**下一片**小（`doc_size < min_size`），不是当前累积片小。后果是：一个 500 字的片后面跟一个 100 字的片，会合并成 600 字；但一个 100 字的片后面跟一个 500 字的片，**不会**合并——那个 100 字的片会被直接 append 出去。所以这个函数消除的是"小片段出现在大片段后面"，消除不了"小片段出现在开头"。这是个真实的不对称，我不打算把它说成设计。

三个参数的实际取值：`chunk_max_size: int = 800`、`chunk_overlap: int = 100`（`config.py:165-166`），合并阈值 300 是 `_merge_small_chunks` 的默认参数。overlap 100 只作用在第二阶段——第一阶段按标题切不需要 overlap，因为标题边界本身就是语义边界，跨边界的上下文由 metadata 里的标题补。

**非 Markdown 走另一条路。** `split_document`（`:119-134`）按扩展名分派，`.md` 走三阶段，其余走 `split_text` 的纯字符递归。这不是偷懒，是承认"结构分块的前提是有结构"。

**分块之后还有一层我做了但不常被算进"chunking"的东西。** `metadata_enricher.py` 给每个 chunk 注入 `doc_tags`（从文件名和 h1 推断的文档级标签，`:56-68`）和 `chunk_summary`（首句 ≤120 字，`:69`）。`chunk_summary` 的注释说是"用于 Rerank / 前端预览"。这解决的是分块的一个固有损失：chunk 被切下来之后就脱离了上下文，一个只说"重启服务"的片段，读者不知道是哪个服务的 SOP。把父级信息塞进 metadata 是最省的补法——比 parent-child 双层索引便宜得多，而且我不需要真的去取父文档。

**我没有的策略，说清楚：** 语义分块（按 embedding 相似度找切点）、父子块双层索引、按 token 而不是字符计的分块、针对表格和代码块的特殊处理——**项目中没有，不能硬套**。字符数和 token 数的偏差在中文语料下不小（`estimate_tokens` 里 CJK 是 1:1，英文约 4:1），我的 800 字符对中文接近 800 token，对英文只有 200 token 左右，同一个参数在两种语料下的实际粒度差四倍。这个我知道，没改，因为语料是中文 SOP。

**项目证据**
- `app/services/document_splitter_service.py:22-30` — `MarkdownHeaderTextSplitter`，第一阶段按 h1/h2
- `app/services/document_splitter_service.py:32-38` — `RecursiveCharacterTextSplitter`，`chunk_size * 2`
- `app/services/document_splitter_service.py:45-81` — `split_markdown` 三阶段流程与元数据注入
- `app/services/document_splitter_service.py:135-176` — `_merge_small_chunks`
- `app/services/document_splitter_service.py:160` — 判据是「下一片小」的不对称
- `app/services/document_splitter_service.py:119-134` — `split_document` 按扩展名分派
- `app/config.py:165-166` — `chunk_max_size = 800`、`chunk_overlap = 100`
- `app/services/metadata_enricher.py:37` — `enrich_documents` 注入 `doc_tags` + `chunk_summary`
- `app/services/metadata_enricher.py:56-68` — `_infer_tags` 从文件名与 h1 推断
- `app/services/metadata_enricher.py:69` — `_extract_summary`，首句 ≤120 字
- `app/services/conversation_memory_service.py:48-54` — CJK 1:1 与英文 4:1 的 token 偏差
- 语义分块、父子块双层索引、按 token 分块、表格与代码块特殊处理：项目中没有，不能硬套

#### B35 · SSE 流式和异步工具调用怎么实现

**回答**

两条流式路径我都写了，而且它们的**粒度不一样**——这个区别是这题的核心，混在一起说就等于没答。

**token 级流式，只有链路一有。** `rag_agent_service.py:321-324` 用 `self.agent.astream(..., stream_mode="messages")`，拿到的是 `(token, metadata)` 对，逐 token 往外推。这是用户视觉上的"打字机效果"。

**节点级流式，是 rag_v2 的。** `service.py:94` 用 `astream(stream_mode="updates")`，每个节点执行完吐一个 patch，服务层把 patch 里的字段翻译成语义事件：`sub_queries` → `{"type": "sub_queries"}`、`documents` → `{"type": "retrieved", "data": {"node", "count"}}`、`deduped_documents` → `used_documents`、`answer`、`validation`（`service.py:99-115`）。用户看到的不是字在长出来，而是"正在改写查询→检索到 12 条→用了 6 条→答案→质检结果"这样的进度。

选 `updates` 而不是 `messages` 是被架构逼的：`rag_v2` 的答案在 `generate_node` 内部一次 `ainvoke` 生成，节点返回时答案已经是完整字符串了。要 token 级就得让 `generate_node` 本身变成生成器，那和 LangGraph 的节点契约（`state -> patch`）冲突。所以 `rag_v2` 的"流式"交付的是**过程可见性**，不是**首字延迟**。这个取舍我认。

**同一个 service 里还有第三种 stream_mode，用途完全不同。** `service.py:62` 的非流式 `query` 用 `astream(stream_mode="values")`——不是为了流式输出，是为了**超时能拿到部分结果**。理由在模块注释 `service.py:15-27`：`ainvoke` 被 `asyncio.timeout` 取消时整个结果全丢，`values` 每个 super-step 吐完整快照，把最后一个留在手里，超时时就能回答"已经拿到了什么"和"卡在哪一步"。三种 mode 各解一个问题：`messages` 解首字延迟、`updates` 解过程可见、`values` 解部分结果。

**SSE 层的两个真实约束。** 一是**状态码已经发出去了改不了**。`service.py:118-131` 的注释写得明确："SSE 已经返回了 HTTP 200，改不了状态码，只能用事件告知降级"，所以超时时 yield 一个带 `degrade_reason: TOTAL_BUDGET_EXCEEDED` 的 error 事件而不是抛异常——注释还补了为什么不抛："前面已经 yield 过若干有效事件，此时抛异常会让 SSE 连接以错误方式断开，客户端可能丢掉已收到的内容"。

二是**流式路径也必须落库**。`chat_v2.py:200-232` 在收到终止 chunk 时先 `append_exchange` 写对话、再 `create_trace` 拿 trace_id、再 `flush_spans(db, trace_id=trace.id)`，最后才 yield `{"type": "done"}`。顺序是有讲究的：span 落盘必须等 trace 建好，因为要外键。注释写的是"和非流式路径同一时机"——两条路径的落库时机对齐，避免只有非流式有 trace。

**异步工具调用。** `_run_one_tool_call`（`nodes.py:381-434`）里一律 `await tool.ainvoke(args)`，不分同步异步。注释 `nodes.py:386-390` 给了理由："MCP 工具是异步的，而本地 `get_current_time` 是同步的——`BaseTool.ainvoke` 对同步工具会自动"（在线程池里跑）。所以调用侧不需要写 `if iscoroutine` 分支，`time_tool.py` 那个纯同步函数和 MCP 的异步工具在同一个 `asyncio.gather` 里混跑。

并行是 `_run_tool_calls`（`nodes.py:341-361`）的 `asyncio.gather`，理由是"两个工具调用之间没有依赖，串行执行会让耗时变成两次 MCP 往返之和——而本节点的耗时直接计入总预算"。

**一个必须说的事实：** 上面这个异步工具调用链（`tool_facts_node` → `_run_tool_calls` → `_run_one_tool_call`）**没有接进图**。`graph.py:35-41` 只注册了五个节点，`tool_facts_node` 全仓除了自己的定义行没有任何调用点。所以"与检索并行的工具事实分支"是写完了但没接线的状态——代码、注释、配置开关（`enable_tool_facts`、`tool_facts_budget_seconds`）、state 字段（`state.py:42`）全齐，就是没在 `graph.py` 里 `add_node`。`generate_node:502-509` 已经在读 `state["tool_facts"]` 了，读到的永远是空列表。

这个我如实说，不包装成"预留扩展点"。真实情况是接线漏了。B21 里同一个事实。

**项目证据**
- `app/services/rag_agent_service.py:321-324` — `stream_mode="messages"`，token 级流式
- `app/agent/rag_v2/service.py:94` — `stream_mode="updates"`，节点级事件
- `app/agent/rag_v2/service.py:99-115` — patch 字段翻译成五种语义事件
- `app/agent/rag_v2/service.py:62` — `stream_mode="values"`，为部分结果而非流式
- `app/agent/rag_v2/service.py:15-27` — 三种 mode 取舍的完整推理
- `app/agent/rag_v2/service.py:118-131` — SSE 200 已发出，用事件而非异常告知降级
- `app/api/chat_v2.py:200-232` — 流式路径落库顺序：对话 → trace → span → done
- `app/api/chat.py:83-141` — 链路一的六种 SSE 事件类型
- `app/agent/rag_v2/nodes.py:381-390` — `ainvoke` 统一同步与异步工具
- `app/agent/rag_v2/nodes.py:414` — `content = await tool.ainvoke(args)`
- `app/agent/rag_v2/nodes.py:341-361` — `asyncio.gather` 并行与理由
- `app/agent/rag_v2/graph.py:35-41` — 只注册五个节点，`tool_facts_node` 不在其中
- `app/agent/rag_v2/nodes.py:502-509` — `generate_node` 读 `tool_facts`，实际恒为空
- `tool_facts_node` 的接线：项目中缺失，是漏接而非预留

### 题库 B-5 · 安全、可控性与评估（9 题）

> 来源：`https://www.cnblogs.com/itech/p/20111938`

#### B36 · Prompt Injection 怎么防

**回答**

先说我项目里**有**的三层，再说最大的那个洞——因为那个洞比这三层重要。

有的部分是**结构隔离与来源声明**。`nodes.py:619-625` 拼 prompt 时把不同来源的内容装进不同的 XML 风格标签：`<question>` `<answer>` `<evidence>` `<conversation_memory>`，并且在标签之外用一句自然语言明确各自的效力——"仅将 `<evidence>` 中的内容作为知识库可验证证据；会话记忆只能帮助判断对象指代"。`MEMORY_POLICY`（`nodes.py:106-108`）把这条声明抽成常量复用，防的是模型把会话记忆里的历史数值当成实时事实。

第二层是**下游不信任模型输出的形状**。`_parse_validation_result`（`nodes.py:784-814`）不管模型返回什么，都过一遍 `_coerce_score` 和 `_coerce_str_list` 做类型强转，解析不出来就落 `_default_validation`。这一层挡的不是攻击者，是"模型输出污染下游逻辑"——但它顺带也挡住了一部分注入后果：注入能改模型说什么，改不了下游读到的字段类型。

第三层是 `aiops_report.py:64-80` 的 `normalize_model_evidence_source`。模型如果把自己的推断标成 `SOP 3: xxx` 这种像真实来源的标签，validator 无条件把它归回 `EvidenceSource.MODEL`。这条是**真的在防冒充**：模型不能自己给自己颁发"可溯源证据"的身份。

**最大的洞：知识库内容是完全可信输入的假设，而这个假设没有任何东西在守。** 上传接口 `api/file.py:48` 只做了 `_sanitize_filename`（`:299-314`，把空格和特殊字符换成下划线），处理的是**文件名**，不是**内容**。文档内容经 `document_splitter_service.split_document` 切块后直接进 Milvus，检索出来直接经 `_format_context`（`nodes.py:903-948`）拼进 prompt——中间没有任何一处检查过内容里有没有"忽略以上指令"这类文本。加上项目**完全没有鉴权层**（全仓无 `HTTPBearer` / `api_key` / `current_user`），意味着任何能访问到这个服务的人都能上传一份文档，让它在之后每一次相关检索里被注入到 prompt 里。这是一条完整的间接注入链，而我这三层防的都不是它。

诚实的结论：我做的是**来源隔离和输出净化**，不是**注入检测**。注入检测、内容过滤、分隔符转义——**项目中没有，不能硬套**。

`[假设]` 要补的话我会按代价从低到高排：先在 `_format_context` 里对拼进 prompt 的 chunk 做一次分隔符转义（把内容里出现的 `</evidence>` 之类闭合标签打掉，这是最便宜也最有效的一步）；然后在上传入口加内容侧的指令特征扫描并标记，命中的走人工确认——复用 `protocol_ingestion` 已有的 `AWAITING_CONFIRMATION` 流程，不用新造机制；最后才是补鉴权，让"谁能往知识库里塞东西"这件事有边界。顺序反过来做的话，前两步在无鉴权的前提下意义不大。

**项目证据**
- `app/agent/rag_v2/nodes.py:619-625` — XML 标签隔离不同来源 + 效力声明
- `app/agent/rag_v2/nodes.py:106-108` — `MEMORY_POLICY` 常量
- `app/agent/rag_v2/nodes.py:784-814` — `_parse_validation_result` 类型强转与兜底
- `app/models/aiops_report.py:64-80` — `normalize_model_evidence_source`，禁止模型自我颁发证据身份
- `app/api/file.py:299-314` — `_sanitize_filename` 只处理文件名
- `app/api/file.py:48` — 上传入口，内容未做任何过滤
- `app/agent/rag_v2/nodes.py:903-948` — `_format_context` 直接拼接检索内容
- 注入检测、内容过滤、分隔符转义、鉴权：项目中没有，不能硬套；补法为 `[假设]`

#### B37 · 越狱防护怎么做

**回答**

**项目中没有，不能硬套。** 全仓没有越狱检测、没有拒答分类器、没有输出侧的策略过滤，也没有接任何内容安全服务。

这题我要特别说明为什么不能用质检层冒充。`validate_answer_node` 看起来像个"输出审查器"，但它审的维度和越狱完全无关：`VALIDATION_SYSTEM_PROMPT`（`nodes.py:79-104`）问的两个问题是 coverage（问题关键点被证据覆盖多少）和 groundedness（答案陈述被证据支撑多少）。一个成功的越狱输出——比如模型被诱导输出了违规内容——如果那段内容恰好在检索证据里有依据，groundedness 会打**高分**，质检层会**放行**。它甚至可能比正常答案更容易过，因为越狱 prompt 常常引导模型复述输入内容。拿它当越狱防护，方向是错的。

真实边界是：这个服务的场景是内部知识库问答和 AIOps 告警诊断，工具全是只读的（查监控、查日志、查知识库、查时间），没有任何写操作工具。越狱的收益上限被**工具集的只读性**限制住了——模型被诱导说了什么，造不成外部副作用。这是"风险低"而不是"有防护"，两者不能混为一谈，面试里说成后者是虚构。

`[假设]` 如果这个服务要对外，我会把防护放在输出侧而不是输入侧，理由是输入侧的模式匹配对改写极其脆弱。具体是在 `validate_answer_node` 之后再挂一个独立的策略检查节点——独立而不是塞进现有质检器，因为两者的失败方向必须相反：证据不足时质检 fail-open 放行答案（`nodes.py:695-718`），而策略检查必须 fail-closed，检查器挂了就不能发。把两种相反的降级策略塞进同一个节点，一定会有一方被写错。

**项目证据**
- 全仓无越狱检测、拒答分类器、内容安全服务调用
- `app/agent/rag_v2/nodes.py:79-104` — 质检器只评 coverage / groundedness，与策略无关
- `app/agent/rag_v2/nodes.py:695-718` — 质检 fail-open，与策略检查所需方向相反
- `app/tools/knowledge_tool.py:54` / `app/tools/time_tool.py` / `mcp_servers/monitor_server.py:124`、`:277` — 工具全为只读
- 越狱防护：项目中没有，不能硬套；补法为 `[假设]`

#### B38 · 工具调用的安全边界怎么划

**回答**

我项目里的边界是**三道真实的硬限制加一个靠"碰巧"成立的前提**，后者是这题真正值得讲的部分。

三道硬限制都在 `nodes.py`。第一是**调用数上限**：`MAX_TOOL_CALLS_PER_TURN = 4`（`:45`），注释里写清了理由——工具调用参数是模型给的，它可以在一轮里请求 20 个调用，每个都是一次 MCP 往返，并发打满会同时压垮 MCP 进程和总预算。关键是超出部分**丢弃并如实记成失败条目**（`_run_tool_calls`，`:361-372`），不是静默截断：模型请求了 6 个却只拿到 4 个结果，它会以为剩下两个"没查到数据"，进而在答案里说"未发现异常"。

第二是**返回内容上限**：`MAX_TOOL_CONTENT_CHARS = 1200`（`:52`）。日志类工具能吐几万字符，原样拼进 prompt 会把知识库证据和会话记忆一起挤出上下文窗口——那等于用实时数据换掉了 SOP 证据。截断而不是丢弃，因为日志头部通常就包含关键错误行。

第三是**工具名白名单**：`_run_one_tool_call`（`:381-398`）用 `by_name` 字典查找，模型请求不存在的工具时走 `logger.warning(f"模型请求了不存在的工具: {name}")` 并返回失败条目，不会尝试反射调用。白名单是 `by_name` 这个 dict 本身，来源是实际注册的工具列表。

**靠"碰巧"成立的前提是：所有工具都是只读的。** `query_cpu_metrics`、`query_memory_metrics`、`search_log`、`get_current_time`、`retrieve_knowledge` 全是查询，仓库里不存在任何一个会改状态的工具。所以我从来没有面对过"模型请求了一个危险操作"这个问题——不是我设计了权限分级挡住了它，是**没有可被滥用的工具**。参数校验也只有 pydantic 从函数签名自动生成的类型检查（`monitor_server.py:126-131` 的 `service_name: str` 之类），没有值域白名单，没有按工具的调用频率限制，没有危险操作的二次确认。

`[假设]` 加第一个写操作工具（比如"重启服务"）的那天，这套边界立刻不够用，必须同时补三样：工具按副作用分级（只读 / 可逆写 / 不可逆），不可逆的强制走人工确认——这里可以复用 `protocol_ingestion` 已有的 `AWAITING_CONFIRMATION` 状态机（`models/protocol_ingestion.py:24`）而不是新造；参数值域白名单（服务名必须在已知列表里，防模型编一个服务名）；以及调用审计要能追到人，而这一步会撞上"没有鉴权层"这个更底层的缺口。

**项目证据**
- `app/agent/rag_v2/nodes.py:40-45` — `MAX_TOOL_CALLS_PER_TURN = 4` 与理由
- `app/agent/rag_v2/nodes.py:361-372` — 超限调用丢弃但如实记为失败
- `app/agent/rag_v2/nodes.py:47-52` — `MAX_TOOL_CONTENT_CHARS = 1200` 与截断理由
- `app/agent/rag_v2/nodes.py:381-398` — `by_name` 白名单查找，未知工具走 warning
- `mcp_servers/monitor_server.py:126-131` — 参数只有签名级类型约束
- `app/models/protocol_ingestion.py:24` — 可复用的人工确认状态
- 工具副作用分级、参数值域白名单、调用频率限制、危险操作二次确认：项目中没有，不能硬套；补法为 `[假设]`

#### B39 · HITL 里哪些操作必须人工确认

**回答**

我项目里有一条**真实跑通的人工确认流程**，也有一处**只表达意图但没有强制力**的字段，两者的区别正是这题的答案。

真实的那条是协议 PDF 入库。状态机 `models/protocol_ingestion.py:24` 有 `AWAITING_CONFIRMATION = "awaiting_confirmation"`，仓储层 `confirm_with_state_trace`（`repositories/protocol_ingestion_repository.py:111-121`）在确认时写入 `confirmed_by` 和 `confirmed_at`，接口层是一对 `/confirm` 和 `/reject`（`api/protocol_pdf.py:122-140`、`:141+`）。这条流程的**强制力来自状态机**：没经过 `confirm_ingestion`，数据就进不了 `protocol_catalog`——不是靠调用方自觉，是靠状态转移的前置条件。

我卡人工确认的判据是**不可逆 + 影响面超出单次请求**。PDF 入库两条都满足：解析结果写进目录表后会被后续所有检索读到，一份解析错的协议会持续污染答案，而且没有回滚脚本。对照着看，知识库文档上传**没有**卡确认——因为它是软删除模型（`knowledge_base.py` 的 `is_deleted`），删错了能恢复，可逆性把它降到了不需要人工的档位。

**只表达意图的那处**是 AIOps 报告里的 `requires_human: bool`（`aiops_report.py:97`）和 `pending_confirmations: list[str]`（`:121-122`）。它们由模型填写，渲染时 `:159` 会在检查项后面加"（需人工确认）"字样、`:179-181` 会把待确认项单独列出来。但这只是**报告文本里的一句提示**——报告是给值班同学看的 Markdown，后面没有任何自动执行环节被这个 flag 门控。它是建议，不是闸门。这两者在面试里必须分清：一个是机制，一个是文案。

还有一个诚实的缺口：`confirmed_by` 是请求体里的字符串，默认值 `"manual-reviewer"`（`api/protocol_pdf.py:28`），任何调用方都能填任何名字。审计字段存在，但**审计链是断的**——因为项目没有鉴权层，服务端无法验证这个名字。所以这条流程满足"有人确认过"的流程要求，不满足"能追溯到具体是谁"的审计要求。

**项目证据**
- `app/models/protocol_ingestion.py:24` — `AWAITING_CONFIRMATION` 状态
- `app/repositories/protocol_ingestion_repository.py:105` — 转入待确认
- `app/repositories/protocol_ingestion_repository.py:111-121` — `confirm_with_state_trace` 写 `confirmed_by` / `confirmed_at`
- `app/api/protocol_pdf.py:122-140` — `/confirm` 接口
- `app/api/protocol_pdf.py:28` — `confirmed_by` 默认 `"manual-reviewer"`，服务端不校验
- `app/models/aiops_report.py:97` — `requires_human`，仅报告文案
- `app/models/aiops_report.py:121-122`、`:159`、`:179-181` — `pending_confirmations` 的渲染
- 身份可验证的审计链、被 `requires_human` 门控的自动执行环节：项目中没有，不能硬套

#### B40 · 目标漂移怎么检测和纠正

**回答**

**项目中没有目标漂移的检测机制，不能硬套。** 没有目标状态对象，没有跨轮的目标一致性校验，也没有偏离后的纠正回路。

不能拿来冒充的两处我要点名。一是 `MEMORY_POLICY`（`nodes.py:106-108`）防的是**记忆污染**——把历史指标写成当前实时值，属于事实时效性问题，跟"任务目标被带偏"是两回事。二是会话摘要 prompt（`conversation_memory_service.py:28-35`）里确实有【任务进展】和【待确认项】两个固定段落，这是仓库里最接近"目标状态"的东西，但它的形态是**一段中文文本**，被整体注入 prompt 而已。没有任何代码读它、比对它、或在它与当前轮不一致时做判断。它是给模型看的上下文，不是给程序用的状态。

真正的原因还是架构：`rag_v2` 是固定五节点 DAG（`graph.py:43-48`），一次请求只走一遍，没有让目标漂移得以发生的多步自主循环。目标漂移是**长程自主执行**的问题，我这个系统在单次请求内就没有"长程"。

对话层面倒是有一个真实的弱化对照：摘要的快照游标机制（`conversation_memory_service.py:104-126`）保证摘要只压缩游标之前的消息，游标之后的原文原样注入。这让"最近几轮用户实际说了什么"永远不会被摘要改写掉——间接降低了多轮对话里目标被摘要模糊化的风险。这是一个副作用，不是设计意图，说成目标漂移防护是夸大。

`[假设]` 如果要做，我会把目标做成**结构化状态而不是文本**：首轮抽取出目标字段存进 state，之后每轮拿当前请求与目标做一致性判定，偏离超阈值时不是自动纠正而是显式向用户确认——自动纠正在目标本身抽错的情况下会把错误固化，代价比漂移更大。

**项目证据**
- `app/agent/rag_v2/graph.py:43-48` — 固定 DAG，单次请求无长程循环
- `app/agent/rag_v2/nodes.py:106-108` — `MEMORY_POLICY` 防的是时效性污染，非目标漂移
- `app/services/conversation_memory_service.py:28-35` — 摘要含【任务进展】【待确认项】，但仅为文本
- `app/services/conversation_memory_service.py:104-126` — 快照游标保留近轮原文
- 目标状态对象、一致性校验、纠正回路：项目中没有，不能硬套；补法为 `[假设]`

#### B41 · Guardrails 用规则还是模型判断

**回答**

我项目里两种都有，而且我能给出一个**不来自文章、来自我自己代码**的判据：**看这道护栏自己会不会挂。**

规则型的三处。`nodes.py:36-37` 的 `MIN_GROUNDEDNESS_SCORE = 0.75` / `MIN_COVERAGE_SCORE = 0.60` 是硬阈值比较；`aiops_first_response_eval.py:91-108` 的 `score_hallucination_controlled` 是纯布尔逻辑——"有 VERIFIED_FACT 证据则允许给根因假设；一条已验证事实都没有却给了根因、且待确认项为空 → 判为无据编造"；`aiops_report.py:64-80` 的 `normalize_model_evidence_source` 是 pydantic validator 层的无条件改写。

模型型的一处：`validate_answer_node`，让 LLM 读问题、答案、证据，输出 coverage 和 groundedness 两个分数。

判据是这样来的。规则护栏是**纯函数**，输入确定输出确定，不依赖外部服务，所以它**不需要降级路径**——没有"阈值比较失败了怎么办"这种情况。模型护栏是一次网络调用，它会超时、会被熔断、会返回不可解析的内容，所以它**必须**配降级路径，而且降级方向还得分两种，这两种在我代码里方向恰好相反：

- **调用失败**（超时、熔断、鉴权）→ fail-open。`nodes.py:695-718` 放行答案、标 `validated=False`、记降级原因。理由写在注释里：拦掉的话质检模型一抖动，每个用户都收到"当前证据还不够支持直接下结论"，而证据明明是好的，挂的是质检器——用假原因掩盖真原因。
- **输出不可解析** → fail-closed。`nodes.py:810-814` 走 `_default_validation("质检器输出不可解析，已按未通过处理")`，`blocked=True`，答案被换成保守回答。

同一个节点、两种失败、两个相反方向。区别在于前者我**知道**质检没跑（服务挂了），后者我**不知道**质检结论是什么（跑了但读不懂）——不知道结论时按未通过处理才是保守的。这个区分我抽成了两个函数，`_unvalidated_validation`（`:848-880`）和 `_default_validation`（`:830-846`），docstring 里明写"只差一个字段，语义却完全相反"。

所以我的结论：**能用规则判的一律用规则**，因为规则不引入新的故障点，也不引入"护栏挂了算过还是算不过"这个必须回答且容易答错的问题。只在判断需要语义理解时才上模型，并且上了就必须同时把两个方向的降级都想清楚。混合结构在我这儿是天然的——模型给分数，规则比阈值，`_coerce_score`（`:816-828`）在中间兜住类型。

**项目证据**
- `app/agent/rag_v2/nodes.py:36-37` — 规则阈值常量
- `tests/eval/aiops_first_response_eval.py:91-108` — `score_hallucination_controlled` 纯规则判定
- `app/models/aiops_report.py:64-80` — pydantic validator 层的规则强制
- `app/agent/rag_v2/nodes.py:79-104` — 模型型护栏的 prompt
- `app/agent/rag_v2/nodes.py:695-718` — 调用失败 fail-open
- `app/agent/rag_v2/nodes.py:810-814` — 输出不可解析 fail-closed
- `app/agent/rag_v2/nodes.py:848-880` — `_unvalidated_validation`，docstring 说明两者语义相反
- `app/agent/rag_v2/nodes.py:830-846` — `_default_validation`
- `app/agent/rag_v2/nodes.py:816-828` — `_coerce_score` 类型兜底

#### B42 · AgentBench / WebArena / SWE-bench 分别评什么

**回答**

这三个 benchmark 我**都没跑过，项目中没有，不能硬套**。原理层面：AgentBench 覆盖多环境下的 Agent 综合能力，WebArena 评网页环境里的真实任务完成率，SWE-bench 评从 issue 到可通过测试的补丁的代码修复能力。这三句是我读来的，不是实践。

我实际有的是**自建离线检索评测**，性质完全不同，得说清边界。`tests/eval/` 下有 `metrics.py` 实现 `hit_at_k`（`:75`）、`recall_at_k`（`:81`）、`reciprocal_rank`（`:90`），`run_eval.py` 是 CLI 入口，支持 `--tag` 打实验名、`--retriever` 切检索器、`--compare A B` 对比两次报告（`:672-700`），报告落 JSON 便于复现。评测集是 `golden_set.jsonl` 18 条人工整理加 `golden_set_expanded.jsonl` 83 条（含基于 SOP 的合成改写）。

它和上面三个 benchmark 的差距在三个层面，任何一个都足以让"我做过 Agent 评测"这句话变成虚构：

其一，**评的对象不同**。我评的是检索召回——给一个问题，看期望的 doc_id 有没有进 Top-K。三个 benchmark 评的是端到端任务完成。我的指标里没有任何一个能回答"这个任务做成了没有"。

其二，**没有环境**。WebArena 有可交互的网页环境，SWE-bench 有能跑测试的代码仓库。我的评测里检索器是被注入的对象，甚至有个 `CheatRetriever`（`run_eval.py:87`）直接按标注返回，用来验证评测框架本身而非系统。没有环境就没有"任务"，只有"查询"。

其三，**生成侧的指标是空的**。见 B43。

另一份 `aiops_first_response_eval.py` 评的是告警首响报告的三个维度（SOP 命中、要点覆盖、幻觉受控），比检索评测更接近"任务质量"，但它跑的是 `_StubLLM`（`run_aiops_eval.py:42`）——桩模型返回预置要点，评的是**编排逻辑**是否正确组装报告，不是真实模型的能力。这个设计是刻意的（离线可复现、不烧 token、CI 里能跑），但它决定了这份评测不能用来回答"模型行不行"。

**项目证据**
- `tests/eval/metrics.py:75`、`:81`、`:90` — `hit_at_k` / `recall_at_k` / `reciprocal_rank`
- `tests/eval/run_eval.py:672-700` — CLI 参数与 `--compare`
- `tests/eval/run_eval.py:87` — `CheatRetriever`，验证框架而非系统
- `tests/eval/golden_set.jsonl` 18 条 / `golden_set_expanded.jsonl` 83 条
- `tests/eval/aiops_first_response_eval.py:110-127` — `score_case` 三维度
- `tests/eval/run_aiops_eval.py:42`、`:72` — `_StubLLM` / `_StubSop`
- AgentBench / WebArena / SWE-bench 实跑经验、交互环境、端到端任务完成率：项目中没有，不能硬套

#### B43 · RAG 检索质量的 Faithfulness 和 Relevance 怎么量化

**回答**

我这题有一个**不太好看但真实**的答案：检索侧的指标是真的跑起来的，生成侧的两个指标**是抛异常的占位函数**。

`tests/eval/metrics.py:174-185`：

```python
def faithfulness_placeholder(*_args, **_kwargs) -> float:
    """TODO(week 2): 接入 ragas.metrics.faithfulness。
    思路:用 LLM 判断 generated_answer 里每个 claim 是否都能由 retrieved_chunks 支撑,
    返回 [支撑的 claim 数 / 总 claim 数]。
    """
    raise NotImplementedError("faithfulness 尚未接入,见 README week 2 计划")

def answer_relevance_placeholder(*_args, **_kwargs) -> float:
    """TODO(week 2): 接入 ragas.metrics.answer_relevancy。"""
    raise NotImplementedError("answer_relevance 尚未接入,见 README week 2 计划")
```

两个都 `raise NotImplementedError`。所以**离线的 Faithfulness / Answer Relevance 量化，项目中没有，不能硬套**。

真的有的是检索侧，而且是可复现的：`hit_at_k`（`:75`）、`recall_at_k`（`:81`）、`reciprocal_rank`（`:90`），`compute_case`（`:96`）默认按 k=(1,3,5,10) 一次算全，`aggregate`（`:120`）出总报告，`run_eval.py --compare A B`（`:692`）比两次实验。我用这套跑出过真实结论：多查询 RRF 正向、CrossEncoder 精排负向。

有意思的是**线上有一个近似替代**，但它不是离线指标，两者不能互换。`validate_answer_node` 让 LLM 输出 `groundedness_score`（答案陈述被证据直接支撑的程度）和 `coverage_score`（问题关键点被证据覆盖的程度）——语义上 groundedness ≈ Faithfulness，coverage ≈ 检索侧的 Relevance。它们落进 `validation` 字典，随 trace 一起入库。差别有三处，每一处都决定它当不了离线指标：

一是**不可复现**。它是线上单次判定，temperature 虽然是 0.0（`:679`）但没有固定评测集，跑两次面对的是不同流量，出的数不可比。离线指标的价值在于 A/B 可比，这个没有。

二是**同模型自评**。生成器和判分器都是 `config.py:46` 的 `qwen-max`——全项目只配了这一个 chat 模型。模型判自己写的答案，偏见方向未知且未测。

三是**它只有单点结论没有分布**。Prometheus 侧只有四个指标（`core/metrics.py`），没有一个记录 groundedness 分数的直方图。所以我知道单次请求的分数，不知道"最近一周 groundedness 的 P50 是多少"——而后者才是质量指标该有的形态。

`[假设]` 补的顺序我想清楚了：先把 `faithfulness_placeholder` 按 docstring 里写的思路实现（claim 级拆分 + 逐条判支撑），跑在现有 101 条评测集上拿到基线；然后把线上 `groundedness_score` 打进直方图，让它从"单次判定"变成"可观测分布"；最后才考虑换judge 模型消除自评偏见——前两步不做，第三步没有对比基准，改了也不知道改好没有。

**项目证据**
- `tests/eval/metrics.py:174-180` — `faithfulness_placeholder` 抛 `NotImplementedError`
- `tests/eval/metrics.py:183-185` — `answer_relevance_placeholder` 抛 `NotImplementedError`
- `tests/eval/metrics.py:75`、`:81`、`:90`、`:96`、`:120` — 检索侧指标与聚合
- `tests/eval/run_eval.py:692` — `--compare` 两次实验对比
- `app/agent/rag_v2/nodes.py:88-89`、`:99-100` — `coverage_score` / `groundedness_score` 定义
- `app/agent/rag_v2/nodes.py:722-737` — 分数落进 `validation` 与阈值
- `app/config.py:46` — 全项目仅一个 `qwen-max`，生成与判分同模型
- `app/core/metrics.py:85-110` — 四个指标，无 groundedness 直方图
- 离线 Faithfulness / Answer Relevance、分数分布可观测、异构 judge：项目中没有，不能硬套；补法为 `[假设]`

#### B44 · LLM-as-Judge 怎么做，有哪些偏见

**回答**

我项目里 `validate_answer_node` 就是一个跑在**线上主链路**上的 LLM-as-Judge，不是离线评测器——这个位置差异决定了它的所有工程约束。

做法很直接：`VALIDATION_SYSTEM_PROMPT`（`nodes.py:79-104`）要求模型只输出严格 JSON，字段是 coverage_score、groundedness_score、两个 pass 布尔、needs_second_retrieval、unsupported_claims、missing_aspects、reason。判分调用 `temperature=0.0, streaming=False`（`:679`），下游按 `MIN_COVERAGE_SCORE = 0.60` / `MIN_GROUNDEDNESS_SCORE = 0.75`（`:36-37`）比阈值。让模型同时输出分数和 pass 布尔，读侧优先取模型的 pass、缺失时才回落阈值比较（`:724-725`），是为了不因为分数刻度理解偏差而误判。

偏见部分，我说三个**我这套实现里确实存在**的，不列通用清单：

**一、同模型自评。** `config.py:46` 全项目只配了 `qwen-max` 一个 chat 模型，生成器和判分器是同一个。模型判自己刚写的答案，倾向于认为自己有依据。我没有做过异构 judge 的对比测量，所以偏见的**方向**我知道、**幅度**我不知道——这一点在面试里必须说，说"我们做了自评所以质量可控"是虚的。

**二、判分器读的是被截断和重排过的证据。** `_format_context`（`:903-948`）按 `rag_document_context_token_budget = 2800`（`config.py:156`）做贪心装载，超预算的文档进不了 judge 的视野。所以 groundedness 评的是"答案能不能被**我喂给它的那部分**证据支撑"，而生成器当时看到的是同一份上下文——两边同源，这让判分器发现不了"证据被裁掉导致答案缺依据"这类问题。它是同一个上下文里的二次检查，不是独立验证。

**三、输出格式偏见落到了兜底逻辑上。** `_parse_validation_result`（`:784-814`）要处理三种情况：整段是 JSON、被 ```json 围栏包裹、文本里嵌了个 `{...}`。这三个分支存在本身就是证据——模型不总是遵守"只输出 JSON"。所有分支都失败时走 `_default_validation("质检器输出不可解析，已按未通过处理")`，fail-closed。这跟调用失败时的 fail-open（`:695-718`）方向相反，理由见 B41。

还有一个刻意的反向决策值得讲：**AIOps 那份评测我故意没用 LLM-as-Judge。** `aiops_first_response_eval.py` 的三个判分器全是规则——`score_sop_hit`（`:60`）关键词匹配、`score_point_coverage`（`:75`）要点覆盖、`score_hallucination_controlled`（`:91`）布尔逻辑，跑的模型是 `_StubLLM`（`run_aiops_eval.py:42`）。理由是评测必须**确定性、可复现、不烧 token、CI 里能跑**，而 LLM judge 四条全违反。线上用 LLM judge 是因为要判语义、允许有噪声；离线评测用规则是因为要可比。同一个项目里两个相反选择，判据是"这个判定需不需要复现"。

**项目证据**
- `app/agent/rag_v2/nodes.py:79-104` — judge 的 prompt 与输出字段
- `app/agent/rag_v2/nodes.py:679` — `temperature=0.0, streaming=False`
- `app/agent/rag_v2/nodes.py:36-37`、`:724-725` — 阈值与 pass 优先级
- `app/config.py:46` — 生成器与判分器同为 `qwen-max`
- `app/agent/rag_v2/nodes.py:903-948` / `app/config.py:156` — judge 看到的证据已被预算裁剪
- `app/agent/rag_v2/nodes.py:784-814` — 三分支 JSON 提取与 fail-closed 兜底
- `app/agent/rag_v2/nodes.py:695-718` — 调用失败 fail-open，方向相反
- `tests/eval/aiops_first_response_eval.py:60`、`:75`、`:91` — 离线判分全为规则
- `tests/eval/run_aiops_eval.py:42` — `_StubLLM`，评编排而非模型
- 异构 judge 对比测量、judge 偏见幅度量化、独立于生成上下文的验证：项目中没有，不能硬套
