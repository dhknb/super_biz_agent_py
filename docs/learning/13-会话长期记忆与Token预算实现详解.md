# 会话长期记忆与 Token 预算实现详解

本文对应项目已实现的持久化会话记忆功能。它解释：为什么原来的聊天虽然保存了历史却没有真正“记住”，本次如何把历史安全地注入 RAG V2，以及如何控制 Token 成本。

## 1. 改造前的问题

原来的新版聊天入口是 `POST /api/chat_v2` 和 `POST /api/chat_v2_stream`。它会把问答写入 PostgreSQL：

```python
ConversationRepository(db).append_exchange(
    request.id,
    user_content=request.question,
    assistant_content=result["answer"],
)
```

但 RAG V2 真正执行时只传入当前问题：

```python
result = await self.graph.ainvoke({"question": question})
```

也就是说：数据库里有历史，前端能恢复历史，模型却看不到历史。

示例：

```text
第 1 轮：服务名是 order-api，环境 prod。
第 2 轮：它 CPU 高应该看什么？
```

第 2 轮只有“它 CPU 高”，模型无法稳定知道“它”是 `order-api`。

## 2. 最终方案

本次采用混合记忆：

```text
conversation_messages：完整原始对话，永久保存、可审计
        +
conversation_memory_snapshots：压缩后的滚动摘要
        +
最近 3 个用户轮次的原始消息
        ↓
固定 Token 上限的 conversation_context
        ↓
RAG V2 改写 → 检索 → 生成 → 证据校验
```

不采用“每轮完整历史都注入”，原因是会话越长，输入 Token、延迟和成本都会线性增长，旧话题还会干扰当前问题。

## 3. 文件清单

| 文件 | 本次作用 |
|---|---|
| `app/models/conversation.py` | 摘要快照 ORM 模型与会话关联 |
| `app/repositories/conversation_repository.py` | 原始消息读取、摘要读写、清空处理 |
| `app/services/conversation_memory_service.py` | Token 估算、压缩、预算控制、上下文渲染 |
| `app/config.py` | 记忆与文档 Token 配置 |
| `.env.example` | 配置示例 |
| `app/api/chat_v2.py` | 请求前加载记忆并传给 RAG |
| `app/agent/rag_v2/state.py` | 新增图状态字段 |
| `app/agent/rag_v2/service.py` | 记忆向 LangGraph 透传 |
| `app/agent/rag_v2/nodes.py` | 改写、生成、校验中的记忆使用与文档裁剪 |
| `migrations/versions/20260823_0007_create_conversation_memory_snapshots.py` | 新表迁移 |
| `tests/unit/test_conversation_memory_service.py` | 记忆单元测试 |

## 4. 原始历史与摘要快照

### 4.1 原始消息仍是事实源

原有模型 `ConversationMessage` 不删除：

```python
class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[str]
    session_id: Mapped[str]
    role: Mapped[ConversationMessageRole]
    content: Mapped[str]
    message_metadata: Mapped[dict | None]
    created_at: Mapped[datetime]
    deleted_at: Mapped[datetime | None]
```

职责：

```text
保存完整会话
恢复前端聊天记录
作为重新生成摘要的来源
支持审计和软删除
```

摘要不会替代原始记录，只决定哪些内容进入模型 Prompt。

### 4.2 新增快照表

文件：`app/models/conversation.py`。

```python
class ConversationMemorySnapshot(Base):
    __tablename__ = "conversation_memory_snapshots"

    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("conversation_sessions.id"),
        primary_key=True,
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    summarized_through_message_id: Mapped[str | None]
    version: Mapped[int] = mapped_column(default=1, nullable=False)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
```

字段解释：

| 字段 | 含义 |
|---|---|
| `session_id` | 主键；一段会话只保存一份当前滚动摘要 |
| `summary` | 压缩后、可直接注入模型的文本 |
| `summarized_through_message_id` | 摘要已经覆盖到哪条原始消息 |
| `version` | 每次更新摘要时递增 |
| 时间字段 | 方便审计摘要的创建与更新 |

`summarized_through_message_id` 用于避免重复压缩。

```text
m1 用户：服务是 order-api
m2 助手：已记录
m3 用户：CPU 高
m4 助手：检查进程
m5 用户：它现在怎么样
```

若摘要已覆盖到 `m4`：

```text
summary = “服务是 order-api，正在排查 CPU 高……”
summarized_through_message_id = m4
```

后续只会处理 `m5` 与新增消息，不会反复摘要 `m1~m4`。

### 4.3 会话一对一关联

```python
class ConversationSession(Base):
    memory_snapshot: Mapped["ConversationMemorySnapshot | None"] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        uselist=False,
    )
```

- `uselist=False` 表示一段会话对应一条摘要快照。
- `delete-orphan` 保证会话关联对象被删除时摘要也会删除。

## 5. 数据库迁移

迁移文件：`migrations/versions/20260823_0007_create_conversation_memory_snapshots.py`。

核心操作：

```python
op.create_table(
    "conversation_memory_snapshots",
    sa.Column("session_id", sa.String(length=64), nullable=False),
    sa.Column("summary", sa.Text(), nullable=False),
    sa.Column("summarized_through_message_id", sa.String(length=64), nullable=True),
    sa.Column("version", sa.Integer(), nullable=False),
    sa.Column("created_at", sa.DateTime(), nullable=False),
    sa.Column("updated_at", sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(["session_id"], ["conversation_sessions.id"]),
    sa.PrimaryKeyConstraint("session_id"),
)
```

迁移只新增一张表，不会删除、覆盖或迁移已有的聊天消息。

执行命令：

```bash
cd /home/dong/projects/super_biz_agent_py
.venv/bin/alembic upgrade head
```

## 6. 配置和 Token 预算

位置：`app/config.py`。

```python
rag_document_context_token_budget: int = 2800
conversation_memory_context_token_budget: int = 1800
conversation_memory_summary_token_budget: int = 700
conversation_memory_compact_threshold_tokens: int = 2400
conversation_memory_recent_turns: int = 3
```

`.env` 中可覆盖：

```dotenv
RAG_DOCUMENT_CONTEXT_TOKEN_BUDGET=2800
CONVERSATION_MEMORY_CONTEXT_TOKEN_BUDGET=1800
CONVERSATION_MEMORY_SUMMARY_TOKEN_BUDGET=700
CONVERSATION_MEMORY_COMPACT_THRESHOLD_TOKENS=2400
CONVERSATION_MEMORY_RECENT_TURNS=3
```

| 配置 | 默认值 | 含义 |
|---|---:|---|
| 会话记忆总预算 | 1800 | 摘要、最近消息、标题、角色前缀合计硬上限 |
| 摘要预算 | 700 | 滚动摘要最多可占 Token 数 |
| 摘要触发阈值 | 2400 | 未摘要旧消息超过该值才调用摘要模型 |
| 最近轮数 | 3 | 仍保留完整原文的最近用户轮次 |
| 文档预算 | 2800 | RAG 检索片段的总 Token 上限 |

这样聊天 10 轮与 100 轮时，模型收到的会话记忆大小都保持在可控范围。

## 7. 核心服务入口

文件：`app/services/conversation_memory_service.py`。

返回对象：

```python
@dataclass(frozen=True)
class ConversationMemoryContext:
    text: str = ""
    estimated_tokens: int = 0
    used_summary: bool = False
    recent_message_count: int = 0
```

| 字段 | 作用 |
|---|---|
| `text` | 最终注入 RAG 的文本 |
| `estimated_tokens` | 便于验证预算是否生效 |
| `used_summary` | 是否用了持久化摘要 |
| `recent_message_count` | 实际保留了多少条最近消息 |

`frozen=True` 让上下文对象构造后不可被意外修改。

### 7.1 Token 估算

```python
def estimate_tokens(text: str) -> int:
    cjk_chars = len(_CJK_RE.findall(text))
    other_chars = max(0, len(text) - cjk_chars)
    return cjk_chars + math.ceil(other_chars / 4)
```

估算规则：

```text
中文：约 1 汉字 = 1 Token
英文、数字、符号：约 4 字符 = 1 Token
```

它不替代真实 tokenizer，但对中文混合运维文本更保守，避免低估导致上下文爆量。

### 7.2 `build_context()` 主流程

```python
async def build_context(self, db: Session, session_id: str):
    repo = ConversationRepository(db)
    messages = repo.list_active_messages(session_id)
    if not messages:
        return ConversationMemoryContext()

    snapshot = repo.get_memory_snapshot(session_id)
    pending_messages = self._messages_after_snapshot(
        messages,
        snapshot.summarized_through_message_id if snapshot else None,
    )
```

先读取完整未软删除消息和已有摘要快照，再根据摘要游标找到“尚未压缩”的消息。

下面继续说明压缩、Prompt 注入与测试。

## 8. 何时压缩：摘要与最近窗口并存

`build_context()` 的核心分段逻辑：

```python
recent_count = self._recent_message_count(pending_messages)
compact_candidates = pending_messages[:-recent_count] if recent_count else []

if self._token_count(compact_candidates) >= self._compact_threshold_tokens:
    summary = await self._summarize(
        existing_summary=snapshot.summary if snapshot else "",
        messages=compact_candidates,
    )
```

这里把未摘要消息分为两部分：

```text
旧消息：可压缩为摘要
最近 3 个用户轮次：保持原文，保留细节
```

为什么最近消息不直接摘要？因为最新对话往往包含：

```text
当前用户的精确问题
刚刚给出的日志片段
刚设置的临时约束
刚刚确认/否定的结论
```

这些信息若立刻压缩，细节容易损失；旧消息则用摘要保留长期状态。

### 8.1 按“用户轮次”保留尾部

```python
def _recent_message_count(self, messages):
    user_messages_seen = 0
    start_index = 0
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role.value == "user":
            user_messages_seen += 1
            if user_messages_seen >= self._recent_turns:
                start_index = index
                break
    return len(messages) - start_index
```

从最新消息反向扫描，数到第 3 条用户消息后，保留从它开始的全部后续内容。

例如：

```text
U1 A1 U2 A2 U3 A3 U4 A4
```

设置 `recent_turns=3` 时保留：

```text
U2 A2 U3 A3 U4 A4
```

这样不会只留下助手答案、丢掉对应问题。

### 8.2 摘要调用不会每轮发生

压缩触发阈值是 `2400 Token`。因此：

```text
短会话：不调用摘要模型
第一次旧消息超过阈值：调用一次摘要模型
后续少量新消息：复用已有摘要
新旧消息再次积累到阈值：再次更新摘要
```

这避免“每轮回答多一次摘要 LLM 调用”的成本和延迟。

## 9. 摘要模型与摘要格式

摘要系统提示词定义在 `conversation_memory_service.py`：

```python
_SUMMARY_SYSTEM_PROMPT = """你负责压缩一段会话记忆，供后续 RAG 问答使用。
只保留用户明确陈述的稳定事实、对象名称、环境、约束、已完成事项、待确认项和偏好。
不要把模型猜测、SOP 内容或历史数值写成实时事实；发生冲突时以较新的用户消息为准。
用中文输出，严格使用以下紧凑标题；缺失项写“无”：
【明确事实】
【任务进展】
【约束与偏好】
【待确认项】"""
```

摘要示例：

```text
【明确事实】
- 服务为 order-api，环境为 prod。
【任务进展】
- 已讨论 CPU 高时的进程和日志排查。
【约束与偏好】
- 用户要求中文、分步骤解释。
【待确认项】
- 尚未获得实时 CPU 指标。
```

### 9.1 为什么要固定结构

固定结构比自由散文更适合记忆：

```text
容易覆盖旧事实
容易发现冲突
便于人工审计
避免摘要掺入无关推测
```

### 9.2 如何调用摘要模型

```python
response = await summarizer.ainvoke(
    [SystemMessage(content=_SUMMARY_SYSTEM_PROMPT), HumanMessage(content=prompt)]
)
raw_summary = getattr(response, "content", None) or str(response)
return self._truncate_text(raw_summary.strip(), self._summary_token_budget)
```

`prompt` 会包含旧摘要与新消息：

```text
已有滚动摘要：...
需要合并的新消息：...
请合并为新的滚动摘要。
```

所以它是增量式摘要：旧摘要 + 新增旧消息 → 新摘要。

默认摘要模型通过项目已有工厂创建：

```python
return llm_factory.create_chat_model(
    model=config.rag_model,
    temperature=0,
    streaming=False,
)
```

- `temperature=0`：减少摘要漂移。
- `streaming=False`：摘要只需一次完整返回。
- 构造函数也允许注入 `summarizer`，测试中用 Fake LLM，不连接真实模型。

### 9.3 摘要失败如何降级

```python
except Exception as exc:
    logger.warning(f"会话记忆摘要失败，降级为最近消息窗口: {exc}")
    return ""
```

摘要失败的链路：

```text
摘要模型异常
→ 不写入坏快照
→ 继续使用最近消息
→ 主聊天与 RAG 继续执行
```

摘要是 Token 优化项，不是主聊天的单点故障。

## 10. 最终记忆文本如何严格控制预算

渲染逻辑：

```python
summary_header = "【会话滚动摘要】\n"
recent_header = "【最近会话原文】\n"

summary_budget = min(
    self._summary_token_budget,
    max(0, self._context_token_budget - estimate_tokens(summary_header)),
)
summary = self._truncate_text(summary, summary_budget)
```

先为标题留出 Token，避免只计算正文、不计算真正发送给模型的标签。

随后为最近消息计算余量：

```python
remaining_budget = max(
    0,
    self._context_token_budget - summary_cost - estimate_tokens(recent_header),
)
recent_messages = self._take_tail_to_budget(pending_messages, remaining_budget)
```

最后再做一次硬性兜底：

```python
text = "\n\n".join(sections)
text = self._truncate_text(text, self._context_token_budget)
```

因此如下内容都算在 1800 Token 内：

```text
摘要正文
最近消息正文
“用户：”“助手：”角色前缀
章节标题
换行和分隔符
```

### 10.1 超长单条消息

如果某条最新消息本身超过剩余预算：

```python
clipped = ConversationMessage(
    id=message.id,
    session_id=message.session_id,
    role=message.role,
    content=self._truncate_text(message.content, max(1, token_budget - 4)),
)
```

这确保一次粘贴巨大的日志也不会击穿记忆预算。完整日志仍保留在原始消息表，只是本轮模型上下文使用裁剪版本。

## 11. Repository 层逐段说明

文件：`app/repositories/conversation_repository.py`。

### 11.1 原始消息读取

```python
def list_active_messages(self, session_id: str) -> list[ConversationMessage]:
    stmt = (
        select(ConversationMessage)
        .where(ConversationMessage.session_id == session_id)
        .where(ConversationMessage.deleted_at.is_(None))
        .order_by(ConversationMessage.created_at.asc(), ConversationMessage.id.asc())
    )
    return list(self.db.scalars(stmt).all())
```

这里返回 ORM 对象而非前端字典，因为摘要服务需要：

```text
message.id：保存摘要游标
message.role：统计用户轮次
message.content：计算 Token、构造摘要
```

`list_session_history()` 改为复用 `list_active_messages()` 再转换为前端格式，减少重复查询逻辑。

### 11.2 摘要快照读写

```python
def get_memory_snapshot(self, session_id: str):
    return self.db.get(ConversationMemorySnapshot, session_id)
```

`session_id` 是快照主键，主键查询即可得到当前会话摘要。

```python
def save_memory_snapshot(...):
    self.get_or_create_session(session_id)
    snapshot = self.get_memory_snapshot(session_id)
    if snapshot is None:
        snapshot = ConversationMemorySnapshot(...)
        self.db.add(snapshot)
    else:
        snapshot.summary = summary
        snapshot.summarized_through_message_id = summarized_through_message_id
        snapshot.version += 1
    self.db.commit()
```

语义：

```text
无快照：INSERT
有快照：UPDATE 摘要、游标，version + 1
```

### 11.3 清空会话

原有消息清空使用软删除；本次同步删除快照：

```python
snapshot = self.get_memory_snapshot(session_id)
if snapshot is not None:
    self.db.delete(snapshot)
```

因此用户清空对话后，旧摘要不会在下一次请求中被再次注入。

## 12. API 层如何接入

文件：`app/api/chat_v2.py`。

新增降级函数：

```python
async def _load_conversation_memory(db: Session, session_id: str) -> str:
    try:
        return (await conversation_memory_service.build_context(db, session_id)).text
    except Exception as exc:
        logger.warning(f"[session {session_id}] 会话记忆加载失败，降级为空记忆: {exc}")
        return ""
```

如果数据库、摘要服务或任何记忆逻辑发生异常，返回空字符串。RAG 仍以原来的无记忆模式执行。

非流式接口：

```python
memory_context = await _load_conversation_memory(db, request.id)
result = await rag_v2_service.query(
    request.question,
    session_id=request.id,
    conversation_context=memory_context,
)
```

流式接口同样在流开始前加载一次：

```python
memory_context = await _load_conversation_memory(db, request.id)
async for chunk in rag_v2_service.query_stream(
    request.question,
    session_id=request.id,
    conversation_context=memory_context,
):
```

本轮问答完成后，仍沿用已有 `append_exchange()` 写入原始消息。因此下一轮请求才会看到本轮问答；这避免本轮问题在回答前被误写入历史。

## 13. RAG V2 的状态与服务透传

### 13.1 图状态

文件：`app/agent/rag_v2/state.py`。

```python
class RAGState(TypedDict, total=False):
    question: str
    conversation_context: str
    sub_queries: List[str]
    documents: Annotated[List[Document], operator.add]
    deduped_documents: List[Document]
    answer: str
    validation: Dict[str, Any]
```

新增的 `conversation_context` 是一个普通字符串状态：

```text
由 API 从 PostgreSQL 构造
在 LangGraph 节点之间共享
不写入 Milvus
不混进检索文档列表
```

### 13.2 服务层

文件：`app/agent/rag_v2/service.py`。

非流式调用变为：

```python
async def query(
    self,
    question: str,
    session_id: str = "",
    conversation_context: str = "",
) -> Dict[str, Any]:
    result = await self.graph.ainvoke(
        {
            "question": question,
            "conversation_context": conversation_context,
        }
    )
```

流式调用同样传入：

```python
async for update in self.graph.astream(
    {
        "question": question,
        "conversation_context": conversation_context,
    },
    stream_mode="updates",
):
```

参数默认 `""`，因此已有直接调用不会中断：

```python
await rag_v2_service.query("查询 Redis 集群状态")
```

无会话记忆时仍正常工作。

## 14. 会话记忆的安全边界

文件：`app/agent/rag_v2/nodes.py`。

新增全局策略：

```python
MEMORY_POLICY = '''会话记忆只用于理解用户已明确说明的对象、任务延续和约束。
它不是知识库证据，更不是实时监控、日志或工具查询结果；不得将历史指标写成当前实时值。'''
```

这是会话记忆最重要的语义隔离。

示例：

```text
历史聊天：昨天 order-api CPU 是 90%
当前问题：现在 CPU 是多少？
```

会话记忆可以帮助识别对象是 `order-api`，但不能让模型回答“现在也是 90%”。要回答实时数值，仍需要：

```text
Prometheus 查询
监控 MCP 工具
日志系统
当前告警 payload
```

## 15. 改写节点：解决“它”“这个服务”等指代

`rewrite_node()`：

```python
question = state["question"]
conversation_context = (state.get("conversation_context") or "").strip()
```

有记忆时，构造显式区块：

```python
memory_block = (
    f"\n\n<conversation_memory>\n{conversation_context}\n</conversation_memory>"
    if conversation_context
    else ""
)
```

并告诉改写模型：

```text
请利用记忆消解代词和省略的对象，但不要把历史信息改写为实时事实。
```

模型输入会类似：

```text
会话记忆只用于理解用户已明确说明的对象、任务延续和约束。

<conversation_memory>
【明确事实】
- 服务是 order-api
</conversation_memory>

当前问题：它为什么 CPU 高？
```

从而生成更准确的检索词：

```text
order-api CPU 使用率高的常见原因
order-api CPU 高时的日志排查步骤
```

而不是去检索模糊的“它为什么 CPU 高”。

## 16. 生成节点：记忆与知识库证据分区

`generate_node()` 分别组织：

```text
<conversation_memory>
会话摘要和最近原文
</conversation_memory>

<context>
RAG 检索出的 SOP / 文档
</context>
```

关键实现：

```python
user_prompt = (
    f"{MEMORY_POLICY}\n\n{memory_block}"
    f"<context>\n{context}\n</context>\n\n问题: {question}"
)
```

职责划分：

| 区块 | 作用 |
|---|---|
| `conversation_memory` | 理解服务名、用户要求、任务延续、指代 |
| `context` | 提供可被 RAG 回答使用的 SOP 和知识文档依据 |
| 当前问题 | 本轮真正要回答的请求 |

这样模型既能知道“它”是哪个服务，也不会把历史中的猜测自动提升为事实。

## 17. 校验节点：历史不算可验证证据

`validate_answer_node()` 同样读取：

```python
conversation_context = (state.get("conversation_context") or "").strip()
```

但校验 Prompt 特别补充：

```python
"仅将 <evidence> 中的内容作为知识库可验证证据；会话记忆只能帮助判断对象指代。"
```

因此 High Precision 面板中的这些结果：

```text
coverage_score
groundedness_score
unsupported_claims
missing_aspects
```

仍只根据 `<evidence>` 中的 RAG 文档判断，不会因为历史聊天里曾出现一个 CPU 数字就让当前回答错误通过校验。

## 18. RAG 文档也有 Token 预算

改造前 `_format_context(docs)` 会把所有最终文档原文拼接。现在增加 `RAG_DOCUMENT_CONTEXT_TOKEN_BUDGET`：

```python
def _format_context(docs: List[Document]) -> str:
    parts: List[str] = []
    used_tokens = 0
    budget = config.rag_document_context_token_budget
    for idx, doc in enumerate(docs, start=1):
        prefix = f"[{idx}] (源子查询: {src})\n"
        available = budget - used_tokens - _estimate_tokens(prefix)
        if available <= 0:
            break
        content = _truncate_to_token_budget(doc.page_content, available)
        parts.append(f"{prefix}{content}")
```

执行效果：

```text
会话记忆：最多 1800 Token
检索文档：最多 2800 Token
```

两类上下文都受限，避免一侧无限增长挤占另一侧，也避免把全部上下文窗口耗尽。

## 19. 清空会话后的效果

调用原有清空接口后：

```text
conversation_messages：标记 deleted_at，软删除
conversation_memory_snapshots：物理删除当前摘要快照
```

因此：

```text
用户清空会话
→ 前端不再显示旧记录
→ 模型不再获得旧摘要
→ 下一轮从空记忆开始
```

## 20. 测试覆盖

新增文件：`tests/unit/test_conversation_memory_service.py`。

覆盖点：

1. 短对话低于阈值时，仅使用最近消息，不调用摘要模型。
2. 超过阈值时，生成摘要并保存快照游标。
3. 已有摘要、少量新消息时，复用摘要而不重复调用模型。
4. 清空会话会同步删除摘要快照。
5. 中文 Token 估算采用保守规则。
6. 最终文本连同标题和角色前缀也严格不超过预算。

`tests/unit/test_rag_v2_nodes.py` 新增：

1. 改写节点收到历史里的 `order-api`。
2. 生成节点明确禁止历史指标当作实时指标。
3. 校验节点明确只有 `<evidence>` 是可验证知识证据。
4. RAG 文档上下文会按 Token 预算裁剪。

`tests/integration/test_chat_v2_api.py` 新增：

```text
API 构建会话记忆
→ 把 conversation_context 传给 rag_v2_service
→ 保持原有对话保存和 trace 保存行为
```

针对性回归结果：

```text
18 passed
```

迁移与模型验证：

```text
20260823_0007 (head)
memory_schema=ok
```

## 21. 故障降级链路

### 摘要模型异常

```text
摘要调用异常
→ 日志 warning
→ 不写入错误摘要
→ 使用最近消息窗口
→ RAG 主流程继续
```

### 会话记忆读取异常

`_load_conversation_memory()` 会捕获异常：

```text
数据库或记忆服务异常
→ 返回空字符串
→ RAG 用无记忆模式继续工作
```

### 检索文档超长

```text
超过文档预算
→ 截断当前文档尾部或停止追加下一文档
→ 保留已选文档的来源子查询
→ 不突破 2800 Token
```

## 22. 运行和手工验证

### 22.1 执行迁移

```bash
cd /home/dong/projects/super_biz_agent_py
.venv/bin/alembic upgrade head
```

迁移会新增：

```text
conversation_memory_snapshots
```

不会删除或覆盖已有对话记录。

### 22.2 启动服务

```bash
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 9900
```

### 22.3 验证对话连续性

同一个会话顺序发送：

```text
服务名称是 order-api，环境是 prod。
它 CPU 高时优先看什么？
它现在 CPU 是多少？
```

预期：

1. 第二句“它”能够被理解为 `order-api`，检索改写会携带服务名。
2. 第三句不能使用旧历史数值当作“现在”的 CPU，应要求监控查询或当前告警数据。
3. 长会话达到阈值后，`conversation_memory_snapshots` 中会创建或更新快照；完整消息仍保留在 `conversation_messages`。

### 22.4 查询快照

```sql
SELECT
  session_id,
  version,
  summarized_through_message_id,
  summary,
  updated_at
FROM conversation_memory_snapshots
ORDER BY updated_at DESC;
```

这能用于确认：

```text
哪些会话已经被压缩
摘要内容是什么
摘要覆盖到哪条消息
摘要更新过几次
```

## 23. 完整请求时序

```text
浏览器发送当前问题（Id / Question）
        ↓
chat_v2.py 读取 session_id 对应的原始历史
        ↓
ConversationMemoryService 读取摘要快照和未压缩消息
        ↓
必要时调用摘要模型，更新 snapshot 游标
        ↓
构造：摘要 + 最近 3 轮原文（<= 1800 Token）
        ↓
RAG V2 rewrite：结合记忆还原对象和指代
        ↓
VectorSearch：检索 SOP / 文档
        ↓
generate：会话记忆与知识文档分区生成答案
        ↓
validate：只将 RAG evidence 视为可验证事实
        ↓
保存当前 user/assistant 原始消息与 RAG trace
        ↓
返回前端
```

## 24. 最终职责边界

| 内容 | 存储 | 是否注入模型 | 作用 |
|---|---|---|---|
| 完整聊天原文 | `conversation_messages` | 不全量注入 | 审计、UI、摘要来源 |
| 滚动摘要 | `conversation_memory_snapshots` | 注入 | 会话长期事实、任务与偏好 |
| 最近 3 轮 | `conversation_messages` | 注入 | 当前细节和连续任务 |
| SOP / 文档 | Milvus | 按检索注入 | 可验证知识依据 |
| 当前 CPU、日志 | 监控 / 日志 / 告警 | 需实时获取 | 实时事实 |

一句话概括：

```text
会话记忆负责“记住对象、任务和约束”；
知识库负责“给出 SOP 和文档证据”；
监控与日志负责“提供当前实时事实”。
```

这样既拥有跨服务重启的会话连续性，又将每轮输入保持在固定 Token 预算内，同时避免历史信息被当成当前生产事实。
