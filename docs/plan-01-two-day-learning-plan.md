# 计划 1：两天学习路线

目标：用两天理解并能独立复述本项目的知识库工程化改造，包括 PostgreSQL 元数据、Redis/RQ 异步索引、Milvus 向量存储、FastAPI 接口边界。

## Day 1：把链路跑通并理解数据建模

### 上午：整体架构和依赖

学习目标：

- 理解为什么上传接口不应该同步做 embedding。
- 理解 PostgreSQL、Redis/RQ、Milvus 各自职责。
- 能画出“上传 -> 建库记录 -> 入队 -> worker 索引 -> Milvus”的流程。

阅读文件：

- `app/api/file.py`
- `app/core/database.py`
- `app/core/task_queue.py`
- `app/models/knowledge_base.py`
- `app/repositories/knowledge_repository.py`

动手任务：

1. 启动基础设施：

   ```bash
   docker compose -f vector-database.yml up -d
   ```

2. 安装依赖并迁移数据库：

   ```bash
   pip install -e .
   alembic upgrade head
   ```

3. 在数据库里确认有三张表：

   - `knowledge_documents`
   - `knowledge_chunks`
   - `index_jobs`

### 下午：接口和 Repository

学习目标：

- 理解 `KnowledgeDocument`、`KnowledgeChunk`、`IndexJob` 三个表的关系。
- 理解 Repository 为什么要隔离数据库操作。
- 理解上传接口为什么返回 `202 accepted`。

动手任务：

1. 启动服务：

   ```bash
   make start-cls
   make start-monitor
   make start-worker
   make start-api
   ```

2. 上传一个 Markdown 文件：

   ```bash
   curl -X POST http://localhost:9900/api/upload \
     -F "file=@aiops-docs/cpu_high_usage.md"
   ```

3. 查询文档列表：

   ```bash
   curl http://localhost:9900/api/documents
   ```

4. 观察数据库里 document 和 job 状态变化：

   - 刚上传：`pending` / `queued`
   - worker 执行中：`indexing` / `running`
   - 成功后：`indexed` / `succeeded`

复盘问题：

- 为什么 `index_jobs` 不能只依赖 RQ 自己的 job 状态？
- 为什么 `content_hash` 对幂等索引有价值？
- 为什么 Milvus metadata 里要放 `document_id` 和 `chunk_id`？

## Day 2：理解 worker、失败恢复和生产化边界

### 上午：RQ Worker 和索引任务

学习目标：

- 理解 RQ worker 是独立进程。
- 理解后台任务里为什么要重新创建数据库 Session。
- 理解索引任务的状态流转和异常处理。

阅读文件：

- `app/services/index_job_queue.py`
- `app/workers/index_worker.py`
- `app/services/document_splitter_service.py`
- `app/services/vector_store_manager.py`

动手任务：

1. 停掉 worker 后上传文件，观察 job 停留在 `queued`。
2. 启动 worker，观察 job 被消费。
3. 故意配置错误的 DashScope key，观察 job 变成 `failed` 并保存错误。
4. 修复配置后调用重建索引：

   ```bash
   curl -X POST http://localhost:9900/api/documents/<document_id>/reindex
   ```

复盘问题：

- 为什么 worker 不能复用 FastAPI 请求里的 `db`？
- 如果 worker 执行到“写入 PostgreSQL chunks 成功，但写入 Milvus 失败”，系统会留下什么状态？
- 为什么重试前要按 `document_id` 删除旧向量？

### 下午：运行、排障和下一步扩展

学习目标：

- 能独立启动完整链路。
- 能判断问题出在 PostgreSQL、Redis、RQ、Milvus、embedding 还是 API。
- 能说出下一阶段要做哪些工程增强。

常用命令：

```bash
make status
make status-mcp
make status-worker
tail -f server.log
tail -f rq_worker.log
docker ps
```

排障顺序：

1. 上传接口是否返回 `202`。
2. PostgreSQL 是否有 `knowledge_documents` 和 `index_jobs` 记录。
3. Redis/RQ worker 是否运行。
4. `rq_worker.log` 是否有异常。
5. Milvus 是否连接成功。
6. DashScope embedding 是否可用。

扩展方向：

- 用 PostgreSQL checkpointer 持久化聊天上下文。
- 给 `documents` 和 `index_jobs` 增加分页和详情接口。
- 给索引任务增加最大重试次数和死信队列。
- 给知识库增加租户字段和权限过滤。
- 用对象存储替代本地 `uploads/`。

最终验收：

- 能讲清楚四个组件职责：FastAPI、PostgreSQL、Redis/RQ、Milvus。
- 能手动上传文件并看到异步索引完成。
- 能解释失败任务如何排查和重试。
- 能解释为什么“前端显示了历史”不等于“模型拿到了上下文”。
