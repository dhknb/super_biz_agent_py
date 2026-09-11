"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Annotated, Any, Dict, List

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "SuperBizAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900
    upload_dir: str = "./uploads"

    # CORS 配置：逗号分隔的来源列表；生产环境必须显式设置
    cors_origins: Annotated[List[str], NoDecode] = [
        "http://localhost:9900",
        "http://127.0.0.1:9900",
    ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # PostgreSQL / Redis / RQ 配置
    database_url: str = (
        "postgresql+psycopg://postgres:postgres@localhost:55432/super_biz_agent"
    )
    redis_url: str = "redis://localhost:6379/0"
    rq_queue_name: str = "knowledge_index"

    # LLM 超时与重试配置
    #
    # 为什么必须有超时：langchain 的 ChatOpenAI / ChatQwen 默认 timeout=None，
    # 也就是「无限等待」。一次 chat_v2 请求串行调用 LLM 三次
    # （rewrite → generate → validate），任何一次卡住整个请求就挂住，
    # SSE 连接不释放，连接池被慢慢吃干 —— 表现为「服务没挂但全站变慢」。
    #
    # 60s 的取值依据：qwen-max 长回答的 p99 在 30s 量级，留一倍余量。
    # 调小会误杀正常的长回答，调大则失去保护意义。
    llm_timeout_seconds: float = 60.0

    # 单次调用的重试次数（由 langchain/openai SDK 内部实现，带指数退避）。
    # 2 次而非默认的 2 以上：重试会成倍放大总耗时，
    # 真正的瞬时抖动一次重试基本就能恢复，多了只是延后失败暴露。
    llm_max_retries: int = 2

    # 整个 chat 请求的总预算。
    # 单次超时 × 3 次串行调用 = 180s 的最坏情况，对交互式接口太久。
    # 总预算是兜底闸门：超了就返回部分结果 + 降级说明，
    # 而不是让用户对着转圈等三分钟。
    chat_total_budget_seconds: float = 90.0

    # 熔断器配置
    #
    # 超时 + 重试 + 降级这套组合拳能处理偶发故障，但在**持续**故障下
    # 每个请求都要先白等 llm_timeout_seconds 才知道下游还是不行。
    # 熔断器负责把「重复确认一个已知的坏消息」的成本降到零。
    # 详见 app/core/circuit_breaker.py 的模块文档。
    #
    # 阈值 5：低于 3 会被偶发抖动误触发（内部服务 QPS 不高，
    # 三次连续失败很可能只是一次网络抽风）；高于 8 则失去意义 ——
    # 8 × 60s 超时 = 8 分钟才熔断，故障早就扩散完了。
    circuit_breaker_failure_threshold: int = 5

    # 冷却 30s：够短，下游恢复后半分钟内就能自动接回；
    # 够长，不会让探针把刚喘上气的下游又打回去。
    circuit_breaker_cooldown_seconds: float = 30.0

    # 运行时降级开关
    #
    # 为什么需要它们：熔断器是**被动**的 —— 它要先失败 N 次才动作。
    # 但有些时候我们**已经知道**某个环节是坏的（正在迁移 Milvus、
    # 质检模型的配额被别的业务吃光了），这时需要一个不用改代码、
    # 不用发版就能立刻关掉它的开关。
    #
    # 两个开关都默认 True：关掉是应急手段，不该是默认状态。
    # 关掉之后链路仍然如实声明降级原因，不会假装「知识库里没有」。
    enable_sop_retrieval: bool = True
    enable_answer_validation: bool = True

    # chat_v2 里的工具调用(实时事实查询)开关。
    #
    # 为什么它需要一个开关,而检索和质检共用不了:
    # 工具节点是 chat_v2 里唯一会**对外产生副作用**的环节 —— 它去查 CLS 日志、
    # 拉监控指标,打在别人的服务上。那些服务出问题时(被别的业务打满、
    # 正在维护),我们需要一个不用发版就能立刻停止打扰它们的开关,
    # 而熔断器只有在**我们这边**看到失败之后才动作 —— 对方还没坏到报错、
    # 只是不希望被打扰的时候,熔断器不会帮忙。
    #
    # 关掉之后 chat_v2 退化成纯知识库问答(也就是加工具之前的形态),
    # 并如实记 feature_disabled,不会假装「工具没查到」。
    enable_tool_facts: bool = True

    # 工具节点的时间预算(秒)。
    #
    # 为什么工具需要一个**独立于总预算**的时限:
    # tool_facts 与检索并行,但 generate 必须等两条都到齐才能开始。
    # 也就是说工具慢就是整条链路慢 —— 它卡在那儿的每一秒,
    # 都是从 generate 和 validate 的预算里偷的。
    #
    # 取 20 秒:留给「选工具的那次 LLM 调用 + 一到两次工具执行」。
    # 比 llm_timeout_seconds(60)短得多是刻意的:工具事实是**增强**,
    # 拿不到就少一份实时数据,而知识库那条链路照样能回答 ——
    # 让一个增强环节吃掉大半个总预算是不划算的交换。
    tool_facts_budget_seconds: float = 20.0

    # 可观测性
    #
    # /metrics 端点开关。默认开着 —— 关掉指标等于关掉发现故障的能力,
    # 那比暴露拓扑的风险大得多。
    #
    # 但这个端点是**无鉴权**的,它吐出的是内部形状:有哪些节点、
    # 各自的耗时分布、降级率、熔断器开没开、请求量。对侦察者来说很有用
    # (比如从 degrade_total 的跳变推断我们哪个依赖正在挂)。
    #
    # 为什么不在应用里加 Basic Auth / bearer token:
    # 那会让抓取端的配置复杂化(密码要发给 Prometheus 并跟着轮转),
    # 而收口这件事在网关/ACL 层做既标准又彻底 ——
    # 让端点只对内网的 Prometheus 可达。**生产环境必须在网关层限制来源。**
    # 这个开关是最后一道保险:真出事了不用发版就能立刻关掉。
    enable_metrics_endpoint: bool = True

    # RAG 配置
    rag_top_k: int = 3
    rag_model: str = "qwen-max"  # 使用快速响应模型，不带扩展思考
    rag_document_context_token_budget: int = 2800

    # 会话记忆：原始消息持久化，模型只注入滚动摘要和有限最近窗口。
    conversation_memory_context_token_budget: int = 1800
    conversation_memory_summary_token_budget: int = 700
    conversation_memory_compact_threshold_tokens: int = 2400
    conversation_memory_recent_turns: int = 3

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # MCP 服务配置
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    # 某个 MCP 服务的工具列举失败后,多久才允许再试一次。
    #
    # 为什么这件事需要一个**独立于熔断器**的冷却时间:
    # 拦截器只包工具「调用」,而 get_tools() 是工具「列举」,走的是
    # load_mcp_tools,压根不经过熔断器(见 app/agent/mcp_tool_provider.py)。
    # 所以列举必须自带退避,否则 MCP 进程没起来时,每个请求都要白等
    # 一次完整的连接超时。
    #
    # 为什么不复用 circuit_breaker_cooldown_seconds:熔断器要连续失败 5 次
    # 才动作,而一次列举失败(连接 + 握手 + list)已经是足够强的证据 ——
    # 等 5 次 × 连接超时太慢了。两者的证据强度和重试经济性都不同。
    #
    # 取 60 秒:比熔断冷却长一些。列举失败通常意味着进程没起来,
    # 那是运维动作(部署、重启)的时间尺度,不是网络抖动的尺度。
    mcp_tool_reload_cooldown_seconds: float = 60.0

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            }
        }


# 全局配置实例
config = Settings()
