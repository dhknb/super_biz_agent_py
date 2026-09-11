"""LLM 工厂类 —— 项目里**唯一**的 LLM 构造入口。

使用 LangChain ChatOpenAI 通过 OpenAI 兼容模式调用阿里云 DashScope，
这种方式便于后续切换到其他支持 OpenAI API 的模型提供商：

- 阿里云 DashScope: https://dashscope.aliyuncs.com/compatible-mode/v1
- OpenAI: https://api.openai.com/v1
- Azure OpenAI: https://{resource}.openai.azure.com
- 其他兼容 OpenAI API 的服务

## 为什么必须收敛到一处（DRY）

改造前项目里有三个各自 new 模型的地方：
    app/core/llm_factory.py:40        ChatOpenAI(...)
    app/services/first_response_service.py:98   ChatQwen(...)
    app/services/rag_agent_service.py:91        ChatQwen(...)

三处都**没有设置任何超时**。这不是巧合，而是分散构造的必然结果：
超时这类横切参数一旦分散，就没有任何一个地方能保证「全都设上了」，
加参数时永远会漏掉一个，而漏掉的那个恰好就是线上挂住的那个。

## 为什么没有超时是严重问题

chat_v2 的一次请求会**串行调用 LLM 三次**（rewrite → generate → validate）。
不传 timeout 时，这条链路是**完全没有超时**的 —— 不是「用了某个默认值」。

实测过程见 docs/learning/16。结论是 langchain-openai 在
`chat_models/base.py` 里把 `request_timeout`（默认 `None`）**无条件**
塞进 openai SDK 的 client 参数，于是 SDK 自带的
`DEFAULT_TIMEOUT = Timeout(connect=5.0, read=600, write=600, pool=600)`
被覆盖成 `None`，底层 httpx 拿到的是 `Timeout(timeout=None)`。
对比同一段代码里的 `max_retries` 用的是 `if ... is not None` 的条件写法，
所以 SDK 的默认重试 2 次能生效 —— 两个参数待遇不同，超时是被覆盖掉的那个。

后果：一旦上游 TCP 连接建立后不再返回数据（限流排队、网关半开连接），
调用会**无限期**挂住 —— 请求挂住 → SSE 挂住 → 连接池被逐渐占满 →
后续所有请求排队。整个服务被一个卡住的上游拖死，
而全项目当时只有三处超时：milvus_timeout、RQ job_timeout、MCP retry_interceptor，
LLM 这条最慢、最不可控的链路反而完全没有防护。

## 单次超时 ≠ 整体预算

`timeout` 管的是单次调用；三次串行调用最坏情况是 3×timeout，
再加上检索耗时，用户侧感知远超预期。所以除了这里的单次超时，
还需要在 graph 调用外层套一个总预算（见 config.chat_total_budget_seconds
与 app/agent/rag_v2/service.py 的 asyncio.timeout 包裹）。
两者是不同层级的防护，缺一不可。
"""

from typing import Any

from langchain_openai import ChatOpenAI

from app.config import config


class LLMFactory:
    """LLM 工厂类 - 使用 OpenAI 兼容模式"""

    # 阿里云 DashScope OpenAI 兼容模式 URL
    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @staticmethod
    def create_chat_model(
        model: str | None = None,
        temperature: float = 0.7,
        streaming: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> ChatOpenAI:
        """构造 ChatOpenAI 实例。

        Args:
            timeout: 单次调用超时（秒）。传 None 走 config.llm_timeout_seconds。
                注意不能用 `timeout or config.x` 这种写法 —— 那样 `timeout=0`
                会被静默替换成默认值。这里用 `is None` 判断，保留显式传 0 的语义。
            max_retries: 失败重试次数。同上，用 `is None` 判断。
        """
        model = model or config.dashscope_model
        base_url = base_url or LLMFactory.DASHSCOPE_BASE_URL
        api_key = api_key or config.dashscope_api_key
        timeout = config.llm_timeout_seconds if timeout is None else timeout
        max_retries = config.llm_max_retries if max_retries is None else max_retries

        # 参考：https://help.aliyun.com/zh/model-studio/getting-started/models
        extra_body = {}
        extra_body["stream"] = streaming

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            extra_body=extra_body if extra_body else None,
        )

        return llm

    @staticmethod
    def create_qwen_model(
        model: str | None = None,
        temperature: float = 0.7,
        streaming: bool = False,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """构造 langchain_qwq 的 ChatQwen 实例。

        为什么不把 ChatQwen 的调用点直接改成 create_chat_model：
        ChatQwen 不只是「另一个 OpenAI 兼容客户端」，它对 Qwen 的
        推理内容（reasoning_content）等字段有专门处理，rag_agent_service
        依赖这些行为。强行换成 ChatOpenAI 会改变运行时语义 ——
        本次目标是补超时，不是换模型客户端（YAGNI，也避免夹带风险）。

        但**超时参数的默认值必须与 ChatOpenAI 路径完全一致**，
        所以两条路径共用同一套 config 默认值，这是收敛的实际意义所在。

        延迟 import langchain_qwq：它是较重的可选依赖，
        放在模块顶层会让所有 import app.core.llm_factory 的地方都被迫加载它。
        """
        from langchain_qwq import ChatQwen

        model = model or config.rag_model
        api_key = api_key or config.dashscope_api_key
        timeout = config.llm_timeout_seconds if timeout is None else timeout
        max_retries = config.llm_max_retries if max_retries is None else max_retries

        return ChatQwen(
            model=model,
            api_key=api_key,
            temperature=temperature,
            streaming=streaming,
            # ChatQwen 继承自 BaseChatOpenAI，其字段名为 request_timeout，
            # 但带有 alias "timeout"（已通过 model_fields 校验）。
            # 这里用 timeout=，与 ChatOpenAI 路径写法保持一致。
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )


# 全局 LLM 工厂实例
llm_factory = LLMFactory()
