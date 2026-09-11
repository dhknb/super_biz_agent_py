"""LLM 工厂单元测试

验证两件事：
1. 参数优先级：显式参数 > config 默认值。
2. **超时与重试一定被注入** —— 这是本次改造的核心目标，
   必须有测试守住，否则以后有人加个新参数把 timeout 漏掉，没人会发现。

不发起真实网络请求：ChatOpenAI 构造时不连网，只做参数校验。
"""

from unittest.mock import patch

import pytest

from app.core.llm_factory import LLMFactory


def _secret(value: object) -> str | None:
    """取出 SecretStr 的明文。

    langchain-openai 会把 api_key 包成 pydantic 的 SecretStr，
    直接和字符串比较永远不相等（repr 是 '**********'）。
    这个小助手让断言意图保持清晰，而不是在每个用例里重复 .get_secret_value()。
    """
    if value is None:
        return None
    getter = getattr(value, "get_secret_value", None)
    return getter() if callable(getter) else str(value)


class TestCreateChatModel:
    def test_returns_chat_openai(self) -> None:
        model = LLMFactory.create_chat_model(
            model="test-model",
            api_key="test-key",
        )
        assert model.model_name == "test-model"

    def test_explicit_api_key_takes_priority(self) -> None:
        model = LLMFactory.create_chat_model(
            model="test-model",
            api_key="explicit-key",
        )
        assert _secret(model.openai_api_key) == "explicit-key"

    def test_explicit_base_url_takes_priority(self) -> None:
        model = LLMFactory.create_chat_model(
            model="test-model",
            api_key="test-key",
            base_url="https://custom-api.example.com/v1",
        )
        assert model.openai_api_base == "https://custom-api.example.com/v1"

    def test_temperature_passed_through(self) -> None:
        model = LLMFactory.create_chat_model(
            model="test-model",
            api_key="test-key",
            temperature=0.2,
        )
        assert model.temperature == 0.2

    def test_streaming_passed_through(self) -> None:
        model = LLMFactory.create_chat_model(
            model="test-model",
            api_key="test-key",
            streaming=False,
        )
        assert model.streaming is False

    def test_default_base_url_is_dashscope(self) -> None:
        model = LLMFactory.create_chat_model(
            model="qwen-max",
            api_key="test-key",
        )
        assert "dashscope" in model.openai_api_base  # type: ignore[operator]

    def test_model_falls_back_to_config_when_none(self) -> None:
        with patch("app.core.llm_factory.config") as mock_config:
            mock_config.dashscope_model = "qwen-plus"
            # 必须一并给出超时相关默认值：factory 现在会读它们，
            # MagicMock 属性会被 pydantic 拒绝（不是合法的 float/int）。
            mock_config.llm_timeout_seconds = 60.0
            mock_config.llm_max_retries = 2
            model = LLMFactory.create_chat_model(model=None, api_key="test-key")
            assert model.model_name == "qwen-plus"


class TestTimeoutAndRetries:
    """超时与重试的注入行为 —— 本次改造的核心不变量。"""

    def test_timeout_defaults_to_config(self) -> None:
        with patch("app.core.llm_factory.config") as mock_config:
            mock_config.dashscope_model = "qwen-max"
            mock_config.llm_timeout_seconds = 45.0
            mock_config.llm_max_retries = 3
            model = LLMFactory.create_chat_model(api_key="test-key")
            assert model.request_timeout == 45.0
            assert model.max_retries == 3

    def test_explicit_timeout_takes_priority(self) -> None:
        model = LLMFactory.create_chat_model(
            model="test-model",
            api_key="test-key",
            timeout=12.5,
            max_retries=0,
        )
        assert model.request_timeout == 12.5
        assert model.max_retries == 0

    @pytest.mark.parametrize("explicit", [0, 0.0])
    def test_zero_timeout_is_not_replaced_by_default(self, explicit: float) -> None:
        """显式传 0 不能被静默改成默认值。

        这条守的是 `timeout or config.x` 这个经典 bug：
        0 是 falsy，用 `or` 会让「显式要求不等待」变成「等 60 秒」。
        factory 里用的是 `is None` 判断，这个用例就是它的护栏。
        """
        model = LLMFactory.create_chat_model(
            model="test-model",
            api_key="test-key",
            timeout=explicit,
        )
        assert model.request_timeout == 0

    def test_timeout_is_always_set(self) -> None:
        """不传任何超时参数时，也必须有非 None 的超时。

        改造前这里是 None —— 上游卡住就无限期挂住，
        一次 chat_v2 请求串行调三次 LLM，任何一次挂住整条链路就死。
        """
        model = LLMFactory.create_chat_model(model="test-model", api_key="test-key")
        assert model.request_timeout is not None
        assert model.request_timeout > 0
