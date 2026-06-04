"""LLM 工厂单元测试

验证 create_chat_model 的参数优先级：显式参数 > config 默认值。
不发起真实网络请求。
"""

from unittest.mock import patch

from app.core.llm_factory import LLMFactory


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
        assert model.openai_api_key == "explicit-key"

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
            model = LLMFactory.create_chat_model(model=None, api_key="test-key")
            assert model.model_name == "qwen-plus"
