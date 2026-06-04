"""配置模块单元测试"""

import os
from unittest.mock import patch

from app.config import Settings


class TestSettingsDefaults:
    def test_default_app_name(self) -> None:
        # 所有默认值测试都显式关闭 .env 读取，避免开发机环境把测试“偷偷改绿”。
        s = Settings(_env_file=None, dashscope_api_key="test")
        assert s.app_name == "SuperBizAgent"

    def test_default_port(self) -> None:
        s = Settings(_env_file=None, dashscope_api_key="test")
        assert s.port == 9900

    def test_default_debug_false(self) -> None:
        s = Settings(_env_file=None, dashscope_api_key="test")
        assert s.debug is False

    def test_default_cors_origins(self) -> None:
        s = Settings(_env_file=None, dashscope_api_key="test")
        assert "http://localhost:9900" in s.cors_origins
        assert "http://127.0.0.1:9900" in s.cors_origins

    def test_default_rag_top_k(self) -> None:
        s = Settings(_env_file=None, dashscope_api_key="test")
        assert s.rag_top_k == 3

    def test_default_chunk_size(self) -> None:
        s = Settings(_env_file=None, dashscope_api_key="test")
        assert s.chunk_max_size == 800
        assert s.chunk_overlap == 100


class TestSettingsEnvOverride:
    def test_env_overrides_app_name(self) -> None:
        # patch.dict 只在当前用例里生效，适合验证环境变量优先级。
        with patch.dict(os.environ, {"APP_NAME": "CustomAgent"}):
            s = Settings(_env_file=None, dashscope_api_key="test")
            assert s.app_name == "CustomAgent"

    def test_env_overrides_port(self) -> None:
        with patch.dict(os.environ, {"PORT": "8080"}):
            s = Settings(_env_file=None, dashscope_api_key="test")
            assert s.port == 8080

    def test_env_overrides_debug(self) -> None:
        with patch.dict(os.environ, {"DEBUG": "true"}):
            s = Settings(_env_file=None, dashscope_api_key="test")
            assert s.debug is True


class TestCorsOrigins:
    def test_cors_from_env_comma_separated(self) -> None:
        with patch.dict(os.environ, {"CORS_ORIGINS": "https://a.com, https://b.com"}):
            s = Settings(_env_file=None, dashscope_api_key="test")
            assert s.cors_origins == ["https://a.com", "https://b.com"]

    def test_cors_from_env_whitespace_tolerant(self) -> None:
        with patch.dict(os.environ, {"CORS_ORIGINS": " https://a.com ,  https://b.com "}):
            s = Settings(_env_file=None, dashscope_api_key="test")
            assert s.cors_origins == ["https://a.com", "https://b.com"]
