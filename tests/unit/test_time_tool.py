"""时间工具单元测试"""

from app.tools.time_tool import get_current_time


class TestGetCurrentTime:
    def test_returns_non_empty_string(self) -> None:
        result = get_current_time.invoke({"timezone": "Asia/Shanghai"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_default_timezone_is_shanghai(self) -> None:
        result = get_current_time.invoke({})
        assert isinstance(result, str)
        # should match YYYY-MM-DD HH:MM:SS format
        import re

        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result)

    def test_invalid_timezone_returns_error_message(self) -> None:
        result = get_current_time.invoke({"timezone": "Mars/Nowhere"})
        assert "获取时间失败" in result
