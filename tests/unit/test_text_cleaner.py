"""TextCleaner 单元测试。"""

from app.services.text_cleaner import TextCleaner


def test_empty_returns_empty() -> None:
    cleaner = TextCleaner()
    assert cleaner.clean("") == ""
    assert cleaner.clean(None) == ""  # type: ignore[arg-type]


def test_bom_stripped() -> None:
    cleaner = TextCleaner()
    assert cleaner.clean("﻿# hello") == "# hello"


def test_zero_width_stripped() -> None:
    cleaner = TextCleaner()
    # 包含零宽连接符、零宽空格、字节顺序标记
    result = cleaner.clean("abc​‌‍﻿def")
    assert result == "abcdef"


def test_control_chars_stripped_but_keep_tab_newline() -> None:
    cleaner = TextCleaner()
    # \t 和 \n 应该被保留(\t 在 whitespace 步骤变成空格)
    result = cleaner.clean("a\x00b\x1fc\td\ne")
    # 控制字符被删,tab→空格,\n保留
    assert "\x00" not in result
    assert "\x1f" not in result
    assert "\n" in result


def test_normalize_newlines() -> None:
    cleaner = TextCleaner()
    assert cleaner.clean("a\r\nb\rc\nd") == "a\nb\nc\nd"


def test_strip_html_tags() -> None:
    cleaner = TextCleaner()
    assert cleaner.clean("<p>hello <b>world</b></p>") == "hello world"


def test_strip_script_and_style_block() -> None:
    cleaner = TextCleaner()
    result = cleaner.clean("a<script>alert(1)</script>b<style>x{y:z}</style>c")
    # script/style 整块删,留下 a b c
    assert "alert" not in result
    assert "{y:z}" not in result
    assert "a" in result and "b" in result and "c" in result


def test_normalize_whitespace() -> None:
    cleaner = TextCleaner()
    # 全角空格 → 半角,多空格折叠,行尾空格删
    assert cleaner.clean("a　 b   c   \nd") == "a b c\nd"


def test_collapse_blank_lines() -> None:
    cleaner = TextCleaner()
    result = cleaner.clean("a\n\n\n\n\nb")
    assert result == "a\n\nb"


def test_keep_markdown_structure() -> None:
    cleaner = TextCleaner()
    md = "# Title\n\n## Section\n\n- item1\n- item2\n\n```python\nx = 1\n```\n"
    result = cleaner.clean(md)
    # 关键标记必须保留
    assert "# Title" in result
    assert "## Section" in result
    assert "- item1" in result
    assert "```python" in result


def test_strip_leading_trailing_whitespace() -> None:
    cleaner = TextCleaner()
    assert cleaner.clean("  \n\n  hello  \n\n  ") == "hello"


def test_real_world_dirty_markdown() -> None:
    """模拟一份从 Windows 复制粘贴的脏 markdown。"""
    cleaner = TextCleaner()
    dirty = (
        "﻿# CPU 告警处理\r\n"
        "\r\n"
        "## 排查步骤\r\n"
        "\r\n"
        "1. 用 top 看进程​\r\n"
        "   <br/>\r\n"
        "2. 用 ps 找异常\r\n"
        "\r\n"
        "\r\n"
        "\r\n"
        "<script>恶意</script>\r\n"
    )
    result = cleaner.clean(dirty)
    assert result.startswith("# CPU 告警处理")
    assert "﻿" not in result
    assert "​" not in result
    assert "<br" not in result
    assert "<script" not in result
    assert "恶意" not in result
    # 3 个空行已折叠成 2 个
    assert "\n\n\n" not in result
