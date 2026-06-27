"""文档入库前的文本清洗服务。

放在 index_worker 的 `read_text` 之后、`split_document` 之前。
目标:把"用户给的脏文本"洗成"对 embedding 友好的干净文本"。

清洗规则(按顺序):
1. 编码残留:BOM (﻿)、零宽空格 (​/‌/‍)、控制字符
2. 换行统一:\r\n / \r → \n
3. HTML 标签:剥掉 <script>/<style>/普通标签
4. 空白规范化:全角空格 → 半角;tab → 空格;多空格折叠为单空格
5. 空行折叠:连续 3+ 空行折叠成 2 个(保留 markdown 段落感)

设计原则:
- 纯函数,无状态,易测
- 保留 markdown 结构(# / ## / - / ``` 等)不动
- 不"过度清洗"——比如不强制全角→半角中文标点(那会改变语义)
"""

from __future__ import annotations

import re

from loguru import logger


# 控制字符:除了 \t \n \r 之外的 0x00-0x1F 和 0x7F
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# 零宽字符
_ZERO_WIDTH_RE = re.compile(r"[​‌‍⁠﻿]")

# HTML 标签(<script>/<style> 整段删,其他只删标签留内容)
_HTML_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# 多空格(不含换行)
_MULTI_SPACE_RE = re.compile(r"[ \t]+")

# 行尾空格
_TRAILING_SPACE_RE = re.compile(r"[ \t]+\n")

# 3+ 空行折叠成 2 个
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")


class TextCleaner:
    """文档入库前的文本清洗。"""

    def clean(self, content: str) -> str:
        """主入口。空内容直接返回 ""。"""
        if not content:
            return ""

        original_len = len(content)

        content = self._strip_encoding_artifacts(content)
        content = self._normalize_newlines(content)
        content = self._strip_html(content)
        content = self._normalize_whitespace(content)
        content = self._collapse_blank_lines(content)

        result = content.strip()
        cleaned_len = len(result)
        if original_len and cleaned_len < original_len:
            logger.debug(f"TextCleaner: {original_len} → {cleaned_len} 字符 (-{original_len - cleaned_len})")
        return result

    # ---- 子步骤 ----

    @staticmethod
    def _strip_encoding_artifacts(text: str) -> str:
        """删除 BOM、零宽、控制字符。"""
        text = _ZERO_WIDTH_RE.sub("", text)
        text = _CONTROL_CHARS_RE.sub("", text)
        return text

    @staticmethod
    def _normalize_newlines(text: str) -> str:
        """Windows / Mac 换行 → Unix。"""
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _strip_html(text: str) -> str:
        """删除 HTML。<script>/<style> 整块删,其他只去标签保留内容。"""
        text = _HTML_SCRIPT_RE.sub("", text)
        text = _HTML_TAG_RE.sub("", text)
        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """全角空格 → 半角;tab → 空格;多空格折叠;行尾空格删掉。"""
        text = text.replace("　", " ")  # 全角空格
        text = _MULTI_SPACE_RE.sub(" ", text)
        text = _TRAILING_SPACE_RE.sub("\n", text)
        return text

    @staticmethod
    def _collapse_blank_lines(text: str) -> str:
        """3+ 个换行折叠成 2 个(保留段落感)。"""
        return _MULTI_BLANK_LINE_RE.sub("\n\n", text)


# 全局单例
text_cleaner = TextCleaner()
