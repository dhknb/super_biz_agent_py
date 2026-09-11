"""持久化会话记忆：滚动摘要 + 有限最近消息窗口。

完整原始消息只保存在 PostgreSQL，作为审计记录；模型上下文仅使用压缩摘要
和未压缩消息的尾部，避免输入 Token 随对话轮数无限增长。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from sqlalchemy.orm import Session

from app.config import config
from app.models.conversation import ConversationMessage
from app.repositories.conversation_repository import ConversationRepository


class SummaryModel(Protocol):
    async def ainvoke(self, messages: Any) -> Any: ...


_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_SUMMARY_SYSTEM_PROMPT = """你负责压缩一段会话记忆，供后续 RAG 问答使用。
只保留用户明确陈述的稳定事实、对象名称、环境、约束、已完成事项、待确认项和偏好。
不要把模型猜测、SOP 内容或历史数值写成实时事实；发生冲突时以较新的用户消息为准。
用中文输出，严格使用以下紧凑标题；缺失项写“无”：
【明确事实】
【任务进展】
【约束与偏好】
【待确认项】"""


@dataclass(frozen=True)
class ConversationMemoryContext:
    """实际交给 RAG 节点的有界上下文。"""

    text: str = ""
    estimated_tokens: int = 0
    used_summary: bool = False
    recent_message_count: int = 0


def estimate_tokens(text: str) -> int:
    """以中文更保守的规则估算 Token，避免上下文预算明显低估。"""
    if not text:
        return 0
    cjk_chars = len(_CJK_RE.findall(text))
    other_chars = max(0, len(text) - cjk_chars)
    return cjk_chars + math.ceil(other_chars / 4)


def _message_token_cost(message: ConversationMessage) -> int:
    return estimate_tokens(message.content) + 4


class ConversationMemoryService:
    def __init__(
        self,
        summarizer: SummaryModel | None = None,
        *,
        context_token_budget: int | None = None,
        summary_token_budget: int | None = None,
        compact_threshold_tokens: int | None = None,
        recent_turns: int | None = None,
    ):
        self._summarizer = summarizer
        self._context_token_budget = (
            context_token_budget
            if context_token_budget is not None
            else config.conversation_memory_context_token_budget
        )
        self._summary_token_budget = (
            summary_token_budget
            if summary_token_budget is not None
            else config.conversation_memory_summary_token_budget
        )
        self._compact_threshold_tokens = (
            compact_threshold_tokens
            if compact_threshold_tokens is not None
            else config.conversation_memory_compact_threshold_tokens
        )
        self._recent_turns = (
            recent_turns
            if recent_turns is not None
            else config.conversation_memory_recent_turns
        )

    async def build_context(
        self,
        db: Session,
        session_id: str,
    ) -> ConversationMemoryContext:
        """读取持久化历史，必要时压缩旧消息，返回固定预算的注入文本。"""
        repo = ConversationRepository(db)
        messages = repo.list_active_messages(session_id)
        if not messages:
            return ConversationMemoryContext()

        snapshot = repo.get_memory_snapshot(session_id)
        pending_messages = self._messages_after_snapshot(
            messages,
            snapshot.summarized_through_message_id if snapshot else None,
        )
        recent_count = self._recent_message_count(pending_messages)
        compact_candidates = pending_messages[:-recent_count] if recent_count else []

        if self._token_count(compact_candidates) >= self._compact_threshold_tokens:
            summary = await self._summarize(
                existing_summary=snapshot.summary if snapshot else "",
                messages=compact_candidates,
            )
            if summary:
                snapshot = repo.save_memory_snapshot(
                    session_id,
                    summary=summary,
                    summarized_through_message_id=compact_candidates[-1].id,
                )
                pending_messages = self._messages_after_snapshot(
                    messages,
                    snapshot.summarized_through_message_id,
                )

        return self._render_context(
            summary=snapshot.summary if snapshot else "",
            pending_messages=pending_messages,
        )

    @staticmethod
    def _messages_after_snapshot(
        messages: Sequence[ConversationMessage],
        summarized_through_message_id: str | None,
    ) -> list[ConversationMessage]:
        if not summarized_through_message_id:
            return list(messages)
        for index, message in enumerate(messages):
            if message.id == summarized_through_message_id:
                return list(messages[index + 1 :])
        # 游标对应的消息已被软删除时，保守地不注入旧原文，避免重复历史。
        return []

    def _recent_message_count(self, messages: Sequence[ConversationMessage]) -> int:
        """按用户轮次保留尾部消息；一轮按用户消息及其后的助手消息计算。"""
        if not messages or self._recent_turns <= 0:
            return 0
        user_messages_seen = 0
        start_index = 0
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role.value == "user":
                user_messages_seen += 1
                if user_messages_seen >= self._recent_turns:
                    start_index = index
                    break
        return len(messages) - start_index

    def _token_count(self, messages: Sequence[ConversationMessage]) -> int:
        return sum(_message_token_cost(message) for message in messages)

    async def _summarize(
        self,
        *,
        existing_summary: str,
        messages: Sequence[ConversationMessage],
    ) -> str:
        try:
            summarizer = self._get_summarizer()
            source = self._render_messages_for_summary(messages)
            existing = self._truncate_text(existing_summary, self._summary_token_budget)
            prompt = (
                f"已有滚动摘要：\n{existing or '（无）'}\n\n"
                f"需要合并的新消息：\n{source}\n\n"
                "请合并为新的滚动摘要。"
            )
            response = await summarizer.ainvoke(
                [SystemMessage(content=_SUMMARY_SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            raw_summary = getattr(response, "content", None) or str(response)
            return self._truncate_text(raw_summary.strip(), self._summary_token_budget)
        except Exception as exc:
            # 摘要是节省 Token 的优化，不可阻断主聊天；失败时退回最近消息窗口。
            logger.warning(f"会话记忆摘要失败，降级为最近消息窗口: {exc}")
            return ""

    def _get_summarizer(self) -> SummaryModel:
        if self._summarizer is not None:
            return self._summarizer
        from app.core.llm_factory import llm_factory

        return llm_factory.create_chat_model(
            model=config.rag_model,
            temperature=0,
            streaming=False,
        )

    @staticmethod
    def _render_messages_for_summary(messages: Sequence[ConversationMessage]) -> str:
        lines = []
        for message in messages:
            role = "用户" if message.role.value == "user" else "助手"
            lines.append(f"{role}：{message.content}")
        return "\n".join(lines)

    def _render_context(
        self,
        *,
        summary: str,
        pending_messages: Sequence[ConversationMessage],
    ) -> ConversationMemoryContext:
        summary_header = "【会话滚动摘要】\n"
        recent_header = "【最近会话原文】\n"
        summary_budget = min(
            self._summary_token_budget,
            max(0, self._context_token_budget - estimate_tokens(summary_header)),
        )
        summary = self._truncate_text(summary, summary_budget)
        summary_cost = estimate_tokens(summary_header + summary) if summary else 0
        # 预留最近消息标题，确保最终渲染文本（含标签）也不会突破总预算。
        remaining_budget = max(
            0,
            self._context_token_budget - summary_cost - estimate_tokens(recent_header),
        )
        recent_messages = self._take_tail_to_budget(pending_messages, remaining_budget)

        sections: list[str] = []
        if summary:
            sections.append(f"{summary_header}{summary}")
        if recent_messages:
            sections.append(
                recent_header
                + self._render_messages_for_summary(recent_messages)
            )
        if not sections:
            return ConversationMemoryContext()

        text = "\n\n".join(sections)
        # 角色前缀、分隔符等是实际 Prompt 的一部分，最后再兜底一次确保硬上限。
        text = self._truncate_text(text, self._context_token_budget)
        return ConversationMemoryContext(
            text=text,
            estimated_tokens=estimate_tokens(text),
            used_summary=bool(summary),
            recent_message_count=len(recent_messages),
        )

    def _take_tail_to_budget(
        self,
        messages: Sequence[ConversationMessage],
        token_budget: int,
    ) -> list[ConversationMessage]:
        selected: list[ConversationMessage] = []
        consumed = 0
        for message in reversed(messages):
            cost = _message_token_cost(message)
            if selected and consumed + cost > token_budget:
                break
            if not selected and cost > token_budget:
                # 单条消息超预算时仍保留尾部，内容裁剪由渲染阶段处理。
                clipped = ConversationMessage(
                    id=message.id,
                    session_id=message.session_id,
                    role=message.role,
                    content=self._truncate_text(message.content, max(1, token_budget - 4)),
                )
                selected.append(clipped)
                break
            selected.append(message)
            consumed += cost
        return list(reversed(selected))

    @staticmethod
    def _truncate_text(text: str, token_budget: int) -> str:
        if estimate_tokens(text) <= token_budget:
            return text
        if token_budget <= 0:
            return ""

        chars: list[str] = []
        used = 0
        for char in text:
            cost = 1 if _CJK_RE.match(char) else 0.25
            if used + cost > token_budget:
                break
            chars.append(char)
            used += cost
        return "".join(chars).rstrip() + "…"


conversation_memory_service = ConversationMemoryService()
