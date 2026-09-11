"""告警首响分析服务 —— 把整条闭环串起来。

这是 Day5 的收口：把前面几天的零件组装成一条可用的诊断链路。

    AlarmEvent
      → SOP 检索（可溯源证据）
      → 构造 prompt（告警上下文 + SOP 证据）
      → LLM 生成结构化 JSON
      → parse_report（失败降级为纯文本）
      → 合并 SOP 证据到报告

设计要点：
1. LLM 与 SOP 检索都**依赖注入**，测试时传假对象，不连真实服务
   （呼应 ADR-000 测试纪律、ADR-004 证据溯源）。
2. 「主流程固定、节点内部智能」——检索是固定步骤，模型只负责理解与组织，
   与产品方向文档 5.2 节的半固定链路一致。
3. 模型推断与已验证事实在报告里始终区分：SOP 证据由代码注入为
   VERIFIED_FACT，模型自己产出的 evidence 视为推断。

关联 ADR：docs/adr/004-sop-retrieval-and-evidence-tracing.md
"""

from __future__ import annotations

import json
import time
from textwrap import dedent
from typing import Any, Callable, Protocol

from loguru import logger

from app.core.breakers import llm_breaker
from app.core.errors import DegradeReason, RetrievalStatus, wrap_llm_exception
from app.core.metrics import count_degrade, observe_llm_call
from app.models.aiops import AlarmEvent
from app.models.aiops_report import (
    Evidence,
    EvidenceType,
    FirstResponseReport,
    parse_report,
)
from app.services.conversation_memory_service import estimate_tokens
from app.services.sop_retrieval_service import SopRetrievalService, sop_retrieval_service


class LLMLike(Protocol):
    """LLM 最小接口：给一段 prompt，返回文本。便于注入假 LLM。"""

    async def ainvoke(self, messages: Any) -> Any: ...


# 阶段回调签名：(阶段名, 该阶段的关键计数)。
# 用 str 而不是 Enum：这些名字要和 DiagnosisTaskStatus 的取值对齐，
# 再定义一个平行的枚举就是两套需要同步维护的真相（DRY）。
PhaseCallback = Callable[[str, dict[str, Any]], None]

# 阶段常量。定成常量而不是散落的字面量，是为了让编排层的 dispatch 表
# 和这里对齐时有唯一来源 —— 拼错一个字符串，静默少一条时间线事件。
#
# 公开（无下划线前缀）是因为编排层要 import 它们来建阶段→状态的映射表。
# 让调用方去写 `"retrieving"` 字面量也能跑，但那就等于把「阶段名」这件事
# 复制成了两份：这里改一个词，编排层的 dispatch 表静默失配，
# 表现是时间线少一条事件 —— 不报错，只是查详情时那一步凭空消失了。
PHASE_RETRIEVING = "retrieving"
PHASE_RETRIEVED = "retrieved"
PHASE_DIAGNOSING = "diagnosing"
PHASE_DIAGNOSED = "diagnosed"


def _elapsed_ms(start: float) -> int:
    """从 perf_counter 起点算出经过的毫秒数。

    用 perf_counter 而不是 datetime 相减：后者受 NTP 校时影响，
    可能算出负数耗时 —— 而耗时是要进指标做分位数统计的。
    """
    return int((time.perf_counter() - start) * 1000)


def _make_emitter(on_phase: PhaseCallback | None) -> PhaseCallback:
    """把可选回调包装成「一定可以直接调用、且一定不抛」的函数。

    两个作用：
    1. 消掉调用点的 `if on_phase is not None:` —— 那会在 analyze 里
       重复三次，且漏写一次就静默少一条事件。
    2. 吞掉回调自身的异常。回调是编排层传进来的，里面会写数据库；
       数据库抖一下不该让告警分析失败 —— 观测设施的故障不能升级成业务故障。
       这和 span_context / job_failure 是同一条原则。
    """
    if on_phase is None:
        return lambda phase, payload: None

    def emit(phase: str, payload: dict[str, Any]) -> None:
        try:
            on_phase(phase, payload)
        except Exception as exc:
            logger.warning(f"阶段回调失败（不影响分析）: phase={phase}: {exc}")

    return emit


_SYSTEM_PROMPT = dedent(
    """
    你是一线值班工程师的告警首响分析助手。给定一条告警和相关 SOP 片段，
    你要产出一份**结构化 JSON** 首响报告，帮助值班同学快速开始第一轮排查。

    严格要求：
    1. 只输出一个 JSON 对象，不要输出多余的解释文字或 markdown 代码块外的内容。
    2. 所有结论必须基于给定信息。**没有证据支撑的判断，放进 pending_confirmations，
       不要写成确定根因。**
    3. recommended_checks 只能是「检查 / 观察」类动作（看日志、查指标、确认配置），
       **禁止**给出重启、删除、扩容等会改变系统状态的高风险命令。
    4. 不要编造主机名、指标值、日志内容。信息不足就在 pending_confirmations 说明。

    JSON 结构：
    {
      "alert_summary": "一句话说清发生了什么",
      "current_judgment": "当前能确定/怀疑的方向",
      "severity_assessment": "严重性评估",
      "recommended_checks": [
        {"order": 1, "action": "...", "reason": "...", "command_hint": "可选只读命令", "requires_human": false}
      ],
      "evidence": [
        {"type": "model_inference", "source": "model", "content": "..."}
      ],
      "root_cause_hypotheses": ["假设1", "假设2"],
      "pending_confirmations": ["待确认项1"],
      "risk_notes": "风险与升级建议"
    }
    """
).strip()


class FirstResponseService:
    """告警首响分析：AlarmEvent → 结构化报告。"""

    def __init__(
        self,
        llm: LLMLike | None = None,
        sop_service: SopRetrievalService | None = None,
    ):
        self._llm = llm
        self._sop_service = sop_service or sop_retrieval_service

    def _get_llm(self) -> LLMLike:
        if self._llm is not None:
            return self._llm
        # 延迟构造，避免 import 期需要 API key。
        # 走 llm_factory 而不是直接 new ChatQwen：超时与重试次数由工厂统一注入。
        # 这里原来是全项目三个「裸构造」之一，都没有 timeout ——
        # 告警首响是值班同学等着看的接口，模型挂住就等于首响无限延迟。
        from app.core.llm_factory import llm_factory

        return llm_factory.create_qwen_model(temperature=0)

    async def analyze(
        self,
        alarm: AlarmEvent,
        *,
        on_phase: PhaseCallback | None = None,
    ) -> FirstResponseReport:
        """对一条告警生成结构化首响报告。

        无论中间哪步出问题，都返回一个可用的 FirstResponseReport
        （必要时是降级报告），绝不抛异常给上层。

        ## on_phase 是干什么的

        本方法内部有两个耗时完全不同量级的阶段：SOP 检索（百毫秒级）
        和 LLM 生成（十秒级）。但从外面看它只是一个 await ——
        编排层无法知道「现在跑到哪一步了」，于是只能在调用**之前**
        把 planning / retrieving / diagnosing 三个状态一次推完。
        结果是时间线上前三个事件的时间戳只差几毫秒，`done` 在几十秒后，
        看时间线的人完全不知道那几十秒花在哪。

        加一个回调就能把真实边界暴露出来：每个阶段**真正开始**时
        回调一次 `(phase, payload)`，编排层据此推进状态并落一条事件。
        payload 里带上该阶段的 `duration_ms` 与关键计数
        （`sop_hit_count` / `prompt_tokens`），让时间线不只有顺序，还有耗时。

        为什么用回调而不是改成异步生成器：
        生成器会把返回值语义搞复杂（要么 `yield` 事件要么 `return` 报告，
        调用方得写 `async for` + 取 `StopAsyncIteration.value`），
        而现有调用方只想拿一份报告。回调是**加法**：不传就是原来的行为，
        现有测试和调用点一行都不用改（KISS + 向后兼容）。

        回调自身的异常一律吞掉：观测设施不该让业务失败。
        """
        emit = _make_emitter(on_phase)

        # 1. 固定步骤：检索 SOP 证据（已带溯源信息）
        #    注意这里要接住 status：证据为空时，「检索挂了」和「知识库没这篇」
        #    必须一路如实传到 prompt 和报告里，不能压成同一个信号。
        #
        #    retrieving 事件在检索**开始前**发，不是结束后 ——
        #    时间线要回答的是「现在在干什么」，事后补报就失去了实时性。
        emit(PHASE_RETRIEVING, {})
        retrieve_start = time.perf_counter()
        sop_evidence, sop_status = self._sop_service.retrieve_sop_evidence(alarm)
        retrieve_ms = _elapsed_ms(retrieve_start)

        # 2. 构造 prompt（告警上下文 + SOP 证据 + 检索状态声明）
        user_prompt = self._build_user_prompt(alarm, sop_evidence, sop_status)

        # 检索这一步真实结束了，把耗时和命中数如实报出去。
        # sop_status 一起带上：0 命中有两种含义（挂了 / 库里没有），
        # 只报数字的话时间线又会把这两件事压成同一个信号。
        emit(
            PHASE_RETRIEVED,
            {
                "duration_ms": retrieve_ms,
                "sop_hit_count": len(sop_evidence),
                "retrieval_status": sop_status.value,
            },
        )

        # 3. 调用 LLM
        #    diagnosing 在这里发，而不是在方法入口 —— 这是本次修复的核心：
        #    以前它在检索都还没开始时就被推进了。
        emit(PHASE_DIAGNOSING, {"prompt_tokens": estimate_tokens(user_prompt)})
        generate_start = time.perf_counter()
        try:
            llm = self._get_llm()
            # 与 rag_v2 三个节点共用同一个 llm_breaker：打的是同一个模型服务,
            # 会同时坏也会同时好。分开数各自的失败次数只会让灵敏度减半,
            # 而 AIOps 首响的 QPS 很低,单独一个熔断器可能永远攒不够阈值。
            with llm_breaker.guard():
                # 计时器在 guard() **内侧**：熔断打开时这次调用根本没发出去,
                # 耗时是微秒级。放外侧会把几百个「没打出去的调用」当成
                # 0 秒样本记进 Histogram —— 熔断期间 P99 反而变好看,
                # 下游挂了而耗时指标显示一切正常。
                with observe_llm_call("aiops_first_response"):
                    response = await llm.ainvoke(
                        [
                            ("system", _SYSTEM_PROMPT),
                            ("user", user_prompt),
                        ]
                    )
            raw_text = getattr(response, "content", None) or str(response)
        except Exception as exc:
            # 用统一的异常分类拿 code / degrade_reason，而不是把 str(exc) 塞进报告：
            # 自由文本没法聚合统计，也分不出「超时」和「上游 5xx」——
            # 前者该调超时阈值或扩容，后者该去看上游。
            wrapped = wrap_llm_exception(exc)
            logger.error(f"首响分析 LLM 调用失败[{wrapped.code}]: {wrapped.message}")
            report = FirstResponseReport(
                alert_summary=f"告警 {alarm.alert_name} 分析失败",
                current_judgment="LLM 调用异常，无法生成分析。",
                is_degraded=True,
                degrade_reason=wrapped.degrade_reason,
                raw_text=f"LLM 调用异常[{wrapped.code}]: {wrapped.message}",
            )
            # 即便 LLM 挂了，SOP 证据仍然有价值，合并进去
            report.evidence = list(sop_evidence)
            # 检索也挂了的话，这里要一并声明：两个故障叠加时不能只报一个。
            self._apply_retrieval_status(report, sop_status)
            self._count_report_degrade(report)
            # 失败也要报耗时。这条比成功路径更重要：LLM 超时的场景下，
            # 它记的就是「白等了多少毫秒才放弃」—— 调超时阈值时唯一的依据。
            emit(
                PHASE_DIAGNOSED,
                {
                    "duration_ms": _elapsed_ms(generate_start),
                    "error_code": wrapped.code,
                    "is_degraded": True,
                },
            )
            return report

        # 4. 解析（失败自动降级，永不抛异常；降级时会标 parse_failed）
        report = parse_report(raw_text)

        # 5. 合并 SOP 证据：代码注入的 SOP 证据是 VERIFIED_FACT，
        #    放在模型自产证据前面，保证「事实优先」
        report.evidence = list(sop_evidence) + [
            ev for ev in report.evidence if ev.type != EvidenceType.VERIFIED_FACT
        ]

        # 6. 检索失败必须体现在报告上：否则这份报告看起来是一份
        #    「正常的、只是没引用 SOP 的分析」，读者无从知道证据链是断的。
        self._apply_retrieval_status(report, sop_status)
        self._count_report_degrade(report)

        # 生成阶段真实结束。这个 duration_ms 就是时间线上那段
        # 「几十秒的空白」的正主 —— 以前它被压在 diagnosing 到 done 之间，
        # 无从知道到底是模型慢还是别的什么慢。
        emit(
            PHASE_DIAGNOSED,
            {
                "duration_ms": _elapsed_ms(generate_start),
                "is_degraded": report.is_degraded,
                "answer_length": len(report.raw_text or ""),
            },
        )
        return report

    @staticmethod
    def _count_report_degrade(report: FirstResponseReport) -> None:
        """把报告的降级状态记进指标。

        为什么要抽成一个方法：`analyze` 有**两条**返回路径
        （LLM 挂掉的早返回、正常走完的返回），两处各写一遍
        `if report.is_degraded: count_degrade(...)` 迟早会漏掉一处，
        而漏掉的通常是早返回那条 —— 也就是故障时唯一会走的那条（DRY）。

        必须在 `_apply_retrieval_status` **之后**调用：
        检索失败的降级标记是在那个方法里才被设上的，
        提前统计会漏掉「LLM 正常但检索挂了」这一类降级。

        为什么不能复用 rag_v2 的埋点切面（instrumentation.py）：
        那一层包的是 LangGraph 节点，而首响分析不是图节点，
        它没有 patch 也没有 span_scope。唯一的共同事实来源是
        `report.is_degraded` —— 所以在这里读它，
        而不是在 rag_v2 那边硬塞一个通道进来。
        """
        if report.is_degraded:
            reason = report.degrade_reason
            count_degrade(reason.value if reason is not None else None)

    @staticmethod
    def _apply_retrieval_status(
        report: FirstResponseReport,
        sop_status: RetrievalStatus,
    ) -> None:
        """把 SOP 检索状态如实落到报告上（就地修改）。

        只有 FAILED 需要处理：OK 和 EMPTY 都是检索服务正常工作的结果，
        报告本身就是可信的，不该标降级。

        为什么要写进 risk_notes 而不只是设个字段：
        `is_degraded` / `degrade_reason` 是给程序看的（指标聚合、告警），
        risk_notes 是给值班同学看的。人不会去读数据库字段，
        他只会读报告正文 —— 所以「本次没有知识库证据」必须出现在正文里，
        否则他仍然会把一份缺证据的分析当成完整分析来用。
        """
        if sop_status is not RetrievalStatus.FAILED:
            return

        report.is_degraded = True
        # 已有更靠前的降级原因（如 LLM 挂了）时不覆盖：
        # 那个原因是报告不可用的**主因**，检索失败作为附加说明进 risk_notes。
        if report.degrade_reason is None:
            report.degrade_reason = DegradeReason.RETRIEVAL_FAILED

        note = (
            "⚠️ 知识库检索服务本次不可用，这份分析**未使用任何 SOP 证据**。"
            "请勿据此判断「该告警没有 SOP」——需要先确认检索服务（向量库）是否正常。"
        )
        report.risk_notes = f"{note}\n\n{report.risk_notes}".strip() if report.risk_notes else note

    def _build_user_prompt(
        self,
        alarm: AlarmEvent,
        sop_evidence: list[Evidence],
        sop_status: RetrievalStatus = RetrievalStatus.OK,
    ) -> str:
        """把告警和 SOP 证据组织成给模型的输入。

        sop_status 决定「没有证据」这件事怎么对模型讲 —— 这是本次修复的关键：
        同样是零条证据，说成「知识库里没有」还是「检索服务挂了」，
        模型给出的结论方向完全相反。
        """
        alarm_block = json.dumps(
            {
                "alert_name": alarm.alert_name,
                "severity": alarm.severity.value,
                "source": alarm.source.value,
                "service": alarm.service,
                "instance": alarm.instance,
                "metric_name": alarm.metric_name,
                "metric_value": alarm.metric_value,
                "summary": alarm.summary,
                "labels": alarm.labels,
                "is_firing": alarm.is_firing,
            },
            ensure_ascii=False,
            indent=2,
        )

        if sop_evidence:
            sop_lines = [
                f"[SOP {i}] {ev.source_title}\n{ev.excerpt}"
                for i, ev in enumerate(sop_evidence, 1)
            ]
            sop_block = "\n\n".join(sop_lines)
        elif sop_status is RetrievalStatus.FAILED:
            # 撒谎链路的修复点。
            #
            # 这里以前和「知识库里没有」共用同一句「未检索到相关 SOP」，
            # 于是模型顺理成章地推断「这个告警没有 SOP」，
            # 报告最后写出「建议补充该告警的 SOP 文档」。
            # 值班同学照着去写文档，而真正挂掉的是向量库。
            #
            # 现在如实告诉模型：证据缺失是**我方工具故障**，不是知识库的事实，
            # 并显式禁止它据此下「没有 SOP」的结论。
            sop_block = (
                "（⚠️ 检索服务不可用，本次分析未使用知识库证据。\n"
                "这**不代表**知识库中没有该告警的 SOP —— 是检索本身失败了。\n"
                "请勿据此建议「补充 SOP 文档」，也不要断言该告警缺少 SOP；\n"
                "请仅基于告警本身做保守分析，并在 pending_confirmations 中\n"
                "写明「知识库检索不可用，需在检索恢复后复核」。）"
            )
        else:
            # 检索服务正常、知识库确实没有相关内容 —— 这是真实结论，可以照实说。
            sop_block = "（未检索到相关 SOP，请基于告警本身分析，并把不确定项放进 pending_confirmations）"

        return dedent(
            f"""
            ## 告警事件
            {alarm_block}

            ## 相关 SOP 片段（已验证事实，可作为排查依据）
            {sop_block}

            请基于以上信息，输出结构化 JSON 首响报告。
            """
        ).strip()


# 全局单例（LLM 与检索器在首次调用时惰性获取）
first_response_service = FirstResponseService()
