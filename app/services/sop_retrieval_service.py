"""SOP 检索服务 —— 把告警事件与知识库 SOP 关联起来。

这是两周 AIOps 首响冲刺里最核心的一块（ADR-004）：
把本项目已有的 RAG 混合检索能力，接入告警诊断链路，
让首响报告的证据**可溯源**——每条 SOP 证据都带来源标题和命中片段。

设计要点：
1. 输入标准化的 AlarmEvent，用 event.retrieval_query() 构造检索语句。
2. 检索结果转成 aiops_report.Evidence（type=VERIFIED_FACT, source=SOP），
   与「模型推断」证据严格区分。
3. 检索器依赖注入（默认走 vector_search_service 单例），
   便于测试隔离，不必真连 Milvus（呼应 ADR-000 的测试纪律）。

关联 ADR：docs/adr/004-sop-retrieval-and-evidence-tracing.md
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

from app.config import config
from app.core.breakers import retrieval_breaker
from app.core.errors import RetrievalStatus, error_code_of
from app.core.metrics import count_retrieval_failure
from app.models.aiops import AlarmEvent
from app.models.aiops_report import Evidence, EvidenceSource, EvidenceType


class DocumentLike(Protocol):
    """检索返回的文档最小接口（page_content + metadata）。

    用 Protocol 而非直接依赖 langchain 的 Document，
    这样测试里可以传任意鸭子类型的假文档。
    """

    page_content: str
    metadata: dict[str, Any]


class Retriever(Protocol):
    """检索器最小接口，便于依赖注入与测试替换。"""

    def retrieve_documents(self, query: str, top_k: int = 3) -> list[Any]: ...


# 一条 SOP 命中片段最多截取的字符数（避免把整篇文档塞进报告）
_MAX_EXCERPT_CHARS = 300


class SopRetrievalService:
    """把告警事件映射到相关 SOP 证据。"""

    def __init__(self, retriever: Retriever | None = None, top_k: int | None = None):
        # 延迟到方法内取默认检索器，避免 import 期触发单例连接 Milvus
        self._retriever = retriever
        self._top_k = top_k if top_k is not None else config.rag_top_k

    def _get_retriever(self) -> Retriever:
        if self._retriever is not None:
            return self._retriever
        # 默认使用项目已有的混合检索单例
        from app.services.vector_search_service import vector_search_service

        return vector_search_service

    def retrieve_sop_evidence(
        self, alarm: AlarmEvent
    ) -> tuple[list[Evidence], RetrievalStatus]:
        """针对一条告警检索 SOP，返回（可溯源证据列表, 检索状态）。

        **为什么要多返回一个 status**

        这个方法以前只返回 list[Evidence]：检索炸了返回 []，检索到 0 条也返回 []。
        调用方只能 `if evidence:`，于是两种完全不同的情况被压成同一个信号：

            Milvus 挂了            → []  → prompt 说「未检索到相关 SOP」
            知识库真的没这篇 SOP   → []  → prompt 说「未检索到相关 SOP」

        第一种情况下这句话是**假的**。模型据此推断「这个告警没有 SOP」，
        报告最后写出「建议补充该告警的 SOP 文档」—— 值班同学于是去写文档，
        而真正该做的是去看向量库为什么挂了。排障方向被带反。

        这不是降级，是撒谎：降级是「我少给你一部分能力，并且告诉你少了什么」，
        撒谎是「把依赖故障伪装成一个正常的业务结论」。

        信息在这一层就丢了，后面无论 prompt 怎么写都救不回来 ——
        所以必须在这里把 failed / empty 分开，让调用方能如实往下传。

        依然不抛异常：诊断链路不能因为检索失败而崩，值班同学宁愿看到
        「没有 SOP 证据的分析 + 明确的失败声明」，也不愿看到 500。
        """
        query = alarm.retrieval_query()

        # 运行时开关：关掉之后如实返回 FAILED，而不是 EMPTY。
        #
        # 为什么是 FAILED —— 因为 EMPTY 的含义是「知识库里确实没有」，
        # 那是一个**业务结论**。开关关着的时候我们根本没去查，
        # 说「库里没有」就是上面那条撒谎链路的翻版，只不过这次撒谎的
        # 起因是我们自己关了开关。FAILED 的措辞（「检索本身没成功，
        # 这不代表知识库里没有」）对开关关闭同样成立。
        #
        # 为什么不新增一个 DISABLED 状态：下游对它的处置和 FAILED
        # 一模一样（标降级 + 在报告里声明证据链断了），
        # 多一个状态会让每个消费者都多一条分支,却没有任何行为差异（YAGNI）。
        # 真正需要区分「是坏了还是被关了」的场合是排障，那看这行日志就够了。
        if not config.enable_sop_retrieval:
            # 这里**刻意不记任何指标**。
            #
            # 不记 retrieval_failures_total:那个指标的含义是「发出去的检索调用
            # 有多少失败了」。开关关着的时候我们一次调用都没发,
            # 把它算成失败会让「检索健康度」这个指标失去意义 ——
            # 明明 Milvus 好得很,失败率却是 100%。
            #
            # 也不记 degrade_total:这次降级已经由**调用方**记了一次
            # (见 first_response_service._count_report_degrade)。
            # 在这里再记一次,同一个逻辑事件会以两个不同的 reason
            # 各记一笔(这里是 feature_disabled、那边是 retrieval_failed),
            # degrade_total 的总数就凭空翻倍。
            #
            # 排障时两个数据源互相矛盾比没有数据更糟 —— 你不知道该信哪个,
            # 只能两边都不信。指标的记账点必须唯一。
            logger.warning(
                f"SOP 检索已被开关关闭(enable_sop_retrieval=false)，"
                f"本次分析无知识库证据: alert={alarm.alert_name}"
            )
            return [], RetrievalStatus.FAILED

        logger.info(f"SOP 检索: alert={alarm.alert_name}, query='{query}'")

        try:
            # 熔断器守在这里而不是守在 vector_search_service 内部：
            # 它要统计的是「这次调用成没成」，而检索内部有自己的
            # 局部降级（混合检索失败退回纯向量检索），那种退回是**成功**，
            # 熔断器不该看见。放在调用点，看到的就是最终结果。
            with retrieval_breaker.guard():
                docs = self._get_retriever().retrieve_documents(query, top_k=self._top_k)
        except Exception as exc:
            # 检索服务不可用：不抛，但**必须**如实标成 FAILED。
            # 这里升级为 error 级别 —— 它代表依赖故障，不是业务上的「没找到」。
            #
            # 熔断打开时走的也是这条分支,一行都不用改:CircuitOpenError
            # 是 AppError 子类,error_code_of 给出 circuit_open,
            # 日志和降级原因自动就是对的 —— 这正是 guard() 选择抛异常
            # 而不是返回布尔值的理由。
            code = error_code_of(exc)
            # 与 rag_v2 的检索失败共用同一个指标:两处打的是同一个 Milvus,
            # 分成两个指标名的话,值班同学得把两条曲线加起来才知道
            # 「检索到底挂得多厉害」—— 而 code 这个 label 已经能区分故障类型,
            # 真要按来源拆分,那属于加 label 而不是加指标。
            count_retrieval_failure(code)
            logger.error(f"SOP 检索失败[{code}]，本次分析无知识库证据: {exc}")
            return [], RetrievalStatus.FAILED

        evidence: list[Evidence] = []
        for doc in docs or []:
            evidence.append(self._doc_to_evidence(doc))

        if not evidence:
            # 检索服务正常，只是知识库里确实没有相关内容。
            # 这是一个**真实的业务结论**，可以放心告诉模型。
            logger.info("SOP 检索完成，知识库中无相关内容")
            return [], RetrievalStatus.EMPTY

        logger.info(f"SOP 检索完成，命中 {len(evidence)} 条证据")
        return evidence, RetrievalStatus.OK

    def _doc_to_evidence(self, doc: DocumentLike) -> Evidence:
        """把一个检索文档转为一条 VERIFIED_FACT 证据。"""
        metadata = getattr(doc, "metadata", None) or {}
        content = getattr(doc, "page_content", "") or ""

        title = self._build_source_title(metadata)
        excerpt = content.strip()[:_MAX_EXCERPT_CHARS]

        return Evidence(
            type=EvidenceType.VERIFIED_FACT,
            source=EvidenceSource.SOP,
            content=f"知识库 SOP 命中：{title}",
            source_title=title,
            excerpt=excerpt,
        )

    @staticmethod
    def _build_source_title(metadata: dict[str, Any]) -> str:
        """从文档元数据里拼一个人可读的来源标题。

        优先用标题层级（h1 > h2 > h3），退回到文件名，最后兜底「未知来源」。
        """
        headers = [
            str(metadata[key])
            for key in ("h1", "h2", "h3")
            if metadata.get(key)
        ]
        if headers:
            file_name = metadata.get("_file_name")
            header_str = " > ".join(headers)
            return f"{file_name} · {header_str}" if file_name else header_str
        return str(metadata.get("_file_name") or "未知来源")


# 全局单例（默认检索器在首次调用时惰性获取）
sop_retrieval_service = SopRetrievalService()
