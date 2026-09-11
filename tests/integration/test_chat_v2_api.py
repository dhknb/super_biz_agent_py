"""chat_v2 接口集成测试"""

from unittest.mock import AsyncMock, MagicMock, patch


class TestChatV2Endpoint:
    def test_chat_v2_returns_validation_payload(self, client) -> None:
        mock_result = {
            "answer": "test reply",
            "sub_queries": ["q1", "q2"],
            "retrieved_count": 2,
            "used_documents": [{"content": "doc", "metadata": {"id": "1"}}],
            "validation": {
                "coverage_score": 0.9,
                "groundedness_score": 0.88,
                "blocked": False,
            },
        }

        with patch("app.api.chat_v2.rag_v2_service") as mock_svc, patch(
            "app.api.chat_v2.conversation_memory_service"
        ) as mock_memory_service, patch(
            "app.api.chat_v2.ConversationRepository"
        ) as mock_repo_cls, patch("app.api.chat_v2.ChatRunTraceRepository") as mock_trace_repo_cls:
            mock_svc.query = AsyncMock(return_value=mock_result)
            mock_memory_service.build_context = AsyncMock()
            mock_memory_service.build_context.return_value.text = "memory"
            mock_repo = MagicMock()
            mock_trace_repo = MagicMock()
            mock_repo_cls.return_value = mock_repo
            mock_trace_repo_cls.return_value = mock_trace_repo
            response = client.post("/api/chat_v2", json={"Id": "s1", "Question": "hello"})
            mock_repo.append_exchange.assert_called_once_with(
                "s1",
                user_content="hello",
                assistant_content="test reply",
                message_metadata={
                    "source": "chat_v2",
                    "high_precision": {
                        "sub_queries": ["q1", "q2"],
                        "retrieved_count": 2,
                        "used_documents": [{"content": "doc", "metadata": {"id": "1"}}],
                        "validation": {
                            "coverage_score": 0.9,
                            "groundedness_score": 0.88,
                            "blocked": False,
                        },
                    },
                },
            )
            mock_trace_repo.create_trace.assert_called_once()
            mock_svc.query.assert_awaited_once_with(
                "hello",
                session_id="s1",
                conversation_context="memory",
            )

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["data"]["success"] is True
        assert data["data"]["answer"] == "test reply"
        assert data["data"]["validation"]["groundedness_score"] == 0.88

    def test_chat_v2_error_returns_real_http_500(self, client_no_raise) -> None:
        """失败时 HTTP 状态码必须是 500，不能是 200。

        改之前这里断言的是 `status_code == 200` + body 里 `code == 500`。
        那个断言恰好把 bug 锁死了：网关、负载均衡、APM、Prometheus
        全都只看 HTTP 状态码，没有一个会去解析 body 里的 code ——
        接口全挂，错误率大盘依然显示 0%。
        """
        with patch("app.api.chat_v2.rag_v2_service") as mock_svc, patch(
            "app.api.chat_v2.conversation_memory_service"
        ) as mock_memory_service, patch(
            "app.api.chat_v2.ChatRunTraceRepository"
        ) as mock_trace_repo_cls:
            mock_svc.query = AsyncMock(side_effect=RuntimeError("boom"))
            mock_memory_service.build_context = AsyncMock()
            mock_memory_service.build_context.return_value.text = ""
            mock_trace_repo = MagicMock()
            mock_trace_repo_cls.return_value = mock_trace_repo
            response = client_no_raise.post(
                "/api/chat_v2", json={"Id": "s2", "Question": "任何问题"}
            )
            # trace 仍然要落库：它是排障的唯一线索，
            # 改成 raise 之后这一点很容易被漏掉，所以必须断言。
            mock_trace_repo.create_trace.assert_called_once()

        assert response.status_code == 500
        data = response.json()
        # 响应体结构保持不变（前端依赖这三个字段）
        assert data["code"] == 500
        assert data["message"] == "error"
        assert data["data"]["success"] is False
        assert data["data"]["answer"] is None
        assert "boom" in data["data"]["errorMessage"]
        # 新增字段：可聚合的错误码 + 修复方向 + 串日志用的 request_id
        assert data["data"]["error_code"] == "runtime_error"
        assert data["data"]["degrade_reason"] == "llm_error"
        # request_id 必须有值。它由 ServerErrorMiddleware 外层的处理器生成，
        # 那时 RequestIdMiddleware 的 finally 已经 reset 了 ContextVar，
        # 只有从 ASGI scope 里取才拿得到 —— 这条断言守的就是那个坑。
        assert data["data"]["request_id"]
        assert response.headers["X-Request-ID"] == data["data"]["request_id"]

    def test_chat_v2_timeout_returns_504(self, client_no_raise) -> None:
        """超时要区分于普通失败：504 而不是 500。

        网关的超时告警、APM 的 Apdex 计算都依赖这个区分。
        全压成 500 的话，「模型慢」和「模型报错」在监控上长得一模一样，
        而这两者的处置完全不同（扩容/调超时 vs 查配额/切模型）。
        """
        with patch("app.api.chat_v2.rag_v2_service") as mock_svc, patch(
            "app.api.chat_v2.conversation_memory_service"
        ) as mock_memory_service, patch(
            "app.api.chat_v2.ChatRunTraceRepository"
        ) as mock_trace_repo_cls:
            mock_svc.query = AsyncMock(side_effect=TimeoutError("模型响应超时"))
            mock_memory_service.build_context = AsyncMock()
            mock_memory_service.build_context.return_value.text = ""
            mock_trace_repo_cls.return_value = MagicMock()
            response = client_no_raise.post(
                "/api/chat_v2", json={"Id": "s4", "Question": "任何问题"}
            )

        assert response.status_code == 504
        data = response.json()
        assert data["code"] == 504
        assert data["data"]["degrade_reason"] == "llm_timeout"


class TestChatV2TraceEndpoint:
    def test_list_bad_cases_returns_trace_summaries(self, client) -> None:
        trace = MagicMock()
        trace.id = "trace-1"
        trace.session_id = "s1"
        trace.source = "chat_v2"
        trace.question = "redis没响应怎么办"
        trace.answer = "根据已有资料无法回答该问题。"
        trace.coverage_score = 0.2
        trace.groundedness_score = 0.3
        trace.is_bad_case = True
        trace.status.value = "succeeded"
        trace.error_message = None
        trace.created_at.isoformat.return_value = "2026-06-29T10:00:00"

        with patch("app.api.chat_v2.ChatRunTraceRepository") as mock_trace_repo_cls:
            mock_trace_repo = MagicMock()
            mock_trace_repo.list_bad_cases.return_value = [trace]
            mock_trace_repo_cls.return_value = mock_trace_repo
            response = client.get("/api/chat_v2/traces/bad-cases")

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["data"][0]["id"] == "trace-1"
        assert data["data"][0]["is_bad_case"] is True

    def test_get_trace_returns_detail(self, client) -> None:
        trace = MagicMock()
        trace.id = "trace-1"
        trace.session_id = "s1"
        trace.source = "chat_v2"
        trace.question = "redis没响应怎么办"
        trace.answer = "根据已有资料无法回答该问题。"
        trace.coverage_score = 0.2
        trace.groundedness_score = 0.3
        trace.is_bad_case = True
        trace.status.value = "succeeded"
        trace.error_message = None
        trace.created_at.isoformat.return_value = "2026-06-29T10:00:00"
        trace.sub_queries = ["redis 排查"]
        trace.retrieved_count = 2
        trace.used_documents = [{"content": "doc"}]
        trace.validation = {"blocked": True}

        with patch("app.api.chat_v2.ChatRunTraceRepository") as mock_trace_repo_cls:
            mock_trace_repo = MagicMock()
            mock_trace_repo.get_trace.return_value = trace
            mock_trace_repo_cls.return_value = mock_trace_repo
            response = client.get("/api/chat_v2/traces/trace-1")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["sub_queries"] == ["redis 排查"]
        assert data["data"]["validation"]["blocked"] is True

    def test_get_trace_returns_404_when_missing(self, client) -> None:
        with patch("app.api.chat_v2.ChatRunTraceRepository") as mock_trace_repo_cls:
            mock_trace_repo = MagicMock()
            mock_trace_repo.get_trace.return_value = None
            mock_trace_repo_cls.return_value = mock_trace_repo
            response = client.get("/api/chat_v2/traces/missing")

        assert response.status_code == 404


class TestChatV2StreamEndpoint:
    def test_chat_v2_stream_emits_validation_event(self, client) -> None:
        async def fake_stream(*_args, **_kwargs):
            yield {"type": "sub_queries", "data": ["q1"]}
            yield {"type": "retrieved", "data": {"node": "retrieve_each", "count": 2}}
            yield {"type": "used_documents", "data": [{"content": "doc", "metadata": {"id": "1"}}]}
            yield {"type": "answer", "data": "fallback"}
            yield {"type": "validation", "data": {"blocked": True, "groundedness_score": 0.4}}
            yield {"type": "complete"}

        with patch("app.api.chat_v2.rag_v2_service") as mock_svc, patch(
            "app.api.chat_v2.conversation_memory_service"
        ) as mock_memory_service, patch(
            "app.api.chat_v2.ChatRunTraceRepository"
        ) as mock_trace_repo_cls, patch("app.api.chat_v2.ConversationRepository") as mock_repo_cls:
            mock_svc.query_stream = fake_stream
            mock_memory_service.build_context = AsyncMock()
            mock_memory_service.build_context.return_value.text = "memory"
            mock_trace_repo = MagicMock()
            mock_repo_cls.return_value = MagicMock()
            mock_trace_repo_cls.return_value = mock_trace_repo
            response = client.post("/api/chat_v2_stream", json={"Id": "s3", "Question": "你好"})
            mock_trace_repo.create_trace.assert_called_once()

        assert response.status_code == 200
        body = response.text
        assert '"type": "validation"' in body
        assert '"groundedness_score": 0.4' in body
