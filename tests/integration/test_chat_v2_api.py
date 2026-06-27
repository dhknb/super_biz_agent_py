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
            "app.api.chat_v2.ConversationRepository"
        ) as mock_repo_cls:
            mock_svc.query = AsyncMock(return_value=mock_result)
            mock_repo = MagicMock()
            mock_repo_cls.return_value = mock_repo
            response = client.post("/api/chat_v2", json={"Id": "s1", "Question": "hello"})
            mock_repo.append_exchange.assert_called_once_with(
                "s1",
                user_content="hello",
                assistant_content="test reply",
                message_metadata={"source": "chat_v2"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["data"]["success"] is True
        assert data["data"]["answer"] == "test reply"
        assert data["data"]["validation"]["groundedness_score"] == 0.88

    def test_chat_v2_error_returns_500(self, client) -> None:
        with patch("app.api.chat_v2.rag_v2_service") as mock_svc:
            mock_svc.query = AsyncMock(side_effect=RuntimeError("boom"))
            response = client.post("/api/chat_v2", json={"Id": "s2", "Question": "任何问题"})

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 500
        assert data["data"]["success"] is False


class TestChatV2StreamEndpoint:
    def test_chat_v2_stream_emits_validation_event(self, client) -> None:
        async def fake_stream(*_args, **_kwargs):
            yield {"type": "sub_queries", "data": ["q1"]}
            yield {"type": "validation", "data": {"blocked": True, "groundedness_score": 0.4}}
            yield {"type": "complete"}

        with patch("app.api.chat_v2.rag_v2_service") as mock_svc:
            mock_svc.query_stream = fake_stream
            response = client.post("/api/chat_v2_stream", json={"Id": "s3", "Question": "你好"})

        assert response.status_code == 200
        body = response.text
        assert '"type": "validation"' in body
        assert '"groundedness_score": 0.4' in body
