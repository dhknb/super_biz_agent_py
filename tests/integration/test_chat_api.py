"""对话接口集成测试

mock RagAgentService，验证路由行为和响应格式。
"""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


class TestChatEndpoint:
    def test_chat_returns_200_with_answer(self, client: TestClient) -> None:
        # Route tests mock the service and repository; persistence is covered separately.
        with patch("app.api.chat.rag_agent_service") as mock_svc, patch(
            "app.api.chat.ConversationRepository"
        ) as mock_repo_cls:
            mock_svc.query = AsyncMock(return_value="test reply")
            mock_repo = MagicMock()
            mock_repo_cls.return_value = mock_repo
            response = client.post(
                "/api/chat",
                json={"Id": "s1", "Question": "hello"},
            )
            mock_repo.append_exchange.assert_called_once_with(
                "s1",
                user_content="hello",
                assistant_content="test reply",
            )
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 200
            assert data["data"]["success"] is True
            assert data["data"]["answer"] == "test reply"

    def test_chat_error_returns_real_http_500(self, client_no_raise: TestClient) -> None:
        """失败时 HTTP 状态码必须如实是 500。

        `/chat` 原来的 except 块除了拼响应体没有任何副作用，所以整块删掉了，
        由全局处理器统一构造错误响应（DRY）。响应体结构不变，另外附上
        error_code / degrade_reason / request_id。
        """
        with patch(
            "app.api.chat.rag_agent_service", new_callable=AsyncMock
        ) as mock_svc:
            mock_svc.query = AsyncMock(side_effect=RuntimeError("boom"))
            response = client_no_raise.post(
                "/api/chat",
                json={"Id": "s2", "Question": "任何问题"},
            )
            assert response.status_code == 500
            data = response.json()
            assert data["code"] == 500
            assert data["data"]["success"] is False
            assert data["data"]["answer"] is None
            assert "boom" in data["data"]["errorMessage"]
            assert data["data"]["error_code"] == "runtime_error"
            assert data["data"]["request_id"]

    def test_chat_missing_required_fields_returns_422(self, client: TestClient) -> None:
        response = client.post("/api/chat", json={})
        assert response.status_code == 422


class TestClearSession:
    def test_clear_returns_success(self, client: TestClient) -> None:
        # clear 接口不需要真实会话存储；只验证路由是否正确调用 service 并返回成功态。
        with patch("app.api.chat.rag_agent_service") as mock_svc, patch(
            "app.api.chat.ConversationRepository"
        ) as mock_repo_cls:
            mock_svc.clear_session.return_value = True
            mock_repo = MagicMock()
            mock_repo_cls.return_value = mock_repo
            response = client.post(
                "/api/chat/clear",
                json={"sessionId": "s1"},
            )
            mock_repo.clear_session.assert_called_once_with("s1")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"


class TestGetSessionInfo:
    def test_session_info_returns_history(self, client: TestClient) -> None:
        # 返回一条假的历史消息，确认接口能正确统计 message_count 并透传 session_id。
        with patch("app.api.chat.rag_agent_service") as mock_svc, patch(
            "app.api.chat.ConversationRepository"
        ) as mock_repo_cls:
            mock_svc.get_session_history.return_value = [
                {"role": "user", "content": "hi"}
            ]
            mock_repo = MagicMock()
            mock_repo.list_session_history.return_value = []
            mock_repo_cls.return_value = mock_repo
            response = client.get("/api/chat/session/test-id")
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "test-id"
            assert data["message_count"] == 1
