"""对话接口集成测试

mock RagAgentService，验证路由行为和响应格式。
"""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


class TestChatEndpoint:
    def test_chat_returns_200_with_answer(self, client: TestClient) -> None:
        # 路由测试只关心 HTTP 层行为，所以把底层 rag_agent_service 整体替换成 mock。
        with patch(
            "app.api.chat.rag_agent_service", new_callable=AsyncMock
        ) as mock_svc:
            mock_svc.query = AsyncMock(return_value="这是测试回复")
            response = client.post(
                "/api/chat",
                json={"Id": "s1", "Question": "你好"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 200
            assert data["data"]["success"] is True
            assert data["data"]["answer"] == "这是测试回复"

    def test_chat_error_returns_500(self, client: TestClient) -> None:
        # 这里故意让 service 抛错，验证 endpoint 会把异常包装成统一响应结构。
        with patch(
            "app.api.chat.rag_agent_service", new_callable=AsyncMock
        ) as mock_svc:
            mock_svc.query = AsyncMock(side_effect=RuntimeError("boom"))
            response = client.post(
                "/api/chat",
                json={"Id": "s2", "Question": "任何问题"},
            )
            assert response.status_code == 200  # endpoint catches, doesn't raise HTTP
            data = response.json()
            assert data["code"] == 500
            assert data["data"]["success"] is False

    def test_chat_missing_required_fields_returns_422(self, client: TestClient) -> None:
        response = client.post("/api/chat", json={})
        assert response.status_code == 422


class TestClearSession:
    def test_clear_returns_success(self, client: TestClient) -> None:
        # clear 接口不需要真实会话存储；只验证路由是否正确调用 service 并返回成功态。
        with patch(
            "app.api.chat.rag_agent_service", new_callable=AsyncMock
        ) as mock_svc:
            mock_svc.clear_session.return_value = True
            response = client.post(
                "/api/chat/clear",
                json={"sessionId": "s1"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"


class TestGetSessionInfo:
    def test_session_info_returns_history(self, client: TestClient) -> None:
        # 返回一条假的历史消息，确认接口能正确统计 message_count 并透传 session_id。
        with patch(
            "app.api.chat.rag_agent_service", new_callable=AsyncMock
        ) as mock_svc:
            mock_svc.get_session_history.return_value = [
                {"role": "user", "content": "hi"}
            ]
            response = client.get("/api/chat/session/test-id")
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "test-id"
            assert data["message_count"] == 1
