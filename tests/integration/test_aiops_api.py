"""AIOps 诊断接口集成测试

mock 掉 aiops_service 的 diagnose 生成器。
"""

import json
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


class TestAIOpsDiagnose:
    def test_diagnose_streams_events(self, client: TestClient) -> None:
        async def fake_events(session_id: str):  # type: ignore[no-untyped-def]
            yield {"type": "status", "stage": "start"}
            yield {"type": "plan", "plan": ["step1"]}
            yield {"type": "complete"}

        mock_svc = AsyncMock()
        mock_svc.diagnose = fake_events

        with patch("app.api.aiops.aiops_service", mock_svc):
            response = client.post(
                "/api/aiops",
                json={"session_id": "test-session"},
            )
            assert response.status_code == 200
            body = response.text
            # SSE should contain our event types
            assert "status" in body or "plan" in body

    def test_diagnose_handles_exception(self, client: TestClient) -> None:
        async def broken(session_id: str):  # type: ignore[no-untyped-def]
            raise RuntimeError("cannot diagnose")
            yield  # unreachable

        mock_svc = AsyncMock()
        mock_svc.diagnose = broken

        with patch("app.api.aiops.aiops_service", mock_svc):
            response = client.post(
                "/api/aiops",
                json={"session_id": "test-session"},
            )
            # The endpoint catches the exception and yields an error event
            assert response.status_code == 200
            body = response.text
            assert "error" in body.lower() or "exception" in body.lower()
