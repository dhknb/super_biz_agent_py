"""健康检查集成测试

用 TestClient，Milvus 全 mock，不连真实服务。
"""

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        # 健康检查是系统最基础的契约，先验证最粗粒度的“接口活着”。
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200

    def test_health_returns_service_info(self, client: TestClient) -> None:
        # 再验证 payload 结构，避免后续改接口时悄悄删字段。
        response = client.get("/health")
        data = response.json()
        assert "data" in data
        assert data["data"]["service"] == "SuperBizAgent"
        assert "version" in data["data"]

    def test_root_returns_200(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200

    def test_docs_accessible(self, client: TestClient) -> None:
        response = client.get("/docs")
        assert response.status_code == 200
