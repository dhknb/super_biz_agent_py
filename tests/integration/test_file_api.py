"""文件上传集成测试"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


class TestFileUpload:
    def test_upload_valid_txt(self, client: TestClient) -> None:
        # 文件内容不重要，这个测试关注的是：
        # 1) 路由能接收 multipart 文件
        # 2) 成功时会把索引分片数写回响应
        with patch(
            "app.api.file.vector_index_service", new_callable=MagicMock
        ) as mock_idx:
            mock_idx.index_single_file.return_value = 3
            response = client.post(
                "/api/upload",
                files={"file": ("test.txt", b"hello world", "text/plain")},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 200
            assert data["data"]["filename"] == "test.txt"
            assert data["data"]["indexed_chunks"] == 3

    def test_upload_rejects_unsupported_extension(self, client: TestClient) -> None:
        # 安全边界测试：接口应该在进入索引流程前就拒绝不支持的扩展名。
        response = client.post(
            "/api/upload",
            files={"file": ("virus.exe", b"bad", "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_upload_empty_filename_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/api/upload",
            files={"file": ("", b"content", "text/plain")},
        )
        assert response.status_code == 400

    def test_upload_no_file_returns_422(self, client: TestClient) -> None:
        response = client.post("/api/upload")
        assert response.status_code == 422


class TestFilenameHelpers:
    def test_get_file_extension(self) -> None:
        from app.api.file import _get_file_extension

        # helper 测试覆盖大小写和“无扩展名”两种边界输入。
        assert _get_file_extension("readme.md") == "md"
        assert _get_file_extension("notes.TXT") == "txt"
        assert _get_file_extension("noext") == ""

    def test_sanitize_filename(self) -> None:
        from app.api.file import _sanitize_filename

        result = _sanitize_filename("my file: with? bad\\chars.txt")
        assert " " not in result
        assert ":" not in result
        assert "?" not in result
        assert "\\" not in result
