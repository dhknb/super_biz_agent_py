"""文件上传集成测试"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


class FakeDocument:
    id = "doc-1"
    filename = "test.txt"
    original_filename = "test.txt"
    status = MagicMock(value="pending")
    version = 1
    file_size = 11
    content_hash = "hash"
    error_message = None


class FakeJob:
    id = "job-1"
    rq_job_id = "rq-1"


class FakeRqJob:
    id = "rq-1"


class FakeRepository:
    def __init__(self, db):
        self.db = db

    def create_document(self, **kwargs):
        doc = FakeDocument()
        doc.filename = kwargs["filename"]
        doc.original_filename = kwargs["original_filename"]
        doc.file_size = kwargs["file_size"]
        doc.content_hash = kwargs["content_hash"]
        return doc

    def create_index_job(self, document_id: str):
        return FakeJob()

    def bind_rq_job(self, job_id: str, rq_job_id: str) -> None:
        return None


class TestFileUpload:
    def test_upload_valid_txt(self, client: TestClient) -> None:
        # 文件上传现在是异步索引：接口只创建 document/job 并投递 RQ。
        with (
            patch("app.api.file.KnowledgeRepository", FakeRepository),
            patch("app.api.file.enqueue_index_job", return_value=FakeRqJob()),
        ):
            response = client.post(
                "/api/upload",
                files={"file": ("test.txt", b"hello world", "text/plain")},
            )
            assert response.status_code == 202
            data = response.json()
            assert data["code"] == 202
            assert data["data"]["filename"] == "test.txt"
            assert data["data"]["document_id"] == "doc-1"
            assert data["data"]["job_id"] == "job-1"
            assert data["data"]["rq_job_id"] == "rq-1"

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
        assert response.status_code in (400, 422)

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
