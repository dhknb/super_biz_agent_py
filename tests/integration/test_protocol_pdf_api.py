"""Protocol PDF ingestion API tests."""

from datetime import datetime
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.models.protocol_ingestion import ProtocolIngestionStatus


class FakeIngestion:
    def __init__(self):
        self.id = "ing-1"
        self.filename = "protocol.pdf"
        self.original_filename = "protocol.pdf"
        self.file_path = "/tmp/protocol.pdf"
        self.file_size = 11
        self.content_hash = "hash"
        self.schema_version = "protocol_pdf_v1"
        self.status = ProtocolIngestionStatus.PENDING
        self.current_phase = None
        self.structured_data = None
        self.validation_result = None
        self.dry_run_plan = None
        self.state_trace = []
        self.confirmed_by = None
        self.confirmed_at = None
        self.error_message = None
        self.created_at = datetime(2026, 6, 24, 1, 2, 3)
        self.updated_at = datetime(2026, 6, 24, 1, 2, 3)


class FakeJob:
    id = "job-1"
    rq_job_id = "rq-1"


class FakeRqJob:
    id = "rq-1"


class FakeRepository:
    ingestion = FakeIngestion()

    def __init__(self, db):
        self.db = db

    def create_ingestion(self, **kwargs):
        ingestion = FakeIngestion()
        ingestion.filename = kwargs["filename"]
        ingestion.original_filename = kwargs["original_filename"]
        ingestion.file_size = kwargs["file_size"]
        ingestion.content_hash = kwargs["content_hash"]
        self.__class__.ingestion = ingestion
        return ingestion

    def create_job(self, ingestion_id: str):
        return FakeJob()

    def bind_rq_job(self, job_id: str, rq_job_id: str) -> None:
        return None

    def list_ingestions(self):
        return [self.__class__.ingestion]

    def get_ingestion(self, ingestion_id: str):
        if ingestion_id == self.__class__.ingestion.id:
            return self.__class__.ingestion
        return None

    def reject(self, ingestion, *, rejected_by: str, reason: str | None = None):
        ingestion.status = ProtocolIngestionStatus.REJECTED
        ingestion.current_phase = "rejected"
        ingestion.error_message = reason
        ingestion.state_trace = [{"phase": "rejected", "actor": rejected_by, "reason": reason}]
        return ingestion


class TestProtocolPdfApi:
    def test_upload_protocol_pdf_enqueues_job(self, client: TestClient) -> None:
        with (
            patch("app.api.protocol_pdf.ProtocolIngestionRepository", FakeRepository),
            patch("app.api.protocol_pdf.enqueue_protocol_pdf_ingestion_job", return_value=FakeRqJob()),
        ):
            response = client.post(
                "/api/protocol-pdfs/upload",
                files={"file": ("protocol.pdf", b"%PDF-1.4\n", "application/pdf")},
            )

        assert response.status_code == 202
        data = response.json()
        assert data["code"] == 202
        assert data["data"]["ingestion_id"] == "ing-1"
        assert data["data"]["job_id"] == "job-1"
        assert data["data"]["rq_job_id"] == "rq-1"
        assert data["data"]["filename"].endswith(".pdf")

    def test_upload_protocol_pdf_rejects_non_pdf(self, client: TestClient) -> None:
        response = client.post(
            "/api/protocol-pdfs/upload",
            files={"file": ("protocol.txt", b"text", "text/plain")},
        )

        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    def test_list_protocol_pdf_ingestions(self, client: TestClient) -> None:
        with patch("app.api.protocol_pdf.ProtocolIngestionRepository", FakeRepository):
            response = client.get("/api/protocol-pdfs")

        assert response.status_code == 200
        data = response.json()
        assert data["data"][0]["id"] == "ing-1"
        assert data["data"][0]["status"] == "pending"

    def test_reject_protocol_pdf_ingestion(self, client: TestClient) -> None:
        with patch("app.api.protocol_pdf.ProtocolIngestionRepository", FakeRepository):
            response = client.post(
                "/api/protocol-pdfs/ing-1/reject",
                json={"rejected_by": "qa", "reason": "schema mismatch"},
            )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["status"] == "rejected"
        assert data["error_message"] == "schema mismatch"

    def test_confirm_protocol_pdf_ingestion(self, client: TestClient) -> None:
        confirmed = FakeIngestion()
        confirmed.status = ProtocolIngestionStatus.COMPLETED
        confirmed.current_phase = "thresholds_saved"
        confirmed.confirmed_by = "qa"
        confirmed.confirmed_at = datetime(2026, 6, 24, 2, 0, 0)
        confirmed.state_trace = [
            {"phase": "protocol_saved", "status": "recorded"},
            {"phase": "equipment_mapped", "status": "recorded"},
            {"phase": "points_saved", "status": "recorded"},
            {"phase": "thresholds_saved", "status": "recorded"},
        ]

        with patch(
            "app.api.protocol_pdf.protocol_pdf_ingestion_service.confirm_ingestion",
            return_value=confirmed,
        ) as confirm:
            response = client.post(
                "/api/protocol-pdfs/ing-1/confirm",
                json={"confirmed_by": "qa"},
            )

        assert response.status_code == 200
        confirm.assert_called_once()
        data = response.json()["data"]
        assert data["status"] == "completed"
        assert [item["phase"] for item in data["state_trace"]] == [
            "protocol_saved",
            "equipment_mapped",
            "points_saved",
            "thresholds_saved",
        ]

    def test_get_protocol_pdf_ingestion_detail(self, client: TestClient) -> None:
        detail = FakeIngestion()
        detail.status = ProtocolIngestionStatus.AWAITING_CONFIRMATION
        detail.current_phase = "awaiting_confirmation"
        detail.structured_data = {"protocol": {"name": "示例协议"}}
        detail.validation_result = {"valid": True, "warnings": []}
        detail.dry_run_plan = {"operation_count": 3}
        detail.state_trace = [{"phase": "protocol_saved", "status": "recorded"}]
        FakeRepository.ingestion = detail

        with patch("app.api.protocol_pdf.ProtocolIngestionRepository", FakeRepository):
            response = client.get("/api/protocol-pdfs/ing-1")

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["structured_data"]["protocol"]["name"] == "示例协议"
        assert data["validation_result"]["valid"] is True
        assert data["dry_run_plan"]["operation_count"] == 3
