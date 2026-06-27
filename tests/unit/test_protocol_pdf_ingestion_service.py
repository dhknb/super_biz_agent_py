"""Protocol PDF ingestion service tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.models.protocol_ingestion import ProtocolIngestionStatus
from app.services.protocol_pdf_ingestion_service import (
    ExtractedPdf,
    ProtocolPdfIngestionService,
    STATE_ORDER,
)


class FakeExtractor:
    def extract(self, file_path: str) -> ExtractedPdf:
        return ExtractedPdf(
            text="""
新设备检测协议
设备: 低压配电柜 A1
温度测点 <= 80 ℃
电流阈值 >= 10 A
""".strip(),
            pages=[
                {
                    "page": 1,
                    "text": "新设备检测协议\n设备: 低压配电柜 A1\n温度测点 <= 80 ℃\n电流阈值 >= 10 A",
                }
            ],
        )


class FakeRepo:
    saved = None
    confirmed = None

    def __init__(self, db):
        self.db = db
        self.ingestion = db.ingestion

    def get_ingestion(self, ingestion_id: str):
        return self.ingestion if ingestion_id == self.ingestion.id else None

    def set_status(self, ingestion, status, *, phase=None, error_message=None):
        ingestion.status = status
        ingestion.current_phase = phase
        ingestion.error_message = error_message
        return ingestion

    def save_dry_run(self, ingestion, **kwargs):
        ingestion.extracted_text = kwargs["extracted_text"]
        ingestion.structured_data = kwargs["structured_data"]
        ingestion.validation_result = kwargs["validation_result"]
        ingestion.dry_run_plan = kwargs["dry_run_plan"]
        ingestion.status = ProtocolIngestionStatus.AWAITING_CONFIRMATION
        ingestion.current_phase = "awaiting_confirmation"
        FakeRepo.saved = ingestion
        return ingestion

    def confirm_with_state_trace(self, ingestion, *, confirmed_by, state_trace):
        ingestion.status = ProtocolIngestionStatus.COMPLETED
        ingestion.confirmed_by = confirmed_by
        ingestion.state_trace = state_trace
        FakeRepo.confirmed = ingestion
        return ingestion


@pytest.fixture
def fake_db():
    return SimpleNamespace(
        ingestion=SimpleNamespace(
            id="ing-1",
            filename="protocol.pdf",
            file_path="/tmp/protocol.pdf",
            status=ProtocolIngestionStatus.PENDING,
            dry_run_plan=None,
            validation_result=None,
            state_trace=[],
        )
    )


def test_process_ingestion_builds_reviewable_dry_run(fake_db):
    service = ProtocolPdfIngestionService(extractor=FakeExtractor())

    with patch(
        "app.services.protocol_pdf_ingestion_service.ProtocolIngestionRepository",
        FakeRepo,
    ):
        ingestion = service.process_ingestion(fake_db, "ing-1")

    assert ingestion.status == ProtocolIngestionStatus.AWAITING_CONFIRMATION
    assert ingestion.current_phase == "awaiting_confirmation"
    assert ingestion.structured_data["protocol"]["name"] == "新设备检测协议"
    assert ingestion.structured_data["devices"][0]["name"] == "低压配电柜 A1"
    assert ingestion.validation_result["valid"] is True
    assert ingestion.dry_run_plan["state_order"] == STATE_ORDER
    assert ingestion.dry_run_plan["operation_count"] >= 4


def test_confirm_ingestion_records_four_ordered_states(fake_db):
    service = ProtocolPdfIngestionService(extractor=FakeExtractor())
    fake_db.ingestion.status = ProtocolIngestionStatus.AWAITING_CONFIRMATION
    fake_db.ingestion.validation_result = {"valid": True, "errors": [], "warnings": []}
    fake_db.ingestion.dry_run_plan = {
        "operations": [
            {"phase": "protocol_saved", "action": "upsert_protocol"},
            {"phase": "equipment_mapped", "action": "map_or_create_device"},
            {"phase": "points_saved", "action": "upsert_detection_point"},
            {"phase": "thresholds_saved", "action": "upsert_threshold_rule"},
        ]
    }

    with patch(
        "app.services.protocol_pdf_ingestion_service.ProtocolIngestionRepository",
        FakeRepo,
    ):
        ingestion = service.confirm_ingestion(fake_db, "ing-1", confirmed_by="qa")

    assert ingestion.status == ProtocolIngestionStatus.COMPLETED
    assert ingestion.confirmed_by == "qa"
    assert [item["phase"] for item in ingestion.state_trace] == STATE_ORDER
    assert all(item["status"] == "recorded" for item in ingestion.state_trace)


def test_confirm_rejects_validation_errors(fake_db):
    service = ProtocolPdfIngestionService(extractor=FakeExtractor())
    fake_db.ingestion.status = ProtocolIngestionStatus.AWAITING_CONFIRMATION
    fake_db.ingestion.validation_result = {"valid": False, "errors": [{"field": "x"}]}
    fake_db.ingestion.dry_run_plan = {"operations": []}

    with patch(
        "app.services.protocol_pdf_ingestion_service.ProtocolIngestionRepository",
        FakeRepo,
    ):
        with pytest.raises(ValueError, match="校验仍存在错误"):
            service.confirm_ingestion(fake_db, "ing-1", confirmed_by="qa")
