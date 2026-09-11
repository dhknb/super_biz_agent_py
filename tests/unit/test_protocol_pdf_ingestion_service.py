"""Protocol PDF ingestion service tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.protocol_catalog import (
    ProtocolDetectionPoint,
    ProtocolEquipmentMapping,
    ProtocolRecord,
    ProtocolThresholdRule,
)
from app.models.protocol_ingestion import ProtocolIngestionStatus
from app.repositories.protocol_ingestion_repository import ProtocolIngestionRepository
from app.services.protocol_pdf_ingestion_service import (
    ExtractedPdf,
    PdfTextExtractor,
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

    def write_protocol_catalog(self, ingestion):
        return [
            {
                "phase": phase,
                "status": "recorded",
                "row_count": 1,
                "row_ids": [f"{phase}-1"],
            }
            for phase in STATE_ORDER
        ]

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


def test_build_structured_draft_skips_extractor_page_markers():
    service = ProtocolPdfIngestionService()

    structured = service.build_structured_draft(
        "[page 1]\n真实协议名称\n设备: 测试设备 A\n温度测点 <= 80 ℃",
        pages=[
            {
                "page": 1,
                "text": "真实协议名称\n设备: 测试设备 A\n温度测点 <= 80 ℃",
            }
        ],
        filename="real.pdf",
    )

    assert structured["protocol"]["name"] == "真实协议名称"


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


def test_confirm_ingestion_writes_protocol_catalog_rows():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    db = session_factory()
    try:
        repo = ProtocolIngestionRepository(db)
        ingestion = repo.create_ingestion(
            filename="protocol.pdf",
            original_filename="protocol.pdf",
            file_path="/tmp/protocol.pdf",
            file_size=123,
            content_hash="hash",
        )
        ingestion.status = ProtocolIngestionStatus.AWAITING_CONFIRMATION
        ingestion.validation_result = {"valid": True, "errors": [], "warnings": []}
        ingestion.structured_data = {
            "protocol": {"name": "新设备检测协议", "source_filename": "protocol.pdf"},
            "devices": [{"name": "低压配电柜 A1", "source_excerpt": "设备: 低压配电柜 A1"}],
            "detection_items": [
                {
                    "name": "温度测点",
                    "measurement_point": "温度测点",
                    "threshold": {
                        "operator": "<=",
                        "value": 80.0,
                        "unit": "℃",
                        "raw": "温度测点 <= 80 ℃",
                    },
                    "source": {"page": 1, "excerpt": "温度测点 <= 80 ℃"},
                    "confidence": 0.55,
                }
            ],
        }
        ingestion.dry_run_plan = ProtocolPdfIngestionService().build_dry_run_plan(
            ingestion.structured_data,
            ingestion.validation_result,
        )
        db.commit()

        service = ProtocolPdfIngestionService(extractor=FakeExtractor())
        confirmed = service.confirm_ingestion(db, ingestion.id, confirmed_by="qa")

        protocol = db.scalar(select(ProtocolRecord).where(ProtocolRecord.name == "新设备检测协议"))
        equipment = db.scalar(
            select(ProtocolEquipmentMapping).where(
                ProtocolEquipmentMapping.equipment_name == "低压配电柜 A1"
            )
        )
        point = db.scalar(
            select(ProtocolDetectionPoint).where(
                ProtocolDetectionPoint.measurement_point == "温度测点"
            )
        )
        threshold = db.scalar(
            select(ProtocolThresholdRule).where(
                ProtocolThresholdRule.measurement_point == "温度测点"
            )
        )

        assert confirmed.status == ProtocolIngestionStatus.COMPLETED
        assert [item["phase"] for item in confirmed.state_trace] == STATE_ORDER
        assert protocol is not None
        assert equipment is not None
        assert equipment.protocol_id == protocol.id
        assert point is not None
        assert point.protocol_id == protocol.id
        assert threshold is not None
        assert threshold.protocol_id == protocol.id
        assert threshold.detection_point_id == point.id
        assert threshold.operator == "<="
        assert threshold.value == 80.0
    finally:
        db.close()
        engine.dispose()


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


def test_pymupdf_extractor_returns_page_text(monkeypatch, tmp_path):
    pdf_path = tmp_path / "demo.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock")

    class FakePage:
        def __init__(self, text: str):
            self._text = text

        def get_text(self, _mode: str) -> str:
            return self._text

    class FakeDoc:
        def __init__(self, pages):
            self._pages = pages

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter(self._pages)

    fake_fitz = MagicMock()
    fake_fitz.open.return_value = FakeDoc(
        [
            FakePage("第一页内容"),
            FakePage("第二页内容"),
        ]
    )
    monkeypatch.setitem(__import__("sys").modules, "fitz", fake_fitz)

    extractor = PdfTextExtractor()
    extracted = extractor.extract(str(pdf_path))

    assert extracted.pages == [
        {"page": 1, "text": "第一页内容"},
        {"page": 2, "text": "第二页内容"},
    ]
    assert "[page 1]\n第一页内容" in extracted.text
    assert "[page 2]\n第二页内容" in extracted.text
