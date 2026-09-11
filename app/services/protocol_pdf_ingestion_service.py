"""Protocol PDF ingestion service."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from app.models.protocol_ingestion import ProtocolIngestionStatus, ProtocolPdfIngestion
from app.repositories.protocol_ingestion_repository import ProtocolIngestionRepository


STATE_ORDER = [
    "protocol_saved",
    "equipment_mapped",
    "points_saved",
    "thresholds_saved",
]


@dataclass(frozen=True)
class ExtractedPdf:
    text: str
    pages: list[dict[str, Any]]


class PdfTextExtractor:
    def extract(self, file_path: str) -> ExtractedPdf:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("缺少 PyMuPDF(fitz) 依赖，无法抽取 PDF 文本") from exc

        pages: list[dict[str, Any]] = []
        with fitz.open(file_path) as doc:
            for index, page in enumerate(doc, start=1):
                page_text = page.get_text("text") or ""
                pages.append({"page": index, "text": page_text})

        text = "\n\n".join(
            f"[page {page['page']}]\n{page['text']}" for page in pages if page["text"].strip()
        )
        return ExtractedPdf(text=text, pages=pages)


class ProtocolPdfIngestionService:
    def __init__(self, extractor: PdfTextExtractor | None = None):
        self.extractor = extractor or PdfTextExtractor()

    def process_ingestion(self, db: Session, ingestion_id: str) -> ProtocolPdfIngestion:
        repo = ProtocolIngestionRepository(db)
        ingestion = repo.get_ingestion(ingestion_id)
        if ingestion is None:
            raise ValueError("protocol PDF ingestion not found")

        repo.set_status(ingestion, ProtocolIngestionStatus.EXTRACTING, phase="extract_pdf")
        extracted = self.extractor.extract(ingestion.file_path)

        repo.set_status(
            ingestion,
            ProtocolIngestionStatus.STRUCTURING,
            phase="structure_with_qwen_schema",
        )
        structured = self.build_structured_draft(
            extracted.text,
            pages=extracted.pages,
            filename=ingestion.filename,
        )

        repo.set_status(
            ingestion,
            ProtocolIngestionStatus.VALIDATING,
            phase="validate_schema_and_rules",
        )
        validation = self.validate_structured_data(structured)
        dry_run_plan = self.build_dry_run_plan(structured, validation)

        return repo.save_dry_run(
            ingestion,
            extracted_text=extracted.text,
            structured_data=structured,
            validation_result=validation,
            dry_run_plan=dry_run_plan,
        )

    def confirm_ingestion(
        self,
        db: Session,
        ingestion_id: str,
        *,
        confirmed_by: str,
    ) -> ProtocolPdfIngestion:
        repo = ProtocolIngestionRepository(db)
        ingestion = repo.get_ingestion(ingestion_id)
        if ingestion is None:
            raise ValueError("protocol PDF ingestion not found")

        if ingestion.status != ProtocolIngestionStatus.AWAITING_CONFIRMATION:
            raise ValueError("协议 PDF 入库任务尚未进入人工确认状态")
        if not ingestion.dry_run_plan:
            raise ValueError("缺少 dry-run 入库计划")

        validation = ingestion.validation_result or {}
        if validation.get("errors"):
            raise ValueError("校验仍存在错误，不能确认写入")

        repo.set_status(ingestion, ProtocolIngestionStatus.WRITING, phase="protocol_saved")
        state_trace = repo.write_protocol_catalog(ingestion)
        operations = ingestion.dry_run_plan.get("operations", [])
        for state in state_trace:
            phase_operations = [
                operation for operation in operations if operation.get("phase") == state["phase"]
            ]
            state["operation_count"] = len(phase_operations)
            state["operations"] = phase_operations

        logger.info(f"协议 PDF 入库确认完成: ingestion_id={ingestion_id}")
        return repo.confirm_with_state_trace(
            ingestion,
            confirmed_by=confirmed_by,
            state_trace=state_trace,
        )

    def build_structured_draft(
        self,
        text: str,
        *,
        pages: list[dict[str, Any]],
        filename: str,
    ) -> dict[str, Any]:
        lines = [
            line.strip()#去掉首尾空行跟换行符
            for line in text.splitlines()
            if line.strip() and not re.fullmatch(r"\[page\s+\d+\]", line.strip(), re.IGNORECASE)
        ]
        protocol_name = lines[0] if lines else Path(filename).stem
        devices = self._extract_devices(lines)
        detection_items = self._extract_detection_items(lines, pages)

        return {
            "schema_version": "protocol_pdf_v1",
            "protocol": {
                "name": protocol_name,
                "source_filename": filename,
            },
            "devices": devices,
            "detection_items": detection_items,
        }

    def validate_structured_data(self, structured_data: dict[str, Any]) -> dict[str, Any]:
        errors: list[dict[str, str]] = []
        warnings: list[dict[str, str]] = []

        protocol = structured_data.get("protocol") or {}
        if not protocol.get("name"):
            errors.append({"field": "protocol.name", "message": "协议名称不能为空"})

        if not structured_data.get("devices"):
            warnings.append({"field": "devices", "message": "未识别到设备/型号，需要人工确认"})

        items = structured_data.get("detection_items") or []
        if not items:
            warnings.append({"field": "detection_items", "message": "未识别到检测项，需要人工补录"})

        seen_points: set[str] = set()
        for index, item in enumerate(items):
            point = item.get("measurement_point") or item.get("name")
            if not point:
                errors.append(
                    {
                        "field": f"detection_items[{index}].measurement_point",
                        "message": "测点不能为空",
                    }
                )
                continue
            if point in seen_points:
                warnings.append(
                    {
                        "field": f"detection_items[{index}].measurement_point",
                        "message": f"测点重复: {point}",
                    }
                )
            seen_points.add(point)

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "needs_review": bool(errors or warnings),
        }

    def build_dry_run_plan(
        self,
        structured_data: dict[str, Any],
        validation_result: dict[str, Any],
    ) -> dict[str, Any]:
        operations: list[dict[str, Any]] = [
            {
                "phase": "protocol_saved",
                "action": "upsert_protocol",
                "target": structured_data.get("protocol", {}),
            }
        ]

        for device in structured_data.get("devices") or []:
            operations.append(
                {
                    "phase": "equipment_mapped",
                    "action": "map_or_create_device",
                    "target": device,
                }
            )

        for item in structured_data.get("detection_items") or []:
            operations.append(
                {
                    "phase": "points_saved",
                    "action": "upsert_detection_point",
                    "target": {
                        "name": item.get("name"),
                        "measurement_point": item.get("measurement_point"),
                        "source": item.get("source"),
                    },
                }
            )
            operations.append(
                {
                    "phase": "thresholds_saved",
                    "action": "upsert_threshold_rule",
                    "target": {
                        "measurement_point": item.get("measurement_point"),
                        "threshold": item.get("threshold"),
                        "source": item.get("source"),
                    },
                }
            )

        return {
            "state_order": STATE_ORDER,
            "can_confirm": validation_result.get("valid", False),
            "requires_human_review": validation_result.get("needs_review", True),
            "operations": operations,
            "operation_count": len(operations),
        }

    def _extract_devices(self, lines: list[str]) -> list[dict[str, Any]]:
        devices: list[dict[str, Any]] = []
        labeled_pattern = re.compile(r"^(适用)?(设备|型号|装置)[:：\s]+(?P<name>.+)$")
        for line in lines:
            match = labeled_pattern.search(line)
            if not match:
                continue
            value = match.group("name").strip()
            if value and value not in {item["name"] for item in devices}:
                devices.append({"name": value, "source_excerpt": line})
        return devices[:10]

    def _extract_detection_items(
        self,
        lines: list[str],
        pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        threshold_pattern = re.compile(
            r"(?P<name>[\w\u4e00-\u9fff（）()/-]{2,30}).{0,12}?"
            r"(?P<op>>=|<=|≤|≥|<|>|=)"
            r"\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>[%℃ΩA-Za-z/]+)?"
        )
        for line in lines:
            match = threshold_pattern.search(line)
            if not match:
                continue
            name = match.group("name").strip(" ：:")
            page = self._find_source_page(line, pages)
            items.append(
                {
                    "name": name,
                    "measurement_point": name,
                    "threshold": {
                        "operator": match.group("op"),
                        "value": float(match.group("value")),
                        "unit": match.group("unit") or "",
                        "raw": match.group(0),
                    },
                    "source": {
                        "page": page,
                        "excerpt": line[:240],
                    },
                    "confidence": 0.55,
                    "needs_review": True,
                }
            )
        return items[:100]

    def _find_source_page(self, line: str, pages: list[dict[str, Any]]) -> int | None:
        for page in pages:
            if line in page.get("text", ""):
                return int(page["page"])
        return None


protocol_pdf_ingestion_service = ProtocolPdfIngestionService()
