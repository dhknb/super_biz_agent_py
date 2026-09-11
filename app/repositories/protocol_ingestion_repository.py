"""Repository for protocol PDF ingestion state."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.protocol_catalog import (
    ProtocolDetectionPoint,
    ProtocolEquipmentMapping,
    ProtocolRecord,
    ProtocolThresholdRule,
)
from app.models.protocol_ingestion import (
    ProtocolIngestionJobStatus,
    ProtocolIngestionStatus,
    ProtocolPdfIngestion,
    ProtocolPdfIngestionJob,
)


class ProtocolIngestionRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_ingestion(
        self,
        *,
        filename: str,
        original_filename: str,
        file_path: str,
        file_size: int,
        content_hash: str,
    ) -> ProtocolPdfIngestion:
        ingestion = ProtocolPdfIngestion(
            filename=filename,
            original_filename=original_filename,
            file_path=file_path,
            file_size=file_size,
            content_hash=content_hash,
            status=ProtocolIngestionStatus.PENDING,
        )
        self.db.add(ingestion)
        self.db.commit()
        self.db.refresh(ingestion)
        return ingestion

    def create_job(self, ingestion_id: str) -> ProtocolPdfIngestionJob:
        job = ProtocolPdfIngestionJob(
            ingestion_id=ingestion_id,
            status=ProtocolIngestionJobStatus.QUEUED,
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def bind_rq_job(self, job_id: str, rq_job_id: str) -> None:
        job = self.get_job(job_id)
        if job is None:
            return
        job.rq_job_id = rq_job_id
        self.db.commit()

    def get_ingestion(self, ingestion_id: str) -> ProtocolPdfIngestion | None:
        return self.db.get(ProtocolPdfIngestion, ingestion_id)

    def get_job(self, job_id: str) -> ProtocolPdfIngestionJob | None:
        return self.db.get(ProtocolPdfIngestionJob, job_id)

    def list_ingestions(self) -> list[ProtocolPdfIngestion]:
        stmt = select(ProtocolPdfIngestion).order_by(ProtocolPdfIngestion.created_at.desc())
        return list(self.db.scalars(stmt).all())#scalars(...) 的意思是只拿模型对象本身。

    def set_status(
        self,
        ingestion: ProtocolPdfIngestion,
        status: ProtocolIngestionStatus,
        *,
        phase: str | None = None,
        error_message: str | None = None,
    ) -> ProtocolPdfIngestion:
        ingestion.status = status
        ingestion.current_phase = phase
        ingestion.error_message = error_message
        self.db.commit()
        self.db.refresh(ingestion)
        return ingestion

    def save_dry_run(#它是在 PDF 已经解析完，但还没有正式入协议库之前调用的。
        self,
        ingestion: ProtocolPdfIngestion,
        *,
        extracted_text: str,
        structured_data: dict[str, Any],
        validation_result: dict[str, Any],
        dry_run_plan: dict[str, Any],
    ) -> ProtocolPdfIngestion:
        ingestion.extracted_text = extracted_text
        ingestion.structured_data = structured_data
        ingestion.validation_result = validation_result
        ingestion.dry_run_plan = dry_run_plan
        ingestion.status = ProtocolIngestionStatus.AWAITING_CONFIRMATION
        ingestion.current_phase = "awaiting_confirmation"
        ingestion.error_message = None
        self.db.commit()
        self.db.refresh(ingestion)
        return ingestion

    def confirm_with_state_trace(
        self,
        ingestion: ProtocolPdfIngestion,
        *,
        confirmed_by: str,
        state_trace: list[dict[str, Any]],
    ) -> ProtocolPdfIngestion:
        ingestion.status = ProtocolIngestionStatus.COMPLETED
        ingestion.current_phase = "thresholds_saved"
        ingestion.confirmed_by = confirmed_by
        ingestion.confirmed_at = datetime.now(UTC).replace(tzinfo=None)
        ingestion.state_trace = state_trace
        ingestion.error_message = None
        self.db.commit()
        self.db.refresh(ingestion)
        return ingestion

    def write_protocol_catalog(self, ingestion: ProtocolPdfIngestion) -> list[dict[str, Any]]:
        structured_data = ingestion.structured_data or {}
        protocol_payload = structured_data.get("protocol") or {}
        protocol_name = protocol_payload.get("name")
        if not protocol_name:
            raise ValueError("协议名称不能为空，不能写入协议库")

        protocol = self._upsert_protocol_record(
            ingestion=ingestion,
            protocol_name=protocol_name,
            protocol_payload=protocol_payload,
            structured_data=structured_data,
        )
        self.db.flush()

        equipment_ids = self._upsert_equipment_mappings(
            protocol_id=protocol.id,
            devices=structured_data.get("devices") or [],
        )
        point_by_measurement = self._upsert_detection_points(
            protocol_id=protocol.id,
            detection_items=structured_data.get("detection_items") or [],
        )
        threshold_ids = self._upsert_threshold_rules(
            protocol_id=protocol.id,
            detection_items=structured_data.get("detection_items") or [],
            point_by_measurement=point_by_measurement,
        )
        self.db.flush()

        now = datetime.now(UTC).replace(tzinfo=None).isoformat()
        return [
            {
                "phase": "protocol_saved",
                "status": "recorded",
                "row_count": 1,
                "row_ids": [protocol.id],
                "at": now,
            },
            {
                "phase": "equipment_mapped",
                "status": "recorded",
                "row_count": len(equipment_ids),
                "row_ids": equipment_ids,
                "at": now,
            },
            {
                "phase": "points_saved",
                "status": "recorded",
                "row_count": len(point_by_measurement),
                "row_ids": [point.id for point in point_by_measurement.values()],
                "at": now,
            },
            {
                "phase": "thresholds_saved",
                "status": "recorded",
                "row_count": len(threshold_ids),
                "row_ids": threshold_ids,
                "at": now,
            },
        ]

    def reject(
        self,
        ingestion: ProtocolPdfIngestion,
        *,
        rejected_by: str,
        reason: str | None = None,
    ) -> ProtocolPdfIngestion:
        trace = list(ingestion.state_trace or [])
        trace.append(
            {
                "phase": "rejected",
                "status": "rejected",
                "actor": rejected_by,
                "reason": reason,
                "at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
            }
        )
        ingestion.status = ProtocolIngestionStatus.REJECTED
        ingestion.current_phase = "rejected"
        ingestion.state_trace = trace
        ingestion.error_message = reason
        self.db.commit()
        self.db.refresh(ingestion)
        return ingestion

    def _upsert_protocol_record(
        self,
        *,
        ingestion: ProtocolPdfIngestion,
        protocol_name: str,
        protocol_payload: dict[str, Any],
        structured_data: dict[str, Any],
    ) -> ProtocolRecord:
        protocol = self.db.scalar(
            select(ProtocolRecord).where(ProtocolRecord.name == protocol_name)
        )
        if protocol is None:
            protocol = ProtocolRecord(name=protocol_name)
            self.db.add(protocol)

        protocol.source_filename = protocol_payload.get("source_filename") or ingestion.filename
        protocol.source_ingestion_id = ingestion.id
        protocol.schema_version = structured_data.get("schema_version") or ingestion.schema_version
        protocol.raw_payload = protocol_payload
        return protocol

    def _upsert_equipment_mappings(
        self,
        *,
        protocol_id: str,
        devices: list[dict[str, Any]],
    ) -> list[str]:
        row_ids: list[str] = []
        for device in devices:
            equipment_name = device.get("name")
            if not equipment_name:
                continue
            mapping = self.db.scalar(
                select(ProtocolEquipmentMapping).where(
                    ProtocolEquipmentMapping.protocol_id == protocol_id,
                    ProtocolEquipmentMapping.equipment_name == equipment_name,
                )
            )
            if mapping is None:
                mapping = ProtocolEquipmentMapping(
                    protocol_id=protocol_id,
                    equipment_name=equipment_name,
                )
                self.db.add(mapping)
                self.db.flush()

            mapping.source_excerpt = device.get("source_excerpt")
            mapping.raw_payload = device
            row_ids.append(mapping.id)
        return row_ids

    def _upsert_detection_points(
        self,
        *,
        protocol_id: str,
        detection_items: list[dict[str, Any]],
    ) -> dict[str, ProtocolDetectionPoint]:
        points: dict[str, ProtocolDetectionPoint] = {}
        for item in detection_items:
            measurement_point = item.get("measurement_point") or item.get("name")
            if not measurement_point:
                continue
            point = self.db.scalar(
                select(ProtocolDetectionPoint).where(
                    ProtocolDetectionPoint.protocol_id == protocol_id,
                    ProtocolDetectionPoint.measurement_point == measurement_point,
                )
            )
            if point is None:
                point = ProtocolDetectionPoint(
                    protocol_id=protocol_id,
                    measurement_point=measurement_point,
                    name=item.get("name") or measurement_point,
                )
                self.db.add(point)
                self.db.flush()

            point.name = item.get("name") or measurement_point
            point.source = item.get("source") or {}
            point.confidence = item.get("confidence")
            point.raw_payload = item
            points[measurement_point] = point
        return points

    def _upsert_threshold_rules(
        self,
        *,
        protocol_id: str,
        detection_items: list[dict[str, Any]],
        point_by_measurement: dict[str, ProtocolDetectionPoint],
    ) -> list[str]:
        row_ids: list[str] = []
        for item in detection_items:
            measurement_point = item.get("measurement_point") or item.get("name")
            threshold = item.get("threshold") or {}
            if not measurement_point or not threshold:
                continue
            rule = self.db.scalar(
                select(ProtocolThresholdRule).where(
                    ProtocolThresholdRule.protocol_id == protocol_id,
                    ProtocolThresholdRule.measurement_point == measurement_point,
                )
            )
            if rule is None:
                rule = ProtocolThresholdRule(
                    protocol_id=protocol_id,
                    measurement_point=measurement_point,
                    operator=threshold.get("operator") or threshold.get("op") or "=",
                    value=float(threshold.get("value", 0)),
                )
                self.db.add(rule)
                self.db.flush()

            point = point_by_measurement.get(measurement_point)
            rule.detection_point_id = point.id if point else None
            rule.operator = threshold.get("operator") or threshold.get("op") or "="
            rule.value = float(threshold.get("value", 0))
            rule.unit = threshold.get("unit") or ""
            rule.raw = threshold.get("raw")
            rule.source = item.get("source") or {}
            rule.raw_payload = threshold
            row_ids.append(rule.id)
        return row_ids
