"""Protocol catalog rows produced by confirmed PDF ingestion."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

def _utcnow_naive() -> datetime:
    """与项目其余模型保持一致：存 naive UTC 时间。"""
    return datetime.now(UTC).replace(tzinfo=None)



class ProtocolRecord(Base):
    __tablename__ = "protocol_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    source_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_ingestion_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("protocol_pdf_ingestions.id"),
        nullable=True,
        index=True,
    )
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False, default="protocol_pdf_v1")
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utcnow_naive,
        onupdate=_utcnow_naive,
    )

    equipment_mappings: Mapped[list["ProtocolEquipmentMapping"]] = relationship(
        back_populates="protocol",
        cascade="all, delete-orphan",
    )
    detection_points: Mapped[list["ProtocolDetectionPoint"]] = relationship(
        back_populates="protocol",
        cascade="all, delete-orphan",
    )
    threshold_rules: Mapped[list["ProtocolThresholdRule"]] = relationship(
        back_populates="protocol",
        cascade="all, delete-orphan",
    )


class ProtocolEquipmentMapping(Base):
    __tablename__ = "protocol_equipment_mappings"
    __table_args__ = (
        UniqueConstraint("protocol_id", "equipment_name", name="uq_protocol_equipment_name"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    protocol_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("protocol_records.id"),
        nullable=False,
        index=True,
    )
    equipment_name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utcnow_naive,
        onupdate=_utcnow_naive,
    )

    protocol: Mapped[ProtocolRecord] = relationship(back_populates="equipment_mappings")


class ProtocolDetectionPoint(Base):
    __tablename__ = "protocol_detection_points"
    __table_args__ = (
        UniqueConstraint("protocol_id", "measurement_point", name="uq_protocol_measurement_point"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    protocol_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("protocol_records.id"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    measurement_point: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utcnow_naive,
        onupdate=_utcnow_naive,
    )

    protocol: Mapped[ProtocolRecord] = relationship(back_populates="detection_points")
    threshold_rules: Mapped[list["ProtocolThresholdRule"]] = relationship(
        back_populates="detection_point",
    )


class ProtocolThresholdRule(Base):
    __tablename__ = "protocol_threshold_rules"
    __table_args__ = (
        UniqueConstraint("protocol_id", "measurement_point", name="uq_protocol_threshold_point"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    protocol_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("protocol_records.id"),
        nullable=False,
        index=True,
    )
    detection_point_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("protocol_detection_points.id"),
        nullable=True,
        index=True,
    )
    measurement_point: Mapped[str] = mapped_column(String(255), nullable=False)
    operator: Mapped[str] = mapped_column(String(16), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utcnow_naive,
        onupdate=_utcnow_naive,
    )

    protocol: Mapped[ProtocolRecord] = relationship(back_populates="threshold_rules")
    detection_point: Mapped[ProtocolDetectionPoint | None] = relationship(
        back_populates="threshold_rules",
    )
