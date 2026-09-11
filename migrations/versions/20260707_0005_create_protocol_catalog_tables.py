"""create protocol catalog tables

Revision ID: 20260707_0005
Revises: 20260629_0004
Create Date: 2026-07-07
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260707_0005"
down_revision: Union[str, None] = "20260629_0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "protocol_records",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("source_filename", sa.String(length=255), nullable=True),
        sa.Column("source_ingestion_id", sa.String(length=64), nullable=True),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("raw_payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["source_ingestion_id"], ["protocol_pdf_ingestions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index(op.f("ix_protocol_records_name"), "protocol_records", ["name"])
    op.create_index(
        op.f("ix_protocol_records_source_ingestion_id"),
        "protocol_records",
        ["source_ingestion_id"],
    )

    op.create_table(
        "protocol_equipment_mappings",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("protocol_id", sa.String(length=64), nullable=False),
        sa.Column("equipment_name", sa.String(length=255), nullable=False),
        sa.Column("source_excerpt", sa.Text(), nullable=True),
        sa.Column("raw_payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["protocol_id"], ["protocol_records.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("protocol_id", "equipment_name", name="uq_protocol_equipment_name"),
    )
    op.create_index(
        op.f("ix_protocol_equipment_mappings_protocol_id"),
        "protocol_equipment_mappings",
        ["protocol_id"],
    )

    op.create_table(
        "protocol_detection_points",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("protocol_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("measurement_point", sa.String(length=255), nullable=False),
        sa.Column("source", sa.JSON(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("raw_payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["protocol_id"], ["protocol_records.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("protocol_id", "measurement_point", name="uq_protocol_measurement_point"),
    )
    op.create_index(
        op.f("ix_protocol_detection_points_protocol_id"),
        "protocol_detection_points",
        ["protocol_id"],
    )

    op.create_table(
        "protocol_threshold_rules",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("protocol_id", sa.String(length=64), nullable=False),
        sa.Column("detection_point_id", sa.String(length=64), nullable=True),
        sa.Column("measurement_point", sa.String(length=255), nullable=False),
        sa.Column("operator", sa.String(length=16), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("unit", sa.String(length=64), nullable=False),
        sa.Column("raw", sa.Text(), nullable=True),
        sa.Column("source", sa.JSON(), nullable=False),
        sa.Column("raw_payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["detection_point_id"], ["protocol_detection_points.id"]),
        sa.ForeignKeyConstraint(["protocol_id"], ["protocol_records.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("protocol_id", "measurement_point", name="uq_protocol_threshold_point"),
    )
    op.create_index(
        op.f("ix_protocol_threshold_rules_detection_point_id"),
        "protocol_threshold_rules",
        ["detection_point_id"],
    )
    op.create_index(
        op.f("ix_protocol_threshold_rules_protocol_id"),
        "protocol_threshold_rules",
        ["protocol_id"],
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_protocol_threshold_rules_protocol_id"),
        table_name="protocol_threshold_rules",
    )
    op.drop_index(
        op.f("ix_protocol_threshold_rules_detection_point_id"),
        table_name="protocol_threshold_rules",
    )
    op.drop_table("protocol_threshold_rules")
    op.drop_index(
        op.f("ix_protocol_detection_points_protocol_id"),
        table_name="protocol_detection_points",
    )
    op.drop_table("protocol_detection_points")
    op.drop_index(
        op.f("ix_protocol_equipment_mappings_protocol_id"),
        table_name="protocol_equipment_mappings",
    )
    op.drop_table("protocol_equipment_mappings")
    op.drop_index(op.f("ix_protocol_records_source_ingestion_id"), table_name="protocol_records")
    op.drop_index(op.f("ix_protocol_records_name"), table_name="protocol_records")
    op.drop_table("protocol_records")
