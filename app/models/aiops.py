"""
AIOps ???????
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class AIOpsRequest(BaseModel):
    """AIOps ????"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "session-123"
            }
        }
    )

    session_id: Optional[str] = Field(
        default="default",
        description="??ID?????????"
    )


class AlertInfo(BaseModel):
    """????"""
    alertname: str
    severity: str
    instance: str
    duration: str
    description: Optional[str] = None


class DiagnosisResponse(BaseModel):
    """?????????"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": 200,
                "message": "success",
                "data": {
                    "status": "completed",
                    "target_alert": {
                        "alertname": "HighCPUUsage",
                        "severity": "critical"
                    },
                    "diagnosis": {
                        "root_cause": "????????",
                        "recommendations": ["????????", "??SQL??"]
                    }
                }
            }
        }
    )

    code: int = 200
    message: str = "success"
    data: Dict[str, Any]
