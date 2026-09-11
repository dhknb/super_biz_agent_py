"""API response models."""

from typing import Any

from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    """Chat response."""

    answer: str = Field(..., description="AI answer")
    session_id: str = Field(..., description="Session ID")


class SessionInfoResponse(BaseModel):
    """Session information response."""

    session_id: str = Field(..., description="Session ID")
    message_count: int = Field(..., description="Message count")
    history: list[dict[str, Any]] = Field(..., description="History messages")


class ApiResponse(BaseModel):
    """Generic API response."""

    status: str = Field(..., description="Status")
    message: str = Field(..., description="Message")
    data: Any | None = Field(None, description="Data")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Version")
