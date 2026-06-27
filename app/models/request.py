"""??????

?? API ??? Pydantic ??
"""

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """????"""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "Id": "session-123",
                "Question": "?????????",
            }
        },
    )

    id: str = Field(..., description="?? ID", alias="Id")
    question: str = Field(..., description="????", alias="Question")


class ClearRequest(BaseModel):
    """??????"""

    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(..., description="?? ID", alias="sessionId")
