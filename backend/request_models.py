
from pydantic import BaseModel, Field
from typing import Optional


class AnalyzeRequest(BaseModel):
    session_id: str = Field(
        default="default",
        description="Unique ID for an agent monitoring session"
    )

    goal: str = Field(
        min_length=5,
        description="The original goal provided by the user"
    )

    action: str = Field(
        min_length=3,
        description="The action performed by the AI agent"
    )

    action_type: Optional[str] = "general"


class AnalyzeResponse(BaseModel):
    session_id: str
    goal: str
    action: str
    similarity: float
    instantaneous_drift: float
    recent_mean: float
    persistence: float
    cumulative_drift: float
    drift_score: float
    status: str
    intervention: str


class ResetRequest(BaseModel):
    session_id: str = "default"
