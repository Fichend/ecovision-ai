from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictionScore(BaseModel):
    label: str
    probability: float = Field(ge=0, le=1)


class ClassificationContext(BaseModel):
    predicted_class: str
    top_predictions: list[PredictionScore]
    confidence: float = Field(ge=0, le=1)
    confidence_level: Literal["high", "medium", "low"]
    uncertainty_message: str
    model_version: str


class ClassificationResponse(ClassificationContext):
    inference_time_ms: float
    gradcam_available: bool
    gradcam_data_url: str | None = None


class SourceReference(BaseModel):
    title: str
    publisher: str
    url: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    conversation_history: list[ChatMessage] = Field(default_factory=list, max_length=12)
    classification: ClassificationContext
    question: str | None = Field(default=None, max_length=2000)
    location: Literal["hong_kong", "mainland_china"] = "hong_kong"
    city: str | None = Field(default=None, max_length=120)


class ChatResponse(BaseModel):
    response: str
    sources: list[SourceReference]
    uncertainty_information: str
    provider: Literal["deepseek", "grounded_fallback"]
    model: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_ready: bool
    model_version: str | None
    device: str | None
    deepseek_configured: bool
    knowledge_version: str
