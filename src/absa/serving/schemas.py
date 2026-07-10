"""Pydantic request/response schemas for the ABSA API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from absa.models.base import AspectPrediction


class AspectOut(BaseModel):
    """One predicted aspect."""

    aspect: str
    category: str | None = None
    sentiment: str
    confidence: float = Field(ge=0.0, le=1.0)
    span: tuple[int, int]

    @classmethod
    def from_prediction(cls, prediction: AspectPrediction) -> AspectOut:
        return cls(
            aspect=prediction.aspect,
            category=prediction.category,
            sentiment=prediction.sentiment,
            confidence=round(float(prediction.confidence), 4),
            span=(prediction.start, prediction.end),
        )


class PredictRequest(BaseModel):
    # Empty text is allowed and returns no aspects (graceful); very long input is
    # capped here and further truncated by the pipeline.
    text: str = Field(
        default="", max_length=20000, examples=["The battery lasts all day but the screen is dim."]
    )


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


class PredictResponse(BaseModel):
    text: str
    track: str
    count: int
    aspects: list[AspectOut]


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    track: str
    device: str
    model_loaded: bool


__all__ = [
    "AspectOut",
    "BatchPredictRequest",
    "BatchPredictResponse",
    "HealthResponse",
    "PredictRequest",
    "PredictResponse",
]
