"""Pydantic request/response schemas for the ABSA API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from absa.models.base import AspectPrediction


class AspectOut(BaseModel):
    """One predicted aspect; ``span`` is [start, end) into the request text."""

    aspect: str
    sentiment: str
    confidence: float = Field(ge=0.0, le=1.0)
    span: tuple[int, int]

    @classmethod
    def from_prediction(cls, prediction: AspectPrediction) -> AspectOut:
        return cls(
            aspect=prediction.aspect,
            sentiment=prediction.sentiment,
            confidence=round(float(prediction.confidence), 4),
            span=(prediction.start, prediction.end),
        )


class CategoryOut(BaseModel):
    """A sentence-level aspect category (SemEval-2014 restaurant categories)."""

    category: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictRequest(BaseModel):
    # Empty text returns no aspects; very long input is capped, then truncated by the pipeline.
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
    categories: list[CategoryOut]


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
    "CategoryOut",
    "HealthResponse",
    "PredictRequest",
    "PredictResponse",
]
