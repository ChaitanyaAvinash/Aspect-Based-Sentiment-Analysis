"""Shared prediction types and the model interface used by both tracks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from absa.data.schema import Polarity


@dataclass
class AspectPrediction:
    """One predicted aspect with its category, sentiment, confidence, and span."""

    aspect: str
    sentiment: Polarity
    confidence: float
    start: int = -1
    end: int = -1
    category: str | None = None

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)

    def to_dict(self) -> dict[str, object]:
        return {
            "aspect": self.aspect,
            "category": self.category,
            "sentiment": self.sentiment,
            "confidence": round(float(self.confidence), 4),
            "span": [self.start, self.end],
        }


@runtime_checkable
class ABSAPipeline(Protocol):
    """End-to-end interface: text in, structured aspect predictions out."""

    def predict(self, text: str) -> list[AspectPrediction]: ...

    def predict_batch(self, texts: list[str]) -> list[list[AspectPrediction]]: ...


__all__ = ["ABSAPipeline", "AspectPrediction"]
