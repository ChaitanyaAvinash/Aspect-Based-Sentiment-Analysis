"""Shared prediction types and the model interface used by both tracks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from absa.data.schema import Polarity


@dataclass
class AspectPrediction:
    """One predicted aspect: sentiment, confidence, and [start, end) span in the input text."""

    aspect: str
    sentiment: Polarity
    confidence: float
    start: int = -1
    end: int = -1

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)

    def to_dict(self) -> dict[str, object]:
        return {
            "aspect": self.aspect,
            "sentiment": self.sentiment,
            "confidence": round(float(self.confidence), 4),
            "span": [self.start, self.end],
        }


@runtime_checkable
class ABSAPipeline(Protocol):
    """End-to-end interface: text in, aspects (+ sentence-level categories) out.

    Categories are sentence-level: SemEval-2014 does not link aspect terms to
    categories, so a per-aspect category would be a guess.
    """

    def predict(self, text: str) -> list[AspectPrediction]: ...

    def predict_batch(self, texts: list[str]) -> list[list[AspectPrediction]]: ...

    def predict_categories(self, text: str) -> list[tuple[str, float]]: ...


@runtime_checkable
class EvaluablePipeline(Protocol):
    """Per-sub-task hooks used for a fair, uniform evaluation of any track."""

    @property
    def category_labels(self) -> list[str]: ...

    def extract_spans(self, text: str) -> list[tuple[int, int, str]]: ...

    def classify_aspect(self, text: str, term: str, span: tuple[int, int]) -> tuple[str, float]: ...

    def predict_categories(self, text: str) -> list[tuple[str, float]]: ...


__all__ = ["ABSAPipeline", "AspectPrediction", "EvaluablePipeline"]
