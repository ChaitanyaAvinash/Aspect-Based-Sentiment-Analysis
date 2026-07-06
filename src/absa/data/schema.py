"""Normalized ABSA data schema.

A single :class:`ABSAExample` is the canonical unit consumed across all tracks
and tasks. SemEval and any other source is parsed into this shape. Aspect terms
(with character spans + polarity) drive ATE/ASC; aspect categories (with
polarity) drive ACD. In SemEval-2014 the two are annotated independently, so we
keep them as separate lists rather than forcing a join.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Polarity = Literal["positive", "negative", "neutral"]
Domain = Literal["restaurants", "laptops", "other"]

#: Sentinel span for implicit / NULL aspects (no character offset).
NULL_SPAN: tuple[int, int] = (-1, -1)


class AspectTerm(BaseModel):
    """An explicit aspect term with its character span and polarity."""

    term: str
    polarity: Polarity
    start: int = -1
    end: int = -1

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)

    @property
    def is_explicit(self) -> bool:
        return self.start >= 0 and self.end > self.start

    def aligns_with(self, text: str) -> bool:
        """True if the recorded span actually points at ``term`` in ``text``."""
        if not self.is_explicit:
            return True
        return text[self.start : self.end] == self.term


class AspectCategory(BaseModel):
    """A coarse aspect category (e.g. ``food``, ``service``) with polarity."""

    category: str
    polarity: Polarity


class ABSAExample(BaseModel):
    """One sentence/review with its aspect-term and aspect-category annotations."""

    id: str
    text: str
    domain: Domain = "other"
    aspect_terms: list[AspectTerm] = Field(default_factory=list)
    aspect_categories: list[AspectCategory] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_spans(self) -> ABSAExample:
        for term in self.aspect_terms:
            if not term.aligns_with(self.text):
                raise ValueError(
                    f"Span {term.span} does not match term {term.term!r} in example {self.id!r}"
                )
        return self

    @property
    def polarities(self) -> list[Polarity]:
        return [t.polarity for t in self.aspect_terms]


__all__ = [
    "NULL_SPAN",
    "ABSAExample",
    "AspectCategory",
    "AspectTerm",
    "Domain",
    "Polarity",
]
