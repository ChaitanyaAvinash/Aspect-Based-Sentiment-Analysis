"""Aspect highlighting + cached examples for the demo UI (pure, testable)."""

from __future__ import annotations

import html

from absa.models.base import AspectPrediction

SENTIMENT_COLORS = {"positive": "#1baf7a", "negative": "#e34948", "neutral": "#eda100"}

EXAMPLE_REVIEWS = [
    "The pizza was delicious but the service was painfully slow.",
    "Battery life is amazing and the screen is gorgeous, though the keyboard feels cheap.",
    "Great ambience and friendly staff, but a bit overpriced for the tiny portions.",
    "Fast performance and a bright display, yet it runs hot and the fan is noisy.",
    "The sushi was fresh, the drinks were cheap, and the waiter was very attentive.",
]


def highlight_html(text: str, aspects: list[AspectPrediction]) -> str:
    """Return HTML with each aspect span wrapped + colored by sentiment."""
    spans = sorted((a for a in aspects if a.start >= 0 and a.end > a.start), key=lambda a: a.start)
    parts: list[str] = []
    cursor = 0
    for a in spans:
        if a.start < cursor:  # skip overlaps
            continue
        parts.append(html.escape(text[cursor : a.start]))
        color = SENTIMENT_COLORS.get(a.sentiment, "#888888")
        segment = html.escape(text[a.start : a.end])
        tip = f"{a.sentiment} {a.confidence:.0%}"
        parts.append(
            f'<span style="background:{color}26;border-bottom:3px solid {color};'
            f'border-radius:4px;padding:0 3px" title="{tip}">{segment}</span>'
        )
        cursor = a.end
    parts.append(html.escape(text[cursor:]))
    return "".join(parts)


__all__ = ["EXAMPLE_REVIEWS", "SENTIMENT_COLORS", "highlight_html"]
