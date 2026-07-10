"""Tests for the demo highlight helper (pure, no heavy deps)."""

from __future__ import annotations

from absa.models.base import AspectPrediction
from absa.serving.render import EXAMPLE_REVIEWS, SENTIMENT_COLORS, highlight_html


def _aspect(term: str, sentiment: str, start: int, end: int) -> AspectPrediction:
    return AspectPrediction(aspect=term, sentiment=sentiment, confidence=0.9, start=start, end=end)


def test_highlight_wraps_and_colors_aspect() -> None:
    text = "The pizza was great"
    out = highlight_html(text, [_aspect("pizza", "positive", 4, 9)])
    assert "pizza" in out
    assert SENTIMENT_COLORS["positive"] in out
    assert out.startswith("The ")


def test_highlight_escapes_html() -> None:
    text = "<b>bad</b> service"
    out = highlight_html(text, [_aspect("service", "negative", 11, 18)])
    assert "&lt;b&gt;" in out
    assert SENTIMENT_COLORS["negative"] in out


def test_highlight_no_aspects_is_escaped_text() -> None:
    assert highlight_html("nothing here", []) == "nothing here"


def test_highlight_skips_overlaps() -> None:
    text = "great battery life here"
    a = _aspect("battery life", "positive", 6, 18)
    b = _aspect("life", "positive", 14, 18)  # overlaps a
    out = highlight_html(text, [a, b])
    assert out.count("border-bottom") == 1


def test_examples_available() -> None:
    assert len(EXAMPLE_REVIEWS) >= 3
    assert all(isinstance(r, str) and r for r in EXAMPLE_REVIEWS)
