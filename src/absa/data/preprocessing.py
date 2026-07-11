"""Stdlib-only text cleaning, offset-preserving tokenization, and BIO tagging."""

from __future__ import annotations

import re
from dataclasses import dataclass

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def strip_html(text: str) -> str:
    """Remove HTML tags and unescape a few common entities."""
    text = _HTML_TAG_RE.sub(" ", text)
    for entity, repl in (("&amp;", "&"), ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")):
        text = text.replace(entity, repl)
    return text


def normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def clean_text(
    text: str,
    *,
    strip_html_tags: bool = True,
    max_chars: int = 2000,
    min_chars: int = 1,
) -> str:
    """Clean and length-guard input; empty in -> "", over-long is truncated."""
    if text is None:
        return ""
    if strip_html_tags:
        text = strip_html(text)
    text = normalize_whitespace(text)
    if len(text) < min_chars:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


@dataclass(frozen=True)
class Token:
    """A token with its character span [start, end) in the source text."""

    text: str
    start: int
    end: int


def tokenize(text: str) -> list[Token]:
    """Regex word tokenizer that preserves character offsets."""
    return [Token(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def spans_to_bio(tokens: list[Token], spans: list[tuple[int, int]]) -> list[str]:
    """Character spans -> token-level BIO tags (B-ASP/I-ASP/O) by overlap."""
    tags = ["O"] * len(tokens)
    for start, end in spans:
        if start < 0 or end <= start:
            continue
        first = True
        for i, tok in enumerate(tokens):
            if tok.start < end and tok.end > start:
                tags[i] = "B-ASP" if first else "I-ASP"
                first = False
    return tags


def looks_english(text: str, threshold: float = 0.6) -> bool:
    """Cheap non-English guard: is the text mostly latin letters?"""
    if not text:
        return True
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    letters = sum(1 for c in text if c.isalpha())
    if letters == 0:
        return True
    return ascii_letters / letters >= threshold


__all__ = [
    "Token",
    "clean_text",
    "looks_english",
    "normalize_whitespace",
    "spans_to_bio",
    "strip_html",
    "tokenize",
]
