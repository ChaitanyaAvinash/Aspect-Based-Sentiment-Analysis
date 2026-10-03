"""Stdlib-only text cleaning, offset-preserving tokenization, and BIO tagging."""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_HTML_ENTITIES = {"&amp;": "&", "&quot;": '"', "&#39;": "'", "&nbsp;": " "}


def strip_html(text: str) -> str:
    """Remove HTML tags and unescape a few common entities."""
    text = _HTML_TAG_RE.sub(" ", text)
    for entity, repl in _HTML_ENTITIES.items():
        text = text.replace(entity, repl)
    return text


def normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


@dataclass(frozen=True)
class CleanedText:
    """Cleaned text plus, per cleaned char, the raw [start, end) it came from."""

    raw: str
    text: str
    origin: tuple[tuple[int, int], ...]

    def to_raw(self, start: int, end: int) -> tuple[int, int]:
        """Map a non-empty span of the cleaned text back onto the raw text."""
        return self.origin[start][0], self.origin[end - 1][1]

    def to_clean(self, start: int, end: int) -> tuple[int, int] | None:
        """Map a raw span onto the cleaned text (None if nothing of it survives)."""
        if start < 0 or end <= start:
            return None
        clean_start = bisect.bisect_right([e for _, e in self.origin], start)
        clean_end = bisect.bisect_left([s for s, _ in self.origin], end)
        return (clean_start, clean_end) if clean_start < clean_end else None


def clean_text_aligned(
    text: str,
    *,
    strip_html_tags: bool = True,
    max_chars: int = 2000,
    min_chars: int = 1,
) -> CleanedText:
    """Same cleaning as :func:`clean_text`, keeping the cleaned -> raw offset map."""
    raw = text or ""
    chars: list[str] = []
    origin: list[tuple[int, int]] = []

    def emit(ch: str, start: int, end: int) -> None:
        if ch.isspace():
            if not chars or chars[-1] == " ":  # leading or repeated whitespace
                return
            ch = " "
        chars.append(ch)
        origin.append((start, end))

    i = 0
    while i < len(raw):
        if strip_html_tags and raw[i] == "<" and (tag := _HTML_TAG_RE.match(raw, i)):
            emit(" ", i, tag.end())
            i = tag.end()
            continue
        if strip_html_tags and raw[i] == "&":
            entity = next((e for e in _HTML_ENTITIES if raw.startswith(e, i)), None)
            if entity:
                emit(_HTML_ENTITIES[entity], i, i + len(entity))
                i += len(entity)
                continue
        emit(raw[i], i, i + 1)
        i += 1

    del chars[max_chars:], origin[max_chars:]
    while chars and chars[-1] == " ":
        chars.pop()
        origin.pop()
    if len(chars) < min_chars:
        return CleanedText(raw=raw, text="", origin=())
    return CleanedText(raw=raw, text="".join(chars), origin=tuple(origin))


def clean_text(
    text: str,
    *,
    strip_html_tags: bool = True,
    max_chars: int = 2000,
    min_chars: int = 1,
) -> str:
    """Clean and length-guard input; empty in -> "", over-long is truncated."""
    return clean_text_aligned(
        text, strip_html_tags=strip_html_tags, max_chars=max_chars, min_chars=min_chars
    ).text


def find_nearest(text: str, term: str, hint: int) -> int:
    """Index of the occurrence of ``term`` closest to ``hint`` (-1 if absent)."""
    hits = [m.start() for m in re.finditer(re.escape(term), text)] if term else []
    return min(hits, key=lambda i: abs(i - hint)) if hits else -1


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
    "CleanedText",
    "Token",
    "clean_text",
    "clean_text_aligned",
    "find_nearest",
    "looks_english",
    "normalize_whitespace",
    "spans_to_bio",
    "strip_html",
    "tokenize",
]
