"""Data loading, SemEval XML parsing, preprocessing, and splits."""

from __future__ import annotations

from absa.data.io import SAMPLE_PATH, load_sample, read_jsonl, write_jsonl
from absa.data.preprocessing import (
    Token,
    clean_text,
    looks_english,
    spans_to_bio,
    tokenize,
)
from absa.data.schema import (
    ABSAExample,
    AspectCategory,
    AspectTerm,
    Domain,
    Polarity,
)
from absa.data.semeval import parse_semeval_string, parse_semeval_xml
from absa.data.splits import SplitRatios, make_splits, sentence_label
from absa.data.stats import label_distribution, summarize_splits

__all__ = [
    "SAMPLE_PATH",
    "ABSAExample",
    "AspectCategory",
    "AspectTerm",
    "Domain",
    "Polarity",
    "SplitRatios",
    "Token",
    "clean_text",
    "label_distribution",
    "load_sample",
    "looks_english",
    "make_splits",
    "parse_semeval_string",
    "parse_semeval_xml",
    "read_jsonl",
    "sentence_label",
    "spans_to_bio",
    "summarize_splits",
    "tokenize",
    "write_jsonl",
]
