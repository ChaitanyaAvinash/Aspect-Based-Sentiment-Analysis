"""Tests for Track B (transformer) encoding + inference helpers.

The pure functions (label alignment, aspect marking, category vocab) are
torch-free and run everywhere. Dataset builders need transformers/datasets and
are skipped when those aren't installed (e.g. the light CI job).
"""

from __future__ import annotations

import pytest

from absa.data import load_sample
from absa.models.transformer import (
    ATE_LABEL2ID,
    align_ate_labels,
    category_vocab,
    mark_aspect,
    trim_span,
)

MARKERS = ("[ASP]", "[/ASP]")


# pure functions (no torch)
def test_align_ate_labels_bio() -> None:
    # [CLS] battery life is great [SEP]  — gold span "battery life" = (0, 12)
    word_ids = [None, 0, 1, 2, 3, None]
    offsets = [(0, 0), (0, 7), (8, 12), (13, 15), (16, 21), (0, 0)]
    labels = align_ate_labels(word_ids, offsets, [(0, 12)])
    assert labels[0] == -100 and labels[-1] == -100
    assert labels[1] == ATE_LABEL2ID["B-ASP"]
    assert labels[2] == ATE_LABEL2ID["I-ASP"]
    assert labels[3] == ATE_LABEL2ID["O"]


def test_align_ate_labels_ignores_continuation_subwords() -> None:
    # word 0 is split into two sub-words; only the first is labelled.
    word_ids = [None, 0, 0, 1, None]
    offsets = [(0, 0), (0, 4), (4, 7), (8, 12), (0, 0)]
    labels = align_ate_labels(word_ids, offsets, [(0, 7)])
    assert labels[1] == ATE_LABEL2ID["B-ASP"]
    assert labels[2] == -100
    assert labels[3] == ATE_LABEL2ID["O"]


def test_mark_aspect_with_valid_span() -> None:
    out = mark_aspect("The pizza was good", "pizza", (4, 9), MARKERS)
    assert "[ASP] pizza [/ASP]" in out


def test_mark_aspect_fallback_when_span_missing() -> None:
    out = mark_aspect("great food", "service", (-1, -1), MARKERS)
    assert out.endswith("[ASP] service [/ASP]")


def test_category_vocab_sorted() -> None:
    vocab = category_vocab(load_sample())
    assert vocab == sorted(vocab)
    assert "food" in vocab


def test_trim_span_strips_punct_and_space() -> None:
    text = "I love the touchscreen functions."
    # span covering "touchscreen functions." including trailing period
    start, end = trim_span(text, 10, len(text))
    assert text[start:end] == "touchscreen functions"


def test_trim_span_keeps_internal_punct() -> None:
    text = "The Wi-Fi is great"
    start, end = trim_span(text, 4, 9)
    assert text[start:end] == "Wi-Fi"


# dataset builders (need transformers + datasets)
@pytest.fixture(scope="module")
def tokenizer():  # type: ignore[no-untyped-def]
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("prajjwal1/bert-tiny", use_fast=True)
    except Exception:
        pytest.skip("tokenizer unavailable offline")


def test_build_ate_dataset(tokenizer) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("datasets")
    from absa.models.transformer import build_ate_dataset

    ds = build_ate_dataset(load_sample()[:5], tokenizer, 64)
    assert set(ds.column_names) >= {"input_ids", "attention_mask", "labels"}
    assert len(ds) == 5
    assert len(ds["input_ids"][0]) == len(ds["labels"][0])


def test_build_asc_dataset(tokenizer) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("datasets")
    from absa.models.transformer import build_asc_dataset

    ds = build_asc_dataset(load_sample()[:5], tokenizer, 64, MARKERS)
    assert all(0 <= label <= 2 for label in ds["labels"])


def test_build_acd_dataset(tokenizer) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("datasets")
    from absa.models.transformer import build_acd_dataset

    examples = load_sample()[:10]
    cats = category_vocab(examples)
    ds = build_acd_dataset(examples, tokenizer, 64, cats)
    assert len(ds["labels"][0]) == len(cats)
