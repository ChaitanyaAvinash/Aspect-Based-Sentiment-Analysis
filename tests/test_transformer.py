"""Tests for Track B (transformer) encoding + inference helpers.

The pure functions (label alignment, aspect marking, category vocab) are
torch-free and run everywhere. Dataset builders need transformers/datasets and
are skipped when those aren't installed (e.g. the light CI job).
"""

from __future__ import annotations

import itertools

import pytest

from absa.data import load_sample
from absa.models.transformer import (
    ATE_LABEL2ID,
    TransformerABSA,
    align_ate_labels,
    category_vocab,
    decode_bio,
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


def test_decode_bio_merges_words_and_subwords() -> None:
    # [CLS] battery life is great [SEP]; "battery" split into two sub-words.
    word_ids = [None, 0, 0, 1, 2, 3, None]
    offsets = [[0, 0], [0, 4], [4, 7], [8, 12], [13, 15], [16, 21], [0, 0]]
    b, i, o = ATE_LABEL2ID["B-ASP"], ATE_LABEL2ID["I-ASP"], ATE_LABEL2ID["O"]
    assert decode_bio(word_ids, offsets, [o, b, o, i, o, b, o]) == [(0, 12), (16, 21)]


def test_decode_bio_ignores_orphan_inside_tag() -> None:
    word_ids = [None, 0, 1, None]
    offsets = [[0, 0], [0, 3], [4, 8], [0, 0]]
    i, o = ATE_LABEL2ID["I-ASP"], ATE_LABEL2ID["O"]
    assert decode_bio(word_ids, offsets, [o, i, o, o]) == []


class _WordTokenizer:
    """Fake tokenizer: one token per whitespace word, plus [CLS]/[SEP]."""

    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": [0] * (len(text.split()) + 2)}


def _windowed(max_length: int) -> TransformerABSA:
    return TransformerABSA(
        tokenizer=_WordTokenizer(),
        ate_model=None,
        asc_model=None,
        acd_model=None,
        max_length=max_length,
    )


def test_windows_keep_short_text_whole() -> None:
    text = "The pizza was great. Service was slow."
    assert _windowed(64)._windows(text) == [(0, len(text))]


def test_windows_pack_long_text_by_sentence() -> None:
    sentence = "The staff were friendly and the room was nice."  # 9 words
    text = " ".join([sentence] * 6) + " Sadly the dessert was stale."
    windows = _windowed(32)._windows(text)  # budget 24 tokens = 2 sentences
    assert len(windows) > 1
    assert windows[0][0] == 0 and windows[-1][1] == len(text)
    assert all(a[1] == b[0] for a, b in itertools.pairwise(windows))
    assert "dessert" in text[windows[-1][0] : windows[-1][1]]


def test_quantized_pipeline_never_batches_sequences() -> None:
    # Dynamic int8 shares activation scales across a batch, so batching would let one
    # aspect's input change another aspect's prediction.
    pipe = _windowed(64)
    assert pipe._batches(3) == [[0, 1, 2]]
    pipe.quantized = True
    assert pipe._batches(3) == [[0], [1], [2]]


def test_int8_roundtrip_is_weights_only(tmp_path) -> None:  # type: ignore[no-untyped-def]
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from absa.models.transformer import load_int8, quantize_int8, save_int8

    config = transformers.BertConfig(
        vocab_size=50,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=3,
    )
    model = transformers.BertForSequenceClassification(config).eval()
    save_int8(model, tmp_path)
    loaded = load_int8(tmp_path, "seq")

    query = loaded.bert.encoder.layer[0].attention.self.query
    assert "quantized" in type(query).__module__
    ids = torch.randint(0, 50, (2, 7))
    expected = quantize_int8(model)(ids).logits
    assert torch.allclose(loaded(ids).logits, expected)


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
