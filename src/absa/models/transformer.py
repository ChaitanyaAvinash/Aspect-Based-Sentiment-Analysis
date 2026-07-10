"""Track B — transformer encoder fine-tuned for ATE / ACD / ASC.

One shared pre-trained encoder (default ``microsoft/deberta-v3-base``, fallback
``bert-base-uncased``) is fine-tuned three ways:

* ATE  -> token classification (BIO) with sub-word label alignment.
* ACD  -> multi-label sequence classification (restaurants only).
* ASC  -> sequence classification with the aspect marked by special tokens.

This module holds the config, dataset encoders, and the inference pipeline
(:class:`TransformerABSA`). Training lives in ``absa.training.hf_trainer``.
Heavy torch/transformers imports are done lazily inside functions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from absa.data.preprocessing import clean_text
from absa.data.schema import ABSAExample, Polarity
from absa.models.base import AspectPrediction

ATE_LABELS: list[str] = ["O", "B-ASP", "I-ASP"]
ATE_LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(ATE_LABELS)}
ATE_ID2LABEL: dict[int, str] = {i: label for label, i in ATE_LABEL2ID.items()}

ASC_LABELS: list[str] = ["negative", "neutral", "positive"]
ASC_LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(ASC_LABELS)}


@dataclass
class TransformerConfig:
    encoder: str = "microsoft/deberta-v3-base"
    fallback_encoder: str = "bert-base-uncased"
    max_length: int = 128
    asc_markers: tuple[str, str] = ("[ASP]", "[/ASP]")


# --------------------------------------------------------------------------- #
# Encoder / tokenizer resolution (primary -> fallback)
# --------------------------------------------------------------------------- #
def resolve_tokenizer(cfg: TransformerConfig) -> tuple[Any, str]:
    """Return (fast tokenizer, encoder_name), falling back if the primary fails."""
    from transformers import AutoTokenizer

    errors: list[str] = []
    for name in (cfg.encoder, cfg.fallback_encoder):
        try:
            tok = AutoTokenizer.from_pretrained(name, use_fast=True)
            if not tok.is_fast:  # offset mapping needed for ATE alignment
                raise RuntimeError(f"{name} has no fast tokenizer")
            return tok, name
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("Could not load any tokenizer:\n" + "\n".join(errors))


# --------------------------------------------------------------------------- #
# ATE — sub-word BIO label alignment
# --------------------------------------------------------------------------- #
def align_ate_labels(
    word_ids: list[int | None],
    offsets: list[tuple[int, int]],
    gold_spans: list[tuple[int, int]],
) -> list[int]:
    """Map gold character spans to per-subword BIO ids (-100 = ignore)."""
    word_span: dict[int, list[int]] = {}
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        start, end = offsets[idx]
        if wid not in word_span:
            word_span[wid] = [start, end]
        else:
            word_span[wid][1] = end

    word_tag: dict[int, str] = dict.fromkeys(word_span, "O")
    for gold_start, gold_end in gold_spans:
        if gold_start < 0 or gold_end <= gold_start:
            continue
        overlapping = [
            wid
            for wid in sorted(word_span)
            if word_span[wid][0] < gold_end and word_span[wid][1] > gold_start
        ]
        for j, wid in enumerate(overlapping):
            word_tag[wid] = "B-ASP" if j == 0 else "I-ASP"

    labels: list[int] = []
    seen: set[int] = set()
    for wid in word_ids:
        if wid is None or wid in seen:
            labels.append(-100)
        else:
            seen.add(wid)
            labels.append(ATE_LABEL2ID[word_tag[wid]])
    return labels


def build_ate_dataset(examples: list[ABSAExample], tokenizer: Any, max_length: int) -> Any:
    from datasets import Dataset

    texts = [ex.text for ex in examples]
    gold = [[t.span for t in ex.aspect_terms if t.is_explicit] for ex in examples]
    enc = tokenizer(texts, truncation=True, max_length=max_length, return_offsets_mapping=True)
    labels = [
        align_ate_labels(enc.word_ids(i), enc["offset_mapping"][i], gold[i])
        for i in range(len(texts))
    ]
    return Dataset.from_dict(
        {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }
    )


# --------------------------------------------------------------------------- #
# ASC — aspect marking
# --------------------------------------------------------------------------- #
def mark_aspect(text: str, term: str, span: tuple[int, int], markers: tuple[str, str]) -> str:
    open_m, close_m = markers
    start, end = span
    if 0 <= start < end <= len(text) and text[start:end] == term:
        return f"{text[:start]}{open_m} {text[start:end]} {close_m}{text[end:]}"
    return f"{text} {open_m} {term} {close_m}"


# SentencePiece attaches the leading space and trailing punctuation to the
# aspect word (e.g. "functions."); gold aspect terms never include these.
_STRIP_CHARS = set(" \t\n\r.,;:!?\"'`()[]{}")


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Strip leading/trailing whitespace + punctuation from a char span."""
    while start < end and text[start] in _STRIP_CHARS:
        start += 1
    while end > start and text[end - 1] in _STRIP_CHARS:
        end -= 1
    return start, end


def build_asc_dataset(
    examples: list[ABSAExample], tokenizer: Any, max_length: int, markers: tuple[str, str]
) -> Any:
    from datasets import Dataset

    texts: list[str] = []
    labels: list[int] = []
    for ex in examples:
        for term in ex.aspect_terms:
            texts.append(mark_aspect(ex.text, term.term, term.span, markers))
            labels.append(ASC_LABEL2ID[term.polarity])
    enc = tokenizer(texts, truncation=True, max_length=max_length)
    return Dataset.from_dict(
        {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}
    )


# --------------------------------------------------------------------------- #
# ACD — multi-label
# --------------------------------------------------------------------------- #
def category_vocab(examples: list[ABSAExample]) -> list[str]:
    cats: set[str] = set()
    for ex in examples:
        cats.update(c.category for c in ex.aspect_categories)
    return sorted(cats)


def build_acd_dataset(
    examples: list[ABSAExample], tokenizer: Any, max_length: int, categories: list[str]
) -> Any:
    from datasets import Dataset

    cat2id = {c: i for i, c in enumerate(categories)}
    texts: list[str] = []
    labels: list[list[float]] = []
    for ex in examples:
        cats = {c.category for c in ex.aspect_categories}
        if not cats:
            continue
        multihot = [0.0] * len(categories)
        for c in cats:
            if c in cat2id:
                multihot[cat2id[c]] = 1.0
        texts.append(ex.text)
        labels.append(multihot)
    enc = tokenizer(texts, truncation=True, max_length=max_length)
    return Dataset.from_dict(
        {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}
    )


# --------------------------------------------------------------------------- #
# Inference pipeline
# --------------------------------------------------------------------------- #
@dataclass
class TransformerABSA:
    """End-to-end Track B pipeline (implements the ABSAPipeline protocol)."""

    tokenizer: Any
    ate_model: Any
    asc_model: Any
    acd_model: Any | None
    categories: list[str] = field(default_factory=list)
    max_length: int = 128
    markers: tuple[str, str] = ("[ASP]", "[/ASP]")
    device: str = "cpu"

    def _move(self, model: Any) -> Any:
        return model.to(self.device).eval()

    def _extract_spans(self, text: str) -> list[tuple[int, int, str]]:
        import torch

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        word_ids = enc.word_ids(0)
        model = self._move(self.ate_model)
        with torch.no_grad():
            logits = model(**{k: v.to(self.device) for k, v in enc.items()}).logits[0]
        pred_ids = logits.argmax(-1).tolist()

        # first sub-word per word -> word tag + word char span
        word_tag: dict[int, str] = {}
        word_span: dict[int, list[int]] = {}
        seen: set[int] = set()
        for idx, wid in enumerate(word_ids):
            if wid is None:
                continue
            start, end = offsets[idx]
            if wid not in word_span:
                word_span[wid] = [start, end]
            else:
                word_span[wid][1] = end
            if wid not in seen:
                seen.add(wid)
                word_tag[wid] = ATE_ID2LABEL.get(pred_ids[idx], "O")

        spans: list[tuple[int, int]] = []
        cur: list[int] | None = None
        for wid in sorted(word_span):
            tag = word_tag.get(wid, "O")
            s, e = word_span[wid]
            if tag == "B-ASP":
                if cur is not None:
                    spans.append((cur[0], cur[1]))
                cur = [s, e]
            elif tag == "I-ASP" and cur is not None:
                cur[1] = e
            else:
                if cur is not None:
                    spans.append((cur[0], cur[1]))
                    cur = None
        if cur is not None:
            spans.append((cur[0], cur[1]))

        result: list[tuple[int, int, str]] = []
        for start, end in spans:
            start, end = trim_span(text, start, end)
            if end > start:
                result.append((start, end, text[start:end]))
        return result

    def _classify_sentiment(
        self, text: str, term: str, span: tuple[int, int]
    ) -> tuple[Polarity, float]:
        import torch

        marked = mark_aspect(text, term, span, self.markers)
        enc = self.tokenizer(
            marked, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        model = self._move(self.asc_model)
        with torch.no_grad():
            logits = model(**{k: v.to(self.device) for k, v in enc.items()}).logits[0]
        probs = torch.softmax(logits, dim=-1)
        idx = int(probs.argmax())
        return ASC_LABELS[idx], float(probs[idx])  # type: ignore[return-value]

    def _category_scores(self, text: str) -> list[tuple[str, float]]:
        if self.acd_model is None or not self.categories:
            return []
        import torch

        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        model = self._move(self.acd_model)
        with torch.no_grad():
            logits = model(**{k: v.to(self.device) for k, v in enc.items()}).logits[0]
        probs = torch.sigmoid(logits).tolist()
        return list(zip(self.categories, probs, strict=True))

    def _top_category(self, text: str) -> str | None:
        scores = self._category_scores(text)
        if not scores:
            return None
        cat, prob = max(scores, key=lambda cp: cp[1])
        return cat if prob >= 0.5 else None

    def predict(self, text: str) -> list[AspectPrediction]:
        cleaned = clean_text(text)
        if not cleaned:
            return []
        category = self._top_category(cleaned)
        predictions: list[AspectPrediction] = []
        for start, end, surface in self._extract_spans(cleaned):
            sentiment, confidence = self._classify_sentiment(cleaned, surface, (start, end))
            predictions.append(
                AspectPrediction(
                    aspect=surface,
                    sentiment=sentiment,
                    confidence=confidence,
                    start=start,
                    end=end,
                    category=category,
                )
            )
        return predictions

    def predict_batch(self, texts: list[str]) -> list[list[AspectPrediction]]:
        return [self.predict(t) for t in texts]

    # --- EvaluablePipeline interface (uniform scoring across tracks) ---
    @property
    def category_labels(self) -> list[str]:
        return list(self.categories)

    def extract_spans(self, text: str) -> list[tuple[int, int, str]]:
        cleaned = clean_text(text)
        return self._extract_spans(cleaned) if cleaned else []

    def classify_aspect(
        self, text: str, term: str, span: tuple[int, int]
    ) -> tuple[Polarity, float]:
        return self._classify_sentiment(clean_text(text), term, span)

    def predict_categories(self, text: str) -> list[tuple[str, float]]:
        return [(c, p) for c, p in self._category_scores(clean_text(text)) if p >= 0.5]

    @classmethod
    def load(cls, directory: str | Path, device: str | None = None) -> TransformerABSA:
        import torch
        from transformers import AutoTokenizer

        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        quantized = bool(meta.get("quantized", False))
        if device is None:
            # int8 dynamic-quantized models are CPU-only.
            device = "cpu" if quantized else ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer")

        if quantized:

            def _load_pt(name: str) -> Any:
                path = directory / f"{name}.pt"
                if not path.exists():
                    return None
                return torch.load(path, map_location="cpu", weights_only=False)

            ate_model = _load_pt("ate")
            asc_model = _load_pt("asc")
            acd_model = _load_pt("acd")
        else:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoModelForTokenClassification,
            )

            ate_model = AutoModelForTokenClassification.from_pretrained(directory / "ate")
            asc_model = AutoModelForSequenceClassification.from_pretrained(directory / "asc")
            acd_dir = directory / "acd"
            acd_model = (
                AutoModelForSequenceClassification.from_pretrained(acd_dir)
                if acd_dir.exists()
                else None
            )
        return cls(
            tokenizer=tokenizer,
            ate_model=ate_model,
            asc_model=asc_model,
            acd_model=acd_model,
            categories=meta.get("categories", []),
            max_length=int(meta.get("max_length", 128)),
            markers=tuple(meta.get("markers", ["[ASP]", "[/ASP]"])),
            device=device,
        )


__all__ = [
    "ASC_LABELS",
    "ATE_LABELS",
    "TransformerABSA",
    "TransformerConfig",
    "align_ate_labels",
    "build_acd_dataset",
    "build_asc_dataset",
    "build_ate_dataset",
    "category_vocab",
    "mark_aspect",
    "resolve_tokenizer",
    "trim_span",
]
