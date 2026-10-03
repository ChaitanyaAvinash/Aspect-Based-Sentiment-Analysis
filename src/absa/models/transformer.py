"""Track B config, dataset encoders, and inference pipeline (ATE/ACD/ASC)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from absa.data.preprocessing import clean_text, clean_text_aligned
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


# Encoder / tokenizer resolution (primary -> fallback)
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


# ATE — sub-word BIO label alignment
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


# ASC — aspect marking
def mark_aspect(text: str, term: str, span: tuple[int, int], markers: tuple[str, str]) -> str:
    open_m, close_m = markers
    start, end = span
    if 0 <= start < end <= len(text) and text[start:end] == term:
        return f"{text[:start]}{open_m} {text[start:end]} {close_m}{text[end:]}"
    return f"{text} {open_m} {term} {close_m}"


# SentencePiece attaches leading space + trailing punctuation to the aspect word.
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


# ACD — multi-label
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


# BIO decoding (shared by inference; pure, testable)
def decode_bio(
    word_ids: list[int | None], offsets: list[list[int]], pred_ids: list[int]
) -> list[tuple[int, int]]:
    """Per-subword predictions -> word-level BIO (first sub-word) -> char spans."""
    word_tag: dict[int, str] = {}
    word_span: dict[int, list[int]] = {}
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        start, end = offsets[idx]
        if wid not in word_span:
            word_span[wid] = [start, end]
            word_tag[wid] = ATE_ID2LABEL.get(pred_ids[idx], "O")
        else:
            word_span[wid][1] = end

    spans: list[tuple[int, int]] = []
    cur: list[int] | None = None
    for wid in sorted(word_span):
        tag = word_tag[wid]
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
    return spans


# int8 CPU artifact: weights only (no pickled modules)
TASK_HEADS: dict[str, str] = {"ate": "token", "asc": "seq", "acd": "seq"}
INT8_WEIGHTS = "model_int8.pt"


def quantize_int8(model: Any) -> Any:
    """Dynamic int8 on every Linear layer (returns a quantized copy)."""
    import torch

    model.eval()
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def save_int8(model: Any, directory: Path) -> None:
    """Write config + quantized state_dict (loadable with ``weights_only=True``)."""
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(directory)
    torch.save(quantize_int8(model).state_dict(), directory / INT8_WEIGHTS)


def load_int8(directory: Path, head: str) -> Any:
    """Rebuild the architecture from config, quantize it the same way, load weights."""
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
    )

    auto = (
        AutoModelForTokenClassification if head == "token" else AutoModelForSequenceClassification
    )
    model = quantize_int8(auto.from_config(AutoConfig.from_pretrained(directory)))
    state = torch.load(directory / INT8_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval()


# Inference pipeline
# Sentence-ish units used to pack long reviews into encoder-sized windows.
_SENTENCE_RE = re.compile(r".+?(?:[.!?]+(?=\s)|$)", re.S)
_MARKER_BUDGET = 8  # tokens reserved for the ASC aspect markers


@dataclass
class TransformerABSA:
    """End-to-end Track B pipeline (implements the ABSAPipeline protocol).

    Spans returned by ``predict``/``extract_spans`` index into the caller's raw text.
    """

    tokenizer: Any
    ate_model: Any
    asc_model: Any
    acd_model: Any | None
    categories: list[str] = field(default_factory=list)
    max_length: int = 128
    markers: tuple[str, str] = ("[ASP]", "[/ASP]")
    device: str = "cpu"
    quantized: bool = False
    # Sigmoid cutoff for ACD. int8 shrinks category probabilities, so export
    # re-tunes this on the validation split and stores it in meta.json.
    acd_threshold: float = 0.5

    def __post_init__(self) -> None:
        for model in (self.ate_model, self.asc_model, self.acd_model):
            if model is not None:
                model.to(self.device).eval()

    def _batches(self, n: int) -> list[list[int]]:
        """Index groups to run through the encoder together.

        Dynamic int8 picks activation scales per input tensor, so in a batch one
        sequence (and its padding) would shift another's prediction; quantized
        models therefore run one sequence per forward pass.
        """
        return [[i] for i in range(n)] if self.quantized else [list(range(n))]

    def _encode(self, texts: list[str], **kwargs: Any) -> Any:
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            **kwargs,
        )

    def _logits(self, model: Any, enc: Any) -> Any:
        import torch

        with torch.no_grad():
            return model(**{k: v.to(self.device) for k, v in enc.items()}).logits

    def _windows(self, text: str) -> list[tuple[int, int]]:
        """Sentence-aligned windows that fit the encoder.

        Text within budget (every SemEval sentence) stays one window; longer reviews
        are packed sentence by sentence instead of being cut off at ``max_length``.
        """
        budget = self.max_length - _MARKER_BUDGET
        if len(self.tokenizer(text)["input_ids"]) <= budget:
            return [(0, len(text))]
        windows: list[tuple[int, int]] = []
        for match in _SENTENCE_RE.finditer(text):
            start, end = match.span()
            merged = text[windows[-1][0] : end] if windows else ""
            if windows and len(self.tokenizer(merged)["input_ids"]) <= budget:
                windows[-1] = (windows[-1][0], end)
            else:
                windows.append((start, end))
        return windows

    def _extract_spans(self, text: str, windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
        chunks = [text[s:e] for s, e in windows]
        spans: list[tuple[int, int]] = []
        for group in self._batches(len(chunks)):
            enc = self._encode([chunks[i] for i in group], return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping").tolist()
            pred_ids = self._logits(self.ate_model, enc).argmax(-1).tolist()
            for row, i in enumerate(group):
                for start, end in decode_bio(enc.word_ids(row), offsets[row], pred_ids[row]):
                    start, end = trim_span(chunks[i], start, end)
                    if end > start:
                        spans.append((windows[i][0] + start, windows[i][0] + end))
        return spans

    def _classify(
        self,
        text: str,
        windows: list[tuple[int, int]],
        aspects: list[tuple[str, tuple[int, int]]],
    ) -> list[tuple[Polarity, float]]:
        """Sentiment per aspect (batched for fp32 models, see ``_batches``)."""
        if not aspects:
            return []
        import torch

        marked: list[str] = []
        for term, (start, end) in aspects:
            w_start, w_end = next((w for w in windows if w[0] <= start < w[1]), windows[0])
            span = (start - w_start, end - w_start)
            marked.append(mark_aspect(text[w_start:w_end], term, span, self.markers))
        logits = torch.cat(
            [
                self._logits(self.asc_model, self._encode([marked[i] for i in group]))
                for group in self._batches(len(marked))
            ]
        )
        confidence, idx = torch.softmax(logits, dim=-1).max(dim=-1)
        return [
            (ASC_LABELS[i], float(c))
            for i, c in zip(idx.tolist(), confidence.tolist(), strict=True)
        ]

    def _category_scores(
        self, text: str, windows: list[tuple[int, int]]
    ) -> list[tuple[str, float]]:
        if self.acd_model is None or not self.categories:
            return []
        import torch

        chunks = [text[s:e] for s, e in windows]
        logits = torch.cat(
            [
                self._logits(self.acd_model, self._encode([chunks[i] for i in group]))
                for group in self._batches(len(chunks))
            ]
        )
        probs = torch.sigmoid(logits).max(dim=0).values.tolist()
        return list(zip(self.categories, probs, strict=True))

    def predict(self, text: str) -> list[AspectPrediction]:
        view = clean_text_aligned(text)
        if not view.text:
            return []
        windows = self._windows(view.text)
        spans = self._extract_spans(view.text, windows)
        labels = self._classify(view.text, windows, [(view.text[s:e], (s, e)) for s, e in spans])
        predictions: list[AspectPrediction] = []
        for (start, end), (sentiment, confidence) in zip(spans, labels, strict=True):
            raw_start, raw_end = view.to_raw(start, end)
            predictions.append(
                AspectPrediction(
                    aspect=view.raw[raw_start:raw_end],
                    sentiment=sentiment,
                    confidence=confidence,
                    start=raw_start,
                    end=raw_end,
                )
            )
        return predictions

    def predict_batch(self, texts: list[str]) -> list[list[AspectPrediction]]:
        return [self.predict(t) for t in texts]

    def category_scores(self, text: str) -> list[tuple[str, float]]:
        """Probability for every category, before thresholding."""
        cleaned = clean_text(text)
        return self._category_scores(cleaned, self._windows(cleaned)) if cleaned else []

    def predict_categories(self, text: str) -> list[tuple[str, float]]:
        scores = self.category_scores(text)
        return sorted(((c, p) for c, p in scores if p >= self.acd_threshold), key=lambda cp: -cp[1])

    # --- EvaluablePipeline interface (uniform scoring across tracks) ---
    # Spans in and out are offsets into the raw ``text``, like gold annotations.
    @property
    def category_labels(self) -> list[str]:
        return list(self.categories)

    def extract_spans(self, text: str) -> list[tuple[int, int, str]]:
        view = clean_text_aligned(text)
        if not view.text:
            return []
        spans = self._extract_spans(view.text, self._windows(view.text))
        raw_spans = [view.to_raw(s, e) for s, e in spans]
        return [(s, e, view.raw[s:e]) for s, e in raw_spans]

    def classify_aspect(
        self, text: str, term: str, span: tuple[int, int]
    ) -> tuple[Polarity, float]:
        view = clean_text_aligned(text)
        clean_span = view.to_clean(*span)
        surface = view.text[clean_span[0] : clean_span[1]] if clean_span else term
        windows = self._windows(view.text)
        return self._classify(view.text, windows, [(surface, clean_span or (-1, -1))])[0]

    @classmethod
    def load(cls, directory: str | Path, device: str | None = None) -> TransformerABSA:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoTokenizer,
        )

        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        quantized = bool(meta.get("quantized", False))
        if quantized and (directory / "ate.pt").exists():
            raise RuntimeError(
                f"{directory} is an old pickled int8 artifact; re-create it with `make export`."
            )
        if device is None:
            # int8 dynamic-quantized models are CPU-only.
            device = "cpu" if quantized else ("cuda" if torch.cuda.is_available() else "cpu")

        models: dict[str, Any] = {}
        for task, head in TASK_HEADS.items():
            task_dir = directory / task
            if not task_dir.exists():
                models[task] = None
            elif quantized:
                models[task] = load_int8(task_dir, head)
            elif head == "token":
                models[task] = AutoModelForTokenClassification.from_pretrained(task_dir)
            else:
                models[task] = AutoModelForSequenceClassification.from_pretrained(task_dir)
        if models["ate"] is None or models["asc"] is None:
            raise FileNotFoundError(f"{directory} is missing the ate/ or asc/ model")

        return cls(
            tokenizer=AutoTokenizer.from_pretrained(directory / "tokenizer"),
            ate_model=models["ate"],
            asc_model=models["asc"],
            acd_model=models["acd"],
            categories=meta.get("categories", []),
            max_length=int(meta.get("max_length", 128)),
            markers=tuple(meta.get("markers", ["[ASP]", "[/ASP]"])),
            device=device,
            quantized=quantized,
            acd_threshold=float(meta.get("acd_threshold", 0.5)),
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
    "decode_bio",
    "load_int8",
    "mark_aspect",
    "quantize_int8",
    "resolve_tokenizer",
    "save_int8",
    "trim_span",
]
