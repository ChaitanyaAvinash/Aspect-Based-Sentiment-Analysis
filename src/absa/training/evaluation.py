"""Uniform ATE/ACD/ASC scoring for any pipeline: ASC/ACD on gold, ATE on predicted spans.

Every model is queried once per example; scores are then computed over the whole
set and per domain (SemEval results are usually reported per domain).
"""

from __future__ import annotations

from dataclasses import dataclass

from absa.data.schema import ABSAExample
from absa.models.base import EvaluablePipeline
from absa.training.metrics import classification_metrics, multilabel_prf, span_prf


@dataclass
class _Predictions:
    spans: list[list[tuple[int, int]]]
    sentiments: list[list[str]]
    categories: list[set[str]]


def _predict(model: EvaluablePipeline, examples: list[ABSAExample]) -> _Predictions:
    has_categories = bool(model.category_labels)
    return _Predictions(
        spans=[[(s, e) for s, e, _ in model.extract_spans(ex.text)] for ex in examples],
        sentiments=[
            [model.classify_aspect(ex.text, t.term, t.span)[0] for t in ex.aspect_terms]
            for ex in examples
        ],
        categories=[
            (
                {c for c, _ in model.predict_categories(ex.text)}
                if has_categories and ex.aspect_categories
                else set()
            )
            for ex in examples
        ],
    )


def _score_ate(examples: list[ABSAExample], pred: _Predictions) -> dict[str, float]:
    gold = [[t.span for t in ex.aspect_terms if t.is_explicit] for ex in examples]
    return span_prf(gold, pred.spans)


def _score_asc(examples: list[ABSAExample], pred: _Predictions) -> dict[str, object]:
    y_true = [t.polarity for ex in examples for t in ex.aspect_terms]
    y_pred = [label for labels in pred.sentiments for label in labels]
    return classification_metrics(y_true, y_pred)


def _score_acd(
    examples: list[ABSAExample], pred: _Predictions, labels: list[str]
) -> dict[str, object]:
    if not labels:
        return {"note": "no category supervision"}
    scored = [i for i, ex in enumerate(examples) if ex.aspect_categories]
    if not scored:
        return {"note": "no category annotations"}
    gold = [{c.category for c in examples[i].aspect_categories} for i in scored]
    result = multilabel_prf(gold, [pred.categories[i] for i in scored], labels)
    result["num_scored"] = len(scored)
    return result


def _score(examples: list[ABSAExample], pred: _Predictions, labels: list[str]) -> dict[str, object]:
    return {
        "counts": {
            "examples": len(examples),
            "aspect_terms": sum(len(ex.aspect_terms) for ex in examples),
            "with_categories": sum(1 for ex in examples if ex.aspect_categories),
        },
        "ate": _score_ate(examples, pred),
        "acd": _score_acd(examples, pred, labels),
        "asc": _score_asc(examples, pred),
    }


def _subset(pred: _Predictions, idx: list[int]) -> _Predictions:
    return _Predictions(
        spans=[pred.spans[i] for i in idx],
        sentiments=[pred.sentiments[i] for i in idx],
        categories=[pred.categories[i] for i in idx],
    )


def evaluate_pipeline(model: EvaluablePipeline, examples: list[ABSAExample]) -> dict[str, object]:
    """Overall scores, plus ``by_domain`` when the examples span several domains."""
    labels = model.category_labels
    pred = _predict(model, examples)
    result = _score(examples, pred, labels)
    domains = sorted({ex.domain for ex in examples})
    if len(domains) > 1:
        by_domain: dict[str, object] = {}
        for domain in domains:
            idx = [i for i, ex in enumerate(examples) if ex.domain == domain]
            by_domain[domain] = _score([examples[i] for i in idx], _subset(pred, idx), labels)
        result["by_domain"] = by_domain
    return result


THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(2, 13))


def tune_threshold(
    scored: list[tuple[set[str], list[tuple[str, float]]]],
    labels: list[str],
    grid: tuple[float, ...] = THRESHOLD_GRID,
) -> tuple[float, dict[float, float]]:
    """Pick the ACD cutoff maximizing micro-F1 over (gold, scores) pairs.

    Use validation data only. Ties go to the value closest to 0.5.
    Returns (best threshold, micro-F1 per threshold).
    """
    gold = [g for g, _ in scored]
    curve: dict[float, float] = {}
    for t in grid:
        pred = [{c for c, p in scores if p >= t} for _, scores in scored]
        curve[t] = float(multilabel_prf(gold, pred, labels)["micro_f1"])  # type: ignore[arg-type]
    best = max(curve, key=lambda t: (curve[t], -abs(t - 0.5)))
    return best, curve


__all__ = ["THRESHOLD_GRID", "evaluate_pipeline", "tune_threshold"]
