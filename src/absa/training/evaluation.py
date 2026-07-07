"""Evaluate a fitted pipeline on the three sub-tasks (ATE / ACD / ASC).

ASC and ACD are scored against gold aspects/categories to isolate each
component from upstream extraction errors; ATE is scored on predicted vs gold
spans (exact match).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from absa.data.schema import ABSAExample
from absa.training.metrics import classification_metrics, multilabel_prf, span_prf

if TYPE_CHECKING:
    from absa.models.baseline import BaselineABSA


def evaluate_ate(model: BaselineABSA, examples: list[ABSAExample]) -> dict[str, float]:
    gold = [[t.span for t in ex.aspect_terms if t.is_explicit] for ex in examples]
    pred = [[(s, e) for s, e, _ in model.ate.predict_spans(ex.text)] for ex in examples]
    return span_prf(gold, pred)


def evaluate_asc(model: BaselineABSA, examples: list[ABSAExample]) -> dict[str, object]:
    y_true: list[str] = []
    y_pred: list[str] = []
    for ex in examples:
        for term in ex.aspect_terms:
            label, _ = model.asc.predict(ex.text, term.term)
            y_true.append(term.polarity)
            y_pred.append(label)
    return classification_metrics(y_true, y_pred)


def evaluate_acd(model: BaselineABSA, examples: list[ABSAExample]) -> dict[str, object]:
    if not model.acd.is_fitted:
        return {"note": "ACD not fitted (no category supervision)"}
    labels = [str(c) for c in model.acd.mlb.classes_]
    scored = [ex for ex in examples if ex.aspect_categories]
    gold = [{c.category for c in ex.aspect_categories} for ex in scored]
    pred = [{c for c, _ in model.acd.predict(ex.text)} for ex in scored]
    result = multilabel_prf(gold, pred, labels)
    result["num_scored"] = len(scored)
    return result


def evaluate_pipeline(model: BaselineABSA, examples: list[ABSAExample]) -> dict[str, object]:
    return {
        "counts": {
            "examples": len(examples),
            "aspect_terms": sum(len(ex.aspect_terms) for ex in examples),
            "with_categories": sum(1 for ex in examples if ex.aspect_categories),
        },
        "ate": evaluate_ate(model, examples),
        "acd": evaluate_acd(model, examples),
        "asc": evaluate_asc(model, examples),
    }


__all__ = ["evaluate_acd", "evaluate_asc", "evaluate_ate", "evaluate_pipeline"]
