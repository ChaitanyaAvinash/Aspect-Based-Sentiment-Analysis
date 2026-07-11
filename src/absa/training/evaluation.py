"""Uniform ATE/ACD/ASC scoring for any pipeline: ASC/ACD on gold, ATE on predicted spans."""

from __future__ import annotations

from absa.data.schema import ABSAExample
from absa.models.base import EvaluablePipeline
from absa.training.metrics import classification_metrics, multilabel_prf, span_prf


def evaluate_ate(model: EvaluablePipeline, examples: list[ABSAExample]) -> dict[str, float]:
    gold = [[t.span for t in ex.aspect_terms if t.is_explicit] for ex in examples]
    pred = [[(s, e) for s, e, _ in model.extract_spans(ex.text)] for ex in examples]
    return span_prf(gold, pred)


def evaluate_asc(model: EvaluablePipeline, examples: list[ABSAExample]) -> dict[str, object]:
    y_true: list[str] = []
    y_pred: list[str] = []
    for ex in examples:
        for term in ex.aspect_terms:
            label, _ = model.classify_aspect(ex.text, term.term, term.span)
            y_true.append(term.polarity)
            y_pred.append(label)
    return classification_metrics(y_true, y_pred)


def evaluate_acd(model: EvaluablePipeline, examples: list[ABSAExample]) -> dict[str, object]:
    labels = model.category_labels
    if not labels:
        return {"note": "no category supervision"}
    scored = [ex for ex in examples if ex.aspect_categories]
    gold = [{c.category for c in ex.aspect_categories} for ex in scored]
    pred = [{c for c, _ in model.predict_categories(ex.text)} for ex in scored]
    result = multilabel_prf(gold, pred, labels)
    result["num_scored"] = len(scored)
    return result


def evaluate_pipeline(model: EvaluablePipeline, examples: list[ABSAExample]) -> dict[str, object]:
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
