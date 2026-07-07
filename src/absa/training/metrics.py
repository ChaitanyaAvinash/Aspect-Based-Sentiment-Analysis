"""Metrics for ATE (span exact-match), ACD (multi-label), ASC (classification)."""

from __future__ import annotations

from collections.abc import Sequence

ASC_LABELS: tuple[str, ...] = ("negative", "neutral", "positive")


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def span_prf(
    gold: Sequence[Sequence[tuple[int, int]]],
    pred: Sequence[Sequence[tuple[int, int]]],
) -> dict[str, float]:
    """Exact-match span P/R/F1 for ATE (SemEval-style)."""
    tp = fp = fn = 0
    for g, p in zip(gold, pred, strict=True):
        gset, pset = set(g), set(p)
        tp += len(gset & pset)
        fp += len(pset - gset)
        fn += len(gset - pset)
    return _prf(tp, fp, fn)


def multilabel_prf(
    gold: Sequence[set[str]],
    pred: Sequence[set[str]],
    labels: Sequence[str],
) -> dict[str, object]:
    """Micro/macro F1 for multi-label ACD, plus per-label P/R/F1."""
    per_label: dict[str, dict[str, float]] = {}
    micro_tp = micro_fp = micro_fn = 0
    macro_f1 = 0.0
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred, strict=True) if label in g and label in p)
        fp = sum(1 for g, p in zip(gold, pred, strict=True) if label not in g and label in p)
        fn = sum(1 for g, p in zip(gold, pred, strict=True) if label in g and label not in p)
        stats = _prf(tp, fp, fn)
        per_label[label] = stats
        micro_tp, micro_fp, micro_fn = micro_tp + tp, micro_fp + fp, micro_fn + fn
        macro_f1 += stats["f1"]
    micro = _prf(micro_tp, micro_fp, micro_fn)
    return {
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
        "macro_f1": macro_f1 / len(labels) if labels else 0.0,
        "per_label": per_label,
    }


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = ASC_LABELS,
) -> dict[str, object]:
    """Accuracy + macro P/R/F1 + per-class report + confusion matrix (ASC)."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    labels = list(labels)
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "labels": labels}
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "per_class": {lbl: report[lbl] for lbl in labels if lbl in report},
        "confusion_matrix": matrix,
        "labels": labels,
    }


__all__ = [
    "ASC_LABELS",
    "classification_metrics",
    "multilabel_prf",
    "span_prf",
]
