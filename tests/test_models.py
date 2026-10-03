"""Tests for Track A (baseline models) and the metric functions.

Skipped entirely when the classical stack (sklearn / sklearn-crfsuite / spaCy)
is not installed, so the light CI import still passes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("sklearn_crfsuite")
pytest.importorskip("spacy")

from absa.data import load_sample  # noqa: E402
from absa.models.base import AspectPrediction  # noqa: E402
from absa.models.baseline import BaselineABSA, _get_nlp  # noqa: E402
from absa.training.evaluation import evaluate_pipeline, tune_threshold  # noqa: E402
from absa.training.metrics import (  # noqa: E402
    classification_metrics,
    multilabel_prf,
    span_prf,
    summarize_runs,
)


@pytest.fixture(scope="module")
def trained() -> tuple[BaselineABSA, list]:
    try:
        _get_nlp()
    except OSError:
        pytest.skip("en_core_web_sm model not installed")
    examples = load_sample()
    model = BaselineABSA().fit(examples)
    return model, examples


# pipeline
def test_predict_returns_valid_predictions(trained: tuple[BaselineABSA, list]) -> None:
    model, _ = trained
    preds = model.predict("The pizza was great but the service was slow.")
    assert isinstance(preds, list)
    for p in preds:
        assert isinstance(p, AspectPrediction)
        assert p.sentiment in {"positive", "negative", "neutral"}
        assert 0.0 <= p.confidence <= 1.0
        assert p.start >= 0 and p.end > p.start
        assert p.to_dict()["aspect"] == p.aspect


def test_predict_spans_index_the_raw_text(trained: tuple[BaselineABSA, list]) -> None:
    # Cleaning collapses whitespace; spans must still point into the caller's text.
    model, _ = trained
    text = "Great place.\n\n  The pizza was great  but the service\u00a0was slow."
    preds = model.predict(text)
    assert preds
    for p in preds:
        assert text[p.start : p.end] == p.aspect
    assert [text[s:e] for s, e, _ in model.extract_spans(text)] == [p.aspect for p in preds]


def test_classify_aspect_accepts_raw_spans(trained: tuple[BaselineABSA, list]) -> None:
    model, _ = trained
    text = "The   pizza was great"
    label, conf = model.classify_aspect(text, "pizza", (6, 11))
    assert label in {"positive", "negative", "neutral"}
    assert 0.0 <= conf <= 1.0


def test_predict_categories_sentence_level(trained: tuple[BaselineABSA, list]) -> None:
    model, _ = trained
    cats = model.predict_categories("The food was delicious and cheap")
    assert all(c in model.category_labels and 0.5 <= p <= 1.0 for c, p in cats)


def test_predict_handles_empty_and_whitespace(trained: tuple[BaselineABSA, list]) -> None:
    model, _ = trained
    assert model.predict("") == []
    assert model.predict("    ") == []


def test_predict_batch(trained: tuple[BaselineABSA, list]) -> None:
    model, _ = trained
    out = model.predict_batch(["Great food", "Terrible service"])
    assert len(out) == 2


def test_asc_confidence_bounds(trained: tuple[BaselineABSA, list]) -> None:
    model, _ = trained
    label, conf = model.asc.predict("The battery life is excellent", "battery life")
    assert label in {"positive", "negative", "neutral"}
    assert 0.0 <= conf <= 1.0


def test_acd_predict_returns_list(trained: tuple[BaselineABSA, list]) -> None:
    model, _ = trained
    cats = model.acd.predict("The food was delicious and cheap")
    assert isinstance(cats, list)
    assert all(isinstance(c, str) and 0.0 <= s <= 1.0 for c, s in cats)


def test_save_load_roundtrip(trained: tuple[BaselineABSA, list], tmp_path: Path) -> None:
    model, _ = trained
    model.save(tmp_path)
    assert (tmp_path / "baseline.joblib").exists()
    assert (tmp_path / "meta.json").exists()
    loaded = BaselineABSA.load(tmp_path)
    text = "Great sushi and friendly staff"
    assert [p.aspect for p in model.predict(text)] == [p.aspect for p in loaded.predict(text)]


def test_evaluate_pipeline_shape(trained: tuple[BaselineABSA, list]) -> None:
    model, examples = trained
    result = evaluate_pipeline(model, examples[:20])
    assert set(result) >= {"ate", "acd", "asc", "counts"}
    assert "f1" in result["ate"]
    assert "macro_f1" in result["asc"]


def test_evaluate_pipeline_by_domain(trained: tuple[BaselineABSA, list]) -> None:
    model, examples = trained
    subset = examples[:30]
    result = evaluate_pipeline(model, subset)
    domains = {ex.domain for ex in subset}
    if len(domains) < 2:
        pytest.skip("sample slice covers a single domain")
    by_domain = result["by_domain"]
    assert set(by_domain) == domains  # type: ignore[arg-type]
    total = sum(d["counts"]["aspect_terms"] for d in by_domain.values())  # type: ignore[union-attr]
    assert total == result["counts"]["aspect_terms"]  # type: ignore[index]


# metrics
def test_span_prf_perfect() -> None:
    assert span_prf([[(0, 5)]], [[(0, 5)]])["f1"] == 1.0


def test_span_prf_partial() -> None:
    r = span_prf([[(0, 5), (6, 9)]], [[(0, 5)]])
    assert r["tp"] == 1 and r["fn"] == 1 and r["fp"] == 0


def test_multilabel_prf() -> None:
    r = multilabel_prf([{"food"}, {"service"}], [{"food"}, {"food"}], ["food", "service"])
    per_label = r["per_label"]
    assert per_label["food"]["tp"] == 1  # type: ignore[index]
    assert 0.0 <= r["micro_f1"] <= 1.0  # type: ignore[operator]


def test_tune_threshold_prefers_best_micro_f1() -> None:
    labels = ["food", "service"]
    # Scores are systematically low (like an int8 model): 0.3 recovers every label.
    scored = [
        ({"food"}, [("food", 0.35), ("service", 0.05)]),
        ({"service"}, [("food", 0.10), ("service", 0.40)]),
        ({"food", "service"}, [("food", 0.32), ("service", 0.31)]),
    ]
    best, curve = tune_threshold(scored, labels)
    assert best == 0.3
    assert curve[0.3] == pytest.approx(1.0)
    assert curve[0.5] == 0.0


def test_tune_threshold_tie_goes_to_half() -> None:
    scored = [({"food"}, [("food", 0.9)])]
    best, _ = tune_threshold(scored, ["food"])
    assert best == 0.5


def test_summarize_runs_mean_and_std() -> None:
    runs = [
        {"asc": {"macro_f1": 0.80}, "ate": {"f1": 0.9}},
        {"asc": {"macro_f1": 0.82}, "ate": {"f1": 0.9}},
        {"asc": {"macro_f1": 0.78}, "acd": {"note": "no category supervision"}},
    ]
    summary = summarize_runs(runs)
    assert summary["asc.macro_f1"]["mean"] == pytest.approx(0.80)
    assert summary["asc.macro_f1"]["std"] == pytest.approx(0.02)
    assert summary["ate.f1"]["n"] == 2
    assert "acd.micro_f1" not in summary


def test_classification_metrics() -> None:
    m = classification_metrics(
        ["positive", "negative", "neutral"], ["positive", "negative", "positive"]
    )
    assert 0.0 <= m["macro_f1"] <= 1.0  # type: ignore[operator]
    assert m["labels"] == ["negative", "neutral", "positive"]
    assert "confusion_matrix" in m
