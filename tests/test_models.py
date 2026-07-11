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
from absa.training.evaluation import evaluate_pipeline  # noqa: E402
from absa.training.metrics import (  # noqa: E402
    classification_metrics,
    multilabel_prf,
    span_prf,
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


def test_classification_metrics() -> None:
    m = classification_metrics(
        ["positive", "negative", "neutral"], ["positive", "negative", "positive"]
    )
    assert 0.0 <= m["macro_f1"] <= 1.0  # type: ignore[operator]
    assert m["labels"] == ["negative", "neutral", "positive"]
    assert "confusion_matrix" in m
