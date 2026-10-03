"""Tests for the FastAPI serving layer (via a stub model, no heavy load)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from absa.models.base import AspectPrediction  # noqa: E402
from absa.serving.app import create_app  # noqa: E402
from absa.serving.service import ModelService  # noqa: E402


class _StubModel:
    def predict(self, text: str) -> list[AspectPrediction]:
        if not text.strip():
            return []
        return [
            AspectPrediction(
                aspect="pizza",
                sentiment="positive",
                confidence=0.95,
                start=0,
                end=5,
            )
        ]

    def predict_categories(self, text: str) -> list[tuple[str, float]]:
        return [("food", 0.9)] if text.strip() else []

    def predict_batch(self, texts: list[str]) -> list[list[AspectPrediction]]:
        return [self.predict(t) for t in texts]


@pytest.fixture
def client() -> Iterator[TestClient]:
    service = ModelService(model=_StubModel(), track="stub", device="cpu")
    with TestClient(create_app(service)) as c:
        yield c


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["track"] == "stub"
    assert body["model_loaded"] is True


def test_predict(client: TestClient) -> None:
    resp = client.post("/predict", json={"text": "pizza was great"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    aspect = body["aspects"][0]
    assert aspect["sentiment"] == "positive"
    assert aspect["span"] == [0, 5]
    assert 0.0 <= aspect["confidence"] <= 1.0
    assert "category" not in aspect
    assert body["categories"] == [{"category": "food", "confidence": 0.9}]


def test_predict_empty_text_is_graceful(client: TestClient) -> None:
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 200
    assert resp.json()["aspects"] == []
    assert resp.json()["categories"] == []


def test_predict_batch(client: TestClient) -> None:
    resp = client.post("/predict/batch", json={"texts": ["good food", "bad service"]})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2


def test_batch_requires_at_least_one_text(client: TestClient) -> None:
    resp = client.post("/predict/batch", json={"texts": []})
    assert resp.status_code == 422


def test_openapi_docs(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert "/predict/batch" in schema["paths"]


def test_bootstrap_sample_service() -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("sklearn_crfsuite")
    pytest.importorskip("spacy")
    from absa.models.baseline import _get_nlp

    try:
        _get_nlp()
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    service = ModelService.bootstrap_sample()
    service.warmup()
    assert service.track == "baseline-sample"
    assert isinstance(service.predict("The food was great"), list)
