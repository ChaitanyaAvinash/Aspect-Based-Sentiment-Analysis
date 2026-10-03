"""FastAPI ABSA service: preload + warmup at startup; OpenAPI at /docs."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request

from absa.config import get_settings
from absa.logging import configure_logging, get_logger
from absa.serving.schemas import (
    AspectOut,
    BatchPredictRequest,
    BatchPredictResponse,
    CategoryOut,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from absa.serving.service import ModelService

log = get_logger("api")


def create_app(service: ModelService | None = None) -> FastAPI:
    """Build the FastAPI app. Inject ``service`` to skip auto-loading (tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        svc = service or ModelService.load(get_settings())
        svc.warmup()
        app.state.service = svc
        log.info("api_ready", track=svc.track, device=svc.device)
        yield

    app = FastAPI(
        title="ABSA API",
        version="0.1.0",
        description="Aspect-Based Sentiment Analysis — extract aspects, categories, "
        "and per-aspect sentiment from review text.",
        lifespan=lifespan,
    )

    def get_service(request: Request) -> ModelService:
        service: ModelService = request.app.state.service
        return service

    def _to_response(text: str, svc: ModelService) -> PredictResponse:
        aspects = [AspectOut.from_prediction(p) for p in svc.predict(text)]
        categories = [
            CategoryOut(category=c, confidence=round(p, 4)) for c, p in svc.predict_categories(text)
        ]
        return PredictResponse(
            text=text,
            track=svc.track,
            count=len(aspects),
            aspects=aspects,
            categories=categories,
        )

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(svc: ModelService = Depends(get_service)) -> HealthResponse:
        return HealthResponse(
            status="ok", track=svc.track, device=svc.device, model_loaded=svc.model is not None
        )

    @app.post("/predict", response_model=PredictResponse, tags=["absa"])
    def predict(req: PredictRequest, svc: ModelService = Depends(get_service)) -> PredictResponse:
        return _to_response(req.text, svc)

    @app.post("/predict/batch", response_model=BatchPredictResponse, tags=["absa"])
    def predict_batch(
        req: BatchPredictRequest, svc: ModelService = Depends(get_service)
    ) -> BatchPredictResponse:
        return BatchPredictResponse(results=[_to_response(t, svc) for t in req.texts])

    return app


app = create_app()

__all__ = ["app", "create_app"]
