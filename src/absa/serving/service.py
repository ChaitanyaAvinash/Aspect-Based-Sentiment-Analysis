"""Model service: resolves, loads, warms up, and serves a pipeline.

Resolution order (best available wins), all on CPU by default (the deployment
target). If nothing is on disk it bootstraps a baseline on the committed sample
so the API always boots — useful for CI and first-run demos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from absa.config import Settings
from absa.logging import get_logger
from absa.models.base import ABSAPipeline, AspectPrediction

log = get_logger("service")


def _serving_device(settings: Settings) -> str:
    return "cuda" if settings.device == "cuda" else "cpu"


@dataclass
class ModelService:
    """Holds the loaded pipeline and metadata used by the API."""

    model: ABSAPipeline
    track: str
    device: str

    @classmethod
    def load(cls, settings: Settings) -> ModelService:
        device = _serving_device(settings)

        quantized = settings.resolve(settings.artifact_path)
        transformer_dir = settings.resolve("artifacts/transformer")
        baseline_dir = settings.resolve("artifacts/baseline")

        prefer_baseline = settings.default_track == "baseline"
        if not prefer_baseline and (quantized / "meta.json").exists():
            return cls._load_transformer(quantized, device, "transformer-int8")
        if not prefer_baseline and (transformer_dir / "meta.json").exists():
            return cls._load_transformer(transformer_dir, device, "transformer")
        if (baseline_dir / "baseline.joblib").exists():
            return cls._load_baseline(baseline_dir)
        if prefer_baseline and (transformer_dir / "meta.json").exists():
            return cls._load_transformer(transformer_dir, device, "transformer")
        return cls.bootstrap_sample()

    @classmethod
    def _load_transformer(cls, directory: Path, device: str, track: str) -> ModelService:
        from absa.models.transformer import TransformerABSA

        log.info("loading_transformer", dir=str(directory), device=device)
        return cls(model=TransformerABSA.load(directory, device=device), track=track, device=device)

    @classmethod
    def _load_baseline(cls, directory: Path) -> ModelService:
        from absa.models.baseline import BaselineABSA

        log.info("loading_baseline", dir=str(directory))
        return cls(model=BaselineABSA.load(directory), track="baseline", device="cpu")

    @classmethod
    def bootstrap_sample(cls) -> ModelService:
        """Train a baseline on the committed sample (offline fallback)."""
        from absa.data.io import load_sample
        from absa.models.baseline import BaselineABSA

        log.warning("bootstrapping_from_sample", note="no trained artifact found")
        model = BaselineABSA().fit(load_sample())
        return cls(model=model, track="baseline-sample", device="cpu")

    def warmup(self) -> None:
        """Run one inference so the first real request isn't slow."""
        try:
            self.model.predict("The service was great and the food was tasty.")
            log.info("warmup_complete", track=self.track, device=self.device)
        except Exception as exc:  # warmup must never crash startup
            log.warning("warmup_failed", error=str(exc)[:160])

    def predict(self, text: str) -> list[AspectPrediction]:
        return self.model.predict(text)

    def predict_batch(self, texts: list[str]) -> list[list[AspectPrediction]]:
        return self.model.predict_batch(texts)


__all__ = ["ModelService"]
