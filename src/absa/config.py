"""Configuration for ABSA.

Two layers:

* :class:`Settings` — runtime/environment settings loaded from ``.env`` /
  environment variables (via ``pydantic-settings``). This is where secrets and
  machine-specific choices live. Never hard-code secrets in source.
* YAML configs under ``configs/`` — declarative experiment configuration
  (data, model, training). Loaded with :func:`load_yaml_config`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from src/absa/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Environment-sourced runtime settings (prefix ``ABSA_``)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ABSA_",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Runtime ---
    environment: str = "development"
    log_level: str = "INFO"
    log_json: bool = False
    seed: int = 42

    # --- Paths (relative to PROJECT_ROOT unless absolute) ---
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    configs_dir: Path = Path("configs")

    # --- Modeling / inference ---
    device: str = "auto"  # auto | cpu | cuda
    default_track: str = "transformer"  # baseline | transformer
    artifact_path: Path = Path("artifacts/absa-transformer-int8")
    max_length: int = Field(default=128, ge=8, le=512)

    # --- Experiment tracking ---
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "absa"

    def resolve(self, path: str | Path) -> Path:
        """Resolve *path* against the project root if it is relative."""
        p = Path(path)
        return p if p.is_absolute() else (PROJECT_ROOT / p)


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict.

    Relative paths are resolved against the project root. The document root
    must be a mapping.
    """
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}: {p}")
    return data


__all__ = ["PROJECT_ROOT", "Settings", "get_settings", "load_yaml_config"]
