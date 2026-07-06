"""Tests for absa.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from absa.config import PROJECT_ROOT, Settings, get_settings, load_yaml_config


def test_default_settings() -> None:
    s = Settings()
    assert s.environment == "development"
    assert s.seed == 42
    assert s.max_length == 128
    assert s.default_track in {"baseline", "transformer"}


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ABSA_SEED", "7")
    monkeypatch.setenv("ABSA_ENVIRONMENT", "test")
    get_settings.cache_clear()
    s = get_settings()
    assert s.seed == 7
    assert s.environment == "test"


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_resolve_relative_path() -> None:
    resolved = Settings().resolve(Path("data"))
    assert resolved.is_absolute()
    assert resolved == PROJECT_ROOT / "data"


def test_resolve_absolute_path(tmp_path: Path) -> None:
    assert Settings().resolve(tmp_path) == tmp_path


def test_max_length_bounds() -> None:
    with pytest.raises(ValueError):
        Settings(max_length=4)


def test_load_yaml_config(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\nb: [1, 2, 3]\n", encoding="utf-8")
    cfg = load_yaml_config(p)
    assert cfg["a"] == 1
    assert cfg["b"] == [1, 2, 3]


def test_load_yaml_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml_config(tmp_path / "nope.yaml")


def test_load_yaml_not_mapping(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml_config(p)


@pytest.mark.parametrize("name", ["data.yaml", "model.yaml", "training.yaml"])
def test_shipped_configs_load(name: str) -> None:
    cfg = load_yaml_config(Path("configs") / name)
    assert isinstance(cfg, dict)
    assert cfg  # non-empty
