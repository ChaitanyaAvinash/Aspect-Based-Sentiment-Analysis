"""Tests for absa.logging."""

from __future__ import annotations

import pytest

from absa.logging import configure_logging, get_logger


def test_configure_json_and_emit(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", json_logs=True)
    get_logger("test").info("hello", key="value")
    out = capsys.readouterr().out
    assert "hello" in out
    assert "value" in out


def test_configure_console(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="DEBUG", json_logs=False)
    get_logger("test").warning("watch-out")
    out = capsys.readouterr().out
    assert "watch-out" in out


def test_get_logger_returns_object() -> None:
    assert get_logger("x") is not None
