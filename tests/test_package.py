"""Smoke tests: package metadata and importability of all subpackages."""

from __future__ import annotations

import importlib

import pytest

import absa


def test_version_present() -> None:
    assert isinstance(absa.__version__, str)
    assert absa.__version__


@pytest.mark.parametrize(
    "module",
    [
        "absa.config",
        "absa.logging",
        "absa.data",
        "absa.models",
        "absa.training",
        "absa.serving",
    ],
)
def test_subpackages_importable(module: str) -> None:
    assert importlib.import_module(module) is not None
