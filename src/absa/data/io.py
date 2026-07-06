"""JSONL read/write for normalized examples, plus the committed sample loader."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from absa.config import PROJECT_ROOT
from absa.data.schema import ABSAExample

SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "reviews.jsonl"


def write_jsonl(examples: Iterable[ABSAExample], path: str | Path) -> int:
    """Write examples as JSON Lines. Returns the number written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(ex.model_dump_json())
            fh.write("\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[ABSAExample]:
    """Read a JSON Lines file of normalized examples."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {p}")
    examples: list[ABSAExample] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(ABSAExample.model_validate_json(line))
    return examples


def load_sample() -> list[ABSAExample]:
    """Load the committed tiny sample set (works fully offline)."""
    return read_jsonl(SAMPLE_PATH)


__all__ = ["SAMPLE_PATH", "load_sample", "read_jsonl", "write_jsonl"]
