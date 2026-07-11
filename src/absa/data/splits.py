"""Deterministic, seeded, stratified train/val/test splitting (stdlib only)."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from absa.data.schema import ABSAExample


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def sentence_label(example: ABSAExample) -> str:
    """Stratification label: majority term polarity, else category, else 'none'."""
    polarities = [t.polarity for t in example.aspect_terms]
    if not polarities:
        polarities = [c.polarity for c in example.aspect_categories]
    if not polarities:
        return "none"
    return Counter(polarities).most_common(1)[0][0]


def make_splits(
    examples: list[ABSAExample],
    ratios: SplitRatios | None = None,
    *,
    seed: int = 42,
    stratify: bool = True,
) -> dict[str, list[ABSAExample]]:
    """Split into train/val/test, deterministic per seed, optionally stratified."""
    ratios = ratios or SplitRatios()
    rng = random.Random(seed)

    groups: dict[str, list[ABSAExample]] = defaultdict(list)
    if stratify:
        for ex in examples:
            groups[sentence_label(ex)].append(ex)
    else:
        groups["_all"] = list(examples)

    out: dict[str, list[ABSAExample]] = {"train": [], "val": [], "test": []}
    for _label, items in sorted(groups.items()):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * ratios.train))
        n_val = int(round(n * ratios.val))
        # Ensure everything is assigned; test gets the remainder.
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        out["train"].extend(items[:n_train])
        out["val"].extend(items[n_train : n_train + n_val])
        out["test"].extend(items[n_train + n_val :])

    for key in out:
        rng.shuffle(out[key])
    return out


__all__ = ["SplitRatios", "make_splits", "sentence_label"]
