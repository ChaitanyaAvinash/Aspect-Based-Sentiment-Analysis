"""Corpus statistics: label distributions and counts for the dataset card."""

from __future__ import annotations

from collections import Counter

from absa.data.schema import ABSAExample


def label_distribution(examples: list[ABSAExample]) -> dict[str, object]:
    """Compute counts used by the dataset card and sanity checks."""
    term_polarity: Counter[str] = Counter()
    category_polarity: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    n_terms = 0
    n_categories = 0
    n_with_terms = 0

    for ex in examples:
        domains[ex.domain] += 1
        if ex.aspect_terms:
            n_with_terms += 1
        for t in ex.aspect_terms:
            term_polarity[t.polarity] += 1
            n_terms += 1
        for c in ex.aspect_categories:
            category_polarity[c.polarity] += 1
            categories[c.category] += 1
            n_categories += 1

    return {
        "num_examples": len(examples),
        "num_aspect_terms": n_terms,
        "num_aspect_categories": n_categories,
        "examples_with_terms": n_with_terms,
        "term_polarity": dict(term_polarity),
        "category_polarity": dict(category_polarity),
        "categories": dict(categories),
        "domains": dict(domains),
    }


def summarize_splits(splits: dict[str, list[ABSAExample]]) -> dict[str, dict[str, object]]:
    """Per-split label distributions."""
    return {name: label_distribution(items) for name, items in splits.items()}


__all__ = ["label_distribution", "summarize_splits"]
