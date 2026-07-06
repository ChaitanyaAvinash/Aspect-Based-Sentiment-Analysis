"""Prepare ABSA datasets -> data/processed/{train,val,test}.jsonl (+ stats.json).

Source resolution (``--source auto``):
  1. Official SemEval XML in data/raw/  (most authoritative; uses gold test split).
  2. Hugging Face mirror   (if online and not ``--offline``).
  3. Committed sample set  (always works, fully offline).

Examples:
  python scripts/prepare_data.py                 # auto (xml -> hf -> sample)
  python scripts/prepare_data.py --source sample # force the offline sample
  python scripts/prepare_data.py --offline       # skip the HF download
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from absa.config import PROJECT_ROOT, get_settings, load_yaml_config
from absa.data.io import load_sample, write_jsonl
from absa.data.schema import ABSAExample, AspectCategory, AspectTerm
from absa.data.semeval import parse_semeval_xml
from absa.data.splits import SplitRatios, make_splits
from absa.data.stats import summarize_splits
from absa.logging import configure_logging, get_logger

log = get_logger("prepare_data")

_POLARITY_ALIASES = {
    "positive": "positive",
    "pos": "positive",
    "2": "positive",
    "negative": "negative",
    "neg": "negative",
    "0": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "1": "neutral",
}


def _domain_from_name(name: str) -> str:
    low = name.lower()
    if "rest" in low:
        return "restaurants"
    if "laptop" in low:
        return "laptops"
    return "other"


# --------------------------------------------------------------------------- #
# Source 1: official SemEval XML in data/raw/
# --------------------------------------------------------------------------- #
def _load_from_xml(
    raw_dir: Path, drop_conflict: bool
) -> tuple[list[ABSAExample], list[ABSAExample]]:
    """Return (train_pool, official_test). test-gold files feed the test split."""
    train_pool: list[ABSAExample] = []
    official_test: list[ABSAExample] = []
    for xml_path in sorted(raw_dir.glob("*.xml")):
        domain = _domain_from_name(xml_path.name)
        examples = parse_semeval_xml(xml_path, domain=domain, drop_conflict=drop_conflict)  # type: ignore[arg-type]
        is_test = any(k in xml_path.name.lower() for k in ("test", "gold"))
        (official_test if is_test else train_pool).extend(examples)
        log.info("parsed_xml", file=xml_path.name, n=len(examples), is_test=is_test)
    return train_pool, official_test


# --------------------------------------------------------------------------- #
# Source 2: Hugging Face mirror
# --------------------------------------------------------------------------- #
def _domain_from_config(name: str, cfg: str | None) -> str:
    return _domain_from_name(f"{name} {cfg or ''}")


def _map_polarity(raw: Any, drop_conflict: bool) -> str | None:
    if raw is None:
        return None
    p = str(raw).strip().lower()
    if p == "conflict":
        return None if drop_conflict else "neutral"
    return _POLARITY_ALIASES.get(p)


def _seq(container: Any, key: str) -> list[Any]:
    """Read a parallel-array field from a datasets Sequence-of-struct row."""
    if isinstance(container, dict):
        val = container.get(key)
        return list(val) if val else []
    return []


def _structured_to_examples(ds: Any, domain: str, drop_conflict: bool) -> list[ABSAExample]:
    """Adapt the SemEval mirror schema (aspects[] + category[] sequences)."""
    examples: list[ABSAExample] = []
    for i, row in enumerate(ds):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        aspects = row.get("aspects") or {}
        terms, pols = _seq(aspects, "term"), _seq(aspects, "polarity")
        froms, tos = _seq(aspects, "from"), _seq(aspects, "to")
        aspect_terms: list[AspectTerm] = []
        for j, raw_term in enumerate(terms):
            pol = _map_polarity(pols[j] if j < len(pols) else None, drop_conflict)
            term = str(raw_term).strip()
            if pol is None or not term or term.upper() == "NULL":
                continue
            start = int(froms[j]) if j < len(froms) else -1
            end = int(tos[j]) if j < len(tos) else -1
            if not (0 <= start < end <= len(text) and text[start:end] == term):
                idx = text.find(term)
                start, end = (idx, idx + len(term)) if idx >= 0 else (-1, -1)
            aspect_terms.append(AspectTerm(term=term, polarity=pol, start=start, end=end))  # type: ignore[arg-type]

        cat_field = row.get("category") or {}
        cats, cpols = _seq(cat_field, "category"), _seq(cat_field, "polarity")
        aspect_categories: list[AspectCategory] = []
        for j, raw_cat in enumerate(cats):
            pol = _map_polarity(cpols[j] if j < len(cpols) else None, drop_conflict)
            cat = str(raw_cat).strip()
            if pol is None or not cat:
                continue
            aspect_categories.append(AspectCategory(category=cat, polarity=pol))  # type: ignore[arg-type]

        rid = str(row.get("id") or f"{domain}-{i:05d}")
        examples.append(
            ABSAExample(
                id=rid,
                text=text,
                domain=domain,  # type: ignore[arg-type]
                aspect_terms=aspect_terms,
                aspect_categories=aspect_categories,
            )
        )
    return examples


def _first_col(cols: set[str], candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in cols), None)


def _decode_polarity(value: Any, class_names: list[str] | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, int) and class_names is not None and 0 <= value < len(class_names):
        value = class_names[value]
    return _POLARITY_ALIASES.get(str(value).strip().lower())


def _dataset_to_examples(ds: Any, domain: str) -> list[ABSAExample]:
    """Adapt a one-aspect-per-row HF dataset into grouped ABSAExamples."""
    cols = set(ds.column_names)
    text_col = _first_col(cols, ["text", "sentence", "Sentence", "review", "content"])
    if text_col is None:
        raise ValueError(f"no text column in {sorted(cols)}")
    term_col = _first_col(cols, ["aspect", "term", "aspectTerm", "target", "aspect_term"])
    pol_col = _first_col(cols, ["polarity", "sentiment", "label", "aspect_polarity"])
    cat_col = _first_col(cols, ["category", "aspectCategory", "aspect_category"])
    if pol_col is None:
        raise ValueError("no polarity column")

    class_names = None
    feat = ds.features.get(pol_col)
    if hasattr(feat, "names"):
        class_names = list(feat.names)

    grouped: dict[str, ABSAExample] = {}
    for row in ds:
        text = (row.get(text_col) or "").strip()
        if not text:
            continue
        polarity = _decode_polarity(row.get(pol_col), class_names)
        if polarity is None:
            continue
        ex = grouped.setdefault(
            text, ABSAExample(id=f"hf-{len(grouped):05d}", text=text, domain=domain)  # type: ignore[arg-type]
        )
        term = str(row.get(term_col) or "").strip() if term_col else ""
        if term and term.upper() != "NULL":
            start = text.find(term)
            ex.aspect_terms.append(
                AspectTerm(
                    term=term,
                    polarity=polarity,  # type: ignore[arg-type]
                    start=start,
                    end=start + len(term) if start >= 0 else -1,
                )
            )
        cat = str(row.get(cat_col) or "").strip() if cat_col else ""
        if cat:
            ex.aspect_categories.append(AspectCategory(category=cat, polarity=polarity))  # type: ignore[arg-type]
    return [ex for ex in grouped.values() if ex.aspect_terms or ex.aspect_categories]


def _is_test_split(split_name: str) -> bool:
    low = split_name.lower()
    return "test" in low or "valid" in low or low == "dev"


def _load_from_hf(
    candidates: list[str], drop_conflict: bool
) -> tuple[list[ABSAExample], list[ABSAExample]]:
    """Return (train_pool, official_test) from the first mirror that loads."""
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError:
        log.warning("datasets_not_installed")
        return [], []

    for name in candidates:
        train_pool: list[ABSAExample] = []
        official_test: list[ABSAExample] = []
        try:
            configs = get_dataset_config_names(name)
        except Exception as exc:  # gated / needs trust_remote_code / offline
            log.warning("hf_configs_failed", dataset=name, error=str(exc)[:160])
            configs = []
        for cfg in configs or [None]:
            try:
                dsd = load_dataset(name, cfg) if cfg else load_dataset(name)
            except Exception as exc:
                log.warning("hf_failed", dataset=name, config=cfg, error=str(exc)[:160])
                continue
            domain = _domain_from_config(name, cfg)
            for split_name, ds in dsd.items():
                cols = set(ds.column_names)
                if "aspects" in cols or "category" in cols:
                    exs = _structured_to_examples(ds, domain, drop_conflict)
                else:
                    exs = _dataset_to_examples(ds, domain)
                (official_test if _is_test_split(split_name) else train_pool).extend(exs)
                log.info("hf_split", dataset=name, config=cfg, split=split_name, n=len(exs))
        if train_pool or official_test:
            log.info("hf_loaded", dataset=name, train=len(train_pool), test=len(official_test))
            return train_pool, official_test
    return [], []


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _write_outputs(splits: dict[str, list[ABSAExample]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        n = write_jsonl(items, out_dir / f"{name}.jsonl")
        log.info("wrote_split", split=name, n=n)
    stats = summarize_splits(splits)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("wrote_stats", path=str(out_dir / "stats.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ABSA datasets.")
    parser.add_argument("--source", choices=["auto", "xml", "hf", "sample"], default="auto")
    parser.add_argument("--offline", action="store_true", help="skip the HF download")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()
    data_cfg = load_yaml_config("configs/data.yaml")
    seed = args.seed if args.seed is not None else settings.seed
    drop_conflict = bool(data_cfg["schema"].get("drop_conflict", True))
    split_cfg = data_cfg["splits"]
    ratios = SplitRatios(train=split_cfg["train"], val=split_cfg["val"], test=split_cfg["test"])
    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "processed"
    raw_dir = PROJECT_ROOT / "data" / "raw"

    train_pool: list[ABSAExample] = []
    official_test: list[ABSAExample] = []
    source_used = "sample"

    if args.source in ("auto", "xml") and any(raw_dir.glob("*.xml")):
        train_pool, official_test = _load_from_xml(raw_dir, drop_conflict)
        source_used = "xml"
    elif args.source == "hf" or (args.source == "auto" and not args.offline):
        candidates = list(data_cfg["dataset"].get("hf_candidates", []))
        train_pool, official_test = _load_from_hf(candidates, drop_conflict)
        if train_pool or official_test:
            source_used = "hf"

    if not train_pool and not official_test:
        log.info("falling_back_to_sample")
        train_pool = load_sample()
        source_used = "sample"

    if official_test:
        # Re-derive train/val from the pool; use gold test verbatim.
        tv_ratios = SplitRatios(
            train=ratios.train / (ratios.train + ratios.val),
            val=ratios.val / (ratios.train + ratios.val),
            test=0.0,
        )
        tv = make_splits(train_pool, tv_ratios, seed=seed)
        splits = {"train": tv["train"], "val": tv["val"], "test": official_test}
    else:
        splits = make_splits(train_pool, ratios, seed=seed)

    _write_outputs(splits, out_dir)
    log.info(
        "done",
        source=source_used,
        train=len(splits["train"]),
        val=len(splits["val"]),
        test=len(splits["test"]),
    )
    print(
        f"[prepare_data] source={source_used}  "
        f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}  "
        f"-> {out_dir}"
    )


if __name__ == "__main__":
    main()
