"""Train a track (baseline|transformer) and write evaluation metrics."""

from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path

from absa.config import PROJECT_ROOT, get_settings, load_yaml_config
from absa.data.io import load_sample, read_jsonl
from absa.data.schema import ABSAExample
from absa.data.splits import make_splits
from absa.logging import configure_logging, get_logger

# Benign scipy/sklearn lbfgs chatter ("Unknown solver options: iprint").
warnings.filterwarnings("ignore", message=".*Unknown solver options.*")

log = get_logger("train")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def _load_splits(data_dir: Path, use_sample: bool, seed: int) -> dict[str, list[ABSAExample]]:
    """Load processed splits, or fall back to a split of the offline sample."""
    train_path = data_dir / "train.jsonl"
    if not use_sample and train_path.exists():
        splits = {
            name: read_jsonl(data_dir / f"{name}.jsonl")
            for name in ("train", "val", "test")
            if (data_dir / f"{name}.jsonl").exists()
        }
        log.info("loaded_processed", **{k: len(v) for k, v in splits.items()})
        return splits
    log.info("using_sample_split")
    return make_splits(load_sample(), seed=seed)


def _train_baseline(
    splits: dict[str, list[ABSAExample]], seed: int
) -> tuple[object, dict[str, object]]:
    from absa.models.baseline import BaselineABSA
    from absa.training.evaluation import evaluate_pipeline

    model_cfg = load_yaml_config("configs/model.yaml")
    model = BaselineABSA.from_config(model_cfg)
    log.info("fitting_baseline", n_train=len(splits["train"]))
    model.fit(splits["train"])
    test = splits.get("test") or splits.get("val") or []
    metrics = evaluate_pipeline(model, test)
    return model, metrics


def _write_metrics(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_baseline(
    args: argparse.Namespace, splits: dict[str, list[ABSAExample]], seed: int
) -> None:
    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "artifacts" / "baseline"
    metrics_out = (
        Path(args.metrics_out)
        if args.metrics_out
        else PROJECT_ROOT / "reports" / "baseline_metrics.json"
    )
    model, metrics = _train_baseline(splits, seed)
    saved = model.save(out_dir)  # type: ignore[attr-defined]
    _write_metrics(
        metrics_out,
        {
            "track": "baseline",
            "seed": seed,
            "splits": {k: len(v) for k, v in splits.items()},
            "metrics": metrics,
        },
    )
    ate_f1 = metrics["ate"]["f1"]  # type: ignore[index]
    asc_f1 = metrics["asc"]["macro_f1"]  # type: ignore[index]
    acd_f1 = metrics["acd"].get("micro_f1", "n/a")  # type: ignore[union-attr]
    log.info("done", model=str(saved), ate_f1=ate_f1, asc_macro_f1=asc_f1, acd_micro_f1=acd_f1)
    print(
        f"[train] baseline saved -> {saved}\n"
        f"        ATE span-F1={ate_f1:.3f}  ASC macro-F1={asc_f1:.3f}  ACD micro-F1={acd_f1}\n"
        f"        metrics -> {metrics_out}"
    )


def _run_transformer(
    args: argparse.Namespace, splits: dict[str, list[ABSAExample]], seed: int
) -> None:
    from absa.training.hf_trainer import train_transformer

    model_cfg = load_yaml_config("configs/model.yaml")
    train_cfg = load_yaml_config("configs/training.yaml")["transformer"]
    train_cfg["seed"] = seed
    out_dir = (
        Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "artifacts" / "transformer"
    )
    metrics_out = (
        Path(args.metrics_out)
        if args.metrics_out
        else PROJECT_ROOT / "reports" / "transformer_metrics.json"
    )
    results = train_transformer(
        splits,
        model_cfg,
        train_cfg,
        out_dir,
        encoder_override=args.encoder,
        max_train=args.max_train,
        epochs=args.epochs,
        mlflow_enabled=not args.no_mlflow,
    )
    _write_metrics(
        metrics_out,
        {
            "track": "transformer",
            "seed": seed,
            "splits": {k: len(v) for k, v in splits.items()},
            "metrics": results,
        },
    )
    ate = results.get("ate", {}).get("f1", "n/a")
    asc = results.get("asc", {}).get("macro_f1", "n/a")
    acd = results.get("acd", {}).get("micro_f1", "n/a")
    print(
        f"[train] transformer ({results.get('encoder')}) saved -> {out_dir}\n"
        f"        ATE span-F1={ate}  ASC macro-F1={asc}  ACD micro-F1={acd}\n"
        f"        metrics -> {metrics_out}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an ABSA track.")
    parser.add_argument("--track", choices=["baseline", "transformer"], default="baseline")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--use-sample", action="store_true", help="train on the offline sample")
    parser.add_argument("--seed", type=int, default=None)
    # transformer-only
    parser.add_argument("--encoder", default=None, help="override the encoder (Track B)")
    parser.add_argument("--max-train", type=int, default=None, help="cap train size (smoke run)")
    parser.add_argument("--epochs", type=float, default=None, help="override epochs (Track B)")
    parser.add_argument("--no-mlflow", action="store_true", help="disable MLflow logging")
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()
    seed = args.seed if args.seed is not None else settings.seed
    _set_seed(seed)

    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "processed"
    splits = _load_splits(data_dir, args.use_sample, seed)

    if args.track == "transformer":
        _run_transformer(args, splits, seed)
    else:
        _run_baseline(args, splits, seed)


if __name__ == "__main__":
    main()
