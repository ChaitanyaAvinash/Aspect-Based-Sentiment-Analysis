"""Evaluate trained tracks on the test split -> reports/comparison.{md,json} + figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from absa.config import PROJECT_ROOT, get_settings
from absa.data.io import load_sample, read_jsonl
from absa.data.schema import ABSAExample
from absa.data.splits import make_splits
from absa.logging import configure_logging, get_logger
from absa.training.evaluation import evaluate_pipeline

log = get_logger("evaluate")

# Validated colorblind-safe categorical palette: blue / aqua.
COLORS = {"baseline": "#2a78d6", "transformer": "#1baf7a", "third": "#eda100"}
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

TASK_KEYS = [
    ("ate", "f1", "ATE\nspan-F1"),
    ("acd", "micro_f1", "ACD\nmicro-F1"),
    ("asc", "macro_f1", "ASC\nmacro-F1"),
]


def _load_test(data_dir: Path, use_sample: bool, seed: int) -> list[ABSAExample]:
    test_path = data_dir / "test.jsonl"
    if not use_sample and test_path.exists():
        return read_jsonl(test_path)
    return make_splits(load_sample(), seed=seed)["test"]


def _load_models() -> dict[str, Any]:
    models: dict[str, Any] = {}
    baseline_dir = PROJECT_ROOT / "artifacts" / "baseline"
    if (baseline_dir / "baseline.joblib").exists():
        from absa.models.baseline import BaselineABSA

        models["baseline"] = BaselineABSA.load(baseline_dir)
        log.info("loaded_baseline")
    transformer_dir = PROJECT_ROOT / "artifacts" / "transformer"
    if (transformer_dir / "meta.json").exists():
        from absa.models.transformer import TransformerABSA

        models["transformer"] = TransformerABSA.load(transformer_dir)
        log.info("loaded_transformer")
    return models


# reports
def _fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "n/a"


def _comparison_markdown(results: dict[str, dict[str, Any]]) -> str:
    tracks = [t for t in ("baseline", "transformer") if t in results]
    names = {"baseline": "Track A — baseline", "transformer": "Track B — deberta-v3"}
    rows = [
        ("ATE — precision", ("ate", "precision")),
        ("ATE — recall", ("ate", "recall")),
        ("ATE — span-F1", ("ate", "f1")),
        ("ACD — micro-F1", ("acd", "micro_f1")),
        ("ACD — macro-F1", ("acd", "macro_f1")),
        ("ASC — accuracy", ("asc", "accuracy")),
        ("ASC — macro-F1", ("asc", "macro_f1")),
    ]
    header = "| Metric | " + " | ".join(names[t] for t in tracks) + " |"
    sep = "|" + "---|" * (len(tracks) + 1)
    lines = ["# Track A vs Track B — SemEval-2014 test", "", header, sep]
    for label, (task, key) in rows:
        cells = [_fmt(results[t].get(task, {}).get(key)) for t in tracks]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "_ASC and ACD are scored on gold aspects/categories; ATE is "
        "predicted-vs-gold span exact match. Both tracks use the same "
        "evaluation code path._",
    ]
    return "\n".join(lines) + "\n"


# figures
def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "font.size": 11,
            "axes.titlesize": 13,
            "figure.dpi": 130,
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _plot_f1_comparison(results: dict[str, dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    tracks = [t for t in ("baseline", "transformer") if t in results]
    labels = [lbl for _, _, lbl in TASK_KEYS]
    x = np.arange(len(labels))
    width = 0.8 / max(len(tracks), 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, track in enumerate(tracks):
        vals = [float(results[track].get(t, {}).get(k, 0) or 0) for t, k, _ in TASK_KEYS]
        offset = (i - (len(tracks) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=track, color=COLORS[track], zorder=3)
        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.015,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Track A vs Track B — SemEval-2014 test")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_confusion(results: dict[str, Any], track: str, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    asc = results.get("asc", {})
    matrix = asc.get("confusion_matrix")
    labels = asc.get("labels")
    if not matrix or not labels:
        return
    matrix_arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(matrix_arr, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"ASC confusion — {track}")
    thresh = matrix_arr.max() / 2 if matrix_arr.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                int(matrix_arr[i, j]),
                ha="center",
                va="center",
                color="white" if matrix_arr[i, j] > thresh else INK,
                fontsize=11,
            )
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_per_class_f1(results: dict[str, dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    tracks = [t for t in ("baseline", "transformer") if t in results]
    classes = ["negative", "neutral", "positive"]
    x = np.arange(len(classes))
    width = 0.8 / max(len(tracks), 1)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for i, track in enumerate(tracks):
        per_class = results[track].get("asc", {}).get("per_class", {})
        vals = [float(per_class.get(c, {}).get("f1-score", 0)) for c in classes]
        offset = (i - (len(tracks) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=track, color=COLORS[track], zorder=3)
        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.015,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )
    ax.set_xticks(x, classes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("ASC per-class F1")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_training_curves(history_path: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    if not history_path.exists():
        return
    histories = json.loads(history_path.read_text(encoding="utf-8"))
    metric_key = {"ate": "eval_f1", "asc": "eval_macro_f1", "acd": "eval_micro_f1"}
    color_by_task = {
        "ate": COLORS["baseline"],
        "asc": COLORS["transformer"],
        "acd": COLORS["third"],
    }
    fig, ax = plt.subplots(figsize=(7, 4.3))
    plotted = False
    for task, key in metric_key.items():
        entries = [e for e in histories.get(task, []) if key in e]
        if not entries:
            continue
        epochs = [e.get("epoch", i + 1) for i, e in enumerate(entries)]
        scores = [e[key] for e in entries]
        ax.plot(
            epochs,
            scores,
            marker="o",
            markersize=6,
            linewidth=2,
            color=color_by_task[task],
            label=f"{task.upper()} ({key.removeprefix('eval_')})",
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation score")
    ax.set_title("Track B — validation score by epoch")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _build_figures(results: dict[str, dict[str, Any]], fig_dir: Path) -> list[str]:
    _style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    _plot_f1_comparison(results, fig_dir / "f1_comparison.png")
    made.append("f1_comparison.png")
    _plot_per_class_f1(results, fig_dir / "asc_per_class_f1.png")
    made.append("asc_per_class_f1.png")
    for track in results:
        _plot_confusion(results[track], track, fig_dir / f"asc_confusion_{track}.png")
        made.append(f"asc_confusion_{track}.png")
    _plot_training_curves(
        PROJECT_ROOT / "artifacts" / "transformer" / "training_history.json",
        fig_dir / "training_curves.png",
    )
    made.append("training_curves.png")
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tracks and build reports.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--use-sample", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()
    seed = args.seed if args.seed is not None else settings.seed
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "processed"

    test = _load_test(data_dir, args.use_sample, seed)
    models = _load_models()
    if not models:
        raise SystemExit("No trained models found in artifacts/. Run scripts/train.py first.")

    results: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        log.info("evaluating", track=name, n=len(test))
        results[name] = evaluate_pipeline(model, test)

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (reports / "comparison.md").write_text(_comparison_markdown(results), encoding="utf-8")
    log.info("wrote_reports", json=str(reports / "comparison.json"))

    figures: list[str] = []
    if not args.skip_figures:
        figures = _build_figures(results, reports / "figures")
        log.info("wrote_figures", figures=figures)

    print("[evaluate] tracks:", ", ".join(results))
    print(f"           reports -> {reports/'comparison.md'}")
    if figures:
        print(f"           figures -> {reports/'figures'} ({len(figures)})")


if __name__ == "__main__":
    main()
