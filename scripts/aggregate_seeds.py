"""Summarize seeded Track B runs (mean +/- std) -> reports/seed_variance.json.

Reads the metrics JSON that ``train.py --metrics-out`` writes per seed (see
``make train-seeds``). These are the Trainer's test-set metrics: ASC/ACD match the
evaluation code, while ATE is seqeval word-level F1 rather than char-exact spans.
"""

from __future__ import annotations

import argparse
import json

from absa.config import PROJECT_ROOT
from absa.training.metrics import summarize_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate seeded training runs.")
    parser.add_argument("--glob", default="reports/seeds/transformer_seed*.json")
    parser.add_argument("--out", default="reports/seed_variance.json")
    args = parser.parse_args()

    paths = sorted(PROJECT_ROOT.glob(args.glob))
    if not paths:
        raise SystemExit(f"No runs match {args.glob}. Run `make train-seeds` first.")
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    summary = summarize_runs([p["metrics"] for p in payloads])

    out = PROJECT_ROOT / args.out
    out.write_text(
        json.dumps(
            {
                "seeds": [p.get("seed") for p in payloads],
                "encoder": payloads[0]["metrics"].get("encoder"),
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[seeds] {len(paths)} runs, encoder={payloads[0]['metrics'].get('encoder')}")
    for name, stats in summary.items():
        print(f"        {name:<14} {stats['mean']:.3f} +/- {stats['std']:.3f}")
    print(f"        -> {out}")


if __name__ == "__main__":
    main()
