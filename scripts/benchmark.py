"""Benchmark CPU inference latency (mean/p50/p95) and artifact size."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from absa.config import get_settings
from absa.logging import configure_logging, get_logger

log = get_logger("benchmark")

SAMPLES = [
    "The battery life is amazing but the keyboard feels cheap.",
    "The pizza was delicious though the service was painfully slow.",
    "Great screen and fast performance for the price.",
    "The trackpad is unresponsive and the fan is far too loud.",
    "Lovely ambience, friendly staff, and reasonable prices.",
]


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU latency + size.")
    parser.add_argument("--source", default=None, help="artifact dir (default: int8 artifact)")
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()
    source = Path(args.source) if args.source else settings.resolve(settings.artifact_path)
    if not (source / "meta.json").exists():
        raise SystemExit(f"No artifact at {source}. Run scripts/export_model.py first.")

    from absa.models.transformer import TransformerABSA

    log.info("loading", source=str(source))
    pipe = TransformerABSA.load(source, device="cpu")

    pipe.predict(SAMPLES[0])  # warmup
    latencies: list[float] = []
    for i in range(args.iters):
        text = SAMPLES[i % len(SAMPLES)]
        t0 = time.perf_counter()
        pipe.predict(text)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    mean = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p95 = latencies[int(0.95 * len(latencies)) - 1]
    size_mb = _dir_size_mb(source)

    log.info(
        "done",
        mean_ms=round(mean, 1),
        p50_ms=round(p50, 1),
        p95_ms=round(p95, 1),
        size_mb=round(size_mb, 1),
    )
    print(f"[benchmark] artifact: {source.name}  ({size_mb:.0f} MB on disk, CPU)")
    print(f"            latency per inference over {args.iters} runs:")
    print(f"              mean {mean:6.1f} ms   p50 {p50:6.1f} ms   p95 {p95:6.1f} ms")
    print(f"            sub-second target: {'MET' if p95 < 1000 else 'NOT MET'} (p95 {p95:.0f} ms)")


if __name__ == "__main__":
    main()
