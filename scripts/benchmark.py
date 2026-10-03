"""Benchmark CPU inference latency (mean/p50/p95) and artifact size -> reports/benchmark.json.

Records the CPU it ran on: the number only means something next to the hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

from absa.config import PROJECT_ROOT, get_settings
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


def _cpu_name() -> str:
    """Marketing CPU name (e.g. "AMD Ryzen 5 7530U"), falling back to platform info."""
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU latency + size.")
    parser.add_argument("--source", default=None, help="artifact dir (default: int8 artifact)")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--out", default=None, help="JSON output (default reports/benchmark.json)")
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

    import torch

    mean = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[-1]
    size_mb = _dir_size_mb(source)
    machine = {
        "cpu": _cpu_name(),
        "logical_cores": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "os": platform.platform(),
    }
    payload = {
        "artifact": source.name,
        "size_mb": round(size_mb, 1),
        "iters": args.iters,
        "latency_ms": {"mean": round(mean, 1), "p50": round(p50, 1), "p95": round(p95, 1)},
        "machine": machine,
    }
    out = Path(args.out) if args.out else PROJECT_ROOT / "reports" / "benchmark.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    log.info(
        "done",
        mean_ms=round(mean, 1),
        p50_ms=round(p50, 1),
        p95_ms=round(p95, 1),
        size_mb=round(size_mb, 1),
    )
    print(f"[benchmark] artifact: {source.name}  ({size_mb:.0f} MB on disk, CPU)")
    print(f"            machine:  {machine['cpu']} ({machine['torch_threads']} torch threads)")
    print(f"            latency per inference over {args.iters} runs:")
    print(f"              mean {mean:6.1f} ms   p50 {p50:6.1f} ms   p95 {p95:6.1f} ms")
    print(f"            sub-second target: {'MET' if p95 < 1000 else 'NOT MET'} (p95 {p95:.0f} ms)")
    print(f"            results -> {out}")


if __name__ == "__main__":
    main()
