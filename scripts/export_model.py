"""Export an int8-quantized CPU artifact from the trained transformer (optional --onnx).

Weights-only format: each task dir holds config.json + a quantized state_dict, so the
artifact never unpickles code and loads with ``torch.load(weights_only=True)``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from absa.config import PROJECT_ROOT, get_settings, load_yaml_config
from absa.logging import configure_logging, get_logger

log = get_logger("export")


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def _export_onnx(pipe: Any, output: Path) -> bool:
    """Best-effort ONNX export of the fp32 sub-models (not used by the demo)."""
    try:
        import torch

        onnx_dir = output / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        dummy = pipe.tokenizer("warmup text", return_tensors="pt", truncation=True, max_length=32)
        args = (dummy["input_ids"], dummy["attention_mask"])
        dyn = {
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        }
        for name, model in (
            ("ate", pipe.ate_model),
            ("asc", pipe.asc_model),
            ("acd", pipe.acd_model),
        ):
            if model is None:
                continue
            model.eval()
            torch.onnx.export(
                model,
                args,
                str(onnx_dir / f"{name}.onnx"),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes=dyn,
                opset_version=14,
            )
        return True
    except Exception as exc:  # ONNX is optional; never fail the int8 export
        log.warning("onnx_export_failed", error=str(exc)[:200])
        return False


def _calibrate_acd(artifact: Path, val_path: Path) -> float | None:
    """Re-tune the ACD cutoff on validation data: int8 shrinks category probabilities,
    so the fp32 cutoff of 0.5 costs recall (test micro-F1 0.822 vs 0.858 at 0.30)."""
    from absa.data.io import read_jsonl
    from absa.models.transformer import TransformerABSA
    from absa.training.evaluation import tune_threshold

    if not val_path.exists():
        log.warning("acd_threshold_untuned", reason=f"{val_path} not found; keeping 0.5")
        return None
    pipe = TransformerABSA.load(artifact, device="cpu")
    if pipe.acd_model is None:
        return None
    scored = [
        ({c.category for c in ex.aspect_categories}, pipe.category_scores(ex.text))
        for ex in read_jsonl(val_path)
        if ex.aspect_categories
    ]
    best, curve = tune_threshold(scored, pipe.category_labels)
    log.info("acd_threshold", best=best, val_micro_f1=round(curve[best], 3), at_half=curve[0.5])
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a CPU (int8) deployment artifact.")
    parser.add_argument("--source", default=None, help="fp32 transformer artifact dir")
    parser.add_argument("--output", default=None, help="output artifact dir")
    parser.add_argument("--version", default=None)
    parser.add_argument("--onnx", action="store_true", help="also emit ONNX graphs")
    parser.add_argument(
        "--val",
        default=str(PROJECT_ROOT / "data" / "processed" / "val.jsonl"),
        help="validation split used to tune the ACD threshold",
    )
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()
    export_cfg = load_yaml_config("configs/training.yaml").get("export", {})
    output = Path(args.output) if args.output else settings.resolve(settings.artifact_path)
    version = args.version or str(export_cfg.get("version_tag", "v1"))

    # Prefer the quantization-friendly bert-base model for deployment; deberta-v3
    # degrades badly under dynamic int8 (its disentangled attention is sensitive).
    if args.source:
        source = Path(args.source)
    else:
        bert_dir = PROJECT_ROOT / "artifacts" / "transformer-bert"
        default_dir = PROJECT_ROOT / "artifacts" / "transformer"
        source = bert_dir if (bert_dir / "meta.json").exists() else default_dir

    if not (source / "meta.json").exists():
        raise SystemExit(
            f"No transformer artifact at {source}. Run `make train-demo` (bert-base) first."
        )
    encoder = json.loads((source / "meta.json").read_text(encoding="utf-8")).get("encoder", "")
    if "deberta" in encoder.lower():
        log.warning(
            "quantizing_deberta",
            note="deberta-v3 loses accuracy under int8; prefer bert-base via `make train-demo`",
        )

    from absa.models.transformer import TransformerABSA, save_int8

    log.info("loading_fp32", source=str(source))
    pipe = TransformerABSA.load(source, device="cpu")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    for name, model in (("ate", pipe.ate_model), ("asc", pipe.asc_model), ("acd", pipe.acd_model)):
        if model is None:
            continue
        save_int8(model, output / name)
        log.info("quantized", model=name)

    pipe.tokenizer.save_pretrained(output / "tokenizer")
    src_meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
    meta = {
        **src_meta,
        "quantized": True,
        "quantization": "dynamic_int8",
        "version": version,
        "source": str(source),
    }
    (output / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    threshold = _calibrate_acd(output, Path(args.val))
    if threshold is not None:
        meta["acd_threshold"] = threshold
        (output / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    onnx_ok = _export_onnx(pipe, output) if args.onnx else False

    fp32_mb, int8_mb = _dir_size_mb(source), _dir_size_mb(output)
    log.info("done", output=str(output), fp32_mb=round(fp32_mb, 1), int8_mb=round(int8_mb, 1))
    print(f"[export] int8 artifact ({version}) -> {output}")
    print(
        f"         size: fp32 {fp32_mb:.0f} MB -> int8 {int8_mb:.0f} MB "
        f"({int8_mb / fp32_mb * 100:.0f}% of fp32)"
    )
    if threshold is not None:
        print(f"         ACD threshold: {threshold} (tuned on {Path(args.val).name})")
    if args.onnx:
        print(f"         onnx: {'exported to onnx/' if onnx_ok else 'failed (see logs)'}")


if __name__ == "__main__":
    main()
