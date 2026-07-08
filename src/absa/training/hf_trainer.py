"""Track B training with HuggingFace ``Trainer`` (ATE / ACD / ASC).

Fine-tunes one shared encoder three ways, evaluates on the test split, saves
each model + the shared tokenizer + ``meta.json`` to the output dir, and logs
params/metrics to MLflow. bf16 mixed precision is used on CUDA (RTX 4070).
Heavy imports are local so ``import absa.training`` stays light.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from absa.config import get_settings
from absa.data.schema import ABSAExample
from absa.logging import get_logger
from absa.models.transformer import (
    ASC_LABEL2ID,
    ASC_LABELS,
    ATE_ID2LABEL,
    ATE_LABEL2ID,
    ATE_LABELS,
    TransformerConfig,
    build_acd_dataset,
    build_asc_dataset,
    build_ate_dataset,
    category_vocab,
    resolve_tokenizer,
)

log = get_logger("hf_trainer")


def set_seed_all(seed: int) -> None:
    from transformers import set_seed

    set_seed(seed)


def _subset(examples: list[ABSAExample], max_n: int | None) -> list[ABSAExample]:
    return examples[:max_n] if max_n else examples


# --------------------------------------------------------------------------- #
# compute_metrics
# --------------------------------------------------------------------------- #
def _ate_compute_metrics(eval_pred: Any) -> dict[str, float]:
    import numpy as np
    from seqeval.metrics import f1_score, precision_score, recall_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_tags: list[list[str]] = []
    pred_tags: list[list[str]] = []
    for pred_row, label_row in zip(preds, labels, strict=True):
        t, p = [], []
        for pred_id, label_id in zip(pred_row, label_row, strict=True):
            if label_id == -100:
                continue
            t.append(ATE_LABELS[label_id])
            p.append(ATE_ID2LABEL.get(int(pred_id), "O"))
        true_tags.append(t)
        pred_tags.append(p)
    return {
        "precision": float(precision_score(true_tags, pred_tags, zero_division=0)),
        "recall": float(recall_score(true_tags, pred_tags, zero_division=0)),
        "f1": float(f1_score(true_tags, pred_tags, zero_division=0)),
    }


def _asc_compute_metrics(eval_pred: Any) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def _acd_compute_metrics(eval_pred: Any) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import f1_score

    logits, labels = eval_pred
    preds = (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(int)
    labels = np.asarray(labels).astype(int)
    return {
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


# --------------------------------------------------------------------------- #
# Trainer wiring
# --------------------------------------------------------------------------- #
def _training_args(
    train_cfg: dict[str, Any],
    out_dir: Path,
    metric: str,
    use_cuda: bool,
    epochs: float,
    run_name: str,
) -> Any:
    from transformers import TrainingArguments

    bf16 = bool(train_cfg.get("bf16", True)) and use_cuda
    return TrainingArguments(
        output_dir=str(out_dir),
        run_name=run_name,
        num_train_epochs=epochs,
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 32)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.1)),
        bf16=bf16,
        fp16=False,
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 2)),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=metric,
        greater_is_better=True,
        logging_strategy="epoch",
        report_to=[],
        seed=int(train_cfg.get("seed", 42)),
        disable_tqdm=False,
    )


def _run_task(
    *,
    model: Any,
    tokenizer: Any,
    train_ds: Any,
    val_ds: Any,
    test_ds: Any,
    collator: Any,
    compute_metrics: Any,
    train_cfg: dict[str, Any],
    out_dir: Path,
    metric: str,
    epochs: float,
    use_cuda: bool,
    patience: int,
    run_name: str,
) -> tuple[Any, dict[str, float], list[dict[str, Any]]]:
    from transformers import EarlyStoppingCallback, Trainer

    args = _training_args(train_cfg, out_dir, metric, use_cuda, epochs, run_name)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    trainer.train()
    raw = trainer.evaluate(test_ds)
    metrics = {k.removeprefix("eval_"): v for k, v in raw.items() if k.startswith("eval_")}
    return trainer.model, metrics, list(trainer.state.log_history)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train_transformer(
    splits: dict[str, list[ABSAExample]],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    output_dir: str | Path,
    *,
    encoder_override: str | None = None,
    max_train: int | None = None,
    epochs: float | None = None,
    mlflow_enabled: bool = True,
) -> dict[str, Any]:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        DataCollatorForTokenClassification,
        DataCollatorWithPadding,
    )

    settings = get_settings()
    tcfg = model_cfg["transformer"]
    markers = tuple(tcfg["asc"].get("aspect_markers", ["[ASP]", "[/ASP]"]))
    tconf = TransformerConfig(
        encoder=encoder_override or tcfg["encoder"],
        fallback_encoder=tcfg.get("fallback_encoder", "bert-base-uncased"),
        max_length=int(tcfg.get("max_length", 128)),
        asc_markers=markers,
    )
    seed = int(train_cfg.get("seed", settings.seed))
    set_seed_all(seed)
    use_cuda = torch.cuda.is_available()
    n_epochs = float(epochs if epochs is not None else train_cfg.get("num_train_epochs", 5))
    patience = int(train_cfg.get("early_stopping_patience", 2))

    tokenizer, encoder = resolve_tokenizer(tconf)
    tokenizer.add_special_tokens({"additional_special_tokens": list(tconf.asc_markers)})
    log.info("resolved_encoder", encoder=encoder, cuda=use_cuda, epochs=n_epochs)

    train_ex = _subset(splits["train"], max_train)
    val_ex = _subset(splits.get("val") or splits["train"], max_train)
    test_ex = splits.get("test") or val_ex
    categories = category_vocab(splits["train"])
    output_dir = Path(output_dir)

    def new_model(head: str, **kwargs: Any) -> Any:
        if head == "token":
            model = AutoModelForTokenClassification.from_pretrained(encoder, **kwargs)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(encoder, **kwargs)
        model.resize_token_embeddings(len(tokenizer))
        return model

    run_ctx: Any = contextlib.nullcontext()
    if mlflow_enabled:
        try:
            import mlflow

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment)
            run_ctx = mlflow.start_run(run_name=f"transformer-{encoder.split('/')[-1]}")
        except Exception as exc:
            log.warning("mlflow_unavailable", error=str(exc)[:160])

    results: dict[str, Any] = {"encoder": encoder, "seed": seed, "epochs": n_epochs}
    with run_ctx:
        _maybe_log_params(
            mlflow_enabled,
            {"encoder": encoder, "epochs": n_epochs, "seed": seed, "max_length": tconf.max_length},
        )

        # ---- ATE ----
        ate_model = new_model(
            "token", num_labels=len(ATE_LABELS), id2label=ATE_ID2LABEL, label2id=ATE_LABEL2ID
        )
        ate_model, ate_metrics, ate_hist = _run_task(
            model=ate_model,
            tokenizer=tokenizer,
            train_ds=build_ate_dataset(train_ex, tokenizer, tconf.max_length),
            val_ds=build_ate_dataset(val_ex, tokenizer, tconf.max_length),
            test_ds=build_ate_dataset(test_ex, tokenizer, tconf.max_length),
            collator=DataCollatorForTokenClassification(tokenizer),
            compute_metrics=_ate_compute_metrics,
            train_cfg=train_cfg,
            out_dir=output_dir / "_ckpt_ate",
            metric="f1",
            epochs=n_epochs,
            use_cuda=use_cuda,
            patience=patience,
            run_name="ate",
        )
        ate_model.save_pretrained(output_dir / "ate")
        results["ate"] = ate_metrics

        # ---- ASC ----
        asc_model = new_model(
            "seq",
            num_labels=len(ASC_LABELS),
            id2label=dict(enumerate(ASC_LABELS)),
            label2id=ASC_LABEL2ID,
        )
        asc_model, asc_metrics, asc_hist = _run_task(
            model=asc_model,
            tokenizer=tokenizer,
            train_ds=build_asc_dataset(train_ex, tokenizer, tconf.max_length, tconf.asc_markers),
            val_ds=build_asc_dataset(val_ex, tokenizer, tconf.max_length, tconf.asc_markers),
            test_ds=build_asc_dataset(test_ex, tokenizer, tconf.max_length, tconf.asc_markers),
            collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=_asc_compute_metrics,
            train_cfg=train_cfg,
            out_dir=output_dir / "_ckpt_asc",
            metric="macro_f1",
            epochs=n_epochs,
            use_cuda=use_cuda,
            patience=patience,
            run_name="asc",
        )
        asc_model.save_pretrained(output_dir / "asc")
        results["asc"] = asc_metrics

        # ---- ACD (restaurants only) ----
        acd_metrics: dict[str, Any] = {"note": "no category supervision"}
        acd_hist: list[dict[str, Any]] = []
        if categories:
            acd_model = new_model(
                "seq", num_labels=len(categories), problem_type="multi_label_classification"
            )
            acd_model, acd_metrics, acd_hist = _run_task(
                model=acd_model,
                tokenizer=tokenizer,
                train_ds=build_acd_dataset(train_ex, tokenizer, tconf.max_length, categories),
                val_ds=build_acd_dataset(val_ex, tokenizer, tconf.max_length, categories),
                test_ds=build_acd_dataset(test_ex, tokenizer, tconf.max_length, categories),
                collator=DataCollatorWithPadding(tokenizer),
                compute_metrics=_acd_compute_metrics,
                train_cfg=train_cfg,
                out_dir=output_dir / "_ckpt_acd",
                metric="micro_f1",
                epochs=n_epochs,
                use_cuda=use_cuda,
                patience=patience,
                run_name="acd",
            )
            acd_model.save_pretrained(output_dir / "acd")
        results["acd"] = acd_metrics

        _save_artifacts(
            output_dir,
            tokenizer,
            tconf,
            categories,
            {"ate": ate_hist, "asc": asc_hist, "acd": acd_hist},
        )
        _maybe_log_metrics(mlflow_enabled, results)

    return results


def _maybe_log_params(enabled: bool, params: dict[str, Any]) -> None:
    if not enabled:
        return
    with contextlib.suppress(Exception):
        import mlflow

        mlflow.log_params(params)


def _maybe_log_metrics(enabled: bool, results: dict[str, Any]) -> None:
    if not enabled:
        return
    with contextlib.suppress(Exception):
        import mlflow

        flat = {
            f"{task}_{k}": v
            for task in ("ate", "acd", "asc")
            for k, v in results.get(task, {}).items()
            if isinstance(v, int | float)
        }
        mlflow.log_metrics(flat)


def _save_artifacts(
    output_dir: Path,
    tokenizer: Any,
    tconf: TransformerConfig,
    categories: list[str],
    histories: dict[str, list[dict[str, Any]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    meta = {
        "track": "transformer",
        "encoder": tconf.encoder,
        "max_length": tconf.max_length,
        "markers": list(tconf.asc_markers),
        "categories": categories,
        "asc_labels": ASC_LABELS,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (output_dir / "training_history.json").write_text(
        json.dumps(histories, indent=2), encoding="utf-8"
    )


__all__ = ["set_seed_all", "train_transformer"]
