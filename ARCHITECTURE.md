# Architecture

Developer overview of the ABSA codebase: what it is, how it is organized, and the
conventions to follow. For results and the full write-up see
`reports/PROJECT_REPORT.md`.

## What this is

**Aspect-Based Sentiment Analysis (ABSA)**: given review text, extract aspects and
predict sentiment per aspect. Three sub-tasks:

- **ATE** — Aspect Term Extraction (spans, e.g. "battery", "service").
- **ACD** — Aspect Category Detection (e.g. `food`, `service`).
- **ASC** — Aspect Sentiment Classification (positive / negative / neutral per aspect).
- End-to-end: `predict(text) -> [{aspect, sentiment, confidence, span}]`, with `span`
  indexing the caller's raw text, plus sentence-level `predict_categories(text)`.
  Categories are not per aspect: SemEval-2014 does not link terms to categories,
  and only restaurants have them.

Two comparable tracks:

- **Track A — classical baseline**: CRF for ATE + TF-IDF/logistic regression for
  ACD and ASC. Fast, CPU-friendly, interpretable.
- **Track B — transformer** (primary): fine-tune `microsoft/deberta-v3-base`
  (fallback `bert-base-uncased`) with the HuggingFace `Trainer`.

## Two-machine split

| | Training desktop | Demo laptop |
|---|---|---|
| Hardware | Ryzen 7 7700X, RTX 4070 12 GB, 16 GB RAM | Ryzen 5 7000U (no GPU), 16 GB RAM |
| Role | fine-tuning (bf16) | inference only, fully offline |
| Install | `requirements-gpu.txt` (CUDA 12.4 torch) | `requirements-cpu.txt` (CPU torch) |
| Loads | training checkpoints | int8-quantized artifact in `artifacts/` |

Design rules:
- All inference (API + demo) runs on CPU, offline; all training uses the GPU.
- The demo path must not require Docker (`make demo` = CPU venv + Streamlit).
- Serving and demo preload the model and run one warmup inference at startup.
- Docker/compose is a separate deliverable, never on the demo path.

## Commands (Makefile / `make.ps1` on Windows)

```
make setup          # CPU venv + install (demo/dev machine)
make setup-gpu      # CUDA venv + install (training desktop)
make prepare-data   # download/parse SemEval -> data/processed
make train          # baseline + transformer (deberta-v3)
make train-demo     # bert-base for the quantized CPU demo
make train-seeds    # deberta-v3 x 3 seeds -> mean/std in reports/seed_variance.json
make evaluate       # metrics, confusion matrices, figures -> reports/
make export         # int8 CPU artifact -> artifacts/
make benchmark      # CPU latency + artifact size
make serve          # FastAPI on :8000 (/docs, /health, /predict)
make demo           # Streamlit demo (CPU, offline)
make test lint type format check
```

Windows without GNU Make: `./make.ps1 <target>`.

## Layout

```
src/absa/
  config.py         pydantic-settings (env/.env) + YAML config loader
  logging.py        structlog setup
  data/             SemEval parsing, preprocessing, splits, io, stats
  models/           baseline.py (Track A), transformer.py (Track B), base interface
  training/         metrics, evaluation, hf_trainer
  serving/          FastAPI app, schemas, model service, demo render helpers
configs/            data.yaml, model.yaml, training.yaml
scripts/            prepare_data, train, evaluate, export_model, benchmark
app/                Streamlit demo
tests/              unit + integration
reports/            figures, results tables, model card, dataset card, paper
docker/             Dockerfile + docker-compose (separate deliverable)
data/{sample,raw,processed}/   sample committed; raw/processed gitignored
artifacts/          trained/exported models (gitignored)
```

## Pipeline

1. `prepare_data.py` fetches SemEval-2014 (HF mirror) or falls back to the
   committed sample, normalizes to `ABSAExample`, and writes stratified
   train/val/test JSONL to `data/processed/`.
2. `train.py --track {baseline,transformer}` fits the models and writes metrics.
   Track B fine-tunes ATE (token classification, BIO), ACD (multi-label), and ASC
   (aspect marked with `[ASP] … [/ASP]`), logging to MLflow.
3. `evaluate.py` scores every track (baseline, deberta, deployed int8) through the
   same code path, overall and per domain, and writes `reports/comparison.{md,json}`,
   `reports/deploy_int8_metrics.json`, and figures.
4. `export_model.py` int8-quantizes a bert-base model into a self-contained CPU
   artifact: per task a `config.json` + quantized `state_dict`, loaded with
   `torch.load(weights_only=True)` (no pickled code). int8 shrinks ACD
   probabilities, so export re-tunes the ACD threshold on the validation split and
   stores it in `meta.json`. `benchmark.py` reports latency, size, and the CPU it
   ran on.
5. `serving/` (FastAPI) and `app/` (Streamlit) load the artifact via
   `ModelService`, preload + warm up, and serve predictions offline.

## Conventions

- **Python 3.12**, `src/` layout, package `absa`.
- **Config**: env/secrets via `absa.config.Settings` (prefix `ABSA_`, `.env`);
  experiment config via YAML in `configs/`. No secrets in code.
- **Logging**: `absa.logging.get_logger`; call `configure_logging()` at entrypoints.
- **Quality gates**: `ruff`, `black` (line length 100), `mypy` (on `src/absa`),
  `pytest` with coverage >= 70%. `pre-commit` enforces locally; CI mirrors it.
- **Reproducibility**: fix all seeds (`ABSA_SEED`, default 42); track runs in MLflow.
- **Heavy imports** (torch/transformers) stay lazy so the base package imports and
  tests collect without the DL stack. Extras split: `classical` (sklearn/spaCy,
  runs in CI) vs `dl` (torch/transformers). `transformer.py` / `hf_trainer.py` are
  omitted from the coverage gate since torch isn't installed in CI.
- Dependencies are pinned; two requirement sets for the CPU/GPU split.
- **Spans** are always offsets into the caller's raw text. Pipelines clean input with
  `clean_text_aligned` and map predicted spans back through its offset table, so the
  demo/API highlight the right characters and evaluation matches gold offsets.
- **Long input** is split into sentence-aligned windows that fit `max_length` instead
  of being truncated (single SemEval sentences always fit in one window).
- **int8 inference** runs one sequence per forward pass: dynamic quantization picks
  activation scales per input tensor, so batching would couple predictions.
