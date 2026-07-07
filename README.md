# ABSA — Aspect-Based Sentiment Analysis

Extract **aspects** from review text and predict **sentiment per aspect**
(positive / negative / neutral), covering three sub-tasks — **ATE** (aspect term
extraction), **ACD** (aspect category detection), and **ASC** (aspect sentiment
classification) — behind one call:

```python
predict("The battery lasts forever but the screen is dim")
# -> [{"aspect": "battery", "category": "BATTERY",  "sentiment": "positive", "confidence": 0.97, "span": [4, 11]},
#     {"aspect": "screen",  "category": "DISPLAY",   "sentiment": "negative", "confidence": 0.93, "span": [37, 43]}]
```

Two comparable tracks are provided so results can be compared head-to-head:

- **Track A — classical baseline**: spaCy/CRF extraction + TF-IDF → LinearSVC/LogReg. CPU, interpretable.
- **Track B — transformer** (primary): fine-tuned `microsoft/deberta-v3-base` (fallback `bert-base-uncased`).

> **Status:** Phase 0 (scaffold) complete. Data, models, serving, and the demo
> land in subsequent phases — see the roadmap below.

## Two target machines

Training and the live demo run on different hardware; the project is designed for both.

| | **Training desktop** | **Demo laptop** |
|---|---|---|
| Hardware | Ryzen 7 7700X · RTX 4070 12 GB · 16 GB | Ryzen 5 7000U (no GPU) · 16 GB |
| Role | fine-tuning (bf16 mixed precision) | **inference only, fully offline** |
| Install | `requirements-gpu.txt` (CUDA 12.4 torch) | `requirements-cpu.txt` (CPU torch) |
| Loads | training checkpoints | int8-quantized artifact from `artifacts/` |

## Quickstart

```bash
# 1. Install (choose one)
make setup          # CPU / demo / dev machine   (or:  ./make.ps1 setup      on Windows)
make setup-gpu      # RTX 4070 training desktop   (or:  ./make.ps1 setup-gpu)

# 2. Prepare data (downloads SemEval-2014 if permitted; else uses committed sample)
make prepare-data

# 3. Train, evaluate, export the CPU artifact
make train
make evaluate
make export

# 4. Serve / demo
make serve          # FastAPI at http://localhost:8000  (/docs, /health, /predict)
make demo           # Streamlit demo (CPU, offline)
```

Without GNU Make on Windows, replace `make <t>` with `./make.ps1 <t>`.

## Deploying to the demo laptop

The laptop runs **inference only, offline, no GPU, no Docker**:

1. On the desktop, train and export: `make train && make export`. This writes an
   **int8-quantized (and optionally ONNX) artifact** to `artifacts/`.
2. Copy the repo **and the `artifacts/` folder** to the laptop (artifacts are
   gitignored because they're large — transfer them manually).
3. On the laptop: `make setup` (installs the **CPU-only** stack), then `make demo`.
4. The demo preloads the artifact and runs a warmup inference at startup, so the
   first live prediction returns in **well under a second** — with no downloads.

## Development

```bash
make test      # pytest + coverage (≥ 70%)
make lint      # ruff + black --check
make type      # mypy
make format    # auto-fix + format
make check     # lint + type + test
```

Config: environment/secrets via `.env` (see `.env.example`, prefix `ABSA_`);
experiment config via YAML in `configs/`. Experiments tracked with MLflow.

## Roadmap

- [x] **P0** Scaffold — packaging, config, logging, tooling, CI, docs.
- [x] **P1** Data — SemEval-2014 parser + fetch, splits, committed sample, dataset card.
- [ ] **P2** Track A baseline — train + evaluate.
- [ ] **P3** Track B transformer — fine-tune, MLflow, checkpoints.
- [ ] **P4** Evaluation — Track A vs B tables, confusion matrices, figures.
- [ ] **P5** Serving — FastAPI `/predict` + `/health`, pydantic schemas.
- [ ] **P5.5** CPU export — int8/ONNX artifact + latency benchmark.
- [ ] **P6** Demo — Streamlit UI (offline, cached examples).
- [ ] **P7** Docker & CI — compose (API + demo), GitHub Actions.
- [ ] **P8** Docs & paper — model card, dataset card, IEEE paper skeleton.

## License

MIT (see `pyproject.toml`).
