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

# 3. Train (deberta-v3 = best, reported), then build the CPU demo model
make train          # baseline + deberta-v3 (best accuracy, for the paper)
make evaluate       # comparison tables + figures -> reports/
make train-demo     # fine-tune bert-base-uncased (quantization-friendly)
make export         # int8-quantize the bert model -> artifacts/absa-transformer-int8

# 4. Serve / demo
make serve          # FastAPI at http://localhost:8000  (/docs, /health, /predict)
make demo           # Streamlit demo (CPU, offline)
```

Without GNU Make on Windows, replace `make <t>` with `./make.ps1 <t>`.

> **Why two transformer models?** `deberta-v3-base` gives the best accuracy and is
> the reported model, but it degrades badly under dynamic int8 quantization (its
> disentangled-attention layers are quant-sensitive). The **demo** therefore ships a
> quantized **`bert-base-uncased`** model, which quantizes cleanly and runs in
> **~46 ms/inference on CPU** (545 MB int8 artifact) — well under the 1-second target.

## Deploying to the demo laptop

The laptop runs **inference only, offline, no GPU, no Docker**:

1. On the desktop: `make train-demo && make export`. This writes the
   **int8-quantized** deployment artifact to `artifacts/absa-transformer-int8/`
   (~545 MB, CPU-only, self-contained).
2. Copy the repo **and the `artifacts/absa-transformer-int8/` folder** to the laptop
   (artifacts are gitignored because they're large — transfer them manually).
3. On the laptop: `make setup` (installs the **CPU-only** stack), then `make demo`.
4. The demo preloads the artifact and runs a warmup inference at startup, so the
   first live prediction returns in **~46 ms** — no downloads, no GPU, no Docker.

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

## Results

Track A vs Track B on the official **SemEval-2014** test split (1,600 sentences),
both scored through the same char-exact evaluation code path
(`reports/comparison.md`, figures in `reports/figures/`):

| Task | Metric | Track A — baseline | Track B — deberta-v3 |
|---|---|---|---|
| ATE | span-F1 | 0.730 | **0.864** |
| ACD | micro-F1 | 0.801 | **0.906** |
| ASC | macro-F1 | 0.599 | **0.803** |
| ASC | accuracy | 0.693 | **0.859** |

Neutral is the hardest sentiment class for both tracks (the classic ABSA
pattern); categories exist only for restaurants in SemEval-2014.

**Deployed CPU model** (int8-quantized `bert-base-uncased`, what the demo runs):
ATE span-F1 **0.803**, ACD micro-F1 **0.821**, ASC macro-F1 **0.723** — between the
baseline and deberta-v3, at ~46 ms/inference on CPU (`reports/deploy_int8_metrics.json`).

## Roadmap

- [x] **P0** Scaffold — packaging, config, logging, tooling, CI, docs.
- [x] **P1** Data — SemEval-2014 parser + fetch, splits, committed sample, dataset card.
- [x] **P2** Track A baseline — CRF (ATE) + TF-IDF/LogReg (ACD/ASC); trained + evaluated.
- [x] **P3** Track B transformer — `deberta-v3-base` fine-tuned (ATE/ACD/ASC), bf16 on RTX 4070, MLflow.
- [x] **P4** Evaluation — unified Track A vs B scoring, confusion matrices, figures in `reports/`.
- [x] **P5** Serving — FastAPI `/predict` (+ batch) & `/health`, pydantic schemas, warmup, `/docs`.
- [x] **P5.5** CPU export — int8-quantized bert artifact (`make export`) + `make benchmark` (46 ms/inference).
- [x] **P6** Demo — Streamlit UI (offline, cached examples, color-coded aspects), loads the int8 artifact.
- [ ] **P7** Docker & CI — compose (API + demo), GitHub Actions.
- [ ] **P8** Docs & paper — model card, dataset card, IEEE paper skeleton.

## License

MIT (see `pyproject.toml`).
