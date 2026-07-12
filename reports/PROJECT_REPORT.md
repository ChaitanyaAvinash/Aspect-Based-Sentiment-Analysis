# ABSA — Aspect-Based Sentiment Analysis: Full Project Report

_A reproducible, end-to-end Aspect-Based Sentiment Analysis system comparing a
classical baseline against a fine-tuned transformer on SemEval-2014, with a
quantized CPU deployment._

Version 0.1.0 · Report generated from the project's actual runs
(`reports/comparison.json`, `reports/deploy_int8_metrics.json`,
`data/processed/stats.json`).

---

## 1. Executive summary

Aspect-Based Sentiment Analysis (ABSA) breaks a review down into
*(aspect, category, sentiment)* triples, so "great food but slow service"
becomes `food → positive`, `service → negative` instead of a single muddy
score. We built a complete ABSA system covering three sub-tasks — **ATE**
(aspect-term extraction), **ACD** (aspect-category detection), and **ASC**
(aspect sentiment classification) — and compared two approaches on the standard
**SemEval-2014 Task 4** benchmark.

**Headline result (SemEval-2014 gold test, 1,600 sentences):** the fine-tuned
transformer beats the classical baseline on every sub-task —
**ATE F1 0.730 → 0.864**, **ACD micro-F1 0.801 → 0.906**, **ASC macro-F1
0.599 → 0.803**. For deployment, we ship a quantized model that runs in
**~46 ms/inference on CPU** (545 MB), fully offline.

The system is fully reproducible: pinned dependencies, fixed seeds, 89 automated
tests at 88% coverage, MLflow experiment tracking, a FastAPI service, and an
offline Streamlit demo.

---

## 2. The idea and motivation

Document-level sentiment ("4 stars") hides the *why*. A restaurant may have
excellent food and terrible service; a laptop great battery but a poor keyboard.
Producers and platforms need **aspect-level** feedback to act on specific
issues. ABSA provides exactly that structure.

**Goals of this project:**
1. A rigorous, fair, reproducible comparison of a **classical** and a
   **transformer** approach on a standard benchmark.
2. A **complete engineering pipeline**: data → training → evaluation → serving →
   demo, not a notebook dump.
3. An honest account of the **accuracy vs. efficiency** trade-offs of running
   modern NLP models on a CPU via quantization.

The system runs across **two machines by design**: training on an RTX 4070
desktop (GPU, bf16 mixed precision), and a live demo on a CPU-only laptop
(inference only, fully offline, no Docker).

---

## 3. Approach

We implement two comparable "tracks", plus a deployment model.

### Track A — classical baseline (CPU, interpretable)
- **ATE:** a linear-chain **CRF** (`sklearn-crfsuite`) over spaCy token features
  (word/shape/affix/POS + context), predicting BIO tags.
- **ACD:** **TF-IDF** (1–2 grams) → one-vs-rest **logistic regression**
  (multi-label).
- **ASC:** **TF-IDF** → **logistic regression** on the sentence with the aspect
  marked.

### Track B — transformer (primary, best accuracy)
A single shared **`microsoft/deberta-v3-base`** encoder fine-tuned three ways
with the HuggingFace `Trainer` (bf16, early stopping):
- **ATE** as token classification (BIO, sub-word label alignment).
- **ACD** as multi-label sequence classification.
- **ASC** as sequence classification with the aspect delimited by `[ASP] … [/ASP]`
  special tokens.

### Deployment model — quantized `bert-base-uncased` (CPU)
The demo/serving path runs an **int8 dynamic-quantized `bert-base-uncased`**
model — see §8 for why this differs from the reported best model.

---

## 4. Data

### Source and splits
We use **SemEval-2014 Task 4** (Restaurants + Laptops), fetched by
`scripts/prepare_data.py` from the Hugging Face mirror
[`jakartaresearch/semeval-absa`](https://huggingface.co/datasets/jakartaresearch/semeval-absa).
The mirror's `validation` split is the **official SemEval-2014 gold test set** and
is used verbatim as our test split; our validation split is re-derived from the
training pool (stratified, seed 42).

| Split | Examples | Aspect terms | Aspect categories | Domains (rest / lap) |
|------:|---------:|-------------:|------------------:|:---------------------|
| train | 5,415 | 5,250 | 3,122 | 2,713 / 2,702 |
| val | 677 | 686 | 396 | 331 / 346 |
| **test (gold)** | **1,600** | **1,758** | **973** | 800 / 800 |

### Label distribution (why neutral is hard)
Aspect-term **polarity** is strongly positive-skewed, and **neutral is the
minority** class:

| | positive | negative | neutral |
|---|---:|---:|---:|
| train terms | 2,796 | 1,489 | 965 |
| test terms | 1,069 | 324 | 365 |

**Categories exist for restaurants only** in SemEval-2014: `food`,
`service`, `price`, `ambience`, `anecdotes/miscellaneous`. Laptops have no
category labels, so ACD is trained and evaluated on restaurant text only.

A tiny **synthetic sample** (35 examples, `data/sample/reviews.jsonl`) is
committed so tests, CI, and the demo run with **zero downloads**.

### Data license
SemEval-2014 Task 4 data is distributed **for research purposes only**. The raw
and processed corpus is **not** committed to this repository (it is gitignored);
only the download/parse tooling and the synthetic sample are versioned. Full
details: `reports/dataset_card.md`.

---

## 5. The codebase

`src/` layout, Python 3.12, package `absa`. 79 tracked files.

```
src/absa/
  config.py            pydantic-settings (env/.env) + YAML config loader
  logging.py           structlog structured logging
  data/
    schema.py          ABSAExample / AspectTerm / AspectCategory (pydantic, span-validated)
    semeval.py         SemEval-2014 / 2015-16 XML parser
    preprocessing.py   cleaning, offset-preserving tokenizer, BIO tagging
    splits.py          deterministic stratified train/val/test split
    io.py, stats.py    JSONL read/write, corpus statistics
  models/
    base.py            AspectPrediction + pipeline interfaces
    baseline.py        Track A: CRF (ATE) + TF-IDF/LogReg (ACD, ASC)
    transformer.py     Track B: encoders, dataset builders, TransformerABSA inference
  training/
    metrics.py         span / multi-label / classification metrics
    evaluation.py      uniform ATE/ACD/ASC scoring (fair across tracks)
    hf_trainer.py      HuggingFace Trainer orchestration (bf16, early stop, MLflow)
  serving/
    schemas.py         pydantic request/response models
    service.py         ModelService: resolve/load/warmup a pipeline
    app.py             FastAPI app (/predict, /predict/batch, /health, /docs)
    render.py          aspect highlighting for the demo
scripts/               prepare_data, train, evaluate, export_model, benchmark
app/streamlit_app.py   offline demo UI with a fast/accurate model switch
configs/               data.yaml, model.yaml, training.yaml
reports/               metrics, figures, model card, dataset card, paper, this report
docker/                Dockerfile + docker-compose (separate deliverable)
tests/                 89 unit + integration tests
```

**Design principles baked in:** no secrets in code (env/`.env`); heavy
imports (torch/spaCy/sklearn) are lazy so the base package stays light and CI runs
without them; fixed seeds everywhere; both tracks evaluated through the *same*
code path for a fair comparison.

---

## 6. Results

All numbers are on the **official SemEval-2014 gold test split (1,600
sentences)**, scored through one **char-exact** evaluation path so Track A and
Track B are directly comparable. ASC and ACD are scored on gold aspects/
categories (isolating each component); ATE is predicted-vs-gold span exact match.

### 6.1 Headline comparison

| Sub-task | Metric | Track A (baseline) | Track B (deberta-v3) | Δ |
|---|---|---:|---:|---:|
| ATE | precision | 0.799 | 0.864 | +0.065 |
| ATE | recall | 0.672 | 0.865 | +0.193 |
| ATE | **span-F1** | **0.730** | **0.864** | **+0.134** |
| ACD | micro-precision | 0.787 | 0.936 | +0.149 |
| ACD | micro-recall | 0.816 | 0.878 | +0.062 |
| ACD | **micro-F1** | **0.801** | **0.906** | **+0.105** |
| ACD | macro-F1 | 0.770 | 0.882 | +0.112 |
| ASC | accuracy | 0.693 | 0.859 | +0.166 |
| ASC | macro-precision | 0.604 | 0.812 | +0.208 |
| ASC | macro-recall | 0.604 | 0.803 | +0.199 |
| ASC | **macro-F1** | **0.599** | **0.803** | **+0.204** |

### 6.2 ASC per-class (precision / recall / F1)

| Class | Baseline P | R | F1 | deberta P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| positive | 0.813 | 0.828 | 0.821 | 0.918 | 0.949 | 0.933 |
| negative | 0.493 | 0.611 | 0.545 | 0.773 | 0.870 | 0.819 |
| **neutral** | 0.507 | 0.373 | **0.430** | 0.744 | 0.589 | **0.657** |

**Neutral is the hardest class** for both models (support 365) — the classic
ABSA pattern. The transformer improves it most (+0.23 F1).

**ASC confusion matrix — deberta-v3** (rows = true, cols = predicted):

| true \\ pred | negative | neutral | positive |
|---|---:|---:|---:|
| negative | 282 | 35 | 7 |
| neutral | 67 | 215 | 83 |
| positive | 16 | 39 | 1014 |

### 6.3 ACD per-category F1

| Category | Baseline F1 | deberta-v3 F1 |
|---|---:|---:|
| food | 0.880 | 0.954 |
| service | 0.820 | 0.936 |
| price | 0.753 | 0.895 |
| anecdotes/miscellaneous | 0.752 | 0.856 |
| ambience | 0.645 | 0.768 |

### 6.4 Deployed CPU model (int8 bert-base) — what the demo runs

| Sub-task | Metric | Value |
|---|---|---:|
| ATE | precision / recall / F1 | 0.770 / 0.840 / **0.803** |
| ACD | micro-F1 / macro-F1 | **0.821** / 0.737 |
| ASC | accuracy / macro-F1 | 0.806 / **0.723** |
| ASC | per-class F1 (pos/neg/neu) | 0.893 / 0.750 / 0.526 |

The deployed model sits **between** the baseline and deberta-v3 — a small
accuracy trade for a large efficiency win.

### 6.5 Efficiency (CPU)

| Model | ATE F1 | ACD micro-F1 | ASC macro-F1 | Artifact size | Latency/inference |
|---|---:|---:|---:|---:|---:|
| deberta-v3 (fp32) | 0.864 | 0.906 | 0.803 | ~2.2 GB | ~270 ms |
| **bert int8 (deployed)** | 0.803 | 0.821 | 0.723 | **545 MB** | **~46 ms** (p95 55 ms) |

Training: single **RTX 4070** (bf16), ~2 min/sub-task for deberta-v3 (~6–7 min
total). Figures (F1 bars, confusion matrices, per-class F1, training curves) are
in `reports/figures/`.

---

## 7. Engineering & reproducibility

- **Tests:** 89 automated tests (unit + API integration), **88% coverage**
  (gate ≥ 70%). `ruff`, `black`, and `mypy` all clean.
- **CI:** GitHub Actions runs lint + type-check + tests on a light install (no
  torch); torch-dependent training code is exercised on the GPU/CPU machines.
- **Reproducibility:** all seeds fixed (42); dependencies pinned; separate
  `requirements-cpu.txt` / `requirements-gpu.txt`; runs logged to **MLflow**
  (`./mlruns`).
- **Config:** environment/secrets via `.env` (prefix `ABSA_`); experiment config
  via YAML in `configs/`.
- **Serving:** FastAPI with pydantic-validated schemas, model preload + warmup,
  OpenAPI docs at `/docs`. The demo runs offline via `make demo` (no Docker);
  Docker/compose is a separate deliverable.

---

## 8. Hurdles faced and how we solved them

1. **Dataset access.** SemEval-2014 has historically been login-gated. The first
   Hugging Face mirror required a config name (`restaurant`/`laptop`) and a second
   required executing untrusted remote code.
   **Solution:** `prepare_data.py` enumerates configs, skips any mirror needing
   `trust_remote_code` (security), and falls back to the committed sample if
   offline — so the pipeline always runs.

2. **DeBERTa-v3 tokenizer.** Its SentencePiece tokenizer needs a fast variant
   (for offset mapping) that must be converted at load time.
   **Solution:** pinned `sentencepiece` + `protobuf`; loading resolves the fast
   `DebertaV2TokenizerFast` with a robust `bert-base-uncased` fallback.

3. **Metric mismatch inflated ATE.** The transformer's SentencePiece attaches
   trailing punctuation to the aspect word (`"functions."`), so token-level
   seqeval scored 0.878 but char-exact span matching collapsed to **0.577**.
   **Solution:** a `trim_span` step strips leading/trailing punctuation on
   decode, recovering ATE to **0.864** — now consistent and comparable to the
   baseline (which uses spaCy, unaffected). Added as a tested helper.

4. **DeBERTa-v3 broke under int8 quantization.** Dynamic int8 quantization
   degraded deberta-v3 catastrophically (it emitted degenerate spans and
   near-random sentiment) — its disentangled-attention projections are
   quant-sensitive. We diagnosed this by confirming in-memory and reloaded
   quantized models were identically broken while fp32 was perfect.
   **Solution:** deploy a quantized **`bert-base-uncased`** (which quantizes
   cleanly — the canonical dynamic-quant target), keeping deberta-v3 fp32 as the
   reported best model. A documented train-big / deploy-small pattern.

5. **CI coverage without a GPU.** Measuring coverage over the whole package while
   torch isn't installed in CI would tank the number.
   **Solution:** split dependency extras into `classical` (CI) vs `dl` (torch);
   omit the torch-only training module from the coverage gate; guard heavy tests
   with `importorskip`.

6. **Environment pinning.** Python 3.12 + numpy 2.x + the full DL stack has
   sharp compatibility edges.
   **Solution:** a single pinned, mutually-compatible dependency set, verified by
   a clean CUDA install of `requirements-gpu.txt`.

---

## 9. Licensing

- **Code:** MIT License (see `LICENSE`).
- **Data:** SemEval-2014 Task 4 — **research use only**; not redistributed by this
  repo (download tooling only).
- **Models / dependencies:** open-source (PyTorch, HuggingFace Transformers,
  scikit-learn, spaCy, FastAPI, Streamlit) under their respective permissive
  licenses. `microsoft/deberta-v3-base` and `bert-base-uncased` are used under
  their model licenses for research/education.

---

## 10. Responsible AI notes

- **Limitations:** English only; restaurant/laptop domains; positive-skewed data;
  neutral is the weakest class; categories are restaurant-only (laptop category
  predictions are not meaningful). Out-of-domain / non-English input is handled
  *gracefully* (empty result, truncation) but not *accurately*.
- **Bias/fairness:** the corpus reflects mostly Western, English-speaking
  reviewers; outputs should not rank or penalize businesses/products without human
  oversight. No personal data is used in training; inference text is not stored.
- **SDG links:** SDG 9 (accessible, efficient NLP infrastructure) and SDG 12
  (aspect-level feedback for more responsive, responsible production).
- Full details: `reports/model_card.md`.

---

## 11. Current standing

- All planned phases complete: scaffold, data, both tracks, evaluation, serving,
  int8 CPU export, offline demo, Docker, and docs.
- Quality gates green: 89 tests, 88% coverage; ruff/black/mypy clean.
- Working artifacts: baseline, deberta-v3 (fp32), and int8 bert (deployed).
- Verified end-to-end on CPU: the FastAPI service and Streamlit demo both boot,
  load the model, and return correct predictions offline.

**Known caveats (honest):** GitHub Actions CI has not yet run against a real
remote; `requirements-cpu.txt` has not been install-verified on a clean CPU venv
(the GPU set was); the ONNX export path is implemented but optional/unvalidated
(int8 is the guaranteed path).

---

## 12. Future scope

- **ASTE** — aspect–sentiment–**opinion triplet** extraction (the named stretch
  goal), for richer structured output.
- **Track C — LLM baseline** — a zero/few-shot large-language-model comparison
  behind an interface, disabled by default, as a modern reference point.
- **Neutral-class improvements** — focal loss, class re-weighting, or targeted
  augmentation to lift the minority class.
- **Cross-domain / multilingual** — evaluate transfer beyond restaurants/laptops
  and English; add SemEval-2015/16 and MAMS.
- **Determinism** — enable `torch.use_deterministic_algorithms` for
  bit-reproducible transformer runs.
- **Deployment hardening** — validate ONNX Runtime export, add request auth/rate
  limiting, and a one-step artifact packaging for the demo laptop.
- **Data versioning** — optional DVC integration.

---

## 13. How to reproduce

```bash
# Install (choose one)
make setup            # CPU / demo machine        (./make.ps1 setup     on Windows)
make setup-gpu        # RTX 4070 training desktop  (./make.ps1 setup-gpu)

make prepare-data     # fetch SemEval-2014 -> data/processed (or offline sample)
make train            # baseline + deberta-v3 (best)
make evaluate         # comparison tables + figures -> reports/
make train-demo       # bert-base for the CPU demo
make export           # int8-quantize -> artifacts/absa-transformer-int8
make serve            # FastAPI on :8000 (/docs)
make demo             # Streamlit demo (CPU, offline)
make check            # lint + type + test
```

---

_This report is generated from the repository's committed metrics. To regenerate
the underlying numbers: `make evaluate` (comparison) and `make benchmark`
(latency/size)._
