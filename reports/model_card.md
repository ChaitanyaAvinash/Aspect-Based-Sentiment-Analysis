# Model Card — ABSA (Aspect-Based Sentiment Analysis)

## Overview

This project ships an ABSA system that, given review text, extracts **aspect
terms**, detects **aspect categories**, and predicts **per-aspect sentiment**
(positive / negative / neutral). Three models are provided:

| Model | Role | ATE span-F1 | ACD micro-F1 | ASC macro-F1 | ASC acc |
|---|---|---|---|---|---|
| **Track A** — CRF + TF-IDF/LogReg | classical baseline | 0.730 | 0.801 | 0.599 | 0.693 |
| **Track B** — `deberta-v3-base` (fp32) | **reported best** | **0.864** | **0.906** | **0.803** | **0.859** |
| **Deployed** — `bert-base-uncased` int8 | CPU demo/serving | 0.803 | 0.821 | 0.723 | 0.806 |

All figures are on the official **SemEval-2014 Task 4** test split (1,600
sentences), scored through one char-exact evaluation path (`reports/comparison.md`).

- **Sub-task modeling (Track B):** ATE = token classification (BIO, sub-word
  aligned); ACD = multi-label sequence classification; ASC = sequence
  classification with the aspect marked by `[ASP] … [/ASP]` tokens.
- **Version / date:** 0.1.0, 2026-07. Seeds fixed (42); runs logged to MLflow.

## Intended use

- **In scope:** research and education on ABSA; analyzing English restaurant and
  laptop reviews; a demonstrable end-to-end NLP pipeline (train → evaluate →
  serve → demo).
- **Out of scope:** high-stakes or automated decision-making; moderation or
  enforcement; non-English text; domains far from restaurants/laptops; treating
  outputs as ground truth without human review.

## Training data

SemEval-2014 Task 4 (Restaurants + Laptops), fetched via the Hugging Face mirror
`jakartaresearch/semeval-absa`. See `reports/dataset_card.md` for splits, label
distribution, license, and limitations. Key facts: English only; positive-skewed
polarity; aspect **categories exist for restaurants only**.

## Metrics & evaluation

Precision / recall / macro-F1, per-class F1, accuracy; separate scores for ATE
(span exact-match), ACD (multi-label micro/macro), and ASC (3-class). Confusion
matrices and per-class breakdowns are in `reports/figures/` and
`reports/comparison.json`. The deployed int8 model's numbers are in
`reports/deploy_int8_metrics.json`.

## Limitations & failure modes

- **Neutral is hardest.** ASC neutral F1 ≈ 0.43 (baseline) / ~0.60 (transformer);
  neutral is the minority class and is often confused with positive/negative.
- **Category is restaurant-only.** ACD was trained on restaurant categories; on
  laptop text it emits meaningless categories (no laptop category labels exist in
  SemEval-2014). Treat `category` as informative only for restaurant-like text.
- **deberta-v3 + int8 is incompatible.** Dynamic int8 quantization severely
  degrades deberta-v3 (its disentangled-attention Linear layers are
  quant-sensitive), so deployment uses a quantized `bert-base-uncased` instead —
  a small accuracy trade-off for a 46 ms/inference CPU model.
- **Domain / language shift.** Accuracy drops on out-of-domain, non-English, very
  long, or code-mixed input; such inputs are handled *gracefully* (empty result,
  truncation) but not accurately.
- **Span boundaries.** Extraction can occasionally over/under-shoot multi-word or
  punctuation-adjacent spans.

## Bias, fairness & risks

- The corpus reflects the demographics and language of its (mostly Western,
  English-speaking) reviewers; sentiment lexical cues may not transfer across
  dialects or cultures. It may under-serve minority dialects or non-native
  English.
- Sentiment models can encode spurious correlations (e.g., cuisine or brand names
  co-occurring with sentiment). No debiasing was applied; do not use outputs to
  rank or penalize businesses/products without human oversight.
- No personal data is used at training time; at inference, submitted text is not
  stored by the service.

## Environmental & compute footprint

- Training: a single RTX 4070 (bf16), ~2 min/sub-task for deberta-v3 (~6–7 min
  total) — small footprint. Baseline trains on CPU in seconds.
- Inference: CPU-only, int8, ~46 ms/inference, 545 MB artifact — cheap to serve.

## Sustainable Development Goals (SDGs)

- **SDG 9 (Industry, Innovation & Infrastructure):** demonstrates accessible,
  reproducible NLP infrastructure — a big model for accuracy, a small quantized
  model for low-cost CPU deployment.
- **SDG 12 (Responsible Consumption & Production):** structured aspect-level
  feedback helps producers act on specific product/service issues (quality,
  service, price), supporting more responsive and efficient production. The
  efficient int8 CPU deployment also reduces the energy cost of serving.

## How to use

```python
from absa.serving.service import ModelService
from absa.config import get_settings

svc = ModelService.load(get_settings())      # loads the int8 CPU artifact
print(svc.predict("The battery lasts all day but the screen is dim."))
```

Or via the API (`make serve`, `POST /predict`) / the demo (`make demo`).
