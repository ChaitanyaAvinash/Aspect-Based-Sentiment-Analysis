# Aspect-Based Sentiment Analysis on SemEval-2014: A Classical vs. Transformer Comparison with a Quantized CPU Deployment

> IEEE-style technical-paper skeleton. Section structure follows IEEEtran; the
> results are pre-filled from this repository's actual runs
> (`reports/comparison.md`, `reports/deploy_int8_metrics.json`). Convert to
> LaTeX (`IEEEtran.cls`, `\documentclass[conference]{IEEEtran}`) for submission.

**Authors:** _ABSA Team_
**Keywords:** aspect-based sentiment analysis, SemEval-2014, DeBERTa, CRF, model quantization

---

## Abstract

Aspect-Based Sentiment Analysis (ABSA) decomposes opinionated text into
(aspect, category, sentiment) structures, enabling fine-grained feedback beyond
document-level polarity. We present a reproducible ABSA system covering three
sub-tasks — aspect term extraction (ATE), aspect category detection (ACD), and
aspect sentiment classification (ASC) — and compare two tracks on the SemEval-2014
Task 4 benchmark: a classical, CPU-friendly baseline (CRF + TF-IDF/logistic
regression) and a fine-tuned transformer (`deberta-v3-base`). The transformer
substantially outperforms the baseline (ATE span-F1 0.73→0.86, ACD micro-F1
0.80→0.91, ASC macro-F1 0.60→0.80). We further study CPU deployment under dynamic
int8 quantization and report a practical finding: DeBERTa-v3 degrades
catastrophically under dynamic quantization, whereas a `bert-base-uncased` variant
quantizes cleanly, yielding a 545 MB, ~46 ms/inference CPU model with only a modest
accuracy cost. We release the full pipeline, evaluation, serving API, and demo.

## 1. Introduction

Document-level sentiment hides disagreement across aspects (e.g., "great food but
slow service"). ABSA addresses this by attributing sentiment to specific aspects.
This work targets three practical goals: (i) a rigorous, reproducible comparison
of a classical and a transformer approach on a standard benchmark; (ii) an
end-to-end system (data → training → evaluation → serving → demo); and (iii) an
honest account of the accuracy/efficiency trade-offs of CPU deployment via
quantization. **Contributions:** (1) a unified, char-exact evaluation of both
tracks across ATE/ACD/ASC; (2) a documented incompatibility between DeBERTa-v3 and
dynamic int8 quantization, with a working alternative; (3) an open, tested,
reproducible implementation.

## 2. Related Work

- **ABSA sub-tasks and benchmarks.** SemEval-2014 Task 4 [1] established the
  Restaurants/Laptops benchmark with aspect-term and aspect-category annotations;
  later editions (2015/2016) unified target+category opinions.
- **Classical approaches.** CRFs for sequence labeling of aspect terms; TF-IDF
  with linear classifiers (LogReg/SVM) for category and polarity.
- **Transformer approaches.** Fine-tuned encoders (BERT [2], RoBERTa, DeBERTa [3])
  dominate ABSA leaderboards; aspect marking and auxiliary-sentence formulations
  improve ASC.
- **Efficient inference.** Dynamic int8 quantization [4] is a standard CPU
  optimization; its effect is architecture-dependent — a gap this paper probes.

## 3. Methodology

### 3.1 Task formulation and data schema
Each example is normalized to `{text, aspect_terms:[{term, polarity, span}],
aspect_categories:[{category, polarity}]}`. Polarity ∈ {positive, negative,
neutral}; `conflict` is dropped.

### 3.2 Track A — classical baseline
- **ATE:** linear-chain CRF over spaCy token features (word/shape/affix/POS +
  context), BIO tags.
- **ACD:** TF-IDF (1–2 grams) → one-vs-rest logistic regression (multi-label).
- **ASC:** TF-IDF → logistic regression on the sentence with the aspect marked.

### 3.3 Track B — transformer
A shared `deberta-v3-base` encoder fine-tuned three ways: ATE as token
classification (BIO, sub-word label alignment); ACD as multi-label sequence
classification; ASC as sequence classification with the aspect delimited by
`[ASP] … [/ASP]` special tokens. Trained with the HuggingFace `Trainer`, bf16
mixed precision, early stopping.

### 3.4 CPU deployment
The trained model is exported as an int8 dynamic-quantized artifact for CPU
inference. We compare DeBERTa-v3 and BERT-base under identical quantization.

## 4. Experimental Setup

- **Data:** SemEval-2014 Task 4 (Restaurants + Laptops); official gold test split
  (1,600 sentences) used verbatim; train/val derived (stratified, seed 42).
- **Hardware:** training on a single NVIDIA RTX 4070 (12 GB, bf16); inference
  benchmarked on CPU.
- **Metrics:** ATE span exact-match P/R/F1; ACD micro/macro-F1; ASC accuracy and
  macro-F1 with per-class breakdown and confusion matrices. Both tracks are scored
  through the *same* char-exact code path for fairness.
- **Reproducibility:** fixed seeds; pinned dependencies; MLflow logging.

## 5. Results

### 5.1 Track A vs. Track B (SemEval-2014 test)

| Sub-task | Metric | Track A (baseline) | Track B (deberta-v3) |
|---|---|---|---|
| ATE | span-F1 | 0.730 | **0.864** |
| ACD | micro-F1 | 0.801 | **0.906** |
| ACD | macro-F1 | 0.770 | **0.882** |
| ASC | accuracy | 0.693 | **0.859** |
| ASC | macro-F1 | 0.599 | **0.803** |

The transformer improves every sub-task; the largest gain is ASC macro-F1
(+0.20), driven by better handling of the minority *neutral* class (see confusion
matrices, `reports/figures/`).

### 5.2 CPU deployment and quantization

| Model | ATE F1 | ACD micro-F1 | ASC macro-F1 | Size | Latency (CPU) |
|---|---|---|---|---|---|
| deberta-v3 fp32 | 0.864 | 0.906 | 0.803 | ~2.2 GB | ~270 ms |
| deberta-v3 **int8** | *broken* | *broken* | *broken* | — | — |
| bert-base **int8** | 0.803 | 0.821 | 0.723 | 545 MB | **~46 ms** |

Dynamic int8 quantization renders DeBERTa-v3 unusable (it emits degenerate spans
and near-random polarity), which we attribute to the quantization sensitivity of
its disentangled-attention projections. BERT-base quantizes cleanly and meets a
sub-100 ms CPU latency target with a modest accuracy trade-off.

## 6. Discussion

- **Neutral is the bottleneck.** Both tracks confuse neutral with the polar
  classes; class imbalance and the subtlety of neutral opinions are the main
  causes — a target for future work (focal loss, data augmentation).
- **Category asymmetry.** SemEval-2014 annotates categories for restaurants only,
  limiting ACD generality.
- **Deploy-small.** A larger model for reported accuracy plus a smaller quantized
  model for deployment is a pragmatic pattern; architecture choice matters for
  quantization robustness.

## 7. Conclusion

We built a reproducible ABSA system, showed a clear transformer advantage on
SemEval-2014, and documented a practical DeBERTa-v3/quantization incompatibility
with a working CPU deployment. Future work: joint triplet extraction (ASTE),
neutral-class improvements, and cross-domain/multilingual evaluation.

## References

[1] M. Pontiki et al., "SemEval-2014 Task 4: Aspect Based Sentiment Analysis," SemEval, 2014.
[2] J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL, 2019.
[3] P. He et al., "DeBERTa: Decoding-enhanced BERT with Disentangled Attention," ICLR, 2021.
[4] B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," CVPR, 2018.
