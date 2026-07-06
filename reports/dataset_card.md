# Dataset Card — ABSA (SemEval-2014 Task 4 + committed sample)

This project uses two datasets: the **real SemEval-2014 Task 4** benchmark
(fetched at data-prep time) and a tiny **committed synthetic sample** that ships
in the repo so tests and the demo run fully offline.

---

## 1. SemEval-2014 Task 4 (primary benchmark)

**Task.** Aspect-Based Sentiment Analysis on customer reviews across two
domains — **Restaurants** and **Laptops**. Annotations cover aspect terms (with
character spans + polarity) and, for restaurants, aspect categories (with
polarity).

**Source / retrieval.** Downloaded by `scripts/prepare_data.py` from the Hugging
Face mirror [`jakartaresearch/semeval-absa`](https://huggingface.co/datasets/jakartaresearch/semeval-absa)
(configs `restaurant`, `laptop`). The mirror exposes `train` and `validation`
splits; the `validation` split corresponds to the **official SemEval-2014 gold
test set** and is used verbatim as our test split. Our validation split is
re-derived from the training pool (stratified, seeded). If the download is
unavailable, prep falls back to the committed sample (§2). The official XML
files can also be dropped into `data/raw/` and are parsed directly.

**License / intended use.** SemEval-2014 Task 4 data is released **for research
purposes**. Restaurant reviews derive from the dataset of Ganu et al. (2009);
laptop reviews derive from customer reviews. Respect the original SemEval terms;
do not redistribute the raw corpus commercially. This repo does **not** commit
the raw/processed SemEval data (it is gitignored) — only the download tooling.

**Normalized schema** (see `absa.data.schema.ABSAExample`):

```json
{
  "id": "3121",
  "text": "But the staff was so horrible to us.",
  "domain": "restaurants",
  "aspect_terms": [{"term": "staff", "polarity": "negative", "start": 8, "end": 13}],
  "aspect_categories": [{"category": "service", "polarity": "negative"}]
}
```

Polarity ∈ {positive, negative, neutral}. The `conflict` label is dropped by
default (`configs/data.yaml: schema.drop_conflict`). Aspect terms whose recorded
span does not match the text are dropped as annotation noise.

**Splits** (produced by `make prepare-data`, seed 42):

| Split | Examples | Aspect terms | Aspect categories | Domains (rest / lap) |
|------:|---------:|-------------:|------------------:|:---------------------|
| train | 5,415 | 5,250 | 3,122 | 2,713 / 2,702 |
| val   |   677 |   686 |   396 | 331 / 346 |
| test  | 1,600 | 1,758 |   973 | 800 / 800 (official gold) |

**Label distribution.**

- *Aspect-term polarity* — train: positive 2,796 · negative 1,489 · neutral 965.
  test: positive 1,069 · negative 324 · neutral 365. Clear **positive skew**.
- *Aspect-category polarity* (restaurants only) — train: positive 1,938 ·
  negative 739 · neutral 445.
- *Categories* (restaurants only in 2014): `food`, `anecdotes/miscellaneous`,
  `service`, `ambience`, `price`. **Laptops have no category annotations** in
  SemEval-2014, so ACD is trained/evaluated on restaurants only.

**Known limitations.**
- Domain-specific (restaurants + laptops); not representative of other domains.
- English only.
- Class imbalance (positive-skewed; neutral is the minority) — report macro-F1,
  not just accuracy.
- Category annotations exist for restaurants only (asymmetry across domains).
- Short, single-sentence reviews; long documents are out of distribution.

---

## 2. Committed sample (`data/sample/reviews.jsonl`)

**Purpose.** A tiny, hand-written **synthetic** set so `pytest`, CI, and the
Streamlit demo run with **zero downloads**. It is *not* SemEval data and must not
be used to report benchmark results.

**Contents.** 35 examples (18 restaurants, 17 laptops), 45 aspect terms, 52
categories, generated reproducibly by `data/sample/build_sample.py` (spans are
computed from the surface form, never hand-counted).

| Field | positive | negative | neutral |
|------|---------:|---------:|--------:|
| aspect-term polarity | 23 | 17 | 5 |
| aspect-category polarity | 26 | 20 | 6 |

Categories span a small controlled vocabulary (`food`, `service`, `ambience`,
`price`, `misc`, plus laptop-oriented `battery`, `design`, `performance`,
`display`, `support`).

**Limitations.** Synthetic, tiny, and clean by construction — useful only for
plumbing/tests/demo, not for measuring model quality.

---

## Reproduction

```bash
make prepare-data                     # auto: xml -> HF mirror -> sample
python scripts/prepare_data.py --offline   # force the offline sample
python data/sample/build_sample.py    # regenerate the committed sample
```
