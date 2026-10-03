# Track comparison — SemEval-2014 test

| Metric | Track A — baseline | Track B — deberta-v3-base | Deployed — bert-base-uncased int8 |
|---|---|---|---|
| ATE — precision | 0.801 | 0.867 | 0.774 |
| ATE — recall | 0.674 | 0.868 | 0.844 |
| ATE — span-F1 | 0.732 | 0.867 | 0.807 |
| ACD — micro-F1 | 0.801 | 0.906 | 0.858 |
| ACD — macro-F1 | 0.770 | 0.882 | 0.810 |
| ASC — accuracy | 0.693 | 0.860 | 0.806 |
| ASC — macro-F1 | 0.599 | 0.804 | 0.724 |

## Laptops

| Metric | Track A — baseline | Track B — deberta-v3-base | Deployed — bert-base-uncased int8 |
|---|---|---|---|
| ATE — precision | 0.772 | 0.828 | 0.722 |
| ATE — recall | 0.583 | 0.815 | 0.777 |
| ATE — span-F1 | 0.664 | 0.821 | 0.749 |
| ACD — micro-F1 | n/a | n/a | n/a |
| ACD — macro-F1 | n/a | n/a | n/a |
| ASC — accuracy | 0.643 | 0.835 | 0.770 |
| ASC — macro-F1 | 0.582 | 0.803 | 0.716 |

## Restaurants

| Metric | Track A — baseline | Track B — deberta-v3-base | Deployed — bert-base-uncased int8 |
|---|---|---|---|
| ATE — precision | 0.815 | 0.888 | 0.803 |
| ATE — recall | 0.726 | 0.898 | 0.881 |
| ATE — span-F1 | 0.768 | 0.893 | 0.840 |
| ACD — micro-F1 | 0.801 | 0.906 | 0.858 |
| ACD — macro-F1 | 0.770 | 0.882 | 0.810 |
| ASC — accuracy | 0.722 | 0.874 | 0.827 |
| ASC — macro-F1 | 0.607 | 0.802 | 0.727 |

_ASC and ACD are scored on gold aspects/categories; ATE is predicted-vs-gold span exact match. All tracks use the same evaluation code path. Aspects labelled `conflict` are dropped from the gold data (`drop_conflict: true`), so ATE is not directly comparable to papers that keep them. Laptops have no category labels in SemEval-2014._
