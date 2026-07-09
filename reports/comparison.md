# Track A vs Track B — SemEval-2014 test

| Metric | Track A — baseline | Track B — deberta-v3 |
|---|---|---|
| ATE — precision | 0.799 | 0.864 |
| ATE — recall | 0.672 | 0.865 |
| ATE — span-F1 | 0.730 | 0.864 |
| ACD — micro-F1 | 0.801 | 0.906 |
| ACD — macro-F1 | 0.770 | 0.882 |
| ASC — accuracy | 0.693 | 0.859 |
| ASC — macro-F1 | 0.599 | 0.803 |

_ASC and ACD are scored on gold aspects/categories; ATE is predicted-vs-gold span exact match. Both tracks use the same evaluation code path._
