"""Track A — classical, CPU-friendly, interpretable baseline.

* ATE: linear-chain CRF (sklearn-crfsuite) over spaCy token features (BIO).
* ACD: TF-IDF + one-vs-rest logistic regression (multi-label), restaurants only.
* ASC: TF-IDF + LogisticRegression/LinearSVC on the sentence with the aspect
  marked.

Heavy imports (spaCy / sklearn / sklearn_crfsuite) are done lazily so that
``import absa.models`` stays cheap and dependency-free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from absa.data.preprocessing import clean_text
from absa.data.schema import ABSAExample, Polarity
from absa.models.base import AspectPrediction

_ASC_MARKER = "[ASP]"
_NLP: Any = None


def _get_nlp() -> Any:
    """Lazily load a lightweight spaCy pipeline (tagger only) as a singleton."""
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    return _NLP


# --------------------------------------------------------------------------- #
# CRF feature engineering
# --------------------------------------------------------------------------- #
def _token_features(tokens: list[Any], i: int) -> dict[str, Any]:
    tok = tokens[i]
    word = tok.text
    feats: dict[str, Any] = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "suffix3": word[-3:],
        "suffix2": word[-2:],
        "prefix2": word[:2],
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "pos": tok.pos_,
        "tag": tok.tag_,
    }
    if i > 0:
        prev = tokens[i - 1]
        feats.update(
            {"-1:lower": prev.text.lower(), "-1:pos": prev.pos_, "-1:istitle": prev.text.istitle()}
        )
    else:
        feats["BOS"] = True
    if i < len(tokens) - 1:
        nxt = tokens[i + 1]
        feats.update(
            {"+1:lower": nxt.text.lower(), "+1:pos": nxt.pos_, "+1:istitle": nxt.text.istitle()}
        )
    else:
        feats["EOS"] = True
    return feats


def _sentence_features(tokens: list[Any]) -> list[dict[str, Any]]:
    return [_token_features(tokens, i) for i in range(len(tokens))]


def _bio_labels(tokens: list[Any], spans: list[tuple[int, int]]) -> list[str]:
    tags = ["O"] * len(tokens)
    for start, end in spans:
        if start < 0 or end <= start:
            continue
        first = True
        for i, tok in enumerate(tokens):
            t_start, t_end = tok.idx, tok.idx + len(tok.text)
            if t_start < end and t_end > start:
                tags[i] = "B-ASP" if first else "I-ASP"
                first = False
    return tags


def _decode_spans(tokens: list[Any], tags: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cur: list[int] | None = None
    for tok, tag in zip(tokens, tags, strict=True):
        t_start, t_end = tok.idx, tok.idx + len(tok.text)
        if tag == "B-ASP":
            if cur is not None:
                spans.append((cur[0], cur[1]))
            cur = [t_start, t_end]
        elif tag == "I-ASP" and cur is not None:
            cur[1] = t_end
        else:
            if cur is not None:
                spans.append((cur[0], cur[1]))
                cur = None
    if cur is not None:
        spans.append((cur[0], cur[1]))
    return spans


# --------------------------------------------------------------------------- #
# Sub-models
# --------------------------------------------------------------------------- #
class BaselineATE:
    """Aspect Term Extraction via CRF."""

    def __init__(self, c1: float = 0.1, c2: float = 0.1, max_iterations: int = 100) -> None:
        self.c1, self.c2, self.max_iterations = c1, c2, max_iterations
        self.crf: Any = None

    def fit(self, examples: list[ABSAExample]) -> BaselineATE:
        import sklearn_crfsuite

        nlp = _get_nlp()
        x_feats: list[list[dict[str, Any]]] = []
        y_tags: list[list[str]] = []
        for ex in examples:
            tokens = list(nlp(ex.text))
            if not tokens:
                continue
            spans = [t.span for t in ex.aspect_terms if t.is_explicit]
            x_feats.append(_sentence_features(tokens))
            y_tags.append(_bio_labels(tokens, spans))
        self.crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=self.c1,
            c2=self.c2,
            max_iterations=self.max_iterations,
            all_possible_transitions=True,
        )
        self.crf.fit(x_feats, y_tags)
        return self

    def predict_spans(self, text: str) -> list[tuple[int, int, str]]:
        if self.crf is None:
            raise RuntimeError("BaselineATE is not fitted")
        tokens = list(_get_nlp()(text))
        if not tokens:
            return []
        tags = self.crf.predict_single(_sentence_features(tokens))
        return [(s, e, text[s:e]) for s, e in _decode_spans(tokens, tags)]


class BaselineACD:
    """Aspect Category Detection via multi-label TF-IDF + one-vs-rest logreg."""

    def __init__(self, max_features: int = 20000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.max_features, self.ngram_range = max_features, ngram_range
        self.vectorizer: Any = None
        self.clf: Any = None
        self.mlb: Any = None

    def fit(self, examples: list[ABSAExample]) -> BaselineACD:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import MultiLabelBinarizer

        texts: list[str] = []
        labels: list[list[str]] = []
        for ex in examples:
            cats = sorted({c.category for c in ex.aspect_categories})
            if cats:  # only examples annotated with categories (restaurants in 2014)
                texts.append(ex.text)
                labels.append(cats)
        if not texts:
            return self  # no category supervision available
        self.mlb = MultiLabelBinarizer()
        y = self.mlb.fit_transform(labels)
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range, max_features=self.max_features
        )
        x = self.vectorizer.fit_transform(texts)
        self.clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
        self.clf.fit(x, y)
        return self

    @property
    def is_fitted(self) -> bool:
        return self.mlb is not None

    def predict(self, text: str, threshold: float = 0.5) -> list[tuple[str, float]]:
        if not self.is_fitted:
            return []
        x = self.vectorizer.transform([text])
        proba = self.clf.predict_proba(x)[0]
        results = [
            (str(cat), float(p))
            for cat, p in zip(self.mlb.classes_, proba, strict=True)
            if p >= threshold
        ]
        return sorted(results, key=lambda cp: -cp[1])

    def top_category(self, text: str) -> str | None:
        if not self.is_fitted:
            return None
        x = self.vectorizer.transform([text])
        proba = self.clf.predict_proba(x)[0]
        idx = int(proba.argmax())
        return str(self.mlb.classes_[idx]) if proba[idx] > 0 else None


class BaselineASC:
    """Aspect Sentiment Classification via TF-IDF + linear classifier."""

    def __init__(
        self,
        classifier: str = "logreg",
        max_features: int = 20000,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self.classifier = classifier
        self.max_features, self.ngram_range = max_features, ngram_range
        self.vectorizer: Any = None
        self.clf: Any = None
        self.classes_: list[str] = []

    def _render(self, text: str, term: str) -> str:
        return f"{text} {_ASC_MARKER} {term}"

    def _make_clf(self) -> Any:
        if self.classifier == "linearsvc":
            from sklearn.svm import LinearSVC

            return LinearSVC(class_weight="balanced")
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000, class_weight="balanced")

    def fit(self, examples: list[ABSAExample]) -> BaselineASC:
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts: list[str] = []
        labels: list[str] = []
        for ex in examples:
            for term in ex.aspect_terms:
                texts.append(self._render(ex.text, term.term))
                labels.append(term.polarity)
        if not texts:
            return self
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range, max_features=self.max_features
        )
        x = self.vectorizer.fit_transform(texts)
        self.clf = self._make_clf()
        self.clf.fit(x, labels)
        self.classes_ = list(self.clf.classes_)
        return self

    def predict(self, text: str, term: str) -> tuple[Polarity, float]:
        if self.clf is None:
            raise RuntimeError("BaselineASC is not fitted")
        import numpy as np

        x = self.vectorizer.transform([self._render(text, term)])
        if hasattr(self.clf, "predict_proba"):
            proba = self.clf.predict_proba(x)[0]
        else:  # LinearSVC -> softmax over decision margins
            scores = np.atleast_1d(np.asarray(self.clf.decision_function(x))[0])
            if scores.shape[0] == 1:  # binary
                s = 1.0 / (1.0 + np.exp(-scores[0]))
                proba = np.array([1.0 - s, s])
            else:
                exp = np.exp(scores - scores.max())
                proba = exp / exp.sum()
        idx = int(np.argmax(proba))
        return self.classes_[idx], float(proba[idx])  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class BaselineABSA:
    """End-to-end Track A pipeline (implements the ABSAPipeline protocol)."""

    ate: BaselineATE = field(default_factory=BaselineATE)
    acd: BaselineACD = field(default_factory=BaselineACD)
    asc: BaselineASC = field(default_factory=BaselineASC)

    @classmethod
    def from_config(cls, model_cfg: dict[str, Any]) -> BaselineABSA:
        base = model_cfg.get("baseline", {})
        acd_cfg = base.get("acd", {})
        asc_cfg = base.get("asc", {})
        return cls(
            ate=BaselineATE(),
            acd=BaselineACD(
                max_features=int(acd_cfg.get("max_features", 20000)),
                ngram_range=tuple(acd_cfg.get("ngram_range", (1, 2))),
            ),
            asc=BaselineASC(
                classifier=str(asc_cfg.get("classifier", "logreg")),
                max_features=int(asc_cfg.get("max_features", 20000)),
                ngram_range=tuple(asc_cfg.get("ngram_range", (1, 2))),
            ),
        )

    def fit(self, examples: list[ABSAExample]) -> BaselineABSA:
        self.ate.fit(examples)
        self.acd.fit(examples)
        self.asc.fit(examples)
        return self

    def predict(self, text: str) -> list[AspectPrediction]:
        cleaned = clean_text(text)
        if not cleaned:
            return []
        category = self.acd.top_category(cleaned)
        predictions: list[AspectPrediction] = []
        for start, end, surface in self.ate.predict_spans(cleaned):
            sentiment, confidence = self.asc.predict(cleaned, surface)
            predictions.append(
                AspectPrediction(
                    aspect=surface,
                    sentiment=sentiment,
                    confidence=confidence,
                    start=start,
                    end=end,
                    category=category,
                )
            )
        return predictions

    def predict_batch(self, texts: list[str]) -> list[list[AspectPrediction]]:
        return [self.predict(t) for t in texts]

    def save(self, directory: str | Path) -> Path:
        import joblib

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "baseline.joblib")
        meta = {
            "track": "baseline",
            "asc_classes": self.asc.classes_,
            "acd_categories": (
                [str(c) for c in self.acd.mlb.classes_] if self.acd.is_fitted else []
            ),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return directory / "baseline.joblib"

    @classmethod
    def load(cls, directory: str | Path) -> BaselineABSA:
        import joblib

        model = joblib.load(Path(directory) / "baseline.joblib")
        if not isinstance(model, cls):
            raise TypeError(f"Expected BaselineABSA, got {type(model)!r}")
        return model


__all__ = ["BaselineABSA", "BaselineACD", "BaselineASC", "BaselineATE"]
