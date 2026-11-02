"""Document classification over OCR text.

* HFDocClassifier   - fine-tuned Hugging Face transformer (PyTorch), primary model.
* SklearnDocClassifier - TF-IDF + logistic regression baseline.
* KeywordDocClassifier - zero-training fallback from schema keywords.

`load_classifier()` picks the best available artefact in `models/`.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np

from . import DOC_TYPES
from .config import settings
from .templates import SCHEMAS

log = logging.getLogger(__name__)

HF_DIR = "doc_classifier_hf"
SK_FILE = "doc_classifier_tfidf.joblib"


class DocClassifier(Protocol):
    name: str

    def predict_proba(self, texts: list[str]) -> np.ndarray: ...


def predict(clf: DocClassifier, text: str) -> tuple[str, float, dict[str, float]]:
    probs = clf.predict_proba([text])[0]
    scores = {t: float(p) for t, p in zip(DOC_TYPES, probs)}
    label = max(scores, key=scores.get)  # type: ignore[arg-type]
    return label, scores[label], scores


def prepare_text(text: str, max_chars: int = 1500) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()[:max_chars]


class KeywordDocClassifier:
    name = "keyword"

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        out = []
        for text in texts:
            t = prepare_text(text, 10_000)
            hits = np.array([sum(kw in t for kw in SCHEMAS[d].keywords) for d in DOC_TYPES], dtype=float)
            e = np.exp(hits - hits.max())
            out.append(e / e.sum())
        return np.vstack(out)


class SklearnDocClassifier:
    name = "tfidf_logreg"

    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    @staticmethod
    def build():
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline

        features = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ])
        return SklearnDocClassifier(Pipeline([("tfidf", features), ("clf", LogisticRegression(C=4.0, max_iter=2000))]))

    def fit(self, texts: list[str], labels: list[str]) -> "SklearnDocClassifier":
        self.pipeline.fit([prepare_text(t) for t in texts], labels)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        probs = self.pipeline.predict_proba([prepare_text(t) for t in texts])
        order = [list(self.pipeline.classes_).index(d) for d in DOC_TYPES]
        return probs[:, order]

    def save(self, path: Path) -> None:
        import joblib

        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path) -> "SklearnDocClassifier":
        import joblib

        return cls(joblib.load(path))


class HFDocClassifier:
    name = "hf_transformer"

    def __init__(self, model_dir: Path, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        id2label = self.model.config.id2label
        self.order = [next(int(i) for i, lab in id2label.items() if lab == d) for d in DOC_TYPES]

    def predict_proba(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer([prepare_text(t) for t in texts[i:i + batch_size]], truncation=True,
                                 max_length=256, padding=True, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**enc).logits
            out.append(logits.softmax(-1).cpu().numpy()[:, self.order])
        return np.vstack(out)


@lru_cache(maxsize=1)
def load_classifier(model_dir: Path | None = None) -> DocClassifier:
    model_dir = model_dir or settings.model_dir
    if (model_dir / HF_DIR / "config.json").exists():
        try:
            return HFDocClassifier(model_dir / HF_DIR)
        except Exception as exc:  # pragma: no cover - fall back rather than fail the request
            log.warning("HF classifier unavailable (%s); falling back", exc)
    if (model_dir / SK_FILE).exists():
        return SklearnDocClassifier.load(model_dir / SK_FILE)
    log.warning("No trained classifier found in %s; using keyword fallback", model_dir)
    return KeywordDocClassifier()
