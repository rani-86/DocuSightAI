"""End-to-end pipeline: preprocess -> OCR -> layout -> classify -> extract -> validate."""
from __future__ import annotations

import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .classifier import DocClassifier, load_classifier, predict
from .config import Settings, settings as default_settings
from .extractor import extract_fields
from .layout import Line, build_layout, reading_order_text
from .normalize import normalize
from .ocr import OCREngine, get_engine, get_refiner
from .preprocessing import PreprocessResult, crop_roi, preprocess
from .schemas import ExtractedField, ExtractionResult, OCRToken
from .tables import extract_table
from .templates import SCHEMAS
from .validation import review_decision, validate


@dataclass
class PipelineArtifacts:
    """Intermediate outputs kept for evaluation and debugging."""

    pre: PreprocessResult
    tokens: list[OCRToken]
    lines: list[Line]
    text: str
    class_scores: dict[str, float]


class DocumentPipeline:
    def __init__(self, engine: Optional[OCREngine] = None, classifier: Optional[DocClassifier] = None,
                 cfg: Settings = default_settings):
        self.cfg = cfg
        self.engine = engine or get_engine(cfg.ocr_engine)
        self.classifier = classifier or load_classifier(cfg.model_dir)

    def run(self, image: np.ndarray, doc_id: Optional[str] = None, doc_type: Optional[str] = None,
            return_artifacts: bool = False):
        timings: dict[str, float] = {}

        @contextmanager
        def timed(stage: str):
            t0 = time.perf_counter()
            yield
            timings[stage] = round((time.perf_counter() - t0) * 1000, 1)

        with timed("preprocess"):
            pre = preprocess(image, self.cfg.max_side)
        with timed("ocr"):
            tokens = self.engine(pre.image)
        with timed("layout"):
            lines = build_layout(tokens, gap_factor=None if self.engine.phrase_level else 1.6)
            text = reading_order_text(lines)
        with timed("classify"):
            label, conf, scores = predict(self.classifier, text)
            if doc_type:  # caller-supplied type overrides the model (e.g. upload slot is known)
                label, conf = doc_type, 1.0
        schema = SCHEMAS[label]
        with timed("extract"):
            fields = extract_fields(lines, schema)
            tables = []
            if schema.table:
                table = extract_table(lines, pre.binary, schema.table)
                if table:
                    tables.append(table)
        if self.cfg.trocr_refine:
            with timed("refine"):
                self._refine(pre, fields, schema)
        with timed("validate"):
            issues = validate(label, conf, fields, tables)
            needs_review, reasons = review_decision(issues)
        timings["total"] = round(sum(timings.values()), 1)

        result = ExtractionResult(
            doc_id=doc_id or uuid.uuid4().hex[:12],
            doc_type=label,
            doc_type_confidence=round(conf, 4),
            fields=fields,
            tables=tables,
            issues=issues,
            needs_review=needs_review,
            review_reasons=reasons,
            ocr_engine=self.engine.name,
            preprocessing=pre.steps,
            image_size=(pre.image.shape[1], pre.image.shape[0]),
            timings_ms=timings,
        )
        if return_artifacts:
            return result, PipelineArtifacts(pre, tokens, lines, text, scores)
        return result

    def _refine(self, pre: PreprocessResult, fields: dict[str, ExtractedField], schema) -> None:
        """Re-read low-confidence field regions with TrOCR and keep the better reading."""
        refiner = get_refiner(self.cfg.trocr_model)
        specs = {s.name: s for s in schema.fields}
        for name, f in fields.items():
            if f.bbox is None or f.confidence >= self.cfg.field_conf_threshold:
                continue
            text, conf = refiner(crop_roi(pre.image, f.bbox))
            spec = specs[name]
            value = normalize(spec.kind, text)
            if value and conf > f.confidence and (spec.pattern is None or re.fullmatch(spec.pattern, value)):
                fields[name] = f.model_copy(update={"value": value, "raw_text": text, "confidence": round(conf, 4),
                                                    "method": f"{f.method}+trocr"})
