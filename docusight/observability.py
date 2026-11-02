"""Lightweight Prometheus-format metrics and PII-safe structured logging."""
from __future__ import annotations

import json
import logging
import re
import threading
from collections import defaultdict

from .schemas import ExtractionResult

LATENCY_BUCKETS_MS = (100, 250, 500, 1000, 2500, 5000, 10000, 20000, 60000)
PII_FIELDS = {"account_number", "ifsc", "pan", "id_number", "gstin", "date_of_birth", "account_holder",
              "employee_name", "name", "father_name", "employee_id"}


def mask(value: str | None) -> str | None:
    """Keep only the last 4 characters of a sensitive value."""
    if not value:
        return value
    return re.sub(r"[A-Za-z0-9]", "*", value[:-4]) + value[-4:] if len(value) > 4 else "****"


def redact(result: ExtractionResult) -> dict:
    """Log-safe summary of a result: PII values are masked, no raw OCR text."""
    return {
        "doc_id": result.doc_id,
        "doc_type": result.doc_type,
        "doc_type_confidence": result.doc_type_confidence,
        "needs_review": result.needs_review,
        "reasons": result.review_reasons,
        "fields": {k: (mask(f.value) if k in PII_FIELDS else f.value) for k, f in result.fields.items()},
        "timings_ms": result.timings_ms,
    }


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters: dict[tuple[str, tuple], float] = defaultdict(float)
        self.hist: dict[tuple[str, str], list[int]] = {}
        self.hist_sum: dict[tuple[str, str], float] = defaultdict(float)

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self.counters[(name, tuple(sorted(labels.items())))] += value

    def observe_ms(self, stage: str, ms: float) -> None:
        key = ("docusight_stage_latency_ms", stage)
        with self._lock:
            buckets = self.hist.setdefault(key, [0] * (len(LATENCY_BUCKETS_MS) + 1))
            for i, b in enumerate(LATENCY_BUCKETS_MS):
                if ms <= b:
                    buckets[i] += 1
            buckets[-1] += 1
            self.hist_sum[key] += ms

    def record(self, result: ExtractionResult) -> None:
        self.inc("docusight_documents_total", doc_type=result.doc_type)
        if result.needs_review:
            self.inc("docusight_review_flagged_total", doc_type=result.doc_type)
        for issue in result.issues:
            self.inc("docusight_validation_issues_total", code=issue.code, doc_type=result.doc_type)
        for stage, ms in result.timings_ms.items():
            self.observe_ms(stage, ms)

    def render(self) -> str:
        out = []
        with self._lock:
            for (name, labels), v in sorted(self.counters.items()):
                lab = ",".join(f'{k}="{val}"' for k, val in labels)
                out.append(f"{name}{{{lab}}} {v}")
            for (name, stage), buckets in sorted(self.hist.items()):
                for b, c in zip(LATENCY_BUCKETS_MS, buckets):
                    out.append(f'{name}_bucket{{stage="{stage}",le="{b}"}} {c}')
                out.append(f'{name}_bucket{{stage="{stage}",le="+Inf"}} {buckets[-1]}')
                out.append(f'{name}_sum{{stage="{stage}"}} {self.hist_sum[(name, stage)]:.1f}')
                out.append(f'{name}_count{{stage="{stage}"}} {buckets[-1]}')
        return "\n".join(out) + "\n"


metrics = Metrics()
