"""FastAPI inference service.

    uvicorn docusight.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from . import DOC_TYPES, __version__
from .classifier import predict
from .config import settings
from .observability import JsonFormatter, metrics, redact
from .pipeline import DocumentPipeline
from .preprocessing import decode_image
from .schemas import ExtractionResult

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
log = logging.getLogger("docusight.api")
log.addHandler(handler)
log.setLevel(logging.INFO)
log.propagate = False

_state: dict = {}
_infer_lock = threading.Lock()  # the Paddle predictor is not thread-safe
_store: "OrderedDict[str, ExtractionResult]" = OrderedDict()  # recent results, in-memory only
STORE_MAX = 500


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    _state["pipeline"] = DocumentPipeline()  # load OCR + classifier once, not per request
    log.info("models loaded", extra={"extra_fields": {
        "ocr_engine": _state["pipeline"].engine.name, "classifier": _state["pipeline"].classifier.name,
        "load_ms": round((time.perf_counter() - t0) * 1000)}})
    yield
    _state.clear()


app = FastAPI(title="DocuSight AI", version=__version__, lifespan=lifespan,
              description="Layout-aware OCR, classification and field extraction for financial documents.")


class ClassifyResponse(BaseModel):
    doc_type: str
    confidence: float
    scores: dict[str, float]


class ReviewItem(BaseModel):
    doc_id: str
    doc_type: str
    reasons: list[str]
    issues: int


async def _read_image(file: UploadFile):
    if file.content_type not in settings.allowed_mime:
        raise HTTPException(415, f"Unsupported content type {file.content_type}")
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, "File too large")
    try:
        return decode_image(data)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _run(image, doc_type: Optional[str]) -> ExtractionResult:
    with _infer_lock:
        return _state["pipeline"].run(image, doc_type=doc_type)


def _remember(result: ExtractionResult) -> None:
    _store[result.doc_id] = result
    while len(_store) > STORE_MAX:
        _store.popitem(last=False)


@app.get("/health")
def health():
    p = _state.get("pipeline")
    return {"status": "ok" if p else "loading", "version": __version__,
            "ocr_engine": p.engine.name if p else None, "classifier": p.classifier.name if p else None}


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    return metrics.render()


@app.post("/v1/documents/extract", response_model=ExtractionResult)
async def extract(file: UploadFile = File(...),
                  doc_type: Optional[str] = Query(None, description=f"Skip classification; one of {DOC_TYPES}")):
    if doc_type and doc_type not in DOC_TYPES:
        raise HTTPException(422, f"doc_type must be one of {DOC_TYPES}")
    image = await _read_image(file)
    try:
        result = await run_in_threadpool(_run, image, doc_type)
    except Exception:
        metrics.inc("docusight_requests_total", endpoint="extract", status="error")
        log.exception("extraction failed")
        raise HTTPException(500, "Extraction failed")
    metrics.inc("docusight_requests_total", endpoint="extract", status="ok")
    metrics.record(result)
    _remember(result)
    log.info("document processed", extra={"extra_fields": redact(result)})
    return result


@app.post("/v1/documents/extract-batch", response_model=list[ExtractionResult])
async def extract_batch(files: list[UploadFile] = File(...)):
    if len(files) > 20:
        raise HTTPException(413, "At most 20 files per batch")
    return [await extract(f, None) for f in files]


@app.post("/v1/documents/classify", response_model=ClassifyResponse)
async def classify(file: UploadFile = File(...)):
    image = await _read_image(file)
    pipe: DocumentPipeline = _state["pipeline"]

    def _classify():
        from .layout import build_layout, reading_order_text
        from .preprocessing import preprocess

        with _infer_lock:
            tokens = pipe.engine(preprocess(image, settings.max_side).image)
        text = reading_order_text(build_layout(tokens, None if pipe.engine.phrase_level else 1.6))
        return predict(pipe.classifier, text)

    label, conf, scores = await run_in_threadpool(_classify)
    return ClassifyResponse(doc_type=label, confidence=round(conf, 4), scores={k: round(v, 4) for k, v in scores.items()})


@app.get("/v1/documents/{doc_id}", response_model=ExtractionResult)
def get_document(doc_id: str):
    if doc_id not in _store:
        raise HTTPException(404, "Unknown or expired doc_id")
    return _store[doc_id]


@app.get("/v1/review-queue", response_model=list[ReviewItem])
def review_queue(limit: int = Query(50, le=500)):
    items = [r for r in reversed(_store.values()) if r.needs_review][:limit]
    return [ReviewItem(doc_id=r.doc_id, doc_type=r.doc_type, reasons=r.review_reasons, issues=len(r.issues)) for r in items]
