"""Evaluate the full pipeline on the held-out set.

Reports classification F1 (fine-tuned transformer vs TF-IDF vs keyword, all on
real OCR text), field-level accuracy, field- and page-level CER, table
row/cell accuracy, review-routing quality, confidence calibration and latency.

    python scripts/evaluate.py --data data/test                         # pre-DIQA ("legacy") gates -> reports/eval_report.md
    python scripts/evaluate.py --mode v2 --per-type 20                  # calibrated DIQA gate, stratified 80-doc subset
    python scripts/evaluate.py --mode v2 --disable-steps denoise,clahe  # ablation of individual preprocessing steps

The legacy configuration keeps its original output names (reports/predictions, reports/eval_report.*) so the published headline numbers
stay reproducible; every other configuration writes to reports/predictions_<tag> and reports/eval_report_<tag>.*.
OCR results are additionally cached by the hash of the preprocessed image (reports/ocr_cache), so configurations that leave a page's pixels
unchanged do not re-run OCR.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import docusight  # noqa: E402,F401
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from docusight import DOC_TYPES  # noqa: E402
from docusight.classifier import KeywordDocClassifier, SK_FILE, SklearnDocClassifier, load_classifier  # noqa: E402
from docusight.config import settings  # noqa: E402
from docusight.metrics import cer_counts, field_match, table_scores  # noqa: E402
from docusight.ocr import CachedEngine  # noqa: E402
from docusight.pipeline import DocumentPipeline  # noqa: E402
from docusight.schemas import ExtractionResult  # noqa: E402
from docusight.synth import load_labels  # noqa: E402
from docusight.templates import SCHEMAS  # noqa: E402
from docusight.validation import review_decision, validate  # noqa: E402


def config_tag(mode: str, disable) -> str:
    """'legacy' = the original pipeline (its outputs keep the original file names)."""
    if mode == "legacy" and not disable:
        return "legacy"
    return mode + ("_off-" + "-".join(sorted(disable)) if disable else "")


def make_pipeline(mode: str, disable, ocr_cache: str | None) -> DocumentPipeline:
    cfg = dataclasses.replace(settings, diqa_mode=mode, preprocess_disable=tuple(disable))
    pipe = DocumentPipeline(cfg=cfg)
    if ocr_cache:
        pipe.engine = CachedEngine(pipe.engine, ocr_cache)
    return pipe


def stratified_subset(records, per_type: int):
    """`per_type` documents of each type, evenly spread over the degradation range (deterministic)."""
    out = []
    for t in DOC_TYPES:
        rs = sorted((r for r in records if r["doc_type"] == t), key=lambda r: (r["degrade_level"], r["doc_id"]))
        idx = np.unique(np.linspace(0, len(rs) - 1, min(per_type, len(rs))).round().astype(int))
        out += [rs[i] for i in idx]
    return out


def run_predictions(records, data_dir: Path, cache_dir: Path, pipe: DocumentPipeline | None = None) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in records if not (cache_dir / f"{r['doc_id']}.json").exists()]
    if not todo:
        return
    pipe = pipe or DocumentPipeline()
    t0 = time.time()
    for i, rec in enumerate(todo, 1):
        img = cv2.imread(str(data_dir / rec["image"]))
        res, art = pipe.run(img, doc_id=rec["doc_id"], return_artifacts=True)
        payload = {"result": res.model_dump(mode="json"), "ocr_text": art.text,
                   "ocr_confidence": float(np.mean([t.confidence for t in art.tokens])) if art.tokens else 0.0,
                   "classifier": pipe.classifier.name}
        (cache_dir / f"{rec['doc_id']}.json").write_text(json.dumps(payload))
        if i % 10 == 0 or i == len(todo):
            print(f"[{i}/{len(todo)}] {rec['doc_id']}  elapsed {time.time() - t0:.0f}s", flush=True)


def pct(x: float) -> float:
    return round(100 * x, 2)


def bucket(level: float) -> str:
    return "light" if level < 0.5 else "medium" if level < 0.8 else "heavy"


def evaluate(records, cache_dir: Path, model_dir: Path) -> dict:
    preds = {r["doc_id"]: json.loads((cache_dir / f"{r['doc_id']}.json").read_text()) for r in records}
    results = {k: ExtractionResult.model_validate(v["result"]) for k, v in preds.items()}
    # Re-run validation on cached OCR/extraction so rule or threshold changes can be
    # evaluated offline without repeating OCR.
    for res in results.values():
        kept = [i for i in res.issues if i.code == "IMAGE_QUALITY"]  # produced by the DIQA gate at inference time, not by validate()
        res.issues = validate(res.doc_type, res.doc_type_confidence, res.fields, res.tables) + kept
        res.needs_review, res.review_reasons = review_decision(res.issues)

    # ---------------- classification (all models scored on the same OCR text)
    y_true = [r["doc_type"] for r in records]
    texts = [preds[r["doc_id"]]["ocr_text"] for r in records]
    cls = {"pipeline": [results[r["doc_id"]].doc_type for r in records]}
    cls["tfidf_logreg"] = [DOC_TYPES[i] for i in SklearnDocClassifier.load(model_dir / SK_FILE).predict_proba(texts).argmax(1)] \
        if (model_dir / SK_FILE).exists() else None
    cls["keyword"] = [DOC_TYPES[i] for i in KeywordDocClassifier().predict_proba(texts).argmax(1)]
    classification = {
        name: {"macro_f1": round(f1_score(y_true, y, average="macro"), 4),
               "per_class_f1": dict(zip(DOC_TYPES, np.round(f1_score(y_true, y, average=None, labels=list(DOC_TYPES)), 4).tolist()))}
        for name, y in cls.items() if y is not None
    }
    classification["pipeline"]["model"] = preds[records[0]["doc_id"]]["classifier"]
    classification["pipeline"]["confusion_matrix"] = {"labels": list(DOC_TYPES),
                                                      "matrix": confusion_matrix(y_true, cls["pipeline"], labels=list(DOC_TYPES)).tolist()}

    # ---------------- fields / tables / CER
    by_field: dict[str, list[bool]] = defaultdict(list)
    by_type: dict[str, list[bool]] = defaultdict(list)
    by_bucket: dict[str, list[bool]] = defaultdict(list)
    by_variant: dict[str, list[bool]] = defaultdict(list)
    by_method: dict[str, list[bool]] = defaultdict(list)
    conf_buckets: dict[str, list[bool]] = defaultdict(list)
    f_edits = f_chars = p_edits = p_chars = 0
    table_m: dict[str, list[dict]] = defaultdict(list)
    doc_rows = []

    for rec in records:
        res = results[rec["doc_id"]]
        schema = SCHEMAS[rec["doc_type"]]
        e, c = cer_counts(rec["text"], preds[rec["doc_id"]]["ocr_text"])
        p_edits, p_chars = p_edits + e, p_chars + c
        doc_errors = 0
        for spec in schema.fields:
            gt = rec["fields"].get(spec.name)
            f = res.fields.get(spec.name) if res.doc_type == rec["doc_type"] else None
            pred = f.value if f else None
            ok = field_match(spec.kind, gt, pred)
            doc_errors += not ok
            for d, k in ((by_field, f"{rec['doc_type']}.{spec.name}"), (by_type, rec["doc_type"]),
                         (by_bucket, bucket(rec["degrade_level"])), (by_variant, f"{rec['doc_type']}.v{rec['layout_variant']}")):
                d[k].append(ok)
            if f and f.value is not None:
                by_method[f.method or "none"].append(ok)
                cb = ">=0.95" if f.confidence >= 0.95 else "0.80-0.95" if f.confidence >= 0.80 else "<0.80"
                conf_buckets[cb].append(ok)
            if gt is not None:
                e, c = cer_counts(gt, pred or "")
                f_edits, f_chars = f_edits + e, f_chars + c
        if schema.table:
            pred_rows = [r.cells for r in res.tables[0].rows] if res.tables and res.doc_type == rec["doc_type"] else []
            ts = table_scores(rec["table"], pred_rows, list(schema.table.columns))
            table_m[rec["doc_type"]].append(ts)
            doc_errors += ts["row_exact"] < 1.0
        doc_rows.append({"doc_id": rec["doc_id"], "anomaly": rec["anomaly"], "errors": doc_errors,
                         "flagged": res.needs_review, "reasons": res.review_reasons, "timings": res.timings_ms,
                         "class_ok": res.doc_type == rec["doc_type"]})

    def acc(d):
        return {k: {"accuracy": pct(np.mean(v)), "n": len(v)} for k, v in sorted(d.items())}

    all_fields = [x for v in by_field.values() for x in v]
    tables = {t: {k: pct(np.mean([m[k] for m in ms])) for k in ("row_exact", "cell_acc", "row_count_ok")} | {"n_docs": len(ms)}
              for t, ms in table_m.items()}

    # ---------------- review routing
    problem = [d["anomaly"] is not None or d["errors"] > 0 or not d["class_ok"] for d in doc_rows]
    flagged = [d["flagged"] for d in doc_rows]
    tp = sum(p and f for p, f in zip(problem, flagged))
    anomalies = [d for d in doc_rows if d["anomaly"]]
    auto = [d for d in doc_rows if not d["flagged"]]
    routing = {
        "docs": len(doc_rows),
        "flag_rate": pct(np.mean(flagged)),
        "straight_through_rate": pct(1 - np.mean(flagged)),
        "problem_docs": int(sum(problem)),
        "problem_recall": pct(tp / max(1, sum(problem))),
        "flag_precision": pct(tp / max(1, sum(flagged))),
        "injected_anomaly_recall": pct(np.mean([d["flagged"] for d in anomalies])) if anomalies else None,
        "injected_anomalies": {k: sum(d["anomaly"] == k for d in anomalies) for k in ("inconsistent", "missing")},
        "auto_approved_docs_fully_correct": pct(np.mean([d["errors"] == 0 and d["class_ok"] for d in auto])) if auto else None,
        "reason_counts": {k: sum(k in d["reasons"] for d in doc_rows) for k in
                          ("MISSING_REQUIRED", "LOW_CONFIDENCE", "FORMAT_INVALID", "INCONSISTENT", "LOW_CLASS_CONFIDENCE", "IMAGE_QUALITY")},
    }

    stages = defaultdict(list)
    for d in doc_rows:
        for k, v in d["timings"].items():
            stages[k].append(v)
    latency = {k: {"p50_ms": round(float(np.percentile(v, 50)), 1), "p95_ms": round(float(np.percentile(v, 95)), 1)}
               for k, v in stages.items()}

    return {
        "dataset": {"docs": len(records), "per_type": {t: sum(r["doc_type"] == t for r in records) for t in DOC_TYPES},
                    "ocr_engine": results[records[0]["doc_id"]].ocr_engine},
        "classification": classification,
        "fields": {"overall_accuracy": pct(np.mean(all_fields)), "n_fields": len(all_fields),
                   "by_doc_type": acc(by_type), "by_field": acc(by_field), "by_degradation": acc(by_bucket),
                   "by_layout_variant": acc(by_variant), "by_extraction_method": acc(by_method),
                   "accuracy_by_confidence": acc(conf_buckets)},
        "cer": {"field_value_cer": round(f_edits / max(1, f_chars), 4), "page_cer": round(p_edits / max(1, p_chars), 4)},
        "tables": tables,
        "review_routing": routing,
        "latency": latency,
        "settings": {"field_conf_threshold": settings.field_conf_threshold, "class_conf_threshold": settings.class_conf_threshold},
        "failures": [d for d in doc_rows if d["errors"] or not d["class_ok"]][:40],
    }


def to_markdown(rep: dict) -> str:
    c, f, r = rep["classification"], rep["fields"], rep["review_routing"]
    lines = [
        "# DocuSight AI - Evaluation Report", "",
        f"Held-out set: **{rep['dataset']['docs']} documents** ({', '.join(f'{k}: {v}' for k, v in rep['dataset']['per_type'].items())}), "
        f"OCR engine: `{rep['dataset']['ocr_engine']}`.", "",
        "## Headline metrics", "",
        "| Metric | Value |", "|---|---|",
        f"| Document classification macro-F1 ({c['pipeline']['model']}) | {c['pipeline']['macro_f1']:.4f} |",
        f"| Field-level accuracy (exact match after normalisation) | {f['overall_accuracy']:.2f}% ({f['n_fields']} fields) |",
        f"| Field-value CER | {rep['cer']['field_value_cer']:.4f} |",
        f"| Page-level OCR CER (reading order) | {rep['cer']['page_cer']:.4f} |",
        f"| Injected-anomaly recall (flagged for review) | {r['injected_anomaly_recall']}% ({sum(r['injected_anomalies'].values())} docs) |",
        f"| Problem-document recall / flag precision | {r['problem_recall']}% / {r['flag_precision']}% |",
        f"| Straight-through rate | {r['straight_through_rate']}% |",
        f"| Auto-approved docs that are fully correct | {r['auto_approved_docs_fully_correct']}% |",
        f"| End-to-end latency p50 / p95 (CPU) | {rep['latency']['total']['p50_ms']:.0f} / {rep['latency']['total']['p95_ms']:.0f} ms |",
        "", "## Classification (same OCR text for every model)", "", "| Model | Macro-F1 |", "|---|---|",
    ]
    lines += [f"| {k}{' (' + v['model'] + ')' if 'model' in v else ''} | {v['macro_f1']:.4f} |" for k, v in c.items()]
    lines += ["", "## Field accuracy by document type", "", "| Doc type | Accuracy | Fields |", "|---|---|---|"]
    lines += [f"| {k} | {v['accuracy']:.2f}% | {v['n']} |" for k, v in f["by_doc_type"].items()]
    lines += ["", "## Field accuracy by field", "", "| Field | Accuracy | n |", "|---|---|---|"]
    lines += [f"| {k} | {v['accuracy']:.2f}% | {v['n']} |" for k, v in f["by_field"].items()]
    lines += ["", "## Robustness by degradation level", "", "| Degradation | Accuracy | Fields |", "|---|---|---|"]
    lines += [f"| {k} | {v['accuracy']:.2f}% | {v['n']} |" for k, v in f["by_degradation"].items()]
    lines += ["", "## Confidence calibration (accuracy of extracted values per confidence band)", "",
              "| Confidence | Accuracy | n |", "|---|---|---|"]
    lines += [f"| {k} | {v['accuracy']:.2f}% | {v['n']} |" for k, v in f["accuracy_by_confidence"].items()]
    lines += ["", "## Tables", "", "| Doc type | Row exact | Cell accuracy | Row count correct |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['row_exact']:.2f}% | {v['cell_acc']:.2f}% | {v['row_count_ok']:.2f}% |" for k, v in rep["tables"].items()]
    lines += ["", "## Review reasons", "", "| Reason | Docs |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in r["reason_counts"].items()]
    lines += ["", "## Latency by stage (CPU)", "", "| Stage | p50 ms | p95 ms |", "|---|---|---|"]
    lines += [f"| {k} | {v['p50_ms']} | {v['p95_ms']} |" for k, v in rep["latency"].items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-type", type=int, default=0, help="stratified subset: N docs per document type (spread over degradation level)")
    ap.add_argument("--mode", choices=["legacy", "v2"], default="legacy", help="preprocessing gate (legacy = pre-DIQA, reproduces the published numbers)")
    ap.add_argument("--disable-steps", default="", help="comma list of steps to switch off: perspective,deskew,despike,denoise,deblur,clahe")
    ap.add_argument("--ocr-cache", default="reports/ocr_cache", help="disk cache of OCR output keyed by preprocessed-image hash ('' to disable)")
    args = ap.parse_args()
    data_dir, out = Path(args.data), Path(args.out)
    records = load_labels(data_dir)
    if args.per_type:
        records = stratified_subset(records, args.per_type)
    if args.limit:
        records = records[:: max(1, len(records) // args.limit)][: args.limit]
    disable = [x for x in args.disable_steps.split(",") if x]
    tag = config_tag(args.mode, disable)
    suffix = "" if tag == "legacy" else f"_{tag}"
    load_classifier.cache_clear()
    run_predictions(records, data_dir, out / f"predictions{suffix}", make_pipeline(args.mode, disable, args.ocr_cache or None))
    rep = evaluate(records, out / f"predictions{suffix}", settings.model_dir)
    rep["config"] = {"mode": args.mode, "disabled_steps": disable, "per_type": args.per_type or None, "tag": tag}
    (out / f"eval_report{suffix}.json").write_text(json.dumps(rep, indent=2))
    (out / f"eval_report{suffix}.md").write_text(to_markdown(rep), encoding="utf-8")
    print(to_markdown(rep))


if __name__ == "__main__":
    main()
