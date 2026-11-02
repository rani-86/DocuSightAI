"""Preprocessing ablation: does each step actually lower error?  Plus validation of the DIQA 'retake' decision.

Configurations (all on the same stratified subset of data/test, N docs per type spread over the degradation range):
    legacy          the original hard-coded gates (pre-DIQA)         - reuses reports/predictions where cached
    v2_all_on       calibrated DIQA gate, every step available
    v2_all_off      no geometry correction, no enhancement (binarisation for table lines only)
    v2_off-<step>   all on except one step: perspective, deskew, despike, denoise, deblur, clahe

Reported per configuration: field accuracy, field-value CER, page CER, latency, and the paired document-level bootstrap 95% CI of the
difference to v2_all_on (so a 'step helps' claim comes with an interval, not just a point estimate).

OCR is the cost (~7 s/page). Results are cached by the hash of the preprocessed image (reports/ocr_cache), so a configuration only pays for
pages whose pixels it actually changed. Use --plan to see the exact number of OCR runs still needed, then run with --per-type 20:

    python scripts/run_ablation.py --plan                 # preprocess-only, prints OCR runs needed + ETA (no OCR)
    python scripts/run_ablation.py --per-type 20          # the full ablation (resumable)
    python scripts/run_ablation.py --per-type 20 --configs v2_all_on,v2_all_off    # a subset of configurations
    python scripts/run_ablation.py --report-only          # rebuild reports/ablation.md from cached results
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import docusight  # noqa: E402,F401
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import evaluate as EV  # noqa: E402
from docusight.config import PREPROCESS_STEPS, settings  # noqa: E402
from docusight.metrics import cer, field_match  # noqa: E402
from docusight.preprocessing import preprocess  # noqa: E402
from docusight.schemas import ExtractionResult  # noqa: E402
from docusight.synth import load_labels  # noqa: E402
from docusight.templates import SCHEMAS  # noqa: E402

RNG = np.random.default_rng(0)
OCR_SECONDS = 7.0


def configs() -> dict[str, tuple[str, list[str]]]:
    c = {"legacy": ("legacy", []), "v2_all_on": ("v2", []), "v2_all_off": ("v2", list(PREPROCESS_STEPS))}
    for s in PREPROCESS_STEPS:
        c[f"v2_off-{s}"] = ("v2", [s])
    return c


def pred_dir(out: Path, name: str, mode: str, disable: list[str]) -> Path:
    return out / "predictions" if name == "legacy" else out / "ablation" / name


def doc_stats(rec, res: ExtractionResult, text: str) -> dict:
    schema, errs = SCHEMAS[rec["doc_type"]], 0
    for spec in schema.fields:
        f = res.fields.get(spec.name) if res.doc_type == rec["doc_type"] else None
        errs += not field_match(spec.kind, rec["fields"].get(spec.name), f.value if f else None)
    return {"n_fields": len(schema.fields), "n_err": errs, "page_cer": cer(rec["text"], text), "bucket": EV.bucket(rec["degrade_level"]),
            "decision": res.quality.decision if res.quality else None, "latency": res.timings_ms, "steps": res.preprocessing}


def plan(records, data_dir: Path, names: list[str], out: Path, cache: Path) -> None:
    """Count the OCR runs still needed by running only the (cheap) preprocessing for every (config, doc)."""
    cfgs = configs()
    seen = {p.stem.split("_", 1)[1] for p in cache.glob("*.json")}
    total_new, report = set(), []
    for name in names:
        mode, disable = cfgs[name]
        if name == "legacy" and all((pred_dir(out, name, mode, disable) / f"{r['doc_id']}.json").exists() for r in records):
            report.append((name, 0, len(records)))
            continue
        cfg = {"mode": mode, "disable": tuple(disable)}
        new = set()
        for r in records:
            pre = preprocess(cv2.imread(str(data_dir / r["image"])), settings.max_side, mode=mode, disable=tuple(disable))
            key = hashlib.sha1(np.ascontiguousarray(pre.image).tobytes() + str(pre.image.shape).encode()).hexdigest()[:20]
            if key not in seen and key not in total_new:
                new.add(key)
        total_new |= new
        report.append((name, len(new), len(records)))
        print(f"  {name:18s} {len(new):4d} new OCR runs (of {len(records)} docs)", flush=True)
    n = len(total_new)
    print(f"\nTotal new OCR runs: {n}  ->  about {n * OCR_SECONDS / 60:.0f} min on this CPU at ~{OCR_SECONDS:.0f} s/page")


def run(records, data_dir: Path, names: list[str], out: Path, ocr_cache: str) -> None:
    cfgs = configs()
    for name in names:
        mode, disable = cfgs[name]
        d = pred_dir(out, name, mode, disable)
        t0 = time.time()
        pipe = EV.make_pipeline(mode, disable, ocr_cache)
        EV.run_predictions(records, data_dir, d, pipe)
        eng = pipe.engine
        print(f"[{name}] done in {time.time() - t0:.0f}s" + (f" (OCR cache hits {eng.hits}, misses {eng.misses})" if hasattr(eng, 'hits') else ""), flush=True)


def paired_bootstrap(a, b, n=5000):
    """Bootstrap over documents of acc(a) - acc(b); a, b = lists of (n_err, n_fields) aligned by document."""
    ea, ta = np.array([x[0] for x in a], float), np.array([x[1] for x in a], float)
    eb, tb = np.array([x[0] for x in b], float), np.array([x[1] for x in b], float)
    idx = RNG.integers(0, len(a), (n, len(a)))
    d = (1 - ea[idx].sum(1) / ta[idx].sum(1)) - (1 - eb[idx].sum(1) / tb[idx].sum(1))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def build_report(records, names: list[str], out: Path, calib_rows_path: Path) -> str:
    cfgs = configs()
    stats, reps = {}, {}
    for name in names:
        mode, disable = cfgs[name]
        d = pred_dir(out, name, mode, disable)
        if not all((d / f"{r['doc_id']}.json").exists() for r in records):
            print(f"skipping {name}: predictions incomplete in {d}")
            continue
        preds = {r["doc_id"]: json.loads((d / f"{r['doc_id']}.json").read_text()) for r in records}
        stats[name] = [doc_stats(r, ExtractionResult.model_validate(preds[r["doc_id"]]["result"]), preds[r["doc_id"]]["ocr_text"]) for r in records]
        reps[name] = EV.evaluate(records, d, settings.model_dir)
    base = "v2_all_on"
    L = ["# Preprocessing ablation", "",
         f"_Generated by `scripts/run_ablation.py`. {len(records)} documents of `data/test` ({len(records) // 4} per type, spread evenly over the degradation range; "
         "PaddleOCR, CPU). Same documents, same classifier and extraction in every row; only preprocessing differs. "
         "Delta = field accuracy minus `v2_all_on`, with a paired document-level bootstrap 95% CI (5,000 resamples). A step 'helps' only if the CI of "
         "(all on - step off) is above 0._", "",
         "| Configuration | Field acc. | Delta vs all_on [95% CI] | Field CER | Page CER | Docs w/ error | Preprocess p50 ms | Total p50 ms |", "|---|---|---|---|---|---|---|---|"]
    for name in names:
        if name not in stats:
            continue
        st, rp = stats[name], reps[name]
        acc = 1 - sum(x["n_err"] for x in st) / sum(x["n_fields"] for x in st)
        if base in stats and name != base:
            lo, hi = paired_bootstrap([(x["n_err"], x["n_fields"]) for x in st], [(x["n_err"], x["n_fields"]) for x in stats[base]])
            delta = f"{(acc - (1 - sum(x['n_err'] for x in stats[base]) / sum(x['n_fields'] for x in stats[base]))) * 100:+.2f} pp [{lo * 100:+.2f}, {hi * 100:+.2f}]"
        else:
            delta = "-"
        L.append(f"| `{name}` | {acc:.2%} | {delta} | {rp['cer']['field_value_cer']:.4f} | {rp['cer']['page_cer']:.4f} | {sum(x['n_err'] > 0 for x in st)}/{len(st)} | "
                 f"{np.median([x['latency'].get('preprocess', 0) for x in st]):.0f} | {np.median([x['latency']['total'] for x in st]):.0f} |")
    L += [""]
    if base in stats:
        L += ["## How often each step fired (`v2_all_on`)", "",
              "| Step | Docs |", "|---|---|"]
        for s in ("perspective_correct", "deskew", "despike", "denoise", "deblur", "clahe"):
            L.append(f"| {s} | {sum(any(t.startswith(s) for t in x['steps']) for x in stats[base])}/{len(stats[base])} |")
        L += ["", "## Decision distribution and error by decision (`v2_all_on`, data/test subset)", "",
              "| Decision | Docs | Mean page CER | Field accuracy |", "|---|---|---|---|"]
        for dcs in ("accept", "enhance", "retake"):
            sub = [x for x in stats[base] if x["decision"] == dcs]
            L.append(f"| {dcs} | {len(sub)} | " + (f"{np.mean([x['page_cer'] for x in sub]):.4f} | {1 - sum(x['n_err'] for x in sub) / sum(x['n_fields'] for x in sub):.2%} |" if sub else "- | - |"))
        L += [""]
        if "legacy" in stats:
            L += ["## Per degradation bucket: field accuracy (legacy vs v2_all_on vs v2_all_off)", "", "| Bucket | n docs | legacy | v2_all_on | v2_all_off |", "|---|---|---|---|---|"]
            for b in ("light", "medium", "heavy"):
                row = [b, sum(x["bucket"] == b for x in stats[base])]
                for nm in ("legacy", "v2_all_on", "v2_all_off"):
                    sub = [x for x in stats.get(nm, []) if x["bucket"] == b]
                    row.append(f"{1 - sum(x['n_err'] for x in sub) / sum(x['n_fields'] for x in sub):.2%}" if sub else "-")
                L.append("| " + " | ".join(map(str, row)) + " |")
            L += [""]
    # retake validation on the calibration images (where retakes actually occur)
    if calib_rows_path.exists():
        rows = json.loads(calib_rows_path.read_text())
        root = Path("data/quality_calib")
        dec = {}
        for r in rows:
            img = cv2.imread(str(root / "images" / f"{r['id']}.png"))
            q = preprocess(img, settings.max_side, mode="v2", disable=()).qinfo
            dec[r["id"]] = q["decision"] if q else None
        L += ["## Retake validation on the controlled calibration images", "",
              "Decision made by the calibrated gate on each calibration image that has an OCR result (`scripts/quality_vs_cer.py`). "
              "'Raw CER' = OCR of the unprocessed image; 'best-variant CER' = the lower of raw and enhanced (what the best available fix achieves). "
              "These are the same images the thresholds were derived from, so this checks the *consistency* of the gate with the experiment, not generalisation "
              "(generalisation is the data/test table above).", "",
              "| Decision | Images | Mean raw CER | Mean best-variant CER | Share with best-variant CER > 5% |", "|---|---|---|---|---|"]
        for dcs in ("accept", "enhance", "retake"):
            sub = [r for r in rows if dec[r["id"]] == dcs]
            if sub:
                best = [min(r["cer_raw"], r["cer_enh"]) if r["cer_enh"] is not None else r["cer_raw"] for r in sub]
                L.append(f"| {dcs} | {len(sub)} | {np.mean([r['cer_raw'] for r in sub]):.4f} | {np.mean(best):.4f} | {np.mean([b > 0.05 for b in best]):.0%} |")
            else:
                L.append(f"| {dcs} | 0 | - | - | - |")
        miss = [r for r in rows if dec[r["id"]] != "retake" and (min(r["cer_raw"], r["cer_enh"]) if r["cer_enh"] is not None else r["cer_raw"]) > 0.05]
        L += ["", f"Images with best-variant CER > 5% that the gate did **not** mark retake (misses): {len(miss)}"
              + (": " + ", ".join(sorted({r['recipe'] for r in miss})) if miss else "") + ".", ""]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--per-type", type=int, default=20)
    ap.add_argument("--configs", default="", help="comma list; default = all")
    ap.add_argument("--ocr-cache", default="reports/ocr_cache")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    data_dir, out = Path(args.data), Path(args.out)
    records = EV.stratified_subset(load_labels(data_dir), args.per_type)
    names = args.configs.split(",") if args.configs else list(configs())
    if args.plan:
        plan(records, data_dir, names, out, Path(args.ocr_cache))
        return
    if not args.report_only:
        run(records, data_dir, names, out, args.ocr_cache)
    text = build_report(records, names, out, out / "quality" / "quality_vs_cer.json")
    (out / "ablation.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
