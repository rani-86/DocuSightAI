"""Link image-quality scores to OCR error, and derive the DIQA thresholds from data.

For every selected calibration image the page is OCR'd (PaddleOCR) as-is ("raw") and, for defects that have a candidate fix, after that
fix ("enhanced"): denoise (non-local means, h from the *measured* sigma) for noise, unsharp-mask deblur for blur, 3x3-median despike for
impulse noise. Page-level CER is computed against the ground-truth page text. Both variants are fed through the pipeline's own
`normalize_size`, so low-resolution pages are up-scaled exactly as in production.

Thresholds are then read off the curves (rule in `derive_thresholds`), not chosen by hand, and written to
docusight/quality_thresholds.json together with the points they were derived from.

OCR is ~7 s/image on CPU, so results are cached on disk (reports/quality/ocr_cache, keyed by image hash + variant + the source of the
functions that produced the variant); an interrupted run resumes where it stopped.

    python scripts/quality_vs_cer.py --pages 2          # ~15 min   (default)  bank_statement_v1 + invoice_v2
    python scripts/quality_vs_cer.py --pages 4          # ~30 min   adds payslip_v0 + kyc_id_v0
    python scripts/quality_vs_cer.py --limit 6          # smoke test
    python scripts/quality_vs_cer.py --report-only      # re-derive thresholds / plots from the cache, no OCR
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import docusight  # noqa: E402,F401
import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import validate_quality as VQ  # noqa: E402
from docusight import quality as Q  # noqa: E402
from docusight.layout import build_layout, reading_order_text  # noqa: E402
from docusight.metrics import cer  # noqa: E402
from docusight import preprocessing as P  # noqa: E402

CER_TARGET = 0.05  # "unacceptable" page CER
BENEFIT = 0.01  # enhancement "helps" if it lowers CER by at least one percentage point
DEFAULT_RECIPES = (["clean"] + [f"gauss_n{s}" for s in (2, 4, 6, 8, 12, 16)] + ["poisson600", "poisson150"]
                   + [f"gblur{b}" for b in (0.5, 1.0, 1.5, 2.0, 3.0)] + [f"motion{n}" for n in (3, 5, 9, 15)]
                   + ["blur2.0_noise8", "blur1.0_noise16"] + [f"lowres{s}" for s in (0.75, 0.5, 0.35, 0.25)]
                   + [f"shadow{s}" for s in (0.3, 0.5, 0.7)] + [f"glare{s}" for s in (0.5, 1.0, 2.0)]
                   + ["sp0.005", "sp0.03", "jpeg50", "jpeg15", "periodic4"] + [f"dark{g}" for g in (0.6, 0.4, 0.25)])
FAMILY = {"noise": ("gauss_n", "poisson"), "blur": ("gblur", "motion"), "combo": ("blur",), "impulse": ("sp",)}


def family(recipe: str) -> str:
    if "_noise" in recipe:
        return "combo"
    if recipe.startswith(("gauss_n", "poisson")):
        return "noise"
    if recipe.startswith(("gblur", "motion")):
        return "blur"
    if recipe.startswith("sp"):
        return "impulse"
    return "none"


# --------------------------------------------------------------------------- enhancement variants

def enhance(gray: np.ndarray, fam: str) -> np.ndarray:
    n = Q.noise_sigma(gray)
    out = gray
    if fam in ("noise", "combo") and n.measurable and n.sigma > 1.5:
        out = P.denoise(out, n.sigma)
    if fam in ("blur", "combo"):
        out = P.deblur(out)
    if fam == "impulse":
        out = Q.despike(gray, n.sigma)[1]
    return out


def _src(*fns) -> str:
    return hashlib.sha1("".join(inspect.getsource(f) for f in fns).encode()).hexdigest()[:10]


VERSION = {"raw": _src(P.normalize_size), "enh": _src(P.normalize_size, P.denoise, P.deblur, Q.despike, Q.noise_sigma, Q.flat_tiles, enhance)}


def ocr_text(engine, gray: np.ndarray) -> tuple[str, float]:
    img = P.normalize_size(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 1600)
    tokens = engine(img)
    lines = build_layout(tokens, gap_factor=None if engine.phrase_level else 1.6)
    return reading_order_text(lines), float(np.mean([t.confidence for t in tokens])) if tokens else 0.0


# --------------------------------------------------------------------------- threshold derivation

def crossing(xs, ys, level, above=True):
    """Smallest x (linear interpolation) at which the (x-sorted) curve first goes above (or below) `level`; None if it never does."""
    pts = sorted(zip(xs, ys))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if (y0 <= level < y1) if above else (y0 >= level > y1):
            return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
    return None if not pts or (pts[0][1] <= level if above else pts[0][1] >= level) else float(pts[0][0])


def crossing_down(xs, ys, level):
    """For scores where LOW is bad (text height, paper brightness): the highest x at which the curve, read from high x to low x, first exceeds `level`."""
    pts = sorted(zip(xs, ys), reverse=True)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 <= level < y1:
            return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
    return None


def derive_thresholds(rows: list[dict]) -> dict:
    """Read thresholds off the recipe-level curves. Each recipe point = mean over the pages of (score, CER).

    enhance_above  : lowest score at which enhancement lowers CER by >= BENEFIT (1 pp) - below it the fix is not worth its cost/side effects.
    retake_above   : lowest score at which the *best available* variant (enhanced if a fix exists, else raw) exceeds CER_TARGET (5%).
    Linear interpolation between neighbouring recipe points; None = never reached in the tested range.
    """
    by = {}
    for r in rows:
        by.setdefault(r["recipe"], []).append(r)
    pts = {k: {m: float(np.mean([x[m] for x in v if x.get(m) is not None])) if any(x.get(m) is not None for x in v) else None
               for m in ("cer_raw", "cer_enh", "sigma", "ew", "text_h", "glare", "glare_ratio", "unevenness", "impulse", "blockiness", "periodic", "paper")}
           for k, v in by.items()}
    out: dict = {"cer_target": CER_TARGET, "benefit_pp": BENEFIT, "evidence": {}}

    def curve(recipes, score, with_enh):
        xs, raw, enh = [], [], []
        for k in recipes:
            if k in pts and pts[k][score] is not None:
                xs.append(pts[k][score])
                raw.append(pts[k]["cer_raw"])
                enh.append(pts[k]["cer_enh"] if (with_enh and pts[k]["cer_enh"] is not None) else pts[k]["cer_raw"])
        return xs, raw, enh

    def setting(name, recipes, score, with_enh):
        xs, raw, enh = curve(recipes, score, with_enh)
        gain = [a - b for a, b in zip(raw, enh)]
        helps_from = crossing(xs, gain, BENEFIT, above=True) if with_enh else None
        stops = None
        if helps_from is not None:  # first score beyond the benefit region where the fix no longer lowers CER by BENEFIT
            tail = [(x, g) for x, g in zip(xs, gain) if x > helps_from]
            stops = crossing([x for x, _ in tail], [g for _, g in tail], BENEFIT, above=False) if tail else None
        out[name] = {"score": score, "enhance_above": helps_from, "enhance_stops_helping_above": stops,
                     "retake_above": crossing(xs, enh, CER_TARGET, above=True), "retake_raw_above": crossing(xs, raw, CER_TARGET, above=True)}
        out["evidence"][name] = [{"recipe": k, score: round(pts[k][score], 4), "cer_raw": round(pts[k]["cer_raw"], 4),
                                  "cer_enh": None if pts[k]["cer_enh"] is None else round(pts[k]["cer_enh"], 4)}
                                 for k in recipes if k in pts and pts[k][score] is not None]

    setting("noise_sigma", ["clean"] + [f"gauss_n{s}" for s in (2, 4, 6, 8, 12, 16)] + ["poisson600", "poisson150"], "sigma", True)
    # Defocus (Gaussian) blur only: the isotropic edge-width estimator under-reads motion blur (see "motion_blur_not_covered"), so mixing the
    # two families on one curve makes it non-monotone and the thresholds meaningless.
    setting("blur_edge_width", ["clean"] + [f"gblur{b}" for b in (0.5, 1.0, 1.5, 2.0, 3.0)], "ew", True)
    out["evidence"]["motion_blur_not_covered"] = [{"recipe": k, "ew": round(pts[k]["ew"], 4), "cer_raw": round(pts[k]["cer_raw"], 4),
                                                   "cer_enh": None if pts[k]["cer_enh"] is None else round(pts[k]["cer_enh"], 4)}
                                                  for k in (f"motion{n}" for n in (3, 5, 9, 15)) if k in pts]
    setting("text_height_px", ["clean"] + [f"lowres{s}" for s in (0.75, 0.5, 0.35, 0.25)], "text_h", False)
    setting("glare_fraction", ["clean", "glare0.5", "glare1.0", "glare2.0"], "glare", False)
    gx, graw, _ = curve(["glare0.5", "glare1.0", "glare2.0"], "glare_ratio", False)
    out["evidence"]["glare_text_loss"] = [{"recipe": k, "glare_ratio": round(pts[k]["glare_ratio"], 3), "cer_raw": round(pts[k]["cer_raw"], 4), "cer_enh": None}
                                          for k in ("glare0.5", "glare1.0", "glare2.0") if k in pts and pts[k]["glare_ratio"] is not None]
    out["glare_text_loss"] = {"score": "glare_ratio", "retake_below": crossing_down(gx, graw, CER_TARGET)}
    setting("illumination_unevenness", ["clean", "shadow0.3", "shadow0.5", "shadow0.7"], "unevenness", False)
    setting("impulse_fraction", ["clean", "sp0.005", "sp0.03"], "impulse", True)
    # text height and paper brightness are *minimums*: CER rises as they fall
    for name, recipes, score in (("text_height_px", ["clean"] + [f"lowres{s}" for s in (0.75, 0.5, 0.35, 0.25)], "text_h"),
                                 ("darkness", ["clean"] + [f"dark{g}" for g in (0.6, 0.4, 0.25)], "paper")):
        xs, raw, _ = curve(recipes, score, False)
        out["evidence"][name] = [{"recipe": k, score: round(pts[k][score], 3), "cer_raw": round(pts[k]["cer_raw"], 4), "cer_enh": None}
                                 for k in recipes if k in pts and pts[k][score] is not None]
        out[name] = {"score": score, "retake_below": crossing_down(xs, raw, CER_TARGET), "cer_at_lowest_tested": raw[int(np.argmin(xs))] if xs else None}
    out["clean_cer"] = pts["clean"]["cer_raw"] if "clean" in pts else None
    return out


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="data/quality_calib")
    ap.add_argument("--pages", type=int, default=2, help="number of non-dense base pages (order: bank_v1, invoice_v2, payslip_v0, kyc_id_v0)")
    ap.add_argument("--recipes", default="", help="comma list overriding the default recipe subset")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (smoke test)")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--thresholds", default="docusight/quality_thresholds.json")
    args = ap.parse_args()
    root, out = Path(args.calib), Path(args.out)
    cache_dir = out / "quality" / "ocr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    feats = {r["id"]: r for r in VQ.load_features(root, out / "quality" / "features_eval.json", 4)}
    pages = json.load(open(root / "pages.json"))
    page_ids = [p for p, v in pages.items() if not v["dense"]][: args.pages]
    recipes = args.recipes.split(",") if args.recipes else DEFAULT_RECIPES
    todo = [f for f in feats.values() if f["page_id"] in page_ids and f["recipe"] in recipes]
    todo.sort(key=lambda f: (f["page_id"], recipes.index(f["recipe"])))
    if args.limit:
        todo = todo[:: max(1, len(todo) // args.limit)][: args.limit]
    print(f"{len(todo)} images from pages {page_ids}")

    engine = None
    rows, t0, n_ocr = [], time.time(), 0
    for i, f in enumerate(todo, 1):
        gray = cv2.imread(str(root / f["image"]), cv2.IMREAD_GRAYSCALE)
        h = hashlib.sha1(gray.tobytes()).hexdigest()[:14]
        fam = family(f["recipe"])
        variants = {"raw": gray}
        if fam != "none":
            variants["enh"] = None  # built lazily only if its OCR is not cached
        res = {}
        for name in variants:
            cp = cache_dir / f"{h}_{name}_{VERSION[name]}.json"
            if cp.exists():
                res[name] = json.loads(cp.read_text())
            elif not args.report_only:
                engine = engine or __import__("docusight.ocr", fromlist=["get_engine"]).get_engine("paddle")
                img = gray if name == "raw" else enhance(gray, fam)
                text, conf = ocr_text(engine, img)
                res[name] = {"text": text, "conf": conf}
                cp.write_text(json.dumps(res[name]))
                n_ocr += 1
        if "raw" not in res:
            continue
        gt = pages[f["page_id"]]["text"]
        rows.append({"id": f["id"], "recipe": f["recipe"], "page": f["page_id"], "family": fam, "true": f["true"],
                     "cer_raw": cer(gt, res["raw"]["text"]), "cer_enh": cer(gt, res["enh"]["text"]) if "enh" in res else None,
                     **{k: f[k] for k in ("sigma", "ew", "text_h", "glare", "glare_ratio", "unevenness", "impulse", "blockiness", "periodic", "measurable", "paper")}})
        if rows[-1]["glare_ratio"] is not None and rows[-1]["glare_ratio"] < 0:
            rows[-1]["glare_ratio"] = None  # feature cache uses -1 for "undefined"
        if n_ocr and (i % 5 == 0 or i == len(todo)):
            print(f"[{i}/{len(todo)}] {f['id']} cer_raw={rows[-1]['cer_raw']:.3f} elapsed {time.time() - t0:.0f}s ({n_ocr} OCR calls)", flush=True)

    (out / "quality" / "quality_vs_cer.json").write_text(json.dumps(rows, indent=1))
    th = derive_thresholds(rows)
    th["derived_from"] = {"script": "scripts/quality_vs_cer.py", "calibration_set": str(root), "pages": page_ids, "n_images": len(rows),
                          "recipes": sorted({r["recipe"] for r in rows}), "ocr_engine": "paddleocr", "date": date.today().isoformat()}
    hint = out / "quality" / "hint_thresholds.json"
    if hint.exists():
        th["hints"] = json.load(open(hint))
    Path(args.thresholds).write_text(json.dumps(th, indent=1))
    write_report(rows, th, out)
    print(json.dumps({k: v for k, v in th.items() if k not in ("evidence",)}, indent=1))


def write_report(rows, th, out: Path) -> None:
    fams = [("noise_sigma", "sigma", ["clean"] + [f"gauss_n{s}" for s in (2, 4, 6, 8, 12, 16)] + ["poisson600", "poisson150"], "estimated noise sigma (gray levels)"),
            ("blur_edge_width", "ew", ["clean"] + [f"gblur{b}" for b in (0.5, 1.0, 1.5, 2.0, 3.0)], "edge width (px), defocus blur"),
            ("text_height_px", "text_h", ["clean"] + [f"lowres{s}" for s in (0.75, 0.5, 0.35, 0.25)], "native text height (px)"),
            ("glare_fraction", "glare", ["clean", "glare0.5", "glare1.0", "glare2.0"], "glare fraction"),
            ("illumination_unevenness", "unevenness", ["clean", "shadow0.3", "shadow0.5", "shadow0.7"], "illumination unevenness"),
            ("impulse_fraction", "impulse", ["clean", "sp0.005", "sp0.03"], "impulse fraction"),
            ("darkness", "paper", ["clean", "dark0.6", "dark0.4", "dark0.25"], "paper brightness (gray level)")]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    for ax, (key, score, recipes, label) in zip(axes.ravel(), fams):  # last panel stays empty
        ev = th["evidence"].get(key, [])
        if not ev:
            continue
        xs = [e[score] for e in ev]
        ax.plot(xs, [e["cer_raw"] * 100 for e in ev], "o-", label="raw", color="tab:red")
        enh = [(e[score], e["cer_enh"] * 100) for e in ev if e["cer_enh"] is not None]
        if enh:
            ax.plot(*zip(*enh), "s--", label="enhanced", color="tab:blue")
        ax.axhline(CER_TARGET * 100, color="k", ls=":", lw=1)
        for k, c in (("enhance_above", "tab:green"), ("retake_above", "tab:purple"), ("retake_below", "tab:purple")):
            v = th.get(key, {}).get(k)
            if v is not None:
                ax.axvline(v, color=c, ls="--", lw=1, label=f"{k}={v:.3g}")
        ax.set(xlabel=label, ylabel="page CER (%)", title=key)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "quality" / "cer_vs_quality.png", dpi=130)
    plt.close(fig)

    L = ["# Image quality vs OCR error, and the thresholds derived from it", "",
         f"_Generated by `scripts/quality_vs_cer.py`. {th['derived_from']['n_images']} calibration images from {len(th['derived_from']['pages'])} base pages "
         f"({', '.join(th['derived_from']['pages'])}), PaddleOCR on CPU. Page-level CER against the ground-truth page text. "
         f"Clean-page CER = {th['clean_cer']:.4f}. 'Unacceptable' = CER > {CER_TARGET:.0%}; enhancement 'helps' if it lowers CER by >= {BENEFIT * 100:.0f} percentage point._", "",
         "Thresholds are read off the curves below (linear interpolation between recipe means); they are stored with their evidence in `docusight/quality_thresholds.json`. "
         "With few pages per point they are coarse: treat each as accurate to roughly one recipe step.", "",
         "![CER vs quality](quality/cer_vs_quality.png)", ""]
    for key, score, recipes, label in fams:
        ev = th["evidence"].get(key, [])
        t = th.get(key, {})
        L += [f"## {key}", "", f"Thresholds: " + ", ".join(f"`{k}` = {v:.4g}" if isinstance(v, float) else f"`{k}` = {v}" for k, v in t.items() if k != "score"), "",
              VQ.md(["recipe", label, "CER raw", "CER enhanced"], [[e["recipe"], f"{e[score]:.3f}", f"{e['cer_raw']:.4f}", "-" if e["cer_enh"] is None else f"{e['cer_enh']:.4f}"] for e in ev]), ""]
    by = {}
    for r in rows:
        by.setdefault(r["recipe"], []).append(r)
    mo = th["evidence"].get("motion_blur_not_covered", [])
    if mo:
        L += ["## Motion blur: a documented miss of the edge-width estimator", "",
              "Edge width is built from the gradient magnitude, which motion blur lowers only across the motion direction, so it under-reads: "
              "a 9 px motion blur (CER ~40%) scores an edge width of ~0.9 px, i.e. looks nearly sharp. The blur gate does **not** catch these pages.", "",
              VQ.md(["recipe", "edge width (px)", "CER raw", "CER enhanced"], [[e["recipe"], f"{e['ew']:.3f}", f"{e['cer_raw']:.4f}", "-" if e["cer_enh"] is None else f"{e['cer_enh']:.4f}"] for e in mo]), ""]
    L += ["## Other defects (raw OCR only)", "", VQ.md(["recipe", "mean CER", "n"], [[k, f"{np.mean([x['cer_raw'] for x in v]):.4f}", len(v)]
          for k, v in by.items() if family(k) == "none" and not k.startswith(("lowres", "glare", "shadow", "clean"))]), ""]
    (out / "quality_vs_cer.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
