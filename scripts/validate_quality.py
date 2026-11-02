"""Validate the image-quality estimators against known ground truth (no OCR).

Inputs : data/quality_calib (evaluation, 364 images) and data/quality_calib_tune (tuning, used ONLY to choose the
         noise-type hint thresholds so the confusion table is not scored on data the thresholds were fitted to).
Outputs: reports/quality/*.png, reports/quality/hint_thresholds.json, reports/quality_validation.md
Cache  : reports/quality/features_<set>.json (keyed by image sha1 + quality.py sha1) - reruns are instant.

    python scripts/validate_quality.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import docusight  # noqa: E402,F401
import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIGMA_VISIBLE = 1.5  # gray levels: "noise is present" cut-off for the gaussian class (a design choice, not fitted; was 2.0 in v1,
#                      which sat exactly on the smallest injected level so sigma=2 images were split by estimator rounding)
DENOISE_ABOVE = 2.0  # validation mimics the pipeline: blur is measured on a denoised image when noise is present


# --------------------------------------------------------------------------- feature extraction

def extract(path: str) -> dict:
    from docusight import quality as Q
    from docusight.preprocessing import blur_score, contrast_score, denoise, noise_score

    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    flat = Q.flat_tiles(gray)
    n = Q.noise_sigma(gray, flat)
    h = Q.noise_hints(gray, flat)
    den = denoise(gray, n.sigma) if n.sigma > DENOISE_ABOVE else gray
    b_raw, b = Q.blur_metrics(gray), Q.blur_metrics(den)
    e, t = Q.exposure_metrics(gray), Q.text_height(den)
    return {
        "legacy_noise": noise_score(gray), "legacy_blur": blur_score(gray), "legacy_contrast": contrast_score(gray),
        "sigma": n.sigma, "sigma_imm": n.sigma_immerkaer, "flat_fraction": n.flat_fraction, "measurable": n.measurable,
        "impulse": h.impulse_fraction, "blockiness": h.jpeg_blockiness, "periodic": h.periodic_peak if h.periodic_peak is not None else 0.0,
        "raw_ew": b_raw.edge_width, "raw_tn": b_raw.tenengrad_norm, "raw_lap": b_raw.lap_var_edge,
        "ew": b.edge_width, "tn": b.tenengrad_norm, "lap": b.lap_var_edge, "edge_density": b.edge_density,
        "paper": e.paper_level, "unevenness": e.unevenness, "shadow_frac": e.shadow_fraction, "glare": e.glare_fraction,
        "glare_ratio": e.glare_edge_ratio if e.glare_edge_ratio is not None else -1.0, "clip_hi": e.clipped_high,
        "text_h": t.text_height_px, "text_h_p25": t.p25,
    }


def estimator_version() -> str:
    """Hash of the source of exactly the functions `extract` calls, so unrelated edits to quality.py / preprocessing.py keep the cache valid."""
    import inspect

    from docusight import preprocessing as P
    from docusight import quality as Q

    fns = [Q._tile_view, Q._immerkaer_abs, Q._tile_sigma, Q.despike, Q.flat_tiles, Q.noise_sigma, Q.impulse_fraction, Q.jpeg_blockiness,
           Q.periodic_peak, Q.noise_hints, Q.blur_metrics, Q.background_map, Q.exposure_metrics, Q.text_height, P.denoise, P.blur_score,
           P.noise_score, P.contrast_score, extract]
    return hashlib.sha1("".join(inspect.getsource(f) for f in fns).encode()).hexdigest()[:12]


def load_features(root: Path, cache: Path, workers: int) -> list[dict]:
    labels = [json.loads(line) for line in open(root / "labels.jsonl", encoding="utf-8")]
    pages = json.load(open(root / "pages.json"))
    ver = estimator_version()
    store = json.load(open(cache)) if cache.exists() else {}
    todo, keys = [], {}
    for r in labels:
        p = root / r["image"]
        keys[r["id"]] = hashlib.sha1(p.read_bytes()).hexdigest()[:12] + ver
        if store.get(r["id"], {}).get("key") != keys[r["id"]]:
            todo.append(r)
    if todo:
        print(f"[{root.name}] computing features for {len(todo)}/{len(labels)} images ...", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r, f in zip(todo, ex.map(extract, [str(root / r["image"]) for r in todo], chunksize=4)):
                store[r["id"]] = {"key": keys[r["id"]], "f": f}
        cache.parent.mkdir(parents=True, exist_ok=True)
        json.dump(store, open(cache, "w"))
    return [{**r, "page": pages[r["page_id"]], **store[r["id"]]["f"]} for r in labels]


# --------------------------------------------------------------------------- helpers

def md(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def true_sigma(r) -> float:
    return float(r["true"].get("noise_sigma", 0.0))


def stats(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope = float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 0 else float("nan")
    return {"pearson": float(pearsonr(x, y)[0]), "spearman": float(spearmanr(x, y)[0]), "mae": float(np.abs(x - y).mean()),
            "bias": float((y - x).mean()), "slope": slope, "n": len(x)}


def fmt_stats(name, s):
    return [name, s["n"], f"{s['pearson']:.3f}", f"{s['spearman']:.3f}", f"{s['mae']:.2f}", f"{s['bias']:+.2f}", f"{s['slope']:.2f}"]


def fit_threshold(scores, is_pos) -> float:
    """Threshold maximising F1 of `score > t` for the positive class."""
    s, y = np.asarray(scores, float), np.asarray(is_pos, bool)
    cand = np.unique(s)
    cand = (cand[:-1] + cand[1:]) / 2 if len(cand) > 1 else cand
    best, bt = -1.0, float(np.median(s))
    for t in cand:
        p = s > t
        tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best or (f1 == best and abs(t) < abs(bt)):
            best, bt = f1, float(t)
    return bt


def classify(f, th) -> str:
    if f["periodic"] > th["periodic_peak"]:
        return "periodic"
    if f["impulse"] > th["impulse_fraction"]:
        return "salt_pepper"
    if f["blockiness"] > th["jpeg_blockiness"]:
        return "jpeg"
    return "gaussian" if f["sigma"] > SIGMA_VISIBLE else "none"


CLASSES = ["none", "gaussian", "salt_pepper", "jpeg", "periodic"]


# --------------------------------------------------------------------------- main analysis

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/quality_calib")
    ap.add_argument("--tune", default="data/quality_calib_tune")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "quality").mkdir(parents=True, exist_ok=True)
    ev = load_features(Path(args.eval), out / "quality" / "features_eval.json", args.workers)
    tu = load_features(Path(args.tune), out / "quality" / "features_tune.json", args.workers)
    L: list[str] = ["# Image-quality estimators: validation against ground truth", "",
                    f"_Generated by `scripts/validate_quality.py`. Evaluation set: {len(ev)} images = "
                    f"{len({r['page_id'] for r in ev})} base pages x {len({r['recipe'] for r in ev})} recipes with known parameters "
                    f"(`data/quality_calib`; 3 of the pages are text-dense). Hint thresholds were fitted on a separate tuning set "
                    f"({len(tu)} images, different seed/pages) and are only *reported* on the evaluation set._", ""]

    L += ["## Revision history (read this first)", "",
          "The estimators were run on the evaluation set once in their first form (kept verbatim in "
          "[`quality_validation_v1_initial.md`](quality_validation_v1_initial.md)). That run exposed real defects, which were debugged "
          "using the separate **tuning set only**, and the evaluation numbers below are from the revised estimators. Because the evaluation set "
          "had already been looked at, treat it as a development-grade check, not a blind test; the blind test is the OCR/CER experiment on "
          "`data/test` (Stage 6). What the first run showed and what changed:", "",
          "| First run (v1) on the evaluation set | Cause | Change |", "|---|---|---|",
          "| Noise-type hints 81.0% accurate; 18 clean/shadow/glare/low-res pages flagged salt-and-pepper | flat tiles were chosen on a median-filtered image, which also erases 1-px strokes and table rules, so those tiles looked like paper with 'impulses' | flatness now ignores isolated impulses only (isolated-pixel rule) instead of median-filtering; per-tile adaptive thresholds |",
          "| JPEG recall 0% (24/28 missed) | blockiness was measured on flat paper only; a perfectly flat region has no blocking even after JPEG; ruled pages gave 'blocky' scores without any JPEG | measured on content with long rules masked, second-difference form; class redefined to q <= 30 with q 50-75 reported as a grey zone |",
          "| Periodic peak threshold fitted to 5e12 | median spectral power ~0 on noise-free paper makes max/median explode | added the 8-bit quantisation-noise floor and plane detrending |",
          "| Dense low-resolution pages: mean sigma 4.29, 42% false alarms | blurred dense text passed the gradient test and was treated as paper | block-mean structure test; such pages now return *not measurable* |",
          "| `sigma = 2` images split between 'gaussian' and 'none' | visibility cut-off equal to the smallest injected level | cut-off 1.5 (a design choice, not fitted) |",
          "| Report text claimed glare 0.5 is not saturated | wrong - measured glare_fraction was 0.085 | corrected |", ""]
    normal = [r for r in ev if not r["page"]["dense"]]
    dense = [r for r in ev if r["page"]["dense"]]
    gaussian_family = lambda r: r["noise_kind"] == "gaussian" or r["recipe"] == "clean" or r["recipe"].startswith("gblur")  # noqa: E731

    # ============ A. noise sigma
    L += ["## A. Noise sigma estimation", "",
          "Ground truth: the injected Gaussian sigma (Poisson recipes: the equivalent std on paper, sqrt(255*L/peak)). "
          "Subset = images whose only noise is additive Gaussian/Poisson (plus the clean and blur-only images, true sigma 0). "
          "Columns: Pearson / Spearman correlation of estimate vs truth, mean absolute error (gray levels), mean bias, and the OLS slope (1.0 = unbiased).", ""]
    rows, fig, axes = [], *plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (title, sub) in zip(axes, [("all pages", [r for r in ev if gaussian_family(r)]),
                                       ("text-dense pages only", [r for r in dense if gaussian_family(r)]),
                                       ("normal pages only", [r for r in normal if gaussian_family(r)])]):
        x = [true_sigma(r) for r in sub]
        for key, label, col in (("legacy_noise", "legacy (whole-page MAD)", "tab:red"), ("sigma", "new: tile MAD", "tab:blue"),
                                ("sigma_imm", "new: tile Immerkaer", "tab:green")):
            y = [r[key] for r in sub]
            ax.scatter(x, y, s=10, alpha=0.45, label=label, color=col)
            rows.append(fmt_stats(f"{label} - {title}", stats(x, y)))
        ax.plot([0, 20], [0, 20], "k--", lw=1)
        ax.set(title=title, xlabel="true sigma (gray levels)")
    axes[0].set_ylabel("estimated sigma")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "quality" / "noise_est_vs_true.png", dpi=130)
    plt.close(fig)
    L += [md(["Estimator - subset", "n", "Pearson", "Spearman", "MAE", "bias", "slope"], rows), "",
          "![noise estimate vs truth](quality/noise_est_vs_true.png)", ""]

    # failure modes on images with NO additive Gaussian noise
    L += ["### Failure modes: images whose true Gaussian sigma is 0", "",
          "Mean estimated sigma (and false-alarm rate = share of images with estimate > 2) per defect class. Ideal = 0.", ""]
    groups = {"clean": lambda r: r["recipe"] == "clean", "gaussian blur": lambda r: r["recipe"].startswith("gblur"),
              "motion blur": lambda r: r["recipe"].startswith("motion"), "salt & pepper": lambda r: r["recipe"].startswith("sp"),
              "JPEG q<=50": lambda r: r["recipe"] in ("jpeg50", "jpeg30", "jpeg15"), "JPEG q>=75": lambda r: r["recipe"] in ("jpeg95", "jpeg75"),
              "shadow": lambda r: r["recipe"].startswith("shadow"), "low resolution": lambda r: r["recipe"].startswith("lowres"),
              "periodic": lambda r: r["recipe"].startswith("periodic")}
    rows = []
    for g, fn in groups.items():
        for pname, pg in (("normal", normal), ("dense", dense)):
            sub = [r for r in pg if fn(r)]
            if sub:
                rows.append([g, pname, len(sub), f"{np.mean([r['legacy_noise'] for r in sub]):.2f}", f"{np.mean([r['sigma'] for r in sub]):.2f}",
                             f"{np.mean([r['legacy_noise'] > 2 for r in sub]):.0%}", f"{np.mean([r['sigma'] > 2 for r in sub]):.0%}"])
    L += [md(["Defect", "Pages", "n", "legacy mean est.", "new mean est.", "legacy false alarm", "new false alarm"], rows), ""]
    sp = [r for r in ev if r["recipe"].startswith("sp")]
    rows = [[f"{d}", f"{np.mean([r['legacy_noise'] for r in sp if r['true']['sp_density'] == d]):.2f}",
             f"{np.mean([r['sigma'] for r in sp if r['true']['sp_density'] == d]):.2f}",
             f"{np.mean([r['impulse'] for r in sp if r['true']['sp_density'] == d]):.4f}"] for d in sorted({r['true']['sp_density'] for r in sp})]
    L += ["Salt-and-pepper (true Gaussian sigma = 0): density vs estimates - the impulse fraction should track density.", "",
          md(["density", "legacy sigma", "new sigma", "impulse fraction"], rows), ""]
    dsub = [r for r in dense if gaussian_family(r)]
    sub = [r for r in normal if gaussian_family(r)]
    L += [f"Dense text pages: flat-tile fraction used = {np.mean([r['flat_fraction'] for r in dsub]):.2f} (normal pages {np.mean([r['flat_fraction'] for r in sub]):.2f}); "
          f"images reported as *not measurable* (too few paper tiles): {np.mean([not r['measurable'] for r in dense]):.0%} of all dense images ({sum(not r['measurable'] for r in dense)}/{len(dense)}: "
          f"{', '.join(sorted({r['recipe'] for r in dense if not r['measurable']}))}) and {np.mean([not r['measurable'] for r in normal]):.0%} of normal ones.", ""]

    # ============ B. blur
    L += ["## B. Blur", "",
          "Ground truth: injected Gaussian blur sigma. Direction: *higher legacy Laplacian variance = sharper*, *higher edge_width = blurrier*; "
          "Spearman is signed so that +1 always means 'perfectly monotone in the right direction'. "
          "New scores are computed on the image **after denoising** (NLM, as the pipeline does whenever sigma > 2).", ""]
    blur_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    def blur_series(page, noise_sigma=0.0):
        seq = []
        for b in blur_levels:
            if noise_sigma == 0:
                rec = "clean" if b == 0 else f"gblur{b}"
            else:
                rec = f"gauss_n{int(noise_sigma)}" if b == 0 else f"blur{b}_noise{int(noise_sigma)}"
            m = [r for r in ev if r["page_id"] == page and r["recipe"] == rec]
            seq.append(m[0] if m else None)
        return seq

    # (key, True if the score INCREASES with blur)
    scores = {"legacy Laplacian variance (raw)": ("legacy_blur", False), "edge_width (denoised)": ("ew", True),
              "tenengrad_norm (denoised)": ("tn", False), "lap_var_edge (denoised)": ("lap", False),
              "edge_width (NOT denoised)": ("raw_ew", True)}
    rows = []
    pages_ids = sorted({r["page_id"] for r in ev})
    for name, (key, up) in scores.items():
        rho, mono = [], 0
        for pg in pages_ids:
            v = np.array([s_[key] for s_ in blur_series(pg)])
            rho.append(spearmanr(blur_levels, v)[0] * (1 if up else -1))
            d = np.diff(v) * (1 if up else -1)
            mono += bool(np.all(d > 0))
        rows.append([name, f"{np.mean(rho):.3f}", f"{mono}/{len(pages_ids)}"])
    L += ["No added noise; 6 blur levels (0, 0.5, 1, 1.5, 2, 3) per base page (a tie between blur 0 and 0.5 counts as not strictly monotone; edge_width has an intrinsic floor of ~0.8 px from anti-aliasing):", "", md(["Score", "mean Spearman (right direction = +1)", "pages strictly monotone over all 6 levels"], rows), ""]

    # content dependence: coefficient of variation across pages at fixed true blur
    rows = []
    for name, (key, _) in scores.items():
        cv = []
        for b in blur_levels:
            rec = "clean" if b == 0 else f"gblur{b}"
            v = np.array([r[key] for r in ev if r["recipe"] == rec])
            cv.append(np.std(v) / np.mean(v))
        rows.append([name] + [f"{c:.2f}" for c in cv])
    L += ["**Content dependence.** Coefficient of variation (std/mean) of each score *across the 7 different pages* at the same true blur. "
          "A good sharpness measure should vary with blur, not with what is printed on the page:", "",
          md(["Score"] + [f"blur {b}" for b in blur_levels], rows), ""]

    # noise inflation
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    noise_levels = [0, 4, 8, 16]
    infl_rows = []
    for ax, (name, key) in zip(axes, [("legacy Laplacian variance (raw)", "legacy_blur"), ("edge_width, denoised (new)", "ew"),
                                      ("edge_width, NOT denoised", "raw_ew")]):
        for ns in noise_levels:
            ys = []
            for b in blur_levels:
                vals = []
                for pg in pages_ids:
                    rr = blur_series(pg, ns)[blur_levels.index(b)]
                    if rr:
                        vals.append(rr[key])
                ys.append(np.mean(vals) if vals else np.nan)
            ax.plot(blur_levels, ys, marker="o", label=f"noise sigma {ns}")
        ax.set(title=name, xlabel="true blur sigma", ylabel=name)
        if key == "legacy_blur":
            ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "quality" / "blur_vs_true_with_noise.png", dpi=130)
    plt.close(fig)
    for b in (1.0, 2.0, 3.0):
        row = [f"blur {b}"]
        for key in ("legacy_blur", "ew", "raw_ew", "tn"):
            ratios = []
            for pg in pages_ids:
                clean_r = blur_series(pg, 0)[blur_levels.index(b)]
                noisy_r = blur_series(pg, 16)[blur_levels.index(b)]
                if clean_r and noisy_r and clean_r[key] > 0:
                    ratios.append(noisy_r[key] / clean_r[key])
            row.append(f"{np.mean(ratios):.2f}x")
        infl_rows.append(row)
    L += ["**Noise inflation.** Score at noise sigma 16 divided by the same page & blur with no noise (1.00x = noise does not move the score; "
          "for the legacy sharpness score a value >1 means noise makes a blurry page look *sharper*):", "",
          md(["True blur", "legacy Laplacian var (raw)", "edge_width (denoised)", "edge_width (NOT denoised)", "tenengrad_norm (denoised)"], infl_rows), "",
          "![blur vs truth under noise](quality/blur_vs_true_with_noise.png)", ""]
    mo = [r for r in ev if r["recipe"].startswith("motion")] + [r for r in ev if r["recipe"] == "clean"]
    rho = spearmanr([r["true"].get("motion_len", 1) for r in mo], [r["ew"] for r in mo])[0]
    L += [f"Motion blur (length 3/5/9/15 px at a random angle, plus clean): Spearman(length, edge_width) = {rho:.3f}. "
          "Edge width is isotropic (it uses the gradient magnitude), so it detects motion blur but cannot tell its direction.", ""]

    # ============ C. noise-type hints
    L += ["## C. Noise-type hints", "",
          f"Rule: periodic peak -> *periodic*; else impulse fraction -> *salt_pepper*; else JPEG blockiness -> *jpeg*; else sigma > {SIGMA_VISIBLE} -> *gaussian*; else *none*. "
          "Thresholds below were fitted (max one-vs-rest F1) on the **tuning set** and are reported on the evaluation set.", ""]
    th = {}
    for key, cls, name in (("periodic", "periodic", "periodic_peak"), ("impulse", "salt_pepper", "impulse_fraction"), ("blockiness", "jpeg", "jpeg_blockiness")):
        pool = [r for r in tu if r["noise_kind"] != "grey"]  # q 50-75 JPEG is a reported grey zone, not used for fitting
        th[name] = fit_threshold([r[key] for r in pool], [r["noise_kind"] == cls for r in pool])
    json.dump({**th, "sigma_visible": SIGMA_VISIBLE, "fitted_on": f"{Path(args.tune).name} ({len(tu)} images)"}, open(out / "quality" / "hint_thresholds.json", "w"), indent=1)
    L += [md(["Hint", "fitted threshold"], [[k, f"{v:.4g}"] for k, v in th.items()]), ""]
    for set_name, data_all in (("Evaluation set", ev), ("(for reference) tuning set - thresholds were fitted here", tu)):
        data = [r for r in data_all if r["noise_kind"] != "grey"]
        cm = np.zeros((5, 5), int)
        for r in data:
            cm[CLASSES.index(r["noise_kind"]), CLASSES.index(classify(r, th))] += 1
        L += [f"**{set_name}** - rows: true class, columns: predicted:", "", md(["true \\ pred"] + CLASSES + ["recall"],
              [[c] + list(cm[i]) + [f"{cm[i, i] / max(1, cm[i].sum()):.0%}"] for i, c in enumerate(CLASSES)]), "",
              f"Overall accuracy {np.trace(cm) / cm.sum():.1%} (n={cm.sum()}).", ""]
        if data_all is ev:
            miss = defaultdict(list)
            for r in data:
                p = classify(r, th)
                if p != r["noise_kind"]:
                    miss[(r["noise_kind"], p)].append(r["recipe"] + ("*" if r["page"]["dense"] else ""))
            L += ["Misclassified (true -> predicted: recipes; `*` = dense page):", ""]
            L += [f"* {t} -> {p}: " + ", ".join(sorted(set(v), key=v.index)[:14]) + (" ..." if len(set(v)) > 14 else "") + f" ({len(v)} images)" for (t, p), v in sorted(miss.items())] or ["* none"]
            L += [""]
    L += ["**JPEG grey zone and blockiness by quality** (mean over pages; threshold above; q 50-75 are excluded from the confusion table "
          "because the grid evidence is smaller than page-to-page variation):", "",
          md(["JPEG quality", "mean blockiness (normal pages)", "mean blockiness (dense pages)", "detected as jpeg (all pages)"],
             [[q, f"{np.mean([r['blockiness'] for r in normal if r['true'].get('jpeg_quality') == q]):.3f}",
               f"{np.mean([r['blockiness'] for r in dense if r['true'].get('jpeg_quality') == q]):.3f}",
               f"{np.mean([r['blockiness'] > th['jpeg_blockiness'] for r in ev if r['true'].get('jpeg_quality') == q]):.0%}"] for q in (95, 75, 50, 30, 15)]),
          f"Clean pages: mean blockiness {np.mean([r['blockiness'] for r in ev if r['recipe'] == 'clean']):.3f} "
          f"(min {min(r['blockiness'] for r in ev if r['recipe'] == 'clean'):.3f}, max {max(r['blockiness'] for r in ev if r['recipe'] == 'clean'):.3f}).", ""]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, (key, ttl) in zip(ax, (("impulse", "impulse fraction"), ("blockiness", "JPEG blockiness"), ("periodic", "periodic peak"))):
        for cls in CLASSES:
            v = [r[key] for r in ev if r["noise_kind"] == cls]
            v = [max(x, 1e-4) for x in v]
            a.scatter([CLASSES.index(cls)] * len(v) + np.random.default_rng(0).normal(0, 0.06, len(v)), v, s=8, alpha=0.5)
        a.axhline(th[{"impulse": "impulse_fraction", "blockiness": "jpeg_blockiness", "periodic": "periodic_peak"}[key]], color="k", ls="--", lw=1)
        a.set_xticks(range(5), CLASSES, rotation=20)
        a.set_title(ttl)
        a.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out / "quality" / "noise_hints.png", dpi=130)
    plt.close(fig)
    L += ["![noise hints](quality/noise_hints.png)", ""]

    # ============ D. exposure + resolution
    L += ["## D. Exposure and resolution estimators", ""]
    sh = [r for r in normal if r["recipe"] == "clean" or r["recipe"].startswith("shadow")]
    s_true = [r["true"].get("shadow_strength", 0.0) for r in sh]
    L += [f"* **Illumination unevenness** vs true shadow strength (0, 0.15, 0.3, 0.5, 0.7; {len(sh)} normal pages): Spearman {spearmanr(s_true, [r['unevenness'] for r in sh])[0]:.3f}; "
          f"mean unevenness by strength: " + ", ".join(f"{s}: {np.mean([r['unevenness'] for r, t in zip(sh, s_true) if t == s]):.2f}" for s in sorted(set(s_true))) + ".",
          f"  Expected for a linear ramp 1 -> (1-s): 1/(1-s) = " + ", ".join(f"{s}: {1 / (1 - s):.2f}" for s in sorted(set(s_true))) + " (the estimator's percentile range shrinks this).",
          f"* On the *dense* pages unevenness for strength 0.7 = {np.mean([r['unevenness'] for r in dense if r['recipe'] == 'shadow0.7']):.2f} "
          f"vs {np.mean([r['unevenness'] for r in normal if r['recipe'] == 'shadow0.7']):.2f} on normal pages; clean dense = {np.mean([r['unevenness'] for r in dense if r['recipe'] == 'clean']):.2f} "
          "(text covers the paper so the background map is less reliable)."]
    gl = [r for r in ev if r["recipe"] == "clean" or r["recipe"].startswith("glare")]
    L += ["* **Glare** (a Gaussian hot spot, strength 0.5/1.0/2.0): " + "; ".join(
        f"strength {s}: glare_fraction {np.mean([r['glare'] for r in gl if r['true'].get('glare_strength', 0) == s]):.4f}, "
        f"text-edge ratio inside the blob vs a ring around it {np.mean([r['glare_ratio'] for r in gl if r['true'].get('glare_strength', 0) == s and r['glare_ratio'] >= 0] or [float('nan')]):.2f}"
        for s in (0, 0.5, 1.0, 2.0)) + ". All three strengths saturate the centre of this paper level (about 220 + 0.5*255 > 250), and the share of edges inside the blob falls as strength rises (text is progressively wiped out)."]
    lr = [r for r in normal if r["recipe"] == "clean" or r["recipe"].startswith("lowres")]
    sc = [r["true"].get("scale", 1.0) for r in lr]
    th_ = [r["text_h"] for r in lr]
    coef = np.polyfit(sc, th_, 1)
    r2 = np.corrcoef(sc, th_)[0, 1] ** 2
    L += [f"* **Text height** vs rendering scale (1, 0.75, 0.5, 0.35, 0.25; {len(lr)} normal pages): linear fit text_height = {coef[0]:.1f} x scale {coef[1]:+.1f}, R^2 = {r2:.3f}. "
          "Median text height by scale: " + ", ".join(f"{s}: {np.median([t for t, s2 in zip(th_, sc) if s2 == s]):.1f}px" for s in sorted(set(sc), reverse=True)) + "."]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].scatter(s_true, [r["unevenness"] for r in sh], s=10)
    ax[0].set(xlabel="true shadow strength", ylabel="unevenness", title="illumination")
    ax[1].scatter(sc, th_, s=10)
    ax[1].set(xlabel="scale", ylabel="median text height (px)", title="resolution")
    for pg in sorted({r["page_id"] for r in ev if r["page"]["dense"]})[:1]:
        pass
    ax[2].scatter([r["true"].get("glare_strength", 0) for r in gl], [r["glare"] for r in gl], s=10)
    ax[2].set(xlabel="glare strength", ylabel="glare_fraction", title="glare")
    fig.tight_layout()
    fig.savefig(out / "quality" / "exposure_resolution.png", dpi=130)
    plt.close(fig)
    L += ["", "![exposure and resolution](quality/exposure_resolution.png)", ""]

    (out / "quality_validation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
