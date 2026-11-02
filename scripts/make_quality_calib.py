"""Build a controlled image-quality calibration set with KNOWN degradation parameters.

Clean pages are rendered with synth.py using a calibration-only vocabulary (disjoint from train/test) and a new seed,
scaled so the paper sits at ~215-230 gray levels (noise on a pure-255 background would be clipped away and become
unobservable), then degraded by a recipe whose true parameters are written to labels.jsonl. Images are lossless PNG
so the pixels the estimators see are exactly what the recipe produced.

    python scripts/make_quality_calib.py --role eval   # -> data/quality_calib/        (used for reporting)
    python scripts/make_quality_calib.py --role tune   # -> data/quality_calib_tune/   (used only to pick hint thresholds)

The default degrade() and data/train, data/test are not touched.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import docusight  # noqa: E402,F401
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from docusight import synth  # noqa: E402

ROLES = {
    # (doc_type, layout variant) or ("dense", font size)
    "eval": {"seed": 20241, "pages": [("bank_statement", 1), ("invoice", 2), ("payslip", 0), ("kyc_id", 0),
                                      ("dense", 14), ("dense", 16), ("dense", 18)], "out": "data/quality_calib"},
    "tune": {"seed": 77123, "pages": [("bank_statement", 2), ("invoice", 0), ("dense", 15)], "out": "data/quality_calib_tune"},
}


def recipes(rng: random.Random) -> list[dict]:
    """Each recipe: name, ops [(op, params)], and the true parameters + noise-type label."""
    R: list[dict] = []

    def add(name, ops, noise_kind="none", **true):
        R.append({"name": name, "ops": ops, "noise_kind": noise_kind, "true": true})

    add("clean", [])
    for s in (2, 4, 6, 8, 12, 16):
        add(f"gauss_n{s}", [("gauss_noise", {"sigma": s})], "gaussian", noise_sigma=float(s))
    for b in (0.5, 1.0, 1.5, 2.0, 3.0):
        add(f"gblur{b}", [("gauss_blur", {"sigma": b})], blur_sigma=float(b))
    for sn in (4, 8, 16):  # blur first (optics), noise after (sensor): noise stays sharp on a blurry page
        for sb in (1.0, 2.0, 3.0):
            add(f"blur{sb}_noise{sn}", [("gauss_blur", {"sigma": sb}), ("gauss_noise", {"sigma": sn})], "gaussian",
                noise_sigma=float(sn), blur_sigma=float(sb))
    for ln in (3, 5, 9, 15):
        ang = round(rng.uniform(0, 180), 1)
        add(f"motion{ln}", [("motion_blur", {"length": ln, "angle_deg": ang})], motion_len=ln, motion_angle=ang)
    for d in (0.002, 0.005, 0.01, 0.03):
        add(f"sp{d}", [("salt_pepper", {"density": d})], "salt_pepper", sp_density=d)
    for peak in (1500, 600, 300, 150):
        add(f"poisson{peak}", [("poisson_noise", {"peak": peak})], "gaussian", poisson_peak=peak)  # sigma filled per page
    for q in (95, 75, 50, 30, 15):
        # class label by evidence: blocking is only visible on text pages at q <= 30; q 50-75 is a reported "grey" zone
        add(f"jpeg{q}", [("jpeg", {"quality": q})], "jpeg" if q <= 30 else ("grey" if q <= 75 else "none"), jpeg_quality=q)
    for st in (0.15, 0.3, 0.5, 0.7):
        ang = round(rng.uniform(0, 360), 1)
        add(f"shadow{st}", [("shadow", {"strength": st, "angle_deg": ang})], shadow_strength=st)
    for st in (0.5, 1.0, 2.0):
        add(f"glare{st}", [("glare", {"cx": 0.5, "cy": 0.4, "radius_frac": 0.12, "strength": st})], glare_strength=st)
    for sc in (0.75, 0.5, 0.35, 0.25):
        add(f"lowres{sc}", [("lowres", {"scale": sc})], scale=sc)
    for amp in (2, 4, 8):
        add(f"periodic{amp}", [("periodic", {"amplitude": amp, "period_px": 6.0, "angle_deg": 0.0})], "periodic", periodic_amp=amp)
    # appended last so the index (and therefore the seed) of every earlier recipe is unchanged
    for g in (0.6, 0.4, 0.25):
        add(f"dark{g}", [("gain", {"factor": g})], gain=g)
    return R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=list(ROLES), default="eval")
    ap.add_argument("--out")
    args = ap.parse_args()
    cfg = ROLES[args.role]
    out = Path(args.out or cfg["out"])
    (out / "images").mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg["seed"])
    gen = synth.Gen(rng, "calib")
    pages, labels = {}, []
    recs = recipes(rng)
    for pi, (kind, arg) in enumerate(cfg["pages"]):
        if kind == "dense":
            canvas, _, _ = synth.dense_text_page(gen, arg)
            page_id, variant = f"{args.role}_dense{arg}", None
        else:
            canvas, _, _ = getattr(gen, kind)(arg, None)
            page_id, variant = f"{args.role}_{kind}_v{arg}", arg
        gray = np.array(canvas.img.convert("L"), np.float32)
        factor = rng.uniform(0.84, 0.90)
        base = gray * factor
        pages[page_id] = {"doc_type": kind, "variant": variant, "dense": kind == "dense", "paper_level": round(255 * factor, 1),
                          "size": [gray.shape[1], gray.shape[0]], "text": canvas.gt_text(), "ink_fraction": round(float((gray < 128).mean()), 4)}
        for ri, rec in enumerate(recs):
            seed = cfg["seed"] * 1000 + pi * 100 + ri
            img = synth.apply_controlled(base, rec["ops"], seed)
            true = dict(rec["true"])
            if "poisson_peak" in true:  # std of Poisson noise on paper level L: sqrt(255 L / peak)
                true["noise_sigma"] = round(float(np.sqrt(255 * pages[page_id]["paper_level"] / true["poisson_peak"])), 3)
            img_id = f"{page_id}__{rec['name']}"
            cv2.imwrite(str(out / "images" / f"{img_id}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            labels.append({"id": img_id, "image": f"images/{img_id}.png", "page_id": page_id, "recipe": rec["name"],
                           "ops": rec["ops"], "noise_kind": rec["noise_kind"], "true": true, "seed": seed})
    (out / "pages.json").write_text(json.dumps(pages, indent=1))
    with open(out / "labels.jsonl", "w", encoding="utf-8") as fh:
        for r in labels:
            fh.write(json.dumps(r) + "\n")

    # vocabulary-disjointness audit against the train/test vocabularies
    used = set(synth.CALIB_FIRST + synth.CALIB_LAST + synth.CALIB_COMPANIES + synth.CALIB_ITEMS)
    other = set(synth.FIRST + synth.LAST + synth.COMPANIES + synth.ITEMS)
    assert not (used & other), f"calibration vocabulary overlaps train/test: {used & other}"
    print(f"wrote {len(labels)} images from {len(pages)} base pages x {len(recs)} recipes to {out}")
    print("calibration vocabulary is disjoint from train/test vocabularies (audited)")


if __name__ == "__main__":
    main()
