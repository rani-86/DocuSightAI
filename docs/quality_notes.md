# Image quality assessment (DIQA): interview notes

Plain-English notes so you can explain and defend every part of the quality work. Every number here is from a script in this repo (cited by file). Where I
do not know something, it says so.

## 0. The 30-second story

"My OCR pipeline decided whether to denoise or sharpen with whole-page numbers and magic thresholds. I tested those numbers on images where I *knew* the
defect, found they were blind or content-dependent, rebuilt them to measure noise only on blank paper and blur only on text edges, validated each against ground truth,
and then set the thresholds from measured OCR error instead of by hand. Two results surprised me: OCR was robust to noise (so the noise gate I was proud of was unnecessary),
and my blur estimator missed motion blur. I kept both in the write-up."

## 1. Why the old numbers were wrong

**Noise score (median of a high-pass residual over the whole page).** A page is mostly white paper. Take each pixel minus its 3x3 median: on exactly-flat paper that is 0.
If more than half the pixels are 0, the *median* of the residual is 0 no matter how noisy the rest is. Result on my 220 test pages: score exactly 0.0 on 203, max 3.0, gate 4, so
denoise fired on 0 pages (`scripts/analyze_degradation.py`). (On calibration images with unclipped paper it did read noise, but 0.8 gray levels too low on average.)

**Contrast score (p95 - p5).** On a text page, 95% of pixels are paper, so the 5th percentile is *also* paper: the "contrast" is about 0 and the "low contrast, apply CLAHE" gate fired on 85% of pages.

**Laplacian variance as a blur score.** Two failures, both measured:
1. *Content dependence.* It is large when there is lots of sharp text and small on a sparse page. At the same true blur, its coefficient of variation across 7 different pages was 0.59-0.91; my edge-width score was 0.01-0.13.
2. *Noise inflation.* The Laplacian is a second derivative, so it amplifies noise. At noise sigma 16 and blur sigma 3 the legacy score was 3,733x higher than for the same blurry page without noise: a blurry noisy page looks sharp.
   It *was* monotone in blur when there was no noise (7/7 pages), so "it is monotone" is not the argument; "it is content-dependent and noise-sensitive" is.

## 2. Noise types: where they come from, what they look like, how I detect them

| Type | Origin | Looks like | Detected here by |
|---|---|---|---|
| Gaussian (read) noise | sensor electronics, amplified at high ISO | fine grain everywhere, independent per pixel | sigma from a high-pass residual on blank-paper tiles (`noise_sigma`) |
| Poisson (shot) noise | photons arrive randomly; worse in low light; strength depends on brightness | grain that is stronger on brighter areas | same estimator (on white paper it behaves like Gaussian with sigma = sqrt(255 x level / peak)) |
| Salt-and-pepper (impulse) | dead/hot pixels, transmission bit errors, dust on a scanner | isolated pure black or white dots | `impulse_fraction`: pixels far from their 3x3 median whose 8 neighbours all agree with that median |
| JPEG blocking | lossy compression works on independent 8x8 blocks | a faint grid, ringing around text | `jpeg_blockiness`: second difference at the 8x8 boundaries vs inside blocks, with long rules masked out |
| Periodic / moire | screen photography, scanner banding, interference | regular stripes or a crosshatch | `periodic_peak`: strongest isolated FFT peak on blank tiles vs the background level |

Important property: they need *different fixes*. Gaussian noise: smoothing (non-local means). Salt-and-pepper: median / despike (a Gaussian blur would smear each dot, not remove it). JPEG: do not sharpen
(it amplifies the grid). So the hints (`noise_types` in the API) matter, not only a single sigma.

## 3. Why noise is estimated on background tiles

Noise is only *observable* where the true signal is constant. On a text stroke you cannot tell noise from the edge. So: cut the page into 32x32 tiles, keep the tiles that are plain paper, and measure the residual there.
Details that matter (and are good to mention):
* **Immerkaer's kernel** [1 -2 1; -2 4 -2; 1 -2 1] has zero response to any locally flat or planar signal, so only noise survives, with standard deviation 6 x sigma. I report both the mean-based estimate (efficient for pure Gaussian) and the **MAD**-based one (robust to heavy tails). They agree within ~0.03 here (both MAE about 0.10).
* **Deciding "flat" adaptively**: noise raises gradients, so a fixed gradient threshold throws away every tile at sigma >= 14. The threshold scales with each tile's own robust sigma (median-based, so a few stroke pixels do not move it), and a block-mean test rejects blurred dense text that is smooth but not flat.
* **Impulses are removed before choosing tiles.** My first version median-filtered first; that also erases 1-px strokes and table rules, so tiles with rules looked like paper with "impulses" and 18 clean pages were flagged salt-and-pepper. (Found by the first validation run; kept in `reports/quality_validation_v1_initial.md`.)
* **Honest "unknown":** if too few paper tiles exist (very dense or very low-res pages) or the paper is clipped to 255, the estimate returns `measurable = False` instead of a number. An earlier fallback used the least-textured tiles anyway and reported a mean sigma of 4.3 on blurred dense low-resolution pages that had no noise at all.

Validated: MAE 0.10 gray levels vs the true injected sigma (legacy 0.92); no false noise on blur, shadow, low-res, JPEG, salt-and-pepper (`scripts/validate_quality.py`, section A).

## 4. Blur

* **Defocus (Gaussian)** blur: the page is a sharp image convolved with a bell curve. **Motion blur**: convolved with a line, so edges soften only *along* the motion direction.
* **Edge width** (my main blur measure): for a step edge of contrast C blurred by a Gaussian of width s, the peak gradient is C / (s sqrt(2 pi)). So width = local contrast / (sqrt(2 pi) x peak gradient). It is contrast-independent and does not depend on how much text there is. It has a floor of ~0.8 px because glyph strokes are 1-3 px wide and anti-aliased, so it is a relative scale, not an absolute focus value.
* **Order matters**: blur is measured *after* denoising, because noise adds high-frequency energy that makes a blurry page look sharp. At noise sigma 16, edge width measured without denoising was 0.81-1.46x its noise-free value; after denoising, 0.85-1.18x (the worst case, true blur 1.0, fell from 1.46x to 1.18x).
* **Known failure (be ready to say this):** edge width is built from the gradient *magnitude*, which motion blur reduces only in one direction, so it under-reads: 9 px motion blur gives CER ~40% but an edge width of 0.91 (looks sharp). The blur gate does not catch motion blur. Fix = directional edge width (measure along several orientations and take the worst). Not done.

## 5. Exposure, resolution, glare

* **Illumination unevenness**: estimate the paper brightness field (shrink the image, "close" away dark strokes, smooth) and take the ratio of its 98th to 2nd percentile. Spearman 0.964 with the true shadow strength; it reads lower than the exact 1/(1-s) because of the percentile range.
* **Text height**: median height of glyph-sized connected components. Linear in the rendering scale (R^2 0.977). Measured in *native* pixels (divided by the up-scale factor), because up-scaling adds no information.
* **Glare**: a large blown-out region on a page whose paper is not itself pure white. Hard part: telling "text wiped out by a flash" from "paper saturated by a lighting gradient" (text still legible). I use the ratio of text-edge density inside the region to just around it (band by band; undefined if nothing is around it). It is a weak cue: **one false retake in 220 test pages** (`test_invoice_0043`). Say this honestly.

## 6. How I validated (this is the "prove it works" part)

1. **Controlled ground truth.** `scripts/make_quality_calib.py` renders clean pages, scales the paper to ~225 so noise is not clipped away, then applies *known* degradations (Gaussian sigma 2-16, Poisson, blur, motion, salt-and-pepper, JPEG, shadow, glare, low-res, periodic, darkening) and saves the true parameters. Different seed and a vocabulary disjoint from train/test.
2. **Tuning vs evaluation split.** The hint thresholds were fitted on a separate 165-image set; results are reported on the 385-image set.
3. **Correlation is not enough.** The legacy noise estimator correlates at r = 0.99 with truth on the calibration set; what separates the estimators is *bias* (slope 0.89 vs 0.98), *false alarms* on other defects, and *behaviour on dense text*. Always show those, not just r.
4. **Failure analysis.** The first run gave 81% hint accuracy, 0% JPEG recall, a nonsense periodic threshold (5e12) and 42% false alarms on dense low-res pages. Each had a diagnosed cause (section 3 and the revision table in `reports/quality_validation.md`). After fixing: 97.6%.
5. **A caveat I state myself:** because I looked at the evaluation set once before fixing, it is development-grade; the independent check is data/test.

## 7. How thresholds were chosen

Not by hand and not by "looks about right". `scripts/quality_vs_cer.py` OCRs the calibration images (raw and, where a fix exists, enhanced), computes page CER against the known text, and reads off:
* **retake** = the score at which the *best available* variant exceeds 5% CER;
* **enhance** = the score at which the fix lowers CER by at least 1 percentage point (linear interpolation between recipe means);
* it also records where the fix **stops helping** (deblur: from 1.04, stops ~1.36, hopeless ~1.45).

Surprises it produced (good interview material):
* **Noise never mattered for PaddleOCR** up to sigma ~19 (CER <= 0.2%) and denoising did not help: the data say delete the noise gate.
* **Darkness and shadows never mattered** (CER ~0 at paper level 55): no retake rule for them even though I had planned one.
* **What does matter**: defocus blur, low resolution (< 5.4 px glyph height), glare that wipes text, and isolated impulses.
* Mixing defocus and motion blur on one curve made the blur thresholds contradictory; that is how I discovered the motion-blur blind spot.

Limits: two base pages per point, one realisation of each defect, one synthetic generator: each threshold is good to about one recipe step. Re-run on real pages before trusting them.

## 8. How the gate works in the pipeline

Measure -> fix -> re-measure, in the right order: JPEG hint on the original decode (resampling destroys the 8x8 grid) -> noise on the resized page before any rotation (interpolation smooths noise) -> despike -> geometry -> denoise -> **blur after denoise** -> deblur -> exposure/text height -> decision.
Decisions: **accept** (nothing above its threshold), **enhance** (a fixable defect; steps listed), **retake** (beyond repair: text too small, glare over text, too blurry). Retake never stops extraction; it sets `needs_review` with reason `IMAGE_QUALITY` and the human-readable reasons are in the response.
On the 220 test pages: 201 accept / 18 enhance (deblur) / 1 retake (false positive). On calibration images: mean raw CER 2.0% (accept), 8.4% (enhance), 19.8% (retake).

## 9. Known weaknesses (say them before they ask)

1. Motion blur is missed (edge width is isotropic).
2. JPEG blocking is only detectable at quality <= ~30 on text pages; q 50-75 is invisible to it. The hint fires on 122/220 test pages, and I cannot verify it per page (the real JPEG quality was not recorded).
3. Strong periodic noise defeats the paper-tile selection (periodic recall 67%).
4. Noise on a saturated page is unobservable -> reported "unmeasurable".
5. The glare cue cannot separate a flash from a saturating lighting gradient (1 false retake in 220).
6. CLAHE is still gated by the legacy contrast rule (fires on 65/80 subset pages) and is *not* calibrated. The ablation shows no measurable effect (0.00 pp, CI +/-0.6), so it is inconclusive, not proven useless.
7. All data are synthetic; thresholds will shift on real photos.

## 10. Likely questions, short answers

* *Why not just use a deep no-reference IQA model (BRISQUE, NIQE, a CNN)?* Those are trained on natural photos, not text pages, and give one opaque score. I needed defect-specific, explainable numbers tied to OCR error and to a specific fix. A learned DIQA model trained on (image, CER) pairs is the obvious next step once there is real labelled data.
* *Why is sigma estimated by a median of tiles and not a mean?* A few tiles with leftover structure would pull a mean up; the median ignores them.
* *Why is MAD more robust than the standard deviation?* It has a 50% breakdown point: up to half the samples can be outliers (impulses, JPEG ringing) without moving it.
* *Why does the blur score change when I denoise first?* Noise adds gradient energy at edges and in flat areas; it makes everything look sharper. Denoising first reduces that bias (in my test the worst case fell from 1.46x to 1.18x; it did not remove it).
* *How do you know it works?* Known-truth images (MAE 0.10 vs 0.92), a split between tuning and evaluation data, failure analysis, and thresholds read from measured CER. And I list where it does not work.
* *How would you improve it?* Directional edge width for motion blur; calibrate on real scans and phone photos; train a small model to predict CER from the quality vector; per-region (not whole-page) decisions so a glare spot flags only the affected fields.

## 11. Where things are

| What | Where |
|---|---|
| Estimators, decision logic, thresholds loader | `docusight/quality.py` |
| Calibrated thresholds + evidence | `docusight/quality_thresholds.json` |
| Pipeline integration | `docusight/preprocessing.py` (`_preprocess_diqa`), `docusight/pipeline.py` |
| Controlled degradations | `docusight/synth.py` (`op_*`, `apply_controlled`, `dense_text_page`), `scripts/make_quality_calib.py` |
| Estimator validation | `scripts/validate_quality.py` -> `reports/quality_validation.md`, `reports/quality/*.png` |
| Quality -> CER -> thresholds | `scripts/quality_vs_cer.py` -> `reports/quality_vs_cer.md` |
| Heavy-vs-medium analysis | `scripts/analyze_degradation.py` -> `reports/degradation_analysis.md` |
| Ablation | `scripts/run_ablation.py` -> `reports/ablation.md` |
| Tests (no OCR) | `tests/test_quality.py` |

## 12. Ablation results (80 data/test docs; see README section 5)

New gate = legacy on end-to-end field accuracy (98.33% both). Geometry correction is what matters (all steps off: -2.29 pp, page CER 0.0077 -> 0.0621; wide CI). Denoise/despike never fired. Deblur: -0.21 pp when removed. CLAHE: no measurable effect. Say plainly that with only 5 erroneous documents the intervals are wide.
