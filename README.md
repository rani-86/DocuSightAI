# DocuSight AI — Layout-Aware Document Intelligence

**Tech stack:** Python · PyTorch · Hugging Face Transformers · OpenCV · PaddleOCR · scikit-learn · FastAPI · Docker

An end-to-end document AI pipeline for lending workflows. It takes a scanned or photographed
**bank statement, invoice, payslip or PAN-style KYC card** and returns the document type, typed key
fields, table rows and a review decision. Every value carries the bounding box of the region it was read
from. A field that is missing, inconsistent or low-confidence sends the document to human review.

```
image/PDF ─► OpenCV preprocessing ─► PaddleOCR ─► layout analysis ─► classifier (HF BERT / TF-IDF)
              perspective · deskew      tokens +     lines → segments      │
              denoise · deblur · CLAHE  boxes + conf  reading order        ▼
              binarise · morphology                              schema for doc type
                                                                         │
            ┌────────────────────────────────────────────────────────────┤
            ▼                                                            ▼
   key-value extraction                                     table recognition
   label → same-segment / right / below                     ruled grid (morphology) or
   + position-aware OCR correction                          header-anchored columns
            └──────────────────────► validation ◄────────────────────────┘
                     required · format · confidence · arithmetic consistency
                                      │
                            ExtractionResult (+ bbox per value) ─► FastAPI / review queue / metrics
```

## Results (held-out set) - pre-DIQA pipeline

Measured on **220 held-out documents** (55 per type). The test set's names, companies and items never
appear in training. Every document went through real PaddleOCR on CPU, after synthetic scan/photo degradation.

| Metric | Value |
|---|---|
| Document classification macro-F1 (fine-tuned BERT, on OCR text) | **1.000** (TF-IDF baseline: 1.000) |
| Field-level accuracy (1,320 fields, exact match after normalisation) | **97.4%** |
| Field-value CER / page-level OCR CER | **0.021 / 0.010** |
| Table cell accuracy: payslip / bank statement / invoice | 100% / 92.2% / 92.9% |
| Injected anomalies (tampered totals, missing fields, minor DOB) flagged | **26 / 26 (100%)** |
| Problem-document recall / flag precision | 85.7% / 90.6% |
| Straight-through rate (no human review) | 75.9% |
| Auto-approved documents that are fully correct | 95.2% |
| Accuracy of values with confidence ≥ 0.95 vs < 0.80 | 99.4% vs 22.2% (the confidence signal is informative) |
| End-to-end latency p50 / p95 (CPU, OCR ≈ 98% of it) | 7.3 s / 13.5 s |

Honest reading of these numbers:
* Classifying four very different document types is easy. Every model, including the keyword prior,
  reaches F1 = 1.0 here. The classifier earns its place on noisier and larger label sets, not this one.
* Most residual errors are OCR errors: a dropped single digit in a borderless table, a clipped last character
  where text touches a cell rule, or "Emp ID:" misread as "D:". Validation is designed to catch these,
  and does in most cases. The remaining silent errors are mostly one-character slips in free-text narrations.

![Source-region overlay: the tampered total is flagged red](reports/overlay_invoice.png)

The full report is in [`reports/eval_report.md`](reports/eval_report.md), with raw numbers in
`reports/eval_report.json`.

## Image quality assessment (DIQA)

The pipeline used to decide whether to denoise / sharpen / boost contrast from three whole-page numbers and three magic thresholds. I tested
those numbers against images with *known* defects and replaced what failed. Everything below comes from scripts in this repo
(`scripts/analyze_degradation.py`, `make_quality_calib.py`, `validate_quality.py`, `quality_vs_cer.py`, `run_ablation.py`); the full reports are in
[`reports/degradation_analysis.md`](reports/degradation_analysis.md), [`reports/quality_validation.md`](reports/quality_validation.md) and
[`reports/quality_vs_cer.md`](reports/quality_vs_cer.md). Interview-style explanations: [`docs/quality_notes.md`](docs/quality_notes.md).

> The headline table above was produced by the **pre-DIQA ("legacy") gates**. It is unchanged and reproducible (`python scripts/evaluate.py`,
> `DOCUSIGHT_DIQA_MODE=legacy`). The new gate is the default in the service; its end-to-end effect on `data/test` is measured by the ablation in section 5
> (same field accuracy as legacy on the 80-document subset).

### 1. What was wrong with the old gates (measured on the 220 held-out documents)

* The legacy noise score was **exactly 0.0 on 203/220 documents** (max 3.0, gate 4), so `denoise` fired on **0/220**. It is the median of a high-pass residual, and on a
  page that is more than half flat paper that median is 0 whatever the noise. The contrast gate (p95-p5 < 150) fired on **187/220 (85%)** because on a text page the 5th
  percentile is still white, so CLAHE was effectively always on. Laplacian variance varies with page content as much as with focus.
* **Heavy vs medium (98.6% vs 95.3% field accuracy) is not a robustness story.** Heavy minus medium = +3.3 pp, document-level bootstrap 95% CI [+0.2, +6.8],
  permutation p = 0.079; nothing survives a correction for three comparisons. Layout mix does not explain it (medium: 26 errors observed vs 17.9 expected from its mix),
  and the error rate is non-monotonic in `degrade_level` (Spearman 0.11, p = 0.11): that scalar only drives some of the randomly drawn artefacts. The honest summary is
  "borderline, and not explained by the metadata I recorded". See [`reports/degradation_analysis.md`](reports/degradation_analysis.md).

### 2. New estimators (`docusight/quality.py`, pure numpy/OpenCV) and how they were validated

Validated on a controlled set with saved true parameters (`data/quality_calib`: 7 base pages x 55 recipes = 385 images; calibration-only vocabulary disjoint from
train/test; 3 pages are text-dense). Hint thresholds were fitted on a *separate* 165-image tuning set. The estimators were revised once after a first run exposed defects
(first-run results are kept in [`reports/quality_validation_v1_initial.md`](reports/quality_validation_v1_initial.md)), so treat the evaluation set as development-grade; the
blind check is the `data/test` experiment.

| Question | Legacy | New | Notes |
|---|---|---|---|
| Noise sigma error (MAE in gray levels; 175 Gaussian/Poisson images) | 0.92 (bias -0.79, slope 0.89) | **0.10** (bias -0.08, slope 0.98) | measured on paper-only tiles, Immerkaer + MAD |
| ...on text-dense pages | 0.80 | **0.07** | text does not leak into the estimate |
| False noise on JPEG q<=50 *dense* pages | 33% of images | **0%** | |
| Salt-and-pepper | invisible (sigma 0) | impulse fraction tracks density (0.0020 / 0.0050 / 0.0099 / 0.0288 for 0.002 / 0.005 / 0.01 / 0.03) | isolated-pixel rule |
| Blur score varies with page content (CV across 7 pages at equal blur) | 0.59-0.91 | edge width **0.01-0.13** | Laplacian variance is monotone in blur but content-dependent |
| Blur score under noise sigma 16 (blur 1 / 2 / 3) | inflated **40x / 687x / 3733x** | edge width (after denoising) **1.18x / 1.16x / 0.85x** | measured after denoising |
| Noise-type hints (none / Gaussian / salt-pepper / JPEG q<=30 / periodic) | n/a | **97.6%** accuracy on 371 images (first run: 81.0%) | thresholds fitted on the tuning set |
| Illumination unevenness vs true shadow strength | n/a | Spearman 0.964 | |
| Text height vs rendering scale | n/a | R^2 0.977 | |

Failure modes I found and kept in the docs rather than hiding: JPEG blocking is detectable only at quality <= ~30 on text pages (q 50-75 is undetected: the evidence is
smaller than page-to-page variation); strong periodic noise defeats the paper-tile test (periodic recall 67%); the isotropic edge-width estimator **misses motion blur**
(9 px motion blur has CER ~40% but an edge width of 0.91, i.e. it looks sharp); noise on a blown-out (255) page is unobservable and is reported as unmeasurable, not zero.

![noise estimate vs truth](reports/quality/noise_est_vs_true.png)
![blur under noise](reports/quality/blur_vs_true_with_noise.png)

### 3. Thresholds are read from OCR error, not chosen

`scripts/quality_vs_cer.py` OCRs 76 calibration images (2 base pages x recipes; raw and, where a fix exists, enhanced), computes page CER, and reads thresholds
off the curves (CER > 5% = unacceptable; a fix "helps" if it lowers CER by at least 1 pp). They are stored with their evidence in `docusight/quality_thresholds.json`.
Clean-page CER is 0.0000. With two pages per point these are coarse (about one recipe step).

| Defect | Finding | Threshold |
|---|---|---|
| Gaussian / Poisson noise | PaddleOCR is robust: CER <= 0.2% up to sigma ~19, and denoising never helps by 1 pp | **no noise gate** (the old `denoise` would have been pointless here) |
| Defocus blur (edge width, px) | deblur helps from 1.04 (CER 4.2% -> 0.3% at blur sigma 1.5), stops helping at ~1.36, unreadable beyond ~1.45 | enhance > **1.04**, retake > **1.45** |
| Low resolution | CER 2.7% at 5.5 px glyph height, 35% at 4 px | retake < **5.4 px** native glyph height |
| Glare | CER 11% only when the text inside the blown area is gone | retake if > **25%** of the page is blown out **and** text-edge ratio < **0.24** |
| Isolated impulses | CER 2.5% at 0.5% density; despike brings it to 0.26% | despike above **0.0022** fraction |
| Dark page / uneven lighting | CER ~0 even at paper level 55 and shadow strength 0.7 | **no retake rule** (not supported by the data) |

Gate consistency check on the calibration images (`reports/ablation.md`): mean raw CER is **2.0% for accept (53 images), 8.4% for enhance (11) and 19.8% for retake (12; 75% of them have CER > 5%)**.
It misses 5 images (CER > 5% but not marked retake: Gaussian blur sigma 2.0, glare 2.0, motion blur 9 and 15). These are the images the thresholds came from, so this checks consistency, not generalisation.

### 4. The gate in the pipeline (`docusight/preprocessing.py`)

Order: JPEG hint on the decoded input -> noise / impulse / periodic on the size-normalised page *before* any rotation or warp (resampling smooths noise) -> despike -> geometry -> denoise -> **blur
measured after denoising** -> deblur -> exposure and text height -> CLAHE -> accept / enhance / retake with human-readable reasons. `retake` never blocks extraction: the page is still read and gets
`needs_review = true` with reason `IMAGE_QUALITY`. The response gains an optional `quality` object (scores, noise types, decision, reasons, steps) and `/metrics` gains
`docusight_quality_decisions_total{decision=...}`. Thresholds can be overridden with `DOCUSIGHT_Q_<NAME>` (for example `DOCUSIGHT_Q_BLUR_ENHANCE=1.2`; `none` disables a check), and
`DOCUSIGHT_DIQA_MODE=legacy` restores the original gates exactly (verified byte-identical on 10 test images).

On the 220 `data/test` pages (no OCR needed): **201 accept, 18 enhance (all deblur, edge width 1.04-1.16), 1 retake**. Preprocessing p50 is 355 ms (was 83 ms; OCR is about 7 s).
The one retake (`test_invoice_0043`, glare 27.7%) is a **false positive**: a lighting gradient saturates the bright side of the page but every character is legible (the legacy run
extracted it without error). The 'jpeg_blocking' hint fires on 122/220 test pages; every test image was JPEG-compressed at quality 45-90, but the per-image quality was not recorded, so I cannot
check that hint page by page. CLAHE still uses the legacy contrast gate (it fires on 187/220) and is **not calibrated**: the ablation (section 5) found no evidence that it helps.

### 5. Preprocessing ablation (80 `data/test` documents, 20 per type, PaddleOCR)

`scripts/run_ablation.py`; full table in [`reports/ablation.md`](reports/ablation.md). Delta = field accuracy minus `v2_all_on`, paired document-level bootstrap 95% CI.

| Configuration | Field acc. | Delta vs all_on [95% CI] | Page CER |
|---|---|---|---|
| `legacy` (pre-DIQA gates) | 98.33% | 0.00 pp | 0.0085 |
| `v2_all_on` (calibrated gate) | 98.33% | - | 0.0077 |
| `v2_all_off` (no geometry, no enhancement) | 96.04% | -2.29 pp [-5.42, +0.83] | 0.0621 |
| off: perspective | 98.33% | 0.00 pp | 0.0140 |
| off: deskew | 96.88% | -1.46 pp [-4.46, +1.55] | 0.0544 |
| off: despike / denoise | 98.33% | 0.00 pp (never fired: 0/80) | 0.0077 |
| off: deblur | 98.12% | -0.21 pp [-0.64, 0.00] | 0.0078 |
| off: clahe | 98.33% | 0.00 pp [-0.62, +0.62] | 0.0093 |

What this does and does not show (80 documents, only 5 of which have any field error, so the intervals are wide):

* **The calibrated gate matches the legacy pipeline end to end** (98.33% both, same 5 documents wrong). I cannot claim it is more accurate on this data; the gain is that the decisions are measured and explainable, and `retake` adds a review signal.
* **Geometry is the part that matters.** Turning off perspective and deskew together costs 2.3 pp field accuracy and raises page CER from 0.0077 to 0.0621 (heavy bucket: 99.4% -> 92.4%). Deskew alone: page CER 0.0544. The field-accuracy CIs still include 0, so the page-CER change is the stronger evidence.
* **Denoise and despike never fired** on these pages (the synthetic noise is mostly erased by the JPEG step in the generator), consistent with the calibration finding that PaddleOCR does not need a noise gate. They are untested on real noisy scans.
* **Deblur** fired on 6/80 pages and helps slightly (-0.21 pp when removed, CI touching 0). Those 6 documents are the weakest group (86% field accuracy), so deblur is not sufficient for them.
* **CLAHE is inconclusive:** it fires on 65/80 pages, removing it changes accuracy by 0.00 pp (CI +/-0.6), page CER is slightly better with it (0.0077 vs 0.0093) but field CER slightly worse (0.0116 vs 0.0100). There is no evidence it helps, so it is a candidate to remove, but this experiment cannot prove that.
* The one `retake` page in this subset was read perfectly (the known glare false positive). Latency columns in `reports/ablation.md` are distorted by the OCR cache (cache hits skip OCR); only the preprocess column is meaningful (legacy 82 ms, new gate about 340-400 ms).

### Limitations specific to image quality

* Thresholds come from two synthetic base pages per point and one synthetic generator; real phone photos (motion blur, specular glare, compression chains, colour casts) will move them. Re-run `scripts/quality_vs_cer.py` on real pages.
* Motion blur is not detected (see above); an anisotropic edge-width estimate is the obvious fix.
* Glare versus a saturating lighting gradient cannot be separated reliably with the edge-ratio cue (one false retake in 220).
* The calibration images are single-pass grayscale PNG/JPEG; real colour JPEG chains and scanner sharpening are not covered, which is why the JPEG hint is only a hint.

## What each part does

| Bullet | Where |
|---|---|
| **OCR + image preprocessing.** Perspective correction for phone photos (page-quad detection + warp), projection-profile deskew, calibrated DIQA gate (despike, non-local-means denoise, unsharp deblur, CLAHE; accept / enhance / retake), adaptive threshold + morphology, ROI cropping | [`preprocessing.py`](docusight/preprocessing.py), [`quality.py`](docusight/quality.py) |
| **OCR engines.** PaddleOCR PP-OCRv5 (primary), Tesseract (fallback), optional TrOCR (Hugging Face) re-read of low-confidence field crops | [`ocr.py`](docusight/ocr.py) |
| **Document classification over 4 categories.** BERT fine-tuned in PyTorch (`google/bert_uncased_L-2_H-128_A-2`) on OCR-noise-augmented text, compared against a scikit-learn TF-IDF + LogisticRegression baseline and a keyword prior | [`classifier.py`](docusight/classifier.py), [`scripts/train_classifier.py`](scripts/train_classifier.py) |
| **Layout-aware key-value extraction.** Tokens → lines → segments (gap analysis). Fuzzy label matching with rival-label disambiguation (e.g. *Opening* vs *Closing Balance*). The value is read from the same segment, the right neighbour or the cell below. Position-aware OCR correction for PAN/IFSC/GSTIN (`O↔0`, `B↔8`, …) | [`layout.py`](docusight/layout.py), [`extractor.py`](docusight/extractor.py), [`normalize.py`](docusight/normalize.py) |
| **Table recognition.** Ruled tables: horizontal/vertical rule detection with morphology gives column boundaries. Borderless tables: header-anchored column spans. Wrapped multi-line narrations are merged into their row | [`tables.py`](docusight/tables.py) |
| **Evaluation on 200+ held-out samples.** Classification macro-F1 and confusion matrix, field-level accuracy, field- and page-level CER, table row/cell accuracy, breakdowns by degradation and layout, confidence calibration, latency | [`scripts/evaluate.py`](scripts/evaluate.py), [`metrics.py`](docusight/metrics.py) |
| **Confidence-based validation + source-region tracking.** Checks for missing required fields, ID format (regex), low confidence and future dates. Cross-field arithmetic: opening + credits − debits = closing, running balance, qty × rate, Σ line items = subtotal, gross − deductions = net, KYC age ≥ 18. Each issue carries the bbox of its source region | [`validation.py`](docusight/validation.py), [`scripts/visualize.py`](scripts/visualize.py) |
| **Serving.** FastAPI (`/v1/documents/extract`, `/classify`, `/extract-batch`, `/review-queue`), Prometheus `/metrics` (per-stage latency histograms, review-flag and issue counters), PII-masked JSON logs, Docker image with baked-in OCR weights | [`api.py`](docusight/api.py), [`observability.py`](docusight/observability.py), [`Dockerfile`](Dockerfile) |

## Data

No public dataset covers Indian lending documents with field-level labels, so
[`synth.py`](docusight/synth.py) generates them:

* 4 document types × 3 layout variants. Label wording, key-value style (`Label: value` / value to the right / value below),
  ruled vs borderless tables, 3 date formats, Indian digit grouping (`12,34,567.00`) and 7 font families all vary.
* Scan and photo degradation: perspective warp on a background, rotation, uneven lighting, Gaussian noise, blur
  and JPEG compression, at a random severity per document.
* About 12% of documents carry **injected anomalies** (tampered totals or balances, missing GSTIN/IFSC/PAN,
  minor DOB), so review routing is measured against known ground truth.
* **Leakage prevention.** Train and test draw names, companies and line items from *disjoint vocabularies*,
  and use different seeds. `scripts/generate_data.py` audits the splits (no shared entities, no duplicate pages)
  and fails if leakage is found. The classifier's validation split is made *per document before
  augmentation*, so noisy copies of a page never straddle the boundary.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_data.py --train-per-type 150 --test-per-type 55   # 600 train / 220 test
python scripts/train_classifier.py                                        # TF-IDF baseline + BERT fine-tune
python scripts/evaluate.py --data data/test                               # pre-DIQA gates -> reports/eval_report.{md,json}
python scripts/evaluate.py --mode v2 --per-type 20                        # calibrated DIQA gate on a stratified subset
python scripts/visualize.py data/test/images/test_invoice_0001.jpg --out reports/overlay.png  # source-region overlay
pytest -q                                                                  # unit + API tests (no OCR models needed)
```

Serve:

```bash
uvicorn docusight.api:app --port 8000
curl -F "file=@data/test/images/test_invoice_0001.jpg;type=image/jpeg" localhost:8000/v1/documents/extract
```

Docker:

```bash
docker build -t docusight-ai . && docker run -p 8000:8000 docusight-ai
```

Configuration is through `DOCUSIGHT_*` environment variables (see [`config.py`](docusight/config.py)), e.g.
`DOCUSIGHT_OCR_ENGINE=tesseract`, `DOCUSIGHT_TROCR_REFINE=1`, `DOCUSIGHT_FIELD_CONF_THRESHOLD=0.85`.

### Sample response (trimmed, `reports/sample_invoice_0001.json`)

This test invoice has a tampered grand total. The issue points back to the exact pixels it came from:

```json
{
  "doc_type": "invoice", "doc_type_confidence": 0.8248,
  "fields": {
    "total_amount": {"value": "141226.52", "raw_text": "1,41,226.52", "confidence": 0.9583,
                     "bbox": [831, 658, 933, 690], "method": "right"}
  },
  "tables": [{"strategy": "header_anchored", "columns": ["item", "qty", "unit_price", "amount"], "rows": ["..."]}],
  "issues": [{"code": "INCONSISTENT", "field": "total_amount", "severity": "error",
              "message": "subtotal + tax != total", "bbox": [831, 658, 933, 690]}],
  "needs_review": true, "review_reasons": ["INCONSISTENT"],
  "preprocessing": ["clahe", "adaptive_threshold+morph_open"],
  "timings_ms": {"preprocess": 73.2, "ocr": 7760.9, "layout": 0.5, "classify": 61.4, "extract": 12.1, "validate": 0.4, "total": 7908.5}
}
```

## Engineering notes

* **Windows DLL clash.** PaddlePaddle and PyTorch ship conflicting DLLs, so torch must load first
  (`docusight/__init__.py`). Paddle 3.3's oneDNN CPU kernels fail on some PIR attributes, so MKL-DNN is
  off by default (`DOCUSIGHT_PADDLE_MKLDNN=1` turns it back on).
* **Phrase-level vs word-level OCR.** PaddleOCR returns phrases, so each detection is kept as one segment.
  Tesseract returns words, which are merged into segments by horizontal gap analysis.
* **Thread safety.** The Paddle predictor is not thread-safe. The API serialises inference behind a lock
  and scales out with processes or replicas rather than threads.
* **PII.** Results are held in memory only, and logs mask IDs, names and DOB (last 4 characters kept).

## Limitations / next steps

* The data is synthetic. The pipeline is template-agnostic (no coordinates are hard-coded), but real scans
  will show a lower ceiling. The next step is labelling real documents in Label Studio and fine-tuning a
  layout model such as LayoutLMv3 for token classification.
* OCR dominates latency on CPU. GPU inference, batching detection across pages, or exporting the PP-OCR
  models to ONNX Runtime/TensorRT would be the first optimisations.
* Only the first page of a PDF is processed.
