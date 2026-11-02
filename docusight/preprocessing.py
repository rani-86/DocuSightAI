"""OpenCV image preprocessing: quality checks, geometric correction and enhancement.

Every step is conditional on a measured quality signal so clean scans are not
degraded by unnecessary filtering. The list of steps actually applied is
returned so it can be logged with each prediction. See docusight/quality.py for the
estimators and the accept / enhance / retake decision (mode "v2"); `blur_score` and
`noise_score` below are the legacy whole-page estimators, kept for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import quality as Q
from .schemas import BBox


@dataclass
class PreprocessResult:
    image: np.ndarray  # enhanced BGR image fed to OCR
    gray: np.ndarray
    binary: np.ndarray  # inverted binary (text = 255), used for table-line detection
    steps: list[str] = field(default_factory=list)
    skew_angle: float = 0.0
    quality: dict[str, float] = field(default_factory=dict)
    qinfo: dict | None = None  # DIQA assessment (v2 mode only): decision, reasons, scores, noise types


# --------------------------------------------------------------------------- io

def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes (or the first page of a PDF) into a BGR array."""
    if data[:4] == b"%PDF":
        return _render_pdf_first_page(data)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unsupported or corrupt image")
    return img


def _render_pdf_first_page(data: bytes) -> np.ndarray:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ValueError("PDF input requires `pip install pypdfium2`") from exc
    pdf = pdfium.PdfDocument(data)
    pil = pdf[0].render(scale=2.0).to_pil()
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------- quality

def blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian; low values indicate a blurry image."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def noise_score(gray: np.ndarray) -> float:
    """Robust noise sigma estimate from the median absolute high-pass residual."""
    residual = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
    return float(1.4826 * np.median(np.abs(residual - np.median(residual))))


def contrast_score(gray: np.ndarray) -> float:
    p5, p95 = np.percentile(gray, (5, 95))
    return float(p95 - p5)


# ------------------------------------------------------------------- geometry

def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype=np.float32)


def find_document_quad(gray: np.ndarray) -> np.ndarray | None:
    """Locate the page outline when the document is photographed on a background."""
    h, w = gray.shape
    scale = 800.0 / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray.copy()
    scale = min(scale, 1.0)
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_total = small.shape[0] * small.shape[1]
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(c)
        if not 0.35 * area_total < area < 0.97 * area_total:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return _order_quad(approx / scale)
    return None


def perspective_correct(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = quad
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, m, (width, height), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def estimate_skew(gray: np.ndarray, max_angle: float = 6.0, step: float = 0.2) -> float:
    """Projection-profile deskew: the angle that maximises row-sum variance."""
    h, w = gray.shape
    scale = 1000.0 / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray
    binv = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    center = (binv.shape[1] / 2, binv.shape[0] / 2)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(binv, m, (binv.shape[1], binv.shape[0]), flags=cv2.INTER_NEAREST)
        return float(np.var(rot.sum(axis=1, dtype=np.float64)))

    coarse = np.arange(-max_angle, max_angle + 1e-6, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(best - 1.0, best + 1.0 + 1e-6, step)
    return float(max(fine, key=score))


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    border = (255, 255, 255) if img.ndim == 3 else 255
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=border)


def crop_roi(img: np.ndarray, bbox: BBox, pad: int = 4) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox
    return img[max(0, y0 - pad): min(h, y1 + pad), max(0, x0 - pad): min(w, x1 + pad)]


# -------------------------------------------------------------- enhancement

def denoise(gray: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, h=float(np.clip(sigma * 1.2, 5, 25)), templateWindowSize=7, searchWindowSize=21)


def deblur(gray: np.ndarray, amount: float = 1.2) -> np.ndarray:
    """Unsharp masking to recover stroke edges."""
    blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
    return cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def binarize(gray: np.ndarray) -> np.ndarray:
    binv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    return cv2.morphologyEx(binv, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def normalize_size(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if max(h, w) < 900:  # upscale tiny images so glyphs are large enough for OCR
        return cv2.resize(img, None, fx=900 / max(h, w), fy=900 / max(h, w), interpolation=cv2.INTER_CUBIC)
    return img


# ---------------------------------------------------------------- pipeline

ENHANCE_STEPS = ("despike", "denoise", "deblur", "clahe")
_warned_missing_thresholds = False


def preprocess(img: np.ndarray, max_side: int = 1600, mode: str | None = None, thresholds=None,
               disable: tuple[str, ...] | None = None) -> PreprocessResult:
    """Preprocess one page for OCR.

    mode "v2" (default): calibrated DIQA gate - measure noise -> denoise -> measure blur on the denoised page -> deblur, then
    decide accept / enhance / retake. mode "legacy": the original hard-coded gates, kept so the pre-DIQA evaluation is
    reproducible (also the automatic fallback if no calibrated thresholds file exists).
    `disable` removes steps (perspective, deskew, despike, denoise, deblur, clahe) for ablation studies; defaults to
    DOCUSIGHT_PREPROCESS_DISABLE.
    """
    global _warned_missing_thresholds
    from .config import settings

    mode = mode or settings.diqa_mode
    disable = tuple(settings.preprocess_disable if disable is None else disable)
    if mode == "legacy":
        return _preprocess_legacy(img, max_side, disable)
    th = thresholds or Q.load_thresholds(settings.quality_thresholds_path or None, dict(settings.quality_overrides))
    if th is None:
        if not _warned_missing_thresholds:
            import logging

            logging.getLogger(__name__).warning("quality_thresholds.json not found - falling back to the legacy preprocessing gates. "
                                                "Run scripts/quality_vs_cer.py to calibrate.")
            _warned_missing_thresholds = True
        return _preprocess_legacy(img, max_side, disable)
    return _preprocess_diqa(img, max_side, th, disable)


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _finish(gray: np.ndarray, steps: list[str], angle: float, quality: dict, **extra) -> PreprocessResult:
    binary = binarize(gray)
    steps.append("adaptive_threshold+morph_open")
    return PreprocessResult(image=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), gray=gray, binary=binary, steps=steps, skew_angle=angle,
                            quality=quality, **extra)


def _geometry(img: np.ndarray, steps: list[str], disable: tuple[str, ...]) -> tuple[np.ndarray, float]:
    gray = _to_gray(img)
    quad = None if "perspective" in disable else find_document_quad(gray)
    if quad is not None:
        img = perspective_correct(img, quad)
        steps.append("perspective_correct")
    angle = 0.0
    if "deskew" not in disable:
        angle = estimate_skew(_to_gray(img))
        if abs(angle) >= 0.3:
            img = rotate(img, angle)
            steps.append(f"deskew({angle:+.1f}deg)")
    return img, angle


def _preprocess_legacy(img: np.ndarray, max_side: int, disable: tuple[str, ...]) -> PreprocessResult:
    """The original gates: noise > 4 -> denoise, blur < 300 -> deblur, contrast < 150 -> CLAHE (all measured before enhancement)."""
    steps: list[str] = []
    img = normalize_size(img, max_side)
    img, angle = _geometry(img, steps, disable)
    gray = _to_gray(img)
    quality = {"blur": blur_score(gray), "noise": noise_score(gray), "contrast": contrast_score(gray)}
    if quality["noise"] > 4.0 and "denoise" not in disable:
        gray = denoise(gray, quality["noise"])
        steps.append("denoise")
    if quality["blur"] < 300 and "deblur" not in disable:
        gray = deblur(gray)
        steps.append("deblur")
    if quality["contrast"] < 150 and "clahe" not in disable:
        gray = enhance_contrast(gray)
        steps.append("clahe")
    return _finish(gray, steps, angle, quality)


def _preprocess_diqa(img: np.ndarray, max_side: int, th, disable: tuple[str, ...]) -> PreprocessResult:
    """Calibrated gate. Measurement points are chosen so each estimator sees the signal it was validated on:

    * JPEG blockiness on the *decoded input* (any resampling destroys the 8x8 grid);
    * noise sigma, impulse fraction and periodic peak after size normalisation but *before* rotation / perspective warp, which
      interpolate and would smooth the noise;
    * blur on the page that will be OCR'd, *after* denoising (noise inflates gradient-based sharpness);
    * exposure and text height on the geometrically corrected page (the photo background is cropped away).
    """
    steps: list[str] = []
    raw_gray = _to_gray(img)
    blockiness = Q.jpeg_blockiness(raw_gray)
    raw_w = raw_gray.shape[1]
    img = normalize_size(img, max_side)
    scale = img.shape[1] / raw_w
    gray = _to_gray(img)

    flat = Q.flat_tiles(gray)
    noise = Q.noise_sigma(gray, flat)
    impulses, n_flat_px = Q.impulse_fraction(flat, gray.shape)
    hints = Q.NoiseHints(impulse_fraction=impulses, jpeg_blockiness=blockiness, periodic_peak=Q.periodic_peak(gray, flat),
                         n_flat_pixels=n_flat_px)

    if th.impulse_enhance is not None and impulses > th.impulse_enhance and "despike" not in disable:
        gray = Q.despike(gray, noise.sigma)[1]  # before any resampling, while impulses are still isolated
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        steps.append("despike")

    img, angle = _geometry(img, steps, disable)
    gray = _to_gray(img)
    quality = {"blur": blur_score(gray), "noise": noise_score(gray), "contrast": contrast_score(gray)}  # legacy values, for comparison

    if th.noise_enhance is not None and noise.measurable and noise.sigma > th.noise_enhance and "denoise" not in disable:
        gray = denoise(gray, noise.sigma)
        steps.append("denoise")
    blur = Q.blur_metrics(gray)  # measured AFTER denoising
    if th.blur_enhance is not None and not np.isnan(blur.edge_width) and blur.edge_width > th.blur_enhance and "deblur" not in disable:
        gray = deblur(gray)
        steps.append("deblur")
    exposure = Q.exposure_metrics(gray)
    text_h = Q.text_height(gray).text_height_px / max(1.0, scale)  # effective (native) glyph height: up-scaling adds no information
    if th.clahe_contrast_below is not None and exposure.contrast < th.clahe_contrast_below and "clahe" not in disable:
        gray = enhance_contrast(gray)
        steps.append("clahe")

    decision = Q.decide(noise, hints, blur, exposure, text_h, th, applied=[s for s in steps if s in ENHANCE_STEPS])
    quality.update({"noise_sigma": noise.sigma, "edge_width": blur.edge_width})
    qinfo = {
        "decision": decision.decision, "reasons": decision.reasons, "steps_applied": decision.enhance,
        "scores": {"noise_sigma": noise.sigma, "noise_measurable": float(noise.measurable), "edge_width": None if np.isnan(blur.edge_width) else blur.edge_width,
                   "tenengrad_norm": blur.tenengrad_norm, "text_height_px": text_h, "paper_level": exposure.paper_level,
                   "illumination_unevenness": exposure.unevenness, "glare_fraction": exposure.glare_fraction, "contrast": exposure.contrast,
                   "clipped_highlights": exposure.clipped_high},
        "noise_types": Q.suspected_noise_types(noise.sigma if noise.measurable else 0.0, hints, th),
        "noise_hints": {"impulse_fraction": hints.impulse_fraction, "jpeg_blockiness": hints.jpeg_blockiness, "periodic_peak": hints.periodic_peak},
        "thresholds_source": th.source,
    }
    return _finish(gray, steps, angle, quality, qinfo=qinfo)
