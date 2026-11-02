"""OpenCV image preprocessing: quality checks, geometric correction and enhancement.

Every step is conditional on a measured quality signal so clean scans are not
degraded by unnecessary filtering. The list of steps actually applied is
returned so it can be logged with each prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .schemas import BBox


@dataclass
class PreprocessResult:
    image: np.ndarray  # enhanced BGR image fed to OCR
    gray: np.ndarray
    binary: np.ndarray  # inverted binary (text = 255), used for table-line detection
    steps: list[str] = field(default_factory=list)
    skew_angle: float = 0.0
    quality: dict[str, float] = field(default_factory=dict)


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

def preprocess(img: np.ndarray, max_side: int = 1600) -> PreprocessResult:
    steps: list[str] = []
    img = normalize_size(img, max_side)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    quad = find_document_quad(gray)
    if quad is not None:
        img = perspective_correct(img, quad)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        steps.append("perspective_correct")

    angle = estimate_skew(gray)
    if abs(angle) >= 0.3:
        img = rotate(img, angle)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        steps.append(f"deskew({angle:+.1f}deg)")

    quality = {"blur": blur_score(gray), "noise": noise_score(gray), "contrast": contrast_score(gray)}
    if quality["noise"] > 4.0:
        gray = denoise(gray, quality["noise"])
        steps.append("denoise")
    if quality["blur"] < 300:
        gray = deblur(gray)
        steps.append("deblur")
    if quality["contrast"] < 150:
        gray = enhance_contrast(gray)
        steps.append("clahe")

    binary = binarize(gray)
    steps.append("adaptive_threshold+morph_open")
    return PreprocessResult(
        image=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        gray=gray,
        binary=binary,
        steps=steps,
        skew_angle=angle,
        quality=quality,
    )
