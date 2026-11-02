"""Runtime settings (override with DOCUSIGHT_* environment variables)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(f"DOCUSIGHT_{name}", default)


PREPROCESS_STEPS = ("perspective", "deskew", "despike", "denoise", "deblur", "clahe")


def _quality_overrides() -> dict:
    """DOCUSIGHT_Q_<THRESHOLD_FIELD>=value  ->  {"threshold_field": value}; e.g. DOCUSIGHT_Q_NOISE_ENHANCE=3.0 (use 'none' to disable a check)."""
    prefix, out = "DOCUSIGHT_Q_", {}
    for k, v in os.environ.items():
        if k.startswith(prefix):
            out[k[len(prefix):].lower()] = None if v.strip().lower() in ("none", "off", "") else float(v)
    return out


def _disabled_steps() -> tuple[str, ...]:
    raw = [s.strip() for s in _env("PREPROCESS_DISABLE", "").split(",") if s.strip()]
    bad = [s for s in raw if s not in PREPROCESS_STEPS]
    if bad:
        raise ValueError(f"DOCUSIGHT_PREPROCESS_DISABLE: unknown step(s) {bad}; choose from {PREPROCESS_STEPS}")
    return tuple(raw)


@dataclass(frozen=True)
class Settings:
    ocr_engine: str = _env("OCR_ENGINE", "paddle")  # paddle | tesseract
    trocr_refine: bool = _env("TROCR_REFINE", "0") == "1"
    trocr_model: str = _env("TROCR_MODEL", "microsoft/trocr-small-printed")
    model_dir: Path = Path(_env("MODEL_DIR", str(ROOT / "models")))
    max_side: int = int(_env("MAX_SIDE", "1600"))

    # Image-quality gate (DIQA). "v2" = calibrated thresholds + accept/enhance/retake; "legacy" = the original hard-coded gates
    # (noise>4, blur<300, contrast<150), kept so the pre-DIQA evaluation stays reproducible. v2 silently falls back to legacy
    # if docusight/quality_thresholds.json does not exist.
    diqa_mode: str = _env("DIQA_MODE", "v2")
    quality_thresholds_path: str = _env("QUALITY_THRESHOLDS", "")
    quality_overrides: tuple = tuple(sorted(_quality_overrides().items(), key=lambda kv: kv[0]))
    preprocess_disable: tuple[str, ...] = _disabled_steps()  # comma list of PREPROCESS_STEPS, for ablations

    # Review-routing thresholds
    field_conf_threshold: float = float(_env("FIELD_CONF_THRESHOLD", "0.80"))
    class_conf_threshold: float = float(_env("CLASS_CONF_THRESHOLD", "0.70"))
    amount_tolerance: float = float(_env("AMOUNT_TOLERANCE", "1.0"))

    allowed_mime: tuple[str, ...] = field(
        default=("image/png", "image/jpeg", "image/tiff", "image/bmp", "application/pdf")
    )
    max_upload_mb: int = int(_env("MAX_UPLOAD_MB", "15"))


settings = Settings()
