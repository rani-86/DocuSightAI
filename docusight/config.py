"""Runtime settings (override with DOCUSIGHT_* environment variables)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(f"DOCUSIGHT_{name}", default)


@dataclass(frozen=True)
class Settings:
    ocr_engine: str = _env("OCR_ENGINE", "paddle")  # paddle | tesseract
    trocr_refine: bool = _env("TROCR_REFINE", "0") == "1"
    trocr_model: str = _env("TROCR_MODEL", "microsoft/trocr-small-printed")
    model_dir: Path = Path(_env("MODEL_DIR", str(ROOT / "models")))
    max_side: int = int(_env("MAX_SIDE", "1600"))

    # Review-routing thresholds
    field_conf_threshold: float = float(_env("FIELD_CONF_THRESHOLD", "0.80"))
    class_conf_threshold: float = float(_env("CLASS_CONF_THRESHOLD", "0.70"))
    amount_tolerance: float = float(_env("AMOUNT_TOLERANCE", "1.0"))

    allowed_mime: tuple[str, ...] = field(
        default=("image/png", "image/jpeg", "image/tiff", "image/bmp", "application/pdf")
    )
    max_upload_mb: int = int(_env("MAX_UPLOAD_MB", "15"))


settings = Settings()
