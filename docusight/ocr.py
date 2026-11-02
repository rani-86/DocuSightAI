"""OCR engine adapters returning word/line tokens with boxes and confidences.

PaddleOCR is the primary engine; Tesseract is a lightweight fallback. TrOCR
(Hugging Face) can optionally re-read low-confidence regions.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Protocol

import numpy as np

from .schemas import OCRToken

log = logging.getLogger(__name__)


class OCREngine(Protocol):
    name: str
    phrase_level: bool  # True if each token is already a phrase rather than a word

    def __call__(self, image: np.ndarray) -> list[OCRToken]: ...


class PaddleEngine:
    """Supports both the PaddleOCR 2.x `.ocr()` and 3.x `.predict()` APIs."""

    name = "paddleocr"
    phrase_level = True

    def __init__(self, lang: str = "en"):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        try:  # 3.x
            self._ocr = PaddleOCR(
                lang=lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,  # geometry is handled by our own preprocessing
                use_textline_orientation=False,
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="en_PP-OCRv5_mobile_rec",
                # Paddle 3.3's oneDNN kernels fail on some PIR attributes on CPU.
                enable_mkldnn=os.getenv("DOCUSIGHT_PADDLE_MKLDNN", "0") == "1",
            )
            self._v3 = True
        except (TypeError, ValueError):  # 2.x
            self._ocr = PaddleOCR(lang=lang, use_angle_cls=False, show_log=False)
            self._v3 = False

    def __call__(self, image: np.ndarray) -> list[OCRToken]:
        tokens: list[OCRToken] = []
        if self._v3:
            for res in self._ocr.predict(image):
                data = res.json["res"] if hasattr(res, "json") else res
                for text, score, poly in zip(data["rec_texts"], data["rec_scores"], data["rec_polys"]):
                    tokens.append(_token(text, score, poly))
        else:
            for line in (self._ocr.ocr(image, cls=False) or [[]])[0] or []:
                poly, (text, score) = line
                tokens.append(_token(text, score, poly))
        return [t for t in tokens if t.text.strip()]


class TesseractEngine:
    name = "tesseract"
    phrase_level = False

    def __init__(self):
        import pytesseract

        self._pt = pytesseract

    def __call__(self, image: np.ndarray) -> list[OCRToken]:
        d = self._pt.image_to_data(image, output_type=self._pt.Output.DICT, config="--psm 11")
        out = []
        for i, text in enumerate(d["text"]):
            conf = float(d["conf"][i])
            if text.strip() and conf >= 0:
                x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
                out.append(OCRToken(text=text, bbox=(x, y, x + w, y + h), confidence=conf / 100))
        return out


class TrOCRRefiner:
    """Re-recognises a cropped text line with a Hugging Face TrOCR model."""

    def __init__(self, model_name: str):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self._torch = torch
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).eval()

    def __call__(self, crop: np.ndarray) -> tuple[str, float]:
        from PIL import Image

        pixel_values = self.processor(images=Image.fromarray(crop[..., ::-1]), return_tensors="pt").pixel_values
        with self._torch.inference_mode():
            out = self.model.generate(pixel_values, max_new_tokens=48, output_scores=True, return_dict_in_generate=True)
        text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
        probs = [s.softmax(-1).max().item() for s in out.scores]
        return text.strip(), float(np.mean(probs)) if probs else 0.0


def _token(text: str, score: float, poly) -> OCRToken:
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return OCRToken(text=str(text).strip(), bbox=(int(x0), int(y0), int(x1), int(y1)), confidence=float(score))


@lru_cache(maxsize=4)
def get_engine(name: str = "paddle") -> OCREngine:
    if name == "paddle":
        return PaddleEngine()
    if name == "tesseract":
        return TesseractEngine()
    raise ValueError(f"Unknown OCR engine: {name}")


@lru_cache(maxsize=1)
def get_refiner(model_name: str) -> TrOCRRefiner:
    return TrOCRRefiner(model_name)
