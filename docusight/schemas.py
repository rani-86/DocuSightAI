"""Typed contracts shared by the pipeline, the API and the evaluator."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

BBox = tuple[int, int, int, int]  # x0, y0, x1, y1 in preprocessed-image pixels


def union_bbox(boxes: list[BBox]) -> Optional[BBox]:
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


class OCRToken(BaseModel):
    text: str
    bbox: BBox
    confidence: float

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class ExtractedField(BaseModel):
    name: str
    value: Optional[str] = None
    raw_text: Optional[str] = None
    confidence: float = 0.0
    bbox: Optional[BBox] = Field(None, description="Source region the value was read from")
    method: Optional[str] = Field(None, description="same_segment | right | below | regex")


class TableRow(BaseModel):
    cells: dict[str, str]
    bbox: Optional[BBox] = None
    confidence: float = 0.0


class Table(BaseModel):
    columns: list[str]
    rows: list[TableRow]
    bbox: Optional[BBox] = None
    strategy: str = Field(..., description="ruled | header_anchored")


class Severity(str, Enum):
    error = "error"
    warning = "warning"


class ValidationIssue(BaseModel):
    code: str  # MISSING_REQUIRED | LOW_CONFIDENCE | FORMAT_INVALID | INCONSISTENT | LOW_CLASS_CONFIDENCE
    field: Optional[str] = None
    severity: Severity
    message: str
    bbox: Optional[BBox] = None


class ExtractionResult(BaseModel):
    doc_id: str
    doc_type: str
    doc_type_confidence: float
    fields: dict[str, ExtractedField]
    tables: list[Table] = []
    issues: list[ValidationIssue] = []
    needs_review: bool = False
    review_reasons: list[str] = []
    ocr_engine: str
    preprocessing: list[str] = []
    image_size: tuple[int, int] = (0, 0)
    timings_ms: dict[str, float] = {}
