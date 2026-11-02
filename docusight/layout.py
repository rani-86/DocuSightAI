"""Geometric layout analysis: OCR tokens -> lines -> segments in reading order."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .schemas import BBox, OCRToken, union_bbox


@dataclass
class Segment:
    """A run of horizontally adjacent tokens on one line (a phrase or a cell)."""

    tokens: list[OCRToken]

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    @property
    def bbox(self) -> BBox:
        return union_bbox([t.bbox for t in self.tokens])  # type: ignore[return-value]

    @property
    def confidence(self) -> float:
        return float(min(t.confidence for t in self.tokens))

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2


@dataclass
class Line:
    segments: list[Segment] = field(default_factory=list)

    @property
    def tokens(self) -> list[OCRToken]:
        return [t for s in self.segments for t in s.tokens]

    @property
    def bbox(self) -> BBox:
        return union_bbox([s.bbox for s in self.segments])  # type: ignore[return-value]

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)


def _vertical_overlap(a: BBox, b: BBox) -> float:
    inter = min(a[3], b[3]) - max(a[1], b[1])
    return inter / max(1, min(a[3] - a[1], b[3] - b[1]))


def group_lines(tokens: list[OCRToken], overlap: float = 0.5) -> list[list[OCRToken]]:
    """Cluster tokens into text lines by vertical overlap, top-to-bottom."""
    lines: list[list[OCRToken]] = []
    for tok in sorted(tokens, key=lambda t: (t.cy, t.bbox[0])):
        for line in reversed(lines[-3:]):
            ref = union_bbox([t.bbox for t in line])
            if _vertical_overlap(tok.bbox, ref) >= overlap:
                line.append(tok)
                break
        else:
            lines.append([tok])
    for line in lines:
        line.sort(key=lambda t: t.bbox[0])
    lines.sort(key=lambda ln: min(t.bbox[1] for t in ln))
    return lines


def build_layout(tokens: list[OCRToken], gap_factor: float | None = 1.6) -> list[Line]:
    """Group tokens into lines, then split each line into segments at wide gaps.

    A gap wider than `gap_factor` x median glyph height marks a column / key-value
    boundary, which is what lets the extractor reason about "value to the right".
    Phrase-level engines (PaddleOCR) already emit one box per phrase, so pass
    `gap_factor=None` to keep every detection as its own segment.
    """
    if not tokens:
        return []
    med_h = float(np.median([t.height for t in tokens])) or 10.0
    out: list[Line] = []
    for toks in group_lines(tokens):
        segs: list[Segment] = [Segment([toks[0]])]
        for tok in toks[1:]:
            gap = tok.bbox[0] - segs[-1].tokens[-1].bbox[2]
            if gap_factor is None or gap > gap_factor * med_h:
                segs.append(Segment([tok]))
            else:
                segs[-1].tokens.append(tok)
        out.append(Line(segs))
    return out


def reading_order_text(lines: list[Line]) -> str:
    return "\n".join(line.text for line in lines)
