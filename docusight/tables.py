"""Table detection and structure recognition.

Two complementary strategies feed a single row-assembly routine:

* ruled   - OpenCV morphology isolates horizontal/vertical rules; vertical rule
            positions give exact column boundaries.
* header_anchored - for borderless tables, the header line is found by fuzzy
            matching column aliases and column boundaries are the midpoints
            between header cells.

Rows are then built line-by-line below the header; a line with no value in the
key column (e.g. a wrapped narration) is merged into the previous row.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .extractor import label_score
from .layout import Line, Segment
from .normalize import amount, clean_text, iso_date
from .schemas import BBox, Table, TableRow, union_bbox
from .templates import TableSpec

HEADER_MIN_SCORE = 85


def detect_rules(binary: np.ndarray) -> tuple[list[int], list[int], Optional[BBox]]:
    """Return x positions of vertical rules, y positions of horizontal rules, table bbox."""
    h, w = binary.shape
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 25), 1)))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 50))))
    grid = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], [], None
    x, y, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if cw < 0.3 * w or ch < 0.05 * h:
        return [], [], None
    v_roi = vert[y:y + ch, x:x + cw]
    h_roi = horiz[y:y + ch, x:x + cw]
    xs = _peaks(v_roi.sum(axis=0), min_strength=0.5 * ch * 255) + x
    ys = _peaks(h_roi.sum(axis=1), min_strength=0.5 * cw * 255) + y
    if len(xs) < 3 or len(ys) < 3:
        return [], [], None
    return xs.tolist(), ys.tolist(), (x, y, x + cw, y + ch)


def _peaks(profile: np.ndarray, min_strength: float, merge: int = 6) -> np.ndarray:
    idx = np.where(profile >= min_strength)[0]
    if idx.size == 0:
        return idx
    groups = np.split(idx, np.where(np.diff(idx) > merge)[0] + 1)
    return np.array([int(g.mean()) for g in groups])


def _find_header(lines: list[Line], spec: TableSpec, region: Optional[BBox]) -> Optional[tuple[int, dict[str, Segment]]]:
    best: Optional[tuple[int, dict[str, Segment]]] = None
    for li, line in enumerate(lines):
        if region and not (region[1] - 5 <= line.bbox[1] <= region[3]):
            continue
        matched: dict[str, Segment] = {}
        for seg in line.segments:
            scored = [(label_score(seg.text, aliases), col) for col, aliases in spec.columns.items() if col not in matched]
            if scored:
                score, col = max(scored)
                if score >= HEADER_MIN_SCORE:
                    matched[col] = seg
        if len(matched) >= max(2, len(spec.columns) - 1) and (best is None or len(matched) > len(best[1])):
            best = (li, matched)
    return best


def _column_spans(header: dict[str, Segment], vrules: list[int], region: Optional[BBox], page_w: int) -> dict[str, tuple[int, int]]:
    cols = sorted(header.items(), key=lambda kv: kv[1].cx)
    if vrules:
        spans = {}
        for col, seg in cols:
            left = max([x for x in vrules if x <= seg.cx], default=0)
            right = min([x for x in vrules if x > seg.cx], default=page_w)
            spans[col] = (left, right)
        return spans
    centers = [seg.cx for _, seg in cols]
    bounds = [0] + [int((a + b) / 2) for a, b in zip(centers, centers[1:])] + [page_w]
    return {col: (bounds[i], bounds[i + 1]) for i, (col, _) in enumerate(cols)}


def _assign(seg: Segment, spans: dict[str, tuple[int, int]]) -> Optional[str]:
    x0, _, x1, _ = seg.bbox
    best, best_ov = None, 0
    for col, (a, b) in spans.items():
        ov = min(x1, b) - max(x0, a)
        if ov > best_ov:
            best, best_ov = col, ov
    return best


def _normalize_cell(col: str, text: str, spec: TableSpec) -> str:
    if col in spec.numeric:
        return amount(text) or ""
    if col == "date":
        return iso_date(text) or clean_text(text)
    return clean_text(text)


def extract_table(lines: list[Line], binary: np.ndarray, spec: TableSpec) -> Optional[Table]:
    page_w = binary.shape[1]
    vrules, _, region = detect_rules(binary)
    found = _find_header(lines, spec, region) or (_find_header(lines, spec, None) if region else None)
    if found is None:
        return None
    header_idx, header = found
    if region and not (region[1] - 5 <= lines[header_idx].bbox[1] <= region[3]):
        region, vrules = None, []
    spans = _column_spans(header, vrules, region, page_w)
    columns = list(spec.columns)
    rows: list[TableRow] = []
    raw_rows: list[dict[str, list[Segment]]] = []
    prev_bottom = lines[header_idx].bbox[3]
    line_h = max(8, lines[header_idx].bbox[3] - lines[header_idx].bbox[1])

    for line in lines[header_idx + 1:]:
        if region and line.bbox[1] > region[3] + 2:
            break
        if not region and line.bbox[1] - prev_bottom > 3.5 * line_h:
            break
        first = clean_text(line.segments[0].text).lower()
        if any(first.startswith(w) for w in spec.stop_words):
            break
        cells: dict[str, list[Segment]] = {}
        for seg in line.segments:
            col = _assign(seg, spans)
            if col:
                cells.setdefault(col, []).append(seg)
        if not cells:
            break
        if spec.key_column not in cells and raw_rows:
            for col, segs in cells.items():  # continuation of a wrapped row
                raw_rows[-1].setdefault(col, []).extend(segs)
        else:
            raw_rows.append(cells)
        prev_bottom = line.bbox[3]

    for cells in raw_rows:
        segs = [s for v in cells.values() for s in v]
        values = {c: _normalize_cell(c, " ".join(s.text for s in cells.get(c, [])), spec) for c in columns}
        rows.append(TableRow(
            cells=values,
            bbox=union_bbox([s.bbox for s in segs]),
            confidence=round(min(s.confidence for s in segs), 4),
        ))
    if not rows:
        return None
    table_bbox = union_bbox([lines[header_idx].bbox] + [r.bbox for r in rows if r.bbox])
    return Table(columns=columns, rows=rows, bbox=table_bbox, strategy="ruled" if vrules else "header_anchored")
