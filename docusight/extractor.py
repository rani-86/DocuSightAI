"""Layout-aware key-value extraction.

For each schema field we locate its label on the page (fuzzy match against
aliases) and read the value from the geometrically most plausible place:
the same segment after a colon, the nearest segment to the right on the same
line, or the segment directly below. Every value keeps the bounding box of the
OCR tokens it came from, so reviewers can jump straight to the source region.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from rapidfuzz import fuzz

from .layout import Line, Segment
from .normalize import normalize
from .schemas import BBox, ExtractedField, OCRToken
from .templates import DocSchema, FieldSpec

LABEL_MIN_SCORE = 86
_DECOR_RE = re.compile(r"\(.*?\)|@?\s*\d+(?:\.\d+)?\s*%|[#*]")


def _clean_label(s: str) -> str:
    s = _DECOR_RE.sub(" ", s.lower())
    return re.sub(r"[^a-z/' ]+", " ", s).strip(" '")


def label_score(text: str, aliases: Iterable[str]) -> float:
    t = re.sub(r"\s+", " ", _clean_label(text))
    if not t:
        return 0.0
    return max(fuzz.ratio(t, a) for a in aliases)


@dataclass
class _LabelHit:
    line_idx: int
    seg_idx: int
    score: float
    remainder: str  # text in the same segment after the label
    remainder_tokens: list[OCRToken]


def _find_labels(lines: list[Line], spec: FieldSpec, rivals: list[str]) -> list[_LabelHit]:
    """Find label occurrences; a segment that matches another field's alias
    better (e.g. "Closing Balance" vs "opening balance") is left to that field."""
    hits: list[_LabelHit] = []
    for li, line in enumerate(lines):
        for si, seg in enumerate(line.segments):
            text = seg.text
            if ":" in text:
                label, _, rest = text.partition(":")
                s = label_score(label, spec.labels)
                if s >= LABEL_MIN_SCORE and s >= label_score(label, rivals):
                    rest_toks = _tokens_after_colon(seg)
                    hits.append(_LabelHit(li, si, s, rest.strip(), rest_toks))
                    continue
            s = label_score(text, spec.labels)
            if s >= LABEL_MIN_SCORE and s >= label_score(text, rivals):
                hits.append(_LabelHit(li, si, s, "", []))
                continue
            # Word-level engines can merge "IFSC Code HDFC0001234" into one segment.
            low = text.lower()
            for alias in sorted(spec.labels, key=len, reverse=True):
                if low.startswith(alias + " ") and len(text) > len(alias) + 1:
                    rest = text[len(alias):].strip()
                    hits.append(_LabelHit(li, si, 90.0, rest, seg.tokens[len(alias.split()):]))
                    break
    return hits


def _tokens_after_colon(seg: Segment) -> list[OCRToken]:
    for i, tok in enumerate(seg.tokens):
        if ":" in tok.text:
            before, _, after = tok.text.partition(":")
            rest = seg.tokens[i + 1:]
            if not after.strip():
                return rest
            # Phrase-level OCR returns "Label: value" as one box; estimate the
            # value's sub-region from character offsets so the source bbox is tight.
            x0, y0, x1, y1 = tok.bbox
            start = len(before) + 1 + (len(after) - len(after.lstrip()))
            cut = x0 + int((x1 - x0) * start / max(1, len(tok.text)))
            sub = OCRToken(text=after.strip(), bbox=(cut, y0, x1, y1), confidence=tok.confidence)
            return [sub] + rest
    return []


def _is_label(text: str, all_aliases: list[str]) -> bool:
    head = text.split(":", 1)[0]
    return label_score(head, all_aliases) >= LABEL_MIN_SCORE


def _right_of(lines: list[Line], hit: _LabelHit) -> Optional[Segment]:
    segs = lines[hit.line_idx].segments
    return segs[hit.seg_idx + 1] if hit.seg_idx + 1 < len(segs) else None


def _below(lines: list[Line], hit: _LabelHit) -> Optional[Segment]:
    label = lines[hit.line_idx].segments[hit.seg_idx].bbox
    lh = label[3] - label[1]
    for line in lines[hit.line_idx + 1: hit.line_idx + 3]:
        if line.bbox[1] - label[3] > 2.5 * lh:
            break
        for seg in line.segments:
            b = seg.bbox
            overlap = min(b[2], label[2]) - max(b[0], label[0])
            if overlap > 0.3 * min(b[2] - b[0], label[2] - label[0]) or abs(b[0] - label[0]) < lh:
                return seg
    return None


def _bbox(tokens: list[OCRToken]) -> Optional[BBox]:
    from .schemas import union_bbox

    return union_bbox([t.bbox for t in tokens])


def extract_field(lines: list[Line], spec: FieldSpec, all_aliases: list[str], rivals: list[str]) -> ExtractedField:
    best = ExtractedField(name=spec.name)
    for hit in _find_labels(lines, spec, rivals):
        label_conf = hit.score / 100.0
        candidates: list[tuple[str, str, list[OCRToken], float]] = []
        if hit.remainder:
            conf = min((t.confidence for t in hit.remainder_tokens), default=lines[hit.line_idx].segments[hit.seg_idx].confidence)
            candidates.append(("same_segment", hit.remainder, hit.remainder_tokens or lines[hit.line_idx].segments[hit.seg_idx].tokens, conf))
        for method, seg in (("right", _right_of(lines, hit)), ("below", _below(lines, hit))):
            if seg is not None and not _is_label(seg.text, all_aliases):
                candidates.append((method, seg.text, seg.tokens, seg.confidence))

        for rank, (method, raw, toks, ocr_conf) in enumerate(candidates):
            value = normalize(spec.kind, raw)
            if value is None:
                continue
            fmt_ok = spec.pattern is None or re.fullmatch(spec.pattern, value) is not None
            # Prefer nearer evidence: same segment > right > below.
            conf = label_conf * ocr_conf * (1.0 if fmt_ok else 0.5) * (1.0 - 0.03 * rank)
            if conf > best.confidence:
                best = ExtractedField(name=spec.name, value=value, raw_text=raw, confidence=round(conf, 4),
                                      bbox=_bbox(toks), method=method)
    if best.value is None and spec.pattern:
        best = _regex_fallback(lines, spec)
    return best


def _regex_fallback(lines: list[Line], spec: FieldSpec) -> ExtractedField:
    """Pattern-only search for strongly formatted IDs whose label was not found."""
    for line in lines:
        for seg in line.segments:
            for tok in seg.tokens:
                value = normalize(spec.kind, tok.text)
                if value and re.fullmatch(spec.pattern, value):  # type: ignore[arg-type]
                    return ExtractedField(name=spec.name, value=value, raw_text=tok.text,
                                          confidence=round(0.8 * tok.confidence, 4), bbox=tok.bbox, method="regex")
    return ExtractedField(name=spec.name)


def extract_fields(lines: list[Line], schema: DocSchema) -> dict[str, ExtractedField]:
    all_aliases = [a for f in schema.fields for a in f.labels]
    if schema.table:
        all_aliases += [a for aliases in schema.table.columns.values() for a in aliases]
    out = {}
    for spec in schema.fields:
        rivals = [a for f in schema.fields if f.name != spec.name for a in f.labels if a not in spec.labels]
        out[spec.name] = extract_field(lines, spec, all_aliases, rivals)
    return out
