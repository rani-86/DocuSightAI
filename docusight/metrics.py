"""Evaluation metrics: CER, field-level accuracy, table accuracy, classification F1."""
from __future__ import annotations

import re
from typing import Optional

from rapidfuzz.distance import Levenshtein


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate = edit distance / reference length."""
    ref, hyp = _norm_ws(reference), _norm_ws(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def cer_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    """(edits, reference chars) so CER can be micro-averaged over a corpus."""
    ref, hyp = _norm_ws(reference), _norm_ws(hypothesis)
    return Levenshtein.distance(ref, hyp), len(ref)


def field_match(kind: str, gt: Optional[str], pred: Optional[str]) -> bool:
    if gt is None or pred is None:
        return gt == pred
    if kind == "text":
        return _norm_ws(gt).lower() == _norm_ws(pred).lower()
    return gt == pred


def table_scores(gt_rows: list[dict], pred_rows: list[dict], columns: list[str]) -> dict[str, float]:
    """Row-aligned table metrics: exact-row accuracy and cell accuracy.

    Rows are aligned by position; missing/extra rows count as fully wrong, so
    over- and under-segmentation are both penalised.
    """
    n = max(len(gt_rows), len(pred_rows))
    if n == 0:
        return {"row_exact": 1.0, "cell_acc": 1.0, "row_count_ok": 1.0}
    exact = cells_ok = 0
    for i in range(n):
        g = gt_rows[i] if i < len(gt_rows) else {}
        p = pred_rows[i] if i < len(pred_rows) else {}
        hits = [field_match("text", g.get(c, ""), p.get(c, "")) if g and p else False for c in columns]
        cells_ok += sum(hits)
        exact += all(hits)
    return {"row_exact": exact / n, "cell_acc": cells_ok / (n * len(columns)), "row_count_ok": float(len(gt_rows) == len(pred_rows))}
