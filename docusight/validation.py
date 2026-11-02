"""Confidence-based validation and cross-field consistency checks.

Produces `ValidationIssue`s (each carrying the source bbox where possible) and
decides whether a document must be routed to human review.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

from .config import settings
from .normalize import age_on, parse_date, to_float
from .schemas import BBox, ExtractedField, Severity, Table, ValidationIssue
from .templates import SCHEMAS


def _issue(code: str, sev: Severity, msg: str, field: Optional[str] = None, bbox: Optional[BBox] = None) -> ValidationIssue:
    return ValidationIssue(code=code, severity=sev, message=msg, field=field, bbox=bbox)


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def _col_sum(table: Table, col: str) -> float:
    return sum(to_float(r.cells.get(col)) or 0.0 for r in table.rows)


def validate(doc_type: str, doc_conf: float, fields: dict[str, ExtractedField], tables: list[Table],
             today: Optional[date] = None) -> list[ValidationIssue]:
    schema = SCHEMAS[doc_type]
    tol = settings.amount_tolerance
    issues: list[ValidationIssue] = []

    if doc_conf < settings.class_conf_threshold:
        issues.append(_issue("LOW_CLASS_CONFIDENCE", Severity.warning,
                             f"Document type '{doc_type}' predicted with confidence {doc_conf:.2f}"))

    for spec in schema.fields:
        f = fields.get(spec.name)
        if f is None or f.value is None:
            if spec.required:
                issues.append(_issue("MISSING_REQUIRED", Severity.error, f"Required field '{spec.name}' not found", spec.name))
            continue
        if spec.pattern and not re.fullmatch(spec.pattern, f.value):
            issues.append(_issue("FORMAT_INVALID", Severity.error,
                                 f"'{spec.name}' value '{f.value}' does not match expected format", spec.name, f.bbox))
        elif f.confidence < settings.field_conf_threshold:
            issues.append(_issue("LOW_CONFIDENCE", Severity.warning,
                                 f"'{spec.name}' extracted with confidence {f.confidence:.2f}", spec.name, f.bbox))
        if spec.kind == "date":
            d = parse_date(f.value)
            if d and d > (today or date.today()):
                issues.append(_issue("INCONSISTENT", Severity.error, f"'{spec.name}' is in the future", spec.name, f.bbox))

    for t in tables:
        numeric = schema.table.numeric if schema.table else ()
        for i, row in enumerate(t.rows):
            empty = [c for c in numeric if not row.cells.get(c)]
            # Bank rows legitimately leave one of debit/credit empty.
            allowed = {"debit", "credit"} if doc_type == "bank_statement" else set()
            if set(empty) - allowed or (allowed and allowed <= set(empty)):
                issues.append(_issue("MISSING_REQUIRED", Severity.warning,
                                     f"Table row {i + 1} is missing {', '.join(empty)}", f"table[{i}]", row.bbox))
            elif row.confidence < settings.field_conf_threshold:
                issues.append(_issue("LOW_CONFIDENCE", Severity.warning,
                                     f"Table row {i + 1} has low OCR confidence ({row.confidence:.2f})", f"table[{i}]", row.bbox))

    v = {k: to_float(f.value) for k, f in fields.items()}
    table = tables[0] if tables else None
    check = _CHECKS.get(doc_type)
    if check:
        issues.extend(check(fields, v, table, tol, today or date.today()))
    return issues


def _bank(fields, v, table, tol, today):
    out = []
    if table and v.get("opening_balance") is not None and v.get("closing_balance") is not None:
        expected = v["opening_balance"] + _col_sum(table, "credit") - _col_sum(table, "debit")
        if not _close(expected, v["closing_balance"], tol):
            out.append(_issue("INCONSISTENT", Severity.error,
                              f"Opening + credits - debits = {expected:.2f} but closing balance is {v['closing_balance']:.2f}",
                              "closing_balance", fields["closing_balance"].bbox))
        running = v["opening_balance"]
        for i, row in enumerate(table.rows):
            bal = to_float(row.cells.get("balance"))
            running += (to_float(row.cells.get("credit")) or 0) - (to_float(row.cells.get("debit")) or 0)
            if bal is not None and not _close(running, bal, tol):
                out.append(_issue("INCONSISTENT", Severity.warning,
                                  f"Row {i + 1}: running balance {running:.2f} != printed balance {bal:.2f}", f"table[{i}]", row.bbox))
                running = bal  # resynchronise so one bad row is not reported repeatedly
    period = fields.get("statement_period")
    if table and period and period.value and ".." in period.value:
        start, end = (parse_date(p) for p in period.value.split(".."))
        for i, row in enumerate(table.rows):
            d = parse_date(row.cells.get("date", ""))
            if d and start and end and not (start <= d <= end):
                out.append(_issue("INCONSISTENT", Severity.warning, f"Row {i + 1} date {d} outside statement period",
                                  f"table[{i}]", row.bbox))
    return out


def _invoice(fields, v, table, tol, today):
    out = []
    if table:
        for i, row in enumerate(table.rows):
            q, p, a = (to_float(row.cells.get(c)) for c in ("qty", "unit_price", "amount"))
            if None not in (q, p, a) and not _close(q * p, a, tol):
                out.append(_issue("INCONSISTENT", Severity.warning, f"Row {i + 1}: qty x price = {q * p:.2f} != {a:.2f}",
                                  f"table[{i}]", row.bbox))
        if v.get("subtotal") is not None and not _close(_col_sum(table, "amount"), v["subtotal"], tol):
            out.append(_issue("INCONSISTENT", Severity.error,
                              f"Line items sum to {_col_sum(table, 'amount'):.2f} but subtotal is {v['subtotal']:.2f}",
                              "subtotal", fields["subtotal"].bbox))
    if None not in (v.get("subtotal"), v.get("tax_amount"), v.get("total_amount")):
        if not _close(v["subtotal"] + v["tax_amount"], v["total_amount"], tol):
            out.append(_issue("INCONSISTENT", Severity.error, "subtotal + tax != total", "total_amount", fields["total_amount"].bbox))
    return out


def _payslip(fields, v, table, tol, today):
    out = []
    if table and v.get("gross_earnings") is not None and not _close(_col_sum(table, "amount"), v["gross_earnings"], tol):
        out.append(_issue("INCONSISTENT", Severity.error,
                          f"Earnings components sum to {_col_sum(table, 'amount'):.2f} but gross is {v['gross_earnings']:.2f}",
                          "gross_earnings", fields["gross_earnings"].bbox))
    if None not in (v.get("gross_earnings"), v.get("total_deductions"), v.get("net_pay")):
        if not _close(v["gross_earnings"] - v["total_deductions"], v["net_pay"], tol):
            out.append(_issue("INCONSISTENT", Severity.error, "gross - deductions != net pay", "net_pay", fields["net_pay"].bbox))
    return out


def _kyc(fields, v, table, tol, today):
    f = fields.get("date_of_birth")
    d = parse_date(f.value) if f and f.value else None
    if d and age_on(d, today) < 18:
        return [_issue("INCONSISTENT", Severity.warning, "Applicant is under 18", "date_of_birth", f.bbox)]
    return []


_CHECKS = {"bank_statement": _bank, "invoice": _invoice, "payslip": _payslip, "kyc_id": _kyc}


def review_decision(issues: list[ValidationIssue]) -> tuple[bool, list[str]]:
    reasons = sorted({i.code for i in issues})
    return bool(issues), reasons
