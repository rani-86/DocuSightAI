from datetime import date

from docusight.metrics import cer, table_scores
from docusight.schemas import ExtractedField, Table, TableRow
from docusight.validation import review_decision, validate


def F(name, value, conf=0.99, bbox=(0, 0, 10, 10)):
    return ExtractedField(name=name, value=value, confidence=conf, bbox=bbox, method="right")


def _invoice(total="14868.00", gstin="27ABCPE1234F1Z5", conf=0.99):
    fields = {"invoice_number": F("invoice_number", "INV-1"), "invoice_date": F("invoice_date", "2025-03-14"),
              "vendor_name": F("vendor_name", "Apex Traders"), "gstin": F("gstin", gstin, conf),
              "subtotal": F("subtotal", "12600.00"), "tax_amount": F("tax_amount", "2268.00"),
              "total_amount": F("total_amount", total)}
    table = Table(columns=["item", "qty", "unit_price", "amount"], strategy="ruled", rows=[
        TableRow(cells={"item": "Chair", "qty": "2.00", "unit_price": "4500.00", "amount": "9000.00"}, confidence=0.99),
        TableRow(cells={"item": "Lamp", "qty": "3.00", "unit_price": "1200.00", "amount": "3600.00"}, confidence=0.99)])
    return fields, [table]


def test_clean_invoice_passes_straight_through():
    fields, tables = _invoice()
    issues = validate("invoice", 0.99, fields, tables, today=date(2026, 1, 1))
    assert issues == []
    assert review_decision(issues) == (False, [])


def test_total_mismatch_is_flagged_with_source_region():
    fields, tables = _invoice(total="15868.00")
    issues = validate("invoice", 0.99, fields, tables, today=date(2026, 1, 1))
    inc = [i for i in issues if i.code == "INCONSISTENT"]
    assert inc and inc[0].field == "total_amount" and inc[0].bbox == (0, 0, 10, 10)


def test_missing_low_conf_and_format_issues():
    fields, tables = _invoice(gstin="27ABCPE1234F1X5")
    fields["vendor_name"] = ExtractedField(name="vendor_name")
    fields["invoice_number"] = F("invoice_number", "INV-1", conf=0.4)
    codes = {(i.code, i.field) for i in validate("invoice", 0.5, fields, tables, today=date(2026, 1, 1))}
    assert ("MISSING_REQUIRED", "vendor_name") in codes
    assert ("FORMAT_INVALID", "gstin") in codes
    assert ("LOW_CONFIDENCE", "invoice_number") in codes
    assert ("LOW_CLASS_CONFIDENCE", None) in codes


def test_bank_balance_reconciliation():
    fields = {"opening_balance": F("opening_balance", "1000.00"), "closing_balance": F("closing_balance", "1400.00")}
    rows = [TableRow(cells={"date": "2025-01-02", "description": "x", "debit": "", "credit": "500.00", "balance": "1500.00"},
                     confidence=0.99),
            TableRow(cells={"date": "2025-01-03", "description": "y", "debit": "100.00", "credit": "", "balance": "1400.00"},
                     confidence=0.99)]
    t = Table(columns=["date", "description", "debit", "credit", "balance"], rows=rows, strategy="header_anchored")
    codes = [i.code for i in validate("bank_statement", 0.99, fields, [t])]
    assert "INCONSISTENT" not in codes
    fields["closing_balance"] = F("closing_balance", "1900.00")
    assert "INCONSISTENT" in [i.code for i in validate("bank_statement", 0.99, fields, [t])]


def test_metrics():
    assert cer("HDFC0001234", "HDFC0001234") == 0
    assert abs(cer("ABCDE", "ABCXE") - 0.2) < 1e-9
    s = table_scores([{"a": "1"}, {"a": "2"}], [{"a": "1"}], ["a"])
    assert s["row_exact"] == 0.5 and s["row_count_ok"] == 0.0
