import numpy as np

from conftest import tok
from docusight.extractor import extract_fields
from docusight.layout import build_layout
from docusight.tables import extract_table
from docusight.templates import SCHEMAS


def test_invoice_fields_same_segment_and_right(invoice_tokens):
    fields = extract_fields(build_layout(invoice_tokens, None), SCHEMAS["invoice"])
    assert fields["invoice_number"].value == "INV-2025-1042"
    assert fields["invoice_number"].method == "same_segment"
    assert fields["invoice_date"].value == "2025-03-14"
    assert fields["vendor_name"].value == "Apex Traders" and fields["vendor_name"].method == "right"
    assert fields["gstin"].value == "27ABCPE1234F1Z5"
    assert fields["tax_amount"].value == "2268.00"  # "GST (18%)" label, not GSTIN
    assert fields["total_amount"].value == "14868.00"


def test_source_region_is_value_not_label(invoice_tokens):
    fields = extract_fields(build_layout(invoice_tokens, None), SCHEMAS["invoice"])
    x0, _, x1, _ = fields["invoice_number"].bbox
    assert x0 > 560 + 10 * len("Invoice No:") - 15  # bbox starts after the label text


def test_value_below_label():
    toks = [tok("Name", 230, 110), tok("Asha Rao", 230, 135), tok("Father's Name", 230, 175), tok("Mohan Rao", 230, 200),
            tok("Date of Birth", 230, 240), tok("09/11/1990", 230, 265), tok("Permanent Account Number", 230, 305),
            tok("ABCPR1234K", 230, 330)]
    f = extract_fields(build_layout(toks, None), SCHEMAS["kyc_id"])
    assert f["name"].value == "Asha Rao" and f["name"].method == "below"
    assert f["father_name"].value == "Mohan Rao"
    assert f["date_of_birth"].value == "1990-11-09"
    assert f["id_number"].value == "ABCPR1234K"


def test_opening_vs_closing_balance_not_confused():
    toks = [tok("Opening Balance: 10,000.00", 60, 100), tok("Closing Balance: 12,500.00", 60, 400)]
    f = extract_fields(build_layout(toks, None), SCHEMAS["bank_statement"])
    assert f["opening_balance"].value == "10000.00"
    assert f["closing_balance"].value == "12500.00"


def test_word_level_tokens_are_merged_into_segments():
    toks = [tok("IFSC", 60, 100, 100), tok("Code", 106, 100, 150), tok("HDFC0001234", 300, 100, 420)]
    f = extract_fields(build_layout(toks, gap_factor=1.6), SCHEMAS["bank_statement"])
    assert f["ifsc"].value == "HDFC0001234" and f["ifsc"].method == "right"


def test_header_anchored_table(invoice_tokens):
    lines = build_layout(invoice_tokens, None)
    table = extract_table(lines, np.zeros((1300, 1000), np.uint8), SCHEMAS["invoice"].table)
    assert table is not None and table.strategy == "header_anchored"
    assert [r.cells for r in table.rows] == [
        {"item": "Office Chair", "qty": "2.00", "unit_price": "4500.00", "amount": "9000.00"},
        {"item": "Desk Lamp", "qty": "3.00", "unit_price": "1200.00", "amount": "3600.00"},
    ]


def test_wrapped_row_is_merged():
    toks = [tok("Date", 60, 300), tok("Description", 200, 300), tok("Debit", 560, 300), tok("Credit", 700, 300),
            tok("Balance", 850, 300),
            tok("03-02-2023", 60, 340), tok("RTGS/1234/Vendor Payment", 200, 340), tok("1,532.28", 700, 340),
            tok("2,49,821.75", 850, 340),
            tok("Received for Consulting", 200, 365),
            tok("04-02-2023", 60, 400), tok("UPI/99/Grocery", 200, 400), tok("500.00", 560, 400), tok("2,49,321.75", 850, 400),
            tok("Closing Balance: 2,49,321.75", 60, 450)]
    table = extract_table(build_layout(toks, None), np.zeros((800, 1000), np.uint8), SCHEMAS["bank_statement"].table)
    assert len(table.rows) == 2
    assert table.rows[0].cells["description"] == "RTGS/1234/Vendor Payment Received for Consulting"
    assert table.rows[0].cells["credit"] == "1532.28" and table.rows[0].cells["debit"] == ""
    assert table.rows[1].cells["debit"] == "500.00"
