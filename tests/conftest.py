import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import docusight  # noqa: E402,F401  (torch-before-paddle import order on Windows)
import pytest  # noqa: E402

from docusight.schemas import OCRToken  # noqa: E402


def tok(text, x0, y0, x1=None, y1=None, conf=0.99):
    x1 = x1 if x1 is not None else x0 + 10 * len(text)
    y1 = y1 if y1 is not None else y0 + 22
    return OCRToken(text=text, bbox=(x0, y0, x1, y1), confidence=conf)


class FakeEngine:
    name = "fake"
    phrase_level = True

    def __init__(self, tokens):
        self.tokens = tokens

    def __call__(self, image):
        return self.tokens


@pytest.fixture
def invoice_tokens():
    return [
        tok("TAX INVOICE", 700, 50),
        tok("Invoice No: INV-2025-1042", 560, 110),
        tok("Invoice Date: 14/03/2025", 560, 150),
        tok("Seller", 60, 220), tok("Apex Traders", 230, 220),
        tok("GSTIN", 60, 260), tok("27ABCPE1234F1Z5", 230, 260),
        tok("Item", 70, 340), tok("Qty", 560, 340), tok("Unit Price", 660, 340), tok("Amount", 860, 340),
        tok("Office Chair", 70, 380), tok("2", 580, 380), tok("4,500.00", 700, 380), tok("9,000.00", 860, 380),
        tok("Desk Lamp", 70, 420), tok("3", 580, 420), tok("1,200.00", 700, 420), tok("3,600.00", 860, 420),
        tok("Subtotal", 620, 480), tok("12,600.00", 860, 480),
        tok("GST (18%)", 620, 520), tok("2,268.00", 860, 520),
        tok("Grand Total", 620, 560), tok("14,868.00", 860, 560),
    ]
