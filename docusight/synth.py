"""Synthetic financial-document generator with ground truth.

Renders bank statements, invoices, payslips and PAN-style KYC cards in several
layout variants, then degrades them like real scans/phone photos (skew,
perspective, blur, noise, JPEG, uneven lighting). A fraction of documents get
injected arithmetic inconsistencies or missing fields so the validator's
review routing can be measured against known anomalies.

Train/test splits draw names, companies and items from *disjoint* vocabularies
(entity-level leakage prevention), on top of disjoint random seeds.
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import string
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import DOC_TYPES

FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/truetype/liberation")]
FONT_FILES = [("arial.ttf", "arialbd.ttf"), ("calibri.ttf", "calibrib.ttf"), ("verdana.ttf", "verdanab.ttf"),
              ("times.ttf", "timesbd.ttf"), ("tahoma.ttf", "tahomabd.ttf"),
              ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"), ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf")]

FIRST = ["Aarav", "Vivaan", "Aditya", "Arjun", "Sai", "Reyansh", "Krishna", "Ishaan", "Rohan", "Kabir", "Ananya", "Diya",
         "Priya", "Sneha", "Kavya", "Meera", "Pooja", "Neha", "Riya", "Isha", "Rahul", "Vikram", "Siddharth", "Nikhil",
         "Aditi", "Shreya", "Tanvi", "Lakshmi", "Harish", "Manoj", "Deepak", "Suresh", "Ramesh", "Gaurav", "Varun",
         "Nandini", "Bhavna", "Swati", "Pallavi", "Komal", "Arnav", "Dhruv", "Yash", "Kunal", "Rani", "Anjali", "Sanjay",
         "Farhan", "Imran", "Joseph", "Thomas", "Gurpreet", "Harleen", "Tenzin", "Lalit", "Mohan", "Kiran", "Asha"]
LAST = ["Sharma", "Verma", "Patel", "Reddy", "Nair", "Iyer", "Gupta", "Singh", "Kumar", "Das", "Mehta", "Joshi",
        "Kulkarni", "Rao", "Menon", "Pillai", "Chopra", "Malhotra", "Banerjee", "Mukherjee", "Bhat", "Shetty", "Agarwal",
        "Saxena", "Mishra", "Pandey", "Yadav", "Chauhan", "Desai", "Shah", "Jain", "Kapoor", "Bose", "Ghosh", "Naidu",
        "Hegde", "Kamath", "Sinha", "Tiwari", "Dubey", "Khan", "Fernandes", "DSouza", "Gill", "Sandhu", "Thakur"]
COMPANIES = ["Apex", "Zenith", "Sunrise", "Bluewave", "Orbit", "Vertex", "Nimbus", "Crest", "Silverline", "Greenleaf",
             "Ironclad", "Lotus", "Pinnacle", "Quantum", "Redwood", "Sapphire", "Trident", "Unity", "Vista", "Willow",
             "Horizon", "Kestrel", "Maple", "Northstar", "Oakridge", "Prism", "Radiant", "Summit", "Tidal", "Evergreen"]
SUFFIX = ["Technologies Pvt Ltd", "Traders", "Enterprises", "Solutions LLP", "Industries Ltd", "Retail Pvt Ltd",
          "Logistics", "Systems Pvt Ltd", "Supplies Co"]
ITEMS = ["Office Chair", "LED Monitor 24in", "Wireless Mouse", "Laptop Stand", "Printer Paper A4", "USB-C Hub",
         "Desk Lamp", "Ethernet Cable 5m", "Mechanical Keyboard", "Webcam HD", "Toner Cartridge", "Whiteboard",
         "Filing Cabinet", "Router AC1200", "SSD 1TB", "Headset", "Extension Board", "Projector Screen",
         "Notebook Pack", "Stapler Heavy", "Server Rack Shelf", "UPS 1kVA", "Water Dispenser", "Paper Shredder"]
BANKS = [("State Bank of India", "SBIN"), ("HDFC Bank", "HDFC"), ("ICICI Bank", "ICIC"), ("Axis Bank", "UTIB"),
         ("Kotak Mahindra Bank", "KKBK"), ("Punjab National Bank", "PUNB"), ("Bank of Baroda", "BARB"),
         ("Canara Bank", "CNRB"), ("Yes Bank", "YESB"), ("IDFC First Bank", "IDFB")]
CREDIT_NARR = ["NEFT/{n}/Salary Credit", "Interest Credit", "NEFT/{n}/Freelance Payment", "Cheque Deposit {n}",
               "RTGS/{n}/Vendor Payment Received for Consulting Services", "IMPS/{n}/Refund from Merchant"]
DEBIT_NARR = ["UPI/{n}/Grocery Store", "ATM WDL/{n}/MG Road", "POS/{n}/Fuel Station", "IMPS/{n}/Rent Transfer",
              "ACH/{n}/Mutual Fund SIP", "UPI/{n}/Electricity Bill", "POS/{n}/Online Shopping", "EMI/{n}/Home Loan",
              "UPI/{n}/Mobile Recharge", "Card Annual Fee", "NACH/{n}/Insurance Premium Auto Debit Monthly"]
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_FULL = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]


def _split_vocab(vocab: list[str], split: str) -> list[str]:
    """Deterministic, disjoint halves of a vocabulary for train vs test."""
    keep = [v for v in vocab if (int(hashlib.md5(v.encode()).hexdigest(), 16) % 3 == 0) == (split == "test")]
    return keep or vocab


# ------------------------------------------------------------------ helpers

def inr(x: float) -> str:
    """Indian digit grouping: 12,34,567.89"""
    neg = x < 0
    whole, frac = f"{abs(x):.2f}".split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return ("-" if neg else "") + f"{whole}.{frac}"


def fmt_date(d: date, style: int) -> str:
    return [f"{d.day:02d}-{d.month:02d}-{d.year}", f"{d.day:02d} {MONTH_ABBR[d.month - 1]} {d.year}",
            f"{d.day:02d}/{d.month:02d}/{d.year}"][style % 3]


def rand_pan(rng: random.Random, holder_last: str = "") -> str:
    letters = string.ascii_uppercase
    return ("".join(rng.choice(letters) for _ in range(3)) + "P" + (holder_last[:1].upper() or rng.choice(letters))
            + f"{rng.randint(0, 9999):04d}" + rng.choice(letters))


def rand_gstin(rng: random.Random) -> str:
    return f"{rng.randint(1, 37):02d}" + rand_pan(rng) + rng.choice("123456789") + "Z" + rng.choice(string.ascii_uppercase + string.digits)


@dataclass
class Canvas:
    w: int
    h: int
    font_pair: tuple[Path, Path]
    size: int
    bg: tuple[int, int, int] = (255, 255, 255)
    img: Image.Image = field(init=False)
    draw: ImageDraw.ImageDraw = field(init=False)
    runs: list[tuple[int, int, str]] = field(default_factory=list)  # (x, y, text) for GT reading order

    def __post_init__(self):
        self.img = Image.new("RGB", (self.w, self.h), self.bg)
        self.draw = ImageDraw.Draw(self.img)

    def font(self, size: Optional[int] = None, bold: bool = False) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(str(self.font_pair[1 if bold else 0]), size or self.size)

    def text(self, x: int, y: int, s: str, size: Optional[int] = None, bold: bool = False, anchor: str = "la",
             fill=(20, 20, 20)) -> tuple[int, int, int, int]:
        f = self.font(size, bold)
        self.draw.text((x, y), s, font=f, fill=fill, anchor=anchor)
        box = self.draw.textbbox((x, y), s, font=f, anchor=anchor)
        self.runs.append((box[0], box[1], s))
        return box

    def hline(self, x0: int, x1: int, y: int, width: int = 2):
        self.draw.line([(x0, y), (x1, y)], fill=(40, 40, 40), width=width)

    def vline(self, x: int, y0: int, y1: int, width: int = 2):
        self.draw.line([(x, y0), (x, y1)], fill=(40, 40, 40), width=width)

    def kv(self, x: int, y: int, label: str, value: str, style: str, value_x: Optional[int] = None) -> int:
        """Draw a key-value pair; returns the y for the next row."""
        if style == "colon":
            self.text(x, y, f"{label}: {value}")
            return y + int(self.size * 1.9)
        if style == "right":
            self.text(x, y, label)
            self.text(value_x or x + 260, y, value)
            return y + int(self.size * 1.9)
        self.text(x, y, label, size=int(self.size * 0.85), fill=(90, 90, 90))  # below
        self.text(x, y + int(self.size * 1.15), value, bold=True)
        return y + int(self.size * 2.8)

    def gt_text(self) -> str:
        rows: list[list[tuple[int, int, str]]] = []
        for run in sorted(self.runs, key=lambda r: (r[1], r[0])):
            if rows and abs(rows[-1][0][1] - run[1]) <= self.size * 0.5:
                rows[-1].append(run)
            else:
                rows.append([run])
        return "\n".join(" ".join(r[2] for r in sorted(row)) for row in rows)


def draw_table(c: Canvas, x0: int, y: int, cols: list[tuple[str, int, str]], rows: list[list[str]], ruled: bool,
               wrap: Optional[dict[int, int]] = None) -> int:
    """cols: (header, width, align l|r). Returns y below the table."""
    row_h = int(c.size * 1.9)
    x1 = x0 + sum(w for _, w, _ in cols)
    lefts = np.cumsum([x0] + [w for _, w, _ in cols]).tolist()
    top = y
    if ruled:
        c.hline(x0, x1, y)
    else:
        c.hline(x0, x1, y + row_h - 4, width=1)

    def put(cells: list[str], yy: int, bold=False):
        for (hdr, w, align), left, val in zip(cols, lefts, cells):
            if not val:
                continue
            if align == "r":
                c.text(left + w - 10, yy + 6, val, bold=bold, anchor="ra")
            else:
                c.text(left + 10, yy + 6, val, bold=bold)

    put([h for h, _, _ in cols], y, bold=True)
    y += row_h
    if ruled:
        c.hline(x0, x1, y)
    for i, r in enumerate(rows):
        cells = list(r)
        extra: list[str] = []
        if wrap:
            for ci, limit in wrap.items():
                if len(cells[ci]) > limit:
                    cut = cells[ci].rfind(" ", 0, limit)
                    cut = cut if cut > 0 else limit
                    extra = [""] * len(cells)
                    extra[ci] = cells[ci][cut:].strip()
                    cells[ci] = cells[ci][:cut].strip()
        put(cells, y)
        y += row_h
        if extra:
            put(extra, y - int(row_h * 0.25))
            y += int(row_h * 0.75)
        if ruled:
            c.hline(x0, x1, y, width=1)
    if ruled:
        for left in lefts:
            c.vline(int(left), top, y)
    return y + 12


# ------------------------------------------------------------- generators

class Gen:
    def __init__(self, rng: random.Random, split: str):
        self.rng = rng
        self.first = _split_vocab(FIRST, split)
        self.last = _split_vocab(LAST, split)
        self.companies = _split_vocab(COMPANIES, split)
        self.items = _split_vocab(ITEMS, split)

    def person(self) -> tuple[str, str]:
        last = self.rng.choice(self.last)
        return f"{self.rng.choice(self.first)} {last}", last

    def company(self) -> str:
        return f"{self.rng.choice(self.companies)} {self.rng.choice(SUFFIX)}"

    def canvas(self, w=1000, h=1300, bg=(255, 255, 255)) -> Canvas:
        fonts = [(d / a, d / b) for a, b in FONT_FILES for d in FONT_DIRS if (d / a).exists() and (d / b).exists()]
        return Canvas(w, h, self.rng.choice(fonts), self.rng.randint(17, 21), bg)

    # -------------------------------------------------------------- bank
    def bank_statement(self, variant: int, anomaly: Optional[str]):
        r, c = self.rng, self.canvas()
        bank, code = r.choice(BANKS)
        holder, _ = self.person()
        acct = "".join(r.choice(string.digits) for _ in range(r.choice([11, 12, 14])))
        ifsc = code + "0" + "".join(r.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        start = date(r.randint(2023, 2026), r.randint(1, 12), 1)
        end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        ds = r.randint(0, 2)
        opening = round(r.uniform(2_000, 250_000), 2)

        n = r.randint(6, 11)
        days = sorted(r.sample(range(start.day, end.day + 1), n))
        rows, bal, gt_rows = [], opening, []
        for d in days:
            is_credit = r.random() < 0.4
            narr = r.choice(CREDIT_NARR if is_credit else DEBIT_NARR).format(n=r.randint(100000, 999999))
            amt = round(r.uniform(150, 60_000 if is_credit else min(40_000, max(500, bal * 0.6))), 2)
            bal = round(bal + amt if is_credit else bal - amt, 2)
            txd = start.replace(day=d)
            rows.append([fmt_date(txd, ds), narr, "" if is_credit else inr(amt), inr(amt) if is_credit else "", inr(bal)])
            gt_rows.append({"date": txd.isoformat(), "description": narr, "debit": "" if is_credit else f"{amt:.2f}",
                            "credit": f"{amt:.2f}" if is_credit else "", "balance": f"{bal:.2f}"})
        closing = bal
        printed_closing = round(closing + r.choice([-1, 1]) * r.uniform(500, 5000), 2) if anomaly == "inconsistent" else closing
        missing = r.choice(["ifsc", "statement_period"]) if anomaly == "missing" else None

        c.text(60, 50, bank, size=c.size + 12, bold=True)
        c.text(60, 50 + c.size + 22, "Account Statement", size=c.size + 2)
        period_txt = f"{fmt_date(start, ds)} to {fmt_date(end, ds)}"
        kvs = [("Account Holder", holder), ("Account Number", acct), ("IFSC Code", ifsc), ("Statement Period", period_txt)]
        if variant == 1:
            kvs = [("Customer Name", holder), ("A/C No", acct), ("IFSC", ifsc), ("Period", period_txt)]
        kvs = [kv for kv in kvs if not (missing == "ifsc" and "IFSC" in kv[0]) and not (missing == "statement_period" and "Period" in kv[0])]
        y = 160
        if variant == 2:  # two-column header block
            left, right = kvs[:2], kvs[2:]
            y2 = y
            for k, v in left:
                y = c.kv(60, y, k, v, "colon")
            for k, v in right:
                y2 = c.kv(540, y2, k, v, "colon")
            y = max(y, y2)
        else:
            for k, v in kvs:
                y = c.kv(60, y, k, v, "right" if variant == 1 else "colon", value_x=300)
        y = c.kv(60, y + 10, "Opening Balance", inr(opening), "right" if variant == 1 else "colon", value_x=300) + 10

        hdrs = [["Date", "Description", "Debit", "Credit", "Balance"],
                ["Txn Date", "Narration", "Withdrawal", "Deposit", "Balance"],
                ["Value Date", "Particulars", "Debit", "Credit", "Balance"]][variant]
        widths = [140, 330, 140, 140, 150]
        cols = [(h, w, "r" if i >= 2 else "l") for i, (h, w) in enumerate(zip(hdrs, widths))]
        y = draw_table(c, 50, y, cols, rows, ruled=(variant == 1), wrap={1: 30})
        c.kv(60, y + 10, "Closing Balance", inr(printed_closing), "right" if variant == 1 else "colon", value_x=300)
        c.text(60, c.h - 80, "This is a computer generated statement and does not require a signature.", size=c.size - 4,
               fill=(110, 110, 110))

        fields = {"account_holder": holder, "account_number": acct, "ifsc": ifsc,
                  "statement_period": f"{start.isoformat()}..{end.isoformat()}",
                  "opening_balance": f"{opening:.2f}", "closing_balance": f"{printed_closing:.2f}"}
        if missing:
            fields[missing] = None
        return c, fields, gt_rows

    # ----------------------------------------------------------- invoice
    def invoice(self, variant: int, anomaly: Optional[str]):
        r, c = self.rng, self.canvas()
        vendor, buyer = self.company(), self.company()
        inv_no = f"{r.choice(['INV', 'BL', 'TX'])}-{r.randint(2023, 2026)}-{r.randint(1000, 99999)}"
        inv_date = date(r.randint(2023, 2026), r.randint(1, 12), r.randint(1, 28))
        gst = rand_gstin(r)
        ds = r.randint(0, 2)
        items = r.sample(self.items, min(len(self.items), r.randint(2, 6)))
        rows, gt_rows, subtotal = [], [], 0.0
        for it in items:
            q = r.randint(1, 12)
            p = round(r.uniform(150, 25_000), 2)
            a = round(q * p, 2)
            subtotal = round(subtotal + a, 2)
            rows.append([it, str(q), inr(p), inr(a)])
            gt_rows.append({"item": it, "qty": f"{q:.2f}", "unit_price": f"{p:.2f}", "amount": f"{a:.2f}"})
        tax = round(subtotal * 0.18, 2)
        total = round(subtotal + tax, 2)
        if anomaly == "inconsistent":
            total = round(total + r.choice([-1, 1]) * r.uniform(200, 3000), 2)
        missing = "gstin" if anomaly == "missing" else None

        c.text(c.w - 60, 50, "TAX INVOICE", size=c.size + 14, bold=True, anchor="ra")
        labels = [("Invoice No", "Invoice Date", "Seller", "GSTIN"),
                  ("Invoice Number", "Date of Issue", "Vendor", "GST No"),
                  ("Bill No", "Bill Date", "Sold By", "GSTIN")][variant]
        style = ["colon", "right", "below"][variant]
        y2 = c.kv(560, 110, labels[0], inv_no, "colon")
        y2 = c.kv(560, y2, labels[1], fmt_date(inv_date, ds), "colon")
        y = y2 + 30
        y = c.kv(60, y, labels[2], vendor, style, value_x=230)
        if not missing:
            y = c.kv(60, y, labels[3], gst, style, value_x=230)
        y += 20
        c.text(60, y, f"Bill To: {buyer}")
        y += 60
        hdr = [["Item", "Qty", "Unit Price", "Amount"], ["Description", "Quantity", "Rate", "Line Total"],
               ["Product", "Qty", "Price", "Amount"]][variant]
        cols = [(hdr[0], 430, "l"), (hdr[1], 110, "r"), (hdr[2], 170, "r"), (hdr[3], 170, "r")]
        y = draw_table(c, 60, y, cols, rows, ruled=(variant != 2))
        tl = [("Subtotal", "GST (18%)", "Grand Total"), ("Sub Total", "Tax Amount", "Total Amount"),
              ("Taxable Value", "Total Tax", "Amount Due")][variant]
        for (lab, val) in zip(tl, (subtotal, tax, total)):
            c.text(620, y + 10, lab, bold=lab == tl[2])
            c.text(c.w - 70, y + 10, inr(val), anchor="ra", bold=lab == tl[2])
            y += int(c.size * 1.9)
        c.text(60, c.h - 90, "Thank you for your business.", size=c.size - 3, fill=(110, 110, 110))
        fields = {"invoice_number": inv_no, "invoice_date": inv_date.isoformat(), "vendor_name": vendor,
                  "gstin": None if missing else gst, "subtotal": f"{subtotal:.2f}", "tax_amount": f"{tax:.2f}",
                  "total_amount": f"{total:.2f}"}
        return c, fields, gt_rows

    # ----------------------------------------------------------- payslip
    def payslip(self, variant: int, anomaly: Optional[str]):
        r, c = self.rng, self.canvas()
        company = self.company()
        name, last = self.person()
        emp_id = f"{r.choice(['EMP', 'E', 'ID'])}{r.randint(1000, 99999)}"
        pan = rand_pan(r, last)
        y_, m_ = r.randint(2023, 2026), r.randint(1, 12)
        basic = round(r.uniform(15_000, 120_000), 0)
        comps = [("Basic Salary", basic), ("HRA", round(basic * 0.4, 0)), ("Special Allowance", round(r.uniform(2_000, 40_000), 0)),
                 ("Conveyance", 1600.0)]
        if r.random() < 0.4:
            comps.append(("Performance Bonus", round(r.uniform(1_000, 25_000), 0)))
        gross = sum(v for _, v in comps)
        deds = [("Provident Fund", round(min(basic, 15_000) * 0.12, 0)), ("Professional Tax", 200.0),
                ("Income Tax (TDS)", round(gross * r.uniform(0.0, 0.15), 0))]
        total_ded = sum(v for _, v in deds)
        net = gross - total_ded
        if anomaly == "inconsistent":
            net += r.choice([-1, 1]) * round(r.uniform(300, 4000), 0)
        missing = "pan" if anomaly == "missing" else None

        c.text(c.w // 2, 50, company, size=c.size + 10, bold=True, anchor="ma")
        c.text(c.w // 2, 50 + c.size + 22, "Payslip", size=c.size + 4, anchor="ma")
        labels = [("Employee Name", "Employee ID", "Pay Period", "PAN"),
                  ("Name of Employee", "Emp Code", "Salary Month", "PAN"),
                  ("Employee Name", "Emp ID", "Month", "PAN")][variant]
        period_txt = [f"{MONTH_FULL[m_ - 1]} {y_}", f"{MONTH_ABBR[m_ - 1]}-{y_}", f"{MONTH_FULL[m_ - 1]}, {y_}"][variant]
        style = ["colon", "right", "colon"][variant]
        y = 160
        pairs = [(labels[0], name), (labels[1], emp_id), (labels[2], period_txt)] + ([] if missing else [(labels[3], pan)])
        if variant == 2:
            yl = yr = y
            for i, (k, v) in enumerate(pairs):
                if i % 2 == 0:
                    yl = c.kv(60, yl, k, v, style)
                else:
                    yr = c.kv(540, yr, k, v, style)
            y = max(yl, yr)
        else:
            for k, v in pairs:
                y = c.kv(60, y, k, v, style, value_x=320)
        y += 20
        hdr = ["Earnings", "Amount"] if variant != 1 else ["Component", "Amount"]
        y = draw_table(c, 60, y, [(hdr[0], 500, "l"), (hdr[1], 220, "r")], [[k, inr(v)] for k, v in comps], ruled=(variant == 0))
        gl = ["Gross Earnings", "Total Earnings", "Gross Salary"][variant]
        c.text(70, y + 6, gl, bold=True)
        c.text(770, y + 6, inr(gross), anchor="ra", bold=True)
        y += 70
        c.text(70, y, "Deductions" if variant != 2 else "Less: Deductions", bold=True)
        y += int(c.size * 1.9)
        for k, v in deds:
            c.text(70, y, k)
            c.text(770, y, inr(v), anchor="ra")
            y += int(c.size * 1.7)
        c.text(70, y + 10, "Total Deductions", bold=True)
        c.text(770, y + 10, inr(total_ded), anchor="ra", bold=True)
        y += 70
        nl = ["Net Pay", "Net Salary", "Take Home"][variant]
        c.text(70, y, nl, size=c.size + 3, bold=True)
        c.text(770, y, inr(net), size=c.size + 3, anchor="ra", bold=True)

        fields = {"employee_name": name, "employee_id": emp_id, "pay_period": f"{y_:04d}-{m_:02d}",
                  "pan": None if missing else pan, "gross_earnings": f"{gross:.2f}", "total_deductions": f"{total_ded:.2f}",
                  "net_pay": f"{net:.2f}"}
        rows = [{"component": k, "amount": f"{v:.2f}"} for k, v in comps]
        return c, fields, rows

    # --------------------------------------------------------------- kyc
    def kyc_id(self, variant: int, anomaly: Optional[str]):
        r = self.rng
        tint = r.choice([(214, 232, 247), (225, 240, 230), (240, 232, 214), (235, 235, 245)])
        c = self.canvas(860, 540, bg=tint)
        c.size = r.randint(20, 23)
        name, last = self.person()
        father = f"{r.choice(self.first)} {last}"
        dob = date(r.randint(1955, 2004), r.randint(1, 12), r.randint(1, 28))
        if anomaly == "inconsistent":  # minor applicant
            dob = date(date.today().year - r.randint(10, 16), r.randint(1, 12), r.randint(1, 28))
        pan = rand_pan(r, last)
        ds = 2 if variant != 1 else r.randint(0, 2)
        c.text(40, 30, "INCOME TAX DEPARTMENT", bold=True, size=c.size + 2)
        c.text(c.w - 40, 30, "GOVT. OF INDIA", bold=True, size=c.size + 2, anchor="ra")
        c.draw.rectangle([40, 120, 190, 300], outline=(90, 90, 90), width=2, fill=tuple(max(0, t - 30) for t in tint))
        x = 230
        if variant == 0:
            y = c.kv(x, 110, "Name", name, "below")
            y = c.kv(x, y, "Father's Name", father, "below")
            y = c.kv(x, y, "Date of Birth", fmt_date(dob, ds), "below")
            c.kv(x, y, "Permanent Account Number", pan, "below")
        elif variant == 1:
            y = 120
            for k, v in [("Name", name), ("Father's Name", father), ("Date of Birth", fmt_date(dob, ds)), ("PAN", pan)]:
                y = c.kv(x, y, k, v, "colon")
        else:
            c.text(x, 95, "Permanent Account Number Card", size=c.size - 2)
            c.text(x, 130, pan, size=c.size + 6, bold=True)
            y = c.kv(x, 185, "Name", name, "below")
            y = c.kv(x, y, "Father's Name", father, "below")
            c.kv(x, y, "Date of Birth", fmt_date(dob, ds), "below")
        c.text(c.w - 60, c.h - 50, "Signature", size=c.size - 4, anchor="ra", fill=(90, 90, 90))
        fields = {"name": name, "father_name": father, "date_of_birth": dob.isoformat(), "id_number": pan}
        return c, fields, []


# --------------------------------------------------------------- degrade

def degrade(img: Image.Image, rng: random.Random, level: float) -> Image.Image:
    """Scan/photo artefacts. level in [0, 1] scales how aggressive they are."""
    if level <= 0:
        return img
    np_rng = np.random.default_rng(rng.randint(0, 2**31))
    if rng.random() < 0.3 * level + 0.1:  # phone photo: page on a background with perspective
        pad = int(0.08 * max(img.size))
        bg = Image.new("RGB", (img.width + 2 * pad, img.height + 2 * pad), tuple(int(v) for v in np_rng.integers(30, 90, 3)))
        bg.paste(img, (pad, pad))
        w, h = bg.size
        j = lambda: rng.uniform(0, pad * 0.6)  # noqa: E731
        src = [(pad, pad), (w - pad, pad), (w - pad, h - pad), (pad, h - pad)]
        dst = [(pad + j(), pad + j()), (w - pad - j(), pad + j()), (w - pad - j(), h - pad - j()), (pad + j(), h - pad - j())]
        img = bg.transform(bg.size, Image.Transform.PERSPECTIVE, _perspective_coeffs(dst, src), Image.Resampling.BICUBIC)
    elif rng.random() < 0.8:
        img = img.rotate(rng.uniform(-3.5, 3.5) * level, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    arr = np.asarray(img).astype(np.float32)
    if rng.random() < 0.6:  # uneven illumination
        h, w = arr.shape[:2]
        gx = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.9, 1.05), w)[None, :, None]
        arr *= gx
    arr = arr * rng.uniform(0.85, 1.1) + rng.uniform(-15, 15)
    if rng.random() < 0.7:
        arr += np_rng.normal(0, rng.uniform(2, 10) * level, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1) * level))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(rng.uniform(45, 90)))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def _perspective_coeffs(src, dst):
    a = []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    b = np.array(dst).reshape(8)
    return np.linalg.solve(np.array(a, dtype=float), b).tolist()


# --------------------------------------------------------------- dataset

def generate(out_dir: Path, n_per_type: int, split: str, seed: int, anomaly_rate: float = 0.12,
             degrade_level: tuple[float, float] = (0.3, 1.0)) -> list[dict]:
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    gen = Gen(rng, split)
    records = []
    for doc_type in DOC_TYPES:
        for i in range(n_per_type):
            variant = i % 3
            roll = rng.random()
            anomaly = None if roll > anomaly_rate else ("inconsistent" if roll < anomaly_rate / 2 else "missing")
            if doc_type == "kyc_id" and anomaly == "missing":
                anomaly = None
            canvas, fields, rows = getattr(gen, doc_type)(variant, anomaly)
            level = rng.uniform(*degrade_level)
            img = degrade(canvas.img, rng, level)
            doc_id = f"{split}_{doc_type}_{i:04d}"
            path = out_dir / "images" / f"{doc_id}.jpg"
            img.save(path, quality=92)
            records.append({
                "doc_id": doc_id, "image": f"images/{doc_id}.jpg", "doc_type": doc_type, "layout_variant": variant,
                "degrade_level": round(level, 3), "anomaly": anomaly, "fields": fields, "table": rows,
                "text": canvas.gt_text(),
            })
    with open(out_dir / "labels.jsonl", "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return records


def load_labels(data_dir: Path) -> list[dict]:
    with open(Path(data_dir) / "labels.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]
