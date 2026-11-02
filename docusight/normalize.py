"""Value normalisation and OCR post-correction per field kind.

Normalised values are canonical strings (ISO dates, plain decimals, upper-case
IDs) so extraction can be scored exactly and validated arithmetically.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

MONTHS = {m.lower(): i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
MONTHS.update({m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September",
     "October", "November", "December"], start=1)})
MONTHS["sept"] = 9

# Position-aware OCR confusion maps: letters that look like digits and vice versa.
TO_DIGIT = str.maketrans({"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "|": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"})
TO_ALPHA = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T", "4": "A"})

_AMOUNT_RE = re.compile(r"-?\d[\d,]*(?:\.\d{1,2})?")
_DATE_PATTERNS = [
    re.compile(r"(?P<d>\d{1,2})[/\-.](?P<m>\d{1,2})[/\-.](?P<y>\d{4})"),
    re.compile(r"(?P<y>\d{4})[/\-.](?P<m>\d{1,2})[/\-.](?P<d>\d{1,2})"),
    re.compile(r"(?P<d>\d{1,2})[\s\-]*(?P<mon>[A-Za-z]{3,9})[,\s\-]*(?P<y>\d{4})"),
    re.compile(r"(?P<mon>[A-Za-z]{3,9})[\s\-]*(?P<d>\d{1,2}),?[\s\-]*(?P<y>\d{4})"),
]


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip(" :;,.-|")
    return s


def amount(s: str) -> Optional[str]:
    s = s.replace("Rs.", " ").replace("Rs", " ").replace("INR", " ").replace("₹", " ")
    s = s.replace(" ", "")
    s = re.sub(r"(?<=\d)[oO](?=[\d,.]|$)", "0", s)  # common o/0 slips inside numbers
    m = _AMOUNT_RE.search(s)
    if not m:
        return None
    try:
        return f"{float(m.group().replace(',', '')):.2f}"
    except ValueError:
        return None


def parse_date(s: str) -> Optional[date]:
    for pat in _DATE_PATTERNS:
        m = pat.search(s)
        if not m:
            continue
        g = m.groupdict()
        try:
            if g.get("m"):
                month = int(g["m"])
            else:
                mon = g["mon"].lower()
                month = MONTHS.get(mon) or MONTHS.get(mon[:3])
            if not month:
                continue
            return date(int(g["y"]), month, int(g["d"]))
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return None


def iso_date(s: str) -> Optional[str]:
    d = parse_date(s)
    return d.isoformat() if d else None


def period(s: str) -> Optional[str]:
    parts = re.split(r"\s+(?:to|-|–|till|until)\s+", s, maxsplit=1, flags=re.I)
    if len(parts) == 2:
        a, b = parse_date(parts[0]), parse_date(parts[1])
        if a and b:
            return f"{a.isoformat()}..{b.isoformat()}"
    return None


def month(s: str) -> Optional[str]:
    m = re.search(r"([A-Za-z]{3,9})[\s,\-']*(\d{4})", s)
    if m and m.group(1).lower() in MONTHS:
        return f"{int(m.group(2)):04d}-{MONTHS[m.group(1).lower()]:02d}"
    m = re.search(r"(\d{1,2})[/\-](\d{4})", s)
    if m and 1 <= int(m.group(1)) <= 12:
        return f"{int(m.group(2)):04d}-{int(m.group(1)):02d}"
    return None


def _alnum(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


def _by_mask(s: str, mask: str) -> str:
    """Correct characters by expected class: A=letter, 9=digit, X=either."""
    out = []
    for ch, kind in zip(s, mask):
        if kind == "A":
            out.append(ch.translate(TO_ALPHA))
        elif kind == "9":
            out.append(ch.translate(TO_DIGIT))
        else:
            out.append(ch)
    return "".join(out)


def pan(s: str) -> Optional[str]:
    v = _alnum(s)
    for i in range(0, max(1, len(v) - 9)):
        cand = v[i:i + 10]
        if len(cand) == 10:
            fixed = _by_mask(cand, "AAAAA9999A")
            if re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", fixed):
                return fixed
    return v[:10] or None


def ifsc(s: str) -> Optional[str]:
    v = _alnum(s)
    if len(v) >= 11:
        v = v[:11]
        return _by_mask(v[:4], "AAAA") + "0" + v[5:]
    return v or None


def gstin(s: str) -> Optional[str]:
    v = _alnum(s)
    if len(v) >= 15:
        v = v[:15]
        return _by_mask(v[:12], "99AAAAA9999A") + v[12] + "Z" + v[14]
    return v or None


def digits(s: str) -> Optional[str]:
    v = _alnum(s).translate(TO_DIGIT)
    v = re.sub(r"\D", "", v)
    return v or None


def code(s: str) -> Optional[str]:
    v = clean_text(s).upper().replace(" ", "")
    return v or None


NORMALIZERS = {
    "text": lambda s: clean_text(s) or None,
    "amount": amount,
    "date": iso_date,
    "period": period,
    "month": month,
    "pan": pan,
    "ifsc": ifsc,
    "gstin": gstin,
    "digits": digits,
    "code": code,
}


def normalize(kind: str, raw: str) -> Optional[str]:
    return NORMALIZERS[kind](raw)


def values_equal(kind: str, a: Optional[str], b: Optional[str]) -> bool:
    """Comparison used by the evaluator (case/space-insensitive for free text)."""
    if a is None or b is None:
        return a == b
    if kind == "text":
        return clean_text(a).lower() == clean_text(b).lower()
    return a == b


def to_float(s: Optional[str]) -> Optional[float]:
    try:
        return float(s) if s is not None else None
    except ValueError:
        return None


def age_on(dob: date, today: Optional[date] = None) -> int:
    today = today or datetime.now().date()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
