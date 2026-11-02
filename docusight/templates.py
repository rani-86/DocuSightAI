"""Per-document-type extraction schemas: fields, label aliases, value formats, tables."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

PAN_RE = r"[A-Z]{5}[0-9]{4}[A-Z]"
IFSC_RE = r"[A-Z]{4}0[A-Z0-9]{6}"
GSTIN_RE = r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    labels: tuple[str, ...]
    kind: str = "text"  # text | amount | date | period | month | pan | ifsc | gstin | digits | code
    required: bool = True
    pattern: Optional[str] = None  # regex the normalised value must fully match


@dataclass(frozen=True)
class TableSpec:
    columns: dict[str, tuple[str, ...]]  # canonical column -> header aliases
    key_column: str  # rows without this column continue the previous row
    numeric: tuple[str, ...] = ()
    stop_words: tuple[str, ...] = ("total", "closing balance", "subtotal", "sub total", "net pay")


@dataclass(frozen=True)
class DocSchema:
    doc_type: str
    fields: tuple[FieldSpec, ...]
    table: Optional[TableSpec] = None
    keywords: tuple[str, ...] = field(default=())


SCHEMAS: dict[str, DocSchema] = {
    "bank_statement": DocSchema(
        "bank_statement",
        fields=(
            FieldSpec("account_holder", ("account holder", "customer name", "name")),
            FieldSpec("account_number", ("account number", "account no", "a/c no"), "digits", pattern=r"[0-9]{9,18}"),
            FieldSpec("ifsc", ("ifsc code", "ifsc"), "ifsc", pattern=IFSC_RE),
            FieldSpec("statement_period", ("statement period", "period"), "period"),
            FieldSpec("opening_balance", ("opening balance",), "amount"),
            FieldSpec("closing_balance", ("closing balance",), "amount"),
        ),
        table=TableSpec(
            columns={
                "date": ("date", "txn date", "value date"),
                "description": ("description", "narration", "particulars"),
                "debit": ("debit", "withdrawal", "dr"),
                "credit": ("credit", "deposit", "cr"),
                "balance": ("balance",),
            },
            key_column="date",
            numeric=("debit", "credit", "balance"),
        ),
        keywords=("statement", "ifsc", "balance", "withdrawal", "narration", "branch"),
    ),
    "invoice": DocSchema(
        "invoice",
        fields=(
            FieldSpec("invoice_number", ("invoice number", "invoice no", "bill no"), "code"),
            FieldSpec("invoice_date", ("invoice date", "date of issue", "bill date"), "date"),
            FieldSpec("vendor_name", ("seller", "vendor", "sold by")),
            FieldSpec("gstin", ("gstin", "gst no"), "gstin", pattern=GSTIN_RE),
            FieldSpec("subtotal", ("subtotal", "sub total", "taxable value"), "amount"),
            FieldSpec("tax_amount", ("gst", "tax amount", "total tax"), "amount"),
            FieldSpec("total_amount", ("grand total", "total amount", "amount due"), "amount"),
        ),
        table=TableSpec(
            columns={
                "item": ("item", "description", "product"),
                "qty": ("qty", "quantity"),
                "unit_price": ("unit price", "rate", "price"),
                "amount": ("amount", "line total"),
            },
            key_column="qty",
            numeric=("qty", "unit_price", "amount"),
            stop_words=("subtotal", "sub total", "taxable value", "total"),
        ),
        keywords=("invoice", "gstin", "tax", "qty", "bill to", "grand total"),
    ),
    "payslip": DocSchema(
        "payslip",
        fields=(
            FieldSpec("employee_name", ("employee name", "name of employee", "name")),
            FieldSpec("employee_id", ("employee id", "emp id", "emp code"), "code"),
            FieldSpec("pay_period", ("pay period", "salary month", "month"), "month"),
            FieldSpec("pan", ("pan",), "pan", pattern=PAN_RE),
            FieldSpec("gross_earnings", ("gross earnings", "total earnings", "gross salary"), "amount"),
            FieldSpec("total_deductions", ("total deductions", "deductions"), "amount"),
            FieldSpec("net_pay", ("net pay", "net salary", "take home"), "amount"),
        ),
        table=TableSpec(
            columns={"component": ("earnings", "component"), "amount": ("amount",)},
            key_column="amount",
            numeric=("amount",),
            stop_words=("gross", "total"),
        ),
        keywords=("payslip", "salary", "earnings", "deductions", "employee", "net pay"),
    ),
    "kyc_id": DocSchema(
        "kyc_id",
        fields=(
            FieldSpec("name", ("name",)),
            FieldSpec("father_name", ("father's name", "fathers name", "father name")),
            FieldSpec("date_of_birth", ("date of birth", "dob"), "date"),
            FieldSpec("id_number", ("permanent account number", "pan", "id number"), "pan", pattern=PAN_RE),
        ),
        keywords=("income tax", "permanent account number", "govt", "signature", "date of birth"),
    ),
}
