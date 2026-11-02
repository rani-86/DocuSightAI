# DocuSight AI - Evaluation Report

Held-out set: **220 documents** (bank_statement: 55, invoice: 55, payslip: 55, kyc_id: 55), OCR engine: `paddleocr`.

## Headline metrics

| Metric | Value |
|---|---|
| Document classification macro-F1 (hf_transformer) | 1.0000 |
| Field-level accuracy (exact match after normalisation) | 97.42% (1320 fields) |
| Field-value CER | 0.0205 |
| Page-level OCR CER (reading order) | 0.0100 |
| Injected-anomaly recall (flagged for review) | 100.0% (26 docs) |
| Problem-document recall / flag precision | 85.71% / 90.57% |
| Straight-through rate | 75.91% |
| Auto-approved docs that are fully correct | 95.21% |
| End-to-end latency p50 / p95 (CPU) | 7290 / 13483 ms |

## Classification (same OCR text for every model)

| Model | Macro-F1 |
|---|---|
| pipeline (hf_transformer) | 1.0000 |
| tfidf_logreg | 1.0000 |
| keyword | 1.0000 |

## Field accuracy by document type

| Doc type | Accuracy | Fields |
|---|---|---|
| bank_statement | 96.67% | 330 |
| invoice | 96.62% | 385 |
| kyc_id | 100.00% | 220 |
| payslip | 97.40% | 385 |

## Field accuracy by field

| Field | Accuracy | n |
|---|---|---|
| bank_statement.account_holder | 98.18% | 55 |
| bank_statement.account_number | 98.18% | 55 |
| bank_statement.closing_balance | 98.18% | 55 |
| bank_statement.ifsc | 92.73% | 55 |
| bank_statement.opening_balance | 98.18% | 55 |
| bank_statement.statement_period | 94.55% | 55 |
| invoice.gstin | 100.00% | 55 |
| invoice.invoice_date | 100.00% | 55 |
| invoice.invoice_number | 100.00% | 55 |
| invoice.subtotal | 94.55% | 55 |
| invoice.tax_amount | 94.55% | 55 |
| invoice.total_amount | 90.91% | 55 |
| invoice.vendor_name | 96.36% | 55 |
| kyc_id.date_of_birth | 100.00% | 55 |
| kyc_id.father_name | 100.00% | 55 |
| kyc_id.id_number | 100.00% | 55 |
| kyc_id.name | 100.00% | 55 |
| payslip.employee_id | 85.45% | 55 |
| payslip.employee_name | 100.00% | 55 |
| payslip.gross_earnings | 100.00% | 55 |
| payslip.net_pay | 100.00% | 55 |
| payslip.pan | 96.36% | 55 |
| payslip.pay_period | 100.00% | 55 |
| payslip.total_deductions | 100.00% | 55 |

## Robustness by degradation level

| Degradation | Accuracy | Fields |
|---|---|---|
| heavy | 98.61% | 431 |
| light | 99.40% | 334 |
| medium | 95.32% | 555 |

## Confidence calibration (accuracy of extracted values per confidence band)

| Confidence | Accuracy | n |
|---|---|---|
| 0.80-0.95 | 100.00% | 44 |
| <0.80 | 22.22% | 9 |
| >=0.95 | 99.36% | 1243 |

## Tables

| Doc type | Row exact | Cell accuracy | Row count correct |
|---|---|---|---|
| bank_statement | 90.79% | 92.24% | 90.91% |
| invoice | 90.30% | 92.92% | 90.91% |
| payslip | 100.00% | 100.00% | 100.00% |

## Review reasons

| Reason | Docs |
|---|---|
| MISSING_REQUIRED | 23 |
| LOW_CONFIDENCE | 11 |
| FORMAT_INVALID | 1 |
| INCONSISTENT | 26 |
| LOW_CLASS_CONFIDENCE | 0 |

## Latency by stage (CPU)

| Stage | p50 ms | p95 ms |
|---|---|---|
| preprocess | 83.4 | 103.9 |
| ocr | 7190.4 | 13375.2 |
| layout | 0.3 | 0.4 |
| classify | 4.8 | 8.0 |
| extract | 9.9 | 14.4 |
| validate | 0.1 | 0.1 |
| total | 7289.7 | 13483.3 |
