# Phase 3: Tax Document Ingestion & Extraction Engine
**Project:** Indian Income Tax Return (ITR-1 SAHAJ) Preparation System  
**Module:** `tax_module.classifiers`, `tax_module.extractors`, `tax_module.ingestion`  
**Date:** November 2026  
**Status:** Completed & Tested (5 unit tests passed, 22 total tax tests passing)

---

## 1. Scope & Capabilities

The **Tax Document Ingestion & Extraction Engine** processes financial documents submitted by individual taxpayers, performs explainable document classification, extracts itemized statutory components with strict **Python `Decimal` precision**, and populates candidate income, deduction, and credit records.

### Supported Document Categories
1. **Form 16 Part A & Part B:** Employer TAN, Employee PAN, Salary u/s 17(1), Standard Deduction u/s 16(ia), Exempt Allowances u/s 10, Professional Tax u/s 16(iii), Net Salary, Chapter VI-A Deductions (80C, 80D, etc.), and TDS deposited.
2. **Bank Interest Certificates:** Savings account interest, fixed deposit interest, and TDS deducted u/s 194A.
3. **AIS / TIS (Annual Information Statement):** Salary income (`S-001`), Savings bank interest (`S-002`), Term deposit interest (`S-003`), Dividend income (`S-004`).
4. **Form 26AS:** TDS from Salary, TDS other than Salary, Advance Tax, Self-Assessment Tax.
5. **Investment Proofs:** 80C, 80D, 80G, and Home Loan interest certificates.

---

## 2. Monetary Precision & Decimal Preservation

The extraction engine relies upon `phase11.money_parser` as the single source of truth for numeric conversion:
- Handles standard Indian grouping (`12,34,567.89`) and Western grouping (`1,234,567.89`).
- Correctly parses cents and paise (`258.75`, `1,234.56`, `₹ 3,281.00`).
- Stores normalized numbers strictly as Python `Decimal` objects and exact Decimal string records in the database, avoiding binary floating-point roundoff errors.

---

## 3. Duplicate Prevention & Lifecycle Integration

- **SHA-256 Hashing:** Every ingested file has its SHA-256 digest computed. Re-uploading an identical document within the same assessment session is rejected immediately with a user-friendly duplicate notification.
- **Traceability:** Every `ExtractedTaxField` preserves:
  - `doc_id`
  - `original_ocr_text`
  - `normalized_decimal`
  - `confidence`
  - `verification_status`
  - `user_corrected_value` (if edited by human reviewer)

---

## 4. Test Suite Verification

- **Suite:** `tests/test_tax_ingestion.py`
- **Result:** 5 passed in 0.53s
- **Scenarios Tested:**
  - Automated classification of Form 16 Part B, Bank Interest Certificates, and AIS/TIS
  - Form 16 extraction with complex salary components and Chapter VI-A deductions
  - Exact Decimal preservation on bank interest (`258.75` and `1,234.56`)
  - End-to-end ingestion pipeline: file hashing, classification, field extraction, candidate `IncomeRecord` and `TaxCreditRecord` generation
  - Duplicate file rejection via SHA-256 digest matching
