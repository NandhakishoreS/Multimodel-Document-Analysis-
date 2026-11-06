# Phase 4: Financial Normalization & Cross-Document Reconciliation
**Project:** Indian Income Tax Return (ITR-1 SAHAJ) Preparation System  
**Module:** `tax_module.reconciliation`  
**Date:** November 2026  
**Status:** Completed & Tested (3 unit tests passed, 25 total tax tests passing)

---

## 1. Statutory Rationale & Architecture

In Indian tax filing, taxpayers frequently possess multiple overlapping documents describing the same earnings:
- **Form 16 vs Monthly Payslips:** Both document salary, but summing them blindly results in catastrophic double-counting of gross income.
- **Form 16 TDS vs Form 26AS TDS:** If the deductor has deposited less tax than stated on Form 16 or delayed quarterly filing, the Centralized Processing Center (CPC) validates credits solely against Form 26AS.
- **Bank Interest Certificates vs AIS (`S-002`/`S-003`):** Banks report interest to the Income Tax Department; taxpayers omitting interest from savings accounts or FDs face Section 133(6) / 143(1) mismatch inquiries.

The **Cross-Document Reconciliation Engine** evaluates these multi-source streams to detect discrepancies, warn users, and provide an auditable resolution workflow.

---

## 2. Reconciliation Rules & Matching Logic

| Comparison Stream | Primary Source | Secondary Source | Action on Discrepancy (> ₹1.00) |
|---|---|---|---|
| **Gross Salary** | Form 16 (Part B Sec 17(1)) | AIS (Code `S-001`) | Flag `REVIEW_REQUIRED`. Form 16 remains authoritative unless employer corrected. |
| **Salary TDS** | Form 16 (Part A) | Form 26AS (Part I) | Flag `REVIEW_REQUIRED`. Warn that Form 26AS determines eligible CPC credit. |
| **Bank Interest** | Bank Interest Certificate | AIS (Code `S-002`/`S-003`) | Flag `REVIEW_REQUIRED`. AIS higher indicates unreported accounts. |
| **Document Overlap** | Form 16 (Annual) | Monthly Salary Slips | Flag `REVIEW_REQUIRED`. Block automatic summation to prevent double-counting. |

---

## 3. Discrepancy Resolution Lifecycle

- When any discrepancy exceeds `DISCREPANCY_TOLERANCE` (₹1.00), the session filing status transitions to `RECONCILIATION_REQUIRED`.
- The taxpayer/reviewer can resolve flags via `resolve_reconciliation_record()`, selecting the authoritative source, providing explanation notes, or adjusting amounts.
- Every resolution is persisted to SQLite and recorded in `tax_audit_log`.

---

## 4. Test Suite Verification

- **Suite:** `tests/test_tax_reconciliation.py`
- **Result:** 3 passed in 0.48s
- **Scenarios Tested:**
  - Perfect salary match between Form 16 and AIS resulting in `MATCHED` status
  - Salary mismatch (₹50,000 difference) triggering `REVIEW_REQUIRED` and state transition to `RECONCILIATION_REQUIRED`
  - Automated detection of Form 16 and salary slip overlap preventing double-counting
  - Resolution lifecycle: status update to `RESOLVED` with auditable resolution notes
