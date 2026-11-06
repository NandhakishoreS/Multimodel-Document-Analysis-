# Phase 15 Architecture & Readiness Audit

**Project:** Multimodal Transformer-Based Intelligent Document Understanding and Information Extraction  
**Phase:** 15 — Duplicate & Anomaly Detection Intelligence  
**Date:** 2026-09-12  
**Baseline Test Suite:** 387 passed, 0 failed, 0 errors, 0 warnings (Phases 1–14 all green)  
**Status:** Pre-Implementation Architecture Audit Completed  

---

## 1. Executive Summary

Phase 15 elevates the institutional financial intelligence platform from basic anomaly indicators into a dedicated, explainable, and actionable **Financial Integrity & Anomaly Intelligence Subsystem**.

### Core Scope:
1. Multi-signal duplicate detection (exact file hash, duplicate invoice ID, duplicate financial signature, strong composite match, near-duplicate record).
2. Deterministic, explainable duplicate confidence and scoring model (no black-box LLMs).
3. Centralized 10-category anomaly detection engine (`ANOMALY_001` through `ANOMALY_010`).
4. Rule-driven deterministic anomaly severity (`INFO`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
5. Multi-signal anomaly aggregation at the document level.
6. Formal investigation lifecycle workflow (`OPEN`, `UNDER_REVIEW`, `RESOLVED`, `DISMISSED`, `CONFIRMED_DUPLICATE`) with an immutable audit history.
7. Non-destructive SQLite schema extensions (`anomaly_findings`, `duplicate_candidates`, `anomaly_review_history`) with dedicated indexes.
8. Controlled, idempotent rescan and recomputation mechanisms.
9. Interactive review queue, side-by-side duplicate comparison, and false-positive dismissal in Streamlit (`app/phase11_app.py`).
10. Enhanced multi-format institutional exports (Excel, CSV, JSON) and full backward compatibility.

### Responsible Analytics Invariants:
- The system is an analytical decision-support tool.
- Anomaly $\ne$ Fraud.
- Duplicate candidate $\ne$ Confirmed duplicate until reviewer sign-off.
- Statistical outlier / round amount $\ne$ Wrongdoing.
- Tax inconsistency $\ne$ Legal non-compliance.
- No unsupported claims of fraud or legal certification will be made.

---

## 2. Audit of Existing Infrastructure

### 2.1 Existing Ingestion & Cryptographic Duplicate Detection (`phase11/ingestion.py`)
- **Current State:**
  - In `IngestionItem.calculate_hash()`, computes SHA-256 digests of raw file bytes.
  - In `DocumentIngestionService.check_duplicate()`, queries `db.get_document_by_hash(file_hash)` to detect identical files.
  - In `DuplicatePolicy`, supports `SKIP`, `WARN`, `PROCESS_ANYWAY`.
  - In `find_duplicate_financials()`, checks whether `(invoice_number and vendor_name)` or `(vendor_name and invoice_date and total_amount)` match existing records.
- **Identified Gaps:**
  - Binary duplicate decision (either exact duplicate or not); no multi-tiered scoring or confidence metric.
  - Does not evaluate near-duplicates (e.g., slight variation in invoice number, small date drift, minor amount discrepancies due to fees).
  - Findings are not persisted in structured relational tables, but merely appended as strings to `doc.anomaly_flags`.
  - No structured review status or reviewer assignment for duplicate candidates.

### 2.2 Existing Anomaly Reporting (`phase11/database.py`)
- **Current State:**
  - `anomaly_candidates()` returns documents with non-empty `anomaly_flags` or round amounts.
  - `advanced_anomaly_report(sigma_threshold=2.0, **filter_kwargs)` calculates sample mean and standard deviation of `total_amount`, detects $\mu + k\sigma$ outliers, groups duplicate invoice numbers, compiles a missing data matrix, and counts round amounts.
- **Identified Gaps:**
  - Computations are calculated on-the-fly without persistent finding identities (`finding_id`).
  - No granular severity levels (`INFO`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
  - Lacks temporal spike tracking (MoM surge > 50% or $>2\times$ rolling average).
  - Lacks category/value distribution mismatch detection.
  - No workflow state to track whether an anomaly was reviewed, dismissed, or resolved.

### 2.3 Existing Schema & Database Design (`phase11/database.py`, `phase11/schema.py`)
- **Current Tables:**
  - `documents`: Stores core entity values, AI baseline values (`ai_vendor_name`, `ai_total_amount`, etc.), processing status, and `anomaly_flags` (JSON string).
  - `line_items`: Per-document itemized rows.
  - `processing_log`: Audit trail table recording actions (`insert`, `update`, `FIELD_UPDATED`, `RECORD_EDITED`, `CLASSIFICATION_MODIFIED`).
- **Required Extensions (Non-Destructive):**
  - Table `anomaly_findings`: Stores atomic anomaly instances with document foreign key, anomaly type, severity, score, confidence, explanation, status, and detection timestamp.
  - Table `duplicate_candidates`: Stores pairwise duplicate linkages (`document_id`, `matched_document_id`) with duplicate type, score, confidence, matched fields, and status.
  - Table `anomaly_review_history`: Immutable log of review actions (`old_status`, `new_status`, `reviewer`, `comment`, `timestamp`).
  - High-speed indexes on document ID, anomaly type, severity, status, and score.

### 2.4 Existing Streamlit UI (`app/phase11_app.py`)
- **Current State:**
  - Page 3 has Tab G displaying basic statistical anomaly scatter charts and duplicate invoice groups.
  - Page 4 ("⚠️ Anomaly Detection") currently displays a basic table and a simple dropdown to flag documents.
- **Proposed Phase 15 Enhancement:**
  - Transform Page 4 into a full **Duplicate & Anomaly Intelligence Center**:
    1. Executive Anomaly & Duplicate KPI Header (Total flagged, open, under review, resolved, dismissed, confirmed duplicates).
    2. Severity & Anomaly Type Visual Distribution.
    3. Multi-Signal Duplicate Investigation Grid (Side-by-side Current vs Matched document comparison).
    4. Review Queue with multi-parameter filtering (Severity, Anomaly Type, Review Status, Score, Category, Vendor, Date).
    5. Investigation & Disposition Modal/Drawer (Change status to `UNDER_REVIEW`, `RESOLVED`, `DISMISSED`, `CONFIRMED_DUPLICATE` with mandatory reviewer attribution and audit comment).
    6. Historical Audit Timeline.
    7. Idempotent Rescan Control (Scan All, Scan Filtered, Scan Single).

---

## 3. Proposed Phase 15 Architecture

### 3.1 Subsystem Modular Structure
We will introduce a dedicated module `phase11/anomaly_engine.py` (or extend `database.py` and `classifier.py`) to maintain high cohesion and separation of concerns:

```text
                               ┌─────────────────────────────────────────┐
                               │       Document Ingestion / DB Stream    │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │   Multi-Signal Duplicate Detector       │
                               │   • Exact File Hash (SHA-256)           │
                               │   • Duplicate Invoice ID (Vendor+Inv)   │
                               │   • Financial Signature (Vendor+Date+Amt│
                               │   • Strong Composite (All 4 match)      │
                               │   • Near-Duplicate (Fuzzy/Tolerance)    │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │   Centralized Anomaly Detection Engine  │
                               │   • ANOMALY_001: Statistical Outlier    │
                               │   • ANOMALY_002: High Invoice Baseline  │
                               │   • ANOMALY_003: Round Amount           │
                               │   • ANOMALY_004: Duplicate Inv Number   │
                               │   • ANOMALY_005: Missing Critical Fields│
                               │   • ANOMALY_006: Tax Inconsistency      │
                               │   • ANOMALY_007: Category Distribution  │
                               │   • ANOMALY_008: Vendor Concentration   │
                               │   • ANOMALY_009: Temporal Spending Surge│
                               │   • ANOMALY_010: Repeated Signature     │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │  Severity Scorer & Document Aggregator  │
                               │  (INFO, LOW, MEDIUM, HIGH, CRITICAL)    │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │   Idempotent Persistence Engine         │
                               │   • anomaly_findings                    │
                               │   • duplicate_candidates                │
                               │   • anomaly_review_history              │
                               └────────────────────┬────────────────────┘
                                                    │
                         ┌──────────────────────────┴──────────────────────────┐
                         ▼                                                     ▼
        ┌──────────────────────────────────┐                 ┌──────────────────────────────────┐
        │  Streamlit Review Center (Page 4)│                 │   Multi-Format Exporter          │
        │  • Summary KPIs                  │                 │   • Excel Workbook (7+ Sheets)   │
        │  • Side-by-Side Duplicate View   │                 │   • Anomaly Findings CSV/JSON    │
        │  • Review Queue & Audit Logging  │                 │   • Duplicate Candidates CSV/JSON│
        └──────────────────────────────────┘                 └──────────────────────────────────┘
```

---

## 4. Multi-Signal Duplicate Detection Matrix

| Duplicate Type | Signals / Fields Evaluated | Candidate Blocking Key | Score | Confidence | Severity |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `EXACT_FILE_DUPLICATE` | `file_hash` SHA-256 match | `file_hash` | 1.00 | 1.00 | `CRITICAL` |
| `STRONG_DUPLICATE_CANDIDATE` | `vendor_name` + `invoice_number` + `invoice_date` + `total_amount` | `vendor_name` + `invoice_number` | 0.95 | 0.95 | `CRITICAL` |
| `DUPLICATE_INVOICE_ID` | `vendor_name` + `invoice_number` | `vendor_name` + `invoice_number` | 0.85 | 0.90 | `HIGH` |
| `DUPLICATE_FINANCIAL_SIGNATURE`| `vendor_name` + `invoice_date` + `total_amount` | `vendor_name` + `invoice_date` | 0.80 | 0.85 | `HIGH` |
| `NEAR_DUPLICATE_RECORD` | `vendor_name` + date within $\pm 3$ days + amount within $\pm 1\%$ (or $\le ₹5$) | `vendor_name` | 0.65 | 0.70 | `MEDIUM` |

*Candidate Blocking*: To prevent $O(N^2)$ comparisons over 10,000 records, comparisons are blocked primarily by normalized `vendor_name`, `invoice_number`, and `invoice_date` prefixes.

---

## 5. Anomaly Types, Scoring & Severity Rules

| Code | Anomaly Type | Condition / Formulation | Base Score | Severity |
| :--- | :--- | :--- | :---: | :---: |
| `ANOMALY_001` | Statistical Outlier | Sample $N \ge 3$, $x > \mu + 2\sigma$ ($2\sigma$ to $3\sigma$: `MEDIUM`; $>3\sigma$: `HIGH`) | 0.70 – 0.90 | `MEDIUM` / `HIGH` |
| `ANOMALY_002` | Unusually High Invoice | $x > 3 \times$ Category Mean or $x > 3 \times$ Vendor Average | 0.75 | `MEDIUM` |
| `ANOMALY_003` | Round Amount | $x > 0$ and $x \pmod{1000} == 0$ (if $x \ge 50,000$: `LOW`; $<50,000$: `INFO`) | 0.30 – 0.50 | `INFO` / `LOW` |
| `ANOMALY_004` | Duplicate Invoice Number | Same `invoice_number` across multiple distinct `document_id`s | 0.85 | `HIGH` |
| `ANOMALY_005` | Missing Critical Fields | Missing `vendor_name`, `invoice_date`, `total_amount`, or `currency` (1 missing: `LOW`; $\ge 2$: `MEDIUM`) | 0.40 – 0.65 | `LOW` / `MEDIUM` |
| `ANOMALY_006` | Tax Inconsistency | Both `subtotal` & `tax_amount` present and $\|(\text{subtotal} + \text{tax}) - \text{total}\| > 1.0$ | 0.60 | `MEDIUM` |
| `ANOMALY_007` | Category/Value Mismatch | Document amount is $> 4\sigma$ above category mean | 0.70 | `MEDIUM` |
| `ANOMALY_008` | Vendor Concentration | Vendor spend accounts for $>25\%$ of total spend | 0.50 | `LOW` |
| `ANOMALY_009` | Temporal Spending Surge | Month spend $> 1.5 \times$ prior month and $> 2 \times$ rolling 3-month avg | 0.65 | `MEDIUM` |
| `ANOMALY_010` | Repeated Financial Signature | Identical (vendor, date, total amount) across distinct invoice numbers | 0.80 | `HIGH` |

### Multi-Signal Aggregation Rule:
- Document Anomaly Score $S = \min\left(1.0, \max(s_i) + 0.1 \times (\text{count} - 1)\right)$.
- Document Severity:
  - If any signal is `CRITICAL` or $\ge 2$ signals are `HIGH` $\rightarrow$ `CRITICAL`.
  - Else if any signal is `HIGH` or $\ge 2$ signals are `MEDIUM` $\rightarrow$ `HIGH`.
  - Else if any signal is `MEDIUM` or $\ge 2$ signals are `LOW` $\rightarrow$ `MEDIUM`.
  - Else if any signal is `LOW` $\rightarrow$ `LOW`.
  - Else $\rightarrow$ `INFO`.

---

## 6. Review Workflow & Audit State Machine

```text
                       ┌──────────────────────┐
                       │         OPEN         │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
            ┌─────────►│     UNDER_REVIEW     │◄─────────┐
            │          └──────────┬───────────┘          │
            │                     │                      │
            ▼                     ▼                      ▼
  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────┐
  │     RESOLVED     │  │    DISMISSED     │  │  CONFIRMED_DUPLICATE  │
  └──────────────────┘  └──────────────────┘  └───────────────────────┘
```

- **Valid Transitions**:
  - `OPEN` $\rightarrow$ `UNDER_REVIEW`, `RESOLVED`, `DISMISSED`, `CONFIRMED_DUPLICATE`
  - `UNDER_REVIEW` $\rightarrow$ `RESOLVED`, `DISMISSED`, `CONFIRMED_DUPLICATE`, `OPEN`
  - `RESOLVED` / `DISMISSED` / `CONFIRMED_DUPLICATE` $\rightarrow$ `UNDER_REVIEW`, `OPEN` (with audit comment)
- **Persistence**: Every transition writes a record to `anomaly_review_history` with:
  `entity_id`, `old_status`, `new_status`, `reviewer`, `comment`, `timestamp`.
- **False-Positive Handling**: `DISMISSED` preserves the finding, explanation, and detection timestamp; it is flagged as dismissed without deleting data. Re-scanning respects dismissed or resolved items.

---

## 7. Idempotency & Re-Scan Strategy

- **Finding Key**: Each finding has a deterministic deduplication key:
  `dedup_key = f"{document_id}:{anomaly_type}"` (or composite key for duplicates).
- **Idempotent Scan**:
  - When a scan is rerun, existing findings for that document are retrieved.
  - If a finding was already reviewed (`UNDER_REVIEW`, `RESOLVED`, `DISMISSED`, `CONFIRMED_DUPLICATE`), its status and history are strictly preserved!
  - Only new or updated findings are upserted.
  - Re-running detection 10 times results in the exact same database state.

---

## 8. Compatibility & Invariant Guarantees

1. **Zero Model Retraining**: No LayoutLMv3 checkpoints or models are modified.
2. **Benchmark Invariant**: SROIE, FUNSD, and CORD benchmarks remain untouched.
3. **Database Safety**: Tables `documents`, `line_items`, and `processing_log` are untouched. The 3 new tables are created with `CREATE TABLE IF NOT EXISTS`.
4. **Fault Isolation**: If anomaly analysis throws an unexpected exception during batch ingestion, the exception is logged, the underlying financial document remains safely saved, and processing continues.
5. **Full Regression Guarantee**: All 387 pre-existing tests across Phases 1–14 continue to pass without changes.
