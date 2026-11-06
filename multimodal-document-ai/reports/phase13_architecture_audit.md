# Phase 13 Architecture & Readiness Audit

**Project:** Multimodal Transformer-Based Intelligent Document Understanding and Information Extraction  
**Phase:** 13 — Advanced Search, Filtering, Editable Financial Records & Audit-Aware Document Management  
**Date:** 2026-09-12  
**Baseline Test Suite:** 299 passed, 0 failed, 0 errors, 0 warnings (194 baseline + 80 Phase 11 + 25 Phase 12)  
**Status:** Pre-Implementation Architecture Audit Completed  

---

## 1. Executive Summary

Phase 13 elevates the institutional financial intelligence platform from batch ingestion and classification (Phases 11–12) into a production-grade **document management, discovery, and human-in-the-loop verification system**. The primary objective is enabling institutional operators to search, filter, sort, inspect, edit, verify, and export hundreds to thousands of financial records while preserving an immutable, granular audit trail and keeping AI-extracted predictions cleanly separated from human-verified final values.

This audit evaluates the existing codebase across:
1. Current search capabilities (`phase11/database.py`)
2. Current database schema and tables (`documents`, `line_items`, `processing_log`)
3. Existing SQLite indexes and query performance
4. Current record editing functionality
5. Existing audit logging mechanism
6. Current Document Registry Streamlit UI (`app/phase11_app.py`)
7. Export and analytics interaction with search/filter state
8. Reusable components vs. required enhancements
9. Architecture risks and mitigation strategies

---

## 2. Audit of Existing Search & Retrieval Capabilities

### 2.1 Current Implementation in `FinancialDocumentDB.search_documents()`
In `phase11/database.py` (lines 459–550):
* **Query Matcher:** Performs a case-insensitive multi-column `LIKE %q%` across `vendor_name`, `filename`, `ocr_raw_text`, `address`, and `invoice_number`.
* **Available Filters:** `category`, `verification_status`, `date_from` (>=), `date_to` (<=), `min_amount` (>=), `max_amount` (<=), `benchmark_domain`, `requires_review`, `processing_status`.
* **Ordering:** Hardcoded to `ORDER BY created_at DESC`.
* **Pagination:** Basic `LIMIT :limit OFFSET :offset`, defaulting to `limit=200, offset=0`.

### 2.2 Gaps & Limitations
1. **No Dedicated Column Filters:** No isolated filter for `vendor_name` or `invoice_number` (currently only search string `q` checks them in an OR clause).
2. **Date Granularity:** Only supports raw string comparison `invoice_date >= date_from` and `<= date_to`. Does not parse month (`YYYY-MM`), fiscal year (`FY2025-26`), or malformed OCR date formats natively.
3. **No Total Result Count for Paginated Queries:** `count_documents()` only supports `category` and `verification_status`. When active filters are applied, the system cannot report total pages or "Showing X to Y of Total Z records" without fetching all rows.
4. **Hardcoded Sort Order:** Does not allow institutional users to sort by invoice date, amount, vendor, category, confidence, or verification status, nor switch between ASC and DESC.
5. **No Full-Text / Tokenized Match:** While `LIKE` works well for SQLite tables up to tens of thousands of rows, queries containing tokenized multi-field terms or specific vendor normalization need robust indexing and query compilation.

---

## 3. Database Schema & Indexing Audit

### 3.1 Current Schema (`data/financial_documents.db`)
* **`documents` Table:**
  * Identity: `document_id` (PK, UUID), `filename`, `source_file`, `document_type`, `benchmark_domain`, `model_name`.
  * Classification: `category`, `sub_category`, `classification_confidence`, `classification_evidence` (JSON text), `ai_category`, `ai_confidence`, `final_category`, `requires_review`, `classification_reason`.
  * Extraction: `vendor_name`, `address`, `invoice_number`, `invoice_date`, `due_date`, `subtotal`, `tax_amount`, `total_amount`, `currency`, `extraction_confidence`, `ocr_word_count`, `ocr_raw_text`.
  * Lifecycle & Ingestion: `verification_status`, `processing_timestamp`, `created_at`, `updated_at`, `notes`, `anomaly_flags` (JSON text), `file_hash`, `file_size`, `processing_status`, `processing_stage`, `processing_error`, `processing_started_at`, `processing_completed_at`, `processing_duration`.
* **`line_items` Table:**
  * `id` (PK AUTOINCREMENT), `document_id` (FK to `documents`), `description`, `quantity`, `unit_price`, `total`, `unit`.
* **`processing_log` Table:**
  * `id` (PK AUTOINCREMENT), `document_id`, `action`, `timestamp`, `details` (TEXT).

### 3.2 Existing Indexes
The following indexes exist in `_DDL_INDEXES`:
* `idx_documents_vendor ON documents(vendor_name)`
* `idx_documents_category ON documents(category)`
* `idx_documents_date ON documents(invoice_date)`
* `idx_documents_status ON documents(verification_status)`
* `idx_documents_amount ON documents(total_amount)`
* `idx_documents_file_hash ON documents(file_hash)`
* `idx_documents_proc_status ON documents(processing_status)`
* `idx_documents_req_review ON documents(requires_review)`
* `idx_line_items_doc ON line_items(document_id)`

### 3.3 Missing Schema Fields & Indexes for Phase 13
1. **AI Value Preservation for Financial Fields:**
   Currently, only `ai_category` and `ai_confidence` are captured in the schema. When a user edits `total_amount`, `vendor_name`, `invoice_date`, or `invoice_number`, the initial AI extraction is overwritten in `documents`.
   *Required New Columns:*
   * `ai_vendor_name TEXT`
   * `ai_invoice_date TEXT`
   * `ai_invoice_number TEXT`
   * `ai_total_amount REAL`
   * `is_modified INTEGER NOT NULL DEFAULT 0`
2. **Verification Status Expansion:**
   Add `"modified"` to `VERIFICATION_STATUSES` (currently `("pending", "verified", "flagged", "rejected")`).
3. **Missing Composite / Dedicated Indexes:**
   * `idx_documents_inv_num ON documents(invoice_number)`: Critical for exact and prefix invoice search.
   * Composite date & category index: `idx_documents_cat_date ON documents(category, invoice_date)` for fast filtered reporting.

---

## 4. Current Record Editing & Audit Mechanism Audit

### 4.1 Existing Editing (`FinancialDocumentDB.update_document` & `update_category`)
* `update_category()` cleanly updates `category`, sets `final_category`, clears `requires_review`, and writes an audit event:
  `action = 'CLASSIFICATION_MODIFIED'`, `details = json.dumps({"previous_category": ..., "new_category": ..., ...})`.
* However, general edits in `update_document()`:
  - Overwrite all fields in the row.
  - Log only `action = 'update'` with empty details.
  - **No field-level diff is recorded in `processing_log`.**
  - **Cannot answer: "What was the previous total amount?", "Who changed the vendor name?", or "When was the due date corrected?"**

### 4.2 Audit Trail Deficiencies
Institutional audit compliance demands:
1. Exact field-level diff: `{"field": "total_amount", "old": 18500.0, "new": 18800.0, "reason": "User correction"}`.
2. Action type: `FIELD_UPDATED` or `RECORD_EDITED`.
3. Ability to retrieve and render the complete chronological timeline for a document (`get_audit_history(document_id)`).

---

## 5. Existing Streamlit UI Audit (`app/phase11_app.py`)

### 5.1 Document Registry Page (Page 2)
* Contains basic search text box and dropdowns for Category, Status, Review, Min/Max amount.
* Results are rendered in a single `st.dataframe` with a hard limit of 500 rows.
* Document selection is done via a single flat dropdown (`doc_names`) which becomes unmanageable when hundreds of documents match.
* Editing form directly updates the document object without validation (e.g. negative amounts or invalid dates are not blocked).
* Discard / Cancel button does not exist (form submit is the only action).
* No pagination controls (Previous / Next / Page size).
* No field-level change history viewer.

### 5.2 Filtered Analytics & Export Disconnect (Pages 3 & 5)
* Page 3 (`Financial Analytics`) queries `db.spend_by_category()` and `db.status_summary()` on the **entire database**, completely ignoring whatever filters the user configured in Page 2.
* Page 5 (`Export & Reports`) exports either all documents or documents matching a simple fresh filter, but filter state is not preserved in `st.session_state` between pages.

---

## 6. Reusable Components vs. Required Changes

| Component | Status | Action in Phase 13 |
| :--- | :--- | :--- |
| `FinancialDocument` dataclass | Active | Add `ai_vendor_name`, `ai_invoice_date`, `ai_invoice_number`, `ai_total_amount`, `is_modified`. Add `modified` to statuses. |
| `FinancialDocumentDB` schema | Active | Non-destructive migration adding new AI value columns & `idx_documents_inv_num`. |
| `FinancialDocumentDB.search_documents` | Active | Extend with dedicated vendor, invoice_number, sort_by, sort_order, and date helper logic. |
| `FinancialDocumentDB.count_search_documents` | New | Add query matching identical WHERE conditions to compute true pagination totals. |
| `FinancialDocumentDB.edit_financial_document` | New | Atomic update with field-level diff logging (`FIELD_UPDATED`) and transaction rollback. |
| `FinancialDocumentDB.get_audit_history` | New | Fetch chronological timeline of insert, classification, edits, and verification. |
| Filter-aware Analytics | New | Add parameter support (`query`, `category`, `date_from`, `date_to`, `vendor`, etc.) to analytics queries. |
| Streamlit Registry UI | Refactor | Modern multi-criteria search bar, structured filters, clear filters, sort selector, pagination, View/Edit/History tabs, validation. |
| Session State Filter Store | New | Synchronize active filters across Registry, Analytics, and Export pages. |
| Missing Fields Detection | New | Rule-based helper highlighting missing essential fields (Vendor, Date, Amount, Category). |

---

## 7. Potential Risks & Mitigation Strategies

1. **Risk: Database Schema Corruption or Incompatibility**
   * *Mitigation:* Use `PRAGMA table_info(documents)` non-destructive migration pattern already verified in Phase 12. Never drop or recreate tables. Default values ensure full backward compatibility with all Phase 11/12 tests.
2. **Risk: Breaking Existing Phase 11/12 Tests (299 Baseline)**
   * *Mitigation:* Keep existing method signatures intact or backwards-compatible with defaults (`update_document` continues to work; new capabilities exposed via `edit_financial_document` and enhanced optional arguments).
3. **Risk: SQL Injection in Dynamic Queries**
   * *Mitigation:* Use 100% parameterized SQLite queries with dictionary binding (`:param_name`). Never concatenate raw user input into SQL strings.
4. **Risk: Performance Degradation on Large Datasets**
   * *Mitigation:* Push all search, filtering, sorting, and pagination logic down to SQLite using indexes. Avoid loading all rows into Python.
5. **Risk: Unsaved Edits Overwritten by Model Predictions**
   * *Mitigation:* Clear separation of `ai_*` fields and user-edited fields. User edits flag `is_modified = 1` and preserve original extraction values.

---

## 8. Audit Conclusion & Go-Ahead Recommendation

The codebase has clean separation of concerns, robust transaction handling, and well-designed dataclass schemas. Phase 13 enhancements can be implemented cleanly through additive schema fields, SQL index expansion, atomic edit/diff logging, filter-aware database queries, and a streamlined Streamlit interface.

All Phase 1–12 regression baselines (299 tests) will remain strictly protected throughout the implementation.
