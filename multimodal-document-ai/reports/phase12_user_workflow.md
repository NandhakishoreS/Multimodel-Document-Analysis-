# Phase 12 Institutional User Workflow Guide
## End-to-End Operational Guide for Financial Document Ingestion & Reporting

**Project**: Multimodal Transformer-Based Intelligent Document Understanding and Information Extraction  
**Version**: Phase 12 (Production Ingestion & Classification Pipeline)

---

## 1. Overview of the Institutional Workflow

Phase 12 provides enterprise accounting, auditing, and administrative teams with a unified interface to ingest, validate, extract, classify, verify, and report financial documents.

The complete operational cycle follows a 7-stage pipeline:

```text
  1. Upload Bills
        ↓ (PDF, PNG, JPG, JPEG)
  2. Ingest & Validate
        ↓ (Format check, size limit, SHA-256 deduplication)
  3. Batch Process
        ↓ (OCR → LayoutLMv3 Inference → Normalization)
  4. Classify & Review
        ↓ (14-category taxonomy, explainability evidence, confidence threshold)
  5. Correct & Audit
        ↓ (Manual category override, audit history preservation)
  6. Search & Filter
        ↓ (Category, review status, date, amount, vendor)
  7. Analyze & Export
        ↓ (Interactive Plotly analytics, PDF executive reports, Excel, CSV, JSON)
```

---

## 2. Step-by-Step User Workflow

### Step 1: Upload Bills & Documents
1. Launch the application:
   ```bash
   streamlit run app_phase11.py
   ```
2. Navigate to **🏠 Process Documents** from the sidebar navigation.
3. In the **Processing Configuration** panel:
   - **Document Domain**: Select `SROIE (Receipts & Bills)` for standard retail receipts and utility bills, `CORD` for dining receipts, or `FUNSD` for tabular forms.
   - **Duplicate Policy**:
     - `Skip Duplicate (Recommended)`: Skips redundant OCR/inference if an identical file hash is found in the database.
     - `Warn & Flag`: Processes the document but adds a duplicate anomaly flag and routes it to manual review.
     - `Process Anyway`: Processes and creates a new record without warning.
   - **Classification Review Threshold**: Set the confidence slider (default `0.60`). Any classification with confidence below this threshold is flagged for human review.
   - **Document Currency**: Select the primary document currency (`INR`, `USD`, `EUR`, `GBP`).
4. In the **Upload Documents** area:
   - Drag and drop or browse files. Multi-format support accepts `.pdf`, `.png`, `.jpg`, `.jpeg`.
   - The **Ingestion Queue Preview** table immediately displays:
     - Filename and detected format
     - File size in KB
     - Integrity validation status (`✅ Valid` or `❌ Rejected`)
     - Duplicate status preview (`✨ New` or `⚠️ Duplicate (doc_id)`)
     - First 12 characters of the cryptographic SHA-256 digest

---

### Step 2: Batch Processing & Execution
1. Click **🚀 Process Documents Batch**.
2. The batch processor coordinates execution with **independent document fault isolation**:
   - Each document proceeds through `VALIDATING` → `OCR` → `EXTRACTING` → `CLASSIFYING` → `SAVING` → `COMPLETED`.
   - If a file is corrupted or unreadable, only that file receives `FAILED` status; all other valid files continue processing without interruption.
3. Real-time stage progress updates appear on screen (`Processing 3/10: invoice.pdf — Stage: OCR`).
4. Upon completion, the **Batch Execution Summary** presents high-level metrics:
   - Total Uploaded, Completed, Duplicates, Failed, Rejected, and Requires Review count.
   - Total processing time and average latency per document.
   - Color-coded Category Breakdown pills.

---

### Step 3: Review Flagged & Low-Confidence Records
1. In the **Processed Document Records** list, inspect individual document cards.
2. Documents requiring human attention are prominently highlighted with a `⚠️ Needs Review` badge:
   - **Low Confidence**: Triggered when classification confidence is below the configured threshold (e.g. `0.42 < 0.60`).
   - **Unknown Category**: Triggered when diagnostic keywords are insufficient to disambiguate the bill category.
   - **Potential Duplicate**: Triggered when an exact SHA-256 hash or identical financial fields (vendor, invoice #, date, total) match an existing record.
3. View the transparent explainability card:
   - Diagnostic keywords that drove the model's decision (e.g., `tangedco`, `kwh`, `meter`, `energy`).
   - Human-readable explainability reason.

---

### Step 4: Correct Classifications & Audit Trail
1. To correct an erroneous or ambiguous category:
   - Expand the document card.
   - Under **Manual Classification Review & Correction**, select the intended category from the dropdown (e.g., change `Miscellaneous` to `Maintenance`).
   - Provide an optional audit comment in the text field (e.g., *"Verified physical bill; HVAC filter replacement"*).
   - Click **💾 Apply Manual Category Override**.
2. **Audit Preservation Invariant**:
   - The initial AI classification (`ai_category`) and AI confidence (`ai_confidence`) are **permanently preserved**.
   - The operational category (`category` and `final_category`) is updated.
   - The `requires_review` flag is cleared.
   - A `CLASSIFICATION_MODIFIED` record is permanently logged in the SQLite `processing_log` table with the operator's ID, timestamp, previous category, and new category.

---

### Step 5: Search & Filter in the Document Registry
1. Navigate to **📋 Document Registry** from the sidebar.
2. Use multi-parameter search filters:
   - **Text Query**: Search across vendor name, filename, OCR text, address, or invoice number.
   - **Category Filter**: Filter by any of the 14 institutional categories.
   - **Verification Status**: Filter by `pending`, `verified`, `flagged`, or `rejected`.
   - **Review Status**: Quick toggle between `All`, `Requires Review Only`, or `No Review Needed`.
   - **Amount Range**: Filter between minimum and maximum spend amounts.
3. Click on any record to open the detailed inspector:
   - Compare `AI Baseline Category` against `Final Category`.
   - Inspect line items, subtotal, tax amount, and currency.
   - Update verification status or delete corrupted records.

---

### Step 6: Analyze Institutional Spend
1. Navigate to **📊 Financial Analytics** from the sidebar.
2. Review interactive Plotly charts:
   - **Monthly Spend Trend**: Line chart tracking expenditures over time.
   - **Spend by Category**: Donut and bar charts showing proportional allocation (e.g., Utilities vs Maintenance vs Travel).
   - **Top Vendors by Spend**: Pareto ranking of major institutional suppliers.
   - **Verification Status Distribution**: Breakdown of verified vs pending review documents.

---

### Step 7: Export & Reporting
1. Navigate to **📤 Export & Reports** from the sidebar.
2. Select desired document filters (date range, category, or status).
3. Choose the export format:
   - **PDF Executive Summary**: Download a structured, print-ready institutional report featuring KPI summary tiles, category spend tables, and document registry listings with Latin-1 sanitization.
   - **Excel Spreadsheet (`.xlsx`)**: Full tabular export with nested line items for corporate ERP integration.
   - **CSV (`.csv`)**: Lightweight data stream for financial pipelines.
   - **JSON (`.json`)**: Complete hierarchical schema export including OCR tokens and confidence scores.

---

## 3. Operational Invariants & Safeguards

| Component | Invariant | Safeguard Implementation |
| :--- | :--- | :--- |
| **Model Weights** | No Retraining | Checkpoints in `models/checkpoints/` are read-only. |
| **Batch Fault Tolerance** | Independent Failures | Each file is isolated in a `try/except` block; corrupt files never crash the queue. |
| **Data Persistence** | Non-Destructive Migration | `PRAGMA table_info` checks column existence before running safe `ALTER TABLE ADD COLUMN`. |
| **Audit Trail** | Immutable Modifications | Classification overrides append `CLASSIFICATION_MODIFIED` entries rather than overwriting history. |
| **Deduplication** | Cryptographic Reliability | SHA-256 hashing detects exact duplicate files regardless of filename changes. |
