# Phase 13 User Workflow Guide

**Document Understanding & Institutional Financial Record Management**  
**Role:** Finance Manager, Comptroller, Senior Auditor, or Operations Clerk  
**Application:** Financial Document AI (`app_phase11.py`)  

---

## 1. End-to-End Institutional User Journey

The institutional lifecycle follows a disciplined, human-in-the-loop progression:

```text
       [1] Ingestion & Batch Processing
                       ↓
       [2] Discovery (Search & Filter)
                       ↓
       [3] Sorting & Controlled Pagination
                       ↓
       [4] Record Inspection & Missing Field Audit
                       ↓
       [5] Record Correction & Validation
                       ↓
       [6] Audit Event Logging & AI Preservation
                       ↓
       [7] Verification Sign-Off
                       ↓
       [8] Filtered Financial Analytics
                       ↓
       [9] Multi-Format Institutional Reporting
```

---

## 2. Step-by-Step Operator Instructions

### Step 1: Ingesting Documents (`🏠 Process Documents`)
1. Navigate to **Process Documents** from the left navigation bar.
2. Select your document domain:
   * **SROIE**: General institutional utility bills, retail receipts, fuel slips.
   * **CORD**: Restaurant, hotel catering, and dining receipts with line items.
   * **FUNSD**: Forms, administrative slips, and agreements.
3. Drag & drop or upload one or multiple documents (`.pdf`, `.png`, `.jpg`, `.jpeg`).
4. Click **Start Batch Ingestion**.
5. The system automatically computes SHA-256 cryptographic hashes to skip duplicate bills, extracts key financial entities via LayoutLMv3, and classifies the bill category with explainability evidence.

---

### Step 2: Finding Documents (`📋 Document Registry`)
1. Navigate to **Document Registry**.
2. **General Keyword Search**:
   * Type any partial term in the keyword search bar: e.g., `electricity`, `tangedco`, `INV-2026`, `45000`.
   * The search scans vendor names, filenames, addresses, invoice numbers, categories, and raw OCR text.
3. **Structured Filters**:
   * **Category**: Filter by any of the 14 standard institutional categories (e.g., `Electricity`, `Water`, `Office Supplies`, `Maintenance`).
   * **Date**: Select your preferred date mode:
     - *Date Range*: Enter start and end dates (e.g., `2026-03-01` to `2026-03-31`).
     - *Exact Date*: Search for a specific transaction date.
     - *Month*: Target a calendar month (e.g., `2026-03`).
     - *Year*: Target an entire year (e.g., `2026`).
     - *Fiscal Year (FY)*: Select an institutional fiscal year (e.g., `FY 2025-26` covering April 1, 2025 – March 31, 2026).
   * **Vendor**: Enter exact or partial vendor name (e.g., `Airtel`).
   * **Invoice Number**: Target a specific bill number.
   * **Amount Range**: Specify minimum and maximum budget limits (e.g., Min `₹10,000`, Max `₹50,000`).
   * **Verification Status**: Filter by `Pending`, `Verified`, `Modified`, `Flagged`, or `Rejected`.
   * **Review Queue**: Filter specifically for documents flagged with `Requires Review Only` (low AI confidence or unclassified).
4. Click **Refresh** to apply filters, or **Clear Filters** to instantly reset to all records.

---

### Step 3: Navigating Results & Pagination
1. Review the result count bar:
   `Showing 1–25 of 1,247 documents (Page 1 of 50)`
2. Sort your records by clicking the **Sort By** dropdown:
   * Options: *Processing Date*, *Invoice Date*, *Total Amount*, *Vendor Name*, *Category*, *Confidence*, or *Status*.
   * Select **Descending** (newest / highest first) or **Ascending** (oldest / lowest first).
3. Use `[⏮ First]`, `[◀ Prev]`, and `[Next ▶]` buttons to page through records smoothly without loading the entire database into memory.
4. Adjust page size (10, 25, 50, 100) according to your screen preference.

---

### Step 4: Inspecting Document Details
1. Select a document from the dropdown selector beneath the table.
2. Review the **Missing Fields Warning**:
   * If required metadata (Vendor Name, Date, Amount, or Category) could not be extracted, the system displays an alert: `⚠️ Missing Required Financial Fields: Vendor Name, Invoice Date`.
3. Open **Tab 1: 👁️ View & Traceability**:
   * **AI Baseline vs Final Verified Values**:
     Compare what the AI model extracted side-by-side with what is currently stored.
   * **Classification Reason & Evidence**: Review the exact keyword tokens that led to the category decision.
   * **Line Items**: Inspect line-by-line product descriptions, unit prices, and totals.
   * **Technical Traceability**: Review SHA-256 hash, duration, and full OCR raw text.

---

### Step 5: Editing & Correcting Financial Records
1. Switch to **Tab 2: ✏️ Edit Financial Record**.
2. Correct any misread or missing fields:
   * Vendor Name
   * Invoice Date (format: `YYYY-MM-DD`)
   * Invoice Number
   * Total Amount (enforces non-negative values)
   * Subtotal and Tax Amount
   * Category (choose from the controlled institutional taxonomy)
   * Currency (`INR`, `USD`, `EUR`, `GBP`)
   * Verification Status (`pending`, `verified`, `modified`, `flagged`, `rejected`)
3. Provide audit accountability:
   * **Modified By**: Enter your username, employee ID, or role (e.g., `audit_lead_04`).
   * **Reason for Change**: Document why the correction was made (e.g., `Corrected OCR vendor misread from receipt banner`).
   * **Additional Notes**: Add any supplementary audit context.
4. Click **💾 Save Changes & Record Audit Event**.
5. The system performs validation:
   * If any amount is negative, saving is blocked with a clear warning.
   * If valid, the database is atomically updated, the document is marked with `✏️ Modified`, and detailed field-level diffs are written to the immutable processing log.

---

### Step 6: Reviewing the Audit Trail
1. Switch to **Tab 3: 📜 Audit History Timeline**.
2. Review the chronological change log:
   * Ingestion timestamp and initial AI values.
   * Every edit event with the exact field changed:
     `Old: ₹45,200.00 ➔ New: ₹45,800.00`
   * Editor username and stated audit reason.
3. This audit trail is permanent and tamper-evident; it cannot be erased by subsequent edits.

---

### Step 7: Verifying & Signing Off Records
1. Once all corrections have been made and verified against the original bill, select Status as **Verified** in the edit form (or use the quick bulk action **Mark Current Page as Verified**).
2. The record badge turns green (`VERIFIED`), and any review alerts are cleared.

---

### Step 8: Filtered Financial Analytics (`📊 Financial Analytics`)
1. Navigate to **Financial Analytics**.
2. If filters were active in the Document Registry, a notification appears:
   `🔍 Displaying metrics for filtered dataset: Category: Electricity | Month: 2026-03`
3. All KPIs, spend distribution charts, top vendor rankings, and monthly trend graphs automatically update to reflect **only the filtered documents**.
4. Click **Clear Filters** if you wish to analyze the complete institutional expenditure across the whole organization.

---

### Step 9: Exporting Institutional Reports (`📤 Export & Reports`)
1. Navigate to **Export & Reports**.
2. Choose your **Export Scope**:
   * **Use Active Filters from Document Registry**: Exports strictly the documents you searched/filtered.
   * **Export All Database Records**: Generates an export of the entire database.
3. Download in your required format:
   * **JSON**: Complete hierarchical data structure including line items and audit metadata.
   * **CSV**: Tabular format compatible with Microsoft Excel, Google Sheets, and ERP tools (includes AI baselines and modification flags).
   * **Excel (`.xlsx`)**: Multi-sheet workbook with Document Records, Line Items, and Category Summary.
   * **PDF Report**: Executive-ready formatted financial summary with KPI cards, category breakdown table, and document inventory.
