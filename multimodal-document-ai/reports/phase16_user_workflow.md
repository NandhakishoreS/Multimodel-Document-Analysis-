# Phase 16 User Workflow Guide

**Document Understanding & Institutional Financial Record Management**  
**Phase:** 16 — Dynamic Institutional Reporting & Executive Document Generation  
**Role:** Chief Financial Officer (CFO), Lead Auditor, Operations Controller, Compliance Director  
**Application Interface:** Financial Document AI (`app/phase11_app.py` — Page 5)  

---

## 1. End-to-End Institutional Reporting Workflow

The Dynamic Institutional Reporting Center allows users to customize, preview, reconcile, and publish formal executive financial dossiers:

```text
┌───────────────────────────────────────────────────────────────────────────┐
│                      [1] Document Registry Filtering                      │
│            (Optional: Set filters on Category, Vendor, FY, etc.)          │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│            [2] Navigate to Page 5: 📤 Export & Reports                     │
│               Select Tab 1: 🏛️ Institutional Intelligence Reports          │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                     [3] Select Report Archetype & Scope                   │
│   - Archetype: Executive / FY / Vendor / Category / Tax / Anomaly / Audit │
│   - Customize Title, Organization ("Acme Corp"), Auditor Remarks          │
│   - Scope: Active Registry Filters / Full Database / Ad-Hoc Filter        │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    [4] Inspect Live Interactive Preview                   │
│   - Metadata Chips: Unique Report ID, Scoped Document Count, Currency     │
│   - Reconciliation Status: Mathematical validation (Category/Monthly sum) │
│   - Executive Observations: Deterministic narrative bullets (Zero LLM)    │
│   - Key Indicators: Total Spend, Average Invoice, Verification Rate       │
│   - Data Preview Expander: Scoped category, vendor, or discrepancy tables │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                     [5] Generate & Distribute Artifacts                   │
│   - 📑 Executive PDF Brief (Publication-grade multi-page PDF)             │
│   - 📗 Institutional Excel (11-sheet workbook, formula-injection safe)    │
│   - 📊 Scoped CSV Table (Normalized UTF-8 records)                        │
│   - 📄 Institutional JSON (Complete machine-readable payload)             │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Operator Walkthrough

### Step 1: Navigating to the Reporting Center
1. Open your browser to the running Streamlit application:
   ```powershell
   streamlit run app.py
   ```
2. In the left sidebar navigation menu, select **📤 Export & Reports** (Page 5).
3. Select **Tab 1: 🏛️ Institutional Intelligence Reports** (selected by default).

---

### Step 2: Configuring Report Parameters & Scope
1. **Choose a Report Archetype**:
   - **Executive Financial Summary**: High-level spending, vendor exposure, category shares, and integrity alerts.
   - **Fiscal Year Performance & Trends**: Indian fiscal year comparison (April 1 – March 31) with YoY variance.
   - **Vendor Exposure & Concentration Intelligence**: Pareto distribution of vendor spend and concentration alerts for vendors representing $>25\%$ of spend.
   - **Category Distribution & HHI Diversity**: Categorical breakdown with Herfindahl-Hirschman Index (HHI) concentration interpretation.
   - **Tax Compliance & Data Quality**: Tax coverage %, effective tax rates, and zero-tax documentation.
   - **Anomaly & Duplicate Investigation Dossier**: Centralized record of open anomaly findings and duplicate candidates.
   - **Document Lifecycle & Decision Audit Trail**: Complete historical record of manual edits, status transitions, and auditor notes.
2. **Review & Customize Metadata**:
   - **Report Title**: Pre-filled with standard title; editable for specific corporate purposes (e.g., *"Q4 FY26 Board Audit Brief"*).
   - **Issuing Entity / Organization**: Defaults to *"Institutional Finance Operations"*; replace with your department or firm name.
   - **Auditor / Executive Remarks**: Add notes such as *"Prepared for the Internal Audit Committee meeting on March 15, 2026."*
3. **Select Analytical Scope**:
   - **Use Active Filters from Document Registry**: Automatically pulls any search, category, status, or date filters currently set on Page 2.
   - **Full Database Population**: Compiles metrics across all documents in the database.
   - **Custom Ad-Hoc Filter**: Expands inline filters to select Category, Verification Status, or Benchmark Domain (SROIE, CORD, FUNSD).

---

### Step 3: Validating Reconciliation & Executive Observations
1. **Check Mathematical Reconciliation Banner**:
   - A green banner `✅ Reconciliation Status: PASS` indicates that the sum of category spend and monthly spend exactly reconciles with total expenditure within a 1.00 INR tolerance.
   - If a yellow banner appears, examine the reported variance notes (e.g., invoices with missing dates or mixed foreign currencies).
2. **Review Deterministic Executive Observations**:
   - Review each bullet point. These observations are calculated algorithmically from verified database numbers without generative LLMs.
   - Note key insights:
     - Top spending category and percentage share.
     - Top vendor exposure and any concentration risk flags ($>25\%$).
     - Month-over-Month (MoM) spending velocity and percentage changes.
     - Annual Fiscal Year (YoY) shifts.
     - Open anomaly alerts and duplicate pairs requiring human sign-off.
3. **Inspect Key Performance Indicator Cards**:
   - Total Spend (formatted in Indian numbering format: e.g., `₹53,19,875.00`).
   - Total Documents Scoped.
   - Average Bill Value.
   - Verification Rate percentage.
4. **Expand Data Preview**:
   - Click **🔍 View Scoped Analytical Tables** to preview category allocations, vendor Pareto rankings, or open findings before downloading.

---

### Step 4: Exporting Publication-Grade Artifacts
In the **5. Institutional Export Artifacts** section, select the desired file format:

1. **📑 Download Institutional PDF**:
   - Produces a multi-page PDF document formatted with a formal corporate header, report metadata, running headers and footers with `Page X of Y` numbering, KPI grid, formatted tables, and an institutional methodology appendix.
   - Ideal for formal executive briefs, board meetings, and external audit distribution.
2. **📗 Download Institutional Excel**:
   - Produces an 11-sheet workbook (`.xlsx`) containing:
     - `Executive Summary`
     - `Financial Records`
     - `Category Analysis`
     - `Vendor Intelligence`
     - `Monthly Trend`
     - `Fiscal Year Analysis`
     - `Tax Analysis`
     - `Anomaly Findings`
     - `Duplicate Candidates`
     - `Audit History`
     - `Report Metadata`
   - **Security Guarantee**: All cells starting with `=`, `+`, `-`, or `@` are automatically escaped to prevent Excel formula injection attacks.
3. **📊 Download Scoped CSV**:
   - Flat tabular export of scoped document records encoded in UTF-8, suitable for ingestion into BI systems (Tableau, PowerBI) or data lakes.
4. **📄 Download Full JSON**:
   - Complete machine-readable JSON structure containing all metadata, KPIs, narrative bullets, reconciliation diagnostics, and row-level document records.

---

### Step 5: Accessing Raw Data Dumps (Tab 2)
If you require raw, unaggregated document dumps as supported in Phase 11:
1. Click **Tab 2: 📦 Raw Document Data Dumps**.
2. Select your scope and download standard flat JSON, CSV, 3-sheet Excel, or basic PDF dumps.
