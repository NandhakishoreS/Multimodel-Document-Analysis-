# Phase 15 User Workflow Guide

**Document Understanding & Institutional Financial Record Management**  
**Phase:** 15 — Duplicate & Anomaly Detection Intelligence  
**Role:** Senior Financial Auditor, Compliance Reviewer, Risk Analyst, Operations Lead  
**Application Interface:** Financial Document AI (`app/phase11_app.py` — Page 4)  

---

## 1. End-to-End Investigation Lifecycle

The Phase 15 intelligence center provides a structured, human-in-the-loop lifecycle for financial discrepancies:

```text
       [1] Ingestion Detection Hook / Manual Rescan
                         ↓
       [2] Multi-Signal Anomaly & Duplicate Detection
                         ↓
       [3] Anomaly Queue Discovery & Severity Prioritization
                         ↓
       [4] Record Inspection & Detailed Evidence Review
                         ↓
       [5] Reviewer Disposition (OPEN → UNDER_REVIEW → RESOLVED / DISMISSED)
                         ↓
       [6] Immutable Audit Logging of Findings
                         ↓
       [7] Side-by-Side Duplicate Candidate Inspection
                         ↓
       [8] Duplicate Confirmation & Record Flagging
                         ↓
       [9] Multi-Format Executive & Audit Export (Excel / CSV / JSON)
```

---

## 2. Operator Workflow Walkthrough

### Step 1: Navigating to the Intelligence Center
1. Launch the application:
   ```powershell
   streamlit run app.py
   ```
2. In the left sidebar navigation, click on **🚨 Anomaly Detection** (Page 4).
3. Observe the three-tab console:
   - **Tab 1: Anomaly Review Queue**
   - **Tab 2: Duplicate Candidates**
   - **Tab 3: Integrity Analytics & Export**

---

### Step 2: Reviewing Anomaly Findings (Tab 1)
1. **Executive Metric Cards**:
   - Inspect the summary cards: *Total Findings*, *Open Findings*, *Under Review*, *Resolved*, and *High / Critical Alerts*.
2. **Filter the Findings**:
   - **Severity Filter**: Select `All`, `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, or `INFO`. Priority should always be given to `CRITICAL` and `HIGH`.
   - **Status Filter**: Filter by `All`, `OPEN`, `UNDER_REVIEW`, `RESOLVED`, or `DISMISSED`.
3. **Select a Finding for Inspection**:
   - Locate the finding in the interactive dataframe.
   - Note the **Finding ID**, **Document ID**, **Type** (`ANOMALY_001` - `ANOMALY_010`), **Severity**, and **Status**.
   - Use the **Select Finding ID to Inspect / Action** dropdown to load its full dossier.

---

### Step 3: Inspecting Anomaly Evidence & Document Context
1. In the **Finding Dossier & Inspection** card:
   - Review the **Anomaly Type** description and human-readable **Explanation**.
   - Inspect the **Severity** and computed **Score**.
   - Check the **Document Details**: Document ID, Vendor, Invoice #, Invoice Date, Total Amount, and current Verification Status.
   - Review **Signal Evidence Details** (e.g., $Z$-score, baseline average, missing field names, tax delta, temporal growth percentage).

---

### Step 4: Adjudicating Finding Disposition
1. In the **Update Finding Status & Audit Notes** section:
   - Select the target **New Status**:
     - `OPEN`: Uninvestigated finding.
     - `UNDER_REVIEW`: Actively being investigated with vendor or internal department.
     - `RESOLVED`: Discrepancy explained, corrected, or verified as legitimate.
     - `DISMISSED`: Determined to be a benign edge case or false positive.
   - Enter the **Reviewer Name** (e.g., `Auditor J. Doe`).
   - Enter detailed **Review Notes / Rationale** (mandatory for audit compliance).
2. Click **Submit Finding Review**:
   - The status is updated in the database.
   - An immutable entry is appended to `anomaly_review_history`.
   - The review queue instantly refreshes.

---

### Step 5: Reviewing Audit History Timeline
1. Scroll down to the **Audit History Timeline** within the dossier.
2. View the chronological log of all transitions on this finding:
   - Timestamp, Reviewer, Previous Status $\rightarrow$ New Status, and Review Notes.

---

### Step 6: Resolving Duplicate Candidates (Tab 2)
1. Switch to **Tab 2: Duplicate Candidates**.
2. **Review Candidates**:
   - Inspect candidate pairs across the 5 detection tiers:
     - `EXACT_FILE_DUPLICATE`: Identical file SHA-256 hash.
     - `DUPLICATE_INVOICE_ID`: Same vendor and invoice number on different documents.
     - `DUPLICATE_FINANCIAL_SIGNATURE`: Same vendor, total amount, and date.
     - `STRONG_DUPLICATE_CANDIDATE`: High composite score ($\ge 0.85$).
     - `NEAR_DUPLICATE_RECORD`: Same vendor, amount, and date within $\pm 3$ days.
3. **Side-by-Side Comparison**:
   - In the **Side-by-Side Candidate Inspection** selector, pick a Candidate ID.
   - Examine the comparison view:
     - Primary vs Matched: Document ID, Vendor, Invoice #, Date, Total Amount.
     - Match Type, Confidence Score, and Match Breakdown details.
4. **Actioning Duplicates**:
   - **Confirm Duplicate**:
     - Click **Confirm Duplicate**.
     - Marks the candidate status as `CONFIRMED_DUPLICATE`.
     - Automatically updates the primary document's verification status to `flagged` in the registry.
     - Records an audit log entry.
   - **Dismiss Candidate**:
     - Click **Dismiss Candidate**.
     - Marks candidate status as `DISMISSED` (e.g., recurring monthly subscription with identical amount).
     - Records audit log entry with reviewer explanation.

---

### Step 7: Integrity Analytics & Rescanning (Tab 3)
1. Switch to **Tab 3: Integrity Analytics & Export**.
2. **Visual Analytics**:
   - View the **Findings by Severity** distribution chart (identifying concentration of High/Critical issues).
   - View the **Findings by Anomaly Type** chart (identifying systematic errors, e.g. frequent tax mismatches).
3. **Rescan Operations**:
   - Under **Anomaly Engine Controls**, review current scan metadata.
   - Click **Run Full Database Rescan**:
     - Evaluates all active documents against `ANOMALY_001` - `ANOMALY_010` and duplicate detection rules.
     - **Idempotency Guarantee**: Any findings previously marked as `UNDER_REVIEW`, `RESOLVED`, or `DISMISSED` retain their human status and audit history.

---

### Step 8: Multi-Format Institutional Exports (Tab 3)
1. Under **Multi-Format Export**:
2. **Executive Excel Workbook (.xlsx)**:
   - Click **Download Anomaly & Duplicate Report (Excel)**.
   - Generates a styled 3-sheet workbook:
     - *Sheet 1: Anomaly Findings* (color-coded severities, finding metadata).
     - *Sheet 2: Duplicate Candidates* (match types, primary vs matched details).
     - *Sheet 3: Review Audit History* (full investigation logs and timestamps).
3. **CSV Export (.csv)**:
   - Click **Download Findings (CSV)** for spreadsheet analysis.
4. **JSON Export (.json)**:
   - Click **Download Findings (JSON)** for machine-to-machine ingestion into enterprise SIEM, ERP, or governance systems.

---

## 3. Responsible Analytics & Reviewer Principles

1. **Non-Defamatory Framing**: Findings represent mathematical outliers, duplicate candidates, or data discrepancies. Never record accusations of fraud or misconduct in review notes.
2. **Contextual Investigation**: A high $Z$-score may represent an authorized annual capital expenditure or enterprise license renewal. Always cross-reference with purchase orders and approvals before confirming flags.
3. **Immutable History**: Once a review note is submitted, it becomes part of the permanent audit trail. Ensure notes are professional, factual, and complete.
