# Phase 14 User Workflow Guide

**Document Understanding & Institutional Financial Record Management**  
**Role:** Chief Financial Officer (CFO), Comptroller, Finance Director, Internal Auditor  
**Application:** Financial Document AI (`app_phase11.py`)  
**Feature:** Advanced Financial Analytics & Institutional Dashboard (Page 3)  

---

## 1. Overview & Institutional Workflow

Phase 14 equips financial decision-makers and audit committees with an interactive, multi-dimensional executive dashboard.

The workflow integrates seamlessly with the existing ingestion, search, and human-in-the-loop verification pipeline:

```text
[1] Ingestion & Batch Processing (Page 1)
                  ↓
[2] Document Registry & Human Verification (Page 2)
                  ↓
[3] Global Search & Filter Configuration (Sidebar)
                  ↓
[4] Executive Financial Analytics Dashboard (Page 3)
     ├── Tab A: 🎯 KPI Overview
     ├── Tab B: 📅 Financial Year Analysis (YoY)
     ├── Tab C: 📈 Trend Analysis (MoM & Rolling Avg)
     ├── Tab D: 🍰 Category Intelligence (HHI)
     ├── Tab E: 🏢 Vendor Intelligence (Concentration)
     ├── Tab F: 💰 Tax & Compliance
     ├── Tab G: ⚠️ Anomaly Analytics
     └── Tab H: 📥 Executive Export (7-Sheet Excel)
                  ↓
[5] Audit Sign-Off & Institutional Reporting (Page 5)
```

---

## 2. Step-by-Step Operator Guide to the 8 Dashboard Tabs

### Tab A: 🎯 KPI Overview
1. Navigate to **Financial Analytics** in the sidebar.
2. The KPI Overview displays 8 primary executive metrics:
   - **Total Spend (₹)**: Gross filtered expenditure.
   - **Document Count**: Total number of invoices/bills matching active filters.
   - **Average Bill Size (₹)**: Mean ticket expenditure.
   - **Verified Spend & Count**: Approved expenditure volume.
   - **Pending Spend & Count**: Awaiting clerk verification.
   - **Flagged Spend & Count**: Held for review or potential irregularity.
   - **Verification Rate (%)**: Ratio of verified documents to total.
   - **Effective Tax Coverage**: Percentage of documents containing readable tax line data.
3. Review the verification progress bar to evaluate month-end closing readiness.

---

### Tab B: 📅 Financial Year Analysis (YoY)
1. Select the **Financial Year Analysis** tab.
2. In the **Compare Fiscal Years** selector, choose:
   - **Base Fiscal Year** (e.g., `FY2024-25`).
   - **Comparison Fiscal Year** (e.g., `FY2025-26`).
3. The dashboard calculates:
   - **Spend Variance**: Absolute ₹ change and % growth.
   - **Volume Variance**: Absolute document count delta and % change.
   - **Average Bill Variance**: Fluctuation in unit ticket price.
4. Review the bar chart comparing multi-year spend side-by-side.

---

### Tab C: 📈 Trend Analysis (MoM & Rolling Averages)
1. Select the **Trend Analysis** tab.
2. View the dual-axis chart:
   - **Monthly Spend Bars**: Total monthly expenditure.
   - **MoM Delta Overlay**: Red/green indicators showing spending spikes or reductions.
   - **3-Month Rolling Average Line**: Smoothed spending trajectory to spot macro trends.
3. Filter by category or vendor in the sidebar to isolate spending velocity for specific budget lines (e.g., Electricity or IT Equipment).

---

### Tab D: 🍰 Category Intelligence & Concentration
1. Select the **Category Intelligence** tab.
2. Review the **Herfindahl-Hirschman Index (HHI)**:
   - `< 1,500`: Low concentration — well-diversified institutional spending.
   - `1,500 – 2,500`: Moderate concentration.
   - `> 2,500`: High concentration — spending is heavily dominated by 1 or 2 categories.
3. Observe the **Diversity Score** ($0-100$) and Top-1/Top-3 expenditure shares.
4. Inspect the interactive Donut Chart to visualize relative budget allocations.

---

### Tab E: 🏢 Vendor Intelligence & Concentration Risk
1. Select the **Vendor Intelligence** tab.
2. Review the Pareto ranking of top vendors by expenditure.
3. Identify **Concentration Risk Warnings**:
   - Any vendor consuming $> 25\%$ of total departmental spend is automatically flagged with an alert banner.
4. Review invoice frequency vs. total spend to distinguish high-volume/low-value suppliers from high-value/low-volume contractors.

---

### Tab F: 💰 Tax & Compliance
1. Select the **Tax & Compliance** tab.
2. Check **Tax Data Availability**:
   - High availability ($>90\%$) indicates strong OCR extraction and vendor compliance.
   - Low availability indicates receipts lacking tax breakdown or manual capture gaps.
3. Inspect the **Effective Tax Rate**: Average ratio of tax to total amount across taxable bills.
4. Review the **Zero-Tax Invoices** table for audit scrutiny (verify whether exemptions, composite schemes, or unregistered dealers are valid).

---

### Tab G: ⚠️ Anomaly Analytics & Completeness
1. Select the **Anomaly Analytics** tab.
2. Adjust the statistical threshold ($2.0\sigma$ or $3.0\sigma$).
3. Examine:
   - **Statistical Outliers**: Invoices whose total amount deviates significantly from the mean, complete with Z-scores.
   - **Duplicate Invoices**: Flagged documents sharing identical invoice numbers across different dates or files.
   - **Data Completeness Matrix**: Field-by-field completeness rates (Vendor Name, Date, Invoice #, Total Amount, Tax, Address) to monitor OCR reliability.

---

### Tab H: 📥 Executive Export (7-Sheet Excel)
1. Select the **Executive Export** tab.
2. Ensure desired global filters (e.g., Fiscal Year or Category) are configured in the sidebar.
3. Click **Generate & Download Dashboard Excel**.
4. The system compiles and delivers a formatted `.xlsx` workbook containing:
   - *Summary KPIs*
   - *Category Analysis*
   - *Vendor Intelligence*
   - *Tax Compliance*
   - *Monthly Spend Trend*
   - *Anomaly Report*
   - *FY Comparison*
5. Submit the generated workbook directly to the audit committee or executive board.
