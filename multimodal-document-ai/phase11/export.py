"""
Phase 11 — Export Engine.

Provides multi-format export of FinancialDocument collections:
    - JSON  (standard json module)
    - CSV   (pandas)
    - Excel (openpyxl via pandas)
    - PDF   (fpdf2)

All export functions accept a list of FinancialDocument objects and either
return a string/bytes payload or write directly to a filepath.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

def _safe_text(s: Any) -> str:
    """Sanitize text to be latin-1 encodable for FPDF core fonts."""
    if s is None:
        return ""
    text = str(s)
    replacements = {
        "\u2014": "-",   # em dash
        "\u2013": "-",   # en dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2022": "*",   # bullet
        "\u2026": "...", # ellipsis
        "\u20b9": "INR ", # Indian Rupee
        "\u20ac": "EUR ", # Euro
        "\u00a3": "GBP ", # Pound
        "\u2265": ">=",  # greater than or equal
        "\u2264": "<=",  # less than or equal
        "\u03c3": "sigma", # sigma
        "\u03bc": "mu",    # mu
        "\u00d7": "x",     # times
        "\u0394": "Delta", # Delta
        "\u2248": "~",     # approx
        "\u00b1": "+/-",   # plus/minus
    }
    for char, rep in replacements.items():
        text = text.replace(char, rep)
    return text.encode("latin-1", "replace").decode("latin-1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_doc(doc: "FinancialDocument") -> Dict[str, Any]:  # type: ignore[name-defined]
    """Convert a FinancialDocument into a flat dict for tabular export."""
    return {
        "document_id":               doc.document_id,
        "filename":                  doc.filename,
        "document_type":             doc.document_type,
        "benchmark_domain":          doc.benchmark_domain,
        "category":                  doc.category,
        "sub_category":              doc.sub_category or "",
        "vendor_name":               doc.vendor_name or "",
        "invoice_number":            doc.invoice_number or "",
        "invoice_date":              doc.invoice_date or "",
        "due_date":                  doc.due_date or "",
        "subtotal":                  doc.subtotal if doc.subtotal is not None else "",
        "tax_amount":                doc.tax_amount if doc.tax_amount is not None else "",
        "total_amount":              doc.total_amount if doc.total_amount is not None else "",
        "currency":                  doc.currency,
        "extraction_confidence":     doc.extraction_confidence,
        "classification_confidence": doc.classification_confidence,
        "ai_category":               getattr(doc, "ai_category", "") or "",
        "ai_confidence":             getattr(doc, "ai_confidence", "") or "",
        "ai_vendor_name":            getattr(doc, "ai_vendor_name", "") or "",
        "ai_total_amount":           getattr(doc, "ai_total_amount", "") if getattr(doc, "ai_total_amount", None) is not None else "",
        "is_modified":               getattr(doc, "is_modified", False),
        "verification_status":       doc.verification_status,
        "address":                   (doc.address or "").replace("\n", " "),
        "line_items_count":          len(doc.line_items),
        "ocr_word_count":            doc.ocr_word_count,
        "processing_timestamp":      doc.processing_timestamp,
        "notes":                     doc.notes or "",
        "anomaly_flags":             ", ".join(doc.anomaly_flags) if doc.anomaly_flags else "",
    }


# ---------------------------------------------------------------------------
# Public exporter class
# ---------------------------------------------------------------------------

class DocumentExporter:
    """
    Multi-format export engine for FinancialDocument collections.

    Usage
    -----
    exporter = DocumentExporter()
    csv_str  = exporter.to_csv(docs)
    exporter.to_excel(docs, Path("report.xlsx"))
    exporter.to_pdf_report(docs, analytics, Path("report.pdf"))
    """

    # -----------------------------------------------------------------------
    # JSON
    # -----------------------------------------------------------------------

    def to_json(self, docs: List[Any], indent: int = 2) -> str:
        """Export documents as a JSON array string."""
        payload = [d.to_dict() for d in docs]
        return json.dumps(payload, indent=indent, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------------

    def to_csv(self, docs: List[Any]) -> str:
        """Export documents as a CSV string (pandas-backed)."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas is required for CSV export.") from exc

        rows = [_flatten_doc(d) for d in docs]
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)

    def to_csv_bytes(self, docs: List[Any]) -> bytes:
        return self.to_csv(docs).encode("utf-8")

    # -----------------------------------------------------------------------
    # Excel
    # -----------------------------------------------------------------------

    def to_excel(self, docs: List[Any], filepath: Optional[Path] = None) -> bytes:
        """
        Export documents to an Excel (.xlsx) workbook.

        Sheets:
            Documents   — flattened document table
            Line Items  — all line items with parent document_id
            Summary     — category totals pivot

        Returns bytes payload (also writes to filepath if provided).
        """
        try:
            import pandas as pd
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill, Alignment
            from openpyxl.utils.dataframe import dataframe_to_rows
        except ImportError as exc:
            raise RuntimeError("openpyxl and pandas are required for Excel export.") from exc

        wb = Workbook()

        # ---- Sheet 1: Documents ----
        ws_docs = wb.active
        ws_docs.title = "Documents"
        rows = [_flatten_doc(d) for d in docs]
        df_docs = pd.DataFrame(rows) if rows else pd.DataFrame()

        header_fill = PatternFill("solid", fgColor="1E3A5F")
        header_font = Font(color="FFFFFF", bold=True)

        if not df_docs.empty:
            for col_idx, col_name in enumerate(df_docs.columns, start=1):
                cell = ws_docs.cell(row=1, column=col_idx, value=col_name)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center")

            for r_idx, row_data in enumerate(df_docs.itertuples(index=False), start=2):
                for c_idx, val in enumerate(row_data, start=1):
                    ws_docs.cell(row=r_idx, column=c_idx, value=val)

            # Auto-fit columns (approximate)
            for col in ws_docs.columns:
                max_len = max(
                    len(str(cell.value or "")) for cell in col
                )
                ws_docs.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

        # ---- Sheet 2: Line Items ----
        ws_items = wb.create_sheet("Line Items")
        item_rows = []
        for d in docs:
            for li in d.line_items:
                item_rows.append({
                    "document_id": d.document_id,
                    "filename":    d.filename,
                    "description": li.description,
                    "quantity":    li.quantity,
                    "unit_price":  li.unit_price,
                    "total":       li.total,
                    "unit":        li.unit or "",
                })
        df_items = pd.DataFrame(item_rows) if item_rows else pd.DataFrame(
            columns=["document_id", "filename", "description", "quantity", "unit_price", "total", "unit"]
        )
        for col_idx, col_name in enumerate(df_items.columns, start=1):
            cell = ws_items.cell(row=1, column=col_idx, value=col_name)
            cell.fill = header_fill
            cell.font = header_font
        for r_idx, row_data in enumerate(df_items.itertuples(index=False), start=2):
            for c_idx, val in enumerate(row_data, start=1):
                ws_items.cell(row=r_idx, column=c_idx, value=val)

        # ---- Sheet 3: Summary ----
        ws_sum = wb.create_sheet("Category Summary")
        if rows:
            df_summary = (
                pd.DataFrame(rows)[["category", "total_amount"]]
                .replace("", None)
                .dropna(subset=["total_amount"])
                .astype({"total_amount": float})
                .groupby("category")
                .agg(count=("total_amount", "count"), total_spend=("total_amount", "sum"), avg_amount=("total_amount", "mean"))
                .reset_index()
                .sort_values("total_spend", ascending=False)
            )
            for col_idx, col_name in enumerate(df_summary.columns, start=1):
                cell = ws_sum.cell(row=1, column=col_idx, value=col_name)
                cell.fill = header_fill
                cell.font = header_font
            for r_idx, row_data in enumerate(df_summary.itertuples(index=False), start=2):
                for c_idx, val in enumerate(row_data, start=1):
                    ws_sum.cell(row=r_idx, column=c_idx, value=val)

        buf = io.BytesIO()
        wb.save(buf)
        xlsx_bytes = buf.getvalue()

        if filepath is not None:
            Path(filepath).write_bytes(xlsx_bytes)

        return xlsx_bytes

    # -----------------------------------------------------------------------
    # PDF
    # -----------------------------------------------------------------------

    def to_pdf_report(
        self,
        docs: List[Any],
        analytics: Optional[Dict[str, Any]] = None,
        filepath: Optional[Path] = None,
        title: str = "Institutional Financial Document Report",
    ) -> bytes:
        """
        Generate a structured PDF summary report using fpdf2.

        Sections:
            1. Report Header (title, generated timestamp, summary KPIs)
            2. Category Breakdown Table
            3. Document Registry Table (top 50)

        Returns bytes payload (also writes to filepath if provided).
        """
        try:
            from fpdf import FPDF
        except ImportError as exc:
            raise RuntimeError("fpdf2 is required for PDF export. Run: pip install fpdf2") from exc

        analytics = analytics or {}
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # ---- Compute summary stats ----
        total_docs = len(docs)
        total_spend = sum(
            d.total_amount for d in docs if d.total_amount is not None
        )
        verified_count = sum(1 for d in docs if d.verification_status == "verified")
        pending_count  = sum(1 for d in docs if d.verification_status == "pending")
        flagged_count  = sum(1 for d in docs if d.verification_status == "flagged")

        # Category breakdown
        cat_totals: Dict[str, Dict[str, float]] = {}
        for d in docs:
            cat = d.category or "Unknown"
            if cat not in cat_totals:
                cat_totals[cat] = {"count": 0, "spend": 0.0}
            cat_totals[cat]["count"] += 1
            if d.total_amount is not None:
                cat_totals[cat]["spend"] += d.total_amount
        cat_rows = sorted(cat_totals.items(), key=lambda x: x[1]["spend"], reverse=True)

        # ---- Build PDF ----
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Header
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_fill_color(30, 58, 95)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 14, _safe_text(title), fill=True, align="C")
        pdf.ln(16)

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, _safe_text(f"Generated: {now}"), align="C")
        pdf.ln(8)

        # KPI row
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Executive Summary")
        pdf.ln(8)
        pdf.set_font("Helvetica", "", 10)

        kpi_data = [
            ("Total Documents", str(total_docs)),
            ("Total Spend",     f"{total_spend:,.2f}"),
            ("Verified",        str(verified_count)),
            ("Pending",         str(pending_count)),
            ("Flagged",         str(flagged_count)),
        ]
        col_w = 38
        for label, value in kpi_data:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(col_w, 8, _safe_text(label + ":"), border=0)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(col_w, 8, _safe_text(value), border=0)
        pdf.ln(10)

        # Category breakdown table
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Spend by Category")
        pdf.ln(8)

        pdf.set_fill_color(30, 58, 95)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(70, 7, "Category",    fill=True, border=1)
        pdf.cell(30, 7, "Count",       fill=True, border=1, align="C")
        pdf.cell(50, 7, "Total Spend", fill=True, border=1, align="R")
        pdf.ln()

        pdf.set_text_color(0, 0, 0)
        fill_flag = False
        for cat, stats in cat_rows:
            pdf.set_fill_color(245, 248, 255) if fill_flag else pdf.set_fill_color(255, 255, 255)
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(70, 6, _safe_text(cat[:35]), fill=True, border=1)
            pdf.cell(30, 6, str(int(stats["count"])), fill=True, border=1, align="C")
            pdf.cell(50, 6, f"{stats['spend']:,.2f}", fill=True, border=1, align="R")
            pdf.ln()
            fill_flag = not fill_flag

        pdf.ln(6)

        # Document registry (top 50)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, _safe_text(f"Document Registry (showing up to 50 of {total_docs})"))
        pdf.ln(8)

        pdf.set_fill_color(30, 58, 95)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)
        headers_w = [50, 30, 25, 30, 25, 20]
        headers_l = ["Vendor", "Date", "Category", "Amount", "Status", "Domain"]
        for h, w in zip(headers_l, headers_w):
            pdf.cell(w, 7, h, fill=True, border=1, align="C")
        pdf.ln()

        pdf.set_text_color(0, 0, 0)
        fill_flag = False
        for d in docs[:50]:
            pdf.set_fill_color(245, 248, 255) if fill_flag else pdf.set_fill_color(255, 255, 255)
            pdf.set_font("Helvetica", "", 7)
            vals = [
                _safe_text((d.vendor_name or "-")[:20]),
                _safe_text((d.invoice_date or "-")[:10]),
                _safe_text((d.category or "-")[:15]),
                _safe_text(f"{d.total_amount:,.2f}" if d.total_amount is not None else "-"),
                _safe_text(d.verification_status[:10]),
                _safe_text(d.benchmark_domain[:6]),
            ]
            for val, w in zip(vals, headers_w):
                pdf.cell(w, 5, str(val), fill=True, border=1)
            pdf.ln()
            fill_flag = not fill_flag

        # Footer
        pdf.ln(8)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 6, "Generated by Phase 11 - Institutional Financial Document Intelligence System", align="C")
        pdf.ln()

        pdf_bytes = bytes(pdf.output())

        if filepath is not None:
            Path(filepath).write_bytes(pdf_bytes)

        return pdf_bytes


    # -------------------------------------------------------------------------
    # Phase 14 — Dashboard Analytics Excel Export
    # -------------------------------------------------------------------------

    def export_dashboard_excel(
        self,
        kpi_data: Dict[str, Any],
        category_data: List[Dict[str, Any]],
        vendor_data: List[Dict[str, Any]],
        tax_data: Dict[str, Any],
        monthly_data: List[Dict[str, Any]],
        anomaly_data: Dict[str, Any],
        fy_data: Optional[List[Dict[str, Any]]] = None,
        filter_description: str = "All Documents",
    ) -> bytes:
        """
        Export a multi-sheet Excel workbook of Phase 14 dashboard analytics.

        Sheets
        ------
        1. Summary KPIs      — key metrics table
        2. Category Analysis — spend by category with HHI
        3. Vendor Analysis   — top vendors with share %
        4. Tax Compliance    — tax coverage and rates
        5. Monthly Trend     — month-by-month spend + MoM delta
        6. Anomaly Report    — sigma outliers and duplicate invoices
        7. FY Comparison     — fiscal year aggregates (if provided)

        Returns bytes of the .xlsx file.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas is required for Excel export") from exc

        output = io.BytesIO()
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        with pd.ExcelWriter(output, engine="openpyxl") as writer:

            # ── Sheet 1: Summary KPIs ────────────────────────────────────
            kpi_rows = [
                {"Metric": "Generated At", "Value": generated_at},
                {"Metric": "Filter Applied", "Value": filter_description},
                {"Metric": "Total Invoice Count", "Value": kpi_data.get("invoice_count", 0)},
                {"Metric": "Total Spend (INR)", "Value": kpi_data.get("total_spend", 0)},
                {"Metric": "Avg Invoice Value (INR)", "Value": kpi_data.get("avg_invoice_value") or "N/A"},
                {"Metric": "Tax Total (INR)",
                 "Value": kpi_data.get("tax_total") if kpi_data.get("tax_total") is not None else "Tax data unavailable"},
                {"Metric": "Effective Tax Rate",
                 "Value": f"{kpi_data['effective_tax_rate']:.2%}" if kpi_data.get("effective_tax_rate") is not None else "N/A"},
                {"Metric": "Verified Count", "Value": kpi_data.get("verified_count", 0)},
                {"Metric": "Verified Rate", "Value": f"{kpi_data.get('verified_rate', 0):.1%}"},
                {"Metric": "Pending Count", "Value": kpi_data.get("pending_count", 0)},
                {"Metric": "Flagged Count", "Value": kpi_data.get("flagged_count", 0)},
                {"Metric": "Modified Count", "Value": kpi_data.get("modified_count", 0)},
                {"Metric": "Avg Classification Confidence",
                 "Value": f"{kpi_data['avg_confidence']:.3f}" if kpi_data.get("avg_confidence") is not None else "N/A"},
            ]
            pd.DataFrame(kpi_rows).to_excel(writer, sheet_name="Summary KPIs", index=False)

            # ── Sheet 2: Category Analysis ────────────────────────────────
            if category_data:
                cat_rows = []
                for c in category_data:
                    cat_rows.append({
                        "Category": c.get("category", ""),
                        "Total Spend (INR)": c.get("total_spend", 0),
                        "Share (%)": round(c.get("share_pct", 0), 2),
                        "Invoice Count": c.get("count", ""),
                        "Avg Amount (INR)": round(c.get("avg_amount", 0), 2) if c.get("avg_amount") else "",
                    })
                pd.DataFrame(cat_rows).to_excel(writer, sheet_name="Category Analysis", index=False)

            # ── Sheet 3: Vendor Analysis ──────────────────────────────────
            if vendor_data:
                vend_rows = [{
                    "Vendor Name": v.get("vendor_name", ""),
                    "Invoice Count": v.get("invoice_count", 0),
                    "Total Spend (INR)": v.get("total_spend", 0),
                    "Share (%)": v.get("share_pct", 0),
                    "Avg Invoice (INR)": round(v.get("avg_invoice") or 0, 2),
                    "Last Invoice Date": v.get("last_invoice_date", ""),
                    "Concentration Risk": "YES" if v.get("concentration_risk") else "NO",
                } for v in vendor_data]
                pd.DataFrame(vend_rows).to_excel(writer, sheet_name="Vendor Analysis", index=False)

            # ── Sheet 4: Tax Compliance ───────────────────────────────────
            tax_summary_rows = [
                {"Metric": "Tax Coverage (%)", "Value": tax_data.get("coverage_pct", 0)},
                {"Metric": "Total Tax (INR)",
                 "Value": tax_data.get("total_tax") if tax_data.get("total_tax") is not None else "Tax data unavailable"},
                {"Metric": "Avg Effective Rate",
                 "Value": f"{tax_data['avg_effective_rate']:.2%}" if tax_data.get("avg_effective_rate") is not None else "N/A"},
                {"Metric": "Min Effective Rate",
                 "Value": f"{tax_data['min_effective_rate']:.2%}" if tax_data.get("min_effective_rate") is not None else "N/A"},
                {"Metric": "Max Effective Rate",
                 "Value": f"{tax_data['max_effective_rate']:.2%}" if tax_data.get("max_effective_rate") is not None else "N/A"},
                {"Metric": "Zero-Tax Documents", "Value": tax_data.get("zero_tax_count", 0)},
                {"Metric": "Missing Tax Data", "Value": tax_data.get("null_tax_count", 0)},
            ]
            tax_df = pd.DataFrame(tax_summary_rows)
            tax_cat_df = pd.DataFrame(tax_data.get("tax_by_category", []))
            tax_df.to_excel(writer, sheet_name="Tax Compliance", index=False, startrow=0)
            if not tax_cat_df.empty:
                tax_cat_df.to_excel(writer, sheet_name="Tax Compliance", index=False,
                                    startrow=len(tax_summary_rows) + 3)

            # ── Sheet 5: Monthly Trend ────────────────────────────────────
            if monthly_data:
                monthly_rows = [{
                    "Month": m.get("month", ""),
                    "Invoice Count": m.get("count", 0),
                    "Total Spend (INR)": m.get("total_spend", 0),
                    "MoM Delta (%)": round(m.get("mom_delta_pct") or 0, 2) if m.get("mom_delta_pct") is not None else "",
                    "3-Month Rolling Avg (INR)": round(m.get("rolling_avg_3m") or 0, 2) if m.get("rolling_avg_3m") is not None else "",
                } for m in monthly_data]
                pd.DataFrame(monthly_rows).to_excel(writer, sheet_name="Monthly Trend", index=False)

            # ── Sheet 6: Anomaly Report ───────────────────────────────────
            anomaly_summary = [
                {"Metric": "Mean Amount", "Value": anomaly_data.get("mean_amount") or "N/A"},
                {"Metric": "Std Dev Amount", "Value": anomaly_data.get("std_amount") or "N/A"},
                {"Metric": "Sigma Threshold", "Value": anomaly_data.get("sigma_threshold", 2.0)},
                {"Metric": "Sigma Outlier Count", "Value": len(anomaly_data.get("sigma_outliers", []))},
                {"Metric": "Duplicate Invoice Groups", "Value": len(anomaly_data.get("duplicate_invoices", []))},
                {"Metric": "Round Amount Docs", "Value": anomaly_data.get("round_amount_count", 0)},
            ]
            pd.DataFrame(anomaly_summary).to_excel(writer, sheet_name="Anomaly Report", index=False, startrow=0)
            if anomaly_data.get("sigma_outliers"):
                out_df = pd.DataFrame(anomaly_data["sigma_outliers"])
                out_df.to_excel(writer, sheet_name="Anomaly Report", index=False,
                                startrow=len(anomaly_summary) + 3)

            # Sheet 7: FY Comparison
            if fy_data:
                pd.DataFrame(fy_data).to_excel(writer, sheet_name="FY Comparison", index=False)

        output.seek(0)
        return output.read()

    def export_anomalies_excel(
        self,
        findings: List[Dict[str, Any]],
        duplicates: Optional[List[Dict[str, Any]]] = None,
        review_history: Optional[List[Dict[str, Any]]] = None,
    ) -> bytes:
        """
        Generate a multi-sheet Excel workbook detailing anomaly findings,
        duplicate candidates, and the audit trail.
        """
        import io
        import pandas as pd

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # Sheet 1: Anomaly Findings
            if findings:
                f_rows = [{
                    "ID": f.get("id"),
                    "Document ID": f.get("document_id"),
                    "Filename": f.get("filename", ""),
                    "Vendor": f.get("vendor_name", ""),
                    "Invoice #": f.get("invoice_number", ""),
                    "Total Amount (INR)": f.get("total_amount", ""),
                    "Anomaly Type": f.get("anomaly_type", ""),
                    "Severity": f.get("severity", ""),
                    "Score": f.get("score", 0.0),
                    "Confidence": f.get("confidence", 0.0),
                    "Status": f.get("status", ""),
                    "Explanation": f.get("explanation", ""),
                    "Detected At": f.get("detected_at", ""),
                } for f in findings]
                pd.DataFrame(f_rows).to_excel(writer, sheet_name="Anomaly Findings", index=False)
            else:
                pd.DataFrame([{"Message": "No anomaly findings to report"}]).to_excel(writer, sheet_name="Anomaly Findings", index=False)

            # Sheet 2: Duplicate Candidates
            if duplicates:
                d_rows = [{
                    "ID": d.get("id"),
                    "Source Doc ID": d.get("document_id"),
                    "Source Filename": d.get("filename", ""),
                    "Source Vendor": d.get("vendor_name", ""),
                    "Source Invoice #": d.get("invoice_number", ""),
                    "Source Amount (INR)": d.get("total_amount", ""),
                    "Matched Doc ID": d.get("matched_document_id"),
                    "Matched Filename": d.get("matched_filename", ""),
                    "Matched Vendor": d.get("matched_vendor_name", ""),
                    "Matched Invoice #": d.get("matched_invoice_number", ""),
                    "Matched Amount (INR)": d.get("matched_total_amount", ""),
                    "Duplicate Type": d.get("duplicate_type", ""),
                    "Score": d.get("score", 0.0),
                    "Confidence": d.get("confidence", 0.0),
                    "Status": d.get("status", ""),
                    "Explanation": d.get("explanation", ""),
                    "Detected At": d.get("detected_at", ""),
                } for d in duplicates]
                pd.DataFrame(d_rows).to_excel(writer, sheet_name="Duplicate Candidates", index=False)
            else:
                pd.DataFrame([{"Message": "No duplicate candidates to report"}]).to_excel(writer, sheet_name="Duplicate Candidates", index=False)

            # Sheet 3: Review History Audit
            if review_history:
                h_rows = [{
                    "ID": h.get("id"),
                    "Document ID": h.get("document_id"),
                    "Entity Type": h.get("entity_type"),
                    "Entity ID": h.get("entity_id"),
                    "Old Status": h.get("old_status"),
                    "New Status": h.get("new_status"),
                    "Reviewer": h.get("reviewer"),
                    "Comment": h.get("comment"),
                    "Timestamp": h.get("timestamp"),
                } for h in review_history]
                pd.DataFrame(h_rows).to_excel(writer, sheet_name="Review Audit Log", index=False)

        output.seek(0)
        return output.read()

    def export_anomalies_csv(self, findings: List[Dict[str, Any]]) -> str:
        """Export anomaly findings as CSV string."""
        import pandas as pd
        if not findings:
            return "document_id,anomaly_type,severity,score,confidence,status,explanation\n"
        df = pd.DataFrame(findings)
        cols = [c for c in ["id", "document_id", "filename", "vendor_name", "invoice_number",
                           "total_amount", "anomaly_type", "severity", "score", "confidence",
                           "status", "explanation", "detected_at"] if c in df.columns]
        return df[cols].to_csv(index=False)

    def export_anomalies_json(
        self,
        findings: List[Dict[str, Any]],
        duplicates: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Export anomaly findings and duplicates as JSON string."""
        payload = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "findings_count": len(findings),
            "findings": findings,
            "duplicate_candidates_count": len(duplicates or []),
            "duplicate_candidates": duplicates or [],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # Phase 16 — Dynamic Institutional Reporting Exporter
    # -------------------------------------------------------------------------

    def export_institutional_pdf(
        self,
        report_data: Dict[str, Any],
        filepath: Optional[Path] = None,
    ) -> bytes:
        """
        Generate a multi-page, executive-grade PDF report using fpdf2.

        Includes:
            - Header with Organization, Title, Report ID, and Timestamp
            - Filter Scope & Consistency Declaration
            - Deterministic Executive Narrative bullets
            - Key Performance Indicators (KPIs) table
            - Category Distribution & HHI Concentration table
            - Vendor Exposure Pareto table
            - Monthly Trend & Spending Velocity table
            - Integrity & Anomaly Discrepancy table
            - Institutional Audit & Methodology Appendix
            - Multi-page header & footer with dynamic page numbering (Page X of Y)
        """
        try:
            from fpdf import FPDF
        except ImportError as exc:
            raise RuntimeError("fpdf2 is required for PDF export. Run: pip install fpdf2") from exc

        meta = report_data.get("metadata", {})
        title = meta.get("report_title", "Institutional Financial Intelligence Report")
        org = meta.get("organization", "Institutional Finance Operations")
        report_id = meta.get("report_id", "RPT-UNKNOWN")
        gen_time = meta.get("generated_at", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        scope_desc = meta.get("scope_description", "Scope: All financial records")

        class _InstitutionalPDF(FPDF):
            def header(self):
                if self.page_no() > 1:
                    self.set_font("Helvetica", "I", 8)
                    self.set_text_color(120, 120, 120)
                    self.cell(0, 6, _safe_text(f"{title} | {org}"), align="L")
                    self.ln(6)
                    self.set_draw_color(220, 220, 220)
                    self.line(10, self.get_y(), 200, self.get_y())
                    self.ln(4)

            def footer(self):
                self.set_y(-12)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(140, 140, 140)
                self.cell(0, 6, _safe_text(f"Page {self.page_no()} of {{nb}} | Ref: {report_id}"), align="C")

        pdf = _InstitutionalPDF()
        pdf.alias_nb_pages()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # ── Title Banner ───────────────────────────────────────────────────
        pdf.set_fill_color(30, 58, 95)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 15)
        pdf.cell(0, 12, _safe_text(title), fill=True, align="C")
        pdf.ln(13)

        # ── Organization & Metadata Subtitle ──────────────────────────────
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(90, 100, 115)
        pdf.cell(95, 5, _safe_text(f"Organization: {org}"), border=0)
        pdf.cell(95, 5, _safe_text(f"Report ID: {report_id}"), border=0, align="R")
        pdf.ln(5)
        pdf.cell(95, 5, _safe_text(f"Generated: {gen_time}"), border=0)
        pdf.cell(95, 5, _safe_text(f"Classification: Institutional / Confidential"), border=0, align="R")
        pdf.ln(6)

        # ── Scope Banner ──────────────────────────────────────────────────
        pdf.set_fill_color(240, 244, 250)
        pdf.set_draw_color(200, 215, 235)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(0, 6, _safe_text(scope_desc), fill=True, border=1)
        pdf.ln(9)

        # ── Section: Executive Narrative ──────────────────────────────────
        pdf.set_text_color(30, 58, 95)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Executive Summary & Key Observations")
        pdf.ln(7)

        narratives = report_data.get("narratives", [])
        pdf.set_text_color(40, 40, 40)
        pdf.set_font("Helvetica", "", 8)
        if narratives:
            for item in narratives:
                pdf.multi_cell(0, 5, _safe_text(f"*  {item}"))
                pdf.ln(1)
        else:
            pdf.cell(0, 5, "No observations recorded for the selected scope.")
            pdf.ln(5)
        pdf.ln(4)

        # ── Section: Key Indicators (KPIs) ────────────────────────────────
        pdf.set_text_color(30, 58, 95)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Key Financial & Operational Indicators")
        pdf.ln(7)

        kpis = report_data.get("kpis", {})
        total_spend = kpis.get("total_spend", 0.0)
        inv_count = kpis.get("invoice_count", 0)
        avg_inv = kpis.get("avg_invoice_value") or 0.0
        tax_total = kpis.get("tax_total")
        eff_rate = kpis.get("effective_tax_rate")
        verified_cnt = kpis.get("verified_count", 0)
        ver_rate = kpis.get("verified_rate", 0.0)

        anom_sum = report_data.get("integrity_analysis", {}).get("summary", {})
        open_anom = anom_sum.get("open_findings", 0)
        dup_cnt = anom_sum.get("total_duplicate_candidates", 0)

        kpi_grid = [
            ("Total Expenditure", f"INR {total_spend:,.2f}"),
            ("Total Document Count", f"{inv_count:,}"),
            ("Average Invoice Value", f"INR {avg_inv:,.2f}" if avg_inv else "N/A"),
            ("Documented Tax Total", f"INR {tax_total:,.2f}" if tax_total is not None else "Unavailable"),
            ("Effective Tax Rate", f"{eff_rate:.2%}" if eff_rate is not None else "N/A"),
            ("Verified Documents", f"{verified_cnt:,} ({ver_rate:.1%})"),
            ("Open Anomaly Findings", f"{open_anom:,}"),
            ("Duplicate Candidate Pairs", f"{dup_cnt:,}"),
        ]

        pdf.set_draw_color(220, 225, 235)
        w_label, w_val = 50, 45
        for i in range(0, len(kpi_grid), 2):
            l1, v1 = kpi_grid[i]
            l2, v2 = kpi_grid[i + 1] if i + 1 < len(kpi_grid) else ("", "")

            pdf.set_fill_color(248, 250, 252)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(70, 80, 95)
            pdf.cell(w_label, 6, _safe_text(l1), border=1, fill=True)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(w_val, 6, _safe_text(v1), border=1)

            if l2:
                pdf.set_fill_color(248, 250, 252)
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(70, 80, 95)
                pdf.cell(w_label, 6, _safe_text(l2), border=1, fill=True)
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(15, 23, 42)
                pdf.cell(w_val, 6, _safe_text(v2), border=1)
            pdf.ln()
        pdf.ln(6)

        # ── Section: Category Distribution ────────────────────────────────
        pdf.set_text_color(30, 58, 95)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Expenditure by Category")
        pdf.ln(7)

        cats = report_data.get("category_analysis", {}).get("breakdown", [])
        pdf.set_fill_color(30, 58, 95)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(75, 6, "Category", fill=True, border=1)
        pdf.cell(30, 6, "Invoices", fill=True, border=1, align="C")
        pdf.cell(45, 6, "Total Spend (INR)", fill=True, border=1, align="R")
        pdf.cell(40, 6, "Share (%)", fill=True, border=1, align="R")
        pdf.ln()

        pdf.set_text_color(15, 23, 42)
        pdf.set_font("Helvetica", "", 7.5)
        if cats:
            for idx, c in enumerate(cats[:10]):
                fill_bg = (idx % 2 == 1)
                pdf.set_fill_color(245, 248, 255) if fill_bg else pdf.set_fill_color(255, 255, 255)
                pdf.cell(75, 5.5, _safe_text(c.get("category", "")[:35]), border=1, fill=fill_bg)
                pdf.cell(30, 5.5, _safe_text(str(c.get("count", ""))), border=1, align="C", fill=fill_bg)
                pdf.cell(45, 5.5, _safe_text(f"{c.get('total_spend', 0):,.2f}"), border=1, align="R", fill=fill_bg)
                pdf.cell(40, 5.5, _safe_text(f"{c.get('share_pct', 0):.1f}%"), border=1, align="R", fill=fill_bg)
                pdf.ln()
        else:
            pdf.cell(190, 5.5, "No category breakdown data available for scope.", border=1, align="C")
            pdf.ln()
        pdf.ln(6)

        # ── Section: Top Vendors ──────────────────────────────────────────
        pdf.set_text_color(30, 58, 95)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Top Vendors by Expenditure (Pareto Summary)")
        pdf.ln(7)

        vendors = report_data.get("vendor_analysis", {}).get("top_vendors", [])
        pdf.set_fill_color(30, 58, 95)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(70, 6, "Vendor Name", fill=True, border=1)
        pdf.cell(25, 6, "Invoices", fill=True, border=1, align="C")
        pdf.cell(40, 6, "Total Spend (INR)", fill=True, border=1, align="R")
        pdf.cell(25, 6, "Share (%)", fill=True, border=1, align="R")
        pdf.cell(30, 6, "Risk Flag", fill=True, border=1, align="C")
        pdf.ln()

        pdf.set_text_color(15, 23, 42)
        pdf.set_font("Helvetica", "", 7.5)
        if vendors:
            for idx, v in enumerate(vendors[:10]):
                fill_bg = (idx % 2 == 1)
                pdf.set_fill_color(245, 248, 255) if fill_bg else pdf.set_fill_color(255, 255, 255)
                risk_str = "REVIEW >25%" if v.get("concentration_risk") or v.get("share_pct", 0) > 25.0 else "Normal"
                pdf.cell(70, 5.5, _safe_text((v.get("vendor_name") or "Unknown")[:32]), border=1, fill=fill_bg)
                pdf.cell(25, 5.5, _safe_text(str(v.get("invoice_count", ""))), border=1, align="C", fill=fill_bg)
                pdf.cell(40, 5.5, _safe_text(f"{v.get('total_spend', 0):,.2f}"), border=1, align="R", fill=fill_bg)
                pdf.cell(25, 5.5, _safe_text(f"{v.get('share_pct', 0):.1f}%"), border=1, align="R", fill=fill_bg)
                pdf.cell(30, 5.5, _safe_text(risk_str), border=1, align="C", fill=fill_bg)
                pdf.ln()
        else:
            pdf.cell(190, 5.5, "No vendor data recorded for the selected scope.", border=1, align="C")
            pdf.ln()
        pdf.ln(6)

        # ── Section: Integrity & Anomaly Discrepancies ─────────────────────
        findings = report_data.get("integrity_analysis", {}).get("findings", [])
        if findings:
            pdf.set_text_color(30, 58, 95)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Financial Discrepancies & Anomaly Findings")
            pdf.ln(7)

            pdf.set_fill_color(30, 58, 95)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(30, 6, "Doc ID", fill=True, border=1)
            pdf.cell(35, 6, "Anomaly Type", fill=True, border=1)
            pdf.cell(20, 6, "Severity", fill=True, border=1, align="C")
            pdf.cell(15, 6, "Score", fill=True, border=1, align="C")
            pdf.cell(25, 6, "Status", fill=True, border=1, align="C")
            pdf.cell(65, 6, "Explanation", fill=True, border=1)
            pdf.ln()

            pdf.set_text_color(15, 23, 42)
            pdf.set_font("Helvetica", "", 7)
            for idx, f in enumerate(findings[:10]):
                fill_bg = (idx % 2 == 1)
                pdf.set_fill_color(245, 248, 255) if fill_bg else pdf.set_fill_color(255, 255, 255)
                pdf.cell(30, 5.5, _safe_text(str(f.get("document_id", ""))[:14]), border=1, fill=fill_bg)
                pdf.cell(35, 5.5, _safe_text(str(f.get("anomaly_type", ""))[:18]), border=1, fill=fill_bg)
                pdf.cell(20, 5.5, _safe_text(str(f.get("severity", ""))), border=1, align="C", fill=fill_bg)
                pdf.cell(15, 5.5, _safe_text(f"{f.get('score', 0):.2f}"), border=1, align="C", fill=fill_bg)
                pdf.cell(25, 5.5, _safe_text(str(f.get("status", ""))), border=1, align="C", fill=fill_bg)
                pdf.cell(65, 5.5, _safe_text(str(f.get("explanation", ""))[:42]), border=1, fill=fill_bg)
                pdf.ln()
            pdf.ln(6)

        # ── Appendix & Methodology ─────────────────────────────────────────
        pdf.set_text_color(30, 58, 95)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Institutional Methodology & Compliance Notice")
        pdf.ln(6)

        consistency = report_data.get("consistency", {})
        c_notes = consistency.get("reconciliation_notes", "Reconciliation validated.")
        pdf.set_font("Helvetica", "I", 7.5)
        pdf.set_text_color(90, 100, 115)
        pdf.multi_cell(
            0, 4.5,
            _safe_text(
                f"1. Mathematical Reconciliation: {c_notes}\n"
                f"2. Statistical Methodology: Sigma outliers are evaluated against sample standard deviation. "
                f"Duplicate candidates are determined via candidate blocking and multi-signal scoring.\n"
                f"3. Responsible Analytics Boundary: Findings represent algorithmic discrepancy alerts for human "
                f"review and do not constitute accusations of fraudulent or unlawful conduct."
            )
        )

        pdf_bytes = bytes(pdf.output())
        if filepath is not None:
            Path(filepath).write_bytes(pdf_bytes)
        return pdf_bytes

    def export_institutional_excel(self, report_data: Dict[str, Any]) -> bytes:
        """
        Generate a comprehensive 11-sheet institutional Excel workbook using openpyxl.

        Sheets:
            1. Executive Summary     — High-level KPIs, narrative observations, reconciliation
            2. Financial Records     — Filtered document list with formula-injection sanitization
            3. Category Analysis     — Category totals, share %, and HHI concentration metrics
            4. Vendor Intelligence   — Vendor rankings, spend, share %, and concentration alerts
            5. Monthly Trend         — Month-by-month spend, sequential growth %, 3M rolling avg
            6. Fiscal Year Analysis  — Indian FY totals and YoY comparative variance
            7. Tax Analysis          — Coverage %, effective tax rates, and category tax totals
            8. Anomaly Findings      — Discrepancy findings with severity, score, and explanation
            9. Duplicate Candidates  — Pairwise candidate matches with primary and matched info
            10. Audit History        — Human-in-the-loop review actions, comments, and timestamps
            11. Report Metadata      — Scope description, filter configuration, and version
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas is required for Excel export") from exc

        def _clean_val(val: Any) -> Any:
            """Sanitize strings against Excel formula injection risks."""
            if isinstance(val, str) and val.startswith(("=", "+", "-", "@")):
                return "'" + val
            return val

        def _clean_df(df: Any) -> Any:
            return df.map(_clean_val)

        output = io.BytesIO()
        meta = report_data.get("metadata", {})
        kpis = report_data.get("kpis", {})
        narratives = report_data.get("narratives", [])
        consistency = report_data.get("consistency", {})

        with pd.ExcelWriter(output, engine="openpyxl") as writer:

            # ── Sheet 1: Executive Summary ────────────────────────────────
            sum_rows = [
                {"Parameter": "Report Title", "Value": meta.get("report_title")},
                {"Parameter": "Report ID", "Value": meta.get("report_id")},
                {"Parameter": "Generated At (UTC)", "Value": meta.get("generated_at")},
                {"Parameter": "Organization", "Value": meta.get("organization")},
                {"Parameter": "Scope", "Value": meta.get("scope_description")},
                {"Parameter": "Reconciliation Status", "Value": "PASS (Consistent)" if consistency.get("is_consistent") else "VARIANCE DETECTED"},
                {"Parameter": "", "Value": ""},
                {"Parameter": "Total Document Count", "Value": kpis.get("invoice_count", 0)},
                {"Parameter": "Total Expenditure (INR)", "Value": kpis.get("total_spend", 0.0)},
                {"Parameter": "Average Invoice Value (INR)", "Value": kpis.get("avg_invoice_value") or "N/A"},
                {"Parameter": "Total Tax (INR)", "Value": kpis.get("tax_total") if kpis.get("tax_total") is not None else "Unavailable"},
                {"Parameter": "Effective Tax Rate", "Value": f"{kpis['effective_tax_rate']:.2%}" if kpis.get("effective_tax_rate") is not None else "N/A"},
                {"Parameter": "Verified Records", "Value": kpis.get("verified_count", 0)},
                {"Parameter": "Verification Rate", "Value": f"{kpis.get('verified_rate', 0):.1%}"},
                {"Parameter": "Pending Verification", "Value": kpis.get("pending_count", 0)},
                {"Parameter": "Flagged Records", "Value": kpis.get("flagged_count", 0)},
            ]
            _clean_df(pd.DataFrame(sum_rows)).to_excel(writer, sheet_name="Executive Summary", index=False)

            if narratives:
                narr_df = pd.DataFrame([{"Executive Observations": n} for n in narratives])
                _clean_df(narr_df).to_excel(writer, sheet_name="Executive Summary", index=False, startrow=len(sum_rows) + 3)

            # ── Sheet 2: Financial Records ────────────────────────────────
            docs = report_data.get("documents", [])
            if docs:
                rec_rows = []
                for d in docs:
                    rec_rows.append({
                        "Document ID": d.get("document_id"),
                        "Filename": d.get("filename"),
                        "Category": d.get("category"),
                        "Vendor": d.get("vendor_name"),
                        "Invoice #": d.get("invoice_number"),
                        "Invoice Date": d.get("invoice_date"),
                        "Total Amount (INR)": d.get("total_amount"),
                        "Tax Amount (INR)": d.get("tax_amount"),
                        "Currency": d.get("currency"),
                        "Verification Status": d.get("verification_status"),
                        "Domain": d.get("benchmark_domain"),
                    })
                _clean_df(pd.DataFrame(rec_rows)).to_excel(writer, sheet_name="Financial Records", index=False)
            else:
                pd.DataFrame([{"Message": "No document records match the selected scope."}]).to_excel(writer, sheet_name="Financial Records", index=False)

            # ── Sheet 3: Category Analysis ────────────────────────────────
            cat_breakdown = report_data.get("category_analysis", {}).get("breakdown", [])
            cat_conc = report_data.get("category_analysis", {}).get("concentration", {})
            if cat_breakdown:
                c_df = pd.DataFrame([{
                    "Category": c.get("category"),
                    "Invoices": c.get("count"),
                    "Total Spend (INR)": c.get("total_spend"),
                    "Average Bill (INR)": c.get("avg_amount"),
                    "Spend Share (%)": round(c.get("share_pct", 0), 2),
                } for c in cat_breakdown])
                _clean_df(c_df).to_excel(writer, sheet_name="Category Analysis", index=False)

                hhi_rows = [
                    {"Metric": "Herfindahl-Hirschman Index (HHI)", "Value": cat_conc.get("hhi")},
                    {"Metric": "Concentration Interpretation", "Value": cat_conc.get("hhi_interpretation")},
                    {"Metric": "Top 1 Category Share (%)", "Value": cat_conc.get("top1_share_pct")},
                    {"Metric": "Top 3 Category Share (%)", "Value": cat_conc.get("top3_share_pct")},
                ]
                _clean_df(pd.DataFrame(hhi_rows)).to_excel(writer, sheet_name="Category Analysis", index=False, startrow=len(cat_breakdown) + 3)
            else:
                pd.DataFrame([{"Message": "No category data available for scope."}]).to_excel(writer, sheet_name="Category Analysis", index=False)

            # ── Sheet 4: Vendor Intelligence ──────────────────────────────
            v_list = report_data.get("vendor_analysis", {}).get("top_vendors", [])
            if v_list:
                v_df = pd.DataFrame([{
                    "Vendor Name": v.get("vendor_name"),
                    "Invoices": v.get("invoice_count"),
                    "Total Spend (INR)": v.get("total_spend"),
                    "Spend Share (%)": v.get("share_pct"),
                    "Average Invoice (INR)": v.get("avg_invoice"),
                    "Last Invoice Date": v.get("last_invoice_date"),
                    "Concentration Risk (>25%)": "YES" if v.get("concentration_risk") else "NO",
                } for v in v_list])
                _clean_df(v_df).to_excel(writer, sheet_name="Vendor Intelligence", index=False)
            else:
                pd.DataFrame([{"Message": "No vendor data recorded."}]).to_excel(writer, sheet_name="Vendor Intelligence", index=False)

            # ── Sheet 5: Monthly Trend ────────────────────────────────────
            m_list = report_data.get("monthly_trends", [])
            if m_list:
                m_df = pd.DataFrame([{
                    "Month": m.get("month"),
                    "Invoices": m.get("count"),
                    "Total Spend (INR)": m.get("total_spend"),
                    "MoM Delta (%)": round(m.get("mom_delta_pct") or 0, 2) if m.get("mom_delta_pct") is not None else "N/A",
                    "3-Month Rolling Avg (INR)": round(m.get("rolling_avg_3m") or 0, 2) if m.get("rolling_avg_3m") is not None else "N/A",
                } for m in m_list])
                _clean_df(m_df).to_excel(writer, sheet_name="Monthly Trend", index=False)
            else:
                pd.DataFrame([{"Message": "No monthly trend data recorded."}]).to_excel(writer, sheet_name="Monthly Trend", index=False)

            # ── Sheet 6: Fiscal Year Analysis ─────────────────────────────
            fy_info = report_data.get("fiscal_year_analysis", {})
            fy_years = fy_info.get("years", [])
            if fy_years:
                fy_df = pd.DataFrame([{
                    "Fiscal Year": y.get("fiscal_year"),
                    "Invoices": y.get("count"),
                    "Total Spend (INR)": y.get("total_spend"),
                    "Avg Amount (INR)": y.get("avg_amount"),
                } for y in fy_years])
                _clean_df(fy_df).to_excel(writer, sheet_name="Fiscal Year Analysis", index=False)

                fy_comp = fy_info.get("comparison")
                if fy_comp:
                    comp_rows = [
                        {"Comparison Metric": f"Spend Change ({fy_comp.get('fy1')} -> {fy_comp.get('fy2')})",
                         "Value": f"{fy_comp.get('delta_spend_pct', 0):.2f}%" if fy_comp.get("delta_spend_pct") is not None else "N/A"},
                        {"Comparison Metric": f"Invoice Count Change",
                         "Value": f"{fy_comp.get('delta_count_pct', 0):.2f}%" if fy_comp.get("delta_count_pct") is not None else "N/A"},
                    ]
                    _clean_df(pd.DataFrame(comp_rows)).to_excel(writer, sheet_name="Fiscal Year Analysis", index=False, startrow=len(fy_years) + 3)
            else:
                pd.DataFrame([{"Message": "No fiscal year data available."}]).to_excel(writer, sheet_name="Fiscal Year Analysis", index=False)

            # ── Sheet 7: Tax Analysis ─────────────────────────────────────
            tax_info = report_data.get("tax_compliance", {})
            if tax_info:
                t_summary = [
                    {"Tax Indicator": "Tax Data Coverage (%)", "Value": tax_info.get("coverage_pct", 0)},
                    {"Tax Indicator": "Total Recorded Tax (INR)", "Value": tax_info.get("total_tax") if tax_info.get("total_tax") is not None else "Unavailable"},
                    {"Tax Indicator": "Average Effective Tax Rate", "Value": f"{tax_info.get('avg_effective_rate', 0):.2%}" if tax_info.get("avg_effective_rate") is not None else "N/A"},
                    {"Tax Indicator": "Zero-Tax Invoices", "Value": tax_info.get("zero_tax_count", 0)},
                    {"Tax Indicator": "Missing Tax Data Invoices", "Value": tax_info.get("null_tax_count", 0)},
                ]
                _clean_df(pd.DataFrame(t_summary)).to_excel(writer, sheet_name="Tax Analysis", index=False)
                t_cats = tax_info.get("tax_by_category", [])
                if t_cats:
                    _clean_df(pd.DataFrame(t_cats)).to_excel(writer, sheet_name="Tax Analysis", index=False, startrow=len(t_summary) + 3)
            else:
                pd.DataFrame([{"Message": "No tax data available."}]).to_excel(writer, sheet_name="Tax Analysis", index=False)

            # ── Sheet 8: Anomaly Findings ─────────────────────────────────
            findings = report_data.get("integrity_analysis", {}).get("findings", [])
            if findings:
                f_df = pd.DataFrame([{
                    "Finding ID": f.get("id"),
                    "Document ID": f.get("document_id"),
                    "Anomaly Type": f.get("anomaly_type"),
                    "Severity": f.get("severity"),
                    "Score": f.get("score"),
                    "Status": f.get("status"),
                    "Explanation": f.get("explanation"),
                    "Detected At": f.get("detected_at"),
                } for f in findings])
                _clean_df(f_df).to_excel(writer, sheet_name="Anomaly Findings", index=False)
            else:
                pd.DataFrame([{"Message": "No anomaly findings to report for this scope."}]).to_excel(writer, sheet_name="Anomaly Findings", index=False)

            # ── Sheet 9: Duplicate Candidates ─────────────────────────────
            dups = report_data.get("integrity_analysis", {}).get("duplicate_candidates", [])
            if dups:
                d_df = pd.DataFrame([{
                    "Candidate ID": d.get("id"),
                    "Source Doc ID": d.get("document_id"),
                    "Matched Doc ID": d.get("matched_document_id"),
                    "Duplicate Type": d.get("duplicate_type"),
                    "Score": d.get("score"),
                    "Status": d.get("status"),
                    "Explanation": d.get("explanation"),
                } for d in dups])
                _clean_df(d_df).to_excel(writer, sheet_name="Duplicate Candidates", index=False)
            else:
                pd.DataFrame([{"Message": "No duplicate candidate pairs identified."}]).to_excel(writer, sheet_name="Duplicate Candidates", index=False)

            # ── Sheet 10: Audit History ───────────────────────────────────
            history = report_data.get("audit_history", [])
            if history:
                h_df = pd.DataFrame([{
                    "Audit ID": h.get("id"),
                    "Document ID": h.get("document_id"),
                    "Entity Type": h.get("entity_type"),
                    "Old Status": h.get("old_status"),
                    "New Status": h.get("new_status"),
                    "Reviewer": h.get("reviewer"),
                    "Comment": h.get("comment"),
                    "Timestamp": h.get("timestamp"),
                } for h in history])
                _clean_df(h_df).to_excel(writer, sheet_name="Audit History", index=False)
            else:
                pd.DataFrame([{"Message": "No audit status transitions on record for this scope."}]).to_excel(writer, sheet_name="Audit History", index=False)

            # ── Sheet 11: Report Metadata ─────────────────────────────────
            active_f = meta.get("active_filters", {})
            meta_rows = [
                {"Metadata Field": "Report ID", "Value": meta.get("report_id")},
                {"Metadata Field": "Report Type", "Value": meta.get("report_type")},
                {"Metadata Field": "Generated At", "Value": meta.get("generated_at")},
                {"Metadata Field": "Application Version", "Value": meta.get("app_version")},
                {"Metadata Field": "Organization", "Value": meta.get("organization")},
                {"Metadata Field": "Scope", "Value": meta.get("scope_description")},
                {"Metadata Field": "Document Count", "Value": meta.get("document_count")},
                {"Metadata Field": "Active Filter Config", "Value": json.dumps(active_f)},
            ]
            _clean_df(pd.DataFrame(meta_rows)).to_excel(writer, sheet_name="Report Metadata", index=False)

        output.seek(0)
        return output.read()

    def export_institutional_csv(self, report_data: Dict[str, Any], section: str = "documents") -> str:
        """
        Export tabular data from report as CSV string with preserved UTF-8 encoding.
        """
        import pandas as pd

        if section == "documents":
            docs = report_data.get("documents", [])
            if docs:
                return pd.DataFrame(docs).to_csv(index=False)
            return "document_id,filename,category,vendor_name,invoice_date,total_amount,verification_status\n"
        
        elif section == "categories":
            cats = report_data.get("category_analysis", {}).get("breakdown", [])
            return pd.DataFrame(cats).to_csv(index=False) if cats else "category,count,total_spend,avg_amount,share_pct\n"
        
        elif section == "vendors":
            vends = report_data.get("vendor_analysis", {}).get("top_vendors", [])
            return pd.DataFrame(vends).to_csv(index=False) if vends else "vendor_name,invoice_count,total_spend,share_pct\n"

        elif section == "anomalies":
            anoms = report_data.get("integrity_analysis", {}).get("findings", [])
            return pd.DataFrame(anoms).to_csv(index=False) if anoms else "id,document_id,anomaly_type,severity,score,status\n"

        # Fallback to KPI summary
        kpis = report_data.get("kpis", {})
        return pd.DataFrame([kpis]).to_csv(index=False)

    def export_institutional_json(self, report_data: Dict[str, Any]) -> str:
        """
        Export complete report dataset as serialized JSON.
        """
        return json.dumps(report_data, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def export_institutional_pdf(report_data: Dict[str, Any], filepath: Optional[Path] = None) -> bytes:
    return DocumentExporter().export_institutional_pdf(report_data, filepath=filepath)

def export_institutional_excel(report_data: Dict[str, Any]) -> bytes:
    return DocumentExporter().export_institutional_excel(report_data)

def export_institutional_csv(report_data: Dict[str, Any], section: str = "documents") -> str:
    return DocumentExporter().export_institutional_csv(report_data, section=section)

def export_institutional_json(report_data: Dict[str, Any]) -> str:
    return DocumentExporter().export_institutional_json(report_data)

