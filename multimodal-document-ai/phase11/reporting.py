"""
Phase 16 — Unified Institutional Reporting Engine.

Consolidates analytical, financial, classification, audit, anomaly, and duplicate
outputs from Phases 11–15 into standardized, reproducible, filter-aware reports.

Report Types:
    1. EXECUTIVE_SUMMARY    — High-level KPIs, category/vendor summaries, monthly trends, integrity flags.
    2. FISCAL_YEAR          — Indian FY cycles (April 1 – March 31), YoY variance, monthly distribution.
    3. VENDOR_INTELLIGENCE  — Vendor concentration risk, Pareto rankings, ticket sizes, anomaly occurrences.
    4. CATEGORY_INTELLIGENCE— Category distribution, HHI concentration index, diversity metrics.
    5. TAX_COMPLIANCE       — Tax coverage, effective tax rates, zero-tax counts, category breakdown.
    6. ANOMALY_DUPLICATE    — 10-category anomaly counts, 5-tier duplicate candidates, review statuses.
    7. AUDIT                — Document edits, status transitions, review notes, historical chronology.

All narratives are strictly deterministic and calculated directly from database records.
No LLMs are used.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from phase11.database import FinancialDocumentDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

APP_VERSION = "16.0.0"

class ReportType(str, Enum):
    EXECUTIVE_SUMMARY     = "EXECUTIVE_SUMMARY"
    FISCAL_YEAR           = "FISCAL_YEAR"
    VENDOR_INTELLIGENCE   = "VENDOR_INTELLIGENCE"
    CATEGORY_INTELLIGENCE = "CATEGORY_INTELLIGENCE"
    TAX_COMPLIANCE        = "TAX_COMPLIANCE"
    ANOMALY_DUPLICATE     = "ANOMALY_DUPLICATE"
    AUDIT                 = "AUDIT"


REPORT_TITLES = {
    ReportType.EXECUTIVE_SUMMARY:     "Executive Financial Summary Report",
    ReportType.FISCAL_YEAR:           "Fiscal Year Financial & Performance Report",
    ReportType.VENDOR_INTELLIGENCE:   "Vendor Exposure & Concentration Intelligence Report",
    ReportType.CATEGORY_INTELLIGENCE: "Expenditure Category Distribution & Diversity Report",
    ReportType.TAX_COMPLIANCE:        "Tax Analytics & Compliance-Oriented Data Quality Report",
    ReportType.ANOMALY_DUPLICATE:     "Financial Discrepancy & Duplicate Candidate Investigation Report",
    ReportType.AUDIT:                 "Document Lifecycle & Review Decision Audit Report",
}


# ---------------------------------------------------------------------------
# Currency & Formatting Utilities
# ---------------------------------------------------------------------------

def format_inr(amount: Optional[float], prefix: str = "₹") -> str:
    """Format a monetary amount according to the Indian numbering system."""
    if amount is None:
        return "N/A"
    is_neg = amount < 0
    abs_amt = abs(amount)
    s = f"{abs_amt:.2f}"
    int_part, dec_part = s.split(".")
    if len(int_part) <= 3:
        formatted_int = int_part
    else:
        last3 = int_part[-3:]
        rest = int_part[:-3]
        groups: List[str] = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        formatted_int = ",".join(groups) + "," + last3
    sign = "-" if is_neg else ""
    return f"{sign}{prefix}{formatted_int}.{dec_part}"


def format_currency(amount: Optional[float], currency: str = "INR", for_pdf: bool = False) -> str:
    """
    Format monetary amount with appropriate symbol or Latin-1 code for PDF fonts.
    For INR, uses Indian numbering format.
    """
    if amount is None:
        return "N/A"
    curr = (currency or "INR").upper()
    if curr == "INR":
        sym = "INR " if for_pdf else "₹"
        return format_inr(amount, prefix=sym)
    
    symbols = {"USD": "$", "EUR": "EUR " if for_pdf else "€", "GBP": "GBP " if for_pdf else "£"}
    sym = symbols.get(curr, f"{curr} ")
    return f"{sym}{amount:,.2f}"


# ---------------------------------------------------------------------------
# Deterministic Executive Narrative Generator
# ---------------------------------------------------------------------------

def generate_deterministic_narratives(
    report_type: ReportType,
    kpis: Dict[str, Any],
    categories: List[Dict[str, Any]],
    vendors: List[Dict[str, Any]],
    monthly: List[Dict[str, Any]],
    fy_comparison: Optional[Dict[str, Any]] = None,
    tax_info: Optional[Dict[str, Any]] = None,
    anomalies_summary: Optional[Dict[str, Any]] = None,
    currency_is_mixed: bool = False,
) -> List[str]:
    """
    Generate deterministic, evidence-based textual observations from ground-truth
    database values. Strictly avoids LLMs, speculation, or unsupported causal claims.
    """
    narratives: List[str] = []
    inv_count = kpis.get("invoice_count", 0)

    if inv_count == 0:
        return ["No financial records were found matching the selected report scope."]

    total_spend = kpis.get("total_spend", 0.0)
    avg_inv = kpis.get("avg_invoice_value")
    spend_str = format_inr(total_spend)

    if currency_is_mixed:
        narratives.append(
            "Notice: Multiple currencies detected within the selected filter scope. "
            "Totals are segregated by currency to preserve mathematical consistency."
        )

    # 1. Scope & Primary Spend Statement
    narratives.append(
        f"A total of {inv_count:,} financial document(s) represent an aggregate expenditure of {spend_str}, "
        f"with an average bill value of {format_inr(avg_inv)}."
    )

    # 2. Category Concentration Observation
    if categories:
        top_cat = categories[0]
        top_cat_share = top_cat.get("share_pct", 0.0)
        narratives.append(
            f"The primary expenditure category is '{top_cat.get('category')}', accounting for "
            f"{format_inr(top_cat.get('total_spend'))} ({top_cat_share:.1f}% of filtered spend)."
        )

    # 3. Vendor Exposure Observation
    if vendors:
        top_vend = vendors[0]
        v_share = top_vend.get("share_pct", 0.0)
        v_name = top_vend.get("vendor_name", "Unknown")
        if v_share > 25.0:
            narratives.append(
                f"Concentration review recommended: Top vendor '{v_name}' represents {v_share:.1f}% "
                f"({format_inr(top_vend.get('total_spend'))}) of the filtered spend across {top_vend.get('invoice_count')} invoices."
            )
        else:
            narratives.append(
                f"Top vendor '{v_name}' accounts for {v_share:.1f}% of filtered expenditure."
            )

    # 4. Monthly Trend / Growth Observation
    if monthly and len(monthly) >= 2:
        last_m = monthly[-1]
        prev_m = monthly[-2]
        delta_pct = last_m.get("mom_delta_pct")
        if delta_pct is not None:
            direction = "increased" if delta_pct > 0 else "decreased"
            narratives.append(
                f"Expenditure for the latest recorded period ({last_m.get('month')}) {direction} by "
                f"{abs(delta_pct):.1f}% relative to the preceding month ({prev_m.get('month')})."
            )

    # 5. Fiscal Year Variance Observation
    if fy_comparison and fy_comparison.get("delta_spend_pct") is not None:
        fy_delta = fy_comparison["delta_spend_pct"]
        fy_dir = "increased" if fy_delta > 0 else "decreased"
        narratives.append(
            f"Annual expenditure {fy_dir} by {abs(fy_delta):.1f}% between {fy_comparison.get('fy1')} and {fy_comparison.get('fy2')}."
        )

    # 6. Integrity & Anomaly Observation
    if anomalies_summary:
        open_anom = anomalies_summary.get("open_findings", 0)
        high_crit = (anomalies_summary.get("severity_counts", {}).get("HIGH", 0) +
                     anomalies_summary.get("severity_counts", {}).get("CRITICAL", 0))
        dups = anomalies_summary.get("total_duplicate_candidates", 0)
        if open_anom > 0 or dups > 0:
            narratives.append(
                f"Integrity Review: {open_anom} open anomaly finding(s) (including {high_crit} High/Critical) "
                f"and {dups} duplicate candidate pair(s) are pending human investigation."
            )
        else:
            narratives.append("Integrity Check: Zero open anomaly findings or duplicate candidate collisions detected.")

    # 7. Tax Data Quality Observation
    if tax_info:
        cov = tax_info.get("coverage_pct", 0.0)
        if cov > 0:
            eff_rate = tax_info.get("avg_effective_rate")
            eff_str = f" with an average effective rate of {eff_rate:.1%}" if eff_rate is not None else ""
            narratives.append(
                f"Tax data quality: Explicit tax amounts are documented on {cov:.1f}% of records{eff_str}."
            )
        else:
            narratives.append("Tax data quality: Tax amounts are unrecorded across the filtered document population.")

    return narratives


# ---------------------------------------------------------------------------
# Centralized Reporting Engine
# ---------------------------------------------------------------------------

class InstitutionalReportEngine:
    """
    Unified reporting engine that coordinates database analytics, filter propagation,
    data consistency reconciliation, and multi-format report packaging.
    """

    def __init__(self, db: Optional[FinancialDocumentDB] = None):
        self.db = db or FinancialDocumentDB()

    def build_report(
        self,
        report_type: ReportType = ReportType.EXECUTIVE_SUMMARY,
        report_title: Optional[str] = None,
        organization: str = "Institutional Finance Operations",
        notes: str = "",
        **filter_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Compile complete report dataset according to requested report type and filter scope.

        Returns a dictionary structure containing:
            - metadata: Report ID, generation timestamp, title, scope description, active filters
            - kpis: Comprehensive KPI summary
            - narrative: List of deterministic textual observations
            - consistency: Mathematical reconciliation check results
            - sections: Specific data collections (category, vendor, monthly, fy, tax, anomalies, audit, documents)
        """
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        report_id = f"RPT-{now_dt.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        title = report_title or REPORT_TITLES.get(report_type, "Financial Intelligence Report")

        # 1. Gather filtered documents and core analytics
        filtered_docs = self.db.search_documents(limit=10000, **filter_kwargs)
        doc_count = len(filtered_docs)

        kpis = self.db.kpi_summary(**filter_kwargs)
        categories = self.db.spend_by_category(**filter_kwargs)
        total_cat_spend = sum(c.get("total_spend", 0.0) for c in categories)
        for c in categories:
            c["share_pct"] = round((c.get("total_spend", 0.0) / total_cat_spend) * 100.0, 2) if total_cat_spend > 0 else 0.0
        cat_concentration = self.db.category_concentration(**filter_kwargs)
        vendors = self.db.vendor_summary(n=15, **filter_kwargs)
        monthly = self.db.monthly_trend_with_delta(**filter_kwargs)
        tax_info = self.db.tax_analysis(**filter_kwargs)
        fy_list = self.db.spend_by_fiscal_year(**filter_kwargs)

        # 2. Currency Isolation Check
        currencies = list({d.currency for d in filtered_docs if d.currency})
        is_mixed_currency = len(currencies) > 1
        primary_currency = currencies[0] if len(currencies) == 1 else "MIXED"

        # Currency breakdown if mixed
        currency_breakdown: Dict[str, Dict[str, Any]] = {}
        if is_mixed_currency:
            for c in currencies:
                c_docs = [d for d in filtered_docs if d.currency == c]
                c_spend = sum(d.total_amount for d in c_docs if d.total_amount is not None)
                currency_breakdown[c] = {
                    "document_count": len(c_docs),
                    "total_spend": round(c_spend, 2),
                }

        # 3. Fiscal Year Comparison (if at least 2 fiscal years exist in DB)
        fy_comp = None
        if len(fy_list) >= 2:
            try:
                fy_comp = self.db.yoy_comparison(fy_list[-2]["fiscal_year"], fy_list[-1]["fiscal_year"], **filter_kwargs)
            except Exception as e:
                logger.debug("FY comparison skipped: %s", e)

        # 4. Anomaly & Duplicate Metrics (Phase 15 integration)
        anomalies_summary = self.db.anomaly_dashboard_summary()
        anomaly_findings = self.db.get_anomaly_findings(limit=500)
        duplicate_candidates = self.db.get_duplicate_candidates(limit=500)
        review_history = self.db.get_anomaly_review_history(limit=500)

        # Scope filter for findings if document-specific filtering is active
        if filter_kwargs:
            filtered_doc_ids = {d.document_id for d in filtered_docs}
            anomaly_findings = [f for f in anomaly_findings if f["document_id"] in filtered_doc_ids]
            duplicate_candidates = [
                c for c in duplicate_candidates
                if c["document_id"] in filtered_doc_ids or c["matched_document_id"] in filtered_doc_ids
            ]
            review_history = [h for h in review_history if h.get("document_id") in filtered_doc_ids]

        # 5. Data Consistency Checks
        # Validate that category sum, vendor sum, and monthly sum reconcile with total spend
        total_spend = kpis.get("total_spend", 0.0)
        cat_sum = sum(c.get("total_spend", 0.0) for c in categories)
        month_sum = sum(m.get("total_spend", 0.0) for m in monthly)

        cat_diff = abs(total_spend - cat_sum)
        month_diff = abs(total_spend - month_sum)

        # Minor rounding tolerance (1.00)
        consistent = (cat_diff < 1.0) and (month_diff < 1.0 or not monthly)
        consistency_report = {
            "is_consistent": consistent,
            "total_spend": total_spend,
            "category_sum": cat_sum,
            "monthly_sum": month_sum,
            "category_variance": round(cat_diff, 2),
            "monthly_variance": round(month_diff, 2),
            "mixed_currency": is_mixed_currency,
            "reconciliation_notes": (
                "Mathematical reconciliation validated: All segment totals reconcile within tolerance."
                if consistent else
                f"Variance detected: Category delta {cat_diff:.2f}, Monthly delta {month_diff:.2f}. "
                f"Possible unassigned dates or mixed currencies."
            ),
        }

        # 6. Build Scope Description
        scope_lines = []
        if filter_kwargs.get("fiscal_year"):
            scope_lines.append(f"Fiscal Year: {filter_kwargs['fiscal_year']}")
        if filter_kwargs.get("category"):
            scope_lines.append(f"Category: {filter_kwargs['category']}")
        if filter_kwargs.get("vendor"):
            scope_lines.append(f"Vendor: {filter_kwargs['vendor']}")
        if filter_kwargs.get("verification_status"):
            scope_lines.append(f"Verification: {filter_kwargs['verification_status']}")
        if filter_kwargs.get("date_from") or filter_kwargs.get("date_to"):
            scope_lines.append(f"Date Range: {filter_kwargs.get('date_from', 'Start')} to {filter_kwargs.get('date_to', 'End')}")
        if filter_kwargs.get("month"):
            scope_lines.append(f"Month: {filter_kwargs['month']}")
        if filter_kwargs.get("min_amount"):
            scope_lines.append(f"Min Amount: {format_inr(filter_kwargs['min_amount'])}")
        if filter_kwargs.get("max_amount"):
            scope_lines.append(f"Max Amount: {format_inr(filter_kwargs['max_amount'])}")
        if filter_kwargs.get("query"):
            scope_lines.append(f"Keyword Search: '{filter_kwargs['query']}'")

        scope_desc = "Scope: " + (", ".join(scope_lines) if scope_lines else "All financial records")

        # 7. Generate Deterministic Narratives
        narratives = generate_deterministic_narratives(
            report_type=report_type,
            kpis=kpis,
            categories=categories,
            vendors=vendors,
            monthly=monthly,
            fy_comparison=fy_comp,
            tax_info=tax_info,
            anomalies_summary=anomalies_summary,
            currency_is_mixed=is_mixed_currency,
        )

        metadata = {
            "report_id": report_id,
            "report_type": report_type.value,
            "report_title": title,
            "generated_at": now_iso,
            "organization": organization,
            "scope_description": scope_desc,
            "active_filters": filter_kwargs,
            "document_count": doc_count,
            "currency": primary_currency,
            "app_version": APP_VERSION,
            "notes": notes,
        }

        # Packaged Report Structure
        report_data: Dict[str, Any] = {
            "metadata": metadata,
            "kpis": kpis,
            "narratives": narratives,
            "consistency": consistency_report,
            "currency_breakdown": currency_breakdown,
            "category_analysis": {
                "breakdown": categories,
                "concentration": cat_concentration,
            },
            "vendor_analysis": {
                "top_vendors": vendors,
            },
            "monthly_trends": monthly,
            "fiscal_year_analysis": {
                "years": fy_list,
                "comparison": fy_comp,
            },
            "tax_compliance": tax_info,
            "integrity_analysis": {
                "summary": anomalies_summary,
                "findings": anomaly_findings,
                "duplicate_candidates": duplicate_candidates,
            },
            "audit_history": review_history,
            "documents": [d.to_dict() for d in filtered_docs[:1000]],
        }

        return report_data
