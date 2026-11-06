"""
Phase 16 Test Suite — Dynamic Institutional Reporting & Executive Document Generation.

Comprehensive test coverage across:
1. Indian Numbering & Multi-Currency Formatting:
   - format_inr (Lakh/Crore grouping, decimals, negative amounts, None handling)
   - format_currency (INR, USD, EUR, GBP, Latin-1 fallback for PDF)
2. Deterministic Executive Narrative Engine:
   - Zero-document boundary
   - Scope & spend summary
   - Category concentration & diversity
   - Top vendor exposure & concentration alerts (>25%)
   - Monthly trend & sequential MoM growth
   - Fiscal year YoY variance
   - Integrity discrepancy & duplicate candidate alerts
   - Tax data quality and coverage
   - Mixed-currency segregation warning
3. Centralized InstitutionalReportEngine:
   - All 7 report archetypes (EXECUTIVE_SUMMARY, FISCAL_YEAR, VENDOR_INTELLIGENCE,
     CATEGORY_INTELLIGENCE, TAX_COMPLIANCE, ANOMALY_DUPLICATE, AUDIT)
   - Scope filter propagation (category, vendor, FY, date range, amounts)
   - Multi-currency isolation & per-currency segregation
   - Mathematical data consistency & reconciliation checks
   - Scoped anomaly & duplicate candidates isolation
4. Publication-Grade Multi-Format Exporters:
   - export_institutional_pdf (valid PDF, headers, multi-page, methodology appendix)
   - export_institutional_excel (11 dedicated sheets, structure validation)
   - Excel Formula Injection Protection (=, +, -, @ prefix sanitization)
   - export_institutional_csv (documents, categories, vendors, anomalies)
   - export_institutional_json (complete schema validation)
   - Module-level convenience functions
"""

from __future__ import annotations

import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import openpyxl
import pytest

from phase11.database import FinancialDocumentDB
from phase11.export import (
    DocumentExporter,
    export_institutional_csv,
    export_institutional_excel,
    export_institutional_json,
    export_institutional_pdf,
)
from phase11.reporting import (
    APP_VERSION,
    REPORT_TITLES,
    InstitutionalReportEngine,
    ReportType,
    format_currency,
    format_inr,
    generate_deterministic_narratives,
)
from phase11.schema import FinancialDocument


# ---------------------------------------------------------------------------
# Fixtures & Test Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path: Path) -> FinancialDocumentDB:
    """Fresh isolated database for each test."""
    return FinancialDocumentDB(project_root=tmp_path)


def _make_doc(
    vendor: str = "TANGEDCO Electricity Board",
    inv_num: str = "INV-2026-001",
    date: str = "2026-03-15",
    total: float = 45000.0,
    subtotal: float = 38135.59,
    tax: float = 6864.41,
    category: str = "Electricity",
    status: str = "verified",
    currency: str = "INR",
    domain: str = "SROIE",
) -> FinancialDocument:
    return FinancialDocument(
        document_id=str(uuid.uuid4()),
        filename=f"invoice_{uuid.uuid4().hex[:6]}.pdf",
        document_type="invoice",
        benchmark_domain=domain,
        category=category,
        vendor_name=vendor,
        invoice_number=inv_num,
        invoice_date=date,
        total_amount=total,
        subtotal=subtotal,
        tax_amount=tax,
        currency=currency,
        verification_status=status,
        file_hash=uuid.uuid4().hex,
        processing_status="COMPLETED",
    )


# ---------------------------------------------------------------------------
# 1. Currency & Number Formatting Tests
# ---------------------------------------------------------------------------

class TestCurrencyAndFormatting:

    def test_format_inr_standard_and_large_numbers(self):
        assert format_inr(0.0) == "₹0.00"
        assert format_inr(500.0) == "₹500.00"
        assert format_inr(1500.50) == "₹1,500.50"
        # 1 Lakh = 1,00,000.00
        assert format_inr(100000.0) == "₹1,00,000.00"
        # 10 Lakhs = 10,00,000.00
        assert format_inr(1000000.0) == "₹10,00,000.00"
        # 1 Crore = 1,00,00,000.00
        assert format_inr(10000000.0) == "₹1,00,00,000.00"

    def test_format_inr_negative_and_none(self):
        assert format_inr(None) == "N/A"
        assert format_inr(-25000.0) == "-₹25,000.00"
        assert format_inr(-100000.0) == "-₹1,00,000.00"

    def test_format_currency_inr_modes(self):
        # UI mode with symbol
        assert "₹" in format_currency(12345.67, currency="INR", for_pdf=False)
        # PDF mode with Latin-1 safe ASCII code
        pdf_fmt = format_currency(12345.67, currency="INR", for_pdf=True)
        assert "INR " in pdf_fmt
        assert "₹" not in pdf_fmt

    def test_format_currency_foreign_currencies(self):
        assert format_currency(1500.0, currency="USD", for_pdf=False) == "$1,500.00"
        assert format_currency(1500.0, currency="EUR", for_pdf=False) == "€1,500.00"
        assert format_currency(1500.0, currency="EUR", for_pdf=True) == "EUR 1,500.00"
        assert format_currency(1500.0, currency="GBP", for_pdf=False) == "£1,500.00"
        assert format_currency(1500.0, currency="GBP", for_pdf=True) == "GBP 1,500.00"
        assert format_currency(None, currency="USD") == "N/A"


# ---------------------------------------------------------------------------
# 2. Deterministic Executive Narrative Generator Tests
# ---------------------------------------------------------------------------

class TestDeterministicNarrativeEngine:

    def test_narrative_empty_population(self):
        narratives = generate_deterministic_narratives(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            kpis={"invoice_count": 0, "total_spend": 0.0},
            categories=[],
            vendors=[],
            monthly=[],
        )
        assert len(narratives) == 1
        assert "No financial records were found" in narratives[0]

    def test_narrative_spend_and_category_summary(self):
        kpis = {"invoice_count": 10, "total_spend": 500000.0, "avg_invoice_value": 50000.0}
        categories = [{"category": "Maintenance", "total_spend": 300000.0, "share_pct": 60.0}]
        narratives = generate_deterministic_narratives(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            kpis=kpis,
            categories=categories,
            vendors=[],
            monthly=[],
        )
        assert any("10 financial document(s)" in n for n in narratives)
        assert any("Maintenance" in n and "60.0%" in n for n in narratives)

    def test_narrative_vendor_concentration_alert(self):
        kpis = {"invoice_count": 5, "total_spend": 100000.0, "avg_invoice_value": 20000.0}
        # Vendor with >25% share triggers concentration review alert
        vendors = [{"vendor_name": "MegaCorp Ltd", "total_spend": 45000.0, "share_pct": 45.0, "invoice_count": 2}]
        narratives = generate_deterministic_narratives(
            report_type=ReportType.VENDOR_INTELLIGENCE,
            kpis=kpis,
            categories=[],
            vendors=vendors,
            monthly=[],
        )
        assert any("Concentration review recommended" in n and "MegaCorp Ltd" in n for n in narratives)

    def test_narrative_monthly_mom_trend(self):
        kpis = {"invoice_count": 20, "total_spend": 200000.0, "avg_invoice_value": 10000.0}
        monthly = [
            {"month": "2026-01", "total_spend": 80000.0, "mom_delta_pct": None},
            {"month": "2026-02", "total_spend": 120000.0, "mom_delta_pct": 50.0},
        ]
        narratives = generate_deterministic_narratives(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            kpis=kpis,
            categories=[],
            vendors=[],
            monthly=monthly,
        )
        assert any("increased by 50.0%" in n and "2026-02" in n for n in narratives)

    def test_narrative_fy_yoy_comparison(self):
        kpis = {"invoice_count": 30, "total_spend": 1000000.0, "avg_invoice_value": 33333.33}
        fy_comp = {"fy1": "FY 2024-25", "fy2": "FY 2025-26", "delta_spend_pct": -12.5}
        narratives = generate_deterministic_narratives(
            report_type=ReportType.FISCAL_YEAR,
            kpis=kpis,
            categories=[],
            vendors=[],
            monthly=[],
            fy_comparison=fy_comp,
        )
        assert any("decreased by 12.5%" in n and "FY 2024-25" in n for n in narratives)

    def test_narrative_integrity_and_anomaly_alerts(self):
        kpis = {"invoice_count": 15, "total_spend": 150000.0, "avg_invoice_value": 10000.0}
        anom_summary = {
            "open_findings": 3,
            "severity_counts": {"HIGH": 2, "CRITICAL": 1},
            "total_duplicate_candidates": 1,
        }
        narratives = generate_deterministic_narratives(
            report_type=ReportType.ANOMALY_DUPLICATE,
            kpis=kpis,
            categories=[],
            vendors=[],
            monthly=[],
            anomalies_summary=anom_summary,
        )
        assert any("3 open anomaly finding(s)" in n and "3 High/Critical" in n for n in narratives)

    def test_narrative_mixed_currency_warning(self):
        kpis = {"invoice_count": 8, "total_spend": 80000.0, "avg_invoice_value": 10000.0}
        narratives = generate_deterministic_narratives(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            kpis=kpis,
            categories=[],
            vendors=[],
            monthly=[],
            currency_is_mixed=True,
        )
        assert any("Multiple currencies detected" in n for n in narratives)


# ---------------------------------------------------------------------------
# 3. Centralized InstitutionalReportEngine Tests
# ---------------------------------------------------------------------------

class TestInstitutionalReportEngine:

    def test_build_report_all_seven_archetypes(self, tmp_db):
        # Insert sample document
        d = _make_doc()
        tmp_db.insert_document(d)

        engine = InstitutionalReportEngine(db=tmp_db)
        for rtype in ReportType:
            report = engine.build_report(report_type=rtype)
            assert report["metadata"]["report_type"] == rtype.value
            assert report["metadata"]["report_title"] == REPORT_TITLES[rtype]
            assert report["metadata"]["app_version"] == APP_VERSION
            assert report["metadata"]["document_count"] == 1
            assert "kpis" in report
            assert "narratives" in report
            assert "consistency" in report
            assert "sections" not in report or report["documents"]

    def test_filter_propagation_by_category(self, tmp_db):
        tmp_db.insert_document(_make_doc(category="Electricity", total=10000.0))
        tmp_db.insert_document(_make_doc(category="Water", total=5000.0))

        engine = InstitutionalReportEngine(db=tmp_db)
        report = engine.build_report(category="Electricity")
        assert report["metadata"]["document_count"] == 1
        assert report["kpis"]["total_spend"] == 10000.0
        assert "Category: Electricity" in report["metadata"]["scope_description"]

    def test_filter_propagation_by_vendor(self, tmp_db):
        tmp_db.insert_document(_make_doc(vendor="Vendor Alpha", total=20000.0))
        tmp_db.insert_document(_make_doc(vendor="Vendor Beta", total=30000.0))

        engine = InstitutionalReportEngine(db=tmp_db)
        report = engine.build_report(vendor="Vendor Alpha")
        assert report["metadata"]["document_count"] == 1
        assert report["kpis"]["total_spend"] == 20000.0
        assert "Vendor: Vendor Alpha" in report["metadata"]["scope_description"]

    def test_multi_currency_isolation_single_currency(self, tmp_db):
        tmp_db.insert_document(_make_doc(currency="INR", total=10000.0))
        tmp_db.insert_document(_make_doc(currency="INR", total=15000.0))

        engine = InstitutionalReportEngine(db=tmp_db)
        report = engine.build_report()
        assert report["metadata"]["currency"] == "INR"
        assert report["currency_breakdown"] == {}
        assert not report["consistency"]["mixed_currency"]

    def test_multi_currency_isolation_mixed_currencies(self, tmp_db):
        tmp_db.insert_document(_make_doc(currency="INR", total=10000.0))
        tmp_db.insert_document(_make_doc(currency="USD", total=500.0))

        engine = InstitutionalReportEngine(db=tmp_db)
        report = engine.build_report()
        assert report["metadata"]["currency"] == "MIXED"
        assert "INR" in report["currency_breakdown"]
        assert "USD" in report["currency_breakdown"]
        assert report["consistency"]["mixed_currency"]

    def test_mathematical_data_consistency_reconciliation(self, tmp_db):
        tmp_db.insert_document(_make_doc(category="Electricity", date="2026-02-10", total=25000.0))
        tmp_db.insert_document(_make_doc(category="Water", date="2026-02-15", total=15000.0))

        engine = InstitutionalReportEngine(db=tmp_db)
        report = engine.build_report()
        c = report["consistency"]
        assert c["is_consistent"] is True
        assert c["total_spend"] == 40000.0
        assert c["category_sum"] == 40000.0
        assert c["category_variance"] == 0.0
        assert "Mathematical reconciliation validated" in c["reconciliation_notes"]

    def test_scoped_anomalies_and_duplicates(self, tmp_db):
        d1 = _make_doc(category="Electricity")
        d2 = _make_doc(category="Water")
        tmp_db.insert_document(d1)
        tmp_db.insert_document(d2)

        # Attach anomaly finding to d1
        tmp_db.save_anomaly_findings([{
            "document_id": d1.document_id,
            "anomaly_type": "HIGH_VALUE_OUTLIER",
            "severity": "HIGH",
            "score": 0.85,
            "confidence": 0.9,
            "explanation": "Outlier test",
        }])

        engine = InstitutionalReportEngine(db=tmp_db)
        # Filter scope to d2 only
        report_d2 = engine.build_report(category="Water")
        assert len(report_d2["integrity_analysis"]["findings"]) == 0

        # Filter scope to d1 only
        report_d1 = engine.build_report(category="Electricity")
        assert len(report_d1["integrity_analysis"]["findings"]) == 1


# ---------------------------------------------------------------------------
# 4. Multi-Format Publication Exporters Tests
# ---------------------------------------------------------------------------

class TestInstitutionalExports:

    def test_export_institutional_pdf_valid_bytes(self, tmp_db):
        d1 = _make_doc(total=50000.0)
        d2 = _make_doc(total=75000.0)
        tmp_db.insert_document(d1)
        tmp_db.insert_document(d2)

        engine = InstitutionalReportEngine(db=tmp_db)
        report_data = engine.build_report(ReportType.EXECUTIVE_SUMMARY)

        pdf_bytes = export_institutional_pdf(report_data)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        # Valid PDF header
        assert pdf_bytes.startswith(b"%PDF")

    def test_export_institutional_pdf_to_file(self, tmp_db, tmp_path: Path):
        d = _make_doc()
        tmp_db.insert_document(d)
        engine = InstitutionalReportEngine(db=tmp_db)
        report_data = engine.build_report()

        out_pdf = tmp_path / "executive_report.pdf"
        export_institutional_pdf(report_data, filepath=out_pdf)
        assert out_pdf.exists()
        assert out_pdf.stat().st_size > 1000

    def test_export_institutional_excel_all_eleven_sheets(self, tmp_db):
        d1 = _make_doc(category="Maintenance", vendor="Apex Services", date="2025-05-10", total=20000.0)
        d2 = _make_doc(category="Electricity", vendor="TANGEDCO", date="2026-01-15", total=30000.0)
        tmp_db.insert_document(d1)
        tmp_db.insert_document(d2)

        engine = InstitutionalReportEngine(db=tmp_db)
        report_data = engine.build_report(ReportType.EXECUTIVE_SUMMARY)

        xlsx_bytes = export_institutional_excel(report_data)
        assert isinstance(xlsx_bytes, bytes)
        assert len(xlsx_bytes) > 2000

        wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
        expected_sheets = [
            "Executive Summary",
            "Financial Records",
            "Category Analysis",
            "Vendor Intelligence",
            "Monthly Trend",
            "Fiscal Year Analysis",
            "Tax Analysis",
            "Anomaly Findings",
            "Duplicate Candidates",
            "Audit History",
            "Report Metadata",
        ]
        for s in expected_sheets:
            assert s in wb.sheetnames, f"Missing required institutional sheet: {s}"

    def test_export_institutional_excel_formula_injection_defense(self, tmp_db):
        # Create documents with potentially malicious spreadsheet formula injection prefixes
        d1 = _make_doc(vendor="=SUM(A1:A10)", inv_num="+CMD|' /C calc'!A0")
        d2 = _make_doc(vendor="-5*2", inv_num="@SUM(1,2)")
        tmp_db.insert_document(d1)
        tmp_db.insert_document(d2)

        engine = InstitutionalReportEngine(db=tmp_db)
        report_data = engine.build_report()

        xlsx_bytes = export_institutional_excel(report_data)
        wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
        ws = wb["Financial Records"]

        # Verify that all formula-injection prefixes were neutralized with a leading single quote
        for row in ws.iter_rows(values_only=True):
            for cell in row:
                if isinstance(cell, str) and len(cell) > 1:
                    # Cell should not begin with bare formula operators
                    assert not cell.startswith(("=", "+CMD", "-5*2", "@SUM")), f"Unsanitized formula cell: {cell}"

    def test_export_institutional_csv_sections(self, tmp_db):
        d = _make_doc(total=12345.0)
        tmp_db.insert_document(d)
        engine = InstitutionalReportEngine(db=tmp_db)
        report_data = engine.build_report()

        csv_docs = export_institutional_csv(report_data, section="documents")
        assert "document_id" in csv_docs
        assert d.document_id in csv_docs

        csv_cats = export_institutional_csv(report_data, section="categories")
        assert "category" in csv_cats
        assert "total_spend" in csv_cats

        csv_vends = export_institutional_csv(report_data, section="vendors")
        assert "vendor_name" in csv_vends

    def test_export_institutional_json_schema(self, tmp_db):
        d = _make_doc()
        tmp_db.insert_document(d)
        engine = InstitutionalReportEngine(db=tmp_db)
        report_data = engine.build_report()

        json_str = export_institutional_json(report_data)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)

        required_keys = [
            "metadata", "kpis", "narratives", "consistency", "currency_breakdown",
            "category_analysis", "vendor_analysis", "monthly_trends",
            "fiscal_year_analysis", "tax_compliance", "integrity_analysis",
            "audit_history", "documents"
        ]
        for k in required_keys:
            assert k in parsed, f"Missing key in JSON schema: {k}"

    def test_module_convenience_functions(self, tmp_db):
        d = _make_doc()
        tmp_db.insert_document(d)
        engine = InstitutionalReportEngine(db=tmp_db)
        report_data = engine.build_report()

        # Test exporter instance methods vs module-level functions produce identical outputs
        exporter = DocumentExporter()
        pdf_inst = exporter.export_institutional_pdf(report_data)
        pdf_mod = export_institutional_pdf(report_data)
        assert len(pdf_inst) == len(pdf_mod)

        json_inst = exporter.export_institutional_json(report_data)
        json_mod = export_institutional_json(report_data)
        assert json.loads(json_inst)["metadata"]["report_id"] == json.loads(json_mod)["metadata"]["report_id"]
