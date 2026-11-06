"""
Phase 14 Test Suite — Advanced Financial Analytics & Institutional Dashboard.

Tests cover:
  - kpi_summary: total spend, count, avg, tax NULL handling, verification rates
  - spend_by_fiscal_year: Indian FY (April–March) grouping
  - yoy_comparison: YoY delta % calculations
  - monthly_trend_with_delta: MoM delta, rolling 3-month average
  - category_concentration: HHI index, share %, diversity interpretation
  - vendor_summary: extended stats, share %, concentration risk flag
  - tax_analysis: coverage %, effective rate, zero-tax docs, NULL handling
  - advanced_anomaly_report: sigma outliers, duplicate invoices, missing data matrix
  - export_dashboard_excel: multi-sheet Excel workbook structure
  - Filter-awareness across all analytics methods
  - Indian FY boundary computation (_fy_bounds static method)
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

from phase11.database import FinancialDocumentDB
from phase11.export import DocumentExporter
from phase11.schema import FinancialDocument


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path: Path) -> FinancialDocumentDB:
    """Fresh isolated database for each test."""
    return FinancialDocumentDB(project_root=tmp_path)


def _make_doc(
    vendor: str = "ACME Corp",
    category: str = "Electricity",
    date: str = "2025-06-15",
    total: float = 10000.0,
    tax: float = 1800.0,
    subtotal: float = 8200.0,
    status: str = "pending",
    inv_num: str = None,
    currency: str = "INR",
) -> FinancialDocument:
    return FinancialDocument(
        document_id=str(uuid.uuid4()),
        filename=f"bill_{uuid.uuid4().hex[:6]}.pdf",
        document_type="invoice",
        benchmark_domain="SROIE",
        category=category,
        vendor_name=vendor,
        invoice_date=date,
        invoice_number=inv_num or f"INV-{uuid.uuid4().hex[:8].upper()}",
        total_amount=total,
        tax_amount=tax,
        subtotal=subtotal,
        currency=currency,
        verification_status=status,
        classification_confidence=0.92,
        processing_timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _seed_db(db: FinancialDocumentDB) -> None:
    """Insert a realistic multi-year, multi-category dataset."""
    docs = [
        # FY 2024-25 (April 2024 – March 2025)
        _make_doc("TANGEDCO", "Electricity", "2024-04-10", 25000, 4500, 20500, "verified"),
        _make_doc("TANGEDCO", "Electricity", "2024-07-15", 27500, 4950, 22550, "verified"),
        _make_doc("BSNL", "Internet/Telecom", "2024-06-20", 5000, 900, 4100, "verified"),
        _make_doc("Airtel", "Internet/Telecom", "2024-09-05", 4500, 810, 3690, "pending"),
        _make_doc("Office Depot", "Office Supplies", "2024-11-01", 8000, 1440, 6560, "pending"),
        _make_doc("Chennai Metro Water", "Water", "2025-01-10", 3000, 540, 2460, "verified"),
        _make_doc("Indane Gas", "Gas", "2025-02-14", 2000, 360, 1640, "pending"),
        _make_doc("TCS Travel", "Transportation", "2025-03-20", 15000, None, None, "flagged"),

        # FY 2025-26 (April 2025 – March 2026)
        _make_doc("TANGEDCO", "Electricity", "2025-04-12", 30000, 5400, 24600, "verified"),
        _make_doc("TANGEDCO", "Electricity", "2025-07-18", 32000, 5760, 26240, "verified"),
        _make_doc("TANGEDCO", "Electricity", "2025-10-09", 28000, 5040, 22960, "modified"),
        _make_doc("BSNL", "Internet/Telecom", "2025-05-25", 5500, 990, 4510, "verified"),
        _make_doc("Airtel", "Internet/Telecom", "2025-08-15", 4800, 864, 3936, "pending"),
        _make_doc("Office Depot", "Office Supplies", "2025-09-01", 9500, 1710, 7790, "verified"),
        _make_doc("Chennai Metro Water", "Water", "2025-12-20", 3200, 576, 2624, "pending"),
        _make_doc("Indane Gas", "Gas", "2026-01-15", 2200, 396, 1804, "verified"),
        _make_doc("Hotel Grand", "Accommodation", "2026-02-10", 45000, None, None, "flagged"),
        _make_doc("TCS Travel", "Transportation", "2026-03-25", 18000, None, None, "pending"),
    ]
    for doc in docs:
        db.insert_document(doc)


# ---------------------------------------------------------------------------
# A. KPI Summary Tests
# ---------------------------------------------------------------------------

class TestKPISummary:

    def test_basic_kpi_structure(self, tmp_db):
        """kpi_summary returns all expected keys."""
        _seed_db(tmp_db)
        kpi = tmp_db.kpi_summary()
        required_keys = [
            "total_spend", "invoice_count", "avg_invoice_value",
            "tax_total", "tax_available_count", "tax_unavailable_count",
            "subtotal_total", "effective_tax_rate",
            "verified_count", "pending_count", "modified_count",
            "flagged_count", "rejected_count",
            "verified_rate", "avg_confidence",
        ]
        for k in required_keys:
            assert k in kpi, f"Missing key: {k}"

    def test_invoice_count(self, tmp_db):
        _seed_db(tmp_db)
        kpi = tmp_db.kpi_summary()
        assert kpi["invoice_count"] == 18

    def test_total_spend_accuracy(self, tmp_db):
        """Total spend equals sum of all total_amounts."""
        _seed_db(tmp_db)
        kpi = tmp_db.kpi_summary()
        expected = sum([
            25000, 27500, 5000, 4500, 8000, 3000, 2000, 15000,
            30000, 32000, 28000, 5500, 4800, 9500, 3200, 2200, 45000, 18000,
        ])
        assert abs(kpi["total_spend"] - expected) < 0.01

    def test_tax_null_handling_unavailable(self, tmp_db):
        """Tax is reported as None when no documents have tax data."""
        doc = _make_doc(total=5000.0, tax=None)
        tmp_db.insert_document(doc)
        kpi = tmp_db.kpi_summary()
        assert kpi["tax_total"] is None
        assert kpi["tax_available_count"] == 0
        assert kpi["tax_unavailable_count"] == 1
        assert kpi["effective_tax_rate"] is None

    def test_tax_available_when_some_have_tax(self, tmp_db):
        """Tax total is computed only for documents that have tax data."""
        db = tmp_db
        db.insert_document(_make_doc(total=10000, tax=1800, subtotal=8200))
        db.insert_document(_make_doc(total=5000, tax=None))
        kpi = db.kpi_summary()
        assert kpi["tax_total"] == 1800.0
        assert kpi["tax_available_count"] == 1
        assert kpi["tax_unavailable_count"] == 1

    def test_verified_rate(self, tmp_db):
        """Verified rate = verified_count / total_count."""
        _seed_db(tmp_db)
        kpi = tmp_db.kpi_summary()
        expected_rate = kpi["verified_count"] / kpi["invoice_count"]
        assert abs(kpi["verified_rate"] - expected_rate) < 1e-6

    def test_status_breakdown_sums_to_total(self, tmp_db):
        _seed_db(tmp_db)
        kpi = tmp_db.kpi_summary()
        status_sum = (
            kpi["verified_count"] + kpi["pending_count"] +
            kpi["modified_count"] + kpi["flagged_count"] + kpi["rejected_count"]
        )
        assert status_sum == kpi["invoice_count"]

    def test_empty_database(self, tmp_db):
        kpi = tmp_db.kpi_summary()
        assert kpi["invoice_count"] == 0
        assert kpi["total_spend"] == 0.0
        assert kpi["tax_total"] is None
        assert kpi["verified_rate"] == 0.0

    def test_filter_awareness(self, tmp_db):
        """kpi_summary respects category filter."""
        _seed_db(tmp_db)
        kpi_all = tmp_db.kpi_summary()
        kpi_elec = tmp_db.kpi_summary(category="Electricity")
        assert kpi_elec["invoice_count"] < kpi_all["invoice_count"]
        assert kpi_elec["total_spend"] < kpi_all["total_spend"]


# ---------------------------------------------------------------------------
# B. Indian Financial Year Tests
# ---------------------------------------------------------------------------

class TestFYBounds:

    def test_fy_label_with_prefix(self):
        start, end = FinancialDocumentDB._fy_bounds("FY 2025-26")
        assert start == "2025-04-01"
        assert end == "2026-03-31"

    def test_fy_label_without_prefix(self):
        start, end = FinancialDocumentDB._fy_bounds("2024-25")
        assert start == "2024-04-01"
        assert end == "2025-03-31"

    def test_fy_label_four_digit_end(self):
        start, end = FinancialDocumentDB._fy_bounds("FY 2023-2024")
        assert start == "2023-04-01"
        assert end == "2024-03-31"

    def test_invalid_fy_label_raises(self):
        with pytest.raises(ValueError):
            FinancialDocumentDB._fy_bounds("not-a-fy")

    def test_spend_by_fiscal_year_count(self, tmp_db):
        _seed_db(tmp_db)
        fy_data = tmp_db.spend_by_fiscal_year()
        # Should have FY 2024-25 and FY 2025-26
        fy_labels = [r["fiscal_year"] for r in fy_data]
        assert len(fy_labels) == 2

    def test_fy_april_date_in_current_year(self, tmp_db):
        """Invoice dated April goes into the FY starting that year."""
        doc = _make_doc(date="2025-04-10", total=10000, tax=None)
        tmp_db.insert_document(doc)
        fy_data = tmp_db.spend_by_fiscal_year()
        assert len(fy_data) == 1
        assert "2025" in fy_data[0]["fiscal_year"]

    def test_fy_march_date_in_previous_year(self, tmp_db):
        """Invoice dated March goes into the FY starting the prior year."""
        doc = _make_doc(date="2025-03-15", total=10000, tax=None)
        tmp_db.insert_document(doc)
        fy_data = tmp_db.spend_by_fiscal_year()
        assert len(fy_data) == 1
        assert "2024" in fy_data[0]["fiscal_year"]


# ---------------------------------------------------------------------------
# C. YoY Comparison Tests
# ---------------------------------------------------------------------------

class TestYoYComparison:

    def test_yoy_structure(self, tmp_db):
        _seed_db(tmp_db)
        result = tmp_db.yoy_comparison("FY 2024-25", "FY 2025-26")
        assert "fy1" in result and "fy2" in result
        assert "fy1_data" in result and "fy2_data" in result
        assert "delta_spend_pct" in result
        assert "delta_count_pct" in result

    def test_yoy_fy1_spend_less_than_fy2(self, tmp_db):
        """Spend in FY2 should be higher than FY1 given the seed data."""
        _seed_db(tmp_db)
        result = tmp_db.yoy_comparison("FY 2024-25", "FY 2025-26")
        assert result["fy2_data"]["total_spend"] > result["fy1_data"]["total_spend"]
        assert result["delta_spend_pct"] is not None
        assert result["delta_spend_pct"] > 0

    def test_yoy_empty_fy_returns_zero(self, tmp_db):
        _seed_db(tmp_db)
        result = tmp_db.yoy_comparison("FY 2020-21", "FY 2025-26")
        assert result["fy1_data"]["total_spend"] == 0
        assert result["delta_spend_pct"] is None  # Division by zero guard

    def test_yoy_count_delta(self, tmp_db):
        _seed_db(tmp_db)
        result = tmp_db.yoy_comparison("FY 2024-25", "FY 2025-26")
        d1 = result["fy1_data"]["count"]
        d2 = result["fy2_data"]["count"]
        expected_pct = (d2 - d1) / d1 * 100 if d1 else None
        if expected_pct is not None:
            assert abs(result["delta_count_pct"] - expected_pct) < 1e-6


# ---------------------------------------------------------------------------
# D. Monthly Trend with Delta Tests
# ---------------------------------------------------------------------------

class TestMonthlyTrend:

    def test_monthly_trend_structure(self, tmp_db):
        _seed_db(tmp_db)
        trend = tmp_db.monthly_trend_with_delta()
        assert len(trend) > 0
        for row in trend:
            assert "month" in row
            assert "count" in row
            assert "total_spend" in row
            assert "mom_delta_pct" in row
            assert "rolling_avg_3m" in row

    def test_first_month_delta_is_none(self, tmp_db):
        _seed_db(tmp_db)
        trend = tmp_db.monthly_trend_with_delta()
        assert trend[0]["mom_delta_pct"] is None

    def test_rolling_avg_starts_at_index_2(self, tmp_db):
        _seed_db(tmp_db)
        trend = tmp_db.monthly_trend_with_delta()
        assert trend[0]["rolling_avg_3m"] is None
        assert trend[1]["rolling_avg_3m"] is None
        if len(trend) >= 3:
            assert trend[2]["rolling_avg_3m"] is not None

    def test_mom_delta_calculation(self, tmp_db):
        """MoM delta = (current - previous) / previous * 100."""
        for date, total in [("2025-05-10", 10000.0), ("2025-06-15", 12000.0)]:
            tmp_db.insert_document(_make_doc(date=date, total=total, tax=None))
        trend = tmp_db.monthly_trend_with_delta()
        assert len(trend) == 2
        expected_delta = (12000 - 10000) / 10000 * 100
        assert abs(trend[1]["mom_delta_pct"] - expected_delta) < 1e-6

    def test_rolling_avg_accuracy(self, tmp_db):
        """Rolling 3m avg should equal sum of 3 months / 3."""
        amounts = [10000.0, 15000.0, 20000.0]
        dates = ["2025-04-01", "2025-05-01", "2025-06-01"]
        for d, t in zip(dates, amounts):
            tmp_db.insert_document(_make_doc(date=d, total=t, tax=None))
        trend = tmp_db.monthly_trend_with_delta()
        assert len(trend) == 3
        expected_avg = sum(amounts) / 3
        assert abs(trend[2]["rolling_avg_3m"] - expected_avg) < 0.01

    def test_trend_ordered_by_month(self, tmp_db):
        _seed_db(tmp_db)
        trend = tmp_db.monthly_trend_with_delta()
        months = [r["month"] for r in trend]
        assert months == sorted(months)


# ---------------------------------------------------------------------------
# E. Category Concentration Tests
# ---------------------------------------------------------------------------

class TestCategoryConcentration:

    def test_concentration_structure(self, tmp_db):
        _seed_db(tmp_db)
        cc = tmp_db.category_concentration()
        assert "categories" in cc
        assert "hhi" in cc
        assert "hhi_interpretation" in cc
        assert "top1_share_pct" in cc
        assert "top3_share_pct" in cc
        assert "category_count" in cc

    def test_share_sums_to_100(self, tmp_db):
        _seed_db(tmp_db)
        cc = tmp_db.category_concentration()
        total_share = sum(r["share_pct"] for r in cc["categories"])
        assert abs(total_share - 100.0) < 0.1  # Allow tiny floating-point drift

    def test_hhi_monopoly(self, tmp_db):
        """Single-category database should have HHI = 10000."""
        for _ in range(5):
            tmp_db.insert_document(_make_doc(category="Electricity", total=10000, tax=None))
        cc = tmp_db.category_concentration()
        assert abs(cc["hhi"] - 10000.0) < 0.1
        assert cc["hhi_interpretation"] == "Highly Concentrated"

    def test_hhi_diversified(self, tmp_db):
        """Equal spend across 20 categories gives HHI = 500 (Diversified)."""
        cats = [f"Category_{i}" for i in range(20)]
        for cat in cats:
            tmp_db.insert_document(_make_doc(category=cat, total=5000, tax=None))
        cc = tmp_db.category_concentration()
        assert cc["hhi"] < 1500
        assert cc["hhi_interpretation"] == "Diversified"

    def test_top1_share_is_largest(self, tmp_db):
        _seed_db(tmp_db)
        cc = tmp_db.category_concentration()
        if cc["categories"]:
            # top1_share_pct is rounded(2); compare with tolerance
            assert abs(cc["top1_share_pct"] - cc["categories"][0]["share_pct"]) < 0.01


    def test_empty_database(self, tmp_db):
        cc = tmp_db.category_concentration()
        assert cc["hhi"] == 0.0
        assert cc["hhi_interpretation"] == "No data"
        assert cc["categories"] == []


# ---------------------------------------------------------------------------
# F. Vendor Summary Tests
# ---------------------------------------------------------------------------

class TestVendorSummary:

    def test_vendor_summary_structure(self, tmp_db):
        _seed_db(tmp_db)
        vendors = tmp_db.vendor_summary(n=5)
        assert len(vendors) <= 5
        for v in vendors:
            assert "vendor_name" in v
            assert "invoice_count" in v
            assert "total_spend" in v
            assert "share_pct" in v
            assert "avg_invoice" in v
            assert "last_invoice_date" in v
            assert "concentration_risk" in v

    def test_share_pct_non_negative(self, tmp_db):
        _seed_db(tmp_db)
        vendors = tmp_db.vendor_summary()
        for v in vendors:
            assert v["share_pct"] >= 0

    def test_concentration_risk_threshold(self, tmp_db):
        """Vendor with >30% share should have concentration_risk = True."""
        # TANGEDCO has most spend across both FYs
        _seed_db(tmp_db)
        vendors = tmp_db.vendor_summary(n=10)
        top = vendors[0]
        if top["share_pct"] > 30:
            assert top["concentration_risk"] is True
        else:
            assert top["concentration_risk"] is False

    def test_vendor_invoice_count_accuracy(self, tmp_db):
        for _ in range(4):
            tmp_db.insert_document(_make_doc(vendor="BigCo Ltd", total=5000, tax=None))
        vendors = tmp_db.vendor_summary()
        bigco = next(v for v in vendors if v["vendor_name"] == "BigCo Ltd")
        assert bigco["invoice_count"] == 4

    def test_vendor_summary_respects_category_filter(self, tmp_db):
        _seed_db(tmp_db)
        all_vendors = tmp_db.vendor_summary()
        elec_vendors = tmp_db.vendor_summary(category="Electricity")
        # Only electricity vendors
        all_names = {v["vendor_name"] for v in all_vendors}
        elec_names = {v["vendor_name"] for v in elec_vendors}
        assert elec_names.issubset(all_names)


# ---------------------------------------------------------------------------
# G. Tax Analysis Tests
# ---------------------------------------------------------------------------

class TestTaxAnalysis:

    def test_tax_analysis_structure(self, tmp_db):
        _seed_db(tmp_db)
        ta = tmp_db.tax_analysis()
        assert "coverage_pct" in ta
        assert "total_tax" in ta
        assert "avg_effective_rate" in ta
        assert "min_effective_rate" in ta
        assert "max_effective_rate" in ta
        assert "zero_tax_count" in ta
        assert "null_tax_count" in ta
        assert "tax_by_category" in ta

    def test_tax_null_when_all_missing(self, tmp_db):
        """total_tax must be None if every document has NULL tax."""
        for _ in range(3):
            tmp_db.insert_document(_make_doc(total=10000, tax=None))
        ta = tmp_db.tax_analysis()
        assert ta["total_tax"] is None
        assert ta["coverage_pct"] == 0.0
        assert ta["null_tax_count"] == 3

    def test_zero_tax_detection(self, tmp_db):
        """Documents with tax_amount=0 but total>0 counted as zero-tax."""
        tmp_db.insert_document(_make_doc(total=10000, tax=0.0, subtotal=10000))
        tmp_db.insert_document(_make_doc(total=5000, tax=900, subtotal=4100))
        ta = tmp_db.tax_analysis()
        assert ta["zero_tax_count"] == 1

    def test_coverage_pct_partial(self, tmp_db):
        tmp_db.insert_document(_make_doc(total=10000, tax=1800, subtotal=8200))
        tmp_db.insert_document(_make_doc(total=5000, tax=None))
        ta = tmp_db.tax_analysis()
        assert ta["coverage_pct"] == 50.0

    def test_effective_rate_computed(self, tmp_db):
        """avg_effective_rate = tax/subtotal for docs with both fields."""
        tmp_db.insert_document(_make_doc(total=11800, tax=1800, subtotal=10000))
        ta = tmp_db.tax_analysis()
        assert ta["avg_effective_rate"] is not None
        assert abs(ta["avg_effective_rate"] - 0.18) < 0.001

    def test_tax_by_category_nonempty(self, tmp_db):
        _seed_db(tmp_db)
        ta = tmp_db.tax_analysis()
        assert len(ta["tax_by_category"]) > 0

    def test_filter_awareness(self, tmp_db):
        _seed_db(tmp_db)
        ta_all = tmp_db.tax_analysis()
        ta_elec = tmp_db.tax_analysis(category="Electricity")
        assert ta_elec["total_tax"] <= ta_all["total_tax"]


# ---------------------------------------------------------------------------
# H. Advanced Anomaly Report Tests
# ---------------------------------------------------------------------------

class TestAdvancedAnomalyReport:

    def test_anomaly_report_structure(self, tmp_db):
        _seed_db(tmp_db)
        ar = tmp_db.advanced_anomaly_report()
        assert "sigma_threshold" in ar
        assert "mean_amount" in ar
        assert "std_amount" in ar
        assert "sigma_outliers" in ar
        assert "duplicate_invoices" in ar
        assert "missing_data_matrix" in ar
        assert "round_amount_count" in ar
        assert "monthly_anomaly_count" in ar

    def test_sigma_threshold_default(self, tmp_db):
        _seed_db(tmp_db)
        ar = tmp_db.advanced_anomaly_report()
        assert ar["sigma_threshold"] == 2.0

    def test_sigma_outlier_detection(self, tmp_db):
        """Document with amount >> mean+2σ must appear in sigma_outliers."""
        for _ in range(10):
            tmp_db.insert_document(_make_doc(total=5000, tax=None))
        # One extreme outlier
        tmp_db.insert_document(_make_doc(total=500000, tax=None))
        ar = tmp_db.advanced_anomaly_report(sigma_threshold=2.0)
        assert len(ar["sigma_outliers"]) >= 1
        outlier_amounts = [o["total_amount"] for o in ar["sigma_outliers"]]
        assert 500000 in outlier_amounts

    def test_no_outliers_when_uniform(self, tmp_db):
        """No sigma outliers when all amounts are equal."""
        for _ in range(10):
            tmp_db.insert_document(_make_doc(total=10000, tax=None))
        ar = tmp_db.advanced_anomaly_report()
        assert len(ar["sigma_outliers"]) == 0

    def test_duplicate_invoice_detection(self, tmp_db):
        """Same invoice_number on two docs should appear in duplicate_invoices."""
        inv = "DUP-2025-001"
        tmp_db.insert_document(_make_doc(inv_num=inv, total=5000, tax=None))
        tmp_db.insert_document(_make_doc(inv_num=inv, total=6000, tax=None))
        ar = tmp_db.advanced_anomaly_report()
        dup_nums = [d["invoice_number"] for d in ar["duplicate_invoices"]]
        assert inv in dup_nums

    def test_missing_data_matrix_fields(self, tmp_db):
        """Missing data matrix must include vendor_name, invoice_date, total_amount."""
        _seed_db(tmp_db)
        ar = tmp_db.advanced_anomaly_report()
        matrix = ar["missing_data_matrix"]
        assert "vendor_name" in matrix
        assert "invoice_date" in matrix
        assert "total_amount" in matrix
        assert "tax_amount" in matrix

    def test_round_amount_detection(self, tmp_db):
        """Amount that is exact multiple of 1000 flagged as round."""
        tmp_db.insert_document(_make_doc(total=50000, tax=None))  # multiple of 1000
        tmp_db.insert_document(_make_doc(total=37250, tax=None))  # not a multiple
        ar = tmp_db.advanced_anomaly_report()
        assert ar["round_amount_count"] >= 1

    def test_custom_sigma_threshold(self, tmp_db):
        """Higher sigma threshold should find fewer (or equal) outliers."""
        for _ in range(20):
            tmp_db.insert_document(_make_doc(total=5000, tax=None))
        tmp_db.insert_document(_make_doc(total=100000, tax=None))
        ar_2sigma = tmp_db.advanced_anomaly_report(sigma_threshold=2.0)
        ar_3sigma = tmp_db.advanced_anomaly_report(sigma_threshold=3.0)
        assert len(ar_2sigma["sigma_outliers"]) >= len(ar_3sigma["sigma_outliers"])

    def test_filter_awareness(self, tmp_db):
        _seed_db(tmp_db)
        ar_all = tmp_db.advanced_anomaly_report()
        ar_elec = tmp_db.advanced_anomaly_report(category="Electricity")
        assert ar_elec["total_count"] <= ar_all["total_count"]


# ---------------------------------------------------------------------------
# I. Dashboard Excel Export Tests
# ---------------------------------------------------------------------------

class TestExportDashboardExcel:

    def _build_sample_data(self, db: FinancialDocumentDB) -> Dict[str, Any]:
        _seed_db(db)
        return {
            "kpi_data":      db.kpi_summary(),
            "category_data": db.category_concentration()["categories"],
            "vendor_data":   db.vendor_summary(n=5),
            "tax_data":      db.tax_analysis(),
            "monthly_data":  db.monthly_trend_with_delta(),
            "anomaly_data":  db.advanced_anomaly_report(),
            "fy_data":       db.spend_by_fiscal_year(),
        }

    def test_export_returns_bytes(self, tmp_db):
        data = self._build_sample_data(tmp_db)
        exporter = DocumentExporter()
        result = exporter.export_dashboard_excel(**data)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_export_is_valid_xlsx(self, tmp_db):
        """Output must be parseable as an xlsx workbook."""
        try:
            import openpyxl
        except ImportError:
            pytest.skip("openpyxl not installed")
        data = self._build_sample_data(tmp_db)
        exporter = DocumentExporter()
        xlsx_bytes = exporter.export_dashboard_excel(**data)
        import io
        wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
        sheet_names = wb.sheetnames
        assert "Summary KPIs" in sheet_names
        assert "Category Analysis" in sheet_names
        assert "Vendor Analysis" in sheet_names
        assert "Tax Compliance" in sheet_names
        assert "Monthly Trend" in sheet_names
        assert "Anomaly Report" in sheet_names
        assert "FY Comparison" in sheet_names

    def test_kpi_sheet_has_content(self, tmp_db):
        """Summary KPIs sheet must have at least 10 rows."""
        try:
            import openpyxl
            import io
        except ImportError:
            pytest.skip("openpyxl not installed")
        data = self._build_sample_data(tmp_db)
        exporter = DocumentExporter()
        wb = openpyxl.load_workbook(io.BytesIO(exporter.export_dashboard_excel(**data)))
        ws = wb["Summary KPIs"]
        assert ws.max_row >= 10

    def test_tax_unavailable_text_when_null(self, tmp_db):
        """When tax is unavailable, KPI sheet should contain the string 'Tax data unavailable'."""
        try:
            import openpyxl
            import io
        except ImportError:
            pytest.skip("openpyxl not installed")
        # Insert document with no tax
        tmp_db.insert_document(_make_doc(total=10000, tax=None))
        kpi = tmp_db.kpi_summary()
        assert kpi["tax_total"] is None
        exporter = DocumentExporter()
        data = {
            "kpi_data": kpi,
            "category_data": [],
            "vendor_data": [],
            "tax_data": tmp_db.tax_analysis(),
            "monthly_data": [],
            "anomaly_data": tmp_db.advanced_anomaly_report(),
        }
        wb = openpyxl.load_workbook(io.BytesIO(exporter.export_dashboard_excel(**data)))
        ws = wb["Summary KPIs"]
        cell_values = [ws.cell(row=r, column=2).value for r in range(1, ws.max_row + 1)]
        assert "Tax data unavailable" in cell_values

    def test_filter_description_in_kpi_sheet(self, tmp_db):
        """filter_description argument appears in the Summary KPIs sheet."""
        try:
            import openpyxl
            import io
        except ImportError:
            pytest.skip("openpyxl not installed")
        _seed_db(tmp_db)
        exporter = DocumentExporter()
        data = self._build_sample_data(tmp_db)
        data["filter_description"] = "Category: Electricity"
        wb = openpyxl.load_workbook(io.BytesIO(exporter.export_dashboard_excel(**data)))
        ws = wb["Summary KPIs"]
        all_values = [ws.cell(row=r, column=2).value for r in range(1, ws.max_row + 1)]
        assert "Category: Electricity" in all_values


# ---------------------------------------------------------------------------
# J. Integration — Filter-Awareness Across All Phase 14 Methods
# ---------------------------------------------------------------------------

class TestPhase14FilterAwareness:

    def test_all_methods_respect_category_filter(self, tmp_db):
        _seed_db(tmp_db)
        # Full dataset
        kpi_all = tmp_db.kpi_summary()
        cat_all  = tmp_db.category_concentration()
        vend_all = tmp_db.vendor_summary()
        tax_all  = tmp_db.tax_analysis()
        trend_all = tmp_db.monthly_trend_with_delta()

        # Filtered
        kpi_f  = tmp_db.kpi_summary(category="Electricity")
        cat_f  = tmp_db.category_concentration(category="Electricity")
        vend_f = tmp_db.vendor_summary(category="Electricity")
        tax_f  = tmp_db.tax_analysis(category="Electricity")
        trend_f = tmp_db.monthly_trend_with_delta(category="Electricity")

        assert kpi_f["invoice_count"] < kpi_all["invoice_count"]
        assert cat_f["category_count"] == 1  # Only "Electricity" category
        assert all(v["vendor_name"] in {"TANGEDCO"} for v in vend_f)
        if tax_all["total_tax"] and tax_f["total_tax"]:
            assert tax_f["total_tax"] <= tax_all["total_tax"]
        elec_spend = sum(r["total_spend"] for r in trend_f)
        all_spend = sum(r["total_spend"] for r in trend_all)
        assert elec_spend <= all_spend

    def test_fiscal_year_filter_isolation(self, tmp_db):
        _seed_db(tmp_db)
        fy25_26 = tmp_db.kpi_summary(fiscal_year="FY 2025-26")
        fy24_25 = tmp_db.kpi_summary(fiscal_year="FY 2024-25")
        all_kpi = tmp_db.kpi_summary()

        assert fy25_26["invoice_count"] + fy24_25["invoice_count"] == all_kpi["invoice_count"]
        assert abs(
            fy25_26["total_spend"] + fy24_25["total_spend"] - all_kpi["total_spend"]
        ) < 0.01
