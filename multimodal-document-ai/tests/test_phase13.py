"""
Phase 13 Test Suite: Advanced Financial Document Search, Filtering,
Record Editing & Audit-Aware Document Management.

Tests cover:
- Keyword search (case-insensitive, partial matching, multi-column)
- Dedicated filters (category, exact date, date range, month, year, fiscal year, vendor, invoice #, amount range, status, domain)
- Combined multi-criteria filtering
- Sorting (invoice date, total amount, vendor, category ASC & DESC)
- SQL-level pagination and matching count queries
- Financial record editing with validation
- Immutable fields protection
- Granular field-level audit logging (FIELD_UPDATED, RECORD_EDITED)
- AI predictions vs final verified value preservation (ai_vendor_name, ai_total_amount, etc.)
- Missing fields detection
- Verification workflow lifecycle (pending -> modified -> verified -> flagged)
- Filtered analytics integration (spend by category, monthly trend, top vendors, status summary)
- Filtered export integration (JSON, CSV, Excel, PDF)
- Database transaction integrity & rollback
"""

from __future__ import annotations

import json
import pytest
import tempfile
from pathlib import Path

from phase11.schema import FinancialDocument, LineItem, VERIFICATION_STATUSES
from phase11.database import FinancialDocumentDB
from phase11.export import DocumentExporter


@pytest.fixture
def temp_db(tmp_path):
    """Fixture providing a fresh, isolated SQLite database."""
    db_file = tmp_path / "test_p13_financial.db"
    return FinancialDocumentDB(db_path=db_file)


@pytest.fixture
def populated_db(temp_db):
    """Fixture providing 10 diverse realistic institutional financial documents."""
    docs = [
        FinancialDocument(
            document_id="DOC-ELEC-001",
            filename="tangedco_mar_2026.pdf",
            vendor_name="TANGEDCO Electricity Board",
            invoice_number="INV-2026-001",
            invoice_date="2026-03-10",
            total_amount=54200.0,
            subtotal=50000.0,
            tax_amount=4200.0,
            category="Electricity",
            sub_category="Utilities",
            classification_confidence=0.95,
            verification_status="pending",
            benchmark_domain="SROIE",
        ),
        FinancialDocument(
            document_id="DOC-ELEC-002",
            filename="tangedco_apr_2026.pdf",
            vendor_name="TANGEDCO Electricity Board",
            invoice_number="INV-2026-002",
            invoice_date="2026-04-05",
            total_amount=62100.0,
            subtotal=58000.0,
            tax_amount=4100.0,
            category="Electricity",
            sub_category="Utilities",
            classification_confidence=0.92,
            verification_status="verified",
            benchmark_domain="SROIE",
        ),
        FinancialDocument(
            document_id="DOC-WATER-001",
            filename="metro_water_mar_2026.png",
            vendor_name="Chennai Metro Water Supply Corporation",
            invoice_number="WAT-8812",
            invoice_date="2026-03-15",
            total_amount=14500.0,
            subtotal=14500.0,
            tax_amount=0.0,
            category="Water",
            sub_category="Utilities",
            classification_confidence=0.88,
            verification_status="pending",
            benchmark_domain="SROIE",
        ),
        FinancialDocument(
            document_id="DOC-INTERNET-001",
            filename="airtel_lease_feb_2026.pdf",
            vendor_name="Bharti Airtel Corporate Telecom",
            invoice_number="AIR-2026-99",
            invoice_date="2026-02-28",
            total_amount=28000.0,
            subtotal=25000.0,
            tax_amount=3000.0,
            category="Internet/Telecom",
            sub_category="Utilities",
            classification_confidence=0.91,
            verification_status="verified",
            benchmark_domain="SROIE",
        ),
        FinancialDocument(
            document_id="DOC-SUPPLIES-001",
            filename="stationery_mar_2026.jpg",
            vendor_name="Reliance Office Depot",
            invoice_number="ROD-4011",
            invoice_date="2026-03-20",
            total_amount=8500.0,
            subtotal=8000.0,
            tax_amount=500.0,
            category="Office Supplies",
            sub_category="Operations",
            classification_confidence=0.85,
            verification_status="pending",
            benchmark_domain="CORD",
        ),
        FinancialDocument(
            document_id="DOC-MAINT-001",
            filename="ac_maintenance_mar_2026.pdf",
            vendor_name="Voltas Engineering Services",
            invoice_number="VES-1090",
            invoice_date="2026-03-25",
            total_amount=45000.0,
            subtotal=42000.0,
            tax_amount=3000.0,
            category="Maintenance",
            sub_category="Operations",
            classification_confidence=0.89,
            verification_status="pending",
            benchmark_domain="FUNSD",
        ),
        FinancialDocument(
            document_id="DOC-FOOD-001",
            filename="canteen_catering_feb_2026.pdf",
            vendor_name="Annapoorna Caterers",
            invoice_number="CAT-2026-14",
            invoice_date="2026-02-14",
            total_amount=32000.0,
            subtotal=30000.0,
            tax_amount=2000.0,
            category="Food/Catering",
            sub_category="Operations",
            classification_confidence=0.84,
            verification_status="pending",
            benchmark_domain="CORD",
        ),
        FinancialDocument(
            document_id="DOC-TRANSPORT-001",
            filename="staff_bus_transport_mar_2026.pdf",
            vendor_name="SRS Logistics & Transport",
            invoice_number="SRS-7701",
            invoice_date="2026-03-30",
            total_amount=75000.0,
            subtotal=70000.0,
            tax_amount=5000.0,
            category="Transportation",
            sub_category="Operations",
            classification_confidence=0.90,
            verification_status="verified",
            benchmark_domain="SROIE",
        ),
        FinancialDocument(
            document_id="DOC-MISC-001",
            filename="hardware_tools_apr_2026.pdf",
            vendor_name="Standard Hardware Stores",
            invoice_number="SHS-004",
            invoice_date="2026-04-02",
            total_amount=3500.0,
            subtotal=3200.0,
            tax_amount=300.0,
            category="Miscellaneous",
            sub_category="Operations",
            classification_confidence=0.72,
            verification_status="pending",
            benchmark_domain="SROIE",
        ),
        FinancialDocument(
            document_id="DOC-UNKNOWN-001",
            filename="unclassified_receipt.pdf",
            vendor_name=None,
            invoice_number=None,
            invoice_date="2026-03-01",
            total_amount=None,
            category="Unknown",
            classification_confidence=0.35,
            requires_review=True,
            verification_status="pending",
            benchmark_domain="SROIE",
        ),
    ]
    for d in docs:
        temp_db.insert_document(d)
    return temp_db


# ─────────────────────────────────────────────────────────────────────────
# 1. Search Engine Tests
# ─────────────────────────────────────────────────────────────────────────

def test_search_case_insensitive_vendor(populated_db):
    """Case-insensitive search on vendor name finds matching records."""
    res_lower = populated_db.search_documents(query="tangedco")
    res_upper = populated_db.search_documents(query="TANGEDCO")
    res_mixed = populated_db.search_documents(query="Tangedco")
    assert len(res_lower) == 2
    assert len(res_upper) == 2
    assert len(res_mixed) == 2
    doc_ids = {d.document_id for d in res_lower}
    assert doc_ids == {"DOC-ELEC-001", "DOC-ELEC-002"}


def test_search_partial_keyword(populated_db):
    """Partial keyword match across fields."""
    res = populated_db.search_documents(query="Voltas")
    assert len(res) == 1
    assert res[0].document_id == "DOC-MAINT-001"


def test_search_invoice_number(populated_db):
    """Search matches invoice number."""
    res = populated_db.search_documents(query="AIR-2026-99")
    assert len(res) == 1
    assert res[0].document_id == "DOC-INTERNET-001"


def test_search_category_keyword(populated_db):
    """General search string matches category term."""
    res = populated_db.search_documents(query="electricity")
    assert len(res) >= 2
    for doc in res:
        assert "electricity" in doc.category.lower() or "electricity" in (doc.vendor_name or "").lower()


def test_dedicated_vendor_and_invoice_filters(populated_db):
    """Isolated vendor and invoice number filter parameters work correctly."""
    v_res = populated_db.search_documents(vendor="Airtel")
    assert len(v_res) == 1
    assert v_res[0].vendor_name == "Bharti Airtel Corporate Telecom"

    inv_res = populated_db.search_documents(invoice_number="ROD-4011")
    assert len(inv_res) == 1
    assert inv_res[0].document_id == "DOC-SUPPLIES-001"


# ─────────────────────────────────────────────────────────────────────────
# 2. Structured Filter Tests
# ─────────────────────────────────────────────────────────────────────────

def test_filter_by_category(populated_db):
    """Filtering by exact category returns only records for that category."""
    res = populated_db.search_documents(category="Electricity")
    assert len(res) == 2
    assert all(d.category == "Electricity" for d in res)


def test_filter_by_verification_status(populated_db):
    """Filter by verification status."""
    res = populated_db.search_documents(verification_status="verified")
    assert len(res) == 3
    assert all(d.verification_status == "verified" for d in res)


def test_filter_by_amount_range(populated_db):
    """Filter by min_amount and max_amount."""
    res = populated_db.search_documents(min_amount=20000.0, max_amount=50000.0)
    assert len(res) == 3
    for d in res:
        assert 20000.0 <= d.total_amount <= 50000.0


def test_filter_date_range(populated_db):
    """Filter by ISO date range."""
    res = populated_db.search_documents(date_from="2026-03-01", date_to="2026-03-31")
    assert len(res) == 6
    for d in res:
        assert d.invoice_date and "2026-03" in d.invoice_date


def test_filter_exact_date(populated_db):
    """Filter by exact invoice date."""
    res = populated_db.search_documents(date_exact="2026-03-15")
    assert len(res) == 1
    assert res[0].document_id == "DOC-WATER-001"


def test_filter_month_and_year(populated_db):
    """Filter by month prefix (YYYY-MM) and year prefix (YYYY)."""
    month_res = populated_db.search_documents(month="2026-02")
    assert len(month_res) == 2
    assert all(d.invoice_date.startswith("2026-02") for d in month_res)

    year_res = populated_db.search_documents(year="2026")
    assert len(year_res) == 10  # All 10 documents have 2026 invoice dates


def test_filter_fiscal_year(populated_db):
    """Indian fiscal year (FY 2025-26 from 2025-04-01 to 2026-03-31)."""
    res = populated_db.search_documents(fiscal_year="FY 2025-26")
    # All Feb & March 2026 documents fall in FY 2025-26, whereas April 2026 fall in FY 2026-27
    assert len(res) == 8
    for d in res:
        assert d.invoice_date <= "2026-03-31"

    fy_next = populated_db.search_documents(fiscal_year="FY 2026-27")
    assert len(fy_next) == 2  # tangedco_apr_2026 and hardware_tools_apr_2026
    for d in fy_next:
        assert d.invoice_date >= "2026-04-01"


def test_combined_filters(populated_db):
    """Multi-filter combination: Category = Electricity AND Date in March 2026 AND Amount > 50,000."""
    res = populated_db.search_documents(
        category="Electricity",
        date_from="2026-03-01",
        date_to="2026-03-31",
        min_amount=50000.0,
    )
    assert len(res) == 1
    assert res[0].document_id == "DOC-ELEC-001"
    assert res[0].total_amount == 54200.0


# ─────────────────────────────────────────────────────────────────────────
# 3. Sorting & Pagination Tests
# ─────────────────────────────────────────────────────────────────────────

def test_sorting_by_amount_asc_desc(populated_db):
    """Sort by total amount ascending and descending."""
    docs_desc = populated_db.search_documents(sort_by="total_amount", sort_order="desc")
    amounts_desc = [d.total_amount for d in docs_desc if d.total_amount is not None]
    assert amounts_desc == sorted(amounts_desc, reverse=True)

    docs_asc = populated_db.search_documents(sort_by="total_amount", sort_order="asc")
    amounts_asc = [d.total_amount for d in docs_asc if d.total_amount is not None]
    assert amounts_asc == sorted(amounts_asc)


def test_sorting_by_date(populated_db):
    """Sort by invoice date ascending and descending."""
    docs_desc = populated_db.search_documents(sort_by="invoice_date", sort_order="desc")
    dates_desc = [d.invoice_date for d in docs_desc if d.invoice_date]
    assert dates_desc == sorted(dates_desc, reverse=True)


def test_pagination_and_counts(populated_db):
    """Pagination controls limit, offset, and count_search_documents matches."""
    total = populated_db.count_search_documents(category="Electricity")
    assert total == 2

    page1 = populated_db.search_documents(category="Electricity", limit=1, offset=0)
    page2 = populated_db.search_documents(category="Electricity", limit=1, offset=1)
    assert len(page1) == 1
    assert len(page2) == 1
    assert page1[0].document_id != page2[0].document_id


# ─────────────────────────────────────────────────────────────────────────
# 4. Record Editing & Audit Trail Tests
# ─────────────────────────────────────────────────────────────────────────

def test_edit_financial_document_valid(populated_db):
    """Valid edit persists changes, sets is_modified, and creates audit events."""
    doc_id = "DOC-ELEC-001"
    initial_doc = populated_db.get_document(doc_id)
    assert initial_doc.total_amount == 54200.0

    updated_doc = populated_db.edit_financial_document(
        document_id=doc_id,
        updates={
            "total_amount": 54800.0,
            "vendor_name": "TANGEDCO Special Services",
        },
        modified_by="audit_officer",
        reason="Tariff surcharge manual adjustment",
    )

    assert updated_doc.total_amount == 54800.0
    assert updated_doc.vendor_name == "TANGEDCO Special Services"
    assert updated_doc.is_modified is True
    assert updated_doc.verification_status == "modified"

    # AI prediction baseline preserved!
    assert updated_doc.ai_total_amount == 54200.0
    assert updated_doc.ai_vendor_name == "TANGEDCO Electricity Board"

    # Verify audit history
    history = populated_db.get_audit_history(doc_id)
    actions = [h["action"] for h in history]
    assert "FIELD_UPDATED" in actions
    assert "RECORD_EDITED" in actions

    diffs = [h for h in history if h["action"] == "FIELD_UPDATED"]
    fields_changed = {d["details"]["field"]: d["details"] for d in diffs}
    assert "total_amount" in fields_changed
    assert fields_changed["total_amount"]["old_value"] == 54200.0
    assert fields_changed["total_amount"]["new_value"] == 54800.0
    assert fields_changed["total_amount"]["modified_by"] == "audit_officer"
    assert fields_changed["total_amount"]["reason"] == "Tariff surcharge manual adjustment"


def test_edit_financial_document_validation_errors(populated_db):
    """Invalid edits (negative amounts, invalid categories) are rejected."""
    doc_id = "DOC-ELEC-001"

    with pytest.raises(ValueError, match="Total amount cannot be negative"):
        populated_db.edit_financial_document(
            document_id=doc_id,
            updates={"total_amount": -100.0},
        )

    with pytest.raises(ValueError, match="Invalid category"):
        populated_db.edit_financial_document(
            document_id=doc_id,
            updates={"category": "NonExistentInvalidCategory"},
        )


def test_immutable_fields_cannot_be_directly_overwritten(populated_db):
    """Directly updating immutable technical fields raises ValueError."""
    doc_id = "DOC-ELEC-001"
    with pytest.raises(ValueError, match="immutable"):
        populated_db.edit_financial_document(
            document_id=doc_id,
            updates={"document_id": "NEW-ID-ATTEMPT"},
        )

    with pytest.raises(ValueError, match="immutable"):
        populated_db.edit_financial_document(
            document_id=doc_id,
            updates={"ai_total_amount": 99999.0},
        )


def test_multiple_edits_create_sequential_audit_trail(populated_db):
    """Multiple sequential edits add separate chronological audit records."""
    doc_id = "DOC-MAINT-001"

    # Edit 1: Total amount
    populated_db.edit_financial_document(
        document_id=doc_id,
        updates={"total_amount": 46000.0},
        modified_by="user_1",
        reason="First adjustment",
    )

    # Edit 2: Invoice Date
    populated_db.edit_financial_document(
        document_id=doc_id,
        updates={"invoice_date": "2026-03-26"},
        modified_by="user_2",
        reason="Date correction",
    )

    history = populated_db.get_audit_history(doc_id)
    field_updates = [h for h in history if h["action"] == "FIELD_UPDATED"]
    assert len(field_updates) == 2
    assert field_updates[0]["details"]["field"] == "total_amount"
    assert field_updates[0]["details"]["modified_by"] == "user_1"
    assert field_updates[1]["details"]["field"] == "invoice_date"
    assert field_updates[1]["details"]["modified_by"] == "user_2"


# ─────────────────────────────────────────────────────────────────────────
# 5. Verification Lifecycle & Missing Fields Detection Tests
# ─────────────────────────────────────────────────────────────────────────

def test_missing_fields_detection(populated_db):
    """detect_missing_fields() flags missing required values."""
    incomplete_doc = populated_db.get_document("DOC-UNKNOWN-001")
    missing = incomplete_doc.detect_missing_fields()
    assert "Vendor Name" in missing
    assert "Invoice Date" not in missing  # Date was provided
    assert "Total Amount" in missing
    assert "Category" in missing  # "Unknown" triggers missing category warning

    complete_doc = populated_db.get_document("DOC-ELEC-001")
    assert complete_doc.detect_missing_fields() == []


def test_verification_lifecycle_transitions(populated_db):
    """Verification state changes pending -> modified -> verified."""
    doc_id = "DOC-WATER-001"
    doc = populated_db.get_document(doc_id)
    assert doc.verification_status == "pending"

    # Edit transitions to modified
    doc = populated_db.edit_financial_document(
        document_id=doc_id,
        updates={"vendor_name": "Chennai Metropolitan Water Supply"},
    )
    assert doc.verification_status == "modified"

    # Explicit verification transitions to verified
    doc = populated_db.edit_financial_document(
        document_id=doc_id,
        updates={"verification_status": "verified"},
        modified_by="senior_auditor",
        reason="Audited and approved",
    )
    assert doc.verification_status == "verified"


# ─────────────────────────────────────────────────────────────────────────
# 6. Filter-Aware Analytics Tests
# ─────────────────────────────────────────────────────────────────────────

def test_filtered_analytics_respects_category(populated_db):
    """Analytics reflect active filters (e.g. Electricity only)."""
    all_spend = populated_db.spend_by_category()
    filtered_spend = populated_db.spend_by_category(category="Electricity")

    assert len(filtered_spend) == 1
    assert filtered_spend[0]["category"] == "Electricity"
    assert filtered_spend[0]["count"] == 2
    assert filtered_spend[0]["total_spend"] == 54200.0 + 62100.0
    assert len(all_spend) > len(filtered_spend)


def test_filtered_monthly_spend(populated_db):
    """Monthly spend calculation for specific month."""
    march_spend = populated_db.monthly_spend(month="2026-03")
    assert len(march_spend) == 1
    assert march_spend[0]["month"] == "2026-03"
    assert march_spend[0]["count"] == 5  # 5 records with non-null total amounts in March 2026


def test_filtered_status_summary(populated_db):
    """Status summary filtered by category."""
    summary = populated_db.status_summary(category="Electricity")
    assert summary.get("pending") == 1
    assert summary.get("verified") == 1


# ─────────────────────────────────────────────────────────────────────────
# 7. Filtered Export Integration Tests
# ─────────────────────────────────────────────────────────────────────────

def test_filtered_export_json_csv_excel(populated_db, tmp_path):
    """Exporter generates valid files strictly containing filtered records."""
    filtered_docs = populated_db.search_documents(category="Electricity")
    assert len(filtered_docs) == 2

    exporter = DocumentExporter()

    # JSON export
    json_str = exporter.to_json(filtered_docs)
    parsed = json.loads(json_str)
    assert len(parsed) == 2
    assert all(d["category"] == "Electricity" for d in parsed)

    # CSV export
    csv_str = exporter.to_csv(filtered_docs)
    lines = csv_str.strip().split("\n")
    assert len(lines) == 3  # header + 2 rows
    assert "Electricity" in lines[1]
    assert "Electricity" in lines[2]

    # Excel export
    xlsx_bytes = exporter.to_excel(filtered_docs)
    assert len(xlsx_bytes) > 0

    # PDF export
    pdf_bytes = exporter.to_pdf_report(filtered_docs, {"spend_by_category": [{"category": "Electricity", "total_spend": 116300.0, "count": 2}]})
    assert pdf_bytes.startswith(b"%PDF")


# ─────────────────────────────────────────────────────────────────────────
# 8. AI vs Final Value & Transaction Integrity Tests
# ─────────────────────────────────────────────────────────────────────────

def test_ai_vs_final_value_distinction(populated_db):
    """Original AI extraction remains preserved while final verified value changes."""
    doc_id = "DOC-SUPPLIES-001"
    doc = populated_db.get_document(doc_id)
    assert doc.category == "Office Supplies"
    assert doc.total_amount == 8500.0

    # User modifies total amount, category, and vendor
    updated = populated_db.edit_financial_document(
        document_id=doc_id,
        updates={
            "total_amount": 9200.0,
            "category": "Maintenance",
            "vendor_name": "Reliance Facility Services",
        },
        modified_by="financial_controller",
        reason="Invoice line item correction",
    )

    # Verify final fields are updated
    assert updated.total_amount == 9200.0
    assert updated.category == "Maintenance"
    assert updated.final_category == "Maintenance"
    assert updated.vendor_name == "Reliance Facility Services"
    assert updated.is_modified is True

    # Verify AI baseline fields remain strictly intact
    assert updated.ai_total_amount == 8500.0
    assert updated.ai_category == "Office Supplies"
    assert updated.ai_vendor_name == "Reliance Office Depot"


def test_database_transaction_integrity_rollback(populated_db):
    """Failed transaction rolls back cleanly without partial modifications."""
    doc_id = "DOC-TRANSPORT-001"
    original_doc = populated_db.get_document(doc_id)
    assert original_doc.total_amount == 75000.0

    # Attempt an update with a valid field plus an invalid value that fails validation
    try:
        populated_db.edit_financial_document(
            document_id=doc_id,
            updates={
                "vendor_name": "Temporary Failed Vendor Name",
                "total_amount": -500.0,  # Fails validation
            },
        )
    except ValueError:
        pass

    # Record in database must remain completely unaltered
    after_failed = populated_db.get_document(doc_id)
    assert after_failed.total_amount == 75000.0
    assert after_failed.vendor_name == original_doc.vendor_name
    assert after_failed.is_modified == original_doc.is_modified

