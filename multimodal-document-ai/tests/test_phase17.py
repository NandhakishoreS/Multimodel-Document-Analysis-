"""
Phase 17 — Final Integration, System Hardening & Project-Wide Acceptance Audit.

Test Suite covering:
1. Pipeline integration — ingestion, extraction, classification, persistence, analytics
2. Data integrity invariants — financial math, AI value separation, audit trail
3. Defensive security — corrupt inputs, path traversal, injection protection, SQL safety
4. Repeatability & idempotency — duplicate uploads, repeated scans, deterministic reports
5. Export validation — PDF structure, Excel sheets, CSV rows, JSON schema
6. Application workflow functions — module imports, core workflow chain
7. Database convenience methods — get_all_documents, update_verification_status, close

Tests are self-contained and use isolated temporary databases.
No models are retrained. No benchmark results are modified.
"""

from __future__ import annotations

import io
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import openpyxl
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from phase11.anomaly_engine import (
    AnomalyType,
    CentralizedAnomalyEngine,
)
from phase11.classifier import BillCategoryClassifier
from phase11.database import FinancialDocumentDB
from phase11.duplicate_detector import MultiSignalDuplicateDetector, DuplicateType
from phase11.export import (
    DocumentExporter,
    export_institutional_csv,
    export_institutional_excel,
    export_institutional_json,
    export_institutional_pdf,
)
from phase11.extractor import FinancialDocumentExtractor
from phase11.ingestion import (
    DocumentIngestionService,
    DuplicatePolicy,
    IngestionConfig,
    IngestionItem,
    ProcessingStatus,
)
from phase11.reporting import (
    APP_VERSION,
    InstitutionalReportEngine,
    ReportType,
    REPORT_TITLES,
    format_inr,
    format_currency,
)
from phase11.schema import FinancialDocument, LineItem


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path: Path) -> FinancialDocumentDB:
    """Fresh isolated database per test."""
    return FinancialDocumentDB(project_root=tmp_path)


def _make_doc(
    vendor: str = "TANGEDCO",
    inv_num: str = "INV-2026-001",
    date: str = "2026-03-15",
    total: float = 45000.0,
    subtotal: float = 38135.59,
    tax: float = 6864.41,
    category: str = "Electricity",
    status: str = "pending",
    currency: str = "INR",
    file_hash: Optional[str] = None,
) -> FinancialDocument:
    return FinancialDocument(
        document_id=str(uuid.uuid4()),
        filename=f"invoice_{uuid.uuid4().hex[:6]}.pdf",
        document_type="invoice",
        benchmark_domain="SROIE",
        category=category,
        vendor_name=vendor,
        invoice_number=inv_num,
        invoice_date=date,
        total_amount=total,
        subtotal=subtotal,
        tax_amount=tax,
        currency=currency,
        verification_status=status,
        file_hash=file_hash or uuid.uuid4().hex,
        processing_status="COMPLETED",
    )


def _create_png_bytes(size: tuple = (100, 100)) -> bytes:
    """Create minimal valid PNG image bytes."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color=(240, 240, 240)).save(buf, format="PNG")
    return buf.getvalue()


def _create_pdf_bytes(text: str = "Test Invoice") -> bytes:
    """Create minimal valid PDF bytes."""
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, text)
    return bytes(pdf.output())


# ===========================================================================
# 1. PIPELINE INTEGRATION TESTS
# ===========================================================================

class TestPipelineIntegration:
    """Verify the complete Phase 11-16 ingestion and processing pipeline."""

    def test_ingestion_item_from_bytes(self):
        """IngestionItem.from_bytes produces correct metadata."""
        data = _create_png_bytes()
        item = IngestionItem.from_bytes("test_bill.png", data)
        assert item.filename == "test_bill.png"
        assert item.file_extension == ".png"
        assert item.file_size == len(data)
        assert len(item.file_hash) == 64  # SHA-256

    def test_ingestion_item_filename_sanitization(self):
        """Filenames with path traversal or null bytes are sanitized."""
        data = _create_png_bytes()
        # Path traversal attempt
        item = IngestionItem.from_bytes("../../etc/passwd.png", data)
        assert "/" not in item.filename
        assert "\\" not in item.filename
        assert item.filename == "passwd.png"

        # Null bytes
        item2 = IngestionItem.from_bytes("bill\x00.png", data)
        assert "\x00" not in item2.filename

    def test_ingestion_empty_file_rejected(self):
        """Empty files must be rejected with REJECTED status."""
        item = IngestionItem.from_bytes("empty.png", b"")
        service = DocumentIngestionService()
        valid = service.validate_item(item)
        assert not valid
        assert item.status == ProcessingStatus.REJECTED
        assert "empty" in item.error_message.lower()

    def test_ingestion_unsupported_extension_rejected(self):
        """Unsupported file types must be rejected."""
        item = IngestionItem.from_bytes("malware.exe", b"MZ" + b"\x00" * 100)
        service = DocumentIngestionService()
        valid = service.validate_item(item)
        assert not valid
        assert item.status == ProcessingStatus.REJECTED

    def test_ingestion_corrupt_image_rejected(self):
        """Corrupt image data fails with FAILED/REJECTED status."""
        item = IngestionItem.from_bytes("corrupt.png", b"\x89PNG\r\n" + b"\xFF" * 100)
        service = DocumentIngestionService()
        valid = service.validate_item(item)
        assert not valid

    def test_document_persistence_and_retrieval(self, tmp_db: FinancialDocumentDB):
        """Documents persist and round-trip correctly."""
        doc = _make_doc(vendor="TANGEDCO", total=45000.0)
        tmp_db.insert_document(doc)
        retrieved = tmp_db.get_document(doc.document_id)
        assert retrieved is not None
        assert retrieved.vendor_name == doc.vendor_name
        assert retrieved.total_amount == doc.total_amount
        assert retrieved.document_id == doc.document_id

    def test_duplicate_file_hash_detection(self, tmp_db: FinancialDocumentDB):
        """Documents with identical SHA-256 are detected as file duplicates."""
        shared_hash = uuid.uuid4().hex
        doc1 = _make_doc(vendor="Vendor A", file_hash=shared_hash)
        tmp_db.insert_document(doc1)
        existing = tmp_db.get_document_by_hash(shared_hash)
        assert existing is not None
        assert existing.document_id == doc1.document_id

    def test_search_by_vendor(self, tmp_db: FinancialDocumentDB):
        """Vendor-based search returns only matching documents."""
        doc1 = _make_doc(vendor="TANGEDCO Electricity")
        doc2 = _make_doc(vendor="Airtel Broadband")
        tmp_db.insert_document(doc1)
        tmp_db.insert_document(doc2)
        results = tmp_db.search_documents(vendor="TANGEDCO")
        assert len(results) == 1
        assert results[0].vendor_name == "TANGEDCO Electricity"

    def test_search_by_category(self, tmp_db: FinancialDocumentDB):
        """Category filter returns only matching documents."""
        doc1 = _make_doc(category="Electricity")
        doc2 = _make_doc(category="Water")
        tmp_db.insert_document(doc1)
        tmp_db.insert_document(doc2)
        results = tmp_db.search_documents(category="Electricity")
        assert len(results) == 1

    def test_search_min_amount_filter(self, tmp_db: FinancialDocumentDB):
        """Amount range filter correctly excludes below-threshold documents."""
        doc_low = _make_doc(vendor="Small Vendor", total=500.0)
        doc_high = _make_doc(vendor="Large Vendor", total=250000.0)
        tmp_db.insert_document(doc_low)
        tmp_db.insert_document(doc_high)
        results = tmp_db.search_documents(min_amount=100000.0)
        assert all(d.total_amount >= 100000.0 for d in results)
        assert any(d.vendor_name == "Large Vendor" for d in results)


# ===========================================================================
# 2. DATA INTEGRITY INVARIANTS
# ===========================================================================

class TestDataIntegrityInvariants:
    """Verify financial math, AI preservation, and audit invariants."""

    def test_financial_non_negative_amounts(self, tmp_db: FinancialDocumentDB):
        """total_amount, tax_amount, subtotal must not be negative."""
        doc = _make_doc(total=45000.0, subtotal=38135.59, tax=6864.41)
        tmp_db.insert_document(doc)
        retrieved = tmp_db.get_document(doc.document_id)
        assert retrieved.total_amount >= 0
        assert retrieved.subtotal >= 0
        assert retrieved.tax_amount >= 0

    def test_tax_math_consistency(self):
        """subtotal + tax approximately equals total within 1.00 tolerance."""
        doc = _make_doc(total=45000.0, subtotal=38135.59, tax=6864.41)
        if doc.subtotal is not None and doc.tax_amount is not None and doc.total_amount is not None:
            discrepancy = abs(doc.subtotal + doc.tax_amount - doc.total_amount)
            assert discrepancy < 1.0, f"Tax math discrepancy: {discrepancy}"

    def test_ai_vendor_preserved_after_edit(self, tmp_db: FinancialDocumentDB):
        """AI-extracted vendor_name is preserved when human edits the field."""
        doc = _make_doc(vendor="AI Detected Vendor")
        doc.ai_vendor_name = "AI Detected Vendor"
        tmp_db.insert_document(doc)
        tmp_db.edit_financial_document(
            document_id=doc.document_id,
            updates={"vendor_name": "Human Corrected Vendor"},
            modified_by="test_auditor",
        )
        reloaded = tmp_db.get_document(doc.document_id)
        assert reloaded.vendor_name == "Human Corrected Vendor"
        assert reloaded.ai_vendor_name == "AI Detected Vendor"

    def test_ai_fields_cannot_be_directly_modified(self, tmp_db: FinancialDocumentDB):
        """edit_financial_document must reject mutation of AI-preserved fields."""
        doc = _make_doc()
        tmp_db.insert_document(doc)
        with pytest.raises(ValueError, match="immutable"):
            tmp_db.edit_financial_document(
                document_id=doc.document_id,
                updates={"ai_vendor_name": "Tampered Value"},
                modified_by="bad_actor",
            )

    def test_document_ids_unique(self, tmp_db: FinancialDocumentDB):
        """Each inserted document must maintain a unique document_id."""
        docs = [_make_doc() for _ in range(10)]
        for doc in docs:
            tmp_db.insert_document(doc)
        ids = {doc.document_id for doc in docs}
        assert len(ids) == 10

    def test_audit_log_created_on_insert(self, tmp_db: FinancialDocumentDB):
        """Inserting a document must create an audit processing_log entry."""
        doc = _make_doc()
        tmp_db.insert_document(doc)
        history = tmp_db.get_audit_history(doc.document_id)
        assert len(history) >= 1
        actions = [h["action"] for h in history]
        assert "insert" in actions

    def test_audit_log_on_edit(self, tmp_db: FinancialDocumentDB):
        """Editing a document must create an audit entry."""
        doc = _make_doc()
        tmp_db.insert_document(doc)
        tmp_db.edit_financial_document(
            document_id=doc.document_id,
            updates={"vendor_name": "New Vendor"},
            modified_by="auditor",
        )
        history = tmp_db.get_audit_history(doc.document_id)
        actions = [h["action"] for h in history]
        assert any("FIELD_UPDATED" in a or "RECORD_EDITED" in a or "edit" in a.lower() for a in actions)

    def test_is_modified_flag_set_on_edit(self, tmp_db: FinancialDocumentDB):
        """is_modified flag must be set to True after editing."""
        doc = _make_doc()
        tmp_db.insert_document(doc)
        tmp_db.edit_financial_document(
            document_id=doc.document_id,
            updates={"notes": "Manual review completed."},
            modified_by="reviewer",
        )
        updated = tmp_db.get_document(doc.document_id)
        assert updated.is_modified == 1


# ===========================================================================
# 3. DEFENSIVE SECURITY HARDENING TESTS
# ===========================================================================

class TestSecurityHardening:
    """Verify defensive input validation and injection protections."""

    def test_sql_injection_in_search_query(self, tmp_db: FinancialDocumentDB):
        """SQL injection patterns in search query must not raise or corrupt DB."""
        doc = _make_doc(vendor="Legitimate Vendor")
        tmp_db.insert_document(doc)
        injection_queries = [
            "' OR '1'='1",
            "'; DROP TABLE documents; --",
            "1; SELECT * FROM sqlite_master; --",
            "'; INSERT INTO documents VALUES (1,2,3); --",
        ]
        for q in injection_queries:
            results = tmp_db.search_documents(query=q)
            assert isinstance(results, list)

    def test_sql_injection_in_vendor_filter(self, tmp_db: FinancialDocumentDB):
        """SQL injection in vendor filter parameter must not cause errors."""
        doc = _make_doc(vendor="Real Vendor")
        tmp_db.insert_document(doc)
        results = tmp_db.search_documents(vendor="'; DROP TABLE documents;--")
        assert isinstance(results, list)

    def test_path_traversal_rejected_by_ingestion(self):
        """Path traversal in filename is sanitized to basename."""
        data = _create_png_bytes()
        item = IngestionItem.from_bytes("../../../etc/cron.d/evil.png", data)
        assert ".." not in item.filename
        assert item.filename == "evil.png"

    def test_null_byte_filename_sanitized(self):
        """Null bytes in filename are stripped during ingestion."""
        data = _create_png_bytes()
        item = IngestionItem.from_bytes("invoice\x00.png", data)
        assert "\x00" not in item.filename

    def test_unicode_filename_handled(self):
        """Unicode filenames are accepted cleanly."""
        data = _create_png_bytes()
        item = IngestionItem.from_bytes("invoice_\u0b95\u0bbe\u0b95\u0bbf\u0ba4\u0bae\u0bcd.png", data)
        assert item.file_extension == ".png"
        assert item.file_size > 0

    def test_excel_formula_injection_protection(self, tmp_db: FinancialDocumentDB):
        """Excel exports must strip formula prefixes from cell values."""
        doc = _make_doc(vendor='=HYPERLINK("http://evil.com","Click Me")')
        tmp_db.insert_document(doc)
        rep = InstitutionalReportEngine(db=tmp_db).build_report(ReportType.EXECUTIVE_SUMMARY)
        excel_bytes = export_institutional_excel(rep)
        wb = openpyxl.load_workbook(io.BytesIO(excel_bytes), data_only=True)
        for sheet in wb.worksheets:
            for row in sheet.iter_rows():
                for cell in row:
                    val = str(cell.value or "")
                    assert not val.startswith("="), f"Formula injection in cell: {val}"

    def test_negative_total_edit_rejected(self, tmp_db: FinancialDocumentDB):
        """Negative total_amount must be rejected by edit_financial_document."""
        doc = _make_doc()
        tmp_db.insert_document(doc)
        with pytest.raises(ValueError):
            tmp_db.edit_financial_document(
                document_id=doc.document_id,
                updates={"total_amount": -1000.0},
                modified_by="test",
            )

    def test_invalid_file_type_rejected(self):
        """Binary files with invalid extension must be rejected."""
        item = IngestionItem.from_bytes("payload.zip", b"PK" + b"\x03\x04" + b"\x00" * 100)
        service = DocumentIngestionService()
        valid = service.validate_item(item)
        assert not valid
        assert item.status == ProcessingStatus.REJECTED


# ===========================================================================
# 4. REPEATABILITY & IDEMPOTENCY TESTS
# ===========================================================================

class TestRepeatabilityAndIdempotency:
    """Verify repeated operations do not corrupt data or produce duplicates."""

    def test_duplicate_upload_warn_policy(self, tmp_db: FinancialDocumentDB):
        """Same file hash uploaded twice under WARN policy stores two records."""
        shared_hash = uuid.uuid4().hex
        doc1 = _make_doc(vendor="Repeated Vendor", file_hash=shared_hash)
        doc2 = _make_doc(vendor="Repeated Vendor", file_hash=shared_hash)
        tmp_db.insert_document(doc1)
        tmp_db.insert_document(doc2)
        all_docs = tmp_db.get_all_documents()
        assert len(all_docs) == 2

    def test_repeated_anomaly_scan_idempotent(self, tmp_db: FinancialDocumentDB):
        """Running anomaly scan twice does not inflate anomaly finding count."""
        doc = _make_doc(total=2500000.0, subtotal=1000000.0, tax=50000.0)
        tmp_db.insert_document(doc)
        engine = CentralizedAnomalyEngine()
        report1 = engine.evaluate_document(doc)
        if report1.findings:
            tmp_db.save_anomaly_findings([f.to_dict() for f in report1.findings])
        count1 = len(tmp_db.get_anomaly_findings())
        report2 = engine.evaluate_document(doc)
        if report2.findings:
            tmp_db.save_anomaly_findings([f.to_dict() for f in report2.findings])
        count2 = len(tmp_db.get_anomaly_findings())
        assert count2 == count1

    def test_repeated_report_generation_deterministic(self, tmp_db: FinancialDocumentDB):
        """Generating the same report twice produces identical narratives."""
        for _ in range(5):
            doc = _make_doc(vendor="Consistent Vendor")
            tmp_db.insert_document(doc)
        eng = InstitutionalReportEngine(db=tmp_db)
        rep1 = eng.build_report(ReportType.EXECUTIVE_SUMMARY)
        rep2 = eng.build_report(ReportType.EXECUTIVE_SUMMARY)
        assert rep1["kpis"]["invoice_count"] == rep2["kpis"]["invoice_count"]
        assert rep1["kpis"]["total_spend"] == rep2["kpis"]["total_spend"]
        assert rep1["narratives"] == rep2["narratives"]

    def test_data_persists_across_db_reconnect(self, tmp_path: Path):
        """Data written to DB persists after closing and reopening connection."""
        db_path = tmp_path / "persist_test.db"
        db1 = FinancialDocumentDB(db_path=db_path)
        doc = _make_doc(vendor="Persistent Vendor")
        db1.insert_document(doc)
        db1.close()
        db2 = FinancialDocumentDB(db_path=db_path)
        retrieved = db2.get_document(doc.document_id)
        assert retrieved is not None
        assert retrieved.vendor_name == "Persistent Vendor"


# ===========================================================================
# 5. EXPORT VALIDATION TESTS
# ===========================================================================

class TestExportValidation:
    """Verify all export formats (PDF, Excel, CSV, JSON) produce valid artifacts."""

    @pytest.fixture
    def report_with_docs(self, tmp_db: FinancialDocumentDB):
        """Create a report with 5 realistic documents."""
        vendors = [
            ("TANGEDCO", "Electricity", 45000.0, 38135.59, 6864.41),
            ("BWSSB Water", "Water", 12500.0, 10593.22, 1906.78),
            ("Dell India", "Equipment", 185000.0, 156779.66, 28220.34),
            ("Starbucks", "Food/Catering", 850.0, 720.34, 129.66),
            ("KPMG", "Professional Services", 220000.0, 186440.68, 33559.32),
        ]
        for v, cat, tot, sub, tax in vendors:
            doc = _make_doc(vendor=v, category=cat, total=tot, subtotal=sub, tax=tax)
            tmp_db.insert_document(doc)
        return InstitutionalReportEngine(db=tmp_db).build_report(ReportType.EXECUTIVE_SUMMARY)

    def test_pdf_export_valid_header(self, report_with_docs):
        """PDF export must start with valid %PDF- header."""
        pdf_bytes = export_institutional_pdf(report_with_docs)
        assert pdf_bytes.startswith(b"%PDF-")
        assert len(pdf_bytes) > 1000

    def test_excel_export_11_sheets(self, report_with_docs):
        """Excel export must produce an 11-sheet workbook."""
        excel_bytes = export_institutional_excel(report_with_docs)
        wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
        assert len(wb.sheetnames) >= 11

    def test_excel_sheet_names_present(self, report_with_docs):
        """Excel workbook must contain expected sheet names."""
        excel_bytes = export_institutional_excel(report_with_docs)
        wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
        expected = {"Executive Summary", "Financial Records", "Category Analysis",
                    "Vendor Intelligence", "Monthly Trend", "Anomaly Findings",
                    "Duplicate Candidates", "Tax Analysis", "Report Metadata"}
        for sheet in expected:
            assert sheet in wb.sheetnames, f"Sheet '{sheet}' missing from workbook"

    def test_csv_export_valid_utf8(self, report_with_docs):
        """CSV export must be valid UTF-8 with column headers."""
        csv_str = export_institutional_csv(report_with_docs)
        assert isinstance(csv_str, str)
        csv_str.encode("utf-8")
        assert len(csv_str.strip()) > 0

    def test_json_export_valid_structure(self, report_with_docs):
        """JSON export must be valid JSON with expected top-level keys."""
        json_str = export_institutional_json(report_with_docs)
        parsed = json.loads(json_str)
        assert "metadata" in parsed
        assert "kpis" in parsed
        assert "narratives" in parsed
        assert "documents" in parsed

    def test_json_export_document_count(self, tmp_db: FinancialDocumentDB):
        """JSON export must contain the expected number of document records."""
        for i in range(3):
            tmp_db.insert_document(_make_doc(vendor=f"Vendor {i}"))
        rep = InstitutionalReportEngine(db=tmp_db).build_report(ReportType.EXECUTIVE_SUMMARY)
        json_str = export_institutional_json(rep)
        parsed = json.loads(json_str)
        assert len(parsed["documents"]) == 3

    def test_all_report_archetypes_generate(self, tmp_db: FinancialDocumentDB):
        """All 7 report archetypes generate without errors."""
        for i in range(5):
            tmp_db.insert_document(_make_doc(vendor=f"Vendor {i}"))
        eng = InstitutionalReportEngine(db=tmp_db)
        for rtype in ReportType:
            rep = eng.build_report(rtype)
            assert rep["metadata"]["report_type"] == rtype.value
            assert len(rep["narratives"]) >= 0


# ===========================================================================
# 6. DATABASE CONVENIENCE METHODS
# ===========================================================================

class TestDatabaseConvenienceMethods:
    """Verify Phase 17 database utility methods."""

    def test_get_all_documents(self, tmp_db: FinancialDocumentDB):
        """get_all_documents returns all persisted documents."""
        for i in range(5):
            tmp_db.insert_document(_make_doc(vendor=f"Vendor {i}"))
        all_docs = tmp_db.get_all_documents()
        assert len(all_docs) == 5

    def test_get_document_count(self, tmp_db: FinancialDocumentDB):
        """get_document_count returns accurate count."""
        for _ in range(7):
            tmp_db.insert_document(_make_doc())
        assert tmp_db.get_document_count() == 7

    def test_update_verification_status(self, tmp_db: FinancialDocumentDB):
        """update_verification_status atomically changes doc status."""
        doc = _make_doc(status="pending")
        tmp_db.insert_document(doc)
        result = tmp_db.update_verification_status(doc.document_id, "verified")
        assert result is True
        updated = tmp_db.get_document(doc.document_id)
        assert updated.verification_status == "verified"

    def test_close_is_safe(self, tmp_db: FinancialDocumentDB):
        """close() must be safe to call without raising."""
        tmp_db.close()


# ===========================================================================
# 7. ANOMALY DETECTION INVARIANTS
# ===========================================================================

class TestAnomalyDetectionInvariants:
    """Verify anomaly engine does not mutate source documents."""

    def test_anomaly_engine_does_not_mutate_document(self):
        """Anomaly scan must not modify the document under evaluation."""
        doc = _make_doc(total=2500000.0, subtotal=1000000.0, tax=50000.0)
        original_total = doc.total_amount
        original_vendor = doc.vendor_name
        engine = CentralizedAnomalyEngine()
        engine.evaluate_document(doc)
        assert doc.total_amount == original_total
        assert doc.vendor_name == original_vendor

    def test_tax_mismatch_detected(self):
        """ANOMALY_006 fires when subtotal + tax does not equal total."""
        doc = _make_doc(total=2500000.0, subtotal=1000000.0, tax=50000.0)
        engine = CentralizedAnomalyEngine()
        report = engine.evaluate_document(doc)
        anomaly_types = [f.anomaly_type for f in report.findings]
        assert AnomalyType.TAX_INCONSISTENCY in anomaly_types

    def test_anomaly_review_status_transition(self, tmp_db: FinancialDocumentDB):
        """Anomaly status transitions from OPEN to RESOLVED are recorded."""
        doc = _make_doc()
        tmp_db.insert_document(doc)
        finding_dict = {
            "document_id": doc.document_id,
            "anomaly_type": AnomalyType.ROUND_AMOUNT,
            "severity": "LOW",
            "score": 0.3,
            "confidence": 0.7,
            "explanation": "Round amount of 45000.0",
            "status": "OPEN",
            "detected_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp_db.save_anomaly_findings([finding_dict])
        findings = tmp_db.get_anomaly_findings()
        assert len(findings) >= 1
        finding_id = findings[0]["id"]
        tmp_db.update_anomaly_status(
            finding_id=finding_id,
            new_status="RESOLVED",
            reviewer="phase17_test",
            comment="Verified as benign round amount.",
        )
        updated = tmp_db.get_anomaly_findings()
        resolved = [f for f in updated if f["id"] == finding_id]
        assert resolved[0]["status"] == "RESOLVED"


# ===========================================================================
# 8. DUPLICATE DETECTION INVARIANTS
# ===========================================================================

class TestDuplicateDetectionInvariants:
    """Verify duplicate detection engine correctness."""

    def test_strong_duplicate_candidate_detected(self):
        """Identical vendor + invoice# + date + amount triggers STRONG_DUPLICATE_CANDIDATE."""
        doc1 = _make_doc(vendor="TANGEDCO", inv_num="INV-001", date="2026-03-15", total=45000.0)
        doc2 = _make_doc(vendor="TANGEDCO", inv_num="INV-001", date="2026-03-15", total=45000.0)
        from phase11.duplicate_detector import evaluate_pair
        match = evaluate_pair(doc1, doc2)
        assert match is not None
        assert match.duplicate_type == DuplicateType.STRONG_DUPLICATE_CANDIDATE

    def test_no_duplicate_for_different_vendors(self):
        """Documents from clearly different vendors are not flagged as duplicates."""
        doc1 = _make_doc(vendor="TANGEDCO", inv_num="INV-001", total=45000.0)
        doc2 = _make_doc(vendor="Dell India", inv_num="INV-999", total=185000.0)
        from phase11.duplicate_detector import evaluate_pair
        match = evaluate_pair(doc1, doc2)
        assert match is None

    def test_scan_all_pairs_finds_pair(self):
        """scan_all_pairs finds at least one duplicate in a collection with known pair."""
        doc1 = _make_doc(vendor="TANGEDCO", inv_num="INV-001", date="2026-03-15", total=45000.0)
        doc2 = _make_doc(vendor="TANGEDCO", inv_num="INV-001", date="2026-03-15", total=45000.0)
        doc3 = _make_doc(vendor="Airtel", inv_num="INV-X99", date="2026-03-10", total=1200.0)
        results = MultiSignalDuplicateDetector.scan_all_pairs([doc1, doc2, doc3])
        assert len(results) >= 1
        assert results[0].duplicate_type == DuplicateType.STRONG_DUPLICATE_CANDIDATE


# ===========================================================================
# 9. REPORTING ENGINE & FORMAT FUNCTIONS
# ===========================================================================

class TestReportingEngine:
    """Verify reporting module functions and formatting utilities."""

    def test_format_inr_zero(self):
        assert format_inr(0.0) == "\u20b90.00"

    def test_format_inr_lakh(self):
        assert format_inr(100000.0) == "\u20b91,00,000.00"

    def test_format_inr_crore(self):
        assert format_inr(10000000.0) == "\u20b91,00,00,000.00"

    def test_format_inr_none(self):
        assert format_inr(None) == "N/A"

    def test_format_inr_negative(self):
        assert "-" in format_inr(-50000.0)

    def test_format_currency_usd(self):
        result = format_currency(1500.0, currency="USD", for_pdf=False)
        assert "$" in result

    def test_format_currency_pdf_mode(self):
        result = format_currency(1500.0, currency="INR", for_pdf=True)
        assert "INR" in result
        assert "\u20b9" not in result

    def test_report_titles_all_types_present(self):
        """REPORT_TITLES dict must cover all ReportType enum values."""
        for rtype in ReportType:
            assert rtype in REPORT_TITLES, f"REPORT_TITLES missing key: {rtype}"

    def test_app_version_is_string(self):
        assert isinstance(APP_VERSION, str)
        assert len(APP_VERSION) > 0


# ===========================================================================
# 10. MODULE IMPORTS & SYSTEM HEALTH
# ===========================================================================

class TestModuleImports:
    """Verify all Phase 11-17 modules import cleanly."""

    def test_phase11_modules_import(self):
        """All phase11 modules must be importable."""
        import phase11.database
        import phase11.schema
        import phase11.classifier
        import phase11.extractor
        import phase11.ingestion
        import phase11.anomaly_engine
        import phase11.duplicate_detector
        import phase11.reporting
        import phase11.export
        assert True

    def test_inference_modules_import(self):
        """Inference and visualization modules import cleanly."""
        import inference.schemas
        import inference.postprocessing
        import visualization.draw_predictions
        assert True

    def test_financial_document_schema_valid(self):
        """FinancialDocument schema can be instantiated with minimal fields."""
        doc = FinancialDocument(
            document_id="test-doc-001",
            filename="test.pdf",
        )
        assert doc.document_id == "test-doc-001"
        assert doc.currency == "INR"
        assert doc.verification_status == "pending"

    def test_final_pipeline_script_exists(self):
        """verify_final_pipeline.py script exists in scripts/ directory."""
        pipeline_script = _PROJECT_ROOT / "scripts" / "verify_final_pipeline.py"
        assert pipeline_script.exists(), "verify_final_pipeline.py not found"


# ===========================================================================
# 11. PERFORMANCE BENCHMARKS (Soft Assertions)
# ===========================================================================

class TestPerformanceBenchmarks:
    """Soft performance checks — warn but do not fail on minor overruns."""

    def test_kpi_query_under_threshold(self, tmp_db: FinancialDocumentDB):
        """KPI query completes within acceptable time for 200 records."""
        for i in range(200):
            doc = _make_doc(vendor=f"Vendor {i % 10}", total=float(10000 + i * 100))
            tmp_db.insert_document(doc)
        t0 = time.perf_counter()
        kpis = tmp_db.kpi_summary()
        elapsed = time.perf_counter() - t0
        assert kpis["invoice_count"] == 200
        assert elapsed < 5.0, f"KPI query took {elapsed:.2f}s"

    def test_search_query_under_threshold(self, tmp_db: FinancialDocumentDB):
        """Multi-filter search completes within acceptable time for 200 records."""
        for i in range(200):
            doc = _make_doc(
                vendor=f"Vendor {i % 5}",
                category="Electricity" if i % 2 == 0 else "Water"
            )
            tmp_db.insert_document(doc)
        t0 = time.perf_counter()
        results = tmp_db.search_documents(category="Electricity", sort_by="created_at", limit=50)
        elapsed = time.perf_counter() - t0
        assert len(results) > 0
        assert elapsed < 5.0, f"Search took {elapsed:.2f}s"

    def test_excel_export_under_threshold(self, tmp_db: FinancialDocumentDB):
        """Excel export for 100 records completes within acceptable time."""
        for i in range(100):
            doc = _make_doc(vendor=f"Vendor {i % 10}")
            tmp_db.insert_document(doc)
        eng = InstitutionalReportEngine(db=tmp_db)
        rep = eng.build_report(ReportType.EXECUTIVE_SUMMARY)
        t0 = time.perf_counter()
        excel_bytes = export_institutional_excel(rep)
        elapsed = time.perf_counter() - t0
        assert len(excel_bytes) > 1000
        assert elapsed < 10.0, f"Excel export took {elapsed:.2f}s"
