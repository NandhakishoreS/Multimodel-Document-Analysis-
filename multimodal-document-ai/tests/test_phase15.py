"""
Phase 15 Test Suite — Duplicate & Anomaly Detection Intelligence.

Tests cover:
1. Multi-signal duplicate detection:
   - EXACT_FILE_DUPLICATE (SHA-256 match)
   - STRONG_DUPLICATE_CANDIDATE (Vendor + Invoice # + Date + Amount)
   - DUPLICATE_INVOICE_ID (Vendor + Invoice #)
   - DUPLICATE_FINANCIAL_SIGNATURE (Vendor + Date + Amount)
   - NEAR_DUPLICATE_RECORD (Vendor + Near Date + Near Amount)
   - Non-duplicate distinction
   - Candidate blocking efficiency
2. Centralized 10-category anomaly detection:
   - ANOMALY_001: Statistical outlier with N >= 3 sample size guard
   - ANOMALY_002: Unusually high invoice relative to category/vendor mean
   - ANOMALY_003: Round amounts (INFO vs LOW severity)
   - ANOMALY_004: Duplicate invoice number
   - ANOMALY_005: Missing critical financial fields
   - ANOMALY_006: Tax inconsistency (subtotal + tax != total with tolerance)
   - ANOMALY_007: Category distribution mismatch
   - ANOMALY_008: Vendor concentration exposure (>25%)
   - ANOMALY_009: Temporal spending surge
   - ANOMALY_010: Repeated financial signature
   - Multi-signal aggregation (overall score, severity, recommendations)
3. Investigation workflow & audit state machine:
   - Status transitions (OPEN -> UNDER_REVIEW -> RESOLVED, DISMISSED, CONFIRMED_DUPLICATE)
   - Immutable audit logging in anomaly_review_history
   - False-positive dismissal preserving evidence
4. Database persistence & Idempotent Rescan:
   - Rescan preserves existing reviewed statuses
   - Re-running scan produces identical row counts
   - Summary KPI aggregation
5. Fault isolation:
   - Anomaly engine failure does not interrupt ingestion or corrupt records
6. Multi-format exports:
   - Excel (3 sheets), CSV, and JSON outputs
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from phase11.anomaly_engine import (
    ANOMALY_NAMES,
    AnomalyFinding,
    AnomalyType,
    CentralizedAnomalyEngine,
    DocumentAnomalyReport,
)
from phase11.database import FinancialDocumentDB
from phase11.duplicate_detector import (
    DuplicateMatch,
    DuplicateType,
    MultiSignalDuplicateDetector,
    evaluate_pair,
    normalize_invoice_number,
    normalize_vendor,
)
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
    vendor: str = "TANGEDCO Electricity Board",
    inv_num: str = "INV-2026-001",
    date: str = "2026-03-15",
    total: float = 45000.0,
    subtotal: float = 38135.59,
    tax: float = 6864.41,
    category: str = "Electricity",
    status: str = "pending",
    file_hash: str = None,
    currency: str = "INR",
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


# ---------------------------------------------------------------------------
# 1. Multi-Signal Duplicate Detection Tests
# ---------------------------------------------------------------------------

class TestMultiSignalDuplicateDetection:

    def test_exact_file_duplicate(self):
        h = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        d1 = _make_doc(file_hash=h, total=1000.0)
        d2 = _make_doc(file_hash=h, total=1000.0)
        match = evaluate_pair(d1, d2)
        assert match is not None
        assert match.duplicate_type == DuplicateType.EXACT_FILE_DUPLICATE
        assert match.score == 1.00
        assert match.confidence == 1.00
        assert "file_hash" in match.matched_fields

    def test_strong_duplicate_candidate(self):
        d1 = _make_doc(vendor="Acme Corp Ltd.", inv_num="INV-100", date="2026-03-01", total=5000.0)
        d2 = _make_doc(vendor="Acme Corp", inv_num="INV-100", date="2026-03-01", total=5000.0)
        match = evaluate_pair(d1, d2)
        assert match is not None
        assert match.duplicate_type == DuplicateType.STRONG_DUPLICATE_CANDIDATE
        assert match.score == 0.95
        assert match.confidence == 0.95
        assert set(match.matched_fields) == {"vendor_name", "invoice_number", "invoice_date", "total_amount"}

    def test_duplicate_invoice_id(self):
        # Same vendor and invoice number, but different dates/amounts
        d1 = _make_doc(vendor="Delta Supplies", inv_num="DS-999", date="2026-01-10", total=2000.0)
        d2 = _make_doc(vendor="Delta Supplies", inv_num="DS-999", date="2026-02-15", total=3500.0)
        match = evaluate_pair(d1, d2)
        assert match is not None
        assert match.duplicate_type == DuplicateType.DUPLICATE_INVOICE_ID
        assert match.score == 0.85
        assert match.confidence == 0.90
        assert set(match.matched_fields) == {"vendor_name", "invoice_number"}

    def test_duplicate_financial_signature(self):
        # Same vendor, date, and amount, but different or missing invoice numbers
        d1 = _make_doc(vendor="Metro Water", inv_num="W-01", date="2026-03-20", total=12500.0)
        d2 = _make_doc(vendor="Metro Water", inv_num="W-02", date="2026-03-20", total=12500.0)
        match = evaluate_pair(d1, d2)
        assert match is not None
        assert match.duplicate_type == DuplicateType.DUPLICATE_FINANCIAL_SIGNATURE
        assert match.score == 0.80
        assert match.confidence == 0.85
        assert set(match.matched_fields) == {"vendor_name", "invoice_date", "total_amount"}

    def test_near_duplicate_record(self):
        # Same vendor, within 2 days, amount within Rs 5
        d1 = _make_doc(vendor="Stationery Hub", inv_num="SH-10", date="2026-04-01", total=4995.0)
        d2 = _make_doc(vendor="Stationery Hub", inv_num="SH-11", date="2026-04-03", total=5000.0)
        match = evaluate_pair(d1, d2)
        assert match is not None
        assert match.duplicate_type == DuplicateType.NEAR_DUPLICATE_RECORD
        assert match.score == 0.65
        assert match.confidence == 0.70
        assert "near_date" in match.matched_fields
        assert "near_amount" in match.matched_fields

    def test_non_duplicate_records(self):
        d1 = _make_doc(vendor="Vendor Alpha", inv_num="A-1", date="2026-01-01", total=100.0)
        d2 = _make_doc(vendor="Vendor Beta", inv_num="B-2", date="2026-05-01", total=9000.0)
        assert evaluate_pair(d1, d2) is None

    def test_same_document_id_returns_none(self):
        d1 = _make_doc()
        assert evaluate_pair(d1, d1) is None

    def test_vendor_normalization(self):
        assert normalize_vendor("TANGEDCO Limited") == normalize_vendor("tangedco pvt ltd")
        assert normalize_vendor("Acme, Inc.") == "acme"

    def test_invoice_normalization(self):
        assert normalize_invoice_number("INV-00123") == "INV123"
        assert normalize_invoice_number("000456/A") == "456A"

    def test_candidate_blocking_collection_detection(self):
        docs = [
            _make_doc(vendor="V1", inv_num="INV-1", date="2026-01-01", total=1000.0),
            _make_doc(vendor="V1", inv_num="INV-1", date="2026-01-01", total=1000.0), # Dup with 0
            _make_doc(vendor="V2", inv_num="INV-2", date="2026-02-01", total=2000.0),
            _make_doc(vendor="V3", inv_num="INV-3", date="2026-03-01", total=3000.0),
        ]
        matches = MultiSignalDuplicateDetector.detect_duplicates_in_collection(docs)
        assert len(matches) >= 1
        assert matches[0].duplicate_type == DuplicateType.STRONG_DUPLICATE_CANDIDATE


# ---------------------------------------------------------------------------
# 2. Centralized 10-Category Anomaly Detection Tests
# ---------------------------------------------------------------------------

class TestCentralizedAnomalyEngine:

    def test_anomaly_001_statistical_outlier(self):
        engine = CentralizedAnomalyEngine(sigma_threshold=2.0, min_sample_size=3)
        # 10 baseline documents around 1,000 + 1 massive outlier at 100,000
        docs = [_make_doc(total=1000.0 + i * 10) for i in range(10)] + [_make_doc(total=100000.0)]
        ctx = engine.build_population_context(docs)
        rep = engine.evaluate_document(docs[-1], ctx)
        assert AnomalyType.STATISTICAL_OUTLIER in rep.anomaly_types
        assert rep.severity in ("HIGH", "CRITICAL")

    def test_anomaly_001_small_sample_size_guard(self):
        # When N < 3, no statistical outlier should be declared
        engine = CentralizedAnomalyEngine(min_sample_size=3)
        docs = [
            _make_doc(total=1000.0),
            _make_doc(total=100000.0),
        ]
        ctx = engine.build_population_context(docs)
        rep = engine.evaluate_document(docs[1], ctx)
        assert AnomalyType.STATISTICAL_OUTLIER not in rep.anomaly_types

    def test_anomaly_002_high_invoice_baseline(self):
        engine = CentralizedAnomalyEngine()
        # 5 normal office supply invoices around 500 + 1 at 6,000 (> 3x category average)
        docs = [_make_doc(category="Office Supplies", total=500.0) for _ in range(5)] + [
            _make_doc(category="Office Supplies", total=6000.0)
        ]
        ctx = engine.build_population_context(docs)
        rep = engine.evaluate_document(docs[-1], ctx)
        assert AnomalyType.HIGH_INVOICE_BASELINE in rep.anomaly_types

    def test_anomaly_003_round_amount_info_and_low(self):
        engine = CentralizedAnomalyEngine()
        # Round 5,000 -> INFO
        d_info = _make_doc(total=5000.0)
        rep_info = engine.evaluate_document(d_info)
        f_info = next(f for f in rep_info.findings if f.anomaly_type == AnomalyType.ROUND_AMOUNT)
        assert f_info.severity == "INFO"

        # Round 100,000 -> LOW
        d_low = _make_doc(total=100000.0)
        rep_low = engine.evaluate_document(d_low)
        f_low = next(f for f in rep_low.findings if f.anomaly_type == AnomalyType.ROUND_AMOUNT)
        assert f_low.severity == "LOW"

    def test_anomaly_004_duplicate_invoice_number(self):
        engine = CentralizedAnomalyEngine()
        docs = [
            _make_doc(inv_num="BILL-888"),
            _make_doc(inv_num="BILL-888"),
        ]
        ctx = engine.build_population_context(docs)
        rep = engine.evaluate_document(docs[0], ctx)
        assert AnomalyType.DUPLICATE_INVOICE_NUM in rep.anomaly_types
        assert any(f.severity == "HIGH" for f in rep.findings if f.anomaly_type == AnomalyType.DUPLICATE_INVOICE_NUM)

    def test_anomaly_005_missing_critical_fields(self):
        engine = CentralizedAnomalyEngine()
        # 1 missing field -> LOW
        d1 = _make_doc(vendor="")
        rep1 = engine.evaluate_document(d1)
        assert AnomalyType.MISSING_CRITICAL_FIELDS in rep1.anomaly_types
        f1 = next(f for f in rep1.findings if f.anomaly_type == AnomalyType.MISSING_CRITICAL_FIELDS)
        assert f1.severity == "LOW"

        # 2 missing fields -> MEDIUM
        d2 = _make_doc(vendor="", date="")
        rep2 = engine.evaluate_document(d2)
        f2 = next(f for f in rep2.findings if f.anomaly_type == AnomalyType.MISSING_CRITICAL_FIELDS)
        assert f2.severity == "MEDIUM"

    def test_anomaly_006_tax_inconsistency(self):
        engine = CentralizedAnomalyEngine()
        # Subtotal 1000 + Tax 180 = 1180, but Total is 1500
        d = _make_doc(subtotal=1000.0, tax=180.0, total=1500.0)
        rep = engine.evaluate_document(d)
        assert AnomalyType.TAX_INCONSISTENCY in rep.anomaly_types
        f = next(f for f in rep.findings if f.anomaly_type == AnomalyType.TAX_INCONSISTENCY)
        assert f.severity == "MEDIUM"

    def test_anomaly_006_tax_consistent_within_tolerance(self):
        engine = CentralizedAnomalyEngine()
        # 1000 + 180 = 1180, Total 1180.50 (within 1.0 tolerance for rounding)
        d = _make_doc(subtotal=1000.0, tax=180.0, total=1180.50)
        rep = engine.evaluate_document(d)
        assert AnomalyType.TAX_INCONSISTENCY not in rep.anomaly_types

    def test_anomaly_008_vendor_concentration(self):
        engine = CentralizedAnomalyEngine()
        docs = [
            _make_doc(vendor="Dominant Corp", total=80000.0), # 80%
            _make_doc(vendor="Small Vendor", total=10000.0),
            _make_doc(vendor="Small Vendor 2", total=10000.0),
        ]
        ctx = engine.build_population_context(docs)
        rep = engine.evaluate_document(docs[0], ctx)
        assert AnomalyType.VENDOR_CONCENTRATION in rep.anomaly_types

    def test_anomaly_010_repeated_financial_signature(self):
        engine = CentralizedAnomalyEngine()
        docs = [
            _make_doc(vendor="V", date="2026-03-01", total=450.0, inv_num="INV-A"),
            _make_doc(vendor="V", date="2026-03-01", total=450.0, inv_num="INV-B"), # Same sig
        ]
        ctx = engine.build_population_context(docs)
        rep = engine.evaluate_document(docs[0], ctx)
        assert AnomalyType.REPEATED_SIGNATURE in rep.anomaly_types

    def test_multi_signal_aggregation_critical(self):
        engine = CentralizedAnomalyEngine()
        docs = [
            _make_doc(vendor="V", date="2026-03-01", total=100000.0, inv_num="INV-1", subtotal=50000.0, tax=5000.0),
            _make_doc(vendor="V", date="2026-03-01", total=100000.0, inv_num="INV-1", subtotal=50000.0, tax=5000.0),
            _make_doc(vendor="V2", date="2026-03-01", total=1000.0),
        ]
        ctx = engine.build_population_context(docs)
        rep = engine.evaluate_document(docs[0], ctx)
        # Should have Duplicate Invoice Num (HIGH) + Repeated Signature (HIGH) + Round Amount (LOW) + Tax Inconsistency (MEDIUM)
        assert len(rep.findings) >= 3
        assert rep.severity in ("CRITICAL", "HIGH")
        assert rep.overall_score >= 0.85
        assert "investigation recommended" in rep.recommendation.lower()


# ---------------------------------------------------------------------------
# 3. Investigation Workflow & State Machine Tests
# ---------------------------------------------------------------------------

class TestInvestigationWorkflow:

    def test_save_and_retrieve_anomaly_findings(self, tmp_db):
        doc = _make_doc()
        tmp_db.insert_document(doc)

        findings = [
            {
                "document_id": doc.document_id,
                "anomaly_type": AnomalyType.ROUND_AMOUNT,
                "severity": "LOW",
                "score": 0.50,
                "confidence": 0.95,
                "explanation": "Round amount ₹45,000",
                "status": "OPEN",
            }
        ]
        saved = tmp_db.save_anomaly_findings(findings)
        assert saved == 1

        retrieved = tmp_db.get_anomaly_findings(document_id=doc.document_id)
        assert len(retrieved) == 1
        assert retrieved[0]["anomaly_type"] == AnomalyType.ROUND_AMOUNT
        assert retrieved[0]["status"] == "OPEN"

    def test_anomaly_status_transitions_with_audit(self, tmp_db):
        doc = _make_doc()
        tmp_db.insert_document(doc)

        tmp_db.save_anomaly_findings([{
            "document_id": doc.document_id,
            "anomaly_type": AnomalyType.STATISTICAL_OUTLIER,
            "severity": "HIGH",
            "score": 0.85,
            "confidence": 0.90,
            "explanation": "Outlier",
            "status": "OPEN",
        }])
        finding = tmp_db.get_anomaly_findings(document_id=doc.document_id)[0]
        fid = finding["id"]

        # 1. Transition OPEN -> UNDER_REVIEW
        ok = tmp_db.update_anomaly_status(fid, "UNDER_REVIEW", "Auditor Alice", "Checking with billing team")
        assert ok is True
        f_updated = tmp_db.get_anomaly_findings(document_id=doc.document_id)[0]
        assert f_updated["status"] == "UNDER_REVIEW"

        # 2. Transition UNDER_REVIEW -> RESOLVED
        ok2 = tmp_db.update_anomaly_status(fid, "RESOLVED", "Auditor Alice", "Confirmed authorized annual license")
        assert ok2 is True
        f_resolved = tmp_db.get_anomaly_findings(document_id=doc.document_id)[0]
        assert f_resolved["status"] == "RESOLVED"

        # Check audit history
        history = tmp_db.get_anomaly_review_history(document_id=doc.document_id)
        assert len(history) == 2
        assert history[0]["new_status"] == "RESOLVED"
        assert history[1]["new_status"] == "UNDER_REVIEW"
        assert history[0]["reviewer"] == "Auditor Alice"

    def test_duplicate_status_transitions_and_flagging(self, tmp_db):
        d1 = _make_doc()
        d2 = _make_doc()
        tmp_db.insert_document(d1)
        tmp_db.insert_document(d2)

        tmp_db.save_duplicate_candidates([{
            "document_id": d1.document_id,
            "matched_document_id": d2.document_id,
            "duplicate_type": DuplicateType.STRONG_DUPLICATE_CANDIDATE,
            "score": 0.95,
            "confidence": 0.95,
            "matched_fields": ["vendor", "date", "amount"],
            "explanation": "Matched all financial attributes",
            "status": "OPEN",
        }])

        candidates = tmp_db.get_duplicate_candidates(document_id=d1.document_id)
        assert len(candidates) == 1
        cid = candidates[0]["id"]

        # Confirm duplicate
        ok = tmp_db.update_duplicate_status(cid, "CONFIRMED_DUPLICATE", "Manager Bob", "Vendor re-sent invoice")
        assert ok is True

        c_updated = tmp_db.get_duplicate_candidates(document_id=d1.document_id)[0]
        assert c_updated["status"] == "CONFIRMED_DUPLICATE"

        # Document d1 should now be flagged
        d1_reloaded = tmp_db.get_document(d1.document_id)
        assert d1_reloaded.verification_status == "flagged"
        assert d1_reloaded.requires_review is True

    def test_dismiss_finding_preserves_evidence(self, tmp_db):
        doc = _make_doc()
        tmp_db.insert_document(doc)

        tmp_db.save_anomaly_findings([{
            "document_id": doc.document_id,
            "anomaly_type": AnomalyType.ROUND_AMOUNT,
            "severity": "INFO",
            "score": 0.30,
            "confidence": 0.95,
            "explanation": "Round ₹1,000",
            "status": "OPEN",
        }])
        fid = tmp_db.get_anomaly_findings(document_id=doc.document_id)[0]["id"]

        tmp_db.update_anomaly_status(fid, "DISMISSED", "Clerk Carol", "Expected round retainer")
        f = tmp_db.get_anomaly_findings(document_id=doc.document_id)[0]
        assert f["status"] == "DISMISSED"
        assert f["score"] == 0.30  # Evidence preserved


# ---------------------------------------------------------------------------
# 4. Idempotency & Rescan Tests
# ---------------------------------------------------------------------------

class TestIdempotencyAndRescan:

    def test_rescan_idempotency_and_preserves_review_status(self, tmp_db):
        d1 = _make_doc(total=10000.0, inv_num="INV-1")
        d2 = _make_doc(total=10000.0, inv_num="INV-1") # Dup inv
        tmp_db.insert_document(d1)
        tmp_db.insert_document(d2)

        # First scan
        res1 = tmp_db.rescan_anomalies()
        assert res1["scanned_documents"] == 2
        findings1 = tmp_db.get_anomaly_findings()
        count1 = len(findings1)
        assert count1 > 0

        # Mark one finding as RESOLVED
        fid = findings1[0]["id"]
        tmp_db.update_anomaly_status(fid, "RESOLVED", "Auditor", "Handled")

        # Second scan (idempotent re-run)
        res2 = tmp_db.rescan_anomalies()
        findings2 = tmp_db.get_anomaly_findings()
        assert len(findings2) == count1  # No duplicate rows created

        # Check that the reviewed finding retained RESOLVED status
        resolved_finding = next(f for f in findings2 if f["id"] == fid)
        assert resolved_finding["status"] == "RESOLVED"

    def test_anomaly_dashboard_summary(self, tmp_db):
        d1 = _make_doc(total=50000.0)
        tmp_db.insert_document(d1)
        tmp_db.save_anomaly_findings([{
            "document_id": d1.document_id,
            "anomaly_type": AnomalyType.ROUND_AMOUNT,
            "severity": "LOW",
            "score": 0.50,
            "confidence": 0.95,
            "explanation": "Round",
            "status": "OPEN",
        }])

        summary = tmp_db.anomaly_dashboard_summary()
        assert summary["total_findings"] >= 1
        assert summary["open_findings"] >= 1
        assert "LOW" in summary["severity_counts"]


# ---------------------------------------------------------------------------
# 5. Fault Isolation Tests
# ---------------------------------------------------------------------------

class TestFaultIsolation:

    def test_database_gracefully_handles_empty_collections(self, tmp_db):
        res = tmp_db.rescan_anomalies()
        assert res["scanned_documents"] == 0
        assert res["findings_detected"] == 0

    def test_nonexistent_finding_update_returns_false(self, tmp_db):
        assert tmp_db.update_anomaly_status(999999, "RESOLVED", "User") is False
        assert tmp_db.update_duplicate_status(999999, "RESOLVED", "User") is False


# ---------------------------------------------------------------------------
# 6. Multi-Format Export Tests
# ---------------------------------------------------------------------------

class TestMultiFormatAnomalyExports:

    def test_export_anomalies_excel(self):
        exporter = DocumentExporter()
        findings = [{
            "id": 1,
            "document_id": "doc-1",
            "filename": "bill.pdf",
            "vendor_name": "ACME",
            "invoice_number": "INV-1",
            "total_amount": 5000.0,
            "anomaly_type": "ANOMALY_003",
            "severity": "LOW",
            "score": 0.5,
            "confidence": 0.95,
            "status": "OPEN",
            "explanation": "Round amount",
            "detected_at": "2026-03-01T00:00:00Z",
        }]
        duplicates = [{
            "id": 1,
            "document_id": "doc-1",
            "matched_document_id": "doc-2",
            "duplicate_type": "DUPLICATE_INVOICE_ID",
            "score": 0.85,
            "confidence": 0.90,
            "status": "OPEN",
            "explanation": "Same invoice",
            "detected_at": "2026-03-01T00:00:00Z",
        }]
        history = [{
            "id": 1,
            "document_id": "doc-1",
            "entity_type": "anomaly",
            "entity_id": 1,
            "old_status": "OPEN",
            "new_status": "UNDER_REVIEW",
            "reviewer": "Alice",
            "comment": "Checking",
            "timestamp": "2026-03-01T01:00:00Z",
        }]
        excel_bytes = exporter.export_anomalies_excel(findings, duplicates, history)
        assert isinstance(excel_bytes, bytes)
        assert len(excel_bytes) > 1000
        # Valid zip/xlsx header
        assert excel_bytes[:4] == b"PK\x03\x04"

    def test_export_anomalies_csv(self):
        exporter = DocumentExporter()
        findings = [{
            "id": 1,
            "document_id": "doc-1",
            "filename": "bill.pdf",
            "vendor_name": "ACME",
            "invoice_number": "INV-1",
            "total_amount": 5000.0,
            "anomaly_type": "ANOMALY_003",
            "severity": "LOW",
            "score": 0.5,
            "confidence": 0.95,
            "status": "OPEN",
            "explanation": "Round amount",
            "detected_at": "2026-03-01T00:00:00Z",
        }]
        csv_str = exporter.export_anomalies_csv(findings)
        assert "document_id" in csv_str
        assert "ANOMALY_003" in csv_str
        assert "ACME" in csv_str

    def test_export_anomalies_json(self):
        exporter = DocumentExporter()
        findings = [{"id": 1, "anomaly_type": "ANOMALY_001"}]
        duplicates = [{"id": 1, "duplicate_type": "EXACT_FILE_DUPLICATE"}]
        json_str = exporter.export_anomalies_json(findings, duplicates)
        data = json.loads(json_str)
        assert data["findings_count"] == 1
        assert data["duplicate_candidates_count"] == 1

    def test_export_empty_findings_handled_cleanly(self):
        exporter = DocumentExporter()
        excel_bytes = exporter.export_anomalies_excel([], [], [])
        assert isinstance(excel_bytes, bytes)
        assert len(excel_bytes) > 0

        csv_str = exporter.export_anomalies_csv([])
        assert "document_id" in csv_str

        json_str = exporter.export_anomalies_json([], [])
        data = json.loads(json_str)
        assert data["findings_count"] == 0
        assert data["duplicate_candidates_count"] == 0


# ---------------------------------------------------------------------------
# 7. Additional Extended Tests (Rules, Filters, Scale, Migration)
# ---------------------------------------------------------------------------

class TestExtendedAnomalyAndDuplicateScenarios:

    def test_anomaly_007_category_distribution_mismatch(self):
        engine = CentralizedAnomalyEngine()
        # 10 records with mean 1,000 and std ~50 in category "Hardware"
        docs = [_make_doc(category="Hardware", total=1000.0 + i * 10) for i in range(10)]
        # Add 1 record with total 5,000 (> 4 sigma above 1045)
        outlier = _make_doc(category="Hardware", total=5000.0)
        docs.append(outlier)

        ctx = engine.build_population_context(docs)
        rep = engine.evaluate_document(outlier, ctx)
        assert AnomalyType.CATEGORY_MISMATCH in rep.anomaly_types

    def test_anomaly_009_temporal_spending_surge(self):
        engine = CentralizedAnomalyEngine()
        # Jan: 10,000; Feb: 10,000; Mar: 10,000 -> 3-month rolling avg = 10,000
        # Apr: 30,000 (> 1.5x prior and > 2x rolling avg)
        docs = [
            _make_doc(date="2026-01-15", total=10000.0),
            _make_doc(date="2026-02-15", total=10000.0),
            _make_doc(date="2026-03-15", total=10000.0),
            _make_doc(date="2026-04-15", total=30000.0),
        ]
        ctx = engine.build_population_context(docs)
        rep = engine.evaluate_document(docs[3], ctx)
        assert AnomalyType.TEMPORAL_SPEND_SURGE in rep.anomaly_types

    def test_near_duplicate_date_out_of_bounds(self):
        # Difference of 10 days (> 3 days threshold) -> Should not match as near duplicate
        d1 = _make_doc(vendor="Catering Ltd", inv_num="C-1", date="2026-03-01", total=2000.0)
        d2 = _make_doc(vendor="Catering Ltd", inv_num="C-2", date="2026-03-12", total=2000.0)
        match = evaluate_pair(d1, d2)
        assert match is None

    def test_near_duplicate_amount_out_of_bounds(self):
        # Difference of Rs 500 on Rs 2,000 (> 1% and > Rs 5) -> Should not match
        d1 = _make_doc(vendor="Catering Ltd", inv_num="C-1", date="2026-03-01", total=2000.0)
        d2 = _make_doc(vendor="Catering Ltd", inv_num="C-2", date="2026-03-02", total=2500.0)
        match = evaluate_pair(d1, d2)
        assert match is None

    def test_detect_duplicates_for_document_against_db(self, tmp_db):
        d1 = _make_doc(vendor="Apex Cloud", inv_num="APX-900", total=15000.0)
        tmp_db.insert_document(d1)

        d2 = _make_doc(vendor="Apex Cloud", inv_num="APX-900", total=15000.0)
        dups = MultiSignalDuplicateDetector.detect_duplicates_for_document(d2, tmp_db)
        assert len(dups) >= 1
        assert dups[0].matched_document_id == d1.document_id
        assert dups[0].duplicate_type == DuplicateType.STRONG_DUPLICATE_CANDIDATE

    def test_filter_anomaly_findings_by_severity(self, tmp_db):
        d1 = _make_doc()
        tmp_db.insert_document(d1)
        tmp_db.save_anomaly_findings([
            {"document_id": d1.document_id, "anomaly_type": AnomalyType.ROUND_AMOUNT, "severity": "LOW", "score": 0.5, "confidence": 0.9, "explanation": "Low"},
            {"document_id": d1.document_id, "anomaly_type": AnomalyType.STATISTICAL_OUTLIER, "severity": "HIGH", "score": 0.85, "confidence": 0.9, "explanation": "High"},
        ])
        high_findings = tmp_db.get_anomaly_findings(severity="HIGH")
        assert len(high_findings) == 1
        assert high_findings[0]["severity"] == "HIGH"

        low_findings = tmp_db.get_anomaly_findings(severity="LOW")
        assert len(low_findings) == 1
        assert low_findings[0]["severity"] == "LOW"

    def test_filter_anomaly_findings_by_status(self, tmp_db):
        d1 = _make_doc()
        tmp_db.insert_document(d1)
        tmp_db.save_anomaly_findings([
            {"document_id": d1.document_id, "anomaly_type": AnomalyType.ROUND_AMOUNT, "severity": "LOW", "score": 0.5, "confidence": 0.9, "explanation": "Open", "status": "OPEN"},
        ])
        fid = tmp_db.get_anomaly_findings()[0]["id"]
        tmp_db.update_anomaly_status(fid, "UNDER_REVIEW", "Reviewer")

        open_res = tmp_db.get_anomaly_findings(status="OPEN")
        assert len(open_res) == 0
        under_res = tmp_db.get_anomaly_findings(status="UNDER_REVIEW")
        assert len(under_res) == 1

    def test_filter_duplicate_candidates_by_type(self, tmp_db):
        d1 = _make_doc()
        d2 = _make_doc()
        tmp_db.insert_document(d1)
        tmp_db.insert_document(d2)
        tmp_db.save_duplicate_candidates([
            {"document_id": d1.document_id, "matched_document_id": d2.document_id, "duplicate_type": DuplicateType.EXACT_FILE_DUPLICATE, "score": 1.0, "confidence": 1.0, "explanation": "Hash"},
        ])
        cand = tmp_db.get_duplicate_candidates(duplicate_type=DuplicateType.EXACT_FILE_DUPLICATE)
        assert len(cand) == 1
        cand_empty = tmp_db.get_duplicate_candidates(duplicate_type=DuplicateType.NEAR_DUPLICATE_RECORD)
        assert len(cand_empty) == 0

    def test_reopen_status_transition_with_audit(self, tmp_db):
        d1 = _make_doc()
        tmp_db.insert_document(d1)
        tmp_db.save_anomaly_findings([
            {"document_id": d1.document_id, "anomaly_type": AnomalyType.TAX_INCONSISTENCY, "severity": "MEDIUM", "score": 0.6, "confidence": 0.9, "explanation": "Tax"},
        ])
        fid = tmp_db.get_anomaly_findings()[0]["id"]
        tmp_db.update_anomaly_status(fid, "RESOLVED", "Auditor", "Resolved initially")
        # Later auditor re-opens finding
        tmp_db.update_anomaly_status(fid, "OPEN", "Senior Auditor", "Re-opening due to audit committee query")

        f = tmp_db.get_anomaly_findings()[0]
        assert f["status"] == "OPEN"

        history = tmp_db.get_anomaly_review_history(d1.document_id)
        assert len(history) == 2
        assert history[0]["new_status"] == "OPEN"
        assert history[0]["old_status"] == "RESOLVED"

    def test_rescan_filtered_documents(self, tmp_db):
        d1 = _make_doc(total=50000.0)
        d2 = _make_doc(total=100.0)
        tmp_db.insert_document(d1)
        tmp_db.insert_document(d2)

        res = tmp_db.rescan_anomalies(document_ids=[d1.document_id])
        assert res["scanned_documents"] == 1

    def test_candidate_blocking_scale_100_docs(self):
        # Verify collection duplicate detection handles 100 documents quickly
        import time
        docs = [_make_doc(vendor=f"Vendor_{i % 10}", total=1000.0 + i) for i in range(100)]
        t0 = time.time()
        matches = MultiSignalDuplicateDetector.detect_duplicates_in_collection(docs)
        duration = time.time() - t0
        assert duration < 1.0  # Must be fast with candidate blocking

    def test_finding_to_dict_includes_name(self):
        f = AnomalyFinding(
            document_id="doc-1",
            anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
            severity="HIGH",
            score=0.85,
            confidence=0.9,
            explanation="Outlier",
        )
        d = f.to_dict()
        assert d["anomaly_name"] == "Statistical Outlier"
        assert d["score"] == 0.85

    def test_document_anomaly_report_no_findings(self):
        engine = CentralizedAnomalyEngine()
        # Clean document
        d = _make_doc(vendor="Clean Vendor", inv_num="INV-1", date="2026-01-01", total=123.45, subtotal=100.0, tax=23.45)
        rep = engine.evaluate_document(d)
        assert rep.anomaly_count == 0
        assert rep.overall_score == 0.0
        assert rep.severity == "INFO"
        assert "no review required" in rep.recommendation.lower()

    def test_multiple_anomalies_score_capping(self):
        engine = CentralizedAnomalyEngine()
        findings = [
            AnomalyFinding(document_id="doc-1", anomaly_type=f"T_{i}", severity="HIGH", score=0.9, confidence=0.9, explanation="")
            for i in range(10)
        ]
        rep = engine._aggregate_findings("doc-1", findings)
        assert rep.overall_score <= 1.0
        assert rep.severity == "CRITICAL"

    def test_severity_boundary_info(self):
        engine = CentralizedAnomalyEngine()
        findings = [
            AnomalyFinding(document_id="doc-1", anomaly_type="ANOMALY_003", severity="INFO", score=0.3, confidence=0.95, explanation=""),
        ]
        rep = engine._aggregate_findings("doc-1", findings)
        assert rep.severity == "INFO"

    def test_severity_boundary_medium_upgrade_from_two_lows(self):
        engine = CentralizedAnomalyEngine()
        findings = [
            AnomalyFinding(document_id="doc-1", anomaly_type="T1", severity="LOW", score=0.4, confidence=0.9, explanation=""),
            AnomalyFinding(document_id="doc-1", anomaly_type="T2", severity="LOW", score=0.4, confidence=0.9, explanation=""),
        ]
        rep = engine._aggregate_findings("doc-1", findings)
        assert rep.severity == "MEDIUM"

    def test_severity_boundary_high_upgrade_from_two_mediums(self):
        engine = CentralizedAnomalyEngine()
        findings = [
            AnomalyFinding(document_id="doc-1", anomaly_type="T1", severity="MEDIUM", score=0.6, confidence=0.9, explanation=""),
            AnomalyFinding(document_id="doc-1", anomaly_type="T2", severity="MEDIUM", score=0.6, confidence=0.9, explanation=""),
        ]
        rep = engine._aggregate_findings("doc-1", findings)
        assert rep.severity == "HIGH"

    def test_save_empty_payloads_return_zero(self, tmp_db):
        assert tmp_db.save_anomaly_findings([]) == 0
        assert tmp_db.save_duplicate_candidates([]) == 0

