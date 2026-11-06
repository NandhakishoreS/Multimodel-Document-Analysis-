"""
Phase 12 — Automated Regression and Unit Tests.

Covers:
    - IngestionItem validation (PDF, PNG, JPG, JPEG, unsupported, oversized, corrupted)
    - Cryptographic SHA-256 file hashing and determinism
    - Duplicate detection (exact hash collisions & secondary financial duplicates)
    - Duplicate handling policies (SKIP, WARN, PROCESS)
    - Independent batch processing (single document failure does not terminate batch)
    - Processing lifecycle statuses (QUEUED, VALIDATING, OCR, EXTRACTING, SAVING, COMPLETED, FAILED, DUPLICATE, REJECTED)
    - Classification explainability and low-confidence review flagging
    - Manual classification correction with audit preservation (ai_category vs final_category)
    - Database schema migration & persistence of Phase 12 fields
    - End-to-end batch ingestion workflow
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest
from PIL import Image

from phase11.classifier import BillCategoryClassifier
from phase11.database import FinancialDocumentDB
from phase11.extractor import FinancialDocumentExtractor
from phase11.ingestion import (
    BatchSummary,
    DocumentIngestionService,
    DuplicatePolicy,
    IngestionConfig,
    IngestionItem,
    ProcessingStatus,
)
from phase11.schema import FinancialDocument, LineItem


# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------

def _create_dummy_image_bytes(format_name: str = "PNG", size: tuple[int, int] = (100, 100)) -> bytes:
    """Generate in-memory image bytes for test documents."""
    img = Image.new("RGB", size, color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format=format_name)
    return buf.getvalue()


def _create_dummy_pdf_bytes() -> bytes:
    """Generate a minimal valid 1-page PDF using fpdf2."""
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, "ELECTRICITY BILL TANGEDCO 250 kWh TOTAL 1500.00")
    return bytes(pdf.output())


class MockEntity:
    def __init__(self, label: str, text: str, confidence: float = 0.95):
        self.label = label
        self.text = text
        self.confidence = confidence


class MockInferenceResult:
    def __init__(
        self,
        doc_type: str = "SROIE",
        raw_entities: Optional[Dict[str, str]] = None,
        ocr_text: str = "",
        model_name: str = "test_layoutlmv3",
    ):
        self.document_type = doc_type
        self.raw_entities_dict = raw_entities or {
            "COMPANY": "TANGEDCO",
            "DATE": "12/09/2026",
            "TOTAL": "1500.00",
            "ADDRESS": "Chennai",
        }
        self.ocr_raw_text = ocr_text or "TANGEDCO electricity bill kwh 250 units total 1500"
        self.model_name = model_name
        self.entities = [
            MockEntity(k, v) for k, v in self.raw_entities_dict.items()
        ]
        self.image_path = "mock.png"


class MockPipeline:
    def __init__(self, should_fail: bool = False, custom_result: Optional[MockInferenceResult] = None):
        self.should_fail = should_fail
        self.custom_result = custom_result or MockInferenceResult()

    def predict_image(self, image: Image.Image) -> MockInferenceResult:
        if self.should_fail:
            raise RuntimeError("OCR or Multimodal Model inference crashed.")
        return self.custom_result


@pytest.fixture
def test_db(tmp_path):
    """Fixture providing a fresh isolated SQLite database."""
    db_file = tmp_path / "test_p12_financials.db"
    return FinancialDocumentDB(db_path=db_file)


@pytest.fixture
def service():
    """Default DocumentIngestionService fixture."""
    return DocumentIngestionService()


# ---------------------------------------------------------------------------
# 1. Ingestion & File Validation Tests
# ---------------------------------------------------------------------------

class TestFileValidation:

    def test_valid_png_ingestion(self, service):
        data = _create_dummy_image_bytes("PNG")
        item = IngestionItem.from_bytes("bill_01.png", data)
        assert service.validate_item(item) is True
        assert item.status == ProcessingStatus.QUEUED
        assert len(item.rendered_images) == 1

    def test_valid_jpg_ingestion(self, service):
        data = _create_dummy_image_bytes("JPEG")
        item = IngestionItem.from_bytes("invoice.jpg", data)
        assert service.validate_item(item) is True
        assert item.status == ProcessingStatus.QUEUED
        assert len(item.rendered_images) == 1

    def test_valid_pdf_ingestion(self, service):
        pdf_data = _create_dummy_pdf_bytes()
        item = IngestionItem.from_bytes("utility.pdf", pdf_data)
        assert service.validate_item(item) is True
        assert item.status == ProcessingStatus.QUEUED
        assert len(item.rendered_images) >= 1
        assert isinstance(item.rendered_images[0], Image.Image)

    def test_unsupported_extension_rejected(self, service):
        data = b"Some text document content"
        item = IngestionItem.from_bytes("contract.docx", data)
        assert service.validate_item(item) is False
        assert item.status == ProcessingStatus.REJECTED
        assert "Unsupported file format" in (item.error_message or "")

    def test_executable_rejected(self, service):
        data = b"MZ\x90\x00\x03\x00\x00\x00"
        item = IngestionItem.from_bytes("payload.exe", data)
        assert service.validate_item(item) is False
        assert item.status == ProcessingStatus.REJECTED

    def test_empty_file_rejected(self, service):
        item = IngestionItem.from_bytes("empty.png", b"")
        assert service.validate_item(item) is False
        assert item.status == ProcessingStatus.REJECTED
        assert "empty" in (item.error_message or "").lower()

    def test_oversized_file_rejected(self):
        cfg = IngestionConfig(max_upload_size_bytes=1024)  # 1 KB limit
        svc = DocumentIngestionService(config=cfg)
        data = b"A" * 2048
        item = IngestionItem.from_bytes("large.png", data)
        assert svc.validate_item(item) is False
        assert item.status == ProcessingStatus.REJECTED
        assert "exceeds maximum limit" in (item.error_message or "").lower()

    def test_corrupted_image_fails_gracefully(self, service):
        # A file with .png extension but completely corrupted byte stream
        corrupt_data = b"\x89PNG\r\n\x1a\nCorruptedGarbageBytes12345"
        item = IngestionItem.from_bytes("corrupt.png", corrupt_data)
        assert service.validate_item(item) is False
        assert item.status == ProcessingStatus.FAILED
        assert "Unable to read" in (item.error_message or "")


# ---------------------------------------------------------------------------
# 2. Cryptographic Hashing Tests
# ---------------------------------------------------------------------------

class TestCryptographicHashing:

    def test_hash_calculation_deterministic(self):
        data1 = b"Invoice content sample 1"
        data2 = b"Invoice content sample 1"
        h1 = IngestionItem.calculate_hash(data1)
        h2 = IngestionItem.calculate_hash(data2)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 produces 64 hex characters

    def test_distinct_content_distinct_hash(self):
        h1 = IngestionItem.calculate_hash(b"Document A")
        h2 = IngestionItem.calculate_hash(b"Document B")
        assert h1 != h2

    def test_item_initializes_with_hash(self):
        data = _create_dummy_image_bytes("PNG")
        item = IngestionItem.from_bytes("test.png", data)
        assert item.file_hash != ""
        assert len(item.file_hash) == 64


# ---------------------------------------------------------------------------
# 3. Duplicate Handling Tests
# ---------------------------------------------------------------------------

class TestDuplicateHandling:

    def test_exact_hash_duplicate_detected(self, test_db, service):
        data = _create_dummy_image_bytes("PNG")
        item1 = IngestionItem.from_bytes("original.png", data)

        pipeline = MockPipeline()
        extractor = FinancialDocumentExtractor()
        clf = BillCategoryClassifier()

        # Process first copy
        res1 = service.process_document(item1, pipeline, extractor, clf, test_db)
        assert res1.status == ProcessingStatus.COMPLETED

        # Attempt to process duplicate with SKIP policy
        item2 = IngestionItem.from_bytes("duplicate_copy.png", data)
        res2 = service.process_document(
            item2, pipeline, extractor, clf, test_db,
            duplicate_policy=DuplicatePolicy.SKIP,
        )
        assert res2.status == ProcessingStatus.DUPLICATE
        assert res2.stage == "SKIPPED_DUPLICATE"
        assert res2.existing_document_id == res1.document_id

    def test_duplicate_policy_warn_flags_document(self, test_db, service):
        data = _create_dummy_image_bytes("PNG")
        item1 = IngestionItem.from_bytes("bill_v1.png", data)

        pipeline = MockPipeline()
        extractor = FinancialDocumentExtractor()
        clf = BillCategoryClassifier()

        res1 = service.process_document(item1, pipeline, extractor, clf, test_db)
        assert res1.status == ProcessingStatus.COMPLETED

        # Process second copy with WARN policy
        item2 = IngestionItem.from_bytes("bill_v2.png", data)
        res2 = service.process_document(
            item2, pipeline, extractor, clf, test_db,
            duplicate_policy=DuplicatePolicy.WARN,
        )
        assert res2.status == ProcessingStatus.COMPLETED
        assert res2.document is not None
        assert any("duplicate" in flag.lower() for flag in res2.document.anomaly_flags)
        assert res2.document.requires_review is True

    def test_financial_duplicate_detected_across_metadata(self, test_db, service):
        # Two documents with different images/hashes, but same vendor, date, and total
        data1 = _create_dummy_image_bytes("PNG", size=(100, 100))
        data2 = _create_dummy_image_bytes("PNG", size=(120, 120))  # different bytes -> different hash

        mock_res = MockInferenceResult(
            raw_entities={"COMPANY": "Vodafone", "DATE": "01/08/2026", "TOTAL": "899.00"},
            ocr_text="Vodafone broadband bill 899",
        )
        pipeline = MockPipeline(custom_result=mock_res)
        extractor = FinancialDocumentExtractor()
        clf = BillCategoryClassifier()

        item1 = IngestionItem.from_bytes("inv1.png", data1)
        service.process_document(item1, pipeline, extractor, clf, test_db)

        item2 = IngestionItem.from_bytes("inv2.png", data2)
        res2 = service.process_document(item2, pipeline, extractor, clf, test_db)

        assert res2.status == ProcessingStatus.COMPLETED
        assert res2.document is not None
        assert any("financial duplicate" in flag.lower() for flag in res2.document.anomaly_flags)
        assert res2.document.requires_review is True


# ---------------------------------------------------------------------------
# 4. Independent Batch Processing Tests
# ---------------------------------------------------------------------------

class TestBatchProcessing:

    def test_batch_processes_independent_documents(self, test_db, service):
        data1 = _create_dummy_image_bytes("PNG", size=(80, 80))
        data2 = _create_dummy_image_bytes("PNG", size=(90, 90))
        data3 = _create_dummy_image_bytes("PNG", size=(100, 100))

        items = [
            IngestionItem.from_bytes("doc1.png", data1),
            IngestionItem.from_bytes("doc2.png", data2),
            IngestionItem.from_bytes("doc3.png", data3),
        ]

        pipeline = MockPipeline()
        extractor = FinancialDocumentExtractor()
        clf = BillCategoryClassifier()

        summary = service.process_batch(items, pipeline, extractor, clf, test_db)

        assert summary.total_uploaded == 3
        assert summary.completed == 3
        assert summary.failed == 0
        assert len(summary.documents) == 3
        assert summary.total_processing_time > 0.0

    def test_single_document_failure_does_not_stop_batch(self, test_db, service):
        # doc1: valid, doc2: corrupted, doc3: valid
        doc1_data = _create_dummy_image_bytes("PNG")
        doc2_corrupt = b"\x89PNG\r\nCorruptedByteStream!!!"
        doc3_data = _create_dummy_image_bytes("PNG", size=(150, 150))

        items = [
            IngestionItem.from_bytes("valid1.png", doc1_data),
            IngestionItem.from_bytes("bad2.png", doc2_corrupt),
            IngestionItem.from_bytes("valid3.png", doc3_data),
        ]

        pipeline = MockPipeline()
        extractor = FinancialDocumentExtractor()
        clf = BillCategoryClassifier()

        summary = service.process_batch(items, pipeline, extractor, clf, test_db)

        assert summary.total_uploaded == 3
        assert summary.completed == 2
        assert summary.failed == 1
        assert summary.rejected == 0
        # Check that valid3 was indeed processed successfully after bad2
        assert items[2].status == ProcessingStatus.COMPLETED

    def test_batch_progress_callback(self, test_db, service):
        items = [
            IngestionItem.from_bytes("doc_a.png", _create_dummy_image_bytes()),
            IngestionItem.from_bytes("doc_b.png", _create_dummy_image_bytes(size=(110, 110))),
        ]
        callbacks_received = []

        def on_progress(current, total, item):
            callbacks_received.append((current, total, item.filename, item.status))

        pipeline = MockPipeline()
        extractor = FinancialDocumentExtractor()
        clf = BillCategoryClassifier()

        service.process_batch(
            items, pipeline, extractor, clf, test_db,
            progress_callback=on_progress,
        )

        assert len(callbacks_received) == 2
        assert callbacks_received[0][0] == 1
        assert callbacks_received[1][0] == 2


# ---------------------------------------------------------------------------
# 5. Classification & Low-Confidence Review Tests
# ---------------------------------------------------------------------------

class TestClassificationAndReview:

    def test_high_confidence_does_not_require_review(self):
        clf = BillCategoryClassifier(review_threshold=0.50)
        result = clf.classify(ocr_text="TANGEDCO electricity bill kwh meter power energy tariff")
        assert result.category == "Electricity"
        assert result.confidence >= 0.50
        assert result.requires_review is False
        assert "Confident classification" in result.reason

    def test_low_confidence_flags_requires_review(self):
        clf = BillCategoryClassifier(review_threshold=0.80)
        # Moderate evidence with confidence ~0.33 < 0.80
        result = clf.classify(ocr_text="bill", vendor_name="TANGEDCO")
        assert result.category == "Electricity"
        assert result.confidence < 0.80
        assert result.requires_review is True
        assert "below review threshold" in result.reason

    def test_unknown_category_always_requires_review(self):
        clf = BillCategoryClassifier()
        result = clf.classify(ocr_text="Unrelated unstructured rambling text without keywords")
        assert result.category == "Unknown"
        assert result.requires_review is True
        assert "Insufficient diagnostic keywords" in result.reason

    def test_configurable_review_threshold_in_classify(self):
        clf = BillCategoryClassifier(review_threshold=0.90)
        # Passing lower explicit threshold overrides default
        result = clf.classify(
            ocr_text="Indane LPG cylinder gas booking",
            review_threshold=0.20,
        )
        assert result.category == "Gas"
        assert result.requires_review is False


# ---------------------------------------------------------------------------
# 6. Manual Classification Correction Tests
# ---------------------------------------------------------------------------

class TestManualClassificationOverride:

    def test_manual_override_preserves_ai_category(self, test_db):
        doc = FinancialDocument(
            document_id="doc-override-001",
            category="Miscellaneous",
            ai_category="Miscellaneous",
            ai_confidence=0.45,
            final_category="Miscellaneous",
            requires_review=True,
            total_amount=500.0,
        )
        test_db.insert_document(doc)

        # Human operator overrides classification
        updated = test_db.update_category(
            document_id="doc-override-001",
            new_category="Maintenance",
            modified_by="auditor_jane",
            notes="Correction verified against physical receipt.",
        )

        assert updated.category == "Maintenance"
        assert updated.final_category == "Maintenance"
        assert updated.ai_category == "Miscellaneous"
        assert updated.ai_confidence == 0.45
        assert updated.requires_review is False

        # Verify in fresh query from DB
        fetched = test_db.get_document("doc-override-001")
        assert fetched.category == "Maintenance"
        assert fetched.final_category == "Maintenance"
        assert fetched.ai_category == "Miscellaneous"
        assert "auditor_jane" not in fetched.category

        # Verify audit log entry
        with test_db._connect() as conn:
            log_entry = conn.execute(
                "SELECT * FROM processing_log WHERE document_id = 'doc-override-001' AND action = 'CLASSIFICATION_MODIFIED'"
            ).fetchone()
            assert log_entry is not None
            details = json.loads(log_entry["details"])
            assert details["previous_category"] == "Miscellaneous"
            assert details["new_category"] == "Maintenance"
            assert details["modified_by"] == "auditor_jane"


# ---------------------------------------------------------------------------
# 7. Database Migration & Persistence Tests
# ---------------------------------------------------------------------------

class TestDatabasePhase12Migration:

    def test_schema_migration_preserves_data(self, tmp_path):
        import sqlite3
        db_path = tmp_path / "legacy.db"

        # 1. Create a simulated legacy Phase 11 database without Phase 12 columns
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL DEFAULT '',
                source_file TEXT NOT NULL DEFAULT '',
                document_type TEXT NOT NULL DEFAULT 'invoice',
                benchmark_domain TEXT NOT NULL DEFAULT '',
                model_name TEXT NOT NULL DEFAULT '',
                category TEXT NOT NULL DEFAULT 'Unknown',
                sub_category TEXT,
                classification_confidence REAL NOT NULL DEFAULT 0.0,
                classification_evidence TEXT NOT NULL DEFAULT '[]',
                vendor_name TEXT,
                address TEXT,
                invoice_number TEXT,
                invoice_date TEXT,
                due_date TEXT,
                subtotal REAL,
                tax_amount REAL,
                total_amount REAL,
                currency TEXT NOT NULL DEFAULT 'INR',
                extraction_confidence REAL NOT NULL DEFAULT 0.0,
                ocr_word_count INTEGER NOT NULL DEFAULT 0,
                ocr_raw_text TEXT NOT NULL DEFAULT '',
                verification_status TEXT NOT NULL DEFAULT 'pending',
                processing_timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                anomaly_flags TEXT NOT NULL DEFAULT '[]'
            );
        """)
        conn.execute("""
            CREATE TABLE line_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                description TEXT,
                quantity REAL,
                unit_price REAL,
                total REAL,
                unit TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE processing_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                action TEXT,
                timestamp TEXT,
                details TEXT
            );
        """)
        conn.execute("INSERT INTO documents (document_id, filename, category, processing_timestamp, created_at, updated_at) VALUES ('legacy-001', 'old.png', 'Utilities', '2026-01-01', '2026-01-01', '2026-01-01')")
        conn.commit()
        conn.close()

        # 2. Open with FinancialDocumentDB - triggering auto-migration
        db = FinancialDocumentDB(db_path=db_path)

        # 3. Verify legacy record intact
        legacy_doc = db.get_document("legacy-001")
        assert legacy_doc is not None
        assert legacy_doc.filename == "old.png"
        assert legacy_doc.category == "Utilities"
        # New fields default properly
        assert legacy_doc.file_hash is None
        assert legacy_doc.requires_review is False

        # 4. Verify new document with Phase 12 fields can be saved and retrieved
        new_doc = FinancialDocument(
            document_id="p12-doc-002",
            filename="new.pdf",
            file_hash="abc123hash",
            file_size=54321,
            processing_status="COMPLETED",
            ai_category="Water",
            final_category="Water",
            requires_review=True,
            classification_reason="Matched water keyword",
        )
        db.insert_document(new_doc)

        retrieved = db.get_document("p12-doc-002")
        assert retrieved.file_hash == "abc123hash"
        assert retrieved.file_size == 54321
        assert retrieved.ai_category == "Water"
        assert retrieved.requires_review is True
        assert retrieved.classification_reason == "Matched water keyword"

    def test_search_by_requires_review_and_status(self, test_db):
        doc1 = FinancialDocument(
            document_id="rev-yes",
            requires_review=True,
            processing_status="COMPLETED",
        )
        doc2 = FinancialDocument(
            document_id="rev-no",
            requires_review=False,
            processing_status="COMPLETED",
        )
        test_db.insert_document(doc1)
        test_db.insert_document(doc2)

        review_needed = test_db.search_documents(requires_review=True)
        assert len(review_needed) == 1
        assert review_needed[0].document_id == "rev-yes"

        no_review = test_db.search_documents(requires_review=False)
        assert len(no_review) == 1
        assert no_review[0].document_id == "rev-no"


# ---------------------------------------------------------------------------
# 8. End-to-End Ingestion Integration Test
# ---------------------------------------------------------------------------

class TestEndToEndIngestion:

    def test_end_to_end_pipeline_flow(self, test_db, service):
        pdf_bytes = _create_dummy_pdf_bytes()
        item = IngestionItem.from_bytes("march_electricity_bill.pdf", pdf_bytes)

        mock_res = MockInferenceResult(
            doc_type="SROIE",
            raw_entities={
                "COMPANY": "TANGEDCO",
                "DATE": "10/03/2026",
                "TOTAL": "4250.00",
                "ADDRESS": "Chennai",
            },
            ocr_text="TANGEDCO electricity bill kwh 450 units meter reading amount 4250.00",
        )
        pipeline = MockPipeline(custom_result=mock_res)
        extractor = FinancialDocumentExtractor()
        clf = BillCategoryClassifier()

        processed_item = service.process_document(
            item=item,
            pipeline=pipeline,
            extractor=extractor,
            classifier=clf,
            db=test_db,
        )

        assert processed_item.status == ProcessingStatus.COMPLETED
        assert processed_item.stage == "COMPLETED"
        assert processed_item.duration_seconds > 0.0

        # Verify persisted document
        doc_in_db = test_db.get_document(processed_item.document_id)
        assert doc_in_db is not None
        assert doc_in_db.filename == "march_electricity_bill.pdf"
        assert doc_in_db.vendor_name == "TANGEDCO"
        assert doc_in_db.total_amount == 4250.0
        assert doc_in_db.category == "Electricity"
        assert doc_in_db.file_hash == processed_item.file_hash
        assert doc_in_db.processing_status == "COMPLETED"
        assert doc_in_db.processing_duration > 0.0
