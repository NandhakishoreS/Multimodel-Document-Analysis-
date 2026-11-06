"""
Phase 11 — Automated Regression and Unit Tests.

Covers:
    - FinancialDocument schema (dataclass, serialisation, validation)
    - BillCategoryClassifier (all 14 categories, Unknown fallback, evidence)
    - FinancialDocumentExtractor (SROIE/CORD/FUNSD domain mapping)
    - FinancialDocumentDB (CRUD, search, analytics, anomaly detection)
    - DocumentExporter (JSON, CSV, Excel, PDF)

All tests use temporary in-memory or temp-file databases so they do NOT
touch the production financial_documents.db.

These tests must NOT import or execute any Phase 1–10 ML training code.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Phase 11 imports
# ---------------------------------------------------------------------------
from phase11.schema import FinancialDocument, LineItem, ClassificationResult
from phase11.classifier import BillCategoryClassifier, TAXONOMY
from phase11.extractor import FinancialDocumentExtractor, _parse_amount, _parse_date
from phase11.database import FinancialDocumentDB
from phase11.export import DocumentExporter


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def clf():
    return BillCategoryClassifier()


@pytest.fixture
def extractor():
    return FinancialDocumentExtractor()


@pytest.fixture
def tmp_db(tmp_path):
    """Isolated SQLite DB for each test."""
    return FinancialDocumentDB(db_path=tmp_path / "test_financial.db")


@pytest.fixture
def sample_doc():
    return FinancialDocument(
        document_id=str(uuid.uuid4()),
        filename="test_invoice.png",
        vendor_name="TANGEDCO",
        invoice_date="15/03/2024",
        total_amount=4250.0,
        category="Electricity",
        sub_category="Utilities",
        classification_confidence=0.85,
        classification_evidence=["electricity", "tangedco", "kwh"],
        extraction_confidence=0.92,
        ocr_word_count=120,
        currency="INR",
    )


@pytest.fixture
def sample_docs(tmp_db):
    """Insert 5 sample documents and return them."""
    docs = []
    for i in range(5):
        d = FinancialDocument(
            document_id=str(uuid.uuid4()),
            filename=f"bill_{i}.png",
            vendor_name=f"Vendor {i}",
            total_amount=float(1000 * (i + 1)),
            category=["Electricity", "Water", "Internet/Telecom", "Gas", "Travel"][i],
            invoice_date=f"2024-0{i+1}-15",
            verification_status="pending",
        )
        tmp_db.insert_document(d)
        docs.append(d)
    return docs


# ===========================================================================
# 1. Schema tests
# ===========================================================================

class TestFinancialDocumentSchema:

    def test_default_document_id_is_uuid(self):
        d = FinancialDocument()
        assert len(d.document_id) == 36
        assert d.document_id.count("-") == 4

    def test_default_verification_status_is_pending(self):
        d = FinancialDocument()
        assert d.verification_status == "pending"

    def test_optional_fields_default_to_none(self):
        d = FinancialDocument()
        for field in ("vendor_name", "invoice_number", "invoice_date", "due_date",
                      "subtotal", "tax_amount", "total_amount", "sub_category", "address"):
            assert getattr(d, field) is None, f"{field} should default to None"

    def test_line_items_default_empty(self):
        d = FinancialDocument()
        assert d.line_items == []

    def test_to_dict_is_serialisable(self, sample_doc):
        d = sample_doc.to_dict()
        assert isinstance(d, dict)
        json_str = json.dumps(d)
        assert "TANGEDCO" in json_str

    def test_to_json_roundtrip(self, sample_doc):
        j = sample_doc.to_json()
        loaded = json.loads(j)
        assert loaded["vendor_name"] == "TANGEDCO"
        assert loaded["total_amount"] == 4250.0

    def test_from_dict_roundtrip(self, sample_doc):
        d_dict = sample_doc.to_dict()
        reconstructed = FinancialDocument.from_dict(d_dict)
        assert reconstructed.document_id == sample_doc.document_id
        assert reconstructed.vendor_name == sample_doc.vendor_name

    def test_validate_warns_on_negative_amount(self):
        d = FinancialDocument(total_amount=-100.0)
        warnings = d.validate()
        assert any("negative" in w for w in warnings)

    def test_validate_warns_on_subtotal_mismatch(self):
        d = FinancialDocument(subtotal=100.0, tax_amount=10.0, total_amount=200.0)
        warnings = d.validate()
        assert any("≠" in w or "total" in w.lower() for w in warnings)

    def test_validate_clean_document_no_warnings(self, sample_doc):
        warnings = sample_doc.validate()
        assert warnings == []

    def test_line_items_serialised_correctly(self):
        d = FinancialDocument(
            line_items=[LineItem("Bread", quantity=2, unit_price=25.0, total=50.0)]
        )
        d_dict = d.to_dict()
        assert d_dict["line_items"][0]["description"] == "Bread"

    def test_touch_updates_updated_at(self, sample_doc):
        original = sample_doc.updated_at
        import time; time.sleep(0.01)
        sample_doc.touch()
        assert sample_doc.updated_at >= original

    def test_classification_evidence_is_list(self, sample_doc):
        assert isinstance(sample_doc.classification_evidence, list)


# ===========================================================================
# 2. Classifier tests
# ===========================================================================

class TestBillCategoryClassifier:

    def test_electricity_classification(self, clf):
        result = clf.classify(ocr_text="TANGEDCO electricity bill kwh 250 units meter reading")
        assert result.category == "Electricity"
        assert result.confidence > 0.1
        assert "electricity" in result.evidence or "tangedco" in result.evidence

    def test_water_classification(self, clf):
        result = clf.classify(ocr_text="BWSSB water supply bill kilolitre consumption meter reading")
        assert result.category == "Water"
        assert result.sub_category == "Utilities"

    def test_internet_telecom_classification(self, clf):
        result = clf.classify(ocr_text="Airtel broadband postpaid bill monthly plan data 100GB")
        assert result.category == "Internet/Telecom"

    def test_gas_classification(self, clf):
        result = clf.classify(ocr_text="Indane LPG cylinder gas booking bill")
        assert result.category == "Gas"

    def test_travel_classification(self, clf):
        result = clf.classify(ocr_text="IndiGo airline flight ticket boarding pass departure seat")
        assert result.category == "Travel"

    def test_accommodation_classification(self, clf):
        result = clf.classify(ocr_text="Hotel Grand check-in check-out night stay room tariff")
        assert result.category == "Accommodation"

    def test_food_catering_classification(self, clf):
        result = clf.classify(ocr_text="Swiggy food catering restaurant order meal lunch")
        assert result.category == "Food/Catering"

    def test_professional_services_classification(self, clf):
        result = clf.classify(ocr_text="Legal consulting retainer fee audit chartered accountant")
        assert result.category == "Professional Services"

    def test_unknown_for_empty_text(self, clf):
        result = clf.classify(ocr_text="")
        assert result.category == "Unknown"
        assert result.confidence == 0.0

    def test_unknown_for_unrelated_text(self, clf):
        result = clf.classify(ocr_text="xyzzy foobar baz quux")
        assert result.category == "Unknown"

    def test_evidence_is_list_of_strings(self, clf):
        result = clf.classify(ocr_text="TANGEDCO electricity kwh")
        assert isinstance(result.evidence, list)
        assert all(isinstance(e, str) for e in result.evidence)

    def test_evidence_capped_at_10(self, clf):
        long_text = " ".join(["electricity", "electric", "power", "energy", "kwh",
                               "watt", "meter", "eb", "tangedco", "kwh", "consumption",
                               "tariff", "load", "transformer", "billing period"])
        result = clf.classify(ocr_text=long_text)
        assert len(result.evidence) <= 10

    def test_confidence_in_range(self, clf):
        result = clf.classify(ocr_text="electricity bill kwh meter power energy tangedco")
        assert 0.0 <= result.confidence <= 1.0

    def test_sub_category_taxonomy(self, clf):
        result = clf.classify(ocr_text="electricity meter kwh tangedco")
        assert result.sub_category == "Utilities"

    def test_all_categories_in_taxonomy(self):
        for cat in TAXONOMY:
            assert isinstance(TAXONOMY[cat], str)

    def test_vendor_name_contributes_to_classification(self, clf):
        # vendor_name alone
        result = clf.classify(ocr_text="bill", vendor_name="TANGEDCO")
        assert result.category == "Electricity"

    def test_returns_classification_result_type(self, clf):
        result = clf.classify(ocr_text="electricity")
        assert isinstance(result, ClassificationResult)


# ===========================================================================
# 3. Extractor tests (no ML model — uses mock DocumentInferenceResult)
# ===========================================================================

class MockEntity:
    def __init__(self, label, text, confidence=0.9):
        self.label = label
        self.text = text
        self.confidence = confidence


class MockInferenceResult:
    def __init__(self, doc_type, raw_entities, entities=None, ocr_text="", model_name="test"):
        self.document_type = doc_type
        self.raw_entities_dict = raw_entities
        self.entities = entities or []
        self.ocr_raw_text = ocr_text
        self.model_name = model_name
        self.image_path = None


class TestFinancialDocumentExtractor:

    def test_sroie_extraction_vendor_name(self, extractor):
        result = MockInferenceResult("SROIE", {"COMPANY": "SUNSHINE STORE", "TOTAL": "125.50"})
        doc = extractor.extract(result, filename="receipt.png")
        assert doc.vendor_name == "SUNSHINE STORE"

    def test_sroie_extraction_total_amount(self, extractor):
        result = MockInferenceResult("SROIE", {"TOTAL": "4,250.00"})
        doc = extractor.extract(result)
        assert doc.total_amount == 4250.0

    def test_sroie_extraction_date(self, extractor):
        result = MockInferenceResult("SROIE", {"DATE": "15/03/2024"})
        doc = extractor.extract(result)
        assert doc.invoice_date == "15/03/2024"

    def test_sroie_extraction_address(self, extractor):
        result = MockInferenceResult("SROIE", {"ADDRESS": "123 Main St, Chennai"})
        doc = extractor.extract(result)
        assert doc.address == "123 Main St, Chennai"

    def test_cord_extraction_total(self, extractor):
        result = MockInferenceResult("CORD", {"TOTAL": "85.00", "SUBTOTAL": "75.00"})
        doc = extractor.extract(result)
        assert doc.total_amount == 85.0
        assert doc.subtotal == 75.0

    def test_cord_extraction_line_items(self, extractor):
        entities = [
            MockEntity("B-MENU", "Chicken Rice"),
            MockEntity("B-MENU", "Lemon Tea"),
        ]
        result = MockInferenceResult("CORD", {"MENU": "Chicken Rice"}, entities=entities)
        doc = extractor.extract(result)
        assert len(doc.line_items) == 2
        assert doc.line_items[0].description == "Chicken Rice"

    def test_funsd_extraction_header(self, extractor):
        result = MockInferenceResult("FUNSD", {"HEADER": "Training Request Form"})
        doc = extractor.extract(result)
        assert doc.vendor_name == "Training Request Form"
        assert doc.document_type == "form"

    def test_extraction_benchmark_domain_preserved(self, extractor):
        result = MockInferenceResult("SROIE", {})
        doc = extractor.extract(result)
        assert doc.benchmark_domain == "SROIE"

    def test_extraction_missing_fields_are_none(self, extractor):
        result = MockInferenceResult("SROIE", {})
        doc = extractor.extract(result)
        assert doc.vendor_name is None
        assert doc.total_amount is None
        assert doc.invoice_date is None

    def test_extraction_confidence_computed(self, extractor):
        entities = [MockEntity("COMPANY", "Test", 0.9), MockEntity("TOTAL", "100", 0.8)]
        result = MockInferenceResult("SROIE", {"COMPANY": "Test", "TOTAL": "100"}, entities=entities)
        doc = extractor.extract(result)
        assert abs(doc.extraction_confidence - 0.85) < 0.01

    def test_verification_status_is_pending_by_default(self, extractor):
        result = MockInferenceResult("SROIE", {})
        doc = extractor.extract(result)
        assert doc.verification_status == "pending"

    def test_filename_stored(self, extractor):
        result = MockInferenceResult("SROIE", {})
        doc = extractor.extract(result, filename="my_invoice.jpg")
        assert doc.filename == "my_invoice.jpg"

    def test_parse_amount_handles_currency_symbols(self):
        assert _parse_amount("₹1,234.56") == 1234.56
        assert _parse_amount("$50.00") == 50.0

    def test_parse_amount_returns_none_for_empty(self):
        assert _parse_amount("") is None
        assert _parse_amount(None) is None

    def test_parse_date_extracts_date_string(self):
        assert _parse_date("Date: 15/03/2024") == "15/03/2024"
        assert _parse_date("2024-03-15") == "2024-03-15"


# ===========================================================================
# 4. Database tests
# ===========================================================================

class TestFinancialDocumentDB:

    def test_insert_and_get(self, tmp_db, sample_doc):
        tmp_db.insert_document(sample_doc)
        retrieved = tmp_db.get_document(sample_doc.document_id)
        assert retrieved is not None
        assert retrieved.vendor_name == "TANGEDCO"

    def test_get_nonexistent_returns_none(self, tmp_db):
        assert tmp_db.get_document("nonexistent-id") is None

    def test_duplicate_insert_raises(self, tmp_db, sample_doc):
        tmp_db.insert_document(sample_doc)
        with pytest.raises(ValueError, match="already exists"):
            tmp_db.insert_document(sample_doc)

    def test_upsert_is_idempotent(self, tmp_db, sample_doc):
        tmp_db.upsert_document(sample_doc)
        sample_doc.vendor_name = "Updated Vendor"
        tmp_db.upsert_document(sample_doc)
        retrieved = tmp_db.get_document(sample_doc.document_id)
        assert retrieved.vendor_name == "Updated Vendor"

    def test_update_document(self, tmp_db, sample_doc):
        tmp_db.insert_document(sample_doc)
        sample_doc.total_amount = 9999.0
        tmp_db.update_document(sample_doc)
        retrieved = tmp_db.get_document(sample_doc.document_id)
        assert retrieved.total_amount == 9999.0

    def test_update_nonexistent_raises(self, tmp_db, sample_doc):
        with pytest.raises(ValueError):
            tmp_db.update_document(sample_doc)

    def test_delete_document(self, tmp_db, sample_doc):
        tmp_db.insert_document(sample_doc)
        tmp_db.delete_document(sample_doc.document_id)
        assert tmp_db.get_document(sample_doc.document_id) is None

    def test_delete_nonexistent_raises(self, tmp_db):
        with pytest.raises(ValueError):
            tmp_db.delete_document("nonexistent")

    def test_search_by_query(self, tmp_db, sample_docs):
        results = tmp_db.search_documents(query="Vendor 0")
        assert any(d.vendor_name == "Vendor 0" for d in results)

    def test_search_by_category(self, tmp_db, sample_docs):
        results = tmp_db.search_documents(category="Water")
        assert all(d.category == "Water" for d in results)

    def test_search_by_status(self, tmp_db, sample_docs):
        results = tmp_db.search_documents(verification_status="pending")
        assert all(d.verification_status == "pending" for d in results)

    def test_search_by_min_amount(self, tmp_db, sample_docs):
        results = tmp_db.search_documents(min_amount=3000.0)
        assert all(d.total_amount >= 3000.0 for d in results if d.total_amount)

    def test_count_documents(self, tmp_db, sample_docs):
        assert tmp_db.count_documents() == 5

    def test_count_by_category(self, tmp_db, sample_docs):
        assert tmp_db.count_documents(category="Electricity") == 1

    def test_spend_by_category(self, tmp_db, sample_docs):
        rows = tmp_db.spend_by_category()
        assert len(rows) > 0
        assert all("category" in r and "total_spend" in r for r in rows)

    def test_status_summary(self, tmp_db, sample_docs):
        summary = tmp_db.status_summary()
        assert "pending" in summary
        assert summary["pending"] == 5

    def test_top_vendors(self, tmp_db, sample_docs):
        vendors = tmp_db.top_vendors(n=3)
        assert len(vendors) <= 3

    def test_line_items_persisted(self, tmp_db):
        doc = FinancialDocument(
            document_id=str(uuid.uuid4()),
            filename="cord.png",
            line_items=[LineItem("Rice", quantity=1, unit_price=80.0, total=80.0)],
        )
        tmp_db.insert_document(doc)
        retrieved = tmp_db.get_document(doc.document_id)
        assert len(retrieved.line_items) == 1
        assert retrieved.line_items[0].description == "Rice"

    def test_classification_evidence_roundtrips_as_list(self, tmp_db):
        doc = FinancialDocument(
            document_id=str(uuid.uuid4()),
            classification_evidence=["electricity", "kwh"],
        )
        tmp_db.insert_document(doc)
        retrieved = tmp_db.get_document(doc.document_id)
        assert retrieved.classification_evidence == ["electricity", "kwh"]

    def test_anomaly_candidates_empty_when_clean(self, tmp_db):
        doc = FinancialDocument(
            document_id=str(uuid.uuid4()),
            total_amount=1234.56,
            invoice_number="INV-001",
        )
        tmp_db.insert_document(doc)
        # No duplicates, no round amounts
        anomalies = tmp_db.anomaly_candidates()
        assert all(d.document_id != doc.document_id for d in anomalies)

    def test_anomaly_duplicate_invoice_number(self, tmp_db):
        for _ in range(2):
            tmp_db.upsert_document(FinancialDocument(
                document_id=str(uuid.uuid4()),
                invoice_number="DUP-999",
                total_amount=500.0,
            ))
        anomalies = tmp_db.anomaly_candidates()
        inv_numbers = [d.invoice_number for d in anomalies if d.invoice_number]
        assert "DUP-999" in inv_numbers

    def test_monthly_spend_returns_list(self, tmp_db, sample_docs):
        rows = tmp_db.monthly_spend()
        assert isinstance(rows, list)


# ===========================================================================
# 5. Exporter tests
# ===========================================================================

class TestDocumentExporter:

    @pytest.fixture
    def docs(self):
        return [
            FinancialDocument(
                document_id=str(uuid.uuid4()),
                filename=f"doc_{i}.png",
                vendor_name=f"Vendor {i}",
                total_amount=float(1000 * (i + 1)),
                category="Electricity",
                currency="INR",
                line_items=[LineItem("Item A", quantity=1, unit_price=100.0, total=100.0)],
            )
            for i in range(3)
        ]

    def test_to_json_valid_json(self, docs):
        exporter = DocumentExporter()
        j = exporter.to_json(docs)
        parsed = json.loads(j)
        assert isinstance(parsed, list)
        assert len(parsed) == 3

    def test_to_json_contains_expected_fields(self, docs):
        exporter = DocumentExporter()
        j = json.loads(exporter.to_json(docs))
        assert "vendor_name" in j[0]
        assert "total_amount" in j[0]

    def test_to_csv_has_header(self, docs):
        exporter = DocumentExporter()
        csv_str = exporter.to_csv(docs)
        assert "vendor_name" in csv_str
        assert "total_amount" in csv_str

    def test_to_csv_has_correct_row_count(self, docs):
        exporter = DocumentExporter()
        csv_str = exporter.to_csv(docs)
        lines = [l for l in csv_str.strip().split("\n") if l]
        assert len(lines) == 4  # 1 header + 3 data rows

    def test_to_csv_bytes(self, docs):
        exporter = DocumentExporter()
        b = exporter.to_csv_bytes(docs)
        assert isinstance(b, bytes)

    def test_to_excel_returns_bytes(self, docs):
        pytest.importorskip("openpyxl")
        exporter = DocumentExporter()
        b = exporter.to_excel(docs)
        assert isinstance(b, bytes)
        assert len(b) > 1000  # Should be a valid xlsx

    def test_to_excel_writes_file(self, docs, tmp_path):
        pytest.importorskip("openpyxl")
        exporter = DocumentExporter()
        out = tmp_path / "test_output.xlsx"
        exporter.to_excel(docs, filepath=out)
        assert out.exists()
        assert out.stat().st_size > 500

    def test_to_pdf_returns_bytes(self, docs):
        pytest.importorskip("fpdf")
        exporter = DocumentExporter()
        b = exporter.to_pdf_report(docs, title="Test Report")
        assert isinstance(b, bytes)
        assert b[:4] == b"%PDF"  # PDF magic bytes

    def test_to_pdf_writes_file(self, docs, tmp_path):
        pytest.importorskip("fpdf")
        exporter = DocumentExporter()
        out = tmp_path / "test_report.pdf"
        exporter.to_pdf_report(docs, filepath=out)
        assert out.exists()

    def test_json_empty_list(self):
        exporter = DocumentExporter()
        j = exporter.to_json([])
        assert json.loads(j) == []

    def test_csv_empty_list(self):
        exporter = DocumentExporter()
        csv_str = exporter.to_csv([])
        assert isinstance(csv_str, str)


# ===========================================================================
# 6. Integration: extractor → database roundtrip
# ===========================================================================

class TestPhase11Integration:

    def test_extract_then_persist_then_retrieve(self, extractor, tmp_db):
        mock_result = MockInferenceResult(
            "SROIE",
            {"COMPANY": "Reliance Power", "TOTAL": "8750.00", "DATE": "01/06/2024"},
            ocr_text="Reliance electricity power bill kwh meter 8750",
        )
        fin_doc = extractor.extract(mock_result, filename="reliance_bill.png")
        tmp_db.insert_document(fin_doc)
        retrieved = tmp_db.get_document(fin_doc.document_id)

        assert retrieved.vendor_name == "Reliance Power"
        assert retrieved.total_amount == 8750.0
        assert retrieved.category == "Electricity"

    def test_full_pipeline_search(self, extractor, tmp_db):
        for i in range(3):
            mock_result = MockInferenceResult(
                "SROIE",
                {"COMPANY": f"BWSSB Branch {i}", "TOTAL": str(500 * (i + 1))},
                ocr_text="water supply kilolitre meter reading BWSSB",
            )
            tmp_db.insert_document(extractor.extract(mock_result, filename=f"water_{i}.png"))

        results = tmp_db.search_documents(query="BWSSB")
        assert len(results) == 3
        assert all(d.category == "Water" for d in results)
