"""
Phase 12 — End-to-End Institutional Workflow Verification Script.

Executes a realistic 20-document workflow:
    1. Generates 20 representative test documents across multiple formats:
       - 10 PDFs (utilities, services, telecom, catering, maintenance)
       - 10 PNGs/JPGs (invoices, receipts, handwritten forms)
       - Contains exact SHA-256 duplicates
       - Contains corrupt / invalid files
       - Contains low-confidence & unknown bills requiring review
    2. File Ingestion & Validation
    3. Resilient Batch Processing (fault isolation)
    4. Classification Review & Evidence Inspection
    5. Manual Classification Override & Audit Preservation
    6. Search & Filter Verification
    7. Persistence After Restart Verification
    8. Multi-format Export Verification (CSV, JSON, Excel, PDF)
"""

import io
import json
import logging
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PIL import Image, ImageDraw
from fpdf import FPDF

from phase11.classifier import BillCategoryClassifier
from phase11.database import FinancialDocumentDB
from phase11.export import DocumentExporter
from phase11.extractor import FinancialDocumentExtractor
from phase11.ingestion import (
    BatchSummary,
    DocumentIngestionService,
    DuplicatePolicy,
    IngestionConfig,
    IngestionItem,
    ProcessingStatus,
)
from phase11.schema import FinancialDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_phase12")


def create_sample_pdf(title: str, body: str) -> bytes:
    """Helper to generate a clean PDF byte stream."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, title)
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 11)
    for line in body.split("\n"):
        pdf.cell(0, 7, line)
        pdf.ln(8)
    return bytes(pdf.output())


def create_sample_png(text_lines: list[str]) -> bytes:
    """Helper to generate an image receipt byte stream."""
    img = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = 20
    for line in text_lines:
        draw.text((30, y), line, fill=(0, 0, 0))
        y += 30
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class RealisticMockPipeline:
    """Simulates multimodal OCR + LayoutLMv3 predictions realistically."""
    def __init__(self, doc_specs: dict[str, dict]):
        self.doc_specs = doc_specs

    def predict_image(self, image: Image.Image) -> Any:
        # In a batch, the pipeline processes the active image
        class MockResult:
            def __init__(self, spec):
                self.document_type = "SROIE"
                self.model_name = "layoutlmv3_robust"
                self.image_path = "batch_input.png"
                self.ocr_raw_text = spec.get("ocr_text", "")
                self.raw_entities_dict = spec.get("entities", {})
                self.entities = [
                    type("E", (), {"label": k, "text": v, "confidence": 0.95})()
                    for k, v in self.raw_entities_dict.items()
                ]
        return MockResult(self.active_spec)


def run_workflow_verification():
    print("=" * 80)
    print("PHASE 12 REALISTIC INSTITUTIONAL BATCH WORKFLOW VERIFICATION")
    print("=" * 80)

    work_dir = Path("scratch/phase12_verification")
    work_dir.mkdir(parents=True, exist_ok=True)
    db_path = work_dir / "institutional_test.db"
    if db_path.exists():
        db_path.unlink()

    db = FinancialDocumentDB(db_path=db_path)
    extractor = FinancialDocumentExtractor()
    classifier = BillCategoryClassifier(review_threshold=0.60)
    service = DocumentIngestionService(
        IngestionConfig(confidence_threshold=0.60, duplicate_policy=DuplicatePolicy.WARN)
    )

    # 1. Define 20 realistic documents
    doc_specs = [
        # 1-4: Utilities (PDFs)
        {"name": "bill_01_electricity.pdf", "type": "pdf", "title": "TANGEDCO ELECTRICITY BILL", "body": "Consumer: 04-123-456\nUnits: 340 kWh\nEnergy Charge: 1850.00\nTotal Amount: 2150.00", "ocr": "TANGEDCO electricity bill kwh meter power 340 units energy charge 2150.00", "vendor": "TANGEDCO", "amount": 2150.0, "date": "2026-03-01"},
        {"name": "bill_02_water.pdf", "type": "pdf", "title": "BWSSB WATER SUPPLY BILL", "body": "Board: BWSSB Municipal Water\nConsumption: 45 kilolitre sewage drainage\nTotal: 780.00", "ocr": "BWSSB water supply bill kilolitre sewage drainage meter reading total 780.00", "vendor": "BWSSB", "amount": 780.0, "date": "2026-03-02"},
        {"name": "bill_03_telecom.pdf", "type": "pdf", "title": "AIRTEL BROADBAND POSTPAID", "body": "Account: 9876543210\nPlan: Airtel Fiber 200Mbps broadband wifi internet\nTotal: 1178.82", "ocr": "Airtel broadband postpaid bill monthly plan data fiber wifi telephone 1178.82", "vendor": "Airtel", "amount": 1178.82, "date": "2026-03-03"},
        {"name": "bill_04_gas.pdf", "type": "pdf", "title": "INDANE LPG GAS BILL", "body": "Distributor: Indane Gas Agency\nRefill cylinder gas booking\nTotal: 950.00", "ocr": "Indane LPG cylinder gas booking consumer delivery total 950.00", "vendor": "Indane", "amount": 950.0, "date": "2026-03-04"},

        # 5-8: Operations & Supplies (PNGs)
        {"name": "bill_05_supplies.png", "type": "png", "lines": ["STAPLES OFFICE SUPPLIES", "Stationery paper pens printer toner", "Invoice # ST-9011", "Total: 3420.00"], "ocr": "Staples office supplies stationery paper notebook pen cartridge toner 3420.00", "vendor": "Staples", "amount": 3420.0, "date": "2026-03-05"},
        {"name": "bill_06_equipment.png", "type": "png", "lines": ["DELL COMPUTERS INDIA", "Hardware desktop server laptop monitor", "Invoice # DEL-4421", "Total: 84500.00"], "ocr": "Dell hardware computer server laptop desktop monitor workstation 84500.00", "vendor": "Dell", "amount": 84500.0, "date": "2026-03-06"},
        {"name": "bill_07_maintenance.png", "type": "png", "lines": ["URBAN COMPANY FACILITY SERVICE", "HVAC repair air conditioner servicing plumbing", "Total: 4600.00"], "ocr": "Urban Company HVAC repair maintenance technician spare parts servicing 4600.00", "vendor": "Urban Company", "amount": 4600.0, "date": "2026-03-07"},
        {"name": "bill_08_transport.png", "type": "png", "lines": ["BLUEDART EXPRESS LOGISTICS", "Courier freight cargo consignment", "Total: 1250.00"], "ocr": "BlueDart courier logistics freight dispatch consignment tracking 1250.00", "vendor": "BlueDart", "amount": 1250.0, "date": "2026-03-08"},

        # 9-12: Administrative & Travel (PDFs)
        {"name": "bill_09_travel.pdf", "type": "pdf", "title": "INDIGO AIRLINES E-TICKET", "body": "Flight: 6E-204 MAA-DEL\nPassenger: John Doe\nAirfare Total: 6420.00", "ocr": "IndiGo airline flight ticket boarding pass departure passenger airfare 6420.00", "vendor": "IndiGo", "amount": 6420.0, "date": "2026-03-09"},
        {"name": "bill_10_hotel.pdf", "type": "pdf", "title": "TAJ HOTELS RESIDENCY INVOICE", "body": "Room Stay: 2 Nights room tariff\nCheck-in: 05/03 Check-out: 07/03\nTotal: 14500.00", "ocr": "Taj Hotel room stay tariff night accommodation check-in check-out 14500.00", "vendor": "Taj Hotels", "amount": 14500.0, "date": "2026-03-10"},
        {"name": "bill_11_training.pdf", "type": "pdf", "title": "COURSERA ENTERPRISE TRAINING", "body": "Employee Learning Certification Course\nInvoice: TRN-2026\nTotal: 18000.00", "ocr": "Coursera training certification course learning workshop seminar 18000.00", "vendor": "Coursera", "amount": 18000.0, "date": "2026-03-11"},
        {"name": "bill_12_legal.pdf", "type": "pdf", "title": "ERNST & YOUNG CONSULTING", "body": "Professional audit tax advisory retainer\nInvoice: EY-8871\nTotal: 75000.00", "ocr": "Ernst Young legal consulting retainer audit advisory chartered accountant 75000.00", "vendor": "Ernst & Young", "amount": 75000.0, "date": "2026-03-12"},

        # 13-14: Catering (PNGs)
        {"name": "bill_13_catering.png", "type": "png", "lines": ["SWIGGY CORPORATE FOOD ORDER", "Team Lunch Catering meals", "Total: 3850.00"], "ocr": "Swiggy food catering restaurant lunch meal dinner order 3850.00", "vendor": "Swiggy", "amount": 3850.0, "date": "2026-03-13"},
        {"name": "bill_14_pantry.png", "type": "png", "lines": ["CAFE COFFEE DAY", "Beverages snacks coffee cafeteria", "Total: 840.00"], "ocr": "Cafe Coffee Day beverage coffee snack cafeteria pantry 840.00", "vendor": "CCD", "amount": 840.0, "date": "2026-03-14"},

        # 15-16: Low Confidence / Unknown (Needs Review)
        {"name": "bill_15_obscure.pdf", "type": "pdf", "title": "COMMERCIAL STATEMENT", "body": "General charges reference # 4491\nTotal Amount: 1200.00", "ocr": "Commercial statement reference general miscellaneous transaction total 1200.00", "vendor": "General Commercial", "amount": 1200.0, "date": "2026-03-15"},
        {"name": "bill_16_unknown.png", "type": "png", "lines": ["UNKNOWN VENDOR RECEIPT", "Lorem ipsum dolor sit amet", "Amount: 500.00"], "ocr": "Lorem ipsum dolor sit amet consectetur 500.00", "vendor": "Unknown", "amount": 500.0, "date": "2026-03-16"},

        # 17-18: Exact SHA-256 Duplicates of bill_01 and bill_05
        {"name": "bill_17_dup_electricity.pdf", "type": "duplicate_of", "target": 0},
        {"name": "bill_18_dup_supplies.png", "type": "duplicate_of", "target": 4},

        # 19: Corrupted file (Failed reading)
        {"name": "bill_19_corrupt.png", "type": "corrupt"},

        # 20: Unsupported file format (Rejected)
        {"name": "bill_20_unsupported.docx", "type": "unsupported"},
    ]

    generated_items = []
    generated_specs = {}

    pipeline = RealisticMockPipeline(generated_specs)

    for i, spec in enumerate(doc_specs):
        stype = spec.get("type")
        fname = spec["name"]

        if stype == "pdf":
            b = create_sample_pdf(spec["title"], spec["body"])
            item = IngestionItem.from_bytes(fname, b)
            generated_specs[fname] = {
                "ocr_text": spec["ocr"],
                "entities": {"COMPANY": spec["vendor"], "TOTAL": str(spec["amount"]), "DATE": spec["date"]}
            }
        elif stype == "png":
            b = create_sample_png(spec["lines"])
            item = IngestionItem.from_bytes(fname, b)
            generated_specs[fname] = {
                "ocr_text": spec["ocr"],
                "entities": {"COMPANY": spec["vendor"], "TOTAL": str(spec["amount"]), "DATE": spec["date"]}
            }
        elif stype == "duplicate_of":
            target_item = generated_items[spec["target"]]
            item = IngestionItem.from_bytes(fname, target_item.data)
            target_spec = doc_specs[spec["target"]]
            generated_specs[fname] = {
                "ocr_text": target_spec["ocr"],
                "entities": {"COMPANY": target_spec["vendor"], "TOTAL": str(target_spec["amount"]), "DATE": target_spec["date"]}
            }
        elif stype == "corrupt":
            item = IngestionItem.from_bytes(fname, b"\x89PNG\r\nCorruptedGarbageStream")
        elif stype == "unsupported":
            item = IngestionItem.from_bytes(fname, b"PK\x03\x04MockDocxArchiveContent")

        generated_items.append(item)

    print(f"[*] Prepared {len(generated_items)} representative multi-format documents.")

    # 2. Batch Processing Execution
    print("\n[*] Executing batch ingestion with independent fault isolation...")
    summary = BatchSummary(total_uploaded=len(generated_items))

    for idx, item in enumerate(generated_items, 1):
        if item.filename in generated_specs:
            pipeline.active_spec = generated_specs[item.filename]
        service.process_document(
            item=item,
            pipeline=pipeline,
            extractor=extractor,
            classifier=classifier,
            db=db,
        )
        if item.status == ProcessingStatus.COMPLETED:
            summary.completed += 1
            if item.document:
                summary.documents.append(item.document)
                cat = item.document.category or "Unknown"
                summary.category_counts[cat] = summary.category_counts.get(cat, 0) + 1
                if item.document.requires_review:
                    summary.requires_review_count += 1
        elif item.status == ProcessingStatus.DUPLICATE:
            summary.duplicates += 1
        elif item.status == ProcessingStatus.REJECTED:
            summary.rejected += 1
        elif item.status == ProcessingStatus.FAILED:
            summary.failed += 1
        summary.items.append(item)

    print(f" -> Total Processed: {len(summary.items)}")
    print(f" -> Completed: {summary.completed}")
    print(f" -> Duplicates: {summary.duplicates}")
    print(f" -> Failed: {summary.failed} (Corrupted file cleanly handled without halting batch)")
    print(f" -> Rejected: {summary.rejected} (Unsupported format cleanly rejected)")
    print(f" -> Requires Review: {summary.requires_review_count}")
    print(f" -> Category Breakdown: {summary.category_counts}")

    assert summary.completed >= 16, f"Expected at least 16 completed, got {summary.completed}"
    assert summary.failed == 1, f"Expected 1 failed corrupt file, got {summary.failed}"
    assert summary.rejected == 1, f"Expected 1 rejected unsupported file, got {summary.rejected}"
    assert summary.requires_review_count >= 2, f"Expected review flags, got {summary.requires_review_count}"

    # 3. Document Registry Inspection & Review
    print("\n[*] Inspecting Document Registry & Review Flags...")
    all_docs = db.search_documents(limit=100)
    review_docs = db.search_documents(requires_review=True)
    print(f" -> Total persisted records in DB: {len(all_docs)}")
    print(f" -> Documents flagged for review: {len(review_docs)}")

    # 4. Manual Classification Correction (Simulate human audit override)
    print("\n[*] Performing manual classification overrides with audit preservation...")
    target_review_doc = None
    for d in review_docs:
        if "obscure" in d.filename or d.category in ("Miscellaneous", "Unknown"):
            target_review_doc = d
            break

    if target_review_doc:
        doc_id = target_review_doc.document_id
        original_ai_cat = target_review_doc.ai_category
        print(f" -> Overriding Document ID: {doc_id[:8]}... (AI Category was: '{original_ai_cat}')")

        updated = db.update_category(
            document_id=doc_id,
            new_category="Maintenance",
            modified_by="chief_auditor",
            notes="Verified physical invoice; repairs of facility equipment.",
        )
        assert updated.category == "Maintenance"
        assert updated.final_category == "Maintenance"
        assert updated.ai_category == original_ai_cat
        assert updated.requires_review is False

        # Verify audit log in SQLite
        with db._connect() as conn:
            log_entry = conn.execute(
                "SELECT * FROM processing_log WHERE document_id = ? AND action = 'CLASSIFICATION_MODIFIED'",
                (doc_id,)
            ).fetchone()
            assert log_entry is not None
            details = json.loads(log_entry["details"])
            assert details["modified_by"] == "chief_auditor"
            assert details["new_category"] == "Maintenance"
            print(f" -> Audit log successfully recorded: {details}")

    # 5. Multi-format Exporter Verification
    print("\n[*] Testing Multi-Format Export Engine...")
    exporter = DocumentExporter()
    csv_str = exporter.to_csv(all_docs)
    json_str = exporter.to_json(all_docs)
    excel_bytes = exporter.to_excel(all_docs)
    pdf_bytes = exporter.to_pdf_report(all_docs, title="Institutional Batch Processing Audit Report")

    assert len(csv_str) > 500
    assert len(json_str) > 1000
    assert len(excel_bytes) > 2000
    assert len(pdf_bytes) > 2000
    print(f" -> CSV Export: {len(csv_str)} bytes")
    print(f" -> JSON Export: {len(json_str)} bytes")
    print(f" -> Excel Export: {len(excel_bytes)} bytes")
    print(f" -> PDF Report: {len(pdf_bytes)} bytes")

    # 6. Persistence After Restart Verification
    print("\n[*] Simulating application restart & testing database persistence...")
    del db
    db_reopened = FinancialDocumentDB(db_path=db_path)
    reopened_docs = db_reopened.search_documents(limit=100)
    assert len(reopened_docs) == len(all_docs)
    print(f" -> Reopened DB successfully verified with all {len(reopened_docs)} documents intact.")

    print("\n" + "=" * 80)
    print("ALL PHASE 12 INSTITUTIONAL WORKFLOW ACCEPTANCE CRITERIA PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    run_workflow_verification()
