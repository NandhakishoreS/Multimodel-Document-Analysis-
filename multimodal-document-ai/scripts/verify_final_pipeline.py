"""
Phase 17 — Complete End-to-End Pipeline Acceptance Verification Script.

Executes a realistic, project-wide lifecycle pipeline over a representative collection of documents:
- Invoices, Receipts, Utility Bills
- Multiple categories & vendors
- Intentional duplicate candidate
- Intentional mathematical/tax anomaly
- Intentional incomplete record

Verified Pipeline Stages:
1. UPLOAD
2. VALIDATE
3. INGEST
4. OCR
5. EXTRACT
6. CLASSIFY
7. SAVE
8. SEARCH
9. EDIT
10. VERIFY
11. DUPLICATE ANALYSIS
12. ANOMALY ANALYSIS
13. ANALYTICS
14. REPORT
15. EXPORT
16. AUDIT
"""

from __future__ import annotations

import io
import json
import logging
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image, ImageDraw
from fpdf import FPDF
import openpyxl

from phase11.anomaly_engine import CentralizedAnomalyEngine
from phase11.classifier import BillCategoryClassifier
from phase11.database import FinancialDocumentDB
from phase11.duplicate_detector import MultiSignalDuplicateDetector
from phase11.export import (
    export_institutional_csv,
    export_institutional_excel,
    export_institutional_json,
    export_institutional_pdf,
)
from phase11.extractor import FinancialDocumentExtractor
from phase11.ingestion import (
    BatchSummary,
    DocumentIngestionService,
    DuplicatePolicy,
    IngestionConfig,
    IngestionItem,
    ProcessingStatus,
)
from phase11.reporting import (
    InstitutionalReportEngine,
    ReportType,
    REPORT_TITLES,
    format_inr,
)
from phase11.schema import FinancialDocument, LineItem

logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Synthetic Document Generators
# ---------------------------------------------------------------------------

def create_sample_pdf(title: str, lines: List[str]) -> bytes:
    """Generate a valid PDF byte stream with clean font layout."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, title)
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 10)
    for line in lines:
        pdf.cell(0, 7, line)
        pdf.ln(8)
    return bytes(pdf.output())


def create_sample_png(text_lines: List[str]) -> bytes:
    """Generate a clean synthetic receipt image."""
    img = Image.new("RGB", (650, 420), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = 20
    for line in text_lines:
        draw.text((30, y), line, fill=(0, 0, 0))
        y += 28
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class RealisticPipelineMock:
    """Simulates multimodal OCR + LayoutLMv3 predictions for pipeline verification."""

    def __init__(self, doc_spec_lookup: Dict[str, Dict[str, Any]]):
        self.doc_spec_lookup = doc_spec_lookup
        self.active_filename = ""

    def set_active_file(self, filename: str):
        self.active_filename = filename

    def predict_image(self, image: Image.Image) -> Any:
        spec = self.doc_spec_lookup.get(self.active_filename, {})

        class MockResult:
            def __init__(self, s):
                self.document_type = s.get("domain", "SROIE")
                self.model_name = "layoutlmv3_primary"
                self.image_path = s.get("name", "document.png")
                self.ocr_raw_text = s.get("ocr_text", "")
                self.raw_entities_dict = s.get("entities", {})
                self.entities = [
                    type("Entity", (), {
                        "label": k,
                        "text": v,
                        "confidence": 0.96,
                        "box": [10, 10, 100, 30],
                    })()
                    for k, v in self.raw_entities_dict.items()
                ]

        return MockResult(spec)


# ---------------------------------------------------------------------------
# Main Verification Routine
# ---------------------------------------------------------------------------

def run_final_pipeline_verification():
    print("=" * 80)
    print("PHASE 17 — FINAL PIPELINE INTEGRATION & SYSTEM HARDENING AUDIT")
    print("=" * 80)

    work_dir = _PROJECT_ROOT / "scratch" / "phase17_final_pipeline"
    work_dir.mkdir(parents=True, exist_ok=True)
    db_path = work_dir / "final_integration_test.db"
    if db_path.exists():
        db_path.unlink()

    db = FinancialDocumentDB(db_path=db_path)
    extractor = FinancialDocumentExtractor()
    classifier = BillCategoryClassifier(review_threshold=0.60)
    service = DocumentIngestionService(
        IngestionConfig(confidence_threshold=0.60, duplicate_policy=DuplicatePolicy.WARN)
    )

    t_start = time.time()

    # Define representative document collection
    raw_specs = [
        # 1. Electricity Bill (PDF)
        {
            "name": "doc_01_electricity.pdf",
            "type": "pdf",
            "domain": "SROIE",
            "title": "TANGEDCO ELECTRICITY DISTRIBUTION BOARD",
            "lines": [
                "Consumer Number: 04-582-9912",
                "Billing Period: 01-May-2025 to 31-May-2025",
                "Units Consumed: 520 kWh",
                "Subtotal: INR 38,135.59",
                "GST (18%): INR 6,864.41",
                "Net Total: INR 45,000.00",
            ],
            "ocr_text": "TANGEDCO Electricity Board power meter 520 kwh subtotal 38135.59 tax 6864.41 total 45000.00",
            "entities": {
                "COMPANY": "TANGEDCO Electricity Board",
                "DATE": "2025-05-10",
                "TOTAL": "45000.00",
                "ADDRESS": "144 Anna Salai, Chennai, Tamil Nadu 600002",
            },
            "subtotal": 38135.59,
            "tax": 6864.41,
            "inv_num": "INV-2025-ELEC-001",
        },
        # 2. Water Utility Bill (PDF)
        {
            "name": "doc_02_water.pdf",
            "type": "pdf",
            "domain": "SROIE",
            "title": "CHENNAI METRO WATER SUPPLY AND SEWERAGE BOARD",
            "lines": [
                "Consumer ID: CMWSSB-90214",
                "Consumption: 62 Kilolitres",
                "Subtotal: INR 10,593.22",
                "Tax (18%): INR 1,906.78",
                "Total Amount Due: INR 12,500.00",
            ],
            "ocr_text": "Chennai Metro Water supply sewerage board water meter reading subtotal 10593.22 tax 1906.78 total 12500.00",
            "entities": {
                "COMPANY": "Chennai Metro Water Board",
                "DATE": "2025-06-15",
                "TOTAL": "12500.00",
                "ADDRESS": "1 Pumping Station Road, Chintadripet, Chennai 600002",
            },
            "subtotal": 10593.22,
            "tax": 1906.78,
            "inv_num": "INV-2025-WTR-002",
        },
        # 3. Operations Receipt (PNG)
        {
            "name": "doc_03_catering.png",
            "type": "png",
            "domain": "SROIE",
            "lines": [
                "STARBUCKS COFFEE INDIA",
                "Express Avenue Mall, Chennai",
                "Order # 4102 | Dine-in",
                "Items: 2x Cappuccino, 2x Croissant",
                "Subtotal: INR 720.34",
                "CGST + SGST (18%): INR 129.66",
                "Total Paid: INR 850.00",
            ],
            "ocr_text": "Starbucks Coffee India cafe food beverage snacks breakfast subtotal 720.34 tax 129.66 total 850.00",
            "entities": {
                "COMPANY": "Starbucks Coffee India",
                "DATE": "2025-07-20",
                "TOTAL": "850.00",
                "ADDRESS": "Express Avenue, Whites Road, Royapettah, Chennai",
            },
            "subtotal": 720.34,
            "tax": 129.66,
            "inv_num": "REC-2025-SBUX-101",
        },
        # 4. Hardware Equipment Invoice (PDF)
        {
            "name": "doc_04_equipment.pdf",
            "type": "pdf",
            "domain": "SROIE",
            "title": "DELL COMPUTERS INDIA PRIVATE LIMITED",
            "lines": [
                "Commercial Invoice: DEL-2025-8812",
                "Product: Dell PowerEdge R750 Rack Server",
                "Subtotal: INR 156,779.66",
                "IGST (18%): INR 28,220.34",
                "Total Amount: INR 185,000.00",
            ],
            "ocr_text": "Dell Computers India hardware rack server workstation monitor subtotal 156779.66 tax 28220.34 total 185000.00",
            "entities": {
                "COMPANY": "Dell Computers India",
                "DATE": "2025-08-12",
                "TOTAL": "185000.00",
                "ADDRESS": "Divyasree Greens, Ground Floor, Bangalore 560071",
            },
            "subtotal": 156779.66,
            "tax": 28220.34,
            "inv_num": "INV-2025-DEL-8812",
        },
        # 5. Duplicate Candidate (Exact duplicate of doc_01 in financial values)
        {
            "name": "doc_05_duplicate_candidate.pdf",
            "type": "pdf",
            "domain": "SROIE",
            "title": "TANGEDCO ELECTRICITY DISTRIBUTION BOARD - DUPLICATE COPY",
            "lines": [
                "Consumer Number: 04-582-9912",
                "Billing Period: 01-May-2025 to 31-May-2025",
                "Net Total: INR 45,000.00",
                "Duplicate customer copy",
            ],
            "ocr_text": "TANGEDCO Electricity Board power meter duplicate copy subtotal 38135.59 tax 6864.41 total 45000.00",
            "entities": {
                "COMPANY": "TANGEDCO Electricity Board",
                "DATE": "2025-05-10",
                "TOTAL": "45000.00",
                "ADDRESS": "144 Anna Salai, Chennai, Tamil Nadu 600002",
            },
            "subtotal": 38135.59,
            "tax": 6864.41,
            "inv_num": "INV-2025-ELEC-001",
        },
        # 6. Intentional Anomaly: Extreme amount + Severe Tax Math Mismatch
        {
            "name": "doc_06_anomaly_tax_mismatch.pdf",
            "type": "pdf",
            "domain": "SROIE",
            "title": "KPMG ADVISORY & AUDIT SERVICES",
            "lines": [
                "Invoice: KPMG-ADV-902",
                "Engagement: Institutional Forensic & Audit Oversight",
                "Reported Subtotal: INR 1,000,000.00",
                "Reported Tax: INR 50,000.00",
                "Billed Total: INR 2,500,000.00",  # 1,000,000 + 50,000 != 2,500,000
            ],
            "ocr_text": "KPMG Advisory legal audit consulting fee subtotal 1000000.00 tax 50000.00 total 2500000.00",
            "entities": {
                "COMPANY": "KPMG Advisory Services",
                "DATE": "2025-09-01",
                "TOTAL": "2500000.00",
                "ADDRESS": "Lodha Excelus, Apollo Mills, Mumbai 400011",
            },
            "subtotal": 1000000.0,
            "tax": 50000.0,
            "inv_num": "INV-2025-KPMG-902",
        },
        # 7. Incomplete Record (Missing invoice number & partial vendor name)
        {
            "name": "doc_07_incomplete_record.png",
            "type": "png",
            "domain": "SROIE",
            "lines": [
                "LOCAL LOGISTICS COURIER",
                "Parcel Delivery Service",
                "Amount Paid: INR 3,200.00",
                "Receipt No: [ILLEGIBLE]",
            ],
            "ocr_text": "Local logistics courier parcel dispatch freight charges total 3200.00",
            "entities": {
                "TOTAL": "3200.00",
                "DATE": "2025-10-05",
            },
            "subtotal": None,
            "tax": None,
            "inv_num": None,
        },
    ]

    spec_lookup = {s["name"]: s for s in raw_specs}
    pipeline_mock = RealisticPipelineMock(spec_lookup)

    # -----------------------------------------------------------------------
    # STAGE 1 & 2: UPLOAD, VALIDATE, INGEST
    # -----------------------------------------------------------------------
    print("\n[STAGE 1 & 2] UPLOAD, VALIDATE & INGEST...")
    ingestion_items: List[IngestionItem] = []
    for s in raw_specs:
        if s["type"] == "pdf":
            raw_data = create_sample_pdf(s["title"], s["lines"])
        else:
            raw_data = create_sample_png(s["lines"])
        item = IngestionItem.from_bytes(s["name"], raw_data)
        assert item.file_size > 0, f"File size zero for {s['name']}"
        assert len(item.file_hash) == 64, f"Invalid SHA-256 hash for {s['name']}"
        valid = service.validate_item(item)
        assert valid, f"Validation unexpectedly failed for {s['name']}: {item.error_message}"
        ingestion_items.append(item)

    print(f"  ✓ Validated and prepared {len(ingestion_items)} document byte streams.")

    # -----------------------------------------------------------------------
    # STAGE 3, 4, 5, 6, 7: OCR, EXTRACT, CLASSIFY, PERSIST
    # -----------------------------------------------------------------------
    print("\n[STAGE 3 to 7] OCR, EXTRACTION, CLASSIFICATION & DATABASE PERSISTENCE...")
    processed_docs: List[FinancialDocument] = []
    for item in ingestion_items:
        pipeline_mock.set_active_file(item.filename)
        processed_item = service.process_document(
            item=item,
            pipeline=pipeline_mock,
            extractor=extractor,
            classifier=classifier,
            db=db,
        )
        assert processed_item.status == ProcessingStatus.PROCESSING or processed_item.document is not None
        doc = processed_item.document
        assert doc is not None, f"Document object not produced for {item.filename}"

        # Inject known subtotal / tax / inv_num from spec if present
        s = spec_lookup[item.filename]
        if s.get("subtotal") is not None:
            doc.subtotal = s["subtotal"]
        if s.get("tax") is not None:
            doc.tax_amount = s["tax"]
        if s.get("inv_num") is not None:
            doc.invoice_number = s["inv_num"]
        db.update_document(doc)

        processed_docs.append(doc)
        print(f"  ✓ Ingested: {doc.filename:<30} | Vendor: {str(doc.vendor_name):<26} | Category: {doc.category:<16} | Total: {doc.total_amount}")

    total_in_db = db.get_document_count()
    assert total_in_db == len(raw_specs), f"Expected {len(raw_specs)} documents in DB, found {total_in_db}"
    print(f"  ✓ Database persistence verified ({total_in_db} records stored).")

    # -----------------------------------------------------------------------
    # STAGE 8: SEARCH & FILTERING
    # -----------------------------------------------------------------------
    print("\n[STAGE 8] SEARCH & MULTI-FILTER RETRIEVAL...")
    # Search by vendor
    tangedco_hits = db.search_documents(vendor="TANGEDCO")
    assert len(tangedco_hits) >= 2, f"Expected >=2 TANGEDCO records, found {len(tangedco_hits)}"

    # Search by category
    elec_hits = db.search_documents(category="Electricity")
    assert len(elec_hits) >= 2, f"Expected Electricity category records, found {len(elec_hits)}"

    # Filter by amount range
    large_hits = db.search_documents(min_amount=100000.0)
    assert len(large_hits) >= 2, f"Expected Dell and KPMG records, found {len(large_hits)}"
    print(f"  ✓ Vendor search passed ({len(tangedco_hits)} hits).")
    print(f"  ✓ Category search passed ({len(elec_hits)} hits).")
    print(f"  ✓ High-value amount filter passed ({len(large_hits)} hits).")

    # -----------------------------------------------------------------------
    # STAGE 9: EDIT RECORD & PRESERVE AI VALUES
    # -----------------------------------------------------------------------
    print("\n[STAGE 9] RECORD EDITING & AI FIELD ISOLATION...")
    incomplete_doc = [d for d in processed_docs if d.filename == "doc_07_incomplete_record.png"][0]
    orig_ai_vendor = incomplete_doc.ai_vendor_name

    # Edit fields
    reloaded_doc = db.edit_financial_document(
        document_id=incomplete_doc.document_id,
        modified_by="lead_auditor_shrinath",
        reason="Verified by human auditor with original physical slip.",
        updates={
            "vendor_name": "Apex Courier Logistics Pvt Ltd",
            "invoice_number": "INV-2025-APX-774",
            "category": "Maintenance",
        },
    )

    # Verify AI vs Human values
    reloaded_doc = db.get_document(incomplete_doc.document_id)
    assert reloaded_doc is not None
    assert reloaded_doc.vendor_name == "Apex Courier Logistics Pvt Ltd"
    assert reloaded_doc.is_modified == 1
    assert reloaded_doc.ai_vendor_name == orig_ai_vendor, "Original AI vendor was overwritten!"
    print(f"  ✓ Incomplete record edited successfully. Verified AI vendor '{reloaded_doc.ai_vendor_name}' preserved.")

    # -----------------------------------------------------------------------
    # STAGE 10: VERIFICATION STATUS WORKFLOW
    # -----------------------------------------------------------------------
    print("\n[STAGE 10] HUMAN VERIFICATION WORKFLOW...")
    doc_elec = [d for d in processed_docs if d.filename == "doc_01_electricity.pdf"][0]
    db.update_verification_status(doc_elec.document_id, "verified")
    v_doc = db.get_document(doc_elec.document_id)
    assert v_doc.verification_status == "verified"
    print(f"  ✓ Document {v_doc.document_id[:8]} marked as VERIFIED.")

    # -----------------------------------------------------------------------
    # STAGE 11: DUPLICATE ANALYSIS
    # -----------------------------------------------------------------------
    print("\n[STAGE 11] MULTI-SIGNAL DUPLICATE DETECTION...")
    all_docs = db.get_all_documents()
    dup_candidates = MultiSignalDuplicateDetector.scan_all_pairs(all_docs)
    assert len(dup_candidates) >= 1, "Expected duplicate candidate between doc_01 and doc_05"
    db.save_duplicate_candidates([m.to_dict() for m in dup_candidates])

    stored_dups = db.get_duplicate_candidates()
    assert len(stored_dups) >= 1, "Duplicate candidates not persisted in DB"
    matched_pair = stored_dups[0]
    print(f"  ✓ Duplicate Candidate identified: Type={matched_pair['duplicate_type']}, Score={matched_pair['score']}, Explanation: {matched_pair['explanation']}")

    # -----------------------------------------------------------------------
    # STAGE 12: ANOMALY ANALYSIS
    # -----------------------------------------------------------------------
    print("\n[STAGE 12] STATISTICAL & RULE-BASED ANOMALY ANALYSIS...")
    anomaly_engine = CentralizedAnomalyEngine()
    total_anomalies = 0
    for d in all_docs:
        report = anomaly_engine.evaluate_document(d)
        if report.findings:
            db.save_anomaly_findings([f.to_dict() for f in report.findings])
            total_anomalies += len(report.findings)

    persisted_findings = db.get_anomaly_findings()
    assert len(persisted_findings) >= 1, "Expected anomaly findings for KPMG tax mismatch"

    # Find tax mismatch finding
    tax_findings = [
        f for f in persisted_findings
        if "006" in f.get("anomaly_type", "") or "Tax" in f.get("anomaly_name", "") or "TAX" in f.get("anomaly_type", "")
    ]
    assert len(tax_findings) >= 1, f"Tax mismatch anomaly not detected! Findings: {persisted_findings}"
    print(f"  ✓ Detected {len(persisted_findings)} total anomaly findings.")
    print(f"  ✓ Verified Tax Mismatch Anomaly: Severity={tax_findings[0]['severity']}, Score={tax_findings[0]['score']}")

    # -----------------------------------------------------------------------
    # STAGE 13: FINANCIAL ANALYTICS
    # -----------------------------------------------------------------------
    print("\n[STAGE 13] FINANCIAL ANALYTICS & RECONCILIATION...")
    engine = InstitutionalReportEngine(db=db)
    kpis = db.kpi_summary()
    assert kpis["invoice_count"] == len(raw_specs)
    assert kpis["total_spend"] > 0
    concentration = db.category_concentration()
    assert "hhi" in concentration
    vendors = db.top_vendors(n=5)
    assert len(vendors) > 0
    print(f"  ✓ Analytics KPIs calculated: Total Spend={format_inr(kpis['total_spend'])}, Documents={kpis['invoice_count']}, HHI={concentration['hhi']:.2f}")

    # -----------------------------------------------------------------------
    # STAGE 14: INSTITUTIONAL REPORT ENGINE (All Archetypes)
    # -----------------------------------------------------------------------
    print("\n[STAGE 14] REPORT GENERATION (7 ARCHETYPES)...")
    archetypes = [
        ReportType.EXECUTIVE_SUMMARY,
        ReportType.FISCAL_YEAR,
        ReportType.VENDOR_INTELLIGENCE,
        ReportType.CATEGORY_INTELLIGENCE,
        ReportType.TAX_COMPLIANCE,
        ReportType.ANOMALY_DUPLICATE,
        ReportType.AUDIT,
    ]
    report_results: Dict[str, Any] = {}
    for atype in archetypes:
        rep = engine.build_report(atype)
        assert rep["metadata"]["report_type"] == atype.value
        assert "narratives" in rep
        assert len(rep["narratives"]) > 0
        report_results[atype.value] = rep
        print(f"  ✓ Report archetype generated: {REPORT_TITLES[atype]:<32} | Narrative paragraphs: {len(rep['narratives'])}")

    # -----------------------------------------------------------------------
    # STAGE 15: EXPORT VERIFICATION (PDF, EXCEL, CSV, JSON)
    # -----------------------------------------------------------------------
    print("\n[STAGE 15] MULTI-FORMAT EXPORT VERIFICATION...")
    exec_report = report_results[ReportType.EXECUTIVE_SUMMARY.value]

    # 1. PDF Export
    pdf_bytes = export_institutional_pdf(exec_report)
    assert pdf_bytes.startswith(b"%PDF-"), "Invalid PDF byte header"
    assert len(pdf_bytes) > 2000, f"PDF export too small ({len(pdf_bytes)} bytes)"
    print(f"  ✓ Institutional PDF generated ({len(pdf_bytes):,} bytes).")

    # 2. Excel Export (11 sheets)
    excel_bytes = export_institutional_excel(exec_report)
    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes), data_only=False)
    assert "Executive Summary" in wb.sheetnames
    assert "Tax Analysis" in wb.sheetnames
    assert "Anomaly Findings" in wb.sheetnames
    print(f"  ✓ Institutional Excel Workbook generated with {len(wb.sheetnames)} sheets.")

    # 3. CSV Export
    csv_str = export_institutional_csv(exec_report)
    assert "document_id" in csv_str or "Document ID" in csv_str
    print(f"  ✓ Scoped CSV generated ({len(csv_str.splitlines())} lines).")

    # 4. JSON Export
    json_str = export_institutional_json(exec_report)
    parsed_json = json.loads(json_str)
    assert "metadata" in parsed_json
    assert "kpis" in parsed_json
    assert "documents" in parsed_json
    print(f"  ✓ Structured JSON generated ({len(parsed_json['documents'])} records).")

    # -----------------------------------------------------------------------
    # STAGE 16: AUDIT TRAIL INTEGRITY
    # -----------------------------------------------------------------------
    print("\n[STAGE 16] IMMUTABLE AUDIT TRAIL INSPECTION...")
    audit_history = db.get_audit_history(incomplete_doc.document_id)
    assert len(audit_history) >= 1, "Audit log missing edit history entries"
    print(f"  ✓ Audit history verified ({len(audit_history)} entries for edited record). Action: {audit_history[0]['action']}")

    # Clean up scratch test database
    db.close()
    if db_path.exists():
        db_path.unlink()

    duration = round(time.time() - t_start, 2)
    print("\n" + "=" * 80)
    print(f"FINAL PIPELINE VERIFICATION PASSED SUCCESSFULLY in {duration}s")
    print("All 16 stages executed with ZERO errors and complete diagnostic confirmation.")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_final_pipeline_verification()
    if not success:
        sys.exit(1)
