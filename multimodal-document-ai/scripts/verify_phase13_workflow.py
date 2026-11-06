"""
Phase 13 End-to-End Acceptance Verification & Performance Benchmark Script.

Executes:
1. Realistic 35-document institutional financial dataset generation.
2. Verification of Tests 1-7 required by Phase 13 specifications:
   - Test 1: Category = Electricity
   - Test 2: Date = March 2026
   - Test 3: Vendor = TANGEDCO
   - Test 4: Amount > ₹50,000
   - Test 5: Electricity AND March 2026 AND Amount > ₹50,000
   - Test 6: Verification Status = Pending Review / Requires Review
   - Test 7: Record modification, AI value preservation, audit history verification.
3. 23-Step End-to-End Acceptance Workflow (Search -> Filter -> Sort -> Paginate -> Edit -> Audit -> Export -> Restart).
4. Performance benchmarks (search time, filtered query time, paginated query time, document update time across 1,000 records).
"""

from __future__ import annotations

import json
import sys
import time
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from phase11.schema import FinancialDocument, LineItem
from phase11.database import FinancialDocumentDB
from phase11.export import DocumentExporter


def generate_realistic_dataset() -> list[FinancialDocument]:
    """Generate 35 realistic institutional financial records across diverse categories and vendors."""
    vendors = [
        ("TANGEDCO Electricity Board", "Electricity", "Utilities", 45000.0, 85000.0),
        ("Adani Electricity Mumbai", "Electricity", "Utilities", 35000.0, 60000.0),
        ("Chennai Metro Water Board", "Water", "Utilities", 8000.0, 22000.0),
        ("Delhi Jal Board", "Water", "Utilities", 12000.0, 28000.0),
        ("Bharti Airtel Enterprise", "Internet/Telecom", "Utilities", 18000.0, 35000.0),
        ("Tata Communications Leased Line", "Internet/Telecom", "Utilities", 40000.0, 75000.0),
        ("Indraprastha Gas Limited", "Gas", "Utilities", 15000.0, 30000.0),
        ("Staples Office Solutions", "Office Supplies", "Operations", 4500.0, 18000.0),
        ("Kores Stationery Mart", "Office Supplies", "Operations", 3200.0, 12500.0),
        ("Dell Computers & Peripherals", "Equipment", "Operations", 65000.0, 185000.0),
        ("Voltas HVAC Maintenance Corp", "Maintenance", "Operations", 25000.0, 55000.0),
        ("Otis Elevator AMC Services", "Maintenance", "Operations", 48000.0, 92000.0),
        ("SRS Logistics Transport", "Transportation", "Operations", 38000.0, 82000.0),
        ("MakeMyTrip Corporate Travel", "Travel", "Administrative", 18000.0, 48000.0),
        ("Taj Coromandel Conference", "Accommodation", "Administrative", 52000.0, 110000.0),
        ("Coursera Institutional Licenses", "Training", "Administrative", 60000.0, 150000.0),
        ("KPMG Audit & Advisory", "Professional Services", "Administrative", 120000.0, 350000.0),
        ("Annapoorna Corporate Catering", "Food/Catering", "Other", 22000.0, 45000.0),
        ("City Hardware Supplies", "Miscellaneous", "Other", 2500.0, 8500.0),
    ]

    docs = []
    doc_idx = 1
    dates = [
        "2026-02-10", "2026-02-18", "2026-02-25",
        "2026-03-02", "2026-03-08", "2026-03-12", "2026-03-15", "2026-03-22", "2026-03-28", "2026-03-31",
        "2026-04-03", "2026-04-10", "2026-04-15",
    ]

    for i in range(35):
        vendor_info = vendors[i % len(vendors)]
        v_name, cat, sub_cat, min_p, max_p = vendor_info
        inv_date = dates[i % len(dates)]
        total = round(min_p + (i * 1234.5) % (max_p - min_p), 2)
        subtotal = round(total / 1.18, 2)
        tax = round(total - subtotal, 2)
        status = "verified" if i % 3 == 0 else ("flagged" if i % 11 == 0 else "pending")
        req_review = True if (i % 7 == 0 or cat == "Unknown") else False

        doc = FinancialDocument(
            document_id=f"DOC-INST-{doc_idx:03d}",
            filename=f"bill_{doc_idx:03d}_{cat.lower().replace('/', '_')}.pdf",
            vendor_name=v_name,
            invoice_number=f"INV-2026-{doc_idx:04d}",
            invoice_date=inv_date,
            total_amount=total,
            subtotal=subtotal,
            tax_amount=tax,
            category=cat,
            sub_category=sub_cat,
            classification_confidence=0.82 + (i % 15) * 0.01,
            verification_status=status,
            requires_review=req_review,
            benchmark_domain="SROIE" if i % 2 == 0 else "CORD",
            file_hash=f"hash_{doc_idx:04d}_sha256_mock",
            processing_duration=0.35 + (i % 5) * 0.08,
            line_items=[
                LineItem(description=f"Item 1 for {cat}", quantity=1.0, unit_price=subtotal, total=subtotal),
            ],
        )
        docs.append(doc)
        doc_idx += 1

    return docs


def run_realistic_search_tests(db: FinancialDocumentDB) -> dict[str, bool]:
    """Execute Tests 1 through 7 specified in Section 27."""
    results = {}

    # Test 1: Category = Electricity
    t1 = db.search_documents(category="Electricity")
    assert len(t1) >= 3, f"Expected >= 3 electricity docs, got {len(t1)}"
    assert all(d.category == "Electricity" for d in t1)
    results["Test 1: Category = Electricity"] = True

    # Test 2: Date = March 2026
    t2 = db.search_documents(month="2026-03")
    assert len(t2) >= 15, f"Expected >= 15 March 2026 docs, got {len(t2)}"
    assert all(d.invoice_date.startswith("2026-03") for d in t2)
    results["Test 2: Date = March 2026"] = True

    # Test 3: Vendor = TANGEDCO
    t3 = db.search_documents(vendor="TANGEDCO")
    assert len(t3) >= 2, f"Expected >= 2 TANGEDCO docs, got {len(t3)}"
    assert all("TANGEDCO" in d.vendor_name for d in t3)
    results["Test 3: Vendor = TANGEDCO"] = True

    # Test 4: Amount > ₹50,000
    t4 = db.search_documents(min_amount=50000.0)
    assert len(t4) >= 5, f"Expected >= 5 docs with amount > 50,000, got {len(t4)}"
    assert all(d.total_amount >= 50000.0 for d in t4)
    results["Test 4: Amount > ₹50,000"] = True

    # Test 5: Electricity AND March 2026 AND Amount > ₹50,000
    t5 = db.search_documents(category="Electricity", month="2026-03", min_amount=50000.0)
    assert len(t5) >= 1, f"Expected >= 1 combined doc, got {len(t5)}"
    for d in t5:
        assert d.category == "Electricity"
        assert d.invoice_date.startswith("2026-03")
        assert d.total_amount >= 50000.0
    results["Test 5: Electricity AND March 2026 AND Amount > ₹50,000"] = True

    # Test 6: Pending Review / Requires Review
    t6 = db.search_documents(requires_review=True)
    assert len(t6) >= 4, f"Expected >= 4 review docs, got {len(t6)}"
    assert all(d.requires_review is True for d in t6)
    results["Test 6: Review Queue Retrieval"] = True

    # Test 7: Open a record and modify Category, Total, Invoice Date
    target_id = t5[0].document_id
    orig_doc = db.get_document(target_id)
    orig_cat = orig_doc.category
    orig_total = orig_doc.total_amount
    orig_date = orig_doc.invoice_date

    updated = db.edit_financial_document(
        document_id=target_id,
        updates={
            "category": "Maintenance",
            "total_amount": 77777.0,
            "invoice_date": "2026-03-29",
        },
        modified_by="institutional_senior_auditor",
        reason="Category reclassification & tariff adjustment",
    )

    # Verification:
    assert updated.category == "Maintenance"
    assert updated.final_category == "Maintenance"
    assert updated.total_amount == 77777.0
    assert updated.invoice_date == "2026-03-29"
    assert updated.is_modified is True

    # Verify AI baseline remained intact
    assert updated.ai_category == orig_cat
    assert updated.ai_total_amount == orig_total
    assert updated.ai_invoice_date == orig_date

    # Verify audit history
    history = db.get_audit_history(target_id)
    field_events = [h for h in history if h["action"] == "FIELD_UPDATED"]
    assert len(field_events) == 3
    changed_fields = {e["details"]["field"] for e in field_events}
    assert changed_fields == {"category", "total_amount", "invoice_date"}
    results["Test 7: Record modification, AI value preservation, audit history"] = True

    return results


def run_e2e_23_step_workflow(db_path: Path) -> list[str]:
    """Execute the exact 23-step workflow specified in Phase 13 requirements."""
    log = []

    # Step 1: Upload / process institutional bills
    db = FinancialDocumentDB(db_path=db_path)
    dataset = generate_realistic_dataset()
    for doc in dataset:
        db.upsert_document(doc)
    log.append(f"Step 1: Upload/process institutional bills ({len(dataset)} documents persisted).")

    # Step 2: Open Document Registry (verify initial document count)
    total_count = db.count_search_documents()
    assert total_count == len(dataset)
    log.append(f"Step 2: Open Document Registry (total records = {total_count}).")

    # Step 3: Search for a vendor
    v_results = db.search_documents(vendor="Airtel")
    assert len(v_results) >= 1
    log.append(f"Step 3: Search for a vendor ('Airtel' -> {len(v_results)} records found).")

    # Step 4: Filter by bill category
    cat_results = db.search_documents(category="Electricity")
    assert len(cat_results) >= 2
    log.append(f"Step 4: Filter by bill category ('Electricity' -> {len(cat_results)} records).")

    # Step 5: Filter by date range
    dr_results = db.search_documents(date_from="2026-03-01", date_to="2026-03-31")
    assert len(dr_results) >= 10
    log.append(f"Step 5: Filter by date range ('2026-03-01' to '2026-03-31' -> {len(dr_results)} records).")

    # Step 6: Filter by amount
    amt_results = db.search_documents(min_amount=50000.0)
    assert len(amt_results) >= 5
    log.append(f"Step 6: Filter by amount (min ₹50,000 -> {len(amt_results)} records).")

    # Step 7: Combine multiple filters
    combo_results = db.search_documents(category="Electricity", month="2026-03", min_amount=40000.0)
    assert len(combo_results) >= 1
    log.append(f"Step 7: Combine multiple filters (Electricity + Mar 2026 + >₹40k -> {len(combo_results)} records).")

    # Step 8: Sort results
    sorted_desc = db.search_documents(sort_by="total_amount", sort_order="desc")
    assert sorted_desc[0].total_amount >= sorted_desc[-1].total_amount
    log.append(f"Step 8: Sort results (by total_amount DESC -> Highest: ₹{sorted_desc[0].total_amount:,.2f}).")

    # Step 9: Navigate through pagination
    page1 = db.search_documents(limit=10, offset=0)
    page2 = db.search_documents(limit=10, offset=10)
    assert len(page1) == 10 and len(page2) == 10
    assert page1[0].document_id != page2[0].document_id
    log.append("Step 9: Navigate through pagination (Page 1: 1-10, Page 2: 11-20 cleanly separated).")

    # Step 10: Open a document
    sel_doc = page1[0]
    log.append(f"Step 10: Open a document ({sel_doc.document_id} — {sel_doc.vendor_name}).")

    # Step 11: Inspect AI extraction
    ai_cat = sel_doc.ai_category or sel_doc.category
    ai_amt = sel_doc.ai_total_amount or sel_doc.total_amount
    log.append(f"Step 11: Inspect AI extraction (AI Category: {ai_cat}, AI Amount: ₹{ai_amt:,.2f}).")

    # Step 12: Edit financial fields
    new_amt = round(sel_doc.total_amount + 2500.0, 2)
    updates = {
        "total_amount": new_amt,
        "category": "Maintenance",
        "invoice_date": "2026-03-31",
    }
    log.append(f"Step 12: Edit financial fields (New Total: ₹{new_amt:,.2f}, Category: Maintenance).")

    # Step 13: Save changes
    edited_doc = db.edit_financial_document(
        document_id=sel_doc.document_id,
        updates=updates,
        modified_by="financial_controller",
        reason="Tariff reconciliation",
    )
    log.append("Step 13: Save changes (Atomic transaction committed).")

    # Step 14: Verify database persistence
    check_doc = db.get_document(sel_doc.document_id)
    assert check_doc.total_amount == new_amt
    assert check_doc.category == "Maintenance"
    log.append("Step 14: Verify database persistence (Retrieved persisted updated values from SQLite).")

    # Step 15: Open change history
    history = db.get_audit_history(sel_doc.document_id)
    assert len(history) >= 2
    log.append(f"Step 15: Open change history ({len(history)} total events retrieved).")

    # Step 16: Confirm old/new values
    field_diffs = [h["details"] for h in history if h["action"] == "FIELD_UPDATED"]
    tot_diff = next(d for d in field_diffs if d["field"] == "total_amount")
    assert tot_diff["old_value"] == sel_doc.total_amount
    assert tot_diff["new_value"] == new_amt
    log.append(f"Step 16: Confirm old/new values (Old: ₹{tot_diff['old_value']:,.2f} -> New: ₹{tot_diff['new_value']:,.2f}).")

    # Step 17: Confirm AI values remain preserved
    assert check_doc.ai_total_amount == sel_doc.total_amount
    assert check_doc.ai_category == sel_doc.category
    assert check_doc.is_modified is True
    log.append(f"Step 17: Confirm AI values remain preserved (AI Baseline: ₹{check_doc.ai_total_amount:,.2f} intact).")

    # Step 18: Mark document Verified
    verified_doc = db.edit_financial_document(
        document_id=sel_doc.document_id,
        updates={"verification_status": "verified"},
        modified_by="chief_auditor",
        reason="Final institutional sign-off",
    )
    assert verified_doc.verification_status == "verified"
    log.append("Step 18: Mark document Verified (Status transitioned to VERIFIED).")

    # Step 19: Generate filtered analytics
    f_spend = db.spend_by_category(category="Maintenance")
    f_m_spend = db.monthly_spend(category="Maintenance")
    assert len(f_spend) == 1
    log.append(f"Step 19: Generate filtered analytics (Maintenance Category Spend: ₹{f_spend[0]['total_spend']:,.2f}).")

    # Step 20: Export filtered records
    exporter = DocumentExporter()
    filtered_list = db.search_documents(category="Maintenance")
    csv_payload = exporter.to_csv(filtered_list)
    json_payload = exporter.to_json(filtered_list)
    assert len(csv_payload) > 50
    assert len(json_payload) > 50
    log.append(f"Step 20: Export filtered records ({len(filtered_list)} Maintenance records exported to CSV & JSON).")

    # Step 21: Restart application (simulate process termination and fresh DB connection)
    del db
    new_db_instance = FinancialDocumentDB(db_path=db_path)
    log.append("Step 21: Restart application (Simulated app restart with brand new DB connection).")

    # Step 22: Repeat search
    post_restart_results = new_db_instance.search_documents(category="Maintenance")
    assert len(post_restart_results) == len(filtered_list)
    log.append(f"Step 22: Repeat search ({len(post_restart_results)} Maintenance records found post-restart).")

    # Step 23: Confirm edited values remain intact
    restarted_doc = new_db_instance.get_document(sel_doc.document_id)
    assert restarted_doc.total_amount == new_amt
    assert restarted_doc.category == "Maintenance"
    assert restarted_doc.verification_status == "verified"
    assert restarted_doc.ai_total_amount == sel_doc.total_amount
    log.append("Step 23: Confirm edited values remain intact (All edited, AI, and verification values intact!).")

    return log


def run_performance_benchmarks(db: FinancialDocumentDB) -> dict[str, float]:
    """Measure search, filtered query, pagination, and document update performance over 1,000 records."""
    # Scale dataset to 1,000 records
    base_docs = generate_realistic_dataset()
    for batch_i in range(30):
        for d in base_docs:
            d_copy = FinancialDocument(
                document_id=f"PERF-{batch_i}-{d.document_id}",
                filename=d.filename,
                vendor_name=d.vendor_name,
                invoice_number=f"INV-PERF-{batch_i}-{d.invoice_number}",
                invoice_date=d.invoice_date,
                total_amount=d.total_amount,
                category=d.category,
                verification_status=d.verification_status,
                requires_review=d.requires_review,
            )
            db.upsert_document(d_copy)

    total_records = db.count_search_documents()

    # 1. Keyword search time
    t0 = time.perf_counter()
    res_search = db.search_documents(query="Electricity")
    t_search = (time.perf_counter() - t0) * 1000.0

    # 2. Multi-criteria filtered query time
    t0 = time.perf_counter()
    res_filter = db.search_documents(category="Electricity", month="2026-03", min_amount=50000.0)
    t_filter = (time.perf_counter() - t0) * 1000.0

    # 3. Paginated query time (Page 10, offset 200, limit 25)
    t0 = time.perf_counter()
    res_page = db.search_documents(limit=25, offset=200, sort_by="total_amount", sort_order="desc")
    t_page = (time.perf_counter() - t0) * 1000.0

    # 4. Document update time
    test_id = res_page[0].document_id
    t0 = time.perf_counter()
    db.edit_financial_document(
        document_id=test_id,
        updates={"total_amount": 99999.0},
        modified_by="perf_benchmark",
        reason="Benchmark update",
    )
    t_update = (time.perf_counter() - t0) * 1000.0

    return {
        "total_records": total_records,
        "keyword_search_ms": round(t_search, 2),
        "multi_filter_query_ms": round(t_filter, 2),
        "paginated_query_ms": round(t_page, 2),
        "document_edit_update_ms": round(t_update, 2),
    }


def main():
    print("=================================================================")
    print("PHASE 13: END-TO-END ACCEPTANCE & PERFORMANCE VERIFICATION")
    print("=================================================================")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "phase13_acceptance.db"
        db = FinancialDocumentDB(db_path=db_file)

        # 1. Populate initial dataset
        dataset = generate_realistic_dataset()
        for d in dataset:
            db.insert_document(d)
        print(f"[OK] Generated and inserted {len(dataset)} realistic institutional documents.")

        # 2. Section 27 Realistic Search Tests
        print("\n--- Running Section 27 Realistic Search Tests ---")
        search_results = run_realistic_search_tests(db)
        for test_name, status in search_results.items():
            print(f"  [PASS] {test_name}: {status}")

        # 3. 23-Step End-to-End Workflow
        print("\n--- Running 23-Step End-to-End Institutional Workflow ---")
        workflow_log = run_e2e_23_step_workflow(db_file)
        for step in workflow_log:
            print(f"  [OK] {step}")

        # 4. Performance Benchmarks
        print("\n--- Running Performance Benchmarks (Scale to 1,000+ Records) ---")
        perf = run_performance_benchmarks(db)
        print(f"  Total Indexed Records: {perf['total_records']:,}")
        print(f"  Keyword Search Query:  {perf['keyword_search_ms']} ms")
        print(f"  Multi-Filter Query:    {perf['multi_filter_query_ms']} ms")
        print(f"  Paginated Query:       {perf['paginated_query_ms']} ms")
        print(f"  Document Edit Update:  {perf['document_edit_update_ms']} ms")

    print("\n=================================================================")
    print("ALL PHASE 13 ACCEPTANCE TESTS & 23-STEP WORKFLOW PASSED SUCCESSFULLY!")
    print("=================================================================")


if __name__ == "__main__":
    main()
