"""
Unit Tests for Phase 4 Cross-Document Financial Reconciliation Engine.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import pytest

from tax_module.constants import (
    FilingStatus,
    TaxDocumentCategory,
)
from tax_module.db import TaxDatabase
from tax_module.models import (
    ExtractedTaxField,
    TaxAssessment,
    TaxDocument,
    TaxpayerProfile,
)
from tax_module.reconciliation import IncomeReconciliationEngine


@pytest.fixture
def recon_db(tmp_path: Path) -> TaxDatabase:
    return TaxDatabase(db_path=tmp_path / "recon_test.db")


def test_reconciliation_matched_salary(recon_db: TaxDatabase):
    engine = IncomeReconciliationEngine(db=recon_db)

    p = TaxpayerProfile(pan="ABCPE1111A")
    recon_db.save_taxpayer_profile(p)
    a = TaxAssessment(taxpayer_id=p.taxpayer_id, assessment_year="2024-25")
    recon_db.save_assessment(a)

    doc1 = TaxDocument(assessment_id=a.assessment_id, document_category=TaxDocumentCategory.FORM_16_B)
    recon_db.save_tax_document(doc1)
    doc2 = TaxDocument(assessment_id=a.assessment_id, document_category=TaxDocumentCategory.AIS_TIS)
    recon_db.save_tax_document(doc2)

    # Identical salary in Form 16 and AIS
    f1 = ExtractedTaxField(
        doc_id=doc1.doc_id,
        assessment_id=a.assessment_id,
        field_name="gross_salary_17_1",
        normalized_decimal=Decimal("1234567.89"),
    )
    f2 = ExtractedTaxField(
        doc_id=doc2.doc_id,
        assessment_id=a.assessment_id,
        field_name="ais_salary_income",
        normalized_decimal=Decimal("1234567.89"),
    )
    recon_db.save_extracted_field(f1)
    recon_db.save_extracted_field(f2)

    records = engine.reconcile_assessment(a.assessment_id)
    assert len(records) == 1
    assert records[0].field_name == "Gross Salary"
    assert records[0].status == "MATCHED"
    assert records[0].difference == Decimal("0.00")


def test_reconciliation_salary_discrepancy(recon_db: TaxDatabase):
    engine = IncomeReconciliationEngine(db=recon_db)

    p = TaxpayerProfile(pan="ABCPE2222B")
    recon_db.save_taxpayer_profile(p)
    a = TaxAssessment(taxpayer_id=p.taxpayer_id, assessment_year="2024-25")
    recon_db.save_assessment(a)

    doc1 = TaxDocument(assessment_id=a.assessment_id, document_category=TaxDocumentCategory.FORM_16_B)
    recon_db.save_tax_document(doc1)
    doc2 = TaxDocument(assessment_id=a.assessment_id, document_category=TaxDocumentCategory.AIS_TIS)
    recon_db.save_tax_document(doc2)

    # Discrepancy: Form 16 says 12,00,000; AIS says 12,50,000
    f1 = ExtractedTaxField(
        doc_id=doc1.doc_id,
        assessment_id=a.assessment_id,
        field_name="gross_salary_17_1",
        normalized_decimal=Decimal("1200000.00"),
    )
    f2 = ExtractedTaxField(
        doc_id=doc2.doc_id,
        assessment_id=a.assessment_id,
        field_name="ais_salary_income",
        normalized_decimal=Decimal("1250000.00"),
    )
    recon_db.save_extracted_field(f1)
    recon_db.save_extracted_field(f2)

    records = engine.reconcile_assessment(a.assessment_id)
    assert len(records) == 1
    assert records[0].status == "REVIEW_REQUIRED"
    assert records[0].difference == Decimal("50000.00")

    # Assessment status transitioned to RECONCILIATION_REQUIRED
    updated_a = recon_db.get_assessment(a.assessment_id)
    assert updated_a.filing_status == FilingStatus.RECONCILIATION_REQUIRED

    # Resolve discrepancy
    ok = engine.resolve_reconciliation_record(
        recon_id=records[0].recon_id,
        chosen_source="Form 16 Part B",
        notes="Employer confirmed AIS includes reimbursement not taxable.",
    )
    assert ok is True


def test_reconciliation_double_counting_warning(recon_db: TaxDatabase):
    engine = IncomeReconciliationEngine(db=recon_db)

    p = TaxpayerProfile(pan="ABCPE3333C")
    recon_db.save_taxpayer_profile(p)
    a = TaxAssessment(taxpayer_id=p.taxpayer_id, assessment_year="2024-25")
    recon_db.save_assessment(a)

    # Ingest both Form 16 and a Salary Slip
    doc_f16 = TaxDocument(assessment_id=a.assessment_id, document_category=TaxDocumentCategory.FORM_16_B)
    doc_slip = TaxDocument(assessment_id=a.assessment_id, document_category=TaxDocumentCategory.SALARY_SLIP)
    recon_db.save_tax_document(doc_f16)
    recon_db.save_tax_document(doc_slip)

    f1 = ExtractedTaxField(
        doc_id=doc_f16.doc_id,
        assessment_id=a.assessment_id,
        field_name="gross_salary_17_1",
        normalized_decimal=Decimal("850000.00"),
    )
    recon_db.save_extracted_field(f1)

    records = engine.reconcile_assessment(a.assessment_id)
    overlap_record = next((r for r in records if r.field_name == "Salary Overlap Warning"), None)
    assert overlap_record is not None
    assert overlap_record.status == "REVIEW_REQUIRED"
    assert "double counting" in overlap_record.resolution_notes.lower()
