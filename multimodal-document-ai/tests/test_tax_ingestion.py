"""
Unit Tests for Phase 3 Tax Document Classification, Extraction, and Ingestion.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import pytest

from tax_module.classifiers import TaxDocumentClassifier
from tax_module.constants import (
    FilingStatus,
    IncomeHead,
    ResidentialStatus,
    TaxDocumentCategory,
    TaxRegime,
)
from tax_module.db import TaxDatabase
from tax_module.extractors import BankInterestExtractor, Form16Extractor
from tax_module.ingestion import TaxDocumentIngestionService
from tax_module.models import TaxAssessment, TaxpayerProfile


@pytest.fixture
def ingestion_service(tmp_path: Path) -> TaxDocumentIngestionService:
    db = TaxDatabase(db_path=tmp_path / "ingestion_test.db")
    return TaxDocumentIngestionService(db=db)


def test_tax_document_classifier():
    classifier = TaxDocumentClassifier()

    f16_text = """
    FORM NO. 16
    PART B - Certificate under section 203 of the Income-tax Act, 1961
    Details of Salary Paid and any other income and tax deducted
    1. Gross Salary: (a) Salary as per provisions contained in sec 17(1)
    2. Less: Standard deduction under section 16(ia)
    3. Total Deductions under Chapter VI-A
    """
    res = classifier.classify_text(f16_text)
    assert res.category == TaxDocumentCategory.FORM_16_B
    assert res.confidence >= 0.85

    bank_cert_text = """
    STATE BANK OF INDIA
    CERTIFICATE OF INTEREST PAID / CREDITED
    Savings Bank Account Interest for Financial Year 2023-24
    Interest credited: INR 258.75
    Tax Deducted at Source (TDS) u/s 194A: NIL
    """
    res2 = classifier.classify_text(bank_cert_text)
    assert res2.category == TaxDocumentCategory.BANK_INTEREST_CERT
    assert res2.confidence >= 0.80

    ais_text = """
    Income-tax Department Portal
    ANNUAL INFORMATION STATEMENT (AIS)
    Information Code S-001: Salary income reported by employer
    Information Code S-002: Interest from savings bank
    """
    res3 = classifier.classify_text(ais_text)
    assert res3.category == TaxDocumentCategory.AIS_TIS
    assert res3.confidence >= 0.80


def test_form16_extraction_with_decimals():
    extractor = Form16Extractor()
    synthetic_f16 = """
    FORM NO. 16 PART B
    Name and address of the Employer: Infosys Limited, Electronics City, Bengaluru
    TAN of the Deductor: BLRI12345E
    PAN of the Employee: ABCPE1234F
    Assessment Year: 2024-25
    
    1. Gross Salary
       (a) Salary as per provisions contained in sec 17(1): INR 12,34,567.89
    2. Less: Allowances exempt under section 10: Rs. 50,000.00
    3. Less: Standard deduction under section 16(ia): 50,000.00
    4. Less: Tax on employment under section 16(iii): 2,400.00
    5. Income chargeable under the head Salaries: 11,32,167.89
    
    Chapter VI-A Deductions:
    Section 80C: 1,50,000.00
    Section 80D: 25,000.00
    
    Total Taxable Income: 9,57,167.89
    Total Tax Deducted and Deposited: 85,450.00
    """
    result = extractor.extract(synthetic_f16)

    assert result.employer_tan == "BLRI12345E"
    assert result.employee_pan == "ABCPE1234F"
    assert result.assessment_year == "2024-25"
    assert result.gross_salary == Decimal("1234567.89")
    assert result.exempt_allowances == Decimal("50000.00")
    assert result.standard_deduction == Decimal("50000.00")
    assert result.professional_tax == Decimal("2400.00")
    assert result.net_salary_income == Decimal("1132167.89")
    assert result.chapter_6a_deductions.get("80C") == Decimal("150000.00")
    assert result.chapter_6a_deductions.get("80D") == Decimal("25000.00")
    assert result.total_tds_deducted == Decimal("85450.00")
    assert result.total_taxable_income == Decimal("957167.89")


def test_bank_interest_extraction_exact_precision():
    extractor = BankInterestExtractor()
    synthetic_cert = """
    HDFC Bank Ltd.
    Certificate of Interest
    Account Number: 50100234567890
    Financial Year: 2023-24
    
    Interest on Savings Deposit: 258.75
    Fixed Deposit Interest: 1,234.56
    Total Interest Credited: 1,493.31
    Tax Deducted at Source: 3,281.00
    """
    result = extractor.extract(synthetic_cert)

    assert result.bank_name == "HDFC Bank"
    assert result.account_number == "50100234567890"
    assert result.savings_interest == Decimal("258.75")
    assert result.term_deposit_interest == Decimal("1234.56")
    assert result.total_interest == Decimal("1493.31")
    assert result.tds_deducted == Decimal("3281.00")


def test_ingestion_service_end_to_end(ingestion_service: TaxDocumentIngestionService):
    # Setup taxpayer and assessment
    p = TaxpayerProfile(pan="ABCPE9999K", first_name="Deepak", last_name="Sharma")
    ingestion_service.db.save_taxpayer_profile(p)
    a = TaxAssessment(taxpayer_id=p.taxpayer_id, assessment_year="2024-25")
    ingestion_service.db.save_assessment(a)

    doc_text = """
    FORM NO. 16 PART B
    Name of Employer: Tech Mahindra Ltd
    TAN of the Deductor: PUNT01234G
    PAN of the Employee: ABCPE9999K
    Gross Salary u/s 17(1): Rs. 8,50,258.75
    Standard deduction under section 16(ia): 50,000.00
    Income chargeable under the head Salaries: 8,00,258.75
    Total Tax Deducted: 45,000.00
    """

    doc, errors = ingestion_service.ingest_document_text(
        assessment_id=a.assessment_id,
        text=doc_text,
        filename="form16.txt",
    )
    assert errors == []
    assert doc is not None
    assert doc.document_category == TaxDocumentCategory.FORM_16_B

    # Status updated from NOT_STARTED to DOCUMENTS_UPLOADED
    a_updated = ingestion_service.db.get_assessment(a.assessment_id)
    assert a_updated.filing_status == FilingStatus.DOCUMENTS_UPLOADED

    # Verify extracted fields in database
    fields = ingestion_service.db.list_extracted_fields_for_doc(doc.doc_id)
    assert len(fields) >= 3
    gross_field = next(f for f in fields if f.field_name == "gross_salary_17_1")
    assert gross_field.normalized_decimal == Decimal("850258.75")

    # Verify generated IncomeRecord
    incomes = ingestion_service.db.list_income_records(a.assessment_id)
    assert len(incomes) == 1
    assert incomes[0].income_head == IncomeHead.SALARY
    assert incomes[0].gross_amount == Decimal("850258.75")
    assert incomes[0].source_name == "Tech Mahindra Ltd"

    # Verify generated TaxCreditRecord
    credits = ingestion_service.db.list_tax_credit_records(a.assessment_id)
    assert len(credits) == 1
    assert credits[0].total_tax_deducted == Decimal("45000.00")
    assert credits[0].deductor_tan == "PUNT01234G"


def test_duplicate_document_rejection(ingestion_service: TaxDocumentIngestionService):
    p = TaxpayerProfile(pan="ZZZPA0000Z")
    ingestion_service.db.save_taxpayer_profile(p)
    a = TaxAssessment(taxpayer_id=p.taxpayer_id, assessment_year="2024-25")
    ingestion_service.db.save_assessment(a)

    text = "STATE BANK OF INDIA Interest Certificate INR 258.75"
    doc1, errors1 = ingestion_service.ingest_document_text(a.assessment_id, text, filename="cert1.txt")
    assert doc1 is not None

    # Re-upload exact identical document
    doc2, errors2 = ingestion_service.ingest_document_text(a.assessment_id, text, filename="cert1_copy.txt")
    assert doc2 is None
    assert any("Duplicate document detected" in e for e in errors2)
