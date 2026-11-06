"""
Unit Tests for Phase 1 Tax Module Domain Models & Database Persistence.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from pathlib import Path
import pytest

from tax_module.constants import (
    FilingStatus,
    FormType,
    IncomeHead,
    ResidentialStatus,
    TaxDocumentCategory,
    TaxRegime,
    VerificationStatus,
    get_ay_from_fy,
    get_fy_from_ay,
    mask_bank_account,
    mask_pan,
)
from tax_module.db import TaxDatabase
from tax_module.models import (
    BankAccount,
    DeductionRecord,
    ExtractedTaxField,
    ITRReturn,
    IncomeRecord,
    ReconciliationRecord,
    TaxAssessment,
    TaxCalculation,
    TaxCreditRecord,
    TaxDocument,
    TaxpayerProfile,
)


# ---------------------------------------------------------------------------
# Test Helpers and Constants
# ---------------------------------------------------------------------------

def test_fy_ay_helpers():
    assert get_fy_from_ay("2024-25") == "2023-24"
    assert get_fy_from_ay("2025-26") == "2024-25"
    assert get_ay_from_fy("2023-24") == "2024-25"
    assert get_ay_from_fy("2024-25") == "2025-26"

    with pytest.raises(ValueError):
        get_fy_from_ay("2024")

    with pytest.raises(ValueError):
        get_ay_from_fy("invalid")


def test_masking_helpers():
    assert mask_pan("ABCDE1234F") == "ABXXXXXX4F"
    assert mask_pan("xyz") == "XXXXXXXXXX"
    assert mask_pan(None) == ""

    assert mask_bank_account("123456789012") == "XXXXXXXX9012"
    assert mask_bank_account("123") == "123"
    assert mask_bank_account(None) == ""


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------

def test_bank_account_model():
    acc = BankAccount(
        account_number="987654321098",
        ifsc_code="HDFC0001234",
        bank_name="HDFC Bank",
        account_type="SAVINGS",
        is_refund_account=True,
    )
    assert acc.masked_account_number() == "XXXXXXXX1098"
    d = acc.to_dict()
    restored = BankAccount.from_dict(d)
    assert restored.account_number == "987654321098"
    assert restored.ifsc_code == "HDFC0001234"
    assert restored.is_refund_account is True


def test_taxpayer_profile_model():
    profile = TaxpayerProfile(
        pan="ABCDE1234F",
        first_name="Rajesh",
        middle_name="Kumar",
        last_name="Sharma",
        dob="1988-06-15",
        gender="M",
        residential_status=ResidentialStatus.RESIDENT,
        email="rajesh.sharma@example.com",
        mobile="9876543210",
        address_flat_door_block="Flat 402",
        address_premises_building="Shanti Heights",
        address_city="Mumbai",
        address_state="Maharashtra",
        address_pincode="400001",
        bank_accounts=[
            BankAccount("123456789", "SBIN0001234", "State Bank of India")
        ]
    )
    assert profile.full_name == "Rajesh Kumar Sharma"
    assert profile.validate_pan() is True
    assert profile.masked_pan == "ABXXXXXX4F"

    # Invalid PAN test
    invalid_profile = TaxpayerProfile(pan="INVALID123")
    assert invalid_profile.validate_pan() is False

    # Roundtrip serialization
    d = profile.to_dict()
    restored = TaxpayerProfile.from_dict(d)
    assert restored.pan == "ABCDE1234F"
    assert restored.full_name == "Rajesh Kumar Sharma"
    assert len(restored.bank_accounts) == 1
    assert restored.bank_accounts[0].bank_name == "State Bank of India"


def test_extracted_tax_field_decimal_fidelity():
    field = ExtractedTaxField(
        field_name="gross_salary",
        original_ocr_text="12,34,567.89",
        normalized_decimal=Decimal("1234567.89"),
        confidence=0.98,
        bounding_box=[10, 20, 100, 40],
    )
    # Check initial effective value
    assert field.effective_value == Decimal("1234567.89")

    # User correction override
    field.user_corrected_value = Decimal("1234568.00")
    field.verification_status = VerificationStatus.USER_CORRECTED
    assert field.effective_value == Decimal("1234568.00")

    # Roundtrip serialization
    d = field.to_dict()
    restored = ExtractedTaxField.from_dict(d)
    assert restored.normalized_decimal == Decimal("1234567.89")
    assert restored.user_corrected_value == Decimal("1234568.00")
    assert restored.effective_value == Decimal("1234568.00")
    assert restored.bounding_box == [10, 20, 100, 40]


def test_income_and_deduction_record_decimals():
    inc = IncomeRecord(
        income_head=IncomeHead.SALARY,
        source_name="Acme Tech India Pvt Ltd",
        gross_amount=Decimal("1500000.50"),
        exempt_allowances=Decimal("50000.00"),
        net_taxable_amount=Decimal("1450000.50"),
    )
    assert inc.gross_amount == Decimal("1500000.50")
    restored_inc = IncomeRecord.from_dict(inc.to_dict())
    assert restored_inc.net_taxable_amount == Decimal("1450000.50")

    ded = DeductionRecord(
        section_code="80C",
        description="PPF & ELSS Mutual Funds",
        claimed_amount=Decimal("180000.75"),
        eligible_amount=Decimal("150000.00"),
    )
    restored_ded = DeductionRecord.from_dict(ded.to_dict())
    assert restored_ded.claimed_amount == Decimal("180000.75")
    assert restored_ded.eligible_amount == Decimal("150000.00")


# ---------------------------------------------------------------------------
# Database Persistence & Isolation Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def test_db(tmp_path: Path) -> TaxDatabase:
    db_file = tmp_path / "test_tax.db"
    return TaxDatabase(db_path=db_file)


def test_db_taxpayer_crud(test_db: TaxDatabase):
    p = TaxpayerProfile(
        pan="ABCDE9999Z",
        first_name="Pooja",
        last_name="Verma",
        dob="1992-11-20",
        bank_accounts=[
            BankAccount("555666777888", "ICIC0000001", "ICICI Bank")
        ]
    )
    test_db.save_taxpayer_profile(p)

    retrieved = test_db.get_taxpayer_profile(p.taxpayer_id)
    assert retrieved is not None
    assert retrieved.pan == "ABCDE9999Z"
    assert retrieved.last_name == "Verma"
    assert len(retrieved.bank_accounts) == 1

    by_pan = test_db.get_taxpayer_by_pan("abcde9999z")
    assert by_pan is not None
    assert by_pan.taxpayer_id == p.taxpayer_id

    # Update profile
    p.address_city = "Bengaluru"
    test_db.save_taxpayer_profile(p)
    updated = test_db.get_taxpayer_profile(p.taxpayer_id)
    assert updated.address_city == "Bengaluru"


def test_db_assessment_lifecycle(test_db: TaxDatabase):
    p = TaxpayerProfile(pan="BKUPR1234K", first_name="Anil", last_name="Kumar")
    test_db.save_taxpayer_profile(p)

    assessment = TaxAssessment(
        taxpayer_id=p.taxpayer_id,
        assessment_year="2024-25",
        financial_year="2023-24",
        filing_status=FilingStatus.NOT_STARTED,
        selected_regime=TaxRegime.NEW,
    )
    test_db.save_assessment(assessment)

    a_fetch = test_db.get_assessment(assessment.assessment_id)
    assert a_fetch is not None
    assert a_fetch.assessment_year == "2024-25"
    assert a_fetch.filing_status == FilingStatus.NOT_STARTED

    # Status update
    test_db.update_filing_status(assessment.assessment_id, FilingStatus.DOCUMENTS_UPLOADED)
    a_updated = test_db.get_assessment(assessment.assessment_id)
    assert a_updated.filing_status == FilingStatus.DOCUMENTS_UPLOADED

    # Check audit log
    audit_events = test_db.get_audit_trail(assessment_id=assessment.assessment_id)
    assert len(audit_events) >= 1
    assert "STATUS_CHANGED_DOCUMENTS_UPLOADED" in audit_events[0].action


def test_db_document_and_extracted_fields(test_db: TaxDatabase):
    p = TaxpayerProfile(pan="AAAPA1111A", first_name="Maya")
    test_db.save_taxpayer_profile(p)
    a = TaxAssessment(taxpayer_id=p.taxpayer_id, assessment_year="2024-25")
    test_db.save_assessment(a)

    doc = TaxDocument(
        assessment_id=a.assessment_id,
        document_category=TaxDocumentCategory.FORM_16_B,
        filename="form16_part_b.pdf",
        file_hash="sha256dummyhash123",
        file_size=1048576,
        ocr_confidence=0.96,
    )
    test_db.save_tax_document(doc)

    doc_fetch = test_db.get_tax_document(doc.doc_id)
    assert doc_fetch is not None
    assert doc_fetch.document_category == TaxDocumentCategory.FORM_16_B

    f1 = ExtractedTaxField(
        doc_id=doc.doc_id,
        assessment_id=a.assessment_id,
        field_name="salary_under_section_17_1",
        original_ocr_text="8,50,000.00",
        normalized_decimal=Decimal("850000.00"),
        confidence=0.97,
    )
    test_db.save_extracted_field(f1)

    fields = test_db.list_extracted_fields_for_doc(doc.doc_id)
    assert len(fields) == 1
    assert fields[0].normalized_decimal == Decimal("850000.00")

    # Correct field
    test_db.update_field_correction(f1.field_id, Decimal("855000.00"))
    fields_corr = test_db.list_extracted_fields_for_doc(doc.doc_id)
    assert fields_corr[0].effective_value == Decimal("855000.00")
    assert fields_corr[0].verification_status == VerificationStatus.USER_CORRECTED


def test_db_full_financial_records_and_calculation(test_db: TaxDatabase):
    p = TaxpayerProfile(pan="ZZZPA9999Z")
    test_db.save_taxpayer_profile(p)
    a = TaxAssessment(taxpayer_id=p.taxpayer_id, assessment_year="2024-25")
    test_db.save_assessment(a)

    # Incomes
    inc1 = IncomeRecord(
        assessment_id=a.assessment_id,
        income_head=IncomeHead.SALARY,
        gross_amount=Decimal("1200000.00"),
        net_taxable_amount=Decimal("1150000.00"),
    )
    inc2 = IncomeRecord(
        assessment_id=a.assessment_id,
        income_head=IncomeHead.OTHER_SOURCES,
        gross_amount=Decimal("25800.75"),
        net_taxable_amount=Decimal("25800.75"),
    )
    test_db.save_income_record(inc1)
    test_db.save_income_record(inc2)

    incomes = test_db.list_income_records(a.assessment_id)
    assert len(incomes) == 2
    assert sum(i.gross_amount for i in incomes) == Decimal("1225800.75")

    # Deductions
    ded = DeductionRecord(
        assessment_id=a.assessment_id,
        section_code="80C",
        claimed_amount=Decimal("150000.00"),
        eligible_amount=Decimal("150000.00"),
    )
    test_db.save_deduction_record(ded)
    assert len(test_db.list_deduction_records(a.assessment_id)) == 1

    # Tax Credits
    credit = TaxCreditRecord(
        assessment_id=a.assessment_id,
        credit_type="TDS_SALARY",
        total_tax_deducted=Decimal("85000.00"),
        claimed_amount=Decimal("85000.00"),
    )
    test_db.save_tax_credit_record(credit)
    assert len(test_db.list_tax_credit_records(a.assessment_id)) == 1

    # Tax Calculation
    calc = TaxCalculation(
        assessment_id=a.assessment_id,
        regime=TaxRegime.NEW,
        gross_total_income=Decimal("1175800.75"),
        total_deductions=Decimal("0.00"),
        total_taxable_income=Decimal("1175800.75"),
        total_tax_liability=Decimal("88483.28"),
        total_tax_credits=Decimal("85000.00"),
        refund_or_payable=Decimal("3483.28"),
    )
    test_db.save_tax_calculation(calc)
    fetched_calc = test_db.get_latest_tax_calculation(a.assessment_id, TaxRegime.NEW)
    assert fetched_calc is not None
    assert fetched_calc.refund_or_payable == Decimal("3483.28")

    # ITR Return
    ret = ITRReturn(
        assessment_id=a.assessment_id,
        assessment_year="2024-25",
        official_json='{"ITR": {"ITR1": {}}}',
        is_valid=True,
    )
    test_db.save_itr_return(ret)
    fetched_ret = test_db.get_latest_itr_return(a.assessment_id)
    assert fetched_ret is not None
    assert fetched_ret.is_valid is True


def test_db_multi_taxpayer_isolation(test_db: TaxDatabase):
    # Taxpayer 1
    p1 = TaxpayerProfile(pan="AAAAA1111A")
    test_db.save_taxpayer_profile(p1)
    a1 = TaxAssessment(taxpayer_id=p1.taxpayer_id, assessment_year="2024-25")
    test_db.save_assessment(a1)
    inc1 = IncomeRecord(assessment_id=a1.assessment_id, gross_amount=Decimal("100000.00"))
    test_db.save_income_record(inc1)

    # Taxpayer 2
    p2 = TaxpayerProfile(pan="BBBBB2222B")
    test_db.save_taxpayer_profile(p2)
    a2 = TaxAssessment(taxpayer_id=p2.taxpayer_id, assessment_year="2024-25")
    test_db.save_assessment(a2)
    inc2 = IncomeRecord(assessment_id=a2.assessment_id, gross_amount=Decimal("200000.00"))
    test_db.save_income_record(inc2)

    # Verify complete isolation
    p1_incomes = test_db.list_income_records(a1.assessment_id)
    assert len(p1_incomes) == 1
    assert p1_incomes[0].gross_amount == Decimal("100000.00")

    p2_incomes = test_db.list_income_records(a2.assessment_id)
    assert len(p2_incomes) == 1
    assert p2_incomes[0].gross_amount == Decimal("200000.00")
