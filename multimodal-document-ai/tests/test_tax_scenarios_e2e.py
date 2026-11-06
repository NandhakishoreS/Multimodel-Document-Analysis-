"""
Phase 12: End-to-End Synthetic Taxpayer Scenarios (Scenarios A through G).
Mandated by Part 13.6 of the Master Project Expansion Specification.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
import pytest

from tax_module.calculator import TaxCalculatorEngine
from tax_module.constants import (
    FilingStatus,
    FormType,
    IncomeHead,
    ResidentialStatus,
    TaxDocumentCategory,
    TaxRegime,
    VerificationStatus,
)
from tax_module.db import TaxDatabase
from tax_module.eligibility import ITR1EligibilityEngine
from tax_module.ingestion import TaxDocumentIngestionService
from tax_module.itr_json.generators import ITR1JSONGenerator
from tax_module.models import (
    BankAccount,
    DeductionRecord,
    ExtractedTaxField,
    IncomeRecord,
    TaxAssessment,
    TaxCreditRecord,
    TaxDocument,
    TaxpayerProfile,
)
from tax_module.onboarding import TaxpayerOnboardingService
from tax_module.reconciliation import IncomeReconciliationEngine


@pytest.fixture
def system(tmp_path: Path):
    db = TaxDatabase(db_path=tmp_path / "scenarios.db")
    return {
        "db": db,
        "onboarding": TaxpayerOnboardingService(db=db),
        "ingestion": TaxDocumentIngestionService(db=db),
        "recon": IncomeReconciliationEngine(db=db),
        "calc": TaxCalculatorEngine(db=db),
        "eligibility": ITR1EligibilityEngine(db=db),
        "generator": ITR1JSONGenerator(db=db),
    }


def test_scenario_a_eligible_salary_and_interest(system):
    """Scenario A: Eligible salaried individual with salary and bank interest."""
    db, onboarding, calc, eligibility, generator = (
        system["db"], system["onboarding"], system["calc"], system["eligibility"], system["generator"]
    )

    profile, _ = onboarding.onboard_taxpayer({
        "pan": "ABCPE1001A",
        "first_name": "Ramesh",
        "last_name": "Patel",
        "dob": "1990-01-01",
        "residential_status": ResidentialStatus.RESIDENT.value,
        "address_city": "Ahmedabad",
        "address_pincode": "380001",
    })
    onboarding.add_bank_account(profile.taxpayer_id, "1234567890", "SBIN0001234", "SBI", is_refund_account=True)
    assessment, _ = onboarding.initialize_assessment(profile.taxpayer_id, "2024-25")

    # Incomes: Salary 8,50,000, Bank interest 15,000
    inc1 = IncomeRecord(assessment_id=assessment.assessment_id, income_head=IncomeHead.SALARY, gross_amount=Decimal("850000.00"))
    inc2 = IncomeRecord(assessment_id=assessment.assessment_id, income_head=IncomeHead.OTHER_SOURCES, gross_amount=Decimal("15000.00"), net_taxable_amount=Decimal("15000.00"))
    db.save_income_record(inc1)
    db.save_income_record(inc2)

    # Eligibility check
    elig = eligibility.evaluate(profile.taxpayer_id, assessment.assessment_id)
    assert elig.eligible is True

    # Calculation
    c_new, c_old, rec_regime, _ = calc.calculate_assessment_taxes(assessment.assessment_id)
    assert c_new.total_taxable_income == Decimal("815000.00") # 8.5L - 50k SD + 15k

    # JSON generation
    ret, errs = generator.generate_itr1_json(assessment.assessment_id)
    assert errs == []
    assert ret.is_valid is True


def test_scenario_b_eligible_salary_interest_and_deductions(system):
    """Scenario B: Eligible salaried individual with salary, bank interest, and supported deductions."""
    db, onboarding, calc, eligibility, generator = (
        system["db"], system["onboarding"], system["calc"], system["eligibility"], system["generator"]
    )

    profile, _ = onboarding.onboard_taxpayer({
        "pan": "ABCPE1002B",
        "first_name": "Sneha",
        "last_name": "Reddy",
        "dob": "1988-07-20",
        "residential_status": ResidentialStatus.RESIDENT.value,
        "address_city": "Hyderabad",
        "address_pincode": "500001",
    })
    onboarding.add_bank_account(profile.taxpayer_id, "9876543210", "HDFC0001234", "HDFC", is_refund_account=True)
    assessment, _ = onboarding.initialize_assessment(profile.taxpayer_id, "2024-25", selected_regime=TaxRegime.OLD)

    # Incomes
    db.save_income_record(IncomeRecord(assessment_id=assessment.assessment_id, income_head=IncomeHead.SALARY, gross_amount=Decimal("1200000.00")))
    db.save_income_record(IncomeRecord(assessment_id=assessment.assessment_id, income_head=IncomeHead.OTHER_SOURCES, gross_amount=Decimal("25000.00"), net_taxable_amount=Decimal("25000.00")))

    # Deductions: 80C 1.5L, 80D 25k, 80TTA 10k
    db.save_deduction_record(DeductionRecord(assessment_id=assessment.assessment_id, section_code="80C", eligible_amount=Decimal("150000.00")))
    db.save_deduction_record(DeductionRecord(assessment_id=assessment.assessment_id, section_code="80D", eligible_amount=Decimal("25000.00")))
    db.save_deduction_record(DeductionRecord(assessment_id=assessment.assessment_id, section_code="80TTA", eligible_amount=Decimal("10000.00")))

    # TDS
    db.save_tax_credit_record(TaxCreditRecord(assessment_id=assessment.assessment_id, claimed_amount=Decimal("80000.00")))

    elig = eligibility.evaluate(profile.taxpayer_id, assessment.assessment_id)
    assert elig.eligible is True

    calc_new, calc_old, _, _ = calc.calculate_assessment_taxes(assessment.assessment_id)
    assert calc_old.total_deductions == Decimal("185000.00")

    ret, errs = generator.generate_itr1_json(assessment.assessment_id)
    assert errs == []
    assert ret.is_valid is True


def test_scenario_c_salaried_with_missing_form16(system):
    """Scenario C: Salaried individual with missing Form 16, falling back to manual entry/slips."""
    db, onboarding, calc = system["db"], system["onboarding"], system["calc"]

    profile, _ = onboarding.onboard_taxpayer({
        "pan": "ABCPE1003C",
        "first_name": "Manoj",
        "last_name": "Bajpayee",
        "dob": "1975-03-10",
        "residential_status": ResidentialStatus.RESIDENT.value,
        "address_city": "Delhi",
        "address_pincode": "110001",
    })
    assessment, _ = onboarding.initialize_assessment(profile.taxpayer_id, "2024-25")

    # No Form 16 uploaded, but salary entered from monthly slips
    inc = IncomeRecord(
        assessment_id=assessment.assessment_id,
        income_head=IncomeHead.SALARY,
        source_name="Startup Co (Slips Only)",
        gross_amount=Decimal("600000.00"),
    )
    db.save_income_record(inc)

    c_new, _, _, _ = calc.calculate_assessment_taxes(assessment.assessment_id)
    # Taxable income = 6L - 50k SD = 5.5L
    assert c_new.total_taxable_income == Decimal("550000.00")
    # Under New Regime, 5.5L <= 7L, full Section 87A rebate wipes tax liability
    assert c_new.total_tax_liability == Decimal("0.00")


def test_scenario_d_fails_itr1_eligibility(system):
    """Scenario D: Individual who fails ITR-1 eligibility due to Capital Gains and >50L income."""
    onboarding, eligibility = system["onboarding"], system["eligibility"]

    profile, _ = onboarding.onboard_taxpayer({
        "pan": "ABCPE1004D",
        "first_name": "Vikram",
        "last_name": "Aditya",
        "dob": "1980-06-15",
        "residential_status": ResidentialStatus.RESIDENT.value,
        "address_city": "Jaipur",
        "address_pincode": "302001",
    })
    assessment, _ = onboarding.initialize_assessment(profile.taxpayer_id, "2024-25")

    elig = eligibility.evaluate(
        taxpayer_id=profile.taxpayer_id,
        assessment_id=assessment.assessment_id,
        extra_declarations={
            "has_capital_gains": True,
            "agricultural_income": "15000.00",
        },
    )
    assert elig.eligible is False
    assert elig.recommended_form == FormType.ITR_2
    assert any("Capital Gains" in f for f in elig.failed_conditions)
    assert any("5,000" in f for f in elig.failed_conditions)


def test_scenario_e_conflicting_salary_and_tds_values(system):
    """Scenario E: Conflicting salary and TDS values across documents trigger reconciliation flag."""
    db, onboarding, recon = system["db"], system["onboarding"], system["recon"]

    profile, _ = onboarding.onboard_taxpayer({
        "pan": "ABCPE1005E",
        "first_name": "Karan",
        "last_name": "Johar",
        "dob": "1985-05-25",
        "residential_status": ResidentialStatus.RESIDENT.value,
        "address_city": "Mumbai",
        "address_pincode": "400050",
    })
    assessment, _ = onboarding.initialize_assessment(profile.taxpayer_id, "2024-25")

    doc1 = TaxDocument(assessment_id=assessment.assessment_id, document_category=TaxDocumentCategory.FORM_16_B)
    doc2 = TaxDocument(assessment_id=assessment.assessment_id, document_category=TaxDocumentCategory.AIS_TIS)
    db.save_tax_document(doc1)
    db.save_tax_document(doc2)

    db.save_extracted_field(ExtractedTaxField(
        doc_id=doc1.doc_id, assessment_id=assessment.assessment_id, field_name="gross_salary_17_1", normalized_decimal=Decimal("1500000.00")
    ))
    db.save_extracted_field(ExtractedTaxField(
        doc_id=doc2.doc_id, assessment_id=assessment.assessment_id, field_name="ais_salary_income", normalized_decimal=Decimal("1550000.00")
    ))

    records = recon.reconcile_assessment(assessment.assessment_id)
    assert len(records) >= 1
    assert any(r.status == "REVIEW_REQUIRED" for r in records)


def test_scenario_f_low_confidence_ocr_requiring_correction(system):
    """Scenario F: Low-confidence OCR extraction requiring user correction with Decimal precision."""
    db, onboarding = system["db"], system["onboarding"]

    profile, _ = onboarding.onboard_taxpayer({
        "pan": "ABCPE1006F",
        "first_name": "Siddharth",
        "last_name": "Malhotra",
        "dob": "1989-02-14",
    })
    assessment, _ = onboarding.initialize_assessment(profile.taxpayer_id, "2024-25")
    doc = TaxDocument(assessment_id=assessment.assessment_id, document_category=TaxDocumentCategory.OTHER)
    db.save_tax_document(doc)

    field = ExtractedTaxField(
        doc_id=doc.doc_id,
        assessment_id=assessment.assessment_id,
        field_name="total_amount",
        original_ocr_text="258.75",
        normalized_decimal=Decimal("258.75"),
        confidence=0.45, # Low confidence
    )
    db.save_extracted_field(field)

    # User corrects value to 258.00
    db.update_field_correction(field.field_id, Decimal("258.00"), notes="Verified from original physical voucher")
    fields = db.list_extracted_fields_for_doc(doc.doc_id)
    f_updated = next(f for f in fields if f.field_id == field.field_id)
    assert f_updated.effective_value == Decimal("258.00")
    assert f_updated.verification_status == VerificationStatus.USER_CORRECTED


def test_scenario_g_complete_return_with_schema_validation(system):
    """Scenario G: Complete return with valid JSON generation matching official ITD schema."""
    db, onboarding, calc, generator = system["db"], system["onboarding"], system["calc"], system["generator"]

    profile, _ = onboarding.onboard_taxpayer({
        "pan": "ABCPE1007G",
        "first_name": "Smriti",
        "last_name": "Irani",
        "dob": "1976-03-23",
        "residential_status": ResidentialStatus.RESIDENT.value,
        "address_city": "New Delhi",
        "address_state": "Delhi",
        "address_pincode": "110001",
    })
    onboarding.add_bank_account(profile.taxpayer_id, "501001234567", "ICIC0000001", "ICICI Bank", is_refund_account=True)
    assessment, _ = onboarding.initialize_assessment(profile.taxpayer_id, "2024-25")

    # Salary 18,00,000, TDS 1,50,000
    db.save_income_record(IncomeRecord(assessment_id=assessment.assessment_id, income_head=IncomeHead.SALARY, gross_amount=Decimal("1800000.00")))
    db.save_tax_credit_record(TaxCreditRecord(assessment_id=assessment.assessment_id, credit_type="TDS_SALARY", deductor_tan="DELI00123A", claimed_amount=Decimal("150000.00")))

    calc.calculate_assessment_taxes(assessment.assessment_id)

    itr_return, errors = generator.generate_itr1_json(assessment.assessment_id)
    assert errors == []
    assert itr_return.is_valid is True

    # Validate output structure
    doc = json.loads(itr_return.official_json)
    assert doc["ITR"]["ITR1"]["PersonalInfo"]["PAN"] == "ABCPE1007G"
    assert doc["ITR"]["ITR1"]["IncomeDeductions"]["GrossSalary"] == 1800000
    assert doc["ITR"]["ITR1"]["ScheduleTDS1"][0]["TAN"] == "DELI00123A"
