"""
Unit Tests for Phase 7 & 8 Official ITR-1 Data Mapping, JSON Generation & Schema Validation.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
import pytest

from tax_module.calculator import TaxCalculatorEngine
from tax_module.constants import (
    FilingStatus,
    IncomeHead,
    TaxRegime,
)
from tax_module.db import TaxDatabase
from tax_module.itr_json.generators.itr1_generator import ITR1JSONGenerator
from tax_module.itr_json.validators.itr1_validator import ITR1SchemaValidator
from tax_module.models import (
    BankAccount,
    IncomeRecord,
    TaxAssessment,
    TaxCreditRecord,
    TaxpayerProfile,
)


@pytest.fixture
def test_setup(tmp_path: Path):
    db = TaxDatabase(db_path=tmp_path / "itr_test.db")
    generator = ITR1JSONGenerator(db=db)
    calculator = TaxCalculatorEngine(db=db)
    validator = ITR1SchemaValidator(assessment_year="2024-25")
    return db, generator, calculator, validator


def test_schema_validator_catches_invalid_json(test_setup):
    _, _, _, validator = test_setup

    # Missing mandatory PersonalInfo
    invalid_dict = {
        "ITR": {
            "ITR1": {
                "CreationInfo": {"SWCreatedBy": "Test", "JSONCreatedBy": "Test", "JSONCreationDate": "2024-05-01"},
                "Form_ITR1": {"FormName": "ITR-1", "Description": "Test", "AssessmentYear": "2024", "SchemaVer": "Ver1.0"},
            }
        }
    }
    is_valid, errors = validator.validate(invalid_dict)
    assert is_valid is False
    assert len(errors) >= 1
    assert any("PersonalInfo" in err for err in errors)


def test_generate_official_itr1_json_end_to_end(test_setup):
    db, generator, calculator, _ = test_setup

    # 1. Taxpayer profile
    profile = TaxpayerProfile(
        pan="ABCPE1234F",
        first_name="Vikas",
        last_name="Khanna",
        dob="1987-03-22",
        gender="M",
        email="vikas.khanna@example.com",
        mobile="9876543210",
        address_flat_door_block="Tower 5, Apt 801",
        address_premises_building="Palm Meadows",
        address_city="Bengaluru",
        address_state="Karnataka",
        address_pincode="560066",
        bank_accounts=[
            BankAccount("123456789012", "HDFC0000001", "HDFC Bank", "SAVINGS", True)
        ]
    )
    db.save_taxpayer_profile(profile)

    # 2. Assessment
    assessment = TaxAssessment(
        taxpayer_id=profile.taxpayer_id,
        assessment_year="2024-25",
        selected_regime=TaxRegime.NEW,
    )
    db.save_assessment(assessment)

    # 3. Incomes: Salary 12,00,000, Bank Interest 35,000
    inc1 = IncomeRecord(
        assessment_id=assessment.assessment_id,
        income_head=IncomeHead.SALARY,
        gross_amount=Decimal("1200000.00"),
    )
    inc2 = IncomeRecord(
        assessment_id=assessment.assessment_id,
        income_head=IncomeHead.OTHER_SOURCES,
        gross_amount=Decimal("35000.00"),
        net_taxable_amount=Decimal("35000.00"),
    )
    db.save_income_record(inc1)
    db.save_income_record(inc2)

    # 4. Tax Credits: TDS 85,000
    cr = TaxCreditRecord(
        assessment_id=assessment.assessment_id,
        credit_type="TDS_SALARY",
        deductor_tan="BLRI12345E",
        deductor_name="Culinary Tech India Pvt Ltd",
        total_tax_deducted=Decimal("85000.00"),
        claimed_amount=Decimal("85000.00"),
    )
    db.save_tax_credit_record(cr)

    # 5. Run Tax Calculator first
    calculator.calculate_assessment_taxes(assessment.assessment_id)

    # 6. Generate Official JSON
    itr_return, errors = generator.generate_itr1_json(
        assessment_id=assessment.assessment_id,
        place_of_filing="Bengaluru",
        father_name="Ram Khanna",
    )

    assert errors == []
    assert itr_return is not None
    assert itr_return.is_valid is True

    # Parse JSON and verify key structural sections
    data = json.loads(itr_return.official_json)
    itr1 = data["ITR"]["ITR1"]
    assert itr1["Form_ITR1"]["FormName"] == "ITR-1"
    assert itr1["PersonalInfo"]["PAN"] == "ABCPE1234F"
    assert itr1["FilingStatus"]["NewTaxRegime"] == "Y"
    assert itr1["IncomeDeductions"]["GrossSalary"] == 1200000
    assert itr1["IncomeDeductions"]["IncomeOthSrc"] == 35000
    assert len(itr1["ScheduleTDS1"]) == 1
    assert itr1["ScheduleTDS1"][0]["TAN"] == "BLRI12345E"
    assert len(itr1["BankAccountDtls"]) == 1
    assert itr1["BankAccountDtls"][0]["RefundFlag"] == "Y"
    assert itr1["Verification"]["Declaration"]["AssesseeVerName"] == "Vikas Khanna"

    # Status updated in database
    a_updated = db.get_assessment(assessment.assessment_id)
    assert a_updated.filing_status == FilingStatus.JSON_GENERATED
