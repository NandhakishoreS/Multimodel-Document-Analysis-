"""
Unit Tests for Phase 5 Statutory ITR-1 (SAHAJ) Eligibility Engine.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import pytest

from tax_module.constants import (
    FormType,
    IncomeHead,
    ResidentialStatus,
)
from tax_module.db import TaxDatabase
from tax_module.eligibility import ITR1EligibilityEngine
from tax_module.models import (
    IncomeRecord,
    TaxAssessment,
    TaxpayerProfile,
)


@pytest.fixture
def eligibility_engine(tmp_path: Path) -> ITR1EligibilityEngine:
    db = TaxDatabase(db_path=tmp_path / "eligibility_test.db")
    return ITR1EligibilityEngine(db=db)


def test_eligible_resident_salaried_individual(eligibility_engine: ITR1EligibilityEngine):
    db = eligibility_engine.db
    p = TaxpayerProfile(
        pan="ABCPE1234F",
        first_name="Anand",
        last_name="Venkatesh",
        residential_status=ResidentialStatus.RESIDENT,
    )
    db.save_taxpayer_profile(p)
    a = TaxAssessment(taxpayer_id=p.taxpayer_id, assessment_year="2024-25")
    db.save_assessment(a)

    inc1 = IncomeRecord(assessment_id=a.assessment_id, income_head=IncomeHead.SALARY, net_taxable_amount=Decimal("1500000.00"))
    inc2 = IncomeRecord(assessment_id=a.assessment_id, income_head=IncomeHead.OTHER_SOURCES, net_taxable_amount=Decimal("35000.00"))
    db.save_income_record(inc1)
    db.save_income_record(inc2)

    res = eligibility_engine.evaluate(p.taxpayer_id, a.assessment_id)
    assert res.eligible is True
    assert res.recommended_form == FormType.ITR_1
    assert len(res.failed_conditions) == 0
    assert res.total_income == Decimal("1535000.00")


def test_disqualified_income_exceeding_50_lakhs(eligibility_engine: ITR1EligibilityEngine):
    db = eligibility_engine.db
    p = TaxpayerProfile(
        pan="ABCPE5555Z",
        first_name="Sunita",
        last_name="Mehta",
        residential_status=ResidentialStatus.RESIDENT,
    )
    db.save_taxpayer_profile(p)
    a = TaxAssessment(taxpayer_id=p.taxpayer_id, assessment_year="2024-25")
    db.save_assessment(a)

    inc = IncomeRecord(assessment_id=a.assessment_id, income_head=IncomeHead.SALARY, net_taxable_amount=Decimal("5200000.00"))
    db.save_income_record(inc)

    res = eligibility_engine.evaluate(p.taxpayer_id, a.assessment_id)
    assert res.eligible is False
    assert res.recommended_form == FormType.ITR_2
    assert any("exceeds ₹50,00,000 threshold" in f for f in res.failed_conditions)


def test_disqualified_non_resident_status(eligibility_engine: ITR1EligibilityEngine):
    db = eligibility_engine.db
    p = TaxpayerProfile(
        pan="ABCPE7777Q",
        first_name="Rohan",
        last_name="Gupta",
        residential_status=ResidentialStatus.NON_RESIDENT,
    )
    db.save_taxpayer_profile(p)
    a = TaxAssessment(taxpayer_id=p.taxpayer_id, assessment_year="2024-25")
    db.save_assessment(a)

    inc = IncomeRecord(assessment_id=a.assessment_id, income_head=IncomeHead.SALARY, net_taxable_amount=Decimal("1000000.00"))
    db.save_income_record(inc)

    res = eligibility_engine.evaluate(p.taxpayer_id, a.assessment_id)
    assert res.eligible is False
    assert res.recommended_form == FormType.ITR_2
    assert any("NON_RESIDENT" in f for f in res.failed_conditions)


def test_disqualified_capital_gains_and_business_income(eligibility_engine: ITR1EligibilityEngine):
    db = eligibility_engine.db
    p = TaxpayerProfile(pan="ABCPE8888P", residential_status=ResidentialStatus.RESIDENT)
    db.save_taxpayer_profile(p)
    a = TaxAssessment(taxpayer_id=p.taxpayer_id, assessment_year="2024-25")
    db.save_assessment(a)

    inc_biz = IncomeRecord(assessment_id=a.assessment_id, income_head=IncomeHead.BUSINESS_PROFESSION, net_taxable_amount=Decimal("200000.00"))
    db.save_income_record(inc_biz)

    res = eligibility_engine.evaluate(p.taxpayer_id, a.assessment_id)
    assert res.eligible is False
    assert res.recommended_form == FormType.ITR_3
    assert any("Business/Professional" in f for f in res.failed_conditions)


def test_disqualified_statutory_exclusions(eligibility_engine: ITR1EligibilityEngine):
    db = eligibility_engine.db
    p = TaxpayerProfile(pan="ABCPE9999M", residential_status=ResidentialStatus.RESIDENT)
    db.save_taxpayer_profile(p)
    a = TaxAssessment(taxpayer_id=p.taxpayer_id, assessment_year="2024-25")
    db.save_assessment(a)

    declarations = {
        "house_property_count": 2,            # Multiple house properties
        "agricultural_income": "12000.00",    # Agri income > 5k
        "is_director_in_company": True,        # Company Director
        "has_foreign_assets_or_income": True, # Foreign Assets
    }
    res = eligibility_engine.evaluate(p.taxpayer_id, a.assessment_id, extra_declarations=declarations)
    assert res.eligible is False
    assert len(res.failed_conditions) >= 4
    assert any("multiple house properties" in f.lower() for f in res.failed_conditions)
    assert any("5,000" in f for f in res.failed_conditions)
    assert any("director" in f.lower() for f in res.failed_conditions)
    assert any("foreign" in f.lower() for f in res.failed_conditions)
