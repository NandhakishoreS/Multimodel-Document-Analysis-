"""
Unit Tests for Phase 6 Deterministic Tax Calculation & Regime Optimization Engine.
"""

from __future__ import annotations

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
from tax_module.models import (
    DeductionRecord,
    IncomeRecord,
    TaxAssessment,
    TaxCreditRecord,
    TaxpayerProfile,
)
from tax_module.tax_rules.ay2024_25 import (
    calculate_new_regime_ay2024_25,
    calculate_old_regime_ay2024_25,
)
from tax_module.tax_rules.ay2025_26 import (
    calculate_new_regime_ay2025_26,
)


def test_new_regime_ay2024_25_rebate_87a_zero_tax():
    # Gross salary = 7,50,000. After standard deduction 50,000, Taxable Income = 7,00,000
    calc = calculate_new_regime_ay2024_25(
        gross_salary=Decimal("750000.00"),
        other_income=Decimal("0.00"),
    )
    assert calc.total_taxable_income == Decimal("700000.00")
    # Slabs: 3L-6L (5% of 3L = 15k) + 6L-7L (10% of 1L = 10k) = 25,000
    assert calc.tax_on_taxable_income == Decimal("25000.00")
    # Full 87A rebate wipes out tax liability
    assert calc.rebate_87a == Decimal("25000.00")
    assert calc.tax_after_rebate == Decimal("0.00")
    assert calc.cess == Decimal("0.00")
    assert calc.total_tax_liability == Decimal("0.00")


def test_new_regime_marginal_relief_87a():
    # Taxable income = 7,10,000 (after standard deduction)
    # Tax without marginal relief would be 15k + 11k = 26,000.
    # Marginal relief caps tax at excess income over 7L (i.e. 10,000).
    calc = calculate_new_regime_ay2024_25(
        gross_salary=Decimal("760000.00"),  # 7,60,000 - 50,000 = 7,10,000
        other_income=Decimal("0.00"),
    )
    assert calc.total_taxable_income == Decimal("710000.00")
    assert calc.tax_on_taxable_income == Decimal("26000.00")
    assert calc.rebate_87a == Decimal("16000.00")
    assert calc.tax_after_rebate == Decimal("10000.00")
    assert calc.cess == Decimal("400.00")  # 4% on 10,000
    assert calc.total_tax_liability == Decimal("10400.00")


def test_old_regime_ay2024_25_deductions_and_slabs():
    # Gross salary 10,00,000. Standard deduction 50,000. 80C 1,50,000. 80D 25,000.
    # Taxable Income = 10,00,000 - 50,000 - 1,50,000 - 25,000 = 7,75,000
    calc = calculate_old_regime_ay2024_25(
        gross_salary=Decimal("1000000.00"),
        other_income=Decimal("0.00"),
        deductions_80c=Decimal("150000.00"),
        deductions_80d=Decimal("25000.00"),
    )
    assert calc.total_taxable_income == Decimal("775000.00")
    # Old Slabs on 7,75,000:
    # 0 - 2.5L: Nil
    # 2.5L - 5L: 5% of 2.5L = 12,500
    # 5L - 7.75L: 20% of 2.75L = 55,000
    # Total tax before cess = 67,500
    assert calc.tax_on_taxable_income == Decimal("67500.00")
    assert calc.rebate_87a == Decimal("0.00")  # Income > 5L
    assert calc.cess == Decimal("2700.00")      # 4% of 67,500 = 2,700
    assert calc.total_tax_liability == Decimal("70200.00")


def test_ay2025_26_revised_new_regime():
    # Gross salary 12,00,000. Standard deduction = 75,000.
    # Taxable income = 11,25,000
    calc = calculate_new_regime_ay2025_26(
        gross_salary=Decimal("1200000.00"),
        other_income=Decimal("0.00"),
    )
    assert calc.total_taxable_income == Decimal("1125000.00")
    # Slabs for AY 2025-26:
    # 0-3L: Nil
    # 3-7L: 5% of 4L = 20,000
    # 7-10L: 10% of 3L = 30,000
    # 10-11.25L: 15% of 1.25L = 18,750
    # Total = 68,750
    assert calc.tax_on_taxable_income == Decimal("68750.00")
    assert calc.cess == Decimal("2750.00")  # 4% of 68,750
    assert calc.total_tax_liability == Decimal("71500.00")


def test_tax_calculator_engine_comparison(tmp_path: Path):
    db = TaxDatabase(db_path=tmp_path / "calc_test.db")
    engine = TaxCalculatorEngine(db=db)

    p = TaxpayerProfile(pan="ABCPE4444X")
    db.save_taxpayer_profile(p)
    a = TaxAssessment(taxpayer_id=p.taxpayer_id, assessment_year="2024-25")
    db.save_assessment(a)

    # Incomes: Salary 15,00,000, Bank interest 25,000
    inc1 = IncomeRecord(assessment_id=a.assessment_id, income_head=IncomeHead.SALARY, gross_amount=Decimal("1500000.00"))
    inc2 = IncomeRecord(assessment_id=a.assessment_id, income_head=IncomeHead.OTHER_SOURCES, gross_amount=Decimal("25000.00"), net_taxable_amount=Decimal("25000.00"))
    db.save_income_record(inc1)
    db.save_income_record(inc2)

    # Deductions: 80C 1,50,000, 80D 25,000
    ded1 = DeductionRecord(assessment_id=a.assessment_id, section_code="80C", eligible_amount=Decimal("150000.00"))
    ded2 = DeductionRecord(assessment_id=a.assessment_id, section_code="80D", eligible_amount=Decimal("25000.00"))
    db.save_deduction_record(ded1)
    db.save_deduction_record(ded2)

    # Prepaid taxes: TDS 1,20,000
    credit = TaxCreditRecord(assessment_id=a.assessment_id, claimed_amount=Decimal("120000.00"))
    db.save_tax_credit_record(credit)

    calc_new, calc_old, recommended, savings = engine.calculate_assessment_taxes(a.assessment_id)

    assert calc_new is not None
    assert calc_old is not None
    assert recommended in (TaxRegime.NEW, TaxRegime.OLD)
    assert savings >= Decimal("0.00")

    # Verify status transition in DB
    updated_a = db.get_assessment(a.assessment_id)
    assert updated_a.filing_status == FilingStatus.TAX_CALCULATED
