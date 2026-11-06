"""
Tax Module — Statutory Tax Computation & Dual-Regime Optimizer Engine (Phase 6).

Coordinates Assessment-Year specific tax slab calculation, Section 87A rebates,
marginal relief, deductions, cess, and side-by-side Old vs New regime optimization.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from tax_module.constants import (
    DEFAULT_ASSESSMENT_YEAR,
    FilingStatus,
    IncomeHead,
    TaxRegime,
)
from tax_module.db import TaxDatabase
from tax_module.models import (
    TaxAssessment,
    TaxCalculation,
)
from tax_module.tax_rules.ay2024_25 import (
    calculate_new_regime_ay2024_25,
    calculate_old_regime_ay2024_25,
)
from tax_module.tax_rules.ay2025_26 import (
    calculate_new_regime_ay2025_26,
)

logger = logging.getLogger(__name__)


class TaxCalculatorEngine:
    """Unified tax calculation engine with regime comparison and optimization."""

    def __init__(self, db: Optional[TaxDatabase] = None):
        self.db = db or TaxDatabase()

    def calculate_assessment_taxes(
        self,
        assessment_id: str,
    ) -> Tuple[TaxCalculation, TaxCalculation, TaxRegime, Decimal]:
        """
        Compute tax under both Old and New regimes for an assessment.
        Returns: (calc_new, calc_old, recommended_regime, savings_amount)
        """
        assessment = self.db.get_assessment(assessment_id)
        if not assessment:
            raise ValueError(f"Assessment '{assessment_id}' not found.")

        incomes = self.db.list_income_records(assessment_id)
        deductions = self.db.list_deduction_records(assessment_id)
        credits = self.db.list_tax_credit_records(assessment_id)

        # Aggregate Incomes
        gross_salary = Decimal("0.00")
        exempt_allowances = Decimal("0.00")
        other_income = Decimal("0.00")

        for inc in incomes:
            if inc.income_head == IncomeHead.SALARY:
                gross_salary += inc.gross_amount
                exempt_allowances += inc.exempt_allowances
            elif inc.income_head in (IncomeHead.OTHER_SOURCES, IncomeHead.HOUSE_PROPERTY):
                other_income += inc.net_taxable_amount

        # Aggregate Deductions
        ded_80c = Decimal("0.00")
        ded_80d = Decimal("0.00")
        ded_80tta = Decimal("0.00")
        ded_other = Decimal("0.00")

        for ded in deductions:
            code = ded.section_code.upper()
            if code == "80C":
                ded_80c += ded.eligible_amount
            elif code == "80D":
                ded_80d += ded.eligible_amount
            elif code in ("80TTA", "80TTB"):
                ded_80tta += ded.eligible_amount
            else:
                ded_other += ded.eligible_amount

        # Aggregate Prepaid Taxes (TDS, Advance Tax, Self Assessment Tax)
        prepaid_taxes = sum((c.claimed_amount for c in credits), Decimal("0.00"))

        ay = assessment.assessment_year

        # Route by Assessment Year
        if ay == "2025-26":
            calc_new = calculate_new_regime_ay2025_26(
                gross_salary=gross_salary,
                other_income=other_income,
                exempt_allowances=exempt_allowances,
                prepaid_taxes=prepaid_taxes,
            )
            calc_old = calculate_old_regime_ay2024_25(
                gross_salary=gross_salary,
                other_income=other_income,
                exempt_allowances=exempt_allowances,
                deductions_80c=ded_80c,
                deductions_80d=ded_80d,
                deductions_80tta=ded_80tta,
                other_deductions=ded_other,
                prepaid_taxes=prepaid_taxes,
            )
        else:  # Default AY 2024-25
            calc_new = calculate_new_regime_ay2024_25(
                gross_salary=gross_salary,
                other_income=other_income,
                exempt_allowances=exempt_allowances,
                prepaid_taxes=prepaid_taxes,
            )
            calc_old = calculate_old_regime_ay2024_25(
                gross_salary=gross_salary,
                other_income=other_income,
                exempt_allowances=exempt_allowances,
                deductions_80c=ded_80c,
                deductions_80d=ded_80d,
                deductions_80tta=ded_80tta,
                other_deductions=ded_other,
                prepaid_taxes=prepaid_taxes,
            )

        calc_new.assessment_id = assessment_id
        calc_old.assessment_id = assessment_id

        # Determine optimal regime
        if calc_new.total_tax_liability <= calc_old.total_tax_liability:
            recommended_regime = TaxRegime.NEW
            savings = calc_old.total_tax_liability - calc_new.total_tax_liability
        else:
            recommended_regime = TaxRegime.OLD
            savings = calc_new.total_tax_liability - calc_old.total_tax_liability

        # Save both calculations to DB
        self.db.save_tax_calculation(calc_new)
        self.db.save_tax_calculation(calc_old)

        # Transition assessment status
        self.db.update_filing_status(
            assessment_id=assessment_id,
            status=FilingStatus.TAX_CALCULATED,
            notes=f"Tax calculated for AY {ay}. Recommended: {recommended_regime.value} Regime (Tax: ₹{min(calc_new.total_tax_liability, calc_old.total_tax_liability):.2f}, Savings: ₹{savings:.2f})",
        )

        return calc_new, calc_old, recommended_regime, savings
