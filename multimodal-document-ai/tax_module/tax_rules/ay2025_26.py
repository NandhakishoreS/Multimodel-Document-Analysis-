"""
Statutory Tax Computation Rules for Assessment Year 2025-26 (Financial Year 2024-25).

Implements statutory amendments under Finance (No. 2) Act 2024:
- Standard Deduction under New Regime increased to ₹75,000.
- Revised New Regime slabs (0-3L 0%, 3-7L 5%, 7-10L 10%, 10-12L 15%, 12-15L 20%, >15L 30%).
- Section 87A rebate and 4% Health & Education Cess with 100% Decimal precision.
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from tax_module.constants import TaxRegime
from tax_module.models import TaxCalculation
from tax_module.tax_rules.ay2024_25 import (
    CESS_RATE,
    OLD_REGIME_SLABS_2024_25,
    REBATE_87A_LIMIT_OLD_2024_25,
    REBATE_87A_MAX_OLD_2024_25,
    calculate_old_regime_ay2024_25,
    compute_tax_on_slabs,
)

ASSESSMENT_YEAR = "2025-26"
FINANCIAL_YEAR = "2024-25"

# Revised New Regime Slabs for AY 2025-26
NEW_REGIME_SLABS_2025_26: List[Tuple[Decimal, Optional[Decimal], Decimal]] = [
    (Decimal("0.00"), Decimal("300000.00"), Decimal("0.00")),
    (Decimal("300000.00"), Decimal("700000.00"), Decimal("0.05")),
    (Decimal("700000.00"), Decimal("1000000.00"), Decimal("0.10")),
    (Decimal("1000000.00"), Decimal("1200000.00"), Decimal("0.15")),
    (Decimal("1200000.00"), Decimal("1500000.00"), Decimal("0.20")),
    (Decimal("1500000.00"), None, Decimal("0.30")),
]

STANDARD_DEDUCTION_NEW_2025_26 = Decimal("75000.00")
REBATE_87A_LIMIT_NEW_2025_26 = Decimal("700000.00")


def calculate_new_regime_ay2025_26(
    gross_salary: Decimal,
    other_income: Decimal,
    exempt_allowances: Decimal = Decimal("0.00"),
    prepaid_taxes: Decimal = Decimal("0.00"),
) -> TaxCalculation:
    """Compute tax under amended New Regime for AY 2025-26."""
    sd = min(STANDARD_DEDUCTION_NEW_2025_26, gross_salary)
    net_salary = max(Decimal("0.00"), gross_salary - sd)
    gross_total_income = net_salary + other_income
    taxable_income = gross_total_income

    tax_on_slabs, slab_breakdown = compute_tax_on_slabs(taxable_income, NEW_REGIME_SLABS_2025_26)

    # Section 87A Rebate & Marginal Relief
    rebate = Decimal("0.00")
    if taxable_income <= REBATE_87A_LIMIT_NEW_2025_26:
        rebate = tax_on_slabs
    elif taxable_income > REBATE_87A_LIMIT_NEW_2025_26:
        excess_income = taxable_income - REBATE_87A_LIMIT_NEW_2025_26
        if tax_on_slabs > excess_income:
            rebate = tax_on_slabs - excess_income

    tax_after_rebate = max(Decimal("0.00"), tax_on_slabs - rebate)
    cess = (tax_after_rebate * CESS_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    total_liability = tax_after_rebate + cess

    refund_or_payable = total_liability - prepaid_taxes

    calc_breakdown = {
        "assessment_year": ASSESSMENT_YEAR,
        "regime": "NEW",
        "gross_salary": str(gross_salary),
        "standard_deduction": str(sd),
        "net_salary": str(net_salary),
        "other_income": str(other_income),
        "gross_total_income": str(gross_total_income),
        "total_deductions": "0.00",
        "total_taxable_income": str(taxable_income),
        "tax_before_rebate": str(tax_on_slabs),
        "rebate_87a": str(rebate),
        "tax_after_rebate": str(tax_after_rebate),
        "cess_4_pct": str(cess),
        "total_tax_liability": str(total_liability),
        "prepaid_taxes": str(prepaid_taxes),
        "refund_or_payable": str(refund_or_payable),
        "slab_breakdown": slab_breakdown,
    }

    return TaxCalculation(
        regime=TaxRegime.NEW,
        gross_total_income=gross_total_income,
        total_deductions=Decimal("0.00"),
        total_taxable_income=taxable_income,
        tax_on_taxable_income=tax_on_slabs,
        rebate_87a=rebate,
        tax_after_rebate=tax_after_rebate,
        cess=cess,
        total_tax_liability=total_liability,
        total_tax_credits=prepaid_taxes,
        refund_or_payable=refund_or_payable,
        breakdown_json=json.dumps(calc_breakdown),
    )
