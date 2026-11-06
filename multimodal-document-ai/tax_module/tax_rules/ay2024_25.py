"""
Statutory Tax Computation Rules for Assessment Year 2024-25 (Financial Year 2023-24).

Implements exact slab computations, Section 87A rebates with marginal relief,
statutory standard deductions, Chapter VI-A deductions, and 4% Health & Education Cess
under both Old and New regimes with 100% Decimal precision.
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from tax_module.constants import TaxRegime
from tax_module.models import TaxCalculation

ASSESSMENT_YEAR = "2024-25"
FINANCIAL_YEAR = "2023-24"
CESS_RATE = Decimal("0.04") # 4% Health & Education Cess


# ---------------------------------------------------------------------------
# Slabs & Statutory Constants for AY 2024-25
# ---------------------------------------------------------------------------

# New Tax Regime (Section 115BAC) Slabs: (Lower, Upper, Rate)
NEW_REGIME_SLABS_2024_25: List[Tuple[Decimal, Optional[Decimal], Decimal]] = [
    (Decimal("0.00"), Decimal("300000.00"), Decimal("0.00")),
    (Decimal("300000.00"), Decimal("600000.00"), Decimal("0.05")),
    (Decimal("600000.00"), Decimal("900000.00"), Decimal("0.10")),
    (Decimal("900000.00"), Decimal("1200000.00"), Decimal("0.15")),
    (Decimal("1200000.00"), Decimal("1500000.00"), Decimal("0.20")),
    (Decimal("1500000.00"), None, Decimal("0.30")),
]

# Old Tax Regime Slabs (General Individual < 60 years)
OLD_REGIME_SLABS_2024_25: List[Tuple[Decimal, Optional[Decimal], Decimal]] = [
    (Decimal("0.00"), Decimal("250000.00"), Decimal("0.00")),
    (Decimal("250000.00"), Decimal("500000.00"), Decimal("0.05")),
    (Decimal("500000.00"), Decimal("1000000.00"), Decimal("0.20")),
    (Decimal("1000000.00"), None, Decimal("0.30")),
]

STANDARD_DEDUCTION_2024_25 = Decimal("50000.00")
REBATE_87A_LIMIT_NEW_2024_25 = Decimal("700000.00")
REBATE_87A_MAX_OLD_2024_25 = Decimal("12500.00")
REBATE_87A_LIMIT_OLD_2024_25 = Decimal("500000.00")


def compute_tax_on_slabs(
    income: Decimal,
    slabs: List[Tuple[Decimal, Optional[Decimal], Decimal]],
) -> Tuple[Decimal, List[Dict[str, Any]]]:
    """Calculate slab-by-slab tax and return total and detailed tier breakdowns."""
    if income <= Decimal("0.00"):
        return Decimal("0.00"), []

    total_tax = Decimal("0.00")
    breakdown: List[Dict[str, Any]] = []

    for lower, upper, rate in slabs:
        if income <= lower:
            break
        taxable_in_slab = (income - lower) if upper is None else min(income - lower, upper - lower)
        tax_in_slab = (taxable_in_slab * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        total_tax += tax_in_slab
        breakdown.append({
            "tier": f"₹{lower} - {'Above' if upper is None else '₹' + str(upper)}",
            "rate": f"{int(rate * 100)}%",
            "taxable_amount": str(taxable_in_slab),
            "tax": str(tax_in_slab),
        })

    return total_tax, breakdown


def calculate_new_regime_ay2024_25(
    gross_salary: Decimal,
    other_income: Decimal,
    exempt_allowances: Decimal = Decimal("0.00"),
    prepaid_taxes: Decimal = Decimal("0.00"),
) -> TaxCalculation:
    """Compute tax under default New Regime (Section 115BAC) for AY 2024-25."""
    # Standard deduction applicable to salary up to gross salary
    sd = min(STANDARD_DEDUCTION_2024_25, gross_salary)
    net_salary = max(Decimal("0.00"), gross_salary - sd)
    gross_total_income = net_salary + other_income

    # Under New Regime, Chapter VI-A deductions are Nil (for basic ITR-1 salaried)
    total_deductions = Decimal("0.00")
    taxable_income = gross_total_income

    tax_on_slabs, slab_breakdown = compute_tax_on_slabs(taxable_income, NEW_REGIME_SLABS_2024_25)

    # Section 87A Rebate & Marginal Relief
    rebate = Decimal("0.00")
    if taxable_income <= REBATE_87A_LIMIT_NEW_2024_25:
        # Full rebate: tax liability becomes zero
        rebate = tax_on_slabs
    elif taxable_income > REBATE_87A_LIMIT_NEW_2024_25:
        # Marginal relief under Section 87A (New Regime):
        # Tax payable cannot exceed excess of income over ₹7,00,000
        excess_income = taxable_income - REBATE_87A_LIMIT_NEW_2024_25
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
        "total_deductions": str(total_deductions),
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
        total_deductions=total_deductions,
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


def calculate_old_regime_ay2024_25(
    gross_salary: Decimal,
    other_income: Decimal,
    exempt_allowances: Decimal = Decimal("0.00"),
    deductions_80c: Decimal = Decimal("0.00"),
    deductions_80d: Decimal = Decimal("0.00"),
    deductions_80tta: Decimal = Decimal("0.00"),
    other_deductions: Decimal = Decimal("0.00"),
    prepaid_taxes: Decimal = Decimal("0.00"),
) -> TaxCalculation:
    """Compute tax under Old Regime for AY 2024-25."""
    # Statutory caps
    capped_80c = min(Decimal("150000.00"), deductions_80c)
    capped_80d = min(Decimal("50000.00"), deductions_80d)
    capped_80tta = min(Decimal("10000.00"), deductions_80tta)

    sd = min(STANDARD_DEDUCTION_2024_25, gross_salary)
    net_salary = max(Decimal("0.00"), gross_salary - exempt_allowances - sd)
    gross_total_income = net_salary + other_income

    total_deductions = capped_80c + capped_80d + capped_80tta + other_deductions
    # Deductions cannot exceed Gross Total Income
    total_deductions = min(total_deductions, gross_total_income)

    taxable_income = max(Decimal("0.00"), gross_total_income - total_deductions)

    tax_on_slabs, slab_breakdown = compute_tax_on_slabs(taxable_income, OLD_REGIME_SLABS_2024_25)

    # Section 87A Rebate (Old Regime: max ₹12,500 if taxable income <= ₹5,00,000)
    rebate = Decimal("0.00")
    if taxable_income <= REBATE_87A_LIMIT_OLD_2024_25:
        rebate = min(tax_on_slabs, REBATE_87A_MAX_OLD_2024_25)

    tax_after_rebate = max(Decimal("0.00"), tax_on_slabs - rebate)
    cess = (tax_after_rebate * CESS_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    total_liability = tax_after_rebate + cess

    refund_or_payable = total_liability - prepaid_taxes

    calc_breakdown = {
        "assessment_year": ASSESSMENT_YEAR,
        "regime": "OLD",
        "gross_salary": str(gross_salary),
        "exempt_allowances": str(exempt_allowances),
        "standard_deduction": str(sd),
        "net_salary": str(net_salary),
        "other_income": str(other_income),
        "gross_total_income": str(gross_total_income),
        "deductions_80c": str(capped_80c),
        "deductions_80d": str(capped_80d),
        "deductions_80tta": str(capped_80tta),
        "total_deductions": str(total_deductions),
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
        regime=TaxRegime.OLD,
        gross_total_income=gross_total_income,
        total_deductions=total_deductions,
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
