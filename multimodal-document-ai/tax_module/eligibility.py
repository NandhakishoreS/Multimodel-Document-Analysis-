"""
Tax Module — ITR-1 (SAHAJ) Statutory Eligibility Engine (Phase 5).

Evaluates whether an individual taxpayer meets all statutory criteria under
the Income Tax Rules to file Form ITR-1 (SAHAJ). Identifies disqualifications
with exact legal explanations and recommends ITR-2 / ITR-3 where appropriate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from tax_module.constants import (
    FormType,
    IncomeHead,
    ResidentialStatus,
)
from tax_module.db import TaxDatabase

logger = logging.getLogger(__name__)

# Statutory Limits for ITR-1
MAX_TOTAL_INCOME_ITR1 = Decimal("5000000.00")      # ₹50 Lakhs
MAX_AGRICULTURAL_INCOME_ITR1 = Decimal("5000.00")   # ₹5,000


@dataclass
class ITR1EligibilityResult:
    """Outcome of statutory eligibility evaluation for ITR-1 (SAHAJ)."""
    eligible: bool
    selected_form: FormType = FormType.ITR_1
    recommended_form: FormType = FormType.ITR_1
    assessment_year: str = ""
    total_income: Decimal = Decimal("0.00")
    passed_conditions: List[str] = field(default_factory=list)
    failed_conditions: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eligible": self.eligible,
            "selected_form": self.selected_form.value,
            "recommended_form": self.recommended_form.value,
            "assessment_year": self.assessment_year,
            "total_income": str(self.total_income),
            "passed_conditions": self.passed_conditions,
            "failed_conditions": self.failed_conditions,
            "explanation": self.explanation,
        }


class ITR1EligibilityEngine:
    """Statutory evaluator for ITR-1 (SAHAJ) individual eligibility."""

    def __init__(self, db: Optional[TaxDatabase] = None):
        self.db = db or TaxDatabase()

    def evaluate(
        self,
        taxpayer_id: str,
        assessment_id: str,
        extra_declarations: Optional[Dict[str, Any]] = None,
    ) -> ITR1EligibilityResult:
        """
        Evaluate full statutory criteria for ITR-1:
        1. Taxpayer is an Individual (PAN 4th char 'P').
        2. Residential status is Resident & Ordinarily Resident (ROR).
        3. Total income does not exceed ₹50,00,000 (50 Lakhs).
        4. No Business or Professional income (PGBP).
        5. No Capital Gains (Short-term or Long-term).
        6. House property: Single house property only (No multiple house properties).
        7. Agricultural income does not exceed ₹5,000.
        8. No Directorship in a company.
        9. No unlisted equity shares held.
        10. No foreign assets, foreign accounts, or foreign income.
        11. No brought-forward or carried-forward losses.
        12. No lottery, betting, gambling, or virtual digital asset (crypto) income.
        """
        profile = self.db.get_taxpayer_profile(taxpayer_id)
        assessment = self.db.get_assessment(assessment_id)
        incomes = self.db.list_income_records(assessment_id)

        if not profile or not assessment:
            return ITR1EligibilityResult(
                eligible=False,
                explanation="Taxpayer profile or assessment session not found in database.",
            )

        passed: List[str] = []
        failed: List[str] = []
        declarations = extra_declarations or {}

        # 1. Taxpayer Status: Individual check
        if profile.pan and len(profile.pan) == 10 and profile.pan[3] == "P":
            passed.append("Status: Natural Person / Individual (PAN 4th character is 'P').")
        else:
            failed.append("Status: Taxpayer is not an Individual. ITR-1 is exclusively available to natural individuals.")

        # 2. Residential Status check
        if profile.residential_status == ResidentialStatus.RESIDENT:
            passed.append("Residential Status: Resident & Ordinarily Resident (ROR).")
        else:
            failed.append(
                f"Residential Status: Taxpayer is {profile.residential_status.value}. "
                "Non-Residents and RNOR individuals are statutorily barred from filing ITR-1 (Must file ITR-2)."
            )

        # 3. Income Head & Total Income checks
        total_income = sum((inc.net_taxable_amount for inc in incomes), Decimal("0.00"))

        if total_income <= MAX_TOTAL_INCOME_ITR1:
            passed.append(f"Income Limit: Total Income ₹{total_income} is within statutory threshold of ₹50,00,000.")
        else:
            failed.append(f"Income Limit: Total Income ₹{total_income} exceeds ₹50,00,000 threshold. Must file ITR-2.")

        # 4. Prohibited Income Heads: Capital Gains, Business
        has_capital_gains = any(inc.income_head == IncomeHead.CAPITAL_GAINS for inc in incomes) or bool(declarations.get("has_capital_gains"))
        if not has_capital_gains:
            passed.append("Income Heads: No capital gains reported.")
        else:
            failed.append("Prohibited Income: Capital Gains reported. ITR-1 does not support Schedule CG (Must file ITR-2).")

        has_business = any(inc.income_head == IncomeHead.BUSINESS_PROFESSION for inc in incomes) or bool(declarations.get("has_business_income"))
        if not has_business:
            passed.append("Income Heads: No business or professional income (PGBP).")
        else:
            failed.append("Prohibited Income: Business/Professional income reported (Must file ITR-3 or ITR-4 SUGAM).")

        # 5. House Property count
        house_property_count = int(declarations.get("house_property_count", 1))
        if house_property_count <= 1:
            passed.append(f"House Property: Owning {house_property_count} house property is permitted in ITR-1.")
        else:
            failed.append(f"House Property: Owning multiple house properties ({house_property_count}) requires ITR-2 Schedule HP.")

        # 6. Agricultural Income
        agri_income = Decimal(str(declarations.get("agricultural_income", "0.00")))
        if agri_income <= MAX_AGRICULTURAL_INCOME_ITR1:
            passed.append(f"Agricultural Income: ₹{agri_income} is within ₹5,000 exempt limit.")
        else:
            failed.append(f"Agricultural Income: ₹{agri_income} exceeds ₹5,000 limit. Requires ITR-2.")

        # 7. Directorship in a company
        is_director = bool(declarations.get("is_director_in_company", False))
        if not is_director:
            passed.append("Directorship: Taxpayer was not a director in any company during the previous year.")
        else:
            failed.append("Directorship: Being a director in any Indian/foreign company requires ITR-2.")

        # 8. Unlisted equity shares
        has_unlisted_shares = bool(declarations.get("holds_unlisted_equity", False))
        if not has_unlisted_shares:
            passed.append("Unlisted Equity: Taxpayer held no unlisted equity shares.")
        else:
            failed.append("Unlisted Equity: Holding unlisted shares at any time during the year requires ITR-2.")

        # 9. Foreign Assets & Foreign Income
        has_foreign_assets = bool(declarations.get("has_foreign_assets_or_income", False))
        if not has_foreign_assets:
            passed.append("Foreign Assets: No foreign assets, bank accounts, or foreign-source income.")
        else:
            failed.append("Foreign Assets: Owning foreign assets/accounts requires Schedule FA in ITR-2.")

        # 10. Carried-forward / Brought-forward losses
        has_losses = bool(declarations.get("has_brought_forward_losses", False))
        if not has_losses:
            passed.append("Losses: No brought-forward or carried-forward loss claims.")
        else:
            failed.append("Losses: Brought-forward house property losses cannot be set off in ITR-1 (Requires ITR-2 Schedule CFL).")

        # 11. Speculative / Virtual Digital Assets
        has_crypto_lottery = bool(declarations.get("has_crypto_or_lottery", False))
        if not has_crypto_lottery:
            passed.append("Special Incomes: No lottery, race horse, or Virtual Digital Asset (VDA/Crypto) income.")
        else:
            failed.append("Special Incomes: VDA (Crypto) or lottery winnings require Schedule VDA / Schedule OS in ITR-2.")

        # Final Determination
        is_eligible = len(failed) == 0
        recommended_form = FormType.ITR_1 if is_eligible else (
            FormType.ITR_3 if has_business else FormType.ITR_2
        )

        explanation = (
            f"Taxpayer {profile.full_name} ({profile.masked_pan}) is fully eligible for ITR-1 (SAHAJ) for AY {assessment.assessment_year}."
            if is_eligible
            else f"Taxpayer {profile.full_name} ({profile.masked_pan}) is NOT eligible for ITR-1 due to {len(failed)} statutory disqualification(s). Recommended return form: {recommended_form.value}."
        )

        result = ITR1EligibilityResult(
            eligible=is_eligible,
            selected_form=FormType.ITR_1,
            recommended_form=recommended_form,
            assessment_year=assessment.assessment_year,
            total_income=total_income,
            passed_conditions=passed,
            failed_conditions=failed,
            explanation=explanation,
        )

        self.db.log_audit_event(
            taxpayer_id=taxpayer_id,
            assessment_id=assessment_id,
            action="ELIGIBILITY_CHECK_PASSED" if is_eligible else "ELIGIBILITY_CHECK_DISQUALIFIED",
            details=explanation,
        )

        return result
