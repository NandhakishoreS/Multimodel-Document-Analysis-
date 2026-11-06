"""
Tax Module — Taxpayer Onboarding & Profile Management Service (Phase 2).

Provides validation, profile lifecycle management, bank account verification,
assessment year binding, and auditable field tracking.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from tax_module.constants import (
    DEFAULT_ASSESSMENT_YEAR,
    SUPPORTED_ASSESSMENT_YEARS,
    FilingStatus,
    FormType,
    ResidentialStatus,
    TaxRegime,
    get_fy_from_ay,
    mask_bank_account,
    mask_pan,
)
from tax_module.db import TaxDatabase
from tax_module.models import BankAccount, TaxAssessment, TaxpayerProfile

logger = logging.getLogger(__name__)

# Statutory Regex Validators
_PAN_REGEX = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$")
_IFSC_REGEX = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")
_PINCODE_REGEX = re.compile(r"^\d{6}$")
_MOBILE_REGEX = re.compile(r"^[6-9]\d{9}$")
_EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class TaxpayerOnboardingService:
    """Service handling taxpayer identity onboarding and assessment initialization."""

    def __init__(self, db: Optional[TaxDatabase] = None):
        self.db = db or TaxDatabase()

    # -----------------------------------------------------------------------
    # Validation Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def validate_profile_inputs(data: Dict[str, Any]) -> List[str]:
        """Validate personal and statutory details for individual taxpayer."""
        errors: List[str] = []

        # PAN validation
        pan = str(data.get("pan", "")).strip().upper()
        if not pan:
            errors.append("PAN is required.")
        elif not _PAN_REGEX.match(pan):
            errors.append(f"Invalid PAN format: '{pan}'. Must follow pattern 5 letters, 4 digits, 1 letter (e.g. ABCDE1234F).")
        elif pan[3] != "P":
            # 4th character 'P' indicates Individual in Indian PAN system
            errors.append(f"PAN 4th character must be 'P' for Individual taxpayers (found '{pan[3]}').")

        # Name validation
        first_name = str(data.get("first_name", "")).strip()
        last_name = str(data.get("last_name", "")).strip()
        if not first_name:
            errors.append("First Name is required.")
        if not last_name:
            errors.append("Last Name / Surname is required.")

        # Date of birth
        dob = str(data.get("dob", "")).strip()
        if not dob:
            errors.append("Date of Birth is required.")
        elif not _DATE_REGEX.match(dob):
            errors.append(f"Date of Birth must be in ISO format YYYY-MM-DD (got '{dob}').")
        else:
            try:
                dob_dt = datetime.strptime(dob, "%Y-%m-%d")
                if dob_dt > datetime.now():
                    errors.append("Date of Birth cannot be in the future.")
            except ValueError:
                errors.append(f"Invalid calendar date for Date of Birth: '{dob}'.")

        # Contact validation
        mobile = str(data.get("mobile", "")).strip()
        if mobile and not _MOBILE_REGEX.match(mobile):
            errors.append(f"Invalid Indian mobile number: '{mobile}'. Must be 10 digits starting with 6-9.")

        email = str(data.get("email", "")).strip()
        if email and not _EMAIL_REGEX.match(email):
            errors.append(f"Invalid email address format: '{email}'.")

        # Address validation
        pincode = str(data.get("address_pincode", "")).strip()
        if pincode and not _PINCODE_REGEX.match(pincode):
            errors.append(f"Invalid PIN code: '{pincode}'. Must be exactly 6 digits.")

        return errors

    @staticmethod
    def validate_bank_account(
        account_number: str,
        ifsc_code: str,
        bank_name: str,
    ) -> List[str]:
        """Validate bank details required for tax refund credit."""
        errors: List[str] = []
        acc = account_number.strip()
        if not acc:
            errors.append("Bank account number is required.")
        elif not (9 <= len(acc) <= 18 and acc.isalnum()):
            errors.append(f"Invalid account number length: '{acc}'. Must be 9 to 18 alphanumeric characters.")

        ifsc = ifsc_code.strip().upper()
        if not ifsc:
            errors.append("IFSC code is required.")
        elif not _IFSC_REGEX.match(ifsc):
            errors.append(f"Invalid IFSC code: '{ifsc}'. Must be 11 characters (e.g. SBIN0001234).")

        if not bank_name.strip():
            errors.append("Bank name is required.")

        return errors

    # -----------------------------------------------------------------------
    # Taxpayer Profile Operations
    # -----------------------------------------------------------------------

    def onboard_taxpayer(self, data: Dict[str, Any]) -> Tuple[Optional[TaxpayerProfile], List[str]]:
        """Validate and persist or update an individual taxpayer profile."""
        errors = self.validate_profile_inputs(data)
        if errors:
            return None, errors

        pan = str(data.get("pan", "")).strip().upper()
        existing = self.db.get_taxpayer_by_pan(pan)
        taxpayer_id = existing.taxpayer_id if existing else data.get("taxpayer_id")

        profile_data = dict(data)
        profile_data["pan"] = pan
        if taxpayer_id:
            profile_data["taxpayer_id"] = taxpayer_id

        profile = TaxpayerProfile.from_dict(profile_data)
        self.db.save_taxpayer_profile(profile)
        self.db.log_audit_event(
            taxpayer_id=profile.taxpayer_id,
            action="TAXPAYER_ONBOARDED" if not existing else "TAXPAYER_PROFILE_UPDATED",
            details=f"Profile saved for PAN {profile.masked_pan} ({profile.full_name})",
        )
        return profile, []

    def add_bank_account(
        self,
        taxpayer_id: str,
        account_number: str,
        ifsc_code: str,
        bank_name: str,
        account_type: str = "SAVINGS",
        is_refund_account: bool = True,
    ) -> Tuple[bool, List[str]]:
        """Add and validate a bank account for the taxpayer profile."""
        errors = self.validate_bank_account(account_number, ifsc_code, bank_name)
        if errors:
            return False, errors

        profile = self.db.get_taxpayer_profile(taxpayer_id)
        if not profile:
            return False, [f"Taxpayer with ID '{taxpayer_id}' not found."]

        # If this is marked as refund account, ensure only one active refund account exists
        if is_refund_account:
            for b in profile.bank_accounts:
                b.is_refund_account = False

        new_acc = BankAccount(
            account_number=account_number.strip(),
            ifsc_code=ifsc_code.strip().upper(),
            bank_name=bank_name.strip(),
            account_type=account_type.upper(),
            is_refund_account=is_refund_account,
        )

        # Replace existing account if already present
        updated_banks = [b for b in profile.bank_accounts if b.account_number != new_acc.account_number]
        updated_banks.append(new_acc)
        profile.bank_accounts = updated_banks

        self.db.save_taxpayer_profile(profile)
        self.db.log_audit_event(
            taxpayer_id=taxpayer_id,
            action="BANK_ACCOUNT_ADDED",
            details=f"Added account {mask_bank_account(account_number)} ({bank_name}, IFSC: {ifsc_code.upper()})",
        )
        return True, []

    # -----------------------------------------------------------------------
    # Assessment Session Operations
    # -----------------------------------------------------------------------

    def initialize_assessment(
        self,
        taxpayer_id: str,
        assessment_year: str = DEFAULT_ASSESSMENT_YEAR,
        selected_regime: TaxRegime = TaxRegime.NEW,
    ) -> Tuple[Optional[TaxAssessment], List[str]]:
        """Bind taxpayer to an Assessment Year and initialize the filing lifecycle."""
        profile = self.db.get_taxpayer_profile(taxpayer_id)
        if not profile:
            return None, [f"Taxpayer with ID '{taxpayer_id}' does not exist."]

        if assessment_year not in SUPPORTED_ASSESSMENT_YEARS:
            return None, [
                f"Unsupported Assessment Year '{assessment_year}'. "
                f"Currently supported years: {', '.join(SUPPORTED_ASSESSMENT_YEARS)}."
            ]

        fy = get_fy_from_ay(assessment_year)

        # Check existing assessment for this taxpayer and year
        existing = self.db.get_assessment_for_year(taxpayer_id, assessment_year)
        if existing:
            return existing, []

        assessment = TaxAssessment(
            taxpayer_id=taxpayer_id,
            assessment_year=assessment_year,
            financial_year=fy,
            filing_status=FilingStatus.NOT_STARTED,
            selected_regime=selected_regime,
            itr_form=FormType.ITR_1,
        )
        self.db.save_assessment(assessment)
        self.db.log_audit_event(
            taxpayer_id=taxpayer_id,
            assessment_id=assessment.assessment_id,
            action="ASSESSMENT_INITIALIZED",
            details=f"Initialized ITR-1 filing session for AY {assessment_year} (FY {fy}) under {selected_regime.value} Regime",
        )
        return assessment, []

    def get_onboarding_status(self, taxpayer_id: str, assessment_id: str) -> Dict[str, Any]:
        """Compute profile completion percentage and readiness for document ingestion."""
        profile = self.db.get_taxpayer_profile(taxpayer_id)
        assessment = self.db.get_assessment(assessment_id)

        if not profile or not assessment:
            return {
                "is_ready": False,
                "completion_percentage": 0,
                "missing_items": ["Taxpayer profile or assessment not found."],
            }

        missing: List[str] = []
        checks = [
            ("PAN", bool(profile.pan and profile.validate_pan())),
            ("Full Name", bool(profile.full_name)),
            ("Date of Birth", bool(profile.dob)),
            ("Residential Status", bool(profile.residential_status)),
            ("Contact Mobile", bool(profile.mobile)),
            ("Contact Email", bool(profile.email)),
            ("Address Details", bool(profile.address_city and profile.address_pincode)),
            ("Bank Account", len(profile.bank_accounts) > 0),
            ("Refund Bank Account", any(b.is_refund_account for b in profile.bank_accounts)),
        ]

        completed = sum(1 for label, ok in checks if ok)
        for label, ok in checks:
            if not ok:
                missing.append(label)

        completion_pct = int((completed / len(checks)) * 100)

        return {
            "is_ready": len(missing) == 0,
            "completion_percentage": completion_pct,
            "taxpayer_name": profile.full_name,
            "masked_pan": profile.masked_pan,
            "assessment_year": assessment.assessment_year,
            "financial_year": assessment.financial_year,
            "regime": assessment.selected_regime.value,
            "filing_status": assessment.filing_status.value,
            "bank_accounts_count": len(profile.bank_accounts),
            "missing_items": missing,
        }
