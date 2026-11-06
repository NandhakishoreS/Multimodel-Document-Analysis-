"""
Tax Module Constants and Enumerations.

Defines filing states, tax regimes, residential statuses, document types,
income heads, and financial/assessment year utilities.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Optional, Tuple


class FilingStatus(str, Enum):
    """Filing workflow lifecycle states (Part 10 of project specification)."""
    NOT_STARTED = "NOT_STARTED"
    DOCUMENTS_UPLOADED = "DOCUMENTS_UPLOADED"
    EXTRACTION_IN_PROGRESS = "EXTRACTION_IN_PROGRESS"
    AWAITING_USER_REVIEW = "AWAITING_USER_REVIEW"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    TAX_CALCULATED = "TAX_CALCULATED"
    ITR_VALIDATION_PENDING = "ITR_VALIDATION_PENDING"
    ITR_READY = "ITR_READY"
    JSON_GENERATED = "JSON_GENERATED"
    IMPORTED_IN_OFFICIAL_UTILITY = "IMPORTED_IN_OFFICIAL_UTILITY"
    SUBMITTED_BY_TAXPAYER = "SUBMITTED_BY_TAXPAYER"
    VERIFICATION_PENDING = "VERIFICATION_PENDING"
    VERIFIED = "VERIFIED"
    REJECTED_OR_CORRECTION_REQUIRED = "REJECTED_OR_CORRECTION_REQUIRED"


class TaxRegime(str, Enum):
    """Tax regimes under the Indian Income Tax Act."""
    OLD = "OLD"
    NEW = "NEW"


class ResidentialStatus(str, Enum):
    """Residential status for tax eligibility."""
    RESIDENT = "RESIDENT"
    RNOR = "RNOR"                  # Resident but Not Ordinarily Resident
    NON_RESIDENT = "NON_RESIDENT"


class IncomeHead(str, Enum):
    """Heads of income under Section 14."""
    SALARY = "SALARY"
    HOUSE_PROPERTY = "HOUSE_PROPERTY"
    OTHER_SOURCES = "OTHER_SOURCES"
    CAPITAL_GAINS = "CAPITAL_GAINS"
    BUSINESS_PROFESSION = "BUSINESS_PROFESSION"


class TaxDocumentCategory(str, Enum):
    """Supported tax document categories for ingestion and extraction."""
    FORM_16_A = "FORM_16_A"
    FORM_16_B = "FORM_16_B"
    SALARY_SLIP = "SALARY_SLIP"
    BANK_INTEREST_CERT = "BANK_INTEREST_CERT"
    BANK_STATEMENT = "BANK_STATEMENT"
    AIS_TIS = "AIS_TIS"
    FORM_26AS = "FORM_26AS"
    TAX_CHALLAN = "TAX_CHALLAN"
    INVESTMENT_80C = "INVESTMENT_80C"
    MEDICAL_80D = "MEDICAL_80D"
    DONATION_80G = "DONATION_80G"
    RENT_RECEIPT = "RENT_RECEIPT"
    HOME_LOAN_INTEREST = "HOME_LOAN_INTEREST"
    OTHER = "OTHER"


class VerificationStatus(str, Enum):
    """Field/Document verification status."""
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"
    USER_CORRECTED = "USER_CORRECTED"
    REJECTED = "REJECTED"


class FormType(str, Enum):
    """Indian Income Tax Return form types."""
    ITR_1 = "ITR-1"  # SAHAJ
    ITR_2 = "ITR-2"
    ITR_3 = "ITR-3"
    ITR_4 = "ITR-4"  # SUGAM


# ---------------------------------------------------------------------------
# Assessment Year / Financial Year Helpers
# ---------------------------------------------------------------------------

SUPPORTED_ASSESSMENT_YEARS: List[str] = [
    "2024-25",
    "2025-26",
]

DEFAULT_ASSESSMENT_YEAR: str = "2024-25"


def get_fy_from_ay(ay: str) -> str:
    """
    Derive the Financial Year (FY / Previous Year) from Assessment Year (AY).
    e.g. '2024-25' -> '2023-24', '2025-26' -> '2024-25'
    """
    m = re.match(r"^(\d{4})-(\d{2})$", ay.strip())
    if not m:
        raise ValueError(f"Invalid Assessment Year format: '{ay}'. Expected 'YYYY-YY'.")
    start_year = int(m.group(1))
    fy_start = start_year - 1
    fy_end = (fy_start + 1) % 100
    return f"{fy_start}-{fy_end:02d}"


def get_ay_from_fy(fy: str) -> str:
    """
    Derive the Assessment Year (AY) from Financial Year (FY).
    e.g. '2023-24' -> '2024-25'
    """
    m = re.match(r"^(\d{4})-(\d{2})$", fy.strip())
    if not m:
        raise ValueError(f"Invalid Financial Year format: '{fy}'. Expected 'YYYY-YY'.")
    start_year = int(m.group(1))
    ay_start = start_year + 1
    ay_end = (ay_start + 1) % 100
    return f"{ay_start}-{ay_end:02d}"


def mask_pan(pan: Optional[str]) -> str:
    """Mask a 10-character PAN leaving only first 2 and last 2 characters visible."""
    if not pan:
        return ""
    p = pan.strip().upper()
    if len(p) == 10:
        return f"{p[:2]}XXXXXX{p[-2:]}"
    return "XXXXXXXXXX"


def mask_bank_account(account_num: Optional[str]) -> str:
    """Mask bank account number showing only last 4 digits."""
    if not account_num:
        return ""
    acc = account_num.strip()
    if len(acc) > 4:
        return f"{'X' * (len(acc) - 4)}{acc[-4:]}"
    return acc
