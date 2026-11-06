"""
Tax Module — AI-Powered Indian Income Tax Return (ITR-1 SAHAJ) Preparation System.

Provides taxpayer profile onboarding, tax document classification & extraction,
cross-document income reconciliation, deterministic statutory tax computation,
ITR-1 eligibility checks, and official-format ITR-1 JSON generation.
"""

from tax_module.constants import (
    DEFAULT_ASSESSMENT_YEAR,
    SUPPORTED_ASSESSMENT_YEARS,
    FilingStatus,
    FormType,
    IncomeHead,
    ResidentialStatus,
    TaxDocumentCategory,
    TaxRegime,
    VerificationStatus,
    get_ay_from_fy,
    get_fy_from_ay,
    mask_bank_account,
    mask_pan,
)
from tax_module.db import TaxDatabase
from tax_module.models import (
    BankAccount,
    DeductionRecord,
    ExtractedTaxField,
    ITRReturn,
    IncomeRecord,
    ReconciliationRecord,
    TaxAssessment,
    TaxAuditEvent,
    TaxCalculation,
    TaxCreditRecord,
    TaxDocument,
    TaxpayerProfile,
    decimal_to_str,
    to_decimal,
)

__all__ = [
    "DEFAULT_ASSESSMENT_YEAR",
    "SUPPORTED_ASSESSMENT_YEARS",
    "FilingStatus",
    "FormType",
    "IncomeHead",
    "ResidentialStatus",
    "TaxDocumentCategory",
    "TaxRegime",
    "VerificationStatus",
    "get_ay_from_fy",
    "get_fy_from_ay",
    "mask_bank_account",
    "mask_pan",
    "BankAccount",
    "DeductionRecord",
    "ExtractedTaxField",
    "ITRReturn",
    "IncomeRecord",
    "ReconciliationRecord",
    "TaxAssessment",
    "TaxAuditEvent",
    "TaxCalculation",
    "TaxCreditRecord",
    "TaxDocument",
    "TaxpayerProfile",
    "TaxDatabase",
    "decimal_to_str",
    "to_decimal",
]
