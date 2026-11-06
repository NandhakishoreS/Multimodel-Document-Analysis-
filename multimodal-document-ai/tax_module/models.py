"""
Tax Module Domain Models and Data Structures.

Defines strongly-typed dataclasses for the entire Indian Income Tax Return
preparation lifecycle. All monetary values are strictly represented as
Python Decimal to guarantee 100% precision without binary float distortion.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from tax_module.constants import (
    DEFAULT_ASSESSMENT_YEAR,
    FilingStatus,
    FormType,
    IncomeHead,
    ResidentialStatus,
    TaxDocumentCategory,
    TaxRegime,
    VerificationStatus,
    get_fy_from_ay,
    mask_bank_account,
    mask_pan,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_decimal(val: Any) -> Optional[Decimal]:
    """Convert input to Decimal safely, returning None if absent or invalid."""
    if val is None:
        return None
    if isinstance(val, Decimal):
        return val
    try:
        s = str(val).strip().replace(",", "")
        if not s:
            return None
        return Decimal(s)
    except (InvalidOperation, ValueError, TypeError):
        return None


def decimal_to_str(val: Optional[Decimal]) -> Optional[str]:
    """Format Decimal as string preserving exact precision."""
    if val is None:
        return None
    return str(val)


# ---------------------------------------------------------------------------
# Bank Account
# ---------------------------------------------------------------------------

@dataclass
class BankAccount:
    """Taxpayer Bank Account details required for ITR-1 refund credit."""
    account_number: str
    ifsc_code: str
    bank_name: str
    account_type: str = "SAVINGS"          # SAVINGS | CURRENT
    is_refund_account: bool = True

    def masked_account_number(self) -> str:
        return mask_bank_account(self.account_number)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BankAccount:
        return cls(
            account_number=str(data.get("account_number", "")),
            ifsc_code=str(data.get("ifsc_code", "")).upper(),
            bank_name=str(data.get("bank_name", "")),
            account_type=str(data.get("account_type", "SAVINGS")),
            is_refund_account=bool(data.get("is_refund_account", True)),
        )


# ---------------------------------------------------------------------------
# Taxpayer Profile
# ---------------------------------------------------------------------------

_PAN_REGEX = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$")


@dataclass
class TaxpayerProfile:
    """Individual taxpayer master profile."""
    taxpayer_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pan: str = ""
    first_name: str = ""
    middle_name: str = ""
    last_name: str = ""
    dob: str = ""                         # ISO YYYY-MM-DD
    gender: str = "M"                     # M | F | O
    residential_status: ResidentialStatus = ResidentialStatus.RESIDENT
    email: str = ""
    mobile: str = ""
    
    # Address components matching ITR-1 Part A
    address_flat_door_block: str = ""
    address_premises_building: str = ""
    address_road_street: str = ""
    address_area_locality: str = ""
    address_city: str = ""
    address_state: str = ""
    address_pincode: str = ""
    
    bank_accounts: List[BankAccount] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    @property
    def full_name(self) -> str:
        parts = [self.first_name, self.middle_name, self.last_name]
        return " ".join(p.strip() for p in parts if p and p.strip())

    @property
    def masked_pan(self) -> str:
        return mask_pan(self.pan)

    def validate_pan(self) -> bool:
        return bool(_PAN_REGEX.match(self.pan.strip().upper()))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["residential_status"] = self.residential_status.value
        d["bank_accounts"] = [acc.to_dict() for acc in self.bank_accounts]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaxpayerProfile:
        res_status = data.get("residential_status", ResidentialStatus.RESIDENT.value)
        try:
            status_enum = ResidentialStatus(res_status)
        except ValueError:
            status_enum = ResidentialStatus.RESIDENT

        bank_list = [
            BankAccount.from_dict(b) if isinstance(b, dict) else b
            for b in data.get("bank_accounts", [])
        ]

        return cls(
            taxpayer_id=str(data.get("taxpayer_id", str(uuid.uuid4()))),
            pan=str(data.get("pan", "")).strip().upper(),
            first_name=str(data.get("first_name", "")),
            middle_name=str(data.get("middle_name", "")),
            last_name=str(data.get("last_name", "")),
            dob=str(data.get("dob", "")),
            gender=str(data.get("gender", "M")),
            residential_status=status_enum,
            email=str(data.get("email", "")),
            mobile=str(data.get("mobile", "")),
            address_flat_door_block=str(data.get("address_flat_door_block", "")),
            address_premises_building=str(data.get("address_premises_building", "")),
            address_road_street=str(data.get("address_road_street", "")),
            address_area_locality=str(data.get("address_area_locality", "")),
            address_city=str(data.get("address_city", "")),
            address_state=str(data.get("address_state", "")),
            address_pincode=str(data.get("address_pincode", "")),
            bank_accounts=bank_list,
            created_at=str(data.get("created_at", _utc_now_iso())),
            updated_at=str(data.get("updated_at", _utc_now_iso())),
        )


# ---------------------------------------------------------------------------
# Tax Assessment
# ---------------------------------------------------------------------------

@dataclass
class TaxAssessment:
    """An individual tax return filing session for a specific Assessment Year."""
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    taxpayer_id: str = ""
    assessment_year: str = DEFAULT_ASSESSMENT_YEAR
    financial_year: str = field(default_factory=lambda: get_fy_from_ay(DEFAULT_ASSESSMENT_YEAR))
    filing_status: FilingStatus = FilingStatus.NOT_STARTED
    selected_regime: TaxRegime = TaxRegime.NEW
    itr_form: FormType = FormType.ITR_1
    filing_type: str = "ORIGINAL"          # ORIGINAL | REVISED
    acknowledgement_number: Optional[str] = None
    filing_date: Optional[str] = None
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "taxpayer_id": self.taxpayer_id,
            "assessment_year": self.assessment_year,
            "financial_year": self.financial_year,
            "filing_status": self.filing_status.value,
            "selected_regime": self.selected_regime.value,
            "itr_form": self.itr_form.value,
            "filing_type": self.filing_type,
            "acknowledgement_number": self.acknowledgement_number,
            "filing_date": self.filing_date,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaxAssessment:
        ay = str(data.get("assessment_year", DEFAULT_ASSESSMENT_YEAR))
        fy = str(data.get("financial_year", get_fy_from_ay(ay)))
        return cls(
            assessment_id=str(data.get("assessment_id", str(uuid.uuid4()))),
            taxpayer_id=str(data.get("taxpayer_id", "")),
            assessment_year=ay,
            financial_year=fy,
            filing_status=FilingStatus(data.get("filing_status", FilingStatus.NOT_STARTED.value)),
            selected_regime=TaxRegime(data.get("selected_regime", TaxRegime.NEW.value)),
            itr_form=FormType(data.get("itr_form", FormType.ITR_1.value)),
            filing_type=str(data.get("filing_type", "ORIGINAL")),
            acknowledgement_number=data.get("acknowledgement_number"),
            filing_date=data.get("filing_date"),
            created_at=str(data.get("created_at", _utc_now_iso())),
            updated_at=str(data.get("updated_at", _utc_now_iso())),
        )


# ---------------------------------------------------------------------------
# Tax Document
# ---------------------------------------------------------------------------

@dataclass
class TaxDocument:
    """Document uploaded for tax preparation (Form 16, interest cert, AIS, etc.)."""
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    assessment_id: str = ""
    document_category: TaxDocumentCategory = TaxDocumentCategory.OTHER
    filename: str = ""
    file_path: str = ""
    file_hash: str = ""                   # SHA-256
    file_size: int = 0
    ocr_raw_text: str = ""
    ocr_confidence: float = 0.0
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    uploaded_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "assessment_id": self.assessment_id,
            "document_category": self.document_category.value,
            "filename": self.filename,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "file_size": self.file_size,
            "ocr_raw_text": self.ocr_raw_text,
            "ocr_confidence": self.ocr_confidence,
            "verification_status": self.verification_status.value,
            "uploaded_at": self.uploaded_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaxDocument:
        cat = data.get("document_category", TaxDocumentCategory.OTHER.value)
        try:
            cat_enum = TaxDocumentCategory(cat)
        except ValueError:
            cat_enum = TaxDocumentCategory.OTHER

        ver = data.get("verification_status", VerificationStatus.UNVERIFIED.value)
        try:
            ver_enum = VerificationStatus(ver)
        except ValueError:
            ver_enum = VerificationStatus.UNVERIFIED

        return cls(
            doc_id=str(data.get("doc_id", str(uuid.uuid4()))),
            assessment_id=str(data.get("assessment_id", "")),
            document_category=cat_enum,
            filename=str(data.get("filename", "")),
            file_path=str(data.get("file_path", "")),
            file_hash=str(data.get("file_hash", "")),
            file_size=int(data.get("file_size", 0)),
            ocr_raw_text=str(data.get("ocr_raw_text", "")),
            ocr_confidence=float(data.get("ocr_confidence", 0.0)),
            verification_status=ver_enum,
            uploaded_at=str(data.get("uploaded_at", _utc_now_iso())),
        )


# ---------------------------------------------------------------------------
# Extracted Tax Field
# ---------------------------------------------------------------------------

@dataclass
class ExtractedTaxField:
    """Individual extracted field with source traceability and Decimal precision."""
    field_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    assessment_id: str = ""
    field_name: str = ""
    original_ocr_text: str = ""
    normalized_decimal: Optional[Decimal] = None
    confidence: float = 0.0
    page_number: int = 1
    bounding_box: Optional[List[int]] = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    user_corrected_value: Optional[Decimal] = None

    @property
    def effective_value(self) -> Optional[Decimal]:
        """User correction overrides OCR extraction if present."""
        if self.user_corrected_value is not None:
            return self.user_corrected_value
        return self.normalized_decimal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_id": self.field_id,
            "doc_id": self.doc_id,
            "assessment_id": self.assessment_id,
            "field_name": self.field_name,
            "original_ocr_text": self.original_ocr_text,
            "normalized_decimal": decimal_to_str(self.normalized_decimal),
            "confidence": self.confidence,
            "page_number": self.page_number,
            "bounding_box": json.dumps(self.bounding_box) if self.bounding_box else None,
            "verification_status": self.verification_status.value,
            "user_corrected_value": decimal_to_str(self.user_corrected_value),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractedTaxField:
        ver = data.get("verification_status", VerificationStatus.UNVERIFIED.value)
        try:
            ver_enum = VerificationStatus(ver)
        except ValueError:
            ver_enum = VerificationStatus.UNVERIFIED

        bbox = data.get("bounding_box")
        if isinstance(bbox, str):
            try:
                bbox = json.loads(bbox)
            except Exception:
                bbox = None

        return cls(
            field_id=str(data.get("field_id", str(uuid.uuid4()))),
            doc_id=str(data.get("doc_id", "")),
            assessment_id=str(data.get("assessment_id", "")),
            field_name=str(data.get("field_name", "")),
            original_ocr_text=str(data.get("original_ocr_text", "")),
            normalized_decimal=to_decimal(data.get("normalized_decimal")),
            confidence=float(data.get("confidence", 0.0)),
            page_number=int(data.get("page_number", 1)),
            bounding_box=bbox,
            verification_status=ver_enum,
            user_corrected_value=to_decimal(data.get("user_corrected_value")),
        )


# ---------------------------------------------------------------------------
# Income Record
# ---------------------------------------------------------------------------

@dataclass
class IncomeRecord:
    """Itemized income entry categorized under a statutory head."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    assessment_id: str = ""
    income_head: IncomeHead = IncomeHead.SALARY
    source_doc_id: Optional[str] = None
    source_name: str = ""                  # Employer name / Bank name
    identifier: Optional[str] = None       # TAN / Account number
    gross_amount: Decimal = Decimal("0.00")
    exempt_allowances: Decimal = Decimal("0.00")
    net_taxable_amount: Decimal = Decimal("0.00")
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "assessment_id": self.assessment_id,
            "income_head": self.income_head.value,
            "source_doc_id": self.source_doc_id,
            "source_name": self.source_name,
            "identifier": self.identifier,
            "gross_amount": decimal_to_str(self.gross_amount),
            "exempt_allowances": decimal_to_str(self.exempt_allowances),
            "net_taxable_amount": decimal_to_str(self.net_taxable_amount),
            "verification_status": self.verification_status.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IncomeRecord:
        head = data.get("income_head", IncomeHead.SALARY.value)
        try:
            head_enum = IncomeHead(head)
        except ValueError:
            head_enum = IncomeHead.SALARY

        ver = data.get("verification_status", VerificationStatus.UNVERIFIED.value)
        try:
            ver_enum = VerificationStatus(ver)
        except ValueError:
            ver_enum = VerificationStatus.UNVERIFIED

        return cls(
            record_id=str(data.get("record_id", str(uuid.uuid4()))),
            assessment_id=str(data.get("assessment_id", "")),
            income_head=head_enum,
            source_doc_id=data.get("source_doc_id"),
            source_name=str(data.get("source_name", "")),
            identifier=data.get("identifier"),
            gross_amount=to_decimal(data.get("gross_amount")) or Decimal("0.00"),
            exempt_allowances=to_decimal(data.get("exempt_allowances")) or Decimal("0.00"),
            net_taxable_amount=to_decimal(data.get("net_taxable_amount")) or Decimal("0.00"),
            verification_status=ver_enum,
        )


# ---------------------------------------------------------------------------
# Deduction Record
# ---------------------------------------------------------------------------

@dataclass
class DeductionRecord:
    """Chapter VI-A deduction claim (80C, 80D, 80CCD, 80TTA, 80G, etc.)."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    assessment_id: str = ""
    section_code: str = "80C"
    description: str = ""
    claimed_amount: Decimal = Decimal("0.00")
    eligible_amount: Decimal = Decimal("0.00")
    source_doc_id: Optional[str] = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "assessment_id": self.assessment_id,
            "section_code": self.section_code,
            "description": self.description,
            "claimed_amount": decimal_to_str(self.claimed_amount),
            "eligible_amount": decimal_to_str(self.eligible_amount),
            "source_doc_id": self.source_doc_id,
            "verification_status": self.verification_status.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeductionRecord:
        ver = data.get("verification_status", VerificationStatus.UNVERIFIED.value)
        try:
            ver_enum = VerificationStatus(ver)
        except ValueError:
            ver_enum = VerificationStatus.UNVERIFIED

        return cls(
            record_id=str(data.get("record_id", str(uuid.uuid4()))),
            assessment_id=str(data.get("assessment_id", "")),
            section_code=str(data.get("section_code", "80C")).upper(),
            description=str(data.get("description", "")),
            claimed_amount=to_decimal(data.get("claimed_amount")) or Decimal("0.00"),
            eligible_amount=to_decimal(data.get("eligible_amount")) or Decimal("0.00"),
            source_doc_id=data.get("source_doc_id"),
            verification_status=ver_enum,
        )


# ---------------------------------------------------------------------------
# Tax Credit Record
# ---------------------------------------------------------------------------

@dataclass
class TaxCreditRecord:
    """Prepaid taxes: TDS from salary/other, Advance Tax, Self-Assessment Tax."""
    credit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    assessment_id: str = ""
    credit_type: str = "TDS_SALARY"        # TDS_SALARY | TDS_OTHER | ADVANCE_TAX | SELF_ASSESSMENT
    deductor_tan: Optional[str] = None
    deductor_name: Optional[str] = None
    total_tax_deducted: Decimal = Decimal("0.00")
    claimed_amount: Decimal = Decimal("0.00")
    challan_bsr_code: Optional[str] = None
    challan_serial_no: Optional[str] = None
    challan_date: Optional[str] = None
    source_doc_id: Optional[str] = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "credit_id": self.credit_id,
            "assessment_id": self.assessment_id,
            "credit_type": self.credit_type,
            "deductor_tan": self.deductor_tan,
            "deductor_name": self.deductor_name,
            "total_tax_deducted": decimal_to_str(self.total_tax_deducted),
            "claimed_amount": decimal_to_str(self.claimed_amount),
            "challan_bsr_code": self.challan_bsr_code,
            "challan_serial_no": self.challan_serial_no,
            "challan_date": self.challan_date,
            "source_doc_id": self.source_doc_id,
            "verification_status": self.verification_status.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaxCreditRecord:
        ver = data.get("verification_status", VerificationStatus.UNVERIFIED.value)
        try:
            ver_enum = VerificationStatus(ver)
        except ValueError:
            ver_enum = VerificationStatus.UNVERIFIED

        return cls(
            credit_id=str(data.get("credit_id", str(uuid.uuid4()))),
            assessment_id=str(data.get("assessment_id", "")),
            credit_type=str(data.get("credit_type", "TDS_SALARY")),
            deductor_tan=data.get("deductor_tan"),
            deductor_name=data.get("deductor_name"),
            total_tax_deducted=to_decimal(data.get("total_tax_deducted")) or Decimal("0.00"),
            claimed_amount=to_decimal(data.get("claimed_amount")) or Decimal("0.00"),
            challan_bsr_code=data.get("challan_bsr_code"),
            challan_serial_no=data.get("challan_serial_no"),
            challan_date=data.get("challan_date"),
            source_doc_id=data.get("source_doc_id"),
            verification_status=ver_enum,
        )


# ---------------------------------------------------------------------------
# Reconciliation Record
# ---------------------------------------------------------------------------

@dataclass
class ReconciliationRecord:
    """Discrepancy audit record between multiple sources (e.g. Form 16 vs 26AS vs AIS)."""
    recon_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    assessment_id: str = ""
    field_name: str = ""
    primary_source: str = ""
    primary_amount: Decimal = Decimal("0.00")
    secondary_source: str = ""
    secondary_amount: Decimal = Decimal("0.00")
    difference: Decimal = Decimal("0.00")
    status: str = "MATCHED"                # MATCHED | REVIEW_REQUIRED | RESOLVED
    resolution_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recon_id": self.recon_id,
            "assessment_id": self.assessment_id,
            "field_name": self.field_name,
            "primary_source": self.primary_source,
            "primary_amount": decimal_to_str(self.primary_amount),
            "secondary_source": self.secondary_source,
            "secondary_amount": decimal_to_str(self.secondary_amount),
            "difference": decimal_to_str(self.difference),
            "status": self.status,
            "resolution_notes": self.resolution_notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReconciliationRecord:
        return cls(
            recon_id=str(data.get("recon_id", str(uuid.uuid4()))),
            assessment_id=str(data.get("assessment_id", "")),
            field_name=str(data.get("field_name", "")),
            primary_source=str(data.get("primary_source", "")),
            primary_amount=to_decimal(data.get("primary_amount")) or Decimal("0.00"),
            secondary_source=str(data.get("secondary_source", "")),
            secondary_amount=to_decimal(data.get("secondary_amount")) or Decimal("0.00"),
            difference=to_decimal(data.get("difference")) or Decimal("0.00"),
            status=str(data.get("status", "MATCHED")),
            resolution_notes=str(data.get("resolution_notes", "")),
        )


# ---------------------------------------------------------------------------
# Tax Calculation
# ---------------------------------------------------------------------------

@dataclass
class TaxCalculation:
    """Comprehensive tax calculation breakdown under Old or New Regime."""
    calc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    assessment_id: str = ""
    regime: TaxRegime = TaxRegime.NEW
    gross_total_income: Decimal = Decimal("0.00")
    total_deductions: Decimal = Decimal("0.00")
    total_taxable_income: Decimal = Decimal("0.00")
    tax_on_taxable_income: Decimal = Decimal("0.00")
    rebate_87a: Decimal = Decimal("0.00")
    tax_after_rebate: Decimal = Decimal("0.00")
    surcharge: Decimal = Decimal("0.00")
    cess: Decimal = Decimal("0.00")         # 4% Health & Education Cess
    total_tax_liability: Decimal = Decimal("0.00")
    total_tax_credits: Decimal = Decimal("0.00")
    refund_or_payable: Decimal = Decimal("0.00")  # Negative = Refund, Positive = Payable
    interest_234a: Decimal = Decimal("0.00")
    interest_234b: Decimal = Decimal("0.00")
    interest_234c: Decimal = Decimal("0.00")
    breakdown_json: str = "{}"
    calculated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calc_id": self.calc_id,
            "assessment_id": self.assessment_id,
            "regime": self.regime.value,
            "gross_total_income": decimal_to_str(self.gross_total_income),
            "total_deductions": decimal_to_str(self.total_deductions),
            "total_taxable_income": decimal_to_str(self.total_taxable_income),
            "tax_on_taxable_income": decimal_to_str(self.tax_on_taxable_income),
            "rebate_87a": decimal_to_str(self.rebate_87a),
            "tax_after_rebate": decimal_to_str(self.tax_after_rebate),
            "surcharge": decimal_to_str(self.surcharge),
            "cess": decimal_to_str(self.cess),
            "total_tax_liability": decimal_to_str(self.total_tax_liability),
            "total_tax_credits": decimal_to_str(self.total_tax_credits),
            "refund_or_payable": decimal_to_str(self.refund_or_payable),
            "interest_234a": decimal_to_str(self.interest_234a),
            "interest_234b": decimal_to_str(self.interest_234b),
            "interest_234c": decimal_to_str(self.interest_234c),
            "breakdown_json": self.breakdown_json,
            "calculated_at": self.calculated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaxCalculation:
        reg = data.get("regime", TaxRegime.NEW.value)
        try:
            reg_enum = TaxRegime(reg)
        except ValueError:
            reg_enum = TaxRegime.NEW

        return cls(
            calc_id=str(data.get("calc_id", str(uuid.uuid4()))),
            assessment_id=str(data.get("assessment_id", "")),
            regime=reg_enum,
            gross_total_income=to_decimal(data.get("gross_total_income")) or Decimal("0.00"),
            total_deductions=to_decimal(data.get("total_deductions")) or Decimal("0.00"),
            total_taxable_income=to_decimal(data.get("total_taxable_income")) or Decimal("0.00"),
            tax_on_taxable_income=to_decimal(data.get("tax_on_taxable_income")) or Decimal("0.00"),
            rebate_87a=to_decimal(data.get("rebate_87a")) or Decimal("0.00"),
            tax_after_rebate=to_decimal(data.get("tax_after_rebate")) or Decimal("0.00"),
            surcharge=to_decimal(data.get("surcharge")) or Decimal("0.00"),
            cess=to_decimal(data.get("cess")) or Decimal("0.00"),
            total_tax_liability=to_decimal(data.get("total_tax_liability")) or Decimal("0.00"),
            total_tax_credits=to_decimal(data.get("total_tax_credits")) or Decimal("0.00"),
            refund_or_payable=to_decimal(data.get("refund_or_payable")) or Decimal("0.00"),
            interest_234a=to_decimal(data.get("interest_234a")) or Decimal("0.00"),
            interest_234b=to_decimal(data.get("interest_234b")) or Decimal("0.00"),
            interest_234c=to_decimal(data.get("interest_234c")) or Decimal("0.00"),
            breakdown_json=str(data.get("breakdown_json", "{}")),
            calculated_at=str(data.get("calculated_at", _utc_now_iso())),
        )


# ---------------------------------------------------------------------------
# ITR Return Artifact
# ---------------------------------------------------------------------------

@dataclass
class ITRReturn:
    """Official generated ITR Return file and validation record."""
    return_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    assessment_id: str = ""
    assessment_year: str = DEFAULT_ASSESSMENT_YEAR
    itr_form: FormType = FormType.ITR_1
    official_json: str = "{}"
    schema_version: str = "1.0"
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "return_id": self.return_id,
            "assessment_id": self.assessment_id,
            "assessment_year": self.assessment_year,
            "itr_form": self.itr_form.value,
            "official_json": self.official_json,
            "schema_version": self.schema_version,
            "is_valid": 1 if self.is_valid else 0,
            "validation_errors": json.dumps(self.validation_errors),
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ITRReturn:
        errors = data.get("validation_errors", [])
        if isinstance(errors, str):
            try:
                errors = json.loads(errors)
            except Exception:
                errors = [errors]

        return cls(
            return_id=str(data.get("return_id", str(uuid.uuid4()))),
            assessment_id=str(data.get("assessment_id", "")),
            assessment_year=str(data.get("assessment_year", DEFAULT_ASSESSMENT_YEAR)),
            itr_form=FormType(data.get("itr_form", FormType.ITR_1.value)),
            official_json=str(data.get("official_json", "{}")),
            schema_version=str(data.get("schema_version", "1.0")),
            is_valid=bool(data.get("is_valid", False)),
            validation_errors=errors,
            generated_at=str(data.get("generated_at", _utc_now_iso())),
        )


# ---------------------------------------------------------------------------
# Tax Audit Event
# ---------------------------------------------------------------------------

@dataclass
class TaxAuditEvent:
    """Immutable audit trail of taxpayer actions and calculation changes."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    taxpayer_id: str = ""
    assessment_id: Optional[str] = None
    action: str = ""                       # e.g., PROFILE_UPDATED, DOC_UPLOADED, FIELD_CORRECTED
    details: str = ""
    timestamp: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaxAuditEvent:
        return cls(
            event_id=str(data.get("event_id", str(uuid.uuid4()))),
            taxpayer_id=str(data.get("taxpayer_id", "")),
            assessment_id=data.get("assessment_id"),
            action=str(data.get("action", "")),
            details=str(data.get("details", "")),
            timestamp=str(data.get("timestamp", _utc_now_iso())),
        )
