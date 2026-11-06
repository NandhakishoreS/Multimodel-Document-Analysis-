"""
Tax Module — Tax Document Field Extraction Engine (Phase 3).

Extracts itemized financial fields, PANs, TANs, salary components, Chapter VI-A
deductions, and TDS credits from tax documents with 100% Decimal precision.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from phase11.money_parser import parse_monetary_string
from tax_module.constants import (
    IncomeHead,
    TaxDocumentCategory,
    VerificationStatus,
)
from tax_module.models import (
    DeductionRecord,
    ExtractedTaxField,
    IncomeRecord,
    TaxCreditRecord,
)

_TAN_REGEX = re.compile(r"\b([A-Z]{4}[0-9]{5}[A-Z]{1})\b")
_PAN_REGEX = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z]{1})\b")
_AY_REGEX = re.compile(r"\b(20\d\d[-/]\d{2})\b")


def _find_amount_near(text: str, patterns: List[str]) -> Tuple[Optional[str], Optional[Decimal]]:
    """Scan text for keywords and extract the adjacent monetary amount."""
    for pattern in patterns:
        m = re.search(
            pattern + r"[\s:=-]+(?:Rs\.?|INR|₹)?\s*([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)",
            text,
            re.IGNORECASE,
        )
        if m:
            raw_val = m.group(1).strip()
            parsed = parse_monetary_string(raw_val)
            if parsed and parsed.value is not None:
                return raw_val, parsed.value
    return None, None


# ---------------------------------------------------------------------------
# Form 16 Part A & Part B Extractor
# ---------------------------------------------------------------------------

@dataclass
class Form16ExtractionResult:
    employer_name: str = ""
    employer_tan: str = ""
    employee_pan: str = ""
    assessment_year: str = ""
    gross_salary: Decimal = Decimal("0.00")
    exempt_allowances: Decimal = Decimal("0.00")
    standard_deduction: Decimal = Decimal("0.00")
    professional_tax: Decimal = Decimal("0.00")
    net_salary_income: Decimal = Decimal("0.00")
    chapter_6a_deductions: Dict[str, Decimal] = field(default_factory=dict)
    total_taxable_income: Decimal = Decimal("0.00")
    total_tds_deducted: Decimal = Decimal("0.00")
    extracted_fields: List[ExtractedTaxField] = field(default_factory=list)


class Form16Extractor:
    """Specialized parser for Form 16 Part A and Part B documents."""

    def extract(self, text: str, doc_id: str = "", assessment_id: str = "") -> Form16ExtractionResult:
        res = Form16ExtractionResult()
        fields: List[ExtractedTaxField] = []

        # TAN of Deductor
        tan_match = _TAN_REGEX.search(text)
        if tan_match:
            res.employer_tan = tan_match.group(1)
            fields.append(ExtractedTaxField(
                doc_id=doc_id,
                assessment_id=assessment_id,
                field_name="employer_tan",
                original_ocr_text=res.employer_tan,
                confidence=0.98,
            ))

        # Employee PAN
        pan_matches = _PAN_REGEX.findall(text)
        for p in pan_matches:
            if p[3] == "P":
                res.employee_pan = p
                fields.append(ExtractedTaxField(
                    doc_id=doc_id,
                    assessment_id=assessment_id,
                    field_name="employee_pan",
                    original_ocr_text=res.employee_pan,
                    confidence=0.98,
                ))
                break

        # Employer Name
        emp_match = re.search(r"name\s*(?:and\s*address)?\s*of\s*(?:the)?\s*employer[\s:=-]+([A-Za-z0-9\s.,&'-]+?)(?:\n|address|tan)", text, re.IGNORECASE)
        if emp_match:
            res.employer_name = emp_match.group(1).strip()
            fields.append(ExtractedTaxField(
                doc_id=doc_id,
                assessment_id=assessment_id,
                field_name="employer_name",
                original_ocr_text=res.employer_name,
                confidence=0.90,
            ))

        # Assessment Year
        ay_match = _AY_REGEX.search(text)
        if ay_match:
            res.assessment_year = ay_match.group(1).replace("/", "-")

        # Monetary fields
        # 1. Gross Salary u/s 17(1)
        raw_gross, val_gross = _find_amount_near(text, [
            r"gross\s*salary(?:\s*(?:as\s*per\s*provisions\s*contained\s*in)?\s*(?:u\/s|sec(?:tion)?\.?)\s*17\(1\))?",
            r"salary\s*as\s*per\s*provisions\s*contained\s*in\s*sec(?:tion)?\.?\s*17\(1\)",
            r"section\s*17\(1\)",
        ])
        if val_gross:
            res.gross_salary = val_gross
            fields.append(ExtractedTaxField(
                doc_id=doc_id,
                assessment_id=assessment_id,
                field_name="gross_salary_17_1",
                original_ocr_text=raw_gross or "",
                normalized_decimal=val_gross,
                confidence=0.95,
            ))

        # 2. Section 10 Exempt Allowances
        raw_exempt, val_exempt = _find_amount_near(text, [
            r"allowances\s*exempt\s*(?:under|u\/s)?\s*sec(?:tion)?\.?\s*10",
            r"exempt\s*allowances",
            r"total\s*exemption\s*u\/s\s*10",
        ])
        if val_exempt:
            res.exempt_allowances = val_exempt
            fields.append(ExtractedTaxField(
                doc_id=doc_id,
                assessment_id=assessment_id,
                field_name="exempt_allowances_sec_10",
                original_ocr_text=raw_exempt or "",
                normalized_decimal=val_exempt,
                confidence=0.92,
            ))

        # 3. Standard Deduction u/s 16(ia)
        raw_sd, val_sd = _find_amount_near(text, [
            r"standard\s*deduction\s*(?:under|u\/s)?\s*sec(?:tion)?\.?\s*16\(ia\)",
            r"standard\s*deduction",
        ])
        if val_sd:
            res.standard_deduction = val_sd
            fields.append(ExtractedTaxField(
                doc_id=doc_id,
                assessment_id=assessment_id,
                field_name="standard_deduction_16_ia",
                original_ocr_text=raw_sd or "",
                normalized_decimal=val_sd,
                confidence=0.96,
            ))

        # 4. Professional Tax u/s 16(iii)
        raw_pt, val_pt = _find_amount_near(text, [
            r"tax\s*on\s*employment\s*(?:under|u\/s)?\s*sec(?:tion)?\.?\s*16\(iii\)",
            r"tax\s*on\s*employment",
            r"professional\s*tax",
        ])
        if val_pt:
            res.professional_tax = val_pt
            fields.append(ExtractedTaxField(
                doc_id=doc_id,
                assessment_id=assessment_id,
                field_name="professional_tax_16_iii",
                original_ocr_text=raw_pt or "",
                normalized_decimal=val_pt,
                confidence=0.92,
            ))

        # 5. Income chargeable under head Salaries
        raw_net, val_net = _find_amount_near(text, [
            r"income\s*chargeable\s*under\s*the\s*head\s*salaries",
            r"income\s*under\s*the\s*head\s*salaries",
            r"net\s*salary\s*income",
        ])
        if val_net:
            res.net_salary_income = val_net
        elif res.gross_salary > Decimal("0.00"):
            # Compute net salary if not explicitly extracted
            res.net_salary_income = max(
                Decimal("0.00"),
                res.gross_salary - res.exempt_allowances - res.standard_deduction - res.professional_tax
            )

        fields.append(ExtractedTaxField(
            doc_id=doc_id,
            assessment_id=assessment_id,
            field_name="net_salary_income",
            original_ocr_text=raw_net or str(res.net_salary_income),
            normalized_decimal=res.net_salary_income,
            confidence=0.95,
        ))

        # 6. Chapter VI-A Deductions
        for sec in ["80C", "80CCC", "80CCD(1)", "80CCD(1B)", "80CCD(2)", "80D", "80E", "80G", "80TTA"]:
            sec_escaped = re.escape(sec)
            raw_sec, val_sec = _find_amount_near(text, [
                rf"section\s*{sec_escaped}",
                rf"sec\s*{sec_escaped}",
                rf"\b{sec_escaped}\b",
            ])
            if val_sec and val_sec > Decimal("0.00"):
                res.chapter_6a_deductions[sec] = val_sec
                fields.append(ExtractedTaxField(
                    doc_id=doc_id,
                    assessment_id=assessment_id,
                    field_name=f"deduction_{sec.lower().replace('(', '_').replace(')', '')}",
                    original_ocr_text=raw_sec or "",
                    normalized_decimal=val_sec,
                    confidence=0.90,
                ))

        # 7. Total Tax Deducted / TDS
        raw_tds, val_tds = _find_amount_near(text, [
            r"total\s*tax\s*deducted\s*and\s*deposited",
            r"total\s*tax\s*deducted",
            r"tax\s*deducted\s*at\s*source",
            r"tds\s*deposited",
            r"tds",
        ])
        if val_tds:
            res.total_tds_deducted = val_tds
            fields.append(ExtractedTaxField(
                doc_id=doc_id,
                assessment_id=assessment_id,
                field_name="total_tds_deducted",
                original_ocr_text=raw_tds or "",
                normalized_decimal=val_tds,
                confidence=0.96,
            ))

        # 8. Total Taxable Income
        raw_taxable, val_taxable = _find_amount_near(text, [
            r"total\s*taxable\s*income",
            r"taxable\s*income",
            r"gross\s*total\s*income",
        ])
        if val_taxable:
            res.total_taxable_income = val_taxable
            fields.append(ExtractedTaxField(
                doc_id=doc_id,
                assessment_id=assessment_id,
                field_name="total_taxable_income",
                original_ocr_text=raw_taxable or "",
                normalized_decimal=val_taxable,
                confidence=0.92,
            ))

        res.extracted_fields = fields
        return res


# ---------------------------------------------------------------------------
# Bank Interest Certificate Extractor
# ---------------------------------------------------------------------------

@dataclass
class BankInterestExtractionResult:
    bank_name: str = ""
    account_number: str = ""
    savings_interest: Decimal = Decimal("0.00")
    term_deposit_interest: Decimal = Decimal("0.00")
    total_interest: Decimal = Decimal("0.00")
    tds_deducted: Decimal = Decimal("0.00")
    extracted_fields: List[ExtractedTaxField] = field(default_factory=list)


class BankInterestExtractor:
    """Extracts interest earned on savings and fixed deposits."""

    def extract(self, text: str, doc_id: str = "", assessment_id: str = "") -> BankInterestExtractionResult:
        res = BankInterestExtractionResult()
        fields: List[ExtractedTaxField] = []

        # Bank Name
        bank_match = re.search(r"\b([A-Za-z\s]+(?:Bank|Cooperative|Grameena))\b", text, re.IGNORECASE)
        if bank_match:
            res.bank_name = bank_match.group(1).strip()
            fields.append(ExtractedTaxField(
                doc_id=doc_id,
                assessment_id=assessment_id,
                field_name="bank_name",
                original_ocr_text=res.bank_name,
                confidence=0.90,
            ))

        # Account Number
        acc_match = re.search(r"account\s*(?:no\.?|number)[\s:=-]+([A-Za-z0-9X\*-]{6,20})", text, re.IGNORECASE)
        if acc_match:
            res.account_number = acc_match.group(1).strip()

        # Savings Account Interest
        raw_sav, val_sav = _find_amount_near(text, [
            r"savings\s*(?:bank)?\s*(?:account)?\s*interest",
            r"interest\s*on\s*savings\s*deposit",
            r"sb\s*interest",
        ])
        if val_sav:
            res.savings_interest = val_sav
            fields.append(ExtractedTaxField(
                doc_id=doc_id,
                assessment_id=assessment_id,
                field_name="savings_interest",
                original_ocr_text=raw_sav or "",
                normalized_decimal=val_sav,
                confidence=0.94,
            ))

        # Term / Fixed Deposit Interest
        raw_fd, val_fd = _find_amount_near(text, [
            r"fixed\s*deposit\s*interest",
            r"term\s*deposit\s*interest",
            r"interest\s*on\s*td\/fd",
            r"fd\s*interest",
        ])
        if val_fd:
            res.term_deposit_interest = val_fd
            fields.append(ExtractedTaxField(
                doc_id=doc_id,
                assessment_id=assessment_id,
                field_name="term_deposit_interest",
                original_ocr_text=raw_fd or "",
                normalized_decimal=val_fd,
                confidence=0.94,
            ))

        # Total Interest
        raw_tot, val_tot = _find_amount_near(text, [
            r"total\s*interest\s*(?:paid|credited)",
            r"total\s*interest",
        ])
        if val_tot:
            res.total_interest = val_tot
        else:
            res.total_interest = res.savings_interest + res.term_deposit_interest

        fields.append(ExtractedTaxField(
            doc_id=doc_id,
            assessment_id=assessment_id,
            field_name="total_interest",
            original_ocr_text=raw_tot or str(res.total_interest),
            normalized_decimal=res.total_interest,
            confidence=0.95,
        ))

        # TDS Deducted under Section 194A
        raw_tds, val_tds = _find_amount_near(text, [
            r"tax\s*deducted\s*at\s*source",
            r"tds\s*deducted",
            r"tds\s*u\/s\s*194a",
        ])
        if val_tds:
            res.tds_deducted = val_tds
            fields.append(ExtractedTaxField(
                doc_id=doc_id,
                assessment_id=assessment_id,
                field_name="interest_tds_194a",
                original_ocr_text=raw_tds or "",
                normalized_decimal=val_tds,
                confidence=0.92,
            ))

        res.extracted_fields = fields
        return res


# ---------------------------------------------------------------------------
# AIS / TIS Extractor
# ---------------------------------------------------------------------------

@dataclass
class AisTisRecord:
    code: str
    description: str
    amount: Decimal
    reported_by: str = ""
    tds: Decimal = Decimal("0.00")


@dataclass
class AisTisExtractionResult:
    salary_income: Decimal = Decimal("0.00")
    savings_interest: Decimal = Decimal("0.00")
    deposit_interest: Decimal = Decimal("0.00")
    dividend_income: Decimal = Decimal("0.00")
    records: List[AisTisRecord] = field(default_factory=list)
    extracted_fields: List[ExtractedTaxField] = field(default_factory=list)


class AisTisExtractor:
    """Extracts reported income entries from AIS and TIS statements."""

    def extract(self, text: str, doc_id: str = "", assessment_id: str = "") -> AisTisExtractionResult:
        res = AisTisExtractionResult()
        fields: List[ExtractedTaxField] = []

        # Salary income
        raw_sal, val_sal = _find_amount_near(text, [
            r"salary\s*income",
            r"salary\s*u\/s\s*192",
            r"information\s*code\s*s-001",
        ])
        if val_sal:
            res.salary_income = val_sal
            res.records.append(AisTisRecord("S-001", "Salary Income", val_sal))
            fields.append(ExtractedTaxField(
                doc_id=doc_id,
                assessment_id=assessment_id,
                field_name="ais_salary_income",
                original_ocr_text=raw_sal or "",
                normalized_decimal=val_sal,
                confidence=0.92,
            ))

        # Savings interest
        raw_sav, val_sav = _find_amount_near(text, [
            r"interest\s*from\s*savings\s*bank",
            r"information\s*code\s*s-002",
        ])
        if val_sav:
            res.savings_interest = val_sav
            res.records.append(AisTisRecord("S-002", "Savings Bank Interest", val_sav))
            fields.append(ExtractedTaxField(
                doc_id=doc_id,
                assessment_id=assessment_id,
                field_name="ais_savings_interest",
                original_ocr_text=raw_sav or "",
                normalized_decimal=val_sav,
                confidence=0.92,
            ))

        # Deposit interest
        raw_dep, val_dep = _find_amount_near(text, [
            r"interest\s*from\s*deposit",
            r"information\s*code\s*s-003",
        ])
        if val_dep:
            res.deposit_interest = val_dep
            res.records.append(AisTisRecord("S-003", "Deposit Interest", val_dep))
            fields.append(ExtractedTaxField(
                doc_id=doc_id,
                assessment_id=assessment_id,
                field_name="ais_deposit_interest",
                original_ocr_text=raw_dep or "",
                normalized_decimal=val_dep,
                confidence=0.92,
            ))

        # Dividend income
        raw_div, val_div = _find_amount_near(text, [
            r"dividend\s*income",
            r"information\s*code\s*s-004",
        ])
        if val_div:
            res.dividend_income = val_div
            res.records.append(AisTisRecord("S-004", "Dividend Income", val_div))
            fields.append(ExtractedTaxField(
                doc_id=doc_id,
                assessment_id=assessment_id,
                field_name="ais_dividend_income",
                original_ocr_text=raw_div or "",
                normalized_decimal=val_div,
                confidence=0.92,
            ))

        res.extracted_fields = fields
        return res
