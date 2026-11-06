"""
Tax Module — Tax Document Classification Engine (Phase 3).

Classifies tax-related PDFs and images into specific statutory categories
(Form 16 Part A/B, Salary Slips, Bank Interest, AIS/TIS, 26AS, Investment Proofs)
using textual signatures, header patterns, and keyword evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tax_module.constants import TaxDocumentCategory


@dataclass
class TaxClassificationResult:
    """Document classification outcome with explainability keywords."""
    category: TaxDocumentCategory
    confidence: float
    evidence: List[str] = field(default_factory=list)
    explanation: str = ""


# Canonical signature keywords per tax document category
_CLASSIFICATION_RULES: Dict[TaxDocumentCategory, List[Tuple[str, float]]] = {
    TaxDocumentCategory.FORM_16_A: [
        (r"form\s*no\.?\s*16\s*part\s*a", 0.95),
        (r"certificate\s*under\s*section\s*203", 0.90),
        (r"tan\s*of\s*the\s*deductor", 0.80),
        (r"quarterly\s*statement", 0.60),
        (r"tds\s*deposited", 0.65),
    ],
    TaxDocumentCategory.FORM_16_B: [
        (r"form\s*no\.?\s*16\s*part\s*b", 0.95),
        (r"details\s*of\s*salary\s*paid", 0.90),
        (r"section\s*17\(1\)", 0.85),
        (r"standard\s*deduction\s*under\s*section\s*16\(ia\)", 0.85),
        (r"chapter\s*vi-?a", 0.75),
        (r"total\s*taxable\s*income", 0.70),
    ],
    TaxDocumentCategory.SALARY_SLIP: [
        (r"salary\s*slip|payslip|pay\s*slip", 0.90),
        (r"basic\s*salary|basic\s*pay", 0.80),
        (r"house\s*rent\s*allowance|hra", 0.75),
        (r"provident\s*fund|epf", 0.70),
        (r"net\s*pay|gross\s*earnings", 0.75),
    ],
    TaxDocumentCategory.FORM_26AS: [
        (r"form\s*26\s*as|form\s*no\.?\s*26as", 0.95),
        (r"annual\s*tax\s*statement", 0.90),
        (r"part\s*i\s*-?\s*details\s*of\s*tax\s*deducted\s*at\s*source", 0.85),
        (r"section\s*192|section\s*194a", 0.75),
        (r"tax\s*credit\s*statement", 0.80),
    ],
    TaxDocumentCategory.AIS_TIS: [
        (r"annual\s*information\s*statement|taxpayer\s*information\s*summary", 0.95),
        (r"\bais\b|\btis\b", 0.80),
        (r"information\s*code\s*s-", 0.90),
        (r"income-tax\s*department\s*portal", 0.70),
        (r"interest\s*from\s*savings\s*bank", 0.75),
    ],
    TaxDocumentCategory.BANK_INTEREST_CERT: [
        (r"interest\s*certificate|certificate\s*of\s*interest", 0.95),
        (r"interest\s*paid|interest\s*credited", 0.80),
        (r"savings\s*account|fixed\s*deposit|term\s*deposit", 0.75),
        (r"financial\s*year\s*20\d\d-\d\d", 0.70),
    ],
    TaxDocumentCategory.BANK_STATEMENT: [
        (r"statement\s*of\s*account|account\s*statement", 0.90),
        (r"opening\s*balance|closing\s*balance", 0.85),
        (r"withdrawal|deposit|balance", 0.70),
        (r"value\s*date|txn\s*date", 0.70),
    ],
    TaxDocumentCategory.INVESTMENT_80C: [
        (r"public\s*provident\s*fund|ppf", 0.90),
        (r"equity\s*linked\s*savings\s*scheme|elss", 0.90),
        (r"life\s*insurance\s*premium|lic", 0.85),
        (r"section\s*80c", 0.80),
        (r"sukanya\s*samriddhi", 0.90),
    ],
    TaxDocumentCategory.MEDICAL_80D: [
        (r"health\s*insurance|mediclaim|medical\s*insurance", 0.90),
        (r"section\s*80d", 0.85),
        (r"premium\s*receipt", 0.75),
        (r"preventive\s*health\s*check-?up", 0.80),
    ],
    TaxDocumentCategory.HOME_LOAN_INTEREST: [
        (r"home\s*loan|housing\s*loan", 0.90),
        (r"provisional\s*interest\s*certificate|interest\s*certificate", 0.80),
        (r"section\s*24\(?b\)?", 0.85),
        (r"principal\s*repayment", 0.80),
    ],
    TaxDocumentCategory.TAX_CHALLAN: [
        (r"challan\s*no\.?\s*\/ITNS\s*280|itns\s*280", 0.95),
        (r"advance\s*tax|self-?assessment\s*tax", 0.85),
        (r"bsr\s*code", 0.90),
        (r"challan\s*tender\s*date", 0.85),
    ],
}


class TaxDocumentClassifier:
    """Classifies OCR extracted text of financial tax documents."""

    def classify_text(self, text: str) -> TaxClassificationResult:
        if not text or not text.strip():
            return TaxClassificationResult(
                category=TaxDocumentCategory.OTHER,
                confidence=0.0,
                evidence=[],
                explanation="Empty text provided.",
            )

        text_lower = text.lower()
        best_category = TaxDocumentCategory.OTHER
        best_score = 0.0
        best_evidence: List[str] = []

        for category, rules in _CLASSIFICATION_RULES.items():
            matched_evidence: List[str] = []
            score_sum = 0.0
            max_weight = 0.0

            for pattern, weight in rules:
                if re.search(pattern, text_lower):
                    matched_evidence.append(pattern)
                    score_sum += weight
                    if weight > max_weight:
                        max_weight = weight

            if matched_evidence:
                # Combined score: blend max match weight and cumulative evidence
                evidence_factor = min(len(matched_evidence) / len(rules), 1.0)
                category_score = (max_weight * 0.7) + (evidence_factor * 0.3)

                if category_score > best_score:
                    best_score = category_score
                    best_category = category
                    best_evidence = matched_evidence

        if best_score < 0.40:
            return TaxClassificationResult(
                category=TaxDocumentCategory.OTHER,
                confidence=best_score,
                evidence=best_evidence,
                explanation="No conclusive tax document signature matched; classified as OTHER.",
            )

        explanation = f"Matched {len(best_evidence)} signatures for {best_category.value}."
        return TaxClassificationResult(
            category=best_category,
            confidence=round(best_score, 2),
            evidence=best_evidence,
            explanation=explanation,
        )
