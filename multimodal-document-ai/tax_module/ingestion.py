"""
Tax Module — Tax Document Ingestion Service (Phase 3).

Coordinates SHA-256 duplicate prevention, OCR text extraction, document classification,
itemized field extraction, and database persistence with full source traceability.
"""

from __future__ import annotations

import hashlib
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tax_module.classifiers import TaxDocumentClassifier
from tax_module.constants import (
    FilingStatus,
    IncomeHead,
    TaxDocumentCategory,
    VerificationStatus,
)
from tax_module.db import TaxDatabase
from tax_module.extractors import (
    AisTisExtractor,
    BankInterestExtractor,
    Form16Extractor,
)
from tax_module.models import (
    DeductionRecord,
    IncomeRecord,
    TaxCreditRecord,
    TaxDocument,
)

logger = logging.getLogger(__name__)


def compute_sha256(content: bytes) -> str:
    """Compute SHA-256 hex digest of file bytes."""
    return hashlib.sha256(content).hexdigest()


class TaxDocumentIngestionService:
    """Ingests tax documents, classifies them, extracts fields, and creates income/credit records."""

    def __init__(self, db: Optional[TaxDatabase] = None):
        self.db = db or TaxDatabase()
        self.classifier = TaxDocumentClassifier()
        self.form16_extractor = Form16Extractor()
        self.interest_extractor = BankInterestExtractor()
        self.ais_extractor = AisTisExtractor()

    def ingest_document_text(
        self,
        assessment_id: str,
        text: str,
        filename: str = "document.txt",
        file_bytes: Optional[bytes] = None,
        forced_category: Optional[TaxDocumentCategory] = None,
    ) -> Tuple[Optional[TaxDocument], List[str]]:
        """
        Ingest text content of a tax document (from OCR, PDF text, or uploaded document).
        Computes SHA-256 hash, checks for duplicates, classifies, extracts fields,
        and saves records.
        """
        if not text or not text.strip():
            return None, ["Document text is empty or could not be extracted."]

        assessment = self.db.get_assessment(assessment_id)
        if not assessment:
            return None, [f"Assessment session '{assessment_id}' does not exist."]

        raw_bytes = file_bytes or text.encode("utf-8")
        file_hash = compute_sha256(raw_bytes)

        # Check for duplicate document in the same assessment
        existing_docs = self.db.list_documents_for_assessment(assessment_id)
        for d in existing_docs:
            if d.file_hash == file_hash:
                return None, [f"Duplicate document detected: '{filename}' has already been uploaded for this assessment session (SHA-256: {file_hash[:8]}...)."]

        # Classification
        if forced_category:
            category = forced_category
            conf = 1.0
        else:
            cls_res = self.classifier.classify_text(text)
            category = cls_res.category
            conf = cls_res.confidence

        doc = TaxDocument(
            assessment_id=assessment_id,
            document_category=category,
            filename=filename,
            file_hash=file_hash,
            file_size=len(raw_bytes),
            ocr_raw_text=text,
            ocr_confidence=conf,
            verification_status=VerificationStatus.UNVERIFIED,
        )
        self.db.save_tax_document(doc)

        # Extraction based on category
        self._extract_and_persist(doc, text)

        # Update assessment filing status if NOT_STARTED
        if assessment.filing_status == FilingStatus.NOT_STARTED:
            self.db.update_filing_status(
                assessment_id=assessment_id,
                status=FilingStatus.DOCUMENTS_UPLOADED,
                notes=f"First document '{filename}' uploaded ({category.value})",
            )

        self.db.log_audit_event(
            taxpayer_id=assessment.taxpayer_id,
            assessment_id=assessment_id,
            action="TAX_DOC_INGESTED",
            details=f"Ingested '{filename}' classified as {category.value} (confidence: {conf:.2f})",
        )

        return doc, []

    def _extract_and_persist(self, doc: TaxDocument, text: str) -> None:
        """Route document to appropriate extractor and generate financial records."""
        cat = doc.document_category

        if cat in (TaxDocumentCategory.FORM_16_A, TaxDocumentCategory.FORM_16_B):
            res16 = self.form16_extractor.extract(text, doc_id=doc.doc_id, assessment_id=doc.assessment_id)
            for f in res16.extracted_fields:
                self.db.save_extracted_field(f)

            # Salary Income Record
            if res16.gross_salary > Decimal("0.00") or res16.net_salary_income > Decimal("0.00"):
                inc = IncomeRecord(
                    assessment_id=doc.assessment_id,
                    income_head=IncomeHead.SALARY,
                    source_doc_id=doc.doc_id,
                    source_name=res16.employer_name or "Form 16 Employer",
                    identifier=res16.employer_tan or None,
                    gross_amount=res16.gross_salary,
                    exempt_allowances=res16.exempt_allowances + res16.standard_deduction + res16.professional_tax,
                    net_taxable_amount=res16.net_salary_income,
                    verification_status=VerificationStatus.UNVERIFIED,
                )
                self.db.save_income_record(inc)

            # Deductions from Form 16 Part B
            for sec_code, claimed_amt in res16.chapter_6a_deductions.items():
                ded = DeductionRecord(
                    assessment_id=doc.assessment_id,
                    section_code=sec_code,
                    description=f"Form 16 reported deduction u/s {sec_code}",
                    claimed_amount=claimed_amt,
                    eligible_amount=claimed_amt,
                    source_doc_id=doc.doc_id,
                    verification_status=VerificationStatus.UNVERIFIED,
                )
                self.db.save_deduction_record(ded)

            # TDS credit from Form 16 Part A/B
            if res16.total_tds_deducted > Decimal("0.00"):
                tc = TaxCreditRecord(
                    assessment_id=doc.assessment_id,
                    credit_type="TDS_SALARY",
                    deductor_tan=res16.employer_tan or None,
                    deductor_name=res16.employer_name or "Form 16 Employer",
                    total_tax_deducted=res16.total_tds_deducted,
                    claimed_amount=res16.total_tds_deducted,
                    source_doc_id=doc.doc_id,
                    verification_status=VerificationStatus.UNVERIFIED,
                )
                self.db.save_tax_credit_record(tc)

        elif cat == TaxDocumentCategory.BANK_INTEREST_CERT:
            res_int = self.interest_extractor.extract(text, doc_id=doc.doc_id, assessment_id=doc.assessment_id)
            for f in res_int.extracted_fields:
                self.db.save_extracted_field(f)

            if res_int.total_interest > Decimal("0.00"):
                inc = IncomeRecord(
                    assessment_id=doc.assessment_id,
                    income_head=IncomeHead.OTHER_SOURCES,
                    source_doc_id=doc.doc_id,
                    source_name=res_int.bank_name or "Bank Interest Certificate",
                    identifier=res_int.account_number or None,
                    gross_amount=res_int.total_interest,
                    exempt_allowances=Decimal("0.00"),
                    net_taxable_amount=res_int.total_interest,
                    verification_status=VerificationStatus.UNVERIFIED,
                )
                self.db.save_income_record(inc)

            if res_int.tds_deducted > Decimal("0.00"):
                tc = TaxCreditRecord(
                    assessment_id=doc.assessment_id,
                    credit_type="TDS_OTHER",
                    deductor_name=res_int.bank_name or "Bank",
                    total_tax_deducted=res_int.tds_deducted,
                    claimed_amount=res_int.tds_deducted,
                    source_doc_id=doc.doc_id,
                    verification_status=VerificationStatus.UNVERIFIED,
                )
                self.db.save_tax_credit_record(tc)

        elif cat == TaxDocumentCategory.AIS_TIS:
            res_ais = self.ais_extractor.extract(text, doc_id=doc.doc_id, assessment_id=doc.assessment_id)
            for f in res_ais.extracted_fields:
                self.db.save_extracted_field(f)

            # AIS other source incomes (savings + deposit + dividend)
            other_total = res_ais.savings_interest + res_ais.deposit_interest + res_ais.dividend_income
            if other_total > Decimal("0.00"):
                inc = IncomeRecord(
                    assessment_id=doc.assessment_id,
                    income_head=IncomeHead.OTHER_SOURCES,
                    source_doc_id=doc.doc_id,
                    source_name="AIS / TIS Statement",
                    gross_amount=other_total,
                    exempt_allowances=Decimal("0.00"),
                    net_taxable_amount=other_total,
                    verification_status=VerificationStatus.UNVERIFIED,
                )
                self.db.save_income_record(inc)
