"""
Tax Module — Cross-Document Financial Reconciliation Engine (Phase 4).

Reconciles income, deductions, and tax credits across multiple reporting sources
(Form 16, Salary Slips, Bank Interest Certificates, AIS/TIS, Form 26AS) to detect
mismatches, missing records, and prevent catastrophic double-counting.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from tax_module.constants import (
    FilingStatus,
    IncomeHead,
    TaxDocumentCategory,
    VerificationStatus,
)
from tax_module.db import TaxDatabase
from tax_module.models import (
    IncomeRecord,
    ReconciliationRecord,
    TaxCreditRecord,
)

logger = logging.getLogger(__name__)

# Discrepancy threshold: amounts differing by more than 1 Rupee trigger review
DISCREPANCY_TOLERANCE = Decimal("1.00")


class IncomeReconciliationEngine:
    """Detects discrepancies across multiple reporting sources and manages reconciliation."""

    def __init__(self, db: Optional[TaxDatabase] = None):
        self.db = db or TaxDatabase()

    def reconcile_assessment(self, assessment_id: str) -> List[ReconciliationRecord]:
        """
        Execute reconciliation algorithms across all ingested records for an assessment.
        Checks:
        1. Form 16 Salary vs AIS Salary
        2. Form 16 TDS vs 26AS TDS
        3. Bank Interest Certificates vs AIS Savings / Deposit Interest
        4. Overlapping salary slips and Form 16 (Double counting prevention)
        """
        docs = self.db.list_documents_for_assessment(assessment_id)
        incomes = self.db.list_income_records(assessment_id)
        credits = self.db.list_tax_credit_records(assessment_id)
        fields = self.db.list_extracted_fields_for_assessment(assessment_id)

        recon_records: List[ReconciliationRecord] = []

        # -------------------------------------------------------------------
        # 1. Salary Reconciliation (Form 16 vs AIS)
        # -------------------------------------------------------------------
        f16_salary_field = next((f for f in fields if f.field_name == "gross_salary_17_1"), None)
        ais_salary_field = next((f for f in fields if f.field_name == "ais_salary_income"), None)

        if f16_salary_field and f16_salary_field.effective_value and ais_salary_field and ais_salary_field.effective_value:
            f16_val = f16_salary_field.effective_value
            ais_val = ais_salary_field.effective_value
            diff = abs(f16_val - ais_val)
            status = "MATCHED" if diff <= DISCREPANCY_TOLERANCE else "REVIEW_REQUIRED"
            notes = "Salary matches between Form 16 and AIS." if status == "MATCHED" else f"Salary difference of ₹{diff} between Form 16 and AIS. Form 16 is authoritative."
            
            r_salary = ReconciliationRecord(
                assessment_id=assessment_id,
                field_name="Gross Salary",
                primary_source="Form 16 (Sec 17(1))",
                primary_amount=f16_val,
                secondary_source="AIS (Code S-001)",
                secondary_amount=ais_val,
                difference=diff,
                status=status,
                resolution_notes=notes,
            )
            self.db.save_reconciliation_record(r_salary)
            recon_records.append(r_salary)

        # -------------------------------------------------------------------
        # 2. TDS Reconciliation (Form 16 Part A/B vs Form 26AS Part I)
        # -------------------------------------------------------------------
        f16_tds_credit = next((c for c in credits if c.credit_type == "TDS_SALARY"), None)
        f26as_tds_credit = next((c for c in credits if c.credit_type == "TDS_26AS_SALARY"), None)

        if f16_tds_credit and f26as_tds_credit:
            f16_tds = f16_tds_credit.total_tax_deducted
            f26as_tds = f26as_tds_credit.total_tax_deducted
            diff = abs(f16_tds - f26as_tds)
            status = "MATCHED" if diff <= DISCREPANCY_TOLERANCE else "REVIEW_REQUIRED"
            notes = "TDS credit matches." if status == "MATCHED" else "TDS mismatch: Form 26AS takes precedence for tax credit in ITD e-filing system."

            r_tds = ReconciliationRecord(
                assessment_id=assessment_id,
                field_name="TDS from Salary",
                primary_source="Form 16 Part A",
                primary_amount=f16_tds,
                secondary_source="Form 26AS Part I",
                secondary_amount=f26as_tds,
                difference=diff,
                status=status,
                resolution_notes=notes,
            )
            self.db.save_reconciliation_record(r_tds)
            recon_records.append(r_tds)

        # -------------------------------------------------------------------
        # 3. Bank Interest Reconciliation (Certificates vs AIS S-002)
        # -------------------------------------------------------------------
        bank_interest_field = next((f for f in fields if f.field_name == "savings_interest"), None)
        ais_interest_field = next((f for f in fields if f.field_name == "ais_savings_interest"), None)

        if bank_interest_field and bank_interest_field.effective_value and ais_interest_field and ais_interest_field.effective_value:
            b_val = bank_interest_field.effective_value
            ais_val = ais_interest_field.effective_value
            diff = abs(b_val - ais_val)
            status = "MATCHED" if diff <= DISCREPANCY_TOLERANCE else "REVIEW_REQUIRED"
            notes = "Savings interest matches." if status == "MATCHED" else f"Savings interest difference of ₹{diff}. AIS higher may indicate additional bank accounts."

            r_interest = ReconciliationRecord(
                assessment_id=assessment_id,
                field_name="Savings Bank Interest",
                primary_source="Bank Interest Certificate",
                primary_amount=b_val,
                secondary_source="AIS (Code S-002)",
                secondary_amount=ais_val,
                difference=diff,
                status=status,
                resolution_notes=notes,
            )
            self.db.save_reconciliation_record(r_interest)
            recon_records.append(r_interest)

        # -------------------------------------------------------------------
        # 4. Double-Counting Prevention (Form 16 vs Salary Slips)
        # -------------------------------------------------------------------
        has_form16 = any(d.document_category in (TaxDocumentCategory.FORM_16_A, TaxDocumentCategory.FORM_16_B) for d in docs)
        has_payslips = any(d.document_category == TaxDocumentCategory.SALARY_SLIP for d in docs)

        if has_form16 and has_payslips:
            # Overlapping documents detected
            r_overlap = ReconciliationRecord(
                assessment_id=assessment_id,
                field_name="Salary Overlap Warning",
                primary_source="Form 16 (Annual)",
                primary_amount=f16_salary_field.effective_value if (f16_salary_field and f16_salary_field.effective_value) else Decimal("0.00"),
                secondary_source="Monthly Salary Slips",
                secondary_amount=Decimal("0.00"),
                difference=Decimal("0.00"),
                status="REVIEW_REQUIRED",
                resolution_notes="Both Form 16 and Salary Slips are present. Form 16 contains full annual earnings; monthly slips must NOT be added to prevent double counting.",
            )
            self.db.save_reconciliation_record(r_overlap)
            recon_records.append(r_overlap)

        # Update filing status if reconciliation review required
        has_review = any(r.status == "REVIEW_REQUIRED" for r in recon_records)
        assessment = self.db.get_assessment(assessment_id)
        if assessment and has_review and assessment.filing_status in (FilingStatus.DOCUMENTS_UPLOADED, FilingStatus.NOT_STARTED):
            self.db.update_filing_status(
                assessment_id=assessment_id,
                status=FilingStatus.RECONCILIATION_REQUIRED,
                notes=f"Reconciliation engine identified {len([r for r in recon_records if r.status == 'REVIEW_REQUIRED'])} discrepancy flags.",
            )

        return recon_records

    def resolve_reconciliation_record(
        self,
        recon_id: str,
        chosen_source: str,
        override_amount: Optional[Decimal] = None,
        notes: str = "",
    ) -> bool:
        """Resolve a discrepancy flag with explicit taxpayer confirmation."""
        recon_records = self.db.list_reconciliation_records(assessment_id="")
        # Direct SQL update
        sql = """
        UPDATE reconciliation_records
        SET status = 'RESOLVED', resolution_notes = ?
        WHERE recon_id = ?;
        """
        res_note = f"Resolved in favor of {chosen_source}. {notes}".strip()
        if override_amount is not None:
            res_note += f" (Adjusted amount: ₹{override_amount})"

        with self.db.connection() as conn:
            conn.execute(sql, (res_note, recon_id))

        self.db.log_audit_event(
            taxpayer_id="",
            action="RECONCILIATION_RESOLVED",
            details=f"Reconciliation flag {recon_id} resolved: {res_note}",
        )
        return True
