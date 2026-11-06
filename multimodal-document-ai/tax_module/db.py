"""
Tax Module SQLite Persistence and Isolation Layer.

Provides ACID-compliant persistence, schema initialization, audit logging,
and strict multi-taxpayer data isolation. All monetary values are serialized
and deserialized with 100% fidelity using Python Decimal string representation.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from tax_module.constants import (
    DEFAULT_ASSESSMENT_YEAR,
    FilingStatus,
    FormType,
    IncomeHead,
    ResidentialStatus,
    TaxDocumentCategory,
    TaxRegime,
    VerificationStatus,
)
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
)

logger = logging.getLogger(__name__)

# Default database location: <project_root>/data/tax_system.db
_DEFAULT_DB_REL_PATH = Path("data") / "tax_system.db"


# ---------------------------------------------------------------------------
# Database DDL
# ---------------------------------------------------------------------------

_DDL_TAXPAYERS = """
CREATE TABLE IF NOT EXISTS taxpayer_profiles (
    taxpayer_id                 TEXT PRIMARY KEY,
    pan                         TEXT NOT NULL UNIQUE,
    first_name                  TEXT NOT NULL DEFAULT '',
    middle_name                 TEXT NOT NULL DEFAULT '',
    last_name                   TEXT NOT NULL DEFAULT '',
    dob                         TEXT NOT NULL DEFAULT '',
    gender                      TEXT NOT NULL DEFAULT 'M',
    residential_status          TEXT NOT NULL DEFAULT 'RESIDENT',
    email                       TEXT NOT NULL DEFAULT '',
    mobile                      TEXT NOT NULL DEFAULT '',
    address_flat_door_block     TEXT NOT NULL DEFAULT '',
    address_premises_building   TEXT NOT NULL DEFAULT '',
    address_road_street         TEXT NOT NULL DEFAULT '',
    address_area_locality       TEXT NOT NULL DEFAULT '',
    address_city                TEXT NOT NULL DEFAULT '',
    address_state               TEXT NOT NULL DEFAULT '',
    address_pincode             TEXT NOT NULL DEFAULT '',
    bank_accounts_json          TEXT NOT NULL DEFAULT '[]',
    created_at                  TEXT NOT NULL,
    updated_at                  TEXT NOT NULL
);
"""

_DDL_ASSESSMENTS = """
CREATE TABLE IF NOT EXISTS tax_assessments (
    assessment_id           TEXT PRIMARY KEY,
    taxpayer_id             TEXT NOT NULL REFERENCES taxpayer_profiles(taxpayer_id) ON DELETE CASCADE,
    assessment_year         TEXT NOT NULL,
    financial_year          TEXT NOT NULL,
    filing_status           TEXT NOT NULL DEFAULT 'NOT_STARTED',
    selected_regime         TEXT NOT NULL DEFAULT 'NEW',
    itr_form                TEXT NOT NULL DEFAULT 'ITR-1',
    filing_type             TEXT NOT NULL DEFAULT 'ORIGINAL',
    acknowledgement_number  TEXT,
    filing_date             TEXT,
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    UNIQUE(taxpayer_id, assessment_year)
);
"""

_DDL_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS tax_documents (
    doc_id                  TEXT PRIMARY KEY,
    assessment_id           TEXT NOT NULL REFERENCES tax_assessments(assessment_id) ON DELETE CASCADE,
    document_category       TEXT NOT NULL DEFAULT 'OTHER',
    filename                TEXT NOT NULL DEFAULT '',
    file_path               TEXT NOT NULL DEFAULT '',
    file_hash               TEXT NOT NULL DEFAULT '',
    file_size               INTEGER NOT NULL DEFAULT 0,
    ocr_raw_text            TEXT NOT NULL DEFAULT '',
    ocr_confidence          REAL NOT NULL DEFAULT 0.0,
    verification_status     TEXT NOT NULL DEFAULT 'UNVERIFIED',
    uploaded_at             TEXT NOT NULL
);
"""

_DDL_EXTRACTED_FIELDS = """
CREATE TABLE IF NOT EXISTS extracted_tax_fields (
    field_id                TEXT PRIMARY KEY,
    doc_id                  TEXT NOT NULL REFERENCES tax_documents(doc_id) ON DELETE CASCADE,
    assessment_id           TEXT NOT NULL REFERENCES tax_assessments(assessment_id) ON DELETE CASCADE,
    field_name              TEXT NOT NULL,
    original_ocr_text       TEXT NOT NULL DEFAULT '',
    normalized_decimal      TEXT,
    confidence              REAL NOT NULL DEFAULT 0.0,
    page_number             INTEGER NOT NULL DEFAULT 1,
    bounding_box            TEXT,
    verification_status     TEXT NOT NULL DEFAULT 'UNVERIFIED',
    user_corrected_value    TEXT
);
"""

_DDL_INCOME_RECORDS = """
CREATE TABLE IF NOT EXISTS income_records (
    record_id               TEXT PRIMARY KEY,
    assessment_id           TEXT NOT NULL REFERENCES tax_assessments(assessment_id) ON DELETE CASCADE,
    income_head             TEXT NOT NULL DEFAULT 'SALARY',
    source_doc_id           TEXT REFERENCES tax_documents(doc_id) ON DELETE SET NULL,
    source_name             TEXT NOT NULL DEFAULT '',
    identifier              TEXT,
    gross_amount            TEXT NOT NULL DEFAULT '0.00',
    exempt_allowances       TEXT NOT NULL DEFAULT '0.00',
    net_taxable_amount      TEXT NOT NULL DEFAULT '0.00',
    verification_status     TEXT NOT NULL DEFAULT 'UNVERIFIED'
);
"""

_DDL_DEDUCTION_RECORDS = """
CREATE TABLE IF NOT EXISTS deduction_records (
    record_id               TEXT PRIMARY KEY,
    assessment_id           TEXT NOT NULL REFERENCES tax_assessments(assessment_id) ON DELETE CASCADE,
    section_code            TEXT NOT NULL DEFAULT '80C',
    description             TEXT NOT NULL DEFAULT '',
    claimed_amount          TEXT NOT NULL DEFAULT '0.00',
    eligible_amount         TEXT NOT NULL DEFAULT '0.00',
    source_doc_id           TEXT REFERENCES tax_documents(doc_id) ON DELETE SET NULL,
    verification_status     TEXT NOT NULL DEFAULT 'UNVERIFIED'
);
"""

_DDL_TAX_CREDITS = """
CREATE TABLE IF NOT EXISTS tax_credit_records (
    credit_id               TEXT PRIMARY KEY,
    assessment_id           TEXT NOT NULL REFERENCES tax_assessments(assessment_id) ON DELETE CASCADE,
    credit_type             TEXT NOT NULL DEFAULT 'TDS_SALARY',
    deductor_tan            TEXT,
    deductor_name           TEXT,
    total_tax_deducted      TEXT NOT NULL DEFAULT '0.00',
    claimed_amount          TEXT NOT NULL DEFAULT '0.00',
    challan_bsr_code        TEXT,
    challan_serial_no       TEXT,
    challan_date            TEXT,
    source_doc_id           TEXT REFERENCES tax_documents(doc_id) ON DELETE SET NULL,
    verification_status     TEXT NOT NULL DEFAULT 'UNVERIFIED'
);
"""

_DDL_RECONCILIATION = """
CREATE TABLE IF NOT EXISTS reconciliation_records (
    recon_id                TEXT PRIMARY KEY,
    assessment_id           TEXT NOT NULL REFERENCES tax_assessments(assessment_id) ON DELETE CASCADE,
    field_name              TEXT NOT NULL,
    primary_source          TEXT NOT NULL DEFAULT '',
    primary_amount          TEXT NOT NULL DEFAULT '0.00',
    secondary_source        TEXT NOT NULL DEFAULT '',
    secondary_amount        TEXT NOT NULL DEFAULT '0.00',
    difference              TEXT NOT NULL DEFAULT '0.00',
    status                  TEXT NOT NULL DEFAULT 'MATCHED',
    resolution_notes        TEXT NOT NULL DEFAULT ''
);
"""

_DDL_TAX_CALCULATIONS = """
CREATE TABLE IF NOT EXISTS tax_calculations (
    calc_id                 TEXT PRIMARY KEY,
    assessment_id           TEXT NOT NULL REFERENCES tax_assessments(assessment_id) ON DELETE CASCADE,
    regime                  TEXT NOT NULL DEFAULT 'NEW',
    gross_total_income      TEXT NOT NULL DEFAULT '0.00',
    total_deductions        TEXT NOT NULL DEFAULT '0.00',
    total_taxable_income    TEXT NOT NULL DEFAULT '0.00',
    tax_on_taxable_income   TEXT NOT NULL DEFAULT '0.00',
    rebate_87a              TEXT NOT NULL DEFAULT '0.00',
    tax_after_rebate        TEXT NOT NULL DEFAULT '0.00',
    surcharge               TEXT NOT NULL DEFAULT '0.00',
    cess                    TEXT NOT NULL DEFAULT '0.00',
    total_tax_liability     TEXT NOT NULL DEFAULT '0.00',
    total_tax_credits       TEXT NOT NULL DEFAULT '0.00',
    refund_or_payable       TEXT NOT NULL DEFAULT '0.00',
    interest_234a           TEXT NOT NULL DEFAULT '0.00',
    interest_234b           TEXT NOT NULL DEFAULT '0.00',
    interest_234c           TEXT NOT NULL DEFAULT '0.00',
    breakdown_json          TEXT NOT NULL DEFAULT '{}',
    calculated_at           TEXT NOT NULL
);
"""

_DDL_ITR_RETURNS = """
CREATE TABLE IF NOT EXISTS itr_returns (
    return_id               TEXT PRIMARY KEY,
    assessment_id           TEXT NOT NULL REFERENCES tax_assessments(assessment_id) ON DELETE CASCADE,
    assessment_year         TEXT NOT NULL,
    itr_form                TEXT NOT NULL DEFAULT 'ITR-1',
    official_json           TEXT NOT NULL DEFAULT '{}',
    schema_version          TEXT NOT NULL DEFAULT '1.0',
    is_valid                INTEGER NOT NULL DEFAULT 0,
    validation_errors       TEXT NOT NULL DEFAULT '[]',
    generated_at            TEXT NOT NULL
);
"""

_DDL_AUDIT_LOG = """
CREATE TABLE IF NOT EXISTS tax_audit_log (
    event_id                TEXT PRIMARY KEY,
    taxpayer_id             TEXT NOT NULL,
    assessment_id           TEXT,
    action                  TEXT NOT NULL,
    timestamp               TEXT NOT NULL,
    details                 TEXT NOT NULL DEFAULT ''
);
"""

_ALL_DDL = [
    _DDL_TAXPAYERS,
    _DDL_ASSESSMENTS,
    _DDL_DOCUMENTS,
    _DDL_EXTRACTED_FIELDS,
    _DDL_INCOME_RECORDS,
    _DDL_DEDUCTION_RECORDS,
    _DDL_TAX_CREDITS,
    _DDL_RECONCILIATION,
    _DDL_TAX_CALCULATIONS,
    _DDL_ITR_RETURNS,
    _DDL_AUDIT_LOG,
]


class TaxDatabase:
    """SQLite DAO managing all tax preparation entities and audit logs."""

    def __init__(self, db_path: Optional[Path | str] = None, project_root: Optional[Path | str] = None):
        if db_path is not None:
            self.db_path = Path(db_path)
        elif project_root is not None:
            self.db_path = Path(project_root) / _DEFAULT_DB_REL_PATH
        else:
            # Sibling to tax_module/.. (project root)
            self.db_path = Path(__file__).resolve().parent.parent / _DEFAULT_DB_REL_PATH

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connection() as conn:
            for ddl in _ALL_DDL:
                conn.execute(ddl)

    # -----------------------------------------------------------------------
    # Taxpayer Profile Operations
    # -----------------------------------------------------------------------

    def save_taxpayer_profile(self, profile: TaxpayerProfile) -> None:
        profile.updated_at = datetime.now(timezone.utc).isoformat()
        banks_json = json.dumps([b.to_dict() for b in profile.bank_accounts])
        sql = """
        INSERT INTO taxpayer_profiles (
            taxpayer_id, pan, first_name, middle_name, last_name, dob, gender,
            residential_status, email, mobile,
            address_flat_door_block, address_premises_building, address_road_street,
            address_area_locality, address_city, address_state, address_pincode,
            bank_accounts_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(taxpayer_id) DO UPDATE SET
            pan = excluded.pan,
            first_name = excluded.first_name,
            middle_name = excluded.middle_name,
            last_name = excluded.last_name,
            dob = excluded.dob,
            gender = excluded.gender,
            residential_status = excluded.residential_status,
            email = excluded.email,
            mobile = excluded.mobile,
            address_flat_door_block = excluded.address_flat_door_block,
            address_premises_building = excluded.address_premises_building,
            address_road_street = excluded.address_road_street,
            address_area_locality = excluded.address_area_locality,
            address_city = excluded.address_city,
            address_state = excluded.address_state,
            address_pincode = excluded.address_pincode,
            bank_accounts_json = excluded.bank_accounts_json,
            updated_at = excluded.updated_at;
        """
        with self.connection() as conn:
            conn.execute(sql, (
                profile.taxpayer_id,
                profile.pan.strip().upper(),
                profile.first_name,
                profile.middle_name,
                profile.last_name,
                profile.dob,
                profile.gender,
                profile.residential_status.value,
                profile.email,
                profile.mobile,
                profile.address_flat_door_block,
                profile.address_premises_building,
                profile.address_road_street,
                profile.address_area_locality,
                profile.address_city,
                profile.address_state,
                profile.address_pincode,
                banks_json,
                profile.created_at,
                profile.updated_at,
            ))

    def get_taxpayer_profile(self, taxpayer_id: str) -> Optional[TaxpayerProfile]:
        sql = "SELECT * FROM taxpayer_profiles WHERE taxpayer_id = ?;"
        with self.connection() as conn:
            row = conn.execute(sql, (taxpayer_id,)).fetchone()
            if not row:
                return None
            return self._row_to_taxpayer(row)

    def get_taxpayer_by_pan(self, pan: str) -> Optional[TaxpayerProfile]:
        sql = "SELECT * FROM taxpayer_profiles WHERE pan = ?;"
        with self.connection() as conn:
            row = conn.execute(sql, (pan.strip().upper(),)).fetchone()
            if not row:
                return None
            return self._row_to_taxpayer(row)

    def list_taxpayers(self) -> List[TaxpayerProfile]:
        sql = "SELECT * FROM taxpayer_profiles ORDER BY created_at DESC;"
        with self.connection() as conn:
            rows = conn.execute(sql).fetchall()
            return [self._row_to_taxpayer(r) for r in rows]

    def _row_to_taxpayer(self, row: sqlite3.Row) -> TaxpayerProfile:
        d = dict(row)
        banks_raw = d.pop("bank_accounts_json", "[]")
        try:
            bank_list = json.loads(banks_raw)
        except Exception:
            bank_list = []
        d["bank_accounts"] = bank_list
        return TaxpayerProfile.from_dict(d)

    # -----------------------------------------------------------------------
    # Tax Assessment Operations
    # -----------------------------------------------------------------------

    def save_assessment(self, assessment: TaxAssessment) -> None:
        assessment.updated_at = datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO tax_assessments (
            assessment_id, taxpayer_id, assessment_year, financial_year,
            filing_status, selected_regime, itr_form, filing_type,
            acknowledgement_number, filing_date, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(assessment_id) DO UPDATE SET
            filing_status = excluded.filing_status,
            selected_regime = excluded.selected_regime,
            itr_form = excluded.itr_form,
            filing_type = excluded.filing_type,
            acknowledgement_number = excluded.acknowledgement_number,
            filing_date = excluded.filing_date,
            updated_at = excluded.updated_at;
        """
        with self.connection() as conn:
            conn.execute(sql, (
                assessment.assessment_id,
                assessment.taxpayer_id,
                assessment.assessment_year,
                assessment.financial_year,
                assessment.filing_status.value,
                assessment.selected_regime.value,
                assessment.itr_form.value,
                assessment.filing_type,
                assessment.acknowledgement_number,
                assessment.filing_date,
                assessment.created_at,
                assessment.updated_at,
            ))

    def get_assessment(self, assessment_id: str) -> Optional[TaxAssessment]:
        sql = "SELECT * FROM tax_assessments WHERE assessment_id = ?;"
        with self.connection() as conn:
            row = conn.execute(sql, (assessment_id,)).fetchone()
            if not row:
                return None
            return TaxAssessment.from_dict(dict(row))

    def get_assessment_for_year(self, taxpayer_id: str, assessment_year: str) -> Optional[TaxAssessment]:
        sql = "SELECT * FROM tax_assessments WHERE taxpayer_id = ? AND assessment_year = ?;"
        with self.connection() as conn:
            row = conn.execute(sql, (taxpayer_id, assessment_year)).fetchone()
            if not row:
                return None
            return TaxAssessment.from_dict(dict(row))

    def list_assessments_for_taxpayer(self, taxpayer_id: str) -> List[TaxAssessment]:
        sql = "SELECT * FROM tax_assessments WHERE taxpayer_id = ? ORDER BY assessment_year DESC;"
        with self.connection() as conn:
            rows = conn.execute(sql, (taxpayer_id,)).fetchall()
            return [TaxAssessment.from_dict(dict(r)) for r in rows]

    def update_filing_status(self, assessment_id: str, status: FilingStatus, notes: str = "") -> None:
        sql = """
        UPDATE tax_assessments
        SET filing_status = ?, updated_at = ?
        WHERE assessment_id = ?;
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            conn.execute(sql, (status.value, now, assessment_id))
        self.log_audit_event(
            taxpayer_id="",
            assessment_id=assessment_id,
            action=f"STATUS_CHANGED_{status.value}",
            details=notes or f"Filing status transitioned to {status.value}",
        )

    # -----------------------------------------------------------------------
    # Document Operations
    # -----------------------------------------------------------------------

    def save_tax_document(self, doc: TaxDocument) -> None:
        sql = """
        INSERT INTO tax_documents (
            doc_id, assessment_id, document_category, filename, file_path,
            file_hash, file_size, ocr_raw_text, ocr_confidence,
            verification_status, uploaded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            document_category = excluded.document_category,
            ocr_raw_text = excluded.ocr_raw_text,
            ocr_confidence = excluded.ocr_confidence,
            verification_status = excluded.verification_status;
        """
        with self.connection() as conn:
            conn.execute(sql, (
                doc.doc_id,
                doc.assessment_id,
                doc.document_category.value,
                doc.filename,
                doc.file_path,
                doc.file_hash,
                doc.file_size,
                doc.ocr_raw_text,
                doc.ocr_confidence,
                doc.verification_status.value,
                doc.uploaded_at,
            ))

    def get_tax_document(self, doc_id: str) -> Optional[TaxDocument]:
        sql = "SELECT * FROM tax_documents WHERE doc_id = ?;"
        with self.connection() as conn:
            row = conn.execute(sql, (doc_id,)).fetchone()
            if not row:
                return None
            return TaxDocument.from_dict(dict(row))

    def list_documents_for_assessment(self, assessment_id: str) -> List[TaxDocument]:
        sql = "SELECT * FROM tax_documents WHERE assessment_id = ? ORDER BY uploaded_at ASC;"
        with self.connection() as conn:
            rows = conn.execute(sql, (assessment_id,)).fetchall()
            return [TaxDocument.from_dict(dict(r)) for r in rows]

    def delete_tax_document(self, doc_id: str) -> None:
        sql = "DELETE FROM tax_documents WHERE doc_id = ?;"
        with self.connection() as conn:
            conn.execute(sql, (doc_id,))

    # -----------------------------------------------------------------------
    # Extracted Field Operations
    # -----------------------------------------------------------------------

    def save_extracted_field(self, f: ExtractedTaxField) -> None:
        norm_str = str(f.normalized_decimal) if f.normalized_decimal is not None else None
        corr_str = str(f.user_corrected_value) if f.user_corrected_value is not None else None
        bbox_str = json.dumps(f.bounding_box) if f.bounding_box else None
        sql = """
        INSERT INTO extracted_tax_fields (
            field_id, doc_id, assessment_id, field_name, original_ocr_text,
            normalized_decimal, confidence, page_number, bounding_box,
            verification_status, user_corrected_value
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(field_id) DO UPDATE SET
            field_name = excluded.field_name,
            original_ocr_text = excluded.original_ocr_text,
            normalized_decimal = excluded.normalized_decimal,
            confidence = excluded.confidence,
            page_number = excluded.page_number,
            bounding_box = excluded.bounding_box,
            verification_status = excluded.verification_status,
            user_corrected_value = excluded.user_corrected_value;
        """
        with self.connection() as conn:
            conn.execute(sql, (
                f.field_id,
                f.doc_id,
                f.assessment_id,
                f.field_name,
                f.original_ocr_text,
                norm_str,
                f.confidence,
                f.page_number,
                bbox_str,
                f.verification_status.value,
                corr_str,
            ))

    def list_extracted_fields_for_doc(self, doc_id: str) -> List[ExtractedTaxField]:
        sql = "SELECT * FROM extracted_tax_fields WHERE doc_id = ? ORDER BY page_number ASC;"
        with self.connection() as conn:
            rows = conn.execute(sql, (doc_id,)).fetchall()
            return [ExtractedTaxField.from_dict(dict(r)) for r in rows]

    def list_extracted_fields_for_assessment(self, assessment_id: str) -> List[ExtractedTaxField]:
        sql = "SELECT * FROM extracted_tax_fields WHERE assessment_id = ?;"
        with self.connection() as conn:
            rows = conn.execute(sql, (assessment_id,)).fetchall()
            return [ExtractedTaxField.from_dict(dict(r)) for r in rows]

    def update_field_correction(self, field_id: str, corrected_value: Decimal, notes: str = "") -> None:
        sql = """
        UPDATE extracted_tax_fields
        SET user_corrected_value = ?, verification_status = ?
        WHERE field_id = ?;
        """
        with self.connection() as conn:
            conn.execute(sql, (str(corrected_value), VerificationStatus.USER_CORRECTED.value, field_id))

    # -----------------------------------------------------------------------
    # Income, Deduction, and Tax Credit Operations
    # -----------------------------------------------------------------------

    def save_income_record(self, inc: IncomeRecord) -> None:
        sql = """
        INSERT INTO income_records (
            record_id, assessment_id, income_head, source_doc_id, source_name,
            identifier, gross_amount, exempt_allowances, net_taxable_amount,
            verification_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(record_id) DO UPDATE SET
            gross_amount = excluded.gross_amount,
            exempt_allowances = excluded.exempt_allowances,
            net_taxable_amount = excluded.net_taxable_amount,
            verification_status = excluded.verification_status;
        """
        with self.connection() as conn:
            conn.execute(sql, (
                inc.record_id,
                inc.assessment_id,
                inc.income_head.value,
                inc.source_doc_id,
                inc.source_name,
                inc.identifier,
                str(inc.gross_amount),
                str(inc.exempt_allowances),
                str(inc.net_taxable_amount),
                inc.verification_status.value,
            ))

    def list_income_records(self, assessment_id: str) -> List[IncomeRecord]:
        sql = "SELECT * FROM income_records WHERE assessment_id = ?;"
        with self.connection() as conn:
            rows = conn.execute(sql, (assessment_id,)).fetchall()
            return [IncomeRecord.from_dict(dict(r)) for r in rows]

    def delete_income_record(self, record_id: str) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM income_records WHERE record_id = ?;", (record_id,))

    def save_deduction_record(self, ded: DeductionRecord) -> None:
        sql = """
        INSERT INTO deduction_records (
            record_id, assessment_id, section_code, description,
            claimed_amount, eligible_amount, source_doc_id, verification_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(record_id) DO UPDATE SET
            claimed_amount = excluded.claimed_amount,
            eligible_amount = excluded.eligible_amount,
            verification_status = excluded.verification_status;
        """
        with self.connection() as conn:
            conn.execute(sql, (
                ded.record_id,
                ded.assessment_id,
                ded.section_code,
                ded.description,
                str(ded.claimed_amount),
                str(ded.eligible_amount),
                ded.source_doc_id,
                ded.verification_status.value,
            ))

    def list_deduction_records(self, assessment_id: str) -> List[DeductionRecord]:
        sql = "SELECT * FROM deduction_records WHERE assessment_id = ?;"
        with self.connection() as conn:
            rows = conn.execute(sql, (assessment_id,)).fetchall()
            return [DeductionRecord.from_dict(dict(r)) for r in rows]

    def save_tax_credit_record(self, tc: TaxCreditRecord) -> None:
        sql = """
        INSERT INTO tax_credit_records (
            credit_id, assessment_id, credit_type, deductor_tan, deductor_name,
            total_tax_deducted, claimed_amount, challan_bsr_code, challan_serial_no,
            challan_date, source_doc_id, verification_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(credit_id) DO UPDATE SET
            total_tax_deducted = excluded.total_tax_deducted,
            claimed_amount = excluded.claimed_amount,
            verification_status = excluded.verification_status;
        """
        with self.connection() as conn:
            conn.execute(sql, (
                tc.credit_id,
                tc.assessment_id,
                tc.credit_type,
                tc.deductor_tan,
                tc.deductor_name,
                str(tc.total_tax_deducted),
                str(tc.claimed_amount),
                tc.challan_bsr_code,
                tc.challan_serial_no,
                tc.challan_date,
                tc.source_doc_id,
                tc.verification_status.value,
            ))

    def list_tax_credit_records(self, assessment_id: str) -> List[TaxCreditRecord]:
        sql = "SELECT * FROM tax_credit_records WHERE assessment_id = ?;"
        with self.connection() as conn:
            rows = conn.execute(sql, (assessment_id,)).fetchall()
            return [TaxCreditRecord.from_dict(dict(r)) for r in rows]

    # -----------------------------------------------------------------------
    # Reconciliation Operations
    # -----------------------------------------------------------------------

    def save_reconciliation_record(self, r: ReconciliationRecord) -> None:
        sql = """
        INSERT INTO reconciliation_records (
            recon_id, assessment_id, field_name, primary_source, primary_amount,
            secondary_source, secondary_amount, difference, status, resolution_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(recon_id) DO UPDATE SET
            status = excluded.status,
            resolution_notes = excluded.resolution_notes;
        """
        with self.connection() as conn:
            conn.execute(sql, (
                r.recon_id,
                r.assessment_id,
                r.field_name,
                r.primary_source,
                str(r.primary_amount),
                r.secondary_source,
                str(r.secondary_amount),
                str(r.difference),
                r.status,
                r.resolution_notes,
            ))

    def list_reconciliation_records(self, assessment_id: str) -> List[ReconciliationRecord]:
        sql = "SELECT * FROM reconciliation_records WHERE assessment_id = ?;"
        with self.connection() as conn:
            rows = conn.execute(sql, (assessment_id,)).fetchall()
            return [ReconciliationRecord.from_dict(dict(r)) for r in rows]

    # -----------------------------------------------------------------------
    # Tax Calculation and Return Operations
    # -----------------------------------------------------------------------

    def save_tax_calculation(self, c: TaxCalculation) -> None:
        sql = """
        INSERT INTO tax_calculations (
            calc_id, assessment_id, regime, gross_total_income, total_deductions,
            total_taxable_income, tax_on_taxable_income, rebate_87a, tax_after_rebate,
            surcharge, cess, total_tax_liability, total_tax_credits,
            refund_or_payable, interest_234a, interest_234b, interest_234c,
            breakdown_json, calculated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(calc_id) DO UPDATE SET
            gross_total_income = excluded.gross_total_income,
            total_deductions = excluded.total_deductions,
            total_taxable_income = excluded.total_taxable_income,
            tax_on_taxable_income = excluded.tax_on_taxable_income,
            rebate_87a = excluded.rebate_87a,
            tax_after_rebate = excluded.tax_after_rebate,
            surcharge = excluded.surcharge,
            cess = excluded.cess,
            total_tax_liability = excluded.total_tax_liability,
            total_tax_credits = excluded.total_tax_credits,
            refund_or_payable = excluded.refund_or_payable,
            interest_234a = excluded.interest_234a,
            interest_234b = excluded.interest_234b,
            interest_234c = excluded.interest_234c,
            breakdown_json = excluded.breakdown_json,
            calculated_at = excluded.calculated_at;
        """
        with self.connection() as conn:
            conn.execute(sql, (
                c.calc_id,
                c.assessment_id,
                c.regime.value,
                str(c.gross_total_income),
                str(c.total_deductions),
                str(c.total_taxable_income),
                str(c.tax_on_taxable_income),
                str(c.rebate_87a),
                str(c.tax_after_rebate),
                str(c.surcharge),
                str(c.cess),
                str(c.total_tax_liability),
                str(c.total_tax_credits),
                str(c.refund_or_payable),
                str(c.interest_234a),
                str(c.interest_234b),
                str(c.interest_234c),
                c.breakdown_json,
                c.calculated_at,
            ))

    def get_latest_tax_calculation(self, assessment_id: str, regime: Optional[TaxRegime] = None) -> Optional[TaxCalculation]:
        if regime:
            sql = "SELECT * FROM tax_calculations WHERE assessment_id = ? AND regime = ? ORDER BY calculated_at DESC LIMIT 1;"
            params = (assessment_id, regime.value)
        else:
            sql = "SELECT * FROM tax_calculations WHERE assessment_id = ? ORDER BY calculated_at DESC LIMIT 1;"
            params = (assessment_id,)
        with self.connection() as conn:
            row = conn.execute(sql, params).fetchone()
            if not row:
                return None
            return TaxCalculation.from_dict(dict(row))

    def save_itr_return(self, ret: ITRReturn) -> None:
        sql = """
        INSERT INTO itr_returns (
            return_id, assessment_id, assessment_year, itr_form, official_json,
            schema_version, is_valid, validation_errors, generated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(return_id) DO UPDATE SET
            official_json = excluded.official_json,
            is_valid = excluded.is_valid,
            validation_errors = excluded.validation_errors,
            generated_at = excluded.generated_at;
        """
        with self.connection() as conn:
            conn.execute(sql, (
                ret.return_id,
                ret.assessment_id,
                ret.assessment_year,
                ret.itr_form.value,
                ret.official_json,
                ret.schema_version,
                1 if ret.is_valid else 0,
                json.dumps(ret.validation_errors),
                ret.generated_at,
            ))

    def get_latest_itr_return(self, assessment_id: str) -> Optional[ITRReturn]:
        sql = "SELECT * FROM itr_returns WHERE assessment_id = ? ORDER BY generated_at DESC LIMIT 1;"
        with self.connection() as conn:
            row = conn.execute(sql, (assessment_id,)).fetchone()
            if not row:
                return None
            return ITRReturn.from_dict(dict(row))

    # -----------------------------------------------------------------------
    # Audit Trail
    # -----------------------------------------------------------------------

    def log_audit_event(
        self,
        taxpayer_id: str,
        action: str,
        details: str = "",
        assessment_id: Optional[str] = None,
    ) -> None:
        event = TaxAuditEvent(
            taxpayer_id=taxpayer_id,
            assessment_id=assessment_id,
            action=action,
            details=details,
        )
        sql = """
        INSERT INTO tax_audit_log (event_id, taxpayer_id, assessment_id, action, timestamp, details)
        VALUES (?, ?, ?, ?, ?, ?);
        """
        with self.connection() as conn:
            conn.execute(sql, (
                event.event_id,
                event.taxpayer_id,
                event.assessment_id,
                event.action,
                event.timestamp,
                event.details,
            ))

    def get_audit_trail(self, assessment_id: Optional[str] = None, taxpayer_id: Optional[str] = None) -> List[TaxAuditEvent]:
        if assessment_id:
            sql = "SELECT * FROM tax_audit_log WHERE assessment_id = ? ORDER BY timestamp DESC;"
            params = (assessment_id,)
        elif taxpayer_id:
            sql = "SELECT * FROM tax_audit_log WHERE taxpayer_id = ? ORDER BY timestamp DESC;"
            params = (taxpayer_id,)
        else:
            sql = "SELECT * FROM tax_audit_log ORDER BY timestamp DESC LIMIT 100;"
            params = ()
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [TaxAuditEvent.from_dict(dict(r)) for r in rows]
