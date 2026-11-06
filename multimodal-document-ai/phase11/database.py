"""
Phase 11 — SQLite Persistence Layer.

Provides CRUD operations, full-text and filtered search, and analytics
queries for FinancialDocument records.

Database location: <project_root>/data/financial_documents.db
(auto-created on first use, including parent directory).

Schema
------
documents        — one row per FinancialDocument
line_items       — child rows linked to documents.document_id
processing_log   — immutable audit trail of all writes

All monetary columns are stored as REAL (Python float).
Lists (classification_evidence, anomaly_flags) are stored as JSON strings.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from phase11.schema import FinancialDocument, LineItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    document_id              TEXT PRIMARY KEY,
    filename                 TEXT NOT NULL DEFAULT '',
    source_file              TEXT NOT NULL DEFAULT '',
    document_type            TEXT NOT NULL DEFAULT 'invoice',
    benchmark_domain         TEXT NOT NULL DEFAULT '',
    model_name               TEXT NOT NULL DEFAULT '',

    category                 TEXT NOT NULL DEFAULT 'Unknown',
    sub_category             TEXT,
    classification_confidence REAL NOT NULL DEFAULT 0.0,
    classification_evidence  TEXT NOT NULL DEFAULT '[]',

    vendor_name              TEXT,
    address                  TEXT,
    invoice_number           TEXT,
    invoice_date             TEXT,
    due_date                 TEXT,

    subtotal                 REAL,
    tax_amount               REAL,
    total_amount             REAL,
    currency                 TEXT NOT NULL DEFAULT 'INR',

    extraction_confidence    REAL NOT NULL DEFAULT 0.0,
    ocr_word_count           INTEGER NOT NULL DEFAULT 0,
    ocr_raw_text             TEXT NOT NULL DEFAULT '',

    verification_status      TEXT NOT NULL DEFAULT 'pending',
    processing_timestamp     TEXT NOT NULL,
    created_at               TEXT NOT NULL,
    updated_at               TEXT NOT NULL,
    notes                    TEXT NOT NULL DEFAULT '',
    anomaly_flags            TEXT NOT NULL DEFAULT '[]',

    -- Phase 12 Ingestion & Processing Metadata
    file_hash                TEXT,
    file_size                INTEGER,
    processing_status        TEXT NOT NULL DEFAULT 'COMPLETED',
    processing_stage         TEXT,
    processing_error         TEXT,
    processing_started_at    TEXT,
    processing_completed_at  TEXT,
    processing_duration      REAL,

    -- Phase 12 Classification Review & Explainability
    ai_category              TEXT,
    ai_confidence            REAL,
    final_category           TEXT,
    requires_review          INTEGER NOT NULL DEFAULT 0,
    classification_reason    TEXT NOT NULL DEFAULT '',

    -- Phase 13 AI Value Preservation & Auditable Editing
    ai_vendor_name           TEXT,
    ai_invoice_date          TEXT,
    ai_invoice_number        TEXT,
    ai_total_amount          REAL,
    is_modified              INTEGER NOT NULL DEFAULT 0
);
"""

_DDL_LINE_ITEMS = """
CREATE TABLE IF NOT EXISTS line_items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    description   TEXT NOT NULL DEFAULT '',
    quantity      REAL,
    unit_price    REAL,
    total         REAL,
    unit          TEXT
);
"""

_DDL_PROCESSING_LOG = """
CREATE TABLE IF NOT EXISTS processing_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   TEXT NOT NULL,
    action        TEXT NOT NULL,   -- 'insert' | 'update' | 'delete' | 'CLASSIFICATION_MODIFIED' | 'FIELD_UPDATED' | 'RECORD_EDITED'
    timestamp     TEXT NOT NULL,
    details       TEXT NOT NULL DEFAULT ''
);
"""

_DDL_ANOMALY_FINDINGS = """
CREATE TABLE IF NOT EXISTS anomaly_findings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    anomaly_type  TEXT NOT NULL,
    severity      TEXT NOT NULL,
    score         REAL NOT NULL,
    confidence    REAL NOT NULL,
    explanation   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'OPEN',
    detected_at   TEXT NOT NULL
);
"""

_DDL_DUPLICATE_CANDIDATES = """
CREATE TABLE IF NOT EXISTS duplicate_candidates (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id         TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    matched_document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    duplicate_type      TEXT NOT NULL,
    score               REAL NOT NULL,
    confidence          REAL NOT NULL,
    matched_fields      TEXT NOT NULL DEFAULT '[]',
    explanation         TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'OPEN',
    detected_at         TEXT NOT NULL
);
"""

_DDL_ANOMALY_REVIEW_HISTORY = """
CREATE TABLE IF NOT EXISTS anomaly_review_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   TEXT NOT NULL,
    entity_type   TEXT NOT NULL DEFAULT 'anomaly', -- 'anomaly' | 'duplicate' | 'document'
    entity_id     INTEGER,                         -- anomaly_findings.id or duplicate_candidates.id
    old_status    TEXT NOT NULL,
    new_status    TEXT NOT NULL,
    reviewer      TEXT NOT NULL,
    comment       TEXT NOT NULL DEFAULT '',
    timestamp     TEXT NOT NULL
);
"""

_DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_documents_vendor      ON documents(vendor_name);",
    "CREATE INDEX IF NOT EXISTS idx_documents_category    ON documents(category);",
    "CREATE INDEX IF NOT EXISTS idx_documents_date        ON documents(invoice_date);",
    "CREATE INDEX IF NOT EXISTS idx_documents_status      ON documents(verification_status);",
    "CREATE INDEX IF NOT EXISTS idx_documents_amount      ON documents(total_amount);",
    "CREATE INDEX IF NOT EXISTS idx_documents_file_hash   ON documents(file_hash);",
    "CREATE INDEX IF NOT EXISTS idx_documents_proc_status ON documents(processing_status);",
    "CREATE INDEX IF NOT EXISTS idx_documents_req_review  ON documents(requires_review);",
    "CREATE INDEX IF NOT EXISTS idx_documents_inv_num     ON documents(invoice_number);",
    "CREATE INDEX IF NOT EXISTS idx_documents_cat_date    ON documents(category, invoice_date);",
    "CREATE INDEX IF NOT EXISTS idx_documents_modified    ON documents(is_modified);",
    "CREATE INDEX IF NOT EXISTS idx_line_items_doc        ON line_items(document_id);",
    "CREATE INDEX IF NOT EXISTS idx_anomaly_findings_doc      ON anomaly_findings(document_id);",
    "CREATE INDEX IF NOT EXISTS idx_anomaly_findings_type     ON anomaly_findings(anomaly_type);",
    "CREATE INDEX IF NOT EXISTS idx_anomaly_findings_severity ON anomaly_findings(severity);",
    "CREATE INDEX IF NOT EXISTS idx_anomaly_findings_status   ON anomaly_findings(status);",
    "CREATE INDEX IF NOT EXISTS idx_anomaly_findings_score    ON anomaly_findings(score);",
    "CREATE INDEX IF NOT EXISTS idx_dup_candidates_doc        ON duplicate_candidates(document_id);",
    "CREATE INDEX IF NOT EXISTS idx_dup_candidates_matched    ON duplicate_candidates(matched_document_id);",
    "CREATE INDEX IF NOT EXISTS idx_dup_candidates_type       ON duplicate_candidates(duplicate_type);",
    "CREATE INDEX IF NOT EXISTS idx_dup_candidates_status     ON duplicate_candidates(status);",
    "CREATE INDEX IF NOT EXISTS idx_review_history_doc        ON anomaly_review_history(document_id);",
]


# ---------------------------------------------------------------------------
# Row → FinancialDocument conversion helpers
# ---------------------------------------------------------------------------

def _row_to_doc(row: sqlite3.Row, line_items: List[sqlite3.Row]) -> FinancialDocument:
    """Convert a SQLite Row (from documents table) + its line_items into FinancialDocument."""
    d = dict(row)
    # Deserialise JSON columns
    for col in ("classification_evidence", "anomaly_flags"):
        raw = d.get(col, "[]")
        try:
            d[col] = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            d[col] = []
    # Deserialise line items
    items = [
        LineItem(
            description=r["description"],
            quantity=r["quantity"],
            unit_price=r["unit_price"],
            total=r["total"],
            unit=r["unit"],
        )
        for r in line_items
    ]
    d["line_items"] = items
    if "requires_review" in d:
        d["requires_review"] = bool(d["requires_review"])
    if "is_modified" in d:
        d["is_modified"] = bool(d["is_modified"])
    return FinancialDocument(**d)


def _doc_to_row(doc: FinancialDocument) -> Dict[str, Any]:
    """Flatten a FinancialDocument into a dict suitable for SQL binding."""
    return {
        "document_id":               doc.document_id,
        "filename":                  doc.filename,
        "source_file":               doc.source_file,
        "document_type":             doc.document_type,
        "benchmark_domain":          doc.benchmark_domain,
        "model_name":                doc.model_name,
        "category":                  doc.category,
        "sub_category":              doc.sub_category,
        "classification_confidence": doc.classification_confidence,
        "classification_evidence":   json.dumps(doc.classification_evidence),
        "vendor_name":               doc.vendor_name,
        "address":                   doc.address,
        "invoice_number":            doc.invoice_number,
        "invoice_date":              doc.invoice_date,
        "due_date":                  doc.due_date,
        "subtotal":                  doc.subtotal,
        "tax_amount":                doc.tax_amount,
        "total_amount":              doc.total_amount,
        "currency":                  doc.currency,
        "extraction_confidence":     doc.extraction_confidence,
        "ocr_word_count":            doc.ocr_word_count,
        "ocr_raw_text":              doc.ocr_raw_text,
        "verification_status":       doc.verification_status,
        "processing_timestamp":      doc.processing_timestamp,
        "created_at":                doc.created_at,
        "updated_at":                doc.updated_at,
        "notes":                     doc.notes,
        "anomaly_flags":             json.dumps(doc.anomaly_flags),
        "file_hash":                 doc.file_hash,
        "file_size":                 doc.file_size,
        "processing_status":         doc.processing_status,
        "processing_stage":          doc.processing_stage,
        "processing_error":          doc.processing_error,
        "processing_started_at":     doc.processing_started_at,
        "processing_completed_at":   doc.processing_completed_at,
        "processing_duration":       doc.processing_duration,
        "ai_category":               doc.ai_category,
        "ai_confidence":             doc.ai_confidence,
        "final_category":            doc.final_category,
        "requires_review":           1 if doc.requires_review else 0,
        "classification_reason":     doc.classification_reason,
        "ai_vendor_name":            doc.ai_vendor_name,
        "ai_invoice_date":           doc.ai_invoice_date,
        "ai_invoice_number":         doc.ai_invoice_number,
        "ai_total_amount":           doc.ai_total_amount,
        "is_modified":               1 if doc.is_modified else 0,
    }


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class FinancialDocumentDB:
    """
    SQLite-backed persistence layer for FinancialDocument records.

    Usage
    -----
    db = FinancialDocumentDB(project_root=Path("."))
    db.insert_document(fin_doc)
    results = db.search_documents(query="electricity", category="Electricity")
    """

    def __init__(self, project_root: Optional[Path] = None, db_path: Optional[Path] = None):
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            root = project_root or Path(__file__).resolve().parent.parent
            self.db_path = root / "data" / "financial_documents.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # -----------------------------------------------------------------------
    # Context manager & connection helper
    # -----------------------------------------------------------------------

    def close(self) -> None:
        """No-op close for resource lifecycle consistency."""
        pass

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Create tables, indexes, and migrate existing schemas if needed."""
        with self._connect() as conn:
            conn.execute(_DDL_DOCUMENTS)
            conn.execute(_DDL_LINE_ITEMS)
            conn.execute(_DDL_PROCESSING_LOG)
            conn.execute(_DDL_ANOMALY_FINDINGS)
            conn.execute(_DDL_DUPLICATE_CANDIDATES)
            conn.execute(_DDL_ANOMALY_REVIEW_HISTORY)

            # Non-destructive migration for existing tables:
            cursor = conn.execute("PRAGMA table_info(documents);")
            existing_cols = {row["name"] for row in cursor.fetchall()}

            new_columns = [
                ("file_hash", "TEXT"),
                ("file_size", "INTEGER"),
                ("processing_status", "TEXT NOT NULL DEFAULT 'COMPLETED'"),
                ("processing_stage", "TEXT"),
                ("processing_error", "TEXT"),
                ("processing_started_at", "TEXT"),
                ("processing_completed_at", "TEXT"),
                ("processing_duration", "REAL"),
                ("ai_category", "TEXT"),
                ("ai_confidence", "REAL"),
                ("final_category", "TEXT"),
                ("requires_review", "INTEGER NOT NULL DEFAULT 0"),
                ("classification_reason", "TEXT NOT NULL DEFAULT ''"),
                ("ai_vendor_name", "TEXT"),
                ("ai_invoice_date", "TEXT"),
                ("ai_invoice_number", "TEXT"),
                ("ai_total_amount", "REAL"),
                ("is_modified", "INTEGER NOT NULL DEFAULT 0"),
            ]
            for col_name, col_def in new_columns:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE documents ADD COLUMN {col_name} {col_def};")

            for idx in _DDL_INDEXES:
                conn.execute(idx)

    # -----------------------------------------------------------------------
    # CRUD
    # -----------------------------------------------------------------------

    def insert_document(self, doc: FinancialDocument) -> None:
        """Persist a new FinancialDocument. Raises ValueError if ID already exists."""
        row = _doc_to_row(doc)
        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row.keys())
        sql = f"INSERT INTO documents ({cols}) VALUES ({placeholders})"
        with self._connect() as conn:
            try:
                conn.execute(sql, row)
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"Document with id '{doc.document_id}' already exists. "
                    "Use update_document() to modify existing records."
                ) from exc
            self._log(conn, doc.document_id, "insert")
            self._insert_line_items(conn, doc)

    def upsert_document(self, doc: FinancialDocument) -> None:
        """Insert or replace an existing document (idempotent)."""
        row = _doc_to_row(doc)
        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row.keys())
        sql = f"INSERT OR REPLACE INTO documents ({cols}) VALUES ({placeholders})"
        with self._connect() as conn:
            conn.execute(sql, row)
            conn.execute("DELETE FROM line_items WHERE document_id = :id", {"id": doc.document_id})
            self._insert_line_items(conn, doc)
            self._log(conn, doc.document_id, "upsert")

    def get_document(self, document_id: str) -> Optional[FinancialDocument]:
        """Retrieve a single document by ID. Returns None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
            if row is None:
                return None
            items = conn.execute(
                "SELECT * FROM line_items WHERE document_id = ?", (document_id,)
            ).fetchall()
        return _row_to_doc(row, items)

    def update_document(self, doc: FinancialDocument) -> None:
        """Update an existing document. Raises ValueError if not found."""
        doc.touch()
        row = _doc_to_row(doc)
        set_clause = ", ".join(f"{k} = :{k}" for k in row.keys() if k != "document_id")
        sql = f"UPDATE documents SET {set_clause} WHERE document_id = :document_id"
        with self._connect() as conn:
            cursor = conn.execute(sql, row)
            if cursor.rowcount == 0:
                raise ValueError(f"Document '{doc.document_id}' not found.")
            conn.execute("DELETE FROM line_items WHERE document_id = :id", {"id": doc.document_id})
            self._insert_line_items(conn, doc)
            self._log(conn, doc.document_id, "update")

    def delete_document(self, document_id: str) -> None:
        """Delete a document and its line items. Raises ValueError if not found."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE document_id = ?", (document_id,)
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Document '{document_id}' not found.")
            self._log(conn, document_id, "delete")

    def get_document_by_hash(self, file_hash: str) -> Optional[FinancialDocument]:
        """Find an existing document by its exact file_hash. Returns newest if multiple exist."""
        if not file_hash:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE file_hash = ? ORDER BY created_at DESC LIMIT 1",
                (file_hash,),
            ).fetchone()
            if row is None:
                return None
            items = conn.execute(
                "SELECT * FROM line_items WHERE document_id = ?", (row["document_id"],)
            ).fetchall()
        return _row_to_doc(row, items)

    def get_all_documents(self, limit: int = 2000) -> List[FinancialDocument]:
        """Return all persisted documents up to the specified limit."""
        return self.search_documents(limit=limit)

    def update_verification_status(
        self,
        document_id: str,
        status: str,
        modified_by: str = "user",
        notes: str = "",
    ) -> bool:
        """Atomically update the verification status of a document with audit tracking."""
        doc = self.edit_financial_document(
            document_id=document_id,
            updates={"verification_status": status.lower()},
            modified_by=modified_by,
            notes=notes,
        )
        return doc is not None

    def find_duplicate_financials(
        self,
        vendor_name: Optional[str] = None,
        invoice_number: Optional[str] = None,
        invoice_date: Optional[str] = None,
        total_amount: Optional[float] = None,
        exclude_id: Optional[str] = None,
    ) -> List[FinancialDocument]:
        """
        Find documents that match core financial fields (invoice duplicate check).
        Matches if same invoice_number and vendor_name, or same vendor + date + total.
        """
        clauses = []
        params: Dict[str, Any] = {}
        if exclude_id:
            params["exclude_id"] = exclude_id

        # Criteria 1: Same invoice_number and vendor
        if invoice_number and vendor_name:
            clauses.append("(invoice_number = :inv_num AND vendor_name = :v_name)")
            params["inv_num"] = invoice_number
            params["v_name"] = vendor_name

        # Criteria 2: Same vendor, date, and total amount
        if vendor_name and invoice_date and total_amount is not None:
            clauses.append("(vendor_name = :v_name2 AND invoice_date = :inv_date AND ABS(total_amount - :tot_amt) < 0.01)")
            params["v_name2"] = vendor_name
            params["inv_date"] = invoice_date
            params["tot_amt"] = total_amount

        if not clauses:
            return []

        where_cond = " OR ".join(clauses)
        if exclude_id:
            where_cond = f"({where_cond}) AND document_id != :exclude_id"

        sql = f"SELECT * FROM documents WHERE {where_cond} ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            results = []
            for r in rows:
                items = conn.execute(
                    "SELECT * FROM line_items WHERE document_id = ?", (r["document_id"],)
                ).fetchall()
                results.append(_row_to_doc(r, items))
        return results

    def update_category(
        self,
        document_id: str,
        new_category: str,
        modified_by: str = "user",
        notes: Optional[str] = None,
    ) -> FinancialDocument:
        """
        Manually correct/update document category while preserving initial AI classification.
        Appends a CLASSIFICATION_MODIFIED record to processing_log.
        """
        doc = self.get_document(document_id)
        if doc is None:
            raise ValueError(f"Document '{document_id}' not found.")

        old_cat = doc.category
        doc.category = new_category
        doc.final_category = new_category
        doc.requires_review = False  # Explicit human review clears the flag
        if notes:
            doc.notes = f"{doc.notes}; {notes}".strip("; ")

        doc.touch()
        row = _doc_to_row(doc)
        set_clause = ", ".join(f"{k} = :{k}" for k in row.keys() if k != "document_id")
        sql = f"UPDATE documents SET {set_clause} WHERE document_id = :document_id"
        with self._connect() as conn:
            conn.execute(sql, row)
            details = json.dumps({
                "previous_category": old_cat,
                "new_category": new_category,
                "ai_category": doc.ai_category,
                "ai_confidence": doc.ai_confidence,
                "modified_by": modified_by,
            })
            self._log(conn, document_id, "CLASSIFICATION_MODIFIED", details)

        return doc

    def edit_financial_document(
        self,
        document_id: str,
        updates: Dict[str, Any],
        modified_by: str = "user",
        reason: str = "",
        notes: Optional[str] = None,
    ) -> FinancialDocument:
        """
        Atomically update financial record fields with validation, diff auditing, and AI value preservation.
        """
        IMMUTABLE_FIELDS = {
            "document_id", "file_hash", "created_at", "processing_timestamp",
            "ai_category", "ai_confidence", "ai_vendor_name", "ai_invoice_date",
            "ai_invoice_number", "ai_total_amount",
        }
        for k in updates:
            if k in IMMUTABLE_FIELDS:
                raise ValueError(f"Field '{k}' is immutable and cannot be directly modified.")

        doc = self.get_document(document_id)
        if doc is None:
            raise ValueError(f"Document '{document_id}' not found.")

        # Preserve initial AI predictions if not yet populated
        if doc.ai_vendor_name is None and doc.vendor_name:
            doc.ai_vendor_name = doc.vendor_name
        if doc.ai_invoice_date is None and doc.invoice_date:
            doc.ai_invoice_date = doc.invoice_date
        if doc.ai_invoice_number is None and doc.invoice_number:
            doc.ai_invoice_number = doc.invoice_number
        if doc.ai_total_amount is None and doc.total_amount is not None:
            doc.ai_total_amount = doc.total_amount
        if doc.ai_category is None:
            doc.ai_category = doc.category
        if doc.ai_confidence is None:
            doc.ai_confidence = doc.classification_confidence

        # Validation
        if "total_amount" in updates and updates["total_amount"] is not None:
            try:
                val = float(updates["total_amount"])
                if val < 0:
                    raise ValueError("Total amount cannot be negative.")
                updates["total_amount"] = val
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid total_amount: {exc}") from exc

        if "subtotal" in updates and updates["subtotal"] is not None:
            try:
                val = float(updates["subtotal"])
                if val < 0:
                    raise ValueError("Subtotal cannot be negative.")
                updates["subtotal"] = val
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid subtotal: {exc}") from exc

        if "tax_amount" in updates and updates["tax_amount"] is not None:
            try:
                val = float(updates["tax_amount"])
                if val < 0:
                    raise ValueError("Tax amount cannot be negative.")
                updates["tax_amount"] = val
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid tax_amount: {exc}") from exc

        if "category" in updates and updates["category"]:
            cat = str(updates["category"]).strip()
            from phase11.classifier import TAXONOMY
            valid_categories = set(TAXONOMY.keys())
            if cat not in valid_categories:
                raise ValueError(f"Invalid category '{cat}'. Must be one of standard taxonomy.")
            updates["category"] = cat

        if "verification_status" in updates and updates["verification_status"]:
            vstat = str(updates["verification_status"]).strip().lower()
            from phase11.schema import VERIFICATION_STATUSES
            if vstat not in VERIFICATION_STATUSES:
                raise ValueError(f"Invalid verification_status '{vstat}'.")
            updates["verification_status"] = vstat

        field_diffs = []
        for field, new_val in updates.items():
            if not hasattr(doc, field):
                continue
            old_val = getattr(doc, field)
            if old_val != new_val:
                field_diffs.append({
                    "field": field,
                    "old_value": old_val,
                    "new_value": new_val,
                    "modified_by": modified_by,
                    "reason": reason,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                setattr(doc, field, new_val)

        if "category" in updates:
            doc.final_category = updates["category"]
            doc.requires_review = False

        if field_diffs:
            doc.is_modified = True
            if "verification_status" not in updates:
                doc.verification_status = "modified"

        if notes:
            doc.notes = f"{doc.notes}; {notes}".strip("; ")

        doc.touch()
        row = _doc_to_row(doc)
        set_clause = ", ".join(f"{k} = :{k}" for k in row.keys() if k != "document_id")
        sql = f"UPDATE documents SET {set_clause} WHERE document_id = :document_id"

        with self._connect() as conn:
            cursor = conn.execute(sql, row)
            if cursor.rowcount == 0:
                raise ValueError(f"Document '{doc.document_id}' not found during update.")
            for diff in field_diffs:
                self._log(conn, document_id, "FIELD_UPDATED", json.dumps(diff))
            if field_diffs:
                self._log(conn, document_id, "RECORD_EDITED", json.dumps({
                    "modified_by": modified_by,
                    "reason": reason,
                    "fields_changed": [d["field"] for d in field_diffs],
                }))

        return doc

    def get_audit_history(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve chronological processing, classification, and field edit log for a document."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM processing_log WHERE document_id = ? ORDER BY timestamp ASC, id ASC",
                (document_id,),
            ).fetchall()
            history = []
            for r in rows:
                details_parsed: Any = r["details"]
                if r["details"]:
                    try:
                        details_parsed = json.loads(r["details"])
                    except Exception:
                        details_parsed = r["details"]
                history.append({
                    "id": r["id"],
                    "document_id": r["document_id"],
                    "action": r["action"],
                    "timestamp": r["timestamp"],
                    "details": details_parsed,
                })
            return history

    # -----------------------------------------------------------------------
    # Search & filter builder
    # -----------------------------------------------------------------------

    def _build_search_conditions(
        self,
        query: str = "",
        category: Optional[str] = None,
        verification_status: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        benchmark_domain: Optional[str] = None,
        requires_review: Optional[bool] = None,
        processing_status: Optional[str] = None,
        vendor: Optional[str] = None,
        invoice_number: Optional[str] = None,
        date_exact: Optional[str] = None,
        month: Optional[str] = None,
        year: Optional[str] = None,
        fiscal_year: Optional[str] = None,
        is_modified: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[List[str], Dict[str, Any]]:
        conditions: List[str] = []
        params: Dict[str, Any] = {}

        if query and query.strip():
            q_clean = query.strip()
            conditions.append(
                "(vendor_name LIKE :q OR filename LIKE :q OR "
                "ocr_raw_text LIKE :q OR address LIKE :q OR invoice_number LIKE :q OR "
                "category LIKE :q OR sub_category LIKE :q)"
            )
            params["q"] = f"%{q_clean}%"

        if category and category != "All":
            conditions.append("category = :category")
            params["category"] = category

        if vendor and vendor != "All" and vendor.strip():
            conditions.append("vendor_name LIKE :vendor_param")
            params["vendor_param"] = f"%{vendor.strip()}%"

        if invoice_number and invoice_number.strip():
            conditions.append("invoice_number LIKE :inv_num_param")
            params["inv_num_param"] = f"%{invoice_number.strip()}%"

        if verification_status and verification_status != "All":
            conditions.append("verification_status = :vstatus")
            params["vstatus"] = verification_status.lower()

        if date_exact and date_exact.strip():
            conditions.append("invoice_date = :date_exact")
            params["date_exact"] = date_exact.strip()
        else:
            if date_from and date_from.strip():
                conditions.append("invoice_date >= :date_from")
                params["date_from"] = date_from.strip()
            if date_to and date_to.strip():
                conditions.append("invoice_date <= :date_to")
                params["date_to"] = date_to.strip()

        if month and month != "All" and month.strip():
            conditions.append("invoice_date LIKE :month_param")
            params["month_param"] = f"{month.strip()}%"

        if year and year != "All" and year.strip():
            conditions.append("invoice_date LIKE :year_param")
            params["year_param"] = f"{year.strip()}%"

        if fiscal_year and fiscal_year != "All" and fiscal_year.strip():
            # Indian Fiscal Year runs from 01-April of first year to 31-March of next year
            fy_clean = fiscal_year.upper().replace("FY", "").strip()
            parts = fy_clean.split("-")
            if len(parts) == 2:
                try:
                    start_yr = int(parts[0])
                    end_str = parts[1]
                    end_yr = int(end_str) if len(end_str) == 4 else (start_yr // 100 * 100 + int(end_str))
                    conditions.append("(invoice_date >= :fy_start AND invoice_date <= :fy_end)")
                    params["fy_start"] = f"{start_yr:04d}-04-01"
                    params["fy_end"] = f"{end_yr:04d}-03-31"
                except ValueError:
                    pass

        if min_amount is not None and min_amount > 0:
            conditions.append("total_amount >= :min_amt")
            params["min_amt"] = min_amount

        if max_amount is not None and max_amount > 0:
            conditions.append("total_amount <= :max_amt")
            params["max_amt"] = max_amount

        if benchmark_domain and benchmark_domain != "All":
            conditions.append("benchmark_domain = :domain")
            params["domain"] = benchmark_domain.upper()

        if requires_review is not None:
            conditions.append("requires_review = :req_rev")
            params["req_rev"] = 1 if requires_review else 0

        if processing_status and processing_status != "All":
            conditions.append("processing_status = :proc_status")
            params["proc_status"] = processing_status

        if is_modified is not None:
            conditions.append("is_modified = :is_mod")
            params["is_mod"] = 1 if is_modified else 0

        return conditions, params

    def search_documents(
        self,
        query: str = "",
        category: Optional[str] = None,
        verification_status: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        benchmark_domain: Optional[str] = None,
        requires_review: Optional[bool] = None,
        processing_status: Optional[str] = None,
        # Phase 13 extensions
        vendor: Optional[str] = None,
        invoice_number: Optional[str] = None,
        date_exact: Optional[str] = None,
        month: Optional[str] = None,
        year: Optional[str] = None,
        fiscal_year: Optional[str] = None,
        is_modified: Optional[bool] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        limit: int = 200,
        offset: int = 0,
        **kwargs,
    ) -> List[FinancialDocument]:
        """
        Search documents with structured filters, sorting, and pagination.
        """
        conditions, params = self._build_search_conditions(
            query=query,
            category=category,
            verification_status=verification_status,
            date_from=date_from,
            date_to=date_to,
            min_amount=min_amount,
            max_amount=max_amount,
            benchmark_domain=benchmark_domain,
            requires_review=requires_review,
            processing_status=processing_status,
            vendor=vendor,
            invoice_number=invoice_number,
            date_exact=date_exact,
            month=month,
            year=year,
            fiscal_year=fiscal_year,
            is_modified=is_modified,
        )

        params["limit"] = limit
        params["offset"] = offset

        ALLOWED_SORTS = {
            "created_at": "created_at",
            "invoice_date": "invoice_date",
            "total_amount": "total_amount",
            "vendor_name": "vendor_name",
            "category": "category",
            "confidence": "classification_confidence",
            "classification_confidence": "classification_confidence",
            "verification_status": "verification_status",
            "processing_duration": "processing_duration",
        }
        col_sort = ALLOWED_SORTS.get(sort_by, "created_at")
        direction = "ASC" if str(sort_order).strip().upper() == "ASC" else "DESC"

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
            SELECT * FROM documents {where}
            ORDER BY {col_sort} IS NULL ASC, {col_sort} {direction}, created_at DESC
            LIMIT :limit OFFSET :offset
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            docs: List[FinancialDocument] = []
            for row in rows:
                items = conn.execute(
                    "SELECT * FROM line_items WHERE document_id = ?", (row["document_id"],)
                ).fetchall()
                docs.append(_row_to_doc(row, items))
        return docs

    def count_search_documents(
        self,
        query: str = "",
        category: Optional[str] = None,
        verification_status: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        benchmark_domain: Optional[str] = None,
        requires_review: Optional[bool] = None,
        processing_status: Optional[str] = None,
        vendor: Optional[str] = None,
        invoice_number: Optional[str] = None,
        date_exact: Optional[str] = None,
        month: Optional[str] = None,
        year: Optional[str] = None,
        fiscal_year: Optional[str] = None,
        is_modified: Optional[bool] = None,
        **kwargs,
    ) -> int:
        """Return total document count matching the search and filter criteria for pagination."""
        conditions, params = self._build_search_conditions(
            query=query,
            category=category,
            verification_status=verification_status,
            date_from=date_from,
            date_to=date_to,
            min_amount=min_amount,
            max_amount=max_amount,
            benchmark_domain=benchmark_domain,
            requires_review=requires_review,
            processing_status=processing_status,
            vendor=vendor,
            invoice_number=invoice_number,
            date_exact=date_exact,
            month=month,
            year=year,
            fiscal_year=fiscal_year,
            is_modified=is_modified,
        )
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"SELECT COUNT(*) FROM documents {where}"
        with self._connect() as conn:
            result = conn.execute(sql, params).fetchone()
        return result[0] if result else 0

    def count_documents(
        self,
        category: Optional[str] = None,
        verification_status: Optional[str] = None,
    ) -> int:
        """Return total document count (backwards compatible with Phase 11)."""
        return self.count_search_documents(
            category=category,
            verification_status=verification_status,
        )

    # Alias for count_documents
    get_document_count = count_documents

    # -----------------------------------------------------------------------
    # Analytics (Filter-Aware)
    # -----------------------------------------------------------------------

    def spend_by_category(self, **filter_kwargs) -> List[Dict[str, Any]]:
        """Total spend and document count grouped by category, respecting active filters."""
        conditions, params = self._build_search_conditions(**filter_kwargs)
        conditions.append("total_amount IS NOT NULL")
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
            SELECT
                category,
                COUNT(*)            AS count,
                SUM(total_amount)   AS total_spend,
                AVG(total_amount)   AS avg_amount
            FROM documents
            {where}
            GROUP BY category
            ORDER BY total_spend DESC
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def monthly_spend(self, **filter_kwargs) -> List[Dict[str, Any]]:
        """Aggregate spend by month derived from invoice_date string (YYYY-MM prefix), respecting active filters."""
        conditions, params = self._build_search_conditions(**filter_kwargs)
        conditions.append("total_amount IS NOT NULL")
        conditions.append("invoice_date IS NOT NULL")
        conditions.append("LENGTH(invoice_date) >= 7")
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
            SELECT
                SUBSTR(invoice_date, 1, 7) AS month,
                COUNT(*)                   AS count,
                SUM(total_amount)          AS total_spend
            FROM documents
            {where}
            GROUP BY month
            ORDER BY month
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def top_vendors(self, n: int = 10, **filter_kwargs) -> List[Dict[str, Any]]:
        """Top N vendors by total spend, respecting active filters."""
        conditions, params = self._build_search_conditions(**filter_kwargs)
        conditions.append("vendor_name IS NOT NULL")
        conditions.append("total_amount IS NOT NULL")
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params["n"] = n
        sql = f"""
            SELECT
                vendor_name,
                COUNT(*)          AS count,
                SUM(total_amount) AS total_spend
            FROM documents
            {where}
            GROUP BY vendor_name
            ORDER BY total_spend DESC
            LIMIT :n
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def status_summary(self, **filter_kwargs) -> Dict[str, int]:
        """Count of documents per verification_status, respecting active filters."""
        conditions, params = self._build_search_conditions(**filter_kwargs)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
            SELECT verification_status, COUNT(*) AS n
            FROM documents
            {where}
            GROUP BY verification_status
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return {r["verification_status"]: r["n"] for r in rows}

    def anomaly_candidates(self) -> List[FinancialDocument]:
        """
        Return documents that meet any anomaly heuristic:
        - Same invoice_number used more than once
        - total_amount is a suspiciously round number (multiple of 1000)
        - Very high amounts (> mean + 3σ) — computed in Python
        """
        with self._connect() as conn:
            # Duplicate invoice numbers
            dup_sql = """
                SELECT document_id FROM documents
                WHERE invoice_number IS NOT NULL
                  AND invoice_number IN (
                      SELECT invoice_number FROM documents
                      WHERE invoice_number IS NOT NULL
                      GROUP BY invoice_number HAVING COUNT(*) > 1
                  )
            """
            dup_ids = {r[0] for r in conn.execute(dup_sql).fetchall()}

            # Round amounts
            round_sql = """
                SELECT document_id FROM documents
                WHERE total_amount IS NOT NULL
                  AND total_amount > 0
                  AND (CAST(total_amount AS INTEGER) % 1000 = 0)
            """
            round_ids = {r[0] for r in conn.execute(round_sql).fetchall()}

            all_ids = dup_ids | round_ids
            docs: List[FinancialDocument] = []
            for doc_id in all_ids:
                row = conn.execute("SELECT * FROM documents WHERE document_id = ?", (doc_id,)).fetchone()
                if row:
                    items = conn.execute("SELECT * FROM line_items WHERE document_id = ?", (doc_id,)).fetchall()
                    docs.append(_row_to_doc(row, items))
        return docs

    # -----------------------------------------------------------------------
    # Phase 14 — Advanced Analytics Engine
    # -----------------------------------------------------------------------

    @staticmethod
    def _fy_bounds(fy_label: str) -> Tuple[str, str]:
        """
        Convert an Indian Financial Year label such as 'FY 2025-26' or '2025-26'
        into (start_date, end_date) ISO strings: ('2025-04-01', '2026-03-31').
        """
        clean = fy_label.upper().replace("FY", "").strip()
        parts = clean.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid FY label: {fy_label!r}")
        start_yr = int(parts[0])
        end_part = parts[1].strip()
        end_yr = int(end_part) if len(end_part) == 4 else (start_yr // 100 * 100 + int(end_part))
        return (f"{start_yr:04d}-04-01", f"{end_yr:04d}-03-31")

    def kpi_summary(self, **filter_kwargs) -> Dict[str, Any]:
        """
        Return a comprehensive KPI summary dict in a single SQL pass.

        Keys returned
        -------------
        total_spend         : float | None
        invoice_count       : int
        avg_invoice_value   : float | None
        tax_total           : float | None   — None if NO rows have tax_amount set
        tax_available_count : int            — rows where tax_amount IS NOT NULL
        tax_unavailable_count: int           — rows where tax_amount IS NULL
        subtotal_total      : float | None
        effective_tax_rate  : float | None   — tax_total/subtotal_total if computable
        verified_count      : int
        pending_count       : int
        modified_count      : int
        flagged_count       : int
        rejected_count      : int
        verified_rate       : float          — 0.0–1.0
        avg_confidence      : float | None
        """
        conditions, params = self._build_search_conditions(**filter_kwargs)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
            SELECT
                COUNT(*)                                              AS invoice_count,
                SUM(CASE WHEN total_amount IS NOT NULL THEN total_amount ELSE 0 END) AS total_spend,
                AVG(total_amount)                                     AS avg_invoice_value,
                SUM(CASE WHEN tax_amount IS NOT NULL THEN tax_amount ELSE 0 END)     AS tax_total,
                COUNT(CASE WHEN tax_amount IS NOT NULL THEN 1 END)   AS tax_available_count,
                COUNT(CASE WHEN tax_amount IS NULL THEN 1 END)       AS tax_unavailable_count,
                SUM(CASE WHEN subtotal IS NOT NULL THEN subtotal ELSE 0 END)         AS subtotal_total,
                COUNT(CASE WHEN verification_status = 'verified'  THEN 1 END)        AS verified_count,
                COUNT(CASE WHEN verification_status = 'pending'   THEN 1 END)        AS pending_count,
                COUNT(CASE WHEN verification_status = 'modified'  THEN 1 END)        AS modified_count,
                COUNT(CASE WHEN verification_status = 'flagged'   THEN 1 END)        AS flagged_count,
                COUNT(CASE WHEN verification_status = 'rejected'  THEN 1 END)        AS rejected_count,
                AVG(classification_confidence)                       AS avg_confidence,
                COUNT(CASE WHEN total_amount IS NOT NULL THEN 1 END) AS spend_count
            FROM documents {where}
        """
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        if row is None:
            return {}
        r = dict(row)
        count = r["invoice_count"] or 0
        tax_avail = r["tax_available_count"] or 0
        # If no rows have tax data at all, report as unavailable
        tax_total: Optional[float] = float(r["tax_total"]) if tax_avail > 0 else None
        subtotal_total: Optional[float] = float(r["subtotal_total"]) if r["subtotal_total"] else None
        effective_tax_rate: Optional[float] = None
        if tax_total is not None and subtotal_total and subtotal_total > 0:
            effective_tax_rate = tax_total / subtotal_total
        verified_count = r["verified_count"] or 0
        return {
            "total_spend":            float(r["total_spend"]) if r["total_spend"] else 0.0,
            "invoice_count":          count,
            "avg_invoice_value":      float(r["avg_invoice_value"]) if r["avg_invoice_value"] is not None else None,
            "tax_total":              tax_total,
            "tax_available_count":    tax_avail,
            "tax_unavailable_count":  r["tax_unavailable_count"] or 0,
            "subtotal_total":         subtotal_total,
            "effective_tax_rate":     effective_tax_rate,
            "verified_count":         verified_count,
            "pending_count":          r["pending_count"] or 0,
            "modified_count":         r["modified_count"] or 0,
            "flagged_count":          r["flagged_count"] or 0,
            "rejected_count":         r["rejected_count"] or 0,
            "verified_rate":          verified_count / count if count > 0 else 0.0,
            "avg_confidence":         float(r["avg_confidence"]) if r["avg_confidence"] is not None else None,
        }

    def spend_by_fiscal_year(self, **filter_kwargs) -> List[Dict[str, Any]]:
        """
        Aggregate spend by Indian Financial Year (April 1 – March 31).

        Returns list of dicts with keys:
            fiscal_year  (str)  e.g. 'FY 2025-26'
            count        (int)
            total_spend  (float)
            avg_amount   (float)
        """
        conditions, params = self._build_search_conditions(**filter_kwargs)
        conditions_copy = list(conditions)
        conditions_copy.append("total_amount IS NOT NULL")
        conditions_copy.append("invoice_date IS NOT NULL")
        conditions_copy.append("LENGTH(invoice_date) >= 7")
        where = ("WHERE " + " AND ".join(conditions_copy)) if conditions_copy else ""
        # Derive FY from invoice_date: if month >= 04, FY start = this year, else FY start = last year
        sql = f"""
            SELECT
                CASE
                    WHEN CAST(SUBSTR(invoice_date, 6, 2) AS INTEGER) >= 4
                    THEN 'FY ' || SUBSTR(invoice_date, 1, 4) || '-' || SUBSTR(CAST(CAST(SUBSTR(invoice_date, 1, 4) AS INTEGER) + 1 AS TEXT), 3, 2)
                    ELSE 'FY ' || CAST(CAST(SUBSTR(invoice_date, 1, 4) AS INTEGER) - 1 AS TEXT) || '-' || SUBSTR(invoice_date, 3, 2)
                END AS fiscal_year,
                COUNT(*)            AS count,
                SUM(total_amount)   AS total_spend,
                AVG(total_amount)   AS avg_amount
            FROM documents {where}
            GROUP BY fiscal_year
            ORDER BY fiscal_year
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def yoy_comparison(self, fy1: str, fy2: str, **filter_kwargs) -> Dict[str, Any]:
        """
        Compare two Indian Financial Years side-by-side.

        Parameters
        ----------
        fy1 : str   e.g. 'FY 2024-25'  (baseline / prior year)
        fy2 : str   e.g. 'FY 2025-26'  (comparison / current year)

        Returns dict with keys:
            fy1, fy2 — label strings
            fy1_data, fy2_data — dicts (total_spend, count, avg_amount, tax_total)
            delta_spend_pct      — % change in total_spend (None if fy1 spend = 0)
            delta_count_pct      — % change in invoice count
            delta_avg_pct        — % change in avg amount
        """
        base_conditions, base_params = self._build_search_conditions(**filter_kwargs)

        def _fy_stats(fy_label: str) -> Dict[str, Any]:
            start, end = self._fy_bounds(fy_label)
            conds = list(base_conditions) + [
                "invoice_date >= :fy_s", "invoice_date <= :fy_e", "total_amount IS NOT NULL"
            ]
            prms = dict(base_params)
            prms["fy_s"] = start
            prms["fy_e"] = end
            where = "WHERE " + " AND ".join(conds)
            sql = f"""
                SELECT COUNT(*) AS count, SUM(total_amount) AS total_spend,
                       AVG(total_amount) AS avg_amount,
                       SUM(CASE WHEN tax_amount IS NOT NULL THEN tax_amount ELSE 0 END) AS tax_total,
                       COUNT(CASE WHEN tax_amount IS NOT NULL THEN 1 END) AS tax_available_count
                FROM documents {where}
            """
            with self._connect() as conn:
                row = conn.execute(sql, prms).fetchone()
            r = dict(row) if row else {}
            return {
                "total_spend": float(r.get("total_spend") or 0),
                "count": int(r.get("count") or 0),
                "avg_amount": float(r.get("avg_amount") or 0) if r.get("avg_amount") is not None else None,
                "tax_total": float(r.get("tax_total") or 0) if (r.get("tax_available_count") or 0) > 0 else None,
            }

        d1 = _fy_stats(fy1)
        d2 = _fy_stats(fy2)

        def _delta_pct(old: float, new: float) -> Optional[float]:
            if old == 0:
                return None
            return (new - old) / old * 100.0

        return {
            "fy1": fy1,
            "fy2": fy2,
            "fy1_data": d1,
            "fy2_data": d2,
            "delta_spend_pct":  _delta_pct(d1["total_spend"],    d2["total_spend"]),
            "delta_count_pct":  _delta_pct(float(d1["count"]),   float(d2["count"])),
            "delta_avg_pct":    _delta_pct(d1["avg_amount"] or 0, d2["avg_amount"] or 0),
        }

    def monthly_trend_with_delta(self, **filter_kwargs) -> List[Dict[str, Any]]:
        """
        Monthly spend with Month-over-Month delta % and 3-month rolling average.

        Returns list ordered by month, each entry:
            month            (str)    YYYY-MM
            count            (int)
            total_spend      (float)
            mom_delta_pct    (float | None) — None for first month
            rolling_avg_3m   (float | None) — None for first 2 months
        """
        conditions, params = self._build_search_conditions(**filter_kwargs)
        conditions = list(conditions)
        conditions.append("total_amount IS NOT NULL")
        conditions.append("invoice_date IS NOT NULL")
        conditions.append("LENGTH(invoice_date) >= 7")
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
            SELECT
                SUBSTR(invoice_date, 1, 7) AS month,
                COUNT(*)                   AS count,
                SUM(total_amount)          AS total_spend
            FROM documents {where}
            GROUP BY month
            ORDER BY month
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        results = [dict(r) for r in rows]
        # Compute MoM delta and rolling 3-month average in Python (small result set)
        for i, row in enumerate(results):
            prev = results[i - 1]["total_spend"] if i > 0 else None
            cur_spend = row["total_spend"] or 0.0
            row["total_spend"] = cur_spend
            if prev is not None and prev != 0:
                row["mom_delta_pct"] = (cur_spend - prev) / prev * 100.0
            else:
                row["mom_delta_pct"] = None
            # Rolling 3-month avg
            if i >= 2:
                window = [results[j]["total_spend"] for j in range(i - 2, i + 1)]
                row["rolling_avg_3m"] = sum(window) / 3.0
            else:
                row["rolling_avg_3m"] = None
        return results

    def category_concentration(self, **filter_kwargs) -> Dict[str, Any]:
        """
        Compute category spend concentration metrics.

        Returns:
            categories        : list of {category, total_spend, share_pct}
            hhi               : float  — Herfindahl-Hirschman Index (0–10000)
            hhi_interpretation: str    — 'Highly Concentrated' | 'Moderate' | 'Diversified'
            top1_share_pct    : float
            top3_share_pct    : float
            category_count    : int
        """
        conditions, params = self._build_search_conditions(**filter_kwargs)
        conditions = list(conditions)
        conditions.append("total_amount IS NOT NULL")
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
            SELECT category, SUM(total_amount) AS total_spend
            FROM documents {where}
            GROUP BY category
            ORDER BY total_spend DESC
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        data = [dict(r) for r in rows if r["total_spend"] is not None and r["total_spend"] > 0]
        grand_total = sum(r["total_spend"] for r in data)
        if grand_total == 0:
            return {
                "categories": [],
                "hhi": 0.0,
                "hhi_interpretation": "No data",
                "top1_share_pct": 0.0,
                "top3_share_pct": 0.0,
                "category_count": 0,
            }
        for r in data:
            r["share_pct"] = r["total_spend"] / grand_total * 100.0
        hhi = sum((r["share_pct"]) ** 2 for r in data)
        if hhi >= 2500:
            interp = "Highly Concentrated"
        elif hhi >= 1500:
            interp = "Moderate Concentration"
        else:
            interp = "Diversified"
        top1 = data[0]["share_pct"] if data else 0.0
        top3 = sum(r["share_pct"] for r in data[:3])
        return {
            "categories":          data,
            "hhi":                 round(hhi, 2),
            "hhi_interpretation":  interp,
            "top1_share_pct":      round(top1, 2),
            "top3_share_pct":      round(top3, 2),
            "category_count":      len(data),
        }

    def vendor_summary(self, n: int = 10, **filter_kwargs) -> List[Dict[str, Any]]:
        """
        Extended vendor statistics for top N vendors by total spend.

        Returns list with keys:
            vendor_name, invoice_count, total_spend, share_pct,
            avg_invoice, last_invoice_date, concentration_risk (bool)
        """
        conditions, params = self._build_search_conditions(**filter_kwargs)
        conditions = list(conditions)
        conditions.append("vendor_name IS NOT NULL")
        conditions.append("total_amount IS NOT NULL")
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params["n"] = n

        # Grand total for share computation
        total_sql = f"SELECT SUM(total_amount) AS grand FROM documents {where}"
        vendor_sql = f"""
            SELECT
                vendor_name,
                COUNT(*)              AS invoice_count,
                SUM(total_amount)     AS total_spend,
                AVG(total_amount)     AS avg_invoice,
                MAX(invoice_date)     AS last_invoice_date
            FROM documents {where}
            GROUP BY vendor_name
            ORDER BY total_spend DESC
            LIMIT :n
        """
        with self._connect() as conn:
            grand_row = conn.execute(total_sql, params).fetchone()
            grand_total = float(grand_row["grand"] or 0) if grand_row else 0.0
            rows = conn.execute(vendor_sql, params).fetchall()
        results = []
        for r in rows:
            spend = float(r["total_spend"] or 0)
            share = spend / grand_total * 100.0 if grand_total > 0 else 0.0
            results.append({
                "vendor_name":         r["vendor_name"],
                "invoice_count":       r["invoice_count"],
                "total_spend":         spend,
                "share_pct":           round(share, 2),
                "avg_invoice":         float(r["avg_invoice"] or 0) if r["avg_invoice"] is not None else None,
                "last_invoice_date":   r["last_invoice_date"],
                "concentration_risk":  share > 30.0,
            })
        return results

    def tax_analysis(self, **filter_kwargs) -> Dict[str, Any]:
        """
        Tax coverage and compliance analysis.

        Returns:
            coverage_pct          : float          — % docs with tax_amount set
            total_tax             : float | None   — None if no tax data
            avg_effective_rate    : float | None   — avg(tax/subtotal) where both available
            min_effective_rate    : float | None
            max_effective_rate    : float | None
            zero_tax_count        : int            — docs where tax_amount = 0 but total > 0
            null_tax_count        : int            — docs where tax_amount IS NULL
            tax_by_category       : List[Dict]     — {category, total_tax, avg_tax, count}
        """
        conditions, params = self._build_search_conditions(**filter_kwargs)
        base_where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        summary_sql = f"""
            SELECT
                COUNT(*)                                                          AS total_count,
                COUNT(CASE WHEN tax_amount IS NOT NULL THEN 1 END)               AS tax_count,
                COUNT(CASE WHEN tax_amount IS NULL THEN 1 END)                   AS null_tax_count,
                COUNT(CASE WHEN tax_amount = 0 AND total_amount > 0 THEN 1 END)  AS zero_tax_count,
                SUM(CASE WHEN tax_amount IS NOT NULL THEN tax_amount ELSE 0 END) AS total_tax,
                AVG(CASE WHEN tax_amount IS NOT NULL AND subtotal IS NOT NULL AND subtotal > 0
                         THEN CAST(tax_amount AS REAL) / subtotal ELSE NULL END) AS avg_eff_rate,
                MIN(CASE WHEN tax_amount IS NOT NULL AND subtotal IS NOT NULL AND subtotal > 0
                         THEN CAST(tax_amount AS REAL) / subtotal ELSE NULL END) AS min_eff_rate,
                MAX(CASE WHEN tax_amount IS NOT NULL AND subtotal IS NOT NULL AND subtotal > 0
                         THEN CAST(tax_amount AS REAL) / subtotal ELSE NULL END) AS max_eff_rate
            FROM documents {base_where}
        """
        cat_conds = list(conditions) + ["tax_amount IS NOT NULL"]
        cat_where = "WHERE " + " AND ".join(cat_conds)
        cat_sql = f"""
            SELECT
                category,
                COUNT(*)            AS count,
                SUM(tax_amount)     AS total_tax,
                AVG(tax_amount)     AS avg_tax
            FROM documents {cat_where}
            GROUP BY category
            ORDER BY total_tax DESC
        """
        with self._connect() as conn:
            summary = conn.execute(summary_sql, params).fetchone()
            cat_rows = conn.execute(cat_sql, params).fetchall()
        s = dict(summary) if summary else {}
        total_count = s.get("total_count") or 0
        tax_count = s.get("tax_count") or 0
        coverage = tax_count / total_count if total_count > 0 else 0.0
        total_tax = float(s["total_tax"]) if (s.get("total_tax") is not None and tax_count > 0) else None
        return {
            "coverage_pct":       round(coverage * 100, 2),
            "total_tax":          total_tax,
            "avg_effective_rate": float(s["avg_eff_rate"]) if s.get("avg_eff_rate") is not None else None,
            "min_effective_rate": float(s["min_eff_rate"]) if s.get("min_eff_rate") is not None else None,
            "max_effective_rate": float(s["max_eff_rate"]) if s.get("max_eff_rate") is not None else None,
            "zero_tax_count":     s.get("zero_tax_count") or 0,
            "null_tax_count":     s.get("null_tax_count") or 0,
            "tax_by_category":    [dict(r) for r in cat_rows],
        }

    def advanced_anomaly_report(self, sigma_threshold: float = 2.0, **filter_kwargs) -> Dict[str, Any]:
        """
        Comprehensive anomaly analysis with statistical sigma detection.

        Returns:
            sigma_threshold       : float
            mean_amount           : float | None
            std_amount            : float | None
            sigma_outliers        : List[Dict]  — docs > sigma_threshold * std above mean
            duplicate_invoices    : List[Dict]  — invoice_number → [document_ids, count]
            missing_data_matrix   : Dict[str, int]  — field → count of NULLs
            round_amount_count    : int
            monthly_anomaly_count : List[Dict]  — month → count of flagged docs
        """
        conditions, params = self._build_search_conditions(**filter_kwargs)
        base_where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        # Compute mean and std of total_amount using SQL (Welford not available in SQLite)
        stats_sql = f"""
            SELECT
                COUNT(total_amount)                    AS n,
                AVG(total_amount)                      AS mean_amount,
                SUM(total_amount * total_amount)       AS sum_sq,
                SUM(total_amount)                      AS sum_val
            FROM documents {base_where}
        """
        with self._connect() as conn:
            stats_row = conn.execute(stats_sql, params).fetchone()
            n = int(stats_row["n"] or 0)
            mean_amount: Optional[float] = float(stats_row["mean_amount"]) if stats_row["mean_amount"] is not None else None
            std_amount: Optional[float] = None
            if n > 1 and mean_amount is not None:
                variance = float(stats_row["sum_sq"] or 0) / n - mean_amount ** 2
                std_amount = variance ** 0.5 if variance > 0 else 0.0

            # Sigma outliers
            sigma_outliers: List[Dict[str, Any]] = []
            if mean_amount is not None and std_amount and std_amount > 0:
                threshold_val = mean_amount + sigma_threshold * std_amount
                outlier_conds = list(conditions) + [
                    "total_amount IS NOT NULL", "total_amount > :sigma_thresh"
                ]
                outlier_prms = dict(params)
                outlier_prms["sigma_thresh"] = threshold_val
                out_where = "WHERE " + " AND ".join(outlier_conds)
                out_sql = f"""
                    SELECT document_id, filename, vendor_name, total_amount, invoice_date, verification_status
                    FROM documents {out_where}
                    ORDER BY total_amount DESC
                    LIMIT 50
                """
                for r in conn.execute(out_sql, outlier_prms).fetchall():
                    d = dict(r)
                    if mean_amount and std_amount and std_amount > 0:
                        d["sigma_distance"] = (float(d["total_amount"]) - mean_amount) / std_amount
                    sigma_outliers.append(d)

            # Duplicate invoice numbers
            dup_conds = list(conditions) + ["invoice_number IS NOT NULL"]
            dup_where = ("WHERE " + " AND ".join(dup_conds)) if dup_conds else ""
            dup_sql = f"""
                SELECT invoice_number, COUNT(*) AS cnt,
                       GROUP_CONCAT(document_id) AS doc_ids
                FROM documents {dup_where}
                GROUP BY invoice_number HAVING cnt > 1
                ORDER BY cnt DESC
            """
            dup_rows = conn.execute(dup_sql, params).fetchall()
            duplicate_invoices = [
                {
                    "invoice_number": r["invoice_number"],
                    "count": r["cnt"],
                    "document_ids": (r["doc_ids"] or "").split(","),
                }
                for r in dup_rows
            ]

            # Missing data matrix — count NULLs per key financial field
            null_sql = f"""
                SELECT
                    COUNT(CASE WHEN vendor_name   IS NULL THEN 1 END) AS vendor_name,
                    COUNT(CASE WHEN invoice_date  IS NULL THEN 1 END) AS invoice_date,
                    COUNT(CASE WHEN invoice_number IS NULL THEN 1 END) AS invoice_number,
                    COUNT(CASE WHEN total_amount  IS NULL THEN 1 END) AS total_amount,
                    COUNT(CASE WHEN tax_amount    IS NULL THEN 1 END) AS tax_amount,
                    COUNT(CASE WHEN subtotal      IS NULL THEN 1 END) AS subtotal,
                    COUNT(CASE WHEN address       IS NULL THEN 1 END) AS address,
                    COUNT(*)                                           AS total_count
                FROM documents {base_where}
            """
            null_row = dict(conn.execute(null_sql, params).fetchone() or {})
            total_count = null_row.pop("total_count", 0)
            missing_data_matrix = {
                field: {"null_count": cnt, "null_pct": round(cnt / total_count * 100, 1) if total_count else 0}
                for field, cnt in null_row.items()
            }

            # Round amounts
            round_conds = list(conditions) + [
                "total_amount IS NOT NULL", "total_amount > 0",
                "(CAST(total_amount AS INTEGER) % 1000 = 0)"
            ]
            round_where = "WHERE " + " AND ".join(round_conds)
            round_count = conn.execute(
                f"SELECT COUNT(*) AS n FROM documents {round_where}", params
            ).fetchone()["n"] or 0

            # Monthly anomaly count (flagged docs per month)
            monthly_conds = list(conditions) + [
                "verification_status = 'flagged'",
                "invoice_date IS NOT NULL", "LENGTH(invoice_date) >= 7"
            ]
            monthly_prms = dict(params)
            monthly_where = "WHERE " + " AND ".join(monthly_conds)
            monthly_sql = f"""
                SELECT SUBSTR(invoice_date, 1, 7) AS month, COUNT(*) AS flagged_count
                FROM documents {monthly_where}
                GROUP BY month ORDER BY month
            """
            monthly_anomaly = [dict(r) for r in conn.execute(monthly_sql, monthly_prms).fetchall()]

        return {
            "sigma_threshold":      sigma_threshold,
            "mean_amount":          mean_amount,
            "std_amount":           std_amount,
            "sigma_outliers":       sigma_outliers,
            "duplicate_invoices":   duplicate_invoices,
            "missing_data_matrix":  missing_data_matrix,
            "total_count":          total_count,
            "round_amount_count":   round_count,
            "monthly_anomaly_count": monthly_anomaly,
        }

    # -----------------------------------------------------------------------
    # Phase 15 — Anomaly & Duplicate Intelligence
    # -----------------------------------------------------------------------

    def save_anomaly_findings(self, findings: List[Dict[str, Any]], preserve_reviewed: bool = True) -> int:
        """
        Persist anomaly findings. If preserve_reviewed is True, keeps the status
        of already reviewed findings (UNDER_REVIEW, RESOLVED, DISMISSED, CONFIRMED_DUPLICATE).
        """
        if not findings:
            return 0
        saved = 0
        with self._connect() as conn:
            for f in findings:
                doc_id = f["document_id"]
                a_type = f["anomaly_type"]
                existing = conn.execute(
                    "SELECT id, status FROM anomaly_findings WHERE document_id = ? AND anomaly_type = ?",
                    (doc_id, a_type),
                ).fetchone()

                if existing:
                    old_status = existing["status"]
                    keep_status = preserve_reviewed and old_status in (
                        "UNDER_REVIEW", "RESOLVED", "DISMISSED", "CONFIRMED_DUPLICATE"
                    )
                    status_to_write = old_status if keep_status else f.get("status", "OPEN")
                    conn.execute(
                        """UPDATE anomaly_findings
                           SET severity = ?, score = ?, confidence = ?, explanation = ?,
                               status = ?, detected_at = ?
                           WHERE id = ?""",
                        (
                            f["severity"], float(f["score"]), float(f["confidence"]),
                            f["explanation"], status_to_write, f.get("detected_at", datetime.now(timezone.utc).isoformat()),
                            existing["id"],
                        ),
                    )
                else:
                    conn.execute(
                        """INSERT INTO anomaly_findings
                           (document_id, anomaly_type, severity, score, confidence, explanation, status, detected_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            doc_id, a_type, f["severity"], float(f["score"]),
                            float(f["confidence"]), f["explanation"],
                            f.get("status", "OPEN"), f.get("detected_at", datetime.now(timezone.utc).isoformat()),
                        ),
                    )
                saved += 1
        return saved

    def save_duplicate_candidates(self, candidates: List[Dict[str, Any]], preserve_reviewed: bool = True) -> int:
        """
        Persist duplicate candidates. If preserve_reviewed is True, preserves
        reviewer status for existing candidate pairs.
        """
        if not candidates:
            return 0
        saved = 0
        with self._connect() as conn:
            for c in candidates:
                d1 = c["document_id"]
                d2 = c["matched_document_id"]
                dtype = c["duplicate_type"]
                m_fields = c.get("matched_fields", [])
                m_fields_str = json.dumps(m_fields) if isinstance(m_fields, (list, dict)) else str(m_fields)

                existing = conn.execute(
                    """SELECT id, status FROM duplicate_candidates
                       WHERE document_id = ? AND matched_document_id = ? AND duplicate_type = ?""",
                    (d1, d2, dtype),
                ).fetchone()

                if existing:
                    old_status = existing["status"]
                    keep_status = preserve_reviewed and old_status in (
                        "UNDER_REVIEW", "RESOLVED", "DISMISSED", "CONFIRMED_DUPLICATE"
                    )
                    status_to_write = old_status if keep_status else c.get("status", "OPEN")
                    conn.execute(
                        """UPDATE duplicate_candidates
                           SET score = ?, confidence = ?, matched_fields = ?, explanation = ?,
                               status = ?, detected_at = ?
                           WHERE id = ?""",
                        (
                            float(c["score"]), float(c["confidence"]), m_fields_str,
                            c["explanation"], status_to_write,
                            c.get("detected_at", datetime.now(timezone.utc).isoformat()),
                            existing["id"],
                        ),
                    )
                else:
                    conn.execute(
                        """INSERT INTO duplicate_candidates
                           (document_id, matched_document_id, duplicate_type, score, confidence,
                            matched_fields, explanation, status, detected_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            d1, d2, dtype, float(c["score"]), float(c["confidence"]),
                            m_fields_str, c["explanation"], c.get("status", "OPEN"),
                            c.get("detected_at", datetime.now(timezone.utc).isoformat()),
                        ),
                    )
                saved += 1
        return saved

    def get_anomaly_findings(
        self,
        document_id: Optional[str] = None,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        anomaly_type: Optional[str] = None,
        min_score: Optional[float] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Retrieve anomaly findings with optional filters, joined with document metadata."""
        conds = []
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if document_id:
            conds.append("af.document_id = :doc_id")
            params["doc_id"] = document_id
        if status and status.upper() != "ALL":
            conds.append("af.status = :status")
            params["status"] = status.upper()
        if severity and severity.upper() != "ALL":
            conds.append("af.severity = :severity")
            params["severity"] = severity.upper()
        if anomaly_type and anomaly_type.upper() != "ALL":
            conds.append("af.anomaly_type = :a_type")
            params["a_type"] = anomaly_type
        if min_score is not None and min_score > 0:
            conds.append("af.score >= :min_score")
            params["min_score"] = float(min_score)

        where_clause = ("WHERE " + " AND ".join(conds)) if conds else ""
        sql = f"""
            SELECT af.*, d.filename, d.vendor_name, d.total_amount, d.invoice_date,
                   d.invoice_number, d.category, d.currency, d.verification_status
            FROM anomaly_findings af
            LEFT JOIN documents d ON af.document_id = d.document_id
            {where_clause}
            ORDER BY af.score DESC, af.detected_at DESC
            LIMIT :limit OFFSET :offset
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def get_duplicate_candidates(
        self,
        document_id: Optional[str] = None,
        status: Optional[str] = None,
        duplicate_type: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Retrieve duplicate candidates with both source and matched document metadata."""
        conds = []
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if document_id:
            conds.append("(dc.document_id = :doc_id OR dc.matched_document_id = :doc_id)")
            params["doc_id"] = document_id
        if status and status.upper() != "ALL":
            conds.append("dc.status = :status")
            params["status"] = status.upper()
        if duplicate_type and duplicate_type.upper() != "ALL":
            conds.append("dc.duplicate_type = :dup_type")
            params["dup_type"] = duplicate_type

        where_clause = ("WHERE " + " AND ".join(conds)) if conds else ""
        sql = f"""
            SELECT dc.*,
                   d1.filename AS filename, d1.vendor_name AS vendor_name,
                   d1.total_amount AS total_amount, d1.invoice_date AS invoice_date,
                   d1.invoice_number AS invoice_number, d1.category AS category,
                   d2.filename AS matched_filename, d2.vendor_name AS matched_vendor_name,
                   d2.total_amount AS matched_total_amount, d2.invoice_date AS matched_invoice_date,
                   d2.invoice_number AS matched_invoice_number, d2.category AS matched_category
            FROM duplicate_candidates dc
            LEFT JOIN documents d1 ON dc.document_id = d1.document_id
            LEFT JOIN documents d2 ON dc.matched_document_id = d2.document_id
            {where_clause}
            ORDER BY dc.score DESC, dc.detected_at DESC
            LIMIT :limit OFFSET :offset
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                mf = d.get("matched_fields", "[]")
                try:
                    d["matched_fields"] = json.loads(mf) if isinstance(mf, str) else mf
                except Exception:
                    d["matched_fields"] = []
                results.append(d)
            return results

    def update_anomaly_status(
        self,
        finding_id: int,
        new_status: str,
        reviewer: str,
        comment: str = "",
    ) -> bool:
        """Update an anomaly finding's status and write to immutable review history."""
        new_status = new_status.upper()
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            curr = conn.execute(
                "SELECT * FROM anomaly_findings WHERE id = ?", (finding_id,)
            ).fetchone()
            if not curr:
                return False
            old_status = curr["status"]
            doc_id = curr["document_id"]

            conn.execute(
                """INSERT INTO anomaly_review_history
                   (document_id, entity_type, entity_id, old_status, new_status, reviewer, comment, timestamp)
                   VALUES (?, 'anomaly', ?, ?, ?, ?, ?, ?)""",
                (doc_id, finding_id, old_status, new_status, reviewer, comment, now),
            )
            conn.execute(
                "UPDATE anomaly_findings SET status = ? WHERE id = ?",
                (new_status, finding_id),
            )
            self._log(
                conn,
                doc_id,
                "ANOMALY_STATUS_UPDATED",
                f"Finding #{finding_id} ({curr['anomaly_type']}) transitioned from {old_status} to {new_status} by {reviewer}. Note: {comment}",
            )
        return True

    def update_duplicate_status(
        self,
        candidate_id: int,
        new_status: str,
        reviewer: str,
        comment: str = "",
    ) -> bool:
        """Update a duplicate candidate's status and write to immutable review history."""
        new_status = new_status.upper()
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            curr = conn.execute(
                "SELECT * FROM duplicate_candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
            if not curr:
                return False
            old_status = curr["status"]
            doc_id = curr["document_id"]

            conn.execute(
                """INSERT INTO anomaly_review_history
                   (document_id, entity_type, entity_id, old_status, new_status, reviewer, comment, timestamp)
                   VALUES (?, 'duplicate', ?, ?, ?, ?, ?, ?)""",
                (doc_id, candidate_id, old_status, new_status, reviewer, comment, now),
            )
            conn.execute(
                "UPDATE duplicate_candidates SET status = ? WHERE id = ?",
                (new_status, candidate_id),
            )
            if new_status == "CONFIRMED_DUPLICATE":
                conn.execute(
                    "UPDATE documents SET verification_status = 'flagged', requires_review = 1 WHERE document_id = ?",
                    (doc_id,),
                )
            self._log(
                conn,
                doc_id,
                "DUPLICATE_STATUS_UPDATED",
                f"Duplicate pair #{candidate_id} ({curr['duplicate_type']}) transitioned from {old_status} to {new_status} by {reviewer}. Note: {comment}",
            )
        return True

    def get_anomaly_review_history(
        self,
        document_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Retrieve audit history of review decisions."""
        conds = []
        params: Dict[str, Any] = {"limit": limit}
        if document_id:
            conds.append("document_id = :doc_id")
            params["doc_id"] = document_id
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        sql = f"SELECT * FROM anomaly_review_history {where} ORDER BY timestamp DESC, id DESC LIMIT :limit"
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def rescan_anomalies(
        self,
        document_ids: Optional[List[str]] = None,
        sigma_threshold: float = 2.0,
    ) -> Dict[str, int]:
        """
        Idempotent full or targeted re-scan of documents for anomalies and duplicates.
        Preserves all existing human review decisions (RESOLVED, DISMISSED, etc.).
        """
        from phase11.anomaly_engine import CentralizedAnomalyEngine
        from phase11.duplicate_detector import MultiSignalDuplicateDetector

        all_docs = self.search_documents(limit=50000)
        if not all_docs:
            return {"scanned_documents": 0, "findings_detected": 0, "duplicates_detected": 0}

        target_docs = [d for d in all_docs if d.document_id in set(document_ids)] if document_ids else all_docs

        engine = CentralizedAnomalyEngine(sigma_threshold=sigma_threshold)
        pop_ctx = engine.build_population_context(all_docs)

        # 1. Anomaly detection
        all_findings = []
        for d in target_docs:
            rep = engine.evaluate_document(d, pop_ctx)
            for f in rep.findings:
                all_findings.append(f.to_dict())

        saved_findings = self.save_anomaly_findings(all_findings, preserve_reviewed=True)

        # 2. Duplicate detection
        dup_matches = MultiSignalDuplicateDetector.detect_duplicates_in_collection(all_docs)
        if document_ids:
            doc_id_set = set(document_ids)
            dup_matches = [m for m in dup_matches if m.document_id in doc_id_set or m.matched_document_id in doc_id_set]

        saved_dups = self.save_duplicate_candidates([m.to_dict() for m in dup_matches], preserve_reviewed=True)

        return {
            "scanned_documents": len(target_docs),
            "findings_detected": saved_findings,
            "duplicates_detected": saved_dups,
        }

    def anomaly_dashboard_summary(self) -> Dict[str, Any]:
        """Summary metrics across all anomaly findings and duplicate candidates."""
        with self._connect() as conn:
            stat_rows = conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM anomaly_findings GROUP BY status"
            ).fetchall()
            status_counts = {r["status"]: r["cnt"] for r in stat_rows}

            sev_rows = conn.execute(
                "SELECT severity, COUNT(*) AS cnt FROM anomaly_findings GROUP BY severity"
            ).fetchall()
            severity_counts = {r["severity"]: r["cnt"] for r in sev_rows}

            type_rows = conn.execute(
                "SELECT anomaly_type, COUNT(*) AS cnt FROM anomaly_findings GROUP BY anomaly_type ORDER BY cnt DESC"
            ).fetchall()
            type_counts = {r["anomaly_type"]: r["cnt"] for r in type_rows}

            dup_stat_rows = conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM duplicate_candidates GROUP BY status"
            ).fetchall()
            dup_status_counts = {r["status"]: r["cnt"] for r in dup_stat_rows}

            total_findings = sum(status_counts.values())
            total_duplicates = sum(dup_status_counts.values())

            # Count of unique flagged documents
            unique_flagged_docs = conn.execute(
                "SELECT COUNT(DISTINCT document_id) AS cnt FROM anomaly_findings"
            ).fetchone()["cnt"] or 0

        return {
            "total_findings": total_findings,
            "unique_flagged_docs": unique_flagged_docs,
            "open_findings": status_counts.get("OPEN", 0),
            "under_review_findings": status_counts.get("UNDER_REVIEW", 0),
            "resolved_findings": status_counts.get("RESOLVED", 0),
            "dismissed_findings": status_counts.get("DISMISSED", 0),
            "confirmed_duplicates": dup_status_counts.get("CONFIRMED_DUPLICATE", 0),
            "total_duplicate_candidates": total_duplicates,
            "severity_counts": severity_counts,
            "anomaly_type_counts": type_counts,
            "duplicate_status_counts": dup_status_counts,
        }

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _insert_line_items(self, conn: sqlite3.Connection, doc: FinancialDocument) -> None:
        for item in doc.line_items:
            conn.execute(
                """INSERT INTO line_items (document_id, description, quantity, unit_price, total, unit)
                   VALUES (:doc_id, :desc, :qty, :price, :total, :unit)""",
                {
                    "doc_id": doc.document_id,
                    "desc":   item.description,
                    "qty":    item.quantity,
                    "price":  item.unit_price,
                    "total":  item.total,
                    "unit":   item.unit,
                },
            )

    def _log(self, conn: sqlite3.Connection, document_id: str, action: str, details: str = "") -> None:
        conn.execute(
            "INSERT INTO processing_log (document_id, action, timestamp, details) VALUES (?, ?, ?, ?)",
            (document_id, action, datetime.now(timezone.utc).isoformat(), details),
        )
