"""
Phase 11 — Financial Document Schema.

Defines the normalized FinancialDocument dataclass and ClassificationResult
used throughout the Phase 11 business-intelligence layer.

All fields are Optional; missing information is represented as None rather
than fabricated values.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Classification result (returned by BillCategoryClassifier)
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Bill category classification with explainability evidence."""
    category: str                   # e.g. "Electricity", "Unknown"
    sub_category: Optional[str]     # e.g. "Utilities"
    confidence: float               # 0.0 – 1.0 normalised score
    evidence: List[str]             # Matched keyword tokens that drove the decision
    reason: str = ""                # Human-readable explainability reason
    requires_review: bool = False   # Flag indicating low confidence or Unknown category


# ---------------------------------------------------------------------------
# Line item (individual product/service row on invoice)
# ---------------------------------------------------------------------------

@dataclass
class LineItem:
    """A single line item entry on a financial document."""
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None
    unit: Optional[str] = None


# ---------------------------------------------------------------------------
# Core financial document schema
# ---------------------------------------------------------------------------

# Valid verification status values
VERIFICATION_STATUSES = ("pending", "verified", "modified", "flagged", "rejected")

# Supported currency codes
DEFAULT_CURRENCY = "INR"


@dataclass
class FinancialDocument:
    """
    Normalised representation of an institutional financial document
    (invoice, bill, receipt, form).

    All monetary amounts are stored as Python float (or None if absent).
    All date fields are ISO-8601 strings ("YYYY-MM-DD") or None.

    Fields sourced from SROIE raw_entities_dict:
        COMPANY  → vendor_name
        DATE     → invoice_date
        ADDRESS  → address
        TOTAL    → total_amount (parsed numeric)

    Fields from CORD:
        MENU     → line_items (text description)
        SUBTOTAL → subtotal
        TOTAL    → total_amount

    Fields populated by the classifier:
        category, sub_category, classification_confidence, classification_evidence
    """

    # Identity
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    source_file: str = ""

    # Document type metadata
    document_type: str = "invoice"          # invoice | receipt | bill | form
    benchmark_domain: str = ""              # SROIE | CORD | FUNSD (for traceability)
    model_name: str = ""

    # Business classification
    category: str = "Unknown"              # Electricity | Water | … | Unknown
    sub_category: Optional[str] = None     # Utilities | Operations | Administrative | Other
    classification_confidence: float = 0.0
    classification_evidence: List[str] = field(default_factory=list)

    # Vendor / supplier
    vendor_name: Optional[str] = None
    address: Optional[str] = None

    # Invoice metadata
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None     # ISO date string or raw OCR date
    due_date: Optional[str] = None

    # Financial fields
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    total_amount: Optional[float] = None
    currency: str = DEFAULT_CURRENCY

    # Line items
    line_items: List[LineItem] = field(default_factory=list)

    # Extraction quality
    extraction_confidence: float = 0.0     # Mean entity confidence
    ocr_word_count: int = 0
    ocr_raw_text: str = ""

    # Lifecycle
    verification_status: str = "pending"   # pending | verified | flagged | rejected
    processing_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Free-form notes for human review
    notes: str = ""

    # Anomaly flags (populated by anomaly detector)
    anomaly_flags: List[str] = field(default_factory=list)

    # Phase 12 Ingestion & Processing Metadata
    file_hash: Optional[str] = None
    file_size: Optional[int] = None
    processing_status: str = "COMPLETED"   # QUEUED | VALIDATING | PROCESSING | OCR | EXTRACTING | CLASSIFYING | SAVING | COMPLETED | FAILED | DUPLICATE | REJECTED
    processing_stage: Optional[str] = None
    processing_error: Optional[str] = None
    processing_started_at: Optional[str] = None
    processing_completed_at: Optional[str] = None
    processing_duration: Optional[float] = None

    # Phase 12 Classification Review & Explainability
    ai_category: Optional[str] = None
    ai_confidence: Optional[float] = None
    final_category: Optional[str] = None
    requires_review: bool = False
    classification_reason: str = ""

    # Phase 13 AI Value Preservation & Auditable Editing
    ai_vendor_name: Optional[str] = None
    ai_invoice_date: Optional[str] = None
    ai_invoice_number: Optional[str] = None
    ai_total_amount: Optional[float] = None
    is_modified: bool = False

    # ---------------------------------------------------------------------------
    # Validation & Quality helpers
    # ---------------------------------------------------------------------------

    def detect_missing_fields(self) -> List[str]:
        """Return list of critical financial fields that are missing or unknown."""
        missing = []
        if not self.vendor_name or not self.vendor_name.strip():
            missing.append("Vendor Name")
        if not self.invoice_date or not self.invoice_date.strip():
            missing.append("Invoice Date")
        if self.total_amount is None:
            missing.append("Total Amount")
        if not self.category or self.category == "Unknown":
            missing.append("Category")
        return missing

    def validate(self) -> List[str]:
        """Return list of validation warnings/errors (empty = clean)."""
        warnings: List[str] = []
        if self.verification_status not in VERIFICATION_STATUSES:
            warnings.append(
                f"Invalid verification_status '{self.verification_status}'. "
                f"Must be one of {VERIFICATION_STATUSES}."
            )
        if self.total_amount is not None and self.total_amount < 0:
            warnings.append("total_amount is negative.")
        if self.subtotal is not None and self.subtotal < 0:
            warnings.append("subtotal is negative.")
        if self.tax_amount is not None and self.tax_amount < 0:
            warnings.append("tax_amount is negative.")
        if self.subtotal is not None and self.tax_amount is not None:
            expected = (self.subtotal or 0) + (self.tax_amount or 0)
            if self.total_amount is not None and abs(expected - self.total_amount) > 1.0:
                warnings.append(
                    f"subtotal ({self.subtotal}) + tax ({self.tax_amount}) "
                    f"≠ total ({self.total_amount}). Possible OCR error."
                )
        return warnings

    # ---------------------------------------------------------------------------
    # Serialisation helpers
    # ---------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serialisable dict."""
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        """Export as formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinancialDocument":
        """Reconstruct from a plain dict (e.g., loaded from JSON or SQLite row)."""
        # Deserialise nested line_items if present
        raw_items = data.pop("line_items", []) or []
        line_items = [
            LineItem(**item) if isinstance(item, dict) else item
            for item in raw_items
        ]
        # classification_evidence may be stored as JSON string in SQLite
        ev = data.get("classification_evidence", [])
        if isinstance(ev, str):
            try:
                data["classification_evidence"] = json.loads(ev)
            except Exception:
                data["classification_evidence"] = []
        # anomaly_flags same
        af = data.get("anomaly_flags", [])
        if isinstance(af, str):
            try:
                data["anomaly_flags"] = json.loads(af)
            except Exception:
                data["anomaly_flags"] = []
        if "is_modified" in data:
            data["is_modified"] = bool(data["is_modified"])
        if "requires_review" in data:
            data["requires_review"] = bool(data["requires_review"])
        return cls(line_items=line_items, **data)

    def touch(self) -> None:
        """Update the updated_at timestamp to now."""
        self.updated_at = datetime.now(timezone.utc).isoformat()
