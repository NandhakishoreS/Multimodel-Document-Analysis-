"""
Phase 15 — Multi-Signal Duplicate Detection Engine.

Detects duplicate document and invoice candidates across independent signals:
1. EXACT_FILE_DUPLICATE: Identical cryptographic SHA-256 digest.
2. STRONG_DUPLICATE_CANDIDATE: Exact match across vendor, invoice number, date, and total amount.
3. DUPLICATE_INVOICE_ID: Same vendor and invoice number.
4. DUPLICATE_FINANCIAL_SIGNATURE: Same vendor, date, and total amount.
5. NEAR_DUPLICATE_RECORD: Same vendor with near date (+/- 3 days) and near total amount (+/- 1% or <= Rs 5).

Provides deterministic scoring, candidate blocking to prevent O(N^2) bottlenecks,
and structured explainability.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from phase11.schema import FinancialDocument


# ---------------------------------------------------------------------------
# Duplicate Types & Weights
# ---------------------------------------------------------------------------

class DuplicateType:
    EXACT_FILE_DUPLICATE = "EXACT_FILE_DUPLICATE"
    STRONG_DUPLICATE_CANDIDATE = "STRONG_DUPLICATE_CANDIDATE"
    DUPLICATE_INVOICE_ID = "DUPLICATE_INVOICE_ID"
    DUPLICATE_FINANCIAL_SIGNATURE = "DUPLICATE_FINANCIAL_SIGNATURE"
    NEAR_DUPLICATE_RECORD = "NEAR_DUPLICATE_RECORD"


DUPLICATE_PRIORITY = {
    DuplicateType.EXACT_FILE_DUPLICATE: 5,
    DuplicateType.STRONG_DUPLICATE_CANDIDATE: 4,
    DuplicateType.DUPLICATE_INVOICE_ID: 3,
    DuplicateType.DUPLICATE_FINANCIAL_SIGNATURE: 2,
    DuplicateType.NEAR_DUPLICATE_RECORD: 1,
}

DUPLICATE_METRICS = {
    DuplicateType.EXACT_FILE_DUPLICATE: {"score": 1.00, "confidence": 1.00, "severity": "CRITICAL"},
    DuplicateType.STRONG_DUPLICATE_CANDIDATE: {"score": 0.95, "confidence": 0.95, "severity": "CRITICAL"},
    DuplicateType.DUPLICATE_INVOICE_ID: {"score": 0.85, "confidence": 0.90, "severity": "HIGH"},
    DuplicateType.DUPLICATE_FINANCIAL_SIGNATURE: {"score": 0.80, "confidence": 0.85, "severity": "HIGH"},
    DuplicateType.NEAR_DUPLICATE_RECORD: {"score": 0.65, "confidence": 0.70, "severity": "MEDIUM"},
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class DuplicateMatch:
    """Represents a pairwise duplicate candidate relationship."""
    document_id: str
    matched_document_id: str
    duplicate_type: str
    score: float
    confidence: float
    matched_fields: List[str]
    explanation: str
    status: str = "OPEN"
    detected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Normalization Utilities
# ---------------------------------------------------------------------------

_CORP_SUFFIXES = re.compile(
    r"\b(pvt|ltd|limited|private|inc|incorporated|corp|corporation|llc|llp|gmbh|co|company)\b\.?",
    re.IGNORECASE,
)
_PUNCT_REGEX = re.compile(r"[^\w\s]")


def normalize_vendor(name: Optional[str]) -> str:
    """Normalize vendor name for robust matching: lowercase, strip corporate suffixes and punctuation."""
    if not name or not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = _CORP_SUFFIXES.sub("", s)
    s = _PUNCT_REGEX.sub(" ", s)
    tokens = [t for t in s.split() if len(t) > 1]
    return " ".join(tokens)


def normalize_invoice_number(inv: Optional[str]) -> str:
    """Normalize invoice number by removing punctuation, spaces, and leading zeros."""
    if not inv or not isinstance(inv, str):
        return ""
    s = inv.strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    s = re.sub(r"(?<=[A-Z])0+(?=[0-9])", "", s)
    s = s.lstrip("0")
    return s


def parse_date_safe(date_str: Optional[str]) -> Optional[date]:
    """Parse ISO date (YYYY-MM-DD) or first 10 characters safely."""
    if not date_str or not isinstance(date_str, str):
        return None
    cleaned = date_str.strip()[:10]
    try:
        return datetime.strptime(cleaned, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Pairwise Duplicate Evaluator
# ---------------------------------------------------------------------------

def evaluate_pair(
    doc1: FinancialDocument,
    doc2: FinancialDocument,
) -> Optional[DuplicateMatch]:
    """
    Evaluate two documents across all 5 duplicate signals.
    Returns the highest-priority DuplicateMatch if a match is found, else None.
    """
    if doc1.document_id == doc2.document_id:
        return None

    # Signal 1: Exact File Hash Match
    if doc1.file_hash and doc2.file_hash and doc1.file_hash == doc2.file_hash:
        m = DUPLICATE_METRICS[DuplicateType.EXACT_FILE_DUPLICATE]
        return DuplicateMatch(
            document_id=doc1.document_id,
            matched_document_id=doc2.document_id,
            duplicate_type=DuplicateType.EXACT_FILE_DUPLICATE,
            score=m["score"],
            confidence=m["confidence"],
            matched_fields=["file_hash"],
            explanation=f"Exact cryptographic SHA-256 digest match with {doc2.filename or doc2.document_id[:8]}.",
        )

    # Normalize fields
    v1 = normalize_vendor(doc1.vendor_name)
    v2 = normalize_vendor(doc2.vendor_name)
    has_vendor = bool(v1 and v2 and (v1 == v2 or v1 in v2 or v2 in v1))

    i1 = normalize_invoice_number(doc1.invoice_number)
    i2 = normalize_invoice_number(doc2.invoice_number)
    has_inv = bool(i1 and i2 and i1 == i2)

    d1 = parse_date_safe(doc1.invoice_date)
    d2 = parse_date_safe(doc2.invoice_date)
    has_date = bool(d1 and d2 and d1 == d2)

    amt1 = doc1.total_amount
    amt2 = doc2.total_amount
    has_amt = bool(
        amt1 is not None and amt2 is not None and abs(amt1 - amt2) <= 0.01 and amt1 > 0
    )

    # Signal 2: Strong Composite Candidate (Vendor + Invoice # + Date + Amount)
    if has_vendor and has_inv and has_date and has_amt:
        m = DUPLICATE_METRICS[DuplicateType.STRONG_DUPLICATE_CANDIDATE]
        return DuplicateMatch(
            document_id=doc1.document_id,
            matched_document_id=doc2.document_id,
            duplicate_type=DuplicateType.STRONG_DUPLICATE_CANDIDATE,
            score=m["score"],
            confidence=m["confidence"],
            matched_fields=["vendor_name", "invoice_number", "invoice_date", "total_amount"],
            explanation=(
                f"Strong composite match on vendor ('{doc1.vendor_name}'), "
                f"invoice #{doc1.invoice_number}, date {doc1.invoice_date}, and amount ₹{amt1:,.2f}."
            ),
        )

    # Signal 3: Duplicate Invoice ID (Vendor + Invoice #)
    if has_vendor and has_inv:
        m = DUPLICATE_METRICS[DuplicateType.DUPLICATE_INVOICE_ID]
        return DuplicateMatch(
            document_id=doc1.document_id,
            matched_document_id=doc2.document_id,
            duplicate_type=DuplicateType.DUPLICATE_INVOICE_ID,
            score=m["score"],
            confidence=m["confidence"],
            matched_fields=["vendor_name", "invoice_number"],
            explanation=f"Identical vendor ('{doc1.vendor_name}') and invoice number #{doc1.invoice_number}.",
        )

    # Signal 4: Duplicate Financial Signature (Vendor + Date + Amount)
    if has_vendor and has_date and has_amt:
        m = DUPLICATE_METRICS[DuplicateType.DUPLICATE_FINANCIAL_SIGNATURE]
        return DuplicateMatch(
            document_id=doc1.document_id,
            matched_document_id=doc2.document_id,
            duplicate_type=DuplicateType.DUPLICATE_FINANCIAL_SIGNATURE,
            score=m["score"],
            confidence=m["confidence"],
            matched_fields=["vendor_name", "invoice_date", "total_amount"],
            explanation=(
                f"Duplicate financial signature: vendor '{doc1.vendor_name}', "
                f"date {doc1.invoice_date}, and total amount ₹{amt1:,.2f}."
            ),
        )

    # Signal 5: Near-Duplicate Financial Record
    if has_vendor and d1 and d2 and amt1 is not None and amt2 is not None and amt1 > 0 and amt2 > 0:
        day_diff = abs((d1 - d2).days)
        amt_diff = abs(amt1 - amt2)
        amt_pct_diff = (amt_diff / max(amt1, amt2)) * 100

        # Conditions: within 3 days and amount within 1% or difference <= Rs 5
        if day_diff <= 3 and (amt_pct_diff <= 1.0 or amt_diff <= 5.0):
            m = DUPLICATE_METRICS[DuplicateType.NEAR_DUPLICATE_RECORD]
            return DuplicateMatch(
                document_id=doc1.document_id,
                matched_document_id=doc2.document_id,
                duplicate_type=DuplicateType.NEAR_DUPLICATE_RECORD,
                score=m["score"],
                confidence=m["confidence"],
                matched_fields=["vendor_name", "near_date", "near_amount"],
                explanation=(
                    f"Near-duplicate candidate for vendor '{doc1.vendor_name}': "
                    f"dates differ by {day_diff} day(s) ({doc1.invoice_date} vs {doc2.invoice_date}) "
                    f"and amounts differ by ₹{amt_diff:.2f} ({amt_pct_diff:.1f}%)."
                ),
            )

    return None


# ---------------------------------------------------------------------------
# Collection-Level Multi-Signal Detector with Candidate Blocking
# ---------------------------------------------------------------------------

class MultiSignalDuplicateDetector:
    """
    Orchestrates duplicate candidate detection across single documents
    or batch collections using candidate blocking to ensure high throughput.
    """

    @classmethod
    def detect_duplicates_in_collection(
        cls,
        documents: List[FinancialDocument],
    ) -> List[DuplicateMatch]:
        """
        Detect duplicates within an in-memory collection using candidate blocking.
        Scales to large datasets without full O(N^2) pairwise comparisons.
        """
        if len(documents) < 2:
            return []

        matches: Dict[Tuple[str, str], DuplicateMatch] = {}

        # 1. Block by file_hash
        hash_blocks: Dict[str, List[FinancialDocument]] = {}
        for doc in documents:
            if doc.file_hash:
                hash_blocks.setdefault(doc.file_hash, []).append(doc)

        for _, block in hash_blocks.items():
            if len(block) > 1:
                for i in range(len(block)):
                    for j in range(i + 1, len(block)):
                        match = evaluate_pair(block[i], block[j])
                        if match:
                            cls._record_bidirectional(matches, match)

        # 2. Block by normalized (vendor, invoice_number)
        inv_blocks: Dict[Tuple[str, str], List[FinancialDocument]] = {}
        for doc in documents:
            v_norm = normalize_vendor(doc.vendor_name)
            i_norm = normalize_invoice_number(doc.invoice_number)
            if v_norm and i_norm:
                inv_blocks.setdefault((v_norm, i_norm), []).append(doc)

        for _, block in inv_blocks.items():
            if len(block) > 1:
                for i in range(len(block)):
                    for j in range(i + 1, len(block)):
                        match = evaluate_pair(block[i], block[j])
                        if match:
                            cls._record_bidirectional(matches, match)

        # 3. Block by normalized vendor (for signature and near-duplicates)
        vendor_blocks: Dict[str, List[FinancialDocument]] = {}
        for doc in documents:
            v_norm = normalize_vendor(doc.vendor_name)
            if v_norm:
                vendor_blocks.setdefault(v_norm, []).append(doc)

        for _, block in vendor_blocks.items():
            if len(block) > 1:
                for i in range(len(block)):
                    for j in range(i + 1, len(block)):
                        match = evaluate_pair(block[i], block[j])
                        if match:
                            cls._record_bidirectional(matches, match)

        return list(matches.values())

    @classmethod
    def detect_duplicates_for_document(
        cls,
        doc: FinancialDocument,
        db: Any,
    ) -> List[DuplicateMatch]:
        """
        Find duplicate candidates for a single document against existing records in SQLite.
        Uses targeted indexed queries for candidate blocking.
        """
        if not doc or not doc.document_id:
            return []

        candidates: Dict[str, FinancialDocument] = {}

        # 1. Query by exact file hash
        if doc.file_hash:
            h_doc = db.get_document_by_hash(doc.file_hash)
            if h_doc and h_doc.document_id != doc.document_id:
                candidates[h_doc.document_id] = h_doc

        # 2. Query by invoice number
        if doc.invoice_number:
            inv_docs = db.search_documents(invoice_number=doc.invoice_number, limit=50)
            for d in inv_docs:
                if d.document_id != doc.document_id:
                    candidates[d.document_id] = d

        # 3. Query by vendor
        if doc.vendor_name:
            v_docs = db.search_documents(vendor=doc.vendor_name, limit=100)
            for d in v_docs:
                if d.document_id != doc.document_id:
                    candidates[d.document_id] = d

        # Evaluate candidate pairs
        results: List[DuplicateMatch] = []
        for other in candidates.values():
            match = evaluate_pair(doc, other)
            if match:
                results.append(match)

        # Sort by score descending
        results.sort(key=lambda m: m.score, reverse=True)
        return results

    # Alias for detect_duplicates_in_collection
    scan_all_pairs = detect_duplicates_in_collection

    @staticmethod
    def _record_bidirectional(
        matches: Dict[Tuple[str, str], DuplicateMatch],
        match: DuplicateMatch,
    ) -> None:
        """Store match, upgrading if a higher-priority duplicate type was detected."""
        pair_key = (match.document_id, match.matched_document_id)
        existing = matches.get(pair_key)
        if not existing or DUPLICATE_PRIORITY[match.duplicate_type] > DUPLICATE_PRIORITY[existing.duplicate_type]:
            matches[pair_key] = match
