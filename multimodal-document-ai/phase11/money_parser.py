
"""
Phase 11 -- Shared Monetary Parsing Utilities.

This module is the *single source of truth* for all monetary string parsing
in the financial document pipeline.  It must be imported by any module that
needs to convert OCR-extracted text into a numeric amount.

Design goals
------------
* Decimal-based arithmetic: float is never used internally for calculation;
  callers that require float receive a float only at the boundary.
* Preserve decimal separators: the parser explicitly distinguishes a decimal
  point from a thousands separator.
* Indian number formats: supports lakh/crore grouping (e.g. 12,34,567.89)
  as well as standard international grouping (1,234,567.89).
* Currency-aware: strips INR, Rs, dollar, euro, pound prefixes/suffixes.
* Ambiguity flagging: returns a ParseResult dataclass with warnings.
* No silent zero substitution: missing/unparseable value -> ParseResult(value=None).

Label-aware field selection
---------------------------
select_best_amount(candidates) accepts a list of (label, ocr_text) tuples
and applies a priority ranking based on common financial document field names.

Usage
-----
    from phase11.money_parser import parse_monetary_string, select_best_amount
    result = parse_monetary_string("INR 3,281.00")
    float(result.value)   # -> 3281.0
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Currency stripping
# ---------------------------------------------------------------------------

_CURRENCY_PREFIX_RE = re.compile(
    r"^\s*(?:[\u20b9\$\u20ac\u00a3\u00a5]|(?:RS|INR|USD|EUR|GBP|CNY|JPY)\.?)\s*",
    re.IGNORECASE,
)

_CURRENCY_SUFFIX_RE = re.compile(
    r"\s*(?:INR|USD|EUR|GBP|CNY|JPY)\s*$",
    re.IGNORECASE,
)

_FALLBACK_RE = re.compile(
    r"(?:[\u20b9\$\u20ac\u00a3\u00a5]?\s*)"
    r"(\d{1,3}(?:[,\s]\d{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
)


# ---------------------------------------------------------------------------
# Label priority table
# ---------------------------------------------------------------------------

_LABEL_PRIORITY: Dict[str, int] = {
    "net amount":           100,
    "net payable":          100,
    "net total":            100,
    "amount payable":       100,
    "total payable":         95,
    "total due":             95,
    "grand total":           90,
    "net":                   88,
    "total amount":          85,
    "invoice total":         85,
    "bill total":            85,
    "final amount":          85,
    "final total":           85,
    "amount due":            85,
    "total":                 70,
    "subtotal":              40,
    "sub total":             40,
    "sub-total":             40,
    "vat":                   30,
    "tax":                   30,
    "gst":                   30,
    "service tax":           25,
    "service charge":        25,
    "service charges":       25,
    "cgst":                  25,
    "sgst":                  25,
    "igst":                  25,
    "cess":                  20,
    "surcharge":             20,
    "quantity":             -50,
    "qty":                  -50,
    "total quantity":       -50,
    "units":                -50,
    "pcs":                  -50,
    "pieces":               -50,
    "nos":                  -50,
}


# ---------------------------------------------------------------------------
# ParseResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    """
    Result of parsing a single monetary string.

    Attributes
    ----------
    ocr_text : str
        The raw input string exactly as received from OCR or entity extraction.
    normalised : str
        The string after currency stripping and whitespace normalisation.
    value : Optional[Decimal]
        The parsed monetary value, or None if parsing failed.
    warnings : List[str]
        Human-readable warnings.  Empty list means parse was unambiguous.
    is_ambiguous : bool
        True when the parser could not determine separators unambiguously.
    """

    ocr_text: str
    normalised: str = ""
    value: Optional[Decimal] = None
    warnings: List[str] = field(default_factory=list)
    is_ambiguous: bool = False

    @property
    def as_float(self) -> Optional[float]:
        """Return the parsed value as Python float, or None."""
        return float(self.value) if self.value is not None else None

    @property
    def succeeded(self) -> bool:
        """True if a numeric value was successfully extracted."""
        return self.value is not None


# ---------------------------------------------------------------------------
# Internal numeric conversion
# ---------------------------------------------------------------------------

def _strip_currency(text: str) -> str:
    """Remove currency symbols and codes from both ends of the text."""
    s = _CURRENCY_PREFIX_RE.sub("", text)
    s = _CURRENCY_SUFFIX_RE.sub("", s)
    s = s.strip().lstrip("\u20b9$\u20ac\u00a3\u00a5").strip()
    return s


def _parse_numeric_string(s: str) -> Decimal:
    """
    Convert a cleaned numeric string to Decimal.

    Handles:
      - Standard international:  1,234.56  ->  1234.56
      - Indian lakh/crore:       12,34,567.89  ->  1234567.89
      - No-comma decimal:        3281.00  ->  3281.00
      - Single decimal place:    258.7  ->  258.7
      - Zero values:             0.00  ->  0.00

    Comma disambiguation rule:
      - Period present -> period is decimal separator; commas are thousands.
      - No period, one comma, 1-digit then 2-digit groups -> treat comma as decimal.
      - Otherwise remove all commas (thousands separators).

    Raises
    ------
    InvalidOperation
        If the resulting string is not a valid Decimal literal.
    """
    s = s.replace(" ", "")

    if "." in s:
        return Decimal(s.replace(",", ""))

    parts = s.split(",")
    if len(parts) == 1:
        return Decimal(s)

    last = parts[-1]
    if len(last) <= 2 and len(parts) == 2 and len(parts[0]) == 1:
        # Ambiguous "1,23" -- treat as decimal "1.23"
        return Decimal(f"{parts[0]}.{last}")

    return Decimal(s.replace(",", ""))


# ---------------------------------------------------------------------------
# Public parsing API
# ---------------------------------------------------------------------------

def parse_monetary_string(text: Optional[str]) -> ParseResult:
    """
    Parse a monetary string into a ParseResult.

    This is the canonical parser used throughout the pipeline.

    Supported formats
    -----------------
    - Plain integers:               258, 3281
    - Simple decimals:              258.75, 258.7, 258.00
    - Standard thousands:           1,234.56
    - Indian thousands:             12,34,567.89, 1,23,456
    - Currency prefix:              INR 3,281.00, Rs3281.00
    - Zero-valued:                  0.00

    Returns
    -------
    ParseResult
        Always returns a ParseResult.  Check .succeeded before using .value.
    """
    if text is None or str(text).strip() == "":
        return ParseResult(
            ocr_text=str(text) if text is not None else "",
            warnings=["Input is empty or None."],
        )

    raw = str(text)
    warnings: List[str] = []
    is_ambiguous = False

    stripped = _strip_currency(raw)

    if not stripped:
        return ParseResult(
            ocr_text=raw,
            normalised=stripped,
            warnings=["No numeric content found after stripping currency tokens."],
        )

    normalised = stripped

    # Attempt structured numeric parse
    try:
        value = _parse_numeric_string(normalised)

        if value < Decimal("0"):
            warnings.append(f"Parsed value is negative ({value}). Please verify.")
        if value == Decimal("0") and normalised not in ("0", "0.0", "0.00", "0.000"):
            warnings.append(
                f"Parsed value is zero from non-trivial input {normalised!r}. "
                "Possible OCR misread."
            )

        return ParseResult(
            ocr_text=raw,
            normalised=normalised,
            value=value,
            warnings=warnings,
            is_ambiguous=is_ambiguous,
        )

    except (InvalidOperation, ValueError):
        pass

    # Fallback: find any embedded number
    matches = _FALLBACK_RE.findall(raw)
    for m in matches:
        m = m.strip()
        if not m:
            continue
        try:
            candidate = _parse_numeric_string(_strip_currency(m))
            warnings.append(
                f"Could not parse {raw!r} directly; "
                f"extracted candidate {m!r} from within the text."
            )
            return ParseResult(
                ocr_text=raw,
                normalised=m,
                value=candidate,
                warnings=warnings,
                is_ambiguous=True,
            )
        except (InvalidOperation, ValueError):
            continue

    return ParseResult(
        ocr_text=raw,
        normalised=normalised,
        warnings=[f"Could not parse {raw!r} as a monetary amount."],
    )


def parse_amount_to_float(text: Optional[str]) -> Optional[float]:
    """
    Convenience wrapper: parse and return a float, or None.

    Drop-in replacement for the old _parse_amount() function.
    Warnings are silently discarded; call parse_monetary_string() directly
    if you need audit information.
    """
    result = parse_monetary_string(text)
    return result.as_float


# ---------------------------------------------------------------------------
# Label-aware field selection
# ---------------------------------------------------------------------------

def _label_score(label: str) -> int:
    """Return the priority score for a given label string."""
    normalised = " ".join(label.lower().split())
    if normalised in _LABEL_PRIORITY:
        return _LABEL_PRIORITY[normalised]
    best = 0
    for key, score in _LABEL_PRIORITY.items():
        if key in normalised:
            if score > best:
                best = score
    return best


@dataclass
class FieldCandidate:
    """A single (label, amount) candidate for field selection."""
    label: str
    ocr_text: str
    parse_result: ParseResult
    label_score: int = 0

    def __post_init__(self) -> None:
        self.label_score = _label_score(self.label)


def select_best_amount(
    candidates: Sequence[Tuple[str, str]],
) -> Tuple[Optional[Decimal], List[FieldCandidate]]:
    """
    Select the most appropriate monetary amount from (label, text) pairs.

    Strategy
    --------
    1. Parse all candidates using parse_monetary_string().
    2. Discard candidates where value is None.
    3. Discard candidates whose label score is strictly negative (qty, pcs, ...).
    4. Rank by label_score descending, then by value descending as tie-breaker.
    5. Return the top-ranked value and the full sorted list.

    This function is intentionally general.  Domain knowledge lives in
    _LABEL_PRIORITY: "Net Amount" > "Grand Total" > "Total" > "Subtotal"
    > "Service Tax" > positive scores > "Quantity" (negative score, excluded).

    Parameters
    ----------
    candidates : Sequence[Tuple[str, str]]
        Pairs of (field_label, ocr_text_value).

    Returns
    -------
    (best_value, ranked_candidates)
        best_value : Optional[Decimal]
        ranked_candidates : List[FieldCandidate]
    """
    parsed: List[FieldCandidate] = []
    for label, text in candidates:
        pr = parse_monetary_string(text)
        parsed.append(FieldCandidate(label=label, ocr_text=text, parse_result=pr))

    valid = [c for c in parsed if c.parse_result.succeeded and c.label_score >= 0]

    if not valid:
        parsed.sort(
            key=lambda c: (c.label_score, c.parse_result.value or Decimal(0)),
            reverse=True,
        )
        return None, parsed

    valid.sort(
        key=lambda c: (c.label_score, c.parse_result.value or Decimal(0)),
        reverse=True,
    )

    discarded = [c for c in parsed if c not in valid]
    ranked = valid + discarded

    return valid[0].parse_result.value, ranked

