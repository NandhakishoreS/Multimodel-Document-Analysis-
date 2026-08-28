"""Text preprocessing and lexical feature extraction utilities."""

import re
import unicodedata
from typing import Any, Dict, List, Optional


# Regex patterns for common document token types
REGEX_NUMERIC = re.compile(r"^[-+]?[0-9]+([.,][0-9]+)?$")
REGEX_CURRENCY_AMOUNT = re.compile(r"^[$€£¥₹]?\s*[0-9]+([.,][0-9]{2})?$")
REGEX_DATE = re.compile(
    r"^(\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}|\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})$",
    re.IGNORECASE,
)
REGEX_PHONE = re.compile(r"^(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}$")


def clean_text(
    text: Optional[str],
    unicode_form: str = "NFKC",
    strip_whitespace: bool = True,
    replace_empty: str = "[EMPTY]",
) -> str:
    """
    Standardize text token with Unicode normalization while preserving document OCR content.

    Args:
        text: Raw OCR word text.
        unicode_form: Normalization form ('NFKC', 'NFC', etc.).
        strip_whitespace: Whether to strip leading/trailing whitespace.
        replace_empty: Replacement string if text is empty or None.

    Returns:
        Cleaned text string.
    """
    if text is None:
        return replace_empty

    text = str(text)

    # Unicode normalization
    if unicode_form:
        text = unicodedata.normalize(unicode_form, text)

    # Remove non-printable control characters except standard space
    text = "".join(ch for ch in text if ch.isprintable() or ch in ("\t", "\n"))

    if strip_whitespace:
        text = text.strip()

    if not text:
        return replace_empty

    return text


def extract_lexical_features(word: str) -> Dict[str, Any]:
    """
    Extract informative lexical and orthographic features from a token for text baseline models.

    Args:
        word: Cleaned text token.

    Returns:
        Dict of binary/numerical lexical features.
    """
    cleaned = clean_text(word)
    is_empty = cleaned == "[EMPTY]" or len(cleaned) == 0

    if is_empty:
        return {
            "token_length": 0,
            "is_upper": 0.0,
            "is_lower": 0.0,
            "is_title": 0.0,
            "is_numeric": 0.0,
            "is_currency_amount": 0.0,
            "is_date": 0.0,
            "has_digits": 0.0,
            "has_punct": 0.0,
            "has_colon": 0.0,
            "has_slash_or_dash": 0.0,
            "has_at_symbol": 0.0,
        }

    return {
        "token_length": min(float(len(cleaned)), 30.0),
        "is_upper": 1.0 if cleaned.isupper() else 0.0,
        "is_lower": 1.0 if cleaned.islower() else 0.0,
        "is_title": 1.0 if cleaned.istitle() else 0.0,
        "is_numeric": 1.0 if REGEX_NUMERIC.match(cleaned) else 0.0,
        "is_currency_amount": 1.0 if REGEX_CURRENCY_AMOUNT.match(cleaned) else 0.0,
        "is_date": 1.0 if REGEX_DATE.match(cleaned) else 0.0,
        "has_digits": 1.0 if any(c.isdigit() for c in cleaned) else 0.0,
        "has_punct": 1.0 if any(not c.isalnum() and not c.isspace() for c in cleaned) else 0.0,
        "has_colon": 1.0 if ":" in cleaned else 0.0,
        "has_slash_or_dash": 1.0 if any(c in "-/\\." for c in cleaned) else 0.0,
        "has_at_symbol": 1.0 if "@" in cleaned else 0.0,
    }
