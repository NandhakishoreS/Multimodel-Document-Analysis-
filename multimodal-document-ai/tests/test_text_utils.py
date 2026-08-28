"""Unit tests for text processing and lexical feature extraction."""

from preprocessing.text_utils import clean_text, extract_lexical_features


def test_clean_text():
    assert clean_text("  Hello \t World  ") == "Hello \t World"
    assert clean_text(None) == "[EMPTY]"
    assert clean_text("") == "[EMPTY]"
    assert clean_text("  ") == "[EMPTY]"
    assert clean_text("Total: $12.50") == "Total: $12.50"


def test_extract_lexical_features():
    feats_amount = extract_lexical_features("$45.99")
    assert feats_amount["is_currency_amount"] == 1.0
    assert feats_amount["has_digits"] == 1.0

    feats_upper = extract_lexical_features("INVOICE")
    assert feats_upper["is_upper"] == 1.0
    assert feats_upper["is_lower"] == 0.0

    feats_empty = extract_lexical_features("[EMPTY]")
    assert feats_empty["token_length"] == 0.0
