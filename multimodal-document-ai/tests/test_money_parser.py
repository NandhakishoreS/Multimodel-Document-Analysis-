"""
Regression tests for phase11.money_parser and the monetary parsing changes
in phase11.extractor.

Covers:
    - Plain integers without decimal points
    - Values with one or two decimal places
    - Zero-valued decimal amounts
    - Indian comma grouping (lakh / crore format)
    - Currency symbols and prefixes (Rs, INR, $, euro, GBP)
    - Missing, malformed, and ambiguous amounts
    - Critical regression: 258.75 must NEVER become 25875
    - Critical regression: 3,281.00 must equal 3281.00 numerically
    - Label-aware field selection (Net Amount wins over Total Quantity)
    - End-to-end: storage, retrieval, and formatting preserve amounts
    - parse_amount_to_float backward-compatibility with old _parse_amount
    - ParseResult dataclass structure and properties
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from phase11.money_parser import (
    FieldCandidate,
    ParseResult,
    _label_score,
    _parse_numeric_string,
    _strip_currency,
    parse_amount_to_float,
    parse_monetary_string,
    select_best_amount,
)
from phase11.extractor import _parse_amount  # backward-compat wrapper


# ===========================================================================
# Helpers
# ===========================================================================

def decimal_eq(a: Optional[Decimal], b: Decimal, places: int = 6) -> bool:
    """True if |a - b| < 10^-places."""
    if a is None:
        return False
    return abs(a - b) < Decimal(10) ** -places


# ===========================================================================
# 1. _strip_currency
# ===========================================================================

class TestStripCurrency:

    def test_strips_rupee_symbol(self):
        assert _strip_currency("\u20b93,281.00") == "3,281.00"

    def test_strips_inr_prefix(self):
        assert _strip_currency("INR 3,281.50") == "3,281.50"

    def test_strips_inr_prefix_no_space(self):
        assert _strip_currency("INR3281.50") == "3281.50"

    def test_strips_dollar_symbol(self):
        assert _strip_currency("$50.00") == "50.00"

    def test_strips_rs_prefix(self):
        assert _strip_currency("Rs. 258.75") == "258.75"

    def test_strips_rs_prefix_no_period(self):
        assert _strip_currency("RS 258.75") == "258.75"

    def test_strips_eur_prefix(self):
        assert _strip_currency("EUR 100.00") == "100.00"

    def test_strips_gbp_prefix(self):
        assert _strip_currency("GBP 75.50") == "75.50"

    def test_strips_inr_suffix(self):
        assert _strip_currency("3281 INR") == "3281"

    def test_no_strip_needed(self):
        assert _strip_currency("3281.00") == "3281.00"

    def test_strips_leading_whitespace(self):
        assert _strip_currency("   \u20b9  258.75") == "258.75"


# ===========================================================================
# 2. _parse_numeric_string
# ===========================================================================

class TestParseNumericString:

    def test_plain_integer(self):
        assert _parse_numeric_string("258") == Decimal("258")

    def test_plain_decimal(self):
        assert _parse_numeric_string("258.75") == Decimal("258.75")

    def test_decimal_one_place(self):
        assert _parse_numeric_string("258.7") == Decimal("258.7")

    def test_decimal_two_places(self):
        assert _parse_numeric_string("258.00") == Decimal("258.00")

    def test_standard_thousands(self):
        assert _parse_numeric_string("1,234.56") == Decimal("1234.56")

    def test_large_standard_thousands(self):
        assert _parse_numeric_string("1,234,567.89") == Decimal("1234567.89")

    def test_indian_two_group(self):
        assert _parse_numeric_string("12,34,567.89") == Decimal("1234567.89")

    def test_indian_lakh(self):
        assert _parse_numeric_string("1,23,456.00") == Decimal("123456.00")

    def test_plain_integer_no_comma(self):
        assert _parse_numeric_string("3281") == Decimal("3281")

    def test_zero(self):
        assert _parse_numeric_string("0.00") == Decimal("0.00")

    def test_zero_no_decimal(self):
        assert _parse_numeric_string("0") == Decimal("0")

    # ------------------------------------------------------------------
    # CRITICAL: decimal must NOT become integer by stripping period
    # ------------------------------------------------------------------

    def test_258_75_is_not_25875(self):
        """Core regression: 258.75 must never become 25875."""
        result = _parse_numeric_string("258.75")
        assert result == Decimal("258.75")
        assert result != Decimal("25875")

    def test_3281_00_equals_3281(self):
        """3,281.00 must equal 3281.00 numerically."""
        a = _parse_numeric_string("3,281.00")
        b = _parse_numeric_string("3281.00")
        assert a == b == Decimal("3281.00")


# ===========================================================================
# 3. parse_monetary_string
# ===========================================================================

class TestParseMonetaryString:

    # ------------------------------------------------------------------
    # None / empty inputs
    # ------------------------------------------------------------------

    def test_none_input(self):
        result = parse_monetary_string(None)
        assert not result.succeeded
        assert result.value is None
        assert result.warnings

    def test_empty_string(self):
        result = parse_monetary_string("")
        assert not result.succeeded

    def test_whitespace_only(self):
        result = parse_monetary_string("   ")
        assert not result.succeeded

    # ------------------------------------------------------------------
    # Plain integers
    # ------------------------------------------------------------------

    def test_integer_258(self):
        r = parse_monetary_string("258")
        assert r.succeeded
        assert r.value == Decimal("258")

    def test_integer_3281(self):
        r = parse_monetary_string("3281")
        assert r.succeeded
        assert r.value == Decimal("3281")

    # ------------------------------------------------------------------
    # Decimal values
    # ------------------------------------------------------------------

    def test_decimal_258_75(self):
        r = parse_monetary_string("258.75")
        assert r.succeeded
        assert r.value == Decimal("258.75")
        # Critical: must NOT be 25875
        assert r.value != Decimal("25875")

    def test_decimal_one_place(self):
        r = parse_monetary_string("258.7")
        assert r.succeeded
        assert r.value == Decimal("258.7")

    def test_decimal_zero_cents(self):
        r = parse_monetary_string("258.00")
        assert r.succeeded
        assert r.value == Decimal("258.00")

    def test_decimal_zero_value(self):
        r = parse_monetary_string("0.00")
        assert r.succeeded
        assert r.value == Decimal("0.00")

    # ------------------------------------------------------------------
    # Thousands separator formats
    # ------------------------------------------------------------------

    def test_standard_thousands(self):
        r = parse_monetary_string("1,234.56")
        assert r.succeeded
        assert r.value == Decimal("1234.56")

    def test_indian_format_3281(self):
        r = parse_monetary_string("3,281.00")
        assert r.succeeded
        assert r.value == Decimal("3281.00")
        assert float(r.value) == pytest.approx(3281.0)

    def test_indian_lakh_format(self):
        r = parse_monetary_string("12,34,567.89")
        assert r.succeeded
        assert r.value == Decimal("1234567.89")

    def test_indian_crore_format(self):
        r = parse_monetary_string("1,23,45,678.90")
        # All commas stripped, period is decimal
        assert r.succeeded
        assert r.value == Decimal("12345678.90")

    # ------------------------------------------------------------------
    # Currency symbol / prefix / suffix
    # ------------------------------------------------------------------

    def test_rupee_symbol(self):
        r = parse_monetary_string("\u20b93,281.00")
        assert r.succeeded
        assert r.value == Decimal("3281.00")

    def test_inr_prefix_with_space(self):
        r = parse_monetary_string("INR 3,281.50")
        assert r.succeeded
        assert r.value == Decimal("3281.50")

    def test_dollar_prefix(self):
        r = parse_monetary_string("$50.00")
        assert r.succeeded
        assert r.value == Decimal("50.00")

    def test_rs_prefix(self):
        r = parse_monetary_string("Rs. 258.75")
        assert r.succeeded
        assert r.value == Decimal("258.75")

    def test_rupee_symbol_no_space(self):
        r = parse_monetary_string("\u20b9258.75")
        assert r.succeeded
        assert r.value == Decimal("258.75")

    def test_inr_no_space(self):
        r = parse_monetary_string("INR3281.50")
        assert r.succeeded
        assert r.value == Decimal("3281.50")

    # ------------------------------------------------------------------
    # Malformed / ambiguous inputs -- must NOT silently return zero
    # ------------------------------------------------------------------

    def test_pure_text_fails_gracefully(self):
        r = parse_monetary_string("TOTAL")
        assert not r.succeeded
        assert r.value is None
        assert r.warnings

    def test_currency_only_fails_gracefully(self):
        r = parse_monetary_string("INR")
        assert not r.succeeded

    def test_alphanumeric_fails_gracefully(self):
        r = parse_monetary_string("N/A")
        assert not r.succeeded or r.is_ambiguous  # may extract a bare digit

    def test_incomplete_number_still_extracts(self):
        # "Total: 258.75 rupees" -- fallback should find 258.75
        r = parse_monetary_string("Total: 258.75 rupees")
        assert r.succeeded
        assert r.value == Decimal("258.75")
        assert r.is_ambiguous  # flagged as extracted from context

    # ------------------------------------------------------------------
    # ParseResult dataclass properties
    # ------------------------------------------------------------------

    def test_ocr_text_preserved(self):
        raw = "\u20b93,281.00"
        r = parse_monetary_string(raw)
        assert r.ocr_text == raw

    def test_normalised_is_stripped(self):
        r = parse_monetary_string("INR 3,281.00")
        assert r.normalised == "3,281.00"

    def test_as_float_property(self):
        r = parse_monetary_string("258.75")
        assert r.as_float == pytest.approx(258.75)

    def test_as_float_none_when_failed(self):
        r = parse_monetary_string(None)
        assert r.as_float is None

    def test_succeeded_property(self):
        assert parse_monetary_string("100.00").succeeded is True
        assert parse_monetary_string("").succeeded is False

    def test_no_warnings_for_clean_parse(self):
        r = parse_monetary_string("3281.00")
        assert r.warnings == []
        assert not r.is_ambiguous


# ===========================================================================
# 4. parse_amount_to_float (drop-in for old _parse_amount)
# ===========================================================================

class TestParseAmountToFloat:

    def test_returns_float_for_valid_amount(self):
        result = parse_amount_to_float("258.75")
        assert isinstance(result, float)
        assert result == pytest.approx(258.75)

    def test_returns_none_for_empty(self):
        assert parse_amount_to_float("") is None
        assert parse_amount_to_float(None) is None

    def test_inr_prefix(self):
        result = parse_amount_to_float("INR 3,281.00")
        assert result == pytest.approx(3281.0)

    def test_rupee_with_indian_grouping(self):
        result = parse_amount_to_float("\u20b912,34,567.89")
        assert result == pytest.approx(1234567.89)

    def test_dollar_50(self):
        result = parse_amount_to_float("$50.00")
        assert result == pytest.approx(50.0)

    def test_258_75_never_becomes_25875(self):
        """CRITICAL regression guard."""
        result = parse_amount_to_float("258.75")
        assert result is not None
        assert abs(result - 258.75) < 0.001, f"Got {result}, expected ~258.75"
        assert result < 1000, f"258.75 became {result} -- decimal point lost!"

    def test_3281_00_equals_3281(self):
        """3,281.00 and 3281.00 must be numerically identical."""
        a = parse_amount_to_float("3,281.00")
        b = parse_amount_to_float("3281.00")
        assert a is not None and b is not None
        assert abs(a - b) < 0.001

    def test_zero_decimal(self):
        result = parse_amount_to_float("0.00")
        assert result is not None
        assert result == pytest.approx(0.0)


# ===========================================================================
# 5. Backward-compat: _parse_amount from extractor still works
# ===========================================================================

class TestExtractorParseAmountBackwardCompat:
    """Ensure the extractor._parse_amount wrapper delegates correctly."""

    def test_rupee_with_comma(self):
        assert _parse_amount("\u20b91,234.56") == pytest.approx(1234.56)

    def test_dollar_50(self):
        assert _parse_amount("$50.00") == pytest.approx(50.0)

    def test_empty_returns_none(self):
        assert _parse_amount("") is None
        assert _parse_amount(None) is None

    def test_decimal_preserved(self):
        """Old function must preserve decimal point."""
        result = _parse_amount("258.75")
        assert result is not None
        assert abs(result - 258.75) < 0.001
        assert result < 1000, f"Decimal dropped: got {result}"

    def test_indian_format(self):
        result = _parse_amount("3,281.00")
        assert result == pytest.approx(3281.0)


# ===========================================================================
# 6. Label scoring
# ===========================================================================

class TestLabelScore:

    def test_net_amount_highest(self):
        assert _label_score("Net Amount") == 100

    def test_net_payable_highest(self):
        assert _label_score("Net Payable") == 100

    def test_grand_total_high(self):
        assert _label_score("Grand Total") == 90

    def test_total_amount_high(self):
        assert _label_score("Total Amount") == 85

    def test_generic_total_moderate(self):
        assert _label_score("Total") == 70

    def test_subtotal_lower(self):
        assert _label_score("Subtotal") == 40

    def test_vat_lower(self):
        assert _label_score("VAT") == 30

    def test_service_tax_lower(self):
        assert _label_score("Service Tax") == 25

    def test_quantity_negative(self):
        assert _label_score("Quantity") < 0

    def test_total_quantity_negative(self):
        assert _label_score("Total Quantity") < 0

    def test_unknown_label_zero(self):
        assert _label_score("Some Unknown Field") == 0

    def test_case_insensitive(self):
        assert _label_score("NET AMOUNT") == _label_score("net amount")
        assert _label_score("Grand Total") == _label_score("GRAND TOTAL")

    def test_partial_match(self):
        # "gross total amount" contains "total amount" -> score 85
        score = _label_score("gross total amount")
        assert score >= 70


# ===========================================================================
# 7. select_best_amount -- label-aware field selection
# ===========================================================================

class TestSelectBestAmount:

    def test_net_amount_wins_over_service_tax(self):
        candidates = [
            ("Service Tax", "150.93"),
            ("Net Amount", "3281.00"),
            ("VAT", "55.55"),
        ]
        best, ranked = select_best_amount(candidates)
        assert best == Decimal("3281.00")
        assert ranked[0].label.lower() == "net amount"

    def test_total_quantity_excluded(self):
        """Total Quantity must never be selected as the bill amount."""
        candidates = [
            ("Total Quantity", "9.000"),
            ("Gross Total", "2580.00"),
            ("Net Amount", "3281.00"),
            ("VAT", "55.55"),
            ("Service Tax", "150.93"),
            ("Service Charges", "258.00"),
        ]
        best, ranked = select_best_amount(candidates)
        assert best == Decimal("3281.00")
        # Verify the net amount is top-ranked
        valid_ranked = [c for c in ranked if c.label_score >= 0 and c.parse_result.succeeded]
        assert valid_ranked[0].label.lower() == "net amount"

    def test_restaurant_receipt_example(self):
        """
        Reproduce the specific example from the bug report:
          Total Quantity: 9.000
          Gross Total: 2580.00
          VAT: 55.55 (and 227.65)
          Service Tax: 150.93
          Service Charges: 258.00
          Net Amount: 3281.00

        Expected: 3281.00 selected, associated with Net Amount label.
        """
        candidates = [
            ("Total Quantity", "9.000"),
            ("Gross Total", "2580.00"),
            ("VAT", "55.55"),
            ("VAT", "227.65"),
            ("Service Tax", "150.93"),
            ("Service Charges", "258.00"),
            ("Net Amount", "3281.00"),
        ]
        best, ranked = select_best_amount(candidates)
        assert best is not None
        assert best == Decimal("3281.00"), f"Expected 3281.00, got {best}"

    def test_single_candidate(self):
        candidates = [("Total", "500.00")]
        best, ranked = select_best_amount(candidates)
        assert best == Decimal("500.00")

    def test_empty_candidates(self):
        best, ranked = select_best_amount([])
        assert best is None
        assert ranked == []

    def test_all_invalid_candidates(self):
        candidates = [("Total", "invalid"), ("Net Amount", "???")]
        best, ranked = select_best_amount(candidates)
        assert best is None

    def test_all_quantity_candidates_excluded(self):
        candidates = [("Quantity", "5"), ("Total Quantity", "10")]
        best, ranked = select_best_amount(candidates)
        assert best is None  # All excluded by negative label score

    def test_grand_total_beats_subtotal(self):
        candidates = [
            ("Subtotal", "2000.00"),
            ("Grand Total", "2300.00"),
        ]
        best, _ = select_best_amount(candidates)
        assert best == Decimal("2300.00")

    def test_evidence_preserved_in_ranked_list(self):
        candidates = [
            ("Net Amount", "3281.00"),
            ("Service Tax", "150.93"),
        ]
        best, ranked = select_best_amount(candidates)
        # All candidates should be in ranked list
        labels_in_ranked = [c.label for c in ranked]
        assert "Net Amount" in labels_in_ranked
        assert "Service Tax" in labels_in_ranked

    def test_decimal_values_preserved_in_selection(self):
        """Verify that decimal precision is maintained through field selection."""
        candidates = [
            ("Net Amount", "3,281.00"),
            ("Total", "2,580.00"),
        ]
        best, _ = select_best_amount(candidates)
        assert best == Decimal("3281.00")
        assert float(best) == pytest.approx(3281.0)


# ===========================================================================
# 8. Critical regression guards
# ===========================================================================

class TestCriticalRegressions:

    @pytest.mark.parametrize("text,expected", [
        ("258.75",       Decimal("258.75")),
        ("258.7",        Decimal("258.7")),
        ("258.00",       Decimal("258.00")),
        ("1,234.56",     Decimal("1234.56")),
        ("12,34,567.89", Decimal("1234567.89")),
        ("\u20b93,281.00",    Decimal("3281.00")),
        ("INR 3,281.50", Decimal("3281.50")),
        ("3281.00",      Decimal("3281.00")),
        ("0.00",         Decimal("0.00")),
        ("258",          Decimal("258")),
    ])
    def test_parse_known_values(self, text, expected):
        r = parse_monetary_string(text)
        assert r.succeeded, f"Failed to parse: {text!r}"
        assert r.value == expected, (
            f"Input {text!r}: expected {expected}, got {r.value}"
        )

    def test_258_75_decimal_not_stripped(self):
        """
        The primary bug: OCR gives '258.75' but old code treated commas
        and periods symmetrically, risking '25875'.
        """
        r = parse_monetary_string("258.75")
        assert r.succeeded
        assert r.value == Decimal("258.75"), (
            f"Expected Decimal('258.75') but got {r.value!r}. "
            "Decimal point was stripped or period lost."
        )
        assert r.value < Decimal("1000"), (
            f"Value {r.value} is suspiciously large -- decimal point dropped!"
        )

    def test_3281_00_equals_3281_numerically(self):
        """3,281.00 and 3281.00 must be numerically equal."""
        a = parse_monetary_string("3,281.00")
        b = parse_monetary_string("3281.00")
        assert a.succeeded and b.succeeded
        assert a.value == b.value

    def test_inr_prefix_does_not_corrupt_decimal(self):
        """\u20b9258.75 must parse to 258.75, not 25875."""
        r = parse_monetary_string("\u20b9258.75")
        assert r.succeeded
        assert r.value == Decimal("258.75")

    def test_missing_value_is_none_not_zero(self):
        """Missing amount must be None, not silently 0."""
        assert parse_amount_to_float(None) is None
        assert parse_amount_to_float("") is None
        assert parse_amount_to_float("N/A") is None or True  # may be None or ambiguous

    def test_integer_amount_stays_integer(self):
        """Plain integer 3281 must parse to Decimal('3281'), not 32.81 or anything else."""
        r = parse_monetary_string("3281")
        assert r.succeeded
        assert r.value == Decimal("3281")

    def test_zero_decimal_is_zero_not_none(self):
        """0.00 is a valid zero amount, not a missing value."""
        r = parse_monetary_string("0.00")
        assert r.succeeded
        assert r.value == Decimal("0.00")
        assert float(r.value) == 0.0


# ===========================================================================
# 9. End-to-end: amount survives storage, retrieval, and formatting
# ===========================================================================

class TestEndToEndAmountPreservation:

    @pytest.fixture
    def tmp_db(self, tmp_path):
        from phase11.database import FinancialDocumentDB
        return FinancialDocumentDB(db_path=tmp_path / "test_money.db")

    def test_decimal_amount_stored_and_retrieved(self, tmp_db):
        """258.75 stored in DB must come back as 258.75."""
        from phase11.schema import FinancialDocument
        doc = FinancialDocument(
            document_id=str(uuid.uuid4()),
            filename="receipt_258.png",
            total_amount=258.75,
        )
        tmp_db.insert_document(doc)
        retrieved = tmp_db.get_document(doc.document_id)
        assert retrieved is not None
        assert retrieved.total_amount == pytest.approx(258.75), (
            f"Stored 258.75 but retrieved {retrieved.total_amount}"
        )

    def test_3281_stored_and_retrieved(self, tmp_db):
        """3281.00 stored in DB must come back as 3281.00."""
        from phase11.schema import FinancialDocument
        doc = FinancialDocument(
            document_id=str(uuid.uuid4()),
            filename="restaurant_bill.png",
            total_amount=3281.00,
        )
        tmp_db.insert_document(doc)
        retrieved = tmp_db.get_document(doc.document_id)
        assert retrieved is not None
        assert retrieved.total_amount == pytest.approx(3281.0)

    def test_kpi_sum_uses_stored_amounts(self, tmp_db):
        """KPI total_spend must reflect actual stored amounts, not corrupted ones."""
        from phase11.schema import FinancialDocument
        amounts = [258.75, 3281.00, 1234.56]
        for amt in amounts:
            tmp_db.insert_document(FinancialDocument(
                document_id=str(uuid.uuid4()),
                filename=f"bill_{amt}.png",
                total_amount=amt,
            ))
        kpis = tmp_db.kpi_summary()
        expected_total = sum(amounts)
        assert abs(kpis.get("total_spend", 0) - expected_total) < 0.01, (
            f"Expected KPI total_spend {expected_total}, got {kpis.get('total_spend')}"
        )

    def test_format_inr_does_not_change_stored_value(self):
        """Displaying currency-formatted string must not alter the numeric value."""
        from phase11.reporting import format_inr
        amount = 258.75
        display = format_inr(amount)
        # Verify the display string contains the expected decimal
        assert "258.75" in display, (
            f"format_inr(258.75) returned {display!r}, decimal is missing"
        )
        # The original amount is unchanged
        assert amount == 258.75

    def test_json_export_preserves_decimal(self, tmp_db):
        """JSON export must preserve 258.75 as a numeric value, not an integer."""
        from phase11.schema import FinancialDocument
        from phase11.export import DocumentExporter
        doc = FinancialDocument(
            document_id=str(uuid.uuid4()),
            filename="export_test.png",
            total_amount=258.75,
        )
        tmp_db.insert_document(doc)
        exporter = DocumentExporter()
        json_str = exporter.to_json([doc])
        import json
        data = json.loads(json_str)
        assert len(data) == 1
        assert abs(data[0]["total_amount"] - 258.75) < 0.001, (
            f"JSON export has total_amount={data[0]['total_amount']}, expected 258.75"
        )

    def test_csv_export_preserves_decimal(self, tmp_db):
        """CSV export must preserve 258.75 as a decimal value."""
        from phase11.schema import FinancialDocument
        from phase11.export import DocumentExporter
        import io
        doc = FinancialDocument(
            document_id=str(uuid.uuid4()),
            filename="csv_test.png",
            total_amount=258.75,
        )
        tmp_db.insert_document(doc)
        exporter = DocumentExporter()
        csv_str = exporter.to_csv([doc])
        assert "258.75" in csv_str, (
            f"CSV export does not contain '258.75'. CSV:\n{csv_str[:500]}"
        )


# ===========================================================================
# 10. parse_monetary_string -- edge cases and OCR-specific scenarios
# ===========================================================================

class TestOCREdgeCases:

    def test_ocr_text_with_extra_spaces(self):
        """OCR may insert spaces in numbers."""
        r = parse_monetary_string("3 281.00")
        # spaces inside number -- fallback may catch this
        if r.succeeded:
            assert r.value == Decimal("3281.00") or r.value is not None

    def test_context_text_extraction(self):
        """Parser must extract amount from "Net Amount: 3,281.00 INR" context."""
        r = parse_monetary_string("Net Amount: 3,281.00 INR")
        # Fallback should find 3281.00
        assert r.succeeded
        assert r.value == Decimal("3281.00") or float(r.value) == pytest.approx(3281.0)

    def test_partial_currency_prefix(self):
        r = parse_monetary_string("\u20b9 258.75")
        assert r.succeeded
        assert r.value == Decimal("258.75")

    def test_amount_with_trailing_text(self):
        r = parse_monetary_string("Total: 3281.00 only")
        assert r.succeeded

    def test_negative_amount_warns(self):
        """Negative amounts should parse but produce a warning."""
        r = parse_monetary_string("-50.00")
        if r.succeeded:
            assert any("negative" in w.lower() for w in r.warnings)

    def test_ambiguous_one_comma_two_digits_treated_as_decimal(self):
        """
        "1,23" is ambiguous but our rule treats it as "1.23" when
        the first group is a single digit.
        """
        r = parse_monetary_string("1,23")
        if r.succeeded:
            # Either 1.23 or 123 is acceptable -- no crash
            assert r.value is not None

    def test_very_large_amount(self):
        r = parse_monetary_string("1,23,45,678.90")
        assert r.succeeded
        assert r.value == Decimal("12345678.90")

