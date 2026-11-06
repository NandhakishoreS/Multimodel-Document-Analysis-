"""
Tests for Phase 7 error analysis and taxonomy module (evaluation/phase7_error_analysis.py).
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_site_packages = _PROJECT_ROOT / ".venv312" / "Lib" / "site-packages"
if _site_packages.exists() and str(_site_packages) not in sys.path:
    sys.path.insert(0, str(_site_packages))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "phase7_error_analysis",
    _PROJECT_ROOT / "evaluation" / "phase7_error_analysis.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["phase7_error_analysis"] = _mod
sys.modules["evaluation.phase7_error_analysis"] = _mod
_spec.loader.exec_module(_mod)

ErrorCategory = _mod.ErrorCategory
EntitySpan = _mod.EntitySpan
extract_iob2_entity_spans = _mod.extract_iob2_entity_spans
classify_document_errors = _mod.classify_document_errors
compute_per_entity_error_summary = _mod.compute_per_entity_error_summary


def test_exact_entity_match():
    words = ["STORE", "ABC", "TOTAL", "10.00"]
    bboxes = [[0, 0, 10, 10]] * 4
    gold_bio = ["B-COMPANY", "I-COMPANY", "O", "B-TOTAL"]
    pred_bio = ["B-COMPANY", "I-COMPANY", "O", "B-TOTAL"]

    rec = classify_document_errors("doc1", "sroie", words, bboxes, gold_bio, pred_bio)
    assert rec.tp_count == 2
    assert rec.fp_count == 0
    assert rec.fn_count == 0
    assert rec.total_errors == 0
    assert all(pair.category == ErrorCategory.TRUE_POSITIVE for pair in rec.pairs)


def test_false_negative_detection():
    words = ["STORE", "ABC", "TOTAL", "10.00"]
    bboxes = [[0, 0, 10, 10]] * 4
    gold_bio = ["B-COMPANY", "I-COMPANY", "O", "B-TOTAL"]
    pred_bio = ["O", "O", "O", "B-TOTAL"]  # COMPANY missed

    rec = classify_document_errors("doc2", "sroie", words, bboxes, gold_bio, pred_bio)
    assert rec.tp_count == 1
    assert rec.fn_count >= 1
    missed_pairs = [p for p in rec.pairs if p.category == ErrorCategory.FALSE_NEGATIVE]
    assert len(missed_pairs) == 1
    assert missed_pairs[0].gold.entity_type == "COMPANY"


def test_false_positive_detection():
    words = ["STORE", "ABC", "TOTAL", "10.00"]
    bboxes = [[0, 0, 10, 10]] * 4
    gold_bio = ["O", "O", "O", "B-TOTAL"]
    pred_bio = ["B-COMPANY", "I-COMPANY", "O", "B-TOTAL"]  # COMPANY spurious

    rec = classify_document_errors("doc3", "sroie", words, bboxes, gold_bio, pred_bio)
    assert rec.tp_count == 1
    assert rec.fp_count >= 1
    spurious_pairs = [p for p in rec.pairs if p.category == ErrorCategory.FALSE_POSITIVE]
    assert len(spurious_pairs) == 1
    assert spurious_pairs[0].pred.entity_type == "COMPANY"


def test_boundary_error_detection():
    # Gold: "STORE ABC" [0:2], Pred: "STORE" [0:1]
    words = ["STORE", "ABC", "O"]
    bboxes = [[0, 0, 10, 10]] * 3
    gold_bio = ["B-COMPANY", "I-COMPANY", "O"]
    pred_bio = ["B-COMPANY", "O", "O"]

    rec = classify_document_errors("doc4", "sroie", words, bboxes, gold_bio, pred_bio)
    assert rec.tp_count == 0
    boundary_pairs = [p for p in rec.pairs if p.category == ErrorCategory.BOUNDARY_ERROR]
    assert len(boundary_pairs) == 1
    assert boundary_pairs[0].gold.entity_type == "COMPANY"
    assert boundary_pairs[0].pred.entity_type == "COMPANY"
    assert boundary_pairs[0].gold.end == 2
    assert boundary_pairs[0].pred.end == 1


def test_type_error_detection():
    # Overlapping span, but wrong entity type (e.g. Question vs Answer or Total vs Date)
    words = ["TOTAL", "10.00"]
    bboxes = [[0, 0, 10, 10]] * 2
    gold_bio = ["O", "B-TOTAL"]
    pred_bio = ["O", "B-DATE"]

    rec = classify_document_errors("doc5", "sroie", words, bboxes, gold_bio, pred_bio)
    assert rec.tp_count == 0
    type_pairs = [p for p in rec.pairs if p.category == ErrorCategory.TYPE_ERROR]
    assert len(type_pairs) == 1
    assert type_pairs[0].gold.entity_type == "TOTAL"
    assert type_pairs[0].pred.entity_type == "DATE"


def test_accounting_consistency():
    words = ["A", "B", "C", "D", "E"]
    bboxes = [[0, 0, 10, 10]] * 5
    gold_bio = ["B-COMPANY", "I-COMPANY", "O", "B-TOTAL", "O"]  # 2 gold spans
    pred_bio = ["B-COMPANY", "O", "B-DATE", "B-TOTAL", "O"]      # 3 pred spans (1 boundary error, 1 spurious, 1 TP)

    rec = classify_document_errors("doc6", "sroie", words, bboxes, gold_bio, pred_bio)
    assert len(rec.gold_spans) == 2
    assert len(rec.pred_spans) == 3
    assert rec.tp_count == 1  # TOTAL is exact match
    assert rec.boundary_errors == 1  # COMPANY is boundary error
    assert rec.fp_count == 2  # 1 boundary pred + 1 spurious pred
    assert rec.fn_count == 1  # 1 boundary gold


def test_per_entity_summary():
    words = ["STORE", "ABC", "TOTAL", "10.00"]
    bboxes = [[0, 0, 10, 10]] * 4
    gold_bio = ["B-COMPANY", "I-COMPANY", "O", "B-TOTAL"]
    pred_bio = ["B-COMPANY", "O", "O", "B-TOTAL"]

    rec = classify_document_errors("doc7", "sroie", words, bboxes, gold_bio, pred_bio)
    summary = compute_per_entity_error_summary([rec], ["COMPANY", "TOTAL"])

    comp_row = next(r for r in summary if r["entity"] == "COMPANY")
    total_row = next(r for r in summary if r["entity"] == "TOTAL")

    assert comp_row["gold"] == 1
    assert comp_row["tp"] == 0
    assert comp_row["boundary"] == 1

    assert total_row["gold"] == 1
    assert total_row["tp"] == 1
    assert total_row["f1"] == 1.0
