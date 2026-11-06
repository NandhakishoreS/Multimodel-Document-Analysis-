"""
Tests for Phase 6 unified evaluation module (evaluation/phase6_unified_eval.py).
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
    "phase6_unified_eval",
    _PROJECT_ROOT / "evaluation" / "phase6_unified_eval.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["phase6_unified_eval"] = _mod
sys.modules["evaluation.phase6_unified_eval"] = _mod
_spec.loader.exec_module(_mod)

extract_iob2_spans = _mod.extract_iob2_spans
compute_strict_span_f1 = _mod.compute_strict_span_f1
build_incompatibility_matrix = _mod.build_incompatibility_matrix


def test_extract_iob2_spans_simple():
    tags = ["O", "B-COMPANY", "I-COMPANY", "O", "B-DATE", "O"]
    spans = extract_iob2_spans(tags)
    assert spans == [("COMPANY", 1, 3), ("DATE", 4, 5)]


def test_extract_iob2_spans_consecutive_same_entity():
    tags = ["B-COMPANY", "I-COMPANY", "B-COMPANY", "I-COMPANY"]
    spans = extract_iob2_spans(tags)
    assert spans == [("COMPANY", 0, 2), ("COMPANY", 2, 4)]


def test_extract_iob2_spans_all_O():
    tags = ["O", "O", "O"]
    spans = extract_iob2_spans(tags)
    assert spans == []


def test_compute_strict_span_f1_perfect():
    gold = [["B-COMPANY", "I-COMPANY", "O", "B-TOTAL"]]
    pred = [["B-COMPANY", "I-COMPANY", "O", "B-TOTAL"]]
    result = compute_strict_span_f1(gold, pred)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.macro_f1 == 1.0
    assert result.micro_f1 == 1.0
    assert result.n_gold_spans == 2
    assert result.n_pred_spans == 2


def test_compute_strict_span_f1_boundary_mismatch():
    # Gold: COMPANY tokens 0..2. Pred: COMPANY tokens 0..1 (missing I-COMPANY)
    gold = [["B-COMPANY", "I-COMPANY", "O"]]
    pred = [["B-COMPANY", "O", "O"]]
    result = compute_strict_span_f1(gold, pred)
    # Strict boundary match: span (COMPANY, 0, 2) != (COMPANY, 0, 1)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.macro_f1 == 0.0
    assert result.micro_f1 == 0.0


def test_compute_strict_span_f1_partial_entities():
    # 2 entities in gold: COMPANY and TOTAL. Model gets COMPANY right, misses TOTAL.
    gold = [["B-COMPANY", "O", "B-TOTAL"]]
    pred = [["B-COMPANY", "O", "O"]]
    result = compute_strict_span_f1(gold, pred)
    assert result.precision == 1.0
    assert result.recall == 0.5
    assert result.micro_f1 == pytest.approx(2/3, rel=1e-3)
    # COMPANY F1 = 1.0, TOTAL F1 = 0.0 -> Macro F1 = 0.5
    assert result.macro_f1 == pytest.approx(0.5, rel=1e-3)


def test_incompatibility_matrix():
    matrix = build_incompatibility_matrix()
    assert "SROIE" in matrix
    assert "FUNSD" in matrix
    assert "CORD" in matrix
    for ds_name, info in matrix.items():
        assert info["baseline_vs_layoutlmv3_comparable"] is False
        assert len(info["reasons"]) > 0
