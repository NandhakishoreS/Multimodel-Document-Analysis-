"""
Tests for Phase 8 robustness and perturbation module (evaluation/phase8_robustness.py).
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_site_packages = _PROJECT_ROOT / ".venv312" / "Lib" / "site-packages"
if _site_packages.exists() and str(_site_packages) not in sys.path:
    sys.path.insert(0, str(_site_packages))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "phase8_robustness",
    _PROJECT_ROOT / "evaluation" / "phase8_robustness.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["phase8_robustness"] = _mod
sys.modules["evaluation.phase8_robustness"] = _mod
_spec.loader.exec_module(_mod)

apply_ocr_char_corruption = _mod.apply_ocr_char_corruption
apply_missing_text_masking = _mod.apply_missing_text_masking
apply_layout_bbox_jitter = _mod.apply_layout_bbox_jitter
apply_image_degradation = _mod.apply_image_degradation


def test_original_words_remain_unmutated():
    original_words = ["TOTAL", "100.00", "STORE", "NAME"]
    words_copy = list(original_words)
    corrupted = apply_ocr_char_corruption(original_words, rate=0.8, seed=42)
    assert original_words == words_copy, "Original words list was mutated!"
    assert len(corrupted) == len(original_words)


def test_missing_text_masking():
    original_words = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    masked = apply_missing_text_masking(original_words, rate=0.5, seed=42, mask_token="***")
    assert original_words[0] == "A"
    assert "***" in masked
    assert len(masked) == len(original_words)


def test_bbox_validity_after_jitter():
    # Test coordinates across diverse box sizes, edge boundaries, and jitter severities
    bboxes = [
        [0, 0, 10, 10],         # Near top-left origin
        [990, 990, 1000, 1000], # Near bottom-right edge
        [100, 200, 300, 400],   # Center standard box
        [50, 50, 52, 52],       # Very small 2x2 box
    ]

    for jitter_rate in [0.02, 0.05, 0.10, 0.20]:
        jittered = apply_layout_bbox_jitter(bboxes, max_pct_jitter=jitter_rate, seed=123)
        assert len(jittered) == len(bboxes)
        for b in jittered:
            x0, y0, x1, y1 = b
            assert 0 <= x0 <= 1000, f"x0 out of bounds: {x0}"
            assert 0 <= y0 <= 1000, f"y0 out of bounds: {y0}"
            assert 0 <= x1 <= 1000, f"x1 out of bounds: {x1}"
            assert 0 <= y1 <= 1000, f"y1 out of bounds: {y1}"
            assert x0 < x1, f"x0 >= x1: {x0}, {x1}"
            assert y0 < y1, f"y0 >= y1: {y0}, {y1}"


def test_severity_rate_scaling():
    words = [f"word_{i}" for i in range(100)]
    corrupted_low = apply_ocr_char_corruption(words, rate=0.05, seed=42)
    corrupted_high = apply_ocr_char_corruption(words, rate=0.50, seed=42)

    diff_low = sum(1 for o, c in zip(words, corrupted_low) if o != c)
    diff_high = sum(1 for o, c in zip(words, corrupted_high) if o != c)

    assert diff_low < diff_high, f"High severity rate should alter more words: low={diff_low}, high={diff_high}"


def test_image_degradation():
    img = Image.new("RGB", (100, 100), color="white")
    blurred = apply_image_degradation(img, "blur", severity=2.0)
    assert blurred.size == (100, 100)
    assert blurred is not img  # Must be a new instance

    downscaled = apply_image_degradation(img, "resolution", severity=0.5)
    assert downscaled.size == (100, 100)
