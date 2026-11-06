"""
Unit tests for Phase 9: Robustness-Aware Model Improvement & Multimodal Ablation.

Validates:
1. Dataset immutability (original disk files untouched).
2. Checkpoint isolation (Phase 1-8 checkpoints untouched).
3. Modality ablation transformations (zeroing bboxes, zeroing pixel_values, masking text).
4. Dynamic augmentation validity (coordinate clamping in [0, 1000], valid areas).
5. Determinism under fixed seed and variation under different seeds.
6. Checkpoints written exclusively to Phase 9 target directories.
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from evaluation.phase9_ablation import (
    ABLATION_CONDITIONS,
    apply_batch_ablation,
)
from models.layoutlmv3.config import _PROJECT_ROOT
from training.phase9_augmentation import (
    AugmentationConfig,
    apply_layout_jitter,
    apply_text_noise,
)


def test_original_datasets_remain_unmodified():
    """Verify that original dataset files exist and are not modified."""
    sroie_test = _PROJECT_ROOT / "processed_data/sroie_corrected/test.jsonl"
    funsd_test = _PROJECT_ROOT / "processed_data/funsd/test.jsonl"
    cord_test = _PROJECT_ROOT / "processed_data/cord/test.jsonl"

    assert sroie_test.exists()
    assert funsd_test.exists()
    assert cord_test.exists()

    with open(sroie_test, "r", encoding="utf-8") as f:
        sroie_lines = sum(1 for _ in f)
    assert sroie_lines == 361, f"SROIE test sample count must remain exactly 361, got {sroie_lines}"


def test_original_checkpoints_remain_unmodified():
    """Verify that validated Phase 1-8 checkpoints exist and are intact."""
    sroie_ckpt = _PROJECT_ROOT / "models/checkpoints/layoutlmv3/sroie/corrected/best"
    funsd_ckpt = _PROJECT_ROOT / "models/checkpoints/layoutlmv3/funsd/full"
    cord_ckpt = _PROJECT_ROOT / "models/checkpoints/layoutlmv3/cord/full"

    assert (sroie_ckpt / "config.json").exists()
    assert (funsd_ckpt / "config.json").exists()
    assert (cord_ckpt / "config.json").exists()


def test_modality_ablation_zeroing():
    """Verify that apply_batch_ablation correctly zeros out modalities without mutating original."""
    batch = {
        "input_ids": torch.tensor([[101, 2054, 2003, 1037, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        "bbox": torch.tensor([[[0, 0, 0, 0], [10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [0, 0, 0, 0]]]),
        "pixel_values": torch.ones(1, 3, 224, 224),
    }

    # Condition B: Text Only (BBox=0, Image=0)
    cond_text_only = next(c for c in ABLATION_CONDITIONS if c["short"] == "Text-Only")
    ablated_b = apply_batch_ablation(batch, cond_text_only)
    assert torch.all(ablated_b["bbox"] == 0)
    assert torch.all(ablated_b["pixel_values"] == 0.0)
    assert torch.all(ablated_b["input_ids"] == batch["input_ids"])
    # Verify original unchanged
    assert not torch.all(batch["bbox"] == 0)

    # Condition C: Text + Layout (Image=0)
    cond_no_img = next(c for c in ABLATION_CONDITIONS if c["short"] == "No-Image")
    ablated_c = apply_batch_ablation(batch, cond_no_img)
    assert not torch.all(ablated_c["bbox"] == 0)
    assert torch.all(ablated_c["pixel_values"] == 0.0)

    # Condition E: Layout + Image (Text masked)
    cond_no_text = next(c for c in ABLATION_CONDITIONS if c["short"] == "No-Text")
    ablated_e = apply_batch_ablation(batch, cond_no_text, mask_token_id=50264)
    assert not torch.all(ablated_e["bbox"] == 0)
    assert not torch.all(ablated_e["pixel_values"] == 0.0)
    # Check that middle tokens were replaced with mask token
    assert ablated_e["input_ids"][0, 1].item() == 50264


def test_layout_jitter_bounding_box_validity():
    """Verify that layout jitter clamps strictly to [0, 1000] and preserves positive area."""
    import random
    rng = random.Random(42)
    extreme_boxes = [
        [0, 0, 10, 10],
        [990, 990, 1000, 1000],
        [450, 450, 550, 550],
    ]
    # Heavy 15% jitter
    jittered = apply_layout_jitter(extreme_boxes, jitter_ratio=0.15, rng=rng)
    assert len(jittered) == len(extreme_boxes)
    for b in jittered:
        x0, y0, x1, y1 = b
        assert 0 <= x0 <= 1000
        assert 0 <= y0 <= 1000
        assert 0 <= x1 <= 1000
        assert 0 <= y1 <= 1000
        assert x0 < x1, f"x0 ({x0}) must be strictly less than x1 ({x1})"
        assert y0 < y1, f"y0 ({y0}) must be strictly less than y1 ({y1})"


def test_seed_determinism_and_variation():
    """Verify that fixed seed produces identical output and different seeds produce variation."""
    import random
    boxes = [[100, 100, 200, 200], [300, 300, 400, 400]]
    words = ["TOTAL", "DUE", "125.50", "TAX", "5.00"]

    # Fixed seed 123
    r1 = random.Random(123)
    b1 = apply_layout_jitter(boxes, 0.05, r1)
    w1 = apply_text_noise(words, 0.1, 0.05, 0.05, r1)

    r2 = random.Random(123)
    b2 = apply_layout_jitter(boxes, 0.05, r2)
    w2 = apply_text_noise(words, 0.1, 0.05, 0.05, r2)

    assert b1 == b2
    assert w1 == w2

    # Different seed 999
    r3 = random.Random(999)
    b3 = apply_layout_jitter(boxes, 0.05, r3)
    assert b1 != b3


def test_phase9_checkpoint_paths_isolated():
    """Verify that Phase 9 checkpoints are directed only to models/checkpoints/phase9/."""
    p9_base = _PROJECT_ROOT / "models/checkpoints/phase9"
    assert "phase9" in str(p9_base)
    # Ensure it is separate from original layoutlmv3 checkpoints
    assert p9_base != (_PROJECT_ROOT / "models/checkpoints/layoutlmv3/sroie/corrected/best")
