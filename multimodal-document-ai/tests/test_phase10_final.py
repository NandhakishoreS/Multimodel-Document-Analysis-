"""
Phase 10 Automated Integration & Unit Tests.

Verifies:
1. Pipeline imports and module structure.
2. Checkpoint availability and model loading.
3. Preprocessing and bounding box coordinate validation.
4. Schema conformance of DocumentInferenceResult.
5. BIO decoding and span aggregation.
6. Confidence computation (mean token probability and minimum probability).
7. Bounding-box visualization output.
8. JSON serialization and export.
9. Graceful handling of invalid inputs and edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from PIL import Image

from inference.schemas import DocumentInferenceResult, ExtractedEntity, TokenPrediction
from inference.postprocessing import (
    collapse_token_preds_to_words,
    reconstruct_entity_spans,
    union_bboxes,
)
from inference.final_inference import (
    DEFAULT_CHECKPOINTS,
    UnseenDocumentInferencePipeline,
)
from visualization.draw_predictions import (
    COLOR_PALETTE,
    draw_predictions_on_image,
    get_label_color,
)


# ---------------------------------------------------------------------------
# Test 1: Module Imports and Directory Integrity
# ---------------------------------------------------------------------------

def test_inference_pipeline_imports():
    """Verify all Phase 10 modules import cleanly without error."""
    import inference.final_inference as finf
    import inference.postprocessing as post
    import inference.schemas as sch
    import visualization.draw_predictions as vis

    assert hasattr(finf, "UnseenDocumentInferencePipeline")
    assert hasattr(post, "reconstruct_entity_spans")
    assert hasattr(sch, "DocumentInferenceResult")
    assert hasattr(vis, "draw_predictions_on_image")


# ---------------------------------------------------------------------------
# Test 2: Checkpoint Path Existence
# ---------------------------------------------------------------------------

def test_final_checkpoints_exist():
    """Ensure all primary selected checkpoints are present on disk."""
    root = Path(__file__).resolve().parent.parent

    # SROIE Joint-Augmented (Phase 9 primary)
    sroie_joint = root / "models/checkpoints/phase9/sroie/joint_aug/best"
    assert sroie_joint.exists(), f"Missing primary SROIE checkpoint: {sroie_joint}"

    # SROIE Baseline Corrected
    sroie_base = root / "models/checkpoints/layoutlmv3/sroie/corrected/best"
    assert sroie_base.exists(), f"Missing baseline SROIE checkpoint: {sroie_base}"

    # FUNSD Full
    funsd_full = root / "models/checkpoints/layoutlmv3/funsd/full"
    assert funsd_full.exists(), f"Missing FUNSD checkpoint: {funsd_full}"

    # CORD Full
    cord_full = root / "models/checkpoints/layoutlmv3/cord/full"
    assert cord_full.exists(), f"Missing CORD checkpoint: {cord_full}"


# ---------------------------------------------------------------------------
# Test 3: Union BBoxes Computation
# ---------------------------------------------------------------------------

def test_union_bboxes():
    """Test minimum enclosing bounding box computation."""
    boxes = [
        [10, 20, 30, 40],
        [15, 25, 50, 45],
        [5, 10, 20, 30],
    ]
    u = union_bboxes(boxes)
    assert u == [5, 10, 50, 45]

    # Empty list edge case
    assert union_bboxes([]) == [0, 0, 0, 0]


# ---------------------------------------------------------------------------
# Test 4: BIO Decoding and Entity Span Reconstruction
# ---------------------------------------------------------------------------

def test_reconstruct_entity_spans_logic():
    """Verify BIO sequence decoding into ExtractedEntity objects."""
    words = ["Starbucks", "Coffee", "Co.", "Date:", "12/04/2022", "Total:", "$4.50"]
    labels = ["B-COMPANY", "I-COMPANY", "I-COMPANY", "O", "B-DATE", "O", "B-TOTAL"]
    probs = [0.98, 0.96, 0.94, 0.99, 0.995, 0.95, 0.99]
    orig_boxes = [
        [50, 100, 150, 130],
        [155, 100, 230, 130],
        [235, 100, 280, 130],
        [50, 140, 100, 160],
        [110, 140, 210, 160],
        [50, 200, 100, 220],
        [120, 200, 180, 220],
    ]
    norm_boxes = [[b[0] * 2, b[1] * 2, b[2] * 2, b[3] * 2] for b in orig_boxes]

    entities, tokens, agg_dict = reconstruct_entity_spans(
        words=words,
        word_labels=labels,
        word_probs=probs,
        original_bboxes=orig_boxes,
        normalized_bboxes=norm_boxes,
    )

    assert len(entities) == 3

    # Check COMPANY span
    company = next(e for e in entities if e.label == "COMPANY")
    assert company.text == "Starbucks Coffee Co."
    assert company.token_count == 3
    # Arithmetic mean confidence: (0.98 + 0.96 + 0.94) / 3 = 0.9600
    assert abs(company.confidence - 0.9600) < 1e-3
    assert company.min_confidence == 0.94
    assert company.bbox == [50, 100, 280, 130]

    # Check DATE span
    date_ent = next(e for e in entities if e.label == "DATE")
    assert date_ent.text == "12/04/2022"
    assert date_ent.token_count == 1
    assert abs(date_ent.confidence - 0.995) < 1e-3

    # Check TOTAL span
    total_ent = next(e for e in entities if e.label == "TOTAL")
    assert total_ent.text == "$4.50"

    # Aggregated dict checks
    assert agg_dict["COMPANY"] == "Starbucks Coffee Co."
    assert agg_dict["DATE"] == "12/04/2022"
    assert agg_dict["TOTAL"] == "$4.50"


# ---------------------------------------------------------------------------
# Test 5: Broken BIO Handling (I without B)
# ---------------------------------------------------------------------------

def test_broken_bio_recovery():
    """Verify that an orphan I- tag starts a new span rather than crashing."""
    words = ["Main", "Street", "Suite", "100"]
    labels = ["I-ADDRESS", "I-ADDRESS", "I-ADDRESS", "I-ADDRESS"]
    probs = [0.90, 0.92, 0.91, 0.89]
    boxes = [[0, 0, 10, 10]] * 4

    entities, _, _ = reconstruct_entity_spans(
        words=words,
        word_labels=labels,
        word_probs=probs,
        original_bboxes=boxes,
        normalized_bboxes=boxes,
    )
    assert len(entities) == 1
    assert entities[0].label == "ADDRESS"
    assert entities[0].text == "Main Street Suite 100"


# ---------------------------------------------------------------------------
# Test 6: Subword to Word Pooling
# ---------------------------------------------------------------------------

def test_collapse_token_preds_to_words():
    """Verify first-subword pooling convention."""
    id2label = {0: "O", 1: "B-COMPANY", 2: "I-COMPANY", 3: "B-TOTAL"}
    # Token IDs: [CLS, B-COMPANY, I-COMPANY, O, B-TOTAL, I-TOTAL, SEP]
    token_label_ids = [0, 1, 2, 0, 3, 0, 0]
    token_probs = [0.99, 0.95, 0.40, 0.88, 0.99, 0.30, 0.99]
    word_ids = [None, 0, 0, 1, 2, 2, None]

    word_labels, word_probs = collapse_token_preds_to_words(
        token_label_ids=token_label_ids,
        token_probs=token_probs,
        word_ids=word_ids,
        id2label=id2label,
        num_words=3,
    )

    assert len(word_labels) == 3
    assert word_labels[0] == "B-COMPANY"
    assert word_probs[0] == 0.95  # First subword probability used
    assert word_labels[1] == "O"
    assert word_labels[2] == "B-TOTAL"
    assert word_probs[2] == 0.99


# ---------------------------------------------------------------------------
# Test 7: Output Schema & JSON Serialization
# ---------------------------------------------------------------------------

def test_document_inference_result_schema_and_json():
    """Verify serialization to valid JSON conforming to output requirements."""
    entity = ExtractedEntity(
        label="COMPANY",
        text="Acme Corp",
        confidence=0.9876,
        min_confidence=0.9700,
        bbox=[100, 50, 300, 90],
        normalized_bbox=[100, 50, 300, 90],
        start_word=0,
        end_word=1,
        token_count=2,
    )

    res = DocumentInferenceResult(
        document_type="SROIE",
        model_name="sroie_robust",
        image_path="sample.png",
        image_size=(800, 1000),
        ocr_word_count=20,
        entities=[entity],
        raw_entities_dict={"COMPANY": "Acme Corp"},
        ocr_raw_text="Acme Corp Receipt",
    )

    d = res.to_dict()
    assert d["document_type"] == "SROIE"
    assert d["entities"][0]["label"] == "COMPANY"
    assert d["entities"][0]["text"] == "Acme Corp"
    assert d["entities"][0]["confidence"] == 0.9876

    json_str = res.to_json()
    parsed = json.loads(json_str)
    assert parsed["document_type"] == "SROIE"
    assert parsed["aggregated_fields"]["COMPANY"] == "Acme Corp"


# ---------------------------------------------------------------------------
# Test 8: Visualization Rendering
# ---------------------------------------------------------------------------

def test_draw_predictions_on_image():
    """Verify visualization overlay draws onto PIL image without throwing."""
    img = Image.new("RGB", (400, 400), color="white")
    entity = ExtractedEntity(
        label="TOTAL",
        text="$25.00",
        confidence=0.99,
        min_confidence=0.99,
        bbox=[50, 100, 150, 140],
        normalized_bbox=[125, 250, 375, 350],
        start_word=0,
        end_word=0,
        token_count=1,
    )

    annotated = draw_predictions_on_image(img, [entity])
    assert isinstance(annotated, Image.Image)
    assert annotated.size == (400, 400)
    assert annotated.mode == "RGB"

    # Verify color palette has entries for core classes
    for c in ["COMPANY", "DATE", "ADDRESS", "TOTAL", "HEADER", "QUESTION", "ANSWER", "MENU", "SUBTOTAL"]:
        color = get_label_color(c)
        assert len(color) == 3
        assert all(0 <= val <= 255 for val in color)


# ---------------------------------------------------------------------------
# Test 9: Edge Case & Invalid Input Handling
# ---------------------------------------------------------------------------

def test_empty_image_ocr_handling():
    """Verify blank image produces empty result without crashing."""
    pipe = UnseenDocumentInferencePipeline(dataset_type="sroie", use_robust_model=False)
    blank_img = Image.new("RGB", (200, 200), color="white")
    res = pipe.predict_image(blank_img)

    assert res.ocr_word_count == 0
    assert len(res.entities) == 0
    assert res.raw_entities_dict == {}


def test_invalid_image_type_handling():
    """Verify passing invalid object raises TypeError."""
    pipe = UnseenDocumentInferencePipeline(dataset_type="sroie")
    with pytest.raises(TypeError):
        pipe.predict_image(12345)  # type: ignore


def test_nonexistent_image_path_handling():
    """Verify nonexistent image path raises FileNotFoundError."""
    pipe = UnseenDocumentInferencePipeline(dataset_type="sroie")
    with pytest.raises(FileNotFoundError):
        pipe.predict_image("nonexistent_image_12345.png")


def test_unsupported_dataset_type():
    """Verify passing unsupported dataset raises ValueError."""
    with pytest.raises(ValueError):
        UnseenDocumentInferencePipeline(dataset_type="unsupported_benchmark")
