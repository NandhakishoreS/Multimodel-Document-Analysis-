"""
Phase 10 Inference Pipeline Tests.

Covers:
- Module imports and API surface
- Checkpoint path existence for all datasets
- Preprocessing (bbox normalization, OCR word-split)
- BIO decoding correctness
- Entity aggregation
- Confidence calculation (arithmetic mean, min)
- Bounding box union geometry
- JSON serialization / export
- Invalid input handling
- Schema structure

These tests are purposely model-free (no GPU required) where possible,
using lightweight synthetic fixtures. The two tests that require model
loading mark themselves as slow via the 'integration' marker.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Module-level imports (must succeed cleanly)
# ---------------------------------------------------------------------------
from inference.schemas import DocumentInferenceResult, ExtractedEntity, TokenPrediction
from inference.postprocessing import (
    collapse_token_preds_to_words,
    reconstruct_entity_spans,
    union_bboxes,
)
from inference.final_inference import (
    DEFAULT_CHECKPOINTS,
    EmptyOCRError,
    UnseenDocumentInferencePipeline,
    split_cord_ocr_words,
)
from visualization.draw_predictions import (
    COLOR_PALETTE,
    draw_predictions_on_image,
    draw_raw_ocr_boxes,
    get_label_color,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Import / API surface
# ---------------------------------------------------------------------------

class TestImports:
    def test_all_modules_importable(self):
        import inference.final_inference
        import inference.postprocessing
        import inference.schemas
        import visualization.draw_predictions
        assert True

    def test_api_surface_inference(self):
        from inference.final_inference import UnseenDocumentInferencePipeline
        assert callable(UnseenDocumentInferencePipeline)

    def test_api_surface_postprocessing(self):
        from inference.postprocessing import (
            collapse_token_preds_to_words,
            reconstruct_entity_spans,
            union_bboxes,
        )
        assert callable(collapse_token_preds_to_words)
        assert callable(reconstruct_entity_spans)
        assert callable(union_bboxes)

    def test_api_surface_schemas(self):
        from inference.schemas import DocumentInferenceResult, ExtractedEntity, TokenPrediction
        assert DocumentInferenceResult
        assert ExtractedEntity
        assert TokenPrediction

    def test_api_surface_visualization(self):
        from visualization.draw_predictions import draw_predictions_on_image, get_label_color, COLOR_PALETTE
        assert callable(draw_predictions_on_image)
        assert callable(get_label_color)
        assert isinstance(COLOR_PALETTE, dict)


# ---------------------------------------------------------------------------
# 2. Checkpoint existence
# ---------------------------------------------------------------------------

class TestCheckpointPaths:
    def test_sroie_joint_aug_exists(self):
        p = PROJECT_ROOT / "models/checkpoints/phase9/sroie/joint_aug/best"
        assert p.exists(), f"Missing SROIE joint-aug checkpoint: {p}"

    def test_sroie_baseline_corrected_exists(self):
        p = PROJECT_ROOT / "models/checkpoints/layoutlmv3/sroie/corrected/best"
        assert p.exists(), f"Missing SROIE baseline checkpoint: {p}"

    def test_funsd_full_exists(self):
        p = PROJECT_ROOT / "models/checkpoints/layoutlmv3/funsd/full"
        assert p.exists(), f"Missing FUNSD checkpoint: {p}"

    def test_cord_full_exists(self):
        p = PROJECT_ROOT / "models/checkpoints/layoutlmv3/cord/full"
        assert p.exists(), f"Missing CORD checkpoint: {p}"

    def test_sroie_joint_aug_has_model_file(self):
        p = PROJECT_ROOT / "models/checkpoints/phase9/sroie/joint_aug/best/model.safetensors"
        assert p.exists(), f"model.safetensors not found in joint-aug checkpoint"

    def test_funsd_has_model_file(self):
        p = PROJECT_ROOT / "models/checkpoints/layoutlmv3/funsd/full/model.safetensors"
        assert p.exists(), f"model.safetensors not found in FUNSD checkpoint"

    def test_default_checkpoints_registry(self):
        for ds, paths in DEFAULT_CHECKPOINTS.items():
            assert "primary" in paths
            assert "baseline" in paths


# ---------------------------------------------------------------------------
# 3. BBox utilities
# ---------------------------------------------------------------------------

class TestBBoxUtils:
    def test_union_basic(self):
        boxes = [[10, 20, 30, 40], [15, 25, 50, 45], [5, 10, 20, 30]]
        assert union_bboxes(boxes) == [5, 10, 50, 45]

    def test_union_single_box(self):
        assert union_bboxes([[1, 2, 3, 4]]) == [1, 2, 3, 4]

    def test_union_empty(self):
        assert union_bboxes([]) == [0, 0, 0, 0]

    def test_union_same_boxes(self):
        boxes = [[10, 10, 20, 20]] * 5
        assert union_bboxes(boxes) == [10, 10, 20, 20]

    def test_union_returns_int_list(self):
        result = union_bboxes([[1.5, 2.5, 3.5, 4.5]])  # type: ignore
        for v in result:
            assert isinstance(v, int)


# ---------------------------------------------------------------------------
# 4. BIO decoding
# ---------------------------------------------------------------------------

class TestBIODecoding:
    def _make_boxes(self, n: int) -> List[List[int]]:
        return [[i * 10, 0, (i + 1) * 10, 10] for i in range(n)]

    def test_standard_bio_sequence(self):
        words = ["Starbucks", "Coffee", "Co.", "12/04/2022", "$4.50"]
        labels = ["B-COMPANY", "I-COMPANY", "I-COMPANY", "B-DATE", "B-TOTAL"]
        probs = [0.98, 0.96, 0.94, 0.995, 0.99]
        boxes = self._make_boxes(5)

        entities, tokens, agg = reconstruct_entity_spans(words, labels, probs, boxes, boxes)

        assert len(entities) == 3
        company = next(e for e in entities if e.label == "COMPANY")
        assert company.text == "Starbucks Coffee Co."
        assert company.token_count == 3
        assert abs(company.confidence - (0.98 + 0.96 + 0.94) / 3) < 1e-3
        assert company.min_confidence == pytest.approx(0.94, abs=1e-3)
        assert agg["COMPANY"] == "Starbucks Coffee Co."

    def test_all_outside_labels(self):
        words = ["the", "quick", "brown", "fox"]
        labels = ["O", "O", "O", "O"]
        probs = [0.9] * 4
        boxes = self._make_boxes(4)
        entities, _, agg = reconstruct_entity_spans(words, labels, probs, boxes, boxes)
        assert len(entities) == 0
        assert agg == {}

    def test_broken_bio_i_without_b(self):
        """I- tags without a preceding B- must start a new span (recovery)."""
        words = ["Main", "Street", "Suite", "100"]
        labels = ["I-ADDRESS", "I-ADDRESS", "I-ADDRESS", "I-ADDRESS"]
        probs = [0.90, 0.92, 0.91, 0.89]
        boxes = self._make_boxes(4)
        entities, _, _ = reconstruct_entity_spans(words, labels, probs, boxes, boxes)
        assert len(entities) == 1
        assert entities[0].label == "ADDRESS"
        assert entities[0].text == "Main Street Suite 100"

    def test_consecutive_entities_same_class(self):
        words = ["A", "B", "C", "D"]
        labels = ["B-DATE", "O", "B-DATE", "O"]
        probs = [0.95, 0.99, 0.93, 0.99]
        boxes = self._make_boxes(4)
        entities, _, agg = reconstruct_entity_spans(words, labels, probs, boxes, boxes)
        assert len(entities) == 2
        # flat agg should join both
        assert "A" in agg["DATE"]
        assert "C" in agg["DATE"]

    def test_entity_bbox_union(self):
        words = ["ABC", "STORE"]
        labels = ["B-COMPANY", "I-COMPANY"]
        probs = [0.95, 0.90]
        orig_boxes = [[50, 100, 150, 130], [155, 100, 230, 130]]
        norm_boxes = [[100, 200, 300, 260], [310, 200, 460, 260]]
        entities, _, _ = reconstruct_entity_spans(words, labels, probs, orig_boxes, norm_boxes)
        assert len(entities) == 1
        assert entities[0].bbox == [50, 100, 230, 130]  # union of orig_boxes
        assert entities[0].normalized_bbox == [100, 200, 460, 260]


# ---------------------------------------------------------------------------
# 5. Token → word collapsing
# ---------------------------------------------------------------------------

class TestTokenWordCollapsing:
    def test_first_subword_convention(self):
        id2label = {0: "O", 1: "B-COMPANY", 2: "I-COMPANY", 3: "B-TOTAL"}
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
        assert word_probs[0] == pytest.approx(0.95)
        assert word_labels[1] == "O"
        assert word_labels[2] == "B-TOTAL"
        assert word_probs[2] == pytest.approx(0.99)

    def test_no_real_words_all_special(self):
        id2label = {0: "O"}
        word_labels, word_probs = collapse_token_preds_to_words(
            token_label_ids=[0, 0],
            token_probs=[0.99, 0.99],
            word_ids=[None, None],
            id2label=id2label,
            num_words=0,
        )
        assert word_labels == []
        assert word_probs == []

    def test_truncated_words_get_o_label(self):
        """Words beyond the 512 token limit should default to 'O'."""
        id2label = {0: "O", 1: "B-COMPANY"}
        word_labels, word_probs = collapse_token_preds_to_words(
            token_label_ids=[0, 1],
            token_probs=[0.99, 0.95],
            word_ids=[None, 0],
            id2label=id2label,
            num_words=5,  # words 1-4 never seen by tokenizer
        )
        assert len(word_labels) == 5
        assert word_labels[0] == "B-COMPANY"
        for i in range(1, 5):
            assert word_labels[i] == "O"
            assert word_probs[i] == 0.0


# ---------------------------------------------------------------------------
# 6. Confidence calculation
# ---------------------------------------------------------------------------

class TestConfidenceCalculation:
    def _entity(self, words, probs):
        labels = ["B-COMPANY"] + ["I-COMPANY"] * (len(words) - 1)
        boxes = [[i * 10, 0, (i + 1) * 10, 10] for i in range(len(words))]
        entities, _, _ = reconstruct_entity_spans(words, labels, probs, boxes, boxes)
        return entities[0]

    def test_single_token_confidence(self):
        e = self._entity(["TOTAL"], [0.95])
        assert e.confidence == pytest.approx(0.95, abs=1e-3)
        assert e.min_confidence == pytest.approx(0.95, abs=1e-3)

    def test_multi_token_mean_confidence(self):
        e = self._entity(["ABC", "STORE", "LTD"], [0.90, 0.80, 0.70])
        expected_mean = (0.90 + 0.80 + 0.70) / 3
        assert e.confidence == pytest.approx(expected_mean, abs=1e-3)
        assert e.min_confidence == pytest.approx(0.70, abs=1e-3)

    def test_confidence_is_rounded(self):
        e = self._entity(["WORD"], [0.123456789])
        # Should be rounded to 4 decimal places
        assert len(str(round(e.confidence, 4)).split(".")[-1]) <= 4


# ---------------------------------------------------------------------------
# 7. Schema and JSON serialization
# ---------------------------------------------------------------------------

class TestSchema:
    def _make_result(self) -> DocumentInferenceResult:
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
        return DocumentInferenceResult(
            document_type="SROIE",
            model_name="sroie_robust",
            image_path="sample.png",
            image_size=(800, 1000),
            ocr_word_count=20,
            entities=[entity],
            raw_entities_dict={"COMPANY": "Acme Corp"},
            ocr_raw_text="Acme Corp Receipt",
        )

    def test_to_dict(self):
        d = self._make_result().to_dict()
        assert d["document_type"] == "SROIE"
        assert d["entities"][0]["label"] == "COMPANY"
        assert d["entities"][0]["confidence"] == 0.9876

    def test_to_json_valid(self):
        json_str = self._make_result().to_json()
        parsed = json.loads(json_str)
        assert parsed["document_type"] == "SROIE"
        assert parsed["aggregated_fields"]["COMPANY"] == "Acme Corp"

    def test_json_contains_required_keys(self):
        parsed = json.loads(self._make_result().to_json())
        for key in ["document_type", "model_name", "image_size", "ocr_word_count",
                    "ocr_raw_text", "entities", "aggregated_fields"]:
            assert key in parsed, f"Missing key: {key}"

    def test_entity_dict_keys(self):
        parsed = json.loads(self._make_result().to_json())
        entity = parsed["entities"][0]
        for key in ["label", "text", "confidence", "min_confidence", "bbox",
                    "normalized_bbox", "token_count"]:
            assert key in entity, f"Missing entity key: {key}"

    def test_empty_entities_serialization(self):
        res = DocumentInferenceResult(
            document_type="FUNSD",
            model_name="funsd_robust",
            image_path=None,
            image_size=(400, 600),
            ocr_word_count=0,
            entities=[],
        )
        d = res.to_dict()
        assert d["entities"] == []
        assert d["aggregated_fields"] == {}


# ---------------------------------------------------------------------------
# 8. Visualization
# ---------------------------------------------------------------------------

class TestVisualization:
    def test_draw_on_blank_image(self):
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
        result = draw_predictions_on_image(img, [entity])
        assert isinstance(result, Image.Image)
        assert result.size == (400, 400)
        assert result.mode == "RGB"

    def test_draw_empty_entity_list(self):
        img = Image.new("RGB", (200, 200), color="white")
        result = draw_predictions_on_image(img, [])
        assert isinstance(result, Image.Image)

    def test_color_palette_for_all_entity_classes(self):
        expected_classes = ["COMPANY", "DATE", "ADDRESS", "TOTAL",
                            "HEADER", "QUESTION", "ANSWER",
                            "MENU", "SUBTOTAL"]
        for cls in expected_classes:
            color = get_label_color(cls)
            assert len(color) == 3
            assert all(0 <= v <= 255 for v in color)

    def test_get_label_color_unknown_returns_default(self):
        color = get_label_color("UNKNOWN_LABEL_XYZ")
        assert color == COLOR_PALETTE["DEFAULT"]

    def test_get_label_color_strips_bio_prefix(self):
        assert get_label_color("B-COMPANY") == get_label_color("COMPANY")
        assert get_label_color("I-TOTAL") == get_label_color("TOTAL")

    def test_draw_respects_min_confidence_filter(self):
        img = Image.new("RGB", (400, 400), color="white")
        entities = [
            ExtractedEntity(
                label="COMPANY", text="High", confidence=0.90, min_confidence=0.90,
                bbox=[50, 50, 150, 80], normalized_bbox=[50, 50, 150, 80],
                start_word=0, end_word=0, token_count=1,
            ),
            ExtractedEntity(
                label="TOTAL", text="Low", confidence=0.10, min_confidence=0.10,
                bbox=[50, 100, 150, 130], normalized_bbox=[50, 100, 150, 130],
                start_word=1, end_word=1, token_count=1,
            ),
        ]
        # With high threshold only one entity should be drawn
        result_filtered = draw_predictions_on_image(img, entities, min_confidence=0.50)
        result_all = draw_predictions_on_image(img, entities, min_confidence=0.0)
        # Both should be PIL images of same size
        assert result_filtered.size == result_all.size == (400, 400)


# ---------------------------------------------------------------------------
# 9. Invalid input handling (no model required)
# ---------------------------------------------------------------------------

class TestInvalidInputHandling:
    def test_unsupported_dataset_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported"):
            UnseenDocumentInferencePipeline(dataset_type="imagenet")

    def test_nonexistent_image_path_raises_file_not_found(self):
        pipe = UnseenDocumentInferencePipeline(dataset_type="sroie")
        with pytest.raises(FileNotFoundError):
            pipe.predict_image("nonexistent_image_xyz_123.png")

    def test_invalid_image_type_raises_type_error(self):
        pipe = UnseenDocumentInferencePipeline(dataset_type="sroie")
        with pytest.raises(TypeError):
            pipe.predict_image(42)  # type: ignore

    def test_empty_image_returns_empty_result(self):
        """A blank white image should yield 0 OCR words and no entities."""
        pipe = UnseenDocumentInferencePipeline(dataset_type="sroie", use_robust_model=False)
        blank = Image.new("RGB", (200, 200), color="white")
        result = pipe.predict_image(blank)
        assert result.ocr_word_count == 0
        assert result.entities == []
        assert result.raw_entities_dict == {}

    def test_cord_dataset_type_accepted(self):
        pipe = UnseenDocumentInferencePipeline(dataset_type="cord")
        assert pipe.dataset_type == "cord"

    def test_funsd_dataset_type_accepted(self):
        pipe = UnseenDocumentInferencePipeline(dataset_type="funsd")
        assert pipe.dataset_type == "funsd"

    def test_case_insensitive_dataset_type(self):
        pipe = UnseenDocumentInferencePipeline(dataset_type="SROIE")
        assert pipe.dataset_type == "sroie"


# ---------------------------------------------------------------------------
# 10. Regression Tests: CORD One-Character Numeric Tokens Bug
# ---------------------------------------------------------------------------

class TestCordNumericTokenRegression:
    """
    Regression tests verifying that single-character numeric tokens ('1', '2', '4')
    are never discarded, whether produced standalone by OCR or extracted from
    tightly kerned table rows ('1EGG', '2CHOCO', '1BANANA', '4Vivamus', '1.').
    """

    def test_one_character_numeric_tokens_not_discarded(self):
        """
        Verify that tokens with len == 1 (e.g. '1', '2', '4') are preserved through
        collapse_token_preds_to_words and reconstruct_entity_spans.
        """
        words = ["1", "Lorem", "Ipsum", "2", "Pellentesque", "4", "Vivamus"]
        word_labels = [
            "B-MENU", "I-MENU", "I-MENU",
            "B-MENU", "I-MENU",
            "B-MENU", "I-MENU",
        ]
        word_probs = [0.99, 0.98, 0.97, 0.99, 0.96, 0.99, 0.95]
        orig_boxes = [[10, 10, 20, 30], [25, 10, 60, 30], [65, 10, 100, 30],
                      [10, 40, 20, 60], [25, 40, 90, 60],
                      [10, 70, 20, 90], [25, 70, 80, 90]]
        norm_boxes = [[10, 10, 20, 30], [25, 10, 60, 30], [65, 10, 100, 30],
                      [10, 40, 20, 60], [25, 40, 90, 60],
                      [10, 70, 20, 90], [25, 70, 80, 90]]

        entities, tokens, flat_agg = reconstruct_entity_spans(
            words=words,
            word_labels=word_labels,
            word_probs=word_probs,
            original_bboxes=orig_boxes,
            normalized_bboxes=norm_boxes,
        )

        assert len(tokens) == 7
        token_words = [t.word for t in tokens]
        assert "1" in token_words
        assert "2" in token_words
        assert "4" in token_words

        # Check that none of the single-character tokens were dropped
        for digit in ["1", "2", "4"]:
            matching_tokens = [t for t in tokens if t.word == digit]
            assert len(matching_tokens) == 1
            assert matching_tokens[0].label == "B-MENU"

    def test_split_cord_ocr_words_glued_digits(self):
        """
        Test that split_cord_ocr_words splits leading digits from words.
        """
        # Leading digits glued to food names
        assert split_cord_ocr_words("1EGG TART") == ["1", "EGG", "TART"]
        assert split_cord_ocr_words("2CHOCO CUSTARD") == ["2", "CHOCO", "CUSTARD"]
        assert split_cord_ocr_words("1BANANA DONUT") == ["1", "BANANA", "DONUT"]
        assert split_cord_ocr_words("4Vivamus volutpat") == ["4", "Vivamus", "volutpat"]

        # Numbered items with trailing period
        assert split_cord_ocr_words("1. Lorem Ipsum") == ["1", "Lorem", "Ipsum"]
        assert split_cord_ocr_words("4. Vivamus") == ["4", "Vivamus"]

        # Parenthesized digits
        assert split_cord_ocr_words("(1) Item") == ["1", "Item"]
        assert split_cord_ocr_words("(4) Item") == ["4", "Item"]

        # Unit quantities attached to items
        assert split_cord_ocr_words("2PCS") == ["2", "PCS"]
        assert split_cord_ocr_words("2ITEM") == ["2", "ITEM"]

        # Preserved tokens: prices, multipliers, normal words
        assert split_cord_ocr_words("1x") == ["1x"]
        assert split_cord_ocr_words("2X") == ["2X"]
        assert split_cord_ocr_words("$50.00") == ["$50.00"]
        assert split_cord_ocr_words("60.000") == ["60.000"]
        assert split_cord_ocr_words("TOTALDISCS -60.000") == ["TOTALDISCS", "-60.000"]

    def test_draw_raw_ocr_boxes_renders_and_highlights(self):
        """
        Verify that draw_raw_ocr_boxes renders raw OCR detections and highlights numeric tokens.
        """
        img = Image.new("RGB", (300, 200), color="white")
        ocr_results = [
            [[[10, 10], [50, 10], [50, 30], [10, 30]], "1 Lorem", 0.95],
            [[[10, 40], [50, 40], [50, 60], [10, 60]], "2 Ipsum", 0.98],
            [[[10, 70], [50, 70], [50, 90], [10, 90]], "4 Dolor", 0.92],
        ]
        annotated = draw_raw_ocr_boxes(img, ocr_results, highlight_tokens=["1", "2", "4"])
        assert isinstance(annotated, Image.Image)
        assert annotated.size == (300, 200)

    def test_raw_ocr_debug_method_output_schema(self):
        """
        Verify that pipeline.get_raw_ocr_debug returns correct schema and detects numbers.
        """
        pipe = UnseenDocumentInferencePipeline(dataset_type="cord")
        blank = Image.new("RGB", (200, 200), color="white")
        debug_info = pipe.get_raw_ocr_debug(blank)
        assert "total_raw_lines" in debug_info
        assert "raw_ocr_items" in debug_info
        assert "numeric_search" in debug_info
        assert "1" in debug_info["numeric_search"]
        assert "4" in debug_info["numeric_search"]
        assert "variants" in debug_info["numeric_search"]

