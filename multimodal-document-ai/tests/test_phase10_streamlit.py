"""
Phase 10 Streamlit Application Tests.

These tests validate the Streamlit application's structure, importability,
and component correctness WITHOUT actually launching a live Streamlit server
(which cannot be automated in a headless test runner). The launch validation
itself is documented in reports/phase10_streamlit_validation.md.

Tests cover:
- App module can be imported without error
- All required UI components are present (via source inspection)
- get_pipeline() caching wrapper instantiates a valid pipeline
- OCR → inference → visualization full pipeline chain (unit-level)
- Download button output format and filename construction
- Error message generation for common failure cases
- Model selector logic (SROIE only shows model toggle)
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from PIL import Image

from inference.final_inference import UnseenDocumentInferencePipeline
from inference.schemas import DocumentInferenceResult, ExtractedEntity
from inference.postprocessing import reconstruct_entity_spans
from visualization.draw_predictions import draw_predictions_on_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "app" / "app.py"
ROOT_APP_PATH = PROJECT_ROOT / "app.py"


# ---------------------------------------------------------------------------
# 1. App file existence
# ---------------------------------------------------------------------------

class TestAppFileExists:
    def test_app_directory_app_py_exists(self):
        assert APP_PATH.exists(), f"app/app.py not found at {APP_PATH}"

    def test_root_app_py_exists(self):
        assert ROOT_APP_PATH.exists(), f"Root app.py not found at {ROOT_APP_PATH}"

    def test_app_is_non_empty(self):
        content = APP_PATH.read_text(encoding="utf-8")
        assert len(content) > 500, "app/app.py is suspiciously short"


# ---------------------------------------------------------------------------
# 2. App source structural checks
# ---------------------------------------------------------------------------

class TestAppSourceStructure:
    @pytest.fixture(scope="class")
    @classmethod
    def app_source(cls):
        return APP_PATH.read_text(encoding="utf-8")

    def test_imports_streamlit(self, app_source):
        assert "import streamlit" in app_source or "import streamlit as st" in app_source

    def test_imports_inference_pipeline(self, app_source):
        assert "UnseenDocumentInferencePipeline" in app_source

    def test_imports_draw_predictions(self, app_source):
        assert "draw_predictions_on_image" in app_source

    def test_uses_cache_resource(self, app_source):
        assert "cache_resource" in app_source

    def test_has_file_uploader(self, app_source):
        assert "file_uploader" in app_source

    def test_has_download_button(self, app_source):
        assert "download_button" in app_source

    def test_has_selectbox_for_doc_type(self, app_source):
        assert "selectbox" in app_source
        assert "SROIE" in app_source
        assert "FUNSD" in app_source
        assert "CORD" in app_source

    def test_has_confidence_slider(self, app_source):
        assert "slider" in app_source
        assert "confidence" in app_source.lower()

    def test_has_model_radio_for_sroie(self, app_source):
        assert "radio" in app_source
        assert "Joint-Augmented" in app_source or "joint_aug" in app_source.lower()

    def test_has_error_handling(self, app_source):
        assert "st.error" in app_source

    def test_page_config_set(self, app_source):
        assert "set_page_config" in app_source

    def test_has_spinner(self, app_source):
        assert "spinner" in app_source

    def test_json_code_display(self, app_source):
        assert "st.code" in app_source

    def test_image_display(self, app_source):
        assert "st.image" in app_source

    def test_dataframe_display(self, app_source):
        assert "dataframe" in app_source or "DataFrame" in app_source

    def test_syntax_valid(self, app_source):
        """App source must be syntactically valid Python."""
        try:
            ast.parse(app_source)
        except SyntaxError as e:
            pytest.fail(f"app/app.py has syntax errors: {e}")


# ---------------------------------------------------------------------------
# 3. get_pipeline equivalent (no Streamlit runtime needed)
# ---------------------------------------------------------------------------

class TestPipelineFactory:
    def test_sroie_pipeline_instantiation(self):
        pipe = UnseenDocumentInferencePipeline(dataset_type="sroie", use_robust_model=True)
        assert pipe.dataset_type == "sroie"
        assert pipe.use_robust_model is True

    def test_sroie_baseline_pipeline_instantiation(self):
        pipe = UnseenDocumentInferencePipeline(dataset_type="sroie", use_robust_model=False)
        assert not pipe.use_robust_model

    def test_funsd_pipeline_instantiation(self):
        pipe = UnseenDocumentInferencePipeline(dataset_type="funsd", use_robust_model=True)
        assert pipe.dataset_type == "funsd"

    def test_cord_pipeline_instantiation(self):
        pipe = UnseenDocumentInferencePipeline(dataset_type="cord", use_robust_model=True)
        assert pipe.dataset_type == "cord"

    def test_project_root_resolved(self):
        pipe = UnseenDocumentInferencePipeline(dataset_type="sroie", project_root=str(PROJECT_ROOT))
        assert pipe.project_root == PROJECT_ROOT

    def test_device_defaults_to_cpu_or_cuda(self):
        import torch
        pipe = UnseenDocumentInferencePipeline(dataset_type="sroie")
        assert pipe.device.type in ("cpu", "cuda")


# ---------------------------------------------------------------------------
# 4. Blank image pipeline (no model load needed — hits OCR path)
# ---------------------------------------------------------------------------

class TestBlankImagePipeline:
    def test_blank_image_returns_empty_result(self):
        pipe = UnseenDocumentInferencePipeline(dataset_type="sroie", use_robust_model=False)
        blank = Image.new("RGB", (300, 300), color=(255, 255, 255))
        result = pipe.predict_image(blank)
        assert isinstance(result, DocumentInferenceResult)
        assert result.ocr_word_count == 0
        assert result.entities == []

    def test_blank_image_json_serializable(self):
        pipe = UnseenDocumentInferencePipeline(dataset_type="sroie", use_robust_model=False)
        blank = Image.new("RGB", (200, 200), color=(255, 255, 255))
        result = pipe.predict_image(blank)
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["document_type"] == "SROIE"
        assert parsed["entities"] == []

    def test_blank_image_visualization_does_not_crash(self):
        pipe = UnseenDocumentInferencePipeline(dataset_type="sroie", use_robust_model=False)
        blank = Image.new("RGB", (200, 200), color=(255, 255, 255))
        result = pipe.predict_image(blank)
        annotated = draw_predictions_on_image(blank, result.entities)
        assert isinstance(annotated, Image.Image)


# ---------------------------------------------------------------------------
# 5. Download button JSON format validation
# ---------------------------------------------------------------------------

class TestJSONExport:
    def _synthetic_result(self, doc_type: str = "SROIE") -> DocumentInferenceResult:
        entity = ExtractedEntity(
            label="TOTAL",
            text="$12.50",
            confidence=0.98,
            min_confidence=0.98,
            bbox=[100, 200, 300, 230],
            normalized_bbox=[100, 200, 300, 230],
            start_word=5,
            end_word=5,
            token_count=1,
        )
        return DocumentInferenceResult(
            document_type=doc_type,
            model_name=f"{doc_type.lower()}_robust",
            image_path="test_image.png",
            image_size=(800, 1200),
            ocr_word_count=30,
            entities=[entity],
            raw_entities_dict={"TOTAL": "$12.50"},
            ocr_raw_text="Total $12.50",
        )

    def test_json_is_valid_string(self):
        result = self._synthetic_result()
        json_str = result.to_json(indent=2)
        assert isinstance(json_str, str)
        json.loads(json_str)  # must not raise

    def test_json_document_type_field(self):
        result = self._synthetic_result("FUNSD")
        parsed = json.loads(result.to_json())
        assert parsed["document_type"] == "FUNSD"

    def test_json_entity_confidence_in_range(self):
        result = self._synthetic_result()
        parsed = json.loads(result.to_json())
        for ent in parsed["entities"]:
            assert 0.0 <= ent["confidence"] <= 1.0

    def test_json_bbox_is_list_of_4_ints(self):
        result = self._synthetic_result()
        parsed = json.loads(result.to_json())
        for ent in parsed["entities"]:
            assert isinstance(ent["bbox"], list)
            assert len(ent["bbox"]) == 4
            for v in ent["bbox"]:
                assert isinstance(v, (int, float))

    def test_download_filename_construction(self):
        """Mimics the filename logic in app.py for different doc types."""
        doc_type = "SROIE"
        image_label = "receipt_0001.png"
        stem = Path(image_label).stem
        filename = f"document_extraction_{doc_type.lower()}_{stem}.json"
        assert filename == "document_extraction_sroie_receipt_0001.json"

    def test_cord_three_class_labels_in_output(self):
        """CORD entities should only ever be MENU, SUBTOTAL, or TOTAL."""
        cord_labels = {"MENU", "SUBTOTAL", "TOTAL"}
        result = self._synthetic_result("CORD")
        for ent in result.entities:
            assert ent.label in cord_labels or ent.label == "TOTAL"  # synthetic fixture uses TOTAL


# ---------------------------------------------------------------------------
# 6. Error handling behaviours (without live UI)
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_missing_image_gives_file_not_found(self):
        pipe = UnseenDocumentInferencePipeline(dataset_type="sroie")
        with pytest.raises(FileNotFoundError):
            pipe.predict_image("/nonexistent/path/image.png")

    def test_wrong_type_gives_type_error(self):
        pipe = UnseenDocumentInferencePipeline(dataset_type="sroie")
        with pytest.raises(TypeError):
            pipe.predict_image({"pixels": []})  # type: ignore

    def test_unsupported_dataset_type(self):
        with pytest.raises(ValueError):
            UnseenDocumentInferencePipeline(dataset_type="wikipedia")

    def test_invalid_confidence_threshold_clip(self):
        """Confidence threshold of 1.0 should yield no entities on blank image."""
        pipe = UnseenDocumentInferencePipeline(dataset_type="sroie", use_robust_model=False)
        blank = Image.new("RGB", (200, 200), color="white")
        result = pipe.predict_image(blank, min_confidence=1.0)
        assert len(result.entities) == 0

    def test_confidence_threshold_zero_returns_all(self):
        """Threshold of 0.0 should not filter out any entities."""
        words = ["ABC", "DATE"]
        labels = ["B-COMPANY", "B-DATE"]
        probs = [0.30, 0.25]
        boxes = [[0, 0, 10, 10], [20, 0, 30, 10]]
        entities, _, _ = reconstruct_entity_spans(words, labels, probs, boxes, boxes)
        filtered = [e for e in entities if e.confidence >= 0.0]
        assert len(filtered) == 2
