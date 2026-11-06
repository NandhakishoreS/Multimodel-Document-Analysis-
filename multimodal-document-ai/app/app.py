"""
Streamlit Web Application: Multimodal Intelligent Document Understanding & Information Extraction.

Provides an interactive web interface for unseen document upload, OCR extraction,
multimodal entity extraction, bounding-box visualization, and structured JSON export.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from PIL import Image

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from inference.final_inference import UnseenDocumentInferencePipeline
from visualization.draw_predictions import draw_predictions_on_image, draw_raw_ocr_boxes

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_app")


# Page layout and aesthetics
st.set_page_config(
    page_title="Multimodal Document AI — Phase 10",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for premium presentation
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .entity-tag {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .ocr-box {
        background-color: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        max-height: 250px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


# Cache inference pipelines in session memory to prevent reloading weights on every interaction
@st.cache_resource(show_spinner=False)
def get_pipeline(dataset_type: str, use_robust_model: bool) -> UnseenDocumentInferencePipeline:
    return UnseenDocumentInferencePipeline(
        dataset_type=dataset_type,
        use_robust_model=use_robust_model,
        project_root=_PROJECT_ROOT,
    )


# Sidebar Configuration
st.sidebar.title("⚙️ Model & Benchmark")

doc_type = st.sidebar.selectbox(
    "Document Benchmark / Domain",
    options=["SROIE", "FUNSD", "CORD"],
    index=0,
    help="Select the domain schema: SROIE (Receipts), FUNSD (Forms), or CORD (Dining receipts).",
)

if doc_type == "SROIE":
    model_choice = st.sidebar.radio(
        "Model Variant",
        options=["Joint-Augmented (Phase 9 Robust)", "Clean Baseline (Phase 5 Corrected)"],
        index=0,
        help="Compare Phase 9 Joint-Augmented model against the Phase 5 clean baseline.",
    )
    use_robust = "Joint-Augmented" in model_choice
else:
    st.sidebar.info(f"Using validated {doc_type} Full Multimodal LayoutLMv3 checkpoint.")
    use_robust = True

confidence_threshold = st.sidebar.slider(
    "Minimum Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.05,
    help="Filter out extracted entity spans with average token confidence below this threshold.",
)

# Header Section
st.markdown('<div class="main-header">📄 Multimodal Intelligent Document Understanding & Information Extraction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-grade unseen document inference pipeline using OCR + 2D Spatial Layout + LayoutLMv3 Multimodal Transformer.</div>', unsafe_allow_html=True)

# Sample image finder
sample_images: Dict[str, List[Path]] = {
    "SROIE": sorted(list((_PROJECT_ROOT / "processed_data/sroie/images").glob("*.png")))[:10],
    "FUNSD": sorted(list((_PROJECT_ROOT / "processed_data/funsd/images").glob("*.png")))[:10],
    "CORD": sorted(list((_PROJECT_ROOT / "processed_data/cord/images").glob("*.png")))[:10],
}

# Image Input Mode
input_mode = st.radio(
    "Document Input Source",
    options=["Upload Unseen Document Image", "Select Preloaded Benchmark Sample"],
    horizontal=True,
)

selected_image: Optional[Image.Image] = None
image_label = ""

if input_mode == "Upload Unseen Document Image":
    uploaded_file = st.file_uploader(
        "Upload a document image (PNG, JPG, JPEG, TIFF, BMP)",
        type=["png", "jpg", "jpeg", "tiff", "bmp"],
    )
    if uploaded_file is not None:
        try:
            selected_image = Image.open(uploaded_file).convert("RGB")
            image_label = uploaded_file.name
        except Exception as e:
            st.error(f"Error opening image file: {e}")
else:
    avail_samples = sample_images.get(doc_type, [])
    if avail_samples:
        sample_names = [p.name for p in avail_samples]
        chosen_sample_name = st.selectbox("Choose sample image:", options=sample_names)
        chosen_path = next(p for p in avail_samples if p.name == chosen_sample_name)
        selected_image = Image.open(chosen_path).convert("RGB")
        image_label = chosen_path.name
    else:
        st.warning("No sample images found for this benchmark.")

# Execution Section
if selected_image is not None:
    col_run, col_info = st.columns([1, 4])
    with col_run:
        run_btn = st.button("🚀 Run Document Understanding", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(f"Running OCR & Multimodal Inference ({doc_type})..."):
            try:
                pipeline = get_pipeline(dataset_type=doc_type.lower(), use_robust_model=use_robust)
                result = pipeline.predict_image(selected_image, min_confidence=confidence_threshold)
                st.session_state["inference_result"] = result
                st.session_state["processed_image"] = selected_image
                st.session_state["image_label"] = image_label
            except Exception as ex:
                st.error(f"Inference failed: {ex}")
                logger.exception("Inference error")

# Render Results
if "inference_result" in st.session_state:
    res = st.session_state["inference_result"]
    active_img = st.session_state["processed_image"]

    st.markdown("---")

    # Metrics Summary Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Benchmark Domain", res.document_type)
    with m2:
        st.metric("OCR Words Detected", res.ocr_word_count)
    with m3:
        st.metric("Entities Extracted", len(res.entities))
    with m4:
        avg_conf = (
            sum(e.confidence for e in res.entities) / len(res.entities)
            if res.entities else 0.0
        )
        st.metric("Mean Entity Confidence", f"{avg_conf:.1%}")

    # Dual Column: Image Visualizations & Extracted Information
    col_vis, col_data = st.columns([1, 1])

    with col_vis:
        st.subheader("🖼️ Document Visualizations")
        vis_tab1, vis_tab2, vis_tab3 = st.tabs(["Bounding-Box Annotations", "Original Document", "Raw OCR Boxes (Debug)"])

        with vis_tab1:
            annotated_img = draw_predictions_on_image(
                active_img,
                res.entities,
                min_confidence=confidence_threshold,
            )
            st.image(annotated_img, use_container_width=True, caption="Detected Entities with Bounding Boxes")

        with vis_tab2:
            st.image(active_img, use_container_width=True, caption=f"Original: {st.session_state.get('image_label', '')}")

        with vis_tab3:
            pipeline_active = get_pipeline(dataset_type=doc_type.lower(), use_robust_model=use_robust)
            if not hasattr(pipeline_active, "get_raw_ocr_debug"):
                # Dynamically bind get_raw_ocr_debug if pipeline was cached from older class definition
                import types
                from inference.final_inference import UnseenDocumentInferencePipeline as _CurrentPipe
                if hasattr(_CurrentPipe, "get_raw_ocr_debug"):
                    pipeline_active.get_raw_ocr_debug = types.MethodType(_CurrentPipe.get_raw_ocr_debug, pipeline_active)

            if hasattr(pipeline_active, "get_raw_ocr_debug"):
                raw_debug = pipeline_active.get_raw_ocr_debug(active_img)
                ocr_results_list = [
                    [item["polygon"], item["text"], item["confidence"]]
                    for item in raw_debug.get("raw_ocr_items", [])
                ]
            else:
                # Direct fallback to RapidOCR if method is unavailable on cached instance
                import numpy as _np
                from rapidocr_onnxruntime import RapidOCR
                _ocr = RapidOCR()
                _ocr_res, _ = _ocr(_np.array(active_img.convert("RGB")))
                ocr_results_list = [
                    [it[0], str(it[1]), float(it[2])] for it in (_ocr_res or [])
                ]

            raw_box_img = draw_raw_ocr_boxes(
                active_img,
                ocr_results_list,
                highlight_tokens=["1", "2", "3", "4"],
            )
            st.image(raw_box_img, use_container_width=True, caption="Raw OCR Boxes ('1'-'4' Highlighted in Green)")

    with col_data:
        st.subheader("📋 Extracted Structured Entities")

        if res.entities:
            # Table of detected entities
            table_data = [
                {
                    "Entity Class": e.label,
                    "Extracted Text": e.text,
                    "Confidence": f"{e.confidence:.2%}",
                    "Word Count": e.token_count,
                    "Bounding Box": str(e.bbox),
                }
                for e in res.entities
            ]
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
        else:
            st.info("No entities detected above the confidence threshold.")

        st.subheader("🧩 Aggregated Document Fields")
        if res.raw_entities_dict:
            for k, v in res.raw_entities_dict.items():
                st.markdown(f"**{k}**: `{v}`")
        else:
            st.write("None")

    # OCR vs Multimodal Semantic Understanding Section
    st.markdown("---")
    st.subheader("🔍 OCR Transcription vs. Multimodal Semantic Understanding")

    with st.expander("Show Raw OCR Transcript ('What text is present?')", expanded=False):
        st.markdown("""
        > **Scientific Clarification**:  
        > **OCR** merely transcribes character symbols from pixel coordinates without contextual comprehension.  
        > **LayoutLMv3 Multimodal AI** synthesizes lexical semantics, 2D spatial positioning, and visual patch context to assign semantic roles (`COMPANY`, `ADDRESS`, `DATE`, `TOTAL`, `MENU`, `SUBTOTAL`).
        """)
        if res.ocr_raw_text:
            st.markdown(f'<div class="ocr-box">{res.ocr_raw_text}</div>', unsafe_allow_html=True)
        else:
            st.write("No raw OCR text available.")

    with st.expander("🔬 Raw OCR Debug & Numeric Token Inspector (Phase 10 CORD)", expanded=False):
        pipeline_active = get_pipeline(dataset_type=doc_type.lower(), use_robust_model=use_robust)
        raw_debug = pipeline_active.get_raw_ocr_debug(active_img)
        st.markdown(f"**Total Raw OCR Lines Detected**: `{raw_debug['total_raw_lines']}`")
        st.markdown("**Numeric Token Search ('1', '2', '3', '4' & variants)**:")
        st.json(raw_debug["numeric_search"])

    # Structured Output & JSON Download Section
    st.markdown("---")
    st.subheader("💾 Structured Output & JSON Export")

    json_output = res.to_json(indent=2)
    col_dl, col_blank = st.columns([1, 3])
    with col_dl:
        st.download_button(
            label="📥 Download Structured JSON",
            data=json_output,
            file_name=f"document_extraction_{res.document_type.lower()}_{Path(st.session_state.get('image_label', 'doc')).stem}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.code(json_output, language="json")

else:
    st.info("👆 Upload or select a document image above and click **'Run Document Understanding'** to begin extraction.")
