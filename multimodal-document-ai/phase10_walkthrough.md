# Phase 10: Complete System Walkthrough & Execution Guide

**Project**: Multimodal Transformer-Based Intelligent Document Understanding and Information Extraction  
**Date**: 2026-09-09  
**Status**: VALIDATED & PRODUCTION READY  

---

## 1. Environment Setup

The project runs on Windows 11 with Python 3.12 (using `.venv312`) and CUDA-accelerated PyTorch:

```powershell
# Navigate to project directory
cd c:\MDA\multimodal-document-ai

# Activate existing Python 3.12 virtual environment
& ".\.venv312\Scripts\Activate.ps1"

# Verify Python version and CUDA GPU detection
& ".\.venv312\Scripts\python.exe" -c "import torch; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 2. Dataset Requirements & Structure

Preprocessed datasets are stored under `processed_data/` in standardized JSONL format:

- **SROIE (Corrected)**: `processed_data/sroie_corrected/`
  - `train.jsonl` (533 samples)
  - `val.jsonl` (59 samples)
  - `test.jsonl` (335 deduplicated samples)
  - Images pool: `processed_data/sroie/images/`
- **FUNSD**: `processed_data/funsd/`
  - `train.jsonl` (119 samples)
  - `val.jsonl` (30 samples)
  - `test.jsonl` (50 samples)
  - Images pool: `processed_data/funsd/images/`
- **CORD (Simplified 3-Class)**: `processed_data/cord/`
  - `train.jsonl` (640 samples)
  - `val.jsonl` (80 samples)
  - `test.jsonl` (80 samples)
  - Images pool: `processed_data/cord/images/`

---

## 3. Model & Checkpoint Locations

| Benchmark | Checkpoint Role | Disk Path | Validated Clean Macro F1 |
|---|---|---|---|
| **SROIE** | **Primary (Robust Joint-Augmented)** | `models/checkpoints/phase9/sroie/joint_aug/best/` | **0.8837** |
| **SROIE** | Preserved Baseline (Phase 5 Corrected) | `models/checkpoints/layoutlmv3/sroie/corrected/best/` | **0.8785** |
| **FUNSD** | Primary (Full Multimodal) | `models/checkpoints/layoutlmv3/funsd/full/` | **0.7179** |
| **CORD** | Primary (Full Multimodal 3-Class) | `models/checkpoints/layoutlmv3/cord/full/` | **0.9733** |

---

## 4. Evaluation & Benchmarking

To run the unified cross-dataset evaluation across all benchmarks:

```powershell
# Cross-dataset unified evaluation
& ".\.venv312\Scripts\python.exe" scripts/run_phase6_evaluation.py

# 14-condition robustness stress evaluation
& ".\.venv312\Scripts\python.exe" scripts/run_phase8_robustness.py

# Multimodal ablation and robustness improvement evaluation
& ".\.venv312\Scripts\python.exe" scripts/run_phase9_pipeline.py
```

---

## 5. Unseen Document Inference (Python API)

To run the end-to-end unseen document inference pipeline directly in Python:

```python
from pathlib import Path
from inference.final_inference import UnseenDocumentInferencePipeline
from visualization.draw_predictions import draw_predictions_on_image

# 1. Initialize pipeline for SROIE using the primary robust model
pipeline = UnseenDocumentInferencePipeline(
    dataset_type="sroie",
    use_robust_model=True,
)

# 2. Run inference on an unseen image file
image_path = "processed_data/sroie/images/X00016469612.png"
result = pipeline.predict_image(image_path, min_confidence=0.20)

# 3. Print extracted entities and confidence
print(f"Document Type: {result.document_type}")
print(f"OCR Words Extracted: {result.ocr_word_count}")
for entity in result.entities:
    print(f"  [{entity.label}] {entity.text} (Conf: {entity.confidence:.2%})")

# 4. Generate annotated bounding-box image
annotated_image = draw_predictions_on_image(image_path, result.entities)
annotated_image.save("extracted_receipt.png")

# 5. Export structured JSON
with open("extracted_receipt.json", "w", encoding="utf-8") as f:
    f.write(result.to_json(indent=2))
```

---

## 6. Interactive Streamlit Web Application

Launch the interactive web interface:

```powershell
& ".\.venv312\Scripts\streamlit.exe" run app.py
```

- Local URL: `http://localhost:8501`
- Network URL: `http://192.168.x.x:8501`

### Application Capabilities:
1. **Sidebar Navigation**: Switch between SROIE, FUNSD, and CORD. For SROIE, toggle between the Joint-Augmented model and the Phase 5 Baseline.
2. **Confidence Slider**: Filter out entities with average token confidence below a custom threshold ($0.0 \dots 1.0$).
3. **Dual Input Mode**: Upload any unseen receipt/form image from your local drive, or select from preloaded benchmark test samples.
4. **Interactive Visualization**: View the original image alongside color-coded bounding box annotations with entity tags and confidence scores.
5. **Raw OCR Inspection**: Compare the raw text transcript against semantic entity assignments.
6. **One-Click JSON Export**: Download formatted structured JSON with a single click.

---

## 7. Automated Testing Suite

To run the complete 105-item automated test suite covering all project phases:

```powershell
& ".\.venv312\Scripts\pytest.exe" tests/ -v
```

Expected output:
```
============================ 105 passed in 18.56s =============================
```

To run Phase 10 integration tests specifically:
```powershell
& ".\.venv312\Scripts\pytest.exe" tests/test_phase10_final.py -v
```

---

## 8. Artifacts Directory

Key summary reports and visualizations are stored under `reports/`:
- Comprehensive Research Report: [`reports/FINAL_PROJECT_REPORT.md`](file:///c:/MDA/multimodal-document-ai/reports/FINAL_PROJECT_REPORT.md)
- Pre-Flight Audit: [`reports/phase10_preflight_audit.md`](file:///c:/MDA/multimodal-document-ai/reports/phase10_preflight_audit.md)
- Model Selection Rationale: [`reports/phase10_final_model_selection.md`](file:///c:/MDA/multimodal-document-ai/reports/phase10_final_model_selection.md)
- Reproducibility Audit: [`reports/phase10_reproducibility_report.md`](file:///c:/MDA/multimodal-document-ai/reports/phase10_reproducibility_report.md)
- Confidence Methodology: [`reports/phase10_confidence_methodology.md`](file:///c:/MDA/multimodal-document-ai/reports/phase10_confidence_methodology.md)
- Robustness Summary: [`reports/phase10_robustness_summary.md`](file:///c:/MDA/multimodal-document-ai/reports/phase10_robustness_summary.md)
- Limitations Catalog: [`reports/phase10_limitations.md`](file:///c:/MDA/multimodal-document-ai/reports/phase10_limitations.md)
- System Architecture Diagram: [`reports/phase10_architecture.png`](file:///c:/MDA/multimodal-document-ai/reports/phase10_architecture.png)
- Publication Figures: [`reports/phase10_figures/`](file:///c:/MDA/multimodal-document-ai/reports/phase10_figures/)
