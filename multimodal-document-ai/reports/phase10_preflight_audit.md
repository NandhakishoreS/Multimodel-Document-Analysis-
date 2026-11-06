# Phase 10 Pre-Flight Audit

**Project**: Multimodal Transformer-Based Intelligent Document Understanding and Information Extraction  
**Date**: 2026-09-09  
**Status**: AUDIT COMPLETE — PRE-FLIGHT VERIFIED  

---

## 1. Project Structure Overview

The repository is structured systematically across data ingestion, preprocessing, multimodal modeling, evaluation, and reporting:

```
c:\MDA\multimodal-document-ai\
├── configs/                  # Dataset label schemas and model hyperparameter YAMLs
│   ├── labels/               # Specific BIO label definitions (sroie, funsd, cord)
│   ├── layoutlmv3.yaml       # Transformer architecture & training defaults
│   └── preprocessing.yaml    # Coordinate normalization & image resizing parameters
├── datasets/                 # Raw dataset download and exploration scripts
│   ├── sroie/                # SROIE ICDAR 2019 dataset ingestion
│   ├── funsd/                # FUNSD form understanding dataset ingestion
│   └── cord/                 # CORD receipt understanding dataset ingestion
├── docs/                     # Historical project documentation (Phase 1–5)
├── evaluation/               # Unified evaluation and phase-specific analysis modules
│   ├── metrics.py            # Word/Entity level precision, recall, F1, CER, WER
│   ├── layoutlmv3_metrics.py # Seqeval entity-level evaluation wrappers
│   ├── phase6_unified_eval.py# Cross-dataset unified benchmarking protocol
│   ├── phase7_error_analysis.py # Confusion matrices, span breakdown, error taxonomy
│   ├── phase8_robustness.py  # 14-condition noise and layout perturbation engine
│   └── phase9_ablation.py    # 7-condition modality ablation engine
├── experiments/              # Experiment logs and training summary JSONs
├── inference/                # LayoutLMv3 single-sample inference and entity reconstruction
│   └── layoutlmv3_predict.py # Token-to-word BIO span aggregation logic
├── models/                   # Model architectures and saved checkpoints
│   ├── baseline/             # Heuristic, Layout, and Text-only baselines
│   ├── layoutlmv3/           # PyTorch LayoutLMv3 classifier and dataset loaders
│   ├── checkpoints/
│   │   ├── layoutlmv3/       # Historical baseline checkpoints (sroie/corrected, funsd/full, cord/full)
│   │   └── phase9/           # Robustness-aware checkpoints (layout_aug, ocr_aug, joint_aug)
│   └── saved/                # Joblib serialized traditional baseline classifiers
├── preprocessing/            # Document parsing, bounding box normalization, and token alignment
│   ├── bbox_utils.py         # Box normalization to [0, 1000] and spatial utilities
│   ├── text_utils.py         # String cleansing, whitespace normalization
│   ├── label_mapping.py      # BIO label schema conversions
│   ├── sroie_processor.py    # Historical SROIE processor (deprecated due to duplicate bug)
│   ├── sroie_corrected_processor.py # Validated Phase 5 corrected processor
│   ├── funsd_processor.py    # Form document parser
│   ├── cord_processor.py     # Simplified CORD receipt parser (3 classes, valid_line)
│   └── unified_dataset.py    # Common JSONL dataset iterator
├── processed_data/           # Formatted JSONL train/val/test splits and image pools
│   ├── sroie/                # Historical SROIE processed data
│   ├── sroie_corrected/      # Verified SROIE corrected splits (533 train, 59 val, 335 test)
│   ├── funsd/                # FUNSD splits (119 train, 30 val, 50 test)
│   └── cord/                 # CORD splits (640 train, 80 val, 80 test)
├── reports/                  # Markdown reports, CSV result tables, and publication figures
├── scripts/                  # Reproducibility runners and diagnostic scripts
├── tests/                    # Pytest test suite covering all functional phases
├── training/                 # Training pipelines for baselines and LayoutLMv3
│   ├── train_layoutlmv3.py   # Standard LayoutLMv3 training loop
│   └── phase9_augmentation.py# Dynamic in-memory jitter and OCR corruption transforms
└── utils/                    # Visualization utilities
    └── visualization.py      # Bounding box rendering on document images
```

---

## 2. Available Datasets & Preprocessing Status

| Dataset | Splits (Train / Val / Test) | Image Count | Domain | Annotations | Active Formulation |
|---|---|---|---|---|---|
| **SROIE (Corrected)** | 533 / 59 / 335 | 927 receipts | Scanned store receipts | Word-level OCR + 4 entities (`COMPANY`, `DATE`, `ADDRESS`, `TOTAL`) | **Phase 5 Corrected Formulation** (removes duplicate image leakage and fixes word token alignment) |
| **FUNSD** | 119 / 30 / 50 | 199 forms | Noisy scanned office forms | Word-level OCR + 3 entities (`HEADER`, `QUESTION`, `ANSWER`) | Full dataset with validated entity IDs |
| **CORD** | 640 / 80 / 80 | 800 receipts | Indonesian dining receipts | Word-level OCR + 3 classes (`MENU`, `SUBTOTAL`, `TOTAL`) | **Simplified 3-class formulation** using `valid_line` words only; zero background `O` tokens |

---

## 3. Available Model Checkpoints

### Historical Baselines (Phases 4–6)
1. **SROIE Baseline (Corrected)**:
   - Path: `models/checkpoints/layoutlmv3/sroie/corrected/best/`
   - Validated Metrics: Precision = 0.8689, Recall = 0.8889, Macro F1 = 0.8785, Micro F1 = 0.8788
2. **FUNSD Full Baseline**:
   - Path: `models/checkpoints/layoutlmv3/funsd/full/`
   - Validated Metrics: Precision = 0.7804, Recall = 0.8196, Macro F1 = 0.7179, Micro F1 = 0.7995
3. **CORD Full Baseline**:
   - Path: `models/checkpoints/layoutlmv3/cord/full/`
   - Validated Metrics: Precision = 0.9748, Recall = 0.9763, Macro F1 = 0.9733, Micro F1 = 0.9756

### Robustness-Aware Models (Phase 9)
1. **SROIE Layout-Augmented**:
   - Path: `models/checkpoints/phase9/sroie/layout_aug/best/`
   - Validated Metrics: Clean Macro F1 = 0.8856, Avg Stressed F1 = 0.8501, Worst-Case F1 = 0.6979
2. **SROIE OCR-Augmented**:
   - Path: `models/checkpoints/phase9/sroie/ocr_aug/best/`
   - Validated Metrics: Clean Macro F1 = 0.8794, Avg Stressed F1 = 0.8358, Worst-Case F1 = 0.7112
3. **SROIE Joint-Augmented**:
   - Path: `models/checkpoints/phase9/sroie/joint_aug/best/`
   - Validated Metrics: Clean Macro F1 = 0.8837, Avg Stressed F1 = 0.8710, Worst-Case F1 = 0.8100
   - Composite Utility Score: **0.8578** (Rank 1 / Pareto-optimal)

---

## 4. Primary Checkpoints Selected for Final Inference

To balance accuracy, real-world robustness, and domain fidelity, the primary checkpoints selected for production deployment in Phase 10 are:

1. **SROIE**: `models/checkpoints/phase9/sroie/joint_aug/best/`
   - *Rationale*: Outperforms baseline on clean test data (+0.53 pp) and provides dramatic resilience against layout jitter (+18.9 pp at ±10% jitter) and text masking (+10.1 pp at 20% masking). Baseline (`models/checkpoints/layoutlmv3/sroie/corrected/best/`) is retained as the baseline reference model for comparative analysis.
2. **FUNSD**: `models/checkpoints/layoutlmv3/funsd/full/`
   - *Rationale*: Primary validated model achieving 0.7179 Macro F1 on noisy scanned forms.
3. **CORD**: `models/checkpoints/layoutlmv3/cord/full/`
   - *Rationale*: Validated model achieving 0.9733 Macro F1 under the documented 3-class receipt formulation.

---

## 5. Existing Inference Capabilities

- `inference/layoutlmv3_predict.py` implements:
  - `reconstruct_entities()`: Converts word-level BIO predictions into structured span dictionaries with average token probabilities.
  - `token_preds_to_word_preds()`: Maps subword token predictions back to word-level using first-subword pooling convention.
  - `predict_document()`: Executes forward pass given a dictionary containing pre-extracted `words`, `bboxes`, and `image_path`.
- **Limitation of Existing Inference**: It requires pre-existing OCR words and bounding boxes in JSONL format. It does not accept a raw unannotated image file directly. Phase 10 must introduce an end-to-end unseen image inference engine.

---

## 6. Existing Streamlit & Visualization Capabilities

- `utils/visualization.py` provides:
  - `visualize_predictions()`: Draws bounding boxes around detected entities on PIL images, color-coded by entity type with text labels.
- `app/` directory exists but is currently empty.
- Phase 10 will build a comprehensive, production-grade interactive Streamlit web application supporting document upload, OCR extraction, model inference, bounding-box overlay, confidence inspection, and structured JSON export.

---

## 7. Known Limitations & Scientific Caveats

1. **SROIE Preprocessing Discontinuity**: Phase 1–4 used an uncorrected SROIE dataset exhibiting duplicate images across train and test splits (Macro F1 = 0.8828). Phase 5 established the corrected formulation (Macro F1 = 0.8785) which is the true primary baseline.
2. **CORD Task Simplification**: CORD evaluation only contains `valid_line` words belonging to `MENU`, `SUBTOTAL`, and `TOTAL` without `O` (background) tokens. Its 0.9733 Macro F1 cannot be directly compared to standard end-to-end CORD benchmarks.
3. **Baseline Comparison Metric Incompatibility**: Baseline models (LogReg/TF-IDF) were evaluated on token-level micro-averages, whereas LayoutLMv3 is evaluated using entity-level Seqeval span metrics. Direct F1 comparison must note this discrepancy.
4. **FUNSD Scale**: FUNSD comprises only 149 training/val and 50 test documents, resulting in higher variance across layout perturbations.

---

## 8. Pre-Flight Verification Verdict

- [x] All directories inspected.
- [x] Datasets and splits validated.
- [x] All 6 model checkpoints confirmed present and accessible.
- [x] No modifications made during audit.
- [x] Status: **PASSED — READY FOR PHASE 10 IMPLEMENTATION**.
