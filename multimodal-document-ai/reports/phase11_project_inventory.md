# Phase 11: Complete Project Inventory & Artifact Catalog

**Project**: Multimodal Transformer-Based Intelligent Document Understanding and Information Extraction  
**Date**: 2026-09-09  
**Audit Status**: COMPLETE PROJECT INVENTORY VERIFIED  

---

## 1. Project Phase-by-Phase Traceability Matrix

| Phase | Title / Scope | Key Deliverables & Scripts | Core Artifacts & Reports | Verification Status |
|---|---|---|---|---|
| **Phase 1** | Project Inception, Environment & Raw Data Ingestion | `datasets/sroie/`, `datasets/funsd/`, `datasets/cord/`, `scripts/validate_phase1.py` | `docs/phase1.md`, raw data splits | VERIFIED |
| **Phase 2** | Text Extraction, Bounding Box Normalization & Alignment | `preprocessing/bbox_utils.py`, `preprocessing/text_utils.py`, `preprocessing/sroie_processor.py` | `processed_data/sroie/`, `processed_data/funsd/`, `processed_data/cord/`, `docs/phase2.md` | VERIFIED |
| **Phase 3** | Baseline Model Architecture (Heuristic, Linear & Spatial) | `models/baseline/`, `training/train_baseline.py`, `training/evaluate_baseline.py` | `models/saved/*.joblib`, `reports/baseline_results.csv`, `docs/phase3.md` | VERIFIED |
| **Phase 4** | Multimodal LayoutLMv3 Token Classification Fine-Tuning | `models/layoutlmv3/`, `training/train_layoutlmv3.py`, `configs/layoutlmv3.yaml` | `models/checkpoints/layoutlmv3/`, `reports/figures/`, `docs/phase4.md` | VERIFIED |
| **Phase 5** | Forensic Data Quality Audit & SROIE Preprocessing Remediation | `preprocessing/sroie_corrected_processor.py`, `scripts/audit_dataset_splits.py`, `scripts/train_sroie_corrected.py` | `processed_data/sroie_corrected/`, `reports/sroie_corrected_training_report.md`, `reports/phase5_split_audit.csv` | **CRITICAL AUDIT PASS** (Identified & excised 58 duplicate images; corrected subword alignment) |
| **Phase 6** | Cross-Dataset Unified Evaluation & Modality Ablation Benchmark | `evaluation/phase6_unified_eval.py`, `scripts/run_phase6_evaluation.py`, `scripts/run_phase6_ablation.py` | `reports/phase6_unified_results.csv`, `reports/phase6_unified_results.md`, `reports/phase6_final_report.md` | VERIFIED |
| **Phase 7** | Multi-Class Error Analysis, Confusion Matrices & Taxonomy | `evaluation/phase7_error_analysis.py`, `scripts/run_phase7_error_analysis.py` | `reports/phase7_error_analysis_report.md`, `reports/error_analysis/`, `reports/phase7_entity_error_comparison.png` | VERIFIED |
| **Phase 8** | Adversarial Robustness & Operational Stress Evaluation | `evaluation/phase8_robustness.py`, `scripts/run_phase8_robustness.py` | `reports/phase8_robustness_report.md`, `reports/phase8_robustness_results.csv`, `reports/phase8_sroie_entity_robustness.csv` | **CRITICAL FAILURE DISCOVERY** (FUNSD -65.4%, SROIE -23.5% under ±10% layout jitter) |
| **Phase 9** | Robustness-Aware Model Improvement & Multimodal Ablation | `training/phase9_augmentation.py`, `evaluation/phase9_ablation.py`, `scripts/run_phase9_pipeline.py` | `models/checkpoints/phase9/sroie/`, `reports/phase9_robustness_improvement_report.md`, `reports/phase9_model_comparison.csv` | **REMEDIATION PASS** (+18.92 pp gain under layout jitter via Joint Augmentation) |
| **Phase 10** | Unseen-Document Inference, Streamlit App & Final Validation | `inference/final_inference.py`, `visualization/draw_predictions.py`, `app/app.py`, `app.py` | `reports/FINAL_PROJECT_REPORT.md`, `reports/phase10_architecture.png`, `reports/phase10_figures/`, `tests/test_phase10_final.py` | **INTEGRATION PASS** (105 tests passing; live web app operational) |

---

## 2. Dataset Inventory

1. **SROIE (ICDAR 2019 Receipt Benchmark)**:
   - Historical Path: `processed_data/sroie/` (contains 58 duplicate images across train/test; legacy only).
   - **Authoritative Corrected Path**: `processed_data/sroie_corrected/` (533 train, 59 val, 335 deduplicated test receipts).
   - Label Schema: `configs/labels/sroie.yaml` (9 BIO classes: `O`, `B-COMPANY`, `I-COMPANY`, `B-DATE`, `I-DATE`, `B-ADDRESS`, `I-ADDRESS`, `B-TOTAL`, `I-TOTAL`).
2. **FUNSD (Form Understanding in Noisy Scanned Documents)**:
   - Path: `processed_data/funsd/` (119 train, 30 val, 50 test forms).
   - Label Schema: `configs/labels/funsd.yaml` (7 BIO classes: `O`, `B-HEADER`, `I-HEADER`, `B-QUESTION`, `I-QUESTION`, `B-ANSWER`, `I-ANSWER`).
3. **CORD (Consolidated Receipt Dataset — Simplified 3-Class Formulation)**:
   - Path: `processed_data/cord/` (640 train, 80 val, 80 test receipts).
   - Label Schema: `configs/labels/cord.yaml` (13 BIO classes; evaluation focused on `MENU`, `SUBTOTAL`, `TOTAL`).

---

## 3. Model Checkpoint Inventory

| Model Identifier | Relative Path | Parameters | Base Weights | Validation F1 | Test Macro F1 | Status |
|---|---|---|---|---|---|---|
| `sroie_corrected_baseline` | `models/checkpoints/layoutlmv3/sroie/corrected/best/` | 125M | `microsoft/layoutlmv3-base` | 0.9090 | 0.8785 | Preserved Reference Baseline |
| `sroie_layout_aug` | `models/checkpoints/phase9/sroie/layout_aug/best/` | 125M | `microsoft/layoutlmv3-base` | 0.9102 | 0.8856 | Phase 9 Ablation Checkpoint |
| `sroie_ocr_aug` | `models/checkpoints/phase9/sroie/ocr_aug/best/` | 125M | `microsoft/layoutlmv3-base` | 0.8994 | 0.8794 | Phase 9 Ablation Checkpoint |
| `sroie_joint_aug` | `models/checkpoints/phase9/sroie/joint_aug/best/` | 125M | `microsoft/layoutlmv3-base` | 0.9115 | **0.8837** | **PRIMARY PRODUCTION SROIE MODEL** |
| `funsd_full` | `models/checkpoints/layoutlmv3/funsd/full/` | 125M | `microsoft/layoutlmv3-base` | 0.6946 | **0.7179** | **PRIMARY PRODUCTION FUNSD MODEL** |
| `cord_full` | `models/checkpoints/layoutlmv3/cord/full/` | 125M | `microsoft/layoutlmv3-base` | 0.9712 | **0.9733** | **PRIMARY PRODUCTION CORD MODEL** |
| Historical uncorrected SROIE | `experiments/layoutlmv3_sroie_full_20260908_091136/` | 125M | `microsoft/layoutlmv3-base` | - | 0.8828 | Historical / Deprecated |

---

## 4. Test Suite Inventory

Total: **105 automated unit and integration tests** located in `tests/`:
- `tests/test_baselines.py` (2 tests): Logistic regression baselines.
- `tests/test_bbox_utils.py` (5 tests): Coordinate normalization, clamping, geometry.
- `tests/test_dataset.py` (8 tests): JSONL loading, schema verification, batching.
- `tests/test_entity_reconstruction.py` (12 tests): Word-level BIO span reconstruction.
- `tests/test_label_mapping.py` (2 tests): Bi-directional schema consistency.
- `tests/test_metrics.py` (3 tests): Precision, recall, F1, CER, WER computations.
- `tests/test_model.py` (19 tests): LayoutLMv3 architecture, tensor shapes, gradient checkpointing.
- `tests/test_phase6_unified_eval.py` (7 tests): Cross-dataset unified benchmarking protocol.
- `tests/test_phase7_error_analysis.py` (7 tests): Error distribution taxonomy and confusion matrices.
- `tests/test_phase8_robustness.py` (5 tests): Perturbation engines and determinism.
- `tests/test_phase9_robustness_improvement.py` (6 tests): Training augmentation transforms and immutability.
- `tests/test_text_utils.py` (2 tests): OCR text cleansing and whitespace normalization.
- `tests/test_token_alignment.py` (15 tests): First-subword alignment convention and special token handling.
- `tests/test_phase10_final.py` (12 tests): Unseen image inference, RapidOCR integration, schemas, postprocessing, confidence, visualization, and error handling.

---

## 5. Artifact Directory Inventory

- Figures: `reports/figures/`, `reports/phase9/visualizations/`, `reports/phase10_figures/`, `reports/phase10_architecture.png`.
- Detailed CSV Results: `reports/phase6_unified_results.csv`, `reports/phase8_robustness_results.csv`, `reports/phase9_ablation_results.csv`, `reports/phase9_robustness_results.csv`, `reports/phase9_model_comparison.csv`, `reports/phase10_robustness_summary.csv`.
- Markdown Reports: 15 formal research reports spanning all project phases.
