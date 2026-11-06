# Multimodal Document Understanding — Final Validated Experimental Results

**Project**: Multimodal Transformer-Based Intelligent Document Understanding and Information Extraction  
**Date of Freeze**: 2026-09-09  
**Status**: OFFICIALLY AUDITED & FROZEN FOR DEFENSE / THESIS  

---

## 1. Primary Cross-Dataset Benchmark Results

The table below presents the verified primary models across the three target benchmark domains. Every row represents **one specific model evaluated under its official protocol**:

| Benchmark | Document Domain | Primary Model Checkpoint | Precision | Recall | Macro F1 | Micro F1 | Test Split Size | Essential Methodological Notes & Caveats |
|---|---|---|---|---|---|---|---|---|
| **SROIE** | Scanned Store Receipts | `models/checkpoints/phase9/sroie/joint_aug/best/` | **0.8694** | **0.8980** | **0.8837** | **0.8835** | 335 receipts | **Primary production model**. Trained with joint layout coordinate jitter and OCR noise; recovers +18.9 pp under worst-case spatial jitter. Evaluated on Phase 5 clean deduplicated test split. |
| **FUNSD** | Noisy Scanned Office Forms | `models/checkpoints/layoutlmv3/funsd/full/` | **0.7804** | **0.8196** | **0.7179** | **0.7995** | 50 forms | **Primary production model**. Evaluated across 50 scanned test forms containing complex multi-column layouts and 29.1% background tokens. |
| **CORD** | Commercial Dining Receipts | `models/checkpoints/layoutlmv3/cord/full/` | **0.9748** | **0.9763** | **0.9733** | **0.9756** | 80 receipts | **Simplified 3-Class Formulation**. Evaluated on `valid_line` words collapsed into `MENU`, `SUBTOTAL`, and `TOTAL` with zero background `O` tokens. **Not directly comparable to full-document 30-class CORD KIE benchmarks.** |

---

## 2. SROIE Model Comparison: Baseline vs. Augmented Models

To prevent any metric conflation, all SROIE model variants evaluated on the identical Phase 5 clean deduplicated test split (335 receipts) are detailed individually below:

| Model Variant | Training Intervention | Precision | Recall | Macro F1 | Micro F1 | Avg Stressed F1 (14 Conditions) | Worst-Case F1 (High Noise Floor) | Composite Utility Score | Selection Decision |
|---|---|---|---|---|---|---|---|---|---|
| **Joint-Augmented** | Layout Jitter + OCR Noise | **0.8694** | **0.8980** | **0.8837** | **0.8835** | **0.8710** | **0.8100** | **0.8578** | **SELECTED PRIMARY MODEL** |
| **Layout-Augmented** | Multi-Scale Layout Jitter Only | 0.8730 | 0.8987 | 0.8856 | 0.8857 | 0.8501 | 0.6979 | 0.8186 | Ablation Candidate |
| **OCR-Augmented** | Character Noise + Masking Only | 0.8711 | 0.8882 | 0.8794 | 0.8796 | 0.8358 | 0.7112 | 0.8159 | Ablation Candidate |
| **Corrected Baseline** | Standard Training (Clean) | 0.8689 | 0.8889 | 0.8785 | 0.8788 | 0.8109 | 0.6716 | 0.7961 | Reference Baseline (Preserved) |
| *Historical Baseline* | *Uncorrected Old Preprocessing* | *0.8792* | *0.8878* | *0.8828* | *0.8835* | *N/A* | *N/A* | *N/A* | *Deprecated (Contains 58 duplicate images)* |

### Critical Clarifications:
1. **Separation of Metrics**: The numbers `P = 0.8689, R = 0.8889, Macro F1 = 0.8785, Micro F1 = 0.8788` belong exclusively to the **Phase 5 Corrected Baseline**. The numbers `P = 0.8694, R = 0.8980, Macro F1 = 0.8837, Micro F1 = 0.8835` belong exclusively to the **Phase 9 Joint-Augmented Model**. They are separate models with distinct weights and checkpoints.
2. **Historical Discontinuity**: The historical SROIE 0.8828 Macro F1 resulted from an uncorrected upstream split defect (58 duplicate images shared between train and test). It is cataloged strictly for historical traceability and is permanently superseded by the leak-free corrected formulation.

---

## 3. Per-Entity Extraction Accuracy (SROIE)

| Entity Class | Corrected Baseline (Clean) | Joint-Augmented (Clean) | Clean Delta | Baseline (±10% Jitter Stress) | Joint-Augmented (±10% Jitter Stress) | Stress Recovery Delta |
|---|---|---|---|---|---|---|
| **COMPANY** | 0.9049 | 0.9027 | -0.0022 | 0.5000 | **0.8862** | **+38.62 pp** |
| **ADDRESS** | 0.8434 | 0.8540 | +0.0106 | 0.6235 | **0.8251** | **+20.16 pp** |
| **TOTAL** | 0.7725 | 0.7851 | +0.0126 | 0.5962 | **0.7542** | **+15.80 pp** |
| **DATE** | 0.9930 | 0.9930 | 0.0000 | 0.9667 | **0.9916** | **+2.49 pp** |
| **Macro Average** | **0.8785** | **0.8837** | **+0.0052** | **0.6716** | **0.8608** | **+18.92 pp** |

---

## 4. Methodological Boundaries & Limitations

1. **Protocol Incompatibility with ML Baselines**: Baseline Logistic Regression models evaluated token-level classifications (excluding background tokens), yielding Micro F1 ~0.92 and Macro F1 ~0.26. LayoutLMv3 was evaluated using strict **Seqeval entity-level span F1** (requiring exact word boundary matching). The baselines demonstrate the insufficiency of 1D text features, but numerical deltas must reflect the protocol distinction.
2. **CORD 3-Class Scope**: CORD's 0.9733 Macro F1 reflects closed-set line-item categorization on pre-segmented lines, not unconstrained open-domain KIE.
3. **Upstream OCR Bounding**: Downstream semantic accuracy depends fundamentally on upstream OCR recall. Missed words cannot be classified by the transformer.
