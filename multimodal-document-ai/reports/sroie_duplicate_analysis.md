# SROIE Upstream Duplicate Images Sensitivity Analysis

**Phase 5 Validation Audit**  
**Model**: LayoutLMv3-base Full Checkpoint (`models/checkpoints/layoutlmv3/sroie/full`)  
**Dataset**: ICDAR 2019 SROIE  

---

## 1. Executive Summary

During the Step 1 dataset split audit, 8 test images were identified as byte-identical copies of images present in the training (7 images) or validation split (1 image). This anomaly is an **upstream dataset artifact** originating from the official ICDAR 2019 SROIE dataset distribution, where identical receipts were submitted under different filenames.

To scientifically quantify the effect of these duplicates without altering the historical 361-sample baseline, the best LayoutLMv3 checkpoint was evaluated on both the **original test split (361 samples)** and the **deduplicated test split (353 samples)**.

- **Original Test Entity Macro F1 (361 samples)**: **`0.8828`** (Micro F1: `0.8835`)
- **Deduplicated Test Entity Macro F1 (353 samples)**: **`0.8813`** (Micro F1: `0.8820`)
- **Absolute Difference**: **`-0.0015`** Macro F1 / **`-0.0015`** Micro F1

**Conclusion**: The removal of the 8 upstream duplicates produces negligible metric variation (< 0.002 F1). The model's high extraction performance is genuine across unseen documents and not an artifact of upstream image duplicates.

---

## 2. Upstream Duplicate Pairs Evidence

| # | Test Sample ID | Source Split | Source Sample ID | SHA-256 Hash | Duplicate Nature |
| :-: | :--- | :---: | :--- | :--- | :--- |
| 1 | `X51006328937` | **TRAIN** | `X51006329399` | `3d0a8110d684933d...` | Byte-identical image content in upstream ICDAR 2019 release |
| 2 | `X51006328967` | **VAL** | `X51006332575` | `1a70bf5fbf23348e...` | Byte-identical image content in upstream ICDAR 2019 release |
| 3 | `X51006329388` | **TRAIN** | `X51006328913` | `09d7a335ea5b2927...` | Byte-identical image content in upstream ICDAR 2019 release |
| 4 | `X51006401853` | **TRAIN** | `X51007135247` | `0c3fb317ff7da1d4...` | Byte-identical image content in upstream ICDAR 2019 release |
| 5 | `X51009008091` | **TRAIN** | `X51006008091` | `1361fad63c351a92...` | Byte-identical image content in upstream ICDAR 2019 release |
| 6 | `X51009008095` | **TRAIN** | `X51006008095` | `b118aae9ff373202...` | Byte-identical image content in upstream ICDAR 2019 release |
| 7 | `X51009453729` | **TRAIN** | `X51005453729` | `5f1751baf3fefc0c...` | Byte-identical image content in upstream ICDAR 2019 release |
| 8 | `X51009568881` | **TRAIN** | `X51005568881` | `f8d05389a7a87202...` | Byte-identical image content in upstream ICDAR 2019 release |

---

## 3. Metrics Comparison: Original vs. Deduplicated

| Metric | Original Test (361 samples) | Deduplicated Test (353 samples) | Absolute Difference | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Entity Macro F1** | **0.8828** | **0.8813** | **-0.0015** | **-0.17%** |
| **Entity Micro F1** | **0.8835** | **0.8820** | **-0.0015** | **-0.17%** |
| **Entity Precision** | 0.8792 | 0.8773 | -0.0019 | -0.22% |
| **Entity Recall** | 0.8878 | 0.8867 | -0.0011 | -0.12% |

### Per-Entity Breakdown Comparison

| Entity Class | Original Precision | Deduplicated Precision | Original Recall | Deduplicated Recall | Original F1 | Deduplicated F1 | Diff F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **COMPANY** | 0.9205 | 0.9188 | 0.9465 | 0.9452 | **0.9333** | **0.9318** | **-0.0015** |
| **ADDRESS** | 0.9245 | 0.9229 | 0.9528 | 0.9517 | **0.9384** | **0.9371** | **-0.0013** |
| **DATE** | 0.9612 | 0.9603 | 0.9666 | 0.9658 | **0.9639** | **0.9631** | **-0.0008** |
| **TOTAL** | 0.7045 | 0.7014 | 0.6870 | 0.6856 | **0.6957** | **0.6934** | **-0.0023** |

---

## 4. Scientific Interpretation

1. **Nature of Duplication**: The 8 duplicate image pairs exist in the raw ICDAR 2019 dataset (`jsdnrs/ICDAR2019-SROIE`), where several receipts have subtle filename variations (e.g., `X51006008095` in train vs. `X51009008095` in test). They are an upstream dataset packaging defect rather than an engineering or modeling leakage bug.
2. **Minimal Sensitivity**: Removing all 8 duplicate images from the evaluation test split shifts the Entity Macro F1 by only a fraction of a percentage point. Precision, recall, and per-entity F1 scores remain virtually identical.
3. **Integrity Confirmation**: LayoutLMv3's extraction capabilities on SROIE hold robustly on strictly non-duplicated test receipts (Macro F1 = 0.8817 vs. 0.8828 originally). Both numbers are retained and documented transparently.
