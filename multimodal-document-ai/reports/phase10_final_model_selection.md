# Phase 10: Final Model & Checkpoint Selection

**Project**: Multimodal Transformer-Based Intelligent Document Understanding and Information Extraction  
**Date**: 2026-09-09  
**Status**: OFFICIALLY SELECTED & FROZEN FOR PRODUCTION INFERENCE  

---

## 1. Executive Summary

This document establishes and justifies the final model checkpoints selected for production deployment, unseen document inference, and interactive Streamlit demonstration across the three benchmark domains: SROIE (receipts), FUNSD (forms), and CORD (restaurant receipts).

Every selection is justified by empirical evidence from Phase 4–5 baseline training, Phase 6 unified cross-dataset evaluation, Phase 7 comprehensive error analysis, Phase 8 perturbation stress testing, and Phase 9 robustness-aware optimization.

---

## 2. Checkpoint Selection Summary Table

| Benchmark | Selected Final Model | Checkpoint Path | Architecture | Clean Macro F1 | Primary Strength | Baseline Preserved for Comparison |
|---|---|---|---|---|---|---|
| **SROIE** | **Joint-Augmented LayoutLMv3** | `models/checkpoints/phase9/sroie/joint_aug/best/` | `microsoft/layoutlmv3-base` (125M params) | **0.8837** | Resilience against spatial jitter (+18.9 pp) and text masking (+10.1 pp); highest composite utility (0.8578) | `models/checkpoints/layoutlmv3/sroie/corrected/best/` (0.8785) |
| **FUNSD** | **Full Multimodal LayoutLMv3** | `models/checkpoints/layoutlmv3/funsd/full/` | `microsoft/layoutlmv3-base` (125M params) | **0.7179** | High entity recall (0.8196) and solid question/answer association on scanned forms | N/A (single validated target) |
| **CORD** | **Full Multimodal LayoutLMv3** | `models/checkpoints/layoutlmv3/cord/full/` | `microsoft/layoutlmv3-base` (125M params) | **0.9733** | Near-ceiling extraction on line-item receipts | N/A (single validated target) |

---

## 3. Dataset-by-Dataset Detailed Selection Rationale

### 3.1 SROIE: Joint-Augmented LayoutLMv3 (Phase 9)

#### Historical Context & The Corrected Formulation
In Phase 4, the initial SROIE model achieved a reported Macro F1 of $0.8828$. However, Phase 5 rigorous data auditing uncovered two critical methodological defects in that initial pipeline:
1. **Duplicate Image Leakage**: Upstream dataset duplication caused 58 identical or near-identical receipt images to appear in both training and test splits.
2. **Subword Label Misalignment**: Punctuation and whitespace tokenization discrepancies artificially inflated certain span accuracies.

Phase 5 corrected these issues, establishing the **Corrected SROIE Formulation** (`processed_data/sroie_corrected/`, 533 train, 59 val, 335 test) with a genuine, leak-free clean test Macro F1 of **0.8785** (Micro F1 = 0.8788). **This corrected formulation supersedes all historical uncorrected results.**

#### Why Joint-Augmented Supersedes the Clean Baseline
Phase 8 stress testing revealed extreme operational fragility in the clean baseline model:
- Under $\pm 10\%$ bounding-box coordinate jitter, SROIE Macro F1 collapsed by **23.5%** ($0.8784 \rightarrow 0.6716$), with the `COMPANY` entity falling by **44.8%** ($0.9049 \rightarrow 0.5000$).
- Under 20% random word masking, `ADDRESS` dropped to $0.6087$.

In Phase 9, three robustness-aware models were trained using dynamic in-memory augmentations. As evaluated across 14 stress conditions:
- **Baseline**: Clean F1 = 0.8784, Avg Stressed F1 = 0.8109, Worst-Case F1 = 0.6716 (Utility = 0.7961)
- **Layout-Augmented**: Clean F1 = 0.8856, Avg Stressed F1 = 0.8501, Worst-Case F1 = 0.6979 (Utility = 0.8186)
- **OCR-Augmented**: Clean F1 = 0.8794, Avg Stressed F1 = 0.8358, Worst-Case F1 = 0.7112 (Utility = 0.8159)
- **Joint-Augmented**: Clean F1 = **0.8837**, Avg Stressed F1 = **0.8710**, Worst-Case F1 = **0.8100** (Utility = **0.8578**)

The **Joint-Augmented** model is selected as the primary production model because:
1. It achieves the **highest worst-case floor** ($0.8100$, a +13.84 pp gain over baseline).
2. It achieves the **highest average stressed performance** ($0.8710$).
3. It exhibits **zero clean performance regression** (+0.53 pp over baseline clean test F1).
4. Statistical validation (paired Wilcoxon test, $p = 3.24 \times 10^{-37}$, Cohen's $d = 0.919$) confirms this improvement is statistically significant.

*Preserved Baseline*: The baseline checkpoint `models/checkpoints/layoutlmv3/sroie/corrected/best/` is preserved to enable side-by-side comparative inspection in the Streamlit application.

---

### 3.2 FUNSD: Full Multimodal LayoutLMv3

#### Selection Rationale
FUNSD represents complex, unstructured office forms characterized by dense tabular structures, handwritten annotations, and diverse layouts. The full multimodal LayoutLMv3 checkpoint (`models/checkpoints/layoutlmv3/funsd/full/`):
- Achieves **0.7179 Macro F1** and **0.7995 Micro F1** (Precision = 0.7804, Recall = 0.8196).
- Excels on `QUESTION` ($F_1 = 0.8144$) and `ANSWER` ($F_1 = 0.7679$).
- Handles multi-column layouts via joint 2D coordinate embeddings.
- Serves as the primary validated form understanding engine.

---

### 3.3 CORD: Full Multimodal LayoutLMv3

#### Selection Rationale & Crucial Methodological Caveat
The CORD receipt checkpoint (`models/checkpoints/layoutlmv3/cord/full/`):
- Achieves **0.9733 Macro F1** and **0.9756 Micro F1** (Precision = 0.9748, Recall = 0.9763).
- Delivers near-perfect extraction across commercial line items: `MENU` ($F_1 = 0.9782$), `SUBTOTAL` ($F_1 = 0.9577$), and `TOTAL` ($F_1 = 0.9839$).

> [!WARNING]
> **Essential Scientific Caveat on CORD Formulation**:  
> The CORD experiment in this repository uses a **simplified 3-class formulation**:
> 1. It filters tokens exclusively to `valid_line` words belonging to `MENU`, `SUBTOTAL`, and `TOTAL`.
> 2. It contains **zero background (`O`) tokens** in the evaluated sequence.
> Consequently, this 0.9733 F1 reflects classification accuracy across pre-segmented line items and **must not be characterized as directly comparable to standard full-document CORD KIE benchmarks** (which evaluate 30+ nested sub-categories over uncurated raw receipt text).

---

## 4. Verification and Checkpoint Integrity

All selected checkpoints:
1. Exist in their respective frozen paths with complete model weights (`model.safetensors` or `pytorch_model.bin`), configuration (`config.json`), and tokenizer configurations.
2. Have been verified via automated hash verification; no weights have been retrained or modified during Phase 10.
3. Are fully integrated into the unified inference pipeline `inference/final_inference.py`.
