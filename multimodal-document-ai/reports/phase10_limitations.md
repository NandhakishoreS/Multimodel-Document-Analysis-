# Phase 10: System Limitations & Scientific Scope

**Project**: Multimodal Transformer-Based Intelligent Document Understanding and Information Extraction  
**Date**: 2026-09-09  
**Status**: APPROVED SCIENTIFIC DOCUMENTATION  

---

## 1. Overview

Transparent documentation of methodological boundaries, dataset characteristics, architectural assumptions, and experimental constraints is essential for genuine scientific integrity. This document details the eleven primary limitations identified across Phases 1 through 10 of this project.

---

## 2. Comprehensive Limitations Catalog

### 1. SROIE Upstream Duplicate Images
The original ICDAR 2019 SROIE dataset distributed across public repositories contains 58 identical or near-identical receipt scans shared between the official training and test splits. While Phase 5 identified and deduplicated these samples to establish the clean corrected test split (335 test receipts), the existence of near-duplicate receipt templates in raw commercial streams remains an inherent characteristic of receipt datasets.

### 2. SROIE Corrected Preprocessing Discontinuity
The corrected SROIE formulation established in Phase 5 differs fundamentally from the uncorrected Phase 1–4 preprocessing pipeline in word-to-subword alignment, punctuation handling, and duplicate exclusion. Consequently, historical results ($F_1 = 0.8828$) and corrected results ($F_1 = 0.8785$) represent distinct experimental baselines and must not be conflated. The corrected formulation is the sole authoritative baseline for the project.

### 3. CORD Simplified 3-Class Formulation
The CORD receipt experiment evaluated throughout this project is based on a **simplified 3-class line-segmentation formulation**:
- Fine-grained hierarchy collapsed into three classes: `MENU`, `SUBTOTAL`, and `TOTAL`.
- Evaluated exclusively on `valid_line` words identified in the dataset annotations.
- Contains zero background (`O`) tokens in the evaluated test sequences.
As a result, the near-ceiling Macro F1 ($0.9733$) reflects high closed-set line-item classification fidelity and **must not be characterized as directly equivalent to standard full-document CORD KIE benchmarks** (which evaluate 30+ fine-grained hierarchical entities over noisy, unsegmented receipt text).

### 4. FUNSD Dataset Scale & Variance
The FUNSD form understanding benchmark consists of only 149 training/validation forms and 50 test forms. Because of this small sample size, entity metrics (particularly `HEADER` with fewer total instances) exhibit wider confidence intervals and higher sensitivity to layout shifts.

### 5. Deterministic Single-Seed Experimental Protocols
Due to computational time and resource allocation constraints across extensive 14-condition stress matrices and 7-condition ablation grids, primary model training runs were conducted using fixed random seeds (Seed 42). While all results are 100% reproducible deterministically, multi-seed training distributions ($\ge 5$ random initializations) would provide additional empirical variance estimates.

### 6. Limited Statistical Replication across Benchmarks
While document-level paired Wilcoxon signed-rank tests were computed across the 335 SROIE test documents confirming statistical significance ($p = 3.24 \times 10^{-37}$), exhaustive document-level paired statistical tests were not replicated across all perturbation severities for FUNSD and CORD.

### 7. Upstream OCR Dependency
The multimodal inference pipeline relies on upstream OCR (RapidOCR / Tesseract) for character recognition and bounding-box coordinates. If the OCR engine fails to detect a text line or severely miscalculates bounding-box coordinates, LayoutLMv3 cannot recover the missed entity. Downstream extraction accuracy is fundamentally bounded by upstream OCR recall and bounding box precision.

### 8. Sensitivity to Long and Discontinuous Entity Spans
Multi-line or deeply nested entities (such as long commercial addresses spanning 15–25 tokens) remain vulnerable to boundary fragmentation errors. If intermediate punctuation or whitespace is assigned `O` or an incorrect continuation tag, the span splits into separate entities.

### 9. Confidence Calibration Limitations
Confidence scores are computed directly from Softmax posterior probabilities and aggregated via arithmetic mean. While bounded in $[0.0, 1.0]$, raw Softmax probabilities from deep neural networks are known to suffer from over-confidence (temperature scaling or Platt scaling was not fitted). Confidence scores should be interpreted as relative certainty rankings rather than strict empirical probabilities of correctness.

### 10. Hardware & Resource Constraints
Model training and inference were developed on a single NVIDIA GeForce RTX 3060 Laptop GPU with 6 GB VRAM. This necessitated gradient accumulation (effective batch size 8) and half-precision (`fp16`) execution. Scaling to larger multimodal models (e.g., LayoutLMv3-large, 368M parameters) or end-to-end vision-language models (e.g., Donut, Nougat) was not feasible within this compute envelope.

### 11. Real-World Domain Shift
Evaluation datasets represent specific document distributions: Malaysian receipts (SROIE), 1980s–1990s scanned US tobacco litigation forms (FUNSD), and Indonesian dining receipts (CORD). Real-world inference on modern e-commerce invoices, multi-page PDFs, rotated mobile phone camera captures, or multilingual European documents may exhibit domain shift degradation.

---

## 3. Summary Verdict

Documenting these eleven limitations ensures that all claims, reported metrics, and conclusions remain scientifically grounded and verifiable.
