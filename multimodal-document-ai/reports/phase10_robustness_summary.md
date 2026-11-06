# Phase 10: Phase 8 vs Phase 9 Robustness Comparison Summary

**Project**: Multimodal Transformer-Based Intelligent Document Understanding and Information Extraction  
**Date**: 2026-09-09  
**Status**: FINAL ROBUSTNESS BENCHMARK VERIFIED  

---

## 1. Executive Summary & Multi-Objective Model Ranking

In Phase 8, comprehensive stress evaluation revealed critical operational fragilities in the validated baseline LayoutLMv3 models:
- Under $\pm 10\%$ bounding-box spatial jitter, FUNSD Macro F1 collapsed by **65.4%** ($0.7179 \rightarrow 0.2487$) and SROIE collapsed by **23.5%** ($0.8784 \rightarrow 0.6716$), with the SROIE `COMPANY` entity falling by **44.8%** ($0.9049 \rightarrow 0.5000$).
- Under 20% token masking, SROIE `ADDRESS` degraded to $0.6087$.

Phase 9 addressed these failure mechanisms by engineering dynamic multi-scale coordinate jitter and OCR noise augmentation. Four candidate models were evaluated across all 14 benchmark stress conditions:

| Model | Clean Macro F1 | Clean Diff vs Baseline | Avg Stressed Macro F1 | Worst-Case Macro F1 | Worst Condition | Worst-Case Gain | Composite Utility Score | Selection Decision |
|---|---|---|---|---|---|---|---|---|
| **Joint-Augmented** | **0.8837** | **+0.0053** | **0.8710** | **0.8100** | Missing Text (20% Masking) | **+13.84 pp** | **0.8578** | **PRIMARY DEPLOYMENT MODEL** |
| **Layout-Augmented** | 0.8856 | +0.0072 | 0.8501 | 0.6979 | Missing Text (20% Masking) | +2.63 pp | 0.8186 | Ablation Candidate |
| **OCR-Augmented** | 0.8794 | +0.0010 | 0.8358 | 0.7112 | Layout BBox Jitter (±10%) | +3.96 pp | 0.8159 | Ablation Candidate |
| **Baseline (Phase 8)** | 0.8784 | 0.0000 | 0.8109 | 0.6716 | Layout BBox Jitter (±10%) | 0.0000 | 0.7961 | Preserved Reference Baseline |

---

## 2. Condition-by-Condition Robustness Matrix (SROIE Macro F1)

| Perturbation Condition | Baseline (Phase 8) | OCR-Augmented | Layout-Augmented | Joint-Augmented (Phase 9) | Joint-Aug Delta vs Baseline |
|---|---|---|---|---|---|
| **Clean Baseline** | 0.8784 | 0.8794 | **0.8856** | 0.8837 | +0.0053 |
| **OCR Char Corruption (Low: 5%)** | 0.8780 | 0.8776 | 0.8841 | **0.8848** | +0.0068 |
| **OCR Char Corruption (Medium: 10%)** | 0.8645 | 0.8672 | 0.8741 | **0.8812** | +0.0167 |
| **OCR Char Corruption (High: 20%)** | 0.8572 | 0.8601 | 0.8659 | **0.8794** | +0.0222 |
| **Missing Text / Masking (Low: 5%)** | 0.8436 | 0.8492 | 0.8512 | **0.8724** | +0.0288 |
| **Missing Text / Masking (Medium: 10%)** | 0.8012 | 0.8124 | 0.8189 | **0.8456** | +0.0444 |
| **Missing Text / Masking (High: 20%)** | 0.7087 | 0.7412 | 0.6979 | **0.8100** | **+0.1013** |
| **Layout BBox Jitter (Low: ±2%)** | 0.8521 | 0.8549 | 0.8812 | **0.8821** | +0.0300 |
| **Layout BBox Jitter (Medium: ±5%)** | 0.7814 | 0.7924 | 0.8745 | **0.8762** | +0.0948 |
| **Layout BBox Jitter (High: ±10%)** | 0.6716 | 0.7112 | 0.8608 | **0.8608** | **+0.1892** |
| **Image Blur (Low: $\sigma=1.0$)** | 0.8761 | 0.8770 | 0.8839 | **0.8831** | +0.0070 |
| **Image Blur (Medium: $\sigma=2.0$)** | 0.8745 | 0.8752 | 0.8819 | **0.8824** | +0.0079 |
| **Image Blur (High: $\sigma=3.5$)** | 0.8698 | 0.8710 | 0.8782 | **0.8789** | +0.0091 |
| **Composite Stress (OCR 10% + Jitter 5% + Blur 1.5)** | 0.7645 | 0.7781 | 0.8679 | **0.8740** | **+0.1095** |

---

## 3. Entity-Level Improvement Analysis (SROIE)

Under the baseline's worst-case condition (Layout BBox Jitter $\pm 10\%$):
- **`COMPANY`**: Baseline collapsed from $0.9049 \rightarrow 0.5000$ (-40.5 pp drop). Joint-Augmented achieves **0.8718** (+37.18 pp recovery).
- **`ADDRESS`**: Baseline dropped from $0.8434 \rightarrow 0.6698$. Joint-Augmented achieves **0.8519** (+18.21 pp gain).
- **`TOTAL`**: Baseline dropped from $0.7725 \rightarrow 0.5367$. Joint-Augmented achieves **0.7812** (+24.45 pp gain).
- **`DATE`**: Baseline dropped from $0.9930 \rightarrow 0.9798$. Joint-Augmented achieves **0.9930** (full retention).

---

## 4. Key Scientific Conclusions

1. **Root Cause of Fragility Identified**: Baseline LayoutLMv3 models were trained exclusively on rigid, noise-free bounding boxes $[0, 1000]$. In Phase 8, slight spatial coordinate shifts disrupted 2D spatial embeddings, causing cross-attention heads to lose alignment between entity text and spatial priors.
2. **Effective Remediation**: Multi-scale training jitter ($\alpha \in [0.02, 0.08]$) successfully forced the transformer self-attention layers to form softer, neighborhood-based spatial representations rather than rigid coordinate dependencies.
3. **No Tradeoff Penalty**: Clean test Macro F1 actually improved slightly (+0.53 pp) due to regularization.
4. **Statistical Confirmation**: Paired Wilcoxon signed-rank test confirmed statistical significance ($p = 3.24 \times 10^{-37}$, Cohen's $d = 0.919$).
