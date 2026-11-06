# Phase 9: Robustness-Aware Model Improvement & Multimodal Ablation Report

**Multimodal Document Understanding & Information Extraction Project**  
**Execution Date**: 2026-09-09  
**Status**: VALIDATED & COMPLETE  

---

## 1. Executive Summary

Phase 9 investigates the functional failure mechanisms discovered in Phase 8 and evaluates targeted robustness interventions. Through a controlled 7-condition multimodal ablation study across SROIE, FUNSD, and CORD, we establish the true functional contributions of Text, Layout, and Image modalities. Dynamic layout jitter and OCR noise augmentation were applied during training to remediate the severe layout fragility (FUNSD -65.4%, SROIE -23.5%). The resulting **Joint-Augmented** achieves a +18.9 pp improvement under worst-case layout stress while maintaining clean performance within tolerance. All original datasets and Phase 1-8 checkpoints remain 100% intact.

## 2. Research Questions

1. Does the image modality contribute functionally, given Gaussian blur caused negligible degradation in Phase 8?
2. Why does LayoutLMv3 collapse under coordinate jitter, and can dynamic jitter augmentation fix this?
3. How resilient are token predictions to OCR corruption, and does text noise augmentation help?
4. Can robustness improve without degrading clean test accuracy?
5. Which model configuration provides the best clean + robustness tradeoff?

## 3. Phase 8 Findings Under Investigation

- **FUNSD Collapse**: Layout BBox Jitter (+/-10%) degraded FUNSD Macro F1 by 65.4% (0.7179 -> 0.2487).
- **SROIE COMPANY Collapse**: COMPANY entity F1 dropped 44.8% (0.9049 -> 0.5000) under +/-10% jitter.
- **CORD Sensitivity**: CORD dropped 0.9733 -> 0.8209 (-15.7%) under +/-10% jitter.
- **Modality Asymmetry**: Image blur caused <=1.3% drop; ablation needed to distinguish robustness from redundancy.

## 4. Experimental Setup

- **Model**: `LayoutLMv3ForTokenClassification` (`microsoft/layoutlmv3-base`, 125M parameters)
- **Training**: 10 epochs, SROIE corrected (533 train / 93 val), batch=2, grad_accum=4, lr=5e-5, cosine schedule, fp16
- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU (6.4 GB VRAM)
- **Evaluation**: Strict entity-level IOB2 span F1, in-memory perturbations, seed=42

## 5. Baseline Verification

| Benchmark | Target Macro F1 | Measured | Diff | Status |
|---|:---:|:---:|:---:|:---:|
| **SROIE** | 0.8785 | **0.8784** | 0.0001 | PASS |
| **FUNSD** | 0.7179 | **0.7179** | 0.0000 | PASS |
| **CORD** | 0.9733 | **0.9733** | 0.0000 | PASS |

## 6. Phase 8 Failure Reproduction

Re-evaluation under identical protocol confirmed exact reproduction of Phase 8 degradations. Layout Jitter (+/-10%) remains the single worst condition across all benchmarks.

## 7. Multimodal Modality Ablation Study

| Dataset   | Condition       | Modalities            |   Precision |   Recall |   Macro F1 |   Micro F1 |
|:----------|:----------------|:----------------------|------------:|---------:|-----------:|-----------:|
| SROIE     | Full Multimodal | Text + Layout + Image |      0.8689 |   0.8889 |     0.8784 |     0.8788 |
| SROIE     | Text Only       | Text                  |      0.6382 |   0.4969 |     0.5297 |     0.5587 |
| SROIE     | Text + Layout   | Text + Layout         |      0.852  |   0.877  |     0.8646 |     0.8643 |
| SROIE     | Text + Image    | Text + Image          |      0.6648 |   0.492  |     0.5291 |     0.5655 |
| SROIE     | Layout + Image  | Layout + Image        |      0.1126 |   0.1377 |     0.1201 |     0.1239 |
| SROIE     | No-Image Model  | Text + Layout         |      0.852  |   0.877  |     0.8646 |     0.8643 |
| SROIE     | No-Layout Model | Text + Image          |      0.6648 |   0.492  |     0.5291 |     0.5655 |
| FUNSD     | Full Multimodal | Text + Layout + Image |      0.7804 |   0.8196 |     0.7179 |     0.7995 |
| FUNSD     | Text Only       | Text                  |      0.3103 |   0.1004 |     0.1015 |     0.1517 |
| FUNSD     | Text + Layout   | Text + Layout         |      0.7422 |   0.813  |     0.6808 |     0.776  |
| FUNSD     | Text + Image    | Text + Image          |      0.2155 |   0.0745 |     0.0784 |     0.1107 |
| FUNSD     | Layout + Image  | Layout + Image        |      0.257  |   0.2276 |     0.2215 |     0.2414 |
| FUNSD     | No-Image Model  | Text + Layout         |      0.7422 |   0.813  |     0.6808 |     0.776  |
| FUNSD     | No-Layout Model | Text + Image          |      0.2155 |   0.0745 |     0.0784 |     0.1107 |
| CORD      | Full Multimodal | Text + Layout + Image |      0.9748 |   0.9763 |     0.9733 |     0.9756 |
| CORD      | Text Only       | Text                  |      0.5733 |   0.4423 |     0.4779 |     0.4994 |
| CORD      | Text + Layout   | Text + Layout         |      0.9763 |   0.9763 |     0.9737 |     0.9763 |
| CORD      | Text + Image    | Text + Image          |      0.631  |   0.5134 |     0.5547 |     0.5661 |
| CORD      | Layout + Image  | Layout + Image        |      0.3381 |   0.1795 |     0.137  |     0.2345 |
| CORD      | No-Image Model  | Text + Layout         |      0.9763 |   0.9763 |     0.9737 |     0.9763 |
| CORD      | No-Layout Model | Text + Image          |      0.631  |   0.5134 |     0.5547 |     0.5661 |

### 7.1 Scientific Findings

**Image Modality Contribution** (Full vs No-Image on clean data):

- **SROIE**: 0.8784 -> 0.8646 (Delta=-0.0138)
- **FUNSD**: 0.7179 -> 0.6808 (Delta=-0.0371)
- **CORD**: 0.9733 -> 0.9737 (Delta=+0.0004)

**Critical finding**: Image blur robustness does NOT mean the image modality is redundant. The ablation confirms vision provides positive signal (especially FUNSD), but text + layout carry primary semantic information. Blur is tolerated because low-frequency spatial structure survives Gaussian blur.

**Layout Modality Contribution** (Full vs No-Layout on clean data):

- **SROIE**: 0.8784 -> 0.5291 (Delta=-0.3493)
- **FUNSD**: 0.7179 -> 0.0784 (Delta=-0.6395)
- **CORD**: 0.9733 -> 0.5547 (Delta=-0.4186)

Spatial 2D position is **indispensable** — removing layout causes catastrophic drops across all datasets.

## 8. Layout Robustness Experiments

**Root Cause**: LayoutLMv3 embeds absolute [x0,y0,x1,y1] coordinates (normalized 0-1000) directly into attention. Small jitter shifts disrupt the learned absolute-position alignment between entity blocks. **Intervention**: Dynamic multi-scale jitter augmentation (2%-8% per sample, p=0.6) forces the model to learn coordinate-invariant spatial attention patterns.

## 9. OCR Robustness Experiments

**Augmentation**: Character-level OCR confusion (O<->0, l<->1, S<->5, B<->8, Z<->2) at 8% rate, typographical noise at 4% rate, and token masking at 5% rate (per-word probability 0.6). Prevents subword fragmentation from propagating into span boundary errors.

## 10. Robustness-Aware Training Summary

| Model | Augmentation | Best Val Macro F1 | Checkpoint |
|---|---|:---:|---|
| Baseline | None | 0.8785 | `layoutlmv3/sroie/corrected/best/` |
| Layout-Augmented | Layout jitter [2%-8%], p=0.6 | 0.9113 | `phase9/sroie/layout_aug/best/` |
| OCR-Augmented | OCR confusion + masking | 0.8924 | `phase9/sroie/ocr_aug/best/` |
| Joint-Augmented | Layout jitter + OCR noise | 0.8976 | `phase9/sroie/joint_aug/best/` |

## 11. Complete 14-Condition Robustness Evaluation Matrix (SROIE)

| Model            | Condition                            | Type         | Severity       |   Precision |   Recall |   Macro F1 |   Delta F1 |   Rel Drop % |
|:-----------------|:-------------------------------------|:-------------|:---------------|------------:|---------:|-----------:|-----------:|-------------:|
| Baseline         | Clean Baseline                       | Clean        | nan            |      0.8689 |   0.8889 |     0.8784 |     0      |         0    |
| Baseline         | OCR Char Corruption (Low: 5%)        | OCR Noise    | Low (5%)       |      0.8682 |   0.8882 |     0.878  |    -0.0004 |         0.05 |
| Baseline         | OCR Char Corruption (Medium: 10%)    | OCR Noise    | Medium (10%)   |      0.8551 |   0.8742 |     0.8645 |    -0.0139 |         1.58 |
| Baseline         | OCR Char Corruption (High: 20%)      | OCR Noise    | High (20%)     |      0.8502 |   0.8644 |     0.8572 |    -0.0212 |         2.41 |
| Baseline         | Missing Text / Masking (Low: 5%)     | Missing Text | Low (5%)       |      0.8374 |   0.8491 |     0.8436 |    -0.0348 |         3.96 |
| Baseline         | Missing Text / Masking (Medium: 10%) | Missing Text | Medium (10%)   |      0.7972 |   0.7939 |     0.7964 |    -0.082  |         9.34 |
| Baseline         | Missing Text / Masking (High: 20%)   | Missing Text | High (20%)     |      0.7144 |   0.6974 |     0.7087 |    -0.1697 |        19.32 |
| Baseline         | Layout BBox Jitter (Low: ±2%)        | Layout Noise | Low (±2%)      |      0.8116 |   0.8281 |     0.8196 |    -0.0588 |         6.69 |
| Baseline         | Layout BBox Jitter (Medium: ±5%)     | Layout Noise | Medium (±5%)   |      0.7354 |   0.7652 |     0.7518 |    -0.1266 |        14.41 |
| Baseline         | Layout BBox Jitter (High: ±10%)      | Layout Noise | High (±10%)    |      0.651  |   0.6883 |     0.6716 |    -0.2068 |        23.54 |
| Baseline         | Image Blur (Low: σ=1.0)              | Image Blur   | Low (σ=1.0)    |      0.8689 |   0.8889 |     0.8784 |     0      |         0    |
| Baseline         | Image Blur (Medium: σ=2.0)           | Image Blur   | Medium (σ=2.0) |      0.8689 |   0.8889 |     0.8784 |     0      |         0    |
| Baseline         | Image Blur (High: σ=3.5)             | Image Blur   | High (σ=3.5)   |      0.8672 |   0.8896 |     0.878  |    -0.0004 |         0.05 |
| Baseline         | Composite Multi-Modal Stress         | Composite    | Medium         |      0.6993 |   0.7282 |     0.7153 |    -0.1631 |        18.57 |
| Layout-Augmented | Clean Baseline                       | Clean        | nan            |      0.873  |   0.8987 |     0.8856 |     0      |         0    |
| Layout-Augmented | OCR Char Corruption (Low: 5%)        | OCR Noise    | Low (5%)       |      0.8708 |   0.8903 |     0.8802 |    -0.0054 |         0.61 |
| Layout-Augmented | OCR Char Corruption (Medium: 10%)    | OCR Noise    | Medium (10%)   |      0.8591 |   0.8777 |     0.8679 |    -0.0177 |         2    |
| Layout-Augmented | OCR Char Corruption (High: 20%)      | OCR Noise    | High (20%)     |      0.8547 |   0.8714 |     0.8629 |    -0.0227 |         2.56 |
| Layout-Augmented | Missing Text / Masking (Low: 5%)     | Missing Text | Low (5%)       |      0.8337 |   0.8407 |     0.8371 |    -0.0485 |         5.48 |
| Layout-Augmented | Missing Text / Masking (Medium: 10%) | Missing Text | Medium (10%)   |      0.7928 |   0.789  |     0.7916 |    -0.094  |        10.61 |
| Layout-Augmented | Missing Text / Masking (High: 20%)   | Missing Text | High (20%)     |      0.7034 |   0.6876 |     0.6979 |    -0.1877 |        21.19 |
| Layout-Augmented | Layout BBox Jitter (Low: ±2%)        | Layout Noise | Low (±2%)      |      0.8621 |   0.891  |     0.8761 |    -0.0095 |         1.07 |
| Layout-Augmented | Layout BBox Jitter (Medium: ±5%)     | Layout Noise | Medium (±5%)   |      0.8524 |   0.8882 |     0.8699 |    -0.0157 |         1.77 |
| Layout-Augmented | Layout BBox Jitter (High: ±10%)      | Layout Noise | High (±10%)    |      0.8435 |   0.8777 |     0.8602 |    -0.0254 |         2.87 |
| Layout-Augmented | Image Blur (Low: σ=1.0)              | Image Blur   | Low (σ=1.0)    |      0.873  |   0.898  |     0.8852 |    -0.0004 |         0.05 |
| Layout-Augmented | Image Blur (Medium: σ=2.0)           | Image Blur   | Medium (σ=2.0) |      0.8717 |   0.8973 |     0.8842 |    -0.0014 |         0.16 |
| Layout-Augmented | Image Blur (High: σ=3.5)             | Image Blur   | High (σ=3.5)   |      0.8692 |   0.8959 |     0.8823 |    -0.0033 |         0.37 |
| Layout-Augmented | Composite Multi-Modal Stress         | Composite    | Medium         |      0.8395 |   0.8735 |     0.8561 |    -0.0295 |         3.33 |
| OCR-Augmented    | Clean Baseline                       | Clean        | nan            |      0.8711 |   0.8882 |     0.8794 |     0      |         0    |
| OCR-Augmented    | OCR Char Corruption (Low: 5%)        | OCR Noise    | Low (5%)       |      0.8689 |   0.8896 |     0.8791 |    -0.0003 |         0.03 |
| OCR-Augmented    | OCR Char Corruption (Medium: 10%)    | OCR Noise    | Medium (10%)   |      0.8634 |   0.8875 |     0.8753 |    -0.0041 |         0.47 |
| OCR-Augmented    | OCR Char Corruption (High: 20%)      | OCR Noise    | High (20%)     |      0.851  |   0.8819 |     0.8665 |    -0.0129 |         1.47 |
| OCR-Augmented    | Missing Text / Masking (Low: 5%)     | Missing Text | Low (5%)       |      0.8578 |   0.8812 |     0.8693 |    -0.0101 |         1.15 |
| OCR-Augmented    | Missing Text / Masking (Medium: 10%) | Missing Text | Medium (10%)   |      0.8508 |   0.8728 |     0.8617 |    -0.0177 |         2.01 |
| OCR-Augmented    | Missing Text / Masking (High: 20%)   | Missing Text | High (20%)     |      0.8088 |   0.8246 |     0.8162 |    -0.0632 |         7.19 |
| OCR-Augmented    | Layout BBox Jitter (Low: ±2%)        | Layout Noise | Low (±2%)      |      0.8245 |   0.8505 |     0.8373 |    -0.0421 |         4.79 |
| OCR-Augmented    | Layout BBox Jitter (Medium: ±5%)     | Layout Noise | Medium (±5%)   |      0.7465 |   0.782  |     0.7645 |    -0.1149 |        13.07 |
| OCR-Augmented    | Layout BBox Jitter (High: ±10%)      | Layout Noise | High (±10%)    |      0.6978 |   0.7261 |     0.7112 |    -0.1682 |        19.13 |
| OCR-Augmented    | Image Blur (Low: σ=1.0)              | Image Blur   | Low (σ=1.0)    |      0.8712 |   0.8889 |     0.8798 |     0.0004 |         0.05 |
| OCR-Augmented    | Image Blur (Medium: σ=2.0)           | Image Blur   | Medium (σ=2.0) |      0.8711 |   0.8882 |     0.8794 |     0      |         0    |
| OCR-Augmented    | Image Blur (High: σ=3.5)             | Image Blur   | High (σ=3.5)   |      0.8698 |   0.8868 |     0.878  |    -0.0014 |         0.16 |
| OCR-Augmented    | Composite Multi-Modal Stress         | Composite    | Medium         |      0.7293 |   0.7645 |     0.7475 |    -0.1319 |        15    |
| Joint-Augmented  | Clean Baseline                       | Clean        | nan            |      0.8694 |   0.898  |     0.8837 |     0      |         0    |
| Joint-Augmented  | OCR Char Corruption (Low: 5%)        | OCR Noise    | Low (5%)       |      0.8723 |   0.8973 |     0.8847 |     0.001  |         0.11 |
| Joint-Augmented  | OCR Char Corruption (Medium: 10%)    | OCR Noise    | Medium (10%)   |      0.8659 |   0.898  |     0.8821 |    -0.0016 |         0.18 |
| Joint-Augmented  | OCR Char Corruption (High: 20%)      | OCR Noise    | High (20%)     |      0.8659 |   0.8931 |     0.8795 |    -0.0042 |         0.48 |
| Joint-Augmented  | Missing Text / Masking (Low: 5%)     | Missing Text | Low (5%)       |      0.8538 |   0.8854 |     0.8696 |    -0.0141 |         1.6  |
| Joint-Augmented  | Missing Text / Masking (Medium: 10%) | Missing Text | Medium (10%)   |      0.8444 |   0.8763 |     0.8604 |    -0.0233 |         2.64 |
| Joint-Augmented  | Missing Text / Masking (High: 20%)   | Missing Text | High (20%)     |      0.7945 |   0.8239 |     0.81   |    -0.0737 |         8.34 |
| Joint-Augmented  | Layout BBox Jitter (Low: ±2%)        | Layout Noise | Low (±2%)      |      0.8629 |   0.8882 |     0.8752 |    -0.0085 |         0.96 |
| Joint-Augmented  | Layout BBox Jitter (Medium: ±5%)     | Layout Noise | Medium (±5%)   |      0.8647 |   0.8889 |     0.8766 |    -0.0071 |         0.8  |
| Joint-Augmented  | Layout BBox Jitter (High: ±10%)      | Layout Noise | High (±10%)    |      0.85   |   0.8791 |     0.8643 |    -0.0194 |         2.2  |
| Joint-Augmented  | Image Blur (Low: σ=1.0)              | Image Blur   | Low (σ=1.0)    |      0.8687 |   0.8966 |     0.8826 |    -0.0011 |         0.12 |
| Joint-Augmented  | Image Blur (Medium: σ=2.0)           | Image Blur   | Medium (σ=2.0) |      0.8677 |   0.898  |     0.8829 |    -0.0008 |         0.09 |
| Joint-Augmented  | Image Blur (High: σ=3.5)             | Image Blur   | High (σ=3.5)   |      0.8666 |   0.8987 |     0.8828 |    -0.0009 |         0.1  |
| Joint-Augmented  | Composite Multi-Modal Stress         | Composite    | Medium         |      0.8544 |   0.8896 |     0.8719 |    -0.0118 |         1.34 |

## 12. Statistical Significance Analysis

**Test**: Paired Wilcoxon Signed-Rank Test — Baseline vs Joint-Augmented on 361 SROIE receipts under Layout Jitter (+/-10%).

| Test | Result | Interpretation |
|---|---|---|
| Wilcoxon Signed-Rank | W=13285, p=3.2384e-37 | Highly significant (alpha=0.001) |
| Cohen's d | d=0.919 | Large effect size |
| 95% CI (Delta F1) | [0.1674, 0.2098] | Entirely positive |
| Mean F1 Delta | +0.1886 (+18.86 pp) | Meaningful improvement |

## 13. Clean vs. Robustness Tradeoff & Model Selection

**Selection formula** (predefined before any training):

```
Score = 0.40 * F1_clean + 0.30 * F1_avg_stress + 0.30 * F1_worst_stress
        - 2.0 * max(0, F1_clean_baseline - F1_clean)
```

| Model            |   Clean F1 |   Delta Baseline |   Avg Stress F1 |   Worst F1 | Worst Condition                    |   Penalty |   Score |
|:-----------------|-----------:|-----------------:|----------------:|-----------:|:-----------------------------------|----------:|--------:|
| Joint-Augmented  |     0.8837 |           0.0053 |          0.871  |     0.81   | Missing Text / Masking (High: 20%) |         0 |  0.8578 |
| Layout-Augmented |     0.8856 |           0.0072 |          0.8501 |     0.6979 | Missing Text / Masking (High: 20%) |         0 |  0.8186 |
| OCR-Augmented    |     0.8794 |           0.001  |          0.8358 |     0.7112 | Layout BBox Jitter (High: ±10%)    |         0 |  0.8159 |
| Baseline         |     0.8784 |           0      |          0.8109 |     0.6716 | Layout BBox Jitter (High: ±10%)    |         0 |  0.7961 |

### Best Model: **Joint-Augmented**

Highest composite utility score. Successfully repaired layout jitter vulnerability while keeping clean SROIE performance within tolerance.

## 14. SROIE Per-Entity Analysis Under Layout Jitter (+/-10%)

| Model            | Entity   |   F1 (Jitter +/-10%) |   Delta F1 |
|:-----------------|:---------|---------------------:|-----------:|
| Baseline         | ADDRESS  |               0.6235 |    -0.2199 |
| Baseline         | COMPANY  |               0.5    |    -0.4049 |
| Baseline         | DATE     |               0.9667 |    -0.0263 |
| Baseline         | TOTAL    |               0.5962 |    -0.1763 |
| Layout-Augmented | ADDRESS  |               0.8197 |    -0.037  |
| Layout-Augmented | COMPANY  |               0.8824 |    -0.0245 |
| Layout-Augmented | DATE     |               0.9889 |    -0.0055 |
| Layout-Augmented | TOTAL    |               0.75   |    -0.0344 |
| OCR-Augmented    | ADDRESS  |               0.6384 |    -0.1995 |
| OCR-Augmented    | COMPANY  |               0.6446 |    -0.26   |
| OCR-Augmented    | DATE     |               0.9737 |    -0.0165 |
| OCR-Augmented    | TOTAL    |               0.5882 |    -0.1966 |
| Joint-Augmented  | ADDRESS  |               0.8251 |    -0.0289 |
| Joint-Augmented  | COMPANY  |               0.8862 |    -0.0165 |
| Joint-Augmented  | DATE     |               0.9916 |    -0.0014 |
| Joint-Augmented  | TOTAL    |               0.7542 |    -0.0309 |

## 15. Limitations

1. Robustness augmentation applied only to SROIE; FUNSD/CORD models not retrained in Phase 9.
2. Perturbations are simulated approximations, not physical scanner artifacts.
3. All models remain closed-set classifiers constrained to predefined label schemas.
4. Single training seed (42); multi-seed confidence intervals not reported.

## 16. Conclusions

1. **Layout is indispensable**: Removing 2D coordinates causes catastrophic drops across all benchmarks.
2. **Image is active but secondary**: Vision provides positive signal (especially FUNSD), but blur robustness does not imply redundancy.
3. **Layout fragility = training distribution narrowness**: Dynamic jitter augmentation closes this gap without architectural changes.
4. **Joint augmentation dominates**: Combined layout + text robustness training is complementary.
5. **No clean performance regression**: All Phase 9 models stay within predefined tolerance.

## 17. Reproducibility Checklist

- [x] Baseline reproduced within +/-0.0020 tolerance (SROIE, FUNSD, CORD)
- [x] Phase 8 failure modes reproduced
- [x] Multimodal ablation completed (7 conditions x 3 datasets)
- [x] Robustness-aware models trained (3 checkpoints in models/checkpoints/phase9/)
- [x] 14-condition stress evaluation completed across 4 candidate models
- [x] Statistical significance confirmed (Wilcoxon p=3.24e-37, Cohen's d=0.919)
- [x] Original datasets unmodified
- [x] Original Phase 1-8 checkpoints unmodified
- [x] Automated test suite passing
- [x] All Phase 6/7/8 artifacts unchanged

## 18. Phase 9 Artifact Index

| Artifact | Path |
|---|---|
| Ablation CSV | `reports/phase9_ablation_results.csv` |
| Robustness Matrix CSV | `reports/phase9_robustness_results.csv` |
| Entity Results CSV | `reports/phase9_entity_results.csv` |
| Model Comparison CSV | `reports/phase9_model_comparison.csv` |
| Ablation Contributions Plot | `reports/phase9/visualizations/ablation_modality_contributions.png` |
| Layout Jitter Sensitivity Plot | `reports/phase9/visualizations/layout_jitter_sensitivity_comparison.png` |
| Clean vs Robustness Tradeoff | `reports/phase9/visualizations/clean_vs_robustness_tradeoff.png` |
| SROIE Entity Improvement Plot | `reports/phase9/visualizations/sroie_entity_improvement.png` |
| Repaired Sample Visualizations | `reports/phase9/visualizations/repaired_sample_*.png` |
| Layout-Augmented Checkpoint | `models/checkpoints/phase9/sroie/layout_aug/best/` |
| OCR-Augmented Checkpoint | `models/checkpoints/phase9/sroie/ocr_aug/best/` |
| Joint-Augmented Checkpoint | `models/checkpoints/phase9/sroie/joint_aug/best/` |
| Phase 9 Final Report | `reports/phase9_robustness_improvement_report.md` |
