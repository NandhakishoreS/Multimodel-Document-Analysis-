# Phase 6: LayoutLMv3 Modality Ablation Report

## 1. Ablation Methodology

> **IMPORTANT**: This is an **input-zeroing sensitivity study**, NOT an architectural ablation.
> LayoutLMv3 deeply fuses text, layout, and image modalities in the first transformer layer.
> Removing a modality cleanly would require retraining from scratch.
> Instead, we zero the relevant input tensor and measure the performance change.
> This shows how sensitive the model's predictions are to each input signal,
> but does NOT yield the same conclusions as training separate single-modality models.


## 2. Ablation Conditions

| Condition | pixel_values | bbox | Modalities Active |
|---|:---:|:---:|---|
| Full (text + layout + image) | Normal | Normal | All three input modalities active. pixel_values and bbox as provided by the data... |
| No-Image (text + layout only) | Zeroed | Normal | pixel_values zeroed (all zeros). bbox unchanged. Removes visual patch signal. Th... |
| No-Layout (text + image only) | Normal | Zeroed | bbox zeroed to (0,0,0,0) for all tokens. pixel_values unchanged. Removes spatial... |
| Text-Approx (text only) | Zeroed | Zeroed | Both pixel_values and bbox zeroed. Approximates text-only input within the Layou... |

## 3. Results by Dataset

### CORD

| Condition | Precision | Recall | Macro F1 | Micro F1 | Delta vs Full | Gold Spans |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Full (text + layout + image) | 0.9748 | 0.9763 | **0.9733** | 0.9756 | +0.0000 | 1309 |
| No-Image (text + layout only) | 0.9763 | 0.9763 | 0.9737 | 0.9763 | +0.0004 | 1309 |
| No-Layout (text + image only) | 0.6310 | 0.5134 | 0.5547 | 0.5661 | -0.4186 | 1309 |
| Text-Approx (text only) | 0.5733 | 0.4423 | 0.4779 | 0.4994 | -0.4954 | 1309 |

> **CORD Caveat**: CORD ablation uses the simplified 3-class task formulation. Results are not comparable to full-document CORD KIE benchmarks.

### FUNSD

| Condition | Precision | Recall | Macro F1 | Micro F1 | Delta vs Full | Gold Spans |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Full (text + layout + image) | 0.7804 | 0.8196 | **0.7179** | 0.7995 | +0.0000 | 1973 |
| No-Image (text + layout only) | 0.7422 | 0.8130 | 0.6808 | 0.7760 | -0.0371 | 1973 |
| No-Layout (text + image only) | 0.2155 | 0.0745 | 0.0784 | 0.1107 | -0.6395 | 1973 |
| Text-Approx (text only) | 0.3103 | 0.1004 | 0.1015 | 0.1517 | -0.6164 | 1973 |

### SROIE

| Condition | Precision | Recall | Macro F1 | Micro F1 | Delta vs Full | Gold Spans |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Full (text + layout + image) | 0.8689 | 0.8889 | **0.8784** | 0.8788 | +0.0000 | 1431 |
| No-Image (text + layout only) | 0.8520 | 0.8770 | 0.8646 | 0.8643 | -0.0138 | 1431 |
| No-Layout (text + image only) | 0.6648 | 0.4920 | 0.5291 | 0.5655 | -0.3493 | 1431 |
| Text-Approx (text only) | 0.6382 | 0.4969 | 0.5297 | 0.5587 | -0.3487 | 1431 |

## 4. Interpretation

- A large negative Delta vs Full indicates strong reliance on that modality.
- A small Delta may indicate robustness OR insensitivity to that zeroing strategy.
- These results apply to **this specific fine-tuned model** only.
- Do NOT generalize: a different training setup may produce different sensitivities.
- The Text-Approx condition is the most aggressive — both layout and image removed.
  The model was never trained in this regime, so degradation is expected and informative.

## 5. Limitations

- Zeroing pixel_values is an out-of-distribution perturbation (model saw real images during training).
- Zeroing bbox to (0,0,0,0) is also OOD (all tokens appear at same spatial location).
- A properly controlled ablation would require training separate models per condition.
- No confidence intervals — single-run results.
