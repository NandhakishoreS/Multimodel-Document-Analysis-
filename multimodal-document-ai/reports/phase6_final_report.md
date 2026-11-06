# Phase 6 Final Report: Controlled Evaluation, Unified Metrics & Ablation Study

**Generated**: 2026-09-09  

**Phase**: 6  

**Project**: Multimodal Transformer-Based Intelligent Document Understanding


---

## 1. Objective

Phase 6 establishes a scientifically consistent evaluation framework for comparing three model types (Text-Only, Text+Layout, LayoutLMv3) across three datasets (SROIE, FUNSD, CORD). The key goal is to transparently document metric incompatibilities between Phase 3 baselines and Phase 4/5 LayoutLMv3 models, provide a unified span-level F1 metric where feasible, conduct a controlled LayoutLMv3 modality ablation study, and preserve all historical results for reproducibility without silently altering them.

## 2. Existing Evaluation Problems

| Problem | Impact |
|---|---|
| Baselines use token-level sklearn F1; LayoutLMv3 uses entity-span seqeval F1 | Numbers are not numerically comparable |
| SROIE baselines evaluated on old preprocessing; LayoutLMv3 on corrected data | Different input distributions |
| CORD uses 3-class simplified task with o_ratio=0.0 | Not equivalent to full-document CORD KIE |
| Baseline B/I labels treated independently; LayoutLMv3 groups them into spans | Macro F1 computed over different unit granularities |

## 3. Unified Evaluation Methodology

Phase 6 defines a **strict entity-level IOB2 span F1** as the primary unified metric. A predicted span is correct if and only if both the entity type AND both span boundaries (start and end token index) match the gold span exactly. This is implemented in `evaluation/phase6_unified_eval.py::compute_strict_span_f1`. Special tokens and padding are excluded before evaluation. O tokens are never counted in any span.

This metric is applied to LayoutLMv3 predictions (via the existing seqeval-compatible pipeline). For baselines, the historical token-level sklearn results are preserved and clearly marked as incompatible. Baseline per-document predictions were not saved in Phase 3 and baselines are not re-run on corrected SROIE data to preserve Phase 3 integrity.

## 4. Dataset-Specific Label Mappings

| Dataset | N Labels | Entity Types | O Ratio (Test) | Notes |
|---|:---:|---|:---:|---|
| SROIE (corrected) | 9 | COMPANY, DATE, ADDRESS, TOTAL | ~7.5% entity | IOB2 BIO scheme |
| FUNSD | 7 | HEADER, QUESTION, ANSWER | ~29.1% O | IOB2 BIO scheme |
| CORD | 13 (trained) | MENU, SUBTOTAL, TOTAL *(evaluated)* | 0.0% | Simplified 3-class; VOID/PAYMENT/ETC have zero gold spans |

## 5. Baseline Comparison

> **Warning**: Baseline Macro F1 values are token-level sklearn metrics. LayoutLMv3 Macro F1 values are entity-span seqeval metrics. These are NOT directly numerically comparable.


| Dataset | Method | Eval Type | Precision | Recall | Macro F1 | Micro F1 | Notes |
|---|---|---|:---:|:---:|:---:|:---:|---|

| SROIE | Text-Only Baseline (TF-IDF + LR) | token/sklearn | 0.2419 | 0.3097 | **0.2616** | 0.9194 | HISTORICAL token-level sklearn F1 on OLD SROIE preprocessing. NOT directly compa... |
| SROIE | Text+Layout Baseline (Spatial + LR) | token/sklearn | 0.2456 | 0.3134 | **0.2660** | 0.9275 | HISTORICAL token-level sklearn F1 on OLD SROIE preprocessing. NOT directly compa... |
| FUNSD | Text-Only Baseline (TF-IDF + LR) | token/sklearn | 0.3474 | 0.4186 | **0.3512** | 0.4019 | Historical token-level sklearn F1. NOT directly comparable to LayoutLMv3 entity-... |
| FUNSD | Text+Layout Baseline (Spatial + LR) | token/sklearn | 0.4840 | 0.5464 | **0.4996** | 0.5405 | Historical token-level sklearn F1. NOT directly comparable to LayoutLMv3 entity-... |
| CORD | Text-Only Baseline (TF-IDF + LR) | token/sklearn | 0.3222 | 0.3399 | **0.3267** | 0.6842 | Historical token-level sklearn F1 on 3-class simplified CORD. NOT directly compa... |
| CORD | Text+Layout Baseline (Spatial + LR) | token/sklearn | 0.3694 | 0.3886 | **0.3767** | 0.8107 | Historical token-level sklearn F1 on 3-class simplified CORD. NOT directly compa... |
| SROIE | LayoutLMv3-base (Corrected Preprocessing) | entity-span/seqeval | 0.8689 | 0.8889 | **0.8785** | 0.8788 | Corrected SROIE preprocessing. Best checkpoint: Epoch 9. Val Macro F1=0.9090. De... |
| FUNSD | LayoutLMv3-base (Full Fine-Tuning) | entity-span/seqeval | 0.7804 | 0.8196 | **0.7179** | 0.7995 | Best checkpoint: Epoch 11. Val Macro F1=0.6946. FUNSD has meaningful I-tag suppo... |
| CORD | LayoutLMv3-base (Full Fine-Tuning) | entity-span/seqeval | 0.9748 | 0.9763 | **0.9733** | 0.9756 | CORD TASK-FORMULATION CAVEAT: Fine-grained CORD labels collapsed to 3 classes (M... |

## 6. LayoutLMv3 Comparison

The following results are all entity-span seqeval F1 and are internally comparable:

| Dataset | Macro F1 | Micro F1 | Precision | Recall |
|---|:---:|:---:|:---:|:---:|
| SROIE (Corrected) | **0.8785** | 0.8788 | 0.8689 | 0.8889 |
| FUNSD | **0.7179** | 0.7995 | 0.7804 | 0.8196 |
| CORD *(simplified task)* | **0.9733** | 0.9756 | 0.9748 | 0.9763 |

## 7. Ablation Results

| dataset | condition | modalities_active | precision | recall | macro_f1 | micro_f1 | delta_vs_full | n_gold_spans |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SROIE | Full (text + layout + image) | text+layout+image | 0.8689 | 0.8889 | 0.8784 | 0.8788 | 0.0 | 1431 |
| SROIE | No-Image (text + layout only) | text+layout | 0.852 | 0.877 | 0.8646 | 0.8643 | -0.0138 | 1431 |
| SROIE | No-Layout (text + image only) | text+image | 0.6648 | 0.492 | 0.5291 | 0.5655 | -0.3493 | 1431 |
| SROIE | Text-Approx (text only) | text-only-approx | 0.6382 | 0.4969 | 0.5297 | 0.5587 | -0.3487 | 1431 |
| FUNSD | Full (text + layout + image) | text+layout+image | 0.7804 | 0.8196 | 0.7179 | 0.7995 | 0.0 | 1973 |
| FUNSD | No-Image (text + layout only) | text+layout | 0.7422 | 0.813 | 0.6808 | 0.776 | -0.0371 | 1973 |
| FUNSD | No-Layout (text + image only) | text+image | 0.2155 | 0.0745 | 0.0784 | 0.1107 | -0.6395 | 1973 |
| FUNSD | Text-Approx (text only) | text-only-approx | 0.3103 | 0.1004 | 0.1015 | 0.1517 | -0.6164 | 1973 |
| CORD | Full (text + layout + image) | text+layout+image | 0.9748 | 0.9763 | 0.9733 | 0.9756 | 0.0 | 1309 |
| CORD | No-Image (text + layout only) | text+layout | 0.9763 | 0.9763 | 0.9737 | 0.9763 | 0.0004 | 1309 |
| CORD | No-Layout (text + image only) | text+image | 0.631 | 0.5134 | 0.5547 | 0.5661 | -0.4186 | 1309 |
| CORD | Text-Approx (text only) | text-only-approx | 0.5733 | 0.4423 | 0.4779 | 0.4994 | -0.4954 | 1309 |


See `reports/phase6_ablation_report.md` for full analysis.

## 8. CORD Task-Formulation Caveat

> **IMPORTANT**: CORD evaluation in this project uses a simplified task formulation:
> - Fine-grained CORD labels are collapsed into 3 main classes: MENU, SUBTOTAL, TOTAL.
> - Only `valid_line` tokens are evaluated.
> - The evaluated test set has effectively **zero O/background tokens** (o_ratio = 0.0).
> - This is a simplified 3-class closed-set line classification/KIE formulation.
> - **CORD results in this project are NOT equivalent to published full-document CORD KIE benchmark results.**
> - Do not use the `0.9733` Macro F1 figure to claim state-of-the-art performance.

## 9. SROIE Corrected Result Explanation

Phase 5 identified and corrected multiple preprocessing errors in the original SROIE pipeline:
- Multi-word tokens were not decomposed into individual words with interpolated bounding boxes.
- TOTAL matching was overly greedy (substring matching instead of exact numeric matching).
- Multi-line ADDRESS spans were not correctly aligned.

The corrected model (Epoch 9 checkpoint) achieves:
- **Validation Macro F1**: 0.9090
- **Test Macro F1**: 0.8785 (full 361 samples)
- **Test Macro F1**: 0.8789 (deduplicated 353 samples)
- Difference from duplicate sensitivity: +0.0004 (negligible)

The old SROIE checkpoint (Macro F1 = 0.8828) is preserved as a historical reference.

## 10. Limitations

1. **Single training run**: No repeated experiments with different seeds. Confidence intervals are not reported.
2. **Metric incompatibility**: Baseline and LayoutLMv3 metrics cannot be combined in a unified table without caveats.
3. **SROIE baselines not re-run**: Phase 3 baselines were trained on old preprocessing. Re-running would require retraining and risks disturbing Phase 3 results.
4. **Ablation via input-zeroing**: Modality ablation zeroes input tensors rather than removing modality pathways from the architecture. Results show modality sensitivity of the fine-tuned model, not generalizable architectural conclusions.
5. **CORD task formulation**: The 3-class simplified CORD pipeline limits comparability with published benchmarks.
6. **No CORD baseline re-run on seqeval**: CORD baseline predictions per-document not saved in Phase 3, so unified span F1 on baselines is Not evaluated.

## 11. Conclusions

Despite metric incompatibilities, the direction of improvement from LayoutLMv3 over baselines is unambiguous across all three datasets. LayoutLMv3's multimodal pretraining (text + layout + image) consistently outperforms both text-only and text+layout baselines.

Key findings:
- **SROIE (corrected)**: LayoutLMv3 Macro F1 = 0.8785 (entity-span). Baseline Macro F1 = 0.26 (token-level) — incompatible but directionally clear.
- **FUNSD**: LayoutLMv3 Macro F1 = 0.7179. Baseline = 0.35 (token). Layout features help even in the baseline (+14.8 pp from text-only to text+layout).
- **CORD**: LayoutLMv3 Macro F1 = 0.9733 on simplified 3-class task. Not comparable to full-document CORD benchmarks.
- **Ablation**: Removing image (pixel_values=0) and/or layout (bbox=0) degrades performance, confirming LayoutLMv3's reliance on multimodal signals. See ablation report for details.

## 12. Generated Artifact Paths

| Artifact | Path |
|---|---|
| Core eval module | `evaluation/phase6_unified_eval.py` |
| Main runner | `scripts/run_phase6_evaluation.py` |
| Ablation runner | `scripts/run_phase6_ablation.py` |
| Audit report | `reports/phase6_evaluation_protocol_audit.md` |
| Unified results CSV | `reports/phase6_unified_results.csv` |
| Unified results Markdown | `reports/phase6_unified_results.md` |
| Ablation results CSV | `reports/phase6_ablation_results.csv` |
| Ablation report | `reports/phase6_ablation_report.md` |
| Final comparison plot | `reports/phase6_final_comparison.png` |
| Ablation comparison plot | `reports/phase6_ablation_comparison.png` |
