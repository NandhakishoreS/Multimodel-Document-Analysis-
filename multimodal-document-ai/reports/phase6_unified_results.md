# Phase 6: Unified Evaluation Results

> **Important**: Baselines use token-level sklearn F1 (Phase 3). LayoutLMv3 uses entity-span seqeval F1 (Phase 4/5). These are NOT directly comparable. See `phase6_evaluation_protocol_audit.md` for full details.

> **CORD Caveat**: CORD results use a simplified 3-class closed-set formulation. NOT comparable to published full-document CORD benchmark results.


## SROIE

| Method | Eval Type | Precision | Recall | Macro F1 | Micro F1 | Comparable | Notes |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| Text-Only Baseline (TF-IDF + LR) | token/sklearn | 0.2419 | 0.3097 | **0.2616** | 0.9194 | ✗ (incompatible) | HISTORICAL token-level sklearn F1 on OLD SROIE preprocessing. NOT directly comparable to corrected LayoutLMv3 (different... |
| Text+Layout Baseline (Spatial + LR) | token/sklearn | 0.2456 | 0.3134 | **0.2660** | 0.9275 | ✗ (incompatible) | HISTORICAL token-level sklearn F1 on OLD SROIE preprocessing. NOT directly comparable to corrected LayoutLMv3 (different... |
| LayoutLMv3-base (Corrected Preprocessing) | entity-span/seqeval | 0.8689 | 0.8889 | **0.8785** | 0.8788 | ✓ | Corrected SROIE preprocessing. Best checkpoint: Epoch 9. Val Macro F1=0.9090. Deduplicated test (353 samples) Macro F1=0... |
| LayoutLMv3-base (Old Preprocessing — historical) | entity-span/seqeval | 0.8792 | 0.8878 | **0.8828** | 0.8835 | ✓ | HISTORICAL — old (uncorrected) SROIE preprocessing. Included for reproducibility only. Primary result is corrected check... |

### LayoutLMv3-base (Corrected Preprocessing) — Per-Entity Breakdown

| Entity | Precision | Recall | F1 | Gold Spans |
|---|:---:|:---:|:---:|:---:|
| ADDRESS | 0.8253 | 0.8624 | 0.8434 | 356 |
| COMPANY | 0.8786 | 0.9328 | 0.9049 | 357 |
| DATE | 0.9862 | 1.0000 | 0.9930 | 357 |
| TOTAL | 0.7835 | 0.7618 | 0.7725 | 361 |

### LayoutLMv3-base (Old Preprocessing — historical) — Per-Entity Breakdown

| Entity | Precision | Recall | F1 | Gold Spans |
|---|:---:|:---:|:---:|:---:|
| ADDRESS | 0.9245 | 0.9528 | 0.9384 | 360 |
| COMPANY | 0.9205 | 0.9465 | 0.9333 | 355 |
| DATE | 0.9612 | 0.9666 | 0.9639 | 359 |
| TOTAL | 0.7045 | 0.6870 | 0.6957 | 361 |

## FUNSD

| Method | Eval Type | Precision | Recall | Macro F1 | Micro F1 | Comparable | Notes |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| Text-Only Baseline (TF-IDF + LR) | token/sklearn | 0.3474 | 0.4186 | **0.3512** | 0.4019 | ✗ (incompatible) | Historical token-level sklearn F1. NOT directly comparable to LayoutLMv3 entity-span F1. Same test split (50 docs). Dire... |
| Text+Layout Baseline (Spatial + LR) | token/sklearn | 0.4840 | 0.5464 | **0.4996** | 0.5405 | ✗ (incompatible) | Historical token-level sklearn F1. NOT directly comparable to LayoutLMv3 entity-span F1. Same test split (50 docs). Dire... |
| LayoutLMv3-base (Full Fine-Tuning) | entity-span/seqeval | 0.7804 | 0.8196 | **0.7179** | 0.7995 | ✓ | Best checkpoint: Epoch 11. Val Macro F1=0.6946. FUNSD has meaningful I-tag support (multi-word spans common). 29.06% O-t... |

### LayoutLMv3-base (Full Fine-Tuning) — Per-Entity Breakdown

| Entity | Precision | Recall | F1 | Gold Spans |
|---|:---:|:---:|:---:|:---:|
| ANSWER | 0.7571 | 0.8286 | 0.7912 | 805 |
| HEADER | 0.5120 | 0.5378 | 0.5246 | 119 |
| QUESTION | 0.8311 | 0.8446 | 0.8378 | 1049 |

## CORD

| Method | Eval Type | Precision | Recall | Macro F1 | Micro F1 | Comparable | Notes |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| Text-Only Baseline (TF-IDF + LR) | token/sklearn | 0.3222 | 0.3399 | **0.3267** | 0.6842 | ✗ (incompatible) | Historical token-level sklearn F1 on 3-class simplified CORD. NOT directly comparable to LayoutLMv3 entity-span F1. CORD... |
| Text+Layout Baseline (Spatial + LR) | token/sklearn | 0.3694 | 0.3886 | **0.3767** | 0.8107 | ✗ (incompatible) | Historical token-level sklearn F1 on 3-class simplified CORD. NOT directly comparable to LayoutLMv3 entity-span F1. CORD... |
| LayoutLMv3-base (Full Fine-Tuning) | entity-span/seqeval | 0.9748 | 0.9763 | **0.9733** | 0.9756 | ✓ | CORD TASK-FORMULATION CAVEAT: Fine-grained CORD labels collapsed to 3 classes (MENU/SUBTOTAL/TOTAL). Only valid_line tok... |

### LayoutLMv3-base (Full Fine-Tuning) — Per-Entity Breakdown

| Entity | Precision | Recall | F1 | Gold Spans |
|---|:---:|:---:|:---:|:---:|
| MENU | 0.9775 | 0.9786 | 0.9780 | 887 |
| SUBTOTAL | 0.9530 | 0.9930 | 0.9726 | 143 |
| TOTAL | 0.9781 | 0.9606 | 0.9693 | 279 |

---

## CORD Task-Formulation Caveat

CORD TASK-FORMULATION CAVEAT: Fine-grained CORD labels collapsed to 3 classes (MENU/SUBTOTAL/TOTAL). Only valid_line tokens evaluated. O-token ratio = 0.0 (effectively no background tokens in evaluated test set). This is a simplified 3-class closed-set line classification formulation, NOT equivalent to full-document CORD KIE evaluation. Results are NOT directly comparable to published full-document CORD benchmark numbers.

