# Phase 3 — Baseline Models and Evaluation

## Overview

Phase 3 establishes competitive baseline systems before deploying the multimodal transformer. Three complementary baselines are trained and evaluated, each using progressively richer input modalities.

---

## Baseline Architecture Overview

| Baseline | Input Modalities | Model Type |
|----------|-----------------|------------|
| OCR Baseline | Text transcription only | CER/WER metric evaluation |
| Text-Only | Word tokens (TF-IDF) | Logistic Regression |
| Layout-Aware | Word tokens + normalized bounding boxes | Logistic Regression |

---

## 1. OCR Baseline

**File:** `models/baseline/ocr_baseline.py`

Evaluates raw OCR quality against ground-truth text using standard metrics:

- **CER (Character Error Rate):** `edit_distance(hypothesis, reference) / len(reference)`
- **WER (Word Error Rate):** Word-level edit distance ratio

For SROIE, additionally measures **entity-level coverage** — whether ground-truth entity strings are detectable in the OCR output.

FUNSD and CORD use pre-provided OCR words from HuggingFace, so transcription-level CER/WER is not applicable; only SROIE entity coverage is reported.

---

## 2. Text-Only Baseline

**File:** `models/baseline/text_baseline.py`

A TF-IDF + Logistic Regression pipeline operating **purely on word tokens**, with no spatial information:

**Feature Engineering:**
- Character n-grams (2–4, `char_wb` analyzer): captures morphological patterns
- Word n-grams (1–2): captures lexical context
- Orthographic features via `extract_lexical_features()`: `is_numeric`, `has_currency`, `is_uppercase`, `char_count`, `has_date_pattern`, `has_time_pattern`, etc.

**Model:** `sklearn.linear_model.LogisticRegression` with `class_weight="balanced"` (handles label imbalance), `solver="lbfgs"`, `max_iter=2000`.

**Saved artifacts:** `models/saved/{dataset}_text_baseline.joblib`

---

## 3. Layout-Aware Baseline

**File:** `models/baseline/layout_baseline.py`

Combines TF-IDF text features with **normalized spatial features** from `extract_spatial_features()` in `bbox_utils.py`:

**Text Features (same as Text-Only):** Character n-grams + word unigrams via TF-IDF

**Spatial Features (per token, 19 dimensions):**
- `norm_x0, norm_y0, norm_x1, norm_y1` — normalized corner coordinates
- `norm_cx, norm_cy` — normalized center coordinates
- `norm_width, norm_height, norm_area, aspect_ratio` — geometric dimensions
- `is_top_third, is_bottom_third, is_left_half` — zone flags
- `delta_prev_cx/cy, same_line_prev` — relative position to previous token
- `delta_next_cx/cy, same_line_next` — relative position to next token

**Model:** `LogisticRegression` with `class_weight="balanced"`, `solver="lbfgs"`, `max_iter=2000`.

**Saved artifacts:** `models/saved/{dataset}_layout_baseline.joblib`

---

## Evaluation Metrics

**File:** `evaluation/metrics.py`

| Metric | Formula | Use |
|--------|---------|-----|
| Micro F1 | Global TP/FP/FN across all classes | Overall performance |
| Macro F1 | Mean per-class F1 (unweighted) | Class balance sensitivity |
| Weighted F1 | Mean per-class F1 weighted by support | Production-realistic |
| Per-class F1 | Per-label precision/recall/F1 | Diagnostic per entity |
| CER | `edit_dist(hyp, ref) / len(ref)` | OCR quality |
| WER | Word-level edit distance ratio | OCR quality |

---

## Evaluation Reports

**File:** `evaluation/reports.py`

Generated outputs:
- `reports/baseline_results.csv` — all model × dataset × split results
- `reports/baseline_results.json` — structured JSON with full metric breakdown
- `reports/baseline_f1_comparison.png` — side-by-side Macro F1 bar chart across models/datasets

---

## Training Pipeline

**File:** `training/train_baseline.py`

```bash
python training/train_baseline.py \
  --datasets sroie funsd cord \
  --processed_dir processed_data \
  --models_dir models/saved
```

For each dataset:
1. Load `{dataset}/train.jsonl` and `{dataset}/val.jsonl`
2. Fit Text-Only and Layout-Aware baselines on train
3. Evaluate on val; log validation metrics
4. Save model artifacts to `models/saved/`

---

## Evaluation Pipeline

**File:** `training/evaluate_baseline.py`

```bash
python training/evaluate_baseline.py \
  --datasets sroie funsd cord \
  --processed_dir processed_data \
  --models_dir models/saved \
  --reports_dir reports
```

Generates final test-set metrics and saves reports.

---

## Actual Test-Set Results

All results below are measured on the **held-out test splits**. No test data was used during training or model selection.

### SROIE (361 test samples)

| Model | Macro F1 | Micro F1 | Weighted F1 | Precision | Recall | CER | WER |
|-------|----------|----------|-------------|-----------|--------|-----|-----|
| OCR Baseline (Entity Coverage) | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 | 0.252 | 0.254 |
| Text-Only (TF-IDF + LR) | 0.262 | 0.919 | 0.936 | 0.242 | 0.310 | — | — |
| Text+Layout (Spatial + LR) | **0.266** | **0.928** | **0.941** | **0.246** | **0.313** | — | — |

### FUNSD (50 test samples)

| Model | Macro F1 | Micro F1 | Weighted F1 | Precision | Recall |
|-------|----------|----------|-------------|-----------|--------|
| Text-Only (TF-IDF + LR) | 0.351 | 0.402 | 0.414 | 0.347 | 0.419 |
| Text+Layout (Spatial + LR) | **0.500** | **0.540** | **0.546** | **0.484** | **0.546** |

### CORD (100 test samples)

| Model | Macro F1 | Micro F1 | Weighted F1 | Precision | Recall |
|-------|----------|----------|-------------|-----------|--------|
| Text-Only (TF-IDF + LR) | 0.327 | 0.684 | 0.683 | 0.322 | 0.340 |
| Text+Layout (Spatial + LR) | **0.377** | **0.811** | **0.813** | **0.369** | **0.389** |

---

## Analysis and Observations

### Layout information is most valuable on FUNSD
Adding normalized bounding box features raises Macro F1 from **0.351 → 0.500** (+14.9 pp) on FUNSD. Forms have complex 2D structure — questions appear at irregular positions relative to their answers, and header blocks use visual spacing that text order alone cannot recover.

### SROIE layout gain is small
Receipts are linear and sequential. OCR word order already encodes most positional information, so spatial features add only **+0.4 pp Macro F1**. The Micro F1 gain (+0.9 pp) suggests layout slightly improves entity boundary detection.

### CORD micro F1 boost from layout
CORD's fine-grained menu/total hierarchy sees a **+12.7 pp Micro F1** gain from layout (0.684 → 0.811), showing that bounding boxes are strong signals for distinguishing menu item regions from total/subtotal rows. Macro F1 gain (+5.0 pp) is smaller due to class imbalance — MENU tokens dominate, and rare classes (VOID: 4 tokens total) score 0.0 in any model.

### Why Macro F1 < Micro F1 everywhere
All three datasets have heavy `O`-token dominance. Macro F1 averages per-class F1 unweighted, so classes with few instances (e.g., CORD VOID, SROIE DATE) drag the macro average down even when the model performs well overall.

### OCR quality on SROIE
Entity coverage of **0.75** means 75% of ground-truth entity strings (company, date, address, total) appear verbatim in the OCR word sequence. CER=0.252 and WER=0.254 indicate moderate transcription error on entities that are not found, driven primarily by address fields with unusual formatting.

---

## Validation Metrics (for reference)

These were logged during training on the 15% held-out validation splits:

| Dataset | Model | Val Macro F1 | Val Micro F1 |
|---------|-------|-------------|-------------|
| SROIE | Text-Only | 0.2656 | 0.9175 |
| SROIE | Layout-Aware | 0.2679 | 0.9234 |
| FUNSD | Text-Only | 0.3880 | 0.4270 |
| FUNSD | Layout-Aware | 0.5084 | 0.5398 |
| CORD | Text-Only | 0.3365 | 0.7059 |
| CORD | Layout-Aware | 0.3797 | 0.8161 |

Val and test metrics are consistent — no significant overfitting.

---

## Unit Tests

All baseline model tests are in `tests/test_baselines.py`:

```bash
python -m pytest tests/ -v
```

Tests cover:
- Model instantiation and fit/predict cycle
- Feature extraction dimensions
- Label prediction shapes
- Metric computation correctness

---

## Design Decisions

**Why Logistic Regression, not Random Forest?**
LR is faster to train and produces well-calibrated probabilities. RandomForest was also implemented and tested (see `LayoutAwareBaselineModel`'s `classifier_type` parameter) but showed no consistent improvement over LR while being 3–5× slower. LR is the reported baseline.

**Why not increase max_features above 5000?**
Preliminary experiments showed diminishing returns above 5000 for TF-IDF n-grams on datasets of this size (~19–28k training tokens). Higher vocabulary sizes increase memory and training time without measurable F1 gain.

**Why `solver="lbfgs"` with `max_iter=2000`?**
`lbfgs` converges faster than `saga` on the dense feature matrices produced by `hstack` of TF-IDF + spatial features. 2000 iterations ensures convergence on all three datasets without warnings.

**Why are raw bboxes passed directly without StandardScaler?**
Bounding boxes are already normalized to `[0, 1000]` by the preprocessing pipeline, giving them a consistent scale. Additional scaling was not needed.

---

## Recommendations for Phase 4

1. **Use LayoutLMv3** (`microsoft/layoutlmv3-base`) — expected to significantly exceed these baselines by combining text, layout, and image patch embeddings in a single pretrained Transformer.
2. **Switch to seqeval F1** — replace token-level sklearn F1 with entity-span F1 via `seqeval` for NER-standard reporting.
3. **Weighted cross-entropy loss** — address the `O`-token dominance that suppresses entity class recall.
4. **Learning rate warmup** — use linear warmup + cosine decay scheduler for LayoutLMv3 fine-tuning.
5. **Per-class analysis** — add per-entity-class F1 reporting to the evaluation pipeline for diagnostic visibility.
