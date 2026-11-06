# Phase 5 — Systematic Evaluation and Analysis

## Overview

Phase 5 performs comprehensive evaluation of the LayoutLMv3 model against the Phase-3
baselines, establishes an ablation study on input modalities, generates error analysis,
and tests robustness of the inference pipeline.

---

## Hardware

| Property | Value |
|----------|-------|
| CPU | AMD Ryzen 7 5000-series |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| VRAM | 6 GB |
| Python | 3.12 (separate .venv312 for CUDA support) |
| PyTorch | 2.4.1+cu121 |
| CUDA driver | 12.3 |
| Transformers | 5.16.1 |
| Seed | 42 |

Note: The original venv used Python 3.14, which has no PyTorch CUDA wheels available
(see [pytorch/pytorch#169929](https://github.com/pytorch/pytorch/issues/169929)).
A Python 3.12 virtual environment (`.venv312/`) was created to enable GPU training.

---

## Phase 4 Bugs Found and Fixed

### Bug 1: SROIE label alignment failure

**Root cause:** The SROIE HuggingFace dataset delivers OCR text as multi-word
strings per "word" entry (e.g., the entire company name `"SINNATHAMBY HOLDINGS SDN. BHD. (249336-T)"` is a single OCR word). The original `align_sroie_entity_labels()` split
both the entity value and the OCR words on spaces and tried to find a multi-token
window match, which never succeeded on single-token OCR "words".

**Result before fix:** B-COMPANY: 3/533 samples (0.6%), B-ADDRESS: 0/533 (0%).

**Fix:** Switched to substring containment matching: an entity value is found in a
word if `normalize(entity) in normalize(word)` or `normalize(word) in normalize(entity)`.

**Result after fix:** B-COMPANY: 522/533 (97.9%), B-ADDRESS: 530/533 (99.4%),
B-DATE: 532/533 (99.8%), B-TOTAL: 532/533 (99.8%).

### Bug 2: word_ids reconstruction heuristic

**Root cause:** The training evaluation loop needed to convert token-level predictions
back to word-level labels, which requires knowing which token belongs to which word.
`word_ids` were produced during tokenization but not stored in the dataset batch.
The heuristic used to reconstruct them from `(label_id == -100)` was unreliable —
the [SEP] token has `label_id == -100` and `attention_mask == 1`, causing it to be
misidentified as a continuation subword.

**Fix:** `LayoutLMv3DocumentDataset.__getitem__()` now stores `word_ids` as an int64
tensor (with `WORD_ID_PAD = -1` for `None` positions). The training/evaluation loops
use `word_ids_tensor_to_list()` to recover the correct word_ids rather than any
heuristic. This eliminated the systematic word-level label misalignment that caused
seqeval to produce 0.0 F1.

### Bug 3: FUNSD image filename collision

**Root cause (from Phase 2):** FUNSD images were saved as `{row_index}.png`, and row
indices reset per split (0–148 for train, 0–49 for test). Train row 42 and test row 42
are different documents but were writing to the same file.

**Fix:** Image filenames now include the split prefix: `funsd_{split}_{id}.png`.
FUNSD was re-preprocessed to regenerate the correct image paths in JSONL files.

---

## Phase 4 Validation Results

### Overfit test (8 samples, 30 epochs, CPU)

| Epoch | Train Loss | Train macro_f1 |
|-------|-----------|---------------|
| 1 | 2.016 | 0.026 |
| 9 | 0.227 | 0.233 |
| 13 | 0.152 | 0.552 |
| 18 | 0.087 | 0.871 |
| 22 | 0.057 | 0.933 |

**Verdict: PASS** — The model memorises the 8 training samples. All pipeline
components (tokenization, label alignment, forward pass, loss, backward pass,
checkpoint saving) are verified correct.

---

## Evaluation Methodology

### Primary metric: seqeval entity-span F1

All LayoutLMv3 evaluations use **seqeval IOB2 entity-span F1**:

- A prediction is correct only if both the entity type and the exact span boundaries
  (start/end word positions) match the ground truth.
- This is stricter than token-level F1 and is the standard for NER benchmarks.
- Computed via `seqeval.metrics.f1_score()` with the IOB2 scheme.

### Secondary metric: token-level sklearn F1

Used for diagnostic comparison with Phase-3 baselines:
- Micro F1, Macro F1, Weighted F1 via `sklearn.metrics.precision_recall_fscore_support`
- Note: token-level F1 is not directly comparable to seqeval entity-span F1.

### Phase 3 baselines used token-level sklearn F1.
### Phase 4 LayoutLMv3 uses seqeval entity-span F1.

This difference is documented in all comparison tables.

---

## Dataset Splits Used

| Dataset | Train | Val | Test |
|---------|-------|-----|------|
| SROIE | 533 | 93 | 361 |
| FUNSD | 127 | 22 | 50 |
| CORD | 800 | 100 | 100 |

No test data used during training or checkpoint selection. Best checkpoint selected
by validation macro F1.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | microsoft/layoutlmv3-base |
| Checkpoint (base) | cfbbbff0762e6aab37086fdd4739ad14fe7d5db4 |
| Optimizer | AdamW |
| Learning rate | 5e-5 (SROIE, FUNSD); 3e-5 (CORD) |
| Batch size | 4 |
| Gradient accumulation | 2 |
| Effective batch size | 8 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Max seq length | 512 |
| fp16 | true (GPU) |
| Seed | 42 |
| Max epochs | 10 (SROIE, CORD); 12 (FUNSD) |
| Early stopping | patience=3 (full stage only) |

---

## Phase 5 Evaluation Results

*Results populated after GPU training completes.*

### SROIE

| Model | Metric type | Macro F1 | Micro F1 | Precision | Recall |
|-------|-------------|----------|----------|-----------|--------|
| OCR Baseline | entity coverage | 0.750 | — | 0.750 | 0.750 |
| Text-Only TF-IDF | token-level | 0.262 | 0.919 | 0.242 | 0.310 |
| Text+Layout TF-IDF | token-level | 0.266 | 0.928 | 0.246 | 0.313 |
| LayoutLMv3 (full) | seqeval span | — | — | — | — |

Run: `python training/train_layoutlmv3.py --dataset sroie --stage full`
Then: `python evaluation/evaluate_layoutlmv3.py --dataset sroie --stage full`

### FUNSD

| Model | Metric type | Macro F1 | Micro F1 |
|-------|-------------|----------|----------|
| Text-Only TF-IDF | token-level | 0.351 | 0.402 |
| Text+Layout TF-IDF | token-level | 0.500 | 0.540 |
| LayoutLMv3 (full) | seqeval span | — | — |

### CORD

| Model | Metric type | Macro F1 | Micro F1 |
|-------|-------------|----------|----------|
| Text-Only TF-IDF | token-level | 0.327 | 0.684 |
| Text+Layout TF-IDF | token-level | 0.377 | 0.811 |
| LayoutLMv3 (full) | seqeval span | — | — |

*Published benchmarks for LayoutLMv3-base: FUNSD ~0.92, CORD ~0.96 (seqeval).*

---

## Ablation Study

### Input modality contribution

| Input modality | Model | SROIE Macro F1 | FUNSD Macro F1 | CORD Macro F1 | Metric |
|----------------|-------|---------------|---------------|--------------|--------|
| Text only | TF-IDF + LR | 0.262 | 0.351 | 0.327 | token-level |
| Text + Layout | TF-IDF + LR | 0.266 | 0.500 | 0.377 | token-level |
| Text + Layout + Image | LayoutLMv3-base | — | — | — | seqeval span |

Key finding from Phase 3: Layout information gives the largest improvement on FUNSD
(+14.9 pp macro F1), confirming that 2D spatial structure is essential for form
understanding. SROIE's linear receipt layout shows minimal layout benefit (+0.4 pp).

**Limitation:** The ablation is not perfectly controlled — baseline models use TF-IDF
with no sequence modelling, while LayoutLMv3 uses pretrained Transformer representations
with cross-modal attention. A fully controlled ablation would require training
LayoutLMv3 variants with image patches masked and bboxes zeroed. This is left for
future work.

---

## Error Analysis

Error analysis visualizations are saved to `reports/error_analysis/{dataset}/` after
running `evaluate_layoutlmv3.py`.

### Expected error categories (from Phase 3 analysis + domain knowledge)

**SROIE:**
- Multi-line addresses — single OCR "word" spans multiple lines; entity may truncate
- Total variants — some receipts use "TOTAL AMOUNT" vs "TOTAL" with different formatting
- Date formats — timestamps in the OCR word can contain extra characters beyond the
  date value (e.g., `"06/02/2018 07:09:46 PM"` when entity is `"06/02/2018"`)
- Company name truncation — OCR word may include registration numbers in parentheses

**FUNSD:**
- Dense multi-column layouts — tokens from different logical regions may appear
  adjacent in reading order
- Handwritten fields — OCR quality is lower for handwritten content
- Ambiguous question/answer boundaries — short answers can blend with next question

**CORD:**
- Multi-line menu items — an item name may span multiple OCR words with different categories
- Subtotal vs Total confusion — both share similar visual patterns
- Non-standard receipt formats — some merchants use unusual label styles

---

## Robustness Tests

Run after full training:
```bash
python scripts/robustness_test.py --dataset sroie --stage full --n 20
```

Tests 9 perturbation types:
1. baseline (no perturbation)
2. low_res_50% (downscale to 50% then upscale)
3. low_res_25%
4. noise_5% (Gaussian noise, 5% of pixel range)
5. noise_15%
6. rotation_3deg
7. rotation_10deg
8. blur_radius2 (Gaussian blur)
9. blur_radius5

Results saved to `reports/robustness_{dataset}.json`.

**Known limitation:** LayoutLMv3's image processor normalises and resizes to 224×224,
which partially compensates for resolution and blur perturbations. True robustness
would require testing with corrupted OCR word quality (wrong words, missing words),
which requires a separate OCR perturbation pipeline not implemented here.

---

## Conclusions

### What was achieved
1. Two root-cause bugs in Phase 4 diagnosed and fixed (label alignment + word_ids)
2. All three datasets correctly preprocessed with proper entity label coverage
3. Overfit test confirms full training pipeline is correct (F1 rises to 0.93 on 8 samples)
4. Pipeline ready for GPU full training with fp16 and gradient accumulation
5. Complete evaluation framework with seqeval metrics, error analysis, confusion matrices
6. Robustness testing script implemented and ready to execute

### Main findings (from Phase 3 baselines)
- Layout information matters most for FUNSD (+14.9 pp macro F1 from layout)
- CORD shows largest micro F1 gain from layout (+12.7 pp)
- SROIE is largely linearly structured; layout adds minimal benefit
- LayoutLMv3's multimodal pretraining is expected to substantially exceed these baselines

### Known limitations
1. **CPU-only training was too slow** — Python 3.14 has no CUDA PyTorch wheels.
   GPU training requires the `.venv312/` environment with Python 3.12.
2. **seqeval vs token-level metrics** — Phase 3 and Phase 4 use different primary metrics.
   Direct numerical comparison requires explicit metric alignment.
3. **No controlled ablation** — A proper ablation would train LayoutLMv3 variants with
   image and layout inputs individually masked.
4. **SROIE label schema is B-only** — Because each entity is a single multi-word OCR token,
   no I- tags appear in the SROIE processed data. This is correct for this dataset's OCR
   granularity but means seqeval treats every entity as a single-token span.

---

## Commands to Reproduce

```bash
# 1. Set up GPU environment
C:\Users\srina\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv312
.venv312\Scripts\pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
.venv312\Scripts\python scripts/setup_venv312.py

# 2. Run SROIE smoke test (GPU)
.venv312\Scripts\python training/train_layoutlmv3.py --dataset sroie --stage smoke

# 3. Run full training on all datasets
.venv312\Scripts\python training/train_layoutlmv3.py --dataset sroie --stage full
.venv312\Scripts\python training/train_layoutlmv3.py --dataset funsd --stage full
.venv312\Scripts\python training/train_layoutlmv3.py --dataset cord  --stage full

# 4. Evaluate all datasets
.venv312\Scripts\python evaluation/evaluate_layoutlmv3.py \
  --dataset sroie funsd cord --stage full

# 5. Generate plots
.venv312\Scripts\python scripts/generate_plots.py

# 6. Robustness tests
.venv312\Scripts\python scripts/robustness_test.py --dataset sroie --stage full

# 7. Final results report
.venv312\Scripts\python scripts/final_results_report.py
```
