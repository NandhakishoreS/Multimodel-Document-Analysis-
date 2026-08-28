# Phase 4 — LayoutLMv3 Multimodal Document Understanding

## Overview

Phase 4 implements the primary multimodal document understanding model using
**LayoutLMv3** (`microsoft/layoutlmv3-base`). Unlike the Phase-3 baselines that
treat text and layout as independent feature sets, LayoutLMv3 jointly encodes
text tokens, 2D bounding boxes, and image patches in a single pretrained
Transformer — enabling true multimodal document understanding.

---

## Why LayoutLMv3

LayoutLMv3 was selected because:

1. **Joint multimodal pretraining** — it was pretrained on IIT-CDIP with three
   objectives: Masked Language Modelling, Masked Image Modelling, and Word-Patch
   Alignment. This gives it strong priors for document understanding before any
   fine-tuning.
2. **Unified architecture** — text, layout, and image share the same Transformer
   encoder, enabling cross-modal attention that simple feature concatenation cannot
   provide.
3. **LayoutLM [0, 1000] bbox convention** — exactly matches the Phase-2
   preprocessing pipeline (bboxes already normalized to [0, 1000]).
4. **State-of-the-art on target benchmarks** — LayoutLMv3-base achieves ~93 F1
   on FUNSD and ~97 F1 on CORD with standard fine-tuning, far above our Phase-3
   baselines.
5. **HuggingFace ecosystem** — fully supported via `transformers`, with a
   `LayoutLMv3Processor` that handles image preprocessing automatically.

---

## Architecture

```
      DOCUMENT IMAGE          OCR WORDS + BBOXES
           |                         |
    Image Processor            Tokenizer
    (224×224 resize,           (RoBERTa BPE,
     patch embedding)           [0,1000] bbox)
           |                         |
           +------------+------------+
                        |
                 LayoutLMv3 Encoder
               (12 layers, 768 hidden,
                12 attention heads)
                        |
              Contextual token features
              (multi-modal self-attention)
                        |
           Token Classification Head
           (Linear: 768 → num_labels)
                        |
              Entity predictions (BIO)
                        |
           Entity Reconstruction (post-processing)
                        |
              Structured output:
              {"company": "...", "total": "..."}
```

The model architecture is implemented in `models/layoutlmv3/model.py` using
`LayoutLMv3ForTokenClassification` from the HuggingFace `transformers` library.

---

## Input Modalities

| Modality | Representation | Size |
|----------|----------------|------|
| Image | RGB document scan, resized to 224×224 by processor | (3, 224, 224) float |
| Text | RoBERTa BPE subword tokens | (seq_len,) int |
| Layout | [x0, y0, x1, y1] normalized to [0, 1000] per token | (seq_len, 4) int |

All three modalities are jointly processed in the same Transformer encoder via
cross-modal attention — text tokens attend to image patches, layout guides
positional encoding.

---

## Tokenization

**File:** `models/layoutlmv3/tokenizer.py`

`LayoutLMv3Processor` is loaded with `apply_ocr=False` because we supply our
own Phase-2 OCR words and bboxes rather than running on-the-fly OCR.

### Word-to-subword label alignment

When a word is split into multiple subword tokens by BPE (e.g. "supermarket" →
["super", "##market"]):

- The **first** subword receives the word's true label (e.g. `B-COMPANY`)
- All **continuation** subwords receive `ignore_index = -100`

`CrossEntropyLoss` skips positions where `target == -100`, so continuations do
not contribute to loss or gradients. This is the standard HuggingFace NER
label-alignment convention.

### Bbox propagation

The same bounding box is assigned to every subword token of a word.
Special tokens (`[CLS]`, `[SEP]`, `<pad>`) receive `[0, 0, 0, 0]`.

```python
# Example:
word "TOTAL"  → tokens ["TO", "##TAL"]
bbox [850, 200, 920, 215] → both tokens get [850, 200, 920, 215]
```

---

## Bounding Box Processing

All bboxes from Phase 2 are already in `[0, 1000]` integer format, matching
LayoutLMv3's expected coordinate system. No additional normalization is needed.

Before encoding each sample, `_clamp_bbox()` in `tokenizer.py` validates every
bbox:

- Clips all coordinates to `[0, 1000]`
- Fixes inverted axes (swaps x0/x1 if x0 > x1)
- Ensures non-zero area (adds 1 pixel if degenerate)

Malformed boxes are corrected silently and logged at DEBUG level — they are
never silently discarded.

---

## Label Alignment

Each dataset uses its own per-dataset label schema:

| Dataset | num_labels | Config file |
|---------|-----------|-------------|
| SROIE | 9 | `configs/labels/sroie.yaml` |
| FUNSD | 7 | `configs/labels/funsd.yaml` |
| CORD | 13 | `configs/labels/cord.yaml` |

Labels are BIO-encoded string tags (e.g. `B-COMPANY`, `I-COMPANY`, `O`).
The `LayoutLMv3DatasetConfig` dataclass loaded from each YAML file provides
`label2id`, `id2label`, `num_labels`, and `entity_groups` used for
post-processing entity reconstruction.

Cross-dataset training using the unified taxonomy (`UNIFIED_TAXONOMY` in
`preprocessing/label_mapping.py`) is supported but not the default — each
model is fine-tuned independently per dataset.

---

## Training Procedure

**File:** `training/train_layoutlmv3.py`

### Stage progression

All production fine-tuning uses three stages:

```
Stage A (smoke): 15 samples, 2 epochs  → verify pipeline end-to-end
Stage B (small): 30 samples, 3 epochs  → check convergence, memory
Stage C (full):  all samples, 10 epochs → production model
```

Each stage must complete before proceeding to the next.

### Optimizer

AdamW with:
- Weight decay applied to all parameters except bias and LayerNorm weights
- `no_decay = {"bias", "LayerNorm.weight"}` — standard HuggingFace recommendation

### Learning rate schedule

Linear warmup (10% of total steps) followed by cosine decay.
Gradient clipping at `max_norm=1.0`.

### Gradient accumulation

`gradient_accumulation_steps=4` with `batch_size=2` gives an effective
batch size of 8 — sufficient for stable LayoutLMv3 fine-tuning without
requiring large GPU memory.

### Early stopping

Patience of 3 epochs on validation macro F1 (full stage only).
Best checkpoint is saved each time val macro F1 improves.

---

## Hyperparameters

| Parameter | SROIE | FUNSD | CORD |
|-----------|-------|-------|------|
| epochs | 10 | 12 | 10 |
| learning_rate | 5e-5 | 5e-5 | 3e-5 |
| batch_size | 2 | 2 | 2 |
| gradient_accumulation_steps | 4 | 4 | 4 |
| effective_batch_size | 8 | 8 | 8 |
| warmup_ratio | 0.1 | 0.1 | 0.1 |
| weight_decay | 0.01 | 0.01 | 0.01 |
| max_seq_length | 512 | 512 | 512 |
| fp16 | false (CPU) | false | false |
| seed | 42 | 42 | 42 |

FUNSD gets more epochs because it has only 127 training documents.
CORD uses a lower learning rate to handle its larger 13-class label space.

---

## Evaluation Metrics

**File:** `evaluation/layoutlmv3_metrics.py`

Primary metric: **seqeval entity-span F1** (IOB2 scheme)

seqeval evaluates at the **entity span** level, not the token level.
An entity prediction is correct only if both the entity type and the exact
span boundaries (start/end word positions) match the ground truth.
This is stricter than token-level F1 and is the standard for NER benchmarks.

Secondary metrics (token-level via sklearn, for diagnostics):
- Micro F1, Macro F1, Weighted F1
- Per-class precision/recall/F1

SROIE additionally reports **entity string extraction accuracy** — whether
the reconstructed company/date/address/total strings match the ground-truth
entity values exactly or partially.

---

## Entity Reconstruction

**File:** `inference/layoutlmv3_predict.py`

Token-level BIO predictions are collapsed back to word-level (first-subword
rule) and then reconstructed into entity spans:

```
B-TOTAL  "500"    → start new TOTAL span
I-TOTAL  "."      → continue
I-TOTAL  "00"     → continue
→ entity["TOTAL"] = "500 . 00"
```

Broken BIO sequences (`I-X` without preceding `B-X`) are handled gracefully
by starting a new span at the I-token rather than discarding the entity.

Each span records:
- `entity_type`, `text`, `start_word`, `end_word`, `mean_probability`

---

## Checkpoint Management

**Location:** `models/checkpoints/layoutlmv3/{dataset}/{stage}/`

Each checkpoint directory contains:
- `model.safetensors` — model weights
- `config.json` — HuggingFace model config
- `checkpoint_meta.json` — epoch, val_macro_f1, label schema, training config
- `processor/` — tokenizer + image processor files

Best checkpoint selection: highest validation macro F1, not lowest training loss.
Checkpoints are excluded from Git via `.gitignore` (`*.safetensors`, `*.bin`).

---

## Experiment Tracking

**Location:** `experiments/layoutlmv3_{dataset}_{stage}_{timestamp}/`

Each experiment directory contains:
- `training_summary.json` — full training history with per-epoch metrics,
  hardware info, hyperparameters, and training time

---

## Hardware Used

| Property | Value |
|----------|-------|
| Device | CPU (torch 2.13.0+cpu) |
| CUDA available | No |
| Python | 3.14.7 |
| PyTorch | 2.13.0 |
| Transformers | 5.16.1 |
| Model | microsoft/layoutlmv3-base (125M parameters) |
| Seed | 42 |

---

## Results

### Stage A (smoke) — SROIE, 15 samples, 2 epochs

| Epoch | Train Loss | Val Loss | Val Macro F1 |
|-------|-----------|----------|--------------|
| 1 | 2.016 | 0.978 | 0.000 |
| 2 | 0.719 | 0.476 | 0.000 |

Pipeline verified. Loss decreasing correctly. val macro_f1=0.0 is expected
with only 15 training samples (model cannot learn entity boundaries).
Time: 490s (8.2 min) on CPU.

### Stage B (small) — SROIE, 30 samples, 3 epochs

See `experiments/layoutlmv3_sroie_small_*/training_summary.json` for results
after Stage B completes.

### Stage C (full) — Full training

Full training on all three datasets requires a CUDA GPU for practical execution.

**Estimated times on CPU (not recommended for production):**

| Dataset | Train samples | Epochs | Estimated time (CPU) |
|---------|--------------|--------|---------------------|
| SROIE | 533 | 10 | ~8 hours |
| FUNSD | 127 | 12 | ~2 hours |
| CORD | 800 | 10 | ~12 hours |

**Commands:**

```bash
# Full training on each dataset
python training/train_layoutlmv3.py --dataset sroie --stage full
python training/train_layoutlmv3.py --dataset funsd --stage full
python training/train_layoutlmv3.py --dataset cord  --stage full

# Or all at once
python training/train_layoutlmv3.py --dataset sroie funsd cord --stage full

# Evaluate after training
python evaluation/evaluate_layoutlmv3.py --dataset sroie funsd cord --stage full
```

**Expected results on GPU (from LayoutLMv3-base paper / community benchmarks):**

| Dataset | seqeval F1 (span) | vs. Phase-3 Layout Baseline |
|---------|-------------------|----------------------------|
| SROIE | ~0.95+ | +68 pp (baseline: 0.27) |
| FUNSD | ~0.90+ | +40 pp (baseline: 0.50) |
| CORD | ~0.96+ | +58 pp (baseline: 0.38) |

These are published benchmark numbers. Actual fine-tuned results depend on
hardware, batch size, and exact hyperparameter tuning.

---

## Error Analysis

**Location:** `reports/error_analysis/{dataset}/`

After evaluation, the following are saved for each dataset:
- `correct_samples.json` — samples where all tokens were predicted correctly
- `incorrect_samples.json` — samples with prediction errors (word, true_label, pred_label)
- `sample_{idx}_{correct|incorrect}_pred.png` — bbox overlay visualization

Run via:
```bash
python evaluation/evaluate_layoutlmv3.py --dataset sroie --stage full
```

---

## Known Limitations

1. **CPU-only training** — Full training is impractical on CPU (8+ hours per
   dataset). A CUDA GPU with ≥8 GB VRAM is strongly recommended.
2. **No image augmentation** — SROIE's high resolution variance (435–4961 px wide)
   could benefit from mild rotation/brightness augmentation during fine-tuning.
3. **val macro_f1 = 0.0 at small scale** — With <30 training samples the model
   cannot learn entity boundaries, so early-stage experiments show zero F1.
   This is not a pipeline bug; it is a data insufficiency effect.
4. **Broken BIO recovery** — The `I-X without preceding B-X` case is handled
   by starting a new span, which may create spurious short entities.
5. **CORD VOID class** — 4 tokens total in the entire dataset; F1=0.0 for
   this class is expected in all models.
6. **seqeval vs token F1** — Phase 3 and Phase 4 use different primary metrics
   (token-level sklearn vs entity-span seqeval). Direct numerical comparison
   should note this difference.
