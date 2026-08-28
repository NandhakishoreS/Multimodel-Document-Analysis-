# Multimodal Transformer-Based Intelligent Document Understanding and Information Extraction

A research and engineering pipeline for understanding complex scanned documents — receipts, forms, and invoices — using multimodal information: document images, OCR text, and spatial bounding box layout.

---

## Project Objective

Build an AI system capable of extracting structured information from complex documents by combining:

1. **Document image** — visual features from scanned pages
2. **OCR text** — word-level transcriptions
3. **Spatial layout** — normalized bounding boxes encoding 2D position

The eventual goal is to fine-tune a pretrained multimodal document Transformer (LayoutLMv3) and serve it through a Streamlit application. Phases 1–3 establish the data foundation and measurable baselines before that step.

---

## Datasets

| Dataset | Source | Task | Samples | Splits |
|---------|--------|------|---------|--------|
| **SROIE** | `jsdnrs/ICDAR2019-SROIE` (HuggingFace) | Key Information Extraction | 987 | Train 533 / Val 93 / Test 361 |
| **FUNSD** | `nielsr/funsd` (HuggingFace) | Form Entity Labeling (NER) | 199 | Train 127 / Val 22 / Test 50 |
| **CORD** | `naver-clova-ix/cord-v2` (HuggingFace) | Receipt Entity Labeling (NER) | 1,000 | Train 800 / Val 100 / Test 100 |

### Dataset Descriptions

**SROIE (ICDAR 2019 Scanned Receipts OCR and Information Extraction)**
- Scanned retail receipts annotated with four entity types: `company`, `date`, `address`, `total`
- Labels encoded in BIO format: `O`, `B-COMPANY`, `I-COMPANY`, `B-DATE`, `I-DATE`, `B-ADDRESS`, `I-ADDRESS`, `B-TOTAL`, `I-TOTAL`
- High image resolution variance (435–4961 px wide)
- Avg 53.7 words per document

**FUNSD (Form Understanding in Noisy Scanned Documents)**
- Noisy scanned government and business forms with hierarchical entity structure
- Labels: `O`, `B-HEADER`, `I-HEADER`, `B-QUESTION`, `I-QUESTION`, `B-ANSWER`, `I-ANSWER`
- Explicit relational links between question and answer blocks
- Avg 153.7 words per document — highest token density of the three datasets

**CORD (Consolidated Receipt Dataset v2)**
- Fine-grained receipt understanding across restaurant and retail domains
- Labels: `O`, `B/I-MENU`, `B/I-TOTAL`, `B/I-SUBTOTAL`, `B/I-VOID`, `B/I-PAYMENT`, `B/I-ETC`
- Official train/val/test splits provided
- Avg 23.9 words per document

---

## Directory Structure

```
multimodal-document-ai/
├── app/                        # (Phase 4) Streamlit inference application
├── configs/
│   └── preprocessing.yaml      # Preprocessing config: bbox range, splits, seeds
├── datasets/
│   ├── sroie/
│   │   ├── download_sroie.py   # Download SROIE from HuggingFace
│   │   └── explore_sroie.py    # Dataset statistics and visualizations
│   ├── funsd/
│   │   ├── download_funsd.py
│   │   └── explore_funsd.py
│   └── cord/
│       ├── download_cord.py
│       └── explore_cord.py
├── docs/
│   ├── phase1.md               # Phase 1 design decisions and results
│   ├── phase2.md               # Phase 2 preprocessing documentation
│   └── phase3.md               # Phase 3 baseline results
├── evaluation/
│   ├── metrics.py              # Precision, Recall, F1, CER, WER
│   └── reports.py              # CSV/JSON/PNG report generation
├── inference/                  # (Phase 4) Inference utilities
├── models/
│   ├── baseline/
│   │   ├── ocr_baseline.py     # OCR quality evaluator (CER/WER/entity coverage)
│   │   ├── text_baseline.py    # TF-IDF + Logistic Regression (text only)
│   │   └── layout_baseline.py  # TF-IDF + Spatial features + LR (text + layout)
│   └── saved/                  # Trained model artifacts (.joblib)
├── notebooks/                  # Exploratory notebooks
├── preprocessing/
│   ├── bbox_utils.py           # normalize_bbox, denormalize_bbox, extract_spatial_features
│   ├── cord_processor.py       # CORD → unified schema
│   ├── funsd_processor.py      # FUNSD → unified schema
│   ├── label_mapping.py        # Per-dataset + unified taxonomy label encoders
│   ├── sroie_processor.py      # SROIE → unified schema
│   ├── text_utils.py           # clean_text, extract_lexical_features
│   └── unified_dataset.py      # PyTorch Dataset + full preprocessing pipeline
├── processed_data/
│   ├── sroie/                  # train.jsonl, val.jsonl, test.jsonl, images/
│   ├── funsd/
│   └── cord/
├── reports/
│   ├── baseline_results.csv    # Final test-set metrics for all baselines
│   ├── baseline_results.json
│   ├── baseline_f1_comparison.png
│   ├── sroie_report.{json,csv}
│   ├── funsd_report.{json,csv}
│   └── cord_report.{json,csv}
├── scripts/
│   ├── validate_phase1.py      # Phase 1 dataset integrity checks
│   └── validate_phase2.py      # Phase 2 schema and bbox validation
├── tests/
│   └── test_*.py               # Unit tests for preprocessing and models
├── training/
│   ├── train_baseline.py       # Train text-only and layout-aware baselines
│   └── evaluate_baseline.py    # Evaluate on test splits, generate reports
├── utils/
│   └── visualization.py        # Document bbox overlays, label/F1 charts
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Environment Setup

### Prerequisites

- Python 3.10+
- Git

### Create and activate virtual environment

```bash
cd multimodal-document-ai
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Dependencies (`requirements.txt`)

```
datasets>=2.18.0
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.24.0
pillow>=10.0.0
matplotlib>=3.8.0
seaborn>=0.13.0
jiwer>=3.0.0
rapidfuzz>=3.0.0
pyyaml>=6.0.0
tqdm>=4.66.0
pytest>=8.0.0
```

---

## Phase 1 — Dataset Setup and Exploration

Downloads all three datasets from HuggingFace, computes per-dataset statistics, generates visualizations, and validates integrity.

### Run dataset downloads

```bash
python datasets/sroie/download_sroie.py
python datasets/funsd/download_funsd.py
python datasets/cord/download_cord.py
```

### Run dataset exploration (generates reports/)

```bash
python datasets/sroie/explore_sroie.py
python datasets/funsd/explore_funsd.py
python datasets/cord/explore_cord.py
```

### Run Phase 1 validation

```bash
python scripts/validate_phase1.py
```

Expected output:
```
SROIE Validation: PASSED (987/987 valid)
FUNSD Validation: PASSED (199/199 valid)
CORD Validation:  PASSED (1000/1000 valid)
ALL PHASE 1 VALIDATION CHECKS PASSED SUCCESSFULLY!
```

### Phase 1 reports generated

| File | Content |
|------|---------|
| `reports/sroie_report.json` | Sample counts, resolution stats, entity distribution |
| `reports/sroie_report.csv` | Per-sample statistics |
| `reports/sroie_entity_distribution.png` | Entity frequency bar chart |
| `reports/funsd_report.json` | Label distribution, word count stats |
| `reports/funsd_label_distribution.png` | NER tag frequency chart |
| `reports/cord_report.json` | CORD category distribution |
| `reports/cord_label_distribution.png` | BIO label frequency chart |

---

## Phase 2 — Data Preprocessing and Unification

Converts all three datasets into a unified JSONL schema with normalized bounding boxes, clean text, and BIO labels.

### Unified sample schema

```json
{
  "id":         "X51005806703",
  "dataset":    "sroie",
  "split":      "train",
  "image_path": "processed_data/sroie/images/X51005806703.png",
  "width":      1000,
  "height":     1000,
  "words":      ["COMPANY", "NAME", "..."],
  "bboxes":     [[12, 34, 120, 58], ...],
  "labels":     ["B-COMPANY", "I-COMPANY", "..."],
  "entities":   {"company": "COMPANY NAME", "date": "01/01/2019", ...},
  "relations":  []
}
```

Bounding boxes are normalized to `[0, 1000]` on both axes.

### Run preprocessing pipeline

```bash
# All datasets
python -m preprocessing.unified_dataset

# Single dataset
python -m preprocessing.unified_dataset --datasets sroie
python -m preprocessing.unified_dataset --datasets funsd
python -m preprocessing.unified_dataset --datasets cord
```

### Run Phase 2 validation

```bash
python scripts/validate_phase2.py
```

Expected output:
```
SROIE PASSED: 987 samples, 53,012 tokens across splits {'train': 533, 'val': 93, 'test': 361}
FUNSD PASSED: 199 samples, 30,595 tokens across splits {'train': 127, 'val': 22, 'test': 50}
CORD  PASSED: 1000 samples, 23,909 tokens across splits {'train': 800, 'val': 100, 'test': 100}
ALL PHASE 2 VALIDATION CHECKS PASSED SUCCESSFULLY!
```

### Split strategy

| Dataset | Val Split Source | Seed |
|---------|-----------------|------|
| SROIE | 15% of official train, random | 42 |
| FUNSD | 15% of official train, random | 42 |
| CORD | Official HuggingFace `validation` split | N/A |

---

## Phase 3 — Baseline Models

Establishes measurable baselines before building the final LayoutLMv3 model.

### Train baselines

```bash
python training/train_baseline.py \
  --datasets sroie funsd cord \
  --processed_dir processed_data \
  --models_dir models/saved
```

### Evaluate baselines on test splits

```bash
python training/evaluate_baseline.py \
  --datasets sroie funsd cord \
  --processed_dir processed_data \
  --models_dir models/saved \
  --reports_dir reports
```

### Baseline test results

| Dataset | Model | Macro F1 | Micro F1 | Weighted F1 |
|---------|-------|----------|----------|-------------|
| SROIE | OCR Entity Coverage | 0.750 | 0.750 | 0.750 |
| SROIE | Text-Only (TF-IDF + LR) | 0.262 | 0.919 | 0.936 |
| SROIE | Text+Layout (Spatial + LR) | **0.266** | **0.928** | **0.941** |
| FUNSD | Text-Only (TF-IDF + LR) | 0.351 | 0.402 | 0.414 |
| FUNSD | Text+Layout (Spatial + LR) | **0.500** | **0.540** | **0.546** |
| CORD | Text-Only (TF-IDF + LR) | 0.327 | 0.684 | 0.683 |
| CORD | Text+Layout (Spatial + LR) | **0.377** | **0.811** | **0.813** |

### Reports generated

| File | Content |
|------|---------|
| `reports/baseline_results.csv` | Full test metrics for all models/datasets |
| `reports/baseline_results.json` | Structured JSON with per-class breakdowns |
| `reports/baseline_f1_comparison.png` | Side-by-side Macro F1 bar chart |

---

## Evaluation Methodology

All metrics are computed on the **official held-out test split** of each dataset. Validation splits are used only during training for model selection.

| Metric | Description |
|--------|-------------|
| Micro F1 | Global TP/FP/FN across all token classes |
| Macro F1 | Unweighted mean of per-class F1 — sensitive to class imbalance |
| Weighted F1 | Per-class F1 weighted by support — production-realistic |
| Precision | Macro-averaged across entity classes |
| Recall | Macro-averaged across entity classes |
| CER | Character Error Rate (OCR baseline only) |
| WER | Word Error Rate (OCR baseline only) |

The `O` (background) label is **included** in all metrics unless explicitly filtered, which is why Micro F1 values are high even for weaker models on SROIE and CORD (where `O` dominates the token distribution).

---

## Key Findings

**Layout information matters most for FUNSD:**
Adding normalized bounding box features raises Macro F1 from 0.351 → 0.500 (+14.9 pp) on FUNSD. Forms have complex 2D structure that text order alone cannot capture.

**SROIE layout gain is modest:**
Receipts are largely linear/sequential, so spatial features add only +0.4 pp Macro F1 over text-only. The main gain is in Micro F1 (+0.9 pp) reflecting better entity boundary detection.

**CORD benefits from layout on micro F1:**
CORD's fine-grained menu/total hierarchy sees the largest absolute micro F1 boost (+12.7 pp) from layout, though macro gain is limited by the class imbalance (MENU dominates).

**Macro F1 is systematically lower than Micro F1:**
All datasets have heavy `O`-token dominance. The macro metric penalizes poor performance on rare entity classes (e.g., CORD's VOID with only 4 instances).

**These baselines deliberately set a low bar:**
TF-IDF has no cross-token context, no image features, and no pretrained representations. LayoutLMv3 is expected to exceed these numbers substantially.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Known Limitations

1. **No image features in baselines** — Phases 1–3 use only OCR text and bounding boxes. Visual CNN features are reserved for Phase 4 (LayoutLMv3).
2. **SROIE macro F1 is misleadingly low** — The dataset has heavy `O` dominance (~82% of tokens). The model correctly classifies most tokens but struggles on minority entity classes.
3. **FUNSD val split is small (22 documents)** — Validation F1 has high variance. Test results on 50 documents are more reliable.
4. **CORD VOID class** — Only 4 VOID-labeled tokens exist across the entire dataset. Any model will score 0.0 on this class; macro F1 is penalized accordingly.
5. **No cross-dataset evaluation** — Each model is trained and evaluated within its own dataset. The unified taxonomy (`label_mapping.py`) is defined for Phase 4 cross-dataset training.
6. **TF-IDF is bag-of-words** — No positional context between adjacent tokens. A window-based or sequence model would perform better even before using Transformers.

---

## Recommendations for Phase 4 (LayoutLMv3)

1. **Use LayoutLMv3** (`microsoft/layoutlmv3-base`) — jointly encodes text tokens, bounding boxes, and image patches in a single Transformer. Expected to significantly outperform these baselines.
2. **Fine-tune per dataset** — Start with individual fine-tuning on SROIE, FUNSD, CORD separately before attempting cross-dataset transfer.
3. **Address class imbalance** — Use weighted loss or focal loss. The `O` token dominance will cause the model to underpredict rare entity classes.
4. **Use seqeval for NER evaluation** — Switch from token-level sklearn F1 to `seqeval` entity-span F1, which is the standard for NER benchmarks and gives more interpretable results.
5. **Image augmentation** — Apply light augmentation (rotation ±5°, brightness, blur) during fine-tuning to handle scan quality variance, especially for SROIE's high resolution variance.
6. **Streamlit app** — Expose the fine-tuned model through a Streamlit interface accepting uploaded document images, running OCR (via pytesseract or EasyOCR), and displaying extracted entities with bounding box overlays.

---

## Reproducibility

All preprocessing uses `seed=42` for train/val splits. Training uses `random_state=42`. Configuration is managed through `configs/preprocessing.yaml`. Raw datasets are never modified.

```bash
# Full reproducible pipeline from scratch
python datasets/sroie/download_sroie.py
python datasets/funsd/download_funsd.py
python datasets/cord/download_cord.py
python -m preprocessing.unified_dataset
python scripts/validate_phase2.py
python training/train_baseline.py --datasets sroie funsd cord
python training/evaluate_baseline.py --datasets sroie funsd cord
```

---

## Attribution

- SROIE dataset: ICDAR 2019 Competition on Scanned Receipt OCR and Information Extraction
- FUNSD dataset: Guillaume Jaume, Hazim Kemal Ekenel, Jean-Philippe Thiran — "FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents", ICDAR 2019
- CORD dataset: Seunghyun Park et al. — "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing", NeurIPS 2019 Workshop

---

## Phase 4 — LayoutLMv3 Multimodal Transformer

Phase 4 fine-tunes `microsoft/layoutlmv3-base` on each dataset using the
multimodal pipeline built in Phases 1–3. It produces:
- A per-dataset fine-tuned model in `models/checkpoints/layoutlmv3/`
- Test-set seqeval span F1 metrics
- Entity-level extraction accuracy (SROIE)
- Error analysis visualizations in `reports/error_analysis/`
- A combined model comparison report in `reports/model_comparison.csv`

### Install Phase 4 dependencies

```bash
pip install -r requirements.txt
```

New packages added in Phase 4: `transformers>=4.38.0`, `seqeval>=1.2.2`,
`accelerate>=0.27.0`, `sentencepiece>=0.2.0`, `timm>=0.9.0`.

### Stage A — Smoke test (always run first)

```bash
python training/train_layoutlmv3.py --dataset sroie --stage smoke
```

Trains 15 samples for 2 epochs. Verifies forward pass, loss, backward pass,
checkpoint save. Expected completion: ~8 min on CPU, <1 min on GPU.

### Stage B — Small experiment

```bash
python training/train_layoutlmv3.py --dataset sroie --stage small
```

Trains 30 samples for 3 epochs. Verifies convergence and memory usage.
Expected completion: ~25 min on CPU.

### Stage C — Full training

```bash
# Train one dataset at a time
python training/train_layoutlmv3.py --dataset sroie --stage full
python training/train_layoutlmv3.py --dataset funsd --stage full
python training/train_layoutlmv3.py --dataset cord  --stage full

# Or all together
python training/train_layoutlmv3.py --dataset sroie funsd cord --stage full
```

> **Note:** Full training on CPU takes 8+ hours per dataset.
> A CUDA GPU with ≥8 GB VRAM is strongly recommended.
> With a GPU, all three datasets train in under 2 hours total.

### Evaluate on test splits

```bash
python evaluation/evaluate_layoutlmv3.py --dataset sroie funsd cord --stage full
```

Generates:
- `reports/layoutlmv3_{dataset}_per_entity.json`
- `reports/layoutlmv3_{dataset}_classification_report.txt`
- `reports/model_comparison.csv` (includes Phase 3 baseline results)
- `reports/error_analysis/{dataset}/correct_samples.json`
- `reports/error_analysis/{dataset}/incorrect_samples.json`
- `reports/error_analysis/{dataset}/sample_*_pred.png`

### Phase 4 directory structure additions

```
models/
└── layoutlmv3/
    ├── __init__.py
    ├── config.py       — LayoutLMv3DatasetConfig + LayoutLMv3TrainingConfig
    ├── tokenizer.py    — Processor wrapper, label/bbox alignment
    ├── dataset.py      — PyTorch Dataset + DataLoader factory
    └── model.py        — LayoutLMv3Classifier wrapper

models/checkpoints/layoutlmv3/
└── {dataset}/{stage}/  — Saved weights + processor + checkpoint_meta.json

training/
└── train_layoutlmv3.py — Smoke/small/full training pipeline

evaluation/
├── layoutlmv3_metrics.py    — seqeval span F1, entity accuracy
└── evaluate_layoutlmv3.py   — Checkpoint loading + test evaluation

inference/
└── layoutlmv3_predict.py    — Entity reconstruction with probabilities

configs/
├── layoutlmv3.yaml          — Training hyperparameters
└── labels/
    ├── sroie.yaml            — 9-class label schema
    ├── funsd.yaml            — 7-class label schema
    └── cord.yaml             — 13-class label schema

experiments/
└── layoutlmv3_{dataset}_{stage}_{timestamp}/
    └── training_summary.json

reports/
├── model_comparison.csv          — Phase 3 + Phase 4 results side-by-side
├── layoutlmv3_*_per_entity.json
├── layoutlmv3_*_classification_report.txt
└── error_analysis/{dataset}/
    ├── correct_samples.json
    ├── incorrect_samples.json
    └── sample_*_pred.png

tests/
├── test_token_alignment.py       — 15 alignment tests
├── test_entity_reconstruction.py — 12 reconstruction tests
├── test_dataset.py               — 8 dataset tests
└── test_model.py                 — 19 config/model tests
```

### Phase 4 results

| Dataset | Model | Macro F1 | Micro F1 | Notes |
|---------|-------|----------|----------|-------|
| SROIE | OCR Baseline | 0.750 | 0.750 | Phase 3 |
| SROIE | Text-Only (TF-IDF) | 0.262 | 0.919 | Phase 3 |
| SROIE | Text+Layout (TF-IDF) | 0.266 | 0.928 | Phase 3 |
| SROIE | LayoutLMv3 (smoke) | — | — | 2 epochs, 15 samples — pipeline verified |
| SROIE | LayoutLMv3 (full) | — | — | Run: `--stage full` |
| FUNSD | Text-Only (TF-IDF) | 0.351 | 0.402 | Phase 3 |
| FUNSD | Text+Layout (TF-IDF) | 0.500 | 0.540 | Phase 3 |
| FUNSD | LayoutLMv3 (full) | — | — | Run: `--stage full` |
| CORD | Text-Only (TF-IDF) | 0.327 | 0.684 | Phase 3 |
| CORD | Text+Layout (TF-IDF) | 0.377 | 0.811 | Phase 3 |
| CORD | LayoutLMv3 (full) | — | — | Run: `--stage full` |

> Phase 4 F1 cells populate after running `evaluate_layoutlmv3.py`.
> Published benchmarks: LayoutLMv3-base achieves ~0.92 on FUNSD, ~0.96 on CORD.

### Running all Phase 4 tests

```bash
python -m pytest tests/test_token_alignment.py tests/test_entity_reconstruction.py \
                 tests/test_dataset.py tests/test_model.py -v
```

**54/54 tests pass** (verified, no network access required).
