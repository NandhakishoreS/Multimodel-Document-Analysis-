# Phase 2 — Data Preprocessing and Unification

## Overview

Phase 2 transforms the raw HuggingFace datasets into a **unified JSON-Lines (JSONL) schema** suitable for training all baseline and transformer models. The pipeline handles bounding-box normalization, text cleaning, cross-dataset label alignment, and reproducible train/val/test splitting.

---

## Unified Sample Schema

Every processed sample follows this structure:

```json
{
  "doc_id": "string",
  "dataset": "sroie | funsd | cord",
  "split": "train | val | test",
  "image_path": "processed_data/<dataset>/images/<id>.png",
  "width": 1000,
  "height": 1000,
  "words": ["token1", "token2", ...],
  "bboxes": [[x0, y0, x1, y1], ...],
  "labels": ["B-COMPANY", "I-COMPANY", "O", ...]
}
```

- `bboxes` are normalized to the **[0, 1000]** integer range on both axes.
- `labels` use the dataset's native BIO label scheme (aligned to unified taxonomy in Phase 3).

---

## Spatial Normalization

All bounding boxes are normalized to a canonical `[0, 1000] × [0, 1000]` coordinate space:

```
x_norm = round(x_raw / image_width  * 1000)
y_norm = round(y_raw / image_height * 1000)
```

This ensures model inputs are resolution-invariant. The normalization is implemented in [`preprocessing/bbox_utils.py`](file:///c:/MDA/multimodal-document-ai/preprocessing/bbox_utils.py):

- `normalize_bbox(bbox, width, height)` ? `[x0, y0, x1, y1]` in [0, 1000]
- `denormalize_bbox(bbox, width, height)` ? pixel coordinates
- `validate_bbox(bbox)` ? checks for valid, non-degenerate boxes
- `compute_iou(bbox_a, bbox_b)` ? Intersection-over-Union
- `get_relative_spatial_features(anchor, target)` ? 8-dimensional spatial context feature vector

---

## Text Normalization

Implemented in [`preprocessing/text_utils.py`](file:///c:/MDA/multimodal-document-ai/preprocessing/text_utils.py):

- **Unicode NFKC normalization** — standardizes characters across encodings
- **Whitespace stripping** — removes leading/trailing/extra spaces
- **Lexical feature extraction** — is_numeric, is_alpha, has_digit, has_punct, is_uppercase, char_n-grams
- **Regex features** — currency, date, time, email, URL pattern matching

---

## Label Mapping

Implemented in [`preprocessing/label_mapping.py`](file:///c:/MDA/multimodal-document-ai/preprocessing/label_mapping.py):

### Dataset-Native Encoders

| Dataset | Labels |
|---------|--------|
| SROIE   | O, B-COMPANY, I-COMPANY, B-DATE, I-DATE, B-ADDRESS, I-ADDRESS, B-TOTAL, I-TOTAL |
| FUNSD   | O, B-HEADER, I-HEADER, B-QUESTION, I-QUESTION, B-ANSWER, I-ANSWER |
| CORD    | O, B-MENU, I-MENU, B-TOTAL, I-TOTAL, B-SUBTOTAL, I-SUBTOTAL, B-VOID, I-VOID |

### Unified Cross-Dataset Taxonomy

For cross-dataset evaluation, labels are aligned to a shared taxonomy:

| Unified Label   | SROIE Source     | FUNSD Source   | CORD Source  |
|-----------------|-----------------|----------------|-------------|
| O               | O               | O              | O           |
| B-KEY_VALUE     | B-COMPANY, etc. | B-QUESTION     | —           |
| B-ENTITY        | —               | B-ANSWER       | B-MENU, etc.|
| B-STRUCTURE     | —               | B-HEADER       | —           |
| B-AMOUNT        | B-TOTAL         | —              | B-TOTAL     |

---

## Dataset Processors

| Processor | Description |
|-----------|-------------|
| [`sroie_processor.py`](file:///c:/MDA/multimodal-document-ai/preprocessing/sroie_processor.py) | Converts SROIE global KV entities to token-level BIO tags via sequence matching |
| [`funsd_processor.py`](file:///c:/MDA/multimodal-document-ai/preprocessing/funsd_processor.py) | Maps FUNSD word-level annotations to unified schema |
| [`cord_processor.py`](file:///c:/MDA/multimodal-document-ai/preprocessing/cord_processor.py) | Parses CORD's nested `ground_truth` JSON string, converts quad polygons to bboxes |

---

## Split Strategy

| Dataset | Train | Val | Test |
|---------|-------|-----|------|
| SROIE   | 533 (85% of 626 train) | 93 (15% of 626 train) | 361 (official test) |
| FUNSD   | 127 (85% of 149 train) | 22 (15% of 149 train) | 50 (official test) |
| CORD    | 800 (official) | 100 (official) | 100 (official) |

- SROIE/FUNSD: random 85/15 train/val split with **seed=42** for reproducibility
- CORD: uses the official HuggingFace `validation` split directly
- No data leakage — unique document IDs verified across splits

---

## Processed Output

```
processed_data/
+-- sroie/
¦   +-- train.jsonl  (533 samples)
¦   +-- val.jsonl    (93 samples)
¦   +-- test.jsonl   (361 samples)
¦   +-- images/      (987 PNGs)
+-- funsd/
¦   +-- train.jsonl  (127 samples)
¦   +-- val.jsonl    (22 samples)
¦   +-- test.jsonl   (50 samples)
¦   +-- images/      (199 PNGs)
+-- cord/
    +-- train.jsonl  (800 samples)
    +-- val.jsonl    (100 samples)
    +-- test.jsonl   (100 samples)
    +-- images/      (1000 PNGs)
```

---

## UnifiedDocumentDataset (PyTorch)

[`preprocessing/unified_dataset.py`](file:///c:/MDA/multimodal-document-ai/preprocessing/unified_dataset.py) provides `UnifiedDocumentDataset`, a `torch.utils.data.Dataset` subclass:

```python
from preprocessing.unified_dataset import UnifiedDocumentDataset

ds = UnifiedDocumentDataset("processed_data/sroie/train.jsonl", load_images=True)
sample = ds[0]
# sample keys: doc_id, dataset, split, image_path, width, height, words, bboxes, labels, image
```

---

## Configuration

[`configs/preprocessing.yaml`](file:///c:/MDA/multimodal-document-ai/configs/preprocessing.yaml):

```yaml
bbox_normalization:
  range: 1000        # normalize to [0, 1000]

dataset_split:
  random_seed: 42
  validation_ratio: 0.15

paths:
  processed_data_dir: processed_data
```
