# Phase 1 — Dataset Setup and Exploration

## Overview

Phase 1 establishes the data foundation for the Multimodal Transformer-Based Intelligent Document Understanding system. Three benchmark datasets are downloaded, explored, and statistically analyzed.

---

## Datasets

### 1. SROIE (ICDAR 2019)
- **Source:** `jsdnrs/ICDAR2019-SROIE` (HuggingFace)
- **Task:** Key Information Extraction from scanned receipts
- **Labels:** `company`, `date`, `address`, `total`
- **Splits:** Train: 626 | Test: 361 | **Total: 987**
- **Avg Words/Doc:** 53.7 | Total Words: 53,012
- **Image Resolution:** 435–4961 px wide, 605–7016 px tall (high variance)
- **Entity Coverage:** company/date 100%, address/total 99.9%

### 2. FUNSD (Form Understanding)
- **Source:** `nielsr/funsd` (HuggingFace)
- **Task:** Semantic entity labeling on noisy scanned forms
- **Labels:** B/I-HEADER, B/I-QUESTION, B/I-ANSWER, O
- **Splits:** Train: 149 | Test: 50 | **Total: 199**
- **Avg Words/Doc:** 153.7 | Total Words: 30,595
- **Image Resolution:** Standardized to 1000px height
- **Label Imbalance:** I-ANSWER dominates (30.3%)

### 3. CORD-v2 (Consolidated Receipt Dataset)
- **Source:** `naver-clova-ix/cord-v2` (HuggingFace)
- **Task:** Fine-grained receipt understanding
- **Labels:** B/I-MENU, B/I-TOTAL, B/I-SUBTOTAL, B/I-VOID
- **Splits:** Train: 800 | Val: 100 | Test: 100 | **Total: 1000**
- **Avg Words/Doc:** 23.9 | Total Words: 23,909
- **MENU dominance:** 58.4% of all tokens

---

## Cross-Dataset Comparison

| Metric            | SROIE  | FUNSD  | CORD   |
|-------------------|--------|--------|--------|
| Total Samples     | 987    | 199    | 1,000  |
| Avg Words/Doc     | 53.7   | 153.7  | 23.9   |
| Task Type         | KIE    | NER    | NER    |
| Native Val Split  | No     | No     | Yes    |
| Resolution Variance | High | Low    | Medium |

---

## Reports Generated

| Report | Path |
|--------|------|
| SROIE JSON | `reports/sroie_report.json` |
| SROIE CSV  | `reports/sroie_report.csv` |
| SROIE entity dist. PNG | `reports/sroie_entity_distribution.png` |
| FUNSD JSON | `reports/funsd_report.json` |
| FUNSD label dist. PNG  | `reports/funsd_label_distribution.png` |
| CORD JSON  | `reports/cord_report.json` |
| CORD label dist. PNG   | `reports/cord_label_distribution.png` |

---

## Phase 1 Validation Results

```
==========================================
STARTING PHASE 1 DATASET INTEGRITY CHECKS
==========================================
SROIE Validation: PASSED (987/987 valid)
FUNSD Validation: PASSED (199/199 valid)
CORD Validation:  PASSED (1000/1000 valid)
==========================================
ALL PHASE 1 VALIDATION CHECKS PASSED SUCCESSFULLY!
==========================================
```

---

## Key Findings

1. CORD and SROIE are 5-10x larger than FUNSD — expect lower baseline reliability on FUNSD.
2. FUNSD has the highest token density per document (154 words avg).
3. SROIE has high resolution variance requiring normalization.
4. All datasets show class imbalance; CORD's VOID labels (4 tokens total) require zero-shot handling in evaluation.
5. Only CORD provides an official validation split; SROIE/FUNSD use seed=42 random split (15% val).
