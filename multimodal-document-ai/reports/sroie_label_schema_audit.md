# SROIE Label Schema and Label Alignment Audit

**Phase 5 — Step 2 Audit Report**  
**Dataset**: ICDAR 2019 SROIE (jsdnrs/ICDAR2019-SROIE)  
**Evaluator & Model**: LayoutLMv3-base Full Checkpoint (`models/checkpoints/layoutlmv3/sroie/full`)  
**Audit Date**: 2026-09-08  

---

## 1. Executive Summary & Audit Verdict

* **Audit Verdict**: **`PASS WITH CAVEAT`** (Significant Preprocessing Alignment Artifact Discovered)
* **Underlying Bug**: In `preprocessing/sroie_processor.py`, `align_sroie_entity_labels()` employs a greedy substring matching rule (`nw in norm_val`) that caused **93 out of 361 test samples (25.8%)** and **144 out of 533 train samples (27.0%)** to incorrectly assign the `B-TOTAL` label to stray single/double-character digits (e.g. `'1'`, `'0'`, `'$'`) appearing earlier on the receipt rather than the true total price line.
* **Absence of `I-` Tags**: Because SROIE OCR words were ingested as line-level strings and the matching loop terminates immediately upon finding a substring (`break`), **0 tokens** in train, val, or test were assigned `I-COMPANY`, `I-ADDRESS`, `I-DATE`, or `I-TOTAL`. The effective dataset exercises only 5 labels (`O`, `B-COMPANY`, `B-ADDRESS`, `B-DATE`, `B-TOTAL`).
* **Multi-line Address Truncation**: When addresses span multiple OCR lines, only the first line matching the substring receives `B-ADDRESS`; subsequent lines are labeled `O`.

---

## 2. Label Schemas & ID Consistency

### 2.1 Config vs Model Checkpoint Mapping
* **Config Path**: `configs/labels/sroie.yaml`
* **Model Checkpoint**: `models/checkpoints/layoutlmv3/sroie/full/config.json`
* **Total Labels Defined**: 9

| Label ID | Config Label String | Checkpoint `id2label` | Match Status |
| :---: | :--- | :--- | :---: |
| 0 | `O` | `O` | **MATCH** |
| 1 | `B-COMPANY` | `B-COMPANY` | **MATCH** |
| 2 | `I-COMPANY` | `I-COMPANY` | **MATCH** |
| 3 | `B-DATE` | `B-DATE` | **MATCH** |
| 4 | `I-DATE` | `I-DATE` | **MATCH** |
| 5 | `B-ADDRESS` | `B-ADDRESS` | **MATCH** |
| 6 | `I-ADDRESS` | `I-ADDRESS` | **MATCH** |
| 7 | `B-TOTAL` | `B-TOTAL` | **MATCH** |
| 8 | `I-TOTAL` | `I-TOTAL` | **MATCH** |

Label IDs are 100% consistent across `configs/labels/sroie.yaml`, `preprocessing/label_mapping.py`, training dataloaders, checkpoint configs, and evaluation scripts.

---

## 3. Empirical Split Label Distributions

### 3.1 Token Counts by Label
| Label | Train (533 docs) | Val (93 docs) | Test (361 docs) | Total Tokens | % of All Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `O` | 26,369 | 4,770 | 17,951 | 49,090 | 92.60% |
| `B-TOTAL` | 532 | 93 | 361 | 986 | 1.86% |
| `B-ADDRESS` | 530 | 93 | 360 | 983 | 1.85% |
| `B-DATE` | 532 | 92 | 359 | 983 | 1.85% |
| `B-COMPANY` | 522 | 93 | 355 | 970 | 1.83% |
| `I-COMPANY` | 0 | 0 | 0 | 0 | 0.00% |
| `I-ADDRESS` | 0 | 0 | 0 | 0 | 0.00% |
| `I-DATE` | 0 | 0 | 0 | 0 | 0.00% |
| `I-TOTAL` | 0 | 0 | 0 | 0 | 0.00% |
| **Total Words** | **28,485** | **5,141** | **19,386** | **53,012** | **100.00%** |

### 3.2 BIO/IOB2 Sequence Validity
* **Illegal Transitions Check**:
  - `I-X` without preceding `B-X` or `I-X`: **0**
  - `I-X` following `B-Y` ($X \neq Y$): **0**
  - Sequence integrity: Formally valid BIO/IOB2 sequences across 100% of samples (since all entity spans are single-word B- tags).

---

## 4. Analysis of Preprocessing & Alignment Anomalies

### 4.1 Root Cause of the Missing `I-` Tags
In raw ICDAR 2019 SROIE (`jsdnrs/ICDAR2019-SROIE`), OCR lines were packaged as entries in `words` (e.g. `'TAN CHONG MOTOR ASSEMBLIES SDN BHD'` is a single entry in `words`).  
In `preprocessing/sroie_processor.py`:
```python
for idx, nw in enumerate(norm_words):
    if labels[idx] != "O":
        continue
    if nw == norm_val or norm_val in nw or nw in norm_val:
        labels[idx] = tag_b
        break
```
Because the multi-word string was matched as a single entry in `words`, only `tag_b` (`B-COMPANY`) was assigned, followed by `break`. As a result:
- 46.5% of OCR entries in SROIE contain multiple whitespace-separated words.
- All entities in the processed JSONL are 1 token in length.
- The `I-` labels have 0 support in training and test sets.

### 4.2 The Greedy Substring Matching Flaw in `B-TOTAL`
The condition `nw in norm_val` was intended to match partially truncated OCR values. However:
- If ground truth total is `'193.00'`, any early OCR token such as `'1'`, `'9'`, `'3'`, or `'.'` satisfies `'1' in '193.00' == True`.
- Because the loop runs top-to-bottom through `norm_words`, the **first** token on the receipt that is a substring of the total amount receives `B-TOTAL`!
- **Empirical Quantification**:
  - In Test: **93 / 361 samples (25.8%)** have `B-TOTAL` assigned to a 1-2 character token that does not equal the true total (e.g., `'1'`, `'0'`, `':'`).
  - In Train: **144 / 533 samples (27.0%)** have this exact misalignment.

#### Concrete Example from Test Set: Sample `X00016469670`
* **Ground Truth Total**: `'193.00'`
* **Token [26]**: `'1'` $\rightarrow$ Labeled `B-TOTAL` (because `'1' in '193.00'`)
* **Token [30]**: `'TOTAL 193.00'` $\rightarrow$ Labeled `O`
* **Result**: The ground truth annotation in `test.jsonl` was corrupted by pointing to token `'1'` instead of the real total line `'TOTAL 193.00'`. When the trained LayoutLMv3 model correctly predicts `'TOTAL 193.00'` as TOTAL, strict span evaluation penalizes it with both a False Positive and a False Negative!

This precisely explains the lower reported test metric for TOTAL:
* `COMPANY` F1: `0.9333`
* `ADDRESS` F1: `0.9384`
* `DATE` F1: `0.9639`
* `TOTAL` F1: **`0.6957`** (dragged down by the ~26% misaligned ground-truth labels).

### 4.3 Multi-Line Address Truncation
For addresses that span 3–5 lines on the receipt (e.g. `X00016469670`), `nw in norm_val` matches the first line (`'NO 2 & 4'`), tags it `B-ADDRESS`, and terminates. The subsequent address lines (`'JALAN BAYU 4'`, `'BANDAR SERI ALAM'`, `'MASAI, JOHOR'`) remain `O`.

---

## 5. Handling of Special Tokens, Padding, and Word Reconstruction

1. **Subword Tokenization**: `LayoutLMv3TokenizerWrapper` applies the first-subword rule:
   - First subword token receives the word's label ID (`0..8`).
   - Subsequent subwords receive `IGNORE_INDEX = -100`.
   - Special tokens (`[CLS]`, `[SEP]`, `<pad>`) receive `IGNORE_INDEX = -100` and `bbox = [0, 0, 0, 0]`.
2. **Evaluation Word Reconstruction**: `evaluate_layoutlmv3.py` invokes `extract_word_level_labels()` and `token_preds_to_word_preds()`:
   - Uses stored tensor `word_ids` to map subwords back to word positions.
   - Ignores all tokens where `lid == -100` or `wid is None`.
   - Passes clean word-level sequences to `seqeval`.
   - Special tokens and padding are strictly excluded from all metric calculations.

---

## 6. Audit Summary

| Check Item | Requirement | Status | Evidence / Notes |
| :--- | :--- | :---: | :--- |
| **Label ID Mapping** | Config, Model, Dataset match | **PASS** | 9 labels identical across YAML, checkpoint, and code. |
| **BIO Validity** | Valid transitions, no illegal I- | **PASS** | 0 illegal transitions across all 987 samples. |
| **Special Token Handling** | Excluded from evaluation | **PASS** | `-100` mask and word-level collapsing correctly applied. |
| **Entity Representation** | Correct, contiguous spans | **CAVEAT** | Single-word lines only; 0 `I-` labels generated. |
| **TOTAL Label Alignment** | Ground truth matches real total | **CAVEAT** | 25.8% test samples misaligned due to greedy `nw in norm_val`. |
