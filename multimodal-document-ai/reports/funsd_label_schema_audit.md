# FUNSD Label Schema and Label Alignment Audit

**Phase 5 — Step 2 Audit Report**  
**Dataset**: ICDAR 2019 FUNSD (`nielsr/funsd`)  
**Evaluator & Model**: LayoutLMv3-base Full Checkpoint (`models/checkpoints/layoutlmv3/funsd/full`)  
**Audit Date**: 2026-09-08  

---

## 1. Executive Summary & Audit Verdict

* **Audit Verdict**: **`PASS`** (Flawless Label Alignment & Schema Fidelity)
* **Finding**: The FUNSD dataset demonstrates perfect alignment between raw HuggingFace annotations and processed JSONL datasets.
* **Schema Integrity**: The 7-class IOB2 schema (`O`, `B-HEADER`, `I-HEADER`, `B-QUESTION`, `I-QUESTION`, `B-ANSWER`, `I-ANSWER`) is completely preserved across all train, validation, and test splits.
* **BIO Transitions**: Exactly **0 illegal transitions** across all 199 documents and 30,595 words.
* **Relation Metadata**: The raw dataset contains no entity relation links in the token NER feature (`relations: []`). Relation metadata is not conflated with token classification labels.
* **ID Mapping**: Checkpoint `id2label` matches `configs/labels/funsd.yaml` 1-to-1.

---

## 2. Label Schemas & ID Consistency

### 2.1 Config vs Raw HuggingFace vs Checkpoint
* **Config Path**: `configs/labels/funsd.yaml`
* **Source**: `nielsr/funsd` feature `ner_tags`
* **Model Checkpoint**: `models/checkpoints/layoutlmv3/funsd/full/config.json`
* **Total Labels Defined**: 7

| Label ID | Raw HF `ner_tags` Name | Config Label String | Checkpoint `id2label` | Match Status |
| :---: | :--- | :--- | :--- | :---: |
| 0 | `O` | `O` | `O` | **MATCH** |
| 1 | `B-HEADER` | `B-HEADER` | `B-HEADER` | **MATCH** |
| 2 | `I-HEADER` | `I-HEADER` | `I-HEADER` | **MATCH** |
| 3 | `B-QUESTION` | `B-QUESTION` | `B-QUESTION` | **MATCH** |
| 4 | `I-QUESTION` | `I-QUESTION` | `I-QUESTION` | **MATCH** |
| 5 | `B-ANSWER` | `B-ANSWER` | `B-ANSWER` | **MATCH** |
| 6 | `I-ANSWER` | `I-ANSWER` | `I-ANSWER` | **MATCH** |

---

## 3. Empirical Split Label Distributions

### 3.1 Token Counts by Label Across Splits
| Label | Train (127 docs) | Val (22 docs) | Test (50 docs) | Total Tokens | % of All Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `I-ANSWER` | 5,817 | 970 | 2,485 | 9,272 | 30.31% |
| `I-QUESTION` | 3,450 | 558 | 1,584 | 5,592 | 18.28% |
| `O` | 3,035 | 597 | 2,385 | 6,017 | 19.67% |
| `B-QUESTION` | 2,781 | 472 | 1,070 | 4,323 | 14.13% |
| `B-ANSWER` | 2,366 | 366 | 809 | 3,541 | 11.57% |
| `I-HEADER` | 847 | 188 | 255 | 1,290 | 4.22% |
| `B-HEADER` | 372 | 69 | 119 | 560 | 1.83% |
| **Total Words** | **18,668** | **3,220** | **8,707** | **30,595** | **100.00%** |

### 3.2 Entity Span Counts (B- Tags)
| Entity Category | Train Spans | Val Spans | Test Spans | Total Spans |
| :--- | :---: | :---: | :---: | :---: |
| `QUESTION` | 2,781 | 472 | 1,070 | 4,323 |
| `ANSWER` | 2,366 | 366 | 809 | 3,541 |
| `HEADER` | 372 | 69 | 119 | 560 |
| **Total Spans** | **5,519** | **907** | **1,998** | **8,424** |

### 3.3 BIO/IOB2 Sequence Validity
* **Illegal Transitions Check**:
  - `I-X` without preceding `B-X` or `I-X`: **0**
  - `I-X` following `B-Y` ($X \neq Y$): **0**
  - `I-X` following `O`: **0**
  - **Verdict**: 100% valid BIO/IOB2 sequences across all splits.

---

## 4. Verification Checklist & Invariants

1. **Entity Boundaries & Multi-word Spans**: Multi-word questions, answers, and headers properly maintain `B-` followed by contiguous `I-` tokens matching the original document structure.
2. **No Silent Loss**: The total number of words in processed data exactly equals the raw token count in the HuggingFace source splits.
3. **Relation Information**: The `relations` list in processed JSONL is empty (`[]`). Token classification relies strictly on the 7-class NER tags without leaking graph edge connectivity.
4. **Special Tokens & Padding**: 
   - Assigned `IGNORE_INDEX = -100` during tokenization.
   - Evaluated strictly at the word level via `extract_word_level_labels()`.
   - Special tokens and padding are excluded from seqeval entity F1 and sklearn token metrics.

---

## 5. Audit Summary

| Check Item | Requirement | Status | Evidence / Notes |
| :--- | :--- | :---: | :--- |
| **Label ID Mapping** | Config, Model, Dataset match | **PASS** | 7 labels match 100% across all files. |
| **BIO Validity** | Valid transitions, no illegal I- | **PASS** | 0 illegal transitions across 30,595 tokens. |
| **Span Preservation** | Contiguous B/I spans | **PASS** | Multi-word questions, answers, headers intact. |
| **Entity Loss** | No dropped tokens or entities | **PASS** | Word counts and label distributions match raw HF. |
| **Relation Leakage** | Relations not used as labels | **PASS** | Pure token classification schema verified. |
| **Special Token Masking** | Padding excluded | **PASS** | Special tokens mapped to -100 and stripped. |
