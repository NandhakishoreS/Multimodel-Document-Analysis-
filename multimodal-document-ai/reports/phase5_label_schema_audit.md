# Cross-Dataset Label Schema & Alignment Audit

**Phase 5 — Step 2 Cross-Dataset Synthesis**  
**Datasets**: SROIE, FUNSD, CORD  
**Model Architecture**: LayoutLMv3-base Token Classification  
**Audit Date**: 2026-09-08  

---

## 1. Cross-Dataset Comparison Matrix

| Dataset | Original Labels | Processed Labels | Final Model Labels | BIO Valid? | Entity Coverage | Special-Token Handling | Potential Issues & Caveats |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| **SROIE** | 4 string entity fields (`company`, `date`, `address`, `total`) + raw OCR boxes | 5 active labels (`O`, `B-COMPANY`, `B-DATE`, `B-ADDRESS`, `B-TOTAL`); 0 `I-` labels | 9 classes in config/model (`O` + 4 B/I pairs) | **Yes** (0 illegal transitions) | High entity span recall, but single-word lines only; addresses truncated to line 1 | First subword labeled; specials & continuations receive `-100`; word-level seqeval | **1.** `nw in norm_val` in `align_sroie_entity_labels()` caused greedy matching on 25.8% test samples for `B-TOTAL`, tagging early digits (e.g. `'1'`) rather than the true price line. <br>**2.** Zero `I-` labels exist in dataset. |
| **FUNSD** | 7 raw NER tags (`O`, `B/I-HEADER`, `B/I-QUESTION`, `B/I-ANSWER`) | Exact 7 IOB2 labels matching raw HF dataset | 7 classes (`O` + 3 B/I pairs) | **Yes** (0 illegal transitions) | 100% full coverage; multi-word spans preserved across questions, answers, headers | First subword labeled; specials & continuations receive `-100`; word-level seqeval | **None.** Clean pass across all splits, schemas, and metrics. |
| **CORD** | 24 fine-grained hierarchical categories in `valid_line` (e.g. `menu.nm`, `total.total_price`) | 3 active coarse categories (`MENU`, `SUBTOTAL`, `TOTAL`); 6 active labels (`B/I-`) | 13 classes defined in config/model | **Yes** (0 illegal transitions) | 100% of words belong to entity lines; zero unannotated background text | First subword labeled; specials & continuations receive `-100`; word-level seqeval | **1.** Words are pre-filtered from `valid_line`; 0% `O` tokens exist. Model performs 3-way line classification without background rejection. <br>**2.** 24 fine-grained classes collapsed into 3 coarse classes. Explains high 0.9733 F1. |

---

## 2. In-Depth Comparability Analysis

### 2.1 Why the Datasets Are NOT Directly Comparable
Although all three datasets are framed as token classification under an IOB2 tag schema, their underlying task definitions and evaluation regimes differ fundamentally:

1. **Background Noise Ratio (`O` Token Dominance)**:
   - **SROIE**: `O` accounts for **92.60%** of all tokens. The model must perform severe needle-in-a-haystack filtering.
   - **FUNSD**: `O` accounts for **19.67%** of tokens. Real-world forms have moderate background text, but questions and answers are dense.
   - **CORD**: `O` accounts for **0.00%** of tokens. Because only annotated `valid_line` entries were extracted during preprocessing, background noise is completely absent.

2. **Entity Granularity & Structural Complexity**:
   - **SROIE**: Single-instance document-level KIE. Each receipt typically has exactly one Company, one Date, one Address, and one Total.
   - **FUNSD**: Dense, multi-entity form understanding with complex 2D reading order, checkboxes, and multi-column tables.
   - **CORD**: Coarse receipt line classification (Menu vs Subtotal vs Total) without entity-linking or fine-grained key-value extraction.

3. **Subword & Span Alignment Semantics**:
   - In SROIE, OCR "words" are multi-word text blocks from the OCR engine, resulting in zero `I-` tags.
   - In FUNSD and CORD, words are individual space-separated tokens, properly exercising `B-` and `I-` transitions.

---

## 3. Classification of Dataset Audit Status

* **SROIE**: **`PASS WITH CAVEAT`**  
  - *Reason*: Schema and label IDs are consistent, and BIO sequences are valid. However, the greedy substring matching flaw in `sroie_processor.py` mislabeled the ground-truth `B-TOTAL` on 25.8% of test samples, artificially lowering the reported test TOTAL F1 to 0.6957.
* **FUNSD**: **`PASS`**  
  - *Reason*: Flawless schema alignment, zero data loss, zero illegal transitions, exact 1-to-1 match with HuggingFace annotations.
* **CORD**: **`PASS WITH CAVEAT`**  
  - *Reason*: Formally valid and leak-free in inputs, but the dataset is heavily simplified (100% pre-filtered valid lines, 0% background `O` tokens, 24 fine-grained classes merged into 3 coarse classes), explaining the high 0.9733 F1.
