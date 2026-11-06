# CORD Label Schema and Label Alignment Audit

**Phase 5 — Step 2 Audit Report**  
**Dataset**: Consolidated Receipt Dataset (CORD-v2, `naver-clova-ix/cord-v2`)  
**Evaluator & Model**: LayoutLMv3-base Full Checkpoint (`models/checkpoints/layoutlmv3/cord/full`)  
**Audit Date**: 2026-09-08  

---

## 1. Executive Summary & Audit Verdict

* **Audit Verdict**: **`PASS WITH CAVEAT`** (Structural Dataset Simplification via Filtering)
* **High Score Explanation (0.9733 F1)**: The near-perfect performance on CORD is directly explained by the dataset construction in `preprocessing/cord_processor.py`:
  1. **Zero Negative (`O`) Tokens**: Only words belonging to `valid_line` in the raw ground truth JSON were extracted into the processed dataset. All unannotated background text on the receipts (store headers, addresses, dates, cash register metadata, footers) was omitted. As a result, the model was evaluated purely on pre-filtered valid receipt entity lines and never had to reject background noise.
  2. **Coarse Category Merging**: 24 fine-grained hierarchical categories (e.g., `menu.nm`, `menu.price`, `menu.cnt`, `total.cashprice`, `sub_total.subtotal_price`) were collapsed into just 3 primary coarse categories: `MENU`, `SUBTOTAL`, and `TOTAL`.
  3. **No Background Ambiguity**: In the test split (100 receipts), **100% of tokens** belong to `MENU` (58.3%), `TOTAL` (26.4%), or `SUBTOTAL` (15.2%).
* **No Direct Metadata Leakage**: The model inputs (`input_ids`, `bbox`, `pixel_values`) do not receive raw CORD hierarchy tags (`valid_line`, `repeating_symbol`, `roi`, `meta`). However, pre-filtering the input words to only valid lines represents an implicit structural simplification.

---

## 2. Category Collapsing & Mapping Traceability

### 2.1 Traceable Mapping from Raw CORD to Coarse Labels
In `preprocessing/cord_processor.py`, `map_cord_category_to_coarse()` implements the following mapping:

| Raw Fine-Grained CORD Category (Sample) | Prefix Match Rule | Coarse Output Category | Model BIO Labels |
| :--- | :--- | :---: | :---: |
| `menu.nm`, `menu.price`, `menu.cnt`, `menu.unitprice`, `menu.discountprice`, `menu.sub.nm`, `menu.sub.price`, `menu.sub.cnt`, `menu.itemsubtotal`, `menu.num`, `menu.etc` | `cat.startswith("menu")` | **`MENU`** | `B-MENU`, `I-MENU` |
| `sub_total.subtotal_price`, `sub_total.service_price`, `sub_total.tax_price`, `sub_total.discount_price`, `sub_total.etc` | `cat.startswith("sub_total")` or `cat.startswith("subtotal")` | **`SUBTOTAL`** | `B-SUBTOTAL`, `I-SUBTOTAL` |
| `total.total_price`, `total.cashprice`, `total.changeprice`, `total.creditcardprice`, `total.emoneyprice`, `total.menuqty_cnt`, `total.menutype_cnt`, `total.total_etc` | `cat.startswith("total")` | **`TOTAL`** | `B-TOTAL`, `I-TOTAL` |
| `void.*` | `cat.startswith("void")` | **`VOID`** | `B-VOID`, `I-VOID` (Train only: 2 words) |
| `payment.*`, `pay.*` | `cat.startswith("payment")` or `cat.startswith("pay")` | **`PAYMENT`** | `B-PAYMENT`, `I-PAYMENT` (0 occurrences in splits) |
| `etc`, `table`, `container` | Fallback | **`ETC`** | `B-ETC`, `I-ETC` (0 occurrences in splits) |

---

## 3. Label Schemas & ID Consistency

### 3.1 Config vs Model Checkpoint Mapping
* **Config Path**: `configs/labels/cord.yaml`
* **Model Checkpoint**: `models/checkpoints/layoutlmv3/cord/full/config.json`
* **Total Labels Defined**: 13

| Label ID | Config Label String | Checkpoint `id2label` | Present in Test Split? |
| :---: | :--- | :--- | :---: |
| 0 | `O` | `O` | No (0 tokens) |
| 1 | `B-MENU` | `B-MENU` | **Yes (887 spans)** |
| 2 | `I-MENU` | `I-MENU` | **Yes (487 tokens)** |
| 3 | `B-TOTAL` | `B-TOTAL` | **Yes (279 spans)** |
| 4 | `I-TOTAL` | `I-TOTAL` | **Yes (344 tokens)** |
| 5 | `B-SUBTOTAL` | `B-SUBTOTAL` | **Yes (143 spans)** |
| 6 | `I-SUBTOTAL` | `I-SUBTOTAL` | **Yes (216 tokens)** |
| 7 | `B-VOID` | `B-VOID` | No (only 2 in train) |
| 8 | `I-VOID` | `I-VOID` | No (only 2 in train) |
| 9 | `B-PAYMENT` | `B-PAYMENT` | No (0 tokens) |
| 10 | `I-PAYMENT` | `I-PAYMENT` | No (0 tokens) |
| 11 | `B-ETC` | `B-ETC` | No (0 tokens) |
| 12 | `I-ETC` | `I-ETC` | No (0 tokens) |

Label IDs are 100% consistent across `configs/labels/cord.yaml`, `preprocessing/label_mapping.py`, checkpoint `config.json`, and the evaluation scripts.

---

## 4. Empirical Split Label Distributions

### 4.1 Token Counts Across Splits
| Label | Train (800 docs) | Val (100 docs) | Test (100 docs) | Total Tokens | % of All Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `B-MENU` | 7,515 | 806 | 887 | 9,208 | 38.51% |
| `I-MENU` | 3,969 | 444 | 487 | 4,900 | 20.49% |
| `I-TOTAL` | 2,692 | 320 | 344 | 3,356 | 14.04% |
| `B-TOTAL` | 2,287 | 278 | 279 | 2,844 | 11.90% |
| `I-SUBTOTAL` | 1,717 | 203 | 216 | 2,136 | 8.93% |
| `B-SUBTOTAL` | 1,183 | 135 | 143 | 1,461 | 6.11% |
| `B-VOID` | 2 | 0 | 0 | 2 | 0.01% |
| `I-VOID` | 2 | 0 | 0 | 2 | 0.01% |
| `O` | 0 | 0 | 0 | 0 | 0.00% |
| **Total Words** | **19,367** | **2,186** | **2,356** | **23,909** | **100.00%** |

### 4.2 Entity Spans (B- Tags) in Test Split
* **MENU**: 887 spans (67.8%)
* **TOTAL**: 279 spans (21.3%)
* **SUBTOTAL**: 143 spans (10.9%)
* **Total Spans**: **1,309 spans** across 100 test receipts.

These exact support numbers (887 MENU, 143 SUBTOTAL, 279 TOTAL) match the reported evaluation metrics perfectly.

### 4.3 BIO/IOB2 Sequence Validity
* **Illegal Transitions Check**:
  - `I-X` without preceding `B-X` or `I-X`: **0**
  - `I-X` following `B-Y` ($X \neq Y$): **0**
  - **Verdict**: 100% valid BIO/IOB2 sequences across all 1,000 documents.

---

## 5. Potential Issues & Scientific Implications

1. **Absence of Background / Negative Class**: In typical end-to-end receipt extraction (such as SROIE), the model must distinguish target fields from irrelevant text (`O` tokens make up 92.6% of SROIE). In CORD, because non-target text is pre-filtered out, every single token belongs to one of the 3 target classes. This substantially eases the classification task.
2. **Coarse Collapsing vs Fine-Grained Benchmarks**: Standard academic benchmarks on CORD often evaluate fine-grained categories (30+ classes) or full KIE. Merging into 3 coarse classes simplifies the task to distinguishing Menu items from Subtotal and Total lines.
3. **No Label Leakage in Inputs**: Ground truth metadata is not fed to the model. The model truly uses OCR text, bounding boxes, and images to make its predictions.

---

## 6. Audit Summary

| Check Item | Requirement | Status | Evidence / Notes |
| :--- | :--- | :---: | :--- |
| **Label ID Mapping** | Config, Model, Dataset match | **PASS** | 13 labels consistent across config and checkpoint. |
| **BIO Validity** | Valid transitions, no illegal I- | **PASS** | 0 illegal transitions across 23,909 tokens. |
| **Mapping Traceability** | Traceable to raw CORD annotations | **PASS** | `map_cord_category_to_coarse()` fully documented. |
| **Metadata Leakage** | Grouping metadata not in model input | **PASS** | Model receives only words, bboxes, and pixel values. |
| **Background / O Handling** | Real-world document noise representation | **CAVEAT** | All words pre-filtered to valid lines; 0% O tokens. |
| **Category Granularity** | Fine-grained vs Coarse | **CAVEAT** | 24 fine-grained classes collapsed into 3 coarse classes. |
