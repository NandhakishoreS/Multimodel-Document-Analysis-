# Phase 6: Evaluation Protocol Audit

## 1. Baseline Evaluation Protocol (Phase 3)

**Models**: Text-Only Baseline (TF-IDF + LR) & Text+Layout Baseline (Spatial + LR)  

**Implementation**: models/baseline/text_baseline.py, models/baseline/layout_baseline.py, evaluation/metrics.py::compute_token_classification_metrics


| Question | Answer |
|---|---|
| **Q1_tokens_evaluated** | All word tokens in the test split. No tokenizer subwords — each word is one feature vector predicting one label. |
| **Q2_O_tokens** | O tokens are excluded from macro/micro averaging when ignore_labels=['O'] is passed. However, positions where true=O but pred=X (false positives for class X) ARE counted. |
| **Q3_special_tokens** | Not applicable — no tokenizer subwords. |
| **Q4_evaluation_level** | TOKEN-LEVEL. Each token is an independent classification. sklearn precision_recall_fscore_support is used. |
| **Q5_bio_spans** | B/I tags treated as INDEPENDENT LABELS. No span extraction. B-COMPANY and I-COMPANY are two distinct classes. |
| **Q6_macro_f1** | sklearn macro F1: mean of per-class F1 over non-O labels (B-COMPANY, I-COMPANY, B-DATE, I-DATE, etc.). NOT equivalent to seqeval entity-type macro F1. |
| **Q7_micro_f1** | sklearn micro F1: TP/(TP+FP) over all non-O labels globally. |
| **Q8_same_samples** | FUNSD and CORD: same test split counts as LayoutLMv3. SROIE: baselines used OLD preprocessing — NOT the same preprocessed data as corrected LayoutLMv3. |
| **Q9_label_mapping** | preprocessing/label_mapping.py::get_dataset_label_encoder. Label sets match configs/labels/*.yaml but evaluation level differs. |

## 2. LayoutLMv3 Evaluation Protocol (Phase 4/5)

**Model**: LayoutLMv3-base (microsoft/layoutlmv3-base fine-tuned)  

**Primary Metric**: seqeval entity-span F1 (IOB2 scheme)  

**Implementation**: evaluation/evaluate_layoutlmv3.py, evaluation/layoutlmv3_metrics.py::compute_seqeval_metrics, training/train_layoutlmv3.py::extract_word_level_labels, inference/layoutlmv3_predict.py::token_preds_to_word_preds


| Question | Answer |
|---|---|
| **Q1_tokens_evaluated** | Word-level labels only. Subword tokens aggregated to word level via word_ids. Only first subword of each word contributes. CLS/SEP excluded via word_ids=None check. |
| **Q2_O_tokens** | O tokens are included in word-level sequences passed to seqeval, but seqeval ignores O tokens in span extraction. O tokens do NOT contribute to any span-level metric. |
| **Q3_special_tokens** | YES — excluded via word_ids=None in extract_word_level_labels and token_preds_to_word_preds. |
| **Q4_evaluation_level** | ENTITY-SPAN LEVEL. seqeval IOB2 scheme. A prediction is correct only if entire span boundary AND entity type match. |
| **Q5_bio_spans** | B/I tags grouped into entity spans by seqeval. Strict IOB2: span starts at B-X, extends through I-X tokens. |
| **Q6_macro_f1** | seqeval macro F1: mean of per-entity-TYPE F1 (e.g., COMPANY, DATE, ADDRESS, TOTAL for SROIE). NOT the same as sklearn macro F1 over B/I labels. |
| **Q7_micro_f1** | seqeval micro F1: TP/(TP+FP) globally across all entity spans. |
| **Q8_same_samples** | Within each dataset, yes (SROIE corrected: 361, FUNSD: 50, CORD: 100). SROIE corrected LayoutLMv3 NOT evaluated on same preprocessed data as Phase 3 baselines. |
| **Q9_label_mapping** | SROIE: 9 labels. FUNSD: 7 labels. CORD: 13 labels. CORD seqeval measures only MENU/SUBTOTAL/TOTAL (VOID/PAYMENT/ETC have zero gold spans; o_ratio=0.0 confirms no O tokens in CORD test). |

## 3. Incompatibility Matrix

| Dataset | Baseline vs LayoutLMv3 | Key Reasons | Direction Valid? |
|---|:---:|---|:---:|
| SROIE | NOT COMPARABLE | Evaluation level: token-level (baseline) vs entity-span (LayoutLMv3).; Preprocessing: old SROIE data (baseline) vs corrected SROIE (LayoutLMv3).... | Yes |
| FUNSD | NOT COMPARABLE | Same test split (50 docs) but evaluation level differs.; Token-level (baseline) vs entity-span (LayoutLMv3).... | Yes |
| CORD | NOT COMPARABLE | Same test split (100 docs) but evaluation level differs.; Token-level (baseline) vs entity-span (LayoutLMv3).... | Yes |

## 4. Summary of Key Incompatibilities

The fundamental incompatibility is that **baseline models use token-level sklearn F1** while **LayoutLMv3 uses entity-span seqeval F1**. This causes:

- **SROIE baselines**: macro F1 ~0.26 because B/I tags for rarely-occurring entity tokens are evaluated independently. The ~92.5% O-token ratio means the denominator of token micro F1 is heavily O-dominated.

- **LayoutLMv3**: macro F1 ~0.88 because COMPANY/DATE/ADDRESS/TOTAL entity types are evaluated at span level, grouping all B/I tokens into coherent spans.

- The **direction of improvement** is unambiguous but the **magnitude** cannot be interpreted numerically without a common metric.
