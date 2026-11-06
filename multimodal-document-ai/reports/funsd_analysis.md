# FUNSD Comprehensive LayoutLMv3 Training, Evaluation & Baseline Analysis Report

**Phase 5 — Full FUNSD LayoutLMv3 Evaluation & Validation Protocol**  
**Date**: September 8, 2026  
**Artifact Path**: `reports/funsd_analysis.md`  
**Dataset**: ICDAR 2019 FUNSD Benchmark (Form Understanding in Noisy Scanned Documents)  
**Model**: `microsoft/layoutlmv3-base` (Full Fine-Tuning Checkpoint, Epoch 11 Best Validation)  

---

## 1. Executive Summary

The fine-tuning of `microsoft/layoutlmv3-base` on the ICDAR 2019 FUNSD dataset was fully executed and evaluated across the official 50-document test set:
- **Entity-Level Macro F1 (Seqeval)**: **`0.7179`** (71.79%)
- **Entity-Level Micro F1 (Seqeval)**: **`0.7995`** (79.95%)
- **Entity-Level Precision**: **`0.7804`** (78.04%)
- **Entity-Level Recall**: **`0.8196`** (81.96%)
- **Token-Level Weighted F1**: **`0.8298`** (82.98%)

### Comparison Against Phase 3 Baselines

| Model | Evaluation Level | Macro F1 | Micro F1 | Precision | Recall | Delta vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Text-Only (TF-IDF + LR)** | Token Level | 0.3512 | 0.4019 | 0.3474 | 0.4186 | — |
| **Text + Layout (Spatial + LR)** | Token Level | 0.4996 | 0.5405 | 0.4854 | 0.5516 | +14.84 pp from layout |
| **LayoutLMv3-base (Full)** | **Entity (Seqeval)** | **0.7179** | **0.7995** | **0.7804** | **0.8196** | **+21.83 pp Macro / +25.90 pp Micro** |

**Key Takeaway**: On FUNSD's complex multi-column forms, LayoutLMv3 substantially outperforms both the text-only and spatial linear baselines. While Phase 3 demonstrated that spatial layout adds +14.8 pp to text alone, LayoutLMv3's multimodal cross-attention between 2D bounding boxes, vision patches, and RoBERTa text representations delivers an additional **+21.83 percentage point gain** in Macro F1 and **+25.90 percentage point gain** in Micro F1.

---

## 2. Hardware & Training Configuration

- **GPU**: `NVIDIA GeForce RTX 3060 Laptop GPU` (6.44 GB VRAM)
- **Framework & Libraries**: PyTorch `2.5.1+cu121`, Transformers `4.46.3` (Python 3.12 `.venv312`)
- **Optimization**: AdamW (`lr = 5.0e-5`, weight decay = 0.01, cosine warmup = 10%)
- **Precision**: FP16 Mixed Precision (`torch.amp.autocast`)
- **Batch Sizing**: Batch size `2` per forward pass, `gradient_accumulation_steps = 4` &rarr; **Effective Batch Size = `8`**
- **Dataset Splits**: Train = 127 samples, Validation = 22 samples, Test = 50 samples
- **Label Schema**: 7 classes (`O`, `B-HEADER`, `I-HEADER`, `B-QUESTION`, `I-QUESTION`, `B-ANSWER`, `I-ANSWER`)

---

## 3. Training Progression & Checkpoint Selection

Training was conducted for the full 12 epochs on all 127 training samples, with Seqeval validation evaluated at the end of each epoch on the official 22-sample validation split:

| Epoch | Train Loss | Val Loss | Val Precision | Val Recall | Val Macro F1 | Val Micro F1 | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 1.800838 | 1.469285 | 0.2695 | 0.2558 | 0.1783 | 0.2624 | Initial checkpoint |
| **2** | 1.099899 | 0.951082 | 0.5639 | 0.7244 | 0.5304 | 0.6342 | Checkpoint saved |
| **3** | 0.773440 | 0.972007 | 0.5776 | 0.6770 | 0.5016 | 0.6234 | No improvement (1/3) |
| **4** | 0.534824 | 0.772932 | 0.6974 | 0.7674 | 0.5954 | 0.7307 | Checkpoint saved |
| **5** | 0.422396 | 0.760179 | 0.6822 | 0.7431 | 0.6132 | 0.7113 | Checkpoint saved |
| **6** | 0.348748 | 0.780780 | 0.7202 | 0.7916 | 0.6594 | 0.7542 | Checkpoint saved |
| **7** | 0.277679 | 0.863499 | 0.7152 | 0.7751 | 0.6530 | 0.7439 | No improvement (1/3) |
| **8** | 0.207529 | 0.876848 | 0.7354 | 0.8181 | 0.6721 | 0.7745 | Checkpoint saved |
| **9** | 0.154873 | 0.872290 | 0.7433 | 0.8203 | 0.6695 | 0.7799 | No improvement (1/3) |
| **10** | 0.134716 | 0.934364 | 0.7404 | 0.8115 | 0.6915 | 0.7743 | Checkpoint saved |
| **11** | **0.115034** | **0.918548** | **0.7455** | **0.8236** | **0.6946** | **0.7826** | **★ BEST CHECKPOINT** |
| **12** | 0.108277 | 0.923879 | 0.7450 | 0.8214 | 0.6938 | 0.7813 | No improvement (1/3) |

- **Best Validation Epoch**: Epoch 11 (`val_macro_f1 = 0.6946`, `val_micro_f1 = 0.7826`)
- **Total Training Time**: 357.5 seconds (5.96 minutes)
- **Checkpoint Path**: [models/checkpoints/layoutlmv3/funsd/full](file:///c:/MDA/multimodal-document-ai/models/checkpoints/layoutlmv3/funsd/full)

---

## 4. Test Set Evaluation (50 Documents, 1,973 Ground-Truth Spans)

The best checkpoint was evaluated on the complete 50-document FUNSD test set using strict Seqeval IOB2 span evaluation:

### Per-Entity Breakdown

| Entity Class | Precision | Recall | F1-Score | Support (Entity Spans) | Operational Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **QUESTION** | **0.831** | **0.845** | **0.838** | 1,049 | Strongest class; form field prompts have distinct lexical patterns (e.g., "Name:", "Date:", "Title:"). |
| **ANSWER** | **0.757** | **0.829** | **0.791** | 805 | High recall; primary errors are subtle boundary shifts on long multi-token filled values. |
| **HEADER** | **0.512** | **0.538** | **0.525** | 119 | Smallest class (6% of spans); high variance across diverse form document banner headers. |
| **Micro Average** | **0.780** | **0.820** | **0.800** | 1,973 | Global entity extraction across all form elements. |
| **Macro Average** | **0.700** | **0.737** | **0.718** | 1,973 | Unweighted mean across the three entity categories. |

### Classification Report

```
              precision    recall  f1-score   support

      ANSWER       0.76      0.83      0.79       805
      HEADER       0.51      0.54      0.52       119
    QUESTION       0.83      0.84      0.84      1049

   micro avg       0.78      0.82      0.80      1973
   macro avg       0.70      0.74      0.72      1973
weighted avg       0.78      0.82      0.80      1973
```

---

## 5. Token Confusion Matrix & Error Characteristics

Token-level predictions across all 8,707 test word tokens:

```
True / Pred         O    B-HEADER    I-HEADER  B-QUESTION  I-QUESTION    B-ANSWER    I-ANSWER
O               1,939          18          13         130          80          91         114
B-HEADER           27          57          11          10           1          11           2
I-HEADER           61           7         147          14          14           4           8
B-QUESTION         97           6           2         829          66          47          23
I-QUESTION        116           2           5          68       1,234          44         115
B-ANSWER          102           1           1          54          12         539         100
I-ANSWER          258           6          17          56         126          91       1,931
```

### Key Error Mechanisms
1. **Question vs. Answer Boundary Bleed**: In dense forms without visual separator lines, the boundary between questions and their immediately adjacent answers can blur (e.g. 54 `B-ANSWER` tokens misclassified as `B-QUESTION`).
2. **Continuation (`I-`) Tag Transition Errors**: When answers span multiple lines or table cells, 100 `B-ANSWER` tokens were predicted as `I-ANSWER`, representing slight span-onset shifts.
3. **Background (`O`) Confusion**: Because form instructions, footer legal notices, and stamps are labeled `O`, 258 `I-ANSWER` tokens were predicted as `O`, usually where scan noise or low-resolution degraded character legibility.

---

## 6. Generated Visualizations & Artifacts

1. **Training Curves**: [reports/figures/training_curves_funsd_full.png](file:///c:/MDA/multimodal-document-ai/reports/figures/training_curves_funsd_full.png)
2. **Loss Curves**: [reports/figures/funsd_full_loss_curves.png](file:///c:/MDA/multimodal-document-ai/reports/figures/funsd_full_loss_curves.png)
3. **F1 Curves**: [reports/figures/funsd_full_f1_curves.png](file:///c:/MDA/multimodal-document-ai/reports/figures/funsd_full_f1_curves.png)
4. **Token Confusion Matrix (Raw)**: [reports/figures/funsd_layoutlmv3_confusion_matrix.png](file:///c:/MDA/multimodal-document-ai/reports/figures/funsd_layoutlmv3_confusion_matrix.png)
5. **Token Confusion Matrix (Normalized)**: [reports/figures/funsd_layoutlmv3_confusion_matrix_normalized.png](file:///c:/MDA/multimodal-document-ai/reports/figures/funsd_layoutlmv3_confusion_matrix_normalized.png)
6. **Per-Entity Metrics JSON**: [reports/layoutlmv3_funsd_per_entity.json](file:///c:/MDA/multimodal-document-ai/reports/layoutlmv3_funsd_per_entity.json)
7. **Test Predictions JSON**: [reports/layoutlmv3_funsd_test_predictions.json](file:///c:/MDA/multimodal-document-ai/reports/layoutlmv3_funsd_test_predictions.json)
8. **Full Experiment Summary**: [reports/layoutlmv3_funsd_full_experiment_summary.json](file:///c:/MDA/multimodal-document-ai/reports/layoutlmv3_funsd_full_experiment_summary.json)
