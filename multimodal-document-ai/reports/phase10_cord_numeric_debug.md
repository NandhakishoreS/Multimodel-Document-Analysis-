# Phase 10: CORD Numeric Token OCR Diagnostic & Resolution Report

**Project**: Multimodal Transformer-Based Intelligent Document Understanding and Information Extraction  
**Component**: Phase 10 Production Inference Pipeline & Streamlit Web Interface  
**Investigation**: Standalone Numeric Token Disappearance on CORD Receipts  
**Date**: 2026-09-12  
**Status**: **DIAGNOSED, RESOLVED & VALIDATED (194/194 TESTS PASSING)**

---

## 1. Executive Summary

During Phase 10 inference evaluation on dining receipt documents from the CORD dataset, small standalone numeric tokens representing table row numbers or line item quantities (e.g., `"1"`, `"4"`) visible in the invoice were not consistently appearing as structured extracted entities, while other digits such as `"2"` were detected and extracted.

This investigation traced the full end-to-end inference lifecycle:
```
Input Image 
  → OCR Detection & Recognition 
  → Line Word Tokenization 
  → Coordinate Bounding Box Interpolation & Normalization 
  → LayoutLMv3 Multimodal Encoding 
  → Token Classification 
  → First-Subword Word Collapsing 
  → BIO Entity Span Decoding 
  → Confidence Filtering 
  → Streamlit Display
```

The root cause was isolated, resolved, and verified across real CORD receipt images and synthetic receipt benchmarks without retraining models or modifying Phase 1–9 checkpoints.

---

## 2. Root Cause Analysis

### 2.1 The Disappearance Stage: OCR Word Tokenization & Kerning Aggregation

Tracing raw RapidOCR output against CORD receipt images revealed that the disappearance was **not** caused by downstream confidence thresholds, BIO span decoding, or entity length filtering (which correctly preserve tokens of length 1).

Instead, the issue occurred at the **OCR text splitting stage** inside `run_ocr` due to inter-character kerning in receipt line items:

1. **Tight Kerning in Line Items**: In CORD receipts, printed item descriptions frequently appear with minimal horizontal whitespace between the row number/quantity digit and the description text:
   - Row 1: `1EGG TART` (quantity `1` immediately adjacent to `EGG`)
   - Row 2: `2CHOCO CUSTARD PASTRY` (quantity `2` immediately adjacent to `CHOCO`)
   - Row 3: `1BANANA DONUT` (quantity `1` immediately adjacent to `BANANA`)
   - Row 4: `1CREAMCHEESE BREAD` (quantity `1` immediately adjacent to `CREAMCHEESE`)
   - Row 5: `1FRANKFRUT SAUSAGE ROLL` (quantity `1` immediately adjacent to `FRANKFRUT`)

2. **Asymmetric Whitespace Recognition**: In some rows (such as `2 TWIST DONUT` in `cord_test_35`), the digit `"2"` had a slightly wider gap (2–3 additional pixels), prompting the OCR recognition model to insert a whitespace character: `"2 TWIST DONUT"`. In adjacent rows (`1BANANA`, `1CREAMCHEESE`), the tighter kerning caused OCR to emit words without whitespace.

3. **Naive Whitespace Tokenization (`text.split()`)**: The original `run_ocr` implementation split OCR line strings using standard Python `text.split()`.
   - When `"2 TWIST DONUT"` was processed, `text.split()` produced `['2', 'TWIST', 'DONUT']`. The token `'2'` became an independent word token, received its own bounding box, and was correctly classified by LayoutLMv3 as `B-MENU` (confidence 99.87%).
   - When `"1BANANA DONUT"` or `"1EGG TART"` was processed, `text.split()` produced `['1BANANA', 'DONUT']` or `['1EGG', 'TART']`. The digit `'1'` was never emitted as an independent word token; it remained fused into the food item name. Consequently, `'1'` never appeared as a standalone structured entity.

4. **Punctuation and Enclosure Variants**: Table numbering formats such as `"1."`, `"4."`, `"(1)"`, or unit quantities like `"2PCS"`, `"2ITEM"` similarly failed to yield pure standalone numeric tokens under naive whitespace splitting.

---

## 3. Diagnostic Decision: Why CASE A / OCR Preprocessing Alone Was Insufficient

Per the diagnostic protocol:
- **Upscaling Experiments (1.5x, 2.0x, 3.0x)**: Upscaling CORD receipts before OCR changed the stroke aspect ratio for thin vertical characters. On `cord_test_6`, 2x upscaling caused RapidOCR's recognition engine to misrecognize the thin vertical stroke `'1'` as the Chinese character `三` (`\u4e09`). On `cord_test_92`, 2x upscaling caused a clearly visible standalone `'4'` to be lost completely.
- **Contrast Enhancement**: Aggressive contrast enhancement degraded character edges on low-contrast thermal receipts.
- **Conclusion**: Baseline RapidOCR at native image resolution preserves raw character strokes with the highest fidelity. The required intervention was in **token boundary resolution and bounding box interpolation for CORD receipt lines**.

---

## 4. The Fix: Targeted CORD OCR Word Splitting & Box Partitioning

### 4.1 Implementation in `inference/final_inference.py`

A dedicated token-partitioning function `split_cord_ocr_words(text: str)` was implemented:

```python
def split_cord_ocr_words(text: str) -> List[str]:
    raw_tokens = text.split()
    refined: List[str] = []
    for token in raw_tokens:
        # Match table row numbering with trailing dot (e.g. '1.' -> '1', '4.' -> '4')
        m_dot = re.match(r"^(\d+)\.$", token)
        # Match parenthesized digits (e.g. '(1)' -> '1', '(4)' -> '4')
        m_paren = re.match(r"^\((\d+)\)$", token)
        # Match leading digits glued to letters, excluding single multiplier units like '1x', '2X'
        m_lead = re.match(r"^(\d+)([A-Za-z].*)$", token)

        if m_dot:
            refined.append(m_dot.group(1))
        elif m_paren:
            refined.append(m_paren.group(1))
        elif m_lead and m_lead.group(2).lower() not in ("x", "k", "l"):
            refined.append(m_lead.group(1))
            refined.append(m_lead.group(2))
        else:
            refined.append(token)
    return refined
```

In `run_ocr`:
```python
if self.dataset_type == "cord":
    line_words = split_cord_ocr_words(text)
else:
    line_words = text.split()
```

### 4.2 Proportional Coordinate Interpolation

When a fused token such as `"1EGG"` is partitioned into `["1", "EGG"]`, the existing horizontal character-width interpolation logic calculates exact pixel and normalized `[0, 1000]` bounding boxes for each sub-token:
$$\text{char\_w} = \frac{x_{\max} - x_{\min}}{\text{total\_chars}}$$
The digit `'1'` receives an accurate bounding box spanning $[x_{\min}, y_{\min}, x_{\min} + \text{char\_w}, y_{\max}]$, and `'EGG'` receives the remainder.

---

## 5. Before vs. After Extraction Comparison

### Sample 1: `cord_test_35.png` (Multi-item Receipt Table)

| Item Line | Raw OCR Text | Before (Extracted Entities) | After (Extracted Entities) |
|---|---|---|---|
| Row 1 | `2 TWIST DONUT` | `[MENU] '2'`, `[MENU] 'TWIST DONUT'` | `[MENU] '2'`, `[MENU] 'TWIST DONUT'` |
| Row 2 | `1BANANA DONUT` | `[MENU] '1BANANA DONUT'` (digit lost) | **`[MENU] '1'`**, `[MENU] 'BANANA DONUT'` |
| Row 3 | `1CREAMCHEESE BREAD` | `[MENU] '1CREAMCHEESE BREAD'` (digit lost) | **`[MENU] '1'`**, `[MENU] 'CREAMCHEESE BREAD'` |
| Row 4 | `1FRANKFRUT SAUSAGE ROLL` | `[MENU] '1FRANKFRUT SAUSAGE ROLL'` (digit lost) | **`[MENU] '1'`**, `[MENU] 'FRANKFRUT SAUSAGE ROLL'` |

### Sample 2: `cord_test_6.png` (Egg Tart & Choco Custard)

| Item Line | Raw OCR Text | Before | After |
|---|---|---|---|
| Row 1 | `1EGG TART` | `[MENU] '1EGG TART'` | **`[MENU] '1'`**, `[MENU] 'EGG TART'` |
| Row 2 | `2CHOCO CUSARD PASTRY` | `[MENU] '2CHOCO CUSARD PASTRY'` | **`[MENU] '2'`**, `[MENU] 'CHOCO CUSARD PASTRY'` |

### Sample 3: `cord_test_97.png` (Dim Sum / Street Food Receipt)

| Item Line | Raw OCR Text | Before | After |
|---|---|---|---|
| Line 1 | `2` | `[MENU] '2'` (conf=0.9987) | `[MENU] '2'` (conf=0.9987) |
| Line 7 | `3` | `[MENU] '3'` (conf=0.9986) | `[MENU] '3'` (conf=0.9986) |

### Sample 4: Synthetic Table Benchmark (User Example Table)

```
1   Lorem Ipsum Dolor              $50.00
2   Pellentesque id neque ligula   $20.00
3   Interdum et malesuada fames    $30.00
4   Vivamus volutpat faucibus      $40.00
```

- **Extraction**:
  - `[MENU] '1'` (conf: 99.84%)
  - `[MENU] '2'` (conf: 99.85%)
  - `[MENU] '3'` (conf: 99.85%)
  - `[MENU] '4'` (conf: 99.82%)
  - All food description entities and price entities extracted cleanly.

---

## 6. Scope of Fix: CORD-Specific vs. Global

> [!IMPORTANT]
> **This fix is strictly CORD-specific.**  
> - When `dataset_type == "cord"`, `split_cord_ocr_words` activates to decouple glued table numbers and item quantities.
> - When `dataset_type == "sroie"` or `dataset_type == "funsd"`, standard `text.split()` whitespace tokenization is preserved identically. This prevents unwanted splitting of alphanumeric unit numbers in Malaysian addresses (e.g., `12A Jalan...`) or form codes (e.g., `1A`, `2B`).
> - Zero checkpoints were modified.
> - Zero datasets were modified.
> - Phase 1–9 baseline metrics remain 100% unaltered.

---

## 7. Visual Debugging Artifacts Generated

The following visual artifacts were generated and saved to `reports/visualizations/`:

1. `reports/visualizations/cord_test_6_raw_ocr_debug.png`: Raw OCR bounding boxes with green highlights on row numbers.
2. `reports/visualizations/cord_test_6_predictions_annotated.png`: Multimodal LayoutLMv3 predicted entity bounding boxes including `'1'` and `'2'`.
3. `reports/visualizations/cord_test_35_raw_ocr_debug.png`: Raw OCR bounding boxes showing individual row items.
4. `reports/visualizations/cord_test_35_predictions_annotated.png`: Annotated predictions showing `'2'`, `'1'`, `'1'`, `'1'` extracted as distinct `MENU` entities.
5. `reports/visualizations/cord_test_97_raw_ocr_debug.png`: Raw OCR boxes on standalone digits `'2'` and `'3'`.
6. `reports/visualizations/cord_test_97_predictions_annotated.png`: Clean predictions on standalone items.

---

## 8. Verification & Test Suite Status

A comprehensive suite of regression tests was added to `tests/test_phase10_inference.py`:
- `test_one_character_numeric_tokens_not_discarded`: Confirms that tokens with `len(w) == 1` are not dropped during word collapsing or BIO span decoding.
- `test_split_cord_ocr_words_glued_digits`: Confirms that all glued variants (`1EGG`, `2CHOCO`, `1BANANA`, `4Vivamus`, `1.`, `4.`, `(1)`, `2PCS`) are correctly decomposed while prices (`50.000`), multipliers (`1x`), and normal words are preserved.
- `test_draw_raw_ocr_boxes_renders_and_highlights`: Confirms that raw OCR debug rendering operates without error.
- `test_raw_ocr_debug_method_output_schema`: Confirms that `get_raw_ocr_debug` produces the required diagnostic schema.

### Test Results
- **Phase 10 Inference Tests**: 50 passed / 0 failed (up from 46)
- **Phase 10 Streamlit Tests**: 39 passed / 0 failed
- **Full Project Test Suite**: **194 passed / 0 failed in 39.03s** (0 warnings)
- **Streamlit Web Application**: Live at `http://localhost:8501`, featuring the interactive Raw OCR Debug Tab and Numeric Token Inspector.
