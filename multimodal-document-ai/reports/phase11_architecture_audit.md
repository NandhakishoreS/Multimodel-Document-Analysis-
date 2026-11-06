# Phase 11 Architecture Audit
## Existing System Inspection Before Implementation

**Date**: September 12, 2026
**Purpose**: Document the exact structure of Phases 1-10 before adding the Phase 11 business-intelligence layer.

---

## 1. Repository Directory Structure

```
c:\MDA\multimodal-document-ai\
├── app.py                       # Root Streamlit entry-point (exec-delegates to app/app.py)
├── requirements.txt             # Production Python dependencies
├── pytest.ini                   # Test configuration
│
├── app/
│   └── app.py                   # Full 327-line Streamlit application (Phase 10)
│
├── inference/
│   ├── final_inference.py       # UnseenDocumentInferencePipeline (516 lines)
│   ├── schemas.py               # DocumentInferenceResult, ExtractedEntity, TokenPrediction
│   ├── postprocessing.py        # BIO span reconstruction, entity aggregation
│   └── layoutlmv3_predict.py    # Low-level LayoutLMv3 batch predict runner
│
├── visualization/
│   └── draw_predictions.py      # draw_predictions_on_image(), draw_raw_ocr_boxes()
│
├── preprocessing/               # SROIE/FUNSD/CORD processors (DO NOT MODIFY)
├── models/checkpoints/          # Trained weights (DO NOT MODIFY)
├── training/                    # Training code (DO NOT MODIFY)
├── evaluation/                  # Phase 6-9 evaluation runners
├── tests/                       # 17 test files, 194 tests (all passing)
├── configs/                     # YAML configs for training
├── processed_data/              # Tokenized benchmark datasets
└── reports/                     # All phase reports and artifacts
```

---

## 2. Core Inference Pipeline (inference/final_inference.py)

### UnseenDocumentInferencePipeline

**Constructor parameters**:
- `dataset_type: str` — "sroie" | "funsd" | "cord" (case-insensitive)
- `use_robust_model: bool` — selects Phase 9 joint-aug (True) or Phase 5 corrected baseline (False)
- `device: Optional[str]` — CUDA/CPU
- `project_root: Optional[Union[str, Path]]`

**Key public methods**:
- `predict_image(image_input, min_confidence)` → `DocumentInferenceResult`
- `predict_sample(sample_dict, image, min_confidence)` → `DocumentInferenceResult`
- `run_ocr(image)` → `(words, orig_bboxes, norm_bboxes, raw_text)`
- `get_raw_ocr_debug(image_input)` → `Dict`

**Checkpoint paths**:
- SROIE: models/checkpoints/phase9/sroie/joint_aug/best (primary)
- SROIE: models/checkpoints/layoutlmv3/sroie/corrected/best (baseline)
- FUNSD: models/checkpoints/layoutlmv3/funsd/full
- CORD: models/checkpoints/layoutlmv3/cord/full

---

## 3. Data Schemas (inference/schemas.py)

### ExtractedEntity (dataclass)
- `label: str` — e.g. "COMPANY", "DATE", "TOTAL", "ADDRESS", "MENU", "SUBTOTAL"
- `text: str` — reconstructed text span
- `confidence: float` — geometric mean over tokens
- `bbox: List[int]` — [x0, y0, x1, y1] in original image pixel space
- `normalized_bbox: List[int]` — [0, 1000] normalized
- `start_word, end_word, token_count: int`
- `min_confidence: float`

### DocumentInferenceResult (dataclass)
- `document_type: str` — "SROIE", "FUNSD", "CORD"
- `model_name: str`
- `image_path: Optional[str]`
- `image_size: Tuple[int, int]`
- `ocr_word_count: int`
- `entities: List[ExtractedEntity]`
- `tokens: List[TokenPrediction]`
- `raw_entities_dict: Dict[str, str]` — e.g. {"COMPANY": "...", "DATE": "...", "TOTAL": "..."}
- `ocr_raw_text: str`

### raw_entities_dict per benchmark
- SROIE: COMPANY, DATE, ADDRESS, TOTAL
- FUNSD: HEADER, QUESTION, ANSWER
- CORD: MENU, SUBTOTAL, TOTAL

---

## 4. Existing Streamlit Application (app/app.py)

Single-page, 327 lines.

**Components**:
- Sidebar: dataset selector, model variant radio, confidence slider
- Input: file uploader OR benchmark sample selector
- Action: "Run Document Understanding" button
- Results: 4 metric cards, 3-tab visualization, entity table, aggregated fields, JSON download

**State management**: st.session_state (inference_result, processed_image, image_label)
**Caching**: @st.cache_resource on get_pipeline()
**Entry**: app.py (root) exec()-delegates to app/app.py

---

## 5. Visualization (visualization/draw_predictions.py)

- `draw_predictions_on_image(image, entities, ...)` → PIL.Image with colored bounding boxes
- `draw_raw_ocr_boxes(image, ocr_results, ...)` → PIL.Image with OCR overlay
- COLOR_PALETTE: COMPANY=blue, DATE=green, ADDRESS=amber, TOTAL=red, MENU=soft-blue, etc.

---

## 6. Dependencies (requirements.txt)

Already present: pandas, numpy, matplotlib, seaborn, torch, transformers, PIL, sklearn, streamlit, pytest, rapidfuzz, pyyaml, tqdm

New dependencies needed for Phase 11:
- openpyxl — Excel export
- fpdf2 — PDF report generation
- plotly — interactive financial analytics charts
- sqlite3 — built into Python stdlib, no install needed

---

## 7. What Phase 11 Must NOT Touch

| Component | Reason |
|:---|:---|
| inference/final_inference.py | Validated pipeline |
| inference/schemas.py | Validated data contracts |
| models/checkpoints/ | Trained weights |
| preprocessing/*.py | Validated processors |
| training/ | Training code |
| evaluation/ | Evaluation runners |
| tests/ | 194 passing tests |
| app/app.py | Phase 10 Streamlit UI |
| processed_data/ | Benchmark splits |

---

## 8. Phase 11 Integration Strategy

Phase 11 is a NEW application/business-intelligence layer that:
1. Calls UnseenDocumentInferencePipeline.predict_image() to get DocumentInferenceResult
2. Maps raw_entities_dict + ExtractedEntity spans into FinancialDocument schema
3. Persists via new database/ layer (SQLite)
4. Classifies via new phase11/ classifier (rule-based, evidence-scored)
5. Presents via new app/phase11_app.py (multi-page Streamlit)
6. Exports via new phase11/export.py (PDF, Excel, CSV, JSON)

Phase 11 Streamlit is launched via new app_phase11.py root entry-point (no conflict with existing app.py).

---

## 9. New Modules to Create

```
phase11/
├── __init__.py
├── schema.py        # FinancialDocument dataclass
├── classifier.py    # Evidence-based bill category classifier
├── extractor.py     # Maps DocumentInferenceResult to FinancialDocument
├── database.py      # SQLite CRUD layer
└── export.py        # PDF / Excel / CSV / JSON export

app/
└── phase11_app.py   # Multi-page Streamlit financial intelligence UI

app_phase11.py       # New root entry-point
```
