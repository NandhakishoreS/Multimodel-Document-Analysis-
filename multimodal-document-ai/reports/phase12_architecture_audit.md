# Phase 12 Architecture Audit
## Multi-Document Ingestion & Intelligent Bill Classification

**Date**: September 12, 2026
**Project**: Multimodal Transformer-Based Intelligent Document Understanding and Information Extraction
**Current Baseline**: Phases 1–11 Complete (274 tests passed, 0 failed, 0 errors, 0 warnings)

---

## 1. Executive Summary & Audit Scope

This architecture audit assesses the end-to-end repository structure, data flows, persistence layer, inference pipelines, and Streamlit interfaces to establish a rigorous, non-disruptive integration plan for **Phase 12: Multi-Document Ingestion & Intelligent Bill Classification**.

### Core Invariants & Preservation Directives:
1. **Zero Retraining / Checkpoint Modification**: Phase 1–9 LayoutLMv3 model weights in `models/checkpoints/` remain completely untouched.
2. **Benchmark Integrity**: No modification to benchmark datasets (`SROIE`, `CORD`, `FUNSD`) or evaluation splits.
3. **No Duplicate Inference Implementations**: All multimodal inference will flow strictly through the validated `UnseenDocumentInferencePipeline` (`inference/final_inference.py`).
4. **Preserve Phase 11 Functionality & Data**: The existing SQLite database schema (`data/financial_documents.db`), `FinancialDocument` dataclass, and audit history must be cleanly extended via non-destructive migrations.
5. **Preserve Full Test Suite**: All 274 existing automated tests (194 baseline + 80 Phase 11) must continue passing without regression or weakening.

---

## 2. Current Ingestion Flow & Supported File Formats

### Current Capabilities (Phases 10–11):
- **Phase 10 App (`app/app.py`)**: Accepts single image uploads (`png`, `jpg`, `jpeg`, `tiff`, `bmp`) via Streamlit's `file_uploader`.
- **Phase 11 App (`app/phase11_app.py`)**: Accepts multi-file image uploads (`png`, `jpg`, `jpeg`, `tiff`, `bmp`).
- **File Ingestion Limitations Prior to Phase 12**:
  - **No PDF Support**: Uploading a `.pdf` file is either disallowed by the file uploader or fails PIL image opening.
  - **No Cryptographic Hashing**: File deduplication is nonexistent; identical files re-run full OCR and model inference and insert duplicate rows into the database.
  - **No Structured Validation Layer**: Invalid, corrupted, or oversized files raise unhandled exceptions during processing.
  - **Batch Vulnerability**: If one document in a loop fails, the entire loop or UI render can crash or leave the batch partially committed without structured status reporting.
  - **No Stage Tracking**: Processing does not track granular stages (`VALIDATING`, `OCR`, `EXTRACTING`, `CLASSIFYING`, `SAVING`).

---

## 3. Existing Inference Entry Points & Model Pipeline

The existing inference layer in `inference/final_inference.py` provides the canonical entry point:

```text
Input Document Image (PIL.Image / Path)
        ↓
UnseenDocumentInferencePipeline.run_ocr() [RapidOCR / polygon parsing / CORD number refinement]
        ↓
UnseenDocumentInferencePipeline.predict_image()
        ↓
Tokenization & 2D Bounding Box Normalization [LayoutLMv3TokenizerWrapper]
        ↓
Multimodal Forward Pass [LayoutLMv3Classifier]
        ↓
Token Prediction to Word BIO Alignment [collapse_token_preds_to_words()]
        ↓
Entity Reconstruction [reconstruct_entity_spans()]
        ↓
DocumentInferenceResult (schemas.py)
```

### Reusable Pipeline Components:
- `UnseenDocumentInferencePipeline` (`inference/final_inference.py`): Must be reused as-is without duplicating model loaders or tokenizer logic.
- `DocumentInferenceResult` (`inference/schemas.py`): Immutable contract output representing raw multimodal extractions across SROIE, CORD, and FUNSD.

---

## 4. Current Phase 11 Processing & Persistence Flow

In Phase 11, the extracted inference result is translated into the business domain:

```text
DocumentInferenceResult
        ↓
FinancialDocumentExtractor.from_inference_result() (phase11/extractor.py)
        ↓
BillCategoryClassifier.classify() (phase11/classifier.py)
        ↓
FinancialDocument (phase11/schema.py)
        ↓
FinancialDocumentDB.insert_document() (phase11/database.py)
        ↓
processing_log audit entry (phase11/database.py)
```

### Key Assets:
1. **`phase11.schema.FinancialDocument`**:
   - Stores identity (`document_id`, `filename`), classification (`category`, `confidence`, `evidence`), vendor/monetary fields (`total_amount`, `subtotal`, `tax_amount`), line items, and lifecycle status (`verification_status`).
2. **`phase11.classifier.BillCategoryClassifier`**:
   - 14-category taxonomy with keyword evidence extraction, calibrated score saturation, and explainable audit trails.
3. **`phase11.database.FinancialDocumentDB`**:
   - Manages SQLite persistence (`documents`, `line_items`, `processing_log`) with foreign keys, WAL mode, and audit logging.

---

## 5. Database Integration & Required Phase 12 Schema Migration

Existing `documents` table columns:
- `document_id`, `filename`, `source_file`, `document_type`, `benchmark_domain`, `model_name`
- `category`, `sub_category`, `classification_confidence`, `classification_evidence`
- `vendor_name`, `address`, `invoice_number`, `invoice_date`, `due_date`
- `subtotal`, `tax_amount`, `total_amount`, `currency`
- `extraction_confidence`, `ocr_word_count`, `ocr_raw_text`
- `verification_status`, `processing_timestamp`, `created_at`, `updated_at`, `notes`, `anomaly_flags`

### Required Phase 12 Extensions:
To support robust multi-document batch ingestion, tracking, deduplication, and classification review without disrupting existing data, the following optional columns will be added safely via `ALTER TABLE`:
- `file_hash` (`TEXT`): Cryptographic SHA-256 hash for exact file duplicate detection.
- `file_size` (`INTEGER`): File size in bytes.
- `processing_status` (`TEXT`): `QUEUED`, `VALIDATING`, `PROCESSING`, `OCR`, `EXTRACTING`, `CLASSIFYING`, `SAVING`, `COMPLETED`, `FAILED`, `DUPLICATE`, `REJECTED`.
- `processing_stage` (`TEXT`): Last active stage during processing.
- `processing_error` (`TEXT`): Human-readable error message if failed/rejected.
- `processing_started_at` (`TEXT`): ISO timestamp.
- `processing_completed_at` (`TEXT`): ISO timestamp.
- `processing_duration` (`REAL`): Duration in seconds.
- `ai_category` (`TEXT`): Preserved initial AI classification before any human override.
- `ai_confidence` (`REAL`): Preserved initial AI classification confidence.
- `final_category` (`TEXT`): Final operational category (human-edited or AI accepted).
- `requires_review` (`INTEGER`): Boolean flag (1/0) indicating low classification confidence or anomalies.

---

## 6. Streamlit Integration Architecture

Currently, the Streamlit interface is served via two endpoints:
- `app.py` -> `app/app.py`: Phase 10 research model inspection app.
- `app_phase11.py` -> `app/phase11_app.py`: Phase 11 5-page institutional financial intelligence app.

### Phase 12 Streamlit Strategy:
- Enhance `app/phase11_app.py` and `app_phase11.py` seamlessly:
  - Page 1 (**Process Documents**): Upgrade to institutional multi-document batch ingestion UI supporting drag-and-drop PDF/PNG/JPG uploads, real-time stage progress bars, duplicate detection modal/actions, batch summaries, and inline failure isolation.
  - Page 2 (**Document Registry**): Support filtering by `requires_review`, `ai_category` vs `final_category`, manual category correction with audit history preservation, and duplicate identification.
  - Configuration Controls: Add configurable confidence threshold (`CLASSIFICATION_CONFIDENCE_THRESHOLD`), maximum file size (`MAX_UPLOAD_SIZE_MB`), and duplicate policy (`skip` | `process_anyway`).

---

## 7. Reusable Components & Components That Must Not Be Duplicated

| Component | Location | Action in Phase 12 |
| :--- | :--- | :--- |
| Multimodal Model Checkpoints | `models/checkpoints/` | **STRICTLY READ-ONLY**. Must never be retrained or modified. |
| Multimodal Predictor | `inference/final_inference.py` | **REUSE DIRECTLY**. Never reimplement LayoutLMv3 inference. |
| Entity Schema | `inference/schemas.py` | **REUSE DIRECTLY**. Contract between inference and extraction. |
| Financial Document Extractor | `phase11/extractor.py` | **REUSE DIRECTLY**. Normalizes SROIE/CORD/FUNSD to FinancialDocument. |
| Bill Classifier | `phase11/classifier.py` | **REUSE & EXTEND**. Add configurable low-confidence review logic. |
| Database Layer | `phase11/database.py` | **NON-DESTRUCTIVE EXTENSION**. Add schema migration and hash lookup. |
| Export Engine | `phase11/export.py` | **REUSE DIRECTLY**. Exports registry data including Phase 12 fields. |

---

## 8. Potential Phase 12 Technical Risks & Mitigations

1. **PDF Rendering Dependencies on Windows**:
   - *Risk*: Traditional `pdf2image` relies on external C++ Poppler binaries which fail on Windows unless manually installed.
   - *Mitigation*: Installed and verified `pypdfium2` and `pypdf`, which include self-contained wheels with zero system dependencies.
2. **Database Schema Breakage**:
   - *Risk*: Modifying table schemas could corrupt or invalidate existing Phase 11 records.
   - *Mitigation*: Implement safe auto-migration in `_init_schema()` using `PRAGMA table_info` and non-destructive `ALTER TABLE ADD COLUMN`.
3. **Batch Processing Failure Cascades**:
   - *Risk*: An unhandled exception on image 3 of 20 crashes the entire run.
   - *Mitigation*: Wrap per-document execution in strict try-except blocks, recording `FAILED` status, logging the stack trace, and yielding control to continue the remaining queue.
4. **Regression in Full Test Suite**:
   - *Risk*: Adding fields to `FinancialDocument` dataclass breaks existing Phase 11 tests.
   - *Mitigation*: All new fields have defaults (`None`, `False`, or standard defaults), preserving complete positional and keyword backwards-compatibility with `FinancialDocument(...)`.
