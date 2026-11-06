"""
Phase 12 — Multi-Document Ingestion & Processing Orchestration.

Provides an institutional-grade document ingestion layer supporting:
    - Formats: PDF (via pypdfium2), PNG, JPG, JPEG
    - Validation: file type, file size, integrity checks
    - Deduplication: SHA-256 cryptographic file hashing & financial duplicate detection
    - Processing lifecycle: QUEUED -> VALIDATING -> OCR -> EXTRACTING -> CLASSIFYING -> SAVING -> COMPLETED
    - Independent batch processing: faults in one document never terminate the batch
    - Comprehensive batch summary analytics
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

from phase11.classifier import BillCategoryClassifier
from phase11.database import FinancialDocumentDB
from phase11.extractor import FinancialDocumentExtractor
from phase11.schema import FinancialDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}


class ProcessingStatus(str, Enum):
    """Normalized document processing lifecycle states."""
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    PROCESSING = "PROCESSING"
    OCR = "OCR"
    EXTRACTING = "EXTRACTING"
    CLASSIFYING = "CLASSIFYING"
    SAVING = "SAVING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DUPLICATE = "DUPLICATE"
    REJECTED = "REJECTED"


class DuplicatePolicy(str, Enum):
    """Policy when an exact file_hash collision is encountered."""
    SKIP = "skip"                # Skip processing, link to existing record
    PROCESS_ANYWAY = "process"   # Re-run inference and insert new record
    WARN = "warn"                # Process but flag as duplicate in anomaly flags


@dataclass
class IngestionConfig:
    """Ingestion configuration options."""
    max_upload_size_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES
    allowed_extensions: set[str] = field(default_factory=lambda: set(ALLOWED_EXTENSIONS))
    confidence_threshold: float = 0.60
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.WARN
    default_currency: str = "INR"


# ---------------------------------------------------------------------------
# Ingestion Item & Batch Summary
# ---------------------------------------------------------------------------

@dataclass
class IngestionItem:
    """Encapsulates a single document candidate for processing."""
    filename: str
    data: bytes
    document_id: Optional[str] = None
    file_extension: str = ""
    file_size: int = 0
    file_hash: str = ""
    status: ProcessingStatus = ProcessingStatus.QUEUED
    stage: str = "INITIALIZED"
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0
    is_duplicate: bool = False
    existing_document_id: Optional[str] = None
    rendered_images: List[Image.Image] = field(default_factory=list)
    document: Optional[FinancialDocument] = None

    @classmethod
    def from_bytes(cls, filename: str, data: bytes) -> "IngestionItem":
        clean_name = Path(str(filename).replace("\x00", "").replace("\\", "/")).name.strip()
        if not clean_name:
            clean_name = "document"
        ext = Path(clean_name).suffix.lower()
        size = len(data)
        item = cls(
            filename=clean_name,
            data=data,
            file_extension=ext,
            file_size=size,
        )
        item.file_hash = cls.calculate_hash(data)
        return item

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "IngestionItem":
        p = Path(path)
        data = p.read_bytes()
        return cls.from_bytes(p.name, data)

    @staticmethod
    def calculate_hash(data: bytes) -> str:
        """Compute SHA-256 digest of raw byte stream."""
        return hashlib.sha256(data).hexdigest()


@dataclass
class BatchSummary:
    """Summary metrics of a batch ingestion run."""
    total_uploaded: int = 0
    completed: int = 0
    failed: int = 0
    duplicates: int = 0
    rejected: int = 0
    requires_review_count: int = 0
    category_counts: Dict[str, int] = field(default_factory=dict)
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    items: List[IngestionItem] = field(default_factory=list)
    documents: List[FinancialDocument] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Ingestion Service
# ---------------------------------------------------------------------------

class DocumentIngestionService:
    """
    Robust multi-document ingestion and batch orchestration service.
    """

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()

    def validate_item(self, item: IngestionItem) -> bool:
        """
        Validate file extension, file size, and basic integrity.
        Returns True if valid, False if rejected.
        """
        item.status = ProcessingStatus.VALIDATING
        item.stage = "VALIDATING"

        # 1. Extension validation
        if item.file_extension not in self.config.allowed_extensions:
            item.status = ProcessingStatus.REJECTED
            item.error_message = (
                f"Unsupported file format '{item.file_extension}'. "
                f"Allowed formats: {', '.join(sorted(self.config.allowed_extensions))}."
            )
            logger.warning("Rejected file '%s': %s", item.filename, item.error_message)
            return False

        # 2. Size validation
        if item.file_size > self.config.max_upload_size_bytes:
            max_mb = self.config.max_upload_size_bytes / (1024 * 1024)
            item.status = ProcessingStatus.REJECTED
            item.error_message = (
                f"File size ({item.file_size / (1024 * 1024):.2f} MB) exceeds "
                f"maximum limit of {max_mb:.1f} MB."
            )
            logger.warning("Rejected file '%s': %s", item.filename, item.error_message)
            return False

        if item.file_size == 0:
            item.status = ProcessingStatus.REJECTED
            item.error_message = "File is empty (0 bytes)."
            logger.warning("Rejected file '%s': %s", item.filename, item.error_message)
            return False

        # 3. Integrity verification & image rendering
        try:
            images = self.render_to_images(item.data, item.file_extension)
            if not images:
                item.status = ProcessingStatus.FAILED
                item.error_message = "No renderable pages found in document."
                return False
            item.rendered_images = images
        except Exception as exc:
            item.status = ProcessingStatus.FAILED
            item.error_message = f"Unable to read or parse document: {str(exc)}"
            logger.error("Corruption check failed for '%s': %s", item.filename, exc)
            return False

        item.status = ProcessingStatus.QUEUED
        item.stage = "VALIDATED"
        return True

    def render_to_images(self, data: bytes, extension: str) -> List[Image.Image]:
        """
        Convert file bytes into a list of PIL Images.
        Handles PDFs (via pypdfium2) and standard image formats.
        """
        ext = extension.lower()
        if ext == ".pdf":
            try:
                import pypdfium2 as pdfium
            except ImportError as exc:
                raise RuntimeError("pypdfium2 is required for PDF rendering.") from exc

            doc = pdfium.PdfDocument(data)
            if len(doc) == 0:
                raise ValueError("PDF contains 0 pages.")
            images: List[Image.Image] = []
            for i in range(len(doc)):
                page = doc[i]
                # Render at 200 DPI scale for crisp OCR
                pil_img = page.render(scale=200 / 72).to_pil().convert("RGB")
                images.append(pil_img)
            return images
        else:
            # Standard image format (PNG, JPG, JPEG)
            img = Image.open(io.BytesIO(data))
            img.verify()  # Verify image header/integrity
            # Reopen for actual pixel operations since verify() closes file
            img = Image.open(io.BytesIO(data)).convert("RGB")
            return [img]

    def check_duplicate(
        self,
        item: IngestionItem,
        db: FinancialDocumentDB,
    ) -> Tuple[bool, Optional[FinancialDocument]]:
        """
        Check if an exact cryptographic duplicate exists in the database.
        """
        existing = db.get_document_by_hash(item.file_hash)
        if existing is not None:
            item.is_duplicate = True
            item.existing_document_id = existing.document_id
            return True, existing
        return False, None

    def process_document(
        self,
        item: IngestionItem,
        pipeline: Any,
        extractor: FinancialDocumentExtractor,
        classifier: BillCategoryClassifier,
        db: FinancialDocumentDB,
        duplicate_policy: Optional[DuplicatePolicy] = None,
    ) -> IngestionItem:
        """
        Process a single IngestionItem through the complete pipeline with strict error isolation.
        """
        policy = duplicate_policy or self.config.duplicate_policy
        t0 = time.time()
        item.started_at = datetime.now(timezone.utc).isoformat()
        item.status = ProcessingStatus.PROCESSING

        try:
            # 1. Validation check
            if not item.rendered_images:
                valid = self.validate_item(item)
                if not valid:
                    item.duration_seconds = round(time.time() - t0, 3)
                    item.completed_at = datetime.now(timezone.utc).isoformat()
                    return item

            # 2. Duplicate detection
            is_dup, existing_doc = self.check_duplicate(item, db)
            if is_dup:
                if policy == DuplicatePolicy.SKIP:
                    item.status = ProcessingStatus.DUPLICATE
                    item.stage = "SKIPPED_DUPLICATE"
                    item.error_message = (
                        f"Exact file duplicate of document ID: {item.existing_document_id}."
                    )
                    item.document = existing_doc
                    item.duration_seconds = round(time.time() - t0, 3)
                    item.completed_at = datetime.now(timezone.utc).isoformat()
                    return item

            # 3. Multimodal inference (OCR + LayoutLMv3)
            item.stage = "OCR_AND_INFERENCE"
            item.status = ProcessingStatus.OCR
            primary_image = item.rendered_images[0]

            item.status = ProcessingStatus.EXTRACTING
            inference_result = pipeline.predict_image(primary_image)

            # 4. Normalized Extraction & Bill Classification
            item.status = ProcessingStatus.CLASSIFYING
            fin_doc = extractor.extract(
                inference_result=inference_result,
                filename=item.filename,
            )

            # Populate Phase 12 Ingestion metadata
            item.document_id = fin_doc.document_id
            fin_doc.file_hash = item.file_hash
            fin_doc.file_size = item.file_size
            fin_doc.currency = self.config.default_currency

            # Classify with explainability and review threshold
            clf_result = classifier.classify(
                ocr_text=fin_doc.ocr_raw_text,
                vendor_name=fin_doc.vendor_name,
                entity_text=" ".join(fin_doc.classification_evidence),
                review_threshold=self.config.confidence_threshold,
            )
            fin_doc.category = clf_result.category
            fin_doc.sub_category = clf_result.sub_category
            fin_doc.classification_confidence = clf_result.confidence
            fin_doc.classification_evidence = clf_result.evidence
            fin_doc.ai_category = clf_result.category
            fin_doc.ai_confidence = clf_result.confidence
            fin_doc.final_category = clf_result.category
            fin_doc.requires_review = clf_result.requires_review
            fin_doc.classification_reason = clf_result.reason

            # Financial duplicate check (secondary check across vendor/amount/date)
            fin_dups = db.find_duplicate_financials(
                vendor_name=fin_doc.vendor_name,
                invoice_number=fin_doc.invoice_number,
                invoice_date=fin_doc.invoice_date,
                total_amount=fin_doc.total_amount,
                exclude_id=fin_doc.document_id,
            )
            if fin_dups:
                warning = f"Possible financial duplicate of document {fin_dups[0].document_id}"
                if warning not in fin_doc.anomaly_flags:
                    fin_doc.anomaly_flags.append(warning)
                fin_doc.requires_review = True

            if is_dup and policy == DuplicatePolicy.WARN:
                dup_warn = f"Exact SHA-256 duplicate of document {item.existing_document_id}"
                if dup_warn not in fin_doc.anomaly_flags:
                    fin_doc.anomaly_flags.append(dup_warn)
                fin_doc.requires_review = True

            # 5. Persistence
            item.status = ProcessingStatus.SAVING
            item.stage = "SAVING"
            duration = round(time.time() - t0, 3)
            fin_doc.processing_status = "COMPLETED"
            fin_doc.processing_stage = "COMPLETED"
            fin_doc.processing_started_at = item.started_at
            fin_doc.processing_completed_at = datetime.now(timezone.utc).isoformat()
            fin_doc.processing_duration = duration

            db.insert_document(fin_doc)

            # Phase 15 Anomaly & Duplicate Intelligence (fault-isolated)
            try:
                from phase11.duplicate_detector import MultiSignalDuplicateDetector
                from phase11.anomaly_engine import CentralizedAnomalyEngine

                dup_matches = MultiSignalDuplicateDetector.detect_duplicates_for_document(fin_doc, db)
                if dup_matches:
                    db.save_duplicate_candidates([m.to_dict() for m in dup_matches])
                    fin_doc.requires_review = True

                engine = CentralizedAnomalyEngine()
                report = engine.evaluate_document(fin_doc)
                if report.findings:
                    db.save_anomaly_findings([f.to_dict() for f in report.findings])
                    if report.severity in ("HIGH", "CRITICAL"):
                        fin_doc.requires_review = True
            except Exception as anom_exc:
                logger.warning("Anomaly evaluation non-fatal notice for '%s': %s", item.filename, anom_exc)

            item.status = ProcessingStatus.COMPLETED
            item.stage = "COMPLETED"
            item.duration_seconds = duration
            item.completed_at = fin_doc.processing_completed_at
            item.document = fin_doc

        except Exception as exc:
            duration = round(time.time() - t0, 3)
            item.status = ProcessingStatus.FAILED
            item.error_message = f"Failure at stage {item.stage}: {str(exc)}"
            item.duration_seconds = duration
            item.completed_at = datetime.now(timezone.utc).isoformat()
            logger.error("Processing failed for '%s': %s", item.filename, exc, exc_info=True)

        return item

    def process_batch(
        self,
        items: List[IngestionItem],
        pipeline: Any,
        extractor: FinancialDocumentExtractor,
        classifier: BillCategoryClassifier,
        db: FinancialDocumentDB,
        progress_callback: Optional[Callable[[int, int, IngestionItem], None]] = None,
    ) -> BatchSummary:
        """
        Process a collection of IngestionItems independently with progress reporting.
        Guarantees that a failure in one document does not terminate the remaining batch.
        """
        summary = BatchSummary(total_uploaded=len(items))
        start_time = time.time()

        for idx, item in enumerate(items, 1):
            # Process single document independently
            self.process_document(
                item=item,
                pipeline=pipeline,
                extractor=extractor,
                classifier=classifier,
                db=db,
            )

            # Accumulate statistics
            if item.status == ProcessingStatus.COMPLETED:
                summary.completed += 1
                if item.document:
                    summary.documents.append(item.document)
                    cat = item.document.category or "Unknown"
                    summary.category_counts[cat] = summary.category_counts.get(cat, 0) + 1
                    if item.document.requires_review:
                        summary.requires_review_count += 1
            elif item.status == ProcessingStatus.DUPLICATE:
                summary.duplicates += 1
            elif item.status == ProcessingStatus.REJECTED:
                summary.rejected += 1
            elif item.status == ProcessingStatus.FAILED:
                summary.failed += 1

            summary.items.append(item)

            if progress_callback:
                try:
                    progress_callback(idx, len(items), item)
                except Exception as cb_exc:
                    logger.warning("Progress callback raised exception: %s", cb_exc)

        summary.total_processing_time = round(time.time() - start_time, 3)
        if len(items) > 0:
            summary.average_processing_time = round(
                summary.total_processing_time / len(items), 3
            )

        return summary
