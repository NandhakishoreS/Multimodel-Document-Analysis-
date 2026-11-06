"""
Phase 11 — Institutional Financial Document Intelligence & Searchable Reporting.

New application/business-intelligence layer built on top of the validated
Phase 1–10 multimodal document understanding pipeline.

Public API:
    from phase11.schema import FinancialDocument, ClassificationResult
    from phase11.classifier import BillCategoryClassifier
    from phase11.extractor import FinancialDocumentExtractor
    from phase11.database import FinancialDocumentDB
    from phase11.export import DocumentExporter
    from phase11.ingestion import (
        DocumentIngestionService,
        IngestionConfig,
        IngestionItem,
        BatchSummary,
        ProcessingStatus,
        DuplicatePolicy,
    )
"""

from phase11.schema import FinancialDocument, ClassificationResult, LineItem
from phase11.classifier import BillCategoryClassifier
from phase11.extractor import FinancialDocumentExtractor
from phase11.database import FinancialDocumentDB
from phase11.export import DocumentExporter
from phase11.ingestion import (
    DocumentIngestionService,
    IngestionConfig,
    IngestionItem,
    BatchSummary,
    ProcessingStatus,
    DuplicatePolicy,
)

__all__ = [
    "FinancialDocument",
    "ClassificationResult",
    "LineItem",
    "BillCategoryClassifier",
    "FinancialDocumentExtractor",
    "FinancialDocumentDB",
    "DocumentExporter",
    "DocumentIngestionService",
    "IngestionConfig",
    "IngestionItem",
    "BatchSummary",
    "ProcessingStatus",
    "DuplicatePolicy",
]
