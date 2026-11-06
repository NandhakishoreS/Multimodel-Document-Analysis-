"""
Phase 11 — Financial Document Extractor.

Maps a DocumentInferenceResult (produced by the validated Phase 1–10
inference pipeline) into a FinancialDocument (Phase 11 schema).

This module is the primary integration point between the Phase 10 ML system
and the Phase 11 business-intelligence layer. It must NOT modify or import
anything from the training, evaluation, or preprocessing stacks.

Supported benchmark domains and their entity mappings:
    SROIE  → COMPANY=vendor_name, DATE=invoice_date, ADDRESS=address, TOTAL=total_amount
    CORD   → MENU=line_items, SUBTOTAL=subtotal, TOTAL=total_amount
    FUNSD  → HEADER=document_type hint, QUESTION/ANSWER pairs are parsed for KV fields
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from phase11.classifier import BillCategoryClassifier
from phase11.money_parser import (
    parse_amount_to_float,
    parse_monetary_string,
    select_best_amount,
)
from phase11.schema import FinancialDocument, LineItem

# ---------------------------------------------------------------------------
# Regex helpers for date parsing
# ---------------------------------------------------------------------------

# Matches dates in common formats: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, DD.MM.YYYY
_DATE_RE = re.compile(
    r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})"
)

# Singleton classifier instance (stateless — safe to share)
_DEFAULT_CLASSIFIER = BillCategoryClassifier()


def _parse_amount(text: str) -> Optional[float]:
    """
    Extract the first plausible monetary amount from a string.

    Delegates to the canonical money_parser module so that decimal points,
    Indian comma grouping, and currency symbols are all handled correctly.
    Returns float or None if no amount found.

    .. deprecated::
        Call ``parse_amount_to_float`` from ``phase11.money_parser`` directly
        in new code; this wrapper exists for backward-compatibility with tests
        that import ``_parse_amount`` from this module.
    """
    return parse_amount_to_float(text)


def _parse_date(text: str) -> Optional[str]:
    """
    Extract the first date-like string from text.
    Returns the raw string (not normalised to ISO) to preserve OCR fidelity.
    """
    if not text:
        return None
    m = _DATE_RE.search(text)
    return m.group(0) if m else text.strip() or None


def _mean_confidence(entities: List[Any]) -> float:
    """Compute mean confidence over a list of ExtractedEntity objects."""
    if not entities:
        return 0.0
    return sum(e.confidence for e in entities) / len(entities)


# ---------------------------------------------------------------------------
# SROIE extraction
# ---------------------------------------------------------------------------

def _extract_sroie(
    raw_entities: Dict[str, str],
    entities: List[Any],
    ocr_text: str,
    filename: str,
    model_name: str,
    doc_id: str,
    classifier: BillCategoryClassifier,
) -> FinancialDocument:
    vendor = raw_entities.get("COMPANY") or None
    date_raw = raw_entities.get("DATE") or None
    address = raw_entities.get("ADDRESS") or None
    total_raw = raw_entities.get("TOTAL") or None

    total_amount = _parse_amount(total_raw) if total_raw else None
    invoice_date = _parse_date(date_raw) if date_raw else None

    clf_result = classifier.classify(
        ocr_text=ocr_text,
        vendor_name=vendor,
        entity_text=" ".join(raw_entities.values()),
    )

    return FinancialDocument(
        document_id=doc_id,
        filename=filename,
        source_file=filename,
        document_type="receipt",
        benchmark_domain="SROIE",
        model_name=model_name,
        vendor_name=vendor,
        address=address,
        invoice_date=invoice_date,
        total_amount=total_amount,
        category=clf_result.category,
        sub_category=clf_result.sub_category,
        classification_confidence=clf_result.confidence,
        classification_evidence=clf_result.evidence,
        ai_category=clf_result.category,
        ai_confidence=clf_result.confidence,
        final_category=clf_result.category,
        requires_review=clf_result.requires_review,
        classification_reason=clf_result.reason,
        extraction_confidence=round(_mean_confidence(entities), 4),
        ocr_word_count=len(ocr_text.split()),
        ocr_raw_text=ocr_text,
    )


# ---------------------------------------------------------------------------
# CORD extraction
# ---------------------------------------------------------------------------

def _extract_cord(
    raw_entities: Dict[str, str],
    entities: List[Any],
    ocr_text: str,
    filename: str,
    model_name: str,
    doc_id: str,
    classifier: BillCategoryClassifier,
) -> FinancialDocument:
    subtotal_raw = raw_entities.get("SUBTOTAL") or None
    total_raw = raw_entities.get("TOTAL") or None
    menu_raw = raw_entities.get("MENU") or None

    subtotal = _parse_amount(subtotal_raw) if subtotal_raw else None
    total_amount = _parse_amount(total_raw) if total_raw else None

    # Build line items from all MENU entities
    line_items: List[LineItem] = []
    for e in entities:
        label = getattr(e, "label", "")
        if label.upper().replace("B-", "").replace("I-", "") == "MENU":
            line_items.append(LineItem(description=e.text.strip()))

    clf_result = classifier.classify(
        ocr_text=ocr_text,
        vendor_name=None,
        entity_text=menu_raw or "",
    )

    return FinancialDocument(
        document_id=doc_id,
        filename=filename,
        source_file=filename,
        document_type="receipt",
        benchmark_domain="CORD",
        model_name=model_name,
        subtotal=subtotal,
        total_amount=total_amount,
        line_items=line_items,
        category=clf_result.category,
        sub_category=clf_result.sub_category,
        classification_confidence=clf_result.confidence,
        classification_evidence=clf_result.evidence,
        ai_category=clf_result.category,
        ai_confidence=clf_result.confidence,
        final_category=clf_result.category,
        requires_review=clf_result.requires_review,
        classification_reason=clf_result.reason,
        extraction_confidence=round(_mean_confidence(entities), 4),
        ocr_word_count=len(ocr_text.split()),
        ocr_raw_text=ocr_text,
    )


# ---------------------------------------------------------------------------
# FUNSD extraction
# ---------------------------------------------------------------------------

def _extract_funsd(
    raw_entities: Dict[str, str],
    entities: List[Any],
    ocr_text: str,
    filename: str,
    model_name: str,
    doc_id: str,
    classifier: BillCategoryClassifier,
) -> FinancialDocument:
    header = raw_entities.get("HEADER") or None
    answer = raw_entities.get("ANSWER") or None

    clf_result = classifier.classify(
        ocr_text=ocr_text,
        vendor_name=header,
        entity_text=answer or "",
    )

    return FinancialDocument(
        document_id=doc_id,
        filename=filename,
        source_file=filename,
        document_type="form",
        benchmark_domain="FUNSD",
        model_name=model_name,
        vendor_name=header,
        category=clf_result.category,
        sub_category=clf_result.sub_category,
        classification_confidence=clf_result.confidence,
        classification_evidence=clf_result.evidence,
        ai_category=clf_result.category,
        ai_confidence=clf_result.confidence,
        final_category=clf_result.category,
        requires_review=clf_result.requires_review,
        classification_reason=clf_result.reason,
        extraction_confidence=round(_mean_confidence(entities), 4),
        ocr_word_count=len(ocr_text.split()),
        ocr_raw_text=ocr_text,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FinancialDocumentExtractor:
    """
    Converts a DocumentInferenceResult into a FinancialDocument.

    This class is the sole integration point with the Phase 10 inference
    pipeline and must not import training or evaluation code.

    Usage
    -----
    from inference.final_inference import UnseenDocumentInferencePipeline
    from phase11.extractor import FinancialDocumentExtractor

    pipeline  = UnseenDocumentInferencePipeline(dataset_type="sroie")
    extractor = FinancialDocumentExtractor()

    result    = pipeline.predict_image(image)
    fin_doc   = extractor.extract(result, filename="invoice.png")
    """

    def __init__(self, classifier: Optional[BillCategoryClassifier] = None):
        self._classifier = classifier or _DEFAULT_CLASSIFIER

    def extract(
        self,
        inference_result: Any,
        filename: str = "",
        document_id: Optional[str] = None,
    ) -> FinancialDocument:
        """
        Map a DocumentInferenceResult → FinancialDocument.

        Parameters
        ----------
        inference_result : DocumentInferenceResult
            Output from UnseenDocumentInferencePipeline.predict_image().
        filename : str
            Original filename of the uploaded document.
        document_id : Optional[str]
            Explicit UUID; auto-generated if None.

        Returns
        -------
        FinancialDocument (verification_status="pending" by default)
        """
        doc_id = document_id or str(uuid.uuid4())

        domain = getattr(inference_result, "document_type", "SROIE").upper()
        raw_entities: Dict[str, str] = getattr(inference_result, "raw_entities_dict", {}) or {}
        entities: List[Any] = getattr(inference_result, "entities", []) or []
        ocr_text: str = getattr(inference_result, "ocr_raw_text", "") or ""
        model_name: str = getattr(inference_result, "model_name", "") or ""

        fn = filename or (Path(getattr(inference_result, "image_path", "") or "").name)

        if domain == "SROIE":
            return _extract_sroie(raw_entities, entities, ocr_text, fn, model_name, doc_id, self._classifier)
        elif domain == "CORD":
            return _extract_cord(raw_entities, entities, ocr_text, fn, model_name, doc_id, self._classifier)
        elif domain == "FUNSD":
            return _extract_funsd(raw_entities, entities, ocr_text, fn, model_name, doc_id, self._classifier)
        else:
            # Generic fallback: use whatever entities are available
            clf_result = self._classifier.classify(ocr_text=ocr_text)
            return FinancialDocument(
                document_id=doc_id,
                filename=fn,
                source_file=fn,
                document_type="invoice",
                benchmark_domain=domain,
                model_name=model_name,
                category=clf_result.category,
                sub_category=clf_result.sub_category,
                classification_confidence=clf_result.confidence,
                classification_evidence=clf_result.evidence,
                ai_category=clf_result.category,
                ai_confidence=clf_result.confidence,
                final_category=clf_result.category,
                requires_review=clf_result.requires_review,
                classification_reason=clf_result.reason,
                extraction_confidence=round(_mean_confidence(entities), 4),
                ocr_word_count=len(ocr_text.split()),
                ocr_raw_text=ocr_text,
            )
