"""Preprocessing package for Multimodal Document Understanding."""

from .bbox_utils import normalize_bbox, denormalize_bbox, validate_bbox, quad_to_bbox, extract_spatial_features, compute_iou
from .text_utils import clean_text, extract_lexical_features
from .label_mapping import (
    SROIE_LABELS,
    FUNSD_LABELS,
    CORD_LABELS,
    UNIFIED_TAXONOMY,
    DATASET_TO_UNIFIED_MAP,
    LabelEncoder,
    get_dataset_label_encoder,
)
from .unified_dataset import UnifiedDocumentDataset, run_preprocessing_pipeline

__all__ = [
    "normalize_bbox",
    "denormalize_bbox",
    "validate_bbox",
    "quad_to_bbox",
    "extract_spatial_features",
    "compute_iou",
    "clean_text",
    "extract_lexical_features",
    "SROIE_LABELS",
    "FUNSD_LABELS",
    "CORD_LABELS",
    "UNIFIED_TAXONOMY",
    "DATASET_TO_UNIFIED_MAP",
    "LabelEncoder",
    "get_dataset_label_encoder",
    "UnifiedDocumentDataset",
    "run_preprocessing_pipeline",
]
