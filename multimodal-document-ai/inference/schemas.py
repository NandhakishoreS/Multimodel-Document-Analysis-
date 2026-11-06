"""
Inference data structures and schemas for Phase 10 Unseen Document Pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TokenPrediction:
    """Detailed token/word-level prediction."""
    word: str
    label: str
    confidence: float
    bbox: List[int]  # [x0, y0, x1, y1] in original image coordinates or normalized [0, 1000]
    word_index: int


@dataclass
class ExtractedEntity:
    """Reconstructed semantic entity span."""
    label: str
    text: str
    confidence: float
    bbox: List[int]  # Minimum bounding rectangle [x0, y0, x1, y1] encompassing all tokens in original image coordinates
    normalized_bbox: List[int]  # Normalized [0, 1000] bounding box
    start_word: int
    end_word: int
    token_count: int
    min_confidence: float = 1.0


@dataclass
class DocumentInferenceResult:
    """Complete document-level inference result."""
    document_type: str  # SROIE, FUNSD, or CORD
    model_name: str
    image_path: Optional[str]
    image_size: Tuple[int, int]  # (width, height)
    ocr_word_count: int
    entities: List[ExtractedEntity] = field(default_factory=list)
    tokens: List[TokenPrediction] = field(default_factory=list)
    raw_entities_dict: Dict[str, str] = field(default_factory=dict)
    ocr_raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "document_type": self.document_type,
            "model_name": self.model_name,
            "image_size": {
                "width": self.image_size[0],
                "height": self.image_size[1],
            },
            "ocr_word_count": self.ocr_word_count,
            "ocr_raw_text": self.ocr_raw_text,
            "entities": [
                {
                    "label": e.label,
                    "text": e.text,
                    "confidence": round(e.confidence, 4),
                    "min_confidence": round(e.min_confidence, 4),
                    "bbox": e.bbox,
                    "normalized_bbox": e.normalized_bbox,
                    "token_count": e.token_count,
                }
                for e in self.entities
            ],
            "aggregated_fields": self.raw_entities_dict,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export result as formatted JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
