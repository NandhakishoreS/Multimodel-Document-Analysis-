"""
Phase 9: Multimodal Modality Ablation Module.

Provides controlled modality ablation utilities for LayoutLMv3:
- Condition A: Full Multimodal (Text + Layout + Image)
- Condition B: Text Only (Text active, BBox zeroed, Image zeroed)
- Condition C / F: Text + Layout / No-Image (Text active, BBox active, Image zeroed)
- Condition D / G: Text + Image / No-Layout (Text active, BBox zeroed, Image active)
- Condition E: Layout + Image / No-Text (Text masked to [MASK], BBox active, Image active)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger("phase9_ablation")

ABLATION_CONDITIONS: List[Dict[str, Any]] = [
    {
        "id": "A",
        "name": "Full Multimodal",
        "short": "Full",
        "modalities": "Text + Layout + Image",
        "zero_layout": False,
        "zero_image": False,
        "mask_text": False,
        "description": "All three input channels active (unperturbed baseline).",
    },
    {
        "id": "B",
        "name": "Text Only",
        "short": "Text-Only",
        "modalities": "Text",
        "zero_layout": True,
        "zero_image": True,
        "mask_text": False,
        "description": "Pure textual semantics: bbox set to [0,0,0,0], image zeroed.",
    },
    {
        "id": "C",
        "name": "Text + Layout",
        "short": "No-Image",
        "modalities": "Text + Layout",
        "zero_layout": False,
        "zero_image": True,
        "mask_text": False,
        "description": "Bimodal text and 2D spatial layout; image zeroed.",
    },
    {
        "id": "D",
        "name": "Text + Image",
        "short": "No-Layout",
        "modalities": "Text + Image",
        "zero_layout": True,
        "zero_image": False,
        "mask_text": False,
        "description": "Bimodal text and visual patches; bbox set to [0,0,0,0].",
    },
    {
        "id": "E",
        "name": "Layout + Image",
        "short": "No-Text",
        "modalities": "Layout + Image",
        "zero_layout": False,
        "zero_image": False,
        "mask_text": True,
        "description": "Pure spatial and visual features; text tokens replaced with [MASK].",
    },
    {
        "id": "F",
        "name": "No-Image Model",
        "short": "No-Image-Alias",
        "modalities": "Text + Layout",
        "zero_layout": False,
        "zero_image": True,
        "mask_text": False,
        "description": "Explicit verification of vision-free performance.",
    },
    {
        "id": "G",
        "name": "No-Layout Model",
        "short": "No-Layout-Alias",
        "modalities": "Text + Image",
        "zero_layout": True,
        "zero_image": False,
        "mask_text": False,
        "description": "Explicit verification of spatial-free performance.",
    },
]


def apply_batch_ablation(
    batch: Dict[str, torch.Tensor],
    condition: Dict[str, Any],
    mask_token_id: int = 50264,
) -> Dict[str, torch.Tensor]:
    """
    Apply modality ablation to a batch in-place or on clone without mutating source dataset.
    """
    ablated: Dict[str, torch.Tensor] = {
        "input_ids": batch["input_ids"].clone(),
        "attention_mask": batch["attention_mask"].clone(),
        "bbox": batch["bbox"].clone(),
        "pixel_values": batch["pixel_values"].clone(),
        "labels": batch.get("labels"),
        "word_ids": batch.get("word_ids"),
    }

    # 1. Zero out layout (all coordinates set to [0, 0, 0, 0])
    if condition.get("zero_layout", False):
        ablated["bbox"] = torch.zeros_like(ablated["bbox"])

    # 2. Zero out image (pixel_values set to 0.0)
    if condition.get("zero_image", False):
        ablated["pixel_values"] = torch.zeros_like(ablated["pixel_values"])

    # 3. Mask out text (all non-special tokens replaced with mask_token_id)
    if condition.get("mask_text", False):
        # preserve [CLS] (usually id=0), [SEP] (usually id=2), and padding (id=1)
        special_mask = (ablated["input_ids"] == 0) | (ablated["input_ids"] == 1) | (ablated["input_ids"] == 2)
        ablated["input_ids"][~special_mask] = mask_token_id

    return ablated
