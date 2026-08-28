"""SROIE dataset processor: parses raw SROIE records into the unified document representation."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import numpy as np

from preprocessing.bbox_utils import normalize_bbox, validate_bbox
from preprocessing.text_utils import clean_text
from preprocessing.label_mapping import SROIE_LABELS

logger = logging.getLogger(__name__)


def align_sroie_entity_labels(
    words: List[str],
    entities: Dict[str, Any],
) -> List[str]:
    """
    Generate token-level BIO tags by matching OCR words against SROIE key entities.
    Entities: company, date, address, total.
    """
    labels = ["O"] * len(words)
    cleaned_words = [clean_text(w).lower() for w in words]
    doc_text_joined = " ".join(cleaned_words)

    # Order of entity matching: company, address, date, total
    for ent_type in ["company", "address", "date", "total"]:
        ent_val = str(entities.get(ent_type, "")).strip()
        if not ent_val:
            continue

        ent_tokens = [clean_text(t).lower() for t in ent_val.split() if clean_text(t).strip()]
        if not ent_tokens:
            continue

        # Substring / sub-sequence matching
        k = len(ent_tokens)
        tag_prefix = ent_type.upper()

        for i in range(len(cleaned_words) - k + 1):
            window = cleaned_words[i : i + k]
            if window == ent_tokens:
                # Check if unassigned
                if all(labels[i + j] == "O" for j in range(k)):
                    labels[i] = f"B-{tag_prefix}"
                    for j in range(1, k):
                        labels[i + j] = f"I-{tag_prefix}"

    # Fallback partial matching for total & date if exact multi-word sequence missed
    total_val = str(entities.get("total", "")).strip().lower()
    if total_val and "B-TOTAL" not in labels:
        for idx, w in enumerate(cleaned_words):
            if w == total_val or (len(total_val) > 2 and total_val in w):
                if labels[idx] == "O":
                    labels[idx] = "B-TOTAL"
                    break

    date_val = str(entities.get("date", "")).strip().lower()
    if date_val and "B-DATE" not in labels:
        for idx, w in enumerate(cleaned_words):
            if w == date_val or (len(date_val) > 3 and date_val in w):
                if labels[idx] == "O":
                    labels[idx] = "B-DATE"
                    break

    return labels


def process_sroie_sample(
    item: Dict[str, Any],
    split_name: str,
    output_images_dir: Path,
) -> Dict[str, Any]:
    """
    Convert a raw SROIE HuggingFace item into the unified JSON schema.
    """
    sample_id = str(item.get("key", f"sroie_{split_name}"))
    img: Image.Image = item["image"]
    width, height = img.size

    # Save image to processed image directory
    image_filename = f"{sample_id}.png"
    image_path = output_images_dir / image_filename
    if not image_path.exists():
        img.save(image_path)

    raw_words = item.get("words", [])
    raw_bboxes = item.get("bboxes", [])
    entities = item.get("entities", {})

    cleaned_words: List[str] = []
    norm_bboxes: List[List[int]] = []

    for w, b in zip(raw_words, raw_bboxes):
        w_clean = clean_text(w)
        b_norm = normalize_bbox(b, width=width, height=height)
        cleaned_words.append(w_clean)
        norm_bboxes.append(b_norm)

    labels = align_sroie_entity_labels(cleaned_words, entities)

    return {
        "id": sample_id,
        "dataset": "sroie",
        "split": split_name,
        "image_path": str(image_path.as_posix()),
        "width": width,
        "height": height,
        "words": cleaned_words,
        "bboxes": norm_bboxes,
        "labels": labels,
        "entities": {
            "company": str(entities.get("company", "")),
            "date": str(entities.get("date", "")),
            "address": str(entities.get("address", "")),
            "total": str(entities.get("total", "")),
        },
        "relations": [],
    }
