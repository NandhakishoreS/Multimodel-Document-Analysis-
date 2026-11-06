"""SROIE dataset processor: parses raw SROIE records into the unified document representation."""

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

from preprocessing.bbox_utils import normalize_bbox
from preprocessing.text_utils import clean_text

logger = logging.getLogger(__name__)


def _normalize_str(s: str) -> str:
    """NFKC-normalise, lowercase, collapse whitespace."""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def align_sroie_entity_labels(
    words: List[str],
    entities: Dict[str, Any],
) -> List[str]:
    """
    Generate token-level BIO tags by matching SROIE entity values against OCR words.

    The SROIE HuggingFace dataset presents OCR "words" that are already
    multi-word strings (e.g. the entire company name is a single word entry).
    The entity values in the annotations are substrings of — or equal to —
    those OCR word strings.

    Strategy (applied in priority order):
    1. Exact match: normalised(word) == normalised(entity_value)
    2. Contains match: normalised(entity_value) in normalised(word)
       or normalised(word) in normalised(entity_value)  [for truncated values]
    3. Multi-word sequence match: entity_value tokens align with consecutive words.

    Each word can only be assigned one entity type (first match wins).
    """
    labels = ["O"] * len(words)
    norm_words = [_normalize_str(w) for w in words]

    # Priority order: longer / more specific entities first to prevent greedy
    # short matches from consuming tokens needed by longer entities.
    entity_order = ["address", "company", "date", "total"]

    for ent_type in entity_order:
        raw_val = str(entities.get(ent_type, "")).strip()
        if not raw_val:
            continue

        norm_val = _normalize_str(raw_val)
        if not norm_val:
            continue

        tag_b = f"B-{ent_type.upper()}"
        tag_i = f"I-{ent_type.upper()}"

        # ── Strategy 1 & 2: single-word match ─────────────────────────────
        # Most SROIE OCR words are multi-word strings that contain the entity
        # as a substring, or vice-versa.
        for idx, nw in enumerate(norm_words):
            if labels[idx] != "O":
                continue
            if nw == norm_val or norm_val in nw or nw in norm_val:
                labels[idx] = tag_b
                break

        # ── Strategy 3: multi-word sequence fallback ──────────────────────
        # Only attempted if no single-word match was found.
        if tag_b not in labels:
            val_tokens = norm_val.split()
            k = len(val_tokens)
            for i in range(len(norm_words) - k + 1):
                window = norm_words[i: i + k]
                if window == val_tokens:
                    if all(labels[i + j] == "O" for j in range(k)):
                        labels[i] = tag_b
                        for j in range(1, k):
                            labels[i + j] = tag_i
                        break

        # ── Strategy 4: partial-token overlap (last resort for total/date) ─
        if tag_b not in labels and ent_type in ("total", "date"):
            for idx, nw in enumerate(norm_words):
                if labels[idx] != "O":
                    continue
                # entity value appears as a contiguous substring inside the word
                if len(norm_val) >= 3 and norm_val in nw:
                    labels[idx] = tag_b
                    break
                # word appears as a contiguous substring inside entity value
                if len(nw) >= 3 and nw in norm_val:
                    labels[idx] = tag_b
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
            "date":    str(entities.get("date", "")),
            "address": str(entities.get("address", "")),
            "total":   str(entities.get("total", "")),
        },
        "relations": [],
    }
