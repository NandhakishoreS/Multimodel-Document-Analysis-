"""FUNSD dataset processor: parses raw FUNSD records into the unified document representation."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

from preprocessing.bbox_utils import normalize_bbox
from preprocessing.text_utils import clean_text
from preprocessing.label_mapping import FUNSD_LABELS

logger = logging.getLogger(__name__)


def process_funsd_sample(
    item: Dict[str, Any],
    split_name: str,
    output_images_dir: Path,
) -> Dict[str, Any]:
    """
    Convert a raw FUNSD item into the unified JSON schema.
    """
    sample_id = str(item.get("id", f"funsd_{split_name}"))
    img: Image.Image = item["image"]
    width, height = img.size

    # Save image — prefix filename with split name so train/val/test index
    # collisions (e.g. FUNSD train row-42 vs test row-42 are different docs)
    # never clobber each other on disk.
    image_filename = f"funsd_{split_name}_{sample_id}.png"
    image_path = output_images_dir / image_filename
    if not image_path.exists():
        img.save(image_path)

    raw_words = item.get("words", [])
    raw_bboxes = item.get("bboxes", [])
    ner_tags = item.get("ner_tags", [])

    cleaned_words: List[str] = []
    norm_bboxes: List[List[int]] = []
    labels: List[str] = []

    for idx, (w, b) in enumerate(zip(raw_words, raw_bboxes)):
        w_clean = clean_text(w)
        b_norm = normalize_bbox(b, width=width, height=height)

        tag_id = ner_tags[idx] if idx < len(ner_tags) else 0
        tag_name = FUNSD_LABELS[tag_id] if tag_id < len(FUNSD_LABELS) else "O"

        cleaned_words.append(w_clean)
        norm_bboxes.append(b_norm)
        labels.append(tag_name)

    return {
        "id": sample_id,
        "dataset": "funsd",
        "split": split_name,
        "image_path": str(image_path.as_posix()),
        "width": width,
        "height": height,
        "words": cleaned_words,
        "bboxes": norm_bboxes,
        "labels": labels,
        "entities": {},
        "relations": [],
    }
