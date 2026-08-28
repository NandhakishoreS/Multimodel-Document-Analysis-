"""CORD dataset processor: parses raw CORD records into the unified document representation."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from preprocessing.bbox_utils import normalize_bbox, quad_to_bbox, validate_bbox
from preprocessing.text_utils import clean_text

logger = logging.getLogger(__name__)


def map_cord_category_to_coarse(category: str) -> str:
    """Map fine-grained CORD categories (e.g., 'menu.nm', 'total.total_price') to standard coarse label."""
    cat = str(category).lower().strip()
    if cat.startswith("menu"):
        return "MENU"
    elif cat.startswith("sub_total") or cat.startswith("subtotal"):
        return "SUBTOTAL"
    elif cat.startswith("total"):
        return "TOTAL"
    elif cat.startswith("void"):
        return "VOID"
    elif cat.startswith("payment") or cat.startswith("pay"):
        return "PAYMENT"
    elif cat in ("etc", "table", "container"):
        return "ETC"
    return "ETC"


def parse_cord_ground_truth(
    gt_raw: Any,
    img_width: int,
    img_height: int,
) -> Tuple[List[str], List[List[int]], List[str]]:
    """
    Parse CORD ground truth JSON structure into flattened words, bounding boxes, and BIO labels.
    """
    if isinstance(gt_raw, str):
        try:
            gt_data = json.loads(gt_raw)
        except Exception:
            return [], [], []
    elif isinstance(gt_raw, dict):
        gt_data = gt_raw
    else:
        return [], [], []

    words: List[str] = []
    bboxes: List[List[int]] = []
    labels: List[str] = []

    valid_lines = gt_data.get("valid_line", [])
    for line in valid_lines:
        raw_cat = line.get("category", "etc")
        coarse_cat = map_cord_category_to_coarse(raw_cat)
        line_words = line.get("words", [])

        for word_idx, w_item in enumerate(line_words):
            text = str(w_item.get("text", "")).strip()
            if not text:
                continue

            quad = w_item.get("quad", {})
            try:
                bbox = quad_to_bbox(quad)
                bbox = validate_bbox(bbox, width=img_width, height=img_height)
            except Exception:
                continue

            prefix = "B-" if word_idx == 0 else "I-"
            label = f"{prefix}{coarse_cat}"

            words.append(text)
            bboxes.append(bbox)
            labels.append(label)

    return words, bboxes, labels


def process_cord_sample(
    item: Dict[str, Any],
    split_name: str,
    output_images_dir: Path,
    sample_index: int = 0,
) -> Dict[str, Any]:
    """
    Convert a raw CORD item into the unified JSON schema.
    """
    sample_id = f"cord_{split_name}_{sample_index}"
    img: Image.Image = item["image"]
    width, height = img.size

    # Save image
    image_filename = f"{sample_id}.png"
    image_path = output_images_dir / image_filename
    if not image_path.exists():
        img.save(image_path)

    gt_raw = item.get("ground_truth", {})
    words, raw_bboxes, labels = parse_cord_ground_truth(gt_raw, img_width=width, img_height=height)

    cleaned_words: List[str] = []
    norm_bboxes: List[List[int]] = []

    for w, b in zip(words, raw_bboxes):
        w_clean = clean_text(w)
        b_norm = normalize_bbox(b, width=width, height=height)
        cleaned_words.append(w_clean)
        norm_bboxes.append(b_norm)

    return {
        "id": sample_id,
        "dataset": "cord",
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
