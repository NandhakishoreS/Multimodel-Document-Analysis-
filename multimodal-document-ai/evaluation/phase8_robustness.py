"""
Phase 8: Robustness Testing, Stress Evaluation & Sensitivity Analysis Module.

Provides:
1. In-memory perturbation generators:
   - OCR character corruption and substitution (Low=5%, Medium=10%, High=20%)
   - Missing / masked text (Low=5%, Medium=10%, High=20%)
   - Layout bounding-box jitter (+/-2%, +/-5%, +/-10%) with strict valid box clamping
   - Image degradations: Gaussian blur, downsampling/resolution reduction, brightness/contrast variation
   - Composite multi-modal degradation
2. RobustnessDocumentDataset wrapper (ensures original test data is never mutated)
3. Delta and degradation metric calculations
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

# OCR confusable characters
OCR_CONFUSIONS = {
    "0": ["O", "o", "D"],
    "O": ["0", "Q", "C"],
    "o": ["0", "c", "e"],
    "1": ["I", "l", "|", "i"],
    "I": ["1", "l", "|", "T"],
    "l": ["1", "I", "|"],
    "5": ["S", "s"],
    "S": ["5", "$"],
    "s": ["5"],
    "8": ["B"],
    "B": ["8"],
    "2": ["Z", "z"],
    "Z": ["2"],
    ",": ["."],
    ".": [","],
    "-": ["_"],
}


# ---------------------------------------------------------------------------
# Perturbation Engines
# ---------------------------------------------------------------------------

def apply_ocr_char_corruption(
    words: List[str],
    rate: float = 0.10,
    seed: Optional[int] = 42,
) -> List[str]:
    """
    Simulate OCR character corruption and substitution with probability `rate` per word.
    Original list is NOT mutated.
    """
    if rate <= 0.0:
        return list(words)

    rng = random.Random(seed)
    corrupted = []
    for w in words:
        if not w:
            corrupted.append(w)
            continue
        if rng.random() < rate:
            chars = list(w)
            char_idx = rng.randint(0, len(chars) - 1)
            c = chars[char_idx]
            if c in OCR_CONFUSIONS:
                chars[char_idx] = rng.choice(OCR_CONFUSIONS[c])
            elif c.isalpha():
                # Random lowercase/uppercase or adjacent typo
                chars[char_idx] = rng.choice(["e", "a", "o", "i", "n", "r", "s", "t"])
            elif c.isdigit():
                chars[char_idx] = rng.choice(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
            corrupted.append("".join(chars))
        else:
            corrupted.append(w)
    return corrupted


def apply_missing_text_masking(
    words: List[str],
    rate: float = 0.10,
    seed: Optional[int] = 42,
    mask_token: str = "***",
) -> List[str]:
    """
    Simulate missing or unreadable OCR tokens by replacing words with mask_token
    at given probability `rate`. Preserves word count and alignment.
    Original list is NOT mutated.
    """
    if rate <= 0.0:
        return list(words)

    rng = random.Random(seed)
    return [mask_token if rng.random() < rate and w else w for w in words]


def apply_layout_bbox_jitter(
    bboxes: List[List[int]],
    max_pct_jitter: float = 0.05,
    seed: Optional[int] = 42,
    img_box_bound: int = 1000,
) -> List[List[int]]:
    """
    Simulate bounding box localization noise by shifting coordinates by uniform
    random offset in [-max_delta, +max_delta], where max_delta = max_pct_jitter * 1000.
    Ensures boxes remain strictly valid:
      0 <= x0 < x1 <= 1000
      0 <= y0 < y1 <= 1000
    Original list is NOT mutated.
    """
    if max_pct_jitter <= 0.0:
        return [list(b) for b in bboxes]

    rng = random.Random(seed)
    max_delta = int(max_pct_jitter * img_box_bound)
    jittered = []

    for b in bboxes:
        if len(b) != 4:
            jittered.append(list(b))
            continue
        x0, y0, x1, y1 = b
        dx0 = rng.randint(-max_delta, max_delta)
        dy0 = rng.randint(-max_delta, max_delta)
        dx1 = rng.randint(-max_delta, max_delta)
        dy1 = rng.randint(-max_delta, max_delta)

        new_x0 = max(0, min(img_box_bound - 2, x0 + dx0))
        new_y0 = max(0, min(img_box_bound - 2, y0 + dy0))
        new_x1 = max(new_x0 + 1, min(img_box_bound, x1 + dx1))
        new_y1 = max(new_y0 + 1, min(img_box_bound, y1 + dy1))

        # Enforce minimum box dimensions of 1x1
        if new_x1 <= new_x0:
            new_x1 = min(img_box_bound, new_x0 + 1)
        if new_y1 <= new_y0:
            new_y1 = min(img_box_bound, new_y0 + 1)

        jittered.append([new_x0, new_y0, new_x1, new_y1])

    return jittered


def apply_image_degradation(
    img: Image.Image,
    degradation_type: str,
    severity: float,
) -> Image.Image:
    """
    Apply realistic image degradation to PIL Image.
    Returns a new image instance (original is not modified in-place).
    """
    out_img = img.copy()

    if degradation_type == "blur":
        # severity is blur radius sigma (e.g. 1.0, 2.0, 3.5)
        out_img = out_img.filter(ImageFilter.GaussianBlur(radius=severity))

    elif degradation_type == "resolution":
        # severity is downscale factor (e.g. 0.5 downscales by 2x then upscales)
        w, h = out_img.size
        small_w = max(16, int(w * severity))
        small_h = max(16, int(h * severity))
        out_img = out_img.resize((small_w, small_h), Image.Resampling.BILINEAR)
        out_img = out_img.resize((w, h), Image.Resampling.BILINEAR)

    elif degradation_type == "brightness":
        # severity is brightness factor (e.g. 0.6 = darker, 1.4 = washed out)
        enhancer = ImageEnhance.Brightness(out_img)
        out_img = enhancer.enhance(severity)

    elif degradation_type == "contrast":
        # severity is contrast factor (e.g. 0.5 = low contrast)
        enhancer = ImageEnhance.Contrast(out_img)
        out_img = enhancer.enhance(severity)

    return out_img


# ---------------------------------------------------------------------------
# Robustness Document Dataset Wrapper
# ---------------------------------------------------------------------------

class RobustnessDocumentDataset(torch.utils.data.Dataset):
    """
    A Dataset wrapper that decorates an existing LayoutLMv3DocumentDataset.
    Applies on-the-fly perturbations to deep-copied samples during __getitem__.
    Guarantees:
    - Base dataset instance and files on disk are completely unmutated.
    - Ground truth annotations (labels, words for gold spans) remain unchanged.
    """

    def __init__(
        self,
        base_dataset: Any,
        perturbation_config: Dict[str, Any],
        seed: int = 42,
    ) -> None:
        self.base_dataset = base_dataset
        self.perturbation_config = perturbation_config
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Deepcopy sample dictionary so original is never touched
        sample = copy.deepcopy(self.base_dataset.samples[idx])
        sample_seed = self.seed + idx

        cfg = self.perturbation_config
        ptype = cfg.get("type", "clean")

        # 1. Text Perturbations
        if ptype == "ocr_char":
            sample["words"] = apply_ocr_char_corruption(
                sample["words"], rate=cfg.get("rate", 0.10), seed=sample_seed
            )
        elif ptype == "missing_text":
            sample["words"] = apply_missing_text_masking(
                sample["words"], rate=cfg.get("rate", 0.10), seed=sample_seed
            )

        # 2. Layout BBox Perturbations
        elif ptype == "layout_jitter":
            sample["bboxes"] = apply_layout_bbox_jitter(
                sample["bboxes"], max_pct_jitter=cfg.get("jitter", 0.05), seed=sample_seed
            )

        # 3. Composite Perturbation
        elif ptype == "composite":
            sample["words"] = apply_ocr_char_corruption(
                sample["words"], rate=cfg.get("ocr_rate", 0.10), seed=sample_seed
            )
            sample["bboxes"] = apply_layout_bbox_jitter(
                sample["bboxes"], max_pct_jitter=cfg.get("layout_jitter", 0.05), seed=sample_seed
            )

        # Tokenize perturbed sample
        # Note: If image perturbation is requested, encode with perturbed image
        encoding = self._encode_sample_with_image_perturbation(sample, cfg)

        raw_word_ids: List[Optional[int]] = encoding.get("word_ids", [])
        WORD_ID_PAD = -100
        word_ids_tensor = torch.tensor(
            [wid if wid is not None else WORD_ID_PAD for wid in raw_word_ids],
            dtype=torch.long,
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "bbox": encoding["bbox"].squeeze(0),
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"].squeeze(0),
            "word_ids": word_ids_tensor,
            "sample_id": sample.get("id", str(idx)),
        }

    def _encode_sample_with_image_perturbation(
        self,
        sample: Dict[str, Any],
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Encode sample, applying image degradation if configured."""
        img_deg_type = cfg.get("img_type")
        img_deg_sev = cfg.get("img_severity")

        if img_deg_type and img_deg_sev:
            # Load and degrade image in memory
            rel_path = sample.get("image_path", "")
            full_path = self.base_dataset.project_root / rel_path
            if full_path.exists():
                orig_img = Image.open(full_path).convert("RGB")
                deg_img = apply_image_degradation(orig_img, img_deg_type, img_deg_sev)

                # Custom encode using tokenizer_wrapper processor
                processor = self.base_dataset.tokenizer_wrapper.processor
                words = sample.get("words", [])
                boxes = sample.get("bboxes", [])
                word_labels = [
                    self.base_dataset.dataset_config.label2id.get(lbl, 0)
                    for lbl in sample.get("labels", [])
                ]
                enc = processor(
                    deg_img,
                    words,
                    boxes=boxes,
                    word_labels=word_labels,
                    truncation=True,
                    padding="max_length",
                    max_length=self.base_dataset.tokenizer_wrapper.max_seq_length,
                    return_tensors="pt",
                )
                return {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "bbox": enc["bbox"],
                    "pixel_values": enc["pixel_values"],
                    "labels": enc["labels"],
                    "word_ids": enc.word_ids(0) if hasattr(enc, "word_ids") else [],
                }

        # Default encoding without image perturbation
        return self.base_dataset.tokenizer_wrapper.encode_sample(
            sample=sample,
            label2id=self.base_dataset.dataset_config.label2id,
            project_root=self.base_dataset.project_root,
            return_tensors="pt",
        )


# ---------------------------------------------------------------------------
# Metric Calculation Utilities
# ---------------------------------------------------------------------------

@dataclass
class RobustnessResultRow:
    dataset: str
    condition_name: str
    degradation_type: str
    severity: str
    precision: float
    recall: float
    macro_f1: float
    micro_f1: float
    delta_macro_f1: float
    delta_micro_f1: float
    relative_macro_deg_pct: float
    per_entity: Dict[str, float] = field(default_factory=dict)
    notes: str = ""
