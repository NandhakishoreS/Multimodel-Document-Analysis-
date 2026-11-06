"""
Phase 9: Robustness-Aware Data Augmentation Module.

Provides in-memory augmentation transforms for LayoutLMv3 training:
1. Dynamic Layout Jitter: Multi-scale 2D coordinate shifts with strict [0, 1000] clamping.
2. Dynamic OCR & Typographical Noise: Character confusion, random swaps/drops, and token masking.
3. Joint Multimodal Augmentation: Combines spatial layout perturbations and textual corruptions.
4. RobustnessAugmentedDataset: PyTorch Dataset wrapper guaranteeing zero disk mutation.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from models.layoutlmv3.config import LayoutLMv3DatasetConfig, _PROJECT_ROOT
from models.layoutlmv3.dataset import WORD_ID_PAD
from models.layoutlmv3.tokenizer import LayoutLMv3TokenizerWrapper

logger = logging.getLogger("phase9_augmentation")

# Common OCR confusion dictionary
OCR_CONFUSIONS: Dict[str, str] = {
    "0": "O", "O": "0", "o": "0",
    "1": "l", "l": "1", "I": "1", "i": "1",
    "5": "S", "S": "5", "s": "5",
    "8": "B", "B": "8",
    "2": "Z", "Z": "2",
    "6": "G", "G": "6",
    "9": "g", "g": "9",
    "c": "e", "e": "c",
    "m": "rn", "rn": "m",
    "cl": "d", "d": "cl",
    "vv": "w", "w": "vv",
}


@dataclass
class AugmentationConfig:
    """Configuration parameters for robustness-aware data augmentation."""
    enable_layout_jitter: bool = False
    layout_jitter_prob: float = 0.5
    layout_jitter_min: float = 0.02
    layout_jitter_max: float = 0.08

    enable_text_noise: bool = False
    text_noise_prob: float = 0.5
    ocr_confusion_rate: float = 0.08
    typo_rate: float = 0.04
    token_mask_rate: float = 0.05

    seed: int = 42


def apply_layout_jitter(
    bboxes: List[List[int]],
    jitter_ratio: float,
    rng: random.Random,
) -> List[List[int]]:
    """
    Apply uniform random jitter to bounding box coordinates.
    Guarantees [0, 1000] clamping and non-zero area (x0 < x1, y0 < y1).
    """
    jittered: List[List[int]] = []
    max_shift = int(round(jitter_ratio * 1000.0))
    if max_shift <= 0:
        return [list(b) for b in bboxes]

    for b in bboxes:
        dx0 = rng.randint(-max_shift, max_shift)
        dy0 = rng.randint(-max_shift, max_shift)
        dx1 = rng.randint(-max_shift, max_shift)
        dy1 = rng.randint(-max_shift, max_shift)

        x0 = max(0, min(998, b[0] + dx0))
        y0 = max(0, min(998, b[1] + dy0))
        x1 = max(x0 + 1, min(1000, b[2] + dx1))
        y1 = max(y0 + 1, min(1000, b[3] + dy1))

        jittered.append([x0, y0, x1, y1])

    return jittered


def apply_text_noise(
    words: List[str],
    ocr_confusion_rate: float,
    typo_rate: float,
    token_mask_rate: float,
    rng: random.Random,
) -> List[str]:
    """
    Apply OCR confusion, typo noise, and token masking to word tokens.
    Deep-copies words so input remains unaltered.
    """
    corrupted: List[str] = []
    for w in words:
        if not w:
            corrupted.append(w)
            continue

        # 1. Word-level token masking
        if rng.random() < token_mask_rate:
            corrupted.append("***")
            continue

        chars = list(w)
        mutated = False

        # 2. Character-level OCR confusion
        for idx in range(len(chars)):
            c = chars[idx]
            if c in OCR_CONFUSIONS and rng.random() < ocr_confusion_rate:
                chars[idx] = OCR_CONFUSIONS[c]
                mutated = True

        # 3. Typographical perturbation (swap, drop, insert)
        if rng.random() < typo_rate and len(chars) > 1:
            choice = rng.choice(["swap", "drop", "repeat"])
            pos = rng.randint(0, len(chars) - 2)
            if choice == "swap":
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            elif choice == "drop" and len(chars) > 2:
                chars.pop(pos)
            elif choice == "repeat":
                chars.insert(pos, chars[pos])
            mutated = True

        corrupted.append("".join(chars) if mutated else w)

    return corrupted


class RobustnessAugmentedDataset(Dataset):
    """
    PyTorch Dataset wrapper applying dynamic in-memory augmentations during training.
    Disk files and original annotations are guaranteed to remain 100% immutable.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        dataset_config: LayoutLMv3DatasetConfig,
        tokenizer_wrapper: LayoutLMv3TokenizerWrapper,
        aug_config: Optional[AugmentationConfig] = None,
        is_training: bool = True,
        project_root: Optional[Path] = None,
    ) -> None:
        self.samples = samples
        self.dataset_config = dataset_config
        self.tokenizer_wrapper = tokenizer_wrapper
        self.aug_config = aug_config or AugmentationConfig()
        self.is_training = is_training
        self.project_root = project_root or _PROJECT_ROOT
        self.rng = random.Random(self.aug_config.seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Deep copy to guarantee zero mutation of original dataset sample
        sample = copy.deepcopy(self.samples[idx])

        if self.is_training:
            cfg = self.aug_config

            # Apply Layout Jitter
            if cfg.enable_layout_jitter and self.rng.random() < cfg.layout_jitter_prob:
                jitter_severity = self.rng.uniform(cfg.layout_jitter_min, cfg.layout_jitter_max)
                sample["bboxes"] = apply_layout_jitter(sample.get("bboxes", []), jitter_severity, self.rng)

            # Apply Text Noise
            if cfg.enable_text_noise and self.rng.random() < cfg.text_noise_prob:
                sample["words"] = apply_text_noise(
                    sample.get("words", []),
                    ocr_confusion_rate=cfg.ocr_confusion_rate,
                    typo_rate=cfg.typo_rate,
                    token_mask_rate=cfg.token_mask_rate,
                    rng=self.rng,
                )

        try:
            encoding = self.tokenizer_wrapper.encode_sample(
                sample=sample,
                label2id=self.dataset_config.label2id,
                project_root=self.project_root,
                return_tensors="pt",
            )
        except Exception as exc:
            logger.error("Augmented tokenization failed for sample %s: %s", sample.get("id", idx), exc)
            seq_len = self.tokenizer_wrapper.max_seq_length
            encoding = {
                "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
                "attention_mask": torch.zeros(1, seq_len, dtype=torch.long),
                "bbox": torch.zeros(1, seq_len, 4, dtype=torch.long),
                "pixel_values": torch.zeros(1, 3, 224, 224, dtype=torch.float),
                "labels": torch.full((1, seq_len), -100, dtype=torch.long),
                "word_ids": [None] * seq_len,
            }

        raw_word_ids = encoding.get("word_ids", [])
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
