"""
PyTorch Dataset for LayoutLMv3 token classification.

Wraps the Phase-2 processed JSONL files and applies the LayoutLMv3
tokenizer/processor on each sample, producing ready-to-use tensors.

Design:
- Lazy tokenization: each sample is tokenized in __getitem__, not up-front.
  This avoids OOM on large datasets and allows subset sampling for smoke/small stages.
- Subset mode: pass num_samples to cap the dataset size (used by smoke and small stages).
- Images are loaded from processed_data/{dataset}/images/ using the relative
  image_path stored in the JSONL, resolved against the project root.
- word_ids are stored as a padded int64 tensor (None → -1) so that the training
  loop can correctly reconstruct word-level label sequences without heuristics.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

from models.layoutlmv3.config import LayoutLMv3DatasetConfig, _PROJECT_ROOT
from models.layoutlmv3.tokenizer import LayoutLMv3TokenizerWrapper

logger = logging.getLogger(__name__)

# Sentinel used in the word_ids tensor to represent None (special/padding tokens).
WORD_ID_PAD: int = -1


class LayoutLMv3DocumentDataset(Dataset):
    """
    PyTorch Dataset that converts Phase-2 JSONL samples into LayoutLMv3 tensors.

    Parameters
    ----------
    jsonl_path : str or Path
    dataset_config : LayoutLMv3DatasetConfig
    tokenizer_wrapper : LayoutLMv3TokenizerWrapper
    split : str
    num_samples : int or None
    seed : int
    project_root : Path or None

    Returns per __getitem__
    -----------------------
    dict with keys:
        input_ids      : Tensor (seq_len,)         int64
        attention_mask : Tensor (seq_len,)         int64
        bbox           : Tensor (seq_len, 4)       int64
        pixel_values   : Tensor (3, 224, 224)      float32
        labels         : Tensor (seq_len,)         int64
        word_ids       : Tensor (seq_len,)         int64  — -1 for special/padding tokens
        sample_id      : str
    """

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        dataset_config: LayoutLMv3DatasetConfig,
        tokenizer_wrapper: LayoutLMv3TokenizerWrapper,
        split: str = "train",
        num_samples: Optional[int] = None,
        seed: int = 42,
        project_root: Optional[Path] = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.dataset_config = dataset_config
        self.tokenizer_wrapper = tokenizer_wrapper
        self.split = split
        self.project_root = project_root or _PROJECT_ROOT

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.jsonl_path}")

        self.samples: List[Dict[str, Any]] = self._load_jsonl(self.jsonl_path)

        if num_samples is not None and num_samples < len(self.samples):
            rng = random.Random(seed)
            self.samples = rng.sample(self.samples, num_samples)
            logger.info(
                "[%s/%s] Subsampled to %d samples (seed=%d)",
                dataset_config.dataset, split, len(self.samples), seed,
            )
        else:
            logger.info(
                "[%s/%s] Loaded %d samples from %s",
                dataset_config.dataset, split, len(self.samples), self.jsonl_path.name,
            )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        try:
            encoding = self.tokenizer_wrapper.encode_sample(
                sample=sample,
                label2id=self.dataset_config.label2id,
                project_root=self.project_root,
                return_tensors="pt",
            )
        except Exception as exc:
            logger.error(
                "Tokenization failed for sample %s: %s — returning empty encoding",
                sample.get("id", idx), exc,
            )
            encoding = self._empty_encoding()

        # word_ids list from tokenizer: List[Optional[int]]
        # Convert to int64 tensor replacing None with WORD_ID_PAD (-1).
        raw_word_ids: List[Optional[int]] = encoding.get("word_ids", [])
        word_ids_tensor = torch.tensor(
            [wid if wid is not None else WORD_ID_PAD for wid in raw_word_ids],
            dtype=torch.long,
        )

        item: Dict[str, Any] = {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "bbox":           encoding["bbox"].squeeze(0),
            "pixel_values":   encoding["pixel_values"].squeeze(0),
            "labels":         encoding["labels"].squeeze(0),
            "word_ids":       word_ids_tensor,   # (seq_len,) with -1 for specials
            "sample_id":      sample.get("id", str(idx)),
        }
        return item

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def _empty_encoding(self) -> Dict[str, Any]:
        """Return a zero-filled encoding as a fallback for failed samples."""
        seq_len = self.tokenizer_wrapper.max_seq_length
        return {
            "input_ids":      torch.zeros(1, seq_len, dtype=torch.long),
            "attention_mask": torch.zeros(1, seq_len, dtype=torch.long),
            "bbox":           torch.zeros(1, seq_len, 4, dtype=torch.long),
            "pixel_values":   torch.zeros(1, 3, 224, 224, dtype=torch.float),
            "labels":         torch.full((1, seq_len), -100, dtype=torch.long),
            "word_ids":       torch.full((1, seq_len), WORD_ID_PAD, dtype=torch.long),
        }

    def get_raw_sample(self, idx: int) -> Dict[str, Any]:
        """Return the raw JSONL dict (no tokenization) for error analysis."""
        return self.samples[idx]


def build_dataloaders(
    dataset_name: str,
    dataset_config: LayoutLMv3DatasetConfig,
    tokenizer_wrapper: LayoutLMv3TokenizerWrapper,
    processed_data_dir: Union[str, Path],
    batch_size: int = 2,
    num_workers: int = 0,
    num_train_samples: Optional[int] = None,
    seed: int = 42,
    project_root: Optional[Path] = None,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Build train / val / test DataLoaders for a given dataset.
    """
    from torch.utils.data import DataLoader

    root = Path(processed_data_dir)
    loaders: Dict[str, DataLoader] = {}

    split_configs = [
        ("train", num_train_samples, True),
        ("val",   None,              False),
        ("test",  None,              False),
    ]

    for split_name, n_samples, shuffle in split_configs:
        jsonl_path = root / dataset_name / f"{split_name}.jsonl"
        if not jsonl_path.exists():
            logger.warning("Missing split file: %s — skipping", jsonl_path)
            continue

        ds = LayoutLMv3DocumentDataset(
            jsonl_path=jsonl_path,
            dataset_config=dataset_config,
            tokenizer_wrapper=tokenizer_wrapper,
            split=split_name,
            num_samples=n_samples,
            seed=seed,
            project_root=project_root,
        )

        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        logger.info(
            "Built %s DataLoader: %d samples, %d batches (batch_size=%d)",
            split_name, len(ds), len(loaders[split_name]), batch_size,
        )

    return loaders
