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


class LayoutLMv3DocumentDataset(Dataset):
    """
    PyTorch Dataset that converts Phase-2 JSONL samples into LayoutLMv3 tensors.

    Parameters
    ----------
    jsonl_path : str or Path
        Path to the processed split file, e.g. processed_data/sroie/train.jsonl.
    dataset_config : LayoutLMv3DatasetConfig
        Label schema for this dataset (loaded from configs/labels/*.yaml).
    tokenizer_wrapper : LayoutLMv3TokenizerWrapper
        Wraps the HuggingFace processor; handles tokenization and label alignment.
    split : str
        "train", "val", or "test" — used for logging only.
    num_samples : int or None
        If set, randomly subsample this many records (for smoke/small stages).
        Sampling uses a fixed seed for reproducibility.
    seed : int
        Random seed for subsampling (default 42).
    project_root : Path or None
        Used to resolve relative image_path values in JSONL records.
        Defaults to the package-level _PROJECT_ROOT.

    Returns per __getitem__
    -----------------------
    dict with keys:
        input_ids      : Tensor (seq_len,)
        attention_mask : Tensor (seq_len,)
        bbox           : Tensor (seq_len, 4)
        pixel_values   : Tensor (3, 224, 224)
        labels         : Tensor (seq_len,)
        sample_id      : str  — original document id, for error analysis
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

        # Load all raw JSONL records into memory (dicts, not tensors yet).
        self.samples: List[Dict[str, Any]] = self._load_jsonl(self.jsonl_path)

        # Optionally subsample for smoke/small stage experiments.
        if num_samples is not None and num_samples < len(self.samples):
            rng = random.Random(seed)
            self.samples = rng.sample(self.samples, num_samples)
            logger.info(
                "[%s/%s] Subsampled to %d / %d samples (seed=%d)",
                dataset_config.dataset, split, len(self.samples),
                len(self._load_jsonl(self.jsonl_path)), seed,
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

        # Squeeze batch dimension added by the processor (we handle batching in DataLoader).
        item: Dict[str, Any] = {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "bbox":           encoding["bbox"].squeeze(0),
            "pixel_values":   encoding["pixel_values"].squeeze(0),
            "labels":         encoding["labels"].squeeze(0),
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

    Parameters
    ----------
    dataset_name : str
        "sroie" | "funsd" | "cord"
    dataset_config : LayoutLMv3DatasetConfig
    tokenizer_wrapper : LayoutLMv3TokenizerWrapper
    processed_data_dir : str or Path
        Root of processed_data/, e.g. "processed_data".
    batch_size : int
        Per-device batch size.
    num_workers : int
        DataLoader worker processes.  0 = main process (safe on Windows).
    num_train_samples : int or None
        Subsample train split (smoke/small experiments).
    seed : int
    project_root : Path or None

    Returns
    -------
    dict with keys "train", "val", "test" mapping to DataLoader instances.
    """
    from torch.utils.data import DataLoader

    root = Path(processed_data_dir)
    loaders: Dict[str, DataLoader] = {}

    split_configs = [
        ("train", num_train_samples, True),   # (split, num_samples, shuffle)
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
            # collate_fn not needed — __getitem__ already returns fixed-length tensors.
            pin_memory=False,  # CPU-only machine; pin_memory has no benefit.
        )
        logger.info(
            "Built %s DataLoader: %d samples, %d batches (batch_size=%d)",
            split_name, len(ds), len(loaders[split_name]), batch_size,
        )

    return loaders
