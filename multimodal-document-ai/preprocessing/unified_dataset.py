"""Unified Document Dataset loader and end-to-end preprocessing orchestrator."""

import argparse
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# hf_datasets is imported lazily inside run_preprocessing_pipeline() to avoid
# the local `datasets/` directory shadowing the HuggingFace `datasets` package
# when this module is imported by other scripts (e.g. train_baseline.py) that
# add the project root to sys.path before importing.
# ---------------------------------------------------------------------------

from preprocessing.sroie_processor import process_sroie_sample
from preprocessing.funsd_processor import process_funsd_sample
from preprocessing.cord_processor import process_cord_sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class UnifiedDocumentDataset(Dataset):
    """
    Standard PyTorch / Python Dataset for accessing normalized multimodal document samples.
    """

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        load_images: bool = False,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.load_images = load_images
        self.samples: List[Dict[str, Any]] = []

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Processed dataset file not found: {self.jsonl_path}")

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    self.samples.append(json.loads(line_str))

        logger.info(f"Loaded {len(self.samples)} samples from {self.jsonl_path.name}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = dict(self.samples[idx])
        if self.load_images and "image_path" in item:
            img_p = Path(item["image_path"])
            if img_p.exists():
                item["image"] = Image.open(img_p).convert("RGB")
            else:
                item["image"] = None
        return item


def split_list_with_seed(
    items: List[Any],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Any], List[Any]]:
    """Split a dataset list into train and validation subsets reproducibly."""
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    val_items = shuffled[:val_size]
    train_items = shuffled[val_size:]
    return train_items, val_items


def run_preprocessing_pipeline(
    config_path: str = "configs/preprocessing.yaml",
    target_datasets: Optional[List[str]] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Execute full preprocessing pipeline for SROIE, FUNSD, and CORD.
    Writes normalized JSONL files and saves images into processed_data/.
    """
    # Load HuggingFace datasets package (lazy import avoids local `datasets/` shadow).
    # We temporarily strip project root from sys.path so Python finds the installed
    # HuggingFace `datasets` package rather than the local `datasets/` directory.
    import sys as _sys
    _proj = str(Path(__file__).resolve().parent.parent)
    _removed = [p for p in list(_sys.path) if p == _proj or p == ""]
    for _p in _removed:
        _sys.path.remove(_p)
    _sys.modules.pop("datasets", None)
    import datasets as hf_datasets  # noqa: F401 – intentional lazy import
    for _p in _removed:
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_root = Path(config.get("paths", {}).get("processed_data_dir", "processed_data"))
    seed = config.get("dataset_split", {}).get("random_seed", 42)
    val_ratio = config.get("dataset_split", {}).get("validation_ratio", 0.15)

    stats: Dict[str, Dict[str, int]] = {}
    datasets_to_process = target_datasets or ["sroie", "funsd", "cord"]

    # 1. SROIE Preprocessing
    if "sroie" in datasets_to_process:
        logger.info("=== Preprocessing SROIE ===")
        out_dir = processed_root / "sroie"
        img_dir = out_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        ds_sroie = hf_datasets.load_dataset("jsdnrs/ICDAR2019-SROIE")
        raw_train = list(ds_sroie["train"])
        raw_test = list(ds_sroie["test"])

        train_items, val_items = split_list_with_seed(raw_train, val_ratio=val_ratio, seed=seed)

        splits_map = {"train": train_items, "val": val_items, "test": raw_test}
        stats["sroie"] = {}

        for split_name, items in splits_map.items():
            jsonl_file = out_dir / f"{split_name}.jsonl"
            logger.info(f"Processing SROIE {split_name} ({len(items)} samples) -> {jsonl_file.name}...")
            with open(jsonl_file, "w", encoding="utf-8") as f_out:
                for itm in items:
                    record = process_sroie_sample(itm, split_name=split_name, output_images_dir=img_dir)
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["sroie"][split_name] = len(items)

    # 2. FUNSD Preprocessing
    if "funsd" in datasets_to_process:
        logger.info("=== Preprocessing FUNSD ===")
        out_dir = processed_root / "funsd"
        img_dir = out_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        ds_funsd = hf_datasets.load_dataset("nielsr/funsd")
        raw_train = list(ds_funsd["train"])
        raw_test = list(ds_funsd["test"])

        train_items, val_items = split_list_with_seed(raw_train, val_ratio=val_ratio, seed=seed)

        splits_map = {"train": train_items, "val": val_items, "test": raw_test}
        stats["funsd"] = {}

        for split_name, items in splits_map.items():
            jsonl_file = out_dir / f"{split_name}.jsonl"
            logger.info(f"Processing FUNSD {split_name} ({len(items)} samples) -> {jsonl_file.name}...")
            with open(jsonl_file, "w", encoding="utf-8") as f_out:
                for itm in items:
                    record = process_funsd_sample(itm, split_name=split_name, output_images_dir=img_dir)
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["funsd"][split_name] = len(items)

    # 3. CORD Preprocessing
    if "cord" in datasets_to_process:
        logger.info("=== Preprocessing CORD ===")
        out_dir = processed_root / "cord"
        img_dir = out_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        ds_cord = hf_datasets.load_dataset("naver-clova-ix/cord-v2")
        splits_map = {
            "train": list(ds_cord["train"]),
            "val": list(ds_cord["validation"]),
            "test": list(ds_cord["test"]),
        }
        stats["cord"] = {}

        for split_name, items in splits_map.items():
            jsonl_file = out_dir / f"{split_name}.jsonl"
            logger.info(f"Processing CORD {split_name} ({len(items)} samples) -> {jsonl_file.name}...")
            with open(jsonl_file, "w", encoding="utf-8") as f_out:
                for idx, itm in enumerate(items):
                    record = process_cord_sample(itm, split_name=split_name, output_images_dir=img_dir, sample_index=idx)
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["cord"][split_name] = len(items)

    logger.info("=== Preprocessing Pipeline Completed Successfully ===")
    logger.info(f"Processed summary: {stats}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multimodal dataset preprocessing pipeline")
    parser.add_argument("--config", type=str, default="configs/preprocessing.yaml", help="Path to YAML config")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to process (sroie, funsd, cord)")
    args = parser.parse_args()

    run_preprocessing_pipeline(config_path=args.config, target_datasets=args.datasets)
