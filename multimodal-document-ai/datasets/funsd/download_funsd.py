"""Download and cache the official FUNSD (Form Understanding in Noisy Scanned Documents) dataset."""

import argparse
import logging
from pathlib import Path
from typing import Optional
import datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_FUNSD_HF_REPO = "nielsr/funsd"


def download_funsd(
    target_dir: Optional[Path] = None,
    hf_repo: str = DEFAULT_FUNSD_HF_REPO,
    save_images_locally: bool = False,
) -> datasets.DatasetDict:
    """
    Download and cache the FUNSD dataset from Hugging Face or local fallback.

    Args:
        target_dir: Optional directory to store extracted raw files.
        hf_repo: HuggingFace repository identifier.
        save_images_locally: Whether to save individual image files to disk.

    Returns:
        HuggingFace DatasetDict containing 'train' and 'test' splits.
    """
    logger.info(f"Loading FUNSD dataset from {hf_repo}...")
    try:
        dataset = datasets.load_dataset(hf_repo)
        logger.info(f"Successfully loaded FUNSD splits: {list(dataset.keys())}")
        for split in dataset:
            logger.info(f"Split '{split}': {len(dataset[split])} samples")
    except Exception as e:
        logger.error(f"Failed to load FUNSD from Hugging Face: {e}")
        logger.info(
            "Manual download instructions:\n"
            "1. Download FUNSD dataset from https://guillaumejaume.github.io/FUNSD/\n"
            "2. Extract dataset zip to raw_data/funsd/\n"
        )
        raise

    if target_dir and save_images_locally:
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        for split_name, split_data in dataset.items():
            split_dir = target_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving {split_name} samples to {split_dir}...")
            for idx, item in enumerate(split_data):
                sample_id = item.get("id", f"sample_{idx}")
                img = item["image"]
                img_path = split_dir / f"{sample_id}.png"
                if not img_path.exists():
                    img.save(img_path)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FUNSD dataset")
    parser.add_argument("--target_dir", type=str, default="raw_data/funsd", help="Local directory for raw data")
    parser.add_argument("--save_images", action="store_true", help="Export images to local directory")
    args = parser.parse_args()

    download_funsd(target_dir=Path(args.target_dir), save_images_locally=args.save_images)
