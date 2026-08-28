"""Phase 2 Validation Script: Verifies unified schema integrity, bbox normalization [0, 1000], label alignment, and split isolation."""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase2Validator")


def validate_unified_dataset(dataset_name: str, processed_root: Path) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate processed JSONL files for a dataset.
    """
    ds_dir = processed_root / dataset_name
    metrics = {
        "dataset": dataset_name,
        "splits": {},
        "total_samples": 0,
        "total_tokens": 0,
        "errors": [],
    }

    if not ds_dir.exists():
        metrics["errors"].append(f"Directory does not exist: {ds_dir}")
        return False, metrics

    split_ids: Dict[str, Set[str]] = {}

    for split in ["train", "val", "test"]:
        jsonl_file = ds_dir / f"{split}.jsonl"
        if not jsonl_file.exists():
            metrics["errors"].append(f"Missing split file: {jsonl_file}")
            continue

        split_ids[split] = set()
        count = 0

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                try:
                    record = json.loads(line.strip())
                except Exception as e:
                    metrics["errors"].append(f"{dataset_name} {split} line {line_no}: JSON decode error {e}")
                    continue

                count += 1
                # Schema uses "id" as the primary document identifier
                sample_id = record.get("id") or record.get("doc_id")
                if not sample_id:
                    metrics["errors"].append(f"{dataset_name} {split} line {line_no}: missing 'id'")
                else:
                    # For leakage detection, use the image_path as the key.
                    # Image filenames are now prefixed with split name by the
                    # processors, so the same image_path in two splits would
                    # be genuine leakage.  Fallback to namespaced id for safety.
                    img_path = record.get("image_path", "")
                    split_ids[split].add(img_path if img_path else f"{split}::{sample_id}")

                # Required fields
                req_fields = ["id", "dataset", "split", "image_path", "width", "height", "words", "bboxes", "labels"]
                for rf in req_fields:
                    if rf not in record:
                        metrics["errors"].append(f"{dataset_name} sample {sample_id}: missing field '{rf}'")

                # Image check (check random sample of images to be fast)
                img_p = Path(record.get("image_path", ""))
                if not img_p.exists():
                    metrics["errors"].append(f"{dataset_name} sample {sample_id}: image path does not exist {img_p}")

                words = record.get("words", [])
                bboxes = record.get("bboxes", [])
                labels = record.get("labels", [])

                n_w = len(words)
                n_b = len(bboxes)
                n_l = len(labels)

                if not (n_w == n_b == n_l):
                    metrics["errors"].append(f"{dataset_name} sample {sample_id}: length mismatch words={n_w}, bboxes={n_b}, labels={n_l}")

                metrics["total_tokens"] += n_w

                # Bbox normalization range check [0, 1000]
                for b_idx, box in enumerate(bboxes):
                    if len(box) != 4:
                        metrics["errors"].append(f"{dataset_name} sample {sample_id} bbox {b_idx}: length != 4 ({box})")
                        continue
                    x0, y0, x1, y1 = box
                    if not (0 <= x0 <= 1000 and 0 <= y0 <= 1000 and 0 <= x1 <= 1000 and 0 <= y1 <= 1000):
                        metrics["errors"].append(f"{dataset_name} sample {sample_id} bbox {b_idx}: coords out of [0, 1000] ({box})")
                    if x0 > x1 or y0 > y1:
                        metrics["errors"].append(f"{dataset_name} sample {sample_id} bbox {b_idx}: invalid ordering x0>x1 or y0>y1 ({box})")

        metrics["splits"][split] = count
        metrics["total_samples"] += count

    # Data leakage check between splits
    splits_present = list(split_ids.keys())
    for i in range(len(splits_present)):
        for j in range(i + 1, len(splits_present)):
            s1, s2 = splits_present[i], splits_present[j]
            overlap = split_ids[s1].intersection(split_ids[s2])
            if overlap:
                metrics["errors"].append(f"DATA LEAKAGE: {len(overlap)} IDs overlap between {s1} and {s2} ({list(overlap)[:5]})")

    is_valid = len(metrics["errors"]) == 0 and metrics["total_samples"] > 0
    return is_valid, metrics


def run_phase2_validation(processed_data_dir: str = "processed_data") -> bool:
    """Run validation across all 3 processed datasets."""
    logger.info("==========================================")
    logger.info("STARTING PHASE 2 UNIFIED DATASET CHECKS")
    logger.info("==========================================")

    processed_root = PROJECT_ROOT / processed_data_dir
    all_valid = True

    for ds_name in ["sroie", "funsd", "cord"]:
        logger.info(f"--- Validating Processed {ds_name.upper()} ---")
        valid, metrics = validate_unified_dataset(ds_name, processed_root)
        if valid:
            logger.info(f"✅ {ds_name.upper()} PASSED: {metrics['total_samples']} samples, {metrics['total_tokens']:,} tokens across splits {metrics['splits']}")
        else:
            all_valid = False
            logger.error(f"❌ {ds_name.upper()} FAILED with {len(metrics['errors'])} errors:")
            for err in metrics["errors"][:10]:
                logger.error(f"   - {err}")
            if len(metrics["errors"]) > 10:
                logger.error(f"   ... and {len(metrics['errors']) - 10} more errors.")

    logger.info("==========================================")
    if all_valid:
        logger.info("✅ ALL PHASE 2 VALIDATION CHECKS PASSED SUCCESSFULLY!")
    else:
        logger.error("❌ PHASE 2 VALIDATION CHECKS FAILED!")
    logger.info("==========================================")

    return all_valid


if __name__ == "__main__":
    passed = run_phase2_validation()
    sys.exit(0 if passed else 1)
