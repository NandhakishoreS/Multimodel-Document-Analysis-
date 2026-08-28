"""Phase 1 Validation Script: Verifies datasets loading, image accessibility, and annotation integrity."""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import datasets
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.bbox_utils import validate_bbox, quad_to_bbox
from preprocessing.cord_processor import parse_cord_ground_truth

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase1Validator")


def validate_sroie() -> Tuple[bool, Dict[str, int]]:
    """Validate SROIE dataset integrity."""
    logger.info("--- Validating SROIE Dataset ---")
    metrics = {"samples_checked": 0, "valid_samples": 0, "errors": 0}
    try:
        ds = datasets.load_dataset("jsdnrs/ICDAR2019-SROIE")
    except Exception as e:
        logger.error(f"Failed to load SROIE dataset: {e}")
        return False, metrics

    for split in ds:
        for idx, item in enumerate(ds[split]):
            metrics["samples_checked"] += 1
            img = item.get("image")
            if not isinstance(img, Image.Image):
                logger.error(f"SROIE {split}[{idx}]: image is not a valid PIL Image")
                metrics["errors"] += 1
                continue

            w, h = img.size
            if w <= 0 or h <= 0:
                logger.error(f"SROIE {split}[{idx}]: invalid dimensions ({w}, {h})")
                metrics["errors"] += 1
                continue

            words = item.get("words", [])
            bboxes = item.get("bboxes", [])
            if len(words) != len(bboxes):
                logger.error(f"SROIE {split}[{idx}]: words count ({len(words)}) != bboxes count ({len(bboxes)})")
                metrics["errors"] += 1
                continue

            for box in bboxes:
                if len(box) != 4 or box[0] > box[2] or box[1] > box[3]:
                    logger.error(f"SROIE {split}[{idx}]: invalid bbox {box}")
                    metrics["errors"] += 1
                    break

            metrics["valid_samples"] += 1

    success = metrics["errors"] == 0 and metrics["samples_checked"] > 0
    logger.info(f"SROIE Validation: {'PASSED' if success else 'FAILED'} ({metrics['valid_samples']}/{metrics['samples_checked']} valid)")
    return success, metrics


def validate_funsd() -> Tuple[bool, Dict[str, int]]:
    """Validate FUNSD dataset integrity."""
    logger.info("--- Validating FUNSD Dataset ---")
    metrics = {"samples_checked": 0, "valid_samples": 0, "errors": 0}
    try:
        ds = datasets.load_dataset("nielsr/funsd")
    except Exception as e:
        logger.error(f"Failed to load FUNSD dataset: {e}")
        return False, metrics

    for split in ds:
        for idx, item in enumerate(ds[split]):
            metrics["samples_checked"] += 1
            img = item.get("image")
            if not isinstance(img, Image.Image):
                logger.error(f"FUNSD {split}[{idx}]: image is not a valid PIL Image")
                metrics["errors"] += 1
                continue

            w, h = img.size
            words = item.get("words", [])
            bboxes = item.get("bboxes", [])
            ner_tags = item.get("ner_tags", [])

            if not (len(words) == len(bboxes) == len(ner_tags)):
                logger.error(f"FUNSD {split}[{idx}]: length mismatch words={len(words)}, bboxes={len(bboxes)}, tags={len(ner_tags)}")
                metrics["errors"] += 1
                continue

            for box in bboxes:
                if len(box) != 4 or box[0] > box[2] or box[1] > box[3]:
                    logger.error(f"FUNSD {split}[{idx}]: invalid bbox {box}")
                    metrics["errors"] += 1
                    break

            metrics["valid_samples"] += 1

    success = metrics["errors"] == 0 and metrics["samples_checked"] > 0
    logger.info(f"FUNSD Validation: {'PASSED' if success else 'FAILED'} ({metrics['valid_samples']}/{metrics['samples_checked']} valid)")
    return success, metrics


def validate_cord() -> Tuple[bool, Dict[str, int]]:
    """Validate CORD dataset integrity."""
    logger.info("--- Validating CORD Dataset ---")
    metrics = {"samples_checked": 0, "valid_samples": 0, "errors": 0}
    try:
        ds = datasets.load_dataset("naver-clova-ix/cord-v2")
    except Exception as e:
        logger.error(f"Failed to load CORD dataset: {e}")
        return False, metrics

    for split in ds:
        for idx, item in enumerate(ds[split]):
            metrics["samples_checked"] += 1
            img = item.get("image")
            if not isinstance(img, Image.Image):
                logger.error(f"CORD {split}[{idx}]: image is not a valid PIL Image")
                metrics["errors"] += 1
                continue

            w, h = img.size
            gt = item.get("ground_truth")
            words, bboxes, labels = parse_cord_ground_truth(gt, img_width=w, img_height=h)

            if len(words) != len(bboxes) or len(words) != len(labels):
                logger.error(f"CORD {split}[{idx}]: parsed length mismatch words={len(words)}, bboxes={len(bboxes)}, labels={len(labels)}")
                metrics["errors"] += 1
                continue

            metrics["valid_samples"] += 1

    success = metrics["errors"] == 0 and metrics["samples_checked"] > 0
    logger.info(f"CORD Validation: {'PASSED' if success else 'FAILED'} ({metrics['valid_samples']}/{metrics['samples_checked']} valid)")
    return success, metrics


def run_phase1_validation() -> bool:
    """Run validation across all 3 datasets."""
    logger.info("==========================================")
    logger.info("STARTING PHASE 1 DATASET INTEGRITY CHECKS")
    logger.info("==========================================")

    sroie_ok, _ = validate_sroie()
    funsd_ok, _ = validate_funsd()
    cord_ok, _ = validate_cord()

    all_passed = sroie_ok and funsd_ok and cord_ok

    logger.info("==========================================")
    if all_passed:
        logger.info("✅ ALL PHASE 1 VALIDATION CHECKS PASSED SUCCESSFULLY!")
    else:
        logger.error("❌ PHASE 1 VALIDATION CHECKS FAILED! Review error logs above.")
    logger.info("==========================================")

    return all_passed


if __name__ == "__main__":
    passed = run_phase1_validation()
    sys.exit(0 if passed else 1)
