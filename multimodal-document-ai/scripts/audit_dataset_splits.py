"""
Phase 5 — Step 1: Audit Dataset Splits & Leakage Detection
Audits SROIE, FUNSD, and CORD for sample counts, ID overlap, image duplicates, and text overlap.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("split_audit")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = _PROJECT_ROOT / "processed_data"
REPORTS_DIR = _PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_CSV = REPORTS_DIR / "phase5_split_audit.csv"

DATASETS = ["sroie", "funsd", "cord"]
SPLITS = ["train", "val", "test"]

def hash_file(path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def normalize_text(words: List[str]) -> str:
    return " ".join(words).strip().lower()

def audit_dataset(dataset_name: str) -> Dict[str, any]:
    logger.info("Auditing dataset: %s", dataset_name.upper())
    ds_dir = PROCESSED_DIR / dataset_name
    
    samples_by_split: Dict[str, List[dict]] = {}
    ids_by_split: Dict[str, Set[str]] = {}
    img_hashes_by_split: Dict[str, Dict[str, str]] = {}  # hash -> sample_id
    text_hashes_by_split: Dict[str, Dict[str, str]] = {} # text_hash -> sample_id

    for split in SPLITS:
        jsonl_path = ds_dir / f"{split}.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Missing {jsonl_path}")
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]
        samples_by_split[split] = samples
        ids_by_split[split] = set(s["id"] for s in samples)
        
        # Check internal duplicate IDs
        if len(ids_by_split[split]) != len(samples):
            logger.warning("Dataset %s split %s has duplicate IDs internally! Count=%d, Unique=%d",
                           dataset_name, split, len(samples), len(ids_by_split[split]))
        
        # Image hashes
        img_hashes = {}
        text_hashes = {}
        for s in samples:
            img_p = Path(s["image_path"])
            if not img_p.is_absolute():
                img_p = _PROJECT_ROOT / img_p
            if img_p.exists():
                h = hash_file(img_p)
                img_hashes[s["id"]] = h
            else:
                logger.warning("Image missing: %s", img_p)
            
            # Normalized text hash
            norm_t = normalize_text(s.get("words", []))
            th = hashlib.md5(norm_t.encode("utf-8")).hexdigest()
            text_hashes[s["id"]] = th
            
        img_hashes_by_split[split] = img_hashes
        text_hashes_by_split[split] = text_hashes

    # Check ID Overlaps
    train_ids = ids_by_split["train"]
    val_ids = ids_by_split["val"]
    test_ids = ids_by_split["test"]

    train_test_id_overlap = train_ids.intersection(test_ids)
    val_test_id_overlap = val_ids.intersection(test_ids)
    train_val_id_overlap = train_ids.intersection(val_ids)

    # Check Image Hash Overlaps across splits
    def find_hash_overlaps(split1: str, split2: str) -> List[Tuple[str, str, str]]:
        overlaps = []
        map1 = img_hashes_by_split[split1]
        map2 = img_hashes_by_split[split2]
        inv_map2 = {}
        for sid, h in map2.items():
            inv_map2.setdefault(h, []).append(sid)
        for sid1, h1 in map1.items():
            if h1 in inv_map2:
                for sid2 in inv_map2[h1]:
                    overlaps.append((sid1, sid2, h1))
        return overlaps

    train_test_img_overlap = find_hash_overlaps("train", "test")
    val_test_img_overlap = find_hash_overlaps("val", "test")
    train_val_img_overlap = find_hash_overlaps("train", "val")

    # Check Text Hash Overlaps across splits
    def find_text_overlaps(split1: str, split2: str) -> List[Tuple[str, str, str]]:
        overlaps = []
        map1 = text_hashes_by_split[split1]
        map2 = text_hashes_by_split[split2]
        inv_map2 = {}
        for sid, th in map2.items():
            inv_map2.setdefault(th, []).append(sid)
        for sid1, th1 in map1.items():
            if th1 in inv_map2:
                for sid2 in inv_map2[th1]:
                    overlaps.append((sid1, sid2, th1))
        return overlaps

    train_test_text_overlap = find_text_overlaps("train", "test")
    val_test_text_overlap = find_text_overlaps("val", "test")
    train_val_text_overlap = find_text_overlaps("train", "val")

    total_leakage = (
        len(train_test_id_overlap) + len(val_test_id_overlap) +
        len(train_test_img_overlap) + len(val_test_img_overlap)
    )

    result = {
        "dataset": dataset_name.upper(),
        "train_samples": len(samples_by_split["train"]),
        "val_samples": len(samples_by_split["val"]),
        "test_samples": len(samples_by_split["test"]),
        "train_test_id_overlap": len(train_test_id_overlap),
        "val_test_id_overlap": len(val_test_id_overlap),
        "train_val_id_overlap": len(train_val_id_overlap),
        "train_test_img_overlap": len(train_test_img_overlap),
        "val_test_img_overlap": len(val_test_img_overlap),
        "train_val_img_overlap": len(train_val_img_overlap),
        "train_test_text_overlap": len(train_test_text_overlap),
        "val_test_text_overlap": len(val_test_text_overlap),
        "train_val_text_overlap": len(train_val_text_overlap),
        "total_test_leakage_count": total_leakage,
        "details": {
            "train_test_id_overlap_items": list(train_test_id_overlap),
            "val_test_id_overlap_items": list(val_test_id_overlap),
            "train_test_img_overlap_items": train_test_img_overlap,
            "val_test_img_overlap_items": val_test_img_overlap,
            "train_test_text_overlap_items": train_test_text_overlap,
        }
    }
    return result

def main():
    print("\n" + "=" * 75)
    print("PHASE 5 — STEP 1: AUDIT DATASET SPLITS & LEAKAGE CHECK")
    print("=" * 75)
    
    results = []
    has_leakage = False

    for ds in DATASETS:
        res = audit_dataset(ds)
        results.append(res)
        if res["total_test_leakage_count"] > 0:
            has_leakage = True
            logger.error("CRITICAL LEAKAGE DETECTED in %s: %s", ds, res["details"])

    # Format Table for console
    print("\nDataset | Train | Validation | Test | Duplicate/Overlap Count")
    print("-" * 65)
    for r in results:
        print(f"{r['dataset']:<7} | {r['train_samples']:<5} | {r['val_samples']:<10} | {r['test_samples']:<4} | {r['total_test_leakage_count']:<23}")
    print("-" * 65)

    # Save to CSV
    fieldnames = [
        "Dataset", "Train", "Validation", "Test",
        "Train_Test_ID_Overlap", "Val_Test_ID_Overlap", "Train_Val_ID_Overlap",
        "Train_Test_Image_Overlap", "Val_Test_Image_Overlap", "Train_Val_Image_Overlap",
        "Train_Test_Text_Overlap", "Val_Test_Text_Overlap",
        "Total_Test_Leakage_Count"
    ]
    with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for r in results:
            writer.writerow([
                r["dataset"], r["train_samples"], r["val_samples"], r["test_samples"],
                r["train_test_id_overlap"], r["val_test_id_overlap"], r["train_val_id_overlap"],
                r["train_test_img_overlap"], r["val_test_img_overlap"], r["train_val_img_overlap"],
                r["train_test_text_overlap"], r["val_test_text_overlap"],
                r["total_test_leakage_count"]
            ])
    logger.info("Split audit saved to %s", AUDIT_CSV)

    if has_leakage:
        print("\n[CRITICAL WARNING] LEAKAGE DETECTED! STOPPING FURTHER EXECUTION.")
        return False
    else:
        print("\n[VERIFIED PASS] Zero train/test or val/test ID or image overlap found across all datasets.")
        return True

if __name__ == "__main__":
    main()
