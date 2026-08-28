"""Exploration, statistical analysis, and report generation for the CORD dataset."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import datasets

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.visualization import visualize_document, plot_label_distribution
from preprocessing.bbox_utils import quad_to_bbox, validate_bbox
from preprocessing.label_mapping import CORD_LABELS
from preprocessing.cord_processor import parse_cord_ground_truth, map_cord_category_to_coarse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)





def explore_cord(
    hf_repo: str = "naver-clova-ix/cord-v2",
    reports_dir: str = "reports",
    sample_vis_dir: str = "reports/visualizations/cord",
) -> Dict[str, Any]:
    """
    Perform comprehensive statistical analysis on CORD and generate reports.
    """
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    vis_path = Path(sample_vis_dir)
    vis_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading CORD dataset for exploration...")
    dataset = datasets.load_dataset(hf_repo)

    stats: Dict[str, Any] = {
        "dataset_name": "CORD-v2",
        "splits": {},
        "total_samples": 0,
        "label_statistics": {lbl: 0 for lbl in CORD_LABELS},
        "image_resolution_statistics": {},
        "word_statistics": {},
        "bbox_statistics": {},
        "data_quality": {
            "empty_words_samples": 0,
            "bbox_mismatch_samples": 0,
            "malformed_bboxes": 0,
            "duplicate_sample_ids": 0,
        },
    }

    all_widths: List[int] = []
    all_heights: List[int] = []
    all_word_counts: List[int] = []
    all_bbox_counts: List[int] = []
    seen_ids: set = set()
    sample_records: List[Dict[str, Any]] = []

    for split_name, split_data in dataset.items():
        split_samples = len(split_data)
        stats["splits"][split_name] = split_samples
        stats["total_samples"] += split_samples

        for idx, item in enumerate(split_data):
            sample_id = f"cord_{split_name}_{idx}"
            if sample_id in seen_ids:
                stats["data_quality"]["duplicate_sample_ids"] += 1
            seen_ids.add(sample_id)

            img = item["image"]
            w, h = img.size
            all_widths.append(w)
            all_heights.append(h)

            gt_raw = item.get("ground_truth", {})
            words, bboxes, labels = parse_cord_ground_truth(gt_raw, img_width=w, img_height=h)

            all_word_counts.append(len(words))
            all_bbox_counts.append(len(bboxes))

            if len(words) == 0:
                stats["data_quality"]["empty_words_samples"] += 1
            if len(words) != len(bboxes) or len(words) != len(labels):
                stats["data_quality"]["bbox_mismatch_samples"] += 1

            for box in bboxes:
                if len(box) != 4 or box[0] > box[2] or box[1] > box[3]:
                    stats["data_quality"]["malformed_bboxes"] += 1

            sample_label_counts = {lbl: 0 for lbl in CORD_LABELS}
            for lbl in labels:
                if lbl in stats["label_statistics"]:
                    stats["label_statistics"][lbl] += 1
                    sample_label_counts[lbl] += 1
                else:
                    stats["label_statistics"]["O"] += 1
                    sample_label_counts["O"] += 1

            rec = {
                "sample_id": sample_id,
                "split": split_name,
                "width": w,
                "height": h,
                "num_words": len(words),
                "num_bboxes": len(bboxes),
            }
            rec.update(sample_label_counts)
            sample_records.append(rec)

            # Save first 3 visualization samples per split
            if idx < 3:
                vis_file = vis_path / f"{split_name}_{sample_id}.png"
                visualize_document(
                    image=img,
                    words=words,
                    bboxes=bboxes,
                    labels=labels,
                    title=f"CORD ({split_name}) - ID: {sample_id}",
                    save_path=vis_file,
                )

    # Resolution stats
    stats["image_resolution_statistics"] = {
        "width_min": int(np.min(all_widths)),
        "width_max": int(np.max(all_widths)),
        "width_mean": float(np.mean(all_widths)),
        "width_median": float(np.median(all_widths)),
        "height_min": int(np.min(all_heights)),
        "height_max": int(np.max(all_heights)),
        "height_mean": float(np.mean(all_heights)),
        "height_median": float(np.median(all_heights)),
    }

    # Word & Bbox stats
    stats["word_statistics"] = {
        "words_min": int(np.min(all_word_counts)),
        "words_max": int(np.max(all_word_counts)),
        "words_mean": float(np.mean(all_word_counts)),
        "words_median": float(np.median(all_word_counts)),
        "total_words": int(np.sum(all_word_counts)),
    }
    stats["bbox_statistics"] = {
        "bboxes_min": int(np.min(all_bbox_counts)),
        "bboxes_max": int(np.max(all_bbox_counts)),
        "bboxes_mean": float(np.mean(all_bbox_counts)),
        "bboxes_median": float(np.median(all_bbox_counts)),
        "total_bboxes": int(np.sum(all_bbox_counts)),
    }

    # Save JSON Report
    json_path = reports_path / "cord_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved CORD report JSON to {json_path}")

    # Save CSV Report
    df = pd.DataFrame(sample_records)
    csv_path = reports_path / "cord_report.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CORD sample-level CSV report to {csv_path}")

    # Plot Label Distribution
    plot_label_distribution(
        stats["label_statistics"],
        title="CORD Label Distribution",
        save_path=reports_path / "cord_label_distribution.png",
    )

    logger.info(f"CORD Exploration Complete: {stats['total_samples']} samples analyzed.")
    return stats


if __name__ == "__main__":
    explore_cord()
