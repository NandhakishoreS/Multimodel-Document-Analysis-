"""Exploration, statistical analysis, and report generation for the FUNSD dataset."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from PIL import Image
import datasets

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.visualization import visualize_document, plot_label_distribution
from preprocessing.label_mapping import FUNSD_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def explore_funsd(
    hf_repo: str = "nielsr/funsd",
    reports_dir: str = "reports",
    sample_vis_dir: str = "reports/visualizations/funsd",
) -> Dict[str, Any]:
    """
    Perform comprehensive statistical analysis on FUNSD and generate reports.
    """
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    vis_path = Path(sample_vis_dir)
    vis_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading FUNSD dataset for exploration...")
    dataset = datasets.load_dataset(hf_repo)

    stats: Dict[str, Any] = {
        "dataset_name": "FUNSD",
        "splits": {},
        "total_samples": 0,
        "label_statistics": {lbl: 0 for lbl in FUNSD_LABELS},
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
            sample_id = item.get("id", f"{split_name}_{idx}")
            if sample_id in seen_ids:
                stats["data_quality"]["duplicate_sample_ids"] += 1
            seen_ids.add(sample_id)

            img = item["image"]
            w, h = img.size
            all_widths.append(w)
            all_heights.append(h)

            words = item.get("words", [])
            bboxes = item.get("bboxes", [])
            ner_tags = item.get("ner_tags", [])

            all_word_counts.append(len(words))
            all_bbox_counts.append(len(bboxes))

            if len(words) == 0:
                stats["data_quality"]["empty_words_samples"] += 1
            if len(words) != len(bboxes) or len(words) != len(ner_tags):
                stats["data_quality"]["bbox_mismatch_samples"] += 1

            for box in bboxes:
                if len(box) != 4 or box[0] > box[2] or box[1] > box[3]:
                    stats["data_quality"]["malformed_bboxes"] += 1

            # Label mapping from ner_tags integers to label strings
            labels_str = []
            sample_label_counts = {lbl: 0 for lbl in FUNSD_LABELS}
            for tag_id in ner_tags:
                tag_name = FUNSD_LABELS[tag_id] if tag_id < len(FUNSD_LABELS) else "O"
                labels_str.append(tag_name)
                stats["label_statistics"][tag_name] = stats["label_statistics"].get(tag_name, 0) + 1
                sample_label_counts[tag_name] += 1

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
                    labels=labels_str,
                    title=f"FUNSD ({split_name}) - ID: {sample_id}",
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
    json_path = reports_path / "funsd_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved FUNSD report JSON to {json_path}")

    # Save CSV Report
    df = pd.DataFrame(sample_records)
    csv_path = reports_path / "funsd_report.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved FUNSD sample-level CSV report to {csv_path}")

    # Plot Label Distribution
    plot_label_distribution(
        stats["label_statistics"],
        title="FUNSD Label Distribution",
        save_path=reports_path / "funsd_label_distribution.png",
    )

    logger.info(f"FUNSD Exploration Complete: {stats['total_samples']} samples analyzed.")
    return stats


if __name__ == "__main__":
    explore_funsd()
