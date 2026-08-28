"""Training script for Baseline Models (Text-Only and Text+Layout) across SROIE, FUNSD, and CORD."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline.text_baseline import TextOnlyBaselineModel
from models.baseline.layout_baseline import LayoutAwareBaselineModel
from preprocessing.unified_dataset import UnifiedDocumentDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainBaseline")


def load_samples_from_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL dataset records into list of dictionaries."""
    samples = []
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing data split file: {jsonl_path}")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
    return samples


def train_baselines_for_dataset(
    dataset_name: str,
    processed_dir: Path,
    models_dir: Path,
) -> Dict[str, str]:
    """
    Train both Text-Only and Text+Layout baselines for a specific dataset.

    Returns:
        Dict mapping model type to saved artifact file path.
    """
    logger.info(f"==================================================")
    logger.info(f"TRAINING BASELINES FOR DATASET: {dataset_name.upper()}")
    logger.info(f"==================================================")

    train_jsonl = processed_dir / dataset_name / "train.jsonl"
    val_jsonl = processed_dir / dataset_name / "val.jsonl"

    train_samples = load_samples_from_jsonl(train_jsonl)
    val_samples = load_samples_from_jsonl(val_jsonl) if val_jsonl.exists() else []

    logger.info(f"Loaded {len(train_samples)} training samples, {len(val_samples)} validation samples.")

    saved_paths: Dict[str, str] = {}

    # 1. Text-Only Baseline
    logger.info(f"--- Training Text-Only Baseline for {dataset_name.upper()} ---")
    text_model = TextOnlyBaselineModel(dataset_name=dataset_name, max_features=5000, random_state=42)
    text_model.fit(train_samples)

    if val_samples:
        val_metrics = text_model.evaluate(val_samples)
        logger.info(
            f"Text-Only Val Metrics: Macro-F1={val_metrics['macro_f1']:.4f}, "
            f"Micro-F1={val_metrics['micro_f1']:.4f}, Weighted-F1={val_metrics['weighted_f1']:.4f}"
        )

    text_model_path = models_dir / f"{dataset_name}_text_baseline.joblib"
    text_model.save(text_model_path)
    saved_paths["text_only"] = str(text_model_path)

    # 2. Text + Layout Baseline
    logger.info(f"--- Training Layout-Aware Baseline for {dataset_name.upper()} ---")
    layout_model = LayoutAwareBaselineModel(
        dataset_name=dataset_name,
        max_features=5000,
        classifier_type="logistic_regression",
        random_state=42,
    )
    layout_model.fit(train_samples)

    if val_samples:
        val_metrics_layout = layout_model.evaluate(val_samples)
        logger.info(
            f"Layout-Aware Val Metrics: Macro-F1={val_metrics_layout['macro_f1']:.4f}, "
            f"Micro-F1={val_metrics_layout['micro_f1']:.4f}, Weighted-F1={val_metrics_layout['weighted_f1']:.4f}"
        )

    layout_model_path = models_dir / f"{dataset_name}_layout_baseline.joblib"
    layout_model.save(layout_model_path)
    saved_paths["layout_aware"] = str(layout_model_path)

    return saved_paths


def run_training_pipeline(
    processed_dir: str = "processed_data",
    models_dir: str = "models/saved",
    datasets: Optional[List[str]] = None,
) -> Dict[str, Dict[str, str]]:
    """Run baseline training across all datasets."""
    p_dir = PROJECT_ROOT / processed_dir
    m_dir = PROJECT_ROOT / models_dir
    m_dir.mkdir(parents=True, exist_ok=True)

    target_datasets = datasets or ["sroie", "funsd", "cord"]
    all_saved: Dict[str, Dict[str, str]] = {}

    for ds in target_datasets:
        saved = train_baselines_for_dataset(ds, processed_dir=p_dir, models_dir=m_dir)
        all_saved[ds] = saved

    logger.info("==================================================")
    logger.info("BASELINE TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("==================================================")
    return all_saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--processed_dir", type=str, default="processed_data", help="Directory with processed JSONL files")
    parser.add_argument("--models_dir", type=str, default="models/saved", help="Directory to save trained models")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to train on")
    args = parser.parse_args()

    run_training_pipeline(processed_dir=args.processed_dir, models_dir=args.models_dir, datasets=args.datasets)
