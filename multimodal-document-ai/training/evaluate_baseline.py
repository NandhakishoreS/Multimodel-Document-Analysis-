"""Evaluation script for Baseline Models (OCR, Text-Only, and Layout-Aware) across SROIE, FUNSD, and CORD."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline.ocr_baseline import OCRBaselineEvaluator
from models.baseline.text_baseline import TextOnlyBaselineModel
from models.baseline.layout_baseline import LayoutAwareBaselineModel
from evaluation.reports import save_baseline_results
from training.train_baseline import load_samples_from_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvaluateBaseline")


def evaluate_baselines_pipeline(
    processed_dir: str = "processed_data",
    models_dir: str = "models/saved",
    reports_dir: str = "reports",
    datasets: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Run complete evaluation on test splits across OCR, Text-Only, and Layout-Aware baselines.
    """
    p_dir = PROJECT_ROOT / processed_dir
    m_dir = PROJECT_ROOT / models_dir
    r_dir = PROJECT_ROOT / reports_dir
    r_dir.mkdir(parents=True, exist_ok=True)

    target_datasets = datasets or ["sroie", "funsd", "cord"]
    all_evaluation_records: List[Dict[str, Any]] = []

    ocr_evaluator = OCRBaselineEvaluator()

    for ds_name in target_datasets:
        logger.info(f"==================================================")
        logger.info(f"EVALUATING TEST BENCHMARKS FOR: {ds_name.upper()}")
        logger.info(f"==================================================")

        test_jsonl = p_dir / ds_name / "test.jsonl"
        if not test_jsonl.exists():
            logger.error(f"Missing test split file: {test_jsonl}")
            continue

        test_samples = load_samples_from_jsonl(test_jsonl)
        logger.info(f"Loaded {len(test_samples)} test samples for {ds_name.upper()}.")

        # ----------------------------------------------------
        # 1. OCR Baseline Evaluation
        # ----------------------------------------------------
        if ds_name == "sroie":
            logger.info("--- Evaluating OCR Baseline (SROIE Entity Transcription) ---")
            ocr_res = ocr_evaluator.evaluate_sroie_entities_in_ocr(test_samples)
            logger.info(f"SROIE OCR Entity Coverage: {ocr_res['overall_entity_coverage']:.4f}")

            # Compute average CER/WER on SROIE entities
            cers = [v["cer"] for v in ocr_res["per_entity_results"].values() if v["total_entities"] > 0]
            wers = [v["wer"] for v in ocr_res["per_entity_results"].values() if v["total_entities"] > 0]
            avg_cer = float(sum(cers) / max(1, len(cers)))
            avg_wer = float(sum(wers) / max(1, len(wers)))

            all_evaluation_records.append({
                "dataset": ds_name.upper(),
                "model_type": "OCR Baseline (Entity Coverage)",
                "split": "test",
                "macro_f1": ocr_res["overall_entity_coverage"],
                "micro_f1": ocr_res["overall_entity_coverage"],
                "weighted_f1": ocr_res["overall_entity_coverage"],
                "precision": ocr_res["overall_entity_coverage"],
                "recall": ocr_res["overall_entity_coverage"],
                "cer": avg_cer,
                "wer": avg_wer,
                "total_tokens_or_samples": len(test_samples),
            })

        # ----------------------------------------------------
        # 2. Text-Only Baseline Evaluation
        # ----------------------------------------------------
        text_model_path = m_dir / f"{ds_name}_text_baseline.joblib"
        if text_model_path.exists():
            logger.info(f"--- Evaluating Text-Only Baseline ({ds_name.upper()}) ---")
            text_model = TextOnlyBaselineModel.load(text_model_path)
            text_metrics = text_model.evaluate(test_samples)

            logger.info(
                f"Text-Only Test Results ({ds_name.upper()}): "
                f"Macro-F1={text_metrics['macro_f1']:.4f}, "
                f"Micro-F1={text_metrics['micro_f1']:.4f}, "
                f"Weighted-F1={text_metrics['weighted_f1']:.4f}, "
                f"Precision={text_metrics['macro_precision']:.4f}, "
                f"Recall={text_metrics['macro_recall']:.4f}"
            )

            all_evaluation_records.append({
                "dataset": ds_name.upper(),
                "model_type": "Text-Only Baseline (TF-IDF + LR)",
                "split": "test",
                "macro_f1": text_metrics["macro_f1"],
                "micro_f1": text_metrics["micro_f1"],
                "weighted_f1": text_metrics["weighted_f1"],
                "precision": text_metrics["macro_precision"],
                "recall": text_metrics["macro_recall"],
                "cer": 0.0,
                "wer": 0.0,
                "total_tokens_or_samples": text_metrics["total_tokens"],
            })
        else:
            logger.warning(f"Model file not found: {text_model_path}")

        # ----------------------------------------------------
        # 3. Layout-Aware Baseline Evaluation
        # ----------------------------------------------------
        layout_model_path = m_dir / f"{ds_name}_layout_baseline.joblib"
        if layout_model_path.exists():
            logger.info(f"--- Evaluating Layout-Aware Baseline ({ds_name.upper()}) ---")
            layout_model = LayoutAwareBaselineModel.load(layout_model_path)
            layout_metrics = layout_model.evaluate(test_samples)

            logger.info(
                f"Layout-Aware Test Results ({ds_name.upper()}): "
                f"Macro-F1={layout_metrics['macro_f1']:.4f}, "
                f"Micro-F1={layout_metrics['micro_f1']:.4f}, "
                f"Weighted-F1={layout_metrics['weighted_f1']:.4f}, "
                f"Precision={layout_metrics['macro_precision']:.4f}, "
                f"Recall={layout_metrics['macro_recall']:.4f}"
            )

            all_evaluation_records.append({
                "dataset": ds_name.upper(),
                "model_type": "Text+Layout Baseline (Spatial + LR)",
                "split": "test",
                "macro_f1": layout_metrics["macro_f1"],
                "micro_f1": layout_metrics["micro_f1"],
                "weighted_f1": layout_metrics["weighted_f1"],
                "precision": layout_metrics["macro_precision"],
                "recall": layout_metrics["macro_recall"],
                "cer": 0.0,
                "wer": 0.0,
                "total_tokens_or_samples": layout_metrics["total_tokens"],
            })
        else:
            logger.warning(f"Model file not found: {layout_model_path}")

    # Save comprehensive reports
    save_baseline_results(all_evaluation_records, output_dir=str(r_dir))

    logger.info("==================================================")
    logger.info("BASELINE EVALUATION PIPELINE COMPLETED")
    logger.info(f"Results summary saved to {r_dir / 'baseline_results.csv'}")
    logger.info("==================================================")

    return all_evaluation_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline models")
    parser.add_argument("--processed_dir", type=str, default="processed_data", help="Processed data directory")
    parser.add_argument("--models_dir", type=str, default="models/saved", help="Trained models directory")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Reports directory")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to evaluate")
    args = parser.parse_args()

    evaluate_baselines_pipeline(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        datasets=args.datasets,
    )
