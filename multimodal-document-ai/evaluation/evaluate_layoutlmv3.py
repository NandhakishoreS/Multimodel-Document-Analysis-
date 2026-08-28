"""
LayoutLMv3 test-set evaluation script.

Loads the best checkpoint for a dataset, runs inference on the test split,
computes seqeval + token-level metrics, generates error analysis samples,
and writes all reports to reports/.

Usage
-----
# Evaluate SROIE (full stage checkpoint):
python evaluation/evaluate_layoutlmv3.py --dataset sroie --stage full

# Evaluate all datasets:
python evaluation/evaluate_layoutlmv3.py --dataset sroie funsd cord --stage full
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.layoutlmv3.config import load_dataset_config, load_training_config
from models.layoutlmv3.model import LayoutLMv3Classifier
from models.layoutlmv3.tokenizer import LayoutLMv3TokenizerWrapper
from models.layoutlmv3.dataset import LayoutLMv3DocumentDataset
from evaluation.layoutlmv3_metrics import (
    compute_seqeval_metrics,
    compute_token_metrics,
    compute_sroie_entity_accuracy,
    collect_predictions,
)
from inference.layoutlmv3_predict import predict_batch_documents
from training.train_layoutlmv3 import (
    _reconstruct_word_ids_from_labels,
    _extract_true_word_labels,
)
from inference.layoutlmv3_predict import token_preds_to_word_preds
from utils.visualization import visualize_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("evaluate_layoutlmv3")


# ---------------------------------------------------------------------------
# Find best checkpoint for a dataset + stage
# ---------------------------------------------------------------------------

def find_best_checkpoint(
    dataset_name: str,
    stage: str,
    checkpoints_root: Path,
) -> Optional[Path]:
    """
    Find the checkpoint directory for a given dataset and stage.

    Looks for models/checkpoints/layoutlmv3/{dataset}/{stage}/
    and reads checkpoint_meta.json to verify it exists.
    """
    ckpt_dir = checkpoints_root / dataset_name / stage
    if not ckpt_dir.exists():
        logger.error("Checkpoint directory not found: %s", ckpt_dir)
        return None

    meta_path = ckpt_dir / "checkpoint_meta.json"
    if not meta_path.exists():
        logger.error("checkpoint_meta.json not found in %s", ckpt_dir)
        return None

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    logger.info(
        "Found checkpoint: %s | epoch=%d | val_macro_f1=%.4f",
        ckpt_dir, meta.get("epoch", -1), meta.get("val_macro_f1", 0.0),
    )
    return ckpt_dir


# ---------------------------------------------------------------------------
# Full evaluation on test split
# ---------------------------------------------------------------------------

def evaluate_dataset(
    dataset_name: str,
    stage: str,
    config_path: str = "configs/layoutlmv3.yaml",
    reports_dir: Optional[str] = None,
    num_error_samples: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate the best checkpoint on the test split of a dataset.

    Returns a results dict suitable for appending to reports/baseline_results.csv.
    """
    training_cfg = load_training_config(config_path, dataset_name=dataset_name)
    dataset_cfg  = load_dataset_config(dataset_name, labels_dir=training_cfg.labels_dir)

    ckpt_root = training_cfg.resolved_checkpoint_dir()
    ckpt_dir  = find_best_checkpoint(dataset_name, stage, ckpt_root)
    if ckpt_dir is None:
        logger.error("No checkpoint found for %s/%s — skipping evaluation.", dataset_name, stage)
        return {}

    out_reports = Path(reports_dir) if reports_dir else training_cfg.resolved_reports_dir()
    out_reports.mkdir(parents=True, exist_ok=True)
    error_dir = out_reports / "error_analysis" / dataset_name
    error_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model and processor ──────────────────────────────────────────────
    processor_dir = ckpt_dir / "processor"
    if not processor_dir.exists():
        processor_dir_str = training_cfg.model_checkpoint  # Fall back to original
    else:
        processor_dir_str = str(processor_dir)

    tokenizer_wrapper = LayoutLMv3TokenizerWrapper.from_pretrained(
        checkpoint=processor_dir_str,
        max_seq_length=training_cfg.max_seq_length,
    )
    classifier = LayoutLMv3Classifier.load_checkpoint(
        checkpoint_dir=ckpt_dir,
        dataset_config=dataset_cfg,
    )
    classifier.eval_mode()

    # ── Load test split ───────────────────────────────────────────────────────
    test_jsonl = training_cfg.resolved_processed_dir() / dataset_name / "test.jsonl"
    test_dataset = LayoutLMv3DocumentDataset(
        jsonl_path=test_jsonl,
        dataset_config=dataset_cfg,
        tokenizer_wrapper=tokenizer_wrapper,
        split="test",
        project_root=_PROJECT_ROOT,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=training_cfg.batch_size, shuffle=False, num_workers=0
    )

    # ── Run inference ─────────────────────────────────────────────────────────
    logger.info("Running inference on %d test samples…", len(test_dataset))
    all_true_word_seqs: List[List[str]] = []
    all_pred_word_seqs: List[List[str]] = []

    with torch.no_grad():
        for batch in test_loader:
            logits = classifier.forward(
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                bbox           = batch["bbox"],
                pixel_values   = batch["pixel_values"],
            )["logits"]

            pred_ids_batch   = torch.argmax(logits, dim=-1)
            probs_batch      = torch.softmax(logits, dim=-1).max(dim=-1).values
            labels_batch     = batch["labels"]
            attn_batch       = batch["attention_mask"]
            batch_size       = batch["input_ids"].shape[0]

            for i in range(batch_size):
                word_ids = _reconstruct_word_ids_from_labels(
                    labels_batch[i].cpu().tolist(),
                    attn_batch[i].cpu().tolist(),
                )
                true_words = _extract_true_word_labels(
                    labels_batch[i].cpu().tolist(), word_ids, dataset_cfg.id2label
                )
                pred_words, _ = token_preds_to_word_preds(
                    token_label_ids = pred_ids_batch[i].cpu().tolist(),
                    token_probs     = probs_batch[i].cpu().tolist(),
                    word_ids        = word_ids,
                    id2label        = dataset_cfg.id2label,
                    num_words       = len(true_words),
                )
                if true_words:
                    all_true_word_seqs.append(true_words)
                    all_pred_word_seqs.append(pred_words)

    # ── Compute metrics ───────────────────────────────────────────────────────
    seqeval_metrics = compute_seqeval_metrics(all_true_word_seqs, all_pred_word_seqs)
    flat_true = [lbl for seq in all_true_word_seqs for lbl in seq]
    flat_pred = [lbl for seq in all_pred_word_seqs for lbl in seq]
    token_metrics = compute_token_metrics(flat_true, flat_pred, labels=dataset_cfg.labels)

    logger.info(
        "TEST RESULTS [%s] — seqeval micro_f1=%.4f | macro_f1=%.4f",
        dataset_name.upper(),
        seqeval_metrics["micro_f1"], seqeval_metrics["macro_f1"],
    )
    logger.info("\n%s", seqeval_metrics.get("classification_report", ""))

    # ── SROIE entity accuracy ─────────────────────────────────────────────────
    entity_accuracy: Dict[str, Any] = {}
    if dataset_name == "sroie":
        raw_test_samples = test_dataset.samples
        pred_results = predict_batch_documents(
            samples=raw_test_samples[:50],  # Cap for speed
            classifier=classifier,
            tokenizer_wrapper=tokenizer_wrapper,
            project_root=_PROJECT_ROOT,
        )
        predicted_entities = [r["entities"] for r in pred_results]
        entity_accuracy = compute_sroie_entity_accuracy(
            samples=raw_test_samples[:50],
            predicted_entities=predicted_entities,
        )
        logger.info("SROIE entity accuracy: %s", entity_accuracy)

    # ── Per-class report ──────────────────────────────────────────────────────
    per_entity_path = out_reports / f"layoutlmv3_{dataset_name}_per_entity.json"
    with open(per_entity_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "stage": stage,
                "seqeval": seqeval_metrics,
                "token_level": token_metrics,
                "sroie_entity_accuracy": entity_accuracy,
            },
            f, indent=2,
        )
    logger.info("Per-entity report saved to %s", per_entity_path)

    # ── Classification report ─────────────────────────────────────────────────
    cls_report_path = out_reports / f"layoutlmv3_{dataset_name}_classification_report.txt"
    with open(cls_report_path, "w", encoding="utf-8") as f:
        f.write(f"LayoutLMv3 — {dataset_name.upper()} — Test Set\n")
        f.write("=" * 60 + "\n\n")
        f.write(seqeval_metrics.get("classification_report", "N/A"))
    logger.info("Classification report saved to %s", cls_report_path)

    # ── Error analysis ────────────────────────────────────────────────────────
    _run_error_analysis(
        dataset_name=dataset_name,
        test_dataset=test_dataset,
        all_true_word_seqs=all_true_word_seqs,
        all_pred_word_seqs=all_pred_word_seqs,
        error_dir=error_dir,
        num_samples=num_error_samples,
    )

    return {
        "dataset":       dataset_name.upper(),
        "model_type":    f"LayoutLMv3 ({stage})",
        "split":         "test",
        "macro_f1":      seqeval_metrics["macro_f1"],
        "micro_f1":      seqeval_metrics["micro_f1"],
        "weighted_f1":   token_metrics.get("weighted_f1", 0.0),
        "precision":     seqeval_metrics["precision"],
        "recall":        seqeval_metrics["recall"],
        "cer":           0.0,
        "wer":           0.0,
        "total_tokens_or_samples": seqeval_metrics["num_tokens_true"],
    }


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def _run_error_analysis(
    dataset_name: str,
    test_dataset: LayoutLMv3DocumentDataset,
    all_true_word_seqs: List[List[str]],
    all_pred_word_seqs: List[List[str]],
    error_dir: Path,
    num_samples: int = 10,
) -> None:
    """
    Save representative correct/incorrect prediction samples with visualizations.

    Saves:
    - error_analysis/{dataset}/correct_samples.json
    - error_analysis/{dataset}/incorrect_samples.json
    - error_analysis/{dataset}/sample_{id}_pred.png  (up to num_samples/2 each)
    """
    correct_records: List[Dict] = []
    incorrect_records: List[Dict] = []
    max_each = num_samples // 2

    for idx, (true_seq, pred_seq) in enumerate(zip(all_true_word_seqs, all_pred_word_seqs)):
        if idx >= len(test_dataset.samples):
            break

        raw_sample = test_dataset.get_raw_sample(idx)
        words = raw_sample.get("words", [])
        bboxes = raw_sample.get("bboxes", [])

        # Align lengths
        min_len = min(len(words), len(true_seq), len(pred_seq))
        words_sub = words[:min_len]
        true_sub  = true_seq[:min_len]
        pred_sub  = pred_seq[:min_len]
        bboxes_sub = bboxes[:min_len]

        errors = sum(1 for t, p in zip(true_sub, pred_sub) if t != p)
        is_correct = errors == 0

        record = {
            "sample_id":   raw_sample.get("id", str(idx)),
            "dataset":     dataset_name,
            "num_words":   min_len,
            "num_errors":  errors,
            "error_rate":  round(errors / max(1, min_len), 4),
            "words":       words_sub,
            "true_labels": true_sub,
            "pred_labels": pred_sub,
        }

        if is_correct and len(correct_records) < max_each:
            correct_records.append(record)
            _save_pred_visualization(raw_sample, pred_sub, bboxes_sub, error_dir, idx, "correct")
        elif not is_correct and len(incorrect_records) < max_each:
            incorrect_records.append(record)
            _save_pred_visualization(raw_sample, pred_sub, bboxes_sub, error_dir, idx, "incorrect")

        if len(correct_records) >= max_each and len(incorrect_records) >= max_each:
            break

    # Save JSON records
    for name, records in [("correct", correct_records), ("incorrect", incorrect_records)]:
        out_path = error_dir / f"{name}_samples.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        logger.info("Error analysis: saved %d %s samples to %s", len(records), name, out_path)


def _save_pred_visualization(
    raw_sample: Dict[str, Any],
    pred_labels: List[str],
    bboxes: List[List[int]],
    error_dir: Path,
    idx: int,
    tag: str,
) -> None:
    """Save a prediction visualization for one sample."""
    try:
        img_path_str = raw_sample.get("image_path", "")
        img_path = _PROJECT_ROOT / img_path_str if img_path_str else None

        if img_path and img_path.exists():
            from PIL import Image as PILImage
            img = PILImage.open(img_path).convert("RGB")
        else:
            return  # Skip if image unavailable

        words = raw_sample.get("words", [])
        min_len = min(len(words), len(pred_labels), len(bboxes))

        fig = visualize_document(
            image=img,
            words=words[:min_len],
            bboxes=bboxes[:min_len],
            labels=pred_labels[:min_len],
            title=f"[{tag.upper()}] {raw_sample.get('id', idx)} — Predicted",
            normalized=True,
            save_path=error_dir / f"sample_{idx}_{tag}_pred.png",
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as exc:
        logger.debug("Could not save visualization for sample %s: %s", idx, exc)


# ---------------------------------------------------------------------------
# Model comparison report
# ---------------------------------------------------------------------------

def generate_model_comparison(
    layoutlmv3_results: List[Dict[str, Any]],
    reports_dir: Path,
) -> Path:
    """
    Append LayoutLMv3 results to Phase-3 baseline_results.csv and
    generate a combined model_comparison.csv.
    """
    baseline_csv = reports_dir / "baseline_results.csv"
    comparison_csv = reports_dir / "model_comparison.csv"

    rows: List[Dict] = []

    # Load existing Phase-3 baseline results.
    if baseline_csv.exists():
        baseline_df = pd.read_csv(baseline_csv)
        rows.extend(baseline_df.to_dict("records"))

    # Append LayoutLMv3 results.
    rows.extend(layoutlmv3_results)

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(comparison_csv, index=False)
    logger.info("Model comparison saved to %s", comparison_csv)

    # Also save JSON for completeness.
    comparison_json = reports_dir / "model_comparison.json"
    with open(comparison_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # Print summary table.
    display_cols = ["dataset", "model_type", "macro_f1", "micro_f1", "precision", "recall"]
    available = [c for c in display_cols if c in comparison_df.columns]
    logger.info("\n%s", comparison_df[available].to_string(index=False))

    return comparison_csv


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_evaluation(
    datasets: List[str],
    stage: str,
    config_path: str = "configs/layoutlmv3.yaml",
) -> List[Dict[str, Any]]:
    all_results: List[Dict[str, Any]] = []
    training_cfg = load_training_config(config_path)
    reports_dir  = training_cfg.resolved_reports_dir()

    for ds_name in datasets:
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATING %s [%s]", ds_name.upper(), stage.upper())
        logger.info("=" * 60)

        result = evaluate_dataset(
            dataset_name=ds_name,
            stage=stage,
            config_path=config_path,
            reports_dir=str(reports_dir),
        )
        if result:
            all_results.append(result)

    if all_results:
        generate_model_comparison(all_results, reports_dir)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LayoutLMv3 on test splits")
    parser.add_argument("--dataset", nargs="+", default=["sroie"],
                        choices=["sroie", "funsd", "cord"])
    parser.add_argument("--stage", default="full",
                        choices=["smoke", "small", "full"])
    parser.add_argument("--config", default="configs/layoutlmv3.yaml")
    args = parser.parse_args()

    run_evaluation(
        datasets=args.dataset,
        stage=args.stage,
        config_path=args.config,
    )
