"""
LayoutLMv3 test-set evaluation script.

Loads the best checkpoint for a dataset, runs inference on the test split,
computes seqeval + token-level metrics, generates error analysis samples,
and writes all reports to reports/.

Usage
-----
python evaluation/evaluate_layoutlmv3.py --dataset sroie --stage full
python evaluation/evaluate_layoutlmv3.py --dataset sroie funsd cord --stage full
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.layoutlmv3.config import load_dataset_config, load_training_config
from models.layoutlmv3.model import LayoutLMv3Classifier
from models.layoutlmv3.tokenizer import LayoutLMv3TokenizerWrapper
from models.layoutlmv3.dataset import LayoutLMv3DocumentDataset, WORD_ID_PAD
from evaluation.layoutlmv3_metrics import (
    compute_seqeval_metrics,
    compute_token_metrics,
    compute_sroie_entity_accuracy,
)
from inference.layoutlmv3_predict import (
    token_preds_to_word_preds,
    predict_batch_documents,
    reconstruct_entities,
)
from training.train_layoutlmv3 import (
    word_ids_tensor_to_list,
    extract_word_level_labels,
)
from utils.visualization import visualize_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("evaluate_layoutlmv3")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def find_best_checkpoint(
    dataset_name: str,
    stage: str,
    checkpoints_root: Path,
) -> Optional[Path]:
    ckpt_dir = checkpoints_root / dataset_name / stage
    if not ckpt_dir.exists():
        logger.error("Checkpoint not found: %s", ckpt_dir)
        return None
    meta_path = ckpt_dir / "checkpoint_meta.json"
    if not meta_path.exists():
        logger.error("checkpoint_meta.json missing in %s", ckpt_dir)
        return None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    logger.info(
        "Checkpoint: %s | epoch=%d | val_macro_f1=%.4f",
        ckpt_dir, meta.get("epoch", -1), meta.get("val_macro_f1", 0.0),
    )
    return ckpt_dir


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    dataset_name: str,
    stage: str,
    config_path: str = "configs/layoutlmv3.yaml",
    reports_dir: Optional[str] = None,
    num_error_samples: int = 10,
) -> Dict[str, Any]:
    training_cfg = load_training_config(config_path, dataset_name=dataset_name)
    dataset_cfg  = load_dataset_config(dataset_name, labels_dir=training_cfg.labels_dir)

    ckpt_root = training_cfg.resolved_checkpoint_dir()
    ckpt_dir  = find_best_checkpoint(dataset_name, stage, ckpt_root)
    if ckpt_dir is None:
        logger.error("No checkpoint for %s/%s — skipping.", dataset_name, stage)
        return {}

    out_reports = Path(reports_dir) if reports_dir else training_cfg.resolved_reports_dir()
    out_reports.mkdir(parents=True, exist_ok=True)
    error_dir = out_reports / "error_analysis" / dataset_name
    error_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_reports / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Load processor + model ────────────────────────────────────────────
    proc_dir = ckpt_dir / "processor"
    proc_src = str(proc_dir) if proc_dir.exists() else training_cfg.model_checkpoint
    tokenizer_wrapper = LayoutLMv3TokenizerWrapper.from_pretrained(
        checkpoint=proc_src, max_seq_length=training_cfg.max_seq_length
    )
    classifier = LayoutLMv3Classifier.load_checkpoint(
        checkpoint_dir=ckpt_dir, dataset_config=dataset_cfg
    )
    classifier.eval_mode()

    # ── Load test dataset ─────────────────────────────────────────────────
    test_jsonl = training_cfg.resolved_processed_dir() / dataset_name / "test.jsonl"
    test_dataset = LayoutLMv3DocumentDataset(
        jsonl_path=test_jsonl,
        dataset_config=dataset_cfg,
        tokenizer_wrapper=tokenizer_wrapper,
        split="test",
        project_root=_PROJECT_ROOT,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )

    logger.info("Running inference on %d test samples…", len(test_dataset))

    all_true_word_seqs: List[List[str]] = []
    all_pred_word_seqs: List[List[str]] = []

    with torch.no_grad():
        for batch in test_loader:
            outputs = classifier.forward(
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                bbox           = batch["bbox"],
                pixel_values   = batch["pixel_values"],
            )
            logits         = outputs["logits"]
            pred_ids_batch = torch.argmax(logits, dim=-1)
            probs_batch    = torch.softmax(logits, dim=-1).max(dim=-1).values
            labels_batch   = batch["labels"]
            word_ids_batch = batch["word_ids"]
            bs             = batch["input_ids"].shape[0]

            for i in range(bs):
                word_ids_i  = word_ids_tensor_to_list(word_ids_batch[i])
                label_ids_i = labels_batch[i].cpu().tolist()

                true_words = extract_word_level_labels(
                    label_ids_i, word_ids_i, dataset_cfg.id2label
                )
                if not true_words:
                    continue

                pred_words, _ = token_preds_to_word_preds(
                    token_label_ids=pred_ids_batch[i].cpu().tolist(),
                    token_probs=probs_batch[i].cpu().tolist(),
                    word_ids=word_ids_i,
                    id2label=dataset_cfg.id2label,
                    num_words=len(true_words),
                )
                all_true_word_seqs.append(true_words)
                all_pred_word_seqs.append(pred_words)

    # ── Metrics ───────────────────────────────────────────────────────────
    seqeval_m = compute_seqeval_metrics(all_true_word_seqs, all_pred_word_seqs)
    flat_true = [l for s in all_true_word_seqs for l in s]
    flat_pred = [l for s in all_pred_word_seqs for l in s]
    token_m   = compute_token_metrics(flat_true, flat_pred, labels=dataset_cfg.labels)

    logger.info(
        "TEST [%s] seqeval micro_f1=%.4f | macro_f1=%.4f | precision=%.4f | recall=%.4f",
        dataset_name.upper(),
        seqeval_m["micro_f1"], seqeval_m["macro_f1"],
        seqeval_m["precision"], seqeval_m["recall"],
    )
    logger.info("\n%s", seqeval_m.get("classification_report", ""))

    # ── SROIE entity string accuracy ──────────────────────────────────────
    entity_accuracy: Dict[str, Any] = {}
    if dataset_name == "sroie":
        raw_samples = test_dataset.samples
        n_eval = min(100, len(raw_samples))
        pred_results = predict_batch_documents(
            samples=raw_samples[:n_eval],
            classifier=classifier,
            tokenizer_wrapper=tokenizer_wrapper,
            project_root=_PROJECT_ROOT,
        )
        predicted_entities = [r["entities"] for r in pred_results]
        entity_accuracy = compute_sroie_entity_accuracy(
            samples=raw_samples[:n_eval],
            predicted_entities=predicted_entities,
        )
        logger.info("SROIE entity accuracy: %s", entity_accuracy)

    # ── Save reports ──────────────────────────────────────────────────────
    per_entity_path = out_reports / f"layoutlmv3_{dataset_name}_per_entity.json"
    with open(per_entity_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset_name, "stage": stage,
            "seqeval": seqeval_m, "token_level": token_m,
            "sroie_entity_accuracy": entity_accuracy,
        }, f, indent=2)

    cls_report_path = out_reports / f"layoutlmv3_{dataset_name}_classification_report.txt"
    with open(cls_report_path, "w", encoding="utf-8") as f:
        f.write(f"LayoutLMv3 — {dataset_name.upper()} — Test Set\n{'='*60}\n\n")
        f.write(seqeval_m.get("classification_report", "N/A"))

    # ── Confusion matrix ──────────────────────────────────────────────────
    _save_confusion_matrix(
        flat_true, flat_pred, dataset_cfg.labels,
        fig_dir / f"confusion_matrix_{dataset_name}.png",
        dataset_name,
    )

    # ── Error analysis ────────────────────────────────────────────────────
    _run_error_analysis(
        dataset_name, test_dataset,
        all_true_word_seqs, all_pred_word_seqs,
        error_dir, num_error_samples,
    )

    return {
        "dataset":       dataset_name.upper(),
        "model_type":    f"LayoutLMv3-base ({stage})",
        "split":         "test",
        "macro_f1":      seqeval_m["macro_f1"],
        "micro_f1":      seqeval_m["micro_f1"],
        "weighted_f1":   token_m.get("weighted_f1", 0.0),
        "precision":     seqeval_m["precision"],
        "recall":        seqeval_m["recall"],
        "cer":           0.0,
        "wer":           0.0,
        "total_tokens_or_samples": seqeval_m["num_tokens_true"],
        "per_entity":    seqeval_m.get("per_entity", {}),
        "sroie_entity_accuracy": entity_accuracy,
    }


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def _save_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    save_path: Path,
    dataset_name: str,
) -> None:
    try:
        from sklearn.metrics import confusion_matrix
        entity_labels = [l for l in labels if l != "O"]
        if not entity_labels:
            return
        cm = confusion_matrix(y_true, y_pred, labels=entity_labels)
        fig, ax = plt.subplots(figsize=(max(6, len(entity_labels)), max(5, len(entity_labels)-1)))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_xticks(range(len(entity_labels)))
        ax.set_yticks(range(len(entity_labels)))
        ax.set_xticklabels(entity_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(entity_labels, fontsize=8)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {dataset_name.upper()} (entity labels only)")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved to %s", save_path)
    except Exception as exc:
        logger.warning("Could not save confusion matrix: %s", exc)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def _run_error_analysis(
    dataset_name: str,
    test_dataset: LayoutLMv3DocumentDataset,
    all_true: List[List[str]],
    all_pred: List[List[str]],
    error_dir: Path,
    num_samples: int = 10,
) -> None:
    correct_records: List[Dict] = []
    incorrect_records: List[Dict] = []
    max_each = num_samples // 2

    for idx, (true_seq, pred_seq) in enumerate(zip(all_true, all_pred)):
        if idx >= len(test_dataset.samples):
            break
        raw = test_dataset.get_raw_sample(idx)
        words  = raw.get("words", [])
        bboxes = raw.get("bboxes", [])
        n = min(len(words), len(true_seq), len(pred_seq))
        errors = sum(1 for t, p in zip(true_seq[:n], pred_seq[:n]) if t != p)

        record = {
            "sample_id":   raw.get("id", str(idx)),
            "num_words":   n,
            "num_errors":  errors,
            "error_rate":  round(errors / max(1, n), 4),
            "words":       words[:n],
            "true_labels": true_seq[:n],
            "pred_labels": pred_seq[:n],
        }
        if errors == 0 and len(correct_records) < max_each:
            correct_records.append(record)
            _save_vis(raw, pred_seq[:n], bboxes[:n], error_dir, idx, "correct")
        elif errors > 0 and len(incorrect_records) < max_each:
            incorrect_records.append(record)
            _save_vis(raw, pred_seq[:n], bboxes[:n], error_dir, idx, "incorrect")
        if len(correct_records) >= max_each and len(incorrect_records) >= max_each:
            break

    for name, records in [("correct", correct_records), ("incorrect", incorrect_records)]:
        out = error_dir / f"{name}_samples.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        logger.info("Saved %d %s samples → %s", len(records), name, out)


def _save_vis(raw, pred_labels, bboxes, error_dir, idx, tag):
    try:
        img_path = _PROJECT_ROOT / raw.get("image_path", "")
        if not img_path.exists():
            return
        from PIL import Image as PILImage
        img = PILImage.open(img_path).convert("RGB")
        words = raw.get("words", [])
        n = min(len(words), len(pred_labels), len(bboxes))
        fig = visualize_document(
            image=img, words=words[:n], bboxes=bboxes[:n],
            labels=pred_labels[:n],
            title=f"[{tag.upper()}] {raw.get('id', idx)} — Predicted",
            normalized=True,
            save_path=error_dir / f"sample_{idx}_{tag}_pred.png",
        )
        plt.close(fig)
    except Exception as exc:
        logger.debug("Could not save vis for sample %s: %s", idx, exc)


# ---------------------------------------------------------------------------
# Model comparison report
# ---------------------------------------------------------------------------

def generate_model_comparison(
    layoutlmv3_results: List[Dict[str, Any]],
    reports_dir: Path,
) -> Path:
    baseline_csv = reports_dir / "baseline_results.csv"
    comparison_csv = reports_dir / "model_comparison.csv"

    rows: List[Dict] = []
    if baseline_csv.exists():
        rows.extend(pd.read_csv(baseline_csv).to_dict("records"))
    rows.extend(layoutlmv3_results)

    df = pd.DataFrame(rows)
    df.to_csv(comparison_csv, index=False)
    logger.info("Model comparison saved → %s", comparison_csv)

    with open(reports_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    cols = ["dataset", "model_type", "macro_f1", "micro_f1", "precision", "recall"]
    available = [c for c in cols if c in df.columns]
    logger.info("\n%s", df[available].to_string(index=False))
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
        result = evaluate_dataset(ds_name, stage, config_path, str(reports_dir))
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
    run_evaluation(datasets=args.dataset, stage=args.stage, config_path=args.config)
