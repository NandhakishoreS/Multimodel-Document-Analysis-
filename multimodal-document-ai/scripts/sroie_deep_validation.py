"""
Complete SROIE Validation and Error Analysis Script
Implements Steps 1 through 9 of the SROIE validation protocol.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import Counter

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sroie_deep_validation")

REPORTS_DIR = _PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load test dataset, LayoutLMv3 predictions, extractions, and summaries."""
    test_jsonl = _PROJECT_ROOT / "processed_data" / "sroie" / "test.jsonl"
    with open(test_jsonl, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f if line.strip()]

    preds_file = REPORTS_DIR / "layoutlmv3_sroie_test_predictions.json"
    with open(preds_file, "r", encoding="utf-8") as f:
        layoutlm_preds = json.load(f)

    extractions_file = REPORTS_DIR / "layoutlmv3_sroie_entity_extractions.json"
    with open(extractions_file, "r", encoding="utf-8") as f:
        extractions = json.load(f)

    from models.layoutlmv3.config import load_dataset_config
    dataset_cfg = load_dataset_config("sroie", labels_dir=_PROJECT_ROOT / "configs" / "labels")

    return test_samples, layoutlm_preds, extractions, dataset_cfg


def run_step2_per_class_metrics(layoutlm_preds, dataset_cfg):
    """Calculate token-level per-class metrics for LayoutLMv3 across all labels including O."""
    y_true = [lbl for p in layoutlm_preds for lbl in p["true"]]
    y_pred = [lbl for p in layoutlm_preds for lbl in p["pred"]]

    all_labels = dataset_cfg.labels

    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", labels=all_labels, zero_division=0)
    p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", labels=all_labels, zero_division=0)
    p_wtd, r_wtd, f1_wtd, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", labels=all_labels, zero_division=0)

    p_cls, r_cls, f1_cls, supp_cls = precision_recall_fscore_support(y_true, y_pred, average=None, labels=all_labels, zero_division=0)

    rows = []
    for idx, lbl in enumerate(all_labels):
        rows.append({
            "Label": lbl,
            "Precision": round(float(p_cls[idx]), 4),
            "Recall": round(float(r_cls[idx]), 4),
            "F1": round(float(f1_cls[idx]), 4),
            "Support": int(supp_cls[idx]),
        })

    token_summary = {
        "macro_precision": round(float(p_mac), 4),
        "macro_recall": round(float(r_mac), 4),
        "macro_f1": round(float(f1_mac), 4),
        "micro_precision": round(float(p_mic), 4),
        "micro_recall": round(float(r_mic), 4),
        "micro_f1": round(float(f1_mic), 4),
        "weighted_f1": round(float(f1_wtd), 4),
        "total_tokens": len(y_true),
        "per_class": rows,
    }

    entity_labels = [lbl for lbl in all_labels if lbl != "O"]
    p_ent_mac, r_ent_mac, f1_ent_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", labels=entity_labels, zero_division=0)
    token_summary["entity_only_macro_f1"] = round(float(f1_ent_mac), 4)

    return token_summary, y_true, y_pred, all_labels


def run_step5_confusion_matrix(y_true, y_pred, all_labels):
    """Generate raw and normalized confusion matrices for LayoutLMv3."""
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    # Raw Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("LayoutLMv3 Token Confusion Matrix — SROIE Test Set", fontsize=13, fontweight="bold", pad=12)
    fig.colorbar(im)
    tick_marks = np.arange(len(all_labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(all_labels, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)

    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    fig.tight_layout()
    raw_path = FIGURES_DIR / "sroie_layoutlmv3_confusion_matrix.png"
    fig.savefig(raw_path, dpi=150)
    plt.close(fig)
    logger.info("Saved raw confusion matrix to %s", raw_path)

    # Normalized Confusion Matrix (by true row support)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(cm.astype("float"), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_title("LayoutLMv3 Normalized Confusion Matrix — SROIE Test Set", fontsize=13, fontweight="bold", pad=12)
    fig.colorbar(im)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(all_labels, fontsize=9)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.2f}" if row_sums[i] > 0 else "-",
                    ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=8)

    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    fig.tight_layout()
    norm_path = FIGURES_DIR / "sroie_layoutlmv3_confusion_matrix_normalized.png"
    fig.savefig(norm_path, dpi=150)
    plt.close(fig)
    logger.info("Saved normalized confusion matrix to %s", norm_path)

    return cm, cm_norm


def run_step6_prediction_distribution(y_true, y_pred, all_labels):
    """Generate comparison table between true and predicted label counts."""
    true_cnt = Counter(y_true)
    pred_cnt = Counter(y_pred)

    rows = []
    for lbl in all_labels:
        t_c = true_cnt.get(lbl, 0)
        p_c = pred_cnt.get(lbl, 0)
        diff = p_c - t_c
        diff_pct = round((diff / max(1, t_c)) * 100, 2)
        rows.append({
            "Label": lbl,
            "True Count": t_c,
            "Predicted Count": p_c,
            "Difference": diff,
            "Diff %": diff_pct,
        })
    return rows


def run_step8_structured_extraction_verification(test_samples, extractions):
    """
    Calculate field-level exact and partial match metrics for SROIE structured extractions.
    Fields: company, date, address, total.
    """
    fields = ["company", "date", "address", "total"]
    extractions_by_id = {e["sample_id"]: e for e in extractions}

    stats = {f: {"exact": 0, "partial": 0, "total_gt": 0, "total_pred": 0} for f in fields}

    for s in test_samples:
        sid = s["id"]
        gt_ents = s.get("entities", {})
        pred_ents = extractions_by_id.get(sid, {})

        for f in fields:
            gt_val = str(gt_ents.get(f, "")).strip().lower()
            pred_val = str(pred_ents.get(f, "")).strip().lower()

            if gt_val:
                stats[f]["total_gt"] += 1
            if pred_val:
                stats[f]["total_pred"] += 1

            if gt_val and pred_val:
                norm_gt = " ".join(gt_val.split())
                norm_pred = " ".join(pred_val.split())

                if norm_gt == norm_pred:
                    stats[f]["exact"] += 1
                    stats[f]["partial"] += 1
                else:
                    gt_toks = set(norm_gt.split())
                    pred_toks = set(norm_pred.split())
                    jaccard = len(gt_toks & pred_toks) / max(1, len(gt_toks | pred_toks))
                    if norm_gt in norm_pred or norm_pred in norm_gt or jaccard >= 0.5:
                        stats[f]["partial"] += 1

    report_rows = []
    for f in fields:
        tot_gt = stats[f]["total_gt"]
        exact = stats[f]["exact"]
        partial = stats[f]["partial"]
        tot_pred = stats[f]["total_pred"]

        em_rate = round(exact / max(1, tot_gt), 4)
        pm_rate = round(partial / max(1, tot_gt), 4)
        precision = round(partial / max(1, tot_pred), 4)
        recall = round(partial / max(1, tot_gt), 4)
        f1 = round(2 * precision * recall / max(1e-6, precision + recall), 4)

        report_rows.append({
            "Field": f.upper(),
            "Exact Match Count": exact,
            "Exact Match Rate": em_rate,
            "Partial Match Count": partial,
            "Partial Match Rate": pm_rate,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Ground Truth Support": tot_gt,
            "Predicted Support": tot_pred,
        })

    return report_rows


def run_step7_error_analysis(test_samples, extractions, layoutlm_preds):
    """
    Identify representative errors and categorize them across error types.
    """
    extractions_by_id = {e["sample_id"]: e for e in extractions}

    error_samples = []
    fields = ["company", "date", "address", "total"]

    for s in test_samples:
        sid = s["id"]
        gt_ents = s.get("entities", {})
        pred_ents = extractions_by_id.get(sid, {})

        for f in fields:
            gt_val = str(gt_ents.get(f, "")).strip()
            pred_val = str(pred_ents.get(f, "")).strip()

            if gt_val and not pred_val:
                error_samples.append({
                    "sample_id": sid,
                    "field": f.upper(),
                    "ground_truth": gt_val,
                    "predicted": "(none)",
                    "error_type": "missed entity",
                    "reason": f"Model failed to detect any tokens for {f.upper()}",
                })
            elif gt_val and pred_val and " ".join(gt_val.lower().split()) != " ".join(pred_val.lower().split()):
                gt_toks = set(gt_val.lower().split())
                pred_toks = set(pred_val.lower().split())
                jaccard = len(gt_toks & pred_toks) / max(1, len(gt_toks | pred_toks))

                if gt_val.lower() in pred_val.lower() or pred_val.lower() in gt_val.lower() or jaccard > 0.4:
                    err_type = "partial extraction / wrong boundary"
                    reason = "Predicted span covers partial entity or includes surrounding noise words."
                elif any(c in gt_val for c in ["/", "-", "."]) and any(c in pred_val for c in ["/", "-", "."]):
                    err_type = "visual ambiguity / OCR artifact"
                    reason = "Punctuation or header prefix captured with entity value."
                else:
                    err_type = "incorrect entity type / confusion"
                    reason = f"Token span assigned incorrect field or confused with another semantic region."

                error_samples.append({
                    "sample_id": sid,
                    "field": f.upper(),
                    "ground_truth": gt_val,
                    "predicted": pred_val,
                    "error_type": err_type,
                    "reason": reason,
                })

    return error_samples


def main():
    logger.info("Loading test samples and model outputs...")
    test_samples, layoutlm_preds, extractions, dataset_cfg = load_data()

    logger.info("Computing Step 2 per-class metrics...")
    step2_summary, y_true, y_pred, all_labels = run_step2_per_class_metrics(layoutlm_preds, dataset_cfg)
    with open(REPORTS_DIR / "sroie_layoutlmv3_per_class_tokens.json", "w", encoding="utf-8") as f:
        json.dump(step2_summary, f, indent=2)

    logger.info("Generating Step 5 confusion matrix...")
    cm, cm_norm = run_step5_confusion_matrix(y_true, y_pred, all_labels)

    logger.info("Generating Step 6 prediction distribution...")
    step6_dist = run_step6_prediction_distribution(y_true, y_pred, all_labels)
    with open(REPORTS_DIR / "sroie_prediction_distribution.json", "w", encoding="utf-8") as f:
        json.dump(step6_dist, f, indent=2)

    logger.info("Performing Step 7 error analysis...")
    error_records = run_step7_error_analysis(test_samples, extractions, layoutlm_preds)
    with open(REPORTS_DIR / "sroie_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(error_records, f, indent=2)
    pd.DataFrame(error_records).to_csv(REPORTS_DIR / "sroie_error_analysis.csv", index=False)
    logger.info("Saved %d error records to sroie_error_analysis.json and CSV", len(error_records))

    logger.info("Performing Step 8 structured extraction verification...")
    step8_results = run_step8_structured_extraction_verification(test_samples, extractions)
    with open(REPORTS_DIR / "sroie_structured_extraction_accuracy.json", "w", encoding="utf-8") as f:
        json.dump(step8_results, f, indent=2)

    logger.info("All SROIE validation analyses successfully executed!")


if __name__ == "__main__":
    main()
