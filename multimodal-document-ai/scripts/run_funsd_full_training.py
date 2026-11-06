"""
Full FUNSD LayoutLMv3 Training + Test Evaluation Orchestrator
============================================================
Phase 5 — Full training experiment on FUNSD form understanding benchmark.

Run from project root:
    python scripts/run_funsd_full_training.py
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.layoutlmv3.config import load_dataset_config, load_training_config
from models.layoutlmv3.dataset import LayoutLMv3DocumentDataset, WORD_ID_PAD
from models.layoutlmv3.model import LayoutLMv3Classifier
from models.layoutlmv3.tokenizer import LayoutLMv3TokenizerWrapper
from evaluation.layoutlmv3_metrics import compute_seqeval_metrics, compute_token_metrics
from inference.layoutlmv3_predict import token_preds_to_word_preds, predict_batch_documents
from training.train_layoutlmv3 import run_training, word_ids_tensor_to_list, extract_word_level_labels

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("run_funsd_full")

DATASET = "funsd"
STAGE   = "full"
CONFIG  = "configs/layoutlmv3.yaml"
REPORTS = _PROJECT_ROOT / "reports"
FIG_DIR = REPORTS / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Pre-flight info
# ---------------------------------------------------------------------------

def print_preflight(cfg, ds_cfg):
    sep = "=" * 65
    print(f"\n{sep}")
    print("  FUNSD LayoutLMv3 Full Training -- Pre-flight Check")
    print(f"{sep}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        device   = "cuda"
    else:
        gpu_name = None
        vram_gb  = None
        device   = "cpu"

    eff_bs = cfg.batch_size * cfg.gradient_accumulation_steps

    print(f"  GPU name        : {gpu_name or '(none - CPU only)'}")
    print(f"  GPU memory      : {f'{vram_gb:.2f} GB' if vram_gb else 'N/A'}")
    print(f"  Effective bs    : {cfg.batch_size} x {cfg.gradient_accumulation_steps} = {eff_bs}")
    print(f"  Learning rate   : {cfg.learning_rate:.2e}")
    print(f"  FP16            : {cfg.fp16}")
    print(f"  Epochs          : {cfg.epochs}")
    print(f"  Early stopping  : patience={cfg.early_stopping_patience}")

    def count_lines(p):
        if not Path(p).exists(): return 0
        with open(p, encoding="utf-8") as f: return sum(1 for _ in f)

    base = cfg.resolved_processed_dir() / DATASET
    n_train = count_lines(base / "train.jsonl")
    n_val   = count_lines(base / "val.jsonl")
    n_test  = count_lines(base / "test.jsonl")
    print(f"  Train samples   : {n_train}")
    print(f"  Val samples     : {n_val}")
    print(f"  Test samples    : {n_test}")
    print(f"  Num labels      : {ds_cfg.num_labels}")
    print(f"  Labels          : {ds_cfg.labels}")
    print(f"{sep}\n")

    return dict(gpu_name=gpu_name, vram_gb=round(vram_gb,2) if vram_gb else None,
                device=device, batch_size=cfg.batch_size,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                effective_batch_size=eff_bs, learning_rate=cfg.learning_rate,
                fp16=cfg.fp16, n_train=n_train, n_val=n_val, n_test=n_test,
                num_labels=ds_cfg.num_labels, labels=ds_cfg.labels)


# ---------------------------------------------------------------------------
# 2. Find best checkpoint
# ---------------------------------------------------------------------------

def find_best_checkpoint(cfg):
    ckpt_dir = cfg.resolved_checkpoint_dir() / DATASET / STAGE
    if not ckpt_dir.exists():
        logger.error("Checkpoint dir not found: %s", ckpt_dir)
        return None, {}
    meta_path = ckpt_dir / "checkpoint_meta.json"
    if not meta_path.exists():
        logger.error("checkpoint_meta.json missing in %s", ckpt_dir)
        return None, {}
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    logger.info("Best checkpoint: epoch=%d val_macro_f1=%.4f dir=%s",
                meta.get("epoch", -1), meta.get("val_macro_f1", 0.0), ckpt_dir)
    return ckpt_dir, meta


# ---------------------------------------------------------------------------
# 3. Test-set evaluation
# ---------------------------------------------------------------------------

def evaluate_test_set(cfg, ds_cfg, ckpt_dir):
    logger.info("=" * 60)
    logger.info("TEST-SET EVALUATION -- %s [%s]", DATASET.upper(), STAGE)
    logger.info("=" * 60)

    proc_dir = ckpt_dir / "processor"
    proc_src = str(proc_dir) if proc_dir.exists() else cfg.model_checkpoint
    tok_wrap = LayoutLMv3TokenizerWrapper.from_pretrained(
        checkpoint=proc_src, max_seq_length=cfg.max_seq_length)
    classifier = LayoutLMv3Classifier.load_checkpoint(
        checkpoint_dir=ckpt_dir, dataset_config=ds_cfg)
    classifier.eval_mode()

    test_jsonl = cfg.resolved_processed_dir() / DATASET / "test.jsonl"
    test_ds = LayoutLMv3DocumentDataset(
        jsonl_path=test_jsonl, dataset_config=ds_cfg,
        tokenizer_wrapper=tok_wrap, split="test", project_root=_PROJECT_ROOT)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    logger.info("Test dataset: %d samples", len(test_ds))

    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = classifier.forward(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                bbox=batch["bbox"], pixel_values=batch["pixel_values"])
            logits = out["logits"]
            pred_ids = torch.argmax(logits, dim=-1)
            probs    = torch.softmax(logits, dim=-1).max(dim=-1).values
            for i in range(batch["input_ids"].shape[0]):
                wids = word_ids_tensor_to_list(batch["word_ids"][i])
                true_w = extract_word_level_labels(
                    batch["labels"][i].cpu().tolist(), wids, ds_cfg.id2label)
                if not true_w: continue
                pred_w, _ = token_preds_to_word_preds(
                    token_label_ids=pred_ids[i].cpu().tolist(),
                    token_probs=probs[i].cpu().tolist(),
                    word_ids=wids, id2label=ds_cfg.id2label, num_words=len(true_w))
                all_true.append(true_w)
                all_pred.append(pred_w)

    seqeval_m = compute_seqeval_metrics(all_true, all_pred)
    flat_t = [l for s in all_true for l in s]
    flat_p = [l for s in all_pred for l in s]
    token_m = compute_token_metrics(flat_t, flat_p, labels=ds_cfg.labels)

    sep = "=" * 65
    print(f"\n{sep}")
    print("  TEST SET RESULTS -- FUNSD LayoutLMv3-base (full)")
    print(f"{sep}")
    print(f"  Seqeval precision : {seqeval_m['precision']:.4f}")
    print(f"  Seqeval recall    : {seqeval_m['recall']:.4f}")
    print(f"  Seqeval macro F1  : {seqeval_m['macro_f1']:.4f}")
    print(f"  Seqeval micro F1  : {seqeval_m['micro_f1']:.4f}")
    print(f"  Token weighted F1 : {token_m.get('weighted_f1', 0.0):.4f}")
    print("\n  Per-entity:")
    for ent, vals in seqeval_m.get("per_entity", {}).items():
        print(f"    {ent:<22} P={vals.get('precision', 0):.3f}  "
              f"R={vals.get('recall', 0):.3f}  F1={vals.get('f1', 0):.3f}  "
              f"support={vals.get('support', 0)}")
    print(f"\n  Classification Report:\n{seqeval_m.get('classification_report', 'N/A')}")
    print(sep)

    o_count = flat_p.count("O")
    o_ratio = o_count / max(1, len(flat_p))
    if o_ratio > 0.95 or seqeval_m["macro_f1"] < 0.05:
        logger.error("WARNING: %.1f%% predictions are 'O'. macro_f1=%.4f.",
                     o_ratio * 100, seqeval_m["macro_f1"])
    else:
        logger.info("Test healthy -- macro_f1=%.4f  O-ratio=%.1f%%",
                    seqeval_m["macro_f1"], o_ratio * 100)

    # Plot Confusion Matrix
    plot_confusion_matrix(flat_t, flat_p, ds_cfg.labels)

    return dict(seqeval=seqeval_m, token_level=token_m,
                num_test_samples=len(test_ds),
                num_sequences=len(all_true),
                o_ratio=round(o_ratio, 4),
                all_true_word_seqs=all_true,
                all_pred_word_seqs=all_pred)


# ---------------------------------------------------------------------------
# 4. Save results & Model Comparison
# ---------------------------------------------------------------------------

def save_results(preflight, train_summary, test_results, ckpt_meta):
    seqeval_m = test_results["seqeval"]

    p = REPORTS / "layoutlmv3_funsd_per_entity.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"dataset": DATASET, "stage": STAGE,
                   "seqeval": {k: v for k, v in seqeval_m.items() if k != "classification_report"},
                   "token_level": test_results["token_level"]}, f, indent=2)
    logger.info("Per-entity -> %s", p)

    p = REPORTS / "layoutlmv3_funsd_classification_report.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"LayoutLMv3 -- FUNSD -- Test Set\n{'='*60}\n\n")
        f.write(seqeval_m.get("classification_report", "N/A"))
    logger.info("Classification report -> %s", p)

    p = REPORTS / "layoutlmv3_funsd_test_predictions.json"
    pairs = [{"true": t, "pred": pr}
             for t, pr in zip(test_results["all_true_word_seqs"],
                              test_results["all_pred_word_seqs"])]
    with open(p, "w", encoding="utf-8") as f:
        json.dump(pairs, f)
    logger.info("Predictions -> %s", p)

    summary = {
        "experiment": "funsd_layoutlmv3_full",
        "timestamp": datetime.now().isoformat(),
        "preflight": preflight,
        "best_checkpoint": ckpt_meta,
        "train_summary": {k: v for k, v in train_summary.items() if k != "epoch_history"},
        "epoch_history": train_summary.get("epoch_history", []),
        "test_results": {
            "num_test_samples": test_results["num_test_samples"],
            "num_sequences_evaluated": test_results["num_sequences"],
            "o_ratio": test_results["o_ratio"],
            "seqeval_precision": seqeval_m["precision"],
            "seqeval_recall": seqeval_m["recall"],
            "seqeval_macro_f1": seqeval_m["macro_f1"],
            "seqeval_micro_f1": seqeval_m["micro_f1"],
            "per_entity": seqeval_m.get("per_entity", {}),
            "token_weighted_f1": test_results["token_level"].get("weighted_f1", 0.0),
        },
    }
    p = REPORTS / "layoutlmv3_funsd_full_experiment_summary.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Full summary -> %s", p)

    # Append to model_comparison.csv
    comp_csv = REPORTS / "model_comparison.csv"
    row = {"dataset": "FUNSD", "model_type": "LayoutLMv3-base (full)", "split": "test",
           "macro_f1": round(seqeval_m["macro_f1"], 4),
           "micro_f1": round(seqeval_m["micro_f1"], 4),
           "precision": round(seqeval_m["precision"], 4),
           "recall": round(seqeval_m["recall"], 4),
           "weighted_f1": round(test_results["token_level"].get("weighted_f1", 0.0), 4),
           "total_tokens_or_samples": seqeval_m.get("num_tokens_true", 0)}
    write_header = not comp_csv.exists()
    with open(comp_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)
    logger.info("model_comparison.csv -> %s", comp_csv)

    # Update model_comparison.json
    comp_json = REPORTS / "model_comparison.json"
    existing = []
    if comp_json.exists():
        try:
            with open(comp_json, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing = [r for r in existing if not (r.get("dataset") == "FUNSD" and r.get("model_type") == "LayoutLMv3-base (full)")]
    existing.append(row)
    with open(comp_json, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------

def plot_training_curves(train_summary):
    history = train_summary.get("epoch_history", [])
    if not history:
        logger.warning("No epoch history -- skipping plots.")
        return

    epochs    = [h["epoch"] for h in history]
    tr_loss   = [h.get("train_loss", float("nan")) for h in history]
    val_loss  = [h.get("val_metrics", {}).get("avg_loss", float("nan")) for h in history]
    val_macro = [h.get("val_metrics", {}).get("macro_f1", float("nan")) for h in history]
    val_micro = [h.get("val_metrics", {}).get("micro_f1", float("nan")) for h in history]
    val_prec  = [h.get("val_metrics", {}).get("precision", float("nan")) for h in history]
    val_rec   = [h.get("val_metrics", {}).get("recall", float("nan")) for h in history]

    # Loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, tr_loss,  "o-",  color="#4895ef", lw=2, label="Train Loss")
    ax.plot(epochs, val_loss, "s--", color="#f72585", lw=2, label="Val Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("FUNSD LayoutLMv3 -- Train/Validation Loss", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIG_DIR / "funsd_full_loss_curves.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    logger.info("Loss curves -> %s", p)

    # F1 curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, val_macro, "o-",  color="#f72585", lw=2, label="Val Macro F1")
    ax.plot(epochs, val_micro, "s--", color="#7209b7", lw=2, label="Val Micro F1")
    ax.plot(epochs, val_prec,  "^:",  color="#4cc9f0", lw=1.5, label="Val Precision")
    ax.plot(epochs, val_rec,   "v:",  color="#4895ef", lw=1.5, label="Val Recall")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title("FUNSD LayoutLMv3 -- Validation Metrics", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIG_DIR / "funsd_full_f1_curves.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    logger.info("F1 curves -> %s", p)

    # Combined
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, tr_loss,  "o-",  color="#4895ef", lw=2, label="Train Loss")
    ax1.plot(epochs, val_loss, "s--", color="#f72585", lw=2, label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, val_macro, "o-",  color="#f72585", lw=2, label="Val Macro F1")
    ax2.plot(epochs, val_micro, "s--", color="#7209b7", lw=2, label="Val Micro F1")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1"); ax2.set_ylim(0, 1.05)
    ax2.set_title("Validation F1"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle("FUNSD LayoutLMv3-base Full Training", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = FIG_DIR / "training_curves_funsd_full.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    logger.info("Combined curves -> %s", p)


def plot_confusion_matrix(flat_true: List[str], flat_pred: List[str], labels: List[str]):
    cm = confusion_matrix(flat_true, flat_pred, labels=labels)

    # Raw CM
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title("FUNSD LayoutLMv3 Token Confusion Matrix", fontweight="bold")
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=8)
    fig.tight_layout()
    p1 = FIG_DIR / "funsd_layoutlmv3_confusion_matrix.png"
    fig.savefig(p1, dpi=150); plt.close(fig)

    # Normalized CM
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-6)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title("FUNSD LayoutLMv3 Normalized Confusion Matrix", fontweight="bold")
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.tight_layout()
    p2 = FIG_DIR / "funsd_layoutlmv3_confusion_matrix_normalized.png"
    fig.savefig(p2, dpi=150); plt.close(fig)
    logger.info("Confusion matrices saved: %s, %s", p1, p2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    logger.info("=" * 65)
    logger.info("FUNSD LayoutLMv3 Full Training Experiment -- START")
    logger.info("=" * 65)

    cfg    = load_training_config(CONFIG, dataset_name=DATASET)
    ds_cfg = load_dataset_config(DATASET, labels_dir=cfg.labels_dir)

    preflight = print_preflight(cfg, ds_cfg)

    if "--eval-only" in sys.argv:
        logger.info("Running in --eval-only mode with existing best checkpoint...")
        summary_candidates = sorted((_PROJECT_ROOT / "experiments").glob("layoutlmv3_funsd_full_*/training_summary.json"))
        if summary_candidates:
            latest_summary = summary_candidates[-1]
            logger.info("Loading training summary from %s", latest_summary)
            with open(latest_summary, "r", encoding="utf-8") as f:
                train_summary = json.load(f)
        else:
            train_summary = {}
    else:
        logger.info("Launching full training...")
        results = run_training(datasets=[DATASET], stage=STAGE, config_path=CONFIG)
        train_summary = results.get(DATASET, {})

        if "error" in train_summary:
            logger.error("Training FAILED: %s", train_summary["error"])
            sys.exit(1)

        logger.info("Training done -- best_val_macro_f1=%.4f epochs=%d time=%.0fs",
                    train_summary.get("best_val_macro_f1", 0.0),
                    train_summary.get("epochs_trained", 0),
                    train_summary.get("training_time_s", 0))

    ckpt_dir, ckpt_meta = find_best_checkpoint(cfg)
    if ckpt_dir is None:
        logger.error("No checkpoint found -- aborting.")
        sys.exit(1)

    print(f"\n  Best checkpoint: epoch={ckpt_meta.get('epoch')}  "
          f"val_macro_f1={ckpt_meta.get('val_macro_f1', 0.0):.4f}")

    test_results = evaluate_test_set(cfg, ds_cfg, ckpt_dir)

    seqeval_m = test_results["seqeval"]
    if seqeval_m["macro_f1"] < 0.01 or test_results["o_ratio"] > 0.98:
        logger.error("DEGENERATE PREDICTIONS -- saving partial results and STOPPING.")
        save_results(preflight, train_summary, test_results, ckpt_meta)
        sys.exit(2)

    save_results(preflight, train_summary, test_results, ckpt_meta)
    plot_training_curves(train_summary)

    elapsed = time.time() - t0
    sep = "=" * 65
    print(f"\n{sep}")
    print("  EXPERIMENT COMPLETE -- FUNSD")
    print(f"  Total wall time  : {elapsed/60:.1f} min")
    print(f"  Best val macro F1: {train_summary.get('best_val_macro_f1', 0.0):.4f}")
    print(f"  Test macro F1    : {seqeval_m['macro_f1']:.4f}")
    print(f"  Test micro F1    : {seqeval_m['micro_f1']:.4f}")
    print(f"  Test precision   : {seqeval_m['precision']:.4f}")
    print(f"  Test recall      : {seqeval_m['recall']:.4f}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
