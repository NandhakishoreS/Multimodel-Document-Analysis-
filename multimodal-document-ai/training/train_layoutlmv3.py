"""
LayoutLMv3 fine-tuning pipeline.

Stages:
  overfit — 8 samples, 30 epochs: intentional memorisation test (F1 must rise).
  smoke   — 15 samples, 2 epochs: pipeline sanity check.
  small   — ~5% of train, 3 epochs: convergence / memory check.
  full    — all samples, configured epochs: production fine-tuning.

Usage
-----
python training/train_layoutlmv3.py --dataset sroie --stage overfit
python training/train_layoutlmv3.py --dataset sroie --stage smoke
python training/train_layoutlmv3.py --dataset sroie --stage full
python training/train_layoutlmv3.py --dataset sroie funsd cord --stage full
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.layoutlmv3.config import (
    LayoutLMv3DatasetConfig,
    LayoutLMv3TrainingConfig,
    load_dataset_config,
    load_training_config,
)
from models.layoutlmv3.dataset import LayoutLMv3DocumentDataset, build_dataloaders, WORD_ID_PAD
from models.layoutlmv3.model import LayoutLMv3Classifier
from models.layoutlmv3.tokenizer import LayoutLMv3TokenizerWrapper
from evaluation.layoutlmv3_metrics import compute_seqeval_metrics
from inference.layoutlmv3_predict import token_preds_to_word_preds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train_layoutlmv3")


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Seed set to %d", seed)


# ─────────────────────────────────────────────────────────────────────────────
# Hardware report
# ─────────────────────────────────────────────────────────────────────────────

def report_hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "gpu_name": None,
        "vram_gb": None,
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        )
        logger.info(
            "Using device: cuda | GPU: %s | VRAM: %.1f GB",
            info["gpu_name"], info["vram_gb"],
        )
    else:
        logger.warning("Using device: cpu — CUDA not available.")
    return info


# ─────────────────────────────────────────────────────────────────────────────
# word_ids helpers (use stored tensor, no heuristics)
# ─────────────────────────────────────────────────────────────────────────────

def word_ids_tensor_to_list(word_ids_t: torch.Tensor) -> List[Optional[int]]:
    """Convert a (seq_len,) int64 word_ids tensor back to List[Optional[int]]."""
    return [None if wid == WORD_ID_PAD else int(wid) for wid in word_ids_t.tolist()]


def extract_word_level_labels(
    label_ids: List[int],
    word_ids: List[Optional[int]],
    id2label: Dict[int, str],
) -> List[str]:
    """
    Produce word-level label strings from token-level label_ids.

    Uses the REAL word_ids (stored in the dataset batch, not reconstructed).
    First-subword rule: only positions where word_id changes assign a label;
    continuation subwords have label_id == -100 and are skipped.
    """
    word_labels: Dict[int, str] = {}
    seen: set = set()
    for lid, wid in zip(label_ids, word_ids):
        if wid is None or lid == -100:
            continue
        if wid not in seen:
            word_labels[wid] = id2label.get(lid, "O")
            seen.add(wid)
    if not word_labels:
        return []
    max_wid = max(word_labels.keys())
    return [word_labels.get(i, "O") for i in range(max_wid + 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_epoch(
    classifier: LayoutLMv3Classifier,
    dataloader: DataLoader,
    dataset_config: LayoutLMv3DatasetConfig,
    split_name: str = "val",
) -> Dict[str, Any]:
    """Run one full evaluation pass; return seqeval metrics + avg_loss."""
    classifier.eval_mode()
    total_loss = 0.0
    num_batches = 0
    all_true_word_seqs: List[List[str]] = []
    all_pred_word_seqs: List[List[str]] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            bbox           = batch["bbox"]
            pixel_values   = batch["pixel_values"]
            labels         = batch["labels"]
            # Real word_ids stored by the Dataset — shape (B, seq_len)
            word_ids_batch = batch["word_ids"]

            outputs = classifier.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                labels=labels,
            )
            if "loss" in outputs:
                total_loss += outputs["loss"].item()
                num_batches += 1

            logits         = outputs["logits"]
            pred_ids_batch = torch.argmax(logits, dim=-1)
            probs_batch    = torch.softmax(logits, dim=-1).max(dim=-1).values
            batch_size     = input_ids.shape[0]

            for i in range(batch_size):
                word_ids_i = word_ids_tensor_to_list(word_ids_batch[i])
                label_ids_i = labels[i].cpu().tolist()

                true_word_labels = extract_word_level_labels(
                    label_ids_i, word_ids_i, dataset_config.id2label
                )
                if not true_word_labels:
                    continue

                pred_word_labels, _ = token_preds_to_word_preds(
                    token_label_ids=pred_ids_batch[i].cpu().tolist(),
                    token_probs=probs_batch[i].cpu().tolist(),
                    word_ids=word_ids_i,
                    id2label=dataset_config.id2label,
                    num_words=len(true_word_labels),
                )
                all_true_word_seqs.append(true_word_labels)
                all_pred_word_seqs.append(pred_word_labels)

    avg_loss = total_loss / max(1, num_batches)
    metrics  = compute_seqeval_metrics(all_true_word_seqs, all_pred_word_seqs)
    metrics["avg_loss"] = round(avg_loss, 6)
    metrics["split"]    = split_name
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_dataset(
    dataset_name: str,
    stage: str,
    config: LayoutLMv3TrainingConfig,
    dataset_config: LayoutLMv3DatasetConfig,
    experiment_dir: Path,
) -> Dict[str, Any]:
    set_seed(config.seed)
    hw_info = report_hardware()

    # Stage parameters
    if stage == "overfit":
        num_train_samples = 8
        epochs            = 30
        eval_on_train     = True   # Evaluate on the same 8 samples to confirm memorisation
    elif stage == "smoke":
        num_train_samples = config.smoke_num_samples
        epochs            = config.smoke_epochs
        eval_on_train     = False
    elif stage == "small":
        num_train_samples = config.small_num_samples
        epochs            = config.small_epochs
        eval_on_train     = False
    else:  # full
        num_train_samples = None
        epochs            = config.epochs
        eval_on_train     = False

    logger.info(
        "=" * 60 + "\nTraining %s | stage=%s | epochs=%d | samples=%s\n" + "=" * 60,
        dataset_name.upper(), stage, epochs,
        num_train_samples if num_train_samples else "all",
    )
    logger.info(
        "lr=%.2e | batch=%d | grad_accum=%d | eff_batch=%d | fp16=%s",
        config.learning_rate, config.batch_size,
        config.gradient_accumulation_steps, config.effective_batch_size,
        config.fp16,
    )

    tokenizer_wrapper = LayoutLMv3TokenizerWrapper.from_pretrained(
        checkpoint=config.model_checkpoint,
        max_seq_length=config.max_seq_length,
    )

    loaders = build_dataloaders(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        tokenizer_wrapper=tokenizer_wrapper,
        processed_data_dir=config.resolved_processed_dir(),
        batch_size=config.batch_size,
        num_workers=0,
        num_train_samples=num_train_samples,
        seed=config.seed,
        project_root=_PROJECT_ROOT,
    )
    train_loader = loaders.get("train")
    # For overfit test, evaluate on train set itself
    eval_loader  = loaders.get("train") if eval_on_train else loaders.get("val")

    if train_loader is None:
        raise RuntimeError(f"No training data for {dataset_name}")

    classifier = LayoutLMv3Classifier(
        dataset_config=dataset_config,
        checkpoint=config.model_checkpoint,
    )

    no_decay = {"bias", "LayerNorm.weight"}
    optimizer = AdamW(
        [
            {
                "params": [p for n, p in classifier.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in classifier.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ],
        lr=config.learning_rate,
    )

    total_steps  = (len(train_loader) // max(1, config.gradient_accumulation_steps)) * epochs
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    logger.info("Scheduler: %d total steps, %d warmup steps", total_steps, warmup_steps)

    use_fp16 = config.fp16 and torch.cuda.is_available()
    scaler   = torch.amp.GradScaler("cuda") if use_fp16 else None
    if use_fp16:
        logger.info("Mixed precision fp16 enabled.")
    else:
        logger.info("Training in fp32.")

    ckpt_dir = config.resolved_checkpoint_dir() / dataset_name / stage
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1   : float = -1.0
    no_improve    : int   = 0
    epoch_history : List[Dict[str, Any]] = []
    train_start   = time.time()

    for epoch in range(1, epochs + 1):
        classifier.train_mode()
        epoch_loss = 0.0
        optimizer.zero_grad()
        step = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            bbox           = batch["bbox"]
            pixel_values   = batch["pixel_values"]
            labels         = batch["labels"]

            if use_fp16:
                with torch.amp.autocast("cuda"):
                    outputs = classifier.forward(
                        input_ids, attention_mask, bbox, pixel_values, labels
                    )
                loss = outputs["loss"] / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = classifier.forward(
                    input_ids, attention_mask, bbox, pixel_values, labels
                )
                loss = outputs["loss"] / config.gradient_accumulation_steps
                loss.backward()

            epoch_loss += outputs["loss"].item()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % max(1, total_steps // 10) == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    logger.info(
                        "Epoch %d/%d | step %d | loss=%.4f | lr=%.2e",
                        epoch, epochs, step,
                        epoch_loss / max(1, batch_idx + 1), lr_now,
                    )

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        logger.info("Epoch %d/%d | avg_train_loss=%.4f", epoch, epochs, avg_train_loss)

        val_metrics: Dict[str, Any] = {}
        if eval_loader and epoch % config.eval_every_n_epochs == 0:
            val_metrics = evaluate_epoch(
                classifier, eval_loader, dataset_config,
                split_name="train_overfit" if eval_on_train else "val",
            )
            val_f1 = val_metrics.get("macro_f1", 0.0)
            logger.info(
                "Epoch %d/%d | %s macro_f1=%.4f | micro_f1=%.4f | loss=%.4f",
                epoch, epochs,
                val_metrics["split"], val_f1,
                val_metrics.get("micro_f1", 0.0),
                val_metrics.get("avg_loss", 0.0),
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                no_improve  = 0
                classifier.save_checkpoint(
                    save_dir=ckpt_dir, epoch=epoch, val_f1=val_f1,
                    extra_metadata={"stage": stage, "train_loss": avg_train_loss},
                )
                tokenizer_wrapper.save_pretrained(ckpt_dir / "processor")
                logger.info("✓ Best checkpoint saved (val macro_f1=%.4f)", val_f1)
            else:
                no_improve += 1
                logger.info(
                    "No improvement %d/%d (best=%.4f)",
                    no_improve, config.early_stopping_patience, best_val_f1,
                )

            if no_improve >= config.early_stopping_patience and stage == "full":
                logger.info(
                    "Early stopping at epoch %d (best_val_f1=%.4f)",
                    epoch, best_val_f1,
                )
                epoch_history.append({
                    "epoch": epoch, "train_loss": avg_train_loss,
                    "val_metrics": {k: v for k, v in val_metrics.items()
                                   if k != "classification_report"},
                    "early_stop": True,
                })
                break

        epoch_history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_metrics": {k: v for k, v in val_metrics.items()
                           if k != "classification_report"},
        })

    training_time_s = time.time() - train_start
    logger.info("Training done in %.1f s (%.1f min)", training_time_s, training_time_s / 60)

    # Overfit test verdict
    if stage == "overfit":
        if best_val_f1 > 0.5:
            logger.info("✓ OVERFIT TEST PASSED — macro_f1=%.4f on train subset", best_val_f1)
        elif best_val_f1 > 0.1:
            logger.warning("⚠ OVERFIT TEST PARTIAL — macro_f1=%.4f (expected >0.5)", best_val_f1)
        else:
            logger.error("✗ OVERFIT TEST FAILED — macro_f1=%.4f after %d epochs", best_val_f1, epochs)

    summary = {
        "experiment_id":      experiment_dir.name,
        "dataset":            dataset_name,
        "stage":              stage,
        "model_checkpoint":   config.model_checkpoint,
        "best_val_macro_f1":  round(best_val_f1, 4),
        "epochs_trained":     len(epoch_history),
        "training_time_s":    round(training_time_s, 1),
        "hardware":           hw_info,
        "config": {
            "learning_rate":               config.learning_rate,
            "batch_size":                  config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size":        config.effective_batch_size,
            "weight_decay":                config.weight_decay,
            "warmup_ratio":                config.warmup_ratio,
            "fp16":                        config.fp16,
            "seed":                        config.seed,
            "max_seq_length":              config.max_seq_length,
        },
        "epoch_history": epoch_history,
        "checkpoint_dir": str(ckpt_dir),
    }

    experiment_dir.mkdir(parents=True, exist_ok=True)
    summary_path = experiment_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved to %s", summary_path)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    datasets: List[str],
    stage: str,
    config_path: str = "configs/layoutlmv3.yaml",
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}

    for ds_name in datasets:
        logger.info("\n" + "=" * 60)
        logger.info("DATASET: %s | STAGE: %s", ds_name.upper(), stage.upper())
        logger.info("=" * 60)

        training_cfg = load_training_config(config_path, dataset_name=ds_name)
        dataset_cfg  = load_dataset_config(ds_name, labels_dir=training_cfg.labels_dir)

        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"layoutlmv3_{ds_name}_{stage}_{ts}"
        exp_dir = training_cfg.resolved_experiments_dir() / exp_id

        logger.info("Experiment: %s", exp_id)
        logger.info("Labels: num_labels=%d | %s", dataset_cfg.num_labels, dataset_cfg.labels)

        try:
            summary = train_one_dataset(
                dataset_name=ds_name,
                stage=stage,
                config=training_cfg,
                dataset_config=dataset_cfg,
                experiment_dir=exp_dir,
            )
            results[ds_name] = summary
            logger.info(
                "✓ %s [%s] done — best_val_f1=%.4f | time=%.0f s",
                ds_name.upper(), stage,
                summary.get("best_val_macro_f1", 0.0),
                summary.get("training_time_s", 0),
            )
        except Exception as exc:
            logger.error("Training failed for %s: %s", ds_name, exc, exc_info=True)
            results[ds_name] = {"error": str(exc), "dataset": ds_name, "stage": stage}

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LayoutLMv3")
    parser.add_argument(
        "--dataset", nargs="+", default=["sroie"],
        choices=["sroie", "funsd", "cord"],
    )
    parser.add_argument(
        "--stage", default="smoke",
        choices=["overfit", "smoke", "small", "full"],
    )
    parser.add_argument("--config", default="configs/layoutlmv3.yaml")
    args = parser.parse_args()

    results = run_training(
        datasets=args.dataset,
        stage=args.stage,
        config_path=args.config,
    )

    logger.info("\n" + "=" * 60 + "\nFINAL SUMMARY\n" + "=" * 60)
    for ds, res in results.items():
        if "error" in res:
            logger.error("  %s: FAILED — %s", ds, res["error"])
        else:
            logger.info(
                "  %s: best_val_f1=%.4f | epochs=%d | time=%.0f s",
                ds,
                res.get("best_val_macro_f1", 0.0),
                res.get("epochs_trained", 0),
                res.get("training_time_s", 0),
            )
