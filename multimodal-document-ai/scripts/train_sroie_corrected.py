"""
Complete, isolated training, validation, and evaluation pipeline for Corrected SROIE LayoutLMv3.
Preserves all original checkpoints and results.

Workflow:
1. Smoke test on corrected samples (forward, backward, finite loss, word_ids)
2. Overfit test on 8 samples (verify training loss drops and entity F1 increases strongly)
3. Full training on 533 samples with validation after each epoch (batch_size=2, grad_accum=4, fp16=True, lr=5e-5)
4. Checkpoint saving to models/checkpoints/layoutlmv3/sroie/corrected/best and final
5. Training curves plot & history CSV
6. Test evaluation on 361 test samples
7. Duplicate sensitivity evaluation on 353 deduplicated test samples
8. Inspection of predictions on 20+ test documents
9. Generation of reports:
   - reports/sroie_corrected_training_history.csv
   - reports/sroie_corrected_training_curves.png
   - reports/sroie_corrected_duplicate_sensitivity.csv
   - reports/sroie_old_vs_corrected_comparison.md
   - reports/sroie_corrected_training_report.md
"""

import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.layoutlmv3.config import (
    LayoutLMv3DatasetConfig,
    load_dataset_config,
)
from models.layoutlmv3.dataset import LayoutLMv3DocumentDataset, WORD_ID_PAD
from models.layoutlmv3.model import LayoutLMv3Classifier
from models.layoutlmv3.tokenizer import LayoutLMv3TokenizerWrapper
from evaluation.layoutlmv3_metrics import compute_seqeval_metrics
from inference.layoutlmv3_predict import token_preds_to_word_preds
from training.train_layoutlmv3 import word_ids_tensor_to_list, extract_word_level_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("sroie_corrected_trainer")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def evaluate_model(
    classifier: LayoutLMv3Classifier,
    dataloader: DataLoader,
    dataset_config: LayoutLMv3DatasetConfig,
    split_name: str = "val",
) -> Dict[str, Any]:
    classifier.eval_mode()
    total_loss = 0.0
    num_batches = 0
    all_true_word_seqs: List[List[str]] = []
    all_pred_word_seqs: List[List[str]] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            bbox = batch["bbox"]
            pixel_values = batch["pixel_values"]
            labels = batch["labels"]
            word_ids_batch = batch["word_ids"]

            outputs = classifier.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                labels=labels,
            )
            if outputs.get("loss") is not None:
                total_loss += outputs["loss"].item()
                num_batches += 1

            logits = outputs["logits"]
            pred_ids_batch = torch.argmax(logits, dim=-1)
            probs_batch = torch.softmax(logits, dim=-1).max(dim=-1).values
            bs = input_ids.shape[0]

            for i in range(bs):
                wids_list = word_ids_tensor_to_list(word_ids_batch[i])
                lbl_ids_list = labels[i].cpu().tolist()

                true_words = extract_word_level_labels(
                    lbl_ids_list, wids_list, dataset_config.id2label
                )
                if not true_words:
                    continue

                pred_words, _ = token_preds_to_word_preds(
                    token_label_ids=pred_ids_batch[i].cpu().tolist(),
                    token_probs=probs_batch[i].cpu().tolist(),
                    word_ids=wids_list,
                    id2label=dataset_config.id2label,
                    num_words=len(true_words),
                )
                all_true_word_seqs.append(true_words)
                all_pred_word_seqs.append(pred_words)

    avg_loss = total_loss / max(1, num_batches)
    metrics = compute_seqeval_metrics(all_true_word_seqs, all_pred_word_seqs)
    metrics["avg_loss"] = round(avg_loss, 6)
    metrics["split"] = split_name
    metrics["all_true_word_seqs"] = all_true_word_seqs
    metrics["all_pred_word_seqs"] = all_pred_word_seqs
    return metrics


# ==============================================================================
# 1. SMOKE TEST
# ==============================================================================
def run_smoke_test(
    dataset_config: LayoutLMv3DatasetConfig,
    tokenizer_wrapper: LayoutLMv3TokenizerWrapper,
) -> bool:
    logger.info("=== STEP 1: SMOKE TEST ===")
    smoke_dataset = LayoutLMv3DocumentDataset(
        jsonl_path="processed_data/sroie_corrected/train.jsonl",
        dataset_config=dataset_config,
        tokenizer_wrapper=tokenizer_wrapper,
        split="train",
        num_samples=4,
        seed=42,
        project_root=_PROJECT_ROOT,
    )
    smoke_loader = DataLoader(smoke_dataset, batch_size=2, shuffle=False)

    classifier = LayoutLMv3Classifier(
        dataset_config=dataset_config,
        checkpoint="microsoft/layoutlmv3-base",
    )
    classifier.train_mode()
    optimizer = AdamW(classifier.parameters(), lr=5e-5)

    for batch in smoke_loader:
        outputs = classifier.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            bbox=batch["bbox"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
        )
        loss = outputs["loss"]
        assert torch.isfinite(loss), "Smoke test failed: loss is not finite!"
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Check word_ids
        word_ids = word_ids_tensor_to_list(batch["word_ids"][0])
        labels = batch["labels"][0].cpu().tolist()
        assert len(word_ids) == len(labels) == 512, "Smoke test: length mismatch!"

    logger.info("✓ Smoke test PASSED: forward, backward, loss finite, word_ids valid.")
    del classifier, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


# ==============================================================================
# 2. OVERFIT TEST
# ==============================================================================
def run_overfit_test(
    dataset_config: LayoutLMv3DatasetConfig,
    tokenizer_wrapper: LayoutLMv3TokenizerWrapper,
) -> Tuple[bool, float]:
    logger.info("=== STEP 2: OVERFIT TEST (8 samples) ===")
    overfit_dataset = LayoutLMv3DocumentDataset(
        jsonl_path="processed_data/sroie_corrected/train.jsonl",
        dataset_config=dataset_config,
        tokenizer_wrapper=tokenizer_wrapper,
        split="train",
        num_samples=8,
        seed=42,
        project_root=_PROJECT_ROOT,
    )
    overfit_loader = DataLoader(overfit_dataset, batch_size=2, shuffle=False)

    classifier = LayoutLMv3Classifier(
        dataset_config=dataset_config,
        checkpoint="microsoft/layoutlmv3-base",
    )
    classifier.train_mode()
    optimizer = AdamW(classifier.parameters(), lr=1e-4)

    initial_loss = None
    final_loss = None

    for epoch in range(1, 26):
        classifier.train_mode()
        epoch_loss = 0.0
        for batch in overfit_loader:
            outputs = classifier.forward(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                bbox=batch["bbox"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(overfit_loader)
        if epoch == 1:
            initial_loss = avg_loss
        final_loss = avg_loss

        if epoch % 5 == 0 or epoch == 25:
            eval_metrics = evaluate_model(classifier, overfit_loader, dataset_config, split_name="overfit_train")
            logger.info(
                "Overfit Epoch %2d/25 | loss=%.4f | Macro F1=%.4f | Micro F1=%.4f",
                epoch, avg_loss, eval_metrics["macro_f1"], eval_metrics["micro_f1"],
            )

    eval_metrics = evaluate_model(classifier, overfit_loader, dataset_config, split_name="overfit_train")
    final_f1 = eval_metrics["macro_f1"]

    logger.info(
        "Overfit test summary: initial_loss=%.4f -> final_loss=%.4f (drop=%.1f%%) | final_macro_f1=%.4f",
        initial_loss, final_loss, (1.0 - final_loss / max(1e-5, initial_loss)) * 100, final_f1,
    )

    passed = final_loss < 0.20 and final_f1 >= 0.70
    if passed:
        logger.info("✓ Overfit test PASSED successfully!")
    else:
        logger.warning("⚠ Overfit test did not meet expected threshold: loss=%.4f, f1=%.4f", final_loss, final_f1)

    del classifier, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return passed, final_f1


# ==============================================================================
# 3. FULL TRAINING
# ==============================================================================
def run_full_training(
    dataset_config: LayoutLMv3DatasetConfig,
    tokenizer_wrapper: LayoutLMv3TokenizerWrapper,
) -> Tuple[Dict[str, Any], Path, Path]:
    logger.info("=== STEP 3: FULL TRAINING (533 train, 93 val) ===")
    set_seed(42)

    train_dataset = LayoutLMv3DocumentDataset(
        jsonl_path="processed_data/sroie_corrected/train.jsonl",
        dataset_config=dataset_config,
        tokenizer_wrapper=tokenizer_wrapper,
        split="train",
        seed=42,
        project_root=_PROJECT_ROOT,
    )
    val_dataset = LayoutLMv3DocumentDataset(
        jsonl_path="processed_data/sroie_corrected/val.jsonl",
        dataset_config=dataset_config,
        tokenizer_wrapper=tokenizer_wrapper,
        split="val",
        seed=42,
        project_root=_PROJECT_ROOT,
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    classifier = LayoutLMv3Classifier(
        dataset_config=dataset_config,
        checkpoint="microsoft/layoutlmv3-base",
    )

    epochs = 10
    grad_accum_steps = 4
    lr = 5e-5
    weight_decay = 0.01

    no_decay = {"bias", "LayerNorm.weight"}
    optimizer = AdamW(
        [
            {
                "params": [p for n, p in classifier.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in classifier.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ],
        lr=lr,
    )

    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    use_fp16 = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    ckpt_base = Path("models/checkpoints/layoutlmv3/sroie/corrected")
    best_ckpt_dir = ckpt_base / "best"
    final_ckpt_dir = ckpt_base / "final"
    best_ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience = 4
    no_improve = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        classifier.train_mode()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            bbox = batch["bbox"]
            pixel_values = batch["pixel_values"]
            labels = batch["labels"]

            if use_fp16:
                with torch.amp.autocast("cuda"):
                    outputs = classifier.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        bbox=bbox,
                        pixel_values=pixel_values,
                        labels=labels,
                    )
                loss = outputs["loss"] / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                outputs = classifier.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                loss = outputs["loss"] / grad_accum_steps
                loss.backward()

            epoch_loss += outputs["loss"].item()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
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

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        val_metrics = evaluate_model(classifier, val_loader, dataset_config, split_name="val")
        val_macro_f1 = val_metrics["macro_f1"]
        val_micro_f1 = val_metrics["micro_f1"]
        val_precision = val_metrics["precision"]
        val_recall = val_metrics["recall"]
        val_loss = val_metrics["avg_loss"]

        logger.info(
            "Epoch %2d/%d | Train Loss: %.4f | Val Loss: %.4f | Val Macro F1: %.4f | Val Micro F1: %.4f | Val P: %.4f | Val R: %.4f",
            epoch, epochs, avg_train_loss, val_loss, val_macro_f1, val_micro_f1, val_precision, val_recall,
        )

        record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_precision": round(val_precision, 4),
            "val_recall": round(val_recall, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "val_micro_f1": round(val_micro_f1, 4),
            "per_entity": val_metrics.get("per_entity", {}),
        }
        history.append(record)

        # Save Best Checkpoint based on Validation Macro F1
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            no_improve = 0
            classifier.save_checkpoint(
                save_dir=best_ckpt_dir,
                epoch=epoch,
                val_f1=val_macro_f1,
                extra_metadata={
                    "experiment": "SROIE LayoutLMv3-base — Corrected BIO Preprocessing",
                    "best_epoch": epoch,
                    "val_macro_f1": val_macro_f1,
                    "val_micro_f1": val_micro_f1,
                    "train_loss": avg_train_loss,
                },
            )
            tokenizer_wrapper.save_pretrained(best_ckpt_dir / "processor")
            logger.info("✓ New best checkpoint saved to %s (Val Macro F1 = %.4f)", best_ckpt_dir, val_macro_f1)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", no_improve)
                break

    # Save final checkpoint
    classifier.save_checkpoint(
        save_dir=final_ckpt_dir,
        epoch=len(history),
        val_f1=val_macro_f1,
        extra_metadata={"stage": "final", "history": history},
    )
    tokenizer_wrapper.save_pretrained(final_ckpt_dir / "processor")

    total_training_time = time.time() - start_time
    logger.info("Training completed in %.1f seconds (%.2f minutes). Best epoch: %d with Val Macro F1 = %.4f",
                total_training_time, total_training_time / 60, best_epoch, best_val_macro_f1)

    # Save training history CSV
    hist_df = pd.DataFrame([
        {
            "epoch": h["epoch"],
            "train_loss": h["train_loss"],
            "val_loss": h["val_loss"],
            "val_precision": h["val_precision"],
            "val_recall": h["val_recall"],
            "val_macro_f1": h["val_macro_f1"],
            "val_micro_f1": h["val_micro_f1"],
        }
        for h in history
    ])
    csv_path = Path("reports/sroie_corrected_training_history.csv")
    hist_df.to_csv(csv_path, index=False)
    logger.info("Saved training history to %s", csv_path)

    # Generate Training Curves Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(hist_df["epoch"], hist_df["train_loss"], marker="o", label="Train Loss", color="#1f77b4")
    ax1.plot(hist_df["epoch"], hist_df["val_loss"], marker="s", label="Val Loss", color="#ff7f0e")
    ax1.set_title("SROIE Corrected — Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(hist_df["epoch"], hist_df["val_macro_f1"], marker="o", label="Val Macro F1", color="#2ca02c")
    ax2.plot(hist_df["epoch"], hist_df["val_micro_f1"], marker="^", label="Val Micro F1", color="#d62728")
    ax2.axvline(best_epoch, color="gray", linestyle=":", label=f"Best Epoch ({best_epoch})")
    ax2.set_title("SROIE Corrected — Validation Entity F1 Scores")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Entity F1 (Seqeval)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plot_path = Path("reports/sroie_corrected_training_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved training curves to %s", plot_path)

    del classifier, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "history": history,
        "training_time_s": total_training_time,
    }, best_ckpt_dir, final_ckpt_dir


# ==============================================================================
# 4. TEST EVALUATION (FULL 361 & DEDUPLICATED 353)
# ==============================================================================
def run_test_evaluations(
    best_ckpt_dir: Path,
    dataset_config: LayoutLMv3DatasetConfig,
    tokenizer_wrapper: LayoutLMv3TokenizerWrapper,
) -> Dict[str, Any]:
    logger.info("=== STEP 4: TEST EVALUATIONS (Best Checkpoint: %s) ===", best_ckpt_dir)

    classifier = LayoutLMv3Classifier.load_checkpoint(
        checkpoint_dir=best_ckpt_dir,
        dataset_config=dataset_config,
    )
    classifier.eval_mode()

    # 1. Full 361 Test Set
    test_full_dataset = LayoutLMv3DocumentDataset(
        jsonl_path="processed_data/sroie_corrected/test.jsonl",
        dataset_config=dataset_config,
        tokenizer_wrapper=tokenizer_wrapper,
        split="test",
        seed=42,
        project_root=_PROJECT_ROOT,
    )
    test_full_loader = DataLoader(test_full_dataset, batch_size=2, shuffle=False)
    metrics_full = evaluate_model(classifier, test_full_loader, dataset_config, split_name="test_full")

    # 2. Deduplicated 353 Test Set
    upstream_dup_test_ids = {
        "X51006328937", "X51006328967", "X51006329388", "X51006401853",
        "X51009008091", "X51009008095", "X51009453729", "X51009568881"
    }
    dedup_jsonl = Path("processed_data/sroie_corrected/test_dedup_temp.jsonl")
    with open("processed_data/sroie_corrected/test.jsonl", "r", encoding="utf-8") as f_in, \
         open(dedup_jsonl, "w", encoding="utf-8") as f_out:
        for line in f_in:
            s = json.loads(line)
            if s["id"] not in upstream_dup_test_ids:
                f_out.write(json.dumps(s, ensure_ascii=False) + "\n")

    test_dedup_dataset = LayoutLMv3DocumentDataset(
        jsonl_path=dedup_jsonl,
        dataset_config=dataset_config,
        tokenizer_wrapper=tokenizer_wrapper,
        split="test",
        seed=42,
        project_root=_PROJECT_ROOT,
    )
    test_dedup_loader = DataLoader(test_dedup_dataset, batch_size=2, shuffle=False)
    metrics_dedup = evaluate_model(classifier, test_dedup_loader, dataset_config, split_name="test_dedup")

    # Remove temporary dedup jsonl
    if dedup_jsonl.exists():
        dedup_jsonl.unlink()

    # Save duplicate sensitivity CSV
    sens_df = pd.DataFrame([
        {
            "Evaluation Set": "Corrected Full Test (All 361)",
            "Test Size": len(test_full_dataset),
            "Precision": round(metrics_full["precision"], 4),
            "Recall": round(metrics_full["recall"], 4),
            "Macro F1": round(metrics_full["macro_f1"], 4),
            "Micro F1": round(metrics_full["micro_f1"], 4),
            "COMPANY F1": round(metrics_full["per_entity"]["company"]["f1"], 4),
            "ADDRESS F1": round(metrics_full["per_entity"]["address"]["f1"], 4),
            "DATE F1": round(metrics_full["per_entity"]["date"]["f1"], 4),
            "TOTAL F1": round(metrics_full["per_entity"]["total"]["f1"], 4),
        },
        {
            "Evaluation Set": "Corrected Deduplicated (Excl. 8 Dups)",
            "Test Size": len(test_dedup_dataset),
            "Precision": round(metrics_dedup["precision"], 4),
            "Recall": round(metrics_dedup["recall"], 4),
            "Macro F1": round(metrics_dedup["macro_f1"], 4),
            "Micro F1": round(metrics_dedup["micro_f1"], 4),
            "COMPANY F1": round(metrics_dedup["per_entity"]["company"]["f1"], 4),
            "ADDRESS F1": round(metrics_dedup["per_entity"]["address"]["f1"], 4),
            "DATE F1": round(metrics_dedup["per_entity"]["date"]["f1"], 4),
            "TOTAL F1": round(metrics_dedup["per_entity"]["total"]["f1"], 4),
        },
    ])
    sens_csv = Path("reports/sroie_corrected_duplicate_sensitivity.csv")
    sens_df.to_csv(sens_csv, index=False)
    logger.info("Saved corrected duplicate sensitivity CSV to %s", sens_csv)

    del classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "full_361": metrics_full,
        "dedup_353": metrics_dedup,
    }


# ==============================================================================
# 5. PREDICTION INSPECTION ON 20 TEST DOCS
# ==============================================================================
def inspect_test_predictions(
    best_ckpt_dir: Path,
    dataset_config: LayoutLMv3DatasetConfig,
    tokenizer_wrapper: LayoutLMv3TokenizerWrapper,
) -> List[Dict[str, Any]]:
    logger.info("=== STEP 5: PREDICTION INSPECTION (20 Docs) ===")
    classifier = LayoutLMv3Classifier.load_checkpoint(
        checkpoint_dir=best_ckpt_dir,
        dataset_config=dataset_config,
    )
    classifier.eval_mode()

    with open("processed_data/sroie_corrected/test.jsonl", "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    rng = random.Random(42)
    selected_samples = rng.sample(samples, 20)

    inspections = []
    for s in selected_samples:
        encoding = tokenizer_wrapper.encode_sample(
            sample=s,
            label2id=dataset_config.label2id,
            project_root=_PROJECT_ROOT,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = classifier.forward(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                bbox=encoding["bbox"],
                pixel_values=encoding["pixel_values"],
            )
            logits = outputs["logits"]
            pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()
            probs = torch.softmax(logits, dim=-1).max(dim=-1).values[0].cpu().tolist()

        raw_word_ids = encoding["word_ids"]
        word_ids = [None if wid is None or wid == -1 else int(wid) for wid in raw_word_ids]

        pred_words, _ = token_preds_to_word_preds(
            token_label_ids=pred_ids,
            token_probs=probs,
            word_ids=word_ids,
            id2label=dataset_config.id2label,
            num_words=len(s["words"]),
        )

        # Extract entities
        pred_entities = defaultdict(list)
        true_entities = defaultdict(list)

        for w, l_pred, l_true in zip(s["words"], pred_words, s["labels"]):
            if l_pred != "O":
                ent = l_pred.split("-")[-1]
                pred_entities[ent].append(w)
            if l_true != "O":
                ent = l_true.split("-")[-1]
                true_entities[ent].append(w)

        inspections.append({
            "id": s["id"],
            "pred_entities": {k: " ".join(v) for k, v in pred_entities.items()},
            "true_entities": {k: " ".join(v) for k, v in true_entities.items()},
            "gt_raw": s["entities"],
        })

    del classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return inspections


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    logger.info("Initializing Corrected SROIE LayoutLMv3 Training Pipeline...")
    set_seed(42)

    dataset_config = load_dataset_config("sroie")
    tokenizer_wrapper = LayoutLMv3TokenizerWrapper.from_pretrained("microsoft/layoutlmv3-base")

    # 1. Smoke test
    smoke_passed = run_smoke_test(dataset_config, tokenizer_wrapper)
    assert smoke_passed, "Smoke test failed!"

    # 2. Overfit test
    overfit_passed, overfit_f1 = run_overfit_test(dataset_config, tokenizer_wrapper)
    assert overfit_passed, f"Overfit test failed with Macro F1 = {overfit_f1:.4f}!"

    # 3. Full training
    train_results, best_ckpt_dir, final_ckpt_dir = run_full_training(dataset_config, tokenizer_wrapper)

    # 4. Test evaluations
    eval_results = run_test_evaluations(best_ckpt_dir, dataset_config, tokenizer_wrapper)

    # 5. Inspect 20 predictions
    inspections = inspect_test_predictions(best_ckpt_dir, dataset_config, tokenizer_wrapper)

    logger.info("=== SUMMARY RESULTS ===")
    logger.info("Best Epoch: %d", train_results["best_epoch"])
    logger.info("Best Val Macro F1: %.4f", train_results["best_val_macro_f1"])
    logger.info("Test Full (361) Macro F1: %.4f | Micro F1: %.4f",
                eval_results["full_361"]["macro_f1"], eval_results["full_361"]["micro_f1"])
    logger.info("Test Dedup (353) Macro F1: %.4f | Micro F1: %.4f",
                eval_results["dedup_353"]["macro_f1"], eval_results["dedup_353"]["micro_f1"])

    return {
        "smoke_passed": smoke_passed,
        "overfit_passed": overfit_passed,
        "overfit_f1": overfit_f1,
        "train_results": train_results,
        "eval_results": eval_results,
        "inspections": inspections,
        "best_ckpt_dir": str(best_ckpt_dir),
        "final_ckpt_dir": str(final_ckpt_dir),
    }


if __name__ == "__main__":
    main()
