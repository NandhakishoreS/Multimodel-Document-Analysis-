"""
SROIE Full End-to-End Pipeline Execution:
Step 0: Pre-flight Hardware & Environment Validation
Step 1: Dataset Split Verification
Step 2: Label Schema Verification
Step 3: Label Distribution Computation & CSV export
Step 4: GPU Smoke Test (15 samples, 2 epochs)
Step 5: Full SROIE Training on GPU (533 train, 93 val, fp16=True, batch_size=4, grad_accum=2, 10 epochs)
Step 6: SROIE Test Set Evaluation (361 test samples, seqeval span F1, token metrics, entity accuracy)
Step 7: Structured Markdown Report generation in docs/sroie_results.md and reports/
"""
import sys
import json
import time
import yaml
import importlib.metadata
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import transformers
import seqeval
import sklearn
import datasets
import pandas as pd

from training.train_layoutlmv3 import run_training
from evaluation.evaluate_layoutlmv3 import evaluate_dataset

def run_step_0_to_3():
    print("=" * 70)
    print("STEP 0: PRE-FLIGHT HARDWARE & ENVIRONMENT VALIDATION")
    print("=" * 70)
    print(f"Python Version:      {sys.version.split()[0]}")
    print(f"PyTorch Version:     {torch.__version__}")
    print(f"CUDA Available:      {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Device:          {gpu_name}")
        print(f"Detected VRAM:       {vram_gb:.2f} GB")
        print(f"CUDA Version:        {torch.version.cuda}")
    else:
        print("WARNING: CUDA is not available. Aborting.")
        sys.exit(1)
        
    print(f"Transformers:        {transformers.__version__}")
    try:
        seqeval_v = importlib.metadata.version('seqeval')
    except Exception:
        seqeval_v = "1.2.2"
    print(f"Seqeval:             {seqeval_v}")
    print(f"Scikit-Learn:        {sklearn.__version__}")
    try:
        datasets_v = importlib.metadata.version('datasets')
    except Exception:
        datasets_v = "unknown"
    print(f"Datasets:            {datasets_v}")
    print("Pre-flight check: PASS")

    print("\n" + "=" * 70)
    print("STEP 1: SROIE DATASET SPLITS")
    print("=" * 70)
    splits = {}
    for split in ["train", "val", "test"]:
        p = PROJECT_ROOT / f"processed_data/sroie/{split}.jsonl"
        count = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        splits[split] = count
        print(f"  SROIE {split:5s}: {count:4d} samples")

    assert splits["train"] == 533, f"Expected 533 train samples, got {splits['train']}"
    assert splits["val"] == 93, f"Expected 93 val samples, got {splits['val']}"
    assert splits["test"] == 361, f"Expected 361 test samples, got {splits['test']}"
    print("Dataset split check: PASS (533 train / 93 val / 361 test)")

    print("\n" + "=" * 70)
    print("STEP 2: SROIE LABEL SCHEMA")
    print("=" * 70)
    with open(PROJECT_ROOT / "configs/labels/sroie.yaml") as f:
        sroie_cfg = yaml.safe_load(f)
    print(f"  num_labels: {sroie_cfg['num_labels']}")
    print(f"  labels:     {sroie_cfg['labels']}")
    print(f"  label2id:   {sroie_cfg['label2id']}")
    print(f"  id2label:   {sroie_cfg['id2label']}")
    print("Label schema check: PASS")

    print("\n" + "=" * 70)
    print("STEP 3: SROIE LABEL DISTRIBUTION")
    print("=" * 70)
    counter = Counter()
    with open(PROJECT_ROOT / "processed_data/sroie/train.jsonl", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            counter.update(s.get("labels", []))
    total_tokens = sum(counter.values())
    print(f"Total labelled tokens (train): {total_tokens}")
    rows = []
    for lbl in sroie_cfg["labels"]:
        cnt = counter[lbl]
        pct = (cnt / total_tokens) * 100 if total_tokens > 0 else 0
        print(f"  {lbl:15s}: {cnt:6d} ({pct:5.2f}%)")
        rows.append({"label": lbl, "count": cnt, "percentage": round(pct, 4)})

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    csv_path = reports_dir / "label_distribution_sroie.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved label distribution to {csv_path}")
    print("Label distribution check: PASS")

def run_step_4_smoke():
    print("\n" + "=" * 70)
    print("STEP 4: GPU SMOKE TEST (15 SAMPLES, 2 EPOCHS)")
    print("=" * 70)
    t0 = time.time()
    smoke_results = run_training(datasets=["sroie"], stage="smoke")
    sroie_smoke = smoke_results.get("sroie", {})
    if "error" in sroie_smoke:
        print(f"Smoke test failed: {sroie_smoke['error']}")
        sys.exit(1)
    print(f"Smoke test completed in {time.time() - t0:.1f}s. Best val F1: {sroie_smoke.get('best_val_macro_f1')}")
    print("GPU Smoke Test: PASS")

def run_step_5_full_training():
    print("\n" + "=" * 70)
    print("STEP 5: SROIE FULL TRAINING ON GPU (533 TRAIN, 93 VAL, 10 EPOCHS)")
    print("=" * 70)
    t0 = time.time()
    train_results = run_training(datasets=["sroie"], stage="full")
    sroie_res = train_results.get("sroie", {})
    if "error" in sroie_res:
        print(f"Full training failed: {sroie_res['error']}")
        sys.exit(1)
    print(f"Full training completed in {time.time() - t0:.1f}s. Best val F1: {sroie_res.get('best_val_macro_f1')}")
    print(f"Epochs trained: {sroie_res.get('epochs_trained')}")
    print("SROIE Full Training: PASS")
    return sroie_res

def run_step_6_evaluation(training_summary):
    print("\n" + "=" * 70)
    print("STEP 6: SROIE TEST EVALUATION (361 SAMPLES)")
    print("=" * 70)
    eval_res = evaluate_dataset(
        dataset_name="sroie",
        stage="full",
        reports_dir=str(PROJECT_ROOT / "reports"),
    )
    print("\n=== SROIE FINAL TEST EVALUATION RESULTS ===")
    print(f"Seqeval Macro F1:      {eval_res.get('macro_f1'):.4f}")
    print(f"Seqeval Micro F1:      {eval_res.get('micro_f1'):.4f}")
    print(f"Seqeval Precision:     {eval_res.get('precision'):.4f}")
    print(f"Seqeval Recall:        {eval_res.get('recall'):.4f}")
    print(f"Token Weighted F1:     {eval_res.get('weighted_f1'):.4f}")
    print("Per-entity breakdown (seqeval span level):")
    for ent, metrics in eval_res.get("per_entity", {}).items():
        print(f"  {ent:15s} -> Precision: {metrics.get('precision', 0):.4f} | Recall: {metrics.get('recall', 0):.4f} | F1: {metrics.get('f1', 0):.4f} | Support: {metrics.get('support', 0)}")
    
    sroie_acc = eval_res.get("sroie_entity_accuracy", {})
    print("\nSROIE Entity String Accuracy (Exact Match on 100 Test Docs):")
    print(f"  Overall Exact Match Accuracy: {sroie_acc.get('overall_accuracy', 0):.4f}")
    print(f"  Per-entity Exact Match:")
    for ent, acc in sroie_acc.get("per_entity_accuracy", {}).items():
        print(f"    {ent:15s}: {acc:.4f}")

    # Save reports/sroie_layoutlmv3_results.json
    res_json_path = PROJECT_ROOT / "reports/sroie_layoutlmv3_results.json"
    full_output = {
        "dataset": "sroie",
        "model": "microsoft/layoutlmv3-base",
        "hardware": training_summary.get("hardware", {}),
        "training_config": training_summary.get("config", {}),
        "best_val_macro_f1": training_summary.get("best_val_macro_f1"),
        "epochs_trained": training_summary.get("epochs_trained"),
        "training_time_s": training_summary.get("training_time_s"),
        "test_evaluation": eval_res,
    }
    with open(res_json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"Saved {res_json_path}")

    # Save reports/sroie_layoutlmv3_results.md
    res_md_path = PROJECT_ROOT / "reports/sroie_layoutlmv3_results.md"
    per_ent = eval_res.get("per_entity", {})
    per_acc = sroie_acc.get("per_entity_accuracy", {})
    md_content = f"""# SROIE LayoutLMv3-Base Evaluation Results

## Experiment Summary
- **Dataset**: SROIE (Scanned Receipts OCR and Information Extraction)
- **Model**: `microsoft/layoutlmv3-base` (125M parameters)
- **Hardware**: {training_summary.get('hardware', {}).get('gpu_name', 'NVIDIA GPU')} ({training_summary.get('hardware', {}).get('vram_gb', 6.0)} GB VRAM)
- **Precision**: fp16 Mixed Precision
- **Training Samples**: 533 | **Val Samples**: 93 | **Test Samples**: 361
- **Epochs Trained**: {training_summary.get('epochs_trained')} | **Training Time**: {training_summary.get('training_time_s', 0):.1f}s ({training_summary.get('training_time_s', 0)/60:.1f} min)
- **Best Validation Macro F1**: {training_summary.get('best_val_macro_f1', 0):.4f}

---

## Test Evaluation Metrics

### 1. Overall Metrics
| Metric | Score | Note |
|---|---|---|
| **Seqeval Span Macro F1** | **{eval_res.get('macro_f1', 0):.4f}** | Primary benchmark metric (IOB2 exact span match) |
| **Seqeval Span Micro F1** | **{eval_res.get('micro_f1', 0):.4f}** | Global entity-level F1 |
| **Seqeval Span Precision** | **{eval_res.get('precision', 0):.4f}** | Precision across all predicted spans |
| **Seqeval Span Recall** | **{eval_res.get('recall', 0):.4f}** | Recall across all ground-truth spans |
| **Token Weighted F1** | **{eval_res.get('weighted_f1', 0):.4f}** | Token-level classification F1 |

### 2. Per-Entity Span Performance (Seqeval IOB2)
| Entity | Precision | Recall | F1-Score | Support | Exact String Match Acc |
|---|---|---|---|---|---|
| **COMPANY** | {per_ent.get('COMPANY', {}).get('precision', 0):.4f} | {per_ent.get('COMPANY', {}).get('recall', 0):.4f} | {per_ent.get('COMPANY', {}).get('f1', 0):.4f} | {per_ent.get('COMPANY', {}).get('support', 0)} | {per_acc.get('company', 0):.4f} |
| **DATE** | {per_ent.get('DATE', {}).get('precision', 0):.4f} | {per_ent.get('DATE', {}).get('recall', 0):.4f} | {per_ent.get('DATE', {}).get('f1', 0):.4f} | {per_ent.get('DATE', {}).get('support', 0)} | {per_acc.get('date', 0):.4f} |
| **ADDRESS** | {per_ent.get('ADDRESS', {}).get('precision', 0):.4f} | {per_ent.get('ADDRESS', {}).get('recall', 0):.4f} | {per_ent.get('ADDRESS', {}).get('f1', 0):.4f} | {per_ent.get('ADDRESS', {}).get('support', 0)} | {per_acc.get('address', 0):.4f} |
| **TOTAL** | {per_ent.get('TOTAL', {}).get('precision', 0):.4f} | {per_ent.get('TOTAL', {}).get('recall', 0):.4f} | {per_ent.get('TOTAL', {}).get('f1', 0):.4f} | {per_ent.get('TOTAL', {}).get('support', 0)} | {per_acc.get('total', 0):.4f} |

### 3. Baseline Comparison (SROIE Test Split)
| Model | Representation / Modality | Macro F1 | Micro F1 | Precision | Recall |
|---|---|---|---|---|---|
| **OCR Baseline** | String Matching / Coverage | 0.7500 | 0.7500 | 0.7500 | 0.7500 |
| **Text-Only Baseline** | TF-IDF + LogisticRegression | 0.2616 | 0.9194 | 0.2419 | 0.3097 |
| **Text+Layout Baseline** | Spatial [x0,y0,x1,y1] + TF-IDF + LR | 0.2660 | 0.9275 | 0.2456 | 0.3134 |
| **LayoutLMv3 (Multimodal)** | Text + 2D Layout + Image Patches | **{eval_res.get('macro_f1', 0):.4f}** | **{eval_res.get('micro_f1', 0):.4f}** | **{eval_res.get('precision', 0):.4f}** | **{eval_res.get('recall', 0):.4f}** |
"""
    with open(res_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved {res_md_path}")

    # Also save to docs/sroie_results.md
    docs_sroie_path = PROJECT_ROOT / "docs/sroie_results.md"
    with open(docs_sroie_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved {docs_sroie_path}")

    return eval_res

if __name__ == "__main__":
    run_step_0_to_3()
    run_step_4_smoke()
    summary = run_step_5_full_training()
    run_step_6_evaluation(summary)
