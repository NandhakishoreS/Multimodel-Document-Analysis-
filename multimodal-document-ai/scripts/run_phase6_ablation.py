"""
Phase 6: LayoutLMv3 Modality Ablation Study.

This script performs input-zeroing ablation experiments on the trained
LayoutLMv3 checkpoints to investigate each modality's contribution.

IMPORTANT ARCHITECTURAL NOTE
-----------------------------
LayoutLMv3 uses a deep fusion of three input modalities:
  - input_ids  : text tokens (via text embedding)
  - bbox       : 2D spatial bounding boxes (via spatial embedding)
  - pixel_values: document page images (via patch embedding)

These modalities are fused at the very first layer of the transformer,
not via late fusion. Therefore, truly removing a modality without
retraining is architecturally impossible.

What this script does instead is INPUT ZEROING:
  - No-Image condition  : pixel_values = torch.zeros_like(pixel_values)
  - No-Layout condition : bbox = torch.zeros_like(bbox)  [all boxes = (0,0,0,0)]
  - Text-Approx condition: both pixel_values and bbox zeroed

These are NOT architectural ablations. They are modality SENSITIVITY
experiments on the fine-tuned model. The transformer still processes
all input channels. The result shows how robust the model's predictions
are when a specific input signal is removed (e.g., all bounding boxes
set to the degenerate (0,0,0,0) position).

A zero bbox is a valid bbox coordinate — it will receive a spatial
embedding of [0,0,0,0], which the model was not trained to handle.
Similarly, a zero pixel image is not a valid document image.

Interpretation: A large drop in performance when a modality is zeroed
suggests the model relies on that modality. A small drop may indicate
either: (a) the modality is not critical, OR (b) the model is robust to
this specific zeroing strategy. Do NOT over-interpret.

Usage
-----
cd c:\\MDA\\multimodal-document-ai
python scripts/run_phase6_ablation.py --datasets sroie funsd cord
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_site_packages = _PROJECT_ROOT / ".venv312" / "Lib" / "site-packages"
if _site_packages.exists() and str(_site_packages) not in sys.path:
    sys.path.insert(0, str(_site_packages))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "phase6_unified_eval",
    _PROJECT_ROOT / "evaluation" / "phase6_unified_eval.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["phase6_unified_eval"] = _mod
sys.modules["evaluation.phase6_unified_eval"] = _mod
_spec.loader.exec_module(_mod)

compute_strict_span_f1 = _mod.compute_strict_span_f1
SpanF1Result = _mod.SpanF1Result

WORD_ID_PAD = -100

def word_ids_tensor_to_list(word_ids_t: torch.Tensor) -> List[Optional[int]]:
    return [None if wid == WORD_ID_PAD else int(wid) for wid in word_ids_t.tolist()]

def extract_word_level_labels(
    label_ids: List[int],
    word_ids: List[Optional[int]],
    id2label: Dict[int, str],
) -> List[str]:
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("phase6_ablation")

REPORTS_DIR = _PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------------
# Ablation condition definitions
# ---------------------------------------------------------------------------

ABLATION_CONDITIONS = [
    {
        "name": "Full (text + layout + image)",
        "short": "Full",
        "zero_image": False,
        "zero_layout": False,
        "description": (
            "All three input modalities active. "
            "pixel_values and bbox as provided by the dataset preprocessor."
        ),
    },
    {
        "name": "No-Image (text + layout only)",
        "short": "No-Image",
        "zero_image": True,
        "zero_layout": False,
        "description": (
            "pixel_values zeroed (all zeros). bbox unchanged. "
            "Removes visual patch signal. The patch embedding will receive "
            "all-zero patches. Model was fine-tuned with real images so "
            "this is an out-of-distribution input for the image pathway."
        ),
    },
    {
        "name": "No-Layout (text + image only)",
        "short": "No-Layout",
        "zero_image": False,
        "zero_layout": True,
        "description": (
            "bbox zeroed to (0,0,0,0) for all tokens. pixel_values unchanged. "
            "Removes spatial bounding-box signal. All tokens receive the same "
            "spatial embedding corresponding to position (0,0,0,0)."
        ),
    },
    {
        "name": "Text-Approx (text only)",
        "short": "Text-Approx",
        "zero_image": True,
        "zero_layout": True,
        "description": (
            "Both pixel_values and bbox zeroed. "
            "Approximates text-only input within the LayoutLMv3 architecture. "
            "The model is still a LayoutLMv3 transformer (not a BERT). "
            "This is the most aggressive input-zeroing condition."
        ),
    },
]

# ---------------------------------------------------------------------------
# Checkpoint paths for corrected/validated checkpoints
# ---------------------------------------------------------------------------

CHECKPOINT_MAP = {
    "sroie": _PROJECT_ROOT / "models/checkpoints/layoutlmv3/sroie/corrected/best",
    "funsd": _PROJECT_ROOT / "models/checkpoints/layoutlmv3/funsd/full",
    "cord":  _PROJECT_ROOT / "models/checkpoints/layoutlmv3/cord/full",
}

TEST_DATA_MAP = {
    "sroie": _PROJECT_ROOT / "processed_data/sroie_corrected/test.jsonl",
    "funsd": _PROJECT_ROOT / "processed_data/funsd/test.jsonl",
    "cord":  _PROJECT_ROOT / "processed_data/cord/test.jsonl",
}


# ---------------------------------------------------------------------------
# Single ablation run
# ---------------------------------------------------------------------------

def run_ablation_condition(
    dataset_name: str,
    condition: Dict[str, Any],
    classifier,
    tokenizer_wrapper,
    test_dataset,
    dataset_cfg,
) -> Dict[str, Any]:
    """
    Run one ablation condition on the test split and return metrics.

    Parameters
    ----------
    dataset_name : str
    condition : dict
        One entry from ABLATION_CONDITIONS.
    classifier : LayoutLMv3Classifier
    tokenizer_wrapper : LayoutLMv3TokenizerWrapper
    test_dataset : LayoutLMv3DocumentDataset
    dataset_cfg : LayoutLMv3DatasetConfig

    Returns
    -------
    dict with keys: condition, precision, recall, macro_f1, micro_f1, per_entity,
                    n_gold_spans, n_pred_spans
    """
    from inference.layoutlmv3_predict import token_preds_to_word_preds

    logger.info(
        "  Running condition: %s | zero_image=%s | zero_layout=%s",
        condition["name"], condition["zero_image"], condition["zero_layout"]
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=2, shuffle=False, num_workers=0
    )

    all_true_word_seqs: List[List[str]] = []
    all_pred_word_seqs: List[List[str]] = []

    classifier.eval_mode()
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            bbox           = batch["bbox"].clone()
            pixel_values   = batch["pixel_values"].clone()
            labels_batch   = batch["labels"]
            word_ids_batch = batch["word_ids"]

            # Apply ablation condition
            if condition["zero_image"]:
                pixel_values = torch.zeros_like(pixel_values)
            if condition["zero_layout"]:
                bbox = torch.zeros_like(bbox)

            outputs = classifier.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
            )
            logits      = outputs["logits"]
            pred_ids    = torch.argmax(logits, dim=-1)
            probs       = torch.softmax(logits, dim=-1).max(dim=-1).values
            bs          = input_ids.shape[0]

            for i in range(bs):
                word_ids_i  = word_ids_tensor_to_list(word_ids_batch[i])
                label_ids_i = labels_batch[i].cpu().tolist()

                true_words = extract_word_level_labels(
                    label_ids_i, word_ids_i, dataset_cfg.id2label
                )
                if not true_words:
                    continue

                pred_words, _ = token_preds_to_word_preds(
                    token_label_ids=pred_ids[i].cpu().tolist(),
                    token_probs=probs[i].cpu().tolist(),
                    word_ids=word_ids_i,
                    id2label=dataset_cfg.id2label,
                    num_words=len(true_words),
                )
                all_true_word_seqs.append(true_words)
                all_pred_word_seqs.append(pred_words)

    # Compute strict span F1
    span_result = compute_strict_span_f1(all_true_word_seqs, all_pred_word_seqs)

    logger.info(
        "  → Precision=%.4f | Recall=%.4f | Macro F1=%.4f | Micro F1=%.4f | Gold spans=%d",
        span_result.precision, span_result.recall,
        span_result.macro_f1, span_result.micro_f1,
        span_result.n_gold_spans,
    )

    return {
        "dataset": dataset_name.upper(),
        "condition": condition["name"],
        "modalities_active": (
            "text+layout+image" if not condition["zero_image"] and not condition["zero_layout"]
            else "text+layout" if condition["zero_image"] and not condition["zero_layout"]
            else "text+image" if not condition["zero_image"] and condition["zero_layout"]
            else "text-only-approx"
        ),
        "zero_image": condition["zero_image"],
        "zero_layout": condition["zero_layout"],
        "precision": span_result.precision,
        "recall": span_result.recall,
        "macro_f1": span_result.macro_f1,
        "micro_f1": span_result.micro_f1,
        "n_gold_spans": span_result.n_gold_spans,
        "n_pred_spans": span_result.n_pred_spans,
        "per_entity": span_result.per_entity,
        "description": condition["description"],
    }


# ---------------------------------------------------------------------------
# Run full ablation for one dataset
# ---------------------------------------------------------------------------

def run_dataset_ablation(
    dataset_name: str,
    conditions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Load dataset checkpoint and run all ablation conditions.

    Returns list of result dicts, one per condition.
    """
    from models.layoutlmv3.config import load_dataset_config, load_training_config
    from models.layoutlmv3.model import LayoutLMv3Classifier
    from models.layoutlmv3.tokenizer import LayoutLMv3TokenizerWrapper
    from models.layoutlmv3.dataset import LayoutLMv3DocumentDataset

    logger.info("=" * 50)
    logger.info("ABLATION: %s", dataset_name.upper())
    logger.info("=" * 50)

    ckpt_dir = CHECKPOINT_MAP.get(dataset_name)
    test_jsonl = TEST_DATA_MAP.get(dataset_name)

    if not ckpt_dir or not ckpt_dir.exists():
        logger.error("Checkpoint not found for %s: %s", dataset_name, ckpt_dir)
        return [{
            "dataset": dataset_name.upper(),
            "condition": c["name"],
            "modalities_active": "N/A",
            "precision": None,
            "recall": None,
            "macro_f1": None,
            "micro_f1": None,
            "n_gold_spans": None,
            "n_pred_spans": None,
            "per_entity": {},
            "description": "Not evaluated — checkpoint not found.",
        } for c in conditions]

    if not test_jsonl or not test_jsonl.exists():
        logger.error("Test data not found for %s: %s", dataset_name, test_jsonl)
        return [{
            "dataset": dataset_name.upper(),
            "condition": c["name"],
            "modalities_active": "N/A",
            "precision": None,
            "recall": None,
            "macro_f1": None,
            "micro_f1": None,
            "n_gold_spans": None,
            "n_pred_spans": None,
            "per_entity": {},
            "description": "Not evaluated — test data not found.",
        } for c in conditions]

    # Load config
    config_path = str(_PROJECT_ROOT / "configs/layoutlmv3.yaml")
    training_cfg = load_training_config(config_path, dataset_name=dataset_name)
    dataset_cfg = load_dataset_config(dataset_name, labels_dir=training_cfg.labels_dir)

    # Load processor
    proc_dir = ckpt_dir / "processor"
    proc_src = str(proc_dir) if proc_dir.exists() else training_cfg.model_checkpoint

    logger.info("Loading tokenizer from %s", proc_src)
    tokenizer_wrapper = LayoutLMv3TokenizerWrapper.from_pretrained(
        checkpoint=proc_src, max_seq_length=training_cfg.max_seq_length
    )

    logger.info("Loading model from %s", ckpt_dir)
    classifier = LayoutLMv3Classifier.load_checkpoint(
        checkpoint_dir=ckpt_dir, dataset_config=dataset_cfg
    )

    # Load test dataset
    logger.info("Loading test dataset from %s", test_jsonl)
    test_dataset = torch.utils.data.DataLoader  # placeholder
    from models.layoutlmv3.dataset import LayoutLMv3DocumentDataset
    test_ds = LayoutLMv3DocumentDataset(
        jsonl_path=test_jsonl,
        dataset_config=dataset_cfg,
        tokenizer_wrapper=tokenizer_wrapper,
        split="test",
        project_root=_PROJECT_ROOT,
    )
    logger.info("Test dataset size: %d samples", len(test_ds))

    # Run each condition
    results = []
    for cond in conditions:
        try:
            result = run_ablation_condition(
                dataset_name=dataset_name,
                condition=cond,
                classifier=classifier,
                tokenizer_wrapper=tokenizer_wrapper,
                test_dataset=test_ds,
                dataset_cfg=dataset_cfg,
            )
            results.append(result)
        except Exception as exc:
            logger.error("Condition '%s' failed: %s", cond["name"], exc, exc_info=True)
            results.append({
                "dataset": dataset_name.upper(),
                "condition": cond["name"],
                "modalities_active": "ERROR",
                "precision": None,
                "recall": None,
                "macro_f1": None,
                "micro_f1": None,
                "n_gold_spans": None,
                "n_pred_spans": None,
                "per_entity": {},
                "description": f"FAILED: {exc}",
            })

    return results


# ---------------------------------------------------------------------------
# Write ablation results
# ---------------------------------------------------------------------------

def write_ablation_csv(all_results: List[Dict[str, Any]], out_path: Path) -> None:
    rows = []
    for r in all_results:
        full_f1 = None
        # Find the Full condition for this dataset to compute delta
        dataset = r.get("dataset", "")
        for rr in all_results:
            if rr["dataset"] == dataset and rr["condition"] == "Full (text + layout + image)":
                full_f1 = rr.get("macro_f1")
                break

        mf1 = r.get("macro_f1")
        delta = None
        if mf1 is not None and full_f1 is not None:
            delta = round(mf1 - full_f1, 4)

        rows.append({
            "dataset": r.get("dataset"),
            "condition": r.get("condition"),
            "modalities_active": r.get("modalities_active"),
            "precision": r.get("precision"),
            "recall": r.get("recall"),
            "macro_f1": r.get("macro_f1"),
            "micro_f1": r.get("micro_f1"),
            "delta_vs_full": delta,
            "n_gold_spans": r.get("n_gold_spans"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info("Ablation CSV saved → %s", out_path)


def write_ablation_report(all_results: List[Dict[str, Any]], out_path: Path) -> None:
    lines = []
    lines.append("# Phase 6: LayoutLMv3 Modality Ablation Report\n")

    lines.append("## 1. Ablation Methodology\n")
    lines.append(
        "> **IMPORTANT**: This is an **input-zeroing sensitivity study**, NOT an architectural ablation.\n"
        "> LayoutLMv3 deeply fuses text, layout, and image modalities in the first transformer layer.\n"
        "> Removing a modality cleanly would require retraining from scratch.\n"
        "> Instead, we zero the relevant input tensor and measure the performance change.\n"
        "> This shows how sensitive the model's predictions are to each input signal,\n"
        "> but does NOT yield the same conclusions as training separate single-modality models.\n"
    )

    lines.append("\n## 2. Ablation Conditions\n")
    lines.append("| Condition | pixel_values | bbox | Modalities Active |")
    lines.append("|---|:---:|:---:|---|")
    for c in ABLATION_CONDITIONS:
        pv = "Zeroed" if c["zero_image"] else "Normal"
        bb = "Zeroed" if c["zero_layout"] else "Normal"
        lines.append(f"| {c['name']} | {pv} | {bb} | {c['description'][:80]}... |")
    lines.append("")

    lines.append("## 3. Results by Dataset\n")
    datasets = sorted(set(r.get("dataset", "") for r in all_results))
    for ds in datasets:
        lines.append(f"### {ds}\n")
        ds_results = [r for r in all_results if r.get("dataset") == ds]

        # Find full condition F1 for delta
        full_f1 = None
        for r in ds_results:
            if r["condition"] == "Full (text + layout + image)":
                full_f1 = r.get("macro_f1")
                break

        lines.append("| Condition | Precision | Recall | Macro F1 | Micro F1 | Delta vs Full | Gold Spans |")
        lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for r in ds_results:
            mf1 = r.get("macro_f1")
            prec = r.get("precision")
            rec = r.get("recall")
            micf1 = r.get("micro_f1")
            ngold = r.get("n_gold_spans", "N/A")
            if mf1 is not None and full_f1 is not None:
                delta = mf1 - full_f1
                delta_str = f"{delta:+.4f}"
                bold_f1 = f"**{mf1:.4f}**" if r["condition"] == "Full (text + layout + image)" else f"{mf1:.4f}"
            else:
                delta_str = "N/A"
                bold_f1 = mf1 if mf1 is not None else "Not evaluated"
            prec_str = f"{prec:.4f}" if prec is not None else "N/A"
            rec_str = f"{rec:.4f}" if rec is not None else "N/A"
            micf1_str = f"{micf1:.4f}" if micf1 is not None else "N/A"
            lines.append(
                f"| {r['condition']} | {prec_str} | {rec_str} | {bold_f1} | "
                f"{micf1_str} | {delta_str} | {ngold} |"
            )
        lines.append("")

        if ds == "CORD":
            lines.append(
                "> **CORD Caveat**: CORD ablation uses the simplified 3-class task formulation. "
                "Results are not comparable to full-document CORD KIE benchmarks.\n"
            )

    lines.append("## 4. Interpretation\n")
    lines.append(
        "- A large negative Delta vs Full indicates strong reliance on that modality.\n"
        "- A small Delta may indicate robustness OR insensitivity to that zeroing strategy.\n"
        "- These results apply to **this specific fine-tuned model** only.\n"
        "- Do NOT generalize: a different training setup may produce different sensitivities.\n"
        "- The Text-Approx condition is the most aggressive — both layout and image removed.\n"
        "  The model was never trained in this regime, so degradation is expected and informative.\n"
    )

    lines.append("## 5. Limitations\n")
    lines.append(
        "- Zeroing pixel_values is an out-of-distribution perturbation (model saw real images during training).\n"
        "- Zeroing bbox to (0,0,0,0) is also OOD (all tokens appear at same spatial location).\n"
        "- A properly controlled ablation would require training separate models per condition.\n"
        "- No confidence intervals — single-run results.\n"
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Ablation report saved → %s", out_path)


def plot_ablation_comparison(all_results: List[Dict[str, Any]], out_path: Path) -> None:
    """Publication-quality ablation comparison plot."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    datasets = sorted(set(r.get("dataset", "") for r in all_results if r.get("macro_f1") is not None))
    conditions = [c["short"] for c in ABLATION_CONDITIONS]
    colors = ["#1a4d8c", "#3a7abf", "#7fbfff", "#aaddff"]

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets) + 2, 6), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    fig.suptitle(
        "Phase 6: LayoutLMv3 Modality Sensitivity (Input-Zeroing Ablation)\n"
        r"$\bf{Note}$: Input-zeroing only — NOT an architectural ablation",
        fontsize=12, y=1.02,
    )

    for ax_idx, ds in enumerate(datasets):
        ax = axes[ax_idx]
        ds_results = {r["condition"]: r for r in all_results if r.get("dataset") == ds}

        f1_vals = []
        for c in ABLATION_CONDITIONS:
            r = ds_results.get(c["name"])
            f1_vals.append(r.get("macro_f1") if r else None)

        valid = [(c, f1, col) for c, f1, col in zip(conditions, f1_vals, colors)
                 if f1 is not None]
        if not valid:
            ax.text(0.5, 0.5, "Not evaluated", transform=ax.transAxes, ha="center")
            ax.set_title(ds, fontsize=13, fontweight="bold")
            continue

        conds_v, f1s_v, cols_v = zip(*valid)
        x = np.arange(len(f1s_v))
        bars = ax.bar(x, f1s_v, color=cols_v, edgecolor="black", linewidth=0.8,
                      width=0.6, zorder=3)

        for bar, f1 in zip(bars, f1s_v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{f1:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold"
            )

        ax.set_xticks(x)
        ax.set_xticklabels(conds_v, fontsize=9, ha="center")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Macro F1 (Span)" if ax_idx == 0 else "", fontsize=11)
        ax.set_title(ds, fontsize=13, fontweight="bold")
        ax.yaxis.grid(True, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

        if ds == "CORD":
            ax.text(
                0.5, 0.97, "⚠ Simplified 3-class formulation",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, color="#b04000",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff3cd",
                          edgecolor="#b04000", alpha=0.8)
            )

    legend_elements = [
        mpatches.Patch(facecolor="#1a4d8c", edgecolor="black", label="Full (all modalities)"),
        mpatches.Patch(facecolor="#3a7abf", edgecolor="black", label="No-Image (text+layout)"),
        mpatches.Patch(facecolor="#7fbfff", edgecolor="black", label="No-Layout (text+image)"),
        mpatches.Patch(facecolor="#aaddff", edgecolor="black", label="Text-Approx (text only)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.06), frameon=True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Ablation plot saved → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6: LayoutLMv3 modality ablation study"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["sroie", "funsd", "cord"],
        choices=["sroie", "funsd", "cord"],
        help="Datasets to ablate",
    )
    parser.add_argument(
        "--conditions", nargs="+",
        default=["full", "no-image", "no-layout", "text-approx"],
        choices=["full", "no-image", "no-layout", "text-approx"],
        help="Ablation conditions to run",
    )
    parser.add_argument(
        "--skip-plot", action="store_true",
        help="Skip plot generation"
    )
    args = parser.parse_args()

    # Filter conditions
    cond_map = {
        "full": ABLATION_CONDITIONS[0],
        "no-image": ABLATION_CONDITIONS[1],
        "no-layout": ABLATION_CONDITIONS[2],
        "text-approx": ABLATION_CONDITIONS[3],
    }
    selected_conditions = [cond_map[c] for c in args.conditions]

    logger.info("=" * 60)
    logger.info("PHASE 6: MODALITY ABLATION STUDY")
    logger.info("Datasets: %s", args.datasets)
    logger.info("Conditions: %s", [c["name"] for c in selected_conditions])
    logger.info("=" * 60)
    logger.info(
        "\n*** IMPORTANT: This is an input-zeroing sensitivity study. ***\n"
        "*** NOT an architectural ablation. Results show modality   ***\n"
        "*** sensitivity of the fine-tuned model only.              ***\n"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ds in args.datasets:
        ds_results = run_dataset_ablation(ds, selected_conditions)
        all_results.extend(ds_results)

    # Save results
    logger.info("Writing ablation results…")
    write_ablation_csv(all_results, REPORTS_DIR / "phase6_ablation_results.csv")
    write_ablation_report(all_results, REPORTS_DIR / "phase6_ablation_report.md")

    # Save full results JSON
    json_path = REPORTS_DIR / "phase6_ablation_results_full.json"
    with open(json_path, "w", encoding="utf-8") as f:
        # Convert per_entity dicts (may have non-serializable types) to safe JSON
        safe_results = []
        for r in all_results:
            sr = {k: v for k, v in r.items() if k != "per_entity"}
            sr["per_entity"] = r.get("per_entity", {})
            safe_results.append(sr)
        json.dump(safe_results, f, indent=2, default=str)
    logger.info("Ablation JSON saved → %s", json_path)

    if not args.skip_plot:
        try:
            plot_ablation_comparison(all_results, REPORTS_DIR / "phase6_ablation_comparison.png")
        except Exception as exc:
            logger.warning("Could not generate ablation plot: %s", exc)

    # Summary
    logger.info("=" * 60)
    logger.info("ABLATION COMPLETE")
    logger.info("Results summary:")
    for r in all_results:
        mf1 = r.get("macro_f1")
        f1_str = f"{mf1:.4f}" if mf1 is not None else "N/A"
        logger.info(
            "  %-8s | %-35s | Macro F1 = %s",
            r.get("dataset"), r.get("condition"), f1_str
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
