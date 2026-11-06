"""
Phase 5 visualization: generate all comparison and diagnostic plots.
Reads from reports/model_comparison.csv and experiment training summaries.

Run from project root:
    python scripts/generate_plots.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS  = PROJECT_ROOT / "reports"
FIG_DIR  = REPORTS / "figures"
EXP_DIR  = PROJECT_ROOT / "experiments"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "OCR Baseline":      "#adb5bd",
    "Text-Only":         "#4895ef",
    "Text+Layout":       "#4cc9f0",
    "LayoutLMv3":        "#f72585",
    "LayoutLMv3-base":   "#f72585",
}

def _color(model_type: str) -> str:
    for k, v in COLORS.items():
        if k.lower() in model_type.lower():
            return v
    return "#888888"


# ─── 1. Model comparison bar chart ────────────────────────────────────────────
def plot_model_comparison() -> None:
    csv = REPORTS / "model_comparison.csv"
    if not csv.exists():
        print("model_comparison.csv not found — skipping comparison plot")
        return

    df = pd.read_csv(csv)
    df = df[df["split"] == "test"].dropna(subset=["macro_f1"])
    datasets = df["dataset"].str.upper().unique()

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 6), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"].str.upper() == ds].copy()
        sub = sub.sort_values("macro_f1")
        colors = [_color(m) for m in sub["model_type"]]
        bars = ax.barh(sub["model_type"], sub["macro_f1"], color=colors, edgecolor="white", height=0.6)
        for bar, val in zip(bars, sub["macro_f1"]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)
        ax.set_title(ds, fontweight="bold")
        ax.set_xlabel("Macro F1 (seqeval)")
        ax.set_xlim(0, 1.05)
        ax.axvline(0, color="black", linewidth=0.5)

    fig.suptitle("Model Comparison — Test Set (Macro F1)", fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─── 2. Per-class F1 heatmap for each dataset ─────────────────────────────────
def plot_per_class_f1() -> None:
    for ds in ["sroie", "funsd", "cord"]:
        json_path = REPORTS / f"layoutlmv3_{ds}_per_entity.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            data = json.load(f)

        per_entity = data.get("seqeval", {}).get("per_entity", {})
        if not per_entity:
            per_entity = data.get("token_level", {}).get("per_class", {})
        if not per_entity:
            continue

        labels = list(per_entity.keys())
        metrics = ["precision", "recall", "f1"]
        matrix = np.array([[per_entity[l].get(m, 0) for m in metrics] for l in labels])

        fig, ax = plt.subplots(figsize=(5, max(3, len(labels) * 0.5 + 1)))
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Precision", "Recall", "F1"], fontsize=9)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(3):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        ax.set_title(f"LayoutLMv3 Per-Class Metrics — {ds.upper()}", fontweight="bold")
        fig.tight_layout()
        out = FIG_DIR / f"per_class_f1_{ds}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# ─── 3. Training curves ────────────────────────────────────────────────────────
def plot_training_curves() -> None:
    for exp_dir in sorted(EXP_DIR.glob("layoutlmv3_*_full_*")):
        summary_path = exp_dir / "training_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)

        history = summary.get("epoch_history", [])
        if not history:
            continue

        ds    = summary.get("dataset", "unknown")
        stage = summary.get("stage", "")
        epochs      = [h["epoch"] for h in history]
        train_loss  = [h.get("train_loss", 0) for h in history]
        val_f1      = [h.get("val_metrics", {}).get("macro_f1", 0) for h in history]
        val_loss    = [h.get("val_metrics", {}).get("avg_loss", 0) for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(epochs, train_loss, "o-", color="#4895ef", label="Train Loss")
        ax1.plot(epochs, val_loss,   "s--", color="#f72585", label="Val Loss")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title(f"{ds.upper()} — Loss Curves"); ax1.legend(); ax1.grid(alpha=0.3)

        ax2.plot(epochs, val_f1, "o-", color="#f72585", label="Val Macro F1")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro F1")
        ax2.set_title(f"{ds.upper()} — Validation F1"); ax2.legend(); ax2.grid(alpha=0.3)
        ax2.set_ylim(0, 1.0)

        fig.suptitle(f"LayoutLMv3 Training — {ds.upper()} [{stage}]", fontweight="bold")
        fig.tight_layout()
        out = FIG_DIR / f"training_curves_{ds}_{stage}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# ─── 4. Label distribution comparison ─────────────────────────────────────────
def plot_label_distributions() -> None:
    import json as _json
    for ds in ["sroie", "funsd", "cord"]:
        report = REPORTS / f"{ds}_report.json"
        if not report.exists():
            continue
        with open(report) as f:
            data = _json.load(f)
        label_dist = data.get("label_distribution") or data.get("label_counts") or {}
        if not label_dist:
            continue

        labels = list(label_dist.keys())
        counts = [label_dist[l] for l in labels]
        total  = sum(counts)
        pcts   = [100 * c / total for c in counts]

        fig, ax = plt.subplots(figsize=(max(6, len(labels)), 4))
        bars = ax.bar(labels, pcts, color="#4895ef", edgecolor="white")
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{pct:.1f}%", ha="center", fontsize=8)
        ax.set_ylabel("Percentage (%)")
        ax.set_title(f"{ds.upper()} — Label Distribution")
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = FIG_DIR / f"label_dist_{ds}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating Phase 5 plots...")
    plot_model_comparison()
    plot_per_class_f1()
    plot_training_curves()
    plot_label_distributions()
    print("Done. All plots saved to", FIG_DIR)
