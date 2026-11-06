"""
Generate publication-quality figures for Phase 10:
1. End-to-End System Architecture Diagram (reports/phase10_architecture.png)
2. Final cross-dataset performance
3. Phase 8 vs Phase 9 robustness comparison
4. Modality ablation study
5. SROIE entity-level recovery
6. Error distribution summary
Saved under reports/phase10_figures/
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# Styling defaults
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cbd5e1"
plt.rcParams["axes.linewidth"] = 0.8

output_dir = Path("reports/phase10_figures")
output_dir.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Figure 0: High-Resolution Architecture Diagram
# ==============================================================================
def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def draw_box(x, y, w, h, text, subtext="", color="#e0f2fe", edge="#0284c7", text_color="#0f172a"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            facecolor=color, edgecolor=edge, linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.15 if subtext else 0), text,
                ha="center", va="center", fontsize=10.5, fontweight="bold", color=text_color)
        if subtext:
            ax.text(x + w / 2, y + h / 2 - 0.22, subtext,
                    ha="center", va="center", fontsize=8.5, color="#475569")

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#334155", lw=1.5),
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label,
                    ha="center", va="center", fontsize=8, color="#64748b")

    # Column 1: Input & OCR
    draw_box(0.5, 3.2, 2.0, 1.4, "Document Image", "JPG / PNG / Scanned", "#f8fafc", "#94a3b8")
    draw_arrow(2.6, 3.9, 3.2, 3.9)

    draw_box(3.2, 3.2, 2.0, 1.4, "OCR Engine", "RapidOCR (ONNX)", "#e0e7ff", "#4f46e5")
    draw_arrow(5.3, 3.9, 5.9, 3.9)

    # Column 2: Multimodal Representation Branches
    draw_box(5.9, 5.5, 2.2, 1.2, "1. Text Tokens", "Byte-Pair Encoding", "#ecfdf5", "#059669")
    draw_box(5.9, 3.4, 2.2, 1.2, "2. 2D Layout", "BBoxes [0, 1000]", "#fef3c7", "#d97706")
    draw_box(5.9, 1.3, 2.2, 1.2, "3. Visual Patches", "14x14 Patches (16x16)", "#fce7f3", "#db2777")

    # Fan out arrows from OCR
    draw_arrow(5.2, 4.4, 5.9, 6.1)
    draw_arrow(5.2, 3.9, 5.9, 4.0)
    draw_arrow(2.5, 3.2, 5.9, 1.9, label="Raw Pixels")

    # Column 3: Multimodal Transformer (LayoutLMv3)
    draw_box(8.7, 2.2, 2.2, 3.6, "LayoutLMv3", "12 Transformer Layers\nEarly Multimodal Fusion\n125M Parameters", "#eff6ff", "#2563eb")
    draw_arrow(8.1, 6.1, 8.7, 4.8)
    draw_arrow(8.1, 4.0, 8.7, 4.0)
    draw_arrow(8.1, 1.9, 8.7, 3.2)

    # Column 4: Token Classification & BIO Decoding
    draw_box(11.4, 4.8, 2.2, 1.4, "Token Classification", "Softmax Probabilities\n+ Argmax BIO Labels", "#f1f5f9", "#475569")
    draw_arrow(10.9, 4.5, 11.4, 5.2)

    draw_box(11.4, 2.6, 2.2, 1.6, "Entity Aggregation\n& Confidence", "BIO Span Merging\nMean Probability\nBBox Union", "#e0f2fe", "#0284c7")
    draw_arrow(12.5, 4.8, 12.5, 4.2)

    # Column 5: Output Destinations
    draw_box(11.4, 0.4, 2.2, 1.6, "Structured Output", "Streamlit UI\nBBox Visualizations\nFormatted JSON", "#f0fdf4", "#16a34a")
    draw_arrow(12.5, 2.6, 12.5, 2.0)

    plt.title("Phase 10 End-to-End System Architecture: Multimodal Intelligent Document Understanding",
              fontsize=13, fontweight="bold", pad=15, color="#1e293b")
    plt.tight_layout()
    plt.savefig("reports/phase10_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved reports/phase10_architecture.png")


# ==============================================================================
# Figure 1: Final Cross-Dataset Performance
# ==============================================================================
def generate_cross_dataset_figure():
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    datasets = ["SROIE (Corrected)", "FUNSD", "CORD (Simplified 3-Class)"]
    precision = [0.8689, 0.7804, 0.9748]
    recall = [0.8889, 0.8196, 0.9763]
    macro_f1 = [0.8785, 0.7179, 0.9733]

    x = np.arange(len(datasets))
    width = 0.25

    rects1 = ax.bar(x - width, precision, width, label="Precision", color="#3b82f6", edgecolor="none")
    rects2 = ax.bar(x, recall, width, label="Recall", color="#10b981", edgecolor="none")
    rects3 = ax.bar(x + width, macro_f1, width, label="Macro F1", color="#6366f1", edgecolor="none")

    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("Final Validated Cross-Dataset Evaluation Metrics (Seqeval Entity-Level)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10, fontweight="bold")
    ax.set_ylim(0.5, 1.05)
    ax.legend(frameon=True, facecolor="#f8fafc", loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for rects in [rects1, rects2, rects3]:
        for bar in rects:
            height = bar.get_height()
            ax.annotate(f"{height:.4f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "1_cross_dataset_performance.png", dpi=300)
    plt.close()
    print("Saved 1_cross_dataset_performance.png")


# ==============================================================================
# Figure 2: Phase 8 vs Phase 9 Robustness Comparison
# ==============================================================================
def generate_robustness_figure():
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    mc = pd.read_csv("reports/phase9_model_comparison.csv")
    models = mc["Model"].tolist()
    clean_f1 = mc["Clean_Macro_F1"].tolist()
    avg_f1 = mc["Avg_Stressed_Macro_F1"].tolist()
    worst_f1 = mc["Worst_Case_Macro_F1"].tolist()

    x = np.arange(len(models))
    width = 0.25

    ax.bar(x - width, clean_f1, width, label="Clean Macro F1", color="#0284c7")
    ax.bar(x, avg_f1, width, label="Avg Stressed F1 (14 Conditions)", color="#0d9488")
    ax.bar(x + width, worst_f1, width, label="Worst-Case Floor F1", color="#e11d48")

    ax.set_ylabel("Macro F1", fontsize=11, fontweight="bold")
    ax.set_title("Phase 8 vs Phase 9 Robustness Benchmark Comparison (SROIE)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=True, facecolor="#f8fafc", loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_dir / "2_phase8_vs_phase9_robustness.png", dpi=300)
    plt.close()
    print("Saved 2_phase8_vs_phase9_robustness.png")


# ==============================================================================
# Figure 3: Modality Ablation Contributions
# ==============================================================================
def generate_ablation_figure():
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    df = pd.read_csv("reports/phase9_ablation_results.csv")
    cond_order = ["Full", "No-Image", "Text-Only", "No-Layout", "No-Text"]
    
    sroie_vals = [df[(df["Dataset"]=="SROIE") & (df["Short"]==c)]["Macro_F1"].values[0] for c in cond_order]
    funsd_vals = [df[(df["Dataset"]=="FUNSD") & (df["Short"]==c)]["Macro_F1"].values[0] for c in cond_order]
    cord_vals = [df[(df["Dataset"]=="CORD") & (df["Short"]==c)]["Macro_F1"].values[0] for c in cond_order]

    x = np.arange(len(cond_order))
    width = 0.25

    ax.bar(x - width, sroie_vals, width, label="SROIE", color="#3b82f6")
    ax.bar(x, funsd_vals, width, label="FUNSD", color="#f59e0b")
    ax.bar(x + width, cord_vals, width, label="CORD (Simplified)", color="#10b981")

    ax.set_ylabel("Macro F1", fontsize=11, fontweight="bold")
    ax.set_title("Multimodal Ablation Study: Impact of Stripping Modalities", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["Full Multimodal", "No-Image\n(Text+Layout)", "Text Only", "No-Layout\n(Text+Image)", "No-Text\n(Layout+Image)"], fontsize=9.5, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True, facecolor="#f8fafc", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_dir / "3_modality_ablation_contributions.png", dpi=300)
    plt.close()
    print("Saved 3_modality_ablation_contributions.png")


# ==============================================================================
# Figure 4: SROIE Entity Recovery
# ==============================================================================
def generate_entity_figure():
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    entities = ["COMPANY", "ADDRESS", "TOTAL", "DATE"]
    baseline_jitter = [0.5000, 0.6698, 0.5367, 0.9798]
    joint_aug_jitter = [0.8718, 0.8519, 0.7812, 0.9930]

    x = np.arange(len(entities))
    width = 0.35

    ax.bar(x - width/2, baseline_jitter, width, label="Baseline (Phase 8)", color="#ef4444")
    ax.bar(x + width/2, joint_aug_jitter, width, label="Joint-Augmented (Phase 9)", color="#10b981")

    ax.set_ylabel("Entity F1 Score", fontsize=11, fontweight="bold")
    ax.set_title("Entity Recovery Under Extreme Layout Jitter (±10%)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(entities, fontsize=10, fontweight="bold")
    ax.set_ylim(0.3, 1.05)
    ax.legend(frameon=True, facecolor="#f8fafc", loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for i in range(len(entities)):
        diff = joint_aug_jitter[i] - baseline_jitter[i]
        ax.annotate(f"+{diff*100:.1f}%",
                    xy=(x[i] + width/2, joint_aug_jitter[i]),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#065f46")

    plt.tight_layout()
    plt.savefig(output_dir / "4_sroie_entity_robustness.png", dpi=300)
    plt.close()
    print("Saved 4_sroie_entity_robustness.png")


# ==============================================================================
# Figure 5: Phase 7 Error Taxonomy Distribution
# ==============================================================================
def generate_error_distribution_figure():
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    categories = [
        "Boundary Fragmentation\n(Multi-line Address/Company)",
        "Numeric Class Confusion\n(Total vs Subtotal/Cash)",
        "Spatial Distortion\n(Skewed / Multi-column)",
        "OCR Character Noise\n(Misrecognized Symbols)",
        "Other / Background Leakage"
    ]
    proportions = [38.5, 24.2, 18.0, 12.3, 7.0]
    colors = ["#f87171", "#fb923c", "#fbbf24", "#38bdf8", "#94a3b8"]

    bars = ax.barh(categories[::-1], proportions[::-1], color=colors[::-1], edgecolor="none", height=0.6)
    ax.set_xlabel("Proportion of Total Errors (%)", fontsize=11, fontweight="bold")
    ax.set_title("Phase 7 Error Taxonomy Breakdown across Document Failures", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0, 45)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.1f}%",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "5_error_taxonomy_distribution.png", dpi=300)
    plt.close()
    print("Saved 5_error_taxonomy_distribution.png")


if __name__ == "__main__":
    generate_architecture_diagram()
    generate_cross_dataset_figure()
    generate_robustness_figure()
    generate_ablation_figure()
    generate_entity_figure()
    generate_error_distribution_figure()
    print("All Phase 10 figures generated successfully.")
