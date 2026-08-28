"""Visualization utilities for document images, bounding boxes, OCR words, and labels."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Curated high-contrast aesthetic color palette for labels
LABEL_COLOR_PALETTE = {
    # FUNSD labels
    "HEADER": "#E63946",       # Coral Red
    "B-HEADER": "#E63946",
    "I-HEADER": "#F1A7A9",
    "QUESTION": "#457B9D",     # Steel Blue
    "B-QUESTION": "#457B9D",
    "I-QUESTION": "#A8DADC",
    "ANSWER": "#2A9D8F",       # Persian Green
    "B-ANSWER": "#2A9D8F",
    "I-ANSWER": "#8AB17D",
    # SROIE labels
    "COMPANY": "#9B5DE5",      # Purple
    "B-COMPANY": "#9B5DE5",
    "I-COMPANY": "#CDB4DB",
    "DATE": "#F15BB5",         # Pink
    "B-DATE": "#F15BB5",
    "I-DATE": "#FFC8DD",
    "ADDRESS": "#F77F00",      # Deep Orange
    "B-ADDRESS": "#F77F00",
    "I-ADDRESS": "#FDCB6E",
    "TOTAL": "#00BBF9",        # Electric Blue
    "B-TOTAL": "#00BBF9",
    "I-TOTAL": "#A0E7E5",
    # CORD labels
    "MENU": "#00F5D4",         # Aquamarine
    "B-MENU": "#00F5D4",
    "I-MENU": "#B9FBC0",
    "SUBTOTAL": "#E0AAFF",     # Mauve
    "B-SUBTOTAL": "#E0AAFF",
    "I-SUBTOTAL": "#F1EAFF",
    "VOID": "#D00000",         # Dark Red
    "B-VOID": "#D00000",
    "I-VOID": "#FF758F",
    "PAYMENT": "#3A86FF",      # Royal Blue
    "B-PAYMENT": "#3A86FF",
    "I-PAYMENT": "#8338EC",
    # Generic / Default
    "OTHER": "#6C757D",        # Neutral Slate Grey
    "O": "#6C757D",
}

DEFAULT_COLOR = "#3F37C9"


def get_color_for_label(label: Optional[str]) -> str:
    """Retrieve consistent color code for a given label string."""
    if not label:
        return DEFAULT_COLOR
    clean_label = str(label).strip().upper()
    return LABEL_COLOR_PALETTE.get(clean_label, DEFAULT_COLOR)


def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    """Convert hex string '#RRGGBB' to RGB tuple."""
    hex_clean = hex_code.lstrip("#")
    return tuple(int(hex_clean[i : i + 2], 16) for i in (0, 2, 4))


def visualize_document(
    image: Union[str, Path, Image.Image],
    words: Optional[List[str]] = None,
    bboxes: Optional[List[List[Union[int, float]]]] = None,
    labels: Optional[List[str]] = None,
    relations: Optional[List[Dict[str, int]]] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    normalized: bool = False,
    figsize: Tuple[int, int] = (10, 14),
    dpi: int = 150,
) -> plt.Figure:
    """
    Render document image with overlaid bounding boxes, text tokens, and semantic labels.

    Args:
        image: PIL Image instance or file path to document image.
        words: List of OCR word strings.
        bboxes: List of bounding boxes [x0, y0, x1, y1].
        labels: List of label strings matching words/bboxes.
        relations: List of relation dictionaries (e.g. {'from_id': 0, 'to_id': 1}).
        title: Title for the matplotlib figure.
        save_path: Path to save the resulting visualization.
        normalized: Whether bboxes are in [0, 1000] space (will be un-normalized to image resolution).
        figsize: Figure dimension in inches.
        dpi: Output resolution.

    Returns:
        matplotlib.figure.Figure instance.
    """
    if isinstance(image, (str, Path)):
        img_obj = Image.open(image).convert("RGB")
    else:
        img_obj = image.convert("RGB")

    img_w, img_h = img_obj.size

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(img_obj)
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    bboxes = bboxes or []
    words = words or []
    labels = labels or []

    # Map boxes from [0, 1000] if normalized
    mapped_boxes = []
    for box in bboxes:
        if normalized:
            x0 = (box[0] / 1000.0) * img_w
            y0 = (box[1] / 1000.0) * img_h
            x1 = (box[2] / 1000.0) * img_w
            y1 = (box[3] / 1000.0) * img_h
        else:
            x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
        mapped_boxes.append((x0, y0, x1, y1))

    # Draw boxes
    for idx, (x0, y0, x1, y1) in enumerate(mapped_boxes):
        w = max(1.0, x1 - x0)
        h = max(1.0, y1 - y0)
        lbl = labels[idx] if idx < len(labels) else "O"
        color = get_color_for_label(lbl)

        # Bounding box rectangle
        rect = patches.Rectangle(
            (x0, y0),
            w,
            h,
            linewidth=1.8,
            edgecolor=color,
            facecolor=color,
            alpha=0.25,
        )
        ax.add_patch(rect)

        # Border
        rect_border = patches.Rectangle(
            (x0, y0),
            w,
            h,
            linewidth=1.5,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect_border)

        # Annotate text & label
        if lbl and str(lbl).upper() not in ("O", "OTHER"):
            tag_text = str(lbl).replace("B-", "").replace("I-", "")
            ax.text(
                x0,
                max(0, y0 - 3),
                tag_text,
                fontsize=7,
                fontweight="bold",
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.85,
                ),
            )

    # Draw relations if provided
    if relations:
        for rel in relations:
            from_idx = rel.get("from_id", rel.get("source"))
            to_idx = rel.get("to_id", rel.get("target"))
            if (
                from_idx is not None
                and to_idx is not None
                and from_idx < len(mapped_boxes)
                and to_idx < len(mapped_boxes)
            ):
                b1 = mapped_boxes[from_idx]
                b2 = mapped_boxes[to_idx]
                c1_x = (b1[0] + b1[2]) / 2.0
                c1_y = (b1[1] + b1[3]) / 2.0
                c2_x = (b2[0] + b2[2]) / 2.0
                c2_y = (b2[1] + b2[3]) / 2.0
                ax.annotate(
                    "",
                    xy=(c2_x, c2_y),
                    xytext=(c1_x, c1_y),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#D62828",
                        lw=1.5,
                        mutation_scale=12,
                    ),
                )

    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight", dpi=dpi)
        plt.close(fig)

    return fig


def plot_label_distribution(
    counts: Dict[str, int],
    title: str = "Label Distribution",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot bar chart of label frequency distribution."""
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = [get_color_for_label(lbl) for lbl in labels]

    bars = ax.bar(labels, values, color=colors, edgecolor="#2B2D42", linewidth=1.2, alpha=0.85)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax.set_xlabel("Semantic Category", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_f1_comparison(
    results: Dict[str, Dict[str, float]],
    title: str = "Baseline F1-Score Comparison",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot grouped bar chart comparing Text-Only vs Text+Layout F1 scores.

    Args:
        results: Dict of {dataset_name: {'Text-Only': f1, 'Text+Layout': f1}}
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    datasets = list(results.keys())
    x = np.arange(len(datasets))
    width = 0.35

    text_f1s = [results[d].get("Text-Only", 0.0) for d in datasets]
    layout_f1s = [results[d].get("Text+Layout", 0.0) for d in datasets]

    rects1 = ax.bar(x - width / 2, text_f1s, width, label="Text-Only Baseline", color="#457B9D", edgecolor="#1D3557")
    rects2 = ax.bar(x + width / 2, layout_f1s, width, label="Text+Layout Baseline", color="#2A9D8F", edgecolor="#264653")

    ax.set_ylabel("Macro F1-Score", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True, facecolor="#F8F9FA")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight")
        plt.close(fig)

    return fig
