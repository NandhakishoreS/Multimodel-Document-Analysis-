"""
Visualization engine for document predictions.
Renders bounding boxes, class labels, and confidence tags directly onto original images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from inference.schemas import ExtractedEntity


# Distinctive high-contrast palette for entity classes across SROIE, FUNSD, and CORD
COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    # SROIE classes
    "COMPANY": (37, 99, 235),       # Vibrant Royal Blue
    "DATE": (16, 185, 129),         # Emerald Green
    "ADDRESS": (245, 158, 11),      # Warm Amber
    "TOTAL": (239, 68, 68),         # Ruby Red
    # FUNSD classes
    "HEADER": (147, 51, 234),       # Vivid Purple
    "QUESTION": (6, 182, 212),      # Modern Cyan
    "ANSWER": (16, 185, 129),       # Emerald Green
    # CORD classes
    "MENU": (59, 130, 246),         # Soft Blue
    "SUBTOTAL": (249, 115, 22),     # Orange
    # Default fallback
    "DEFAULT": (107, 114, 128),     # Slate Grey
}


def get_label_color(label: str) -> Tuple[int, int, int]:
    """Retrieve consistent color tuple for an entity label."""
    clean_label = label.upper().replace("B-", "").replace("I-", "")
    return COLOR_PALETTE.get(clean_label, COLOR_PALETTE["DEFAULT"])


def draw_predictions_on_image(
    image: Union[Image.Image, str, Path],
    entities: List[ExtractedEntity],
    min_confidence: float = 0.0,
    line_width: int = 3,
    show_confidence: bool = True,
    fill_alpha: int = 45,
) -> Image.Image:
    """
    Render bounding boxes and label tags for detected entities onto the document image.

    Parameters
    ----------
    image : PIL.Image or filepath
        Input document image.
    entities : List[ExtractedEntity]
        Detected entity spans with bounding boxes and confidence scores.
    min_confidence : float
        Filter out entities below this confidence threshold.
    line_width : int
        Thickness of bounding box borders.
    show_confidence : bool
        Whether to append confidence percentage to the label banner.
    fill_alpha : int
        Transparency (0-255) for semi-transparent box highlights.

    Returns
    -------
    annotated_image : PIL.Image in RGB mode
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGBA")
    else:
        img = image.convert("RGBA")

    # Create overlay layer for translucent bounding box fills
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(img)

    # Attempt to load a default TrueType font, fallback to standard bitmap font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for entity in entities:
        if entity.confidence < min_confidence:
            continue

        x0, y0, x1, y1 = entity.bbox
        # Ensure coordinates are within image bounds
        x0 = max(0, min(x0, img.width - 1))
        y0 = max(0, min(y0, img.height - 1))
        x1 = max(x0 + 1, min(x1, img.width))
        y1 = max(y0 + 1, min(y1, img.height))

        color = get_label_color(entity.label)
        fill_color = (*color, fill_alpha)

        # Draw translucent fill
        draw_overlay.rectangle([x0, y0, x1, y1], fill=fill_color)

        # Draw solid bounding rectangle
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)

        # Render label banner text
        if show_confidence:
            banner_text = f"{entity.label} ({entity.confidence:.2f})"
        else:
            banner_text = entity.label

        # Compute text bounding box
        bbox_text = draw.textbbox((x0, y0), banner_text, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]

        # Draw banner background tag
        banner_y0 = max(0, y0 - text_h - 4)
        banner_y1 = y0
        banner_x1 = min(img.width, x0 + text_w + 8)

        draw.rectangle([x0, banner_y0, banner_x1, banner_y1], fill=color)
        draw.text((x0 + 4, banner_y0 + 1), banner_text, fill=(255, 255, 255), font=font)

    # Alpha composite translucent fill with the image
    final_img = Image.alpha_composite(img, overlay).convert("RGB")
    return final_img


def draw_raw_ocr_boxes(
    image: Union[Image.Image, str, Path],
    ocr_results: List[Any],
    highlight_tokens: Optional[List[str]] = None,
    line_width: int = 2,
    fill_alpha: int = 40,
) -> Image.Image:
    """
    Render all raw OCR detected bounding boxes, text, and confidence onto the document image.
    Highlights specific tokens (such as '1', '2', '3', '4') in vivid emerald green for visual debugging.
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGBA")
    else:
        img = image.convert("RGBA")

    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if highlight_tokens is None:
        highlight_tokens = ["1", "2", "3", "4"]

    default_color = (156, 163, 175)     # Slate gray
    highlight_color = (16, 185, 129)    # Emerald green

    for item in ocr_results:
        poly, text, conf = item
        text_str = str(text).strip()
        poly_np = np.array(poly)
        x0 = int(np.min(poly_np[:, 0]))
        y0 = int(np.min(poly_np[:, 1]))
        x1 = int(np.max(poly_np[:, 0]))
        y1 = int(np.max(poly_np[:, 1]))

        x0 = max(0, min(x0, img.width - 1))
        y0 = max(0, min(y0, img.height - 1))
        x1 = max(x0 + 1, min(x1, img.width))
        y1 = max(y0 + 1, min(y1, img.height))

        words = text_str.split()
        is_highlight = any(h in words for h in highlight_tokens) or any(text_str.startswith(h) for h in highlight_tokens)

        color = highlight_color if is_highlight else default_color
        box_w = line_width + 1 if is_highlight else line_width
        fill_color = (*color, fill_alpha + 30 if is_highlight else fill_alpha)

        draw_overlay.rectangle([x0, y0, x1, y1], fill=fill_color)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=box_w)

        banner_text = f"{text_str} ({conf:.2f})"
        bbox_text = draw.textbbox((x0, y0), banner_text, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]

        banner_y0 = max(0, y0 - text_h - 4)
        banner_y1 = y0
        banner_x1 = min(img.width, x0 + text_w + 6)

        draw.rectangle([x0, banner_y0, banner_x1, banner_y1], fill=color)
        text_color = (0, 0, 0) if is_highlight else (255, 255, 255)
        draw.text((x0 + 3, banner_y0 + 1), banner_text, fill=text_color, font=font)

    return Image.alpha_composite(img, overlay).convert("RGB")

