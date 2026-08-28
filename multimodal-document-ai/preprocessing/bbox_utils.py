"""Bounding box geometry, normalization, validation, and coordinate transformation utilities."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


def validate_bbox(
    bbox: Sequence[Union[int, float]],
    width: Optional[int] = None,
    height: Optional[int] = None,
    clip_to_bounds: bool = True,
) -> List[int]:
    """
    Validate and ensure standard format [x0, y0, x1, y1] with x0 <= x1 and y0 <= y1.

    Args:
        bbox: Bounding box coordinates [x0, y0, x1, y1].
        width: Image width if clipping to boundaries.
        height: Image height if clipping to boundaries.
        clip_to_bounds: Whether to clamp coordinates within [0, width] and [0, height].

    Returns:
        Validated bounding box as list of integers [x0, y0, x1, y1].
    """
    if len(bbox) != 4:
        raise ValueError(f"Expected 4 coordinate values [x0, y0, x1, y1], got {len(bbox)}: {bbox}")

    x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    # Ensure x0 <= x1 and y0 <= y1
    min_x, max_x = min(x0, x1), max(x0, x1)
    min_y, max_y = min(y0, y1), max(y0, y1)

    if clip_to_bounds:
        min_x = max(0.0, min_x)
        min_y = max(0.0, min_y)
        if width is not None and width > 0:
            min_x = min(min_x, float(width))
            max_x = min(max(min_x, max_x), float(width))
        if height is not None and height > 0:
            min_y = min(min_y, float(height))
            max_y = min(max(min_y, max_y), float(height))

    # Ensure at least 1 pixel width/height to avoid zero-area degenerate bboxes
    if max_x <= min_x:
        max_x = min_x + 1.0
    if max_y <= min_y:
        max_y = min_y + 1.0

    return [int(round(min_x)), int(round(min_y)), int(round(max_x)), int(round(max_y))]


def normalize_bbox(
    bbox: Sequence[Union[int, float]],
    width: int,
    height: int,
    target_range: Tuple[int, int] = (0, 1000),
) -> List[int]:
    """
    Normalize pixel bounding box [x0, y0, x1, y1] to target range (default [0, 1000]) for document transformers.

    Args:
        bbox: Pixel bounding box [x0, y0, x1, y1].
        width: Document image width in pixels.
        height: Document image height in pixels.
        target_range: Target scale range, default (0, 1000).

    Returns:
        Normalized bounding box [nx0, ny0, nx1, ny1] in [0, 1000].
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: width={width}, height={height}")

    validated = validate_bbox(bbox, width=width, height=height, clip_to_bounds=True)
    x0, y0, x1, y1 = validated

    t_min, t_max = target_range
    scale_range = t_max - t_min

    nx0 = int(np.clip(round((x0 / width) * scale_range + t_min), t_min, t_max))
    ny0 = int(np.clip(round((y0 / height) * scale_range + t_min), t_min, t_max))
    nx1 = int(np.clip(round((x1 / width) * scale_range + t_min), t_min, t_max))
    ny1 = int(np.clip(round((y1 / height) * scale_range + t_min), t_min, t_max))

    # Ensure valid ordering and non-zero dimension in normalized space
    if nx1 <= nx0:
        nx1 = min(t_max, nx0 + 1)
    if ny1 <= ny0:
        ny1 = min(t_max, ny0 + 1)

    return [nx0, ny0, nx1, ny1]


def denormalize_bbox(
    bbox: Sequence[Union[int, float]],
    width: int,
    height: int,
    source_range: Tuple[int, int] = (0, 1000),
) -> List[int]:
    """
    Convert normalized bounding box back to original pixel coordinates.

    Args:
        bbox: Normalized bounding box [nx0, ny0, nx1, ny1].
        width: Original document image width.
        height: Original document image height.
        source_range: Normalization scale range, default (0, 1000).

    Returns:
        Denormalized pixel coordinates [x0, y0, x1, y1].
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: width={width}, height={height}")

    nx0, ny0, nx1, ny1 = bbox[0], bbox[1], bbox[2], bbox[3]
    s_min, s_max = source_range
    scale = float(s_max - s_min)

    x0 = int(round(((nx0 - s_min) / scale) * width))
    y0 = int(round(((ny0 - s_min) / scale) * height))
    x1 = int(round(((nx1 - s_min) / scale) * width))
    y1 = int(round(((ny1 - s_min) / scale) * height))

    return validate_bbox([x0, y0, x1, y1], width=width, height=height, clip_to_bounds=True)


def quad_to_bbox(quad: Union[Sequence[Union[int, float]], Dict[str, Any]]) -> List[int]:
    """
    Convert polygon / 4-point quadrilateral coordinate representation to axis-aligned bounding box [x0, y0, x1, y1].

    Supports:
        - 8-number list: [x1, y1, x2, y2, x3, y3, x4, y4]
        - List of 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        - Dict with keys like {'x1': ..., 'y1': ..., 'x2': ..., ...}
        - Dict with keys {'quad': {'x1': ..., ...}}
    """
    if isinstance(quad, dict):
        if "quad" in quad and isinstance(quad["quad"], dict):
            quad = quad["quad"]
        if "x1" in quad:
            xs = [quad.get(f"x{i}", 0) for i in range(1, 5)]
            ys = [quad.get(f"y{i}", 0) for i in range(1, 5)]
            return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        elif "box" in quad:
            return validate_bbox(quad["box"])

    if isinstance(quad, (list, tuple)):
        if len(quad) == 4 and all(isinstance(p, (int, float)) for p in quad):
            # Already 4 numbers [x0, y0, x1, y1]
            return validate_bbox(quad)
        elif len(quad) == 8 and all(isinstance(p, (int, float)) for p in quad):
            # 8 coordinates
            xs = [quad[0], quad[2], quad[4], quad[6]]
            ys = [quad[1], quad[3], quad[5], quad[7]]
            return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        elif len(quad) == 4 and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in quad):
            # List of points
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

    raise ValueError(f"Unsupported quad coordinate format: {quad}")


def compute_iou(box1: Sequence[Union[int, float]], box2: Sequence[Union[int, float]]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    inter_x0 = max(x0_1, x0_2)
    inter_y0 = max(y0_1, y0_2)
    inter_x1 = min(x1_1, x1_2)
    inter_y1 = min(y1_1, y1_2)

    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x1_1 - x0_1) * max(0.0, y1_1 - y0_1)
    area2 = max(0.0, x1_2 - x0_2) * max(0.0, y1_2 - y0_2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def extract_spatial_features(
    bbox: Sequence[Union[int, float]],
    prev_bbox: Optional[Sequence[Union[int, float]]] = None,
    next_bbox: Optional[Sequence[Union[int, float]]] = None,
    target_scale: float = 1000.0,
) -> Dict[str, float]:
    """
    Extract geometric and relative spatial layout features for a normalized bounding box.

    Returns dictionary of numerical spatial features suitable for tabular/layout classifiers:
        - normalized coordinates: x0, y0, x1, y1
        - center coordinates: cx, cy
        - dimension: width, height, area, aspect_ratio
        - relative position to prev/next tokens (horizontal & vertical deltas)
    """
    x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    area = w * h

    features = {
        "norm_x0": x0 / target_scale,
        "norm_y0": y0 / target_scale,
        "norm_x1": x1 / target_scale,
        "norm_y1": y1 / target_scale,
        "norm_cx": cx / target_scale,
        "norm_cy": cy / target_scale,
        "norm_width": w / target_scale,
        "norm_height": h / target_scale,
        "norm_area": area / (target_scale * target_scale),
        "aspect_ratio": w / h,
        "is_top_third": 1.0 if cy < (target_scale / 3.0) else 0.0,
        "is_bottom_third": 1.0 if cy > (2 * target_scale / 3.0) else 0.0,
        "is_left_half": 1.0 if cx < (target_scale / 2.0) else 0.0,
    }

    if prev_bbox is not None:
        p_cx = (prev_bbox[0] + prev_bbox[2]) / 2.0
        p_cy = (prev_bbox[1] + prev_bbox[3]) / 2.0
        features["delta_prev_cx"] = (cx - p_cx) / target_scale
        features["delta_prev_cy"] = (cy - p_cy) / target_scale
        features["same_line_prev"] = 1.0 if abs(cy - p_cy) < (h * 0.7) else 0.0
    else:
        features["delta_prev_cx"] = 0.0
        features["delta_prev_cy"] = 0.0
        features["same_line_prev"] = 0.0

    if next_bbox is not None:
        n_cx = (next_bbox[0] + next_bbox[2]) / 2.0
        n_cy = (next_bbox[1] + next_bbox[3]) / 2.0
        features["delta_next_cx"] = (n_cx - cx) / target_scale
        features["delta_next_cy"] = (n_cy - cy) / target_scale
        features["same_line_next"] = 1.0 if abs(n_cy - cy) < (h * 0.7) else 0.0
    else:
        features["delta_next_cx"] = 0.0
        features["delta_next_cy"] = 0.0
        features["same_line_next"] = 0.0

    return features
