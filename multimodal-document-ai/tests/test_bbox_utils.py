"""Unit tests for bounding box utilities."""

import pytest
from preprocessing.bbox_utils import (
    validate_bbox,
    normalize_bbox,
    denormalize_bbox,
    quad_to_bbox,
    compute_iou,
    extract_spatial_features,
)


def test_validate_bbox():
    # Regular box
    assert validate_bbox([10, 20, 100, 200]) == [10, 20, 100, 200]

    # Inverted coordinates
    assert validate_bbox([100, 200, 10, 20]) == [10, 20, 100, 200]

    # Boundary clamping
    assert validate_bbox([-10, -20, 500, 600], width=400, height=500) == [0, 0, 400, 500]

    # Zero area box expansion
    assert validate_bbox([50, 50, 50, 50]) == [50, 50, 51, 51]


def test_normalize_and_denormalize_bbox():
    width, height = 800, 1000
    orig_box = [80, 100, 400, 500]

    norm_box = normalize_bbox(orig_box, width=width, height=height)
    assert len(norm_box) == 4
    assert 0 <= norm_box[0] <= 1000
    assert 0 <= norm_box[1] <= 1000
    assert 0 <= norm_box[2] <= 1000
    assert 0 <= norm_box[3] <= 1000
    assert norm_box[0] <= norm_box[2]
    assert norm_box[1] <= norm_box[3]

    # Expected: [100, 100, 500, 500]
    assert norm_box == [100, 100, 500, 500]

    # Denormalize
    denorm_box = denormalize_bbox(norm_box, width=width, height=height)
    assert denorm_box == orig_box


def test_quad_to_bbox():
    # 8-element list
    quad = [10, 20, 100, 25, 95, 120, 12, 115]
    bbox = quad_to_bbox(quad)
    assert bbox == [10, 20, 100, 120]

    # Dict format
    quad_dict = {"x1": 5, "y1": 10, "x2": 50, "y2": 12, "x3": 48, "y3": 60, "x4": 4, "y4": 58}
    bbox_dict = quad_to_bbox(quad_dict)
    assert bbox_dict == [4, 10, 50, 60]


def test_compute_iou():
    box1 = [0, 0, 100, 100]
    box2 = [0, 0, 100, 100]
    assert compute_iou(box1, box2) == pytest.approx(1.0)

    box3 = [100, 100, 200, 200]
    assert compute_iou(box1, box3) == pytest.approx(0.0)

    box4 = [50, 0, 150, 100]
    # Inter: 50*100 = 5000, Union: 10000+10000-5000 = 15000 -> 1/3
    assert compute_iou(box1, box4) == pytest.approx(1.0 / 3.0)


def test_extract_spatial_features():
    bbox = [100, 200, 300, 400]
    prev_bbox = [50, 100, 150, 200]
    next_bbox = [350, 450, 500, 600]

    feats = extract_spatial_features(bbox, prev_bbox=prev_bbox, next_bbox=next_bbox)
    assert "norm_x0" in feats
    assert "norm_y0" in feats
    assert "aspect_ratio" in feats
    assert "delta_prev_cx" in feats
    assert "delta_next_cx" in feats
    assert feats["norm_width"] == pytest.approx(0.2)
