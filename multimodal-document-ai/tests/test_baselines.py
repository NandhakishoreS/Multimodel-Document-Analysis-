"""Integration tests for baseline models (Text-Only and Layout-Aware)."""

import pytest
from models.baseline.text_baseline import TextOnlyBaselineModel
from models.baseline.layout_baseline import LayoutAwareBaselineModel


def get_dummy_samples():
    return [
        {
            "id": "doc_1",
            "words": ["INVOICE", "Date:", "2023-01-01", "Total:", "$100.00"],
            "bboxes": [
                [100, 50, 400, 100],
                [100, 150, 250, 180],
                [260, 150, 450, 180],
                [100, 800, 250, 830],
                [260, 800, 450, 830],
            ],
            "labels": ["B-HEADER", "B-QUESTION", "B-ANSWER", "B-QUESTION", "B-ANSWER"],
        },
        {
            "id": "doc_2",
            "words": ["RECEIPT", "Date:", "2023-05-12", "Total:", "$25.50"],
            "bboxes": [
                [110, 60, 390, 110],
                [100, 160, 240, 190],
                [250, 160, 440, 190],
                [100, 810, 240, 840],
                [250, 810, 440, 840],
            ],
            "labels": ["B-HEADER", "B-QUESTION", "B-ANSWER", "B-QUESTION", "B-ANSWER"],
        },
    ]


@pytest.fixture
def dummy_samples():
    return get_dummy_samples()


def test_text_baseline_fit_predict(dummy_samples):
    model = TextOnlyBaselineModel(dataset_name="funsd", max_features=100)
    model.fit(dummy_samples)
    assert model.is_fitted

    preds = model.predict(dummy_samples)
    assert len(preds) == 2
    assert len(preds[0]) == len(dummy_samples[0]["words"])

    metrics = model.evaluate(dummy_samples)
    assert "macro_f1" in metrics
    assert metrics["macro_f1"] >= 0.0


def test_layout_baseline_fit_predict(dummy_samples):
    model = LayoutAwareBaselineModel(dataset_name="funsd", max_features=100)
    model.fit(dummy_samples)
    assert model.is_fitted

    preds = model.predict(dummy_samples)
    assert len(preds) == 2
    assert len(preds[0]) == len(dummy_samples[0]["words"])

    metrics = model.evaluate(dummy_samples)
    assert "macro_f1" in metrics
    assert metrics["macro_f1"] >= 0.0
