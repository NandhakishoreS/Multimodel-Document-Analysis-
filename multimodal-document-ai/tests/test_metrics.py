"""Unit tests for evaluation metrics (F1, Precision, Recall, CER, WER)."""

import pytest
from evaluation.metrics import compute_token_classification_metrics, compute_cer, compute_wer


def test_token_classification_metrics():
    y_true = ["B-QUESTION", "I-QUESTION", "B-ANSWER", "O", "O"]
    y_pred = ["B-QUESTION", "I-QUESTION", "B-ANSWER", "O", "O"]

    metrics = compute_token_classification_metrics(y_true, y_pred)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["micro_f1"] == pytest.approx(1.0)


def test_imperfect_metrics():
    y_true = ["B-QUESTION", "I-QUESTION", "B-ANSWER", "O"]
    y_pred = ["B-QUESTION", "O", "B-ANSWER", "O"]

    metrics = compute_token_classification_metrics(y_true, y_pred)
    assert 0.0 < metrics["macro_f1"] < 1.0


def test_cer_and_wer():
    ref = ["TOTAL DUE: $50.00"]
    hyp = ["TOTAL DUE: $50.00"]

    assert compute_cer(ref, hyp) == pytest.approx(0.0)
    assert compute_wer(ref, hyp) == pytest.approx(0.0)

    # Substitution
    hyp_err = ["TOTAL DUE: $58.00"]
    assert compute_cer(ref, hyp_err) > 0.0
    assert compute_wer(ref, hyp_err) > 0.0
