"""Evaluation metrics: Precision, Recall, F1-Score (micro, macro, per-class), CER, and WER."""

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import jiwer


def compute_token_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Optional[List[str]] = None,
    ignore_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute Precision, Recall, F1-Score (Micro, Macro, Weighted) and per-class metrics for token classification.

    Args:
        y_true: List of ground-truth label strings.
        y_pred: List of predicted label strings.
        labels: Optional explicit list of target label classes to evaluate.
        ignore_labels: Optional list of labels to exclude from evaluation (e.g., ['O']).

    Returns:
        Dictionary containing overall and per-class precision, recall, and F1 scores.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})")

    if not y_true:
        return {
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "total_tokens": 0,
            "per_class": {},
        }

    # Filter out ignored labels if specified
    filtered_true = []
    filtered_pred = []
    ignored = set(ignore_labels or [])

    for t, p in zip(y_true, y_pred):
        if t in ignored and p in ignored:
            continue
        filtered_true.append(t)
        filtered_pred.append(p)

    if not filtered_true:
        filtered_true = list(y_true)
        filtered_pred = list(y_pred)

    unique_labels = sorted(list(set(filtered_true) | set(filtered_pred)))
    target_labels = [lbl for lbl in (labels or unique_labels) if lbl not in ignored]

    if not target_labels:
        target_labels = unique_labels

    # Micro metrics
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        filtered_true, filtered_pred, average="micro", labels=target_labels, zero_division=0
    )

    # Macro metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        filtered_true, filtered_pred, average="macro", labels=target_labels, zero_division=0
    )

    # Weighted metrics
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        filtered_true, filtered_pred, average="weighted", labels=target_labels, zero_division=0
    )

    # Per-class metrics
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        filtered_true, filtered_pred, average=None, labels=target_labels, zero_division=0
    )

    per_class_dict = {}
    for idx, lbl in enumerate(target_labels):
        per_class_dict[lbl] = {
            "precision": float(p_class[idx]),
            "recall": float(r_class[idx]),
            "f1": float(f1_class[idx]),
            "support": int(support_class[idx]),
        }

    return {
        "micro_precision": float(p_micro),
        "micro_recall": float(r_micro),
        "micro_f1": float(f1_micro),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "total_tokens": len(filtered_true),
        "per_class": per_class_dict,
    }


def compute_cer(reference_texts: Union[str, List[str]], hypothesis_texts: Union[str, List[str]]) -> float:
    """
    Calculate Character Error Rate (CER) using Levenshtein distance at character level.
    """
    if isinstance(reference_texts, str):
        reference_texts = [reference_texts]
    if isinstance(hypothesis_texts, str):
        hypothesis_texts = [hypothesis_texts]

    if not reference_texts:
        return 0.0

    try:
        cer = jiwer.cer(reference_texts, hypothesis_texts)
        return float(cer)
    except Exception:
        # Fallback character error calculation
        total_dist = 0
        total_len = 0
        for ref, hyp in zip(reference_texts, hypothesis_texts):
            dist = jiwer.process_characters(ref, hyp).substitutions + \
                   jiwer.process_characters(ref, hyp).deletions + \
                   jiwer.process_characters(ref, hyp).insertions
            total_dist += dist
            total_len += max(1, len(ref))
        return float(total_dist / total_len)


def compute_wer(reference_texts: Union[str, List[str]], hypothesis_texts: Union[str, List[str]]) -> float:
    """
    Calculate Word Error Rate (WER).
    """
    if isinstance(reference_texts, str):
        reference_texts = [reference_texts]
    if isinstance(hypothesis_texts, str):
        hypothesis_texts = [hypothesis_texts]

    if not reference_texts:
        return 0.0

    try:
        wer = jiwer.wer(reference_texts, hypothesis_texts)
        return float(wer)
    except Exception:
        total_dist = 0
        total_len = 0
        for ref, hyp in zip(reference_texts, hypothesis_texts):
            r_words = ref.split()
            h_words = hyp.split()
            dist = jiwer.process_words(ref, hyp).substitutions + \
                   jiwer.process_words(ref, hyp).deletions + \
                   jiwer.process_words(ref, hyp).insertions
            total_dist += dist
            total_len += max(1, len(r_words))
        return float(total_dist / total_len)
