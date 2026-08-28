"""
LayoutLMv3 evaluation metrics using seqeval (entity-span F1) and
sklearn (token-level breakdown). Kept separate from the Phase-3
evaluation/metrics.py to avoid modifying existing baseline code.

Primary metric: seqeval macro/micro F1 on entity spans (BIO sequences).
Secondary metrics: token-level per-class precision/recall/F1 via sklearn.
Special metrics: entity-string extraction accuracy for SROIE.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from seqeval.metrics import (
    classification_report as seq_classification_report,
    f1_score as seq_f1,
    precision_score as seq_precision,
    recall_score as seq_recall,
)
from seqeval.scheme import IOB2
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Primary evaluation function (seqeval span-level)
# ---------------------------------------------------------------------------


def compute_seqeval_metrics(
    true_sequences: List[List[str]],
    pred_sequences: List[List[str]],
    ignore_label: str = "O",
    scheme: type = IOB2,
    zero_division: int = 0,
) -> Dict[str, Any]:
    """
    Compute seqeval entity-span F1, precision, and recall.

    Sequences are lists of BIO label strings per document.  Length of each
    sequence must match between true and pred.

    Parameters
    ----------
    true_sequences : List[List[str]]
        Ground-truth BIO label sequences, one list per document.
        Labels must be string BIO tags (e.g. "B-COMPANY", "I-COMPANY", "O").
    pred_sequences : List[List[str]]
        Predicted BIO label sequences, same structure.
    ignore_label : str
        Label to exclude from entity-type reporting (default "O").
    scheme : seqeval scheme class
        IOB2 (default) — matches the BIO encoding used in Phase 2.
    zero_division : int
        Value to use when a metric is undefined (0 or 1).

    Returns
    -------
    dict with keys:
        precision       : float  (micro-averaged over all entity spans)
        recall          : float
        f1              : float
        macro_f1        : float  (macro-averaged — unweighted mean over entity types)
        micro_f1        : float  (alias for f1)
        per_entity      : Dict[str, Dict]  (per entity-type P/R/F1/support)
        classification_report : str  (human-readable seqeval report)
        num_sequences   : int
        num_tokens_true : int
    """
    if not true_sequences or not pred_sequences:
        return _empty_seqeval_metrics()

    # seqeval requires sequences of equal length per document.
    aligned_true, aligned_pred = _align_sequences(true_sequences, pred_sequences)

    try:
        micro_p = seq_precision(aligned_true, aligned_pred, zero_division=zero_division)
        micro_r = seq_recall(aligned_true, aligned_pred, zero_division=zero_division)
        micro_f = seq_f1(aligned_true, aligned_pred, zero_division=zero_division)
        macro_f = seq_f1(
            aligned_true, aligned_pred,
            average="macro", zero_division=zero_division,
        )
        report_str = seq_classification_report(
            aligned_true, aligned_pred,
            zero_division=zero_division,
            output_dict=False,
        )
        report_dict = seq_classification_report(
            aligned_true, aligned_pred,
            zero_division=zero_division,
            output_dict=True,
        )
    except Exception as exc:
        logger.error("seqeval failed: %s", exc)
        return _empty_seqeval_metrics()

    per_entity: Dict[str, Dict[str, float]] = {}
    for entity_type, values in report_dict.items():
        if isinstance(values, dict) and entity_type not in {"micro avg", "macro avg", "weighted avg"}:
            per_entity[entity_type] = {
                "precision": round(float(values.get("precision", 0.0)), 4),
                "recall":    round(float(values.get("recall", 0.0)), 4),
                "f1":        round(float(values.get("f1-score", 0.0)), 4),
                "support":   int(values.get("support", 0)),
            }

    total_tokens = sum(len(s) for s in aligned_true)

    return {
        "precision":             round(float(micro_p), 4),
        "recall":                round(float(micro_r), 4),
        "f1":                    round(float(micro_f), 4),
        "micro_f1":              round(float(micro_f), 4),
        "macro_f1":              round(float(macro_f), 4),
        "per_entity":            per_entity,
        "classification_report": report_str,
        "num_sequences":         len(aligned_true),
        "num_tokens_true":       total_tokens,
    }


# ---------------------------------------------------------------------------
# Token-level metrics (sklearn) — secondary, for diagnostic use
# ---------------------------------------------------------------------------


def compute_token_metrics(
    y_true_flat: List[str],
    y_pred_flat: List[str],
    labels: Optional[List[str]] = None,
    ignore_label: str = "O",
) -> Dict[str, Any]:
    """
    Compute token-level precision/recall/F1 using sklearn.

    This is a secondary metric — seqeval span F1 is the primary.
    Useful for diagnosing per-label performance including "O".

    Parameters
    ----------
    y_true_flat : List[str]  — flat list of true token labels
    y_pred_flat : List[str]  — flat list of predicted token labels
    labels : List[str] or None — label subset to evaluate (None = all)
    ignore_label : str — label excluded from macro averaging

    Returns
    -------
    dict with keys: micro_f1, macro_f1, weighted_f1, per_class (dict)
    """
    if not y_true_flat:
        return {"micro_f1": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "per_class": {}}

    all_labels = sorted(set(y_true_flat) | set(y_pred_flat))
    eval_labels = [l for l in (labels or all_labels) if l != ignore_label]

    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_true_flat, y_pred_flat, average="micro", labels=eval_labels, zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true_flat, y_pred_flat, average="macro", labels=eval_labels, zero_division=0
    )
    _, _, f_weighted, _ = precision_recall_fscore_support(
        y_true_flat, y_pred_flat, average="weighted", labels=eval_labels, zero_division=0
    )

    p_cls, r_cls, f_cls, sup_cls = precision_recall_fscore_support(
        y_true_flat, y_pred_flat, average=None, labels=eval_labels, zero_division=0
    )
    per_class = {
        lbl: {
            "precision": round(float(p_cls[i]), 4),
            "recall":    round(float(r_cls[i]), 4),
            "f1":        round(float(f_cls[i]), 4),
            "support":   int(sup_cls[i]),
        }
        for i, lbl in enumerate(eval_labels)
    }

    return {
        "micro_precision":  round(float(p_micro), 4),
        "micro_recall":     round(float(r_micro), 4),
        "micro_f1":         round(float(f_micro), 4),
        "macro_precision":  round(float(p_macro), 4),
        "macro_recall":     round(float(r_macro), 4),
        "macro_f1":         round(float(f_macro), 4),
        "weighted_f1":      round(float(f_weighted), 4),
        "per_class":        per_class,
    }


# ---------------------------------------------------------------------------
# SROIE entity-string extraction accuracy
# ---------------------------------------------------------------------------


def compute_sroie_entity_accuracy(
    samples: List[Dict[str, Any]],
    predicted_entities: List[Dict[str, str]],
    entity_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute exact-match and partial-match accuracy for SROIE entity extraction.

    Compares the reconstructed entity strings from token predictions against
    the ground-truth entity strings stored in the JSONL `entities` dict.

    Parameters
    ----------
    samples : List[dict]
        Raw JSONL sample dicts (must have `entities` key with ground-truth strings).
    predicted_entities : List[dict]
        Reconstructed entity dicts from post-processing, one per sample.
        Keys are entity types (e.g. "company", "date", "address", "total").
    entity_keys : List[str] or None
        Entity types to evaluate.  Defaults to SROIE keys.

    Returns
    -------
    dict with keys:
        overall_exact_match   : float  (fraction of samples where ALL entities match)
        per_entity : Dict[str, dict]  (exact_match, partial_match, coverage per entity)
    """
    if entity_keys is None:
        entity_keys = ["company", "date", "address", "total"]

    per_entity: Dict[str, Dict[str, Any]] = {k: {"exact": 0, "partial": 0, "total": 0} for k in entity_keys}
    all_match_count = 0

    for sample, pred_ents in zip(samples, predicted_entities):
        gt_ents: Dict[str, str] = sample.get("entities", {})
        sample_all_match = True

        for key in entity_keys:
            gt_val = str(gt_ents.get(key, "")).strip().lower()
            pred_val = str(pred_ents.get(key, "")).strip().lower()

            if not gt_val:
                continue  # No ground truth — skip

            per_entity[key]["total"] += 1

            if gt_val == pred_val:
                per_entity[key]["exact"] += 1
            else:
                sample_all_match = False
                if gt_val and pred_val and (gt_val in pred_val or pred_val in gt_val):
                    per_entity[key]["partial"] += 1
                else:
                    sample_all_match = False

        if sample_all_match:
            all_match_count += 1

    results: Dict[str, Any] = {
        "overall_exact_match": round(all_match_count / max(1, len(samples)), 4),
        "per_entity": {},
    }
    for key in entity_keys:
        total = per_entity[key]["total"]
        results["per_entity"][key] = {
            "exact_match": round(per_entity[key]["exact"] / max(1, total), 4),
            "partial_match": round(per_entity[key]["partial"] / max(1, total), 4),
            "coverage": round(total / max(1, len(samples)), 4),
            "total_gt": total,
        }

    return results


# ---------------------------------------------------------------------------
# Helper to flatten token-level predictions from batch loop
# ---------------------------------------------------------------------------


def collect_predictions(
    all_true_word_labels: List[List[str]],
    all_pred_word_labels: List[List[str]],
    ignore_index_label: str = "O",
) -> Tuple[List[List[str]], List[List[str]], List[str], List[str]]:
    """
    Validate and return both sequence-level lists (for seqeval) and
    flat lists (for token-level sklearn metrics).

    Returns
    -------
    (seq_true, seq_pred, flat_true, flat_pred)
    """
    seq_true: List[List[str]] = []
    seq_pred: List[List[str]] = []
    flat_true: List[str] = []
    flat_pred: List[str] = []

    for true_seq, pred_seq in zip(all_true_word_labels, all_pred_word_labels):
        if len(true_seq) != len(pred_seq):
            min_len = min(len(true_seq), len(pred_seq))
            true_seq = true_seq[:min_len]
            pred_seq = pred_seq[:min_len]
            logger.debug("Trimmed sequence pair to length %d", min_len)

        seq_true.append(true_seq)
        seq_pred.append(pred_seq)
        flat_true.extend(true_seq)
        flat_pred.extend(pred_seq)

    return seq_true, seq_pred, flat_true, flat_pred


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _align_sequences(
    true_seqs: List[List[str]],
    pred_seqs: List[List[str]],
) -> Tuple[List[List[str]], List[List[str]]]:
    """Ensure each true/pred pair has the same length by trimming to minimum."""
    aligned_t, aligned_p = [], []
    for t, p in zip(true_seqs, pred_seqs):
        min_len = min(len(t), len(p))
        aligned_t.append(t[:min_len])
        aligned_p.append(p[:min_len])
    return aligned_t, aligned_p


def _empty_seqeval_metrics() -> Dict[str, Any]:
    return {
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "micro_f1": 0.0, "macro_f1": 0.0,
        "per_entity": {}, "classification_report": "",
        "num_sequences": 0, "num_tokens_true": 0,
    }
