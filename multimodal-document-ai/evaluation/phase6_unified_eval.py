"""
Phase 6: Unified evaluation module for controlled comparison.

Design
------
This module provides a single canonical IOB2 span-level evaluation function
that can be applied consistently across baselines and LayoutLMv3 wherever
the data format permits.

It does NOT modify any Phase 1-5 evaluation code.

Primary metric
--------------
Strict entity-level IOB2 span F1 (seqeval-compatible logic but implemented
explicitly here so the algorithm is fully documented and auditable):

  A predicted span (entity_type, start, end) is correct IFF:
    - entity_type matches the gold span's entity_type exactly
    - start token index matches exactly
    - end token index matches exactly

Special tokens and padding are NEVER counted (they must be filtered
by the caller before passing sequences to this module).

Usage
-----
from evaluation.phase6_unified_eval import (
    compute_strict_span_f1,
    extract_iob2_spans,
    PhaseEvalResult,
    audit_baseline_protocol,
)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SpanF1Result:
    """
    Results from strict entity-level IOB2 span F1 evaluation.

    Fields
    ------
    precision : float
        Micro-precision over all entity spans.
    recall : float
        Micro-recall over all entity spans.
    micro_f1 : float
        Micro-F1 over all entity spans.
    macro_f1 : float
        Unweighted mean of per-entity-type F1 scores.
    per_entity : Dict[str, Dict]
        Per-entity-type: precision, recall, f1, gold_count, pred_count.
    n_gold_spans : int
        Total gold entity spans evaluated.
    n_pred_spans : int
        Total predicted entity spans.
    n_correct_spans : int
        Number of exactly matching spans.
    evaluation_type : str
        Always "strict_entity_span_f1_iob2".
    """

    precision: float = 0.0
    recall: float = 0.0
    micro_f1: float = 0.0
    macro_f1: float = 0.0
    per_entity: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    n_gold_spans: int = 0
    n_pred_spans: int = 0
    n_correct_spans: int = 0
    evaluation_type: str = "strict_entity_span_f1_iob2"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_type": self.evaluation_type,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "micro_f1": round(self.micro_f1, 4),
            "macro_f1": round(self.macro_f1, 4),
            "n_gold_spans": self.n_gold_spans,
            "n_pred_spans": self.n_pred_spans,
            "n_correct_spans": self.n_correct_spans,
            "per_entity": {
                k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                for k, v in self.per_entity.items()
            },
        }


@dataclass
class PhaseEvalResult:
    """
    Standardised result container for Phase 6 comparison table rows.

    Fields
    ------
    dataset : str
    method : str
        Human-readable model name.
    evaluation_type : str
        "strict_entity_span_f1_iob2", "token_level_sklearn", or
        "entity_span_seqeval" (legacy from Phase 4/5).
    precision : float
    recall : float
    macro_f1 : float
    micro_f1 : float
    per_entity : Dict[str, Dict]
    notes : str
        Free-text notes / caveats.
    comparable_to_layoutlmv3 : bool
        True only when evaluation_type matches LayoutLMv3's protocol.
    extra : Dict
        Additional metadata (checkpoint path, split, n_samples, etc.).
    """

    dataset: str = ""
    method: str = ""
    evaluation_type: str = ""
    precision: float = 0.0
    recall: float = 0.0
    macro_f1: float = 0.0
    micro_f1: float = 0.0
    per_entity: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    notes: str = ""
    comparable_to_layoutlmv3: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        row = {
            "dataset": self.dataset,
            "method": self.method,
            "evaluation_type": self.evaluation_type,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "micro_f1": round(self.micro_f1, 4),
            "notes": self.notes,
            "comparable_to_layoutlmv3": self.comparable_to_layoutlmv3,
        }
        return row


# ---------------------------------------------------------------------------
# IOB2 span extraction
# ---------------------------------------------------------------------------


def extract_iob2_spans(
    label_sequence: List[str],
    ignore_label: str = "O",
) -> List[Tuple[str, int, int]]:
    """
    Extract entity spans from an IOB2 label sequence.

    Spans are returned as (entity_type, start_idx, end_idx) tuples
    where start_idx is inclusive and end_idx is exclusive (Python slice
    convention: labels[start:end]).

    Rules
    -----
    - A span starts with a B- tag.
    - A span extends through consecutive I- tags of the SAME entity type.
    - An I- tag that follows an O or a B/I- of a DIFFERENT type is treated
      as an implicit B- (lenient recovery, logged as a warning).
    - O tokens are never included in a span.
    - The label "O" (or whatever ignore_label specifies) is skipped.

    Parameters
    ----------
    label_sequence : List[str]
        Flat sequence of IOB2 tag strings for one document.
        Must already have special tokens removed.
    ignore_label : str
        Background label (default "O").

    Returns
    -------
    List of (entity_type, start, end) tuples, sorted by start index.
    """
    spans: List[Tuple[str, int, int]] = []
    current_entity: Optional[str] = None
    current_start: int = -1

    for idx, label in enumerate(label_sequence):
        if label == ignore_label:
            if current_entity is not None:
                spans.append((current_entity, current_start, idx))
                current_entity = None
                current_start = -1
            continue

        if label.startswith("B-"):
            if current_entity is not None:
                spans.append((current_entity, current_start, idx))
            current_entity = label[2:]
            current_start = idx

        elif label.startswith("I-"):
            entity_type = label[2:]
            if current_entity is None:
                # Orphaned I- tag: treat as implicit B-
                logger.debug(
                    "Orphaned I-%s at position %d — treating as implicit B-",
                    entity_type, idx
                )
                current_entity = entity_type
                current_start = idx
            elif entity_type != current_entity:
                # I- tag changing entity type mid-span: close old, open new
                logger.debug(
                    "I-tag entity mismatch at %d: was %s, now %s",
                    idx, current_entity, entity_type
                )
                spans.append((current_entity, current_start, idx))
                current_entity = entity_type
                current_start = idx
            # else: same entity, extend span (do nothing)

        else:
            # Unknown tag format — treat as O
            logger.warning("Unrecognised label '%s' at position %d — treating as O", label, idx)
            if current_entity is not None:
                spans.append((current_entity, current_start, idx))
                current_entity = None
                current_start = -1

    # Close any open span at end of sequence
    if current_entity is not None:
        spans.append((current_entity, current_start, len(label_sequence)))

    return spans


# ---------------------------------------------------------------------------
# Strict entity span F1
# ---------------------------------------------------------------------------


def compute_strict_span_f1(
    true_sequences: List[List[str]],
    pred_sequences: List[List[str]],
    ignore_label: str = "O",
) -> SpanF1Result:
    """
    Compute strict entity-level IOB2 span F1.

    A predicted span is correct IFF:
      - entity_type matches exactly
      - start token index matches exactly
      - end token index matches exactly

    Evaluation is performed at the document level: each document has an
    independent span set.

    Parameters
    ----------
    true_sequences : List[List[str]]
        Ground-truth IOB2 label sequences, one list per document.
        Special tokens must already be removed by the caller.
    pred_sequences : List[List[str]]
        Predicted IOB2 label sequences, same structure.
    ignore_label : str
        Background label to exclude from span extraction (default "O").

    Returns
    -------
    SpanF1Result
    """
    if not true_sequences:
        return SpanF1Result()

    # Accumulate per-entity counts
    entity_tp: Dict[str, int] = defaultdict(int)
    entity_fp: Dict[str, int] = defaultdict(int)
    entity_fn: Dict[str, int] = defaultdict(int)
    entity_gold: Dict[str, int] = defaultdict(int)
    entity_pred_count: Dict[str, int] = defaultdict(int)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gold = 0
    total_pred = 0

    for doc_idx, (true_seq, pred_seq) in enumerate(zip(true_sequences, pred_sequences)):
        # Align lengths (trim to minimum)
        min_len = min(len(true_seq), len(pred_seq))
        true_seq = true_seq[:min_len]
        pred_seq = pred_seq[:min_len]

        gold_spans: Set[Tuple[str, int, int]] = set(
            extract_iob2_spans(true_seq, ignore_label=ignore_label)
        )
        pred_spans: Set[Tuple[str, int, int]] = set(
            extract_iob2_spans(pred_seq, ignore_label=ignore_label)
        )

        for etype, start, end in gold_spans:
            entity_gold[etype] += 1
        for etype, start, end in pred_spans:
            entity_pred_count[etype] += 1

        correct = gold_spans & pred_spans
        for etype, start, end in correct:
            entity_tp[etype] += 1
        for etype, start, end in (pred_spans - gold_spans):
            entity_fp[etype] += 1
        for etype, start, end in (gold_spans - pred_spans):
            entity_fn[etype] += 1

        total_gold += len(gold_spans)
        total_pred += len(pred_spans)
        total_tp += len(correct)
        total_fp += len(pred_spans - gold_spans)
        total_fn += len(gold_spans - pred_spans)

    # Micro metrics
    micro_p = total_tp / total_pred if total_pred > 0 else 0.0
    micro_r = total_tp / total_gold if total_gold > 0 else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if (micro_p + micro_r) > 0 else 0.0
    )

    # Per-entity metrics
    all_entity_types = sorted(set(entity_gold.keys()) | set(entity_pred_count.keys()))
    per_entity: Dict[str, Dict[str, Any]] = {}
    per_entity_f1_scores: List[float] = []

    for etype in all_entity_types:
        tp = entity_tp[etype]
        fp = entity_fp[etype]
        fn = entity_fn[etype]
        gold_count = entity_gold[etype]
        pred_count = entity_pred_count[etype]

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        per_entity[etype] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "gold_count": gold_count,
            "pred_count": pred_count,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        per_entity_f1_scores.append(f1)

    # Macro F1: unweighted mean over entity types with gold support
    types_with_gold = [etype for etype in all_entity_types if entity_gold[etype] > 0]
    macro_f1_scores = [per_entity[etype]["f1"] for etype in types_with_gold]
    macro_f1 = (
        sum(macro_f1_scores) / len(macro_f1_scores)
        if macro_f1_scores else 0.0
    )

    return SpanF1Result(
        precision=round(micro_p, 4),
        recall=round(micro_r, 4),
        micro_f1=round(micro_f1, 4),
        macro_f1=round(macro_f1, 4),
        per_entity=per_entity,
        n_gold_spans=total_gold,
        n_pred_spans=total_pred,
        n_correct_spans=total_tp,
    )


# ---------------------------------------------------------------------------
# Audit protocol documentation
# ---------------------------------------------------------------------------


def audit_baseline_protocol() -> Dict[str, Any]:
    """
    Return a structured audit of the Phase 3 baseline evaluation protocol.
    Documentation only — does not load any data.
    """
    return {
        "model": "Text-Only Baseline (TF-IDF + LR) & Text+Layout Baseline (Spatial + LR)",
        "implementation_files": [
            "models/baseline/text_baseline.py",
            "models/baseline/layout_baseline.py",
            "evaluation/metrics.py::compute_token_classification_metrics",
        ],
        "audit": {
            "Q1_tokens_evaluated": (
                "All word tokens in the test split. No tokenizer subwords — "
                "each word is one feature vector predicting one label."
            ),
            "Q2_O_tokens": (
                "O tokens are excluded from macro/micro averaging when "
                "ignore_labels=['O'] is passed. However, positions where "
                "true=O but pred=X (false positives for class X) ARE counted."
            ),
            "Q3_special_tokens": "Not applicable — no tokenizer subwords.",
            "Q4_evaluation_level": (
                "TOKEN-LEVEL. Each token is an independent classification. "
                "sklearn precision_recall_fscore_support is used."
            ),
            "Q5_bio_spans": (
                "B/I tags treated as INDEPENDENT LABELS. No span extraction. "
                "B-COMPANY and I-COMPANY are two distinct classes."
            ),
            "Q6_macro_f1": (
                "sklearn macro F1: mean of per-class F1 over non-O labels "
                "(B-COMPANY, I-COMPANY, B-DATE, I-DATE, etc.). NOT equivalent "
                "to seqeval entity-type macro F1."
            ),
            "Q7_micro_f1": (
                "sklearn micro F1: TP/(TP+FP) over all non-O labels globally."
            ),
            "Q8_same_samples": (
                "FUNSD and CORD: same test split counts as LayoutLMv3. "
                "SROIE: baselines used OLD preprocessing — NOT the same "
                "preprocessed data as corrected LayoutLMv3."
            ),
            "Q9_label_mapping": (
                "preprocessing/label_mapping.py::get_dataset_label_encoder. "
                "Label sets match configs/labels/*.yaml but evaluation level differs."
            ),
        },
    }


def audit_layoutlmv3_protocol() -> Dict[str, Any]:
    """
    Return a structured audit of the LayoutLMv3 evaluation protocol (Phase 4/5).
    Documentation only — does not load any data.
    """
    return {
        "model": "LayoutLMv3-base (microsoft/layoutlmv3-base fine-tuned)",
        "implementation_files": [
            "evaluation/evaluate_layoutlmv3.py",
            "evaluation/layoutlmv3_metrics.py::compute_seqeval_metrics",
            "training/train_layoutlmv3.py::extract_word_level_labels",
            "inference/layoutlmv3_predict.py::token_preds_to_word_preds",
        ],
        "primary_metric": "seqeval entity-span F1 (IOB2 scheme)",
        "audit": {
            "Q1_tokens_evaluated": (
                "Word-level labels only. Subword tokens aggregated to word level "
                "via word_ids. Only first subword of each word contributes. "
                "CLS/SEP excluded via word_ids=None check."
            ),
            "Q2_O_tokens": (
                "O tokens are included in word-level sequences passed to seqeval, "
                "but seqeval ignores O tokens in span extraction. "
                "O tokens do NOT contribute to any span-level metric."
            ),
            "Q3_special_tokens": (
                "YES — excluded via word_ids=None in extract_word_level_labels "
                "and token_preds_to_word_preds."
            ),
            "Q4_evaluation_level": (
                "ENTITY-SPAN LEVEL. seqeval IOB2 scheme. A prediction is correct "
                "only if entire span boundary AND entity type match."
            ),
            "Q5_bio_spans": (
                "B/I tags grouped into entity spans by seqeval. "
                "Strict IOB2: span starts at B-X, extends through I-X tokens."
            ),
            "Q6_macro_f1": (
                "seqeval macro F1: mean of per-entity-TYPE F1 "
                "(e.g., COMPANY, DATE, ADDRESS, TOTAL for SROIE). "
                "NOT the same as sklearn macro F1 over B/I labels."
            ),
            "Q7_micro_f1": (
                "seqeval micro F1: TP/(TP+FP) globally across all entity spans."
            ),
            "Q8_same_samples": (
                "Within each dataset, yes (SROIE corrected: 361, FUNSD: 50, CORD: 100). "
                "SROIE corrected LayoutLMv3 NOT evaluated on same preprocessed data "
                "as Phase 3 baselines."
            ),
            "Q9_label_mapping": (
                "SROIE: 9 labels. FUNSD: 7 labels. CORD: 13 labels. "
                "CORD seqeval measures only MENU/SUBTOTAL/TOTAL (VOID/PAYMENT/ETC "
                "have zero gold spans; o_ratio=0.0 confirms no O tokens in CORD test)."
            ),
        },
    }


def build_incompatibility_matrix() -> Dict[str, Any]:
    """
    Document which cross-model comparisons are and are not valid.
    """
    return {
        "SROIE": {
            "baseline_vs_layoutlmv3_comparable": False,
            "reasons": [
                "Evaluation level: token-level (baseline) vs entity-span (LayoutLMv3).",
                "Preprocessing: old SROIE data (baseline) vs corrected SROIE (LayoutLMv3).",
                "B/I labels treated independently (baseline) vs grouped into spans (LayoutLMv3).",
                "Baseline macro F1 depressed by I-tag classes with near-zero support.",
            ],
            "direction_valid": True,
            "note": (
                "Direction of improvement (LayoutLMv3 >> baseline) is unambiguous "
                "even across metric protocols."
            ),
        },
        "FUNSD": {
            "baseline_vs_layoutlmv3_comparable": False,
            "reasons": [
                "Same test split (50 docs) but evaluation level differs.",
                "Token-level (baseline) vs entity-span (LayoutLMv3).",
                "Unified span F1 can be computed for a controlled comparison.",
            ],
            "direction_valid": True,
            "note": (
                "Span F1 applied to baseline predictions yields a controlled comparison "
                "under the same metric."
            ),
        },
        "CORD": {
            "baseline_vs_layoutlmv3_comparable": False,
            "reasons": [
                "Same test split (100 docs) but evaluation level differs.",
                "Token-level (baseline) vs entity-span (LayoutLMv3).",
                "Task formulation: 3-class closed-set (no O tokens). "
                "NOT equivalent to full-document CORD KIE benchmark.",
            ],
            "direction_valid": True,
            "note": (
                "CORD results are NOT comparable to published full-document CORD benchmarks. "
                "Within this project's task formulation, direction of improvement is clear."
            ),
        },
    }
