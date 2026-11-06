"""
Phase 7: Error Analysis, Failure Classification & Prediction Analysis Module.

Provides:
1. Formal Error Taxonomy (TP, FN, FP, Boundary, Type, Partial, OCR, Confusion, Multi-Token, Unknown).
2. Strict span matching engine aligned with Phase 6 IOB2 evaluation.
3. Diagnostic analysis routines for SROIE (TOTAL, ADDRESS, COMPANY, DATE), FUNSD, and CORD.
4. Document-level error accounting and ranking.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class ErrorCategory(str, Enum):
    TRUE_POSITIVE = "TRUE_POSITIVE"
    FALSE_NEGATIVE = "FALSE_NEGATIVE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    BOUNDARY_ERROR = "BOUNDARY_ERROR"
    TYPE_ERROR = "TYPE_ERROR"
    PARTIAL_MATCH = "PARTIAL_MATCH"
    OCR_RELATED = "OCR_RELATED"
    LAYOUT_RELATED = "LAYOUT_RELATED"
    CONFUSION_ERROR = "CONFUSION_ERROR"
    MULTI_TOKEN_ERROR = "MULTI_TOKEN_ERROR"
    UNKNOWN_UNDETERMINED = "UNKNOWN_UNDETERMINED"


@dataclass
class EntitySpan:
    """Represents an extracted entity span over word tokens."""
    entity_type: str
    start: int  # 0-indexed word start (inclusive)
    end: int    # 0-indexed word end (exclusive)
    text: str = ""
    bboxes: List[List[int]] = field(default_factory=list)
    confidence: float = 1.0

    @property
    def token_length(self) -> int:
        return self.end - self.start

    def overlaps_with(self, other: EntitySpan) -> bool:
        return max(self.start, other.start) < min(self.end, other.end)

    def overlap_ratio(self, other: EntitySpan) -> float:
        inter = max(0, min(self.end, other.end) - max(self.start, other.start))
        union = max(self.end, other.end) - min(self.start, other.start)
        return inter / union if union > 0 else 0.0


@dataclass
class MatchedEntityPair:
    """Detailed pairing between gold and predicted entity spans."""
    gold: Optional[EntitySpan]
    pred: Optional[EntitySpan]
    category: ErrorCategory
    notes: str = ""
    detailed_cause: str = ""


@dataclass
class DocumentErrorRecord:
    """Document-level error evaluation and diagnostics."""
    doc_id: str
    dataset: str
    words: List[str]
    bboxes: List[List[int]]
    gold_bio: List[str]
    pred_bio: List[str]
    gold_spans: List[EntitySpan]
    pred_spans: List[EntitySpan]
    pairs: List[MatchedEntityPair]
    tp_count: int
    fp_count: int
    fn_count: int
    boundary_errors: int
    type_errors: int
    partial_matches: int
    total_errors: int


# ---------------------------------------------------------------------------
# Strict IOB2 Span Extraction
# ---------------------------------------------------------------------------

def extract_iob2_entity_spans(
    bio_tags: List[str],
    words: Optional[List[str]] = None,
    bboxes: Optional[List[List[int]]] = None,
    confidences: Optional[List[float]] = None,
) -> List[EntitySpan]:
    """
    Extract entity spans from a word-level IOB2 tag sequence.
    Preserves token text, bounding boxes, and mean confidence if provided.
    """
    spans: List[EntitySpan] = []
    current_type: Optional[str] = None
    start_idx: int = 0

    for i, tag in enumerate(bio_tags):
        if tag.startswith("B-"):
            if current_type is not None:
                span_text = " ".join(words[start_idx:i]) if words else ""
                span_bboxes = bboxes[start_idx:i] if bboxes else []
                span_conf = (
                    sum(confidences[start_idx:i]) / (i - start_idx)
                    if confidences and i > start_idx else 1.0
                )
                spans.append(EntitySpan(
                    entity_type=current_type,
                    start=start_idx,
                    end=i,
                    text=span_text,
                    bboxes=span_bboxes,
                    confidence=span_conf,
                ))
            current_type = tag[2:]
            start_idx = i
        elif tag.startswith("I-"):
            ent = tag[2:]
            if current_type == ent:
                continue
            else:
                if current_type is not None:
                    span_text = " ".join(words[start_idx:i]) if words else ""
                    span_bboxes = bboxes[start_idx:i] if bboxes else []
                    span_conf = (
                        sum(confidences[start_idx:i]) / (i - start_idx)
                        if confidences and i > start_idx else 1.0
                    )
                    spans.append(EntitySpan(
                        entity_type=current_type,
                        start=start_idx,
                        end=i,
                        text=span_text,
                        bboxes=span_bboxes,
                        confidence=span_conf,
                    ))
                current_type = ent
                start_idx = i
        else:  # O or other
            if current_type is not None:
                span_text = " ".join(words[start_idx:i]) if words else ""
                span_bboxes = bboxes[start_idx:i] if bboxes else []
                span_conf = (
                    sum(confidences[start_idx:i]) / (i - start_idx)
                    if confidences and i > start_idx else 1.0
                )
                spans.append(EntitySpan(
                    entity_type=current_type,
                    start=start_idx,
                    end=i,
                    text=span_text,
                    bboxes=span_bboxes,
                    confidence=span_conf,
                ))
                current_type = None

    if current_type is not None:
        n = len(bio_tags)
        span_text = " ".join(words[start_idx:n]) if words else ""
        span_bboxes = bboxes[start_idx:n] if bboxes else []
        span_conf = (
            sum(confidences[start_idx:n]) / (n - start_idx)
            if confidences and n > start_idx else 1.0
        )
        spans.append(EntitySpan(
            entity_type=current_type,
            start=start_idx,
            end=n,
            text=span_text,
            bboxes=span_bboxes,
            confidence=span_conf,
        ))

    return spans


# ---------------------------------------------------------------------------
# Matching and Classification Engine
# ---------------------------------------------------------------------------

def classify_document_errors(
    doc_id: str,
    dataset: str,
    words: List[str],
    bboxes: List[List[int]],
    gold_bio: List[str],
    pred_bio: List[str],
    confidences: Optional[List[float]] = None,
) -> DocumentErrorRecord:
    """
    Match gold entity spans with predicted entity spans and categorize all errors.
    Uses Phase 6 strict exact match for True Positives.
    """
    gold_spans = extract_iob2_entity_spans(gold_bio, words, bboxes)
    pred_spans = extract_iob2_entity_spans(pred_bio, words, bboxes, confidences)

    matched_gold_indices: Set[int] = set()
    matched_pred_indices: Set[int] = set()
    pairs: List[MatchedEntityPair] = []

    # 1. Exact Matches (True Positives)
    for g_idx, g in enumerate(gold_spans):
        for p_idx, p in enumerate(pred_spans):
            if p_idx in matched_pred_indices:
                continue
            if g.entity_type == p.entity_type and g.start == p.start and g.end == p.end:
                matched_gold_indices.add(g_idx)
                matched_pred_indices.add(p_idx)
                pairs.append(MatchedEntityPair(
                    gold=g,
                    pred=p,
                    category=ErrorCategory.TRUE_POSITIVE,
                    notes="Exact entity span match",
                    detailed_cause="Exact boundary and entity type agreement",
                ))
                break

    # 2. Overlapping Matches (Boundary Error, Type Error, Partial Match)
    for g_idx, g in enumerate(gold_spans):
        if g_idx in matched_gold_indices:
            continue
        best_p_idx = None
        best_overlap = 0.0
        for p_idx, p in enumerate(pred_spans):
            if p_idx in matched_pred_indices:
                continue
            ov = g.overlap_ratio(p)
            if ov > best_overlap:
                best_overlap = ov
                best_p_idx = p_idx

        if best_p_idx is not None and best_overlap > 0.0:
            p = pred_spans[best_p_idx]
            matched_gold_indices.add(g_idx)
            matched_pred_indices.add(best_p_idx)

            # Determine category
            if g.entity_type == p.entity_type:
                cat = ErrorCategory.BOUNDARY_ERROR
                cause = f"Same entity {g.entity_type} but boundaries differ: gold [{g.start}:{g.end}] vs pred [{p.start}:{p.end}]"
            else:
                cat = ErrorCategory.TYPE_ERROR
                cause = f"Confused entity type: gold {g.entity_type} [{g.start}:{g.end}] vs pred {p.entity_type} [{p.start}:{p.end}]"

            # Check OCR or multi-token / multi-line indicators
            is_multi_line = False
            if g.bboxes and len(g.bboxes) > 1:
                y_diffs = [abs(g.bboxes[k][1] - g.bboxes[0][1]) for k in range(len(g.bboxes))]
                if max(y_diffs) > 20:
                    is_multi_line = True

            if is_multi_line and g.token_length > 3:
                cause += " | Multi-line entity span"

            pairs.append(MatchedEntityPair(
                gold=g,
                pred=p,
                category=cat,
                notes=f"Overlapping span (IoU={best_overlap:.2f})",
                detailed_cause=cause,
            ))

    # 3. Unmatched Gold Spans (False Negatives)
    for g_idx, g in enumerate(gold_spans):
        if g_idx in matched_gold_indices:
            continue
        # Check if text is corrupted or unreadable
        is_ocr_issue = bool(re.search(r"[^\x20-\x7E]|([A-Z0-9]{15,})", g.text))
        cat = ErrorCategory.FALSE_NEGATIVE
        cause = f"Gold {g.entity_type} '{g.text}' not detected"
        if is_ocr_issue:
            cause += " | Potential OCR noise"
        if g.token_length > 5:
            cause += " | Long multi-token span"

        pairs.append(MatchedEntityPair(
            gold=g,
            pred=None,
            category=cat,
            notes="Missed entity",
            detailed_cause=cause,
        ))

    # 4. Unmatched Predicted Spans (False Positives)
    for p_idx, p in enumerate(pred_spans):
        if p_idx in matched_pred_indices:
            continue
        pairs.append(MatchedEntityPair(
            gold=None,
            pred=p,
            category=ErrorCategory.FALSE_POSITIVE,
            notes="Spurious prediction",
            detailed_cause=f"Predicted {p.entity_type} '{p.text}' with no corresponding gold entity",
        ))

    # Calculate summary counts
    tp_count = sum(1 for pair in pairs if pair.category == ErrorCategory.TRUE_POSITIVE)
    fn_count = sum(1 for pair in pairs if pair.category == ErrorCategory.FALSE_NEGATIVE or (pair.gold is not None and pair.category != ErrorCategory.TRUE_POSITIVE))
    fp_count = sum(1 for pair in pairs if pair.category == ErrorCategory.FALSE_POSITIVE or (pair.pred is not None and pair.category != ErrorCategory.TRUE_POSITIVE))
    boundary_errors = sum(1 for pair in pairs if pair.category == ErrorCategory.BOUNDARY_ERROR)
    type_errors = sum(1 for pair in pairs if pair.category == ErrorCategory.TYPE_ERROR)
    partial_matches = sum(1 for pair in pairs if pair.category in (ErrorCategory.BOUNDARY_ERROR, ErrorCategory.PARTIAL_MATCH))

    # Errors on gold entities: gold not exactly matched + spurious preds
    total_errors = (len(gold_spans) - tp_count) + (len(pred_spans) - tp_count)

    return DocumentErrorRecord(
        doc_id=doc_id,
        dataset=dataset,
        words=words,
        bboxes=bboxes,
        gold_bio=gold_bio,
        pred_bio=pred_bio,
        gold_spans=gold_spans,
        pred_spans=pred_spans,
        pairs=pairs,
        tp_count=tp_count,
        fp_count=fp_count,
        fn_count=fn_count,
        boundary_errors=boundary_errors,
        type_errors=type_errors,
        partial_matches=partial_matches,
        total_errors=total_errors,
    )


# ---------------------------------------------------------------------------
# Per-Entity Summary & Aggregation
# ---------------------------------------------------------------------------

def compute_per_entity_error_summary(
    records: List[DocumentErrorRecord],
    entity_types: List[str],
) -> List[Dict[str, Any]]:
    """
    Compute structured error accounting table across documents for given entity types:
    Entity | Gold | Predicted | TP | FP | FN | Boundary | Partial | Type | OCR-related | Precision | Recall | F1
    """
    stats: Dict[str, Dict[str, int]] = {
        ent: defaultdict(int) for ent in entity_types
    }

    for rec in records:
        for pair in rec.pairs:
            # True Positive
            if pair.category == ErrorCategory.TRUE_POSITIVE and pair.gold is not None:
                ent = pair.gold.entity_type
                if ent in stats:
                    stats[ent]["gold"] += 1
                    stats[ent]["pred"] += 1
                    stats[ent]["tp"] += 1

            # Boundary Error
            elif pair.category == ErrorCategory.BOUNDARY_ERROR and pair.gold is not None:
                ent = pair.gold.entity_type
                if ent in stats:
                    stats[ent]["gold"] += 1
                    stats[ent]["pred"] += 1
                    stats[ent]["fn"] += 1
                    stats[ent]["fp"] += 1
                    stats[ent]["boundary"] += 1
                    stats[ent]["partial"] += 1

            # Type Error
            elif pair.category == ErrorCategory.TYPE_ERROR:
                if pair.gold is not None and pair.gold.entity_type in stats:
                    stats[pair.gold.entity_type]["gold"] += 1
                    stats[pair.gold.entity_type]["fn"] += 1
                    stats[pair.gold.entity_type]["type_error"] += 1
                if pair.pred is not None and pair.pred.entity_type in stats:
                    stats[pair.pred.entity_type]["pred"] += 1
                    stats[pair.pred.entity_type]["fp"] += 1
                    stats[pair.pred.entity_type]["type_error"] += 1

            # False Negative (missed)
            elif pair.category == ErrorCategory.FALSE_NEGATIVE and pair.gold is not None:
                ent = pair.gold.entity_type
                if ent in stats:
                    stats[ent]["gold"] += 1
                    stats[ent]["fn"] += 1
                    if "OCR" in pair.detailed_cause:
                        stats[ent]["ocr"] += 1
                    else:
                        stats[ent]["miss"] += 1

            # False Positive (spurious)
            elif pair.category == ErrorCategory.FALSE_POSITIVE and pair.pred is not None:
                ent = pair.pred.entity_type
                if ent in stats:
                    stats[ent]["pred"] += 1
                    stats[ent]["fp"] += 1
                    stats[ent]["spurious"] += 1

    summary = []
    for ent in entity_types:
        d = stats[ent]
        gold = d["gold"]
        pred = d["pred"]
        tp = d["tp"]
        fp = d["fp"]
        fn = d["fn"]
        p = tp / pred if pred > 0 else 0.0
        r = tp / gold if gold > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        summary.append({
            "entity": ent,
            "gold": gold,
            "pred": pred,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "boundary": d["boundary"],
            "partial": d["partial"],
            "type_error": d["type_error"],
            "ocr_related": d["ocr"],
            "other_errors": d["miss"] + d["spurious"],
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        })

    return summary


# ---------------------------------------------------------------------------
# SROIE Specific Diagnostics
# ---------------------------------------------------------------------------

def analyze_sroie_total(records: List[DocumentErrorRecord]) -> Dict[str, Any]:
    """
    Detailed diagnostic analysis for SROIE TOTAL (F1 = 0.7725).
    Investigates confusion between TOTAL, SUBTOTAL, TAX, CASH, CHANGE, and line items.
    """
    missed_totals = 0
    wrong_numeric_selected = 0
    subtotal_tax_cash_confused = 0
    boundary_mismatch = 0
    multiple_totals_predicted = 0
    case_studies = []

    for rec in records:
        gold_totals = [g for g in rec.gold_spans if g.entity_type == "TOTAL"]
        pred_totals = [p for p in rec.pred_spans if p.entity_type == "TOTAL"]

        if len(pred_totals) > 1:
            multiple_totals_predicted += 1

        if not gold_totals:
            continue

        gt = gold_totals[0]
        if not pred_totals:
            missed_totals += 1
            case_studies.append({
                "doc_id": rec.doc_id,
                "gold_text": gt.text,
                "pred_text": "None",
                "failure_type": "Missed TOTAL",
                "context": " ".join(rec.words[max(0, gt.start - 5):min(len(rec.words), gt.end + 5)]),
            })
            continue

        pt = pred_totals[0]
        if gt.start == pt.start and gt.end == pt.end:
            continue

        # Check if boundaries overlap
        if gt.overlaps_with(pt):
            boundary_mismatch += 1
            case_studies.append({
                "doc_id": rec.doc_id,
                "gold_text": gt.text,
                "pred_text": pt.text,
                "failure_type": "Boundary Mismatch",
                "context": " ".join(rec.words[max(0, min(gt.start, pt.start) - 3):min(len(rec.words), max(gt.end, pt.end) + 3)]),
            })
        else:
            # Check context around predicted number
            pred_ctx = " ".join(rec.words[max(0, pt.start - 4):min(len(rec.words), pt.end + 4)]).upper()
            if any(term in pred_ctx for term in ["SUBTOTAL", "SUB TOTAL", "TAX", "GST", "CASH", "CHANGE"]):
                subtotal_tax_cash_confused += 1
                case_studies.append({
                    "doc_id": rec.doc_id,
                    "gold_text": gt.text,
                    "pred_text": pt.text,
                    "failure_type": "Confused with Subtotal/Tax/Cash",
                    "context": pred_ctx,
                })
            else:
                wrong_numeric_selected += 1
                case_studies.append({
                    "doc_id": rec.doc_id,
                    "gold_text": gt.text,
                    "pred_text": pt.text,
                    "failure_type": "Wrong Numeric Amount Selected",
                    "context": pred_ctx,
                })

    return {
        "missed_totals": missed_totals,
        "boundary_mismatch": boundary_mismatch,
        "subtotal_tax_cash_confused": subtotal_tax_cash_confused,
        "wrong_numeric_selected": wrong_numeric_selected,
        "multiple_totals_predicted": multiple_totals_predicted,
        "sample_cases": case_studies[:10],
    }


def analyze_sroie_address(records: List[DocumentErrorRecord]) -> Dict[str, Any]:
    """
    Detailed diagnostic analysis for SROIE ADDRESS (F1 = 0.8434).
    Calculates average token counts, multi-line span behavior, and short vs long address performance.
    """
    gold_lengths: List[int] = []
    pred_lengths: List[int] = []
    exact_matches = 0
    boundary_errors = 0
    misses = 0
    short_address_matches = 0
    short_address_total = 0
    long_address_matches = 0
    long_address_total = 0

    case_studies = []

    for rec in records:
        g_addrs = [g for g in rec.gold_spans if g.entity_type == "ADDRESS"]
        p_addrs = [p for p in rec.pred_spans if p.entity_type == "ADDRESS"]

        for g in g_addrs:
            gold_lengths.append(g.token_length)
            is_short = g.token_length <= 10
            if is_short:
                short_address_total += 1
            else:
                long_address_total += 1

            matched_pred = None
            for p in p_addrs:
                if g.start == p.start and g.end == p.end:
                    exact_matches += 1
                    if is_short:
                        short_address_matches += 1
                    else:
                        long_address_matches += 1
                    matched_pred = p
                    break
                elif g.overlaps_with(p):
                    matched_pred = p

            if matched_pred is None:
                misses += 1
            elif matched_pred.start != g.start or matched_pred.end != g.end:
                boundary_errors += 1
                case_studies.append({
                    "doc_id": rec.doc_id,
                    "gold_text": g.text,
                    "pred_text": matched_pred.text,
                    "gold_len": g.token_length,
                    "pred_len": matched_pred.token_length,
                    "issue": "Truncated or overextended address boundary",
                })

        for p in p_addrs:
            pred_lengths.append(p.token_length)

    avg_gold_len = sum(gold_lengths) / len(gold_lengths) if gold_lengths else 0.0
    avg_pred_len = sum(pred_lengths) / len(pred_lengths) if pred_lengths else 0.0

    return {
        "avg_gold_token_count": round(avg_gold_len, 2),
        "avg_pred_token_count": round(avg_pred_len, 2),
        "exact_match_count": exact_matches,
        "boundary_error_count": boundary_errors,
        "miss_count": misses,
        "short_address_accuracy": round(short_address_matches / short_address_total, 4) if short_address_total else 0.0,
        "long_address_accuracy": round(long_address_matches / long_address_total, 4) if long_address_total else 0.0,
        "sample_cases": case_studies[:8],
    }


def analyze_sroie_company(records: List[DocumentErrorRecord]) -> Dict[str, Any]:
    """
    Detailed diagnostic analysis for SROIE COMPANY (F1 = 0.9049).
    Buckets entities by token length: 1, 2-3, 4-6, 7+ tokens.
    """
    buckets = {
        "1 token": {"gold": 0, "tp": 0},
        "2-3 tokens": {"gold": 0, "tp": 0},
        "4-6 tokens": {"gold": 0, "tp": 0},
        "7+ tokens": {"gold": 0, "tp": 0},
    }

    boundary_errors = 0
    case_studies = []

    for rec in records:
        g_comps = [g for g in rec.gold_spans if g.entity_type == "COMPANY"]
        p_comps = [p for p in rec.pred_spans if p.entity_type == "COMPANY"]

        for g in g_comps:
            tl = g.token_length
            bucket_key = "1 token" if tl == 1 else "2-3 tokens" if tl <= 3 else "4-6 tokens" if tl <= 6 else "7+ tokens"
            buckets[bucket_key]["gold"] += 1

            matched_exact = False
            overlapping_pred = None
            for p in p_comps:
                if g.start == p.start and g.end == p.end:
                    matched_exact = True
                    break
                elif g.overlaps_with(p):
                    overlapping_pred = p

            if matched_exact:
                buckets[bucket_key]["tp"] += 1
            elif overlapping_pred:
                boundary_errors += 1
                case_studies.append({
                    "doc_id": rec.doc_id,
                    "gold_text": g.text,
                    "pred_text": overlapping_pred.text,
                    "gold_len": g.token_length,
                    "pred_len": overlapping_pred.token_length,
                })

    bucket_results = {}
    for k, v in buckets.items():
        g_cnt = v["gold"]
        tp_cnt = v["tp"]
        acc = tp_cnt / g_cnt if g_cnt else 0.0
        bucket_results[k] = {
            "gold_count": g_cnt,
            "tp_count": tp_cnt,
            "exact_match_rate": round(acc, 4),
        }

    return {
        "bucket_results": bucket_results,
        "boundary_errors": boundary_errors,
        "sample_cases": case_studies[:8],
    }


def analyze_sroie_date(records: List[DocumentErrorRecord]) -> Dict[str, Any]:
    """
    Detailed diagnostic analysis for SROIE DATE (F1 = 0.9930).
    Examines the small number of error cases.
    """
    errors = []
    for rec in records:
        g_dates = [g for g in rec.gold_spans if g.entity_type == "DATE"]
        p_dates = [p for p in rec.pred_spans if p.entity_type == "DATE"]

        for g in g_dates:
            matched = any(g.start == p.start and g.end == p.end for p in p_dates)
            if not matched:
                pred_match = next((p for p in p_dates if g.overlaps_with(p)), None)
                errors.append({
                    "doc_id": rec.doc_id,
                    "gold_text": g.text,
                    "pred_text": pred_match.text if pred_match else "None",
                    "issue": "Boundary mismatch" if pred_match else "Missed date",
                })

    return {
        "total_errors": len(errors),
        "error_cases": errors,
    }


# ---------------------------------------------------------------------------
# FUNSD & CORD Specific Diagnostics
# ---------------------------------------------------------------------------

def analyze_funsd_errors(records: List[DocumentErrorRecord]) -> Dict[str, Any]:
    """
    Diagnostic analysis for FUNSD LayoutLMv3 predictions:
    QUESTION vs ANSWER confusion, HEADER boundaries, spatial adjacency.
    """
    q_as_a = 0
    a_as_q = 0
    header_boundary = 0
    header_miss = 0
    other_confusions = 0

    for rec in records:
        for pair in rec.pairs:
            if pair.category == ErrorCategory.TYPE_ERROR and pair.gold and pair.pred:
                if pair.gold.entity_type == "QUESTION" and pair.pred.entity_type == "ANSWER":
                    q_as_a += 1
                elif pair.gold.entity_type == "ANSWER" and pair.pred.entity_type == "QUESTION":
                    a_as_q += 1
                else:
                    other_confusions += 1
            elif pair.gold and pair.gold.entity_type == "HEADER":
                if pair.category == ErrorCategory.BOUNDARY_ERROR:
                    header_boundary += 1
                elif pair.category == ErrorCategory.FALSE_NEGATIVE:
                    header_miss += 1

    return {
        "question_predicted_as_answer": q_as_a,
        "answer_predicted_as_question": a_as_q,
        "header_boundary_errors": header_boundary,
        "header_misses": header_miss,
        "other_type_confusions": other_confusions,
    }


def analyze_cord_errors(records: List[DocumentErrorRecord]) -> Dict[str, Any]:
    """
    Diagnostic analysis for CORD simplified 3-class task (MENU, SUBTOTAL, TOTAL):
    MENU vs SUBTOTAL, MENU vs TOTAL, SUBTOTAL vs TOTAL confusion.
    """
    menu_as_subtotal = 0
    menu_as_total = 0
    subtotal_as_total = 0
    total_as_subtotal = 0
    total_as_menu = 0
    subtotal_as_menu = 0
    boundary_errors = 0

    for rec in records:
        for pair in rec.pairs:
            if pair.category == ErrorCategory.BOUNDARY_ERROR:
                boundary_errors += 1
            elif pair.category == ErrorCategory.TYPE_ERROR and pair.gold and pair.pred:
                gt = pair.gold.entity_type
                pt = pair.pred.entity_type
                if gt == "MENU" and pt == "SUBTOTAL":
                    menu_as_subtotal += 1
                elif gt == "MENU" and pt == "TOTAL":
                    menu_as_total += 1
                elif gt == "SUBTOTAL" and pt == "TOTAL":
                    subtotal_as_total += 1
                elif gt == "TOTAL" and pt == "SUBTOTAL":
                    total_as_subtotal += 1
                elif gt == "TOTAL" and pt == "MENU":
                    total_as_menu += 1
                elif gt == "SUBTOTAL" and pt == "MENU":
                    subtotal_as_menu += 1

    return {
        "menu_predicted_as_subtotal": menu_as_subtotal,
        "menu_predicted_as_total": menu_as_total,
        "subtotal_predicted_as_total": subtotal_as_total,
        "total_predicted_as_subtotal": total_as_subtotal,
        "total_predicted_as_menu": total_as_menu,
        "subtotal_predicted_as_menu": subtotal_as_menu,
        "boundary_errors": boundary_errors,
    }
