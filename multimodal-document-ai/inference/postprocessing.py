"""
Postprocessing utilities: BIO decoding, bounding box aggregation, and confidence calculation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from inference.schemas import ExtractedEntity, TokenPrediction


def union_bboxes(bboxes: List[List[int]]) -> List[int]:
    """
    Compute minimum bounding box [x0, y0, x1, y1] enclosing all input boxes.
    """
    if not bboxes:
        return [0, 0, 0, 0]
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return [int(x0), int(y0), int(x1), int(y1)]


def collapse_token_preds_to_words(
    token_label_ids: List[int],
    token_probs: List[float],
    word_ids: List[Optional[int]],
    id2label: Dict[int, str],
    num_words: int,
) -> Tuple[List[str], List[float]]:
    """
    Collapse LayoutLMv3 subword token predictions to word-level using the first-subword convention.

    Parameters
    ----------
    token_label_ids : List[int]
        Predicted label ID for each subword token.
    token_probs : List[float]
        Softmax probability of the predicted label for each subword token.
    word_ids : List[Optional[int]]
        Tokenizer word index mapping (None for special tokens).
    id2label : Dict[int, str]
        Mapping from class index to BIO label name.
    num_words : int
        Number of original words extracted by OCR.

    Returns
    -------
    (word_labels, word_probs) each of length num_words
    """
    word_labels: Dict[int, str] = {}
    word_probs_map: Dict[int, float] = {}
    seen_words: set = set()

    for lid, prob, wid in zip(token_label_ids, token_probs, word_ids):
        if wid is None:
            continue
        if wid not in seen_words:
            word_labels[wid] = id2label.get(lid, "O")
            word_probs_map[wid] = float(prob)
            seen_words.add(wid)

    # Fill any unpredicted words (e.g. truncated beyond 512 tokens) with background 'O'
    out_labels = [word_labels.get(i, "O") for i in range(num_words)]
    out_probs = [word_probs_map.get(i, 0.0) for i in range(num_words)]
    return out_labels, out_probs


def reconstruct_entity_spans(
    words: List[str],
    word_labels: List[str],
    word_probs: List[float],
    original_bboxes: List[List[int]],
    normalized_bboxes: List[List[int]],
) -> Tuple[List[ExtractedEntity], List[TokenPrediction], Dict[str, str]]:
    """
    Decode BIO label sequences into ExtractedEntity objects with bounding box unions and confidence scores.

    Parameters
    ----------
    words : List[str]
        OCR extracted word strings.
    word_labels : List[str]
        Predicted BIO labels (e.g. 'B-COMPANY', 'I-COMPANY', 'O').
    word_probs : List[float]
        Per-word confidence scores from model softmax.
    original_bboxes : List[List[int]]
        Bounding boxes in original image pixel coordinates [x0, y0, x1, y1].
    normalized_bboxes : List[List[int]]
        Normalized bounding boxes [0, 1000].

    Returns
    -------
    entities : List[ExtractedEntity]
    tokens : List[TokenPrediction]
    aggregated_dict : Dict[str, str]
    """
    num_words = len(words)
    tokens: List[TokenPrediction] = [
        TokenPrediction(
            word=w,
            label=lbl,
            confidence=round(float(p), 4),
            bbox=b,
            word_index=i,
        )
        for i, (w, lbl, p, b) in enumerate(zip(words, word_labels, word_probs, original_bboxes))
    ]

    entities: List[ExtractedEntity] = []
    aggregated_dict: Dict[str, List[str]] = {}

    current_type: Optional[str] = None
    current_words: List[str] = []
    current_probs: List[float] = []
    current_orig_boxes: List[List[int]] = []
    current_norm_boxes: List[List[int]] = []
    current_start: int = 0

    def _flush_span(end_idx: int) -> None:
        nonlocal current_type, current_words, current_probs
        nonlocal current_orig_boxes, current_norm_boxes, current_start
        if current_type and current_words:
            span_text = " ".join(current_words)
            mean_prob = sum(current_probs) / len(current_probs)
            min_prob = min(current_probs)
            orig_bbox = union_bboxes(current_orig_boxes)
            norm_bbox = union_bboxes(current_norm_boxes)

            entity = ExtractedEntity(
                label=current_type,
                text=span_text,
                confidence=round(float(mean_prob), 4),
                min_confidence=round(float(min_prob), 4),
                bbox=orig_bbox,
                normalized_bbox=norm_bbox,
                start_word=current_start,
                end_word=end_idx - 1,
                token_count=len(current_words),
            )
            entities.append(entity)
            aggregated_dict.setdefault(current_type, []).append(span_text)

        current_type = None
        current_words = []
        current_probs = []
        current_orig_boxes = []
        current_norm_boxes = []
        current_start = 0

    for idx, (word, label, prob, obox, nbox) in enumerate(
        zip(words, word_labels, word_probs, original_bboxes, normalized_bboxes)
    ):
        if label == "O" or not label:
            _flush_span(idx)
        elif label.startswith("B-"):
            _flush_span(idx)
            current_type = label[2:]
            current_words = [word]
            current_probs = [prob]
            current_orig_boxes = [obox]
            current_norm_boxes = [nbox]
            current_start = idx
        elif label.startswith("I-"):
            i_type = label[2:]
            if current_type == i_type:
                # Valid BIO continuation
                current_words.append(word)
                current_probs.append(prob)
                current_orig_boxes.append(obox)
                current_norm_boxes.append(nbox)
            else:
                # Broken BIO: I without matching B
                _flush_span(idx)
                current_type = i_type
                current_words = [word]
                current_probs = [prob]
                current_orig_boxes = [obox]
                current_norm_boxes = [nbox]
                current_start = idx
        else:
            _flush_span(idx)

    _flush_span(num_words)

    # Flatten aggregated spans into space-joined string per entity type
    flat_aggregated: Dict[str, str] = {
        etype: " ".join(texts)
        for etype, texts in aggregated_dict.items()
    }

    return entities, tokens, flat_aggregated
