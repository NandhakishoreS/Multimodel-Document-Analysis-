"""
LayoutLMv3 inference pipeline: entity reconstruction from token predictions.

Accepts a document (image + words + bboxes) and returns:
- Per-token predictions with prediction_probability
- Reconstructed entity spans (BIO → entity string)

Designed to be Streamlit-independent.  The Streamlit app will import
`predict_document()` and render the results.

Entity reconstruction rules:
- B-<TYPE>  → start new entity span
- I-<TYPE>  → continue entity if preceded by B-<TYPE> or I-<TYPE> of same type
- I-<TYPE>  → start new entity if not preceded by matching B/I (broken BIO)
- O         → end any open span
- Subword tokens from the same word are merged back to word-level before reconstruction
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

# Ensure project root on sys.path when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.layoutlmv3.config import LayoutLMv3DatasetConfig, load_dataset_config
from models.layoutlmv3.model import LayoutLMv3Classifier
from models.layoutlmv3.tokenizer import LayoutLMv3TokenizerWrapper, IGNORE_INDEX

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity reconstruction from word-level BIO predictions
# ---------------------------------------------------------------------------


def reconstruct_entities(
    words: List[str],
    word_labels: List[str],
    word_probs: Optional[List[float]] = None,
    entity_groups: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Convert word-level BIO label sequences into structured entity dicts.

    Parameters
    ----------
    words : List[str]
        OCR words (same order as in the JSONL sample).
    word_labels : List[str]
        Predicted BIO label per word (e.g. ["O", "B-COMPANY", "I-COMPANY", ...]).
    word_probs : List[float] or None
        Prediction probability per word (softmax of the argmax class).
    entity_groups : Dict[str, List[str]] or None
        Maps entity_type → [B-TYPE, I-TYPE] from the label config.
        If None, derived automatically from the label strings.

    Returns
    -------
    dict with keys:
        entities : Dict[str, str]
            Entity type → concatenated text (e.g. {"company": "SINNATHAMBY ...", ...})
            When the same entity type appears more than once, values are joined
            by a space.
        spans : List[Dict]
            Detailed span records:
            [{"entity_type": str, "text": str, "start_word": int, "end_word": int,
              "mean_probability": float}, ...]
        token_predictions : List[Dict]
            Per-word output:
            [{"word": str, "label": str, "probability": float}, ...]
    """
    assert len(words) == len(word_labels), (
        f"words length {len(words)} != word_labels length {len(word_labels)}"
    )
    if word_probs is None:
        word_probs = [1.0] * len(words)
    assert len(word_probs) == len(words)

    # Build token_predictions output.
    token_predictions = [
        {"word": w, "label": lbl, "prediction_probability": round(float(p), 4)}
        for w, lbl, p in zip(words, word_labels, word_probs)
    ]

    # Extract spans using BIO logic.
    spans: List[Dict[str, Any]] = []
    entities_accumulator: Dict[str, List[str]] = {}  # entity_type → list of span texts

    current_type: Optional[str] = None
    current_words: List[str] = []
    current_probs: List[float] = []
    current_start: int = 0

    def _flush_span(end_idx: int) -> None:
        nonlocal current_type, current_words, current_probs, current_start
        if current_type and current_words:
            span_text = " ".join(current_words)
            mean_prob = round(float(sum(current_probs) / len(current_probs)), 4)
            spans.append({
                "entity_type": current_type,
                "text": span_text,
                "start_word": current_start,
                "end_word": end_idx - 1,
                "mean_probability": mean_prob,
            })
            entities_accumulator.setdefault(current_type, []).append(span_text)
        current_type = None
        current_words = []
        current_probs = []
        current_start = 0

    for idx, (word, label, prob) in enumerate(zip(words, word_labels, word_probs)):
        if label == "O" or label is None:
            _flush_span(idx)
        elif label.startswith("B-"):
            _flush_span(idx)
            current_type = label[2:]   # Strip "B-" prefix
            current_words = [word]
            current_probs = [prob]
            current_start = idx
        elif label.startswith("I-"):
            i_type = label[2:]
            if current_type == i_type:
                # Valid continuation
                current_words.append(word)
                current_probs.append(prob)
            else:
                # Broken BIO: I without matching B — start new span
                _flush_span(idx)
                current_type = i_type
                current_words = [word]
                current_probs = [prob]
                current_start = idx
        else:
            # Unknown label format — treat as O
            _flush_span(idx)

    _flush_span(len(words))  # Flush any open span at end of sequence

    # Merge multiple spans of same type into a single string.
    entities: Dict[str, str] = {
        etype: " ".join(texts)
        for etype, texts in entities_accumulator.items()
    }

    return {
        "entities": entities,
        "spans": spans,
        "token_predictions": token_predictions,
    }


# ---------------------------------------------------------------------------
# Word-level predictions from token-level model output
# ---------------------------------------------------------------------------


def token_preds_to_word_preds(
    token_label_ids: List[int],
    token_probs: List[float],
    word_ids: List[Optional[int]],
    id2label: Dict[int, str],
    num_words: int,
) -> Tuple[List[str], List[float]]:
    """
    Collapse token-level predictions back to word-level.

    For each word, use the prediction of its *first* subword token
    (the same convention used during label alignment at training time).

    Parameters
    ----------
    token_label_ids : List[int]  — argmax label id per token position
    token_probs     : List[float] — max softmax probability per token position
    word_ids        : List[Optional[int]] — word_ids from the tokenizer
    id2label        : Dict[int, str]
    num_words       : int — expected number of output word predictions

    Returns
    -------
    (word_labels, word_probs) — each of length num_words
    """
    word_labels: Dict[int, str] = {}
    word_probs_map: Dict[int, float] = {}
    seen_word_ids: set = set()

    for tok_idx, (lid, prob, wid) in enumerate(zip(token_label_ids, token_probs, word_ids)):
        if wid is None:
            continue  # Special token
        if wid not in seen_word_ids:
            word_labels[wid] = id2label.get(lid, "O")
            word_probs_map[wid] = prob
            seen_word_ids.add(wid)

    # Fill gaps (truncated sequences may not cover all words).
    out_labels = [word_labels.get(i, "O") for i in range(num_words)]
    out_probs  = [word_probs_map.get(i, 0.0) for i in range(num_words)]
    return out_labels, out_probs


# ---------------------------------------------------------------------------
# High-level document prediction entry point
# ---------------------------------------------------------------------------


def predict_document(
    sample: Dict[str, Any],
    classifier: LayoutLMv3Classifier,
    tokenizer_wrapper: LayoutLMv3TokenizerWrapper,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run LayoutLMv3 inference on a single document sample dict.

    Parameters
    ----------
    sample : dict
        Phase-2 JSONL sample (keys: id, words, bboxes, labels, image_path, …).
    classifier : LayoutLMv3Classifier
    tokenizer_wrapper : LayoutLMv3TokenizerWrapper
    project_root : Path or None

    Returns
    -------
    dict with keys:
        sample_id          : str
        words              : List[str]
        word_labels_true   : List[str]   — ground-truth (from sample["labels"])
        word_labels_pred   : List[str]   — model predictions
        word_probabilities : List[float] — per-word max softmax probability
        entities           : Dict[str, str]   — reconstructed entities
        spans              : List[Dict]       — detailed span records
        token_predictions  : List[Dict]       — per-word word/label/prob
    """
    words: List[str] = sample.get("words", [])
    true_labels: List[str] = sample.get("labels", ["O"] * len(words))
    num_words = len(words)

    # Tokenize (no label alignment needed for inference — pass label2id=None
    # to skip label tensor, but we still need word_ids for decoding).
    encoding = tokenizer_wrapper.encode_sample(
        sample=sample,
        label2id=None,   # No label tensor during inference
        project_root=project_root,
        return_tensors="pt",
    )
    word_ids: List[Optional[int]] = encoding["word_ids"]

    # Move tensors to model device and run inference.
    classifier.eval_mode()
    with torch.no_grad():
        pred_ids, probs = classifier.predict_batch(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            bbox=encoding["bbox"],
            pixel_values=encoding["pixel_values"],
        )

    # pred_ids shape: (1, seq_len); probs shape: (1, seq_len, num_labels)
    token_label_ids: List[int] = pred_ids[0].cpu().tolist()
    token_max_probs: List[float] = probs[0].max(dim=-1).values.cpu().tolist()

    # Collapse to word level.
    word_labels_pred, word_probs = token_preds_to_word_preds(
        token_label_ids=token_label_ids,
        token_probs=token_max_probs,
        word_ids=word_ids,
        id2label=classifier.dataset_config.id2label,
        num_words=num_words,
    )

    # Reconstruct entities from word-level BIO predictions.
    entity_groups = dict(classifier.dataset_config.entity_groups)
    reconstruction = reconstruct_entities(
        words=words,
        word_labels=word_labels_pred,
        word_probs=word_probs,
        entity_groups=entity_groups,
    )

    return {
        "sample_id":          sample.get("id", "unknown"),
        "words":              words,
        "word_labels_true":   true_labels,
        "word_labels_pred":   word_labels_pred,
        "word_probabilities": word_probs,
        "entities":           reconstruction["entities"],
        "spans":              reconstruction["spans"],
        "token_predictions":  reconstruction["token_predictions"],
    }


def predict_batch_documents(
    samples: List[Dict[str, Any]],
    classifier: LayoutLMv3Classifier,
    tokenizer_wrapper: LayoutLMv3TokenizerWrapper,
    project_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Run predict_document() on a list of samples.

    Returns
    -------
    List of result dicts, one per sample.
    """
    results = []
    for sample in samples:
        try:
            result = predict_document(
                sample=sample,
                classifier=classifier,
                tokenizer_wrapper=tokenizer_wrapper,
                project_root=project_root,
            )
            results.append(result)
        except Exception as exc:
            logger.error("Prediction failed for sample %s: %s", sample.get("id", "?"), exc)
            results.append({
                "sample_id": sample.get("id", "?"),
                "words": sample.get("words", []),
                "word_labels_true": sample.get("labels", []),
                "word_labels_pred": ["O"] * len(sample.get("words", [])),
                "word_probabilities": [0.0] * len(sample.get("words", [])),
                "entities": {},
                "spans": [],
                "token_predictions": [],
                "error": str(exc),
            })
    return results
