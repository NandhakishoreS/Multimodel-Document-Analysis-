"""
LayoutLMv3 tokenizer wrapper with correct word-to-subword label and bbox alignment.

Key design decisions:
- LayoutLMv3Processor.apply_ocr=False: we supply Phase-2 words & bboxes directly.
- For every word split into N subword tokens, the first subword gets the true
  word label and subsequent subwords get ignore_index (-100) so the loss ignores
  them.  This is the standard HuggingFace NER label-alignment convention.
- Bounding boxes are duplicated for each subword of the same word (same spatial
  position).  Special tokens ([CLS], [SEP], padding) receive bbox [0,0,0,0].
- Images are handled by the processor's built-in image feature extractor
  (LayoutLMv3ImageProcessor), which resizes to 224×224 and normalises.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import LayoutLMv3Processor

logger = logging.getLogger(__name__)

# Label id assigned to sub-word continuation tokens and special tokens.
# CrossEntropyLoss ignores positions where target == -100 by default.
IGNORE_INDEX: int = -100

# Bbox placeholder for special tokens ([CLS], [SEP], <pad>).
PAD_BBOX: List[int] = [0, 0, 0, 0]


def align_labels_to_tokens(
    word_labels: List[str],
    word_ids: List[Optional[int]],
    label2id: Dict[str, int],
    ignore_index: int = IGNORE_INDEX,
) -> List[int]:
    """
    Align word-level string labels to subword-level integer label ids.

    Parameters
    ----------
    word_labels : List[str]
        One label per *word* (e.g. ["O", "B-COMPANY", "I-COMPANY", ...]).
    word_ids : List[Optional[int]]
        Output of `encoding.word_ids()` — maps each token position to the
        word index it originated from.  None for special tokens.
    label2id : Dict[str, int]
        Label string → integer id.
    ignore_index : int
        Value to assign to continuation sub-words and special tokens.

    Returns
    -------
    List[int]
        Token-level integer labels, same length as word_ids.

    Notes
    -----
    Only the *first* subword of each word receives the word's label.
    All subsequent subwords for the same word get ignore_index.
    Special tokens (None word_id) also get ignore_index.

    Example
    -------
    word_labels = ["O", "B-TOTAL"]
    word "25.00" → tokens ["25", ".", "00"]  (word_id 1 three times)
    result       → [ignore, label_id("O"), label_id("B-TOTAL"), ignore, ignore, ignore]
                    ^CLS      ^25           ^.                    ^.cont  ^00     ^SEP
    """
    token_labels: List[int] = []
    previous_word_id: Optional[int] = None

    for word_id in word_ids:
        if word_id is None:
            # Special token: [CLS], [SEP], <pad>
            token_labels.append(ignore_index)
        elif word_id != previous_word_id:
            # First subword of this word → use the word's label
            if word_id < len(word_labels):
                lbl_str = word_labels[word_id]
                token_labels.append(label2id.get(lbl_str, label2id.get("O", 0)))
            else:
                # word_id out of range (should not happen with well-formed data)
                logger.warning(
                    "word_id %d exceeds word_labels length %d — assigning O",
                    word_id, len(word_labels),
                )
                token_labels.append(label2id.get("O", 0))
        else:
            # Continuation subword of the same word → ignore
            token_labels.append(ignore_index)

        previous_word_id = word_id

    return token_labels


def align_bboxes_to_tokens(
    word_bboxes: List[List[int]],
    word_ids: List[Optional[int]],
    pad_bbox: List[int] = PAD_BBOX,
) -> List[List[int]]:
    """
    Propagate word-level bounding boxes to every subword token of that word.

    Special tokens receive pad_bbox ([0, 0, 0, 0]).

    Parameters
    ----------
    word_bboxes : List[List[int]]
        One [x0, y0, x1, y1] bbox per word (normalized to [0, 1000]).
    word_ids : List[Optional[int]]
        Token → word index mapping from the tokenizer.
    pad_bbox : List[int]
        Bbox to assign to special / padding tokens.

    Returns
    -------
    List[List[int]]
        Token-level bboxes, same length as word_ids.
    """
    token_bboxes: List[List[int]] = []
    for word_id in word_ids:
        if word_id is None or word_id >= len(word_bboxes):
            token_bboxes.append(list(pad_bbox))
        else:
            token_bboxes.append(list(word_bboxes[word_id]))
    return token_bboxes


class LayoutLMv3TokenizerWrapper:
    """
    Wraps the HuggingFace LayoutLMv3Processor for Phase 4 document tokenization.

    Handles:
    - Loading the processor (tokenizer + image processor) from a checkpoint.
    - Calling the processor with apply_ocr=False using Phase-2 OCR words & bboxes.
    - Subword label alignment (first-token rule, ignore_index for continuations).
    - Bbox propagation to every subword token.
    - Image loading and resizing via the processor's built-in image pipeline.
    - Correct tensor shapes for LayoutLMv3ForTokenClassification.

    Usage
    -----
    wrapper = LayoutLMv3TokenizerWrapper.from_pretrained("microsoft/layoutlmv3-base")
    encoding = wrapper.encode_sample(sample, label2id=label2id)
    # encoding keys: input_ids, attention_mask, bbox, pixel_values, labels
    """

    def __init__(self, processor: LayoutLMv3Processor, max_seq_length: int = 512) -> None:
        self.processor = processor
        self.max_seq_length = max_seq_length

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = "microsoft/layoutlmv3-base",
        max_seq_length: int = 512,
    ) -> "LayoutLMv3TokenizerWrapper":
        """Load processor from a HuggingFace checkpoint or local directory."""
        logger.info("Loading LayoutLMv3Processor from '%s'…", checkpoint)
        processor = LayoutLMv3Processor.from_pretrained(
            checkpoint,
            apply_ocr=False,  # We supply OCR words & bboxes from Phase 2.
        )
        logger.info("Processor loaded.")
        return cls(processor=processor, max_seq_length=max_seq_length)

    # ------------------------------------------------------------------
    # Core encode method
    # ------------------------------------------------------------------

    def encode_sample(
        self,
        sample: Dict[str, Any],
        label2id: Optional[Dict[str, int]] = None,
        project_root: Optional[Path] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, Any]:
        """
        Tokenize one sample dict from the Phase-2 JSONL schema.

        Parameters
        ----------
        sample : dict
            Keys: id, dataset, split, image_path, width, height,
                  words, bboxes, labels, entities, relations.
        label2id : dict or None
            Label string → int id.  If None, labels tensor is omitted.
        project_root : Path or None
            Prepended to image_path (stored as relative Posix path in JSONL).
        return_tensors : str
            "pt" (default) for PyTorch tensors, "np" for numpy.

        Returns
        -------
        dict with keys:
            input_ids        : (1, seq_len) int64
            attention_mask   : (1, seq_len) int64
            bbox             : (1, seq_len, 4) int64
            pixel_values     : (1, 3, 224, 224) float32
            labels           : (1, seq_len) int64  [only if label2id provided]
            word_ids         : List[Optional[int]]  [always included, not a tensor]
        """
        words: List[str] = sample.get("words", [])
        bboxes: List[List[int]] = sample.get("bboxes", [])
        word_labels: List[str] = sample.get("labels", ["O"] * len(words))

        # ---- Validate bbox count matches words ----
        if len(bboxes) != len(words):
            logger.warning(
                "Sample %s: bbox count %d != word count %d — padding/trimming bboxes",
                sample.get("id", "?"), len(bboxes), len(words),
            )
            bboxes = _reconcile_lengths(bboxes, len(words), PAD_BBOX)

        # ---- Validate bbox coordinates ----
        bboxes = [_clamp_bbox(b) for b in bboxes]

        # ---- Load image ----
        image = self._load_image(sample, project_root)

        # ---- Handle empty word list edge case ----
        if not words:
            words = ["[EMPTY]"]
            bboxes = [PAD_BBOX]
            word_labels = ["O"]

        # ---- Call processor ----
        # LayoutLMv3Processor(apply_ocr=False) expects:
        #   images, text (words list), boxes (bboxes), word_labels (optional)
        # It returns: input_ids, attention_mask, bbox, pixel_values
        # We do NOT pass word_labels to the processor — we align them ourselves
        # to use ignore_index for continuation tokens, which the processor does not do.
        encoding = self.processor(
            images=image,
            text=words,
            boxes=bboxes,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors,
        )

        # ---- Build word_ids list ----
        # word_ids() maps each token position to its originating word index.
        # Returns None for special tokens.
        word_ids: List[Optional[int]] = encoding.word_ids(batch_index=0)

        # ---- Align bboxes to token level ----
        # The processor already fills bbox for us, but we verify/override to
        # ensure special-token bboxes are exactly [0,0,0,0].
        aligned_bboxes = align_bboxes_to_tokens(bboxes, word_ids)
        if return_tensors == "pt":
            encoding["bbox"] = torch.tensor([aligned_bboxes], dtype=torch.long)
        else:
            import numpy as np
            encoding["bbox"] = np.array([aligned_bboxes])

        # ---- Align labels ----
        result: Dict[str, Any] = dict(encoding)
        result["word_ids"] = word_ids

        if label2id is not None:
            token_label_ids = align_labels_to_tokens(
                word_labels=word_labels,
                word_ids=word_ids,
                label2id=label2id,
            )
            if return_tensors == "pt":
                result["labels"] = torch.tensor([token_label_ids], dtype=torch.long)
            else:
                import numpy as np
                result["labels"] = np.array([token_label_ids])

        return result

    def encode_batch(
        self,
        samples: List[Dict[str, Any]],
        label2id: Optional[Dict[str, int]] = None,
        project_root: Optional[Path] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a list of samples and stack into a batched dict of tensors.

        Returns
        -------
        dict with keys: input_ids, attention_mask, bbox, pixel_values, labels
        Each tensor has shape (batch_size, seq_len) or (batch_size, 3, 224, 224).
        """
        encoded = [
            self.encode_sample(s, label2id=label2id, project_root=project_root)
            for s in samples
        ]
        batch: Dict[str, Any] = {}
        tensor_keys = ["input_ids", "attention_mask", "bbox", "pixel_values"]
        if label2id is not None:
            tensor_keys.append("labels")

        for key in tensor_keys:
            tensors = [e[key].squeeze(0) for e in encoded if key in e]
            if tensors:
                batch[key] = torch.stack(tensors, dim=0)

        return batch

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_image(
        self,
        sample: Dict[str, Any],
        project_root: Optional[Path],
    ) -> Image.Image:
        """
        Load the document image from image_path stored in the sample dict.

        The JSONL stores image_path as a relative Posix path from the project root.
        """
        img_path_str: str = sample.get("image_path", "")
        img_path = Path(img_path_str)

        if not img_path.is_absolute() and project_root is not None:
            img_path = project_root / img_path

        if img_path.exists():
            try:
                return Image.open(img_path).convert("RGB")
            except Exception as exc:
                logger.warning("Could not open image %s: %s — using blank image", img_path, exc)

        # Fallback: blank white image at recorded dimensions
        w = sample.get("width", 224)
        h = sample.get("height", 224)
        logger.warning(
            "Image not found for sample %s (%s) — using blank %dx%d RGB image",
            sample.get("id", "?"), img_path, w, h,
        )
        return Image.new("RGB", (w, h), color=(255, 255, 255))

    def save_pretrained(self, save_dir: Union[str, Path]) -> None:
        """Save the processor to a local directory (for checkpointing)."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.processor.save_pretrained(str(save_dir))
        logger.info("Processor saved to %s", save_dir)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _clamp_bbox(bbox: List[int]) -> List[int]:
    """
    Clamp all bbox coordinates to [0, 1000].
    Ensures x0 <= x1 and y0 <= y1.
    Does NOT raise — malformed boxes are silently corrected and logged at DEBUG.
    """
    if len(bbox) != 4:
        return [0, 0, 1, 1]
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(int(x0), 1000))
    y0 = max(0, min(int(y0), 1000))
    x1 = max(0, min(int(x1), 1000))
    y1 = max(0, min(int(y1), 1000))
    # Fix inverted coordinates
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    # Ensure non-zero area
    if x1 == x0:
        x1 = min(1000, x0 + 1)
    if y1 == y0:
        y1 = min(1000, y0 + 1)
    return [x0, y0, x1, y1]


def _reconcile_lengths(
    lst: List[Any],
    target_len: int,
    pad_value: Any,
) -> List[Any]:
    """Trim or pad a list to target_len."""
    if len(lst) >= target_len:
        return lst[:target_len]
    return lst + [pad_value] * (target_len - len(lst))
