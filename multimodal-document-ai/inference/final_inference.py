"""
Final Unseen Document Inference Pipeline.

Performs end-to-end multimodal information extraction:
Input Image -> OCR -> Bounding-box Extraction -> LayoutLMv3 Encoding ->
Model Inference -> Word BIO Decoding -> Entity Aggregation -> Confidence Calculation -> Structured Output
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor

from models.layoutlmv3.config import LayoutLMv3DatasetConfig, load_dataset_config
from models.layoutlmv3.model import LayoutLMv3Classifier
from models.layoutlmv3.tokenizer import LayoutLMv3TokenizerWrapper
from preprocessing.bbox_utils import normalize_bbox
from inference.schemas import DocumentInferenceResult, ExtractedEntity, TokenPrediction
from inference.postprocessing import collapse_token_preds_to_words, reconstruct_entity_spans

logger = logging.getLogger(__name__)


# Checkpoint directory registry
DEFAULT_CHECKPOINTS: Dict[str, Dict[str, str]] = {
    "sroie": {
        "primary": "models/checkpoints/phase9/sroie/joint_aug/best",
        "baseline": "models/checkpoints/layoutlmv3/sroie/corrected/best",
    },
    "funsd": {
        "primary": "models/checkpoints/layoutlmv3/funsd/full",
        "baseline": "models/checkpoints/layoutlmv3/funsd/full",
    },
    "cord": {
        "primary": "models/checkpoints/layoutlmv3/cord/full",
        "baseline": "models/checkpoints/layoutlmv3/cord/full",
    },
}


def split_cord_ocr_words(text: str) -> List[str]:
    """
    Split CORD OCR text line into individual word tokens.

    CORD receipt tables frequently contain standalone row numbers and quantities
    (e.g., '1 EGG', '2 TWIST', '4 Vivamus') where tight inter-character kerning causes
    OCR engines to concatenate leading digits to adjacent words (e.g. '1EGG', '2CHOCO',
    '1BANANA', '4Vivamus', '2PCS') or append punctuation (e.g. '1.', '4.', '(1)').
    This function cleanly separates these standalone numeric tokens so they are available
    as distinct word tokens to the multimodal model.
    """
    raw_tokens = text.split()
    refined: List[str] = []
    for token in raw_tokens:
        # Match table row numbering with trailing dot (e.g. '1.' -> '1', '4.' -> '4')
        m_dot = re.match(r"^(\d+)\.$", token)
        # Match parenthesized digits (e.g. '(1)' -> '1', '(4)' -> '4')
        m_paren = re.match(r"^\((\d+)\)$", token)
        # Match leading digits glued to letters, excluding single multiplier units like '1x', '2X'
        m_lead = re.match(r"^(\d+)([A-Za-z].*)$", token)

        if m_dot:
            refined.append(m_dot.group(1))
        elif m_paren:
            refined.append(m_paren.group(1))
        elif m_lead and m_lead.group(2).lower() not in ("x", "k", "l"):
            refined.append(m_lead.group(1))
            refined.append(m_lead.group(2))
        else:
            refined.append(token)
    return refined


class EmptyOCRError(ValueError):
    """Raised when OCR detects zero text in an input document."""
    pass


class UnseenDocumentInferencePipeline:
    """
    Production-ready inference pipeline for previously unseen document images.
    """

    def __init__(
        self,
        dataset_type: str = "sroie",
        use_robust_model: bool = True,
        device: Optional[str] = None,
        project_root: Optional[Union[str, Path]] = None,
    ):
        self.dataset_type = dataset_type.lower()
        if self.dataset_type not in DEFAULT_CHECKPOINTS:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}. Supported: sroie, funsd, cord")

        self.use_robust_model = use_robust_model
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent

        # Select target checkpoint
        variant = "primary" if use_robust_model else "baseline"
        rel_ckpt = DEFAULT_CHECKPOINTS[self.dataset_type][variant]
        self.checkpoint_dir = self.project_root / rel_ckpt

        if not self.checkpoint_dir.exists():
            # Fallback to baseline if primary path not found
            fallback_rel = DEFAULT_CHECKPOINTS[self.dataset_type]["baseline"]
            self.checkpoint_dir = self.project_root / fallback_rel

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Lazy loaded components
        self._dataset_config: Optional[LayoutLMv3DatasetConfig] = None
        self._classifier: Optional[LayoutLMv3Classifier] = None
        self._tokenizer: Optional[LayoutLMv3TokenizerWrapper] = None
        self._image_processor: Optional[AutoImageProcessor] = None
        self._ocr_engine: Optional[Any] = None

    def _load_model_components(self) -> None:
        """Initialize and load the model, tokenizer, and image processor."""
        if self._classifier is not None:
            return

        logger.info("Loading configuration for %s...", self.dataset_type)
        self._dataset_config = load_dataset_config(self.dataset_type)

        logger.info("Loading LayoutLMv3 classifier from %s...", self.checkpoint_dir)
        self._classifier = LayoutLMv3Classifier(
            dataset_config=self._dataset_config,
            checkpoint=str(self.checkpoint_dir),
            device=self.device,
        )
        self._classifier.model.eval()

        proc_dir = self.checkpoint_dir / "processor"
        proc_src = str(proc_dir) if proc_dir.exists() else "microsoft/layoutlmv3-base"

        logger.info("Loading tokenizer from %s...", proc_src)
        self._tokenizer = LayoutLMv3TokenizerWrapper.from_pretrained(
            proc_src,
            max_seq_length=512,
        )

        logger.info("Loading image processor from %s...", proc_src)
        try:
            self._image_processor = AutoImageProcessor.from_pretrained(proc_src)
        except Exception:
            self._image_processor = AutoImageProcessor.from_pretrained("microsoft/layoutlmv3-base")

    def _get_ocr_engine(self) -> Any:
        """Initialize RapidOCR engine on demand."""
        if self._ocr_engine is None:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr_engine = RapidOCR()
        return self._ocr_engine

    def run_ocr(
        self, image: Image.Image
    ) -> Tuple[List[str], List[List[int]], List[List[int]], str]:
        """
        Execute OCR detection and recognition on a document image.

        Returns
        -------
        words : List[str]
        original_bboxes : List[List[int]] ([x0, y0, x1, y1] in image pixel space)
        normalized_bboxes : List[List[int]] ([0, 1000] space)
        raw_text : str (full OCR text transcript)
        """
        ocr = self._get_ocr_engine()
        img_np = np.array(image.convert("RGB"))
        ocr_results, _ = ocr(img_np)

        if not ocr_results:
            return [], [], [], ""

        width, height = image.size
        words: List[str] = []
        original_bboxes: List[List[int]] = []
        normalized_bboxes: List[List[int]] = []
        raw_lines: List[str] = []

        for item in ocr_results:
            # item format: [polygon_coords, text, confidence]
            poly, text, conf = item
            text = str(text).strip()
            if not text:
                continue

            raw_lines.append(text)
            poly_np = np.array(poly)
            x_min = int(np.min(poly_np[:, 0]))
            y_min = int(np.min(poly_np[:, 1]))
            x_max = int(np.max(poly_np[:, 0]))
            y_max = int(np.max(poly_np[:, 1]))

            # Ensure valid rectangle
            x_min = max(0, min(x_min, width - 1))
            y_min = max(0, min(y_min, height - 1))
            x_max = max(x_min + 1, min(x_max, width))
            y_max = max(y_min + 1, min(y_max, height))

            # Split line into individual words
            if self.dataset_type == "cord":
                line_words = split_cord_ocr_words(text)
            else:
                line_words = text.split()
            if not line_words:
                continue

            # Interpolate word bounding boxes horizontally across the line
            line_w = x_max - x_min
            total_chars = max(1, sum(len(w) for w in line_words) + len(line_words) - 1)
            char_w = line_w / total_chars

            cur_x = x_min
            for w in line_words:
                w_len = len(w)
                w_x0 = int(cur_x)
                w_x1 = int(min(x_max, cur_x + w_len * char_w))
                w_box = [w_x0, y_min, max(w_x0 + 1, w_x1), y_max]
                norm_box = normalize_bbox(w_box, width, height)

                words.append(w)
                original_bboxes.append(w_box)
                normalized_bboxes.append(norm_box)

                cur_x += (w_len + 1) * char_w

        return words, original_bboxes, normalized_bboxes, "\n".join(raw_lines)

    def get_raw_ocr_debug(
        self, image_input: Union[Image.Image, str, Path]
    ) -> Dict[str, Any]:
        """
        Extract detailed debug information from raw OCR detection and recognition.

        Returns
        -------
        Dict containing:
        - total_raw_lines: int
        - raw_ocr_items: List of dicts with token_index, text, confidence, polygon, bbox
        - numeric_search: Detected matches for '1', '2', '3', '4' and common variants ('1.', '2.', '3.', '4.')
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError(f"Expected PIL.Image or path, got {type(image_input)}")

        ocr = self._get_ocr_engine()
        img_np = np.array(img)
        ocr_results, _ = ocr(img_np)

        raw_items = []
        numeric_matches: Dict[str, List[Dict[str, Any]]] = {
            "1": [], "2": [], "3": [], "4": [], "variants": []
        }
        target_digits = {"1", "2", "3", "4"}
        target_variants = {"1.", "2.", "3.", "4.", "1)", "2)", "3)", "4)", "(1)", "(2)", "(3)", "(4)"}

        for idx, item in enumerate(ocr_results or []):
            poly, text, conf = item
            text_str = str(text).strip()
            poly_np = np.array(poly)
            poly_list = poly_np.tolist() if hasattr(poly_np, "tolist") else []
            bbox = [
                int(np.min(poly_np[:, 0])),
                int(np.min(poly_np[:, 1])),
                int(np.max(poly_np[:, 0])),
                int(np.max(poly_np[:, 1])),
            ]
            raw_items.append({
                "token_index": idx,
                "text": text_str,
                "confidence": round(float(conf), 4),
                "polygon": poly_list,
                "bbox": bbox,
            })

            words = text_str.split()
            for w in words:
                if w in target_digits:
                    numeric_matches[w].append({"item_index": idx, "token": w, "text": text_str, "confidence": round(float(conf), 4), "bbox": bbox})
                elif w in target_variants:
                    numeric_matches["variants"].append({"item_index": idx, "token": w, "text": text_str, "confidence": round(float(conf), 4), "bbox": bbox})
                elif any(w.startswith(d) for d in target_digits) and len(w) > 1:
                    numeric_matches["variants"].append({"item_index": idx, "token": w, "text": text_str, "confidence": round(float(conf), 4), "bbox": bbox})

        return {
            "total_raw_lines": len(raw_items),
            "raw_ocr_items": raw_items,
            "numeric_search": numeric_matches,
        }

    @torch.no_grad()
    def predict_image(
        self,
        image_input: Union[Image.Image, str, Path],
        min_confidence: float = 0.0,
    ) -> DocumentInferenceResult:
        """
        Run end-to-end inference on an unseen document image.

        Parameters
        ----------
        image_input : PIL.Image or path to image
        min_confidence : float, threshold to filter low-confidence entity spans

        Returns
        -------
        DocumentInferenceResult
        """
        # Load and validate image
        image_path_str: Optional[str] = None
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image not found at {path}")
            image = Image.open(path).convert("RGB")
            image_path_str = str(path)
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError(f"Expected PIL.Image or path, got {type(image_input)}")

        self._load_model_components()

        # Step 1: Run OCR
        words, orig_boxes, norm_boxes, raw_text = self.run_ocr(image)

        if not words:
            # Handle empty document gracefully
            return DocumentInferenceResult(
                document_type=self.dataset_type.upper(),
                model_name=f"{self.dataset_type}_{'robust' if self.use_robust_model else 'baseline'}",
                image_path=image_path_str,
                image_size=image.size,
                ocr_word_count=0,
                entities=[],
                tokens=[],
                raw_entities_dict={},
                ocr_raw_text="",
            )

        # Step 2: Tokenize and align using processor
        encoding = self._tokenizer.processor(
            images=image,
            text=words,
            boxes=norm_boxes,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        bbox = encoding["bbox"].to(self.device)
        pixel_vals = encoding["pixel_values"].to(self.device)
        word_ids = encoding.word_ids(batch_index=0)

        # Step 3: Model forward pass
        outputs = self._classifier.forward(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            pixel_values=pixel_vals,
        )

        logits = outputs["logits"].squeeze(0)  # [seq_len, num_labels]
        probs = F.softmax(logits, dim=-1)
        max_probs, pred_ids = torch.max(probs, dim=-1)

        token_label_ids = pred_ids.cpu().tolist()
        token_probs = max_probs.cpu().tolist()

        # Step 5: Collapse subwords to word-level predictions
        word_labels, word_probs = collapse_token_preds_to_words(
            token_label_ids=token_label_ids,
            token_probs=token_probs,
            word_ids=word_ids,
            id2label=self._dataset_config.id2label,
            num_words=len(words),
        )

        # Step 6: Decode BIO entity spans with confidence
        entities, tokens, flat_agg = reconstruct_entity_spans(
            words=words,
            word_labels=word_labels,
            word_probs=word_probs,
            original_bboxes=orig_boxes,
            normalized_bboxes=norm_boxes,
        )

        if min_confidence > 0.0:
            entities = [e for e in entities if e.confidence >= min_confidence]

        return DocumentInferenceResult(
            document_type=self.dataset_type.upper(),
            model_name=f"{self.dataset_type}_{'robust' if self.use_robust_model else 'baseline'}",
            image_path=image_path_str,
            image_size=image.size,
            ocr_word_count=len(words),
            entities=entities,
            tokens=tokens,
            raw_entities_dict=flat_agg,
            ocr_raw_text=raw_text,
        )

    @torch.no_grad()
    def predict_sample(
        self,
        sample: Dict[str, Any],
        image: Optional[Image.Image] = None,
        min_confidence: float = 0.0,
    ) -> DocumentInferenceResult:
        """
        Run inference using pre-extracted words and bboxes (e.g. from JSONL benchmark samples).
        """
        self._load_model_components()

        words: List[str] = sample.get("words", [])
        norm_boxes: List[List[int]] = sample.get("bboxes", [])

        # Load image if not provided
        if image is None:
            img_path = sample.get("image_path")
            if img_path:
                full_path = self.project_root / img_path if not Path(img_path).is_absolute() else Path(img_path)
                image = Image.open(full_path).convert("RGB")
            else:
                image = Image.new("RGB", (1000, 1000), color="white")

        w_img, h_img = image.size
        # Denormalize boxes to original pixel space
        orig_boxes: List[List[int]] = []
        for nb in norm_boxes:
            x0 = int(nb[0] * w_img / 1000.0)
            y0 = int(nb[1] * h_img / 1000.0)
            x1 = int(nb[2] * w_img / 1000.0)
            y1 = int(nb[3] * h_img / 1000.0)
            orig_boxes.append([x0, y0, x1, y1])

        encoding = self._tokenizer.processor(
            images=image,
            text=words,
            boxes=norm_boxes,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        bbox = encoding["bbox"].to(self.device)
        pixel_vals = encoding["pixel_values"].to(self.device)
        word_ids = encoding.word_ids(batch_index=0)

        outputs = self._classifier.forward(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            pixel_values=pixel_vals,
        )

        logits = outputs["logits"].squeeze(0)
        probs = F.softmax(logits, dim=-1)
        max_probs, pred_ids = torch.max(probs, dim=-1)

        token_label_ids = pred_ids.cpu().tolist()
        token_probs = max_probs.cpu().tolist()

        word_labels, word_probs = collapse_token_preds_to_words(
            token_label_ids=token_label_ids,
            token_probs=token_probs,
            word_ids=word_ids,
            id2label=self._dataset_config.id2label,
            num_words=len(words),
        )

        entities, tokens, flat_agg = reconstruct_entity_spans(
            words=words,
            word_labels=word_labels,
            word_probs=word_probs,
            original_bboxes=orig_boxes,
            normalized_bboxes=norm_boxes,
        )

        if min_confidence > 0.0:
            entities = [e for e in entities if e.confidence >= min_confidence]

        return DocumentInferenceResult(
            document_type=self.dataset_type.upper(),
            model_name=f"{self.dataset_type}_{'robust' if self.use_robust_model else 'baseline'}",
            image_path=str(sample.get("image_path", "")),
            image_size=image.size,
            ocr_word_count=len(words),
            entities=entities,
            tokens=tokens,
            raw_entities_dict=flat_agg,
            ocr_raw_text=" ".join(words),
        )
