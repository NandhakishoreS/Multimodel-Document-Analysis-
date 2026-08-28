"""
LayoutLMv3 token classification model wrapper.

Wraps microsoft/layoutlmv3-base with a token-classification head for
training, evaluation, and inference.  Designed to be dataset-agnostic —
num_labels and the label maps are injected at construction time.

Key features:
- Loads from a HuggingFace checkpoint OR a local directory.
- Configures id2label / label2id from the dataset label schema.
- Exposes a unified forward() for training (loss included) and inference.
- Gradient checkpointing can be enabled to reduce peak memory.
- Checkpoint save/load utilities that store model + processor + config.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import LayoutLMv3Config, LayoutLMv3ForTokenClassification

from models.layoutlmv3.config import LayoutLMv3DatasetConfig, _PROJECT_ROOT

logger = logging.getLogger(__name__)


class LayoutLMv3Classifier:
    """
    Wrapper around LayoutLMv3ForTokenClassification.

    Parameters
    ----------
    dataset_config : LayoutLMv3DatasetConfig
        Label schema — provides num_labels, id2label, label2id.
    checkpoint : str
        HuggingFace model id or local path to load weights from.
    gradient_checkpointing : bool
        Enable gradient checkpointing to trade compute for memory.
        Useful on very limited GPU/CPU RAM.
    device : str or torch.device or None
        Target device.  Auto-detected if None.

    Attributes
    ----------
    model : LayoutLMv3ForTokenClassification
        The underlying HuggingFace model.
    device : torch.device
    dataset_config : LayoutLMv3DatasetConfig
    """

    def __init__(
        self,
        dataset_config: LayoutLMv3DatasetConfig,
        checkpoint: str = "microsoft/layoutlmv3-base",
        gradient_checkpointing: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.dataset_config = dataset_config
        self.checkpoint = checkpoint

        # ---- Device detection ----
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info("Using device: cuda | GPU: %s | VRAM: %.1f GB", gpu_name, vram_gb)
            else:
                self.device = torch.device("cpu")
                logger.info("Using device: cpu (CUDA not available)")
        else:
            self.device = torch.device(device)

        # ---- Load model ----
        logger.info(
            "Loading LayoutLMv3ForTokenClassification from '%s' "
            "(num_labels=%d)…",
            checkpoint, dataset_config.num_labels,
        )
        start = time.time()
        self.model: LayoutLMv3ForTokenClassification = (
            LayoutLMv3ForTokenClassification.from_pretrained(
                checkpoint,
                num_labels=dataset_config.num_labels,
                id2label=dataset_config.id2label,
                label2id=dataset_config.label2id,
                ignore_mismatched_sizes=True,
            )
        )
        elapsed = time.time() - start
        logger.info("Model loaded in %.1f s", elapsed)

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")

        self.model.to(self.device)

        # Log total and trainable parameters.
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            "Parameters: %s total, %s trainable",
            f"{total:,}", f"{trainable:,}",
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run a forward pass through LayoutLMv3.

        Parameters
        ----------
        input_ids      : (batch, seq_len)
        attention_mask : (batch, seq_len)
        bbox           : (batch, seq_len, 4)
        pixel_values   : (batch, 3, 224, 224)
        labels         : (batch, seq_len) or None
            If provided, the HuggingFace model computes CrossEntropyLoss
            internally (ignoring positions where label == -100).

        Returns
        -------
        dict with keys:
            loss   : scalar Tensor (only if labels provided)
            logits : (batch, seq_len, num_labels)
        """
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            bbox=bbox.to(self.device),
            pixel_values=pixel_values.to(self.device),
            labels=labels.to(self.device) if labels is not None else None,
        )
        result: Dict[str, torch.Tensor] = {"logits": outputs.logits}
        if outputs.loss is not None:
            result["loss"] = outputs.loss
        return result

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on a batch and return predicted label ids and probabilities.

        Returns
        -------
        pred_ids     : (batch, seq_len) int64 — argmax of logits
        probs        : (batch, seq_len, num_labels) float32 — softmax probabilities
        """
        self.model.eval()
        outputs = self.forward(input_ids, attention_mask, bbox, pixel_values)
        logits = outputs["logits"]                        # (B, S, num_labels)
        probs = torch.softmax(logits, dim=-1)            # (B, S, num_labels)
        pred_ids = torch.argmax(probs, dim=-1)           # (B, S)
        return pred_ids, probs

    def decode_predictions(
        self,
        pred_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_ids_list: Optional[List[List[Optional[int]]]] = None,
    ) -> List[List[str]]:
        """
        Convert prediction id tensors to label string lists, one per sample.

        Filters out special-token positions (attention_mask == 0) and
        optionally filters out continuation sub-word positions.

        Parameters
        ----------
        pred_ids       : (batch, seq_len)
        attention_mask : (batch, seq_len)
        word_ids_list  : list of word_ids for each sample in the batch.
            If provided, only first-subword positions are returned (word-level output).

        Returns
        -------
        List[List[str]] — one label list per sample in the batch.
        """
        batch_preds: List[List[str]] = []
        batch_size = pred_ids.shape[0]

        for i in range(batch_size):
            ids = pred_ids[i].cpu().tolist()
            mask = attention_mask[i].cpu().tolist()
            word_ids = word_ids_list[i] if word_ids_list else None

            preds: List[str] = []
            prev_word_id: Optional[int] = None

            for tok_idx, (lid, m) in enumerate(zip(ids, mask)):
                if m == 0:
                    continue  # Padding token

                if word_ids is not None:
                    wid = word_ids[tok_idx]
                    if wid is None:
                        continue  # CLS / SEP
                    if wid == prev_word_id:
                        continue  # Continuation sub-word — skip
                    prev_word_id = wid

                preds.append(self.dataset_config.id2label.get(lid, "O"))

            batch_preds.append(preds)

        return batch_preds

    # ------------------------------------------------------------------
    # Checkpoint utilities
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        save_dir: Union[str, Path],
        epoch: int,
        val_f1: float,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save model weights, config, and metadata to a checkpoint directory.

        Saves:
        - pytorch_model.bin / model.safetensors (via save_pretrained)
        - config.json (HuggingFace model config)
        - checkpoint_meta.json (epoch, val_f1, label schema)

        Parameters
        ----------
        save_dir : str or Path
            Directory to save into.
        epoch : int
            Epoch number (1-indexed).
        val_f1 : float
            Validation macro F1 score that triggered this save.
        extra_metadata : dict or None
            Additional fields to store in checkpoint_meta.json.

        Returns
        -------
        Path to the save directory.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(save_path))
        logger.info("Model weights saved to %s", save_path)

        meta = {
            "epoch": epoch,
            "val_macro_f1": round(val_f1, 6),
            "dataset": self.dataset_config.dataset,
            "num_labels": self.dataset_config.num_labels,
            "checkpoint_source": self.checkpoint,
            "labels": self.dataset_config.labels,
            "label2id": self.dataset_config.label2id,
            "id2label": {str(k): v for k, v in self.dataset_config.id2label.items()},
        }
        if extra_metadata:
            meta.update(extra_metadata)

        meta_path = save_path / "checkpoint_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("Checkpoint metadata saved to %s", meta_path)

        return save_path

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_dir: Union[str, Path],
        dataset_config: LayoutLMv3DatasetConfig,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "LayoutLMv3Classifier":
        """
        Load a previously saved checkpoint.

        Parameters
        ----------
        checkpoint_dir : str or Path
            Directory containing model weights and checkpoint_meta.json.
        dataset_config : LayoutLMv3DatasetConfig
            Label schema (loaded from configs/labels/*.yaml).
        device : str or torch.device or None

        Returns
        -------
        LayoutLMv3Classifier with loaded weights.
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        logger.info("Loading checkpoint from %s…", checkpoint_dir)
        instance = cls(
            dataset_config=dataset_config,
            checkpoint=str(checkpoint_dir),
            device=device,
        )
        return instance

    def train_mode(self) -> None:
        """Set model to training mode."""
        self.model.train()

    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()

    def parameters(self):
        """Expose model parameters for optimizer construction."""
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()
