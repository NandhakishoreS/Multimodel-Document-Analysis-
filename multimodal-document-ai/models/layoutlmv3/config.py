"""
Configuration dataclasses and YAML loaders for Phase 4 LayoutLMv3 training.

Design:
- LayoutLMv3DatasetConfig  — per-dataset label schema loaded from configs/labels/*.yaml
- LayoutLMv3TrainingConfig — full training run configuration loaded from configs/layoutlmv3.yaml
- load_dataset_config()    — factory that reads the right labels yaml for a given dataset
- load_training_config()   — loads layoutlmv3.yaml, applies per-dataset overrides
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel for project root — all relative paths are resolved from here.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _project_path(p: str) -> Path:
    """Resolve a path string relative to the project root."""
    return (_PROJECT_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# Dataset label configuration
# ---------------------------------------------------------------------------

@dataclass
class LayoutLMv3DatasetConfig:
    """
    Holds the label schema for one dataset, loaded from configs/labels/<dataset>.yaml.

    Attributes
    ----------
    dataset : str
        Dataset name: "sroie" | "funsd" | "cord".
    task : str
        Human-readable task description.
    num_labels : int
        Total number of label classes (including "O").
    labels : List[str]
        Ordered list of label strings; index == label id.
    label2id : Dict[str, int]
        Label string → integer id mapping.
    id2label : Dict[int, str]
        Integer id → label string mapping.
    entity_groups : Dict[str, List[str]]
        Maps entity type name to its BIO label variants, used for
        post-processing entity reconstruction.
    ignore_index : int
        The PyTorch loss ignore index for sub-word continuation tokens.
        Always -100 — matches CrossEntropyLoss default.
    """

    dataset: str
    task: str
    num_labels: int
    labels: List[str]
    label2id: Dict[str, int]
    id2label: Dict[int, str]
    entity_groups: Dict[str, List[str]] = field(default_factory=dict)
    ignore_index: int = -100

    def __post_init__(self) -> None:
        # Ensure id2label uses int keys (YAML may deserialise as strings).
        self.id2label = {int(k): v for k, v in self.id2label.items()}
        # Validate consistency.
        assert len(self.labels) == self.num_labels, (
            f"{self.dataset}: labels list length {len(self.labels)} "
            f"!= declared num_labels {self.num_labels}"
        )
        assert set(self.label2id.keys()) == set(self.labels), (
            f"{self.dataset}: label2id keys do not match labels list"
        )

    @property
    def o_label_id(self) -> int:
        """Integer id of the 'O' (outside/background) label."""
        return self.label2id["O"]


def load_dataset_config(
    dataset_name: str,
    labels_dir: str = "configs/labels",
) -> LayoutLMv3DatasetConfig:
    """
    Load label schema from configs/labels/<dataset_name>.yaml.

    Parameters
    ----------
    dataset_name : str
        One of "sroie", "funsd", "cord".
    labels_dir : str
        Directory containing label YAML files, relative to project root.

    Returns
    -------
    LayoutLMv3DatasetConfig
    """
    yaml_path = _project_path(labels_dir) / f"{dataset_name.lower()}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Label config not found: {yaml_path}. "
            f"Expected configs/labels/{dataset_name}.yaml"
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return LayoutLMv3DatasetConfig(
        dataset=raw["dataset"],
        task=raw.get("task", "token_classification"),
        num_labels=raw["num_labels"],
        labels=raw["labels"],
        label2id=raw["label2id"],
        id2label=raw["id2label"],
        entity_groups=raw.get("entity_groups", {}),
    )


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class LayoutLMv3TrainingConfig:
    """
    Full training run configuration, loaded from configs/layoutlmv3.yaml.

    Attributes mirror the YAML structure with dataset-level overrides applied.
    """

    # Model
    model_checkpoint: str = "microsoft/layoutlmv3-base"

    # Processor
    apply_ocr: bool = False
    max_seq_length: int = 512

    # Training hyperparameters
    epochs: int = 10
    learning_rate: float = 5e-5
    batch_size: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    fp16: bool = False
    seed: int = 42
    early_stopping_patience: int = 3
    eval_every_n_epochs: int = 1

    # Checkpointing
    checkpoint_dir: str = "models/checkpoints/layoutlmv3"
    save_best_only: bool = True
    checkpoint_metric: str = "macro_f1"
    checkpoint_metric_mode: str = "max"

    # Paths
    processed_data_dir: str = "processed_data"
    labels_dir: str = "configs/labels"
    experiments_dir: str = "experiments"
    reports_dir: str = "reports"

    # Stage settings
    smoke_num_samples: int = 15
    smoke_epochs: int = 2
    small_num_samples: int = 30
    small_epochs: int = 3

    # Dataset — set at runtime
    dataset_name: Optional[str] = None
    num_labels: Optional[int] = None

    @property
    def effective_batch_size(self) -> int:
        """Batch size × gradient accumulation steps."""
        return self.batch_size * self.gradient_accumulation_steps

    def resolved_checkpoint_dir(self) -> Path:
        return _project_path(self.checkpoint_dir)

    def resolved_processed_dir(self) -> Path:
        return _project_path(self.processed_data_dir)

    def resolved_experiments_dir(self) -> Path:
        return _project_path(self.experiments_dir)

    def resolved_reports_dir(self) -> Path:
        return _project_path(self.reports_dir)


def load_training_config(
    config_path: str = "configs/layoutlmv3.yaml",
    dataset_name: Optional[str] = None,
) -> LayoutLMv3TrainingConfig:
    """
    Load training configuration from YAML, optionally applying per-dataset overrides.

    Parameters
    ----------
    config_path : str
        Path to layoutlmv3.yaml, relative to project root.
    dataset_name : str or None
        If provided, per-dataset overrides from configs.datasets.<name> are applied.

    Returns
    -------
    LayoutLMv3TrainingConfig
    """
    yaml_path = _project_path(config_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Training config not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    model_cfg = raw.get("model", {})
    proc_cfg = raw.get("processor", {})
    train_cfg = raw.get("training", {})
    ckpt_cfg = raw.get("checkpointing", {})
    paths_cfg = raw.get("paths", {})
    stages_cfg = raw.get("stages", {})

    # Apply dataset-specific overrides when requested.
    ds_overrides: Dict[str, Any] = {}
    if dataset_name:
        ds_overrides = raw.get("datasets", {}).get(dataset_name.lower(), {})

    cfg = LayoutLMv3TrainingConfig(
        model_checkpoint=model_cfg.get("checkpoint", "microsoft/layoutlmv3-base"),
        apply_ocr=proc_cfg.get("apply_ocr", False),
        max_seq_length=proc_cfg.get("max_seq_length", 512),
        epochs=ds_overrides.get("epochs", train_cfg.get("epochs", 10)),
        learning_rate=float(ds_overrides.get("learning_rate", train_cfg.get("learning_rate", 5e-5))),
        batch_size=train_cfg.get("batch_size", 2),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.1)),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        fp16=train_cfg.get("fp16", False),
        seed=train_cfg.get("seed", 42),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 3),
        eval_every_n_epochs=train_cfg.get("eval_every_n_epochs", 1),
        checkpoint_dir=ckpt_cfg.get("save_dir", "models/checkpoints/layoutlmv3"),
        save_best_only=ckpt_cfg.get("save_best_only", True),
        checkpoint_metric=ckpt_cfg.get("metric", "macro_f1"),
        checkpoint_metric_mode=ckpt_cfg.get("mode", "max"),
        processed_data_dir=paths_cfg.get("processed_data_dir", "processed_data"),
        labels_dir=paths_cfg.get("labels_dir", "configs/labels"),
        experiments_dir=paths_cfg.get("experiments_dir", "experiments"),
        reports_dir=paths_cfg.get("reports_dir", "reports"),
        smoke_num_samples=stages_cfg.get("smoke", {}).get("num_samples", 15),
        smoke_epochs=stages_cfg.get("smoke", {}).get("epochs", 2),
        small_num_samples=stages_cfg.get("small", {}).get("num_samples", 30),
        small_epochs=stages_cfg.get("small", {}).get("epochs", 3),
        dataset_name=dataset_name,
        num_labels=ds_overrides.get("num_labels"),
    )

    logger.info(
        "Loaded training config: checkpoint=%s, lr=%g, batch=%d, "
        "grad_accum=%d, effective_batch=%d, epochs=%d, fp16=%s",
        cfg.model_checkpoint, cfg.learning_rate, cfg.batch_size,
        cfg.gradient_accumulation_steps, cfg.effective_batch_size,
        cfg.epochs, cfg.fp16,
    )
    return cfg
