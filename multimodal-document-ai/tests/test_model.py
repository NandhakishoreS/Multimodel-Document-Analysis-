"""
Tests for LayoutLMv3Classifier and config loading.

The model-forward tests use a tiny mock model to avoid downloading
the 500 MB microsoft/layoutlmv3-base checkpoint in CI.
Config and label-schema tests run without network access.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.layoutlmv3.config import (
    LayoutLMv3DatasetConfig,
    LayoutLMv3TrainingConfig,
    load_dataset_config,
    load_training_config,
)


# ---------------------------------------------------------------------------
# LayoutLMv3DatasetConfig tests
# ---------------------------------------------------------------------------

class TestLayoutLMv3DatasetConfig:
    def _make_config(self, **overrides):
        defaults = dict(
            dataset="sroie", task="kie", num_labels=9,
            labels=["O","B-COMPANY","I-COMPANY","B-DATE","I-DATE",
                    "B-ADDRESS","I-ADDRESS","B-TOTAL","I-TOTAL"],
            label2id={"O":0,"B-COMPANY":1,"I-COMPANY":2,"B-DATE":3,"I-DATE":4,
                      "B-ADDRESS":5,"I-ADDRESS":6,"B-TOTAL":7,"I-TOTAL":8},
            id2label={0:"O",1:"B-COMPANY",2:"I-COMPANY",3:"B-DATE",4:"I-DATE",
                      5:"B-ADDRESS",6:"I-ADDRESS",7:"B-TOTAL",8:"I-TOTAL"},
        )
        defaults.update(overrides)
        return LayoutLMv3DatasetConfig(**defaults)

    def test_valid_config_initialises(self):
        cfg = self._make_config()
        assert cfg.num_labels == 9
        assert cfg.o_label_id == 0
        assert cfg.ignore_index == -100

    def test_id2label_keys_coerced_to_int(self):
        """YAML may parse int keys as strings — __post_init__ must fix them."""
        cfg = self._make_config(
            id2label={"0":"O","1":"B-COMPANY","2":"I-COMPANY",
                      "3":"B-DATE","4":"I-DATE","5":"B-ADDRESS",
                      "6":"I-ADDRESS","7":"B-TOTAL","8":"I-TOTAL"}
        )
        assert cfg.id2label[0] == "O"
        assert cfg.id2label[7] == "B-TOTAL"

    def test_mismatch_raises(self):
        with pytest.raises(AssertionError):
            self._make_config(num_labels=5)  # Wrong num_labels

    def test_label2id_mismatch_raises(self):
        with pytest.raises(AssertionError):
            self._make_config(
                label2id={"O": 0, "B-WRONG": 1}  # Doesn't match labels list
            )


class TestLoadDatasetConfig:
    def test_load_sroie(self):
        cfg = load_dataset_config("sroie")
        assert cfg.dataset == "sroie"
        assert cfg.num_labels == 9
        assert "B-COMPANY" in cfg.label2id
        assert "O" in cfg.labels

    def test_load_funsd(self):
        cfg = load_dataset_config("funsd")
        assert cfg.dataset == "funsd"
        assert cfg.num_labels == 7
        assert "B-QUESTION" in cfg.label2id

    def test_load_cord(self):
        cfg = load_dataset_config("cord")
        assert cfg.dataset == "cord"
        assert cfg.num_labels == 13
        assert "B-MENU" in cfg.label2id

    def test_unknown_dataset_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset_config("nonexistent_dataset_xyz")

    def test_entity_groups_present(self):
        """Entity groups are needed for entity reconstruction."""
        cfg = load_dataset_config("sroie")
        assert "company" in cfg.entity_groups
        assert "total" in cfg.entity_groups

    def test_labels_ordered_match_ids(self):
        """labels[i] must equal id2label[i] for all i."""
        for ds in ["sroie", "funsd", "cord"]:
            cfg = load_dataset_config(ds)
            for i, lbl in enumerate(cfg.labels):
                assert cfg.id2label[i] == lbl, (
                    f"{ds}: labels[{i}]={lbl} != id2label[{i}]={cfg.id2label[i]}"
                )


class TestLoadTrainingConfig:
    def test_load_defaults(self):
        cfg = load_training_config()
        assert cfg.model_checkpoint == "microsoft/layoutlmv3-base"
        assert cfg.apply_ocr is False
        assert cfg.batch_size >= 1
        assert 0 < cfg.warmup_ratio < 1

    def test_dataset_override_applied(self):
        """FUNSD overrides epochs to 12."""
        cfg = load_training_config(dataset_name="funsd")
        assert cfg.epochs == 12
        assert cfg.dataset_name == "funsd"

    def test_cord_lr_override(self):
        """CORD overrides learning_rate to 3e-5."""
        cfg = load_training_config(dataset_name="cord")
        assert abs(cfg.learning_rate - 3e-5) < 1e-9

    def test_effective_batch_size(self):
        cfg = load_training_config()
        assert cfg.effective_batch_size == cfg.batch_size * cfg.gradient_accumulation_steps

    def test_smoke_stage_params(self):
        cfg = load_training_config()
        assert cfg.smoke_num_samples == 15
        assert cfg.smoke_epochs == 2

    def test_small_stage_params(self):
        cfg = load_training_config()
        assert cfg.small_num_samples == 30
        assert cfg.small_epochs == 3


# ---------------------------------------------------------------------------
# LayoutLMv3Classifier — unit tests using mocked HuggingFace model
# ---------------------------------------------------------------------------

class TestLayoutLMv3ClassifierForward:
    """
    Tests LayoutLMv3Classifier.forward() and predict_batch() using a
    mock HuggingFace model to avoid downloading the real checkpoint.
    """

    def _make_dataset_config(self):
        return LayoutLMv3DatasetConfig(
            dataset="sroie", task="kie", num_labels=9,
            labels=["O","B-COMPANY","I-COMPANY","B-DATE","I-DATE",
                    "B-ADDRESS","I-ADDRESS","B-TOTAL","I-TOTAL"],
            label2id={"O":0,"B-COMPANY":1,"I-COMPANY":2,"B-DATE":3,"I-DATE":4,
                      "B-ADDRESS":5,"I-ADDRESS":6,"B-TOTAL":7,"I-TOTAL":8},
            id2label={0:"O",1:"B-COMPANY",2:"I-COMPANY",3:"B-DATE",4:"I-DATE",
                      5:"B-ADDRESS",6:"I-ADDRESS",7:"B-TOTAL",8:"I-TOTAL"},
        )

    @patch("models.layoutlmv3.model.LayoutLMv3ForTokenClassification.from_pretrained")
    def test_forward_returns_loss_and_logits(self, mock_from_pretrained):
        B, S, L = 2, 16, 9
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.loss   = torch.tensor(1.5)
        mock_output.logits = torch.randn(B, S, L)
        mock_model.return_value = mock_output
        mock_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.named_parameters.return_value = iter([("w", torch.tensor([1.0]))])
        mock_model.to.return_value = mock_model
        mock_model.__call__ = mock_model
        mock_from_pretrained.return_value = mock_model

        from models.layoutlmv3.model import LayoutLMv3Classifier
        cfg = self._make_dataset_config()
        classifier = LayoutLMv3Classifier(
            dataset_config=cfg,
            checkpoint="mock_checkpoint",
            device="cpu",
        )
        classifier.model = mock_model

        result = classifier.forward(
            input_ids      = torch.zeros(B, S, dtype=torch.long),
            attention_mask = torch.ones(B, S, dtype=torch.long),
            bbox           = torch.zeros(B, S, 4, dtype=torch.long),
            pixel_values   = torch.zeros(B, 3, 224, 224),
            labels         = torch.zeros(B, S, dtype=torch.long),
        )
        assert "loss" in result
        assert "logits" in result
        assert result["logits"].shape == (B, S, L)

    @patch("models.layoutlmv3.model.LayoutLMv3ForTokenClassification.from_pretrained")
    def test_decode_predictions_shape(self, mock_from_pretrained):
        """decode_predictions should return one label list per sample."""
        B, S, L = 2, 8, 9
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.parameters.return_value = iter([])
        mock_model.named_parameters.return_value = iter([])
        mock_from_pretrained.return_value = mock_model

        from models.layoutlmv3.model import LayoutLMv3Classifier
        cfg = self._make_dataset_config()
        classifier = LayoutLMv3Classifier(
            dataset_config=cfg, checkpoint="mock", device="cpu"
        )

        pred_ids       = torch.zeros(B, S, dtype=torch.long)
        attention_mask = torch.ones(B, S, dtype=torch.long)

        preds = classifier.decode_predictions(pred_ids, attention_mask)
        assert len(preds) == B
        assert all(isinstance(seq, list) for seq in preds)

    @patch("models.layoutlmv3.model.LayoutLMv3ForTokenClassification.from_pretrained")
    def test_save_checkpoint_writes_files(self, mock_from_pretrained, tmp_path):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.parameters.return_value = iter([])
        mock_model.named_parameters.return_value = iter([])
        mock_model.save_pretrained = MagicMock()
        mock_from_pretrained.return_value = mock_model

        from models.layoutlmv3.model import LayoutLMv3Classifier
        cfg = self._make_dataset_config()
        classifier = LayoutLMv3Classifier(
            dataset_config=cfg, checkpoint="mock", device="cpu"
        )

        save_dir = tmp_path / "ckpt"
        classifier.save_checkpoint(save_dir=save_dir, epoch=3, val_f1=0.75)

        meta = json.loads((save_dir / "checkpoint_meta.json").read_text())
        assert meta["epoch"] == 3
        assert abs(meta["val_macro_f1"] - 0.75) < 1e-4
        assert meta["dataset"] == "sroie"
        assert meta["num_labels"] == 9
