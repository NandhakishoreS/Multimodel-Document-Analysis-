"""
Integration tests for LayoutLMv3DocumentDataset.

These tests do NOT download the model checkpoint — they use a mocked
tokenizer wrapper so the tests run offline without heavy dependencies.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.layoutlmv3.config import LayoutLMv3DatasetConfig
from models.layoutlmv3.dataset import LayoutLMv3DocumentDataset


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

LABEL_SCHEMA = LayoutLMv3DatasetConfig(
    dataset="sroie",
    task="kie",
    num_labels=9,
    labels=["O","B-COMPANY","I-COMPANY","B-DATE","I-DATE",
            "B-ADDRESS","I-ADDRESS","B-TOTAL","I-TOTAL"],
    label2id={"O":0,"B-COMPANY":1,"I-COMPANY":2,"B-DATE":3,"I-DATE":4,
              "B-ADDRESS":5,"I-ADDRESS":6,"B-TOTAL":7,"I-TOTAL":8},
    id2label={0:"O",1:"B-COMPANY",2:"I-COMPANY",3:"B-DATE",4:"I-DATE",
              5:"B-ADDRESS",6:"I-ADDRESS",7:"B-TOTAL",8:"I-TOTAL"},
)

SEQ_LEN = 512

def _make_fake_encoding(seq_len: int = SEQ_LEN) -> Dict[str, Any]:
    """Return a fake processor encoding dict (batch_size=1)."""
    return {
        "input_ids":      torch.zeros(1, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        "bbox":           torch.zeros(1, seq_len, 4, dtype=torch.long),
        "pixel_values":   torch.zeros(1, 3, 224, 224, dtype=torch.float),
        "labels":         torch.zeros(1, seq_len, dtype=torch.long),
        "word_ids":       [0] * seq_len,
    }


def _make_fake_sample(n_words: int = 5, idx: int = 0) -> Dict[str, Any]:
    return {
        "id": f"sample_{idx}",
        "dataset": "sroie",
        "split": "train",
        "image_path": "processed_data/sroie/images/dummy.png",
        "width": 100,
        "height": 100,
        "words": [f"word{i}" for i in range(n_words)],
        "bboxes": [[i*10, 0, i*10+8, 10] for i in range(n_words)],
        "labels": ["O"] * n_words,
        "entities": {},
        "relations": [],
    }


def _write_jsonl(path: Path, samples: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLayoutLMv3DocumentDataset:
    def _make_mock_wrapper(self):
        wrapper = MagicMock()
        wrapper.max_seq_length = SEQ_LEN
        wrapper.encode_sample.return_value = _make_fake_encoding()
        return wrapper

    def test_len_matches_jsonl(self, tmp_path):
        samples = [_make_fake_sample(idx=i) for i in range(7)]
        jsonl = tmp_path / "train.jsonl"
        _write_jsonl(jsonl, samples)

        ds = LayoutLMv3DocumentDataset(
            jsonl_path=jsonl,
            dataset_config=LABEL_SCHEMA,
            tokenizer_wrapper=self._make_mock_wrapper(),
        )
        assert len(ds) == 7

    def test_getitem_returns_required_keys(self, tmp_path):
        samples = [_make_fake_sample()]
        jsonl = tmp_path / "train.jsonl"
        _write_jsonl(jsonl, samples)

        ds = LayoutLMv3DocumentDataset(
            jsonl_path=jsonl,
            dataset_config=LABEL_SCHEMA,
            tokenizer_wrapper=self._make_mock_wrapper(),
        )
        item = ds[0]
        for key in ["input_ids", "attention_mask", "bbox", "pixel_values", "labels"]:
            assert key in item, f"Missing key: {key}"

    def test_tensor_shapes(self, tmp_path):
        samples = [_make_fake_sample()]
        jsonl = tmp_path / "train.jsonl"
        _write_jsonl(jsonl, samples)

        ds = LayoutLMv3DocumentDataset(
            jsonl_path=jsonl,
            dataset_config=LABEL_SCHEMA,
            tokenizer_wrapper=self._make_mock_wrapper(),
        )
        item = ds[0]
        assert item["input_ids"].shape      == torch.Size([SEQ_LEN])
        assert item["attention_mask"].shape == torch.Size([SEQ_LEN])
        assert item["bbox"].shape           == torch.Size([SEQ_LEN, 4])
        assert item["pixel_values"].shape   == torch.Size([3, 224, 224])
        assert item["labels"].shape         == torch.Size([SEQ_LEN])

    def test_subsampling(self, tmp_path):
        samples = [_make_fake_sample(idx=i) for i in range(20)]
        jsonl = tmp_path / "train.jsonl"
        _write_jsonl(jsonl, samples)

        ds = LayoutLMv3DocumentDataset(
            jsonl_path=jsonl,
            dataset_config=LABEL_SCHEMA,
            tokenizer_wrapper=self._make_mock_wrapper(),
            num_samples=5,
            seed=42,
        )
        assert len(ds) == 5

    def test_subsampling_reproducible(self, tmp_path):
        """Same seed gives same subset."""
        samples = [_make_fake_sample(idx=i) for i in range(20)]
        jsonl = tmp_path / "train.jsonl"
        _write_jsonl(jsonl, samples)

        wrapper = self._make_mock_wrapper()
        ds1 = LayoutLMv3DocumentDataset(jsonl, LABEL_SCHEMA, wrapper, num_samples=5, seed=42)
        ds2 = LayoutLMv3DocumentDataset(jsonl, LABEL_SCHEMA, wrapper, num_samples=5, seed=42)
        ids1 = [s["id"] for s in ds1.samples]
        ids2 = [s["id"] for s in ds2.samples]
        assert ids1 == ids2

    def test_missing_jsonl_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LayoutLMv3DocumentDataset(
                jsonl_path=tmp_path / "nonexistent.jsonl",
                dataset_config=LABEL_SCHEMA,
                tokenizer_wrapper=self._make_mock_wrapper(),
            )

    def test_sample_id_in_item(self, tmp_path):
        samples = [_make_fake_sample(idx=42)]
        jsonl = tmp_path / "train.jsonl"
        _write_jsonl(jsonl, samples)

        ds = LayoutLMv3DocumentDataset(
            jsonl_path=jsonl,
            dataset_config=LABEL_SCHEMA,
            tokenizer_wrapper=self._make_mock_wrapper(),
        )
        assert ds[0]["sample_id"] == "sample_42"

    def test_get_raw_sample(self, tmp_path):
        samples = [_make_fake_sample(n_words=3)]
        jsonl = tmp_path / "train.jsonl"
        _write_jsonl(jsonl, samples)

        ds = LayoutLMv3DocumentDataset(
            jsonl_path=jsonl,
            dataset_config=LABEL_SCHEMA,
            tokenizer_wrapper=self._make_mock_wrapper(),
        )
        raw = ds.get_raw_sample(0)
        assert raw["words"] == ["word0", "word1", "word2"]
