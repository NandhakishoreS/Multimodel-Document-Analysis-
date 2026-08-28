"""Tests for word-to-subword label and bbox alignment (tokenizer.py)."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.layoutlmv3.tokenizer import (
    align_labels_to_tokens,
    align_bboxes_to_tokens,
    _clamp_bbox,
    IGNORE_INDEX,
    PAD_BBOX,
)


LABEL2ID = {
    "O": 0, "B-COMPANY": 1, "I-COMPANY": 2,
    "B-TOTAL": 7, "I-TOTAL": 8,
}


class TestAlignLabelsToTokens:
    def test_all_single_tokens(self):
        """Each word becomes exactly one token — no continuations."""
        word_labels = ["O", "B-COMPANY", "I-COMPANY"]
        word_ids    = [None, 0, 1, 2, None]  # CLS, word0, word1, word2, SEP
        result = align_labels_to_tokens(word_labels, word_ids, LABEL2ID)
        assert result == [IGNORE_INDEX, 0, 1, 2, IGNORE_INDEX]

    def test_subword_continuation_gets_ignore(self):
        """Word split into two subwords: first gets label, second gets -100."""
        word_labels = ["B-TOTAL"]
        # CLS  word0_sub0  word0_sub1  SEP
        word_ids = [None, 0, 0, None]
        result = align_labels_to_tokens(word_labels, word_ids, LABEL2ID)
        assert result[0] == IGNORE_INDEX   # CLS
        assert result[1] == LABEL2ID["B-TOTAL"]  # first subword
        assert result[2] == IGNORE_INDEX   # continuation
        assert result[3] == IGNORE_INDEX   # SEP

    def test_three_subwords(self):
        """Word split into 3 subwords: only first gets label."""
        word_labels = ["B-COMPANY"]
        word_ids    = [None, 0, 0, 0, None]
        result = align_labels_to_tokens(word_labels, word_ids, LABEL2ID)
        assert result == [IGNORE_INDEX, LABEL2ID["B-COMPANY"],
                          IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]

    def test_unknown_label_maps_to_o(self):
        """Unknown label string falls back to O index."""
        word_labels = ["B-UNKNOWN_ENTITY"]
        word_ids    = [None, 0, None]
        result = align_labels_to_tokens(word_labels, word_ids, LABEL2ID)
        assert result[1] == LABEL2ID["O"]

    def test_all_special_tokens(self):
        """All special tokens → all IGNORE_INDEX."""
        word_ids = [None, None, None]
        result = align_labels_to_tokens([], word_ids, LABEL2ID)
        assert result == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]

    def test_multiple_words_with_subwords(self):
        """
        3 words: word0 (1 tok), word1 (2 toks), word2 (1 tok)
        [CLS, 0, 1, 1, 2, SEP]
        """
        word_labels = ["O", "B-COMPANY", "I-COMPANY"]
        word_ids    = [None, 0, 1, 1, 2, None]
        result = align_labels_to_tokens(word_labels, word_ids, LABEL2ID)
        expected = [
            IGNORE_INDEX,
            LABEL2ID["O"],
            LABEL2ID["B-COMPANY"],
            IGNORE_INDEX,          # continuation of word1
            LABEL2ID["I-COMPANY"],
            IGNORE_INDEX,
        ]
        assert result == expected

    def test_out_of_range_word_id(self):
        """word_id pointing beyond word_labels length logs warning and uses O."""
        word_labels = ["O"]
        word_ids    = [None, 0, 5, None]  # word_id=5 is out of range
        result = align_labels_to_tokens(word_labels, word_ids, LABEL2ID)
        assert result[2] == LABEL2ID["O"]


class TestAlignBboxesToTokens:
    def test_basic_propagation(self):
        """Each word's bbox is duplicated to all its subword tokens."""
        bboxes   = [[10, 20, 30, 40], [50, 60, 70, 80]]
        word_ids = [None, 0, 0, 1, None]
        result   = align_bboxes_to_tokens(bboxes, word_ids)
        assert result[0] == PAD_BBOX          # CLS
        assert result[1] == [10, 20, 30, 40]  # word0 subword0
        assert result[2] == [10, 20, 30, 40]  # word0 subword1 (same bbox)
        assert result[3] == [50, 60, 70, 80]  # word1
        assert result[4] == PAD_BBOX          # SEP

    def test_out_of_range_word_id(self):
        bboxes   = [[1, 2, 3, 4]]
        word_ids = [None, 0, 99, None]  # word_id=99 out of range
        result   = align_bboxes_to_tokens(bboxes, word_ids)
        assert result[2] == PAD_BBOX  # Falls back to pad

    def test_empty(self):
        result = align_bboxes_to_tokens([], [None, None])
        assert result == [PAD_BBOX, PAD_BBOX]


class TestClampBbox:
    def test_valid_bbox_unchanged(self):
        assert _clamp_bbox([10, 20, 100, 200]) == [10, 20, 100, 200]

    def test_clamps_over_1000(self):
        b = _clamp_bbox([0, 0, 1500, 800])
        assert b[2] == 1000

    def test_fixes_inverted_x(self):
        b = _clamp_bbox([300, 0, 100, 50])  # x0 > x1
        assert b[0] <= b[2]

    def test_fixes_zero_area(self):
        b = _clamp_bbox([100, 100, 100, 100])  # zero area
        assert b[2] > b[0] or b[3] > b[1]

    def test_wrong_length_returns_default(self):
        assert _clamp_bbox([1, 2, 3]) == [0, 0, 1, 1]
