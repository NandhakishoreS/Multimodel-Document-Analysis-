"""Tests for entity reconstruction from BIO predictions (layoutlmv3_predict.py)."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.layoutlmv3_predict import reconstruct_entities, token_preds_to_word_preds


class TestReconstructEntities:
    def test_single_entity(self):
        words  = ["25", ".00"]
        labels = ["B-TOTAL", "I-TOTAL"]
        result = reconstruct_entities(words, labels)
        assert result["entities"].get("TOTAL") == "25 .00"
        assert len(result["spans"]) == 1
        assert result["spans"][0]["entity_type"] == "TOTAL"
        assert result["spans"][0]["start_word"] == 0
        assert result["spans"][0]["end_word"] == 1

    def test_multiple_entity_types(self):
        words  = ["ACME", "Corp", "01/01/2024", "100.00"]
        labels = ["B-COMPANY", "I-COMPANY", "B-DATE", "B-TOTAL"]
        result = reconstruct_entities(words, labels)
        assert "COMPANY" in result["entities"]
        assert "DATE" in result["entities"]
        assert "TOTAL" in result["entities"]
        assert result["entities"]["COMPANY"] == "ACME Corp"
        assert result["entities"]["DATE"] == "01/01/2024"
        assert result["entities"]["TOTAL"] == "100.00"

    def test_o_labels_produce_no_entity(self):
        words  = ["the", "quick", "fox"]
        labels = ["O", "O", "O"]
        result = reconstruct_entities(words, labels)
        assert result["entities"] == {}
        assert result["spans"] == []

    def test_broken_bio_i_without_b(self):
        """I-TOTAL without preceding B-TOTAL should still create a span."""
        words  = ["100", ".00"]
        labels = ["I-TOTAL", "I-TOTAL"]  # Broken BIO
        result = reconstruct_entities(words, labels)
        # Should recover gracefully — either merges into one span or makes two
        assert "TOTAL" in result["entities"]

    def test_repeated_entity_same_type_joined(self):
        """Two B-COMPANY spans → values joined by space."""
        words  = ["ACME", "O", "Corp"]
        labels = ["B-COMPANY", "O", "B-COMPANY"]
        result = reconstruct_entities(words, labels)
        # Both "ACME" and "Corp" are separate B-COMPANY spans
        ent_val = result["entities"].get("COMPANY", "")
        assert "ACME" in ent_val
        assert "Corp" in ent_val

    def test_probabilities_attached(self):
        words  = ["100.00"]
        labels = ["B-TOTAL"]
        probs  = [0.97]
        result = reconstruct_entities(words, labels, word_probs=probs)
        assert result["spans"][0]["mean_probability"] == pytest.approx(0.97, abs=1e-3)

    def test_token_predictions_format(self):
        words  = ["A", "B"]
        labels = ["B-DATE", "O"]
        result = reconstruct_entities(words, labels)
        preds  = result["token_predictions"]
        assert len(preds) == 2
        assert preds[0]["word"] == "A"
        assert preds[0]["label"] == "B-DATE"
        assert "prediction_probability" in preds[0]

    def test_empty_input(self):
        result = reconstruct_entities([], [])
        assert result["entities"] == {}
        assert result["spans"] == []
        assert result["token_predictions"] == []

    def test_b_immediately_follows_b_same_type(self):
        """Two consecutive B tags of same type → two separate spans."""
        words  = ["100", "200"]
        labels = ["B-TOTAL", "B-TOTAL"]
        result = reconstruct_entities(words, labels)
        assert len(result["spans"]) == 2


class TestTokenPredsToWordPreds:
    def test_basic_first_subword_rule(self):
        """
        word 0 → tokens 1,2 (subword split); word 1 → token 3; specials at 0,4.
        token_labels[1]=1 (B-COMPANY) is the first subword of word0 → word0 label.
        token_labels[3]=2 (I-COMPANY) is the first (only) token of word1 → word1 label.
        """
        id2label     = {0: "O", 1: "B-COMPANY", 2: "I-COMPANY"}
        # CLS=None, word0_sub0=0, word0_sub1=0, word1=1, SEP=None
        token_labels = [0, 1, 2, 2, 0]   # token[3]=2 => word1 gets "I-COMPANY"
        token_probs  = [0.9, 0.95, 0.8, 0.85, 0.9]
        word_ids     = [None, 0, 0, 1, None]
        labels, probs = token_preds_to_word_preds(
            token_labels, token_probs, word_ids, id2label, num_words=2
        )
        assert labels == ["B-COMPANY", "I-COMPANY"]
        # word0: first subword is token[1] with prob 0.95
        assert probs[0] == pytest.approx(0.95, abs=1e-3)
        # word1: token[3] with prob 0.85
        assert probs[1] == pytest.approx(0.85, abs=1e-3)

    def test_truncated_sequence_fills_o(self):
        """If sequence is truncated, missing words get 'O'."""
        id2label     = {0: "O", 1: "B-TOTAL"}
        token_labels = [0, 1]
        token_probs  = [0.9, 0.95]
        word_ids     = [None, 0]   # Only word 0 appears
        labels, probs = token_preds_to_word_preds(
            token_labels, token_probs, word_ids, id2label, num_words=3
        )
        assert len(labels) == 3
        assert labels[1] == "O"   # word 1 not seen → O
        assert labels[2] == "O"   # word 2 not seen → O

    def test_all_special_tokens_gives_all_o(self):
        id2label    = {0: "O"}
        word_ids    = [None, None, None]
        labels, _   = token_preds_to_word_preds([0, 0, 0], [0.9]*3, word_ids, id2label, 0)
        assert labels == []
