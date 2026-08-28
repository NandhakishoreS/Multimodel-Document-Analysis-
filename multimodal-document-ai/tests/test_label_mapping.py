"""Unit tests for label encoding and dataset registries."""

import pytest
from preprocessing.label_mapping import (
    LabelEncoder,
    get_dataset_label_encoder,
    SROIE_LABELS,
    FUNSD_LABELS,
    CORD_LABELS,
    UNIFIED_TAXONOMY,
)


def test_label_encoder():
    encoder = LabelEncoder(FUNSD_LABELS)
    assert encoder.encode("B-QUESTION") == FUNSD_LABELS.index("B-QUESTION")
    assert encoder.decode(FUNSD_LABELS.index("B-ANSWER")) == "B-ANSWER"
    # Unknown fallback to 'O'
    assert encoder.encode("NON_EXISTENT") == FUNSD_LABELS.index("O")


def test_dataset_encoders():
    sroie_enc = get_dataset_label_encoder("sroie")
    assert "B-COMPANY" in sroie_enc.labels

    funsd_enc = get_dataset_label_encoder("funsd")
    assert "B-HEADER" in funsd_enc.labels

    cord_enc = get_dataset_label_encoder("cord")
    assert "B-MENU" in cord_enc.labels

    with pytest.raises(ValueError):
        get_dataset_label_encoder("invalid_dataset")
