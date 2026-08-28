"""Baseline 3: Multimodal Text + Layout-Aware Baseline Model."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

from preprocessing.text_utils import clean_text, extract_lexical_features
from preprocessing.bbox_utils import extract_spatial_features
from preprocessing.label_mapping import get_dataset_label_encoder, LabelEncoder
from evaluation.metrics import compute_token_classification_metrics

logger = logging.getLogger(__name__)


class LayoutAwareBaselineModel:
    """
    Multimodal Text + Spatial Layout baseline model combining lexical TF-IDF, orthographic features,
    and 2D bounding box geometric & relational layout features.
    """

    def __init__(
        self,
        dataset_name: str,
        max_features: int = 5000,
        classifier_type: str = "logistic_regression",
        random_state: int = 42,
    ):
        self.dataset_name = dataset_name
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.label_encoder: LabelEncoder = get_dataset_label_encoder(dataset_name)

        # Vectorizers
        self.tfidf_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=max_features,
            sublinear_tf=True,
        )
        self.tfidf_word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
        )
        self.dict_vectorizer_text = DictVectorizer(sparse=True)
        self.dict_vectorizer_spatial = DictVectorizer(sparse=True)

        if classifier_type == "random_forest":
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=25,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            self.classifier = LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                random_state=self.random_state,
                C=1.0,
            )
        self.is_fitted = False

    def _extract_multimodal_features(
        self,
        samples: List[Dict[str, Any]],
        fit_vectorizers: bool = False,
    ) -> Tuple[sp.csr_matrix, np.ndarray, List[str]]:
        """
        Extract combined token-level text + spatial layout feature matrix.
        """
        tokens: List[str] = []
        lexical_dicts: List[Dict[str, Any]] = []
        spatial_dicts: List[Dict[str, float]] = []
        labels: List[str] = []

        for sample in samples:
            words = sample.get("words", [])
            bboxes = sample.get("bboxes", [])
            lbls = sample.get("labels", ["O"] * len(words))

            for idx, w in enumerate(words):
                w_clean = clean_text(w)
                box = bboxes[idx] if idx < len(bboxes) else [0, 0, 10, 10]
                prev_box = bboxes[idx - 1] if idx > 0 else None
                next_box = bboxes[idx + 1] if idx + 1 < len(bboxes) else None

                tokens.append(w_clean)
                lexical_dicts.append(extract_lexical_features(w_clean))
                spatial_dicts.append(extract_spatial_features(box, prev_bbox=prev_box, next_bbox=next_box))
                labels.append(lbls[idx] if idx < len(lbls) else "O")

        if fit_vectorizers:
            feat_char = self.tfidf_char.fit_transform(tokens)
            feat_word = self.tfidf_word.fit_transform(tokens)
            feat_lex = self.dict_vectorizer_text.fit_transform(lexical_dicts)
            feat_spatial = self.dict_vectorizer_spatial.fit_transform(spatial_dicts)
        else:
            feat_char = self.tfidf_char.transform(tokens)
            feat_word = self.tfidf_word.transform(tokens)
            feat_lex = self.dict_vectorizer_text.transform(lexical_dicts)
            feat_spatial = self.dict_vectorizer_spatial.transform(spatial_dicts)

        X = sp.hstack([feat_char, feat_word, feat_lex, feat_spatial], format="csr")
        y = np.array([self.label_encoder.encode(lbl) for lbl in labels], dtype=np.int64)

        return X, y, labels

    def fit(self, train_samples: List[Dict[str, Any]]) -> "LayoutAwareBaselineModel":
        """Fit layout-aware baseline model on training samples."""
        logger.info(f"Extracting multimodal text+layout features for {len(train_samples)} samples ({self.dataset_name})...")
        X_train, y_train, _ = self._extract_multimodal_features(train_samples, fit_vectorizers=True)
        logger.info(f"Training {self.classifier_type} on {X_train.shape[0]} tokens with {X_train.shape[1]} features...")
        self.classifier.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, samples: List[Dict[str, Any]]) -> List[List[str]]:
        """Predict token labels for document samples using text + layout features."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict()")

        predictions_by_sample: List[List[str]] = []
        for sample in samples:
            words = sample.get("words", [])
            bboxes = sample.get("bboxes", [])
            if not words:
                predictions_by_sample.append([])
                continue

            tokens = [clean_text(w) for w in words]
            lex_dicts = [extract_lexical_features(w) for w in tokens]
            spatial_dicts = []
            for idx in range(len(tokens)):
                box = bboxes[idx] if idx < len(bboxes) else [0, 0, 10, 10]
                prev_box = bboxes[idx - 1] if idx > 0 else None
                next_box = bboxes[idx + 1] if idx + 1 < len(bboxes) else None
                spatial_dicts.append(extract_spatial_features(box, prev_bbox=prev_box, next_bbox=next_box))

            feat_char = self.tfidf_char.transform(tokens)
            feat_word = self.tfidf_word.transform(tokens)
            feat_lex = self.dict_vectorizer_text.transform(lex_dicts)
            feat_spatial = self.dict_vectorizer_spatial.transform(spatial_dicts)

            X = sp.hstack([feat_char, feat_word, feat_lex, feat_spatial], format="csr")
            y_pred_ids = self.classifier.predict(X)
            y_pred_labels = [self.label_encoder.decode(pid) for pid in y_pred_ids]
            predictions_by_sample.append(y_pred_labels)

        return predictions_by_sample

    def evaluate(
        self,
        test_samples: List[Dict[str, Any]],
        ignore_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate model predictions on test samples."""
        y_true_all: List[str] = []
        y_pred_all: List[str] = []

        preds = self.predict(test_samples)
        for sample, sample_preds in zip(test_samples, preds):
            words = sample.get("words", [])
            labels = sample.get("labels", ["O"] * len(words))
            for t_lbl, p_lbl in zip(labels, sample_preds):
                y_true_all.append(t_lbl)
                y_pred_all.append(p_lbl)

        metrics = compute_token_classification_metrics(
            y_true=y_true_all,
            y_pred=y_pred_all,
            labels=self.label_encoder.labels,
            ignore_labels=ignore_labels,
        )
        return metrics

    def save(self, file_path: Union[str, Path]) -> None:
        """Save model pipeline to disk."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        logger.info(f"Saved LayoutAwareBaselineModel to {p}")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "LayoutAwareBaselineModel":
        """Load saved model pipeline from disk."""
        return joblib.load(file_path)
