"""Baseline 1: OCR and Text Extraction Evaluation."""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from evaluation.metrics import compute_cer, compute_wer

logger = logging.getLogger(__name__)


class OCRBaselineEvaluator:
    """
    Evaluator for OCR accuracy, text completeness, and error rates (CER & WER).
    """

    def __init__(self):
        pass

    def evaluate_transcription_fidelity(
        self,
        ground_truth_texts: List[str],
        extracted_ocr_texts: List[str],
    ) -> Dict[str, float]:
        """
        Compute CER and WER between ground truth text lines and OCR extractions.

        Args:
            ground_truth_texts: List of reference text strings.
            extracted_ocr_texts: List of hypothesis OCR strings.

        Returns:
            Dictionary with 'cer', 'wer', 'exact_match_ratio'.
        """
        if len(ground_truth_texts) != len(extracted_ocr_texts):
            raise ValueError("Ground truth and hypothesis lists must have identical lengths")

        if not ground_truth_texts:
            return {"cer": 0.0, "wer": 0.0, "exact_match_ratio": 0.0, "samples": 0}

        cer_val = compute_cer(ground_truth_texts, extracted_ocr_texts)
        wer_val = compute_wer(ground_truth_texts, extracted_ocr_texts)

        exact_matches = sum(
            1 for ref, hyp in zip(ground_truth_texts, extracted_ocr_texts) if ref.strip().lower() == hyp.strip().lower()
        )
        exact_ratio = exact_matches / max(1, len(ground_truth_texts))

        return {
            "cer": float(cer_val),
            "wer": float(wer_val),
            "exact_match_ratio": float(exact_ratio),
            "samples": len(ground_truth_texts),
        }

    def evaluate_sroie_entities_in_ocr(
        self,
        samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate what percentage of ground truth entities are fully captured by the OCR words in SROIE.
        """
        results_by_entity: Dict[str, Dict[str, float]] = {}
        entity_keys = ["company", "date", "address", "total"]

        for k in entity_keys:
            refs = []
            hyps = []
            found_count = 0
            total_count = 0

            for s in samples:
                doc_text = " ".join(s.get("words", [])).lower()
                ent_val = str(s.get("entities", {}).get(k, "")).strip()
                if ent_val:
                    total_count += 1
                    refs.append(ent_val)
                    if ent_val.lower() in doc_text:
                        found_count += 1
                        hyps.append(ent_val)
                    else:
                        # Find closest match or partial
                        hyps.append("")

            results_by_entity[k] = {
                "entity_coverage": float(found_count / max(1, total_count)),
                "total_entities": total_count,
                "cer": compute_cer(refs, hyps) if refs else 0.0,
                "wer": compute_wer(refs, hyps) if refs else 0.0,
            }

        overall_coverage = float(
            np.mean([v["entity_coverage"] for v in results_by_entity.values() if v["total_entities"] > 0])
        )

        return {
            "overall_entity_coverage": overall_coverage,
            "per_entity_results": results_by_entity,
        }
