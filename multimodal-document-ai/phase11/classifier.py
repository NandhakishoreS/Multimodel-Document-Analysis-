"""
Phase 11 — Bill Category Classifier.

Evidence-based, deterministic bill categorisation that operates on the raw
OCR text and extracted entity text of a financial document.

Design principles:
- Each category carries a weighted keyword vocabulary.
- A document score is the sum of weights for all matched keywords (normalised
  to [0, 1] by dividing by the theoretical maximum for that category).
- The winning category must exceed MIN_SCORE_THRESHOLD; otherwise the result
  is "Unknown".
- All matched keywords are returned as human-readable evidence tokens.
- The classifier is stateless and requires no model weights or training data.
  It can be replaced by an ML classifier later without changing the API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from phase11.schema import ClassificationResult


# ---------------------------------------------------------------------------
# Minimum confidence threshold — below this score the result is "Unknown"
# ---------------------------------------------------------------------------
MIN_SCORE_THRESHOLD = 0.10   # At least 10 % of max-possible category score

# ---------------------------------------------------------------------------
# Taxonomy hierarchy
# ---------------------------------------------------------------------------
TAXONOMY: Dict[str, str] = {
    # Utilities
    "Electricity":           "Utilities",
    "Water":                 "Utilities",
    "Internet/Telecom":      "Utilities",
    "Gas":                   "Utilities",
    # Operations
    "Office Supplies":       "Operations",
    "Equipment":             "Operations",
    "Maintenance":           "Operations",
    "Transportation":        "Operations",
    # Administrative
    "Travel":                "Administrative",
    "Accommodation":         "Administrative",
    "Training":              "Administrative",
    "Professional Services": "Administrative",
    # Other
    "Food/Catering":         "Other",
    "Miscellaneous":         "Other",
    "Unknown":               "Other",
}

# ---------------------------------------------------------------------------
# Keyword vocabularies with weights
#   weight 3 → strong signal (category-exclusive term)
#   weight 2 → moderate signal (common but usually diagnostic)
#   weight 1 → weak/supporting signal (generic word)
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS: Dict[str, List[Tuple[str, int]]] = {
    "Electricity": [
        ("electricity", 3), ("electric", 3), ("power", 2), ("energy", 2),
        ("kwh", 3), ("watt", 2), ("unit", 1), ("meter", 2), ("eb", 3),
        ("tangedco", 3), ("bescom", 3), ("msedcl", 3), ("kseb", 3),
        ("tsspdcl", 3), ("apepdcl", 3), ("pdb", 2), ("transformer", 2),
        ("load", 1), ("tariff", 2), ("consumption", 2), ("demand charge", 2),
        ("fixed charge", 1), ("connected load", 2), ("billing period", 1),
    ],
    "Water": [
        ("water", 3), ("water supply", 3), ("sewage", 2), ("drainage", 2),
        ("meter reading", 2), ("kilolitre", 3), ("kl", 2), ("litre", 2),
        ("consumption", 2), ("water board", 3), ("bwssb", 3), ("cmwssb", 3),
        ("twad", 3), ("municipal water", 3), ("tap", 1), ("pipe", 1),
    ],
    "Internet/Telecom": [
        ("internet", 3), ("broadband", 3), ("telecom", 3), ("telephone", 3),
        ("mobile", 2), ("data plan", 3), ("wifi", 2), ("fiber", 2),
        ("airtel", 3), ("jio", 3), ("bsnl", 3), ("tata", 2), ("vodafone", 3),
        ("vi", 2), ("sim", 2), ("recharge", 2), ("postpaid", 2), ("prepaid", 2),
        ("bill", 1), ("subscription", 1), ("monthly plan", 2),
    ],
    "Gas": [
        ("gas", 3), ("lpg", 3), ("png", 3), ("cng", 3), ("cylinder", 3),
        ("cooking gas", 3), ("indane", 3), ("hp gas", 3), ("bharat gas", 3),
        ("mcf", 2), ("scm", 2), ("natural gas", 3), ("pipeline gas", 3),
    ],
    "Office Supplies": [
        ("stationery", 3), ("pen", 2), ("paper", 2), ("printer", 2),
        ("toner", 3), ("cartridge", 3), ("office", 1), ("supplies", 2),
        ("envelope", 2), ("stapler", 2), ("file", 1), ("folder", 2),
        ("notebook", 2), ("desk", 1), ("chair", 1),
    ],
    "Equipment": [
        ("equipment", 3), ("machinery", 3), ("hardware", 2), ("machine", 2),
        ("instrument", 2), ("device", 2), ("laptop", 3), ("computer", 3),
        ("server", 3), ("projector", 3), ("scanner", 3), ("camera", 2),
        ("purchase", 1), ("installation", 2), ("warranty", 2),
    ],
    "Maintenance": [
        ("maintenance", 3), ("repair", 3), ("service charge", 2), ("amc", 3),
        ("annual maintenance", 3), ("overhauling", 3), ("spare", 2),
        ("parts", 1), ("technician", 2), ("labour", 2), ("labor", 2),
        ("cleaning", 2), ("pest control", 3), ("housekeeping", 3),
    ],
    "Transportation": [
        ("transport", 3), ("freight", 3), ("logistics", 3), ("courier", 3),
        ("shipping", 3), ("fuel", 2), ("vehicle", 2), ("truck", 2),
        ("bus", 2), ("cab", 2), ("taxi", 2), ("rental", 2), ("hire", 2),
        ("auto", 1), ("kilometre", 2), ("km", 1), ("petrol", 2), ("diesel", 2),
    ],
    "Travel": [
        ("travel", 3), ("flight", 3), ("airline", 3), ("airways", 3),
        ("train", 2), ("railway", 3), ("bus ticket", 3), ("boarding pass", 3),
        ("itinerary", 3), ("passport", 2), ("visa", 2), ("departure", 2),
        ("arrival", 2), ("seat", 1), ("class", 1), ("ticket", 2),
    ],
    "Accommodation": [
        ("hotel", 3), ("accommodation", 3), ("lodging", 3), ("resort", 3),
        ("inn", 2), ("guesthouse", 3), ("room", 2), ("check-in", 3),
        ("check-out", 3), ("night", 2), ("stay", 2), ("tariff", 2),
        ("bed", 1), ("breakfast", 1), ("suite", 2),
    ],
    "Training": [
        ("training", 3), ("workshop", 3), ("seminar", 3), ("conference", 3),
        ("course", 2), ("certification", 3), ("programme", 2), ("program", 2),
        ("learning", 2), ("education", 2), ("coaching", 3), ("tuition", 3),
        ("faculty", 2), ("registration", 2), ("delegate", 2),
    ],
    "Professional Services": [
        ("consulting", 3), ("consultancy", 3), ("professional", 2),
        ("legal", 3), ("audit", 3), ("accounting", 3), ("lawyer", 3),
        ("advocate", 3), ("chartered accountant", 3), ("ca ", 2),
        ("advisory", 3), ("retainer", 3), ("fee", 2), ("service", 1),
        ("engagement", 2), ("contract", 2), ("outsourcing", 3),
    ],
    "Food/Catering": [
        ("food", 3), ("catering", 3), ("restaurant", 3), ("canteen", 3),
        ("meal", 2), ("lunch", 2), ("dinner", 2), ("breakfast", 2),
        ("snack", 2), ("beverage", 2), ("refreshment", 3), ("menu", 2),
        ("order", 1), ("kitchen", 2), ("tiffin", 3), ("cafe", 3),
        ("swiggy", 3), ("zomato", 3),
    ],
    "Miscellaneous": [
        ("miscellaneous", 3), ("misc", 2), ("other", 1), ("general", 1),
        ("sundry", 2), ("incidental", 2), ("petty cash", 3),
    ],
}

# Precompile patterns for speed
_COMPILED: Dict[str, List[Tuple[re.Pattern, int]]] = {
    cat: [(re.compile(r"\b" + re.escape(kw.lower()) + r"\b"), w)
          for kw, w in kws]
    for cat, kws in CATEGORY_KEYWORDS.items()
}

# Reference score for normalisation (sum of top-3 keyword weights in category)
_MAX_SCORES: Dict[str, float] = {
    cat: float(sum(sorted([w for _, w in kws], reverse=True)[:3]))
    for cat, kws in CATEGORY_KEYWORDS.items()
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class BillCategoryClassifier:
    """
    Stateless, evidence-based document bill classifier.

    Usage
    -----
    clf = BillCategoryClassifier()
    result = clf.classify(ocr_text="TANGEDCO ELECTRICITY BILL kwh 250 units")
    print(result.category)     # "Electricity"
    print(result.confidence)   # ~0.87
    print(result.evidence)     # ["tangedco", "electricity", "kwh", "units"]
    """

    def __init__(
        self,
        min_score_threshold: float = MIN_SCORE_THRESHOLD,
        review_threshold: float = 0.60,
        extra_keywords: Optional[Dict[str, List[Tuple[str, int]]]] = None,
    ):
        self.min_score_threshold = min_score_threshold
        self.review_threshold = review_threshold
        self._compiled = dict(_COMPILED)
        self._max_scores = dict(_MAX_SCORES)
        if extra_keywords:
            for cat, kws in extra_keywords.items():
                if cat not in self._compiled:
                    self._compiled[cat] = []
                    self._max_scores[cat] = 0.0
                patterns = [(re.compile(r"\b" + re.escape(kw.lower()) + r"\b"), w)
                            for kw, w in kws]
                self._compiled[cat].extend(patterns)
                self._max_scores[cat] += sum(w for _, w in kws)

    def classify(
        self,
        ocr_text: str = "",
        vendor_name: Optional[str] = None,
        entity_text: Optional[str] = None,
        review_threshold: Optional[float] = None,
    ) -> ClassificationResult:
        """
        Classify a document into a bill category.

        Parameters
        ----------
        ocr_text : str
            Full raw OCR transcript of the document (most informative signal).
        vendor_name : Optional[str]
            Extracted vendor/company name (additional signal).
        entity_text : Optional[str]
            Concatenated extracted entity spans (additional signal).
        review_threshold : Optional[float]
            Confidence cutoff below which requires_review is set to True.
            Defaults to self.review_threshold (0.60).

        Returns
        -------
        ClassificationResult with category, sub_category, confidence, evidence, reason, requires_review.
        """
        # Build composite text corpus
        parts = [ocr_text or ""]
        if vendor_name:
            parts.append(vendor_name)
        if entity_text:
            parts.append(entity_text)
        corpus = " ".join(parts).lower()

        threshold = review_threshold if review_threshold is not None else self.review_threshold

        best_category = "Unknown"
        best_norm_score = 0.0
        best_evidence: List[str] = []

        for cat, patterns in self._compiled.items():
            raw_score = 0.0
            evidence: List[str] = []
            for pattern, weight in patterns:
                matches = pattern.findall(corpus)
                if matches:
                    raw_score += weight
                    evidence.append(matches[0])  # record one instance per keyword

            max_score = self._max_scores.get(cat, 1.0) or 1.0
            norm_score = min(raw_score / max_score, 1.0)

            if norm_score > best_norm_score:
                best_norm_score = norm_score
                best_category = cat
                best_evidence = list(dict.fromkeys(evidence))  # deduplicate, preserve order

        if best_norm_score < self.min_score_threshold:
            return ClassificationResult(
                category="Unknown",
                sub_category="Other",
                confidence=0.0,
                evidence=[],
                reason="Insufficient diagnostic keywords detected. Routed to manual review.",
                requires_review=True,
            )

        conf = round(best_norm_score, 4)
        evidence_capped = best_evidence[:10]
        requires_rev = (conf < threshold) or (best_category == "Unknown")
        if requires_rev:
            reason = (
                f"Confidence ({conf:.2f}) is below review threshold ({threshold:.2f}). "
                f"Matched keywords: {', '.join(evidence_capped)}."
            )
        else:
            reason = (
                f"Confident classification ({conf:.2f} >= {threshold:.2f}) "
                f"based on keywords: {', '.join(evidence_capped)}."
            )

        return ClassificationResult(
            category=best_category,
            sub_category=TAXONOMY.get(best_category, "Other"),
            confidence=conf,
            evidence=evidence_capped,
            reason=reason,
            requires_review=requires_rev,
        )
