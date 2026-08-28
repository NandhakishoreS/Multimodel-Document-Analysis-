"""Label mappings, unified semantic taxonomy, and dataset-specific label registries."""

from typing import Dict, List, Optional, Set, Tuple

# ==============================================================================
# DATASET-SPECIFIC LABEL REGISTRIES
# ==============================================================================

# SROIE: Task 3 Key Information Extraction (BIO Format)
SROIE_LABELS = [
    "O",
    "B-COMPANY",
    "I-COMPANY",
    "B-DATE",
    "I-DATE",
    "B-ADDRESS",
    "I-ADDRESS",
    "B-TOTAL",
    "I-TOTAL",
]

# FUNSD: Form Understanding in Noisy Scanned Documents (BIO Format)
FUNSD_LABELS = [
    "O",
    "B-HEADER",
    "I-HEADER",
    "B-QUESTION",
    "I-QUESTION",
    "B-ANSWER",
    "I-ANSWER",
]

# CORD: Consolidated Receipt Dataset (Standardized Coarse Categories BIO)
CORD_LABELS = [
    "O",
    "B-MENU",
    "I-MENU",
    "B-TOTAL",
    "I-TOTAL",
    "B-SUBTOTAL",
    "I-SUBTOTAL",
    "B-VOID",
    "I-VOID",
    "B-PAYMENT",
    "I-PAYMENT",
    "B-ETC",
    "I-ETC",
]

# Unified Cross-Dataset Coarse Taxonomy
UNIFIED_TAXONOMY = [
    "O",            # Background / unannotated text
    "HEADER",       # Document titles, section headers
    "QUESTION",     # Form prompts, field keys (e.g., 'Date:', 'Total:')
    "ANSWER",       # Field values, filled responses
    "ITEM",         # Line items, menu entries, itemized products
    "TOTAL",        # Totals, subtotals, grand amounts
    "METADATA",     # Date, address, company/vendor names
    "OTHER",        # Miscellaneous / Void
]

# Mapping from Dataset Specific Labels to Unified Taxonomy
DATASET_TO_UNIFIED_MAP = {
    # SROIE
    "O": "O",
    "B-COMPANY": "METADATA",
    "I-COMPANY": "METADATA",
    "B-DATE": "METADATA",
    "I-DATE": "METADATA",
    "B-ADDRESS": "METADATA",
    "I-ADDRESS": "METADATA",
    "B-TOTAL": "TOTAL",
    "I-TOTAL": "TOTAL",
    # FUNSD
    "B-HEADER": "HEADER",
    "I-HEADER": "HEADER",
    "B-QUESTION": "QUESTION",
    "I-QUESTION": "QUESTION",
    "B-ANSWER": "ANSWER",
    "I-ANSWER": "ANSWER",
    # CORD
    "B-MENU": "ITEM",
    "I-MENU": "ITEM",
    "B-TOTAL": "TOTAL",
    "I-TOTAL": "TOTAL",
    "B-SUBTOTAL": "TOTAL",
    "I-SUBTOTAL": "TOTAL",
    "B-VOID": "OTHER",
    "I-VOID": "OTHER",
    "B-PAYMENT": "ANSWER",
    "I-PAYMENT": "ANSWER",
    "B-ETC": "OTHER",
    "I-ETC": "OTHER",
}


class LabelEncoder:
    """Manages label-to-ID and ID-to-label encoding for a given dataset label set."""

    def __init__(self, labels: List[str]):
        self.labels = list(labels)
        self.label_to_id: Dict[str, int] = {lbl: idx for idx, lbl in enumerate(self.labels)}
        self.id_to_label: Dict[int, str] = {idx: lbl for idx, lbl in enumerate(self.labels)}

    def encode(self, label: str) -> int:
        """Convert label string to integer ID, defaulting to 'O' index if unknown."""
        return self.label_to_id.get(label, self.label_to_id.get("O", 0))

    def decode(self, idx: int) -> str:
        """Convert integer ID to label string."""
        return self.id_to_label.get(idx, "O")

    def __len__(self) -> int:
        return len(self.labels)


def get_dataset_label_encoder(dataset_name: str) -> LabelEncoder:
    """Get the appropriate LabelEncoder instance for a specific dataset."""
    name = dataset_name.lower().strip()
    if name == "sroie":
        return LabelEncoder(SROIE_LABELS)
    elif name == "funsd":
        return LabelEncoder(FUNSD_LABELS)
    elif name == "cord":
        return LabelEncoder(CORD_LABELS)
    elif name == "unified":
        return LabelEncoder(UNIFIED_TAXONOMY)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Expected 'sroie', 'funsd', 'cord', or 'unified'.")


# ==============================================================================
# DOCUMENTATION OF SEMANTIC DIFFERENCES
# ==============================================================================

DATASET_SEMANTIC_NOTES = {
    "SROIE": (
        "Focus: Key Information Extraction (KIE) from scanned receipts.\n"
        "Characteristics:\n"
        "- Flat entity extraction: Company name, Date, Address, and Total amount.\n"
        "- Document structures are single-page shopping receipts with linear receipt layouts.\n"
        "- Annotations do not include relational question-answer linking or table cell grids."
    ),
    "FUNSD": (
        "Focus: Form Understanding in Noisy Scanned Documents.\n"
        "Characteristics:\n"
        "- Hierarchical form structure: Questions (keys), Answers (values), Headers, and Other.\n"
        "- Explicit relational linking: directed edges connecting question blocks to their corresponding answers.\n"
        "- Documents are complex government/business forms with checkboxes, multi-column blocks, and diverse layouts."
    ),
    "CORD": (
        "Focus: Consolidated Receipt Dataset for complex receipt understanding.\n"
        "Characteristics:\n"
        "- Fine-grained itemized hierarchy: Menu items (name, count, price, sub-item), Totals, Subtotals, Taxes, Payment info.\n"
        "- Highly varied non-standard layouts from restaurants, supermarkets, and stores across multi-lingual contexts.\n"
        "- Annotations are line/word-level nested JSON groups rather than flat regex entities."
    ),
}
