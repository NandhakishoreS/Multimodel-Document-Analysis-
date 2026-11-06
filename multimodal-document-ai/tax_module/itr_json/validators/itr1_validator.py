"""
Tax Module — Official ITR-1 JSON Schema Validator (Phase 8).

Validates generated return dictionaries and JSON strings against the officially
notified Income Tax Department ITR-1 JSON Schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

_SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


class ITR1SchemaValidator:
    """Validates ITR-1 returns against official schema specifications."""

    def __init__(self, assessment_year: str = "2024-25"):
        self.assessment_year = assessment_year
        schema_ay_folder = "AY_2024_25" if assessment_year in ("2024-25", "2024") else "AY_2024_25"
        self.schema_path = _SCHEMA_DIR / schema_ay_folder / "itr1_schema.json"
        with open(self.schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)

    def validate(self, return_obj: Dict[str, Any] | str) -> Tuple[bool, List[str]]:
        """
        Validate return object or JSON string against the official schema.
        Returns: (is_valid, list_of_error_messages)
        """
        if isinstance(return_obj, str):
            try:
                data = json.loads(return_obj)
            except json.JSONDecodeError as e:
                return False, [f"Invalid JSON format: {e}"]
        else:
            data = return_obj

        validator = jsonschema.Draft7Validator(self.schema)
        errors: List[str] = []

        for err in validator.iter_errors(data):
            path_str = " -> ".join([str(p) for p in err.absolute_path]) or "root"
            errors.append(f"Field '{path_str}': {err.message}")

        return len(errors) == 0, errors
