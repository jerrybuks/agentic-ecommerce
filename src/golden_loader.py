"""Read-only loader for Shoplytic's fixed golden evaluation fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIELD_NAMES = (
    "id",
    "category",
    "query",
    "expected_agent",
    "expected_behavior",
    "min_quality_score",
)

CASES_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "golden_data"
    / "test_cases.json"
)


def _read_cases() -> list[dict[str, Any]]:
    """Read and validate the fixed fixture without importing Shoplytic code."""
    with CASES_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    cases = payload.get("test_cases")
    if not isinstance(cases, list):
        raise TypeError("Golden fixture must contain a test_cases list")

    expected_keys = set(FIELD_NAMES)
    validated: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise TypeError(f"Golden case {index} must be an object")
        if set(case) != expected_keys:
            raise ValueError(
                f"Golden case {index} must contain exactly {FIELD_NAMES}; got {tuple(case)}"
            )
        validated.append({field: case[field] for field in FIELD_NAMES})

    return validated


def list_golden_cases(category: str | None = None) -> list[dict[str, Any]]:
    """Return source-ordered cases, optionally filtered by exact category."""
    cases = _read_cases()
    if category is None:
        return cases
    return [case for case in cases if case["category"] == category]
