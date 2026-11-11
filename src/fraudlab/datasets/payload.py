"""Turn one transaction into the exact input every model receives.

Rules (the same for every provider and baseline):
- only the configured feature columns, in config order;
- never the label, ID or time column;
- floats rounded to 4 decimal places, missing values as null;
- the payload fingerprint (hash) is stored with every prediction.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fraudlab.config import REPO_ROOT
from fraudlab.datasets.load import LoadedDataset

QUESTIONS_PATH = REPO_ROOT / "configs" / "questions-v1.json"
FLOAT_DECIMALS = 4
RENDER_VERSION = "1"  # bump if the rendering rules above change


@dataclass(frozen=True)
class Question:
    version: str
    id: str
    question: str
    answer: str
    criteria_true: str
    criteria_false: str


def load_question(path: Path = QUESTIONS_PATH) -> Question:
    raw = json.loads(path.read_text())
    return Question(
        version=raw["version"],
        id=raw["id"],
        question=raw["question"],
        answer=raw["answer"],
        criteria_true=raw["criteria"]["true"],
        criteria_false=raw["criteria"]["false"],
    )


def _clean(value: Any) -> Any:
    """Convert pandas/numpy values to plain JSON values."""
    if value is None or value is pd.NA:
        return None
    if hasattr(value, "item"):  # numpy scalar -> Python scalar
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, FLOAT_DECIMALS)
    if isinstance(value, int | str):
        return value
    return str(value)


def render_case(ds: LoadedDataset, row_position: int) -> dict[str, Any]:
    """The fields of one transaction, ready to send. `row_position` is a position in ds.frame."""
    row = ds.frame.iloc[row_position]
    return {name: _clean(row[name]) for name in ds.features}


def payload_hash(fields: dict[str, Any]) -> str:
    text = json.dumps(fields, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def field_descriptions(ds: LoadedDataset) -> dict[str, str]:
    """Descriptions for the features that have one; sent once with the question."""
    described = ds.config.dataset.feature_descriptions
    return {name: described[name] for name in ds.features if name in described}


def case_id(ds: LoadedDataset, row_position: int) -> str:
    """Stable case identifier for results. Kept next to predictions, never sent to models."""
    id_col = ds.config.dataset.id_column
    return str(ds.frame.iloc[row_position][id_col]) if id_col else f"row-{row_position}"
