"""Validate expensive-work options before any model execution or output creation."""

from __future__ import annotations

import math
from collections.abc import Collection


def validate_names(names: list[str], allowed: Collection[str]) -> None:
    if not names:
        raise ValueError("at least one provider is required")
    if len(set(names)) != len(names):
        raise ValueError("provider names must be unique")
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise ValueError(f"unknown model(s): {', '.join(unknown)}")


def finite_number(value: float, name: str, *, positive: bool = False) -> None:
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        kind = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a finite {kind} number")


def positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
