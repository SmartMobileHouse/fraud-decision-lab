"""The common contract every provider follows.

A provider gets a Task (one transaction + the question) and returns a
Prediction (a fraud score or a clear failure status, plus latency, tokens and
cost). Provider differences stay inside each adapter.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from fraudlab.config import REPO_ROOT

PRICES_PATH = REPO_ROOT / "configs" / "prices-2026-11-06.yaml"

Status = Literal["ok", "error", "timeout", "refusal", "invalid", "rate_limited", "not_run", "permission_blocked"]


@dataclass(frozen=True)
class Task:
    """Everything a provider may see for one case. Identical for every provider."""

    case_id: str  # for our records only; adapters must never send it
    fields: dict[str, Any]
    payload_hash: str
    question: str
    criteria_true: str
    criteria_false: str
    descriptions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    case_id: str
    provider: str
    model_requested: str
    model_reported: str | None
    fraud_score: float | None
    status: Status
    attempts: int
    latency_ms: float | None  # last attempt only
    total_latency_ms: float | None  # all attempts, including retry waits
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    cost_source: Literal["reported", "usage_calculated", "estimated", "none"]  # "reported" is legacy, not an invoice
    request_id: str | None
    payload_hash: str
    error: str | None = None  # short, sanitised; never contains keys
    raw_answer: str | None = None  # original answer with credential redaction and a 4,000-character limit
    attempt_details: str | None = None  # JSON metadata for every HTTP attempt; never bodies or headers
    cost_unknown_attempts: int = 0
    cached_input_tokens: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Price:
    input_per_m: float
    output_per_m: float
    cached_input_per_m: float | None
    source: str

    def cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        cached = min(cached_tokens, input_tokens) if self.cached_input_per_m is not None else 0
        return (
            (input_tokens - cached) * self.input_per_m
            + cached * (self.cached_input_per_m or 0.0)
            + output_tokens * self.output_per_m
        ) / 1_000_000


def load_prices(path: Path = PRICES_PATH) -> dict[str, Price]:
    raw = yaml.safe_load(path.read_text())["providers"]
    return {
        name: Price(
            input_per_m=float(p["input"]),
            output_per_m=float(p["output"]),
            cached_input_per_m=float(p["cached_input"]) if "cached_input" in p else None,
            source=str(p["source"]),
        )
        for name, p in raw.items()
    }


class MissingCredentials(RuntimeError):
    """A live provider was requested but its keys are not set."""


class DecisionProvider(ABC):
    """Base class for every provider adapter."""

    name: str  # e.g. "jev"
    model: str  # exact model id requested
    required_env: tuple[str, ...] = ()

    def credentials(self) -> dict[str, str]:
        missing = [k for k in self.required_env if not os.environ.get(k, "").strip()]
        if missing:
            raise MissingCredentials(f"{self.name}: missing {', '.join(missing)} (see .env.example)")
        return {k: os.environ[k].strip() for k in self.required_env}

    @abstractmethod
    async def evaluate(self, task: Task) -> Prediction:
        """Score one task. Must never raise for provider-side problems: return a failure status instead."""
