"""Helpers shared by HTTP adapters: turn a call outcome into a Prediction."""

from __future__ import annotations

import json
import math
from typing import Any

import httpx

from fraudlab.providers.base import DecisionProvider, Prediction, Price, Status, Task, load_prices
from fraudlab.providers.http import MAX_RETRIES, CallOutcome, redact_text


def valid_probability(value: Any) -> float | None:
    """A finite number in [0, 1], or None."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


def status_for_http_failure(outcome: CallOutcome) -> Status:
    if outcome.timed_out:
        return "timeout"
    if outcome.status_code == 429:
        return "rate_limited"
    return "error"


def estimate_tokens(obj: Any) -> int:
    """Rough token estimate (about 3 characters per token for JSON). Used only for limits and budgets."""
    return math.ceil(len(json.dumps(obj, ensure_ascii=False)) / 3)


RAW_ANSWER_LIMIT = 4_000


def raw_text(value: Any) -> str | None:
    """The answer as text (or compact JSON), credential-redacted and capped in length."""
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    text = redact_text(text)
    return text if len(text) <= RAW_ANSWER_LIMIT else text[:RAW_ANSWER_LIMIT] + "…[truncated]"


def _usage_values(usage: dict[str, Any]) -> tuple[int, int, int] | None:
    it, ot = usage.get("input_tokens"), usage.get("output_tokens")
    if not isinstance(it, int) or isinstance(it, bool) or it < 0:
        return None
    if not isinstance(ot, int) or isinstance(ot, bool) or ot < 0:
        return None
    details = usage.get("input_tokens_details")
    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else usage.get("cached_input_tokens", 0)
    if not isinstance(cached, int) or isinstance(cached, bool) or cached < 0:
        return None
    return it, ot, min(cached, it)


def build_prediction(
    *,
    task: Task,
    provider: str,
    model: str,
    price: Price,
    outcome: CallOutcome | None,
    status: Status,
    score: float | None = None,
    model_reported: str | None = None,
    usage: dict[str, Any] | None = None,
    error: str | None = None,
    raw_answer: Any = None,
) -> Prediction:
    input_tokens = output_tokens = None
    cached = 0
    cost: float | None = None
    cost_source = "none"
    unknown = 0
    attempt_details = None
    if outcome is not None and outcome.attempt_details:
        history = []
        known_costs = []
        for detail in outcome.attempt_details:
            values = _usage_values(detail)
            item = dict(detail)
            if values is None:
                unknown += 1
                item["cost_usd"] = None
            else:
                it, ot, ct = values
                input_tokens = (input_tokens or 0) + it
                output_tokens = (output_tokens or 0) + ot
                cached += ct
                item["cost_usd"] = price.cost(it, ot, ct)
                known_costs.append(item["cost_usd"])
            history.append(item)
        attempt_details = json.dumps(history, separators=(",", ":"))
        if known_costs:
            cost, cost_source = sum(known_costs), "usage_calculated"
    elif usage and (values := _usage_values(usage)) is not None:
        input_tokens, output_tokens, cached = values
        cost, cost_source = price.cost(*values), "usage_calculated"
    elif outcome is not None:
        # Missing usage never proves that a rejection or timeout was free.
        unknown = outcome.attempts
    return Prediction(
        case_id=task.case_id,
        provider=provider,
        model_requested=model,
        model_reported=redact_text(model_reported) if model_reported is not None else None,
        fraud_score=score if status == "ok" else None,
        status=status,
        attempts=outcome.attempts if outcome else 0,
        latency_ms=outcome.latency_ms if outcome else None,
        total_latency_ms=outcome.total_latency_ms if outcome else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        cost_source=cost_source,  # type: ignore[arg-type]
        request_id=outcome.request_id if outcome else None,
        payload_hash=task.payload_hash,
        error=redact_text(error) if error is not None else (outcome.error if outcome and status != "ok" else None),
        raw_answer=raw_text(raw_answer),
        attempt_details=attempt_details,
        cost_unknown_attempts=unknown,
        cached_input_tokens=cached if input_tokens is not None else None,
    )


class HttpProvider(DecisionProvider):
    """Base for adapters that call a provider over HTTP with the shared client."""

    def __init__(self, client: httpx.AsyncClient, price: Price | None = None, timeout_s: float = 30.0) -> None:
        self.client = client
        self.price = price or load_prices()[self.name]
        self.timeout_s = timeout_s
        self.max_retries = MAX_RETRIES  # the load test sets 0 to see raw provider behaviour

    def _result(self, task: Task, outcome: CallOutcome | None, status: Status, **details: Any) -> Prediction:
        return build_prediction(
            task=task, provider=self.name, model=self.model, price=self.price, outcome=outcome, status=status, **details
        )
