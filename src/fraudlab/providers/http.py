"""One shared HTTP layer for every provider, so timing and retries are measured the same way.

- Timeout per attempt (default 30 s).
- Up to 2 retries on 429, 5xx (incl. TypeSafe 529) and timeouts, with jittered
  exponential backoff; a Retry-After header is honoured (capped).
- Never retries 400/401/403/404/413/415/422: those will not fix themselves.
- Public errors omit response bodies. Responses retained in memory redact known
  credentials; per-attempt records contain status/timing/token metadata only.
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import httpx

RETRYABLE = {429, 500, 502, 503, 504, 529}
MAX_RETRIES = 2
BASE_BACKOFF_S = 1.0
MAX_BACKOFF_S = 20.0
REQUEST_ID_HEADERS = ("x-request-id", "request-id", "cf-ray", "x-typesafe-request-id")


@dataclass(frozen=True)
class CallOutcome:
    status_code: int | None  # None when no response arrived (timeout / network)
    body: Any  # parsed JSON, or None
    attempts: int
    latency_ms: float | None  # last attempt
    total_latency_ms: float
    request_id: str | None
    error: str | None  # set when the call ultimately failed
    timed_out: bool = False
    attempt_details: tuple[dict[str, Any], ...] = ()


def redact_text(text: str, secrets: Iterable[str] = ()) -> str:
    """Remove known credentials before any response-derived text is retained or printed."""
    values = set(secrets)
    values.update(
        value
        for name, value in os.environ.items()
        if any(part in name.upper() for part in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "ACCOUNT_ID"))
    )
    for value in sorted((v for v in values if len(v) >= 4), key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    return re.sub(r"\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{12,}|AKIA[A-Z0-9]{16})\b", "[REDACTED]", text)


def _redact(value: Any, secrets: Iterable[str]) -> Any:
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, dict):
        return {redact_text(str(key), secrets): _redact(item, secrets) for key, item in value.items()}
    return value


def _attempt_metadata(status_code: int | None, body: Any, latency_ms: float, timed_out: bool = False) -> dict[str, Any]:
    """Usage is retained before retries, without recording arbitrary provider content."""
    data = body
    if isinstance(data, dict) and isinstance(data.get("result"), dict):
        data = data["result"]
    usage = data.get("usage") if isinstance(data, dict) else None
    result: dict[str, Any] = {"status_code": status_code, "latency_ms": latency_ms, "timed_out": timed_out}
    if isinstance(usage, dict):
        for name in ("input_tokens", "output_tokens"):
            value = usage.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                result[name] = value
        details = usage.get("input_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        if isinstance(cached, int) and not isinstance(cached, bool) and cached >= 0:
            result["cached_input_tokens"] = cached
    return result


def _request_id(headers: httpx.Headers) -> str | None:
    for name in REQUEST_ID_HEADERS:
        if name in headers:
            return headers[name]
    return None


def _backoff(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            seconds = float(retry_after)
            if math.isfinite(seconds) and seconds >= 0:
                return min(seconds, MAX_BACKOFF_S)
        except ValueError:
            pass
    return min(BASE_BACKOFF_S * 2**attempt, MAX_BACKOFF_S) * random.uniform(0.5, 1.0)  # noqa: S311


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_s: float = 30.0,
    max_retries: int = MAX_RETRIES,
) -> CallOutcome:
    if not math.isfinite(timeout_s) or timeout_s <= 0 or max_retries < 0:
        raise ValueError("timeout must be finite and positive; retries must be nonnegative")
    started = time.perf_counter()
    last_latency: float | None = None
    attempt = 0
    history: list[dict[str, Any]] = []
    secret_values = [
        value for name, value in headers.items() if name.lower() in ("authorization", "x-api-key", "api-key")
    ]
    secret_values += [value.split(" ", 1)[1] for value in secret_values if " " in value]
    while True:
        attempt += 1
        t0 = time.perf_counter()
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=timeout_s)
        except httpx.TimeoutException:
            last_latency = (time.perf_counter() - t0) * 1000
            history.append(_attempt_metadata(None, None, last_latency, timed_out=True))
            if attempt <= max_retries:
                await asyncio.sleep(_backoff(attempt - 1, None))
                continue
            return CallOutcome(
                None,
                None,
                attempt,
                last_latency,
                _ms(started),
                None,
                "timeout",
                timed_out=True,
                attempt_details=tuple(history),
            )
        except httpx.HTTPError as exc:  # network problems: DNS, connection refused, ...
            last_latency = (time.perf_counter() - t0) * 1000
            history.append(_attempt_metadata(None, None, last_latency))
            if attempt <= max_retries:
                await asyncio.sleep(_backoff(attempt - 1, None))
                continue
            return CallOutcome(
                None,
                None,
                attempt,
                last_latency,
                _ms(started),
                None,
                type(exc).__name__,
                attempt_details=tuple(history),
            )

        last_latency = (time.perf_counter() - t0) * 1000
        request_id = _request_id(response.headers)
        if request_id is not None:
            request_id = redact_text(request_id, secret_values)
        try:
            body = _redact(response.json(), secret_values)
        except ValueError:
            body = None
        history.append(_attempt_metadata(response.status_code, body, last_latency))
        if response.status_code in RETRYABLE and attempt <= max_retries:
            await asyncio.sleep(_backoff(attempt - 1, response.headers.get("retry-after")))
            continue
        error = None
        if response.status_code >= 400:
            error = f"HTTP {response.status_code} (provider response omitted)"
        return CallOutcome(
            response.status_code,
            body,
            attempt,
            last_latency,
            _ms(started),
            request_id,
            error,
            attempt_details=tuple(history),
        )


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
