"""Append-only call log (attempts.jsonl), written before AND after every provider call.

Used to reuse known successful answers and retain all attempts across sessions:
- a case with a finished "ok" record is never sent again;
- a call that started but never finished (crash, kill) is counted as
  "possibly billed": its reserved cost is added to spent on resume;
- failed cases (timeouts, errors) may be retried on resume.
"""

from __future__ import annotations

import json
import math
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from fraudlab.providers.base import Prediction
from fraudlab.providers.http import redact_text


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass
class LedgerSummary:
    completed: dict[tuple[str, str], Prediction] = field(default_factory=dict)  # (provider, case_id) -> ok prediction
    last: dict[tuple[str, str], Prediction] = field(default_factory=dict)  # latest finished record of any status
    spent_usd: float = 0.0
    estimated_usd: float = 0.0  # unknown-cost and possibly-billed calls
    unfinished: int = 0  # started, never finished
    by_provider: dict[str, dict[str, Any]] = field(default_factory=dict)


def prediction_costs(pred: Prediction, reserved_usd: float) -> tuple[float, float]:
    """Known usage-derived cost and an estimate for attempts lacking billing evidence.

    Legacy predictions with only final usage cannot reconstruct earlier HTTP
    retries. Their missing retry evidence is separately marked in the summary.
    """
    known = float(pred.cost_usd or 0.0)
    unknown = int(pred.cost_unknown_attempts or 0)
    if pred.cost_usd is None and pred.attempts > 0:
        estimated = reserved_usd
    else:
        estimated = reserved_usd * min(unknown, pred.attempts) / max(1, pred.attempts)
    return known, float(estimated)


def _latencies(values: list[float]) -> dict[str, float | None]:
    valid = [v for v in values if math.isfinite(v) and v >= 0]
    return {
        "p50_ms": float(np.percentile(valid, 50)) if valid else None,
        "p95_ms": float(np.percentile(valid, 95)) if valid else None,
        "p99_ms": float(np.percentile(valid, 99)) if len(valid) >= 1_000 else None,
    }


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record: dict[str, Any]) -> None:
        # A crash may leave a partial final line. Separate it from the next event
        # so the first resumed record is not lost with the malformed line.
        with self.path.open("a+") as f:
            f.seek(0, 2)
            size = f.tell()
            if size:
                f.seek(size - 1)
                if f.read(1) != "\n":
                    f.seek(0, 2)
                    f.write("\n")
            f.seek(0, 2)
            f.write(json.dumps(record) + "\n")
            f.flush()

    def start(self, provider: str, case_id: str, reserved_usd: float, *, context: dict[str, Any] | None = None) -> str:
        attempt_id = uuid.uuid4().hex
        self._append({"event": "start", "attempt_id": attempt_id, "provider": provider, "case_id": case_id,
                      "reserved_usd": reserved_usd, "at": _now(), "context": context or {}})  # fmt: skip
        return attempt_id

    def finish(self, attempt_id: str, prediction: Prediction) -> None:
        safe = prediction.as_dict()
        for name in ("error", "raw_answer", "model_reported", "request_id"):
            if isinstance(safe.get(name), str):
                safe[name] = redact_text(safe[name])
        self._append({"event": "finish", "attempt_id": attempt_id, "at": _now(), "prediction": safe})

    def summary(self) -> LedgerSummary:
        out = LedgerSummary()
        if not self.path.is_file():
            return out
        starts: dict[str, dict[str, Any]] = {}
        finished: set[str] = set()
        first: dict[tuple[str, str], str] = {}
        stats: dict[str, dict[str, Any]] = {}

        def entry(provider: str) -> dict[str, Any]:
            if provider not in stats:
                stats[provider] = {
                    "known_cost_usd": 0.0,
                    "estimated_cost_usd": 0.0,
                    "cost_accounted_usd": 0.0,
                    "unknown_cost_calls": 0,
                    "legacy_unverified_zero_cost_calls": 0,
                    "calls": 0,
                    "statuses": Counter(),
                    "first_pass_ok": 0,
                    "http_attempts": 0,
                    "http_attempts_complete": True,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_input_tokens": 0,
                    "token_usage_complete": True,
                    "http_statuses": Counter(),
                    "unpriced_http_attempts": 0,
                    "unfinished_calls": 0,
                    "_cases": set(),
                    "_ok_cases": set(),
                    "_lat_all": [],
                    "_lat_ok": [],
                    "_lat_failed": [],
                }
            return stats[provider]

        with self.path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:  # a line cut off by a crash
                    continue
                attempt_id = rec["attempt_id"]
                if rec["event"] == "start":
                    starts[attempt_id] = rec
                    first.setdefault((rec["provider"], rec["case_id"]), attempt_id)
                elif rec["event"] == "finish" and attempt_id not in finished:
                    finished.add(attempt_id)
                    pred = Prediction(**rec["prediction"])
                    key = (pred.provider, pred.case_id)
                    out.last[key] = pred
                    if pred.status == "ok":
                        out.completed[key] = pred
                    start = starts.pop(attempt_id, {})
                    reserved = start.get("reserved_usd", 0.0)
                    known, estimated = prediction_costs(pred, reserved)
                    legacy_zero = (
                        pred.attempts > 0
                        and pred.cost_usd == 0.0
                        and pred.cost_source == "none"
                        and not pred.attempt_details
                        and pred.provider != "mock"
                    )
                    out.spent_usd += known + estimated
                    out.estimated_usd += estimated
                    st = entry(pred.provider)
                    st["known_cost_usd"] += known
                    st["estimated_cost_usd"] += estimated
                    st["unknown_cost_calls"] += int(
                        estimated > 0 or (pred.cost_usd is None and pred.attempts > 0) or legacy_zero
                    )
                    st["legacy_unverified_zero_cost_calls"] += int(legacy_zero)
                    st["calls"] += 1
                    st["statuses"][pred.status] += 1
                    st["_cases"].add(pred.case_id)
                    if pred.status == "ok":
                        st["_ok_cases"].add(pred.case_id)
                        if first.get(key, attempt_id) == attempt_id and pred.attempts <= 1:
                            st["first_pass_ok"] += 1
                    st["http_attempts"] += pred.attempts if pred.provider != "mock" else 0
                    for name in ("input_tokens", "output_tokens", "cached_input_tokens"):
                        value = getattr(pred, name)
                        if value is not None:
                            st[name] += value
                        elif pred.attempts > 0 and pred.provider != "mock":
                            st["token_usage_complete"] = False
                    if pred.total_latency_ms is not None:
                        st["_lat_all"].append(pred.total_latency_ms)
                        st["_lat_ok" if pred.status == "ok" else "_lat_failed"].append(pred.total_latency_ms)
                    try:
                        details = json.loads(pred.attempt_details) if pred.attempt_details else []
                    except (ValueError, TypeError):
                        details = []
                    if pred.provider == "mock":
                        continue
                    if not isinstance(details, list) or len(details) != pred.attempts:
                        st["http_attempts_complete"] = False
                        st["unpriced_http_attempts"] += max(
                            0, pred.attempts - (1 if pred.cost_usd is not None and not legacy_zero else 0)
                        )
                    else:
                        for detail in details:
                            code = detail.get("status_code")
                            status = (
                                str(code)
                                if code is not None
                                else ("timeout" if detail.get("timed_out") else "network_error")
                            )
                            st["http_statuses"][status] += 1
                            st["unpriced_http_attempts"] += int(detail.get("cost_usd") is None)
        for attempt_id, rec in starts.items():
            if attempt_id not in finished:
                out.unfinished += 1
                out.spent_usd += rec["reserved_usd"]
                out.estimated_usd += rec["reserved_usd"]
                st = entry(rec["provider"])
                st["estimated_cost_usd"] += rec["reserved_usd"]
                st["unknown_cost_calls"] += 1
                st["unfinished_calls"] += 1
                st["calls"] += 1
                st["statuses"]["unfinished"] += 1
                st["http_attempts_complete"] = False
                st["token_usage_complete"] = False
                st["_cases"].add(rec["case_id"])
        for provider, st in stats.items():
            cases = len(st.pop("_cases"))
            st["cases"] = cases
            st["eventual_ok"] = len(st.pop("_ok_cases"))
            st["first_pass_coverage"] = st["first_pass_ok"] / cases if cases else 0.0
            st["eventual_coverage"] = st["eventual_ok"] / cases if cases else 0.0
            st["cost_accounted_usd"] = st["known_cost_usd"] + st["estimated_cost_usd"]
            st["cost_evidence_complete"] = st["unknown_cost_calls"] == 0 and st["unpriced_http_attempts"] == 0
            st["latency_all_ms"] = _latencies(st.pop("_lat_all"))
            st["latency_success_ms"] = _latencies(st.pop("_lat_ok"))
            st["latency_failure_ms"] = _latencies(st.pop("_lat_failed"))
            st["statuses"] = dict(st["statuses"])
            st["http_statuses"] = dict(st["http_statuses"])
            out.by_provider[provider] = st
        return out
