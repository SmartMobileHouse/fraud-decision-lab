"""Estimated budget reservations and complete accounting across resumed sessions."""

from __future__ import annotations

import asyncio
import random
from dataclasses import replace
from pathlib import Path

import pytest

from fraudlab.benchmark.budget import Budget
from fraudlab.benchmark.ledger import Ledger, prediction_costs
from fraudlab.providers.base import Prediction


def pred(case: str, status: str = "ok", cost: float | None = 0.001, attempts: int = 1) -> Prediction:
    return Prediction(
        case_id=case, provider="jev", model_requested="m", model_reported="m",
        fraud_score=0.5 if status == "ok" else None, status=status, attempts=attempts,  # type: ignore[arg-type]
        latency_ms=1.0, total_latency_ms=1.0, input_tokens=1, output_tokens=0, cost_usd=cost,
        cost_source="reported" if cost is not None else "none", request_id=None, payload_hash="h",
    )  # fmt: skip


async def test_refuses_when_reservation_would_exceed_cap() -> None:
    b = Budget(0.010)
    assert await b.reserve(0.006)
    assert not await b.reserve(0.006)  # 0.012 > 0.010
    await b.settle(0.006, 0.001)  # real cost much lower than worst case
    assert await b.reserve(0.006)  # room again: 0.001 + 0.006 <= 0.010
    assert b.state.refused == 1


async def test_unknown_cost_keeps_full_reservation() -> None:
    b = Budget(1.0)
    await b.reserve(0.2)
    await b.settle(0.2, None)
    assert b.state.spent_usd == pytest.approx(0.2) and b.state.estimated_usd == pytest.approx(0.2)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.0])
async def test_invalid_spending_amounts_are_refused(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        Budget(value)
    b = Budget(1.0)
    with pytest.raises(ValueError, match="finite"):
        await b.reserve(value)
    with pytest.raises(ValueError, match="finite"):
        await b.settle(0.1, value)


async def test_usage_overrun_is_visible_and_stops_new_reservations() -> None:
    b = Budget(0.01)
    assert await b.reserve(0.01)
    await b.settle(0.01, 0.012, estimated_usd=0.003)
    assert b.state.spent_usd == pytest.approx(0.015)
    assert b.state.estimated_usd == pytest.approx(0.003)
    assert b.state.overrun_usd == pytest.approx(0.005)
    assert b.exhausted and not await b.reserve(0.001)


def test_partial_retry_usage_keeps_known_cost_and_unknown_reservation() -> None:
    p = replace(pred("c", cost=0.001, attempts=2), cost_unknown_attempts=1)
    assert prediction_costs(p, 0.02) == pytest.approx((0.001, 0.01))


async def test_cap_holds_under_heavy_concurrency() -> None:
    b = Budget(0.050)
    rng = random.Random(0)

    async def one_call() -> None:
        worst = 0.004
        if await b.reserve(worst):
            await asyncio.sleep(rng.random() / 1000)
            await b.settle(worst, rng.choice([worst, worst / 2, None]))  # real, cheaper, or unknown

    await asyncio.gather(*(one_call() for _ in range(500)))
    assert b.state.spent_usd + b.state.reserved_usd <= 0.050 + 1e-12
    assert b.state.refused > 0


def test_ledger_resume_skips_ok_and_counts_unfinished(tmp_path: Path) -> None:
    led = Ledger(tmp_path / "attempts.jsonl")
    a = led.start("jev", "c1", 0.01)
    led.finish(a, pred("c1"))  # done
    b = led.start("jev", "c2", 0.01)
    led.finish(b, pred("c2", status="timeout", cost=None, attempts=3))  # may have been billed
    led.start("jev", "c3", 0.01)  # crash: never finished
    s = Ledger(tmp_path / "attempts.jsonl").summary()
    assert set(s.completed) == {("jev", "c1")}
    assert s.unfinished == 1
    assert s.spent_usd == pytest.approx(0.001 + 0.01 + 0.01)
    assert s.estimated_usd == pytest.approx(0.02)
    assert s.last[("jev", "c2")].status == "timeout"


def test_ledger_survives_a_line_cut_off_by_a_crash(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    led = Ledger(path)
    led.finish(led.start("jev", "c1", 0.01), pred("c1"))
    with path.open("a") as f:
        f.write('{"event": "finish", "attempt_id": "x", "predic')  # partial line
    assert set(Ledger(path).summary().completed) == {("jev", "c1")}
    led.finish(led.start("jev", "c2", 0.01), pred("c2"))
    assert set(led.summary().completed) == {("jev", "c1"), ("jev", "c2")}


def test_failed_then_ok_counts_as_completed(tmp_path: Path) -> None:
    led = Ledger(tmp_path / "a.jsonl")
    led.finish(led.start("jev", "c1", 0.01), pred("c1", status="error", cost=None, attempts=1))
    led.finish(led.start("jev", "c1", 0.01), pred("c1"))
    s = led.summary()
    assert ("jev", "c1") in s.completed and s.last[("jev", "c1")].status == "ok"
    st = s.by_provider["jev"]
    assert st["calls"] == 2 and st["statuses"] == {"error": 1, "ok": 1}
    assert st["first_pass_coverage"] == 0.0 and st["eventual_coverage"] == 1.0
    assert st["known_cost_usd"] == pytest.approx(0.001)
    assert st["estimated_cost_usd"] == pytest.approx(0.01)
    assert st["cost_accounted_usd"] == pytest.approx(s.spent_usd)
    assert st["latency_failure_ms"]["p95_ms"] == 1.0
    led.finish(
        led.start("jev", "c2", 0.01),
        replace(pred("c2", status="rate_limited", cost=0.0, attempts=3), cost_source="none"),
    )
    legacy = led.summary().by_provider["jev"]
    assert legacy["legacy_unverified_zero_cost_calls"] == 1
    assert legacy["unknown_cost_calls"] == 2 and not legacy["cost_evidence_complete"]
    assert legacy["cost_accounted_usd"] == pytest.approx(s.spent_usd)  # do not invent historical charges
