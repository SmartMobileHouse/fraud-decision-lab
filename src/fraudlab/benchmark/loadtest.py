"""`fraudlab loadtest`: steady request rates, like production traffic (synthetic cases only).

For each provider and each rate (e.g. 1, 5, 20 requests per second), requests are
sent on a fixed schedule for N seconds WITHOUT waiting for earlier answers
("open loop"). Retries are off, so "too busy" (429) answers are counted honestly.
Measured: achieved rate, p50/p95 latency, share of 429s, other errors, cost.
Quality metrics are not computed here.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from fraudlab.benchmark.baseline_run import _manifest
from fraudlab.benchmark.budget import Budget
from fraudlab.benchmark.ledger import Ledger, prediction_costs
from fraudlab.benchmark.provenance import atomic_json, atomic_parquet, json_hash, source_hash
from fraudlab.benchmark.runner import _blocked, _provider_settings
from fraudlab.benchmark.validation import finite_number, validate_names
from fraudlab.config import OUTPUTS, REPO_ROOT, load_env
from fraudlab.datasets.config import load_config
from fraudlab.datasets.load import load_dataset
from fraudlab.datasets.payload import load_question
from fraudlab.datasets.split import split_dataset
from fraudlab.providers.base import PRICES_PATH, MissingCredentials, Prediction, Task
from fraudlab.providers.common import HttpProvider
from fraudlab.providers.registry import LIVE_PROVIDERS, make_provider, make_task, worst_case_cost

SYNTHETIC_CONFIG = REPO_ROOT / "configs" / "datasets" / "synthetic.yaml"


class LoadTestRefused(RuntimeError):
    """The load test was not allowed to start."""


@dataclass(frozen=True)
class StepResult:
    provider: str
    target_rps: float
    seconds: float
    scheduled: int
    sent: int  # scheduled minus those refused by the budget
    ok: int
    rate_limited: int
    errors: int
    timeouts: int
    achieved_rps: float  # answered requests per second of wall time
    p50_ms: float | None
    p95_ms: float | None
    cost_usd: float
    known_cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0
    unknown_cost_calls: int = 0
    p95_all_ms: float | None = None
    budget_refused: int = 0
    wall_seconds: float = 0.0


def _summarise(provider: str, rate: float, seconds: float, scheduled: int, preds: list[Prediction],
               wall_s: float, estimated_cost: float = 0.0) -> StepResult:  # fmt: skip
    ok = [p for p in preds if p.status == "ok"]
    lat = np.array([p.latency_ms for p in ok if p.latency_ms is not None])
    all_lat = [p.total_latency_ms for p in preds if p.total_latency_ms is not None]
    known_cost = sum(p.cost_usd or 0.0 for p in preds)
    return StepResult(
        provider=provider, target_rps=rate, seconds=seconds, scheduled=scheduled, sent=len(preds), ok=len(ok),
        rate_limited=sum(p.status == "rate_limited" for p in preds),
        errors=sum(p.status in ("error", "invalid", "refusal") for p in preds),
        timeouts=sum(p.status == "timeout" for p in preds),
        achieved_rps=len(ok) / wall_s if wall_s > 0 else 0.0,
        p50_ms=float(np.percentile(lat, 50)) if len(lat) else None,
        p95_ms=float(np.percentile(lat, 95)) if len(lat) else None,
        cost_usd=known_cost,
        known_cost_usd=known_cost,
        estimated_cost_usd=estimated_cost,
        unknown_cost_calls=sum(p.cost_usd is None or bool(p.cost_unknown_attempts) for p in preds),
        p95_all_ms=float(np.percentile(all_lat, 95)) if all_lat else None,
        budget_refused=scheduled-len(preds),
        wall_seconds=wall_s,
    )  # fmt: skip


async def _step(
    provider: HttpProvider,
    tasks: list[Task],
    rate: float,
    seconds: float,
    budget: Budget,
    step_log: list[dict[str, Any]],
    ledger: Ledger,
) -> StepResult:
    total = int(rate * seconds)
    start = time.perf_counter()
    jobs: list[asyncio.Task[Prediction | None]] = []
    estimates: list[float] = []

    async def one(i: int) -> Prediction | None:
        task = tasks[i % len(tasks)]
        reserve = worst_case_cost(provider, task, attempts=1)
        if not await budget.reserve(reserve):
            return None
        attempt_id = ledger.start(provider.name, task.case_id, reserve, context={"target_rps": rate})
        try:
            pred = await provider.evaluate(task)
        except Exception as exc:
            pred = replace(
                _blocked(task, provider.name, provider.model, "error", f"adapter error: {type(exc).__name__}"),
                attempts=1,
            )
        ledger.finish(attempt_id, pred)
        known, estimated = prediction_costs(pred, reserve)
        await budget.settle(reserve, known, estimated_usd=estimated)
        estimates.append(estimated)
        step_log.append(
            {
                **pred.as_dict(),
                "target_rps": rate,
                "reserved_usd": reserve,
                "known_cost_usd": known,
                "estimated_cost_usd": estimated,
            }
        )
        return pred

    for i in range(total):  # open loop: fire on schedule, never wait for answers
        delay = start + i / rate - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        jobs.append(asyncio.create_task(one(i)))
    results = [r for r in await asyncio.gather(*jobs) if r is not None]
    wall = time.perf_counter() - start
    return _summarise(provider.name, rate, seconds, total, results, wall, sum(estimates))


async def run_loadtest(
    providers: list[str], rates: list[float], seconds: float, max_cost_usd: float, outputs: Path | None = None
) -> tuple[Path, list[StepResult], Budget]:
    try:
        validate_names(providers, LIVE_PROVIDERS)
        finite_number(seconds, "seconds", positive=True)
        finite_number(max_cost_usd, "max-cost-usd")
        if not rates or len(set(rates)) != len(rates):
            raise ValueError("rates must be nonempty and unique")
        for rate in rates:
            finite_number(rate, "rate", positive=True)
            finite_number(rate * seconds, "scheduled calls", positive=True)
            if rate * seconds < 1:
                raise ValueError("each rate step must schedule at least one request")
    except ValueError as exc:
        raise LoadTestRefused(str(exc)) from exc
    load_env()
    ds = load_dataset(load_config(SYNTHETIC_CONFIG))
    if not ds.config.dataset.external_processing_allowed:
        raise LoadTestRefused("dataset does not allow external processing")
    question = load_question()
    split = split_dataset(ds)
    tasks = [make_task(ds, int(i), question) for i in split.holdout.index[:500]]

    budget = Budget(max_cost_usd)
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=50)
    results: list[StepResult] = []
    calls: list[dict[str, Any]] = []  # every call, kept for later inspection
    async with httpx.AsyncClient(limits=limits) as client:
        objs = []
        for name in providers:
            p = make_provider(name, client)
            if not isinstance(p, HttpProvider):
                raise LoadTestRefused(f"{name} is not a live provider")
            try:
                p.credentials()
            except MissingCredentials as exc:
                raise LoadTestRefused(str(exc)) from exc
            p.max_retries = 0
            objs.append(p)
        worst = sum(
            max(worst_case_cost(p, t, attempts=1) for t in tasks) * int(r * seconds) for p in objs for r in rates
        )
        if worst > max_cost_usd:
            raise LoadTestRefused(f"estimated maximum cost ${worst:.4f} exceeds --max-cost-usd ${max_cost_usd:.4f}")
        started = datetime.now(UTC)
        run_id = started.strftime("%Y%m%dT%H%M%SZ") + "-loadtest"
        out = (outputs or OUTPUTS) / run_id
        out.mkdir(parents=True)
        provider_settings = await _provider_settings(providers, True)
        for settings in provider_settings.values():
            settings["max_retries"] = 0
        manifest = _manifest(
            run_id,
            ds,
            split,
            {
                "kind": "loadtest",
                "started_utc": started.isoformat(),
                "state": "running",
                "source_sha256": source_hash(),
                "question": asdict(question),
                "provider_settings": provider_settings,
                "prices": PRICES_PATH.read_text(),
                "providers": providers,
                "rates": rates,
                "seconds": seconds,
                "input_payloads_sha256": json_hash([t.payload_hash for t in tasks]),
                "measurement": {
                    "schedule": "open loop",
                    "retries": 0,
                    "warmup": "none",
                    "input_cases": len(tasks),
                    "input_reuse": "cycle through first holdout cases",
                    "client_max_connections": 200,
                    "client_max_keepalive_connections": 50,
                    "latency": "request including client/network time; success and all-call summaries",
                    "throughput": "successful responses / wall time including response drain",
                },
                "estimated_maximum_cost_usd": worst,
                "budget": asdict(budget.state),
                "budget_kind": "estimated spending guard; not a billing guarantee",
            },
        )
        atomic_json(out / "manifest.json", manifest)
        ledger = Ledger(out / "attempts.jsonl")
        for p in objs:  # one provider at a time, so providers do not slow each other down
            for rate in rates:
                results.append(await _step(p, tasks, rate, seconds, budget, calls, ledger))
                atomic_parquet(out / "calls.parquet", pd.DataFrame(calls))
                atomic_json(
                    out / "loadtest.json",
                    {
                        **manifest,
                        "budget": asdict(budget.state),
                        "steps": [asdict(r) for r in results],
                        "operational": ledger.summary().by_provider,
                    },
                )
                await asyncio.sleep(2)  # short pause between steps

    manifest.update(state="complete", completed_utc=datetime.now(UTC).isoformat(), budget=asdict(budget.state))
    atomic_json(out / "manifest.json", manifest)
    summary = {**manifest, "steps": [asdict(r) for r in results], "operational": ledger.summary().by_provider}
    atomic_json(out / "loadtest.json", summary)
    pd.DataFrame([asdict(r) for r in results]).to_csv(out / "loadtest.csv", index=False)
    return out, results, budget


def loadtest_table(results: list[StepResult]) -> str:
    lines = [f"{'provider':<17}{'target/s':>9}{'got/s':>8}{'sent':>7}{'ok':>7}{'429':>6}{'err':>6}{'t/o':>5}"
             f"{'p50 ms':>9}{'p95 ms':>9}{'known $':>10}{'est. $':>10}"]  # fmt: skip
    for r in results:
        p50 = f"{r.p50_ms:.0f}" if r.p50_ms is not None else "-"
        p95 = f"{r.p95_ms:.0f}" if r.p95_ms is not None else "-"
        lines.append(
            f"{r.provider:<17}{r.target_rps:>9g}{r.achieved_rps:>8.1f}{r.sent:>7}{r.ok:>7}{r.rate_limited:>6}"
            f"{r.errors:>6}{r.timeouts:>5}{p50:>9}{p95:>9}{r.known_cost_usd:>10.5f}{r.estimated_cost_usd:>10.5f}"
        )
    return "\n".join(lines)
