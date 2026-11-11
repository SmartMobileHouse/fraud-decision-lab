"""`fraudlab probe`: a tiny live check of each provider on SYNTHETIC cases only.

Safety:
- needs --allow-external-requests and --max-cost-usd;
- always uses the bundled synthetic dataset (never real data);
- checks an estimated cost before starting and reserves spending before each call;
- saves predictions and credential-redacted response bodies privately under outputs/.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from fraudlab.benchmark.baseline_run import _manifest
from fraudlab.benchmark.budget import Budget
from fraudlab.benchmark.ledger import Ledger, prediction_costs
from fraudlab.benchmark.provenance import atomic_json, json_hash, source_hash
from fraudlab.benchmark.runner import _blocked, _provider_settings
from fraudlab.benchmark.validation import finite_number, positive_integer, validate_names
from fraudlab.config import OUTPUTS, REPO_ROOT, load_env
from fraudlab.datasets.config import load_config
from fraudlab.datasets.load import load_dataset
from fraudlab.datasets.payload import load_question
from fraudlab.datasets.split import split_dataset
from fraudlab.providers.base import PRICES_PATH, MissingCredentials, Prediction
from fraudlab.providers.http import redact_text
from fraudlab.providers.registry import LIVE_PROVIDERS, make_provider, make_task, worst_case_cost

SYNTHETIC_CONFIG = REPO_ROOT / "configs" / "datasets" / "synthetic.yaml"


class ProbeRefused(RuntimeError):
    """The probe was not allowed to start (budget, flags or credentials)."""


async def run_probe(
    providers: list[str], cases: int, max_cost_usd: float, outputs: Path | None = None
) -> tuple[Path, list[Prediction]]:
    outputs = outputs or OUTPUTS
    try:
        validate_names(providers, LIVE_PROVIDERS)
        positive_integer(cases, "cases")
        if cases > 20:
            raise ValueError("cases must be <= 20")
        finite_number(max_cost_usd, "max-cost-usd")
    except ValueError as exc:
        raise ProbeRefused(str(exc)) from exc
    load_env()

    ds = load_dataset(load_config(SYNTHETIC_CONFIG))
    if not ds.config.dataset.external_processing_allowed:  # defensive: synthetic is always allowed
        raise ProbeRefused("dataset does not allow external processing")
    question = load_question()
    split = split_dataset(ds)
    holdout = split.holdout.index
    tasks = [make_task(ds, int(i), question) for i in holdout[:cases]]

    captured: list[dict[str, Any]] = []
    budget = Budget(max_cost_usd)

    async def capture(response: httpx.Response) -> None:
        await response.aread()
        auth = response.request.headers.get("authorization", "")
        secrets = [auth, auth.partition(" ")[2]]
        captured.append(
            {
                "url": redact_text(str(response.request.url), secrets),
                "status": response.status_code,
                "body": redact_text(response.text, secrets)[:20_000],
            }
        )

    async with httpx.AsyncClient(event_hooks={"response": [capture]}) as client:
        objs = [make_provider(name, client) for name in providers]
        for p in objs:
            try:
                p.credentials()
            except MissingCredentials as exc:
                raise ProbeRefused(str(exc)) from exc
        worst = sum(worst_case_cost(p, t) for p in objs for t in tasks)
        if worst > max_cost_usd:
            raise ProbeRefused(f"estimated maximum cost ${worst:.4f} exceeds --max-cost-usd ${max_cost_usd:.4f}")

        started = datetime.now(UTC)
        run_id = started.strftime("%Y%m%dT%H%M%SZ") + "-probe"
        out = outputs / run_id
        (out / "private_raw").mkdir(parents=True, mode=0o700)
        manifest = _manifest(
            run_id,
            ds,
            split,
            {
                "kind": "probe",
                "started_utc": started.isoformat(),
                "state": "running",
                "source_sha256": source_hash(),
                "question": asdict(question),
                "provider_settings": await _provider_settings(providers, True),
                "prices": PRICES_PATH.read_text(),
                "input_payloads_sha256": json_hash([t.payload_hash for t in tasks]),
                "budget": asdict(budget.state),
                "budget_kind": "estimated spending guard; not a billing guarantee",
            },
        )
        atomic_json(out / "manifest.json", manifest)
        ledger = Ledger(out / "attempts.jsonl")

        results: list[Prediction] = []
        for p in objs:  # one provider at a time, one case at a time: gentle on rate limits
            for t in tasks:
                reserve = worst_case_cost(p, t)
                if not await budget.reserve(reserve):
                    results.append(_blocked(t, p.name, p.model, "not_run", "estimated spending limit exhausted"))
                    continue
                attempt_id = ledger.start(p.name, t.case_id, reserve)
                try:
                    pred = await p.evaluate(t)
                except Exception as exc:
                    pred = replace(
                        _blocked(t, p.name, p.model, "error", f"adapter error: {type(exc).__name__}"), attempts=1
                    )
                ledger.finish(attempt_id, pred)
                known, estimated = prediction_costs(pred, reserve)
                await budget.settle(reserve, known, estimated_usd=estimated)
                results.append(pred)

    with (out / "predictions.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r.as_dict()) + "\n")
    raw_path = out / "private_raw" / "responses.jsonl"
    with raw_path.open("w") as f:
        for c in captured:
            f.write(json.dumps(c) + "\n")
    raw_path.chmod(0o600)
    manifest.update(state="complete", completed_utc=datetime.now(UTC).isoformat(), budget=asdict(budget.state))
    atomic_json(out / "manifest.json", manifest)
    summary = {**manifest, "providers": providers, "cases": cases, "estimated_maximum_cost_usd": worst,
               "known_cost_usd": sum(r.cost_usd or 0 for r in results),
               "estimated_cost_usd": budget.state.estimated_usd, "spent_usd": budget.state.spent_usd,
               "operational": ledger.summary().by_provider}  # fmt: skip
    atomic_json(out / "probe.json", summary)
    return out, results


def probe_table(results: list[Prediction]) -> str:
    lines = [f"{'provider':<17}{'status':<13}{'score':>7}{'latency':>10}{'tokens in/out':>15}{'cost $':>11}  model"]
    for r in results:
        score = f"{r.fraud_score:.3f}" if r.fraud_score is not None else "-"
        latency = f"{r.total_latency_ms:.0f} ms" if r.total_latency_ms is not None else "-"
        tokens = f"{r.input_tokens}/{r.output_tokens}" if r.input_tokens is not None else "-"
        cost = f"{r.cost_usd:.7f}" if r.cost_usd is not None else "-"
        lines.append(
            f"{r.provider:<17}{r.status:<13}{score:>7}{latency:>10}{tokens:>15}{cost:>11}  {r.model_reported or ''}"
        )
        if r.status != "ok":
            lines.append(f"{'':<17}↳ {redact_text(r.error or r.status)}; details kept in private output")
    return "\n".join(lines)


def probe_sync(providers: list[str], cases: int, max_cost_usd: float) -> tuple[Path, list[Prediction]]:
    return asyncio.run(run_probe(providers, cases, max_cost_usd))
