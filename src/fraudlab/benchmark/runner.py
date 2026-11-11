"""`fraudlab run`: score one dataset's holdout with any mix of models and save everything.

Models: "rules" and "xgboost" (local, free), "mock" (offline fake), and the live
providers (jev, openai_decisions, clef, clef_flash, llm).

Safety:
- live providers need --allow-external-requests, --max-cost-usd and keys;
- datasets with external_processing_allowed: false are never sent: every case
  is recorded as permission_blocked, whatever the flags say;
- an estimated spending guard and a call log that reuses compatible successful answers.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from fraudlab.baselines.rules import score_rules
from fraudlab.baselines.xgb import train_and_score
from fraudlab.benchmark.baseline_run import _manifest
from fraudlab.benchmark.budget import Budget
from fraudlab.benchmark.ledger import Ledger, prediction_costs
from fraudlab.benchmark.provenance import (
    atomic_json,
    atomic_parquet,
    file_hash,
    frame_hash,
    json_hash,
    run_lock,
    source_hash,
)
from fraudlab.benchmark.validation import finite_number, positive_integer, validate_names
from fraudlab.config import OUTPUTS, load_env
from fraudlab.datasets.config import ConfigError
from fraudlab.datasets.leakage import check_leakage
from fraudlab.datasets.load import LoadedDataset
from fraudlab.datasets.payload import load_question
from fraudlab.datasets.split import DatasetSplit, split_dataset
from fraudlab.evaluate.bootstrap import DEFAULT_RESAMPLES, METRICS, paired_bootstrap
from fraudlab.evaluate.metrics import evaluate
from fraudlab.providers.base import PRICES_PATH, DecisionProvider, MissingCredentials, Prediction, Task
from fraudlab.providers.common import HttpProvider
from fraudlab.providers.registry import LIVE_PROVIDERS, make_provider, make_task, worst_case_cost

LOCAL_MODELS = ("rules", "xgboost")
REFERENCE = "xgboost"


class RunRefused(RuntimeError):
    """The run was not allowed to start. The message says why and how to fix it."""


@dataclass
class RunOptions:
    providers: list[str]
    allow_external_requests: bool = False
    max_cost_usd: float | None = None
    concurrency: int = 8  # calls in flight per provider
    rate_per_s: float = 4.0  # max new calls per second per provider (~240/min)
    max_records: int | None = None  # score only a seeded sample of the holdout (smoke runs)
    resamples: int = DEFAULT_RESAMPLES
    resume_run_id: str | None = None
    progress: Callable[[str], None] | None = None


@dataclass
class _RateLimiter:
    rate_per_s: float
    _next: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + 1.0 / self.rate_per_s
        if start > now:
            await asyncio.sleep(start - now)


def _blocked(task: Task, provider: str, model: str, status: str, error: str) -> Prediction:
    return Prediction(
        case_id=task.case_id, provider=provider, model_requested=model, model_reported=None, fraud_score=None,
        status=status, attempts=0, latency_ms=None, total_latency_ms=None, input_tokens=None,  # type: ignore[arg-type]
        output_tokens=None, cost_usd=None, cost_source="none", request_id=None, payload_hash=task.payload_hash,
        error=error,
    )  # fmt: skip


async def _run_provider(
    provider: DecisionProvider, tasks: list[Task], budget: Budget, ledger: Ledger,
    done: dict[tuple[str, str], Prediction], opts: RunOptions,
) -> list[Prediction]:  # fmt: skip
    semaphore = asyncio.Semaphore(opts.concurrency)
    limiter = _RateLimiter(opts.rate_per_s)
    finished = 0

    async def one(task: Task) -> Prediction:
        nonlocal finished
        previous = done.get((provider.name, task.case_id))
        if previous is not None:
            if previous.payload_hash != task.payload_hash or previous.model_requested != provider.model:
                raise RunRefused("resume refused: a saved payload or model differs")
            return previous
        reserve = worst_case_cost(provider, task)
        async with semaphore:
            if not await budget.reserve(reserve):
                return _blocked(task, provider.name, provider.model, "not_run", "budget exhausted")
            if isinstance(provider, HttpProvider):  # speed limit only for real providers
                await limiter.wait()
            attempt_id = ledger.start(provider.name, task.case_id, reserve)
            try:
                pred = await provider.evaluate(task)
            except Exception as exc:  # an adapter bug must not stop the run or leak details
                pred = _blocked(task, provider.name, provider.model, "error", f"adapter error: {type(exc).__name__}")
                pred = Prediction(**{**pred.as_dict(), "attempts": 1})
            ledger.finish(attempt_id, pred)
            known, estimated = prediction_costs(pred, reserve)
            await budget.settle(reserve, known, estimated_usd=estimated)
        finished += 1
        if opts.progress and finished % 250 == 0:
            opts.progress(
                f"  {provider.name}: {finished:,}/{len(tasks):,} done, accounted ${budget.state.spent_usd:.4f}"
            )
        return pred

    return list(await asyncio.gather(*(one(t) for t in tasks)))


def _latency_stats(preds: list[Prediction]) -> dict[str, float | None]:
    values = np.array([p.total_latency_ms for p in preds if p.status == "ok" and p.total_latency_ms is not None])
    if len(values) == 0:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None}
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)) if len(values) >= 1_000 else None,  # unstable below 1,000
    }


def _scores(preds: list[Prediction]) -> np.ndarray:
    return np.array([np.nan if p.fraud_score is None else p.fraud_score for p in preds], dtype=float)


def _validate_options(opts: RunOptions) -> None:
    try:
        validate_names(opts.providers, (*LOCAL_MODELS, *LIVE_PROVIDERS, "mock"))
        positive_integer(opts.resamples, "resamples")
        positive_integer(opts.concurrency, "concurrency")
        if opts.concurrency > 64:
            raise ValueError("concurrency must be <= 64")
        finite_number(opts.rate_per_s, "rate", positive=True)
        if opts.max_cost_usd is not None:
            finite_number(opts.max_cost_usd, "max-cost-usd")
        if opts.max_records is not None:
            positive_integer(opts.max_records, "max-records")
        if opts.resume_run_id and (
            opts.resume_run_id in (".", "..") or Path(opts.resume_run_id).name != opts.resume_run_id
        ):
            raise ValueError("resume must be a run folder name, not a path")
    except ValueError as exc:
        raise RunRefused(str(exc)) from exc


async def _provider_settings(names: list[str], external_allowed: bool) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    async with httpx.AsyncClient() as client:
        for name in names:
            if name in LOCAL_MODELS:
                settings[name] = {"model_requested": name, "kind": "local"}
                continue
            provider = make_provider(name, client)
            if name in LIVE_PROVIDERS and external_allowed:
                try:
                    provider.credentials()
                except MissingCredentials as exc:
                    raise RunRefused(str(exc)) from exc
            entry: dict[str, Any] = {"model_requested": provider.model, "adapter": type(provider).__qualname__}
            for attr in (
                "timeout_s",
                "max_retries",
                "max_output_tokens",
                "reasoning_effort",
                "max_context_tokens",
                "body_model",
                "path_model",
                "seed",
                "fail_every",
            ):
                if hasattr(provider, attr):
                    entry[attr] = getattr(provider, attr)
            endpoint = getattr(provider, "url", None)
            if name in ("openai_decisions", "llm"):
                from fraudlab.providers.openai import API_BASE

                endpoint = API_BASE + ("/decisions" if name == "openai_decisions" else "/responses")
            elif name in ("clef", "clef_flash"):
                endpoint = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/" + str(
                    getattr(provider, "path_model", "")
                )
            entry["endpoint"] = endpoint
            entry["sampling"] = "provider defaults except explicitly recorded controls"
            if isinstance(provider, HttpProvider):
                entry["price"] = asdict(provider.price)
            settings[name] = entry
    return settings


def run_benchmark(ds: LoadedDataset, opts: RunOptions, outputs: Path | None = None) -> tuple[Path, dict[str, Any]]:
    _validate_options(opts)
    outputs = outputs or OUTPUTS
    load_env()
    spec = ds.config.dataset
    send_live = any(p in LIVE_PROVIDERS for p in opts.providers) and spec.external_processing_allowed
    if send_live and not opts.allow_external_requests:
        raise RunRefused("live providers need --allow-external-requests")
    if send_live and opts.max_cost_usd is None:
        raise RunRefused("live providers need --max-cost-usd")

    split = split_dataset(ds)
    check_leakage(ds, split)
    seed = ds.config.split.seed
    positions = split.holdout.index
    if opts.max_records is not None and opts.max_records < len(positions):
        positions = np.sort(np.random.default_rng(seed).choice(positions, size=opts.max_records, replace=False))
    labels = ds.frame[spec.label_column].to_numpy()[positions]
    if labels.sum() == 0:
        raise RunRefused("the scored cases contain no fraud; use more records")
    question = load_question()
    tasks = [make_task(ds, int(i), question) for i in positions]
    ids = [t.case_id for t in tasks]
    provider_configs = asyncio.run(_provider_settings(opts.providers, spec.external_processing_allowed))
    initial = _manifest("pending", ds, split, {})
    started = datetime.now(UTC).isoformat(timespec="microseconds")
    identity = {
        "schema": 1,
        "dataset_config": ds.config.model_dump(mode="json"),
        "input_files_sha256": initial["dataset"]["files_sha256"],
        "frame_sha256": frame_hash(ds.frame),
        "split": split.hashes,
        "case_ids_sha256": _ids_hash(ids),
        "payloads_sha256": json_hash([t.payload_hash for t in tasks]),
        "labels_sha256": json_hash(labels.tolist()),
        "question": asdict(question),
        "descriptions": tasks[0].descriptions,
        "provider_configs": provider_configs,
        "source_sha256": source_hash(),
        "prices_sha256": file_hash(PRICES_PATH),
        "max_records": opts.max_records,
        "resamples": opts.resamples,
    }
    identity_digest = json_hash(identity)
    if opts.resume_run_id:
        out = outputs / opts.resume_run_id
        if not out.is_dir():
            raise RunRefused(f"no run to resume: {out}")
    else:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{spec.name}-run"
        if Path(spec.name).name != spec.name or spec.name in (".", ".."):
            raise RunRefused("dataset name must be a folder-safe name")
        out = outputs / run_id
        try:
            out.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise RunRefused(
                "a run with this timestamp already exists; use a different output directory or resume it"
            ) from exc
    try:
        with run_lock(out):
            return _execute_run(
                ds, opts, out, initial, identity, identity_digest, started, tasks, labels, ids, split, positions
            )
    except ValueError as exc:
        raise RunRefused(str(exc)) from exc


def _execute_run(
    ds: LoadedDataset,
    opts: RunOptions,
    out: Path,
    initial: dict[str, Any],
    identity: dict[str, Any],
    identity_digest: str,
    started: str,
    tasks: list[Task],
    labels: np.ndarray,
    ids: list[str],
    split: DatasetSplit,
    positions: np.ndarray,
) -> tuple[Path, dict[str, Any]]:
    spec = ds.config.dataset
    if opts.resume_run_id:
        path = out / "manifest.json"
        if not path.is_file():
            raise RunRefused("resume refused: initial manifest is missing; legacy ledger-only runs cannot be verified")
        manifest = json.loads(path.read_text())
        if manifest.get("identity_sha256") != identity_digest or manifest.get("identity") != identity:
            raise RunRefused(
                "resume refused: inputs, labels, question, model, code or configuration differ; "
                "legacy runs without an immutable identity cannot be resumed"
            )
    else:
        manifest = initial
        manifest.update(
            run_id=out.name,
            kind="run",
            started_utc=started,
            identity=identity,
            identity_sha256=identity_digest,
            sessions=[],
        )
        manifest["run"] = {
            "providers": opts.providers,
            "scored_cases": len(tasks),
            "case_ids_sha256": _ids_hash(ids),
            "max_records": opts.max_records,
            "concurrency": opts.concurrency,
            "rate_per_s": opts.rate_per_s,
            "allow_external_requests": opts.allow_external_requests,
            "max_cost_usd": opts.max_cost_usd,
            "question_version": identity["question"]["version"],
            "question": identity["question"],
            "provider_configs": identity["provider_configs"],
            "prices": PRICES_PATH.read_text(),
            "connection_pool": "concurrency x remote providers + 10",
        }
    ledger = Ledger(out / "attempts.jsonl")
    past = ledger.summary()
    if opts.max_cost_usd is not None and past.spent_usd > opts.max_cost_usd:
        raise RunRefused("resume budget is below the amount already accounted; increase it before resuming")
    budget = Budget(opts.max_cost_usd or 0.0, past.spent_usd, past.estimated_usd)
    session = {
        "started_utc": started,
        "concurrency": opts.concurrency,
        "rate_per_s": opts.rate_per_s,
        "max_cost_usd": opts.max_cost_usd,
        "allow_external_requests": opts.allow_external_requests,
        "source_sha256": identity["source_sha256"],
        "git": initial["git"],
        "status": "running",
    }
    manifest["sessions"].append(session)
    manifest["status"] = "running"
    manifest["run"]["resumed"] = bool(opts.resume_run_id)
    atomic_json(out / "manifest.json", manifest)
    predictions: dict[str, list[Prediction]] = {}
    try:
        if "xgboost" in opts.providers:
            xgb = train_and_score(ds, split)
            all_hold = {int(p): k for k, p in enumerate(split.holdout.index)}
            xs = xgb.scores[[all_hold[int(p)] for p in positions]]
            predictions["xgboost"] = [_local(t, "xgboost", float(s)) for t, s in zip(tasks, xs, strict=True)]
            manifest["xgboost"] = {
                "params": xgb.params,
                "trees": xgb.best_iteration + 1,
                "train_seconds": xgb.train_seconds,
            }
        if "rules" in opts.providers:
            rules = score_rules(ds, positions)
            if rules.status == "ok" and rules.scores is not None:
                predictions["rules"] = [_local(t, "rules", float(s)) for t, s in zip(tasks, rules.scores, strict=True)]
            else:
                predictions["rules"] = [_blocked(t, "rules", "rules", "not_run", rules.reason) for t in tasks]

        async def remote():
            n_remote = sum(1 for n in opts.providers if n in (*LIVE_PROVIDERS, "mock"))
            pool = max(10, n_remote * opts.concurrency + 10)
            async with httpx.AsyncClient(
                limits=httpx.Limits(max_connections=pool, max_keepalive_connections=pool)
            ) as client:
                to_run = []
                for name in opts.providers:
                    if name not in (*LIVE_PROVIDERS, "mock"):
                        continue
                    provider = make_provider(name, client)
                    if name in LIVE_PROVIDERS and not spec.external_processing_allowed:
                        predictions[name] = [
                            _blocked(
                                t,
                                name,
                                provider.model,
                                "permission_blocked",
                                "dataset is local only (external_processing_allowed: false)",
                            )
                            for t in tasks
                        ]
                    else:
                        to_run.append(provider)
                if opts.progress and any(p.name in LIVE_PROVIDERS for p in to_run):
                    estimate = sum(worst_case_cost(p, t) for p in to_run for t in tasks)
                    opts.progress(
                        f"conservative cost estimate ${estimate:.4f}; guard ${opts.max_cost_usd:.2f}; "
                        "token and billing uncertainty can exceed estimates"
                    )
                results = await asyncio.gather(
                    *[_run_provider(p, tasks, budget, ledger, past.completed, opts) for p in to_run]
                )
                predictions.update({p.name: preds for p, preds in zip(to_run, results, strict=True)})

        asyncio.run(remote())
        accounting = ledger.summary().by_provider
        metrics = _metrics(predictions, labels, ids, ds.config.split.seed, opts.resamples, accounting=accounting)
        metrics.update(run_id=out.name, budget=asdict(budget.state), accounting_version=2)
        metrics["budget"]["overrun_usd"] = budget.state.overrun_usd
        atomic_parquet(
            out / "predictions.parquet", pd.DataFrame([p.as_dict() for preds in predictions.values() for p in preds])
        )
        atomic_parquet(out / "labels.parquet", pd.DataFrame({"case_id": ids, "label": labels}))
        atomic_json(out / "metrics.json", metrics)
        session.update(status="complete", completed_utc=datetime.now(UTC).isoformat(timespec="microseconds"))
        manifest.update(status="complete", completed_utc=session["completed_utc"])
        manifest["run"]["budget"] = asdict(budget.state)
        manifest["artifacts_sha256"] = {
            name: file_hash(out / name)
            for name in ("metrics.json", "predictions.parquet", "labels.parquet", "attempts.jsonl")
            if (out / name).is_file()
        }
        atomic_json(out / "manifest.json", manifest)
        return out, metrics
    except BaseException as exc:
        session.update(status="interrupted", error_type=type(exc).__name__)
        manifest["status"] = "interrupted"
        atomic_json(out / "manifest.json", manifest)
        raise


def _ids_hash(ids: list[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def _local(task: Task, name: str, score: float) -> Prediction:
    return Prediction(
        case_id=task.case_id, provider=name, model_requested=name, model_reported=name, fraud_score=score,
        status="ok", attempts=0, latency_ms=None, total_latency_ms=None, input_tokens=None, output_tokens=None,
        cost_usd=0.0, cost_source="none", request_id=None, payload_hash=task.payload_hash,
    )  # fmt: skip


def _metrics(
    predictions: dict[str, list[Prediction]],
    labels: np.ndarray,
    ids: list[str],
    seed: int,
    resamples: int,
    accounting: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    scored: dict[str, np.ndarray] = {}
    n = len(ids)
    for name, preds in predictions.items():
        statuses = pd.Series([p.status for p in preds]).value_counts().to_dict()
        ok = statuses.get("ok", 0)
        entry: dict[str, Any] = {"statuses": statuses, "coverage": ok / n}
        if accounting is not None:
            operational = dict(accounting.get(name, {}))
            known = float(operational.get("known_cost_usd", sum(p.cost_usd or 0.0 for p in preds)))
            estimated = float(operational.get("estimated_cost_usd", 0.0))
            operational.update(
                first_pass_coverage=operational.get("first_pass_ok", ok if name in LOCAL_MODELS else 0) / n,
                eventual_coverage=ok / n,
            )
            entry.update(
                operational=operational,
                known_cost_usd=known,
                estimated_cost_usd=estimated,
                cost_incomplete=bool(
                    operational.get("unknown_cost_calls", 0) or not operational.get("http_attempts_complete", True)
                ),
                cost_usd=known + estimated,
                cost_per_1k_usd=(known + estimated) / n * 1_000,
            )
        if ok == 0:
            first = next((p.error for p in preds if p.error), "")
            entry["status"] = max(statuses, key=statuses.get) if statuses else "not_run"
            entry["reason"] = first
            models[name] = entry
            continue
        s = _scores(preds)
        scored[name] = s
        m = evaluate(labels, [None if math.isnan(v) else v for v in s], ids, seed=seed).as_dict()
        cost = float(entry["cost_usd"]) if accounting is not None else sum(p.cost_usd or 0.0 for p in preds)
        caught = m["frauds_caught_top_1pct"]
        tokens_in, tokens_out = sum(p.input_tokens or 0 for p in preds), sum(p.output_tokens or 0 for p in preds)
        if accounting is not None:
            operational = entry["operational"]
            tokens_in = operational.get("input_tokens", tokens_in)
            tokens_out = operational.get("output_tokens", tokens_out)
        entry.update(
            status="ok",
            metrics=m,
            latency=_latency_stats(preds),
            cost_usd=cost,
            cost_per_1k_usd=cost / n * 1_000,
            cost_per_fraud_caught_top_1pct_usd=(cost / caught) if caught else None,
            tokens={"input": tokens_in, "output": tokens_out},
        )
        if accounting is not None:
            entry["tokens"].update(
                cached_input=entry["operational"].get("cached_input_tokens"),
                complete=entry["operational"].get("token_usage_complete", name in LOCAL_MODELS),
            )
        models[name] = entry
    reference = REFERENCE if REFERENCE in scored else None
    if scored:
        boot = paired_bootstrap(labels, scored, ids, reference=reference, resamples=resamples, seed=seed)
        for name in scored:
            models[name]["ranges_95"] = {m: asdict(boot.intervals[name][m]) for m in METRICS}
            if name in boot.differences:
                models[name][f"difference_vs_{REFERENCE}_95"] = {m: asdict(boot.differences[name][m]) for m in METRICS}
    return {
        "reference": reference,
        "scored_cases": n,
        "frauds": int(labels.sum()),
        "bootstrap": {"resamples": resamples, "seed": seed, "method": "paired, stratified by class, percentile"},
        "models": models,
    }


def format_run_table(metrics: dict[str, Any]) -> str:
    lines = [
        f"{'model':<17}{'status':<19}{'coverage':>9}{'PR-AUC':>8}{'rec@1%':>8}{'rec@5%':>8}{'p95 ms':>9}{'$/1k':>10}"
        f"  PR-AUC vs xgboost"
    ]
    for name, m in metrics["models"].items():
        if m["status"] != "ok":
            lines.append(f"{name:<17}{m['status']:<19}{m['coverage']:>9.1%}  {m.get('reason', '')}")
            continue
        p, lat = m["metrics"], m["latency"]["p95_ms"]
        diff = m.get(f"difference_vs_{REFERENCE}_95", {}).get("pr_auc")
        diff_text = "(reference)" if name == REFERENCE else ""
        if diff:
            diff_text = f"{diff['low']:+.3f} to {diff['high']:+.3f}"
        lat_text = f"{lat:.0f}" if lat is not None else "-"
        lines.append(
            f"{name:<17}{'ok':<19}{m['coverage']:>9.1%}{p['pr_auc']:>8.3f}{p['recall_top_1pct']:>8.1%}"
            f"{p['recall_top_5pct']:>8.1%}{lat_text:>9}{m['cost_per_1k_usd']:>10.4f}  {diff_text}"
        )
    return "\n".join(lines)


__all__ = ["ConfigError", "RunOptions", "RunRefused", "format_run_table", "run_benchmark"]
