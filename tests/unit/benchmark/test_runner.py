"""`fraudlab run`: blocking, budget, resume and outputs. Offline: HTTP is mocked with respx."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import respx

from fraudlab.benchmark import runner as runner_mod
from fraudlab.benchmark.runner import RunOptions, RunRefused, run_benchmark
from fraudlab.datasets.config import DatasetConfig
from fraudlab.datasets.load import LoadedDataset, load_dataset
from fraudlab.datasets.synthetic import FEATURES, generate_synthetic
from fraudlab.providers import http as http_mod

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_OK = {"model": "jev-1.14.0", "answers": {"fraud": {"type": "noul", "noul": 0.3}},
          "usage": {"input_tokens": 600, "output_tokens": 20}}  # fmt: skip


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-secret")
    monkeypatch.setattr(runner_mod, "load_env", lambda: False)

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(http_mod.asyncio, "sleep", no_sleep)


def make_ds(tmp_path: Path, allowed: bool = True) -> LoadedDataset:
    p = tmp_path / "d.parquet"
    generate_synthetic(4_000, seed=21).to_parquet(p)
    spec = {
        "name": "synthetic-v1", "path": str(p), "label_column": "is_fraud", "id_column": "tx_id",
        "time_column": "event_time", "features": FEATURES, "external_processing_allowed": allowed,
    }  # fmt: skip
    return load_dataset(DatasetConfig.model_validate({"dataset": spec}))


def opts(**kw: object) -> RunOptions:
    base: dict[str, object] = {"providers": ["rules", "xgboost", "mock"], "resamples": 30, "rate_per_s": 1_000}
    return RunOptions(**{**base, **kw})  # type: ignore[arg-type]


def test_offline_run_writes_everything(tmp_path: Path) -> None:
    out, metrics = run_benchmark(make_ds(tmp_path), opts(), outputs=tmp_path / "o")
    assert {"manifest.json", "metrics.json", "predictions.parquet", "labels.parquet"} <= {p.name for p in out.iterdir()}
    assert metrics["reference"] == "xgboost"
    assert set(metrics["models"]) == {"rules", "xgboost", "mock"}
    preds = pd.read_parquet(out / "predictions.parquet")
    assert len(preds) == 3 * metrics["scored_cases"]


def test_live_provider_needs_flag_and_budget(tmp_path: Path) -> None:
    with pytest.raises(RunRefused, match="allow-external-requests"):
        run_benchmark(make_ds(tmp_path), opts(providers=["jev"]), outputs=tmp_path / "o")
    with pytest.raises(RunRefused, match="max-cost-usd"):
        run_benchmark(make_ds(tmp_path), opts(providers=["jev"], allow_external_requests=True), outputs=tmp_path / "o")


@respx.mock
def test_local_only_dataset_is_never_sent(tmp_path: Path) -> None:
    route = respx.post(JEV_URL).respond(200, json=JEV_OK)
    _, metrics = run_benchmark(
        make_ds(tmp_path, allowed=False),
        opts(providers=["xgboost", "jev"], allow_external_requests=True, max_cost_usd=10),
        outputs=tmp_path / "o",
    )
    assert route.call_count == 0
    assert metrics["models"]["jev"]["status"] == "permission_blocked"


@respx.mock
def test_budget_exhaustion_stops_sending(tmp_path: Path) -> None:
    route = respx.post(JEV_URL).respond(200, json=JEV_OK)
    out, metrics = run_benchmark(
        make_ds(tmp_path),
        opts(providers=["xgboost", "jev"], allow_external_requests=True, max_cost_usd=0.0005, max_records=200),
        outputs=tmp_path / "o",
    )
    b = metrics["budget"]
    assert b["refused"] > 0 and route.call_count < 200
    assert b["spent_usd"] + b["reserved_usd"] <= 0.0005 + 1e-12
    assert metrics["models"]["jev"]["statuses"]["not_run"] == b["refused"]


@respx.mock
def test_resume_never_pays_twice(tmp_path: Path) -> None:
    route = respx.post(JEV_URL).respond(200, json=JEV_OK)
    ds = make_ds(tmp_path)
    live = {"providers": ["xgboost", "jev"], "allow_external_requests": True, "max_records": 100}
    out, first = run_benchmark(ds, opts(**live, max_cost_usd=0.001), outputs=tmp_path / "o")
    sent_first = route.call_count
    assert first["budget"]["refused"] > 0  # budget ran out part-way

    _, second = run_benchmark(ds, opts(**live, max_cost_usd=1.0, resume_run_id=out.name), outputs=tmp_path / "o")
    assert route.call_count == 100  # only the missing cases were sent
    assert route.call_count - sent_first == 100 - sent_first
    assert second["models"]["jev"]["coverage"] == 1.0
    starts = [json.loads(x) for x in (out / "attempts.jsonl").read_text().splitlines() if '"start"' in x]
    assert len({s["case_id"] for s in starts}) == len(starts) == 100  # each case sent exactly once
    before = (out / "manifest.json").read_bytes()
    with pytest.raises(RunRefused, match="below"):
        run_benchmark(ds, opts(**live, max_cost_usd=0.0, resume_run_id=out.name), outputs=tmp_path / "o")
    assert (out / "manifest.json").read_bytes() == before


def test_resume_refuses_different_cases(tmp_path: Path) -> None:
    ds = make_ds(tmp_path)
    out, _ = run_benchmark(ds, opts(max_records=100), outputs=tmp_path / "o")
    with pytest.raises(RunRefused, match="differ"):
        run_benchmark(ds, opts(max_records=50, resume_run_id=out.name), outputs=tmp_path / "o")


def test_failures_count_against_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fraudlab.providers import registry
    from fraudlab.providers.mock import MockProvider

    monkeypatch.setattr(registry, "MockProvider", lambda: MockProvider(fail_every=4))
    _, metrics = run_benchmark(make_ds(tmp_path), opts(), outputs=tmp_path / "o")
    m = metrics["models"]["mock"]
    assert m["coverage"] == pytest.approx(0.75, abs=0.01) and m["statuses"]["error"] > 0


@pytest.mark.parametrize(
    "invalid",
    [
        {"resamples": 0},
        {"max_cost_usd": float("nan")},
        {"rate_per_s": float("inf")},
        {"providers": ["mock", "mock"]},
        {"providers": []},
        {"resume_run_id": ".."},
    ],
)
def test_invalid_options_refuse_before_work(tmp_path: Path, invalid: dict) -> None:
    output = tmp_path / "never-created"
    with pytest.raises(RunRefused):
        run_benchmark(make_ds(tmp_path), opts(**invalid), outputs=output)
    assert not output.exists()


def test_resume_rejects_changed_features_or_labels(tmp_path: Path) -> None:
    ds = make_ds(tmp_path)
    options = opts(providers=["mock"], max_records=100, resamples=10)
    out, _ = run_benchmark(ds, options, outputs=tmp_path / "o")
    ds.frame.loc[0, "amount"] += 1  # training input also belongs to experiment identity
    options.resume_run_id = out.name
    with pytest.raises(RunRefused, match="differ"):
        run_benchmark(ds, options, outputs=tmp_path / "o")
    ds.frame.loc[0, "amount"] -= 1
    ds.frame.loc[0, "is_fraud"] = 1 - ds.frame.loc[0, "is_fraud"]
    with pytest.raises(RunRefused, match="differ"):
        run_benchmark(ds, options, outputs=tmp_path / "o")


def test_interrupted_run_resumes_from_initial_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = make_ds(tmp_path)
    options = opts(providers=["mock"], max_records=100, resamples=10)
    original = runner_mod._metrics

    def interrupted(*args, **kwargs):
        raise RuntimeError("simulated interruption after calls")

    monkeypatch.setattr(runner_mod, "_metrics", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        run_benchmark(ds, options, outputs=tmp_path / "o")
    out = next((tmp_path / "o").iterdir())
    before = (out / "attempts.jsonl").read_text()
    initial = json.loads((out / "manifest.json").read_text())
    assert initial["status"] == "interrupted" and initial["identity_sha256"]
    monkeypatch.setattr(runner_mod, "_metrics", original)
    options.resume_run_id = out.name
    _, metrics = run_benchmark(ds, options, outputs=tmp_path / "o")
    assert (out / "attempts.jsonl").read_text() == before
    final = json.loads((out / "manifest.json").read_text())
    assert final["started_utc"] == initial["started_utc"]
    assert final["started_utc"] < final["completed_utc"]
    assert final["status"] == "complete" and len(final["sessions"]) == 2
    assert metrics["models"]["mock"]["coverage"] == 1
