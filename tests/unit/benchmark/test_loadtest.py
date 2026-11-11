"""Load test: open-loop schedule, honest 429 counting, budget and flags. Offline (respx)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from fraudlab.benchmark import loadtest as lt
from fraudlab.cli import app

runner = CliRunner()
JEV_URL = "https://api.typesafe.ai/v1/systemone"
OK = {"model": "jev-1.14.0", "answers": {"fraud": {"type": "noul", "noul": 0.3}},
      "usage": {"input_tokens": 600, "output_tokens": 20}}  # fmt: skip


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-secret")
    monkeypatch.setattr(lt, "load_env", lambda: False)


def test_needs_flag() -> None:
    r = runner.invoke(app, ["loadtest", "--providers", "jev", "--max-cost-usd", "1"])
    assert r.exit_code == 1 and "--allow-external-requests" in r.output


def test_refuses_small_budget() -> None:
    r = runner.invoke(app, ["loadtest", "--providers", "jev", "--max-cost-usd", "0.00001", "--allow-external-requests"])
    assert r.exit_code == 1 and "exceeds" in r.output


@respx.mock
async def test_counts_429_without_retrying(tmp_path: Path) -> None:
    responses = [httpx.Response(200, json=OK), httpx.Response(429)] * 10
    route = respx.post(JEV_URL).mock(side_effect=responses)
    out, results, budget = await lt.run_loadtest(["jev"], [20.0], 1.0, 1.0, outputs=tmp_path)
    r = results[0]
    assert route.call_count == 20  # one call per scheduled request: no retries
    assert r.sent == 20 and r.ok == 10 and r.rate_limited == 10
    assert r.p50_ms is not None and (out / "loadtest.json").is_file()
    import pandas as pd

    calls = pd.read_parquet(out / "calls.parquet")
    assert len(calls) == 20 and set(calls.status) == {"ok", "rate_limited"}
    known_cost = 10 * 600 * 0.042 / 1e6
    assert budget.state.estimated_usd > 0
    assert budget.state.spent_usd == pytest.approx(known_cost + budget.state.estimated_usd)
    assert r.known_cost_usd == pytest.approx(known_cost)
    assert r.estimated_cost_usd == pytest.approx(budget.state.estimated_usd)
    summary = json.loads((out / "loadtest.json").read_text())
    assert summary["dataset"]["publication_allowed"] is True
    assert summary["dataset"]["files_sha256"] and summary["source_sha256"]
    assert summary["provider_settings"]["jev"]["max_retries"] == 0
    assert summary["started_utc"] < summary["completed_utc"]
    assert summary["state"] == "complete" and len(calls.attempt_details.dropna()) == 20


@pytest.mark.parametrize(
    "providers,rates,seconds,cap",
    [
        (["jev", "jev"], [1.0], 1.0, 1.0),
        (["jev"], [float("nan")], 1.0, 1.0),
        (["jev"], [1.0], float("inf"), 1.0),
        (["jev"], [1.0], 1.0, float("nan")),
    ],
)
async def test_invalid_loadtest_options_fail_before_outputs(
    tmp_path: Path,
    providers: list[str],
    rates: list[float],
    seconds: float,
    cap: float,
) -> None:
    with pytest.raises(lt.LoadTestRefused):
        await lt.run_loadtest(providers, rates, seconds, cap, outputs=tmp_path)
    assert not list(tmp_path.iterdir())


async def test_loadtest_refuses_local_only_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = lt.load_config(lt.SYNTHETIC_CONFIG)
    cfg.dataset.external_processing_allowed = False
    monkeypatch.setattr(lt, "load_config", lambda _: cfg)
    with pytest.raises(lt.LoadTestRefused, match="external processing"):
        await lt.run_loadtest(["jev"], [1.0], 1.0, 1.0, outputs=tmp_path)
    assert not list(tmp_path.iterdir())
