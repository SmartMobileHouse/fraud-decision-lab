"""Probe safety (flags, budget, synthetic only) and replay. Offline: HTTP is mocked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import respx
from typer.testing import CliRunner

from fraudlab import probe as probe_mod
from fraudlab.cli import app
from fraudlab.datasets.config import load_config
from fraudlab.datasets.load import load_dataset
from fraudlab.datasets.payload import load_question
from fraudlab.providers.registry import ReplayProvider, make_task
from fraudlab.providers.system_one import JevProvider

runner = CliRunner()
JEV_URL = "https://api.typesafe.ai/v1/systemone"


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-secret")
    monkeypatch.setattr(probe_mod, "load_env", lambda: False)
    monkeypatch.setattr(probe_mod, "OUTPUTS", tmp_path / "outputs")


def test_probe_needs_explicit_flag() -> None:
    r = runner.invoke(app, ["probe", "--providers", "jev", "--max-cost-usd", "1"])
    assert r.exit_code == 1 and "--allow-external-requests" in r.output


def test_probe_refuses_when_budget_too_small() -> None:
    r = runner.invoke(app, ["probe", "--providers", "jev", "--max-cost-usd", "0.0000001", "--allow-external-requests"])
    assert r.exit_code == 1 and "exceeds --max-cost-usd" in r.output


def test_probe_refuses_missing_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    r = runner.invoke(app, ["probe", "--providers", "clef", "--max-cost-usd", "1", "--allow-external-requests"])
    assert r.exit_code == 1 and "CLOUDFLARE" in r.output


@respx.mock
def test_probe_runs_on_synthetic_and_saves_raw_responses(tmp_path: Path) -> None:
    route = respx.post(JEV_URL).respond(
        200, json={"model": "jev-1.14.0", "answers": {"fraud": {"type": "noul", "noul": 0.2}}, "debug": "ts-secret",
                   "usage": {"input_tokens": 300, "output_tokens": 5}},
    )  # fmt: skip
    r = runner.invoke(app, ["probe", "--providers", "jev", "--cases", "2", "--max-cost-usd", "1",
                            "--allow-external-requests"])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert route.call_count == 2
    sent = json.loads(route.calls[0].request.content)
    assert set(sent["state"]) == set(load_dataset(load_config(probe_mod.SYNTHETIC_CONFIG)).features)
    out = next((tmp_path / "outputs").iterdir())
    raw = (out / "private_raw" / "responses.jsonl").read_text()
    first = json.loads(raw.splitlines()[0])
    assert json.loads(first["body"])["answers"]["fraud"]["noul"] == 0.2
    assert "ts-secret" not in raw
    assert len((out / "predictions.jsonl").read_text().splitlines()) == 2


@respx.mock
async def test_probe_runtime_guard_stops_after_usage_overrun(tmp_path: Path) -> None:
    route = respx.post(JEV_URL).respond(
        200,
        json={
            "model": "jev-1.14.0",
            "answers": {"fraud": {"type": "noul", "noul": 0.2}},
            "usage": {"input_tokens": 1_000_000_000, "output_tokens": 0},
        },
    )
    out, predictions = await probe_mod.run_probe(["jev"], 2, 1.0, outputs=tmp_path)
    assert route.call_count == 1 and [p.status for p in predictions] == ["ok", "not_run"]
    summary = json.loads((out / "probe.json").read_text())
    assert summary["known_cost_usd"] == 42.0 and summary["budget"]["spent_usd"] > summary["budget"]["cap_usd"]
    assert "raw:" not in probe_mod.probe_table(predictions)


async def test_replay_returns_saved_and_never_calls_network() -> None:
    ds = load_dataset(load_config(probe_mod.SYNTHETIC_CONFIG))
    task = make_task(ds, 0, load_question())
    import httpx

    with respx.mock:
        respx.post(JEV_URL).respond(200, json={"model": "m", "answers": {"fraud": {"type": "noul", "noul": 0.7}},
                                               "usage": {"input_tokens": 1, "output_tokens": 0}})  # fmt: skip
        async with httpx.AsyncClient() as c:
            saved = await JevProvider(c).evaluate(task)
    replay = ReplayProvider([saved])
    assert await replay.evaluate(task) == saved
    other = make_task(ds, 1, load_question())
    assert (await replay.evaluate(other)).status == "not_run"


def test_task_has_no_label_or_id() -> None:
    ds = load_dataset(load_config(probe_mod.SYNTHETIC_CONFIG))
    task = make_task(ds, 0, load_question())
    assert "is_fraud" not in task.fields and "tx_id" not in task.fields and "scenario" not in task.fields
    assert task.criteria_true and task.criteria_false and task.descriptions
