"""Replay reproduces a run's metrics exactly, without calling any model."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fraudlab.benchmark import runner as runner_mod
from fraudlab.benchmark.replay import ReplayError, replay_run
from fraudlab.benchmark.runner import RunOptions, run_benchmark
from fraudlab.datasets.config import DatasetConfig
from fraudlab.datasets.load import load_dataset
from fraudlab.datasets.synthetic import FEATURES, generate_synthetic
from fraudlab.providers import registry
from fraudlab.providers.mock import MockProvider


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(runner_mod, "load_env", lambda: False)
    monkeypatch.setattr(registry, "MockProvider", lambda: MockProvider(fail_every=5))  # include failures
    p = tmp_path / "d.parquet"
    generate_synthetic(3_000, seed=4).to_parquet(p)
    spec = {"name": "synthetic-v1", "path": str(p), "label_column": "is_fraud", "id_column": "tx_id",
            "time_column": "event_time", "features": FEATURES}  # fmt: skip
    ds = load_dataset(DatasetConfig.model_validate({"dataset": spec}))
    out, _ = run_benchmark(ds, RunOptions(providers=["rules", "xgboost", "mock"], resamples=40), outputs=tmp_path / "o")
    return out


def test_replay_is_identical(run: Path) -> None:
    _, _, differences = replay_run(run.name, outputs=run.parent)
    assert differences == []


def test_replay_detects_tampering(run: Path) -> None:
    preds = pd.read_parquet(run / "predictions.parquet")
    preds.loc[preds.provider == "rules", "fraud_score"] = 0.5  # someone edits the saved answers
    preds.to_parquet(run / "predictions.parquet", index=False)
    _, _, differences = replay_run(run.name, outputs=run.parent)
    assert "models" in differences


def test_replay_needs_all_files(run: Path) -> None:
    (run / "labels.parquet").unlink()
    with pytest.raises(ReplayError, match="labels.parquet"):
        replay_run(run.name, outputs=run.parent)


def test_replay_refuses_baseline_runs(run: Path) -> None:
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["kind"] = "baseline"
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReplayError, match="fraudlab run"):
        replay_run(run.name, outputs=run.parent)
