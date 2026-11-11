"""`fraudlab baseline`: results are saved, complete and consistent."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fraudlab.benchmark.baseline_run import format_table, run_baselines
from fraudlab.datasets.config import DatasetConfig
from fraudlab.datasets.load import LoadedDataset, load_dataset
from fraudlab.datasets.synthetic import FEATURES, generate_synthetic
from fraudlab.evaluate.metrics import evaluate


def make_ds(tmp_path: Path, name: str = "synthetic-v1") -> LoadedDataset:
    p = tmp_path / "d.parquet"
    generate_synthetic(6_000, seed=11).to_parquet(p)
    spec = {
        "name": name, "path": str(p), "label_column": "is_fraud", "id_column": "tx_id",
        "time_column": "event_time", "features": FEATURES, "license_note": "test",
    }  # fmt: skip
    return load_dataset(DatasetConfig.model_validate({"dataset": spec}))


@pytest.fixture
def run(tmp_path: Path) -> tuple[Path, dict]:  # type: ignore[type-arg]
    return run_baselines(make_ds(tmp_path), resamples=50, outputs=tmp_path / "outputs")


def test_writes_all_files(run: tuple[Path, dict]) -> None:  # type: ignore[type-arg]
    out, _ = run
    assert sorted(p.name for p in out.iterdir()) == [
        "labels.parquet", "manifest.json", "metrics.json", "predictions.parquet",
    ]  # fmt: skip


def test_manifest_records_versions_and_fingerprints(run: tuple[Path, dict]) -> None:  # type: ignore[type-arg]
    manifest = json.loads((run[0] / "manifest.json").read_text())
    assert set(manifest["versions"]) >= {"fraudlab", "eval", "rules", "xgb", "render", "xgboost"}
    assert set(manifest["split"]["fingerprints"]) == {"train", "validation", "holdout"}
    assert len(next(iter(manifest["dataset"]["files_sha256"].values()))) == 64


def test_saved_predictions_reproduce_saved_metrics(run: tuple[Path, dict]) -> None:  # type: ignore[type-arg]
    out, metrics = run
    preds = pd.read_parquet(out / "predictions.parquet")
    labels = pd.read_parquet(out / "labels.parquet")
    for model in ("xgboost", "rules"):
        p = preds[preds.model == model].set_index("case_id").loc[labels.case_id]
        again = evaluate(labels.label, p.score.to_numpy(), list(labels.case_id)).as_dict()
        assert again == metrics["models"][model]["metrics"]


def test_rules_not_run_is_recorded_not_faked(tmp_path: Path) -> None:
    out, metrics = run_baselines(make_ds(tmp_path, name="other"), resamples=20, outputs=tmp_path / "o")
    assert metrics["models"]["rules"]["status"] == "not_run"
    assert "metrics" not in metrics["models"]["rules"]
    assert set(pd.read_parquet(out / "predictions.parquet").model) == {"xgboost"}
    assert "not_run" in format_table(metrics)


def test_table_shows_every_model(run: tuple[Path, dict]) -> None:  # type: ignore[type-arg]
    table = format_table(run[1])
    assert "xgboost" in table and "rules" in table and "(reference)" in table
