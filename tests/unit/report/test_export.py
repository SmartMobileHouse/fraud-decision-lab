"""Evidence bundle export: complete, public-safe, refuses restricted datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fraudlab.benchmark import runner as runner_mod
from fraudlab.benchmark.runner import RunOptions, run_benchmark
from fraudlab.datasets.config import DatasetConfig, load_config
from fraudlab.datasets.load import load_dataset
from fraudlab.datasets.synthetic import FEATURES, generate_synthetic
from fraudlab.report.export import RestrictedDatasetError, export_report_data
from fraudlab.report.render import SYNTHETIC_CONFIG


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str, str]:
    monkeypatch.setattr(runner_mod, "load_env", lambda: False)
    outputs = tmp_path / "o"
    synth = load_dataset(load_config(SYNTHETIC_CONFIG))
    s_out, _ = run_benchmark(synth, RunOptions(providers=["rules", "xgboost"], resamples=20, max_records=2_000),
                             outputs=outputs)  # fmt: skip
    p = tmp_path / "private.parquet"
    df = generate_synthetic(3_000, seed=9)
    df["tx_id"] = [f"SECRET-ROW-{i}" for i in range(len(df))]
    df.to_parquet(p)
    spec = {"name": "restricted-real", "path": str(p), "label_column": "is_fraud", "id_column": "tx_id",
            "time_column": "event_time", "features": FEATURES, "external_processing_allowed": False}  # fmt: skip
    r_out, _ = run_benchmark(load_dataset(DatasetConfig.model_validate({"dataset": spec})),
                             RunOptions(providers=["xgboost"], resamples=20), outputs=outputs)  # fmt: skip
    return outputs, s_out.name, r_out.name


def test_bundle_has_metrics_curves_examples(runs: tuple[Path, str, str], tmp_path: Path) -> None:
    outputs, s, _ = runs
    data = json.loads(export_report_data([s], tmp_path / "b.json", outputs=outputs).read_text())
    exp = data["experiments"][s]
    assert data["schema"] == "smh-research-data/1"
    assert exp["models"]["xgboost"]["metrics"]["pr_auc"] > 0
    assert len(exp["curves"]["xgboost"]["precision_recall"]["recall"]) == 101
    assert exp["top_1pct_cases"] == 20
    assert data["examples"]["cases"] and "seed" in data["examples"]["selection"]
    verification = exp["verification"]
    assert len(verification["labels"]) == 2_000
    assert len(verification["predictions"]) == 4_000
    assert all(set(row) == {"case_id", "provider", "fraud_score", "status", "payload_hash"}
               for row in verification["predictions"])  # fmt: skip
    assert exp["provenance"]["git"] is not None


def test_restricted_dataset_is_refused(runs: tuple[Path, str, str], tmp_path: Path) -> None:
    outputs, s, r = runs
    out = tmp_path / "b.json"
    with pytest.raises(RestrictedDatasetError, match="local only"):
        export_report_data([s, r], out, outputs=outputs)
    assert not out.exists()  # nothing written, not even the allowed part


def test_export_provenance_redacts_absolute_paths(tmp_path: Path) -> None:
    from fraudlab.report.export import _public_provenance

    original_path = str(tmp_path / "private-input.csv")
    config = {"dataset": {"path": original_path, "join": {"path": str(tmp_path / "private-identity.csv")}}}
    manifest = {"dataset_config": config, "identity": {"dataset_config": config,
                "input_files_sha256": {original_path: "hash"}}}  # fmt: skip
    public = _public_provenance(manifest)
    assert str(tmp_path) not in json.dumps(public)
    assert manifest["dataset_config"]["dataset"]["path"] == original_path  # original unchanged


def test_legacy_unknown_loadtest_is_not_assumed_public(runs: tuple[Path, str, str], tmp_path: Path) -> None:
    outputs, s, _ = runs
    load = outputs / "legacy-load"
    load.mkdir()
    (load / "loadtest.json").write_text(json.dumps({"steps": []}))
    out = tmp_path / "b.json"
    with pytest.raises(RestrictedDatasetError):
        export_report_data([s], out, loadtest_id=load.name, outputs=outputs)
    assert not out.exists()


def test_resume_keeps_original_start_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_mod, "load_env", lambda: False)
    ds = load_dataset(load_config(SYNTHETIC_CONFIG))
    out, _ = run_benchmark(ds, RunOptions(providers=["rules"], resamples=10, max_records=500), outputs=tmp_path)
    started = json.loads((out / "manifest.json").read_text())["started_utc"]
    run_benchmark(ds, RunOptions(providers=["rules"], resamples=10, max_records=500, resume_run_id=out.name),
                  outputs=tmp_path)  # fmt: skip
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["started_utc"] == started and len(manifest["sessions"]) == 2
