"""The model input: identical for every model and free of the answer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from fraudlab.cli import app
from fraudlab.datasets.config import DatasetConfig, load_config
from fraudlab.datasets.load import load_dataset
from fraudlab.datasets.payload import case_id, load_question, payload_hash, render_case

REPO = Path(__file__).resolve().parents[3]
runner = CliRunner()


@pytest.fixture
def ds(tmp_path: Path):  # type: ignore[no-untyped-def]
    df = pd.DataFrame(
        {
            "tx_id": ["TX-UNIQUE-1", "TX-UNIQUE-2"],
            "ts": [1, 2],
            "amount": [12.345678, np.nan],
            "known": pd.array([True, pd.NA], dtype="boolean"),
            "cat": ["grocery", None],
            "count": np.array([3, 4], dtype="int16"),
            "y": [0, 1],
        }
    )
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    spec = {"name": "t", "path": str(p), "label_column": "y", "id_column": "tx_id", "time_column": "ts"}
    return load_dataset(DatasetConfig.model_validate({"dataset": spec}))


def test_payload_has_only_features(ds) -> None:  # type: ignore[no-untyped-def]
    fields = render_case(ds, 0)
    assert list(fields) == ["amount", "known", "cat", "count"]
    assert not {"y", "tx_id", "ts"} & set(fields)


def test_values_are_plain_json(ds) -> None:  # type: ignore[no-untyped-def]
    assert render_case(ds, 0) == {"amount": 12.3457, "known": True, "cat": "grocery", "count": 3}
    assert render_case(ds, 1) == {"amount": None, "known": None, "cat": None, "count": 4}
    json.dumps(render_case(ds, 1), allow_nan=False)  # must not raise


def test_hash_is_stable_and_sensitive(ds) -> None:  # type: ignore[no-untyped-def]
    a, b = render_case(ds, 0), render_case(ds, 0)
    assert payload_hash(a) == payload_hash(b)
    assert payload_hash(a) != payload_hash(render_case(ds, 1))


def test_case_id_is_reported_but_not_in_payload(ds) -> None:  # type: ignore[no-untyped-def]
    assert case_id(ds, 0) == "TX-UNIQUE-1"
    assert "TX-UNIQUE-1" not in json.dumps(render_case(ds, 0))


def test_question_file() -> None:
    q = load_question()
    assert q.version == "v1" and q.id == "fraud_suspected"


def test_synthetic_payload_matches_feature_list() -> None:
    ds = load_dataset(load_config(REPO / "configs/datasets/synthetic.yaml"))
    fields = render_case(ds, 0)
    assert list(fields) == ds.features
    assert "scenario" not in fields and "is_fraud" not in fields


def test_cli_inspect_and_show_case() -> None:
    r = runner.invoke(app, ["data", "inspect", "-d", str(REPO / "configs/datasets/synthetic.yaml")])
    assert r.exit_code == 0, r.output
    assert "holdout       10,000 rows" in r.output and "ALLOWED" in r.output
    r = runner.invoke(app, ["data", "show-case", "-d", str(REPO / "configs/datasets/synthetic.yaml")])
    assert r.exit_code == 0, r.output
    assert "payload fingerprint" in r.output and '"is_fraud"' not in r.output


def test_cli_reports_bad_config_cleanly(tmp_path: Path) -> None:
    r = runner.invoke(app, ["data", "inspect", "-d", str(tmp_path / "missing.yaml")])
    assert r.exit_code == 1
    assert "not found" in r.output


def test_generate_demo_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "demo.parquet"
    r = runner.invoke(app, ["data", "generate-demo", "--rows", "500", "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert len(pd.read_parquet(out)) == 500
