"""Rules baseline: transparent points, bounded score, honest not_run."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from fraudlab.baselines import rules
from fraudlab.baselines.rules import SYNTHETIC_RULES, explain, score_rules
from fraudlab.datasets.config import DatasetConfig
from fraudlab.datasets.load import load_dataset
from fraudlab.datasets.synthetic import generate_synthetic

MAX_POINTS = sum(r.points for r in SYNTHETIC_RULES)


def synthetic_ds(tmp_path: Path, df: pd.DataFrame, name: str = "synthetic-v1"):  # type: ignore[no-untyped-def]
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    spec = {"name": name, "path": str(p), "label_column": "is_fraud", "id_column": "tx_id"}
    spec["features"] = [c for c in df.columns if c not in ("is_fraud", "tx_id", "scenario", "event_time")]
    return load_dataset(DatasetConfig.model_validate({"dataset": spec}))


def one_row(**overrides: object) -> pd.DataFrame:
    base = generate_synthetic(1, seed=0).copy()
    quiet = {
        "failed_attempts_24h": 0, "tx_count_1h": 1, "device_known": True, "country_mismatch": False,
        "amount_to_avg_ratio": 1.0, "avg_amount_30d": 40.0, "card_age_days": 400, "hour_of_day": 14,
        "merchant_category": "grocery",
    }  # fmt: skip
    for k, v in {**quiet, **overrides}.items():
        base[k] = pd.Series([v], dtype=base[k].dtype)
    return base


def test_quiet_transaction_scores_zero(tmp_path: Path) -> None:
    ds = synthetic_ds(tmp_path, one_row())
    assert score_rules(ds, np.array([0])).scores.tolist() == [0.0]  # type: ignore[union-attr]
    assert explain(ds, 0) == []


def test_points_add_up(tmp_path: Path) -> None:
    ds = synthetic_ds(tmp_path, one_row(failed_attempts_24h=4, country_mismatch=True))
    score = score_rules(ds, np.array([0])).scores[0]  # type: ignore[index]
    assert score == (3 + 2) / MAX_POINTS
    assert set(explain(ds, 0)) == {"3+ failed attempts in 24h", "country differs from home country"}


def test_unknown_values_do_not_crash_or_count_as_bad(tmp_path: Path) -> None:
    ds = synthetic_ds(tmp_path, one_row(device_known=None, country_mismatch=None, amount_to_avg_ratio=np.nan))
    assert explain(ds, 0) == ["device status missing"]


def test_scores_are_between_0_and_1(tmp_path: Path) -> None:
    df = generate_synthetic(5_000, seed=3)
    s = score_rules(synthetic_ds(tmp_path, df), np.arange(5_000)).scores
    assert s is not None and s.min() >= 0 and s.max() <= 1


def test_unknown_dataset_is_not_run(tmp_path: Path) -> None:
    ds = synthetic_ds(tmp_path, one_row(), name="ulb-creditcard")
    r = score_rules(ds, np.array([0]))
    assert r.status == "not_run" and r.scores is None


def test_rules_do_not_use_the_generator() -> None:
    """The checklist must not import or copy the synthetic generator's formula."""
    tree = ast.parse(Path(rules.__file__).read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "fraudlab.datasets.synthetic" not in imported
