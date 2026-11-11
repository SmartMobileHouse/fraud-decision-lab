"""Train / validation / holdout splitting and leak checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudlab.datasets.config import ConfigError, DatasetConfig
from fraudlab.datasets.leakage import LeakageError, check_leakage
from fraudlab.datasets.load import load_dataset
from fraudlab.datasets.split import split_dataset


def make(
    tmp: Path, df: pd.DataFrame, time: bool = True, sample: int | None = None, features: object = None
) -> DatasetConfig:
    p = tmp / "d.parquet"
    df.to_parquet(p)
    spec: dict[str, object] = {"name": "t", "path": str(p), "label_column": "y", "id_column": "id"}
    if time:
        spec["time_column"] = "t"
    if features is not None:
        spec["features"] = features
    return DatasetConfig.model_validate({"dataset": spec, "holdout_sample": sample})


def frame(n: int = 1000, fraud_every: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "t": np.sort(rng.integers(0, n // 4, n)),  # many equal timestamps
            "amount": rng.random(n),
            "y": (np.arange(n) % fraud_every == 0).astype(int),
        }
    )


def test_parts_cover_every_row_once(tmp_path: Path) -> None:
    ds = load_dataset(make(tmp_path, frame()))
    s = split_dataset(ds)
    all_idx = np.concatenate([p.index for p in s.parts()])
    assert sorted(all_idx) == list(range(1000))


def test_time_split_trains_on_past_and_scores_on_future(tmp_path: Path) -> None:
    ds = load_dataset(make(tmp_path, frame()))
    s = split_dataset(ds)
    t = ds.frame["t"].to_numpy()
    assert t[s.train.index].max() < t[s.validation.index].min()
    assert t[s.validation.index].max() < t[s.holdout.index].min()


def test_equal_timestamps_never_split(tmp_path: Path) -> None:
    df = frame()
    ds = load_dataset(make(tmp_path, df))
    s = split_dataset(ds)
    part_of = {}
    for p in s.parts():
        for i in p.index:
            part_of[i] = p.name
    for _, group in df.groupby("t"):
        assert len({part_of[i] for i in group.index}) == 1


def test_proportions_are_close_to_config(tmp_path: Path) -> None:
    s = split_dataset(load_dataset(make(tmp_path, frame(n=10_000))))
    assert abs(s.train.rows / 10_000 - 0.70) < 0.01
    assert abs(s.holdout.rows / 10_000 - 0.15) < 0.01


def test_split_is_deterministic(tmp_path: Path) -> None:
    cfg = make(tmp_path, frame(), time=False)
    assert split_dataset(load_dataset(cfg)).hashes == split_dataset(load_dataset(cfg)).hashes


def test_random_split_keeps_fraud_rate(tmp_path: Path) -> None:
    s = split_dataset(load_dataset(make(tmp_path, frame(n=10_000), time=False)))
    assert s.method == "random"
    for p in s.parts():
        assert abs(p.fraud_rate - 0.10) < 0.005


def test_part_without_fraud_is_rejected(tmp_path: Path) -> None:
    df = frame()
    df["y"] = 0
    df.loc[:50, "y"] = 1  # fraud only in the oldest rows -> holdout has none
    with pytest.raises(ConfigError, match="no fraud"):
        split_dataset(load_dataset(make(tmp_path, df)))


def test_holdout_sample_is_seeded_and_from_holdout(tmp_path: Path) -> None:
    cfg = make(tmp_path, frame(n=10_000), sample=500)
    s1, s2 = split_dataset(load_dataset(cfg)), split_dataset(load_dataset(cfg))
    assert s1.holdout.rows == 500 and s1.holdout_full_rows > 500
    assert s1.hashes["holdout"] == s2.hashes["holdout"]
    assert set(s1.holdout.index).isdisjoint(s1.train.index)


def test_feature_equal_to_label_is_caught(tmp_path: Path) -> None:
    df = frame()
    df["leak"] = df["y"] * 3.0
    ds = load_dataset(make(tmp_path, df))
    with pytest.raises(LeakageError, match="leak"):
        check_leakage(ds, split_dataset(ds))


def test_text_feature_that_spells_the_label_is_caught(tmp_path: Path) -> None:
    df = frame()
    df["status"] = np.where(df["y"] == 1, "chargeback", "settled")
    ds = load_dataset(make(tmp_path, df))
    with pytest.raises(LeakageError, match="status"):
        check_leakage(ds, split_dataset(ds))


def test_normal_features_pass(tmp_path: Path) -> None:
    ds = load_dataset(make(tmp_path, frame()))
    check_leakage(ds, split_dataset(ds))  # no error
