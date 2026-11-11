"""XGBoost baseline: trained only on the past, deterministic, safe encoders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraudlab.baselines.xgb import FeatureEncoder, train_and_score
from fraudlab.datasets.config import DatasetConfig
from fraudlab.datasets.load import LoadedDataset, load_dataset
from fraudlab.datasets.split import split_dataset
from fraudlab.datasets.synthetic import FEATURES, generate_synthetic


def make_ds(tmp_path: Path, df: pd.DataFrame) -> LoadedDataset:
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    spec = {
        "name": "t", "path": str(p), "label_column": "is_fraud", "id_column": "tx_id",
        "time_column": "event_time", "features": FEATURES,
    }  # fmt: skip
    return load_dataset(DatasetConfig.model_validate({"dataset": spec}))


def test_scores_are_valid_and_beat_random(tmp_path: Path) -> None:
    ds = make_ds(tmp_path, generate_synthetic(8_000, seed=5))
    split = split_dataset(ds)
    r = train_and_score(ds, split)
    assert len(r.scores) == split.holdout.rows
    assert r.scores.min() >= 0 and r.scores.max() <= 1
    y = ds.frame["is_fraud"].to_numpy()[split.holdout.index]
    assert r.scores[y == 1].mean() > 2 * r.scores[y == 0].mean()


def test_training_is_deterministic(tmp_path: Path) -> None:
    ds = make_ds(tmp_path, generate_synthetic(5_000, seed=5))
    split = split_dataset(ds)
    assert np.array_equal(train_and_score(ds, split).scores, train_and_score(ds, split).scores)


def test_holdout_cannot_influence_the_model(tmp_path: Path) -> None:
    """Scramble every holdout row: the trained model must not change at all."""
    df = generate_synthetic(5_000, seed=5)
    ds = make_ds(tmp_path, df)
    split = split_dataset(ds)
    original = train_and_score(ds, split)

    scrambled = ds.frame.copy()
    hold = split.holdout.index
    rng = np.random.default_rng(0)
    scrambled.loc[scrambled.index[hold], "amount"] = rng.random(len(hold)) * 9_999
    scrambled.loc[scrambled.index[hold], "merchant_category"] = "never_seen_before"
    scrambled.loc[scrambled.index[hold], "is_fraud"] = 1 - scrambled["is_fraud"].iloc[hold]
    changed = LoadedDataset(config=ds.config, frame=scrambled, features=ds.features)
    retrained = train_and_score(changed, split)

    assert original.model.get_booster().save_raw("json") == retrained.model.get_booster().save_raw("json")
    assert original.encoder.categories == retrained.encoder.categories


def test_unseen_category_becomes_missing() -> None:
    train = pd.DataFrame({"cat": ["a", "b", "a"], "x": [1.0, 2.0, None]})
    enc = FeatureEncoder.fit(train, ["cat", "x"])
    out = enc.transform(pd.DataFrame({"cat": ["b", "zzz"], "x": [3.0, None]}))
    assert out["cat"].tolist()[0] == "b"
    assert pd.isna(out["cat"].tolist()[1])
    assert np.isnan(out["x"].tolist()[1])


def test_booleans_become_numbers_with_missing() -> None:
    train = pd.DataFrame({"flag": pd.array([True, False, pd.NA], dtype="boolean")})
    out = FeatureEncoder.fit(train, ["flag"]).transform(train)
    assert out["flag"].tolist()[:2] == [1.0, 0.0] and np.isnan(out["flag"].tolist()[2])
