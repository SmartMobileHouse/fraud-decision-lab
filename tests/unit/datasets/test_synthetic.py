"""The synthetic generator and the bundled demo file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraudlab.datasets.config import load_config
from fraudlab.datasets.demo import DEMO_PATH, content_hash
from fraudlab.datasets.leakage import check_leakage
from fraudlab.datasets.load import load_dataset
from fraudlab.datasets.split import split_dataset
from fraudlab.datasets.synthetic import FEATURES, generate_synthetic

REPO = Path(__file__).resolve().parents[3]
# Update only when GENERATOR_VERSION changes on purpose (and regenerate data/demo/synthetic.parquet).
EXPECTED_DEMO_HASH = "377506cd95f6b6ce"


def test_same_seed_same_data() -> None:
    assert content_hash(generate_synthetic(2_000, seed=7)) == content_hash(generate_synthetic(2_000, seed=7))


def test_different_seed_different_data() -> None:
    assert content_hash(generate_synthetic(2_000, seed=7)) != content_hash(generate_synthetic(2_000, seed=8))


def test_fraud_rate_is_about_three_percent() -> None:
    rate = generate_synthetic(20_000, seed=1)["is_fraud"].mean()
    assert 0.02 < rate < 0.04


def test_risky_signals_raise_fraud_rate() -> None:
    df = generate_synthetic(20_000, seed=1)
    known = df["device_known"].astype("boolean")
    unknown_rate = df.loc[known.eq(False).fillna(False), "is_fraud"].mean()
    known_rate = df.loc[known.eq(True).fillna(False), "is_fraud"].mean()
    assert unknown_rate > 2 * known_rate
    assert df.loc[df.failed_attempts_24h >= 3, "is_fraud"].mean() > 5 * df["is_fraud"].mean()


def test_labels_are_not_a_single_visible_rule() -> None:
    df = generate_synthetic(20_000, seed=1)
    # Some 'normal' rows are fraud and some fraud-scenario rows are legitimate.
    assert df.loc[df.scenario == "normal", "is_fraud"].sum() > 0
    assert (df.loc[df.scenario == "card_testing", "is_fraud"] == 0).sum() > 0


def test_has_realistic_missing_values() -> None:
    df = generate_synthetic(20_000, seed=1)
    for col in ("avg_amount_30d", "device_known", "country_mismatch"):
        assert 0.01 < df[col].isna().mean() < 0.15


def test_no_personal_data_columns() -> None:
    cols = set(generate_synthetic(100).columns)
    assert not cols & {"name", "email", "card_number", "pan", "iban", "address", "phone", "ip"}


def test_bundled_demo_file_matches_generator() -> None:
    assert DEMO_PATH.is_file(), "run `fraudlab data generate-demo`"
    assert content_hash(pd.read_parquet(DEMO_PATH)) == EXPECTED_DEMO_HASH


def test_synthetic_config_gives_10k_holdout_and_hides_scenario() -> None:
    ds = load_dataset(load_config(REPO / "configs/datasets/synthetic.yaml"))
    assert ds.features == FEATURES
    assert "scenario" not in ds.features
    split = split_dataset(ds)
    assert split.holdout.rows == 10_000
    check_leakage(ds, split)
