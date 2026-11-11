"""Dataset config validation and loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fraudlab.datasets.config import ConfigError, load_config
from fraudlab.datasets.load import load_dataset

REPO = Path(__file__).resolve().parents[3]


def write(tmp: Path, name: str, text: str) -> Path:
    p = tmp / name
    p.write_text(text)
    return p


def config_yaml(data: Path, extra: str = "", features: str = "all_except_label_id_time") -> str:
    return f"""
dataset:
  name: t
  path: {data}
  label_column: is_fraud
  id_column: tx_id
  time_column: ts
  features: {features}
{extra}
"""


@pytest.fixture
def csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "tx_id": [1, 2, 3, 4],
            "ts": [10, 20, 30, 40],
            "amount": [5.0, 9.5, 120.0, 3.2],
            "is_fraud": [0, 0, 1, 0],
        }
    )
    p = tmp_path / "tx.csv"
    df.to_csv(p, index=False)
    return p


def test_shipped_ulb_config_is_valid() -> None:
    cfg = load_config(REPO / "configs/datasets/ulb.yaml")
    assert cfg.dataset.label_column == "Class"
    assert cfg.dataset.external_processing_allowed is True


def test_loads_and_excludes_label_id_time_from_features(tmp_path: Path, csv: Path) -> None:
    ds = load_dataset(load_config(write(tmp_path, "c.yaml", config_yaml(csv))))
    assert ds.features == ["amount"]
    assert len(ds.frame) == 4
    assert ds.frame["is_fraud"].dtype == "int8"


def test_external_processing_is_off_unless_set(tmp_path: Path, csv: Path) -> None:
    cfg = load_config(write(tmp_path, "c.yaml", config_yaml(csv)))
    assert cfg.dataset.external_processing_allowed is False


def test_label_listed_as_feature_is_rejected(tmp_path: Path, csv: Path) -> None:
    with pytest.raises(ConfigError, match="must not include the label"):
        load_config(write(tmp_path, "c.yaml", config_yaml(csv, features="[amount, is_fraud]")))


def test_split_must_sum_to_one(tmp_path: Path, csv: Path) -> None:
    bad = "split: {train: 0.7, validation: 0.2, holdout: 0.2}"
    with pytest.raises(ConfigError, match="must equal 1.0"):
        load_config(write(tmp_path, "c.yaml", config_yaml(csv, extra=bad)))


def test_unknown_keys_are_rejected(tmp_path: Path, csv: Path) -> None:
    with pytest.raises(ConfigError, match="Extra inputs"):
        load_config(write(tmp_path, "c.yaml", config_yaml(csv, extra="typo_key: 1")))


def test_missing_data_file_explains_where_to_put_it(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, "c.yaml", config_yaml(tmp_path / "nope.csv")))
    with pytest.raises(ConfigError, match="data/external"):
        load_dataset(cfg)


def test_non_binary_label_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    bad = {"tx_id": [1, 2], "ts": [1, 2], "amount": [1.0, 2.0], "is_fraud": [0, 2]}
    pd.DataFrame(bad).to_csv(p, index=False)
    with pytest.raises(ConfigError, match="only 0"):
        load_dataset(load_config(write(tmp_path, "c.yaml", config_yaml(p))))


def test_missing_feature_column_is_reported(tmp_path: Path, csv: Path) -> None:
    cfg = load_config(write(tmp_path, "c.yaml", config_yaml(csv, features="[amount, not_there]")))
    with pytest.raises(ConfigError, match="not_there"):
        load_dataset(cfg)


def test_left_join_keeps_rows_without_a_match(tmp_path: Path, csv: Path) -> None:
    ident = tmp_path / "ident.csv"
    pd.DataFrame({"tx_id": [1, 3], "device": ["ios", "android"]}).to_csv(ident, index=False)
    extra = f"  join: {{path: {ident}, key: tx_id}}"
    ds = load_dataset(load_config(write(tmp_path, "c.yaml", config_yaml(csv, extra=extra))))
    assert len(ds.frame) == 4  # rows 2 and 4 have no identity row but are kept
    assert ds.frame["device"].isna().sum() == 2
    assert "device" in ds.features


def test_duplicate_join_keys_are_rejected(tmp_path: Path, csv: Path) -> None:
    ident = tmp_path / "ident.csv"
    pd.DataFrame({"tx_id": [1, 1], "device": ["a", "b"]}).to_csv(ident, index=False)
    extra = f"  join: {{path: {ident}, key: tx_id}}"
    with pytest.raises(ConfigError, match="duplicates"):
        load_dataset(load_config(write(tmp_path, "c.yaml", config_yaml(csv, extra=extra))))
