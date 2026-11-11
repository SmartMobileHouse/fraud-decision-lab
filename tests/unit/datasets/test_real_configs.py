"""Shipped configs for the real datasets. Data-dependent checks skip when files are absent."""

from __future__ import annotations

from pathlib import Path

import pytest

from fraudlab.datasets.config import load_config
from fraudlab.datasets.leakage import check_leakage
from fraudlab.datasets.load import load_dataset
from fraudlab.datasets.split import split_dataset

REPO = Path(__file__).resolve().parents[3]
CONFIGS = REPO / "configs" / "datasets"


@pytest.mark.parametrize("name", ["ieee.yaml", "ieee-full.yaml"])
def test_ieee_configs_are_local_only(name: str) -> None:
    cfg = load_config(CONFIGS / name)
    assert cfg.dataset.external_processing_allowed is False
    assert cfg.dataset.join is not None and cfg.dataset.join.key == "TransactionID"
    assert cfg.holdout_sample == 10_000


def test_every_shipped_config_is_valid() -> None:
    names = sorted(p.name for p in CONFIGS.glob("*.yaml"))
    assert names == ["ieee-full.yaml", "ieee.yaml", "synthetic.yaml", "ulb.yaml"]
    for p in CONFIGS.glob("*.yaml"):
        load_config(p)


def _require(cfg_name: str) -> Path:
    cfg = load_config(CONFIGS / cfg_name)
    path = cfg.resolve(cfg.dataset.path)
    if not path.is_file():
        pytest.skip(f"{path.name} not downloaded")
    return CONFIGS / cfg_name


def test_ieee_loads_joins_and_splits() -> None:
    ds = load_dataset(load_config(_require("ieee.yaml")))
    assert len(ds.frame) == 590_540
    assert len(ds.features) == 11
    assert set(ds.frame.columns) == set(ds.features) | {"TransactionID", "isFraud", "TransactionDT"}
    assert ds.frame["DeviceType"].isna().mean() > 0.7  # most transactions have no identity row
    split = split_dataset(ds)
    assert split.holdout.rows == 10_000
    check_leakage(ds, split)


def test_ulb_loads_and_splits() -> None:
    ds = load_dataset(load_config(_require("ulb.yaml")))
    assert len(ds.frame) == 284_807
    split = split_dataset(ds)
    check_leakage(ds, split)
