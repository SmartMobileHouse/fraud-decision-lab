"""Split a dataset into train / validation / holdout.

- With a time column: oldest rows go to train, newest to holdout. Rows with the
  same timestamp always stay in the same part.
- Without a time column: a seeded random split that keeps the fraud rate equal
  in every part.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraudlab.datasets.config import ConfigError
from fraudlab.datasets.load import LoadedDataset

PARTS = ("train", "validation", "holdout")


@dataclass(frozen=True)
class SplitPart:
    name: str
    index: np.ndarray  # row positions in LoadedDataset.frame
    rows: int
    frauds: int

    @property
    def fraud_rate(self) -> float:
        return self.frauds / self.rows


@dataclass(frozen=True)
class DatasetSplit:
    method: str  # "time" or "random"
    train: SplitPart
    validation: SplitPart
    holdout: SplitPart  # what gets scored (after optional sampling)
    holdout_full_rows: int  # holdout size before sampling
    hashes: dict[str, str]  # fingerprint of each part, recorded in the run manifest

    def parts(self) -> tuple[SplitPart, SplitPart, SplitPart]:
        return (self.train, self.validation, self.holdout)

    def summary(self) -> str:
        lines = [f"split method: {self.method}"]
        for p in self.parts():
            lines.append(f"  {p.name:<10} {p.rows:>9,} rows  {p.frauds:>7,} fraud  ({p.fraud_rate:.3%})")
        if self.holdout.rows != self.holdout_full_rows:
            lines.append(f"  (holdout sampled from {self.holdout_full_rows:,} rows)")
        return "\n".join(lines)


def _row_keys(ds: LoadedDataset) -> np.ndarray:
    """Stable identifier for each row: the ID column, or the row number."""
    id_col = ds.config.dataset.id_column
    return ds.frame[id_col].to_numpy() if id_col else np.arange(len(ds.frame))


def _time_cuts(times: np.ndarray, fractions: tuple[float, float]) -> tuple[int, int]:
    """Cut positions in time-sorted rows, moved forward so equal timestamps are never split."""
    n = len(times)
    cuts = []
    for frac in fractions:
        cut = int(round(frac * n))
        if 0 < cut < n:
            cut = int(np.searchsorted(times, times[cut - 1], side="right"))
        cuts.append(cut)
    return cuts[0], max(cuts[0], cuts[1])


def _fingerprint(keys: np.ndarray) -> str:
    data = ",".join(map(str, np.sort(keys))).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def split_dataset(ds: LoadedDataset) -> DatasetSplit:
    spec, split_cfg = ds.config.dataset, ds.config.split
    labels = ds.frame[spec.label_column].to_numpy()
    keys = _row_keys(ds)
    n = len(ds.frame)

    if spec.time_column:
        order = np.lexsort((keys, ds.frame[spec.time_column].to_numpy()))  # by time, then key
        times = ds.frame[spec.time_column].to_numpy()[order]
        a, b = _time_cuts(times, (split_cfg.train, split_cfg.train + split_cfg.validation))
        idx = {"train": order[:a], "validation": order[a:b], "holdout": order[b:]}
        method = "time"
    else:
        all_idx = np.arange(n)
        train_idx, rest = train_test_split(
            all_idx, train_size=split_cfg.train, stratify=labels, random_state=split_cfg.seed
        )
        val_share = split_cfg.validation / (split_cfg.validation + split_cfg.holdout)
        val_idx, hold_idx = train_test_split(
            rest, train_size=val_share, stratify=labels[rest], random_state=split_cfg.seed
        )
        idx = {"train": np.sort(train_idx), "validation": np.sort(val_idx), "holdout": np.sort(hold_idx)}
        method = "random"

    holdout_full_rows = len(idx["holdout"])
    sample = ds.config.holdout_sample
    if sample is not None and sample < holdout_full_rows:
        rng = np.random.default_rng(split_cfg.seed)
        idx["holdout"] = np.sort(rng.choice(idx["holdout"], size=sample, replace=False))

    parts = {}
    for name in PARTS:
        part_idx = idx[name]
        frauds = int(labels[part_idx].sum())
        if len(part_idx) == 0:
            raise ConfigError(f"The {name} part is empty; the dataset is too small for this split")
        if frauds == 0 or frauds == len(part_idx):
            raise ConfigError(
                f"The {name} part has {'no fraud' if frauds == 0 else 'no legitimate'} rows; "
                "it cannot be used. Use a larger dataset or a different split."
            )
        parts[name] = SplitPart(name=name, index=part_idx, rows=len(part_idx), frauds=frauds)

    # Safety: the three parts never share a row.
    seen = np.concatenate([parts[p].index for p in PARTS])
    if len(np.unique(seen)) != len(seen):
        raise AssertionError("split parts overlap")  # a bug, not a user error

    return DatasetSplit(
        method=method,
        train=parts["train"],
        validation=parts["validation"],
        holdout=parts["holdout"],
        holdout_full_rows=holdout_full_rows,
        hashes={p: _fingerprint(keys[parts[p].index]) for p in PARTS},
    )


def part_frame(ds: LoadedDataset, part: SplitPart) -> pd.DataFrame:
    """The rows of one part, as a table."""
    return ds.frame.iloc[part.index]
