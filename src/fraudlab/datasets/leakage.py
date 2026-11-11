"""Leak checks: make sure no model input secretly contains the answer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudlab.datasets.config import ConfigError
from fraudlab.datasets.load import LoadedDataset
from fraudlab.datasets.split import DatasetSplit

# A feature this strongly tied to the label on training data is almost surely the answer in disguise.
MAX_ABS_CORRELATION = 0.99


class LeakageError(ConfigError):
    """A feature would reveal the fraud label to the models."""


def check_leakage(ds: LoadedDataset, split: DatasetSplit) -> None:
    spec = ds.config.dataset
    reserved = {c for c in (spec.label_column, spec.id_column, spec.time_column) if c}
    bad = reserved & set(ds.features)
    if bad:
        raise LeakageError(f"Reserved columns used as features: {sorted(bad)}")

    train = ds.frame.iloc[split.train.index]
    label = train[spec.label_column].astype(float)
    for col in ds.features:
        values = train[col]
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            values = values.astype("category").cat.codes.where(values.notna())
        values = values.astype(float)
        mask = values.notna()
        if mask.sum() < 2 or values[mask].nunique() < 2:
            continue
        corr = np.corrcoef(values[mask], label[mask])[0, 1]
        if np.isfinite(corr) and abs(corr) > MAX_ABS_CORRELATION:
            raise LeakageError(
                f"Feature {col!r} matches the label almost perfectly on train (correlation {corr:+.3f}). "
                "It probably contains the answer; remove it from features."
            )
