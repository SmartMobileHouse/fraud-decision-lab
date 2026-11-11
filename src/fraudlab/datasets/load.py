"""Load a dataset described by a config into one table, and check it is usable."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from fraudlab.datasets.config import ALL_FEATURES, ConfigError, DatasetConfig


@dataclass(frozen=True)
class LoadedDataset:
    config: DatasetConfig
    frame: pd.DataFrame
    features: list[str]  # the exact columns every model may see


def _columns_in(path: Path) -> list[str]:
    """Column names from the file header, without reading the data."""
    if path.suffix.lower() == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    return list(pq.read_schema(path).names)


def _read(path: Path, wanted: set[str] | None = None) -> pd.DataFrame:
    """Read a CSV or Parquet file. With `wanted`, read only those columns (saves memory)."""
    if not path.is_file():
        raise ConfigError(f"Data file not found: {path}. Download it into data/external/ (see DATA_LICENSES.md).")
    suffix = path.suffix.lower()
    if suffix not in (".csv", ".parquet"):
        raise ConfigError(f"Unsupported file type {suffix!r} for {path}; use .csv or .parquet")
    columns = None
    if wanted is not None:
        columns = [c for c in _columns_in(path) if c in wanted]  # keep file order
    if suffix == ".csv":
        return pd.read_csv(path, engine="pyarrow", usecols=columns)
    return pd.read_parquet(path, columns=columns)


def load_dataset(config: DatasetConfig) -> LoadedDataset:
    spec = config.dataset
    wanted: set[str] | None = None  # None = read every column
    if spec.features != ALL_FEATURES:
        wanted = set(spec.features) | {c for c in (spec.label_column, spec.id_column, spec.time_column) if c}
        if spec.join is not None:
            wanted.add(spec.join.key)
    frame = _read(config.resolve(spec.path), wanted)

    if spec.join is not None:
        other = _read(config.resolve(spec.join.path), wanted)
        for side, table in (("main file", frame), ("join file", other)):
            if spec.join.key not in table.columns:
                raise ConfigError(f"Join column {spec.join.key!r} is missing from the {side}")
        if other[spec.join.key].duplicated().any():
            raise ConfigError(f"Join column {spec.join.key!r} has duplicates in the join file")
        rows_before = len(frame)
        frame = frame.merge(other, on=spec.join.key, how=spec.join.how, validate="many_to_one")
        if len(frame) != rows_before:  # a left join must never add or drop rows
            raise ConfigError("Join changed the number of rows")

    required = [spec.label_column] + [c for c in (spec.id_column, spec.time_column) if c]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ConfigError(f"Columns not found in the data: {missing}")

    labels = frame[spec.label_column]
    if labels.isna().any():
        raise ConfigError(f"Label column {spec.label_column!r} has missing values")
    if not set(labels.unique()) <= {0, 1}:
        raise ConfigError(f"Label column {spec.label_column!r} must contain only 0 (legit) and 1 (fraud)")
    frame[spec.label_column] = labels.astype("int8")

    if spec.id_column:
        if frame[spec.id_column].isna().any():
            raise ConfigError(f"ID column {spec.id_column!r} has missing values")
        if frame[spec.id_column].duplicated().any():
            raise ConfigError(f"ID column {spec.id_column!r} has duplicate values")
    if spec.time_column and frame[spec.time_column].isna().any():
        raise ConfigError(f"Time column {spec.time_column!r} has missing values")

    if spec.features == ALL_FEATURES:
        reserved = {spec.label_column, spec.id_column, spec.time_column} - {None}
        features = [c for c in frame.columns if c not in reserved]
    else:
        features = list(spec.features)
        absent = [c for c in features if c not in frame.columns]
        if absent:
            raise ConfigError(f"Feature columns not found in the data: {absent}")

    return LoadedDataset(config=config, frame=frame, features=features)
