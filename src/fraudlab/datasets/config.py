"""Dataset config: a small YAML file that describes any labelled tabular dataset.

Example (configs/datasets/ulb.yaml):

    dataset:
      name: ulb-creditcard
      path: data/external/ulb/creditcard.csv
      label_column: Class
      time_column: Time
      features: all_except_label_id_time
      external_processing_allowed: true
    split: {train: 0.70, validation: 0.15, holdout: 0.15, seed: 42}
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from fraudlab.config import REPO_ROOT

ALL_FEATURES: Literal["all_except_label_id_time"] = "all_except_label_id_time"


class ConfigError(ValueError):
    """A dataset config is missing, unreadable or invalid. The message says how to fix it."""


class JoinConfig(BaseModel):
    """A second file merged into the main one (for example IEEE-CIS identity data)."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    key: str  # column present in both files (not "on": YAML reads `on` as true)
    how: Literal["left"] = "left"  # keep every main row, even without a match


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    path: Path
    label_column: str
    id_column: str | None = None  # never sent to models; row number is used when absent
    time_column: str | None = None  # set: split by time; unset: seeded random split
    features: list[str] | Literal["all_except_label_id_time"] = ALL_FEATURES
    feature_descriptions: dict[str, str] = Field(default_factory=dict)
    external_processing_allowed: bool = False  # safe default: local only
    publication_allowed: bool = False  # processing permission does not grant publication rights
    publication_note: str = ""
    license_note: str = ""
    join: JoinConfig | None = None

    @model_validator(mode="after")
    def _label_is_not_a_feature(self) -> DatasetSpec:
        if self.publication_allowed and not self.publication_note.strip():
            raise ValueError("publication_allowed requires a publication_note recording the rights basis")
        if isinstance(self.features, list):
            reserved = {c for c in (self.label_column, self.id_column, self.time_column) if c}
            clash = reserved & set(self.features)
            if clash:
                raise ValueError(f"features must not include the label, id or time column: {sorted(clash)}")
            if len(set(self.features)) != len(self.features):
                raise ValueError("features contains duplicates")
            if not self.features:
                raise ValueError("features must not be empty")
        return self


class SplitSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: float = Field(0.70, gt=0, lt=1)
    validation: float = Field(0.15, gt=0, lt=1)
    holdout: float = Field(0.15, gt=0, lt=1)
    seed: int = 42

    @model_validator(mode="after")
    def _sums_to_one(self) -> SplitSpec:
        total = self.train + self.validation + self.holdout
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"train + validation + holdout must equal 1.0 (got {total:.4f})")
        return self


class DatasetConfig(BaseModel):
    """The whole YAML file."""

    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSpec
    split: SplitSpec = Field(default_factory=SplitSpec)
    holdout_sample: int | None = Field(None, gt=0)  # None = score the whole holdout

    def resolve(self, path: Path) -> Path:
        """Paths in a config are relative to the repository root."""
        return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | Path) -> DatasetConfig:
    """Read and validate a dataset YAML file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Dataset config not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping with a 'dataset:' section")
    try:
        return DatasetConfig.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors())
        raise ConfigError(f"{path} is invalid — {problems}") from exc
