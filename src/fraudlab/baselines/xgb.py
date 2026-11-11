"""XGBoost baseline: supervised, trained on train, stopped on validation, scored once on holdout.

- Settings are fixed below (XGB_VERSION); they are never tuned on the holdout.
- Encoders are fitted on train only. A category never seen in train becomes
  "missing" in validation/holdout, so no information flows back from the future.
- XGBoost handles missing values and categories natively.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from fraudlab.datasets.load import LoadedDataset
from fraudlab.datasets.split import DatasetSplit

XGB_VERSION = "1"
XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 2_000,  # upper limit; early stopping picks the real number
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "enable_categorical": True,
    "eval_metric": "aucpr",
    "early_stopping_rounds": 100,
}


@dataclass
class FeatureEncoder:
    """Turns raw columns into XGBoost-ready columns. Fitted on train only."""

    features: list[str]
    categories: dict[str, list[Any]] = field(default_factory=dict)

    @classmethod
    def fit(cls, train: pd.DataFrame, features: list[str]) -> FeatureEncoder:
        enc = cls(features=features)
        for col in features:
            series = train[col]
            if not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)):
                enc.categories[col] = sorted(series.dropna().astype(str).unique())
        return enc

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in self.features:
            series = frame[col]
            if col in self.categories:
                known = self.categories[col]
                text = series.astype("string")
                text = text.where(text.isin(known))  # unseen in train -> missing
                out[col] = text.astype(pd.CategoricalDtype(known))
            else:
                out[col] = pd.to_numeric(series.astype("Float64"), errors="coerce").astype(float)
        return pd.DataFrame(out, index=frame.index)


@dataclass(frozen=True)
class XGBResult:
    scores: np.ndarray  # holdout fraud scores in [0, 1], in split.holdout.index order
    best_iteration: int
    train_seconds: float
    params: dict[str, Any]
    encoder: FeatureEncoder
    model: xgb.XGBClassifier


def train_and_score(ds: LoadedDataset, split: DatasetSplit, seed: int | None = None) -> XGBResult:
    label = ds.config.dataset.label_column
    seed = ds.config.split.seed if seed is None else seed
    frame = ds.frame
    train, val, hold = (frame.iloc[p.index] for p in split.parts())

    encoder = FeatureEncoder.fit(train, ds.features)
    x_train, x_val, x_hold = (encoder.transform(t) for t in (train, val, hold))

    params = {**XGB_PARAMS, "random_state": seed}
    model = xgb.XGBClassifier(**params)
    started = time.perf_counter()
    model.fit(x_train, train[label], eval_set=[(x_val, val[label])], verbose=False)
    seconds = time.perf_counter() - started

    scores = model.predict_proba(x_hold)[:, 1].astype(float)
    return XGBResult(
        scores=np.clip(scores, 0.0, 1.0),
        best_iteration=int(model.best_iteration),
        train_seconds=seconds,
        params=params,
        encoder=encoder,
        model=model,
    )
