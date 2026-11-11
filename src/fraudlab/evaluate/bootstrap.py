"""Paired bootstrap: how sure are we about each number and each difference?

We redraw the holdout 1,000 times (with replacement), keeping the number of
fraud and legitimate cases fixed, and recompute every metric for every model
on the SAME redraw. The middle 95% of those values is the confidence range.
Because all models share each redraw, the range for "model minus XGBoost"
shows whether a difference is real or luck.

Caveat (stated in the report): redrawing single cases ignores links between
transactions of the same customer or day, and says nothing about future periods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fraudlab.evaluate.metrics import (
    TOP_K,
    MetricError,
    average_precision_sorted,
    prepare_scores,
    ranking,
    tie_break_keys,
)

METRICS = ("pr_auc", "recall_top_1pct", "recall_top_5pct", "precision_top_1pct")
DEFAULT_RESAMPLES = 1_000


@dataclass(frozen=True)
class Interval:
    low: float
    high: float


@dataclass(frozen=True)
class BootstrapResult:
    resamples: int
    seed: int
    reference: str | None
    intervals: dict[str, dict[str, Interval]]  # model -> metric -> 95% range
    differences: dict[str, dict[str, Interval]]  # model -> metric -> range of (model - reference)


def _resample_metrics(y: np.ndarray, s: np.ndarray, rank: np.ndarray, idx: np.ndarray) -> np.ndarray:
    order = idx[np.argsort(rank[idx], kind="stable")]
    y_sorted, n, frauds = y[order], len(idx), y[idx].sum()
    caught = {k: y_sorted[: math.ceil(k * n)].sum() for k in TOP_K}
    return np.array(
        [
            average_precision_sorted(y_sorted, s[order]),
            caught[0.01] / frauds,
            caught[0.05] / frauds,
            caught[0.01] / math.ceil(0.01 * n),
        ]
    )


def paired_bootstrap(
    labels: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    case_ids: list[str],
    reference: str | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 42,
) -> BootstrapResult:
    y = np.asarray(labels, dtype=np.int8)
    fraud_idx, legit_idx = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(fraud_idx) == 0:
        raise MetricError("no fraud cases")
    if reference is not None and reference not in scores_by_model:
        raise MetricError(f"reference model {reference!r} has no scores")

    keys = tie_break_keys(case_ids, seed)
    prepared = {}
    for name, raw in scores_by_model.items():
        s = prepare_scores(raw)
        rank = np.empty(len(y), dtype=np.int64)
        rank[ranking(s, keys)] = np.arange(len(y))  # position of each case in the full ranking
        prepared[name] = (s, rank)

    rng = np.random.default_rng(seed)
    values = {name: np.empty((resamples, len(METRICS))) for name in prepared}
    for b in range(resamples):
        idx = np.concatenate([rng.choice(fraud_idx, len(fraud_idx)), rng.choice(legit_idx, len(legit_idx))])
        for name, (s, rank) in prepared.items():
            values[name][b] = _resample_metrics(y, s, rank, idx)

    def ranges(arr: np.ndarray) -> dict[str, Interval]:
        lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
        return {m: Interval(float(lo[i]), float(hi[i])) for i, m in enumerate(METRICS)}

    intervals = {name: ranges(v) for name, v in values.items()}
    differences = {}
    if reference is not None:
        for name, v in values.items():
            if name != reference:
                differences[name] = ranges(v - values[reference])
    return BootstrapResult(resamples, seed, reference, intervals, differences)
