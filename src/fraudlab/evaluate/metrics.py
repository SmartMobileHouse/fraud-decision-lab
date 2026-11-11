"""Ranking metrics (EVAL_VERSION 1).

Conventions, fixed so every model is scored the same way:
- A model gives each case a score in [0, 1]; higher = more suspicious.
- A case with no score (failed call, timeout, refusal) is ranked LAST, so
  failures count against the model. Coverage reports how many were scored.
- Top-K: K = ceil(k * N). Ties are broken by a fixed hash of (seed, case_id),
  never by row order.
- PR-AUC is average precision with tied scores grouped, identical to
  sklearn.metrics.average_precision_score.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

EVAL_VERSION = "1"
TOP_K = (0.01, 0.05)
FAILED_SCORE = -1.0  # below every valid score, so failures rank last


class MetricError(ValueError):
    """Inputs cannot be scored (for example: no fraud cases)."""


@dataclass(frozen=True)
class Metrics:
    n: int
    frauds: int
    coverage: float
    pr_auc: float
    recall_top_1pct: float
    recall_top_5pct: float
    precision_top_1pct: float
    frauds_caught_top_1pct: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def tie_break_keys(case_ids: Sequence[str], seed: int) -> np.ndarray:
    """A fixed pseudo-random number per case, used only to order equal scores."""
    return np.array(
        [int.from_bytes(hashlib.sha256(f"{seed}:{c}".encode()).digest()[:8], "big") for c in case_ids],
        dtype=np.uint64,
    )


def prepare_scores(scores: Sequence[float | None] | np.ndarray) -> np.ndarray:
    """Validate scores and replace missing ones with FAILED_SCORE."""
    s = np.array([np.nan if v is None else v for v in scores], dtype=float)
    valid = ~np.isnan(s)
    if ((s[valid] < 0) | (s[valid] > 1)).any():
        raise MetricError("scores must be between 0 and 1")
    return np.where(valid, s, FAILED_SCORE)


def ranking(scores: np.ndarray, tie_keys: np.ndarray) -> np.ndarray:
    """Case positions from most to least suspicious."""
    return np.lexsort((tie_keys, -scores))


def average_precision_sorted(labels_sorted: np.ndarray, scores_sorted: np.ndarray) -> float:
    """Average precision for cases already sorted by score (descending); tied scores grouped."""
    positives = labels_sorted.sum()
    if positives == 0:
        raise MetricError("no fraud cases")
    tp = np.cumsum(labels_sorted)
    # Last position of each group of equal scores = one threshold.
    ends = np.flatnonzero(np.r_[scores_sorted[1:] != scores_sorted[:-1], True])
    tp_at = tp[ends]
    precision = tp_at / (ends + 1)
    recall_gain = np.diff(np.r_[0, tp_at]) / positives
    return float(np.sum(precision * recall_gain))


def evaluate(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float | None] | np.ndarray,
    case_ids: Sequence[str],
    seed: int = 42,
) -> Metrics:
    y = np.asarray(labels, dtype=np.int8)
    if len(y) != len(scores) or len(y) != len(case_ids):
        raise MetricError("labels, scores and case_ids must have the same length")
    frauds = int(y.sum())
    if frauds == 0:
        raise MetricError("no fraud cases")
    raw = np.array([np.nan if v is None else v for v in scores], dtype=float)
    s = prepare_scores(raw)
    order = ranking(s, tie_break_keys(case_ids, seed))
    return _metrics_from_order(y, s, order, coverage=float(np.mean(~np.isnan(raw))))


def _metrics_from_order(y: np.ndarray, s: np.ndarray, order: np.ndarray, coverage: float) -> Metrics:
    n, frauds = len(y), int(y.sum())
    y_sorted = y[order]
    caught = {k: int(y_sorted[: math.ceil(k * n)].sum()) for k in TOP_K}
    k1 = math.ceil(0.01 * n)
    return Metrics(
        n=n,
        frauds=frauds,
        coverage=coverage,
        pr_auc=average_precision_sorted(y_sorted, s[order]),
        recall_top_1pct=caught[0.01] / frauds,
        recall_top_5pct=caught[0.05] / frauds,
        precision_top_1pct=caught[0.01] / k1,
        frauds_caught_top_1pct=caught[0.01],
    )
