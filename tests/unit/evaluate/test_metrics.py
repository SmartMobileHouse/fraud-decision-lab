"""Metrics against hand-computed answers and sklearn."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from fraudlab.evaluate.bootstrap import paired_bootstrap
from fraudlab.evaluate.metrics import MetricError, evaluate

IDS = [f"c{i}" for i in range(200)]


def test_hand_computed_example() -> None:
    # 200 cases, 4 frauds. Top 1% = 2 cases, top 5% = 10 cases.
    y = np.zeros(200, dtype=int)
    y[[0, 1, 50, 150]] = 1
    s = np.linspace(0.5, 0.0, 200)  # case 0 most suspicious ... case 199 least
    s[0], s[1], s[50], s[150] = 0.99, 0.10, 0.98, 0.0001
    m = evaluate(y, s, IDS)
    # ranking: c0 (0.99), c50 (0.98), then c2.. by score; c1 at 0.10 falls far down
    assert m.frauds_caught_top_1pct == 2
    assert m.recall_top_1pct == 0.5
    assert m.precision_top_1pct == 1.0
    assert m.recall_top_5pct == 0.5
    assert m.coverage == 1.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pr_auc_matches_sklearn_including_ties(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = (rng.random(200) < 0.1).astype(int)
    s = np.round(rng.random(200), 1)  # only 11 distinct values -> many ties
    assert evaluate(y, s, IDS).pr_auc == pytest.approx(average_precision_score(y, s))


def test_failures_rank_last_and_lower_coverage() -> None:
    y = np.zeros(200, dtype=int)
    y[:2] = 1
    good = evaluate(y, [0.9, 0.9] + [0.1] * 198, IDS)
    failed = evaluate(y, [None, None] + [0.1] * 198, IDS)
    assert good.recall_top_1pct == 1.0
    assert failed.recall_top_1pct == 0.0
    assert failed.coverage == pytest.approx(0.99)


def test_ties_do_not_depend_on_row_order() -> None:
    y = np.zeros(200, dtype=int)
    y[::20] = 1
    s = np.full(200, 0.5)  # every case tied
    a = evaluate(y, s, IDS)
    perm = np.random.default_rng(3).permutation(200)
    b = evaluate(y[perm], s[perm], [IDS[i] for i in perm])
    assert a == b


def test_scores_outside_0_1_are_rejected() -> None:
    with pytest.raises(MetricError, match="between 0 and 1"):
        evaluate([1, 0], [1.5, 0.1], ["a", "b"])


def test_no_fraud_is_rejected() -> None:
    with pytest.raises(MetricError, match="no fraud"):
        evaluate([0, 0], [0.5, 0.1], ["a", "b"])


def test_bootstrap_brackets_point_estimate_and_detects_real_difference() -> None:
    rng = np.random.default_rng(0)
    n = 5_000
    y = (rng.random(n) < 0.05).astype(int)
    ids = [f"x{i}" for i in range(n)]
    strong = np.clip(0.6 * y + rng.random(n) * 0.5, 0, 1)
    weak = rng.random(n)
    r = paired_bootstrap(y, {"xgb": strong, "random": weak}, ids, reference="xgb", resamples=200)
    point = evaluate(y, strong, ids).pr_auc
    assert r.intervals["xgb"]["pr_auc"].low <= point <= r.intervals["xgb"]["pr_auc"].high
    diff = r.differences["random"]["pr_auc"]
    assert diff.high < 0  # random is clearly worse than xgb


def test_bootstrap_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(1_000) < 0.05).astype(int)
    ids = [f"x{i}" for i in range(1_000)]
    s = rng.random(1_000)
    a = paired_bootstrap(y, {"m": s}, ids, resamples=50, seed=7)
    b = paired_bootstrap(y, {"m": s}, ids, resamples=50, seed=7)
    assert a.intervals == b.intervals
