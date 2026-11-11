"""Rules baseline: a transparent fraud-analyst checklist.

Each warning sign adds points; the score is points / maximum points, so it is
in [0, 1]. Rules use common fraud knowledge only. They are NOT derived from the
synthetic generator's formula, and they do not learn anything from the data.

Rules exist only for datasets whose fields have a known meaning. For anonymised
datasets (such as ULB, with V1..V28) the result is `not_run`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudlab.datasets.load import LoadedDataset

RULES_VERSION = "1"


@dataclass(frozen=True)
class Rule:
    name: str
    points: float
    fires: Callable[[pd.DataFrame], pd.Series]  # True where the warning sign is present


def _flag(series: pd.Series) -> pd.Series:
    """True where a nullable boolean is True; unknown (null) counts as False."""
    return series.astype("boolean").fillna(False).astype(bool)


SYNTHETIC_RULES = [
    Rule("3+ failed attempts in 24h", 3, lambda d: d.failed_attempts_24h >= 3),
    Rule("1-2 failed attempts in 24h", 1, lambda d: d.failed_attempts_24h.between(1, 2)),
    Rule("5+ transactions in the last hour", 2, lambda d: d.tx_count_1h >= 5),
    Rule("unknown device", 2, lambda d: _flag(~d.device_known.astype("boolean"))),
    Rule("device status missing", 1, lambda d: d.device_known.isna()),
    Rule("country differs from home country", 2, lambda d: _flag(d.country_mismatch)),
    Rule("amount 5x+ the card's usual", 2, lambda d: d.amount_to_avg_ratio >= 5),
    Rule("amount 3-5x the card's usual", 1, lambda d: d.amount_to_avg_ratio.between(3, 5, inclusive="left")),
    Rule("no spending history", 1, lambda d: d.avg_amount_30d.isna()),
    Rule("card younger than 30 days", 1, lambda d: d.card_age_days < 30),
    Rule("night time (00-06)", 1, lambda d: d.hour_of_day < 6),
    Rule(
        "high-risk merchant (digital goods, gaming, electronics)",
        1,
        lambda d: d.merchant_category.isin(["digital_goods", "gaming", "electronics"]),
    ),
]

PRIVACY_EMAIL_DOMAINS = ["protonmail.com", "anonymous.com", "mail.com"]

IEEE_RULES = [
    Rule("amount over 1,000", 2, lambda d: d.TransactionAmt > 1000),
    Rule("amount 500-1,000", 1, lambda d: d.TransactionAmt.between(500, 1000)),
    Rule("privacy / throwaway email domain", 2, lambda d: d.P_emaildomain.isin(PRIVACY_EMAIL_DOMAINS)),
    Rule(
        "purchaser and recipient emails differ",
        1,
        lambda d: d.P_emaildomain.notna() & d.R_emaildomain.notna() & (d.P_emaildomain != d.R_emaildomain),
    ),
    Rule("credit card (more often abused than debit)", 1, lambda d: d.card6 == "credit"),
    Rule("no purchaser email", 1, lambda d: d.P_emaildomain.isna()),
]

# Dataset config name -> its checklist. Names not listed get `not_run`.
RULESETS: dict[str, list[Rule]] = {
    "synthetic-v1": SYNTHETIC_RULES,
    "ieee-cis": IEEE_RULES,
    "ieee-cis-full": IEEE_RULES,
}


@dataclass(frozen=True)
class RulesResult:
    status: str  # "ok" or "not_run"
    scores: np.ndarray | None
    reason: str = ""


def score_rules(ds: LoadedDataset, rows: np.ndarray) -> RulesResult:
    """Score the given row positions of a dataset."""
    rules = RULESETS.get(ds.config.dataset.name)
    if rules is None:
        return RulesResult("not_run", None, "no rules for this dataset (fields have no known meaning)")
    frame = ds.frame.iloc[rows]
    points = np.zeros(len(frame))
    for rule in rules:
        points += rule.points * rule.fires(frame).fillna(False).to_numpy(dtype=bool)
    return RulesResult("ok", points / sum(r.points for r in rules))


def explain(ds: LoadedDataset, row: int) -> list[str]:
    """Which warning signs fired for one transaction (for report examples)."""
    rules = RULESETS.get(ds.config.dataset.name, [])
    frame = ds.frame.iloc[[row]]
    return [r.name for r in rules if bool(r.fires(frame).fillna(False).iloc[0])]
