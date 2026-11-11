"""Synthetic card transactions with realistic fraud patterns (fictional, no real people).

How labels are made:
1. Each transaction gets a scenario (normal, amount spike, card testing, ...).
2. The scenario shapes its fields (amount, counts, device, country, ...).
3. A hidden risk score is computed from the fields plus random noise.
4. Fraud is drawn from that risk score, tuned to about 3% overall.

So fraud is *likely* when signals look bad, but never guaranteed. No single
visible rule reproduces the label. The scenario column is kept for
explanations only and is never a model input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GENERATOR_VERSION = "1"
DEFAULT_ROWS = 66_667  # with a 70/15/15 split the holdout is exactly 10,000 rows
DEFAULT_SEED = 42
TARGET_FRAUD_RATE = 0.03
DAYS = 30

SCENARIOS = {  # share of all transactions
    "normal": 0.900,
    "noisy_legit_outlier": 0.050,  # unusual but legitimate (holiday, big purchase)
    "amount_spike": 0.0125,  # much larger than this card usually spends
    "card_testing": 0.0125,  # burst of small attempts, several failures
    "new_device_abroad": 0.0125,  # unknown device in another country
    "account_takeover": 0.0125,  # new device, big digital purchase at night
}
CATEGORIES = ["grocery", "restaurants", "fuel", "fashion", "electronics", "travel", "gaming", "digital_goods"]
CATEGORY_P = [0.26, 0.18, 0.12, 0.12, 0.10, 0.08, 0.07, 0.07]
CURRENCIES, CURRENCY_P = ["EUR", "USD", "GBP"], [0.85, 0.10, 0.05]

FEATURES = [
    "amount",
    "currency",
    "merchant_category",
    "hour_of_day",
    "tx_count_1h",
    "tx_count_24h",
    "avg_amount_30d",
    "amount_to_avg_ratio",
    "device_known",
    "country_mismatch",
    "card_age_days",
    "failed_attempts_24h",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic(rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = rows
    names = list(SCENARIOS)
    scenario = rng.choice(names, size=n, p=list(SCENARIOS.values()))
    s = {name: scenario == name for name in names}

    # Time: spread over 30 days, strictly increasing, in seconds.
    event_time = np.sort(rng.choice(DAYS * 86_400, size=n, replace=False))
    hour = (event_time // 3600) % 24
    night = rng.integers(0, 6, n)  # 00:00-05:59
    hour = np.where(s["account_takeover"] & (rng.random(n) < 0.7), night, hour)

    # Card history.
    card_age = np.round(rng.gamma(2.0, 400.0, n)).astype(int) + 1
    card_age = np.where(s["card_testing"] & (rng.random(n) < 0.6), rng.integers(1, 30, n), card_age)
    avg_30d = np.round(rng.lognormal(3.6, 0.7, n), 2)  # typical spend ~ EUR 37

    # Amount: usually near the card's average.
    amount = avg_30d * rng.lognormal(0.0, 0.45, n)
    amount = np.where(s["amount_spike"], avg_30d * rng.uniform(6, 25, n), amount)
    amount = np.where(s["account_takeover"], avg_30d * rng.uniform(3, 12, n), amount)
    amount = np.where(s["card_testing"], rng.uniform(0.5, 5.0, n), amount)
    amount = np.where(s["noisy_legit_outlier"], avg_30d * rng.uniform(3, 10, n), amount)
    amount = np.round(np.clip(amount, 0.5, 20_000), 2)

    # Activity counts (including this transaction).
    tx_1h = 1 + rng.poisson(0.3, n)
    tx_1h = np.where(s["card_testing"], 1 + rng.poisson(8, n), tx_1h)
    tx_1h = np.where(s["noisy_legit_outlier"], 1 + rng.poisson(1.5, n), tx_1h)
    tx_24h = tx_1h + rng.poisson(2.5, n)
    failed = rng.poisson(0.05, n)
    failed = np.where(s["card_testing"], rng.poisson(4, n), failed)
    failed = np.where(s["account_takeover"], rng.poisson(1.2, n), failed)

    # Device and geography.
    device_known = rng.random(n) > 0.10
    new_device = s["new_device_abroad"] | s["account_takeover"]
    device_known = np.where(new_device, rng.random(n) > 0.85, device_known)
    device_known = np.where(s["noisy_legit_outlier"], rng.random(n) > 0.4, device_known)
    country_mismatch = rng.random(n) < 0.04
    country_mismatch = np.where(s["new_device_abroad"], rng.random(n) < 0.9, country_mismatch)
    country_mismatch = np.where(s["noisy_legit_outlier"], rng.random(n) < 0.5, country_mismatch)

    category = rng.choice(CATEGORIES, size=n, p=CATEGORY_P)
    risky_cat = np.array(["electronics", "gaming", "digital_goods"])
    category = np.where(
        (s["account_takeover"] | s["amount_spike"]) & (rng.random(n) < 0.7),
        rng.choice(risky_cat, size=n),
        category,
    )
    category = np.where(s["card_testing"] & (rng.random(n) < 0.8), "digital_goods", category)

    # Hidden risk score -> fraud label.
    ratio = amount / avg_30d
    logit = (
        0.9 * np.log1p(np.maximum(ratio - 1, 0))
        + 0.35 * np.minimum(tx_1h - 1, 10)
        + 0.6 * np.minimum(failed, 6)
        + 1.2 * (~device_known)
        + 1.0 * country_mismatch
        + 0.8 * (card_age < 30)
        + 0.5 * np.isin(category, risky_cat)
        + 0.5 * (hour < 6)
        + 1.5 * (s["amount_spike"] | s["card_testing"] | s["new_device_abroad"] | s["account_takeover"])
        - 1.0 * s["noisy_legit_outlier"]
        + rng.normal(0, 1.0, n)  # noise: same signals do not always mean the same outcome
    )
    lo, hi = -20.0, 5.0  # tune the intercept so the expected fraud rate is TARGET_FRAUD_RATE
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _sigmoid(logit + mid).mean() < TARGET_FRAUD_RATE else (lo, mid)
    is_fraud = (rng.random(n) < _sigmoid(logit + lo)).astype("int8")

    # Missing data, as in real systems.
    avg_obs = np.where((card_age < 30) | (rng.random(n) < 0.03), np.nan, avg_30d)
    ratio_obs = np.where(np.isnan(avg_obs), np.nan, np.round(amount / avg_obs, 3))
    device_obs = pd.array(device_known, dtype="boolean")
    device_obs[rng.random(n) < 0.08] = pd.NA
    country_obs = pd.array(country_mismatch, dtype="boolean")
    country_obs[rng.random(n) < 0.04] = pd.NA

    return pd.DataFrame(
        {
            "tx_id": [f"syn-{i:06d}" for i in range(1, n + 1)],
            "event_time": event_time.astype("int64"),
            "amount": amount,
            "currency": rng.choice(CURRENCIES, size=n, p=CURRENCY_P),
            "merchant_category": category,
            "hour_of_day": hour.astype("int8"),
            "tx_count_1h": tx_1h.astype("int16"),
            "tx_count_24h": tx_24h.astype("int16"),
            "avg_amount_30d": avg_obs,
            "amount_to_avg_ratio": ratio_obs,
            "device_known": device_obs,
            "country_mismatch": country_obs,
            "card_age_days": card_age.astype("int32"),
            "failed_attempts_24h": failed.astype("int16"),
            "scenario": scenario,  # explanation only; never a model input
            "is_fraud": is_fraud,
        }
    )
