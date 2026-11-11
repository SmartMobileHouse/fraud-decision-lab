"""Price snapshot, credentials and the mock provider."""

from __future__ import annotations

import pytest

from fraudlab.providers.base import MissingCredentials, Task, load_prices
from fraudlab.providers.mock import MockProvider

TASK = Task("c1", {"amount": 10.0}, "abc123", "Is this fraud?", "fraud", "legit")


def test_price_snapshot_has_every_provider() -> None:
    prices = load_prices()
    assert set(prices) == {"jev", "openai_decisions", "llm", "clef", "clef_flash", "mock"}
    assert prices["jev"].cost(1_000_000, 500) == pytest.approx(0.042)  # output is free
    assert prices["llm"].cost(1_000_000, 1_000_000) == pytest.approx(0.60)
    assert prices["llm"].cost(1_000_000, 0, cached_tokens=1_000_000) == pytest.approx(0.01)


async def test_mock_is_deterministic_and_bounded() -> None:
    a, b = await MockProvider(seed=1).evaluate(TASK), await MockProvider(seed=1).evaluate(TASK)
    assert a.fraud_score == b.fraud_score and 0 <= (a.fraud_score or 0) <= 1
    assert a.status == "ok" and a.payload_hash == "abc123"


async def test_mock_can_simulate_failures() -> None:
    m = MockProvider(fail_every=2)
    first, second = await m.evaluate(TASK), await m.evaluate(TASK)
    assert first.status == "ok" and second.status == "error" and second.fraud_score is None


def test_missing_credentials_are_named(monkeypatch: pytest.MonkeyPatch) -> None:
    class Needs(MockProvider):
        required_env = ("SOME_KEY_XYZ",)

    monkeypatch.delenv("SOME_KEY_XYZ", raising=False)
    with pytest.raises(MissingCredentials, match="SOME_KEY_XYZ"):
        Needs().credentials()
