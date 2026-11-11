"""Mock provider: deterministic fake scores for tests and the offline demo.

Its numbers are NOT a measurement of any real model and must never be shown as one.
"""

from __future__ import annotations

import hashlib

from fraudlab.providers.base import DecisionProvider, Prediction, Task


class MockProvider(DecisionProvider):
    name = "mock"
    model = "mock-1"

    def __init__(self, seed: int = 0, fail_every: int = 0) -> None:
        self.seed = seed
        self.fail_every = fail_every  # simulate failures for testing (0 = never)
        self._calls = 0

    async def evaluate(self, task: Task) -> Prediction:
        self._calls += 1
        failed = self.fail_every and self._calls % self.fail_every == 0
        digest = hashlib.sha256(f"{self.seed}:{task.payload_hash}".encode()).digest()
        score = int.from_bytes(digest[:8], "big") / 2**64
        return Prediction(
            case_id=task.case_id,
            provider=self.name,
            model_requested=self.model,
            model_reported=self.model,
            fraud_score=None if failed else score,
            status="error" if failed else "ok",
            attempts=1,
            latency_ms=0.0,
            total_latency_ms=0.0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            cost_source="none",
            request_id=None,
            payload_hash=task.payload_hash,
            error="simulated failure" if failed else None,
        )
