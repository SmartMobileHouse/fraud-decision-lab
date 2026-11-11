"""Estimated spending guard for live runs.

Before each case is sent, a conservative estimated cost is reserved. A case is sent only
if  spent + reserved + this reservation <= cap.  After the answer arrives the
reservation is replaced by the usage-derived cost. If the cost is unknown (no usage
reported, or a timeout that may still have been billed), the full reservation
is kept as estimated spend. A tokenizer estimate cannot guarantee a billing cap:
reported usage can exceed its reservation. Such overruns remain visible and
prevent further reservations; use provider-side limits where available.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass


@dataclass
class BudgetState:
    cap_usd: float
    spent_usd: float = 0.0
    reserved_usd: float = 0.0
    estimated_usd: float = 0.0  # part of spent that is an estimate (unknown or possibly-billed calls)
    refused: int = 0  # cases not sent because the budget was exhausted

    @property
    def overrun_usd(self) -> float:
        return max(0.0, self.spent_usd - self.cap_usd)


def _amount(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0")


class Budget:
    def __init__(self, cap_usd: float, already_spent_usd: float = 0.0, already_estimated_usd: float = 0.0) -> None:
        for name, value in (
            ("budget cap", cap_usd),
            ("already spent", already_spent_usd),
            ("already estimated", already_estimated_usd),
        ):
            _amount(value, name)
        if already_estimated_usd > already_spent_usd:
            raise ValueError("already estimated spend must not exceed already spent")
        self.state = BudgetState(cap_usd, spent_usd=already_spent_usd, estimated_usd=already_estimated_usd)
        self._lock = asyncio.Lock()

    async def reserve(self, amount_usd: float) -> bool:
        _amount(amount_usd, "reservation")
        async with self._lock:
            s = self.state
            if s.spent_usd + s.reserved_usd + amount_usd > s.cap_usd + 1e-12:
                s.refused += 1
                return False
            s.reserved_usd += amount_usd
            return True

    async def settle(self, reserved_usd: float, actual_usd: float | None, estimated_usd: float = 0.0) -> None:
        """Account known usage-derived cost plus unknown-cost estimates, even above the cap."""
        _amount(reserved_usd, "reservation")
        _amount(estimated_usd, "estimated cost")
        if actual_usd is not None:
            _amount(actual_usd, "usage-derived cost")
        async with self._lock:
            s = self.state
            s.reserved_usd = max(0.0, s.reserved_usd - reserved_usd)
            if actual_usd is None:
                s.spent_usd += reserved_usd
                s.estimated_usd += reserved_usd
            else:
                s.spent_usd += actual_usd + estimated_usd
                s.estimated_usd += estimated_usd

    @property
    def exhausted(self) -> bool:
        return self.state.refused > 0 or self.state.spent_usd >= self.state.cap_usd
