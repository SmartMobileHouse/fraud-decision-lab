"""The test suite itself must be unable to reach the network."""

from __future__ import annotations

import httpx
import pytest


def test_httpx_request_is_blocked() -> None:
    with pytest.raises((RuntimeError, httpx.ConnectError)):
        httpx.get("https://example.com", timeout=2)
