"""Shared HTTP layer: retries, no retries, timeouts, redaction. All offline (respx)."""

from __future__ import annotations

import httpx
import pytest
import respx

from fraudlab.providers import http as http_mod
from fraudlab.providers.http import post_json

URL = "https://api.example.test/v1/score"
SECRET = "sk-THIS-IS-SECRET-123"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(http_mod.asyncio, "sleep", fake_sleep)
    return waits


async def call() -> http_mod.CallOutcome:
    async with httpx.AsyncClient() as client:
        return await post_json(client, URL, {"Authorization": f"Bearer {SECRET}"}, {"x": 1})


@respx.mock
async def test_success_first_try() -> None:
    respx.post(URL).respond(200, json={"ok": True}, headers={"x-request-id": "req-1"})
    out = await call()
    assert out.status_code == 200 and out.body == {"ok": True}
    assert out.attempts == 1 and out.request_id == "req-1" and out.error is None


@respx.mock
async def test_retries_429_then_succeeds_honouring_retry_after(no_sleep: list[float]) -> None:
    respx.post(URL).mock(
        side_effect=[httpx.Response(429, headers={"retry-after": "3"}), httpx.Response(200, json={"ok": 1})]
    )
    out = await call()
    assert out.status_code == 200 and out.attempts == 2
    assert no_sleep == [3.0]
    assert [a["status_code"] for a in out.attempt_details] == [429, 200]
    assert all("body" not in a for a in out.attempt_details)


@respx.mock
async def test_gives_up_after_two_retries_on_5xx() -> None:
    route = respx.post(URL).respond(503, text="overloaded")
    out = await call()
    assert route.call_count == 3 and out.attempts == 3
    assert out.status_code == 503 and "HTTP 503" in (out.error or "")


@respx.mock
async def test_typesafe_529_is_retried() -> None:
    respx.post(URL).mock(side_effect=[httpx.Response(529), httpx.Response(200, json={})])
    assert (await call()).attempts == 2


@pytest.mark.parametrize("code", [400, 401, 403, 422])
@respx.mock
async def test_client_errors_are_not_retried(code: int) -> None:
    route = respx.post(URL).respond(code, json={"detail": "bad"})
    out = await call()
    assert route.call_count == 1 and out.status_code == code


@respx.mock
async def test_timeout_is_reported_after_retries() -> None:
    route = respx.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    out = await call()
    assert route.call_count == 3 and out.timed_out and out.status_code is None and out.error == "timeout"


@respx.mock
async def test_errors_never_contain_the_key() -> None:
    respx.post(URL).respond(401, json={"error": f"invalid key {SECRET}", "customer": "sensitive record"})
    out = await call()
    assert SECRET not in (out.error or "") and SECRET not in repr(out)
    assert "sensitive record" not in (out.error or "")
    assert "[REDACTED]" in out.body["error"]
