"""Jev and Clef adapters against recorded-shape responses (offline, respx)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from fraudlab.providers import http as http_mod
from fraudlab.providers.base import Task
from fraudlab.providers.system_one import ClefFlashProvider, ClefProvider, JevProvider

TASK = Task(
    case_id="syn-000123",
    fields={"amount": 249.9, "device_known": None, "tx_count_1h": 7},
    payload_hash="h1",
    question="Is this transaction fraudulent rather than legitimate?",
    criteria_true="The transaction is fraudulent",
    criteria_false="The transaction is legitimate",
    descriptions={"amount": "Transaction amount"},
)
JEV_URL = "https://api.typesafe.ai/v1/systemone"
CF_URL = "https://api.cloudflare.com/client/v4/accounts/acct123/ai/run/@cf/cloudflare/clef"
CF_FLASH_URL = "https://api.cloudflare.com/client/v4/accounts/acct123/ai/run/@cf/cloudflare/clef-flash"


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-secret")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-secret")

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(http_mod.asyncio, "sleep", no_sleep)


def ok_body(model: str = "jev-1.14.0", p: float = 0.83) -> dict:  # type: ignore[type-arg]
    return {"model": model, "answers": {"fraud": {"type": "noul", "noul": p}},
            "usage": {"input_tokens": 400, "output_tokens": 12}}  # fmt: skip


@respx.mock
async def test_jev_request_shape_and_success() -> None:
    route = respx.post(JEV_URL).respond(200, json=ok_body())
    async with httpx.AsyncClient() as c:
        pred = await JevProvider(c).evaluate(TASK)
    sent = json.loads(route.calls[0].request.content)
    assert route.calls[0].request.headers["authorization"] == "Bearer ts-secret"
    assert sent["model"] == "jev-1.14.0"
    assert sent["state"] == TASK.fields
    q = sent["questions"]["fraud"]
    assert q["type"] == "noul" and q["criteria"] == {"true": TASK.criteria_true, "false": TASK.criteria_false}
    assert q["instructions"]["field_descriptions"] == {"amount": "Transaction amount"}
    assert "syn-000123" not in route.calls[0].request.content.decode()  # case ID never sent
    assert pred.status == "ok" and pred.fraud_score == 0.83 and pred.model_reported == "jev-1.14.0"
    assert pred.raw_answer == '{"type":"noul","noul":0.83}'
    assert pred.input_tokens == 400 and pred.cost_usd == pytest.approx(400 * 0.042 / 1e6)
    assert pred.cost_source == "usage_calculated"


@pytest.mark.parametrize("first_usage_known", [False, True])
@respx.mock
async def test_retries_retain_each_attempt_cost_and_unknown_usage(first_usage_known: bool) -> None:
    respx.post(JEV_URL).mock(
        side_effect=[
            httpx.Response(503, json=ok_body() if first_usage_known else {"error": "busy"}),
            httpx.Response(200, json=ok_body()),
        ]
    )
    async with httpx.AsyncClient() as c:
        pred = await JevProvider(c).evaluate(TASK)
    details = json.loads(pred.attempt_details or "[]")
    assert [a["status_code"] for a in details] == [503, 200]
    assert pred.input_tokens == (800 if first_usage_known else 400)
    assert pred.cost_usd == pytest.approx((pred.input_tokens or 0) * 0.042 / 1e6)
    assert pred.cost_unknown_attempts == (0 if first_usage_known else 1)
    assert all("body" not in detail for detail in details)


@respx.mock
async def test_clef_unwraps_cloudflare_result_wrapper() -> None:
    respx.post(CF_URL).respond(200, json={"result": ok_body("clef", 0.4), "success": True, "errors": []})
    async with httpx.AsyncClient() as c:
        pred = await ClefProvider(c).evaluate(TASK)
    assert pred.status == "ok" and pred.fraud_score == 0.4 and pred.cost_usd == pytest.approx(400 * 0.24 / 1e6)


@respx.mock
async def test_clef_flash_uses_its_own_model_and_url() -> None:
    route = respx.post(CF_FLASH_URL).respond(200, json=ok_body("clef-flash", 0.1))
    async with httpx.AsyncClient() as c:
        pred = await ClefFlashProvider(c).evaluate(TASK)
    assert json.loads(route.calls[0].request.content)["model"] == "clef-flash"
    assert pred.status == "ok" and pred.provider == "clef_flash"


async def test_clef_flash_refuses_input_that_would_be_truncated() -> None:
    big = Task(**{**TASK.__dict__, "fields": {f"f{i}": "x" * 100 for i in range(1_000)}})
    async with httpx.AsyncClient() as c:
        pred = await ClefFlashProvider(c).evaluate(big)
    assert pred.status == "invalid" and "too long" in (pred.error or "") and pred.attempts == 0


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ({"model": "m", "answers": {"fraud": {"type": "noul", "noul": 1.7}}, "usage": {}}, "out of range"),
        ({"model": "m", "answers": {}, "usage": {}}, "missing answer"),
        ({"model": "m", "answers": {"fraud": {"type": "noul", "noul": "high"}}}, "not a number"),
    ],
)
@respx.mock
async def test_bad_answers_are_invalid_not_guessed(body: dict, reason: str) -> None:  # type: ignore[type-arg]
    respx.post(JEV_URL).respond(200, json=body)
    async with httpx.AsyncClient() as c:
        pred = await JevProvider(c).evaluate(TASK)
    assert pred.status == "invalid" and pred.fraud_score is None, reason
    assert pred.raw_answer is not None  # what the model actually said is kept


@respx.mock
async def test_auth_error_is_not_retried_and_hides_key() -> None:
    route = respx.post(JEV_URL).respond(401, json={"detail": "Missing or invalid API key"})
    async with httpx.AsyncClient() as c:
        pred = await JevProvider(c).evaluate(TASK)
    assert route.call_count == 1 and pred.status == "error" and "401" in (pred.error or "")
    assert "ts-secret" not in json.dumps(pred.as_dict())


@respx.mock
async def test_rate_limit_exhausted_is_reported() -> None:
    respx.post(JEV_URL).respond(429)
    async with httpx.AsyncClient() as c:
        pred = await JevProvider(c).evaluate(TASK)
    assert pred.status == "rate_limited" and pred.attempts == 3
    assert pred.cost_usd is None  # no usage is not evidence of zero billing
    assert pred.cost_unknown_attempts == pred.attempts


@respx.mock
async def test_timeout_is_reported() -> None:
    respx.post(CF_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    async with httpx.AsyncClient() as c:
        pred = await ClefProvider(c).evaluate(TASK)
    assert pred.status == "timeout" and pred.fraud_score is None
    assert pred.cost_usd is None  # unknown: may have been billed


async def test_question_without_descriptions_is_plain_text() -> None:
    from fraudlab.providers.system_one import system_one_body

    plain = Task(**{**TASK.__dict__, "descriptions": {}})
    assert system_one_body("m", plain)["questions"]["fraud"]["instructions"] == TASK.question


@respx.mock
async def test_out_of_range_answer_is_kept_verbatim() -> None:
    respx.post(JEV_URL).respond(200, json={"model": "m", "answers": {"fraud": {"type": "noul", "noul": 1.4}}})
    async with httpx.AsyncClient() as c:
        pred = await JevProvider(c).evaluate(TASK)
    assert pred.status == "invalid" and pred.fraud_score is None
    assert pred.raw_answer == '{"type":"noul","noul":1.4}'


def test_raw_answer_is_capped() -> None:
    from fraudlab.providers.common import RAW_ANSWER_LIMIT, raw_text

    text = raw_text("x" * (RAW_ANSWER_LIMIT + 50))
    assert text is not None and text.endswith("[truncated]") and len(text) < RAW_ANSWER_LIMIT + 20
