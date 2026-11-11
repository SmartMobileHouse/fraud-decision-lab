"""OpenAI Decisions and plain-LLM adapters against documented response shapes (offline, respx)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from fraudlab.providers import http as http_mod
from fraudlab.providers.base import Task
from fraudlab.providers.openai import OpenAIDecisionsProvider, OpenAILLMProvider

TASK = Task(
    case_id="syn-000777",
    fields={"amount": 912.5, "country_mismatch": True, "device_known": None},
    payload_hash="h7",
    question="Is this transaction fraudulent rather than legitimate?",
    criteria_true="The transaction is fraudulent",
    criteria_false="The transaction is legitimate",
    descriptions={"amount": "Transaction amount"},
)
DEC_URL = "https://api.openai.com/v1/decisions"
RESP_URL = "https://api.openai.com/v1/responses"
USAGE = {"input_tokens": 300, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 0, "total_tokens": 300}


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(http_mod.asyncio, "sleep", no_sleep)


def llm_reply(text: str, status: str = "completed", usage: dict | None = None) -> dict:  # type: ignore[type-arg]
    return {
        "model": "gpt-6-luna",
        "status": status,
        "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        "usage": usage or {"input_tokens": 350, "output_tokens": 9, "output_tokens_details": {"reasoning_tokens": 0}},
    }


@respx.mock
async def test_decisions_request_shape_and_success() -> None:
    route = respx.post(DEC_URL).respond(
        200,
        json={"model": "gpt-6-luna", "usage": USAGE,
              "answers": [{"type": "predicate", "name": "fraud", "probability": 0.91}]},
    )  # fmt: skip
    async with httpx.AsyncClient() as c:
        pred = await OpenAIDecisionsProvider(c).evaluate(TASK)
    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "gpt-6-luna"
    assert sent["input"][0]["content"][1] == {"type": "input_json", "data": TASK.fields}
    q = sent["questions"][0]
    assert q["type"] == "predicate" and q["name"] == "fraud"
    assert TASK.criteria_true in q["instructions"] and "amount: Transaction amount" in q["instructions"]
    assert "syn-000777" not in route.calls[0].request.content.decode()
    assert pred.status == "ok" and pred.fraud_score == 0.91
    assert pred.cost_usd == pytest.approx(300 * 0.10 / 1e6)  # output tokens are free on Decisions


@respx.mock
async def test_decisions_refusal_is_reported() -> None:
    respx.post(DEC_URL).respond(200, json={"model": "gpt-6-luna", "usage": USAGE,
                                           "answers": [{"type": "refusal", "name": "fraud"}]})  # fmt: skip
    async with httpx.AsyncClient() as c:
        pred = await OpenAIDecisionsProvider(c).evaluate(TASK)
    assert pred.status == "refusal" and pred.fraud_score is None and pred.cost_usd is not None
    assert pred.raw_answer == '{"type":"refusal","name":"fraud"}'


@respx.mock
async def test_decisions_bad_probability_is_invalid() -> None:
    respx.post(DEC_URL).respond(200, json={"answers": [{"type": "predicate", "probability": -0.2}]})
    async with httpx.AsyncClient() as c:
        assert (await OpenAIDecisionsProvider(c).evaluate(TASK)).status == "invalid"


@respx.mock
async def test_llm_request_forces_strict_json_and_no_reasoning() -> None:
    route = respx.post(RESP_URL).respond(200, json=llm_reply('{"fraud_score": 0.66}'))
    async with httpx.AsyncClient() as c:
        pred = await OpenAILLMProvider(c).evaluate(TASK)
    sent = json.loads(route.calls[0].request.content)
    fmt = sent["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    assert fmt["schema"]["required"] == ["fraud_score"] and fmt["schema"]["additionalProperties"] is False
    assert sent["reasoning"] == {"effort": "none"} and sent["store"] is False
    assert "syn-000777" not in route.calls[0].request.content.decode()
    assert pred.status == "ok" and pred.fraud_score == 0.66
    assert pred.cost_usd == pytest.approx((350 * 0.10 + 9 * 0.50) / 1e6)  # output tokens are billed


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ('{"fraud_score": 1.4}', "invalid"),  # out of range: never clipped or repaired
        ('{"score": 0.5}', "invalid"),
        ("probably fraud", "invalid"),
    ],
)
@respx.mock
async def test_llm_bad_output_is_a_failure(text: str, status: str) -> None:
    respx.post(RESP_URL).respond(200, json=llm_reply(text))
    async with httpx.AsyncClient() as c:
        pred = await OpenAILLMProvider(c).evaluate(TASK)
    assert pred.status == status and pred.fraud_score is None
    assert pred.raw_answer == text  # the original reply is kept verbatim


@respx.mock
async def test_llm_refusal_and_incomplete() -> None:
    refused_part = {"type": "refusal", "refusal": "I can't help"}
    refusal = {
        "model": "gpt-6-luna", "status": "completed", "usage": USAGE,
        "output": [{"type": "message", "content": [refused_part]}],
    }  # fmt: skip
    respx.post(RESP_URL).mock(
        side_effect=[
            httpx.Response(200, json=refusal),
            httpx.Response(
                200,
                json={**llm_reply(""), "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            ),  # fmt: skip
        ]
    )
    async with httpx.AsyncClient() as c:
        p = OpenAILLMProvider(c)
        first, second = await p.evaluate(TASK), await p.evaluate(TASK)
    assert first.status == "refusal" and first.raw_answer == "I can't help"
    assert second.status == "invalid" and "max_output_tokens" in (second.error or "")


@respx.mock
async def test_openai_errors_hide_key() -> None:
    respx.post(DEC_URL).respond(401, json={"error": {"message": "Incorrect API key"}})
    async with httpx.AsyncClient() as c:
        pred = await OpenAIDecisionsProvider(c).evaluate(TASK)
    assert pred.status == "error" and "sk-openai-secret" not in json.dumps(pred.as_dict())
