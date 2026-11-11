"""Adapters for the "System One" request format: TypeSafe Jev and Cloudflare Clef / Clef-flash.

Request (verified against official docs, 2026-11-06):
    {"model": ..., "state": {<transaction fields>},
     "questions": {"fraud": {"type": "noul", "instructions": ..., "criteria": {"true": ..., "false": ...}}}}
Answer: answers.fraud.noul = probability of "yes" (fraud), in [0, 1].

Docs:
- Jev:  https://docs.typesafe.ai/api/system-one
- Clef: https://developers.cloudflare.com/workers-ai/models/clef/
"""

from __future__ import annotations

from typing import Any

from fraudlab.providers.base import Prediction, Task
from fraudlab.providers.common import HttpProvider, estimate_tokens, status_for_http_failure, valid_probability
from fraudlab.providers.http import post_json

QUESTION_NAME = "fraud"


def system_one_body(model: str, task: Task) -> dict[str, Any]:
    instructions: str | dict[str, Any] = task.question
    if task.descriptions:  # the question plus what each field means
        instructions = {"question": task.question, "field_descriptions": task.descriptions}
    return {
        "model": model,
        "state": task.fields,  # the transaction itself; never the case ID or label
        "questions": {
            QUESTION_NAME: {
                "type": "noul",
                "instructions": instructions,
                "criteria": {"true": task.criteria_true, "false": task.criteria_false},
            }
        },
    }


class SystemOneProvider(HttpProvider):
    url: str
    body_model: str  # the "model" value inside the request body
    max_context_tokens: int | None = None  # refuse to send inputs that would be silently cut

    def headers(self) -> dict[str, str]:
        raise NotImplementedError

    def endpoint(self) -> str:
        return self.url

    def unwrap(self, body: Any) -> Any:
        """Remove any provider wrapper around the System One response."""
        return body

    async def evaluate(self, task: Task) -> Prediction:
        body = system_one_body(self.body_model, task)
        if self.max_context_tokens and estimate_tokens(body) > 0.9 * self.max_context_tokens:
            return self._result(
                task, None, "invalid",
                error=f"input too long for {self.model} ({self.max_context_tokens:,}-token context); not sent",
            )  # fmt: skip
        outcome = await post_json(
            self.client,
            self.endpoint(),
            self.headers(),
            body,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
        )
        if outcome.error or outcome.status_code != 200:
            return self._result(task, outcome, status_for_http_failure(outcome))

        data = self.unwrap(outcome.body)
        if not isinstance(data, dict):
            return self._result(task, outcome, "invalid", error="response is not a JSON object", raw_answer=data)
        answer = (data.get("answers") or {}).get(QUESTION_NAME) or {}
        score = valid_probability(answer.get("noul"))
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        reported = data.get("model") if isinstance(data.get("model"), str) else None
        if answer.get("type") != "noul" or score is None:
            return self._result(
                task, outcome, "invalid", usage=usage, model_reported=reported,
                error="answer missing or not a probability in [0, 1]", raw_answer=answer or data,
            )  # fmt: skip
        return self._result(task, outcome, "ok", score=score, usage=usage, model_reported=reported, raw_answer=answer)


class JevProvider(SystemOneProvider):
    name = "jev"
    model = "jev-1.14.0"  # pinned: the alias jev-latest can move
    body_model = "jev-1.14.0"
    url = "https://api.typesafe.ai/v1/systemone"
    required_env = ("TYPESAFE_API_KEY",)
    max_context_tokens = 32_768  # per-question context without the long-context header

    def headers(self) -> dict[str, str]:
        key = self.credentials()["TYPESAFE_API_KEY"]
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


class _CloudflareProvider(SystemOneProvider):
    required_env = ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN")
    path_model: str

    def endpoint(self) -> str:
        account = self.credentials()["CLOUDFLARE_ACCOUNT_ID"]
        return f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{self.path_model}"

    def headers(self) -> dict[str, str]:
        token = self.credentials()["CLOUDFLARE_API_TOKEN"]
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def unwrap(self, body: Any) -> Any:
        # Cloudflare REST wraps the answer: {"result": {...}, "success": true, "errors": [], "messages": []}.
        # Not documented on the Clef pages; confirmed by the live probe on 2026-11-06. Bare shape also accepted.
        if isinstance(body, dict) and "answers" not in body and isinstance(body.get("result"), dict):
            return body["result"]
        return body


class ClefProvider(_CloudflareProvider):
    name = "clef"
    model = "@cf/cloudflare/clef"
    path_model = "@cf/cloudflare/clef"
    body_model = "clef"
    max_context_tokens = 65_536


class ClefFlashProvider(_CloudflareProvider):
    name = "clef_flash"
    model = "@cf/cloudflare/clef-flash"
    path_model = "@cf/cloudflare/clef-flash"
    body_model = "clef-flash"
    max_context_tokens = 24_576  # longer state is silently truncated, so we refuse instead
