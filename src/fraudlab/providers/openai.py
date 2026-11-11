"""OpenAI adapters: Decisions (predicate question) and a plain LLM forced into strict JSON.

Verified against official docs, 2026-11-06:
- Decisions: POST /v1/decisions, model gpt-6-luna, `input` = user message with
  input_text + input_json parts, `questions` = [{"type": "predicate", ...}].
  Answer: answers[0].probability; a refusal is answers[0].type == "refusal".
  https://developers.openai.com/api/docs/guides/decisions
- Responses: POST /v1/responses with text.format = strict json_schema;
  reasoning.effort "none". https://developers.openai.com/api/docs/guides/structured-outputs
"""

from __future__ import annotations

import json
from typing import Any

from fraudlab.providers.base import Prediction, Task
from fraudlab.providers.common import HttpProvider, status_for_http_failure, valid_probability
from fraudlab.providers.http import post_json

API_BASE = "https://api.openai.com/v1"
QUESTION_NAME = "fraud"


def question_text(task: Task) -> str:
    """The question with its yes/no meaning and field descriptions, as plain text."""
    lines = [
        task.question,
        f"Answer yes/true if: {task.criteria_true}.",
        f"Answer no/false if: {task.criteria_false}.",
    ]
    if task.descriptions:
        lines.append("Field meanings:")
        lines += [f"- {name}: {text}" for name, text in task.descriptions.items()]
    return "\n".join(lines)


class _OpenAIProvider(HttpProvider):
    required_env = ("OPENAI_API_KEY",)
    model = "gpt-6-luna"

    def headers(self) -> dict[str, str]:
        key = self.credentials()["OPENAI_API_KEY"]
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


class OpenAIDecisionsProvider(_OpenAIProvider):
    name = "openai_decisions"

    def body(self, task: Task) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Transaction fields:"},
                        {"type": "input_json", "data": task.fields},
                    ],
                }
            ],
            "questions": [{"type": "predicate", "name": QUESTION_NAME, "instructions": question_text(task)}],
        }

    async def evaluate(self, task: Task) -> Prediction:
        outcome = await post_json(
            self.client,
            f"{API_BASE}/decisions",
            self.headers(),
            self.body(task),
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
        )
        if outcome.error or outcome.status_code != 200:
            return self._result(task, outcome, status_for_http_failure(outcome))
        data = outcome.body if isinstance(outcome.body, dict) else {}
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        reported = data.get("model") if isinstance(data.get("model"), str) else None
        answers = data.get("answers") or []
        answer = answers[0] if answers and isinstance(answers[0], dict) else {}
        if answer.get("type") == "refusal":
            return self._result(
                task, outcome, "refusal", usage=usage, model_reported=reported, error="model refused", raw_answer=answer
            )
        score = valid_probability(answer.get("probability"))
        if answer.get("type") != "predicate" or score is None:
            return self._result(
                task, outcome, "invalid", usage=usage, model_reported=reported,
                error="answer missing or not a probability in [0, 1]", raw_answer=answer or data,
            )  # fmt: skip
        return self._result(task, outcome, "ok", score=score, usage=usage, model_reported=reported, raw_answer=answer)


SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "fraud_score": {
            "type": "number",
            "description": "Likelihood from 0 to 1 that the transaction is fraud. Higher is more suspicious.",
        }
    },
    "required": ["fraud_score"],
    "additionalProperties": False,
}


class OpenAILLMProvider(_OpenAIProvider):
    """A general LLM, not a decision model: it must reply with {"fraud_score": <0..1>} and nothing else."""

    name = "llm"
    reasoning_effort = "none"  # classification task; keeps cost and latency low
    max_output_tokens = 64

    def body(self, task: Task) -> dict[str, Any]:
        instructions = (
            "You score payment transactions for fraud review.\n"
            + question_text(task)
            + "\nReply with fraud_score: a number from 0 to 1, the likelihood that the transaction is fraud."
        )
        return {
            "model": self.model,
            "instructions": instructions,
            "input": [{"role": "user", "content": json.dumps({"transaction_fields": task.fields}, ensure_ascii=False)}],
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
            "store": False,
            "text": {"format": {"type": "json_schema", "name": "fraud_score", "strict": True, "schema": SCORE_SCHEMA}},
        }

    async def evaluate(self, task: Task) -> Prediction:
        outcome = await post_json(
            self.client,
            f"{API_BASE}/responses",
            self.headers(),
            self.body(task),
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
        )
        if outcome.error or outcome.status_code != 200:
            return self._result(task, outcome, status_for_http_failure(outcome))
        data = outcome.body if isinstance(outcome.body, dict) else {}
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        reported = data.get("model") if isinstance(data.get("model"), str) else None
        extra: dict[str, Any] = {"usage": usage, "model_reported": reported}

        if data.get("status") not in (None, "completed"):
            reason = (data.get("incomplete_details") or {}).get("reason", data.get("status"))
            return self._result(
                task,
                outcome,
                "invalid",
                error=f"response not completed: {reason}",
                raw_answer=data.get("output"),
                **extra,
            )

        texts, refused, refusal_text = [], False, None
        for item in data.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue  # e.g. reasoning items
            for part in item.get("content") or []:
                if part.get("type") == "refusal":
                    refused, refusal_text = True, part.get("refusal")
                elif part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
        reply = "".join(texts)
        extra["raw_answer"] = refusal_text if refused else reply  # kept verbatim, even when invalid
        if refused:
            return self._result(task, outcome, "refusal", error="model refused", **extra)
        try:
            parsed = json.loads(reply)
        except ValueError:
            return self._result(task, outcome, "invalid", error="reply is not valid JSON", **extra)
        score = valid_probability(parsed.get("fraud_score") if isinstance(parsed, dict) else None)
        if score is None:  # invalid output is recorded as a failure, never repaired
            return self._result(task, outcome, "invalid", error="fraud_score missing or outside [0, 1]", **extra)
        return self._result(task, outcome, "ok", score=score, **extra)
