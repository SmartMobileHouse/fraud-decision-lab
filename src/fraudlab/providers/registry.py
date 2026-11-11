"""Build provider objects by name, and build Tasks from a dataset row."""

from __future__ import annotations

import httpx

from fraudlab.datasets.load import LoadedDataset
from fraudlab.datasets.payload import Question, case_id, field_descriptions, payload_hash, render_case
from fraudlab.providers.base import DecisionProvider, Prediction, Task
from fraudlab.providers.common import HttpProvider, estimate_tokens
from fraudlab.providers.mock import MockProvider
from fraudlab.providers.openai import OpenAIDecisionsProvider, OpenAILLMProvider
from fraudlab.providers.system_one import ClefFlashProvider, ClefProvider, JevProvider

LIVE_PROVIDERS: dict[str, type[HttpProvider]] = {
    "jev": JevProvider,
    "openai_decisions": OpenAIDecisionsProvider,
    "clef": ClefProvider,
    "clef_flash": ClefFlashProvider,
    "llm": OpenAILLMProvider,
}
ALL_PROVIDERS = [*LIVE_PROVIDERS, "mock"]


def make_provider(name: str, client: httpx.AsyncClient) -> DecisionProvider:
    if name == "mock":
        return MockProvider()
    if name not in LIVE_PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; choose from {', '.join(ALL_PROVIDERS)}")
    return LIVE_PROVIDERS[name](client)


def make_task(ds: LoadedDataset, row_position: int, question: Question) -> Task:
    fields = render_case(ds, row_position)
    return Task(
        case_id=case_id(ds, row_position),
        fields=fields,
        payload_hash=payload_hash(fields),
        question=question.question,
        criteria_true=question.criteria_true,
        criteria_false=question.criteria_false,
        descriptions=field_descriptions(ds),
    )


def worst_case_cost(provider: DecisionProvider, task: Task, attempts: int = 3) -> float:
    """Conservative token-based estimate for reservations; not a guaranteed billing upper bound."""
    if not isinstance(provider, HttpProvider):
        return 0.0
    input_tokens = int(estimate_tokens({"fields": task.fields, "descriptions": task.descriptions, "q": task.question}))
    input_tokens = int(input_tokens * 1.5) + 200  # margin for wrappers and tokenizer differences
    output_tokens = getattr(provider, "max_output_tokens", 64)
    return attempts * provider.price.cost(input_tokens, output_tokens)


class ReplayProvider(DecisionProvider):
    """Returns previously saved predictions. Never calls the network."""

    name = "replay"
    model = "replay"

    def __init__(self, saved: list[Prediction]) -> None:
        self.saved = {p.case_id: p for p in saved}

    async def evaluate(self, task: Task) -> Prediction:
        hit = self.saved.get(task.case_id)
        if hit is not None and hit.payload_hash == task.payload_hash:
            return hit
        return Prediction(
            case_id=task.case_id, provider=self.name, model_requested=self.model, model_reported=None,
            fraud_score=None, status="not_run", attempts=0, latency_ms=None, total_latency_ms=None,
            input_tokens=None, output_tokens=None, cost_usd=None, cost_source="none", request_id=None,
            payload_hash=task.payload_hash, error="no compatible saved prediction for this case and payload",
        )  # fmt: skip
