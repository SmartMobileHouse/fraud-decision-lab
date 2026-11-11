"""`fraudlab replay`: recompute a finished run's metrics from its saved predictions and labels.

No model or provider is called. The result must match the saved metrics.json
exactly; any difference is reported.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from fraudlab.benchmark.ledger import Ledger
from fraudlab.benchmark.provenance import file_hash
from fraudlab.benchmark.runner import _ids_hash, _metrics
from fraudlab.config import OUTPUTS
from fraudlab.providers.base import Prediction


class ReplayError(RuntimeError):
    """The run folder is missing files or cannot be replayed."""


def _clean(value: Any) -> Any:
    """Parquet turns missing values into NaN; predictions use None."""
    return None if isinstance(value, float) and math.isnan(value) else value


def replay_run(run_id: str, outputs: Path | None = None) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    out = (outputs or OUTPUTS) / run_id
    needed = ["metrics.json", "predictions.parquet", "labels.parquet", "manifest.json"]
    missing = [n for n in needed if not (out / n).is_file()]
    if missing:
        raise ReplayError(f"{out} is missing {', '.join(missing)}")
    saved = json.loads((out / "metrics.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest.get("kind") != "run":
        raise ReplayError("only runs made by `fraudlab run` can be replayed")
    labels = pd.read_parquet(out / "labels.parquet")
    preds = pd.read_parquet(out / "predictions.parquet")
    ids = labels["case_id"].astype(str).tolist()
    if labels["case_id"].duplicated().any() or not set(labels["label"].unique()) <= {0, 1}:
        raise ReplayError("labels require unique case IDs and binary outcomes")
    if manifest.get("run", {}).get("case_ids_sha256") != _ids_hash(ids):
        raise ReplayError("label case IDs differ from the manifest")

    by_model: dict[str, list[Prediction]] = {}
    for name, rows in preds.groupby("provider", sort=False):
        if rows["case_id"].duplicated().any():
            raise ReplayError(f"duplicate predictions for {name}")
        lookup = {str(r["case_id"]): r for r in rows.to_dict("records")}
        if set(lookup) != set(ids):
            raise ReplayError(f"predictions for {name} do not cover the same cases as the labels")
        by_model[str(name)] = [Prediction(**{k: _clean(v) for k, v in lookup[c].items()}) for c in ids]

    seed = saved["bootstrap"]["seed"]
    resamples = saved["bootstrap"]["resamples"]
    if set(by_model) != set(manifest["run"]["providers"]):
        raise ReplayError("prediction providers differ from the manifest")
    accounting = Ledger(out / "attempts.jsonl").summary() if saved.get("accounting_version") == 2 else None
    again = _metrics(
        by_model,
        labels["label"].to_numpy(),
        ids,
        seed,
        resamples,
        accounting=accounting.by_provider if accounting is not None else None,
    )

    differences = []
    hashes = manifest.get("artifacts_sha256", {})
    if any(not (out / name).is_file() or file_hash(out / name) != digest for name, digest in hashes.items()):
        differences.append("artifact fingerprints")
    if accounting is not None:
        budget = saved.get("budget", {})
        if not (
            math.isclose(accounting.spent_usd, budget.get("spent_usd", 0.0), rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(accounting.estimated_usd, budget.get("estimated_usd", 0.0), rel_tol=1e-12, abs_tol=1e-12)
        ):
            differences.append("budget accounting")
    for key in ("reference", "scored_cases", "frauds", "bootstrap", "models"):
        if json.loads(json.dumps(again[key])) != saved[key]:
            differences.append(key)
    return saved, again, differences
