"""`fraudlab export-report-data`: a public-safe evidence bundle for publishing tools.

Writes one JSON file with every number, curve and example a published report may
use, each under a stable evidence ID. Separate publishing tools read this file;
they never need to read run folders directly.

Safety:
- explicit publication approval is required independently of API processing permission;
- no transaction rows from real datasets: real datasets contribute aggregates and
  curves (downsampled to ~100 points) only; examples come from the synthetic data.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudlab import __version__
from fraudlab.config import OUTPUTS
from fraudlab.report.errors import ReportError
from fraudlab.report.errors import RestrictedDatasetError as RestrictedDatasetError
from fraudlab.report.publication import is_bundled_synthetic, require_publication
from fraudlab.report.render import (
    RunData,
    _budget_curve,
    _examples,
    _load,
    _pr_curve,
    _scores,
)

SCHEMA = "smh-research-data/1"


def _start_from_run_id(run_id: str) -> str | None:
    """Run IDs begin with their start time (YYYYMMDDTHHMMSSZ); older manifests lack started_utc."""
    try:
        return datetime.strptime(run_id[:16], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).isoformat()
    except ValueError:
        return None


def _model_versions(predictions: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, rows in predictions.groupby("provider"):
        reported = sorted({str(v) for v in rows["model_reported"].dropna().unique()})
        requested = sorted({str(v) for v in rows["model_requested"].dropna().unique()})
        out[str(name)] = reported or requested
    return out


def _public_provenance(manifest: dict[str, Any]) -> dict[str, Any]:
    """Execution metadata without raw records or machine-specific data paths."""

    def redact_config_paths(config: dict[str, Any] | None) -> None:
        if not config:
            return
        ds = config.get("dataset", {})
        for item in (ds, ds.get("join") or {}):
            if item.get("path") and Path(item["path"]).is_absolute():
                item["path"] = Path(item["path"]).name

    config = deepcopy(manifest.get("dataset_config"))
    redact_config_paths(config)
    identity = deepcopy(manifest.get("identity"))
    if identity:
        redact_config_paths(identity.get("dataset_config"))
    if identity and "input_files_sha256" in identity:
        identity["input_files_sha256"] = {
            Path(p).name if Path(p).is_absolute() else p: v for p, v in identity["input_files_sha256"].items()
        }
    run = manifest.get("run", {})
    settings = {k: run[k] for k in (
        "providers", "scored_cases", "case_ids_sha256", "max_records", "concurrency", "rate_per_s",
        "allow_external_requests", "max_cost_usd", "connection_pool", "question_version", "question",
        "provider_configs",
    ) if k in run}  # fmt: skip
    if not run:
        settings = {k: manifest[k] for k in (
            "providers", "rates", "seconds", "question", "provider_settings", "prices",
        ) if k in manifest}  # fmt: skip
    return {
        "git": manifest.get("git"),
        "dataset_config": config,
        "execution": settings,
        "xgboost": manifest.get("xgboost"),
        "identity": identity,
        "identity_sha256": manifest.get("identity_sha256"),
        "identity_hash_scope": "original local identity before any absolute input paths are redacted",
        "sessions": manifest.get("sessions", []),
        "artifacts_sha256": manifest.get("artifacts_sha256", {}),
        "source_sha256": manifest.get("source_sha256"),
        "limitation": "Missing historical settings remain unknown; metadata does not freeze provider-side behavior.",
    }


def _verification_records(run: RunData) -> dict[str, Any] | None:
    """Permit independent metric calculation for the fingerprinted fictional dataset only."""
    if not is_bundled_synthetic(run.manifest):
        return None
    from fraudlab.config import REPO_ROOT

    source = pd.read_parquet(REPO_ROOT / "data" / "demo" / "synthetic.parquet", columns=["tx_id", "is_fraud"])
    source = source.set_index("tx_id")["is_fraud"]
    ids = run.labels["case_id"].astype(str)
    if not ids.isin(source.index).all() or not np.array_equal(
        source.loc[ids].to_numpy(), run.labels["label"].to_numpy()
    ):
        raise ReportError("Saved verification labels do not match the bundled fictional dataset")
    columns = ["case_id", "provider", "fraud_score", "status", "payload_hash"]
    predictions = run.predictions[columns].astype(object).where(pd.notna(run.predictions[columns]), None)
    return {
        "scope": "saved final predictions only; operational evidence is summarized separately",
        "labels": run.labels[["case_id", "label"]].to_dict("records"),
        "predictions": predictions[predictions["provider"] != "mock"].to_dict("records"),
    }


def _experiment(run_id: str, outputs: Path) -> dict[str, Any]:
    run = _load(run_id, outputs)
    ds = run.manifest["dataset"]
    m = run.metrics
    versions = _model_versions(run.predictions)
    models: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    for name, entry in m["models"].items():
        if name == "mock":
            continue
        item: dict[str, Any] = {"status": entry["status"], "coverage": entry["coverage"],
                                "statuses": entry.get("statuses", {}), "versions": versions.get(name, [])}  # fmt: skip
        item["operational"] = entry.get("operational")
        item["cost_incomplete"] = entry.get("cost_incomplete", name not in ("rules", "xgboost"))
        item["cost_usd"] = entry.get("cost_usd")
        item["known_cost_usd"] = entry.get("known_cost_usd", entry.get("cost_usd"))
        item["estimated_cost_usd"] = entry.get("estimated_cost_usd")
        item["cost_method"] = (
            "usage × recorded prices, plus separately identified unknown-cost reservations; not invoices"
        )
        if entry["status"] == "ok":
            item.update(
                metrics=entry["metrics"],
                ranges_95=entry["ranges_95"],
                difference_vs_reference_95=entry.get("difference_vs_xgboost_95"),
                cost_usd=entry["cost_usd"],
                cost_per_1k_usd=entry["cost_per_1k_usd"],
                cost_per_fraud_caught_top_1pct_usd=entry.get("cost_per_fraud_caught_top_1pct_usd"),
                tokens=entry.get("tokens"),
            )
            y, s = _scores(run, name)
            bx, by = _budget_curve(y, s, run.labels["case_id"].astype(str).tolist(), m["bootstrap"]["seed"])
            rx, ry = _pr_curve(y, s)
            curves[name] = {
                "review_budget": {"x_pct_reviewed": bx, "y_pct_fraud_caught": by},
                "precision_recall": {
                    "recall": rx,
                    "precision": ry,
                    "method": "interpolated precision envelope; AP is non-interpolated",
                },
            }
        else:
            item["reason"] = entry.get("reason", "")
        models[name] = item
    run_info = run.manifest.get("run", {})
    return {
        "run_id": run_id,
        "started_utc": run.manifest.get("started_utc") or _start_from_run_id(run_id) or run.manifest["created_utc"],
        "manifest_written_utc": run.manifest["created_utc"],
        "dataset": {
            "name": ds["name"], "license_note": ds["license_note"], "rows": ds["rows"], "features": ds["features"],
            "files_sha256": {Path(p).name if Path(p).is_absolute() else p: h for p, h in ds["files_sha256"].items()},
            "external_processing_allowed": ds["external_processing_allowed"],
            "publication_allowed": True,
            "publication_note": ds.get("publication_note", "Original bundled synthetic data (MIT)."),
        },
        "split": run.manifest["split"],
        "scored_cases": m["scored_cases"],
        "frauds": m["frauds"],
        "top_1pct_cases": -(-m["scored_cases"] // 100),
        "reference_model": m.get("reference"),
        "bootstrap": m["bootstrap"],
        "versions": run.manifest["versions"],
        "question_version": run_info.get("question_version"),
        "prices_snapshot": run_info.get("prices"),
        "spent_usd": (m.get("budget") or {}).get("spent_usd"),
        "models": models,
        "curves": curves,
        "provenance": _public_provenance(run.manifest),
        "verification": _verification_records(run),
    }  # fmt: skip


def _loadtest(loadtest_id: str, outputs: Path) -> dict[str, Any]:
    path = outputs / loadtest_id / "loadtest.json"
    if not path.is_file():
        raise ReportError(f"{path} not found")
    data = json.loads(path.read_text())
    require_publication(data, loadtest_id)
    if data.get("state", "complete") != "complete":
        raise ReportError(f"{loadtest_id} is not a completed load test")
    by_provider: dict[str, dict[str, Any]] = {}
    for st in data["steps"]:
        by_provider.setdefault(st["provider"], {})[f"{float(st['target_rps']):g}"] = st
    evidence = "per-call records available" if (path.parent / "calls.parquet").is_file() else "aggregate-only evidence"
    return {
        "run_id": loadtest_id,
        "started_utc": data.get("started_utc") or _start_from_run_id(loadtest_id),
        "seconds_per_rate": data["seconds"],
        "rates": data["rates"],
        "spent_usd": data["budget"]["spent_usd"],
        "steps": data["steps"],
        "by_provider": by_provider,
        "evidence": evidence,
        "provenance": _public_provenance(data),
        "limitation": "Aggregate summaries cannot independently verify individual calls or billed charges.",
    }


def export_report_data(
    run_ids: list[str], out_file: Path, loadtest_id: str | None = None, outputs: Path | None = None
) -> Path:
    outputs = outputs or OUTPUTS
    experiments = {rid: _experiment(rid, outputs) for rid in run_ids}  # raises on restricted datasets
    synthetic = next((rid for rid, e in experiments.items() if e["verification"] is not None), None)
    examples: dict[str, Any] | None = None
    if synthetic:
        examples = {
            "run_id": synthetic,
            "selection": "2 fraud and 2 legitimate holdout cases drawn at random with a fixed seed (7); not curated",
            "cases": _examples(_load(synthetic, outputs), include_mock=False),
        }
    bundle = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "generator": {"tool": "fraudlab", "version": __version__},
        "experiments": experiments,
        "loadtest": _loadtest(loadtest_id, outputs) if loadtest_id else None,
        "examples": examples,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(bundle, indent=2, default=str))
    return out_file
