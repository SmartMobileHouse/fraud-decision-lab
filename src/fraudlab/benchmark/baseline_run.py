"""Run the offline baselines (rules + XGBoost) on one dataset and save the results.

Writes outputs/<run_id>/ (gitignored):
- manifest.json     what was run: versions, dataset fingerprints, split, settings
- metrics.json      every metric, its 95% range, and the difference from XGBoost
- predictions.parquet  one score per holdout case and model (local only)
- labels.parquet    holdout labels, so metrics can be recomputed later (local only)
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost

from fraudlab import __version__
from fraudlab.baselines.rules import RULES_VERSION, score_rules
from fraudlab.baselines.xgb import XGB_VERSION, train_and_score
from fraudlab.config import OUTPUTS, REPO_ROOT
from fraudlab.datasets.leakage import check_leakage
from fraudlab.datasets.load import LoadedDataset
from fraudlab.datasets.payload import RENDER_VERSION, case_id
from fraudlab.datasets.split import DatasetSplit, split_dataset
from fraudlab.evaluate.bootstrap import DEFAULT_RESAMPLES, METRICS, paired_bootstrap
from fraudlab.evaluate.metrics import EVAL_VERSION, evaluate

REFERENCE = "xgboost"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.run(  # noqa: S603
                ["git", *args],  # noqa: S607 - git from PATH is intended
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        except OSError:  # no git installed (e.g. inside the Docker image)
            return ""

    return {"commit": git("rev-parse", "HEAD") or None, "uncommitted_changes": bool(git("status", "--porcelain"))}


def _manifest(run_id: str, ds: LoadedDataset, split: DatasetSplit, extra: dict[str, Any]) -> dict[str, Any]:
    spec = ds.config.dataset
    files = {str(spec.path): _file_sha256(ds.config.resolve(spec.path))}
    if spec.join is not None:
        files[str(spec.join.path)] = _file_sha256(ds.config.resolve(spec.join.path))
    return {
        "run_id": run_id,
        "kind": "baseline",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": _git_state(),
        "versions": {
            "fraudlab": __version__,
            "eval": EVAL_VERSION,
            "rules": RULES_VERSION,
            "xgb": XGB_VERSION,
            "render": RENDER_VERSION,
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",  # XGBoost can differ slightly across platforms
            "xgboost": xgboost.__version__,
        },
        "dataset": {
            "name": spec.name,
            "files_sha256": files,
            "license_note": spec.license_note,
            "external_processing_allowed": spec.external_processing_allowed,
            "publication_allowed": spec.publication_allowed,
            "publication_note": spec.publication_note,
            "rows": len(ds.frame),
            "features": ds.features,
        },
        "dataset_config": ds.config.model_dump(mode="json"),
        "split": {
            "method": split.method,
            "seed": ds.config.split.seed,
            "fractions": ds.config.split.model_dump(exclude={"seed"}),
            "holdout_sample": ds.config.holdout_sample,
            "fingerprints": split.hashes,
            "parts": {p.name: {"rows": p.rows, "frauds": p.frauds, "fraud_rate": p.fraud_rate} for p in split.parts()},
        },
        **extra,
    }


def run_baselines(
    ds: LoadedDataset, resamples: int = DEFAULT_RESAMPLES, outputs: Path = OUTPUTS
) -> tuple[Path, dict[str, Any]]:
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        from fraudlab.datasets.config import ConfigError

        raise ConfigError("resamples must be a positive integer")
    split = split_dataset(ds)
    check_leakage(ds, split)
    seed = ds.config.split.seed
    hold = split.holdout.index
    labels = ds.frame[ds.config.dataset.label_column].to_numpy()[hold]
    ids = [case_id(ds, int(i)) for i in hold]

    xgb_result = train_and_score(ds, split)
    rules_result = score_rules(ds, hold)

    scores: dict[str, np.ndarray] = {REFERENCE: xgb_result.scores}
    models: dict[str, dict[str, Any]] = {
        REFERENCE: {"status": "ok", "kind": "supervised", "trees": xgb_result.best_iteration + 1},
        "rules": {"status": rules_result.status, "kind": "checklist"},
    }
    if rules_result.status == "ok" and rules_result.scores is not None:
        scores["rules"] = rules_result.scores
    else:
        models["rules"]["reason"] = rules_result.reason

    for name, s in scores.items():
        models[name]["metrics"] = evaluate(labels, s, ids, seed=seed).as_dict()
    boot = paired_bootstrap(labels, scores, ids, reference=REFERENCE, resamples=resamples, seed=seed)
    for name in scores:
        models[name]["ranges_95"] = {m: asdict(boot.intervals[name][m]) for m in METRICS}
        if name in boot.differences:
            models[name][f"difference_vs_{REFERENCE}_95"] = {m: asdict(boot.differences[name][m]) for m in METRICS}

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{ds.config.dataset.name}-baseline"
    out = outputs / run_id
    out.mkdir(parents=True, exist_ok=False)

    metrics = {
        "run_id": run_id,
        "eval_version": EVAL_VERSION,
        "reference": REFERENCE,
        "bootstrap": {"resamples": resamples, "seed": seed, "method": "paired, stratified by class, percentile"},
        "models": models,
    }
    extra = {
        "xgboost": {"params": xgb_result.params, "trees": xgb_result.best_iteration + 1,
                    "train_seconds": round(xgb_result.train_seconds, 2)},
    }  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(_manifest(run_id, ds, split, extra), indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(
        [{"case_id": c, "model": name, "score": float(s[i])} for name, s in scores.items() for i, c in enumerate(ids)]
    ).to_parquet(out / "predictions.parquet", index=False)
    pd.DataFrame({"case_id": ids, "label": labels}).to_parquet(out / "labels.parquet", index=False)
    return out, metrics


def format_table(metrics: dict[str, Any]) -> str:
    """Plain-text results table."""

    def pct(v: float) -> str:
        return f"{v:6.1%}"

    lines = [
        f"{'model':<9} {'PR-AUC (95% range)':<22} {'recall@1%':>9} {'recall@5%':>9} {'prec@1%':>8}"
        f"  {'PR-AUC vs xgboost':<24}",
    ]
    for name, m in metrics["models"].items():
        if m["status"] != "ok":
            lines.append(f"{name:<9} {m['status']}: {m.get('reason', '')}")
            continue
        p, r = m["metrics"], m["ranges_95"]["pr_auc"]
        diff = m.get(f"difference_vs_{metrics['reference']}_95", {}).get("pr_auc")
        diff_text = f"{diff['low']:+.3f} to {diff['high']:+.3f}" if diff else "(reference)"
        lines.append(
            f"{name:<9} {p['pr_auc']:.3f} ({r['low']:.3f}-{r['high']:.3f})"
            f"{'':<4} {pct(p['recall_top_1pct']):>9} {pct(p['recall_top_5pct']):>9}"
            f" {pct(p['precision_top_1pct']):>8}  {diff_text}"
        )
    return "\n".join(lines)
