"""`fraudlab report`: build one self-contained HTML page from finished runs.

Public-safety rules:
- only aggregate numbers and curves (downsampled to ~100 points) for real datasets;
- example transactions come ONLY from the bundled synthetic dataset;
- statuses such as not_run / permission_blocked are shown as text, never numbers;
- the mock model is excluded unless explicitly requested.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup, escape
from plotly.offline import get_plotlyjs

from fraudlab import __version__
from fraudlab.benchmark.ledger import Ledger
from fraudlab.benchmark.provenance import file_hash
from fraudlab.config import OUTPUTS, REPO_ROOT
from fraudlab.evaluate.metrics import MetricError, evaluate, ranking, tie_break_keys
from fraudlab.report.errors import ReportError
from fraudlab.report.publication import is_bundled_synthetic, require_publication

TEMPLATES = Path(__file__).parent / "templates"

GITHUB_URL = "https://github.com/SmartMobileHouse/fraud-decision-lab"
# Plain-words titles and one line on what each dataset tests (keyed by dataset config name).
DATASET_INFO = {
    "synthetic-v1": (
        "Synthetic transactions (meaningful fields, simulated fraud)",
        "Readable fields such as amount, device and country, so AI models can reason about them; "
        "fraud labels are simulated, so this shows behaviour, not real-world accuracy.",
    ),
    "ulb-creditcard": (
        "Real card fraud, anonymised fields (ULB)",
        "Real fraud outcomes with anonymised numerical fields; their lack of semantic descriptions "
        "limits this comparison of supervised learning and zero-shot scoring.",
    ),
    "ieee-cis": (
        "Real payments, local only (IEEE-CIS)",
        "Real payments with readable fields; competition rules forbid sending them to AI providers, "
        "so only the local models run.",
    ),
    "ieee-cis-full": (
        "Real payments, all fields, local only (IEEE-CIS)",
        "All 431 fields for a stronger local XGBoost; never sent to AI providers.",
    ),
}
LOCAL_MODELS = ("xgboost", "rules")
SYNTHETIC_NAME = "synthetic-v1"
SYNTHETIC_CONFIG = REPO_ROOT / "configs" / "datasets" / "synthetic.yaml"
# Colour-blind-safe palette (Okabe–Ito based), readable on light and dark backgrounds.
MODEL_COLORS = {
    "xgboost": "#0072B2",
    "rules": "#7F7F7F",
    "jev": "#CC79A7",
    "openai_decisions": "#D55E00",
    "clef": "#E69F00",
    "clef_flash": "#009E73",
    "llm": "#56B4E9",
    "mock": "#BBBBBB",
}
FALLBACK_COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]


def _color(name: str, i: int = 0) -> str:
    return MODEL_COLORS.get(name, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])


MODEL_LABELS = {
    "xgboost": "XGBoost (supervised)",
    "rules": "Rules checklist",
    "jev": "TypeSafe Jev",
    "openai_decisions": "OpenAI Decisions",
    "clef": "Cloudflare Clef",
    "clef_flash": "Cloudflare Clef-flash",
    "llm": "GPT-6-luna (JSON)",
    "mock": "Mock (fake, for testing)",
}


SHORT_LABELS = {"xgboost": "XGBoost", "rules": "Rules", "jev": "Jev", "openai_decisions": "Decisions",
                "clef": "Clef", "clef_flash": "Clef-flash", "llm": "LLM", "mock": "Mock"}  # fmt: skip


@dataclass
class RunData:
    run_id: str
    manifest: dict[str, Any]
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    labels: pd.DataFrame


def _load(run_id: str, outputs: Path) -> RunData:
    out = outputs / run_id
    for name in ("manifest.json", "metrics.json", "predictions.parquet", "labels.parquet"):
        if not (out / name).is_file():
            raise ReportError(f"{out} is missing {name}")
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest.get("kind") != "run":
        raise ReportError(f"{run_id} was not made by `fraudlab run`")
    if manifest.get("status", "complete") != "complete":
        raise ReportError(f"{run_id} is not a completed run")
    require_publication(manifest, run_id)
    for name in ("metrics.json", "predictions.parquet", "labels.parquet", "attempts.jsonl"):
        expected = manifest.get("artifacts_sha256", {}).get(name)
        if expected and (not (out / name).is_file() or file_hash(out / name) != expected):
            raise ReportError(f"{run_id}: saved artifact fingerprint differs for {name}")
    metrics = json.loads((out / "metrics.json").read_text())
    predictions = pd.read_parquet(out / "predictions.parquet")
    labels = pd.read_parquet(out / "labels.parquet")
    _validate_saved_metrics(run_id, metrics, predictions, labels)
    if (out / "attempts.jsonl").is_file():
        # Reconcile historical summaries without replacing the original saved metrics.
        accounting = Ledger(out / "attempts.jsonl").summary().by_provider
        for name, operational in accounting.items():
            if name not in metrics["models"]:
                continue
            entry = metrics["models"][name]
            operational["first_pass_coverage"] = operational["first_pass_ok"] / metrics["scored_cases"]
            operational["eventual_coverage"] = entry["coverage"]
            entry.update(
                operational=operational,
                cost_usd=operational["cost_accounted_usd"],
                known_cost_usd=operational["known_cost_usd"],
                estimated_cost_usd=operational["estimated_cost_usd"],
                cost_per_1k_usd=operational["cost_accounted_usd"] / metrics["scored_cases"] * 1_000,
                cost_incomplete=bool(operational["unknown_cost_calls"] or not operational["http_attempts_complete"]),
            )
            caught = entry.get("metrics", {}).get("frauds_caught_top_1pct")
            entry["cost_per_fraud_caught_top_1pct_usd"] = operational["cost_accounted_usd"] / caught if caught else None
    return RunData(run_id, manifest, metrics, predictions, labels)


def _validate_saved_metrics(
    run_id: str, metrics: dict[str, Any], predictions: pd.DataFrame, labels: pd.DataFrame
) -> None:
    """Validate point metrics without repeating bootstrap or any provider execution."""
    if labels["case_id"].isna().any() or labels["case_id"].astype(str).duplicated().any():
        raise ReportError(f"{run_id}: labels must have unique nonmissing case IDs")
    if predictions[["provider", "case_id"]].isna().any().any():
        raise ReportError(f"{run_id}: predictions have missing model or case IDs")
    if predictions.duplicated(["provider", "case_id"]).any():
        raise ReportError(f"{run_id}: duplicate predictions for a model and case")
    ids = labels["case_id"].astype(str).tolist()
    if not set(labels["label"].unique()) <= {0, 1}:
        raise ReportError(f"{run_id}: saved labels must be binary")
    if metrics["scored_cases"] != len(labels) or metrics["frauds"] != int(labels["label"].sum()):
        raise ReportError(f"{run_id}: saved metric population differs from labels")
    if set(predictions["provider"]) != set(metrics["models"]):
        raise ReportError(f"{run_id}: saved model list differs from predictions")
    for name, entry in metrics["models"].items():
        rows = predictions[predictions["provider"] == name].copy()
        rows["case_id"] = rows["case_id"].astype(str)
        if set(rows["case_id"]) != set(ids):
            raise ReportError(f"{run_id}: predictions for {name} do not cover the label population")
        ok = int((rows["status"] == "ok").sum())
        if not math.isclose(entry["coverage"], ok / len(ids), abs_tol=1e-15):
            raise ReportError(f"{run_id}: saved coverage differs for {name}")
        if not ok:
            if entry["status"] == "ok":
                raise ReportError(f"{run_id}: model {name} has no successful predictions")
            continue
        if entry["status"] != "ok":
            raise ReportError(f"{run_id}: saved model status differs for {name}")
        scores = rows.set_index("case_id").loc[ids, "fraud_score"].to_numpy(dtype=float)
        try:
            again = evaluate(labels["label"].to_numpy(), scores, ids, seed=metrics["bootstrap"]["seed"])
        except MetricError as exc:
            raise ReportError(f"{run_id}: saved predictions for {name} cannot be evaluated") from exc
        for metric, value in again.as_dict().items():
            recorded = entry.get("metrics", {}).get(metric)
            if recorded is None or not math.isclose(value, recorded, rel_tol=1e-12, abs_tol=1e-15):
                raise ReportError(f"{run_id}: saved {metric} differs from predictions for {name}; run replay")


def _scores(run: RunData, model: str) -> tuple[np.ndarray, np.ndarray]:
    preds = run.predictions[run.predictions.provider == model].set_index("case_id")
    y = run.labels["label"].to_numpy()
    s = preds.loc[run.labels["case_id"], "fraud_score"].to_numpy(dtype=float)
    return y, np.where(np.isnan(s), -1.0, s)  # failures rank last, as in the metrics


def _budget_curve(y: np.ndarray, s: np.ndarray, case_ids: list[str], seed: int) -> tuple[list[float], list[float]]:
    """Share of all fraud caught when analysts review the top x% (x from 0.1% to 20%)."""
    order = ranking(s, tie_break_keys(case_ids, seed))
    caught = np.cumsum(y[order])
    xs = np.unique(np.r_[np.linspace(0.001, 0.2, 100), [0.01, 0.05]])
    ks = np.maximum(1, np.ceil(xs * len(y)).astype(int))
    return [float(x * 100) for x in xs], [float(caught[k - 1] / y.sum() * 100) for k in ks]


def _pr_curve(y: np.ndarray, s: np.ndarray) -> tuple[list[float], list[float]]:
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y, s)
    grid = np.linspace(0, 1, 101)  # downsample: aggregate curve only, no row-level detail
    best = [float(precision[recall >= r].max()) if (recall >= r).any() else 0.0 for r in grid]
    return [float(r) for r in grid], best


def _figure(traces: list[go.Scatter], title: str, x_title: str, y_title: str) -> str:
    fig = go.Figure(traces)
    fig.update_layout(  # neutral layout; the page script recolours text and grid for light/dark
        title={"text": title, "font": {"size": 16}}, xaxis_title=x_title, yaxis_title=y_title, template="none",
        height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif", "size": 13},
        margin={"l": 72, "r": 20, "t": 56, "b": 40}, legend={"orientation": "h", "x": 0, "y": -0.32, "yanchor": "top"},
        xaxis={"automargin": True, "title": {"text": x_title, "standoff": 12}},
        yaxis={"automargin": True, "title": {"text": y_title, "standoff": 10}},
    )  # fmt: skip
    fig.update_layout(height=470)
    for trace in fig.data:
        trace.update(line={"width": 2.5, **(trace.line.to_plotly_json() if trace.line else {})})
    return str(fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False}))


def json_html(value: Any) -> Markup:
    """Pretty-printed JSON with syntax-colour spans (keys, strings, numbers, booleans, null). Escaped and safe."""
    text = json.dumps(value, indent=2, ensure_ascii=False)
    token = re.compile(r'("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false)\b|\b(null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')

    def paint(m: re.Match[str]) -> str:
        if m.group(1) is not None:
            css = "k" if m.group(2) else "s"
            return f'<span class="j-{css}">{escape(m.group(1))}</span>{m.group(2) or ""}'
        if m.group(3):
            return f'<span class="j-b">{m.group(3)}</span>'
        if m.group(4):
            return '<span class="j-n">null</span>'
        return f'<span class="j-d">{m.group(5)}</span>'

    out, last = [], 0
    for m in token.finditer(text):
        out.append(str(escape(text[last : m.start()])))
        out.append(paint(m))
        last = m.end()
    out.append(str(escape(text[last:])))
    return Markup("".join(out))  # noqa: S704 - every piece above is escaped


def _bar_figure(labels: list[str], values: list[float], colors: list[str], title: str, y_title: str) -> str:
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=[f"${v:.4f}" for v in values],
                           textposition="outside", cliponaxis=False))  # fmt: skip
    fig.update_layout(
        title={"text": title, "font": {"size": 16}}, yaxis_title=y_title, template="none", height=380,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif", "size": 13},
        margin={"l": 64, "r": 20, "t": 56, "b": 80}, showlegend=False,
    )  # fmt: skip
    return str(fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False}))


def _dataset_section(run: RunData, include_mock: bool) -> dict[str, Any]:
    m, man = run.metrics, run.manifest
    models = []
    color = {}
    for i, (name, entry) in enumerate(m["models"].items()):
        if name == "mock" and not include_mock:
            continue
        color[name] = _color(name, i)
        row: dict[str, Any] = {"name": name, "label": MODEL_LABELS.get(name, name), "status": entry["status"],
                               "coverage": entry["coverage"], "reason": entry.get("reason", ""),
                               "color": color[name], "local": name in LOCAL_MODELS}  # fmt: skip
        operational = entry.get("operational") or {}
        if operational:
            operational = {
                "calls": 0,
                "http_attempts": 0,
                "http_attempts_complete": True,
                "statuses": {},
                "latency_all_ms": {"p50_ms": None, "p95_ms": None, "p99_ms": None},
                **operational,
            }
        cost = operational.get("cost_accounted_usd", entry.get("cost_usd", 0.0))
        row.update(
            cost=cost,
            cost1k=cost / m["scored_cases"] * 1_000,
            known_cost=operational.get("known_cost_usd", entry.get("known_cost_usd", cost)),
            estimated_cost=operational.get("estimated_cost_usd", entry.get("estimated_cost_usd", 0.0)),
            unknown_cost_calls=operational.get("unknown_cost_calls", 0),
            cost_incomplete=entry.get("cost_incomplete", name not in LOCAL_MODELS and not operational),
            operational=operational,
            cost_per_fraud=None,
        )
        if entry["status"] == "ok":
            p, r = entry["metrics"], entry["ranges_95"]
            diff = entry.get("difference_vs_xgboost_95", {}).get("pr_auc")
            row.update(
                pr_auc=p["pr_auc"], pr_lo=r["pr_auc"]["low"], pr_hi=r["pr_auc"]["high"],
                rec1=p["recall_top_1pct"], rec5=p["recall_top_5pct"], prec1=p["precision_top_1pct"],
                caught1=p["frauds_caught_top_1pct"], diff=diff,
                verdict=_verdict(diff) if diff else ("reference" if name == "xgboost" else ""),
                p95=entry["latency"]["p95_ms"],
                cost_per_fraud=cost / p["frauds_caught_top_1pct"] if p["frauds_caught_top_1pct"] else None,
            )  # fmt: skip
        models.append(row)

    budget_traces, pr_traces = [], []
    for name in color:
        if m["models"][name]["status"] != "ok":
            continue
        y, s = _scores(run, name)
        bx, by = _budget_curve(y, s, run.labels["case_id"].astype(str).tolist(), m["bootstrap"]["seed"])
        budget_traces.append(go.Scatter(x=bx, y=by, name=MODEL_LABELS.get(name, name), line={"color": color[name]}))
        rx, ry = _pr_curve(y, s)
        pr_traces.append(go.Scatter(x=rx, y=ry, name=MODEL_LABELS.get(name, name), line={"color": color[name]}))
    budget_traces.append(go.Scatter(x=[0.1, 20], y=[0.1, 20], name="Random guessing",
                                    line={"color": "#B0ADA4", "dash": "dot"}))  # fmt: skip

    # Cost comparison: every model that ran, local ones at $0 (they cost compute time, not API fees).
    priced = [r for r in models if r["status"] == "ok" or r["cost"] or r["unknown_cost_calls"]]
    cost_chart = _bar_figure(
        [SHORT_LABELS.get(r["name"], r["name"]) for r in priced],
        [r["cost1k"] for r in priced],
        [r["color"] for r in priced],
        "Accounted API cost per 1,000 transactions",
        "US dollars",
    )
    ds = man["dataset"]
    parts = man["split"]["parts"]
    title, tests = DATASET_INFO.get(ds["name"], (ds["name"], ""))
    return {
        "name": ds["name"],
        "title": title,
        "tests": tests,
        "anchor": re.sub(r"[^a-z0-9]+", "-", ds["name"].lower()).strip("-"),
        "show_status": any(r["status"] == "ok" and r["coverage"] < 1.0 for r in models),
        "run_id": run.run_id,
        "created": man["created_utc"],
        "scored": m["scored_cases"],
        "frauds": m["frauds"],
        "fraud_rate": m["frauds"] / m["scored_cases"],
        "license": ds["license_note"],
        "external": ds["external_processing_allowed"],
        "features": ds["features"],
        "split_method": man["split"]["method"],
        "parts": parts,
        "split_fractions": man["split"].get("fractions", {}),
        "models": models,
        "budget_chart": _figure(budget_traces, "Fraud caught vs. share of transactions reviewed",
                                "Transactions reviewed (top %)", "Fraud caught (%)"),
        "pr_chart": _figure(pr_traces, "Precision–recall (interpolated envelope)", "Recall", "Precision"),
        "cost_chart": cost_chart,
        "total_cost": sum(r.get("cost", 0.0) or 0.0 for r in priced),
        "known_cost": sum(r["known_cost"] for r in priced),
        "estimated_cost": sum(r["estimated_cost"] for r in priced),
        "cost_incomplete": any(r["cost_incomplete"] for r in priced),
        "operational_available": any(r["operational"] and not r["local"] for r in models),
        "versions": man["versions"],
        "question": man.get("run", {}).get("question_version"),
        "budget": m.get("budget"),
    }  # fmt: skip


FINDING_NAMES = {"synthetic-v1": "synthetic transactions", "ulb-creditcard": "real card fraud (ULB)"}


def _key_findings(sections: list[dict[str, Any]], loadtest: dict[str, Any] | None) -> list[str]:
    """Up to 4 plain sentences answering the headline question. Every number comes from the runs."""
    out: list[str] = []
    ai_sections = [s for s in sections if any(m["status"] == "ok" and not m["local"] for m in s["models"])]
    any_beat, has_ref = False, False
    for s in ai_sections[:2]:
        ok = [m for m in s["models"] if m["status"] == "ok"]
        best = max(ok, key=lambda m: m["pr_auc"])
        best_ai = max((m for m in ok if not m["local"]), key=lambda m: m["pr_auc"])
        any_beat |= any(m.get("verdict", "").startswith("better") for m in ok if not m["local"])
        has_ref |= any(m["name"] == "xgboost" for m in ok)
        name = FINDING_NAMES.get(s["name"], s["title"].split(" (")[0].lower())
        best_name = SHORT_LABELS.get(best["name"], best["label"])
        caught = best_ai["rec5"] * 100
        sentence = (
            f"On {name}, {best_name} ranked fraud best (PR-AUC {best['pr_auc']:.3f}); the best AI model, "
            f"{best_ai['label']}, reached {best_ai['pr_auc']:.3f} and caught {caught:.0f}% of all "
            "fraud in the top 5% of transactions"
        )
        if best["name"] != best_ai["name"]:
            sentence += f", against {best['rec5'] * 100:.0f}% for {best_name}"
        out.append(sentence + ".")
    if not ai_sections:  # e.g. the offline demo: local models only
        for s in [s for s in sections if any(m["status"] == "ok" for m in s["models"])][:2]:
            best = max((m for m in s["models"] if m["status"] == "ok"), key=lambda m: m["pr_auc"])
            name = FINDING_NAMES.get(s["name"], s["title"].split(" (")[0].lower())
            label = SHORT_LABELS.get(best["name"], best["label"])
            out.append(
                f"On {name}, {label} ranked fraud best "
                f"(PR-AUC {best['pr_auc']:.3f}, {best['rec5'] * 100:.0f}% of fraud caught in the top 5%)."
            )
        if out:
            out.append("No AI providers were run for this report.")
    if ai_sections and has_ref and not any_beat:
        out.append("No evaluated AI model showed higher AP than XGBoost in these runs (conditional 95% ranges).")
    extras = []
    if loadtest:
        top = max(loadtest["rates"])
        timed = [r for r in loadtest["rows"] if r["p95"].get(top) is not None]
        if timed:
            fastest = min(timed, key=lambda r: r["p95"][top])
            extras.append(f"lowest recorded successful-request p95 was {fastest['label']} "
                          f"({fastest['p95'][top]:.0f} ms at {top:g} requests per second)")  # fmt: skip
    priced = [m for s in ai_sections[:1] for m in s["models"]
              if m["status"] == "ok" and not m["local"]]  # fmt: skip
    if priced and all(not m["cost_incomplete"] for m in priced):
        cheapest = min(priced, key=lambda m: m["cost1k"])
        extras.append(f"lowest accounted API cost was {cheapest['label']} "
                      f"(${cheapest['cost1k']:.3f} per 1,000 transactions; usage-derived, not an invoice)")  # fmt: skip
    if extras:
        text = "; ".join(extras)
        out.append(text[0].upper() + text[1:] + ".")
    return out[:4]


def _verdict(diff: dict[str, float]) -> str:
    if diff["high"] < 0:
        return "worse than XGBoost"
    if diff["low"] > 0:
        return "better than XGBoost"
    return "no clear difference"


def _examples(run: RunData, include_mock: bool, count: int = 4) -> list[dict[str, Any]]:
    """A few synthetic transactions with every model's score and original answer."""
    from fraudlab.baselines.rules import explain
    from fraudlab.datasets.config import load_config
    from fraudlab.datasets.load import load_dataset
    from fraudlab.datasets.payload import render_case

    ds = load_dataset(load_config(SYNTHETIC_CONFIG))
    position = {str(c): i for i, c in enumerate(ds.frame[ds.config.dataset.id_column].astype(str))}
    # The scenario is never loaded as a model input; read it separately, for explanation only.
    scenario_of = pd.read_parquet(ds.config.resolve(ds.config.dataset.path), columns=["tx_id", "scenario"])
    scenario_of = dict(zip(scenario_of.tx_id.astype(str), scenario_of.scenario, strict=True))
    labels = run.labels.set_index("case_id")["label"]
    preds = run.predictions
    if not include_mock:
        preds = preds[preds.provider != "mock"]
    rng = np.random.default_rng(7)
    frauds = labels[labels == 1].index.to_numpy()
    legit = labels[labels == 0].index.to_numpy()
    chosen = list(rng.choice(frauds, size=min(count // 2, len(frauds)), replace=False)) + list(
        rng.choice(legit, size=min(count - count // 2, len(legit)), replace=False)
    )
    out = []
    for cid in chosen:
        pos = position[str(cid)]
        rows = preds[preds.case_id == cid]
        out.append({
            "case_id": cid,
            "label": "fraud" if labels[cid] == 1 else "legitimate",
            "scenario": str(scenario_of.get(str(cid), "?")),
            "fields": render_case(ds, pos),
            "rules_fired": explain(ds, pos),
            "answers": [
                {"model": MODEL_LABELS.get(r.provider, r.provider), "status": r.status, "color": _color(r.provider),
                 "score": None if r.fraud_score is None or math.isnan(r.fraud_score) else r.fraud_score,
                 "raw": r.raw_answer if isinstance(r.raw_answer, str) else None}
                for r in sorted(rows.itertuples(), key=lambda r: list(MODEL_LABELS).index(r.provider)
                                if r.provider in MODEL_LABELS else 99)
            ],
        })  # fmt: skip
    return out


def _loadtest(loadtest_id: str, outputs: Path) -> dict[str, Any]:
    path = outputs / loadtest_id / "loadtest.json"
    if not path.is_file():
        raise ReportError(f"{path} not found")
    data = json.loads(path.read_text())
    require_publication(data, loadtest_id)
    if data.get("state", "complete") != "complete":
        raise ReportError(f"{loadtest_id} is not a completed load test")
    steps = data["steps"]
    rates = sorted({float(st["target_rps"]) for st in steps})
    providers = list(dict.fromkeys(st["provider"] for st in steps))
    rows, traces = [], []
    for i, p in enumerate(providers):
        by_rate = {float(st["target_rps"]): st for st in steps if st["provider"] == p}
        rows.append({
            "label": MODEL_LABELS.get(p, p), "color": _color(p, i),
            "p50": {r: by_rate[r]["p50_ms"] for r in rates if r in by_rate},
            "p95": {r: by_rate[r]["p95_ms"] for r in rates if r in by_rate},
            "too_busy": sum(st["rate_limited"] for st in by_rate.values()),
            "errors": sum(st["errors"] + st["timeouts"] for st in by_rate.values()),
            "sent": sum(st["sent"] for st in by_rate.values()),
        })  # fmt: skip
        traces.append(go.Bar(name=MODEL_LABELS.get(p, p), x=[f"{r:g} / s" for r in rates],
                             y=[by_rate[r]["p95_ms"] if r in by_rate else None for r in rates],
                             marker_color=_color(p, i)))  # fmt: skip
    fig = go.Figure(traces)
    fig.update_layout(
        barmode="group", template="none", height=470, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title={"text": "Slowest 5% response time at each request rate (log scale)", "font": {"size": 16}},
        font={"family": "ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif", "size": 13},
        xaxis={"type": "category", "title": {"text": "Requests per second", "standoff": 12}, "automargin": True},
        yaxis={"type": "log", "title": {"text": "p95 latency (ms)", "standoff": 10}, "automargin": True,
               "tickvals": [100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000],
               "ticktext": ["100", "200", "300", "500", "1,000", "2,000", "3,000", "5,000", "10,000"]},
        margin={"l": 72, "r": 20, "t": 56, "b": 40}, legend={"orientation": "h", "x": 0, "y": -0.32, "yanchor": "top"},
    )  # fmt: skip
    chart = str(fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False}))
    return {"id": loadtest_id, "seconds": data["seconds"], "rates": rates, "rows": rows,
            "rates_text": ", ".join(f"{r:g}" for r in rates),
            "spent": data["budget"]["spent_usd"], "chart": chart,
            "aggregate_only": not (path.parent / "calls.parquet").is_file()}  # fmt: skip


def build_report(
    run_ids: list[str], out_file: Path, loadtest_id: str | None = None, include_mock: bool = False,
    outputs: Path | None = None,
) -> Path:  # fmt: skip
    outputs = outputs or OUTPUTS
    runs = [_load(r, outputs) for r in run_ids]
    sections = [_dataset_section(r, include_mock) for r in runs]
    load = _loadtest(loadtest_id, outputs) if loadtest_id else None
    synthetic = next((r for r in runs if is_bundled_synthetic(r.manifest)), None)
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html", "j2"]))
    env.filters["pct"] = lambda v, d=1: "—" if v is None else f"{v * 100:.{d}f}%"
    env.filters["json_html"] = json_html
    env.filters["signed"] = lambda v: f"{v:+.2f}".replace("-", "−")
    env.filters["num"] = lambda v, d=3: "—" if v is None else f"{v:.{d}f}"
    now = datetime.now(UTC)
    html = env.get_template("report.html.j2").render(
        title="Fraud Decision Lab — results",
        generated=now.strftime("%Y-%m-%d %H:%M UTC"),
        release_date=f"{now.day} {now.strftime('%b %Y')}",
        year=now.year,
        version=__version__,
        links={"github": GITHUB_URL},
        sections=sections,
        examples=_examples(synthetic, include_mock) if synthetic else [],
        loadtest=load,
        has_restricted=any(r.manifest["dataset"]["name"] in ("ieee-cis", "ieee-cis-full") for r in runs),
        findings=_key_findings(sections, load),
        plotly_js=get_plotlyjs(),
        include_mock=include_mock,
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html)
    return out_file
