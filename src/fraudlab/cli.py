"""Command-line interface: `fraudlab <command>`.

Commands: `doctor`, `version`, `data ...`, `baseline`, `run`, `replay`, `probe`, `loadtest`, `report`.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import stat
import sys
from pathlib import Path

import typer

from fraudlab import __version__
from fraudlab.benchmark.baseline_run import format_table, run_baselines
from fraudlab.benchmark.loadtest import LoadTestRefused, loadtest_table, run_loadtest
from fraudlab.benchmark.replay import ReplayError, replay_run
from fraudlab.benchmark.runner import RunOptions, RunRefused, format_run_table, run_benchmark
from fraudlab.config import (
    ENV_FILE,
    EXPECTED_DATA_FILES,
    PROVIDER_ENV,
    load_env,
    secret_status,
)
from fraudlab.data_cli import data_app
from fraudlab.datasets.config import ConfigError, load_config
from fraudlab.datasets.load import load_dataset
from fraudlab.demo import run_demo
from fraudlab.evaluate.bootstrap import DEFAULT_RESAMPLES
from fraudlab.probe import ProbeRefused, probe_sync, probe_table
from fraudlab.providers.registry import LIVE_PROVIDERS
from fraudlab.report.export import export_report_data
from fraudlab.report.render import ReportError, build_report

app = typer.Typer(
    help="Fraud Decision Lab — benchmark AI decision models against XGBoost for fraud ranking.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(data_app, name="data")

# Display name -> importable module. Checked by import, not by distribution name,
# because XGBoost ships as `xgboost` on macOS and `xgboost-cpu` on Linux.
KEY_LIBRARIES = {
    "xgboost": "xgboost",
    "scikit-learn": "sklearn",
    "pandas": "pandas",
    "numpy": "numpy",
    "httpx": "httpx",
    "pydantic": "pydantic",
    "plotly": "plotly",
}
OK, WARN, FAIL = "ok  ", "warn", "FAIL"


def _line(status: str, label: str, detail: str = "") -> None:
    typer.echo(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))


@app.command()
def baseline(
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Dataset config, e.g. configs/datasets/synthetic.yaml"),
    resamples: int = typer.Option(DEFAULT_RESAMPLES, min=1, help="Bootstrap resamples for the 95% ranges"),
) -> None:
    """Run the offline baselines (rules + XGBoost) on one dataset and save the results. No network."""
    try:
        ds = load_dataset(load_config(dataset))
        out, metrics = run_baselines(ds, resamples=resamples)
    except ConfigError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    holdout = json.loads((out / "manifest.json").read_text())["split"]["parts"]["holdout"]
    typer.echo(f"dataset: {ds.config.dataset.name}   holdout: {holdout['rows']:,} cases, {holdout['frauds']:,} fraud")
    typer.echo(format_table(metrics))
    typer.echo(f"\nSaved to {out}")


@app.command()
def run(
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Dataset config, e.g. configs/datasets/synthetic.yaml"),
    providers: str = typer.Option(
        "rules,xgboost,mock", help="Comma-separated models: rules, xgboost, mock, " + ", ".join(LIVE_PROVIDERS)
    ),
    allow_external_requests: bool = typer.Option(False, help="Required for live providers (paid API calls)"),
    max_cost_usd: float | None = typer.Option(
        None, help="Estimated spending guard; actual billing can exceed token estimates"
    ),
    max_records: int | None = typer.Option(None, min=1, help="Score only a seeded sample of the holdout"),
    concurrency: int = typer.Option(8, min=1, max=64, help="Calls in flight per provider"),
    rate: float = typer.Option(4.0, min=0.1, help="Max new calls per second per provider"),
    resamples: int = typer.Option(DEFAULT_RESAMPLES, min=1, help="Bootstrap resamples"),
    resume: str | None = typer.Option(None, help="Run ID to resume (skips cases already answered)"),
) -> None:
    """Score a holdout; live calls use an estimated spending guard and verified resume identity."""
    opts = RunOptions(
        providers=[p.strip() for p in providers.split(",") if p.strip()],
        allow_external_requests=allow_external_requests, max_cost_usd=max_cost_usd, concurrency=concurrency,
        rate_per_s=rate, max_records=max_records, resamples=resamples, resume_run_id=resume, progress=typer.echo,
    )  # fmt: skip
    try:
        ds = load_dataset(load_config(dataset))
        out, metrics = run_benchmark(ds, opts)
    except (ConfigError, RunRefused) as exc:
        typer.echo(f"Refused: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    scored = f"{metrics['scored_cases']:,} cases, {metrics['frauds']:,} fraud"
    typer.echo(f"dataset: {ds.config.dataset.name}   scored: {scored}")
    typer.echo(format_run_table(metrics))
    b = metrics["budget"]
    if opts.max_cost_usd is not None:
        typer.echo(
            f"\naccounted ${b['spent_usd']:.4f}; guard ${b['cap_usd']:.2f} "
            f"(uncertain reservation ${b['estimated_usd']:.4f}; not an invoice)"
        )
        if b["refused"]:
            typer.echo(f"budget exhausted: {b['refused']:,} calls not sent — resume with a larger --max-cost-usd")
    typer.echo(f"saved to {out}")


@app.command()
def replay(run_id: str = typer.Option(..., "--run-id", help="Folder name under outputs/")) -> None:
    """Check saved prediction metrics offline; does not authenticate execution or verify billing."""
    try:
        saved, again, differences = replay_run(run_id)
    except ReplayError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(format_run_table(again))
    if differences:
        typer.echo(f"\nMISMATCH in: {', '.join(differences)}", err=True)
        raise typer.Exit(code=2)
    typer.echo("\nreplay: saved prediction metrics match (not independent provider or billing verification)")


@app.command()
def loadtest(
    providers: str = typer.Option(",".join(LIVE_PROVIDERS), help="Comma-separated provider names"),
    rates: str = typer.Option("1,5,20", help="Requests per second to test, in order"),
    seconds: float = typer.Option(60, min=1, help="Duration of each rate step"),
    max_cost_usd: float = typer.Option(..., help="Estimated spending guard (not a provider billing limit)"),
    allow_external_requests: bool = typer.Option(False, help="Required: confirms you want paid API calls"),
) -> None:
    """LIVE load test on synthetic cases (paid calls; estimated spending guard; no retries)."""
    if not allow_external_requests:
        typer.echo("Refused: live calls need --allow-external-requests.", err=True)
        raise typer.Exit(code=1)
    names = [p.strip() for p in providers.split(",") if p.strip()]
    try:
        rate_list = [float(r) for r in rates.split(",") if r.strip()]
    except ValueError as exc:
        typer.echo("Refused: rates must be comma-separated finite positive numbers.", err=True)
        raise typer.Exit(code=1) from exc
    try:
        out, results, budget = asyncio.run(run_loadtest(names, rate_list, seconds, max_cost_usd))
    except LoadTestRefused as exc:
        typer.echo(f"Refused: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(loadtest_table(results))
    typer.echo(
        f"\naccounted ${budget.state.spent_usd:.4f}; guard ${max_cost_usd:.2f} "
        f"(uncertain reservation ${budget.state.estimated_usd:.4f}; not an invoice)   saved to {out}"
    )


@app.command()
def report(
    run_id: list[str] = typer.Option(..., "--run-id", help="Run folder(s) under outputs/; repeat for several datasets"),
    loadtest_id: str | None = typer.Option(None, help="Load test folder under outputs/"),
    out: Path = typer.Option(Path("outputs/report.html"), help="Where to write the HTML file"),
    include_mock: bool = typer.Option(False, help="Also show the fake mock model"),
) -> None:
    """Build an offline HTML report for datasets explicitly approved for publication."""
    try:
        path = build_report(run_id, out, loadtest_id=loadtest_id, include_mock=include_mock)
    except ReportError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"report written to {path} ({path.stat().st_size / 1e6:.1f} MB)")


@app.command("export-report-data")
def export_report_data_cmd(
    run_id: list[str] = typer.Option(..., "--run-id", help="Run folder(s) under outputs/; repeat for several"),
    loadtest_id: str | None = typer.Option(None, help="Load test folder under outputs/"),
    out: Path = typer.Option(Path("outputs/report-data.json"), help="Where to write the JSON evidence bundle"),
) -> None:
    """Export a public-safe JSON evidence bundle for publishing tools. Refuses local-only datasets."""
    try:
        path = export_report_data(run_id, out, loadtest_id=loadtest_id)
    except ReportError as exc:
        typer.echo(f"Refused: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"evidence bundle written to {path} ({path.stat().st_size / 1e3:.0f} KB)")


@app.command()
def probe(
    providers: str = typer.Option(",".join(LIVE_PROVIDERS), help="Comma-separated provider names"),
    cases: int = typer.Option(3, min=1, max=20, help="Synthetic cases per provider"),
    max_cost_usd: float = typer.Option(..., help="Estimated spending guard (not a provider billing limit)"),
    allow_external_requests: bool = typer.Option(False, help="Required: confirms you want paid API calls"),
) -> None:
    """Tiny LIVE check on synthetic cases (paid API calls; estimated spending guard)."""
    if not allow_external_requests:
        typer.echo("Refused: live calls need --allow-external-requests.", err=True)
        raise typer.Exit(code=1)
    try:
        out, results = probe_sync([p.strip() for p in providers.split(",") if p.strip()], cases, max_cost_usd)
    except ProbeRefused as exc:
        typer.echo(f"Refused: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(probe_table(results))
    budget = json.loads((out / "probe.json").read_text())["budget"]
    typer.echo(
        f"\naccounted ${budget['spent_usd']:.6f}; guard ${budget['cap_usd']:.2f} "
        f"(uncertain reservation ${budget['estimated_usd']:.6f}; not an invoice)   saved to {out}"
    )
    if any(r.status != "ok" for r in results):
        raise typer.Exit(code=2)


@app.command()
def demo() -> None:
    """One-command offline demo: rules + XGBoost on the bundled synthetic data, then the HTML report."""
    typer.echo("Running rules and XGBoost on 10,000 synthetic transactions (offline, free)...")
    run_dir, report = run_demo()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    typer.echo(format_run_table(metrics))
    typer.echo(f"\nreport: {report}")
    typer.echo("Open it in a browser. To compare AI providers, see `fraudlab run --help` (needs API keys).")


@app.command()
def version() -> None:
    """Print the fraudlab version."""
    typer.echo(__version__)


@app.command()
def doctor() -> None:
    """Check the local setup. Offline: makes no network calls and never prints secrets."""
    failures = 0

    typer.echo("Python & libraries")
    py = sys.version_info
    if (py.major, py.minor) == (3, 12):
        _line(OK, f"Python {py.major}.{py.minor}.{py.micro}")
    else:
        _line(FAIL, f"Python {py.major}.{py.minor}.{py.micro}", "3.12 required")
        failures += 1
    _line(OK, f"fraudlab {__version__}")
    for label, module_name in KEY_LIBRARIES.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            _line(FAIL, label, f"cannot import ({exc.__class__.__name__}) — run `uv sync --frozen`")
            failures += 1
        else:
            _line(OK, f"{label} {getattr(module, '__version__', '?')}")

    typer.echo("Secrets (.env)")
    if load_env(ENV_FILE):
        mode = stat.S_IMODE(ENV_FILE.stat().st_mode)
        if mode & 0o077:
            _line(WARN, ".env found", f"permissions {oct(mode)}; run `chmod 600 .env`")
        else:
            _line(OK, ".env found", "readable only by you")
    else:
        _line(WARN, ".env not found", "copy .env.example to .env to use live providers")

    typer.echo("Providers (live runs only; offline runs need no keys)")
    for provider, names in PROVIDER_ENV.items():
        statuses = [secret_status(n) for n in names]
        missing = [s.name for s in statuses if not s.is_set]
        if missing:
            _line(WARN, provider, "missing " + ", ".join(missing))
        else:
            _line(OK, provider, "keys set (" + ", ".join(f"{s.name}: {s.length} chars" for s in statuses) + ")")

    typer.echo("Datasets (data/external, never committed)")
    for name, files in EXPECTED_DATA_FILES.items():
        missing_files = [f for f in files if not f.is_file()]
        if missing_files:
            _line(WARN, name, "missing " + ", ".join(f.name for f in missing_files))
        else:
            size_mb = sum(f.stat().st_size for f in files) / 1e6
            _line(OK, name, f"{len(files)} file(s), {size_mb:,.0f} MB")

    if failures:
        typer.echo(f"\n{failures} problem(s) must be fixed.")
        raise typer.Exit(code=1)
    typer.echo("\nReady. Offline runs work now; live providers marked warn need credentials before use.")
