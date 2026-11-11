"""`fraudlab demo`: the one-command offline tour.

Runs the free local models (rules + XGBoost) on the bundled synthetic data and
builds the HTML report. No API keys, no downloads, no network.
"""

from __future__ import annotations

from pathlib import Path

from fraudlab.benchmark.runner import RunOptions, run_benchmark
from fraudlab.config import OUTPUTS, REPO_ROOT
from fraudlab.datasets.config import load_config
from fraudlab.datasets.load import load_dataset
from fraudlab.report.render import build_report

SYNTHETIC_CONFIG = REPO_ROOT / "configs" / "datasets" / "synthetic.yaml"


def run_demo(outputs: Path | None = None, resamples: int = 1_000) -> tuple[Path, Path]:
    outputs = outputs or OUTPUTS
    ds = load_dataset(load_config(SYNTHETIC_CONFIG))
    run_dir, _ = run_benchmark(ds, RunOptions(providers=["rules", "xgboost"], resamples=resamples), outputs=outputs)
    report = build_report([run_dir.name], outputs / "demo-report.html", outputs=outputs)
    return run_dir, report
