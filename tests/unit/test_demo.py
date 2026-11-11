"""The offline demo runs end to end and writes a report."""

from __future__ import annotations

from pathlib import Path

import pytest

from fraudlab import demo
from fraudlab.benchmark import runner as runner_mod

SYNTH_H2 = "<h2>Synthetic transactions (meaningful fields, simulated fraud)</h2>"


def test_demo_runs_offline_and_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_mod, "load_env", lambda: False)
    run_dir, report = demo.run_demo(outputs=tmp_path, resamples=20)
    assert (run_dir / "metrics.json").is_file()
    html = report.read_text()
    assert SYNTH_H2 in html and "XGBoost (supervised)" in html and "Rules checklist" in html
