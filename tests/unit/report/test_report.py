"""HTML report: complete, honest and safe to share."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudlab.benchmark import runner as runner_mod
from fraudlab.benchmark.runner import RunOptions, run_benchmark
from fraudlab.datasets.config import DatasetConfig, load_config
from fraudlab.datasets.load import load_dataset
from fraudlab.datasets.synthetic import FEATURES, generate_synthetic
from fraudlab.report.render import SYNTHETIC_CONFIG, ReportError, build_report

SYNTH_H2 = "<h2>Synthetic transactions (meaningful fields, simulated fraud)</h2>"


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str, str]:
    monkeypatch.setattr(runner_mod, "load_env", lambda: False)
    outputs = tmp_path / "o"
    synth = load_dataset(load_config(SYNTHETIC_CONFIG))
    s_out, _ = run_benchmark(synth, RunOptions(providers=["rules", "xgboost", "mock"], resamples=20, max_records=2_000),
                             outputs=outputs)  # fmt: skip
    p = tmp_path / "private.parquet"
    df = generate_synthetic(3_000, seed=9)
    df["tx_id"] = [f"SECRET-ROW-{i}" for i in range(len(df))]  # stands in for a restricted real dataset
    df.to_parquet(p)
    spec = {"name": "restricted-real", "path": str(p), "label_column": "is_fraud", "id_column": "tx_id",
            "time_column": "event_time", "features": FEATURES, "external_processing_allowed": False}  # fmt: skip
    r_out, _ = run_benchmark(load_dataset(DatasetConfig.model_validate({"dataset": spec})),
                             RunOptions(providers=["xgboost", "jev"], resamples=20), outputs=outputs)  # fmt: skip
    return outputs, s_out.name, r_out.name


def test_report_contains_every_section(runs: tuple[Path, str, str], tmp_path: Path) -> None:
    outputs, s, _ = runs
    html = build_report([s], tmp_path / "r.html", outputs=outputs).read_text()
    for text in (SYNTH_H2, "Limitations", "Example cases", "Cost comparison",
                 "Fraud caught vs. share of transactions reviewed", "fraud-decision-lab"):  # fmt: skip
        assert text in html


def test_restricted_results_are_refused_before_writing(runs: tuple[Path, str, str], tmp_path: Path) -> None:
    outputs, s, r = runs
    out = tmp_path / "r.html"
    with pytest.raises(ReportError, match="not approved for publication"):
        build_report([s, r], out, outputs=outputs)
    assert not out.exists()


def test_mock_hidden_by_default(runs: tuple[Path, str, str], tmp_path: Path) -> None:
    outputs, s, _ = runs
    assert "Mock (fake" not in build_report([s], tmp_path / "a.html", outputs=outputs).read_text()
    assert "Mock (fake" in build_report([s], tmp_path / "b.html", outputs=outputs, include_mock=True).read_text()


def test_examples_are_synthetic_only(runs: tuple[Path, str, str], tmp_path: Path) -> None:
    outputs, s, r = runs
    path = outputs / r / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["dataset"].update(publication_allowed=True, publication_note="Fictional test data approved for this test.")
    path.write_text(json.dumps(manifest))
    only_restricted = build_report([r], tmp_path / "r.html", outputs=outputs).read_text()
    assert "Example cases" not in only_restricted
    body = build_report([s], tmp_path / "s.html", outputs=outputs).read_text().split("</script>", 1)[1]
    assert re.findall(r"syn-\d{6}", body)


def test_report_needs_complete_run(runs: tuple[Path, str, str], tmp_path: Path) -> None:
    outputs, s, _ = runs
    (outputs / s / "labels.parquet").unlink()
    with pytest.raises(ReportError, match="labels.parquet"):
        build_report([s], tmp_path / "r.html", outputs=outputs)


def test_curves_are_downsampled() -> None:
    from fraudlab.report.render import _budget_curve, _pr_curve

    y = (pd.Series(range(5_000)) % 30 == 0).astype(int).to_numpy()
    s = pd.Series(range(5_000), dtype=float).to_numpy() / 5_000
    ids = [str(i) for i in range(len(y))]
    assert len(_pr_curve(y, s)[0]) == 101 and len(_budget_curve(y, s, ids, 42)[0]) <= 102


def test_tied_curve_matches_metrics_and_is_order_independent() -> None:
    from fraudlab.evaluate.metrics import evaluate
    from fraudlab.report.render import _budget_curve

    y = (np.arange(200) % 7 == 0).astype(int)
    scores = np.full(200, 0.5)
    ids = [f"case-{i}" for i in range(200)]
    metrics = evaluate(y, scores, ids, seed=19)
    x, curve = _budget_curve(y, scores, ids, seed=19)
    assert curve[x.index(1.0)] == pytest.approx(metrics.recall_top_1pct * 100)
    assert curve[x.index(5.0)] == pytest.approx(metrics.recall_top_5pct * 100)
    order = np.random.default_rng(4).permutation(len(y))
    assert (x, curve) == _budget_curve(y[order], scores[order], [ids[i] for i in order], seed=19)


def test_json_highlighting_is_escaped_and_coloured() -> None:
    from fraudlab.report.render import json_html

    html = str(json_html({"merchant": "<script>alert(1)</script>", "amount": 1.5, "known": True, "avg": None}))
    assert "<script>" not in html and "&lt;script&gt;" in html
    for css in ("j-k", "j-s", "j-d", "j-b", "j-n"):
        assert f'class="{css}"' in html


def test_saved_metric_validation_rejects_stale_metrics_and_duplicate_cases() -> None:
    from fraudlab.evaluate.metrics import evaluate
    from fraudlab.report.render import _validate_saved_metrics

    labels = pd.DataFrame({"case_id": ["a", "b", "c", "d"], "label": [1, 0, 1, 0]})
    scores = [0.9, 0.8, 0.7, 0.6]
    predictions = pd.DataFrame({"case_id": labels.case_id, "provider": "m", "status": "ok", "fraud_score": scores})
    point = evaluate(labels.label.to_numpy(), scores, labels.case_id.tolist()).as_dict()
    metrics = {"scored_cases": 4, "frauds": 2, "bootstrap": {"seed": 42},
               "models": {"m": {"status": "ok", "coverage": 1.0, "metrics": point}}}  # fmt: skip
    _validate_saved_metrics("run", metrics, predictions, labels)
    point["pr_auc"] = 0
    with pytest.raises(ReportError, match="saved pr_auc differs"):
        _validate_saved_metrics("run", metrics, predictions, labels)
    with pytest.raises(ReportError, match="duplicate predictions"):
        _validate_saved_metrics("run", metrics, pd.concat([predictions, predictions.iloc[:1]]), labels)


def test_report_is_unbranded_and_offline(runs: tuple[Path, str, str], tmp_path: Path) -> None:
    outputs, s, _ = runs
    html = build_report([s], tmp_path / "r.html", outputs=outputs).read_text()
    assert "fonts.googleapis" not in html and "fontshare" not in html  # nothing sent to font services
    assert "Key findings" in html
    brands = ("smartmobilehouse.com", "Satoshi", "AI benchmarking", "Report ©", 'aria-label="Smart Mobile House"')
    for brand in brands:
        assert brand not in html, brand


def test_publication_permission_is_separate_from_network_permission() -> None:
    from fraudlab.report.publication import require_publication

    manifest = {"dataset": {"name": "real", "external_processing_allowed": True}}
    with pytest.raises(ReportError, match="not approved for publication"):
        require_publication(manifest, "run")
    manifest["dataset"].update(external_processing_allowed=False, publication_allowed=True,
                               publication_note="Publication rights reviewed separately.")  # fmt: skip
    require_publication(manifest, "run")
    manifest["dataset"]["publication_allowed"] = False
    with pytest.raises(ReportError):
        require_publication(manifest, "run")


def test_dynamic_split_and_loadtest_evidence(runs: tuple[Path, str, str], tmp_path: Path) -> None:
    outputs, s, _ = runs
    manifest_path = outputs / s / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["split"]["fractions"].update(train=0.6, validation=0.2, holdout=0.2)
    manifest_path.write_text(json.dumps(manifest))
    load = outputs / "approved-load"
    load.mkdir()
    summary = {"dataset": manifest["dataset"], "seconds": 7,
               "budget": {"spent_usd": 0.01}, "steps": [{"provider": "jev", "target_rps": 3,
               "p50_ms": 10, "p95_ms": 20, "rate_limited": 0, "errors": 0, "timeouts": 0, "sent": 21}]}  # fmt: skip
    (load / "loadtest.json").write_text(json.dumps(summary))
    html = build_report([s], tmp_path / "dynamic.html", loadtest_id=load.name, outputs=outputs).read_text()
    assert "60/20/20" in html and "at 3 requests per second" in html
    assert "Aggregate-only historical evidence" in html
