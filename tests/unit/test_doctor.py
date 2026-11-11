"""`fraudlab doctor` reports setup state and never leaks secret values."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from fraudlab import cli

runner = CliRunner()
SECRET = "sk-test-THIS-MUST-NEVER-BE-PRINTED-1234567890"
ALL_KEYS = ("TYPESAFE_API_KEY", "OPENAI_API_KEY", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN")


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point doctor at an empty temp repo with no keys and no datasets."""
    monkeypatch.setattr(cli, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(cli, "EXPECTED_DATA_FILES", {"ulb": (tmp_path / "creditcard.csv",)})
    for key in ALL_KEYS:
        monkeypatch.delenv(key, raising=False)
    return tmp_path


def test_doctor_passes_without_keys_or_data(isolated: Path) -> None:
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert ".env not found" in result.output
    assert "missing TYPESAFE_API_KEY" in result.output
    assert "missing creditcard.csv" in result.output


def test_doctor_never_prints_secret_values(isolated: Path) -> None:
    env = isolated / ".env"
    env.write_text("".join(f"{k}={SECRET}\n" for k in ALL_KEYS))
    env.chmod(0o600)
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert SECRET not in result.output
    assert f"TYPESAFE_API_KEY: {len(SECRET)} chars" in result.output
    assert "readable only by you" in result.output


def test_doctor_warns_on_world_readable_env(isolated: Path) -> None:
    env = isolated / ".env"
    env.write_text("OPENAI_API_KEY=x\n")
    env.chmod(0o644)
    result = runner.invoke(cli.app, ["doctor"])
    assert "chmod 600" in result.output


def test_doctor_reports_present_dataset(isolated: Path) -> None:
    (isolated / "creditcard.csv").write_text("Time,Amount,Class\n0,1.0,0\n")
    result = runner.invoke(cli.app, ["doctor"])
    assert "[ok  ] ulb" in result.output


def test_version_matches_package() -> None:
    result = runner.invoke(cli.app, ["version"])
    assert result.output.strip() == cli.__version__


def test_doctor_fails_when_a_library_cannot_be_imported(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "KEY_LIBRARIES", {"xgboost": "xgboost", "missing-lib": "no_such_module_xyz"})
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "[FAIL] missing-lib" in result.output
