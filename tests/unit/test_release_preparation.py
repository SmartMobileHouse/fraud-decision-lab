"""Protect source-only release boundaries without changing Git history or using network."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _script(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_excludes_private_data_and_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _script("prepare_public_source")
    source = tmp_path / "source"
    paths = ["README.md", ".env", "data/external/records.csv", "outputs/results.json", "docs/images/results.png"]
    for relative in paths:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    monkeypatch.setattr(script, "_git", lambda _root, *args: "\0".join(paths) if args[0] == "ls-files" else "fixture")
    monkeypatch.setattr(script.shutil, "which", lambda _: "gitleaks")
    monkeypatch.setattr(script.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))
    destination = tmp_path / "candidate"
    assert script.prepare(source, destination) == 0
    assert sorted(str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()) == [
        "README.md",
        "SOURCE_SNAPSHOT.json",
    ]
    assert all((source / p).exists() for p in paths)


def test_snapshot_refuses_overwrite_and_nested_destination(tmp_path: Path) -> None:
    script = _script("prepare_public_source")
    root = tmp_path / "source"
    root.mkdir()
    with pytest.raises(ValueError, match="outside"):
        script.prepare(root, root / "candidate")
    with pytest.raises(ValueError, match="already exists"):
        script.prepare(root, tmp_path)


def test_snapshot_refuses_symlink_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _script("prepare_public_source")
    root = tmp_path / "source"
    path = root / "src/fraudlab/example.py"
    path.parent.mkdir(parents=True)
    target = tmp_path / "private.py"
    target.write_text("fixture")
    path.symlink_to(target)
    monkeypatch.setattr(script, "_git", lambda *_args: "src/fraudlab/example.py")
    monkeypatch.setattr(script.shutil, "which", lambda _: "gitleaks")
    destination = tmp_path / "candidate"
    with pytest.raises(ValueError, match="symlink"):
        script.prepare(root, destination)
    assert not destination.exists()


def test_history_gate_catches_deleted_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _script("check_release_history")
    monkeypatch.setattr(
        script.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="README.md\ndocs/images/results.png\n"),
    )
    assert script.history_findings(tmp_path) == ["docs/images/results.png"]
