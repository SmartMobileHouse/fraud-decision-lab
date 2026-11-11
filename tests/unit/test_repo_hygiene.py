"""Guards on what may be committed: no secrets, datasets, outputs or plan files."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fraudlab.config import REPO_ROOT

FORBIDDEN_PREFIXES = ("data/external/", "data/processed/", "outputs/", "private/")
FORBIDDEN_NAMES = (".env", "CLAUDE.local.md")
FORBIDDEN_PATTERNS = ("IMPLEMENTATION_PLAN", "PLAN_REVIEW", "PROGRESS.md")
MAX_TRACKED_BYTES = 5_000_000  # anything bigger is probably data


def _tracked_files() -> list[str]:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(  # noqa: S603
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],  # noqa: S607
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def test_no_forbidden_files_would_be_committed() -> None:
    bad = [
        f
        for f in _tracked_files()
        if f.startswith(FORBIDDEN_PREFIXES)
        or Path(f).name in FORBIDDEN_NAMES
        or any(p in f for p in FORBIDDEN_PATTERNS)
    ]
    assert not bad, f"These must not be committed: {bad}"


def test_no_large_files_would_be_committed() -> None:
    big = [
        f for f in _tracked_files() if (REPO_ROOT / f).is_file() and (REPO_ROOT / f).stat().st_size > MAX_TRACKED_BYTES
    ]
    assert not big, f"Unexpectedly large files (data?): {big}"


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        "CLAUDE.local.md",
        "data/external/x.csv",
        "outputs/run/metrics.json",
        "IMPLEMENTATION_PLAN_V0.3.md",
    ],
)
def test_gitignore_covers(path: str) -> None:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO_ROOT, check=False)  # noqa: S603, S607
    assert result.returncode == 0, f"{path} is not gitignored"


def test_public_files_do_not_reveal_private_plan() -> None:
    """Committed files must not point to the private plan folder or progress log."""
    leaks = []
    for f in _tracked_files():
        path = REPO_ROOT / f
        if f.endswith("test_repo_hygiene.py") or not path.is_file() or path.stat().st_size > 1_000_000:
            continue
        text = path.read_text(errors="ignore")
        if "fraud-decision-lab-plan" in text or "PROGRESS.md" in text or "/Volumes/" in text:
            leaks.append(f)
    assert not leaks, f"Private paths found in committed files: {leaks}"
