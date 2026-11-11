"""Prepare a local source-only candidate without carrying private Git history."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT_FILES = {
    ".dockerignore",
    ".env.example",
    ".gitignore",
    ".python-version",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    "DATA_LICENSES.md",
    "Dockerfile",
    "LICENSE",
    "Makefile",
    "README.md",
    "RELEASE.md",
    "SECURITY.md",
    "pyproject.toml",
    "uv.lock",
}


def allowed(path: str) -> bool:
    """Closed file-type/path allowlist; no outputs, third-party records or history."""
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts:
        return False
    if path in ROOT_FILES or path == "data/demo/synthetic.parquet":
        return True
    if path.startswith(("src/fraudlab/", "tests/", "scripts/")):
        return p.suffix in {".py", ".j2", ".html", ".css", ".svg"}
    if path.startswith("configs/datasets/"):
        return p.suffix == ".yaml"
    if len(p.parts) == 2 and p.parts[0] == "configs":
        return (p.name.startswith("prices-") and p.suffix == ".yaml") or (
            p.name.startswith("questions-") and p.suffix == ".json"
        )
    if path.startswith(".github/"):
        return p.suffix in {".md", ".yml", ".yaml"} or p.name == "CODEOWNERS"
    return False


def _git(root: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607 - intended local Git executable
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def prepare(root: Path, destination: Path) -> int:
    root, destination = root.resolve(), destination.expanduser().resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("destination must be outside the source repository")
    if destination.exists():
        raise ValueError("destination already exists; choose a new directory")
    scanner = shutil.which("gitleaks")
    if scanner is None:
        raise ValueError("gitleaks must already be installed; no tools will be downloaded")
    # New unignored source files are included so reviewed fixes need not be committed first.
    paths = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")
    selected = sorted({p for p in paths if p and allowed(p)})
    files: dict[str, str] = {}
    for relative in selected:
        source = root / relative
        if any(p.is_symlink() for p in (source, *source.parents) if p != root):
            raise ValueError(f"symlinks are not included: {relative}")
        if not source.is_file() or not source.resolve().is_relative_to(root):
            raise ValueError(f"allowlisted source is missing or outside the project: {relative}")
    destination.mkdir(parents=True, exist_ok=False)
    for relative in selected:
        data = (root / relative).read_bytes()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files[relative] = hashlib.sha256(data).hexdigest()
    snapshot = {
        "kind": "source-only-review-candidate",
        "publication_approved": False,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_commit": _git(root, "rev-parse", "HEAD"),
        "working_changes_present": bool(_git(root, "status", "--porcelain")),
        "files_sha256": files,
        "limitations": "No Git history, benchmark results, external data or legal/publication approval is included.",
    }
    (destination / "SOURCE_SNAPSHOT.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    result = subprocess.run(  # noqa: S603
        [scanner, "dir", "--no-banner", "--redact", str(destination)],
        check=False,
    )
    print(f"Prepared {len(files)} allowlisted files at {destination}; no Git history copied.")
    if result.returncode:
        print("NOT APPROVED: secret scan failed; candidate retained privately for inspection.")
        return result.returncode
    print("Secret scan passed. Human rights, artifact, environment and GitHub review is still required (RELEASE.md).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New destination outside the source repository")
    args = parser.parse_args()
    try:
        return prepare(Path(__file__).resolve().parents[1], args.out)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Refused: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
