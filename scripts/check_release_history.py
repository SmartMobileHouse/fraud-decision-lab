"""Flag result/data paths reachable in Git before a source-only publication."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RESULT_PREFIXES = ("outputs/", "data/external/", "data/processed/", "private/", "docs/images/")


def history_findings(root: Path) -> list[str]:
    result = subprocess.run(  # noqa: S603
        ["git", "log", "--all", "--format=", "--name-only"],  # noqa: S607
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = set(result.stdout.splitlines())
    findings = [p for p in paths if p.startswith(RESULT_PREFIXES)]
    findings += [p for p in paths if Path(p).name.startswith(".env") and Path(p).name != ".env.example"]
    return sorted(set(findings))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        print("No local Git history: inspect this source snapshot and its actual publication destination separately.")
        return 0
    findings = history_findings(root)
    if findings:
        print("FAIL: historical files require publication review even if absent from HEAD:")
        for path in findings:
            print(f"  {path}")
        print("Use RELEASE.md to prepare a separate source-only candidate. Original history is unchanged.")
        return 1
    print(
        "No flagged historical paths found locally. GitHub-only artifacts and publication rights remain separate gates."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
