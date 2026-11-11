"""One publication gate for every report format, separate from API processing rights."""

from __future__ import annotations

import hashlib
from typing import Any

from fraudlab.config import REPO_ROOT
from fraudlab.report.errors import RestrictedDatasetError


def is_bundled_synthetic(manifest: dict[str, Any]) -> bool:
    """Recognize historical bundled synthetic runs by their input file fingerprint."""
    ds = manifest.get("dataset", {})
    if ds.get("name") != "synthetic-v1":
        return False
    bundled = REPO_ROOT / "data" / "demo" / "synthetic.parquet"
    expected = (ds.get("files_sha256") or {}).get("data/demo/synthetic.parquet")
    if not expected or not bundled.is_file():
        return False
    return hashlib.sha256(bundled.read_bytes()).hexdigest() == expected


def require_publication(manifest: dict[str, Any], run_id: str) -> None:
    """Refuse unapproved artifacts before any HTML or JSON output is written.

    Legacy runs have no publication policy. Only the fingerprinted original
    synthetic dataset can use its known project license as a migration path.
    Network-processing permission is never interpreted as publication consent.
    """
    ds = manifest.get("dataset", {})
    if ds.get("publication_allowed") is True and str(ds.get("publication_note", "")).strip():
        return
    if "publication_allowed" not in ds and is_bundled_synthetic(manifest):
        return
    raise RestrictedDatasetError(
        f"{run_id} uses dataset {ds.get('name')!r}, whose results are not approved for publication "
        "(publication_allowed and a publication_note are required; local only until rights are reviewed)"
    )
