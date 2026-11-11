"""Local experiment identity and atomic checkpoints; no provider requests."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from fraudlab.config import REPO_ROOT


def json_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash() -> str:
    """Bind even uncommitted executable code and the dependency lock to the run."""
    paths = sorted((REPO_ROOT / "src" / "fraudlab").rglob("*.py"))
    paths += [REPO_ROOT / "uv.lock"]
    return json_hash({str(p.relative_to(REPO_ROOT)): file_hash(p) for p in paths})


def frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps([(c, str(frame[c].dtype)) for c in frame.columns]).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


@contextmanager
def run_lock(out: Path):
    """Prevent concurrent resumes; the OS releases the lock after an interruption."""
    import fcntl

    with (out / ".run.lock").open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("this run is already being executed by another process") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
