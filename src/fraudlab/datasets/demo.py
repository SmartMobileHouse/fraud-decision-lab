"""The bundled synthetic dataset: where it lives and how to check it is unchanged."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraudlab.config import REPO_ROOT
from fraudlab.datasets.synthetic import DEFAULT_ROWS, DEFAULT_SEED, generate_synthetic

DEMO_PATH = REPO_ROOT / "data" / "demo" / "synthetic.parquet"


def content_hash(df: pd.DataFrame) -> str:
    """Fingerprint of the table's contents (independent of file format details)."""
    return f"{int(pd.util.hash_pandas_object(df, index=False).sum()) & 0xFFFFFFFFFFFFFFFF:016x}"


def write_demo(path: Path = DEMO_PATH, rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    df = generate_synthetic(rows=rows, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    return df
