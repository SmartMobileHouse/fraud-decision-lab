"""Shared settings: where things live and which credentials each provider needs.

Secrets are read from the environment (optionally loaded from a local `.env`).
Nothing in this module ever returns or logs a secret value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Repository root: src/fraudlab/config.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"
DATA_EXTERNAL = REPO_ROOT / "data" / "external"
OUTPUTS = REPO_ROOT / "outputs"

# Provider id -> environment variables it needs (see plan §5).
PROVIDER_ENV: dict[str, tuple[str, ...]] = {
    "jev": ("TYPESAFE_API_KEY",),
    "openai_decisions": ("OPENAI_API_KEY",),
    "llm": ("OPENAI_API_KEY",),
    "clef": ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"),
    "clef_flash": ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"),
}

# Dataset files the reference runs expect (user-downloaded, never committed).
EXPECTED_DATA_FILES: dict[str, tuple[Path, ...]] = {
    "ulb": (DATA_EXTERNAL / "ulb" / "creditcard.csv",),
    "ieee": (
        DATA_EXTERNAL / "ieee-cis" / "train_transaction.csv",
        DATA_EXTERNAL / "ieee-cis" / "train_identity.csv",
    ),
}


def load_env(env_file: Path = ENV_FILE) -> bool:
    """Load `.env` into the process environment without overriding real env vars.

    Returns True if the file existed.
    """
    if env_file.is_file():
        load_dotenv(env_file, override=False)
        return True
    return False


@dataclass(frozen=True)
class SecretStatus:
    """Presence of one secret. Holds only its name and length, never the value."""

    name: str
    is_set: bool
    length: int


def secret_status(name: str) -> SecretStatus:
    value = os.environ.get(name, "").strip()
    return SecretStatus(name=name, is_set=bool(value), length=len(value))
