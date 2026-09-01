"""Loads CANDIDATE_KEY from the environment or a local .env file.

No third-party .env loader is used (the sandbox this was built in has no pip),
so this is a ~10-line stand-in: it only ever reads a file named `.env` sitting
next to the repo root, and never overrides a variable already set in the real
environment.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_URL = "https://ml-interview.fly.dev"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = _REPO_ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def get_candidate_key(cli_value: str | None = None) -> str:
    key = cli_value or os.environ.get("CANDIDATE_KEY")
    if not key:
        raise SystemExit(
            "No candidate key found. Set CANDIDATE_KEY in a .env file "
            "(see .env.example), export it, or pass --candidate-key."
        )
    return key
