"""Load repository `.env` then optional `.env.local` for host-side scripts."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def _running_in_docker() -> bool:
    return Path("/.dockerenv").exists()


def load_repo_dotenv(repo_root: Path | None = None) -> None:
    """Load ``<repo>/.env`` then ``<repo>/.env.local`` (override), except in Docker.

    - Preserves variables already set in the process environment for ``.env``
      (``override=False``).
    - Applies ``.env.local`` on the host only with ``override=True`` so local
      URLs override docker-oriented values from ``.env``.
    - Skips ``.env.local`` inside a container so bind mounts (e.g. simulator)
      do not apply host-only overrides.
    """
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
    if _running_in_docker():
        return
    local_path = root / ".env.local"
    if local_path.is_file():
        load_dotenv(local_path, override=True)
