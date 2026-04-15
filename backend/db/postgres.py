from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

import psycopg2
from psycopg2.extensions import connection as PGConnection


def _dsn() -> str:
    host = os.getenv("POSTGRES_HOST", "").strip()
    if not host:
        return ""
    port = os.getenv("POSTGRES_PORT", "5432").strip()
    db = os.getenv("POSTGRES_DB", "ecom_support").strip()
    user = os.getenv("POSTGRES_USER", "admin").strip()
    password = os.getenv("POSTGRES_PASSWORD", "").strip()
    return f"host={host} port={port} dbname={db} user={user} password={password}"


def postgres_configured() -> bool:
    return bool(_dsn())


@contextmanager
def get_connection() -> Generator[PGConnection, None, None]:
    dsn = _dsn()
    if not dsn:
        raise RuntimeError("Postgres is not configured (set POSTGRES_HOST, etc.)")
    conn = psycopg2.connect(dsn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
