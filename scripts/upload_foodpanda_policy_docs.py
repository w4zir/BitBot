#!/usr/bin/env python3
"""Upload Foodpanda policy Markdown files to Elasticsearch via the bulk API.

Run from the repository root. Uses only the Python standard library.

Environment (defaults in parentheses; align with `.env.example`):
  ES_HOST (localhost), ES_PORT (9200), ES_SCHEME (http),
  ES_POLICY_INDEX (policy_docs), ES_TIMEOUT_SECONDS (60 for this script).

Examples:
  python scripts/upload_foodpanda_policy_docs.py --dry-run
  python scripts/upload_foodpanda_policy_docs.py --create-index --host localhost
  ES_HOST=elasticsearch python scripts/upload_foodpanda_policy_docs.py --create-index
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_SOURCE = Path("data/policy_docs/foodpanda/policy_docs")


def _env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    if v is None or not str(v).strip():
        return default
    return str(v).strip()


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip(), 10)
    except ValueError:
        return default


def parse_markdown_doc(path: Path) -> tuple[str, str, list[str], str]:
    """Return (title, content, tags, doc_id)."""
    raw = path.read_text(encoding="utf-8")
    m = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
    title = (m.group(1).strip() if m else path.stem.replace("_", " "))
    doc_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", path.stem.lower()).strip("-")
    tags = ["foodpanda", "policy"]
    tags.extend(
        w.lower()
        for w in re.split(r"[_\s]+", path.stem)
        if w.isalpha() and len(w) > 2
    )
    content = raw.strip()
    return title, content, tags, doc_id


def build_bulk_ndjson(paths: list[Path]) -> bytes:
    """Pairs of NDJSON lines; UTF-8; trailing newline."""
    lines: list[str] = []
    for path in paths:
        title, content, tags, doc_id = parse_markdown_doc(path)
        # When posting to /{index}/_bulk, omit _index in the action line.
        action = {"index": {"_id": doc_id}}
        body = {"title": title, "content": content, "tags": tags}
        lines.append(json.dumps(action, ensure_ascii=False))
        lines.append(json.dumps(body, ensure_ascii=False))
    text = "\n".join(lines) + ("\n" if lines else "")
    return text.encode("utf-8")


def http_request(
    method: str,
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    req = Request(url, data=data, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 — intentional ES URL
            body = resp.read().decode("utf-8")
            code = getattr(resp, "status", 200)
            try:
                parsed: dict[str, Any] = json.loads(body) if body else {}
            except json.JSONDecodeError:
                parsed = {}
            return int(code), parsed
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            parsed_err: dict[str, Any] = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed_err = {}
        return int(e.code), parsed_err


def create_index(base: str, index: str, timeout: float) -> None:
    url = f"{base}/{index}"
    code, data = http_request(
        "PUT",
        url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    if code in (200, 201):
        if data.get("acknowledged") is False:
            raise RuntimeError(f"Create index not acknowledged: {data}")
        return
    if code == 400:
        err = data.get("error")
        if isinstance(err, dict) and err.get("type") == "resource_already_exists_exception":
            print(f"Index {index!r} already exists; continuing.")
            return
    raise RuntimeError(f"Create index failed: HTTP {code} {data}")


def post_bulk(base: str, index: str, ndjson: bytes, timeout: float) -> dict[str, Any]:
    url = f"{base}/{index}/_bulk"
    code, data = http_request(
        "POST",
        url,
        data=ndjson,
        headers={"Content-Type": "application/x-ndjson"},
        timeout=timeout,
    )
    if code not in (200, 201):
        raise RuntimeError(f"Bulk request failed: HTTP {code} {data}")
    return data


def summarize_bulk_errors(data: dict[str, Any]) -> list[str]:
    msgs: list[str] = []
    items = data.get("items") or []
    for i, item in enumerate(items):
        for action, result in item.items():
            if not isinstance(result, dict):
                continue
            status = result.get("status", 0)
            if status >= 300:
                err = result.get("error")
                msgs.append(f"item[{i}] {action} status={status} error={err}")
    return msgs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload Foodpanda policy Markdown to Elasticsearch (bulk API)."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Folder with *.md policies (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--index",
        default=_env_str("ES_POLICY_INDEX", "policy_docs"),
        help="Index name (default: ES_POLICY_INDEX or policy_docs)",
    )
    parser.add_argument(
        "--host",
        default=_env_str("ES_HOST", "localhost"),
        help="Elasticsearch host (default: ES_HOST or localhost)",
    )
    parser.add_argument(
        "--port",
        default=_env_str("ES_PORT", "9200"),
        help="Elasticsearch port (default: ES_PORT or 9200)",
    )
    parser.add_argument(
        "--scheme",
        default=_env_str("ES_SCHEME", "http"),
        help="URL scheme (default: ES_SCHEME or http)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(_env_int("ES_TIMEOUT_SECONDS", 60)),
        help="HTTP timeout in seconds (default: ES_TIMEOUT_SECONDS or 60)",
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="PUT the index with an empty body before bulk (idempotent if exists).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files and print summary; do not call Elasticsearch.",
    )
    args = parser.parse_args()

    source_dir: Path = args.source_dir
    if not source_dir.is_dir():
        print(f"error: source directory not found: {source_dir}", file=sys.stderr)
        return 1

    paths = sorted(source_dir.glob("*.md"))
    if not paths:
        print(f"error: no .md files under {source_dir}", file=sys.stderr)
        return 1

    scheme = args.scheme or "http"
    host = args.host
    port = args.port
    index = args.index or "policy_docs"
    base = f"{scheme}://{host}:{port}"

    if args.dry_run:
        print(f"dry-run: would index {len(paths)} document(s) into {index!r}")
        print(f"base URL: {base}")
        for p in paths[:5]:
            _, _, _, doc_id = parse_markdown_doc(p)
            print(f"  - {p.name} -> _id={doc_id}")
        if len(paths) > 5:
            print(f"  ... and {len(paths) - 5} more")
        return 0

    ndjson = build_bulk_ndjson(paths)

    try:
        if args.create_index:
            print(f"Creating index {index!r} ...")
            create_index(base, index, args.timeout)
            print("Index ready.")

        print(f"POST {base}/{index}/_bulk ({len(ndjson)} bytes) ...")
        data = post_bulk(base, index, ndjson, args.timeout)
    except URLError as e:
        print(f"error: connection failed: {e.reason}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if data.get("errors"):
        print("Bulk completed with errors.", file=sys.stderr)
        for line in summarize_bulk_errors(data):
            print(line, file=sys.stderr)
        return 1

    took = data.get("took")
    items = data.get("items") or []
    print(f"ok: indexed {len(items)} item(s), took_ms={took}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
