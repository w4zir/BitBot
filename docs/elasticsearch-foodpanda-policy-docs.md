# Index Foodpanda policy Markdown in Elasticsearch

This guide shows how to load the Markdown files under [`data/policy_docs/foodpanda/policy_docs/`](../data/policy_docs/foodpanda/policy_docs/) into Elasticsearch for BitBot policy retrieval.

## What Elasticsearch must store

The backend retriever (`backend/rag/policy_retriever.py`) runs a `multi_match` query on these fields:

| Field | Role |
|--------|------|
| `title` | Searchable; boosted in queries (`title^2`). |
| `content` | Main body text (your Markdown or plain text). |
| `tags` | Optional keywords (string or array of strings, depending on mapping); included in search. |

Environment variables (see `.env.example`): `ES_HOST`, `ES_PORT`, `ES_SCHEME`, **`ES_POLICY_INDEX`** (default `policy_docs`), `ES_TIMEOUT_SECONDS`. Use the same index name for uploads as in `ES_POLICY_INDEX`.

## Source files

The Foodpanda sample policies are seven Markdown files:

- `01_Returns_And_Refund_Policy.md`
- `02_Shipping_And_Logistics_Policy.md`
- `03_Subscription_Policy.md`
- `04_Loyalty_Program_Policy.md`
- `05_Damaged_Items_Policy.md`
- `06_Fraud_Security_Policy.md`
- `07_Order_Cancellation_Policy.md`

The repository includes a small uploader: [`scripts/upload_foodpanda_policy_docs.py`](../scripts/upload_foodpanda_policy_docs.py). It reads **all** `*.md` files in the folder above (including newly added files), derives `title` from the first `#` heading (or the filename), builds stable `_id` values from the filename stem, sets `tags` from `foodpanda`, `policy`, and filename tokens, and POSTs to the Elasticsearch [`_bulk`](https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-bulk.html) API.

## Prerequisites

- Run commands from the **repository root** (`BitBot/`).
- Elasticsearch **reachable** on HTTP (this project’s Compose disables Elasticsearch security for local dev).
- Python 3.10+ (stdlib only; no extra packages).

Start Elasticsearch if needed:

```bash
docker compose up -d elasticsearch
```

Wait until the cluster responds (e.g. `curl -s http://localhost:9200` on the host when port **9200** is published).

---

## Recommended: upload with the Python script

From the repository root:

1. **Dry run** (parse files, show IDs; no HTTP):

   ```bash
   python scripts/upload_foodpanda_policy_docs.py --dry-run
   ```

2. **Create the index (optional)** and **bulk-upload** documents.

   **Host machine** (Elasticsearch on `localhost:9200`, typical when Compose maps the port):

   ```bash
   python scripts/upload_foodpanda_policy_docs.py --create-index --host localhost
   ```

   Override the index name if you use a custom `ES_POLICY_INDEX`:

   ```bash
   python scripts/upload_foodpanda_policy_docs.py --create-index --host localhost --index my_policy_index
   ```

   **Same Docker network as Elasticsearch** (e.g. script runs in a container, or you set the service hostname). The hostname `elasticsearch` matches `.env.example` and `docker-compose.yml`:

   ```bash
   python scripts/upload_foodpanda_policy_docs.py --create-index --host elasticsearch --port 9200
   ```

The script reads these environment variables when flags are omitted (defaults in parentheses): `ES_HOST` (**localhost** if unset), `ES_PORT` (9200), `ES_SCHEME` (http), `ES_POLICY_INDEX` (policy_docs), `ES_TIMEOUT_SECONDS` (60 for HTTP in this script).

- `--create-index` sends `PUT /{index}` with an empty JSON body. If the index already exists, Elasticsearch returns `resource_already_exists_exception` and the script continues.
- On success you should see: `ok: indexed N item(s), took_ms=...`.
- If any bulk line fails, the script exits with a non-zero status and prints per-item errors.

---

## Verify (matches app query shape)

**Terminal (host):**

```bash
curl -s -X POST "http://localhost:9200/policy_docs/_search" -H "Content-Type: application/json" -d "{\"size\":3,\"query\":{\"multi_match\":{\"query\":\"refund\",\"fields\":[\"title^2\",\"content\",\"tags\"]}}}"
```

Replace `policy_docs` if you used a custom `--index` / `ES_POLICY_INDEX`.

**Docker:**

```bash
docker compose exec -T elasticsearch curl -s -X POST "http://localhost:9200/policy_docs/_search" -H "Content-Type: application/json" -d "{\"size\":3,\"query\":{\"multi_match\":{\"query\":\"refund\",\"fields\":[\"title^2\",\"content\",\"tags\"]}}}"
```

You should see hits whose `_source` includes `title`, `content`, and `tags`.

---

## Backend connectivity

Policy retrieval in the app uses `ES_HOST`, `ES_PORT`, `ES_SCHEME`, and `ES_POLICY_INDEX` from the environment (see `.env.example`). If `ES_HOST` is unset, retrieval returns no documents.

- **Uploader on the host** usually targets `--host localhost` while the **backend in Compose** uses `ES_HOST=elasticsearch`. Both can talk to the **same** cluster; only the hostname differs.
- The **index name** you upload to must match **`ES_POLICY_INDEX`** for the backend to find documents.

---

## Optional: reset the index

To re-import from scratch:

**Terminal (host):**

```bash
curl -s -X DELETE "http://localhost:9200/policy_docs"
```

**Docker:**

```bash
docker compose exec -T elasticsearch curl -s -X DELETE "http://localhost:9200/policy_docs"
```

Then run the upload script again with `--create-index`.

---

## Manual `_bulk` (without the script)

If you prefer raw NDJSON, each document is still **two lines**: an action line, then a JSON source line. Example:

```ndjson
{"index":{"_id":"01-returns-and-refund-policy"}}
{"title":"Global Returns & Refund Policy","content":"...","tags":["foodpanda","policy","returns"]}
```

Post to `POST /{index}/_bulk` with `Content-Type: application/x-ndjson` and a trailing newline after the last line. See the [README](../README.md) section “How to add data to Elasticsearch” for generic `curl` examples.

---

## Troubleshooting

| Issue | What to check |
|--------|----------------|
| Connection refused | `docker compose ps`; ensure Elasticsearch is up; on the host use `--host localhost` if the port is mapped. |
| Script exits with bulk errors | Re-run with a fresh index or fix mapping conflicts; stderr lists failing items. |
| `Create index failed` | Permissions, wrong URL, or cluster not ready. |
| Backend returns no policy hits | Set `ES_HOST` for the backend; index name must match `ES_POLICY_INDEX`. |
| Wrong host from `.env` | `.env` may set `ES_HOST=elasticsearch` (for Compose). For a script on the host, pass `--host localhost` or unset `ES_HOST`. |

For more generic Compose-oriented examples, see the [README](../README.md) section “How to add data to Elasticsearch”.
