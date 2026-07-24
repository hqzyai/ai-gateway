#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if ! docker info >/dev/null 2>&1; then
  if [[ "$(uname -s)" != "Darwin" || ! -d /Applications/Docker.app ]]; then
    echo "Docker is not running" >&2
    exit 1
  fi

  open -a Docker
  for _ in {1..120}; do
    if docker info >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker did not become ready" >&2
  exit 1
fi

docker compose up -d db

for _ in {1..60}; do
  db_health="$(docker inspect -f '{{.State.Health.Status}}' litellm_db 2>/dev/null || true)"
  if [[ "$db_health" == "healthy" ]]; then
    break
  fi
  sleep 1
done

if [[ "${db_health:-}" != "healthy" ]]; then
  docker compose logs --tail 50 db
  echo "PostgreSQL did not become healthy" >&2
  exit 1
fi

uv run --env-file .env python - <<'PY'
from subprocess import DEVNULL, run
from urllib.parse import unquote, urlsplit

from dotenv import dotenv_values

database_url = dotenv_values(".env").get("DATABASE_URL")
if not isinstance(database_url, str):
    raise SystemExit("DATABASE_URL is missing from .env")

url = urlsplit(database_url)
if url.hostname not in {"127.0.0.1", "localhost", "::1"} or (url.port or 5432) != 5432:
    raise SystemExit("DATABASE_URL must point to the local PostgreSQL service on port 5432")

user = unquote(url.username or "")
password = unquote(url.password or "")
database = unquote(url.path.lstrip("/"))
if not user or not password or not database:
    raise SystemExit("DATABASE_URL must include a user, password, and database")


def identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


role_sql = (
    f"DO $do$ BEGIN "
    f"IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = {literal(user)}) THEN "
    f"ALTER ROLE {identifier(user)} WITH LOGIN PASSWORD {literal(password)}; "
    f"ELSE CREATE ROLE {identifier(user)} WITH LOGIN PASSWORD {literal(password)}; "
    f"END IF; END $do$;"
)
run(
    ["docker", "exec", "-i", "litellm_db", "psql", "-U", "llmproxy", "-d", "litellm", "-v", "ON_ERROR_STOP=1", "-q"],
    input=role_sql,
    text=True,
    check=True,
    stdout=DEVNULL,
)

database_exists = run(
    [
        "docker",
        "exec",
        "litellm_db",
        "psql",
        "-U",
        "llmproxy",
        "-d",
        "litellm",
        "-tAc",
        f"SELECT 1 FROM pg_database WHERE datname = {literal(database)}",
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()

if database_exists != "1":
    run(
        ["docker", "exec", "litellm_db", "createdb", "-U", "llmproxy", "-O", user, database],
        check=True,
        stdout=DEVNULL,
    )

ownership_sql = (
    f"ALTER DATABASE {identifier(database)} OWNER TO {identifier(user)}; "
    f"ALTER SCHEMA public OWNER TO {identifier(user)}; "
    f"GRANT ALL ON SCHEMA public TO {identifier(user)};"
)
run(
    ["docker", "exec", "-i", "litellm_db", "psql", "-U", "llmproxy", "-d", database, "-v", "ON_ERROR_STOP=1", "-q"],
    input=ownership_sql,
    text=True,
    check=True,
    stdout=DEVNULL,
)
PY

echo "LiteLLM Dashboard: http://localhost:4000/ui/"
exec uv run --env-file .env python litellm/proxy/proxy_cli.py \
  --host 0.0.0.0 \
  --port 4000 \
  --use_v2_migration_resolver \
  --enforce_prisma_migration_check
