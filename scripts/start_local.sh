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

{
  IFS= read -r compression_backend
  IFS= read -r configured_compression_url
  IFS= read -r configured_compression_token
  IFS= read -r headroom_proxy_port
  IFS= read -r lean_ctx_proxy_port
} < <(
  uv run --env-file .env python - <<'PY'
import os

print(os.environ.get("COMPRESSION_BACKEND", "none"))
print(os.environ.get("COMPRESSION_PROXY_URL", ""))
print(os.environ.get("COMPRESSION_PROXY_TOKEN", ""))
print(os.environ.get("HEADROOM_PROXY_PORT", "8787"))
print(os.environ.get("LEAN_CTX_PROXY_PORT", "4444"))
PY
)

case "$compression_backend" in
  none)
    proxy_config_path="litellm/proxy/dev_config.yaml"
    ;;
  headroom)
    export COMPRESSION_PROXY_URL="${configured_compression_url:-http://127.0.0.1:${headroom_proxy_port}}"
    export COMPRESSION_PROXY_TOKEN="${configured_compression_token:-}"
    proxy_config_path="litellm/proxy/dev_config_headroom.yaml"
    ;;
  lean-ctx)
    if ! command -v lean-ctx >/dev/null 2>&1; then
      echo "lean-ctx is not installed" >&2
      exit 1
    fi
    export COMPRESSION_PROXY_URL="${configured_compression_url:-http://127.0.0.1:${lean_ctx_proxy_port}}"
    export COMPRESSION_PROXY_TOKEN="${configured_compression_token:-$(lean-ctx proxy token)}"
    proxy_config_path="litellm/proxy/dev_config_headroom.yaml"
    ;;
  *)
    echo "COMPRESSION_BACKEND must be 'none', 'headroom', or 'lean-ctx'" >&2
    exit 1
    ;;
esac

if [[ "$compression_backend" == "none" ]]; then
  echo "Compression backend: disabled"
else
  echo "Compression backend: ${compression_backend} (${COMPRESSION_PROXY_URL})"
fi

if [[ ! -x "ui/litellm-dashboard/node_modules/.bin/next" ]]; then
  echo "Dashboard dependencies are missing. Run 'npm install' in ui/litellm-dashboard first" >&2
  exit 1
fi

echo "Building Dashboard for http://localhost:4000/ui/"
npm --prefix ui/litellm-dashboard run build

dashboard_output_path="$repo_dir/ui/litellm-dashboard/out"
if [[ ! -f "$dashboard_output_path/index.html" || ! -f "$dashboard_output_path/token-analytics/index.html" ]]; then
  echo "Dashboard build is incomplete" >&2
  exit 1
fi

export LITELLM_UI_PATH="$dashboard_output_path"
echo "LiteLLM Dashboard: http://localhost:4000/ui/"
echo "Token Analytics: http://localhost:4000/ui/token-analytics/"

exec uv run --env-file .env python litellm/proxy/proxy_cli.py \
  --config "$proxy_config_path" \
  --host 0.0.0.0 \
  --port 4000 \
  --use_v2_migration_resolver \
  --enforce_prisma_migration_check
