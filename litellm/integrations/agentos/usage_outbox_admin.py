from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, TypeAdapter


class OutboxArguments(BaseModel):
    database: str | None = None
    command: str
    call_id: str | None = None


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=rw", uri=True, timeout=30)


def status(path: Path) -> Mapping[str, int | float]:
    now = time.time()
    with _connect(path) as connection:
        pending, dead_letter, oldest_pending = TypeAdapter(tuple[int | None, int | None, float | None]).validate_python(
            connection.execute(
                """
            SELECT
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END),
                SUM(CASE WHEN status = 'dead_letter' THEN 1 ELSE 0 END),
                MIN(CASE WHEN status = 'pending' THEN created_at END)
            FROM usage_outbox
            """
            ).fetchone()
        )
    return {  # mutable-ok: json.dumps emits the operator-facing native JSON status dictionary.
        "pending": int(pending or 0),
        "dead_letter": int(dead_letter or 0),
        "oldest_pending_age_seconds": round(
            max(0.0, now - float(oldest_pending)) if oldest_pending else 0.0,
            3,
        ),
    }


def requeue_dead_letters(path: Path, *, call_id: str | None) -> int:
    where = "status = 'dead_letter'"
    parameters: tuple[object, ...] = (time.time(),)
    if call_id is not None:
        where += " AND call_id = ?"
        parameters += (call_id,)
    with _connect(path) as connection:
        cursor = connection.execute(
            f"""
            UPDATE usage_outbox
            SET status = 'pending', next_attempt_at = 0,
                last_error = NULL, updated_at = ?
            WHERE {where}
            """,
            parameters,
        )
    return cursor.rowcount


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect or requeue usage outbox")
    parser.add_argument(
        "--database",
        default=os.environ.get("AGENTOS_USAGE_OUTBOX_PATH"),
        help="SQLite outbox path (defaults to AGENTOS_USAGE_OUTBOX_PATH)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Print non-sensitive outbox status as JSON")
    requeue = commands.add_parser("requeue", help="Move dead-letter events back to the pending queue")
    selection = requeue.add_mutually_exclusive_group(required=True)
    selection.add_argument("--call-id")
    selection.add_argument("--all", action="store_true")
    args = OutboxArguments.model_validate(vars(parser.parse_args(argv)))
    if not args.database:
        parser.error("--database or AGENTOS_USAGE_OUTBOX_PATH is required")

    try:
        path = Path(args.database)
        result = (
            status(path)
            if args.command == "status"
            else {  # mutable-ok: json.dumps emits the native JSON command result.
                "requeued": requeue_dead_letters(path, call_id=args.call_id)
            }
        )
    except sqlite3.Error:
        sys.stdout.write('{"error":"outbox unavailable"}\n')
        return 1
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
