"""
Track guardrail and policy usage for the dashboard: upsert daily metrics and
insert into SpendLogGuardrailIndex when spend logs are written.
"""

import json
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.proxy.utils import PrismaClient
from litellm.repositories.table_repositories import (
    DailyGuardrailMetricsRepository,
    SpendLogGuardrailIndexRepository,
)


def guardrail_entry_to_action(entry: Mapping[str, object]) -> str:
    status = str(entry.get("guardrail_status") or "").lower()
    if "intervened" in status or "block" in status:
        return "blocked"
    if "fail" in status or "error" in status:
        return "flagged"
    masked = entry.get("masked_entity_count")
    if isinstance(masked, dict) and any(
        int(value or 0) > 0
        for value in TypeAdapter(Mapping[str, int | str | float | None]).validate_python(masked).values()
    ):
        return "masked"
    response = entry.get("guardrail_response")
    if isinstance(response, str) and response.strip().lower() in ("mask", "masked", "redact", "redacted"):
        return "masked"
    if isinstance(response, list):
        for item in TypeAdapter(tuple[object, ...]).validate_python(response):
            if isinstance(item, dict) and str(
                TypeAdapter(Mapping[str, object]).validate_python(item).get("action") or ""
            ).upper() in ("MASK", "REDACT"):
                return "masked"
    return "passed"


def _guardrail_status_to_action(status: Optional[str]) -> str:
    """Map StandardLogging guardrail_status to blocked/passed/flagged."""
    if not status:
        return "passed"
    s = (status or "").lower()
    if "intervened" in s or "block" in s:
        return "blocked"
    if "fail" in s or "error" in s:
        return "flagged"
    return "passed"


def _parse_guardrail_info_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract guardrail_information from spend log payload metadata."""
    meta = payload.get("metadata")
    if not meta:
        return []
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(meta, dict):
        return []
    info = meta.get("guardrail_information") or meta.get("standard_logging_guardrail_information")
    if not isinstance(info, list):
        return []
    return info


def _date_str(dt: datetime) -> str:
    """YYYY-MM-DD in UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


async def process_spend_logs_guardrail_usage(
    prisma_client: PrismaClient,
    logs_to_process: List[Dict[str, Any]],
) -> None:
    """
    After spend logs are written: update DailyGuardrailMetrics and insert
    SpendLogGuardrailIndex rows from guardrail_information in each payload.
    """
    if not logs_to_process:
        return
    daily_guardrail: Dict[tuple, Dict[str, Any]] = defaultdict(
        lambda: {
            "requests_evaluated": 0,
            "passed_count": 0,
            "blocked_count": 0,
            "flagged_count": 0,
            "latency_sum_ms": 0.0,
            "latency_samples": 0,
        }
    )
    index_rows: List[Dict[str, Any]] = []

    for payload in logs_to_process:
        request_id = payload.get("request_id")
        start_time = payload.get("startTime")
        if not request_id or not start_time:
            continue
        if isinstance(start_time, str):
            try:
                start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue
        date_key = _date_str(start_time)

        for entry in _parse_guardrail_info_from_payload(payload):
            guardrail_id = entry.get("guardrail_id") or entry.get("guardrail_name") or ""
            if not guardrail_id:
                continue
            key = (guardrail_id, date_key)
            daily_guardrail[key]["requests_evaluated"] += 1
            action = guardrail_entry_to_action(entry)
            if action == "passed":
                daily_guardrail[key]["passed_count"] += 1
            elif action == "blocked":
                daily_guardrail[key]["blocked_count"] += 1
            else:
                daily_guardrail[key]["flagged_count"] += 1
            duration = entry.get("duration")
            if duration is not None:
                try:
                    latency_ms = float(duration) * 1000
                except (TypeError, ValueError):
                    latency_ms = -1
                if latency_ms >= 0:
                    daily_guardrail[key]["latency_sum_ms"] += latency_ms
                    daily_guardrail[key]["latency_samples"] += 1
            policy_id = entry.get("policy_id")
            index_rows.append(
                {
                    "request_id": request_id,
                    "guardrail_id": guardrail_id,
                    "policy_id": policy_id,
                    "start_time": start_time,
                }
            )

    if not daily_guardrail and not index_rows:
        return

    try:
        # Insert index rows (skip duplicates by request_id + guardrail_id)
        if index_rows:
            index_data = []
            for r in index_rows:
                st = r["start_time"]
                if isinstance(st, str):
                    try:
                        st = datetime.fromisoformat(st.replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        continue
                index_data.append(
                    {
                        "request_id": r["request_id"],
                        "guardrail_id": r["guardrail_id"],
                        "policy_id": r.get("policy_id"),
                        "start_time": st,
                    }
                )
            try:
                await SpendLogGuardrailIndexRepository(prisma_client).table.create_many(
                    data=index_data,
                    skip_duplicates=True,
                )
            except Exception as e:
                verbose_proxy_logger.debug("Guardrail usage tracking: index create_many skipped: %s", e)

        metrics_table = DailyGuardrailMetricsRepository(prisma_client).table
        for (guardrail_id, date_key), agg in daily_guardrail.items():
            n = int(agg["requests_evaluated"])
            if n == 0:
                continue
            where = {
                "guardrail_id_date": {
                    "guardrail_id": guardrail_id,
                    "date": date_key,
                }
            }
            latency_samples = int(agg["latency_samples"])
            batch_avg_latency = float(agg["latency_sum_ms"]) / latency_samples if latency_samples > 0 else None
            existing = await metrics_table.find_unique(where=where)
            existing_avg = getattr(existing, "avg_latency_ms", None) if existing else None
            existing_count = int(getattr(existing, "requests_evaluated", 0) or 0) if existing else 0
            if batch_avg_latency is None:
                combined_avg_latency = existing_avg
            elif existing_avg is None or existing_count == 0:
                combined_avg_latency = batch_avg_latency
            else:
                combined_avg_latency = (float(existing_avg) * existing_count + batch_avg_latency * latency_samples) / (
                    existing_count + latency_samples
                )
            await metrics_table.upsert(
                where=where,
                data={
                    "create": {
                        "guardrail_id": guardrail_id,
                        "date": date_key,
                        "requests_evaluated": n,
                        "passed_count": int(agg["passed_count"]),
                        "blocked_count": int(agg["blocked_count"]),
                        "flagged_count": int(agg["flagged_count"]),
                        "avg_latency_ms": combined_avg_latency,
                    },
                    "update": {
                        "requests_evaluated": {"increment": n},
                        "passed_count": {"increment": int(agg["passed_count"])},
                        "blocked_count": {"increment": int(agg["blocked_count"])},
                        "flagged_count": {"increment": int(agg["flagged_count"])},
                        "avg_latency_ms": combined_avg_latency,
                    },
                },
            )
    except Exception as e:
        verbose_proxy_logger.warning("Guardrail usage tracking failed (non-fatal): %s", e)
