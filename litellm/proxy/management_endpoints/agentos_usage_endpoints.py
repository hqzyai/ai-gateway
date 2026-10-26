"""Private BFF usage query: predicates, totals and a page share one SQL snapshot."""

from __future__ import annotations

import hmac
import json
import math
from collections.abc import Mapping
from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, JsonValue, TypeAdapter, model_validator

TEXT_FIELDS = (
    "model",
    "model_group",
    "model_id",
    "actual_model",
    "provider",
    "key_alias",
    "end_user",
    "request_id",
    "session_id",
    "error_code",
    "error_message",
    "status_filter",
    "call_type",
)
RANGES = ("spend", "tokens", "duration_ms")
EXTRA_METRICS = (
    "cache_creation_input_tokens",
    "compression_saved_tokens",
    "compression_savings_spend",
    "prompt_caching_savings_spend",
    "compression_gross_saved_tokens",
    "compression_extra_input_tokens",
    "compression_requests",
    "compression_gross_savings_spend",
    "compression_extra_input_spend",
)


class UsageQuery(BaseModel):
    start: datetime
    end: datetime
    timezone: int = Field(default=0, ge=-840, le=840)
    filters: Mapping[str, str] = Field(default_factory=dict)
    user_ids: list[str] = Field(  # mutable-ok: Prisma binds this JSON list as a PostgreSQL text[] parameter.
        default_factory=list
    )
    include_unowned: bool = False
    deployment_models: Mapping[str, str] = Field(default_factory=dict)
    include_records: bool = False
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def validate_query(self) -> UsageQuery:
        if not timedelta(0) < self.end - self.start <= timedelta(days=366):
            raise ValueError("Invalid usage date range")
        allowed = frozenset(
            (
                *TEXT_FIELDS,
                "search",
                "api_key",
                "actual_model_aliases",
                *(f"{b}_{n}" for n in RANGES for b in ("min", "max")),
            )
        )
        if frozenset(self.filters) - allowed or any(len(v) > 500 for v in self.filters.values()):
            raise ValueError("Invalid usage filters")
        if self.filters.get("status_filter", "") not in ("", "success", "failure"):
            raise ValueError("Invalid usage status")
        if self.filters.get("actual_model_aliases"):
            aliases = TypeAdapter[JsonValue](JsonValue).validate_json(self.filters["actual_model_aliases"])
            if (
                not isinstance(aliases, list)
                or not 1 <= len(aliases) <= 8
                or any(not isinstance(value, str) or not value.strip() or len(value) > 200 for value in aliases)
                or len(frozenset(str(value) for value in aliases)) != len(aliases)
            ):
                raise ValueError("Invalid actual model aliases")
        for name in RANGES:
            lo, hi = self.filters.get(f"min_{name}"), self.filters.get(f"max_{name}")
            if any(v and (not math.isfinite(float(v)) or float(v) < 0) for v in (lo, hi)):
                raise ValueError("Invalid usage range")
            if lo and hi and float(lo) > float(hi):
                raise ValueError("Invalid usage range")
        return self


def _object(expr: str) -> str:
    # JSON columns and legacy JSON-encoded strings both occur in retained logs.
    return (
        f"CASE WHEN jsonb_typeof({expr}) = 'object' THEN {expr} "
        f"WHEN jsonb_typeof({expr}) = 'string' AND pg_input_is_valid({expr} #>> '{{}}', 'jsonb') "
        f"THEN CASE WHEN jsonb_typeof(({expr} #>> '{{}}')::jsonb) = 'object' "
        f"THEN ({expr} #>> '{{}}')::jsonb ELSE '{{}}'::jsonb END ELSE '{{}}'::jsonb END"
    )


def _number(expr: str) -> str:
    return (
        f"CASE WHEN pg_input_is_valid(({expr})::text, 'double precision') "
        f"AND lower(({expr})::text) NOT IN ('nan', 'infinity', '-infinity', 'inf', '-inf') "
        f"THEN ({expr})::double precision ELSE 0 END"
    )


def build_query(query: UsageQuery) -> tuple[str, tuple[object, ...]]:
    """Only trusted identifiers enter SQL; all caller values are bound parameters."""
    values: tuple[object, ...] = ()

    def bind(value: object) -> str:
        nonlocal values
        values += (value,)
        return f"${len(values)}"

    scope = (
        f"\"startTime\" >= ({bind(query.start)}::timestamptz AT TIME ZONE 'UTC')",
        f"\"startTime\" < ({bind(query.end)}::timestamptz AT TIME ZONE 'UTC')",
    )
    if not query.include_unowned:
        scope += (f'"user" = ANY({bind(query.user_ids)}::text[])',)
    predicates: tuple[str, ...] = ()
    for key, value in query.filters.items():
        if not value:
            continue
        if key == "actual_model_aliases":
            predicates += (f"actual_model = ANY({bind(TypeAdapter(list[str]).validate_json(value))}::text[])",)
        elif key in ("search", "key_alias", "session_id", "error_message"):
            # strpos treats %, _ and backslash literally, unlike LIKE.
            column = (
                "concat_ws(' ', " + ", ".join((*TEXT_FIELDS, '"user"', "api_key")) + ")" if key == "search" else key
            )
            clause = f"strpos(lower({column}), lower({bind(value)})) > 0"
            if key == "session_id":
                scope += (clause,)
            else:
                predicates += (clause,)
        elif key in (*TEXT_FIELDS, "api_key"):
            clause = f"{key} = {bind(value)}"
            if key in ("api_key", "model", "model_group", "model_id", "request_id", "end_user", "call_type"):
                scope += (clause,)
            else:
                predicates += (clause,)
        else:
            bound, name = key.split("_", 1)
            column = (
                'COALESCE(request_duration_ms, EXTRACT(EPOCH FROM ("endTime" - "startTime")) * 1000)'
                if name == "duration_ms"
                else _number("total_tokens" if name == "tokens" else "spend")
            )
            scope += (f"{column} {'>=' if bound == 'min' else '<='} {bind(float(value))}",)

    metrics = tuple((name, _number(name)) for name in ("spend", "prompt_tokens", "completion_tokens", "total_tokens"))
    metrics += (
        ("api_requests", "1"),
        ("successful_requests", "(status_filter = 'success')::int"),
        ("failed_requests", "(status_filter <> 'success')::int"),
    )
    metrics += (
        (
            "cache_read_input_tokens",
            _number(
                "COALESCE(extra->>'cache_read_input_tokens', md->>'cache_read_input_tokens', "
                "usage->>'cache_read_input_tokens', token_details->>'cached_tokens')"
            ),
        ),
    )
    metrics += tuple(
        (name, _number(f"COALESCE(extra->>'{name}', md->>'{name}', usage->>'{name}')")) for name in EXTRA_METRICS
    )
    metric_columns = ", ".join(f"{expr} AS m_{name}" for name, expr in metrics)
    metric_sums = ", ".join(f"'{name}', SUM(m_{name})" for name, _ in metrics)
    shift = bind(query.timezone)
    limit = bind(query.page_size if query.include_records else 0)
    offset = bind((query.page - 1) * query.page_size)
    identities = bind(json.dumps(query.deployment_models))
    historical_upstream = "md #>> '{spend_logs_metadata,agentos_routing_hops,-1,upstream}'"
    mapped_upstream = f"({identities}::jsonb)->>model_id"
    # A saved hop wins if the deployment has since changed its upstream. When
    # the names still agree, the catalog mapping only supplies the vendor namespace.
    historical_identity = (
        f"CASE WHEN regexp_replace({historical_upstream}, '^openai/', '') = "
        f"regexp_replace({mapped_upstream}, '^[^/]+/', '') THEN {mapped_upstream} "
        f"ELSE {historical_upstream} END"
    )
    dimensions = "day, model, model_group, model_id, actual_model, provider, api_key, key_alias, user_id, key_hash"
    # SpendLogs metadata can contain large request context. Extract its small
    # billing subset once, rather than repeatedly decompressing the TOAST value.
    metadata_keys = (
        "agentos_provider",
        "agentos_upstream_model",
        "model_info",
        "spend_logs_metadata",
        "agentos_key_alias",
        "user_api_key_alias",
        "status",
        "error_information",
        "additional_usage_values",
        "usage_object",
        "user_api_key_hash",
        "cache_read_input_tokens",
        *EXTRA_METRICS,
    )
    metadata_columns = ", ".join(f"{key} jsonb" for key in metadata_keys)
    metadata_object = ", ".join(f"'{key}', billing.{key}" for key in metadata_keys)
    sql = f"""
        WITH scoped AS MATERIALIZED (
            SELECT logs.request_id, logs."startTime", logs."endTime", logs.model, logs.model_group, logs.model_id,
                   logs.api_key, logs."user", logs.session_id, logs.end_user, logs.call_type, logs.status,
                   logs.custom_llm_provider, logs.spend, logs.total_tokens, logs.prompt_tokens,
                   logs.completion_tokens, logs.request_duration_ms,
                   jsonb_build_object({metadata_object}) AS md
            FROM "LiteLLM_SpendLogs" AS logs
            CROSS JOIN LATERAL jsonb_to_record({_object("logs.metadata::jsonb")}) AS billing({metadata_columns})
            WHERE {" AND ".join(scope)}
        ), normalized AS MATERIALIZED (
            SELECT *,
                COALESCE(NULLIF(md->>'agentos_upstream_model', ''),
                         NULLIF(md #>> '{{model_info,agentos_upstream_model}}', ''),
                         NULLIF(md #>> '{{spend_logs_metadata,agentos_upstream_model}}', ''),
                         NULLIF({historical_identity}, ''),
                         NULLIF(({identities}::jsonb)->>model_id, ''),
                         '__unattributed__') AS actual_model,
                COALESCE(NULLIF(lower(btrim(md->>'agentos_provider')), ''),
                         NULLIF(lower(btrim(md #>> '{{model_info,litellm_provider}}')), ''),
                         CASE WHEN md #>> '{{model_info,agentos_upstream_model}}' LIKE '%/%'
                              THEN lower(split_part(md #>> '{{model_info,agentos_upstream_model}}', '/', 1)) END,
                         NULLIF(custom_llm_provider, ''), 'unknown') AS provider,
                COALESCE(NULLIF(md->>'agentos_key_alias', ''), md->>'user_api_key_alias', '') AS key_alias,
                COALESCE(NULLIF(status, ''), NULLIF(md->>'status', ''), 'success') AS status_filter,
                ({_object("md->'error_information'")})->>'error_code' AS error_code,
                COALESCE(({_object("md->'error_information'")})->>'error_message',
                         CASE WHEN jsonb_typeof(md->'error_information') = 'string'
                              THEN md->>'error_information' END, '') AS error_message,
                COALESCE(request_duration_ms, EXTRACT(EPOCH FROM ("endTime" - "startTime")) * 1000) AS duration_ms,
                {_object("md->'additional_usage_values'")} AS extra,
                {_object("md->'usage_object'")} AS usage
            FROM scoped
        ), with_details AS MATERIALIZED (
            SELECT *, COALESCE(NULLIF({_object("usage->'prompt_tokens_details'")}, '{{}}'::jsonb),
                               {_object("extra->'prompt_tokens_details'")}) AS token_details
            FROM normalized WHERE {" AND ".join(predicates) or "TRUE"}
        ), filtered AS MATERIALIZED (
            SELECT *, {metric_columns},
                   ("startTime" - ({shift}::int * interval '1 minute'))::date::text AS day,
                   "user" AS user_id, md->>'user_api_key_hash' AS key_hash
            FROM with_details
        ), buckets AS (
            SELECT {dimensions}, jsonb_build_object({metric_sums}) AS metrics
            FROM filtered GROUP BY {dimensions}
        ), records AS (
            SELECT request_id, ("startTime" AT TIME ZONE 'UTC') AS "startTime", model, model_group, model_id, actual_model, provider, key_alias,
                   user_id, status_filter AS status, m_spend AS spend, m_total_tokens AS total_tokens
            FROM filtered ORDER BY "startTime" DESC, request_id DESC LIMIT {limit} OFFSET {offset}
        )
        SELECT COALESCE((SELECT jsonb_agg(to_jsonb(buckets)) FROM buckets), '[]'::jsonb) AS buckets,
               COALESCE((SELECT jsonb_agg(to_jsonb(records) ORDER BY "startTime" DESC, request_id DESC)
                         FROM records), '[]'::jsonb) AS records,
               (SELECT COUNT(*) FROM filtered) AS records_total
    """
    return sql, values


async def require_master_key(request: Request) -> None:
    from litellm.proxy.proxy_server import master_key

    expected = f"Bearer {master_key}" if master_key else ""
    if not expected or not hmac.compare_digest(request.headers.get("authorization", ""), expected):
        raise HTTPException(403, "Usage query requires gateway master key")


router = APIRouter()


class DailyUsageQuery(BaseModel):
    start_date: date | None = None
    end_date: date | None = None
    timezone: int = Field(default=0, ge=-840, le=840)
    user_ids: list[str] = Field(  # mutable-ok: Native daily activity accepts a list of authorized user IDs.
        default_factory=list
    )
    include_unowned: bool = False
    model: str | None = None
    api_key: str | None = None

    @model_validator(mode="after")
    def validate_dates(self) -> DailyUsageQuery:
        if (
            self.start_date
            and self.end_date
            and not timedelta(0) <= self.end_date - self.start_date < timedelta(days=366)
        ):
            raise ValueError("Invalid usage date range")
        return self


@router.post("/agentos/usage/daily", dependencies=(Depends(require_master_key),))
async def query_daily_usage(query: DailyUsageQuery) -> object:
    """Use native daily metrics once for the complete BFF-authorized user scope."""
    from litellm.proxy.management_endpoints.common_daily_activity import (
        get_daily_activity_aggregated,  # pyright: ignore[reportUnknownVariableType]  # Existing native daily aggregation has partially annotated parameters.
    )
    from litellm.proxy.proxy_server import prisma_client

    return await get_daily_activity_aggregated(
        prisma_client=prisma_client,
        table_name="litellm_dailyuserspend",
        entity_id_field="user_id",
        entity_id=None if query.include_unowned else query.user_ids,
        entity_metadata_field=None,
        start_date=query.start_date.isoformat() if query.start_date else None,
        end_date=query.end_date.isoformat() if query.end_date else None,
        model=query.model,
        api_key=query.api_key,
        timezone_offset_minutes=query.timezone,
    )


@router.post("/agentos/usage/query", dependencies=(Depends(require_master_key),))
async def query_usage(query: UsageQuery) -> Mapping[str, object]:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(503, "Usage storage unavailable")
    sql, values = build_query(query)
    rows = TypeAdapter(tuple[Mapping[str, object], ...]).validate_python(await prisma_client.db.query_raw(sql, *values))  # pyright: ignore[reportAny]  # Validate the native Prisma raw-query response at its boundary.
    result = dict(  # mutable-ok: Decode legacy JSON-string fields in the native response dictionary.
        rows[0]
    )
    for key in ("buckets", "records"):
        value = result[key]
        if isinstance(value, str):
            result[key] = TypeAdapter[JsonValue](JsonValue).validate_json(value)
    return result


class PricingImport(BaseModel):
    data: Mapping[str, Mapping[str, JsonValue]] = Field(min_length=1, max_length=10000)


@router.put("/model/cost_map/overrides/import", dependencies=(Depends(require_master_key),))
async def import_pricing_rules(body: PricingImport) -> Mapping[str, object]:
    from litellm.proxy.management_endpoints import model_cost_map_endpoints as native

    try:
        imported = await native.apply_pricing_import(body.data)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, "Invalid pricing snapshot") from exc
    return {  # mutable-ok: FastAPI serializes this native JSON endpoint response.
        "status": "success",
        "imported": imported,
    }
