import json
import os
import re
import sys
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from litellm.proxy.management_endpoints import agentos_usage_endpoints as gateway


def test_usage_sql_keeps_filters_bound_and_savings_versioned():
    variants = (
        ({},)
        + tuple({key: ("success" if key == "status_filter" else "中文 %_\\ injection")} for key in gateway.TEXT_FIELDS)
        + tuple({f"{bound}_{name}": "1.25"} for name in gateway.RANGES for bound in ("min", "max"))
        + (
            {"actual_model_aliases": '["dashscope/model", "model"]'},
            {
                "search": "x",
                "session_id": "y",
                "api_key": "hash",
                "min_tokens": "1",
                "max_tokens": "200",
                "provider": "dashscope",
            },
        )
    )
    for filters in variants:
        for unowned in (False, True):
            for records in (False, True):
                request = gateway.UsageQuery(
                    start="2026-10-01T00:00:00Z",
                    end="2026-10-20T00:00:00Z",
                    filters=filters,
                    user_ids=["u1", "u2"],
                    include_unowned=unowned,
                    include_records=records,
                    page=3,
                    page_size=9,
                    timezone=-480,
                    deployment_models={"deployment": "dashscope/model"},
                )
                sql, parameters = gateway.build_query(request)
                assert "1 AS savings_schema_version" in sql
                assert "中文" not in sql
                assert "optimization_savings" in sql
                assert len(parameters) == max(int(value) for value in re.findall(r"\$(\d+)", sql))


def query(**kwargs):
    return gateway.UsageQuery(start="2026-09-22T00:00:00Z", end="2026-09-29T00:00:00Z", **kwargs)


@pytest.fixture
def db():
    import psycopg

    dsn = os.environ.get("AGENTOS_USAGE_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set AGENTOS_USAGE_TEST_DATABASE_URL to run PostgreSQL usage regressions")
    with psycopg.connect(dsn) as conn:
        # TEMP shadows the real table and the transaction always rolls back.
        conn.execute("""CREATE TEMP TABLE "LiteLLM_SpendLogs" (
            request_id text PRIMARY KEY, "startTime" timestamp, "endTime" timestamp,
            model text, model_group text, model_id text, api_key text, "user" text,
            session_id text, end_user text, call_type text, status text, custom_llm_provider text,
            spend double precision, total_tokens bigint, prompt_tokens bigint, completion_tokens bigint,
            request_duration_ms double precision, metadata jsonb
        )""")
        conn.execute(
            """INSERT INTO "LiteLLM_SpendLogs"
            SELECT 'request-' || n, '2026-09-22 16:00:00'::timestamp, '2026-09-22 16:00:01'::timestamp,
                   'model-a', 'group-a', 'deployment-a', 'hash-a', 'alice', 'SESSION-%%_abc',
                   'customer-a', 'acompletion', 'failure', 'openai', 0.25, 30, 20, 10, NULL,
                   %s::jsonb
            FROM generate_series(1, 10005) AS n""",
            (
                json.dumps(
                    {
                        "agentos_key_alias": "Key 中文 %_Alpha",
                        "user_api_key_alias": "old-alias",
                        "agentos_provider": "siliconflow",
                        "error_information": {"error_code": "429", "error_message": "Rate limit exceeded"},
                        "additional_usage_values": {"compression_saved_tokens": -3, "compression_savings_spend": -0.1},
                        "usage_object": {"prompt_tokens_details": {"cached_tokens": 7}},
                    }
                ),
            ),
        )
        try:
            yield conn
        finally:
            conn.rollback()


def run_query(db, **kwargs):
    sql, values = gateway.build_query(query(**kwargs))
    # Prisma uses $n; psycopg uses positional %s. Preserve repeated references.
    bound = [values[int(match[1]) - 1] for match in re.finditer(r"\$(\d+)", sql)]
    sql = re.sub(r"\$\d+", "%s", sql).replace("LIKE '%/%'", "LIKE '%%/%%'")
    row = db.execute(sql, bound).fetchone()
    return dict(zip(("buckets", "records", "records_total", "savings_schema_version"), row, strict=True))


@pytest.mark.parametrize(
    "prices, expected, unknown",
    [
        ({"input_cost_per_token": 0.01, "cache_read_input_token_cost": 0.002}, 0.056, 0),
        ({}, 0, 1),
    ],
)
def test_saved_price_recovery_and_missing_coverage(db, prices, expected, unknown):
    db.execute(
        'UPDATE "LiteLLM_SpendLogs" SET metadata = %s::jsonb',
        (
            json.dumps(
                {
                    "usage_object": {"prompt_tokens_details": {"cached_tokens": 7}},
                    "model_map_information": {"model_map_value": prices},
                }
            ),
        ),
    )
    result = run_query(db, include_unowned=True, filters={"request_id": "request-1"})
    assert result["savings_schema_version"] == 1
    metrics = result["buckets"][0]["metrics"]
    assert metrics["prompt_caching_savings_spend"] == pytest.approx(expected)
    assert metrics["cache_unpriced_requests"] == unknown
    assert metrics["compression_unpriced_requests"] == 0


@pytest.mark.parametrize(
    ("key", "matching", "excluded"),
    [
        ("search", "中文", "no-match"),
        ("search", "%_", "%X"),
        ("model_group", "group-a", "other-group"),
        ("session_id", "sion-%_a", "session-X"),
        ("key_alias", "%_alpha", "old-alias"),
        ("provider", "siliconflow", "openai"),
        ("model", "model-a", "model-b"),
        ("model_id", "deployment-a", "deployment-b"),
        ("api_key", "hash-a", "hash-b"),
        ("end_user", "customer-a", "customer-b"),
        ("request_id", "request-1", "request-nope"),
        ("call_type", "acompletion", "embedding"),
        ("status_filter", "failure", "success"),
        ("error_code", "429", "500"),
        ("error_message", "RATE LIMIT", "timeout"),
        ("min_spend", "0.25", "0.26"),
        ("max_spend", "0.25", "0"),
        ("min_tokens", "30", "31"),
        ("max_tokens", "30", "0"),
        ("min_duration_ms", "1000", "1001"),
        ("max_duration_ms", "1000", "0"),
    ],
)
def test_predicates_filter_before_count_and_pagination(db, key, matching, excluded):
    result = run_query(db, filters={key: matching}, user_ids=["alice"], include_records=True, page=2, page_size=2)
    assert result["records_total"] == (1 if key == "request_id" else 10005)
    assert len(result["records"]) == (0 if key == "request_id" else 2)
    assert run_query(db, filters={key: excluded}, include_unowned=True)["records_total"] == 0


def test_combined_filters_scope_totals_timezone_and_live_inserts(db):
    filters = {"search": "中文", "model_group": "group-a", "session_id": "abc", "key_alias": "alpha"}
    result = run_query(db, filters=filters, user_ids=["alice"], timezone=-480, include_records=True)
    assert result["records_total"] == 10005
    bucket = result["buckets"][0]
    assert bucket["day"] == "2026-09-23"
    assert bucket["metrics"]["spend"] == pytest.approx(2501.25)
    assert bucket["metrics"]["compression_saved_tokens"] == -30015
    assert bucket["metrics"]["compression_savings_spend"] == pytest.approx(-1000.5)
    assert bucket["metrics"]["cache_read_input_tokens"] == 70035
    assert bucket["metrics"]["failed_requests"] == 10005
    assert all(
        set(row)
        == {
            "request_id",
            "startTime",
            "model",
            "model_group",
            "model_id",
            "actual_model",
            "provider",
            "key_alias",
            "user_id",
            "status",
            "spend",
            "total_tokens",
        }
        for row in result["records"]
    )
    assert run_query(db, filters=filters, user_ids=["bob"])["records_total"] == 0
    assert run_query(db, filters=filters)["records_total"] == 0
    db.execute("DELETE FROM \"LiteLLM_SpendLogs\" WHERE request_id = 'request-1'")
    later = run_query(db, filters=filters, include_unowned=True, page=2, include_records=True)
    assert later["records_total"] == 10004
    assert later["buckets"][0]["metrics"]["api_requests"] == 10004


def test_literal_injection_is_a_bound_value(db):
    assert run_query(db, filters={"search": "' OR 1=1 --"}, include_unowned=True)["records_total"] == 0


@pytest.mark.parametrize("metadata", [None, [], "[]", "null", "invalid json", '{"agentos_key_alias":"legacy"}'])
def test_legacy_or_nonobject_metadata_does_not_break_usage(db, metadata):
    db.execute(
        "UPDATE \"LiteLLM_SpendLogs\" SET metadata = %s::jsonb WHERE request_id = 'request-1'", (json.dumps(metadata),)
    )
    result = run_query(db, filters={"request_id": "request-1"}, include_unowned=True, include_records=True)
    assert result["records_total"] == 1
    assert result["records"][0]["key_alias"] == ("legacy" if metadata == '{"agentos_key_alias":"legacy"}' else "")


@pytest.mark.parametrize(
    "filters",
    [
        {"unknown": "1"},
        {"min_tokens": "nan"},
        {"max_spend": "-1"},
        {"min_spend": "2", "max_spend": "1"},
        {"status_filter": "wrong"},
    ],
)
def test_invalid_filters_are_rejected(filters):
    with pytest.raises(ValueError):
        query(filters=filters)


def test_query_parameters_never_become_sql():
    sql, values = gateway.build_query(query(filters={"search": "private needle"}, user_ids=["alice"]))
    assert "private needle" not in sql
    assert "private needle" in values
    assert "alice" not in sql
    assert "MATERIALIZED" in sql


@pytest.mark.asyncio
async def test_internal_route_requires_master_key_and_uses_one_statement(monkeypatch):
    from types import ModuleType, SimpleNamespace
    from unittest.mock import AsyncMock

    proxy = ModuleType("litellm.proxy.proxy_server")
    proxy.__dict__["master_key"] = "test-master"
    execute = AsyncMock(return_value=[{"buckets": "[]", "records": "[]", "records_total": 0}])
    proxy.__dict__["prisma_client"] = SimpleNamespace(db=SimpleNamespace(query_raw=execute))
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy)
    app = FastAPI()
    app.include_router(gateway.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://gateway") as client:
        for auth in (None, "Bearer virtual-key"):
            response = await client.post(
                "/agentos/usage/query",
                json=query().model_dump(mode="json"),
                headers={"Authorization": auth} if auth else {},
            )
            assert response.status_code == 403
        execute.assert_not_called()
        response = await client.post(
            "/agentos/usage/query",
            json=query().model_dump(mode="json"),
            headers={"Authorization": "Bearer test-master"},
        )
        assert response.status_code == 200
        assert response.json() == {"buckets": [], "records": [], "records_total": 0}
        execute.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("include_unowned", [False, True])
@pytest.mark.parametrize("dates", [{}, {"start_date": "2026-09-22", "end_date": "2026-09-28"}])
async def test_daily_usage_is_one_native_call_with_authorized_user_scope(monkeypatch, include_unowned, dates):
    from types import ModuleType
    from unittest.mock import AsyncMock

    proxy = ModuleType("litellm.proxy.proxy_server")
    proxy.__dict__.update(master_key="test-master", prisma_client=object())
    common = ModuleType("litellm.proxy.management_endpoints.common_daily_activity")
    fetch = AsyncMock(return_value={"results": [], "metadata": {}})
    common.__dict__["get_daily_activity_aggregated"] = fetch
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy)
    monkeypatch.setitem(sys.modules, "litellm.proxy.management_endpoints.common_daily_activity", common)
    app = FastAPI()
    app.include_router(gateway.router)
    body = {
        **dates,
        "timezone": -480,
        "user_ids": ["alice", "bob"],
        "include_unowned": include_unowned,
        "model": "model-a",
        "api_key": "hash-a",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://gateway") as client:
        forbidden = await client.post("/agentos/usage/daily", json=body)
        assert forbidden.status_code == 403
        fetch.assert_not_called()
        response = await client.post("/agentos/usage/daily", json=body, headers={"Authorization": "Bearer test-master"})
    assert response.status_code == 200
    fetch.assert_awaited_once()
    assert fetch.await_args is not None
    params = fetch.await_args.kwargs
    assert params["entity_id"] == (None if include_unowned else ["alice", "bob"])
    assert params["start_date"] == dates.get("start_date") and params["end_date"] == dates.get("end_date")
    assert "include_entity_breakdown" not in params
    assert params["model"] == "model-a" and params["api_key"] == "hash-a"
    assert params["timezone_offset_minutes"] == -480


@pytest.mark.parametrize("start,end", [("2026-09-29", "2026-09-22"), ("2025-09-22", "2026-09-23")])
def test_daily_usage_rejects_invalid_date_ranges(start, end):
    with pytest.raises(ValueError):
        gateway.DailyUsageQuery(start_date=start, end_date=end)


def test_actual_model_filter_is_bound_and_distinct_from_requested_model():
    sql, values = gateway.build_query(
        query(
            filters={"actual_model": "dashscope/qwen3.8-flash"},
            deployment_models={"deployment-a": "dashscope/qwen3.8-flash"},
        )
    )
    assert "actual_model = $" in sql
    assert "actual_model" in sql.split("GROUP BY")[-1]
    assert "dashscope/qwen3.8-flash" not in sql
    assert "dashscope/qwen3.8-flash" in values


def test_actual_model_metadata_and_retained_identity_do_not_use_group_name(db):
    db.execute("UPDATE \"LiteLLM_SpendLogs\" SET status='success'")
    result = run_query(
        db,
        include_unowned=True,
        deployment_models={"deployment-a": "dashscope/qwen3.8-flash"},
        filters={"actual_model": "dashscope/qwen3.8-flash"},
    )
    assert result["records_total"] == 10005
    assert result["buckets"][0]["actual_model"] == "dashscope/qwen3.8-flash"
    assert run_query(db, include_unowned=True, filters={"actual_model": "model-a"})["records_total"] == 0
    db.execute(
        'UPDATE "LiteLLM_SpendLogs" SET metadata=jsonb_set(metadata,\'{model_info}\',\'{"agentos_upstream_model":"siliconflow/actual-model"}\')'
    )
    result = run_query(
        db,
        include_unowned=True,
        deployment_models={"deployment-a": "dashscope/qwen3.8-flash"},
        filters={"actual_model": "siliconflow/actual-model"},
    )
    assert result["records_total"] == 10005


def test_request_time_hop_wins_over_changed_deployment(db):
    db.execute(
        'UPDATE "LiteLLM_SpendLogs" SET metadata=%s::jsonb',
        (json.dumps({"spend_logs_metadata": {"agentos_routing_hops": [{"upstream": "dashscope/old-model"}]}}),),
    )
    result = run_query(
        db,
        include_unowned=True,
        deployment_models={"deployment-a": "dashscope/new-model"},
        filters={"actual_model": "dashscope/old-model"},
    )
    assert result["records_total"] == 10005


def test_actual_model_uses_final_hop_after_fallback(db):
    db.execute(
        'UPDATE "LiteLLM_SpendLogs" SET metadata=%s::jsonb',
        (
            json.dumps(
                {
                    "spend_logs_metadata": {
                        "agentos_routing_hops": [
                            {"upstream": "dashscope/failed-model", "outcome": "failure"},
                            {"upstream": "openai/local-model", "outcome": "success"},
                        ]
                    }
                }
            ),
        ),
    )
    result = run_query(db, include_unowned=True, filters={"actual_model": "openai/local-model"})
    assert result["records_total"] == 10005


@pytest.mark.asyncio
async def test_bulk_pricing_uses_native_repository_and_rolls_back_reload_failure(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    before = {"old": {"input_cost_per_token": 1}}
    repository = SimpleNamespace(set_param=AsyncMock())
    native = SimpleNamespace(
        parse_model_cost_map_overrides=lambda value: value,
        _get_prisma_client=lambda: object(),
        load_model_cost_map_overrides=AsyncMock(return_value=before),
        _load_latest_model_cost_map=lambda: {},
        merge_model_cost_map=lambda base, overrides: {**base, **overrides},
        ConfigRepository=lambda prisma: repository,
        _CONFIG_NAME="model_cost_map_overrides",
        _schedule_cluster_reload=AsyncMock(),
        _apply_cost_map=Mock(),
    )
    updates = {"new": {"input_cost_per_token": 2}}
    from litellm.proxy.management_endpoints import model_cost_map_endpoints as endpoint

    for name, value in vars(native).items():
        monkeypatch.setattr(endpoint, name, value)
    assert await endpoint.apply_pricing_import(updates) == 1
    repository.set_param.assert_awaited_once_with("model_cost_map_overrides", {**before, **updates})
    native._apply_cost_map.assert_called_once_with({**before, **updates})
    repository.set_param.reset_mock()
    native._schedule_cluster_reload.side_effect = RuntimeError("reload unavailable")
    with pytest.raises(RuntimeError, match="reload unavailable"):
        await endpoint.apply_pricing_import(updates)
    assert repository.set_param.await_args_list[-1].args == ("model_cost_map_overrides", before)
    native._apply_cost_map.assert_called_with(before)


@pytest.mark.asyncio
async def test_bulk_pricing_rejects_non_master_keys(monkeypatch):
    from types import ModuleType

    proxy = ModuleType("litellm.proxy.proxy_server")
    proxy.__dict__["master_key"] = "sk-test-master"
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy)
    app = FastAPI()
    app.include_router(gateway.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.put(
            "/model/cost_map/overrides/import", json={"data": {"model": {"input_cost_per_token": 1}}}
        )
        assert response.status_code == 403


def test_actual_model_aliases_are_bound_and_intersect_other_filters():
    aliases = ["openai/qwen3.8-flash", "dashscope/qwen3.8-flash", "qwen3.8-flash"]
    sql, values = gateway.build_query(
        query(filters={"actual_model_aliases": json.dumps(aliases), "provider": "dashscope"})
    )
    assert "actual_model = ANY(" in sql
    assert aliases in values
    assert "provider = $" in sql
    assert "openai/qwen3.8-flash" not in sql


@pytest.mark.parametrize("value", ["{}", "[]", "[1]", '[""]', '["a","a"]', '["a","b","c","d","e","f","g","h","i"]'])
def test_actual_model_aliases_reject_invalid_lists(value):
    with pytest.raises(ValueError):
        query(filters={"actual_model_aliases": value})


def test_alias_union_retains_raw_models_and_intersects_provider_in_postgres(db):
    aliases = ["openai/qwen3.8-flash", "dashscope/qwen3.8-flash", "qwen3.8-flash"]
    for request_id, upstream in [
        ("request-1", aliases[0]),
        ("request-2", aliases[1]),
        ("request-3", aliases[2]),
        ("request-4", "other/qwen3.8-flash"),
    ]:
        db.execute(
            'UPDATE "LiteLLM_SpendLogs" SET metadata=%s::jsonb WHERE request_id=%s',
            (json.dumps({"model_info": {"agentos_upstream_model": upstream}}), request_id),
        )
    result = run_query(db, include_unowned=True, filters={"actual_model_aliases": json.dumps(aliases)})
    assert result["records_total"] == 3
    assert {bucket["actual_model"] for bucket in result["buckets"]} == set(aliases)
    assert (
        run_query(
            db,
            include_unowned=True,
            filters={"actual_model_aliases": json.dumps(aliases), "provider": "no-such-provider"},
        )["records_total"]
        == 0
    )
