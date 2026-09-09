import json
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolRequestParams, CallToolResult, TextContent

from litellm.proxy._experimental.mcp_server.qcc import (
    call_qcc_document_tool,
    get_qcc_catalog,
    get_qcc_preset,
    is_qcc_free_result,
    resolve_qcc_cost_info,
)
from litellm.proxy._experimental.mcp_server.mcp_server_manager import _normalize_mcp_server_cost_info


def document_result(status, pages=None, task_id="document-task"):
    payload = {"task_id": task_id, "status": status}
    if pages is not None:
        payload["details"] = [{"total_pages": pages, "result_md": "document text"}]
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(payload))])


def test_qcc_catalog_covers_all_ten_services_and_one_dollar_per_credit():
    catalog = get_qcc_catalog()
    assert catalog.usd_per_credit == 1
    assert len(catalog.servers) == 10
    assert sum(len(s.mcp_server_cost_info.get("tool_name_to_cost_per_query") or {}) for s in catalog.servers) == 202
    company = get_qcc_preset("https://agent.qcc.com/mcp/company/stream")
    assert company.mcp_server_cost_info["tool_name_to_cost_per_query"]["get_company_registration_info"] == 3
    assert company.mcp_server_cost_info["tool_name_to_cost_per_query"]["get_shareholder_info"] == 20
    assert all(s.auth_type == "bearer_token" and s.transport == "http" for s in catalog.servers)
    assert all(s.mcp_server_cost_info["require_tool_pricing"] for s in catalog.servers)


@pytest.mark.parametrize(
    "url",
    [
        "https://agent.qcc.com.evil.example/mcp/company/stream",
        "http://agent.qcc.com/mcp/company/stream",
        "https://other.example/mcp/company/stream",
        "https://agent.qcc.com/mcp/company/stream?api_key=x",
    ],
)
def test_qcc_pricing_not_applied_to_unrelated_servers(url):
    assert get_qcc_preset(url) is None
    assert resolve_qcc_cost_info(url, None) is None


def test_qcc_config_load_keeps_custom_metadata_and_explicit_zero_price():
    info = {
        "owner": "test",
        "mcp_server_cost_info": {"tool_name_to_cost_per_query": {"get_company_registration_info": 0}},
    }
    _normalize_mcp_server_cost_info(info, "https://agent.qcc.com/mcp/company/stream")
    assert info["owner"] == "test"
    assert info["mcp_server_cost_info"]["tool_name_to_cost_per_query"]["get_company_registration_info"] == 0
    assert info["mcp_server_cost_info"]["tool_name_to_cost_per_query"]["get_shareholder_info"] == 20


@pytest.mark.asyncio
async def test_async_document_is_submitted_once_and_waited_for_actual_pages():
    completed = document_result("success", pages=4)
    call = AsyncMock(side_effect=[document_result("processing"), document_result("processing"), completed])
    sleep = AsyncMock()
    result = await call_qcc_document_tool(
        call,
        CallToolRequestParams(
            name="parse_document", arguments={"file_url": "https://example.com/document.pdf", "wait": False}
        ),
        sleep,
    )
    assert result == completed
    assert [args.args[0].name for args in call.await_args_list] == [
        "parse_document",
        "get_parse_result",
        "get_parse_result",
    ]
    assert call.await_args_list[0].args[0].arguments["wait"] is True
    assert call.await_args_list[1].args[0].arguments == {"task_id": "document-task"}
    assert sleep.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        document_result("failed"),
        document_result("success"),
        document_result("success", pages=-1),
        document_result("processing", task_id=None),
    ],
)
async def test_document_failure_or_missing_usage_never_becomes_billable_success(result):
    call = AsyncMock(return_value=result)
    response = await call_qcc_document_tool(call, CallToolRequestParams(name="parse_document"), AsyncMock())
    assert response.isError is True
    assert call.await_count == 1


@pytest.mark.asyncio
async def test_query_result_does_not_start_another_job_or_poll():
    result = document_result("processing")
    call = AsyncMock(return_value=result)
    response = await call_qcc_document_tool(
        call, CallToolRequestParams(name="get_parse_result", arguments={"task_id": "document-task"}), AsyncMock()
    )
    assert response is result
    assert call.await_count == 1
    assert "wait" not in call.await_args.args[0].arguments


@pytest.mark.asyncio
async def test_wait_rejects_result_for_another_document():
    call = AsyncMock(
        side_effect=[document_result("processing"), document_result("success", pages=99, task_id="other-task")]
    )
    result = await call_qcc_document_tool(call, CallToolRequestParams(name="parse_document"), AsyncMock())
    assert result.isError is True


@pytest.mark.asyncio
async def test_document_polling_cancels_with_request_timeout():
    import asyncio

    call = AsyncMock(return_value=document_result("processing"))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(call_qcc_document_tool(call, CallToolRequestParams(name="parse_document")), timeout=0.01)
    assert call.await_count == 1


def test_qcc_null_override_uses_default_instead_of_restoring_catalog_price():
    configured = {"default_cost_per_query": 2, "tool_name_to_cost_per_query": {"get_company_profile": None}}
    pricing = resolve_qcc_cost_info("https://agent.qcc.com/mcp/company/stream", configured)
    assert pricing["tool_name_to_cost_per_query"]["get_company_profile"] is None
    assert pricing["default_cost_per_query"] == 2


@pytest.mark.parametrize("settlement", ["free_no_match", "free_additional_reason"])
def test_qcc_explicit_free_settlement_family(settlement):
    assert is_qcc_free_result({"_extra": {"settlement": settlement}}, "https://agent.qcc.com")


@pytest.mark.parametrize(
    "response",
    [
        {"_extra": {"settlement": "free_"}},
        {"_extra": {"settlement": ["free_no_match"]}},
        {"_extra": "free_no_match"},
        {"content": [{"type": "text", "text": '{"_extra":{"settlement":"free_no_match"}}'}]},
        {"structuredContent": {"_extra": {"settlement": "free_no_match"}}},
    ],
)
def test_qcc_only_uses_top_level_machine_settlement(response):
    assert not is_qcc_free_result(response, "https://agent.qcc.com")


@pytest.mark.asyncio
async def test_free_document_result_is_preserved_without_pages_or_polling():
    result = CallToolResult(content=[], _extra={"settlement": "free_no_match"})
    call = AsyncMock(return_value=result)
    response = await call_qcc_document_tool(call, CallToolRequestParams(name="parse_document"), AsyncMock())
    assert response is result
    assert not response.isError
    assert call.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("submitted_extra", [None, {"settlement": "charged"}])
async def test_free_result_lookup_does_not_waive_original_document_parse(submitted_extra):
    from types import SimpleNamespace
    from litellm.proxy._experimental.mcp_server.cost_calculator import MCPCostCalculator

    submitted = document_result("processing")
    if submitted_extra is not None:
        submitted = submitted.model_copy(update={"_extra": submitted_extra})
    completed = document_result("success", pages=4).model_copy(update={"_extra": {"settlement": "free_result_lookup"}})
    call = AsyncMock(side_effect=[submitted, completed])
    result = await call_qcc_document_tool(call, CallToolRequestParams(name="parse_document"), AsyncMock())
    assert (result.model_extra or {}).get("_extra") == submitted_extra
    assert not is_qcc_free_result(result, "https://agent.qcc.com")
    assert is_qcc_free_result(completed, "https://agent.qcc.com")
    logging_obj = SimpleNamespace(
        model_call_details={
            "mcp_tool_call_metadata": {
                "name": "parse_document",
                "mcp_server_resource": "https://agent.qcc.com",
                "mcp_server_cost_info": get_qcc_preset(
                    "https://agent.qcc.com/mcp/document/stream"
                ).mcp_server_cost_info,
            }
        }
    )
    assert MCPCostCalculator.calculate_mcp_tool_call_cost(logging_obj, result) == 4
