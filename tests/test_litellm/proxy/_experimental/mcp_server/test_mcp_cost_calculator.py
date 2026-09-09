import json
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))  # Adds the parent directory to the system path

from litellm.proxy._experimental.mcp_server.cost_calculator import MCPCostCalculator


class TestMCPCostCalculator:
    def test_calculate_mcp_tool_call_cost_none_logging_obj(self):
        """Test that when litellm_logging_obj is None, it returns 0.0"""
        result = MCPCostCalculator.calculate_mcp_tool_call_cost(None)
        assert result == 0.0

    def test_calculate_mcp_tool_call_cost_with_tool_specific_cost(self):
        """Test that when a specific tool has a defined cost, it returns that cost"""
        # Mock the litellm_logging_obj
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {
            "mcp_tool_call_metadata": {
                "name": "search_web",
                "mcp_server_cost_info": {
                    "default_cost_per_query": 0.01,
                    "tool_name_to_cost_per_query": {
                        "search_web": 0.05,
                        "generate_code": 0.03,
                    },
                },
            }
        }

        result = MCPCostCalculator.calculate_mcp_tool_call_cost(mock_logging_obj)
        assert result == 0.05

    def test_calculate_mcp_tool_call_cost_with_default_cost(self):
        """Test that when no tool-specific cost is found, it falls back to default cost"""
        # Mock the litellm_logging_obj
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {
            "mcp_tool_call_metadata": {
                "name": "unknown_tool",
                "mcp_server_cost_info": {
                    "default_cost_per_query": 0.02,
                    "tool_name_to_cost_per_query": {"search_web": 0.05},
                },
            }
        }

        result = MCPCostCalculator.calculate_mcp_tool_call_cost(mock_logging_obj)
        assert result == 0.02

    def test_calculate_mcp_tool_call_cost_no_cost_configuration(self):
        """Test that when no cost configuration is provided, it returns 0.0"""
        # Mock the litellm_logging_obj with minimal metadata
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {
            "mcp_tool_call_metadata": {"name": "some_tool", "mcp_server_cost_info": {}}
        }

        result = MCPCostCalculator.calculate_mcp_tool_call_cost(mock_logging_obj)
        assert result == 0.0

    def test_calculate_mcp_tool_call_cost_empty_metadata(self):
        """Test that when metadata is empty or missing, it returns 0.0"""
        # Mock the litellm_logging_obj with empty model_call_details
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}

        result = MCPCostCalculator.calculate_mcp_tool_call_cost(mock_logging_obj)
        assert result == 0.0


@pytest.mark.parametrize("tool_price, expected", [(None, 2.0), (0, 0.0), ("7e-05", 0.00007)])
def test_tool_price_null_zero_and_yaml_scientific_notation(tool_price, expected):
    logging_obj = MagicMock(
        model_call_details={
            "mcp_tool_call_metadata": {
                "name": "lookup",
                "mcp_server_cost_info": {
                    "default_cost_per_query": 2,
                    "tool_name_to_cost_per_query": {"lookup": tool_price},
                },
            },
        }
    )
    assert MCPCostCalculator.calculate_mcp_tool_call_cost(logging_obj) == expected


@pytest.mark.parametrize("price", [-1, float("nan"), float("inf"), True, "invalid"])
@pytest.mark.parametrize("field", ["default_cost_per_query", "tool_name_to_cost_per_query"])
def test_invalid_mcp_prices_rejected_before_persistence(price, field):
    from pydantic import ValidationError
    from litellm.proxy._types import NewMCPServerRequest, UpdateMCPServerRequest
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import _normalize_mcp_server_cost_info

    cost = {field: {"lookup": price} if field == "tool_name_to_cost_per_query" else price}
    payload = {
        "server_id": "test",
        "url": "https://example.com/mcp",
        "transport": "http",
        "mcp_info": {"mcp_server_cost_info": cost},
    }
    for request_type in (NewMCPServerRequest, UpdateMCPServerRequest):
        with pytest.raises(ValidationError):
            request_type(**payload)
    with pytest.raises(ValidationError):
        _normalize_mcp_server_cost_info(payload["mcp_info"])


@pytest.mark.parametrize("structured", [False, True])
def test_usage_billing_flows_through_litellm_completion_cost(structured):
    from mcp.types import CallToolResult, TextContent
    from litellm import completion_cost

    payload = {"details": [{"total_pages": 2}, {"total_pages": 3}]}
    result = CallToolResult(
        content=[] if structured else [TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload if structured else None,
    )
    logging_obj = MagicMock(
        model_call_details={
            "mcp_tool_call_metadata": {
                "name": "parse_document",
                "mcp_server_cost_info": {
                    "tool_name_to_cost_per_unit": {
                        "parse_document": {"unit": "page", "cost_per_unit": 1, "usage_path": "details.*.total_pages"},
                    }
                },
            }
        }
    )
    assert (
        completion_cost(
            model="MCP: qcc_document-parse_document",
            completion_response=result,
            call_type="call_mcp_tool",
            litellm_logging_obj=logging_obj,
        )
        == 5.0
    )


@pytest.mark.parametrize(
    "payload", [{}, {"details": []}, {"details": [{"total_pages": -1}]}, {"details": [{"total_pages": True}]}]
)
def test_missing_or_invalid_usage_is_not_silently_free(payload):
    from mcp.types import CallToolResult, TextContent
    from litellm.proxy._experimental.mcp_server.cost_calculator import get_mcp_usage_quantity

    with pytest.raises(ValueError):
        get_mcp_usage_quantity(
            CallToolResult(content=[TextContent(type="text", text=json.dumps(payload))]), "details.*.total_pages"
        )


def test_failed_result_is_free_even_with_custom_cost_hook():
    from mcp.types import CallToolResult, TextContent

    logging_obj = MagicMock(model_call_details={"response_cost": 99})
    result = CallToolResult(isError=True, content=[TextContent(type="text", text="failed")])
    assert MCPCostCalculator.calculate_mcp_tool_call_cost(logging_obj, result) == 0


def test_explicit_fixed_price_overrides_usage_rate_and_hook_overrides_fixed_price():
    logging_obj = MagicMock(
        model_call_details={
            "mcp_tool_call_metadata": {
                "name": "parse_document",
                "mcp_server_cost_info": {
                    "tool_name_to_cost_per_query": {"parse_document": 0},
                    "tool_name_to_cost_per_unit": {
                        "parse_document": {"unit": "page", "cost_per_unit": 1, "usage_path": "pages"}
                    },
                },
            }
        }
    )
    assert MCPCostCalculator.calculate_mcp_tool_call_cost(logging_obj) == 0
    logging_obj.model_call_details["response_cost"] = 7
    assert MCPCostCalculator.calculate_mcp_tool_call_cost(logging_obj) == 7


def test_require_tool_pricing_allows_free_and_metered_tools_but_rejects_unknown():
    cost = {
        "require_tool_pricing": True,
        "tool_name_to_cost_per_query": {"free": 0},
        "tool_name_to_cost_per_unit": {"parse": {"unit": "page", "cost_per_unit": 1, "usage_path": "pages"}},
    }
    assert MCPCostCalculator.is_tool_priced(cost, "free")
    assert MCPCostCalculator.is_tool_priced(cost, "parse")
    assert not MCPCostCalculator.is_tool_priced(cost, "unknown")
    assert MCPCostCalculator.is_tool_priced({**cost, "default_cost_per_query": 0}, "unknown")


def test_invalid_usage_result_is_an_mcp_failure_before_success_logging():
    from mcp.types import CallToolResult, TextContent
    from litellm.proxy._experimental.mcp_server.cost_calculator import validate_mcp_result_usage

    result = CallToolResult(content=[TextContent(type="text", text='{"status":"success"}')])
    pricing = {"tool_name_to_cost_per_unit": {"parse": {"cost_per_unit": 1, "unit": "page", "usage_path": "pages"}}}
    assert validate_mcp_result_usage(result, pricing, "parse").isError
    assert result.isError is False
    assert (
        validate_mcp_result_usage(result, {**pricing, "tool_name_to_cost_per_query": {"parse": 0}}, "parse") is result
    )


@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("pricing", [{"default_cost_per_query": 5}, {"tool_name_to_cost_per_query": {"lookup": 5}}])
def test_qcc_free_settlement_overrides_fixed_prices_and_response_cost(as_dict, pricing):
    from litellm import completion_cost
    from mcp.types import CallToolResult, TextContent

    response = CallToolResult(
        content=[TextContent(type="text", text='{"无匹配项":"未匹配到搜索关键词"}')],
        isError=False,
        _extra={"settlement": "free_no_match"},
    )
    logging_obj = MagicMock(
        model_call_details={
            "response_cost": 9,
            "mcp_tool_call_metadata": {
                "name": "lookup",
                "mcp_server_resource": "https://agent.qcc.com",
                "mcp_server_cost_info": pricing,
            },
        }
    )
    assert (
        completion_cost(
            model="MCP: lookup",
            completion_response=response.model_dump() if as_dict else response,
            call_type="call_mcp_tool",
            litellm_logging_obj=logging_obj,
        )
        == 0
    )


@pytest.mark.parametrize("resource", [None, "https://other.example", "https://agent.qcc.com.evil.example"])
def test_qcc_free_marker_cannot_waive_other_providers_prices(resource):
    logging_obj = MagicMock(
        model_call_details={
            "mcp_tool_call_metadata": {
                "name": "lookup",
                "mcp_server_resource": resource,
                "mcp_server_cost_info": {"default_cost_per_query": 5},
            },
        }
    )
    assert (
        MCPCostCalculator.calculate_mcp_tool_call_cost(
            logging_obj, {"isError": False, "_extra": {"settlement": "free_no_match"}}
        )
        == 5
    )


@pytest.mark.parametrize("extra", [None, {}, {"settlement": "charged"}, {"cached": True}, {"settlement": False}])
def test_qcc_empty_or_cached_result_without_free_settlement_keeps_price(extra):
    logging_obj = MagicMock(
        model_call_details={
            "mcp_tool_call_metadata": {
                "name": "lookup",
                "mcp_server_resource": "https://agent.qcc.com",
                "mcp_server_cost_info": {"default_cost_per_query": 5},
            },
        }
    )
    assert (
        MCPCostCalculator.calculate_mcp_tool_call_cost(logging_obj, {"isError": False, "content": [], "_extra": extra})
        == 5
    )


def test_qcc_free_metered_result_needs_no_usage_and_keeps_original_response():
    from mcp.types import CallToolResult, TextContent
    from litellm.proxy._experimental.mcp_server.cost_calculator import validate_mcp_result_usage

    result = CallToolResult(content=[TextContent(type="text", text="no match")], _extra={"settlement": "free_no_match"})
    pricing = {"tool_name_to_cost_per_unit": {"parse": {"cost_per_unit": 1, "unit": "page", "usage_path": "pages"}}}
    assert validate_mcp_result_usage(result, pricing, "parse", "https://agent.qcc.com") is result
    assert validate_mcp_result_usage(result, pricing, "parse", "https://other.example").isError
    logging_obj = MagicMock(
        model_call_details={
            "mcp_tool_call_metadata": {
                "name": "parse",
                "mcp_server_resource": "https://agent.qcc.com",
                "mcp_server_cost_info": pricing,
            }
        }
    )
    assert MCPCostCalculator.calculate_mcp_tool_call_cost(logging_obj, result) == 0
