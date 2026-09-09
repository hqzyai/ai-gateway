"""
Cost calculator for MCP tools.
"""

from typing import TYPE_CHECKING, Any, Optional, cast

from pydantic import JsonValue, TypeAdapter

from litellm.types.mcp import MCPServerCostInfo, MCPToolPrice
from litellm.types.utils import StandardLoggingMCPToolCall


def _mcp_result_payload(response: object) -> JsonValue:
    from mcp.types import CallToolResult, TextContent

    result = response if isinstance(response, CallToolResult) else CallToolResult.model_validate(response)
    if result.structuredContent is not None:
        return TypeAdapter(JsonValue).validate_python(result.structuredContent)
    text_content = tuple(block for block in result.content if isinstance(block, TextContent))
    if len(text_content) != 1:
        raise ValueError("Metered MCP billing requires structuredContent or one JSON text result")
    return TypeAdapter(JsonValue).validate_json(text_content[0].text)


def _usage_values(value: JsonValue, path: tuple[str, ...]) -> tuple[float, ...]:
    if not path:
        return (TypeAdapter(MCPToolPrice).validate_python(value),)
    key, *remaining = path
    if key == "*" and isinstance(value, list) and value:
        return tuple(quantity for item in value for quantity in _usage_values(item, tuple(remaining)))
    if isinstance(value, dict) and key in value:
        return _usage_values(value[key], tuple(remaining))
    raise ValueError("MCP billing usage is missing from the configured response path")


def get_mcp_usage_quantity(response: object, usage_path: str) -> float:
    quantity = sum(_usage_values(_mcp_result_payload(response), tuple(usage_path.split("."))))
    return TypeAdapter(MCPToolPrice).validate_python(quantity)


if TYPE_CHECKING:
    from mcp.types import CallToolResult

    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LitellmLoggingObject,
    )
else:
    LitellmLoggingObject = Any


def validate_mcp_result_usage(
    result: "CallToolResult", cost_info: object, tool_name: str, server_resource: str | None = None
) -> "CallToolResult":
    from mcp.types import TextContent

    from litellm.proxy._experimental.mcp_server.qcc import is_qcc_free_result

    if is_qcc_free_result(result, server_resource):
        return result
    config = TypeAdapter(MCPServerCostInfo).validate_python(cost_info or {})
    unit_cost = (config.get("tool_name_to_cost_per_unit") or {}).get(tool_name)
    fixed_price = (config.get("tool_name_to_cost_per_query") or {}).get(tool_name)
    if unit_cost is None or fixed_price is not None or result.isError:
        return result
    try:
        TypeAdapter(MCPToolPrice).validate_python(
            unit_cost["cost_per_unit"] * get_mcp_usage_quantity(result, unit_cost["usage_path"])
        )
    except ValueError:
        return result.model_copy(
            update={
                "isError": True,
                "content": [TextContent(type="text", text="MCP tool returned missing or invalid billing usage")],
            }
        )
    return result


class MCPCostCalculator:
    @staticmethod
    def is_tool_priced(cost_info: object, tool_name: str) -> bool:
        config = TypeAdapter(MCPServerCostInfo).validate_python(cost_info or {})
        if not config.get("require_tool_pricing"):
            return True
        return (
            (config.get("tool_name_to_cost_per_query") or {}).get(tool_name) is not None
            or tool_name in (config.get("tool_name_to_cost_per_unit") or {})
            or config.get("default_cost_per_query") is not None
        )

    @staticmethod
    def calculate_mcp_tool_call_cost(
        litellm_logging_obj: Optional[LitellmLoggingObject],
        completion_response: object = None,
    ) -> float:
        """
        Calculate the cost of an MCP tool call.

        Default is 0.0, unless user specifies a custom cost per request for MCP tools.
        """
        if litellm_logging_obj is None:
            return 0.0

        from litellm.proxy._experimental.mcp_server.utils import extract_mcp_tool_result_error_message

        if completion_response is not None and extract_mcp_tool_result_error_message(completion_response) is not None:
            return 0.0

        from litellm.proxy._experimental.mcp_server.qcc import is_qcc_free_result

        mcp_tool_call_metadata: StandardLoggingMCPToolCall = (
            cast(StandardLoggingMCPToolCall, litellm_logging_obj.model_call_details.get("mcp_tool_call_metadata")) or {}
        )
        if is_qcc_free_result(completion_response, mcp_tool_call_metadata.get("mcp_server_resource")):
            return 0.0

        #########################################################
        # Get the response cost from logging object model_call_details
        # This is set when a user modifies the response in a post_mcp_tool_call_hook
        #########################################################
        response_cost = litellm_logging_obj.model_call_details.get("response_cost", None)
        if response_cost is not None:
            return TypeAdapter(MCPToolPrice).validate_python(response_cost)

        #########################################################
        # Unpack the mcp_tool_call_metadata
        #########################################################
        mcp_server_cost_info = TypeAdapter(MCPServerCostInfo).validate_python(
            mcp_tool_call_metadata.get("mcp_server_cost_info") or {}
        )
        #########################################################
        # User defined cost per query
        #########################################################
        default_cost_per_query = mcp_server_cost_info.get("default_cost_per_query", None)
        tool_name_to_cost_per_query = mcp_server_cost_info.get("tool_name_to_cost_per_query") or {}
        tool_name = mcp_tool_call_metadata.get("name", "")

        #########################################################
        # 1. If tool_name is in tool_name_to_cost_per_query, use the cost per query
        # 2. If tool_name is not in tool_name_to_cost_per_query, use the default cost per query
        # 3. Default to 0.0 if no cost per query is found
        #########################################################
        tool_cost = tool_name_to_cost_per_query.get(tool_name)
        if tool_cost is not None:
            return tool_cost
        unit_cost = (mcp_server_cost_info.get("tool_name_to_cost_per_unit") or {}).get(tool_name)
        if unit_cost is not None:
            return TypeAdapter(MCPToolPrice).validate_python(
                unit_cost["cost_per_unit"] * get_mcp_usage_quantity(completion_response, unit_cost["usage_path"])
            )
        return default_cost_per_query if default_cost_per_query is not None else 0.0
