import asyncio
from collections.abc import Awaitable, Callable
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from mcp.types import CallToolRequestParams, CallToolResult, TextContent
from pydantic import BaseModel, Field, TypeAdapter

from litellm.types.mcp import MCPServerCostInfo


class QCCEnvironmentVariable(BaseModel):
    name: str
    description: str
    secret: bool


class QCCServerPreset(BaseModel):
    name: str
    title: str
    description: str
    category: str
    transport: Literal["http"]
    url: str
    auth_type: Literal["bearer_token"]
    env_vars: tuple[QCCEnvironmentVariable, ...]
    mcp_server_cost_info: MCPServerCostInfo
    timeout: float = 300


class QCCCatalog(BaseModel):
    verified_at: str
    usd_per_credit: float
    servers: tuple[QCCServerPreset, ...]


@lru_cache(maxsize=1)
def get_qcc_catalog() -> QCCCatalog:
    return QCCCatalog.model_validate_json(
        (Path(__file__).parents[2] / "qcc_mcp_catalog.json").read_text(encoding="utf-8")
    )


def get_qcc_preset(url: str | None) -> QCCServerPreset | None:
    if not url:
        return None
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc.lower() != "agent.qcc.com" or parsed.query or parsed.fragment:
        return None
    return next(
        (server for server in get_qcc_catalog().servers if url.rstrip("/") == server.url),
        None,
    )


def resolve_qcc_cost_info(url: str | None, configured: object) -> MCPServerCostInfo | None:
    preset = get_qcc_preset(url)
    if preset is None:
        return TypeAdapter(MCPServerCostInfo).validate_python(configured) if configured is not None else None
    overrides = TypeAdapter(MCPServerCostInfo).validate_python(configured or {})
    return {
        **preset.mcp_server_cost_info,
        **overrides,
        "tool_name_to_cost_per_query": {
            **(preset.mcp_server_cost_info.get("tool_name_to_cost_per_query") or {}),
            **(overrides.get("tool_name_to_cost_per_query") or {}),
        },
    }


def is_qcc_free_result(response: object, server_resource: str | None) -> bool:
    if server_resource != "https://agent.qcc.com":
        return False
    if isinstance(response, CallToolResult):
        extra = (response.model_extra or {}).get("_extra")
    elif isinstance(response, dict):
        extra = response.get("_extra")
    else:
        return False
    if not isinstance(extra, dict):
        return False
    settlement = extra.get("settlement")
    return isinstance(settlement, str) and settlement.startswith("free_") and len(settlement) > len("free_")


class QCCDocumentDetail(BaseModel):
    total_pages: int = Field(ge=1, strict=True)


class QCCDocumentResponse(BaseModel):
    task_id: str | None = None
    status: Literal["processing", "success", "failed"]
    details: tuple[QCCDocumentDetail, ...] = ()


def _read_document_response(result: CallToolResult) -> QCCDocumentResponse:
    if result.structuredContent is not None:
        return QCCDocumentResponse.model_validate(result.structuredContent)
    blocks = tuple(block for block in result.content if isinstance(block, TextContent))
    if len(blocks) != 1:
        raise ValueError("QCC document response must contain one JSON text block")
    return QCCDocumentResponse.model_validate_json(blocks[0].text)


async def call_qcc_document_tool(
    call_tool: Callable[[CallToolRequestParams], Awaitable[CallToolResult]],
    params: CallToolRequestParams,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> CallToolResult:
    result = await call_tool(
        params.model_copy(update={"arguments": {**(params.arguments or {}), "wait": True}})
        if params.name == "parse_document"
        else params
    )
    if result.isError or is_qcc_free_result(result, "https://agent.qcc.com"):
        return result
    try:
        document = _read_document_response(result)
    except ValueError:
        return CallToolResult(
            isError=True,
            content=[
                TextContent(type="text", text="QCC returned an invalid document response; billing usage is unavailable")
            ],
        )
    if document.status == "failed":
        return result.model_copy(update={"isError": True})
    if params.name != "parse_document":
        return result
    if document.status == "success":
        if document.details:
            return result
        return CallToolResult(
            isError=True,
            content=[
                TextContent(type="text", text="QCC returned no document page count; billing usage is unavailable")
            ],
        )
    if not document.task_id:
        return CallToolResult(
            isError=True,
            content=[TextContent(type="text", text="QCC returned a processing document without a task_id")],
        )
    await sleep(2)
    completed = await _wait_for_qcc_document(call_tool, document.task_id, sleep)
    submitted_extra = (result.model_extra or {}).get("_extra")
    return CallToolResult.model_validate(
        {
            **completed.model_dump(by_alias=True, exclude={"_extra"}),
            **({"_extra": submitted_extra} if submitted_extra is not None else {}),
        }
    )


async def _wait_for_qcc_document(
    call_tool: Callable[[CallToolRequestParams], Awaitable[CallToolResult]],
    task_id: str,
    sleep: Callable[[float], Awaitable[None]],
) -> CallToolResult:
    while True:
        result = await call_qcc_document_tool(
            call_tool, CallToolRequestParams(name="get_parse_result", arguments={"task_id": task_id}), sleep
        )
        if result.isError:
            return result
        document = _read_document_response(result)
        if document.task_id != task_id:
            return CallToolResult(
                isError=True, content=[TextContent(type="text", text="QCC returned a mismatched document task_id")]
            )
        if document.status == "success":
            if document.details:
                return result
            return CallToolResult(
                isError=True, content=[TextContent(type="text", text="QCC returned no document page count")]
            )
        await sleep(2)
