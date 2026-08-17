"""
Unit tests for the Headroom guardrail.

Tests cover:
- apply_guardrail compresses messages via /v1/compress and returns them as structured_messages
- x-headroom-bypass: true header causes guardrail to skip compression
- missing or empty messages are passed through unchanged
- response-type input is passed through unchanged
- /v1/compress HTTP error raises HTTPException (fail_closed, the default)
- /v1/compress returning malformed JSON raises HTTPException
- /v1/compress non-2xx surfaces as httpx.HTTPStatusError (raise_for_status),
  not a status_code check on the returned response -- both are handled
- unreachable_fallback="fail_open" forwards the request uncompressed instead of raising
- CCR: headroom_retrieve tool injected when compressed messages contain hashes
- CCR: enable_ccr=false compresses without injecting retrieve or running the agentic loop
- CCR: async_should_run_agentic_loop returns True when response has headroom_retrieve tool calls
- CCR: async_build_agentic_loop_plan calls retrieve endpoint and builds follow-up messages
- successful compression logs messages_before/messages_after into guardrail_response
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

import litellm

from litellm.proxy.guardrails.guardrail_hooks.headroom.headroom import (
    HEADROOM_CCR_SYSTEM_INSTRUCTION,
    HeadroomGuardrail,
    extract_hashes_from_messages,
    has_headroom_retrieve_tool,
    prepare_messages_for_compression,
    unwrap_agent_tool_result_content,
    HEADROOM_RETRIEVE_TOOL_NAME,
)
from litellm.types.utils import GenericGuardrailAPIInputs

FAKE_API_BASE = "https://headroom.example.com"
FAKE_API_KEY = "test-key"

ORIGINAL_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "A" * 5000},
]
COMPRESSED_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "A" * 500},
]
COMPRESSED_MESSAGES_WITH_HASH = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": "Summary. Retrieve more: hash=b573993006976af767214fac",
    },
]


def _make_guardrail(**kwargs) -> HeadroomGuardrail:
    defaults = dict(
        api_base=FAKE_API_BASE,
        api_key=FAKE_API_KEY,
        guardrail_name="headroom",
        default_on=True,
    )
    defaults.update(kwargs)
    return HeadroomGuardrail(**defaults)


def _make_compress_response(messages: list, status: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = {
        "messages": messages,
        "tokens_before": 1000,
        "tokens_after": 100,
        "compression_ratio": 0.1,
        "transforms_applied": ["router:smart_crusher:0.35"],
    }
    mock.text = ""
    return mock


def _make_retrieve_response(original_content: str, status: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = {"original_content": original_content}
    mock.text = original_content
    return mock


def _make_openai_response_with_tool_call(tool_name: str, arguments: dict, tool_id: str = "call_abc123") -> MagicMock:
    fn = MagicMock()
    fn.name = tool_name
    fn.arguments = json.dumps(arguments)

    tc = MagicMock()
    tc.id = tool_id
    tc.type = "function"
    tc.function = fn

    message = MagicMock()
    message.content = None
    message.tool_calls = [tc]

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_logging_obj(call_id: str, followup_model: str = "openai/followup-model") -> MagicMock:
    logging_obj = MagicMock()
    litellm_params = {"metadata": {}}
    logging_obj.litellm_call_id = call_id
    logging_obj.litellm_params = litellm_params
    logging_obj.model_call_details = {
        "litellm_params": litellm_params,
        "agentic_loop_params": {"model": followup_model},
    }
    return logging_obj


def _logged_headroom_response(logging_obj: MagicMock) -> dict:
    entries = logging_obj.litellm_params["metadata"]["standard_logging_guardrail_information"]
    return entries[0]["guardrail_response"]


def _headroom_tool_calls(hash_value: str) -> list[dict]:
    return [
        {
            "id": "call_abc123",
            "type": "function",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "arguments": {"hash": hash_value},
        }
    ]


@pytest.fixture
def guardrail() -> HeadroomGuardrail:
    return _make_guardrail()


def test_unwrap_agent_tool_result_content_hermes_read_file_envelope():
    inner = ("line\n" * 80) + ("def foo():\n    return 1\n" * 40)
    wrapped = json.dumps({"content": inner})
    assert unwrap_agent_tool_result_content(wrapped) == inner


def test_unwrap_agent_tool_result_content_keeps_structured_search_payload():
    payload = {
        "total_count": 40,
        "files": [f"/tmp/file_{i}.py" for i in range(40)],
        "matches_format": "path-grouped",
    }
    wrapped = json.dumps(payload)
    assert unwrap_agent_tool_result_content(wrapped) == wrapped


def test_prepare_messages_for_compression_unwraps_only_tool_roles():
    inner = "x" * 500
    messages = [
        {"role": "user", "content": json.dumps({"content": inner})},
        {"role": "tool", "tool_call_id": "1", "content": json.dumps({"content": inner})},
    ]
    prepared = prepare_messages_for_compression(messages)
    assert prepared[0]["content"] == messages[0]["content"]
    assert prepared[1]["content"] == inner


@pytest.mark.asyncio
async def test_apply_guardrail_unwraps_hermes_tool_json_before_compress(
    guardrail: HeadroomGuardrail,
):
    inner = ("from __future__ import annotations\n" * 50) + ("def f(x):\n    return x\n" * 80)
    wrapped = json.dumps({"content": inner})
    inputs = GenericGuardrailAPIInputs(
        texts=[wrapped],
        structured_messages=[
            {"role": "user", "content": "read the file"},
            {"role": "tool", "tool_call_id": "call_1", "content": wrapped},
        ],
    )
    mock_response = _make_compress_response(
        [
            {"role": "user", "content": "read the file"},
            {"role": "tool", "tool_call_id": "call_1", "content": "stub hash=b573993006976af767214fac"},
        ]
    )
    mock_post = AsyncMock(return_value=mock_response)

    with patch.object(guardrail.async_handler, "post", new=mock_post):
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    sent_payload = mock_post.await_args.kwargs["json"]
    assert sent_payload["messages"][1]["content"] == inner


@pytest.mark.asyncio
async def test_apply_guardrail_compresses_and_returns_structured_messages(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(
        texts=["A" * 5000],
        structured_messages=ORIGINAL_MESSAGES,
    )
    mock_response = _make_compress_response(COMPRESSED_MESSAGES)

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    assert result.get("structured_messages") == COMPRESSED_MESSAGES


@pytest.mark.asyncio
async def test_apply_guardrail_logs_messages_before_and_after_compression(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(
        texts=["A" * 5000],
        structured_messages=ORIGINAL_MESSAGES,
    )
    mock_response = _make_compress_response(COMPRESSED_MESSAGES)
    request_data: dict = {"model": "gpt-4o", "metadata": {}}

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
        )

    logged = request_data["metadata"]["standard_logging_guardrail_information"]
    assert len(logged) == 1
    response = logged[0]["guardrail_response"]
    assert response["messages_before"] == ORIGINAL_MESSAGES
    assert response["messages_after"] == COMPRESSED_MESSAGES
    assert response["tokens_before"] == 1000
    assert response["tokens_after"] == 100
    assert response["tokens_saved"] == 900
    assert response["compression_ratio"] == 0.1
    assert response["transforms_applied"] == ["router:smart_crusher:0.35"]


@pytest.mark.asyncio
async def test_apply_guardrail_preserves_headroom_net_loss_for_aggregation(
    guardrail: HeadroomGuardrail,
):
    from litellm.proxy.spend_tracking.compression_savings import extract_compression_saved_tokens

    logging_obj = _make_logging_obj("call-compression-net-loss")
    compression_response = _make_compress_response(COMPRESSED_MESSAGES)
    compression_response.json.return_value["tokens_before"] = 100
    compression_response.json.return_value["tokens_after"] = 125
    compression_response.json.return_value["tokens_saved"] = 0
    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=compression_response,
    ):
        await guardrail.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(texts=["A" * 5000], structured_messages=ORIGINAL_MESSAGES),
            request_data={"model": "gpt-4o", "metadata": {}},
            input_type="request",
            logging_obj=logging_obj,
        )

    telemetry = _logged_headroom_response(logging_obj)
    assert telemetry["tokens_saved"] == -25
    metadata = {
        "guardrail_information": logging_obj.litellm_params["metadata"]["standard_logging_guardrail_information"]
    }
    assert extract_compression_saved_tokens(metadata) == -25


@pytest.mark.asyncio
async def test_apply_guardrail_injects_retrieve_tool_when_hashes_present(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(
        texts=["A" * 5000],
        structured_messages=ORIGINAL_MESSAGES,
    )
    mock_response = _make_compress_response(COMPRESSED_MESSAGES_WITH_HASH)

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    tools = result.get("tools")
    assert tools is not None
    assert has_headroom_retrieve_tool(tools)
    structured_messages = result.get("structured_messages")
    assert isinstance(structured_messages, list)
    system_message = structured_messages[0]
    assert system_message["role"] == "system"
    system_content = system_message["content"]
    assert isinstance(system_content, str)
    assert HEADROOM_CCR_SYSTEM_INSTRUCTION in system_content

    retrieve_tool = next(
        tool
        for tool in tools
        if isinstance(tool, dict) and tool.get("function", {}).get("name") == HEADROOM_RETRIEVE_TOOL_NAME
    )
    hash_schema = retrieve_tool["function"]["parameters"]["properties"]["hash"]
    assert hash_schema == {
        "type": "string",
        "description": "Select the exact hash from the compression marker.",
        "enum": ["b573993006976af767214fac"],
        "minLength": 12,
        "maxLength": 24,
        "pattern": "^(?:[a-f0-9]{12}|[a-f0-9]{24})$",
    }


@pytest.mark.asyncio
async def test_apply_guardrail_prioritizes_retrieve_before_refetch_tools(
    guardrail: HeadroomGuardrail,
):
    read_file_tool = {"type": "function", "function": {"name": "read_file", "parameters": {}}}
    inputs = GenericGuardrailAPIInputs(
        texts=["A" * 5000],
        structured_messages=ORIGINAL_MESSAGES,
        tools=[read_file_tool],
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=_make_compress_response(COMPRESSED_MESSAGES_WITH_HASH),
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    structured_messages = result.get("structured_messages")
    assert isinstance(structured_messages, list)
    system_content = structured_messages[0]["content"]
    assert isinstance(system_content, str)
    assert "MUST call `headroom_retrieve`" in system_content
    assert "before calling `read_file`" in system_content
    tools = result.get("tools")
    assert isinstance(tools, list)
    assert tools[0] == read_file_tool
    assert has_headroom_retrieve_tool(tools)


@pytest.mark.asyncio
async def test_apply_guardrail_skips_ccr_tool_when_enable_ccr_false():
    """Compress-only mode: still rewrite messages, never inject headroom_retrieve."""
    guardrail = _make_guardrail(enable_ccr=False)
    inputs = GenericGuardrailAPIInputs(
        texts=["A" * 5000],
        structured_messages=ORIGINAL_MESSAGES,
        tools=[{"type": "function", "function": {"name": "search_files", "parameters": {}}}],
    )
    mock_response = _make_compress_response(COMPRESSED_MESSAGES_WITH_HASH)

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    assert result["structured_messages"] == COMPRESSED_MESSAGES_WITH_HASH
    assert not has_headroom_retrieve_tool(result.get("tools") or [])
    assert result.get("tools") == inputs.get("tools")

    response = _make_openai_response_with_tool_call(
        HEADROOM_RETRIEVE_TOOL_NAME,
        {"hash": "b573993006976af767214fac"},
    )
    should_run, _ = await guardrail.async_should_run_agentic_loop(
        response=response,
        model="gpt-4o",
        messages=list(COMPRESSED_MESSAGES_WITH_HASH),
        tools=[{"type": "function", "function": {"name": HEADROOM_RETRIEVE_TOOL_NAME}}],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )
    assert should_run is False


@pytest.mark.asyncio
async def test_apply_guardrail_no_tool_injected_when_no_hashes(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(
        texts=["A" * 5000],
        structured_messages=ORIGINAL_MESSAGES,
    )
    mock_response = _make_compress_response(COMPRESSED_MESSAGES)

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    tools = result.get("tools")
    assert not has_headroom_retrieve_tool(tools or [])


@pytest.mark.asyncio
async def test_apply_guardrail_preserves_existing_tools_when_injecting(
    guardrail: HeadroomGuardrail,
):
    existing_tool = {"type": "function", "function": {"name": "my_tool", "parameters": {}}}
    inputs = GenericGuardrailAPIInputs(
        texts=["A" * 5000],
        structured_messages=ORIGINAL_MESSAGES,
        tools=[existing_tool],
    )
    mock_response = _make_compress_response(COMPRESSED_MESSAGES_WITH_HASH)

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    tools = result.get("tools")
    assert tools is not None
    assert isinstance(tools, list)
    assert any(isinstance(t, dict) and t.get("function", {}).get("name") == "my_tool" for t in tools)
    assert has_headroom_retrieve_tool(tools)


@pytest.mark.asyncio
async def test_apply_guardrail_replaces_client_retrieve_tool_with_request_hash_enum(
    guardrail: HeadroomGuardrail,
):
    client_retrieve_tool = {
        "type": "function",
        "function": {
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "parameters": {
                "type": "object",
                "properties": {"hash": {"type": "string", "enum": ["deadbeef000000000000dead"]}},
            },
        },
    }
    inputs = GenericGuardrailAPIInputs(
        texts=["A" * 5000],
        structured_messages=ORIGINAL_MESSAGES,
        tools=[client_retrieve_tool],
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=_make_compress_response(COMPRESSED_MESSAGES_WITH_HASH),
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    tools = result.get("tools")
    assert isinstance(tools, list)
    assert len(tools) == 1
    hash_schema = tools[0]["function"]["parameters"]["properties"]["hash"]
    assert hash_schema["enum"] == ["b573993006976af767214fac"]
    assert "deadbeef000000000000dead" not in hash_schema["enum"]


@pytest.mark.asyncio
async def test_apply_guardrail_includes_every_current_request_hash_in_enum(
    guardrail: HeadroomGuardrail,
):
    compressed_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                "First hash=ffeeddccbbaa998877665544, "
                "second hash=00112233445566778899aabb, "
                "duplicate hash=ffeeddccbbaa998877665544"
            ),
        },
    ]
    inputs = GenericGuardrailAPIInputs(texts=["A" * 5000], structured_messages=ORIGINAL_MESSAGES)

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=_make_compress_response(compressed_messages),
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    tools = result.get("tools")
    assert isinstance(tools, list)
    hash_schema = tools[0]["function"]["parameters"]["properties"]["hash"]
    assert hash_schema["enum"] == ["00112233445566778899aabb", "ffeeddccbbaa998877665544"]


@pytest.mark.asyncio
async def test_async_should_run_agentic_loop_returns_true_for_retrieve_call(
    guardrail: HeadroomGuardrail,
):
    retrieve_tool_def = [{"type": "function", "function": {"name": HEADROOM_RETRIEVE_TOOL_NAME}}]
    response = _make_openai_response_with_tool_call(
        tool_name=HEADROOM_RETRIEVE_TOOL_NAME,
        arguments={"hash": "b573993006976af767214fac"},
    )

    should_run, ctx = await guardrail.async_should_run_agentic_loop(
        response=response,
        model="gpt-4o",
        messages=[],
        tools=retrieve_tool_def,
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is True
    assert len(ctx["tool_calls"]) == 1
    assert ctx["tool_calls"][0]["arguments"]["hash"] == "b573993006976af767214fac"


@pytest.mark.asyncio
async def test_async_should_run_agentic_loop_returns_false_without_retrieve_tool(
    guardrail: HeadroomGuardrail,
):
    other_tools = [{"type": "function", "function": {"name": "other_tool"}}]
    response = _make_openai_response_with_tool_call(
        tool_name="other_tool",
        arguments={},
    )

    should_run, _ = await guardrail.async_should_run_agentic_loop(
        response=response,
        model="gpt-4o",
        messages=[],
        tools=other_tools,
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is False


@pytest.mark.asyncio
async def test_async_should_run_agentic_loop_true_without_tool_in_request_tools(
    guardrail: HeadroomGuardrail,
):
    """Response headroom_retrieve is enough; request tools may omit the injected tool."""
    response = _make_openai_response_with_tool_call(
        tool_name=HEADROOM_RETRIEVE_TOOL_NAME,
        arguments={"hash": "b573993006976af767214fac"},
    )
    should_run, ctx = await guardrail.async_should_run_agentic_loop(
        response=response,
        model="gpt-4o",
        messages=[],
        tools=[{"type": "function", "function": {"name": "terminal"}}],
        stream=True,
        custom_llm_provider="openai",
        kwargs={},
    )
    assert should_run is True
    assert ctx["tool_calls"][0]["name"] == HEADROOM_RETRIEVE_TOOL_NAME


@pytest.mark.asyncio
async def test_ccr_telemetry_tracks_not_triggered_retrieve_success_and_completed(
    guardrail: HeadroomGuardrail,
):
    call_id = "call-telemetry-success"
    logging_obj = _make_logging_obj(call_id, followup_model="openai/fallback-model")
    request_data = {"model": "gpt-4o", "litellm_call_id": call_id, "metadata": {}}

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=_make_compress_response(COMPRESSED_MESSAGES_WITH_HASH),
    ):
        await guardrail.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(
                texts=["A" * 5000],
                structured_messages=ORIGINAL_MESSAGES,
            ),
            request_data=request_data,
            input_type="request",
            logging_obj=logging_obj,
        )

    telemetry = _logged_headroom_response(logging_obj)
    assert telemetry["ccr_enabled"] is True
    assert telemetry["ccr_status"] == "not_triggered"
    assert telemetry["ccr_hashes_issued"] == 1
    assert telemetry["ccr_hashes_requested"] == 0
    assert telemetry["ccr_hashes_retrieved"] == 0

    response = _make_openai_response_with_tool_call(
        HEADROOM_RETRIEVE_TOOL_NAME,
        {"hash": "b573993006976af767214fac"},
    )
    with patch.object(
        guardrail.async_handler,
        "get",
        new_callable=AsyncMock,
        return_value=_make_retrieve_response("recalled content"),
    ):
        plan = await guardrail.async_build_agentic_loop_plan(
            tools={"tool_calls": _headroom_tool_calls("b573993006976af767214fac")},
            model="gpt-4o",
            messages=list(COMPRESSED_MESSAGES_WITH_HASH),
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=logging_obj,
            stream=False,
            kwargs={"litellm_call_id": call_id},
        )

    assert telemetry["ccr_status"] == "retrieve_success"
    assert telemetry["ccr_hashes_requested"] == 1
    assert telemetry["ccr_hashes_retrieved"] == 1
    assert telemetry["ccr_retrieved_chars"] == len("recalled content")
    assert telemetry["ccr_retrieved_tokens"] > 0
    assert telemetry["tokens_after"] == 200 + telemetry["ccr_retrieved_tokens"]
    assert telemetry["tokens_saved"] == 1000 - telemetry["tokens_after"]
    assert telemetry["ccr_followup_model"] == "openai/fallback-model"
    assert telemetry["ccr_error"] is None

    followup_response = MagicMock()
    followup_response._hidden_params = {
        "additional_headers": {"x-litellm-attempted-fallbacks": "1"},
    }
    await guardrail.async_post_agentic_loop_response_hook(followup_response, plan, {})

    assert telemetry["ccr_status"] == "completed"
    assert telemetry["ccr_fallback_used"] is True
    await guardrail.async_agentic_loop_cleanup_hook(plan, {})
    assert telemetry["ccr_status"] == "completed"
    assert call_id not in guardrail._issued_hashes_by_call_id


@pytest.mark.asyncio
async def test_ccr_retrieval_net_loss_is_not_counted_as_saved_tokens(
    guardrail: HeadroomGuardrail,
):
    from litellm.proxy.spend_tracking.compression_savings import extract_compression_saved_tokens

    call_id = "call-telemetry-net-loss"
    logging_obj = _make_logging_obj(call_id)
    request_data = {"model": "gpt-4o", "litellm_call_id": call_id, "metadata": {}}
    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=_make_compress_response(COMPRESSED_MESSAGES_WITH_HASH),
    ):
        await guardrail.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(
                texts=["A" * 5000],
                structured_messages=ORIGINAL_MESSAGES,
            ),
            request_data=request_data,
            input_type="request",
            logging_obj=logging_obj,
        )

    cached_content = "restored content\n" * 5000
    response = _make_openai_response_with_tool_call(
        HEADROOM_RETRIEVE_TOOL_NAME,
        {"hash": "b573993006976af767214fac"},
    )
    with patch.object(
        guardrail.async_handler,
        "get",
        new_callable=AsyncMock,
        return_value=_make_retrieve_response(cached_content),
    ):
        await guardrail.async_build_agentic_loop_plan(
            tools={"tool_calls": _headroom_tool_calls("b573993006976af767214fac")},
            model="gpt-4o",
            messages=list(COMPRESSED_MESSAGES_WITH_HASH),
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=logging_obj,
            stream=False,
            kwargs={"litellm_call_id": call_id},
        )

    telemetry = _logged_headroom_response(logging_obj)
    assert telemetry["tokens_saved"] < 0
    metadata = {
        "guardrail_information": logging_obj.litellm_params["metadata"]["standard_logging_guardrail_information"]
    }
    assert extract_compression_saved_tokens(metadata) < 0


@pytest.mark.asyncio
async def test_ccr_uses_followup_provider_usage_for_net_savings(
    guardrail: HeadroomGuardrail,
):
    from litellm.proxy.spend_tracking.compression_savings import extract_compression_saved_tokens
    from litellm.types.utils import Usage

    call_id = "call-telemetry-provider-usage"
    logging_obj = _make_logging_obj(call_id)
    request_data = {"model": "gpt-4o", "litellm_call_id": call_id, "metadata": {}}
    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=_make_compress_response(COMPRESSED_MESSAGES_WITH_HASH),
    ):
        await guardrail.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(texts=["A" * 5000], structured_messages=ORIGINAL_MESSAGES),
            request_data=request_data,
            input_type="request",
            logging_obj=logging_obj,
        )

    response = _make_openai_response_with_tool_call(
        HEADROOM_RETRIEVE_TOOL_NAME,
        {"hash": "b573993006976af767214fac"},
    )
    response.usage = Usage(prompt_tokens=120, completion_tokens=5, total_tokens=125)
    with patch.object(
        guardrail.async_handler,
        "get",
        new_callable=AsyncMock,
        return_value=_make_retrieve_response("recalled content"),
    ):
        plan = await guardrail.async_build_agentic_loop_plan(
            tools={"tool_calls": _headroom_tool_calls("b573993006976af767214fac")},
            model="gpt-4o",
            messages=list(COMPRESSED_MESSAGES_WITH_HASH),
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=logging_obj,
            stream=False,
            kwargs={"litellm_call_id": call_id},
        )

    followup_response = MagicMock()
    followup_response.usage = Usage(prompt_tokens=950, completion_tokens=10, total_tokens=960)
    await guardrail.async_post_agentic_loop_response_hook(followup_response, plan, {})

    telemetry = _logged_headroom_response(logging_obj)
    assert telemetry["headroom_compressed_tokens"] == 100
    assert telemetry["ccr_initial_input_tokens"] == 120
    assert telemetry["ccr_followup_input_tokens"] == 950
    assert telemetry["tokens_after"] == 1070
    assert telemetry["tokens_saved"] == -70
    assert followup_response.usage.prompt_tokens == 950
    metadata = {
        "guardrail_information": logging_obj.litellm_params["metadata"]["standard_logging_guardrail_information"]
    }
    assert extract_compression_saved_tokens(metadata) == -70


@pytest.mark.asyncio
async def test_apply_guardrail_bypasses_internal_agentic_followup(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(texts=["A" * 5000], structured_messages=ORIGINAL_MESSAGES)

    result = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data={"litellm_metadata": {"_agentic_loop_depth": 1}},
        input_type="request",
    )

    assert result == inputs


@pytest.mark.asyncio
async def test_ccr_followup_does_not_inherit_headroom_savings(
    guardrail: HeadroomGuardrail,
):
    call_id = "call-followup-no-duplicate-savings"
    logging_obj = _make_logging_obj(call_id)
    request_data = {"model": "gpt-4o", "litellm_call_id": call_id, "metadata": {}}
    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=_make_compress_response(COMPRESSED_MESSAGES_WITH_HASH),
    ):
        await guardrail.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(texts=["A" * 5000], structured_messages=ORIGINAL_MESSAGES),
            request_data=request_data,
            input_type="request",
            logging_obj=logging_obj,
        )

    headroom_entry = logging_obj.litellm_params["metadata"]["standard_logging_guardrail_information"][0]
    other_entry = {"guardrail_provider": "other", "guardrail_response": {"allowed": True}}
    response = _make_openai_response_with_tool_call(
        HEADROOM_RETRIEVE_TOOL_NAME,
        {"hash": "b573993006976af767214fac"},
    )
    with patch.object(
        guardrail.async_handler,
        "get",
        new_callable=AsyncMock,
        return_value=_make_retrieve_response("recalled content"),
    ):
        plan = await guardrail.async_build_agentic_loop_plan(
            tools={"tool_calls": _headroom_tool_calls("b573993006976af767214fac")},
            model="gpt-4o",
            messages=list(COMPRESSED_MESSAGES_WITH_HASH),
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=logging_obj,
            stream=True,
            kwargs={
                "litellm_call_id": call_id,
                "litellm_metadata": {
                    "standard_logging_guardrail_information": [headroom_entry, other_entry],
                    "tenant": "keep-me",
                },
            },
        )

    assert plan.request_patch is not None
    followup_metadata = plan.request_patch.kwargs["litellm_metadata"]
    assert followup_metadata["standard_logging_guardrail_information"] == [other_entry]
    assert followup_metadata["tenant"] == "keep-me"


@pytest.mark.asyncio
async def test_ccr_telemetry_marks_retrieve_failed_for_expired_hash(
    guardrail: HeadroomGuardrail,
):
    call_id = "call-telemetry-retrieve-failed"
    logging_obj = _make_logging_obj(call_id)
    request_data = {"model": "gpt-4o", "litellm_call_id": call_id, "metadata": {}}
    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=_make_compress_response(COMPRESSED_MESSAGES_WITH_HASH),
    ):
        await guardrail.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(
                texts=["A" * 5000],
                structured_messages=ORIGINAL_MESSAGES,
            ),
            request_data=request_data,
            input_type="request",
            logging_obj=logging_obj,
        )

    response = _make_openai_response_with_tool_call(
        HEADROOM_RETRIEVE_TOOL_NAME,
        {"hash": "b573993006976af767214fac"},
    )
    with patch.object(
        guardrail.async_handler,
        "get",
        new_callable=AsyncMock,
        return_value=_make_retrieve_response("", status=404),
    ):
        plan = await guardrail.async_build_agentic_loop_plan(
            tools={"tool_calls": _headroom_tool_calls("b573993006976af767214fac")},
            model="gpt-4o",
            messages=list(COMPRESSED_MESSAGES_WITH_HASH),
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=logging_obj,
            stream=False,
            kwargs={"litellm_call_id": call_id},
        )

    telemetry = _logged_headroom_response(logging_obj)
    assert telemetry["ccr_status"] == "retrieve_failed"
    assert telemetry["ccr_hashes_requested"] == 1
    assert telemetry["ccr_hashes_retrieved"] == 0
    assert telemetry["ccr_retrieved_chars"] == 0
    assert telemetry["ccr_error"] == "not_found_or_expired"
    await guardrail.async_agentic_loop_cleanup_hook(plan, {})
    assert telemetry["ccr_status"] == "retrieve_failed"


@pytest.mark.asyncio
async def test_ccr_telemetry_marks_followup_failed_when_cleanup_runs_before_post_hook(
    guardrail: HeadroomGuardrail,
):
    call_id = "call-telemetry-followup-failed"
    logging_obj = _make_logging_obj(call_id)
    request_data = {"model": "gpt-4o", "litellm_call_id": call_id, "metadata": {}}
    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=_make_compress_response(COMPRESSED_MESSAGES_WITH_HASH),
    ):
        await guardrail.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(
                texts=["A" * 5000],
                structured_messages=ORIGINAL_MESSAGES,
            ),
            request_data=request_data,
            input_type="request",
            logging_obj=logging_obj,
        )

    response = _make_openai_response_with_tool_call(
        HEADROOM_RETRIEVE_TOOL_NAME,
        {"hash": "b573993006976af767214fac"},
    )
    with patch.object(
        guardrail.async_handler,
        "get",
        new_callable=AsyncMock,
        return_value=_make_retrieve_response("recalled content"),
    ):
        plan = await guardrail.async_build_agentic_loop_plan(
            tools={"tool_calls": _headroom_tool_calls("b573993006976af767214fac")},
            model="gpt-4o",
            messages=list(COMPRESSED_MESSAGES_WITH_HASH),
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=logging_obj,
            stream=False,
            kwargs={"litellm_call_id": call_id},
        )

    telemetry = _logged_headroom_response(logging_obj)
    assert telemetry["ccr_status"] == "retrieve_success"
    await guardrail.async_agentic_loop_cleanup_hook(plan, {})
    assert telemetry["ccr_status"] == "followup_failed"
    assert telemetry["ccr_error"] == "followup_failed"


async def test_async_should_run_agentic_loop_returns_false_when_no_retrieve_calls(
    guardrail: HeadroomGuardrail,
):
    retrieve_tool_def = [{"type": "function", "function": {"name": HEADROOM_RETRIEVE_TOOL_NAME}}]
    response = _make_openai_response_with_tool_call(
        tool_name="some_other_function",
        arguments={},
    )

    should_run, _ = await guardrail.async_should_run_agentic_loop(
        response=response,
        model="gpt-4o",
        messages=[],
        tools=retrieve_tool_def,
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is False


@pytest.mark.asyncio
async def test_async_build_agentic_loop_plan_calls_retrieve_and_builds_messages(
    guardrail: HeadroomGuardrail,
):
    original_content = "This is the full compressed content."
    mock_retrieve = _make_retrieve_response(original_content)

    tool_calls = [
        {
            "id": "call_abc123",
            "type": "function",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "arguments": {"hash": "b573993006976af767214fac"},
        }
    ]
    response = _make_openai_response_with_tool_call(
        tool_name=HEADROOM_RETRIEVE_TOOL_NAME,
        arguments={"hash": "b573993006976af767214fac"},
        tool_id="call_abc123",
    )
    messages = [{"role": "user", "content": "What does it say? hash=b573993006976af767214fac"}]
    guardrail._issued_hashes_by_call_id["call-1"] = (
        frozenset({"b573993006976af767214fac"}),
        time.monotonic() + 999,
    )

    with patch.object(
        guardrail.async_handler,
        "get",
        new_callable=AsyncMock,
        return_value=mock_retrieve,
    ) as mock_get:
        plan = await guardrail.async_build_agentic_loop_plan(
            tools={"tool_calls": tool_calls},
            model="gpt-4o",
            messages=messages,
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=None,
            stream=False,
            kwargs={"litellm_call_id": "call-1"},
        )

    assert plan.run_agentic_loop is True
    assert plan.request_patch is not None

    follow_up = plan.request_patch.messages
    assert follow_up is not None

    tool_result_message = next((m for m in follow_up if m.get("role") == "tool"), None)
    assert tool_result_message is not None
    assert tool_result_message["content"] == original_content
    assert tool_result_message["tool_call_id"] == "call_abc123"

    mock_get.assert_called_once()
    call_url = mock_get.call_args.kwargs.get("url") or mock_get.call_args.args[0]
    assert "b573993006976af767214fac" in call_url


@pytest.mark.asyncio
async def test_async_build_agentic_loop_plan_handles_retrieve_404(
    guardrail: HeadroomGuardrail,
):
    mock_retrieve = MagicMock()
    mock_retrieve.status_code = 404

    tool_calls = [
        {
            "id": "call_xyz",
            "type": "function",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "arguments": {"hash": "deadbeef000000000000dead"},
        }
    ]
    response = _make_openai_response_with_tool_call(
        tool_name=HEADROOM_RETRIEVE_TOOL_NAME,
        arguments={"hash": "deadbeef000000000000dead"},
        tool_id="call_xyz",
    )

    messages = [
        {
            "role": "user",
            "content": "Retrieve more: hash=deadbeef000000000000dead",
        }
    ]
    guardrail._issued_hashes_by_call_id["call-1"] = (
        frozenset({"deadbeef000000000000dead"}),
        time.monotonic() + 999,
    )

    with patch.object(
        guardrail.async_handler,
        "get",
        new_callable=AsyncMock,
        return_value=mock_retrieve,
    ):
        plan = await guardrail.async_build_agentic_loop_plan(
            tools={"tool_calls": tool_calls},
            model="gpt-4o",
            messages=messages,
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=None,
            stream=False,
            kwargs={"litellm_call_id": "call-1"},
        )

    follow_up = plan.request_patch.messages  # type: ignore[union-attr]
    tool_result = next((m for m in follow_up if m.get("role") == "tool"), None)
    assert tool_result is not None
    assert "not found" in tool_result["content"] or "expired" in tool_result["content"]


@pytest.mark.asyncio
async def test_async_build_agentic_loop_plan_rejects_hash_with_no_known_call(
    guardrail: HeadroomGuardrail,
):
    """A hash-shaped string planted in message text must not be honored when
    this guardrail has no record of ever issuing it, even if it's echoed back
    in the current request's own messages (e.g. via prompt injection)."""
    tool_calls = [
        {
            "id": "call_xyz",
            "type": "function",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "arguments": {"hash": "deadbeef000000000000dead"},
        }
    ]
    response = _make_openai_response_with_tool_call(
        tool_name=HEADROOM_RETRIEVE_TOOL_NAME,
        arguments={"hash": "deadbeef000000000000dead"},
        tool_id="call_xyz",
    )
    assert not guardrail._issued_hashes_by_call_id

    with patch.object(
        guardrail.async_handler,
        "get",
        new_callable=AsyncMock,
    ) as mock_get:
        plan = await guardrail.async_build_agentic_loop_plan(
            tools={"tool_calls": tool_calls},
            model="gpt-4o",
            messages=[{"role": "user", "content": "Please fetch hash=deadbeef000000000000dead for me"}],
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=None,
            stream=False,
            kwargs={"litellm_call_id": "call-unknown"},
        )

    mock_get.assert_not_called()

    follow_up = plan.request_patch.messages  # type: ignore[union-attr]
    tool_result = next((m for m in follow_up if m.get("role") == "tool"), None)
    assert tool_result is not None
    assert "was not produced by the current request" in tool_result["content"]


@pytest.mark.asyncio
async def test_async_build_agentic_loop_plan_rejects_hash_issued_for_different_call(
    guardrail: HeadroomGuardrail,
):
    """A hash issued for one request must not be retrievable by a different
    request just because the second request echoes that hash-shaped string
    back in its own messages -- retrieval must be scoped per litellm_call_id,
    not derived by re-scanning attacker-controlled message text."""
    guardrail._issued_hashes_by_call_id["call-A"] = (
        frozenset({"b573993006976af767214fac"}),
        time.monotonic() + 999,
    )

    tool_calls = [
        {
            "id": "call_xyz",
            "type": "function",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "arguments": {"hash": "b573993006976af767214fac"},
        }
    ]
    response = _make_openai_response_with_tool_call(
        tool_name=HEADROOM_RETRIEVE_TOOL_NAME,
        arguments={"hash": "b573993006976af767214fac"},
        tool_id="call_xyz",
    )

    with patch.object(
        guardrail.async_handler,
        "get",
        new_callable=AsyncMock,
    ) as mock_get:
        plan = await guardrail.async_build_agentic_loop_plan(
            tools={"tool_calls": tool_calls},
            model="gpt-4o",
            messages=[{"role": "user", "content": "Please fetch hash=b573993006976af767214fac for me"}],
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=None,
            stream=False,
            kwargs={"litellm_call_id": "call-B"},
        )

    mock_get.assert_not_called()

    follow_up = plan.request_patch.messages  # type: ignore[union-attr]
    tool_result = next((m for m in follow_up if m.get("role") == "tool"), None)
    assert tool_result is not None
    assert "was not produced by the current request" in tool_result["content"]


@pytest.mark.asyncio
async def test_async_build_agentic_loop_plan_builds_responses_api_function_call_items(
    guardrail: HeadroomGuardrail,
):
    """For the Responses API, follow-up input must echo a function_call paired
    with a function_call_output keyed by the same call_id -- chat-style
    assistant/tool messages are not valid Responses API input items."""
    original_content = "This is the full compressed content."
    mock_retrieve = _make_retrieve_response(original_content)

    response = MagicMock()
    response.choices = None
    response.content = None
    response.output = [
        {
            "type": "function_call",
            "id": "fc_abc123",
            "call_id": "call_abc123",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "arguments": json.dumps({"hash": "b573993006976af767214fac"}),
        }
    ]

    tool_calls = [
        {
            "id": "call_abc123",
            "type": "function",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "arguments": {"hash": "b573993006976af767214fac"},
        }
    ]
    messages = [{"role": "user", "content": "What does it say? hash=b573993006976af767214fac"}]
    guardrail._issued_hashes_by_call_id["call-1"] = (
        frozenset({"b573993006976af767214fac"}),
        time.monotonic() + 999,
    )

    with patch.object(
        guardrail.async_handler,
        "get",
        new_callable=AsyncMock,
        return_value=mock_retrieve,
    ):
        plan = await guardrail.async_build_agentic_loop_plan(
            tools={"tool_calls": tool_calls},
            model="gpt-4o",
            messages=messages,
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=None,
            stream=False,
            kwargs={"litellm_call_id": "call-1"},
        )

    follow_up = plan.request_patch.messages  # type: ignore[union-attr]
    assert all("role" not in item for item in follow_up if item not in messages)

    function_call_item = next((i for i in follow_up if i.get("type") == "function_call"), None)
    assert function_call_item is not None
    assert function_call_item["call_id"] == "call_abc123"
    assert function_call_item["name"] == HEADROOM_RETRIEVE_TOOL_NAME

    output_item = next((i for i in follow_up if i.get("type") == "function_call_output"), None)
    assert output_item is not None
    assert output_item["call_id"] == "call_abc123"
    assert output_item["output"] == original_content


@pytest.mark.asyncio
async def test_async_build_agentic_loop_plan_builds_anthropic_tool_result_messages(
    guardrail: HeadroomGuardrail,
):
    """For the Anthropic Messages API, follow-up must echo a tool_use content
    block in an assistant message paired with a tool_result content block in a
    user message keyed by the same tool_use_id -- chat-style tool-role
    messages are not valid Anthropic input.

    AnthropicMessagesResponse is a TypedDict, so real responses are plain
    dicts at runtime; a MagicMock response here would pass even if branch
    selection used bare getattr() and silently fell through to the
    chat-completions replay shape for every real Anthropic response.
    """
    original_content = "This is the full compressed content."
    mock_retrieve = _make_retrieve_response(original_content)

    response = {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_abc123",
                "name": HEADROOM_RETRIEVE_TOOL_NAME,
                "input": {"hash": "b573993006976af767214fac"},
            }
        ]
    }

    tool_calls = [
        {
            "id": "toolu_abc123",
            "type": "function",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "arguments": {"hash": "b573993006976af767214fac"},
        }
    ]
    messages = [{"role": "user", "content": "What does it say? hash=b573993006976af767214fac"}]
    guardrail._issued_hashes_by_call_id["call-1"] = (
        frozenset({"b573993006976af767214fac"}),
        time.monotonic() + 999,
    )

    with patch.object(
        guardrail.async_handler,
        "get",
        new_callable=AsyncMock,
        return_value=mock_retrieve,
    ):
        plan = await guardrail.async_build_agentic_loop_plan(
            tools={"tool_calls": tool_calls},
            model="claude-sonnet-4-5",
            messages=messages,
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=None,
            stream=False,
            kwargs={"litellm_call_id": "call-1"},
        )

    follow_up = plan.request_patch.messages  # type: ignore[union-attr]
    assert all(m.get("role") != "tool" for m in follow_up)

    assistant_message = next((m for m in follow_up if m.get("role") == "assistant"), None)
    assert assistant_message is not None
    tool_use_block = next((b for b in assistant_message["content"] if b.get("type") == "tool_use"), None)
    assert tool_use_block is not None
    assert tool_use_block["id"] == "toolu_abc123"

    user_message = follow_up[-1]
    assert user_message["role"] == "user"
    tool_result_block = next((b for b in user_message["content"] if b.get("type") == "tool_result"), None)
    assert tool_result_block is not None
    assert tool_result_block["tool_use_id"] == "toolu_abc123"
    assert tool_result_block["content"] == original_content


def test_extract_hashes_from_messages_finds_hashes():
    messages = [
        {"role": "user", "content": "Retrieve more: hash=b573993006976af767214fac"},
        {"role": "assistant", "content": "Also: hash=aabbccdd001122334455aabb"},
        {"role": "tool", "content": "Compressed payload: <<ccr:e60cbe778006,base64,512B>>"},
    ]
    hashes = extract_hashes_from_messages(messages)
    assert "b573993006976af767214fac" in hashes
    assert "aabbccdd001122334455aabb" in hashes
    assert "e60cbe778006" in hashes


def test_extract_hashes_from_messages_ignores_invalid_hash_lengths():
    messages = [
        {"role": "user", "content": "hash=abc123"},
        {"role": "assistant", "content": "hash=0011223344556"},
        {"role": "tool", "content": "<<ccr:001122334455667788>>"},
    ]
    hashes = extract_hashes_from_messages(messages)
    assert not hashes


def test_extract_hashes_from_list_content_blocks():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hash=b573993006976af767214fac found here"},
            ],
        }
    ]
    hashes = extract_hashes_from_messages(messages)
    assert "b573993006976af767214fac" in hashes


def test_has_headroom_retrieve_tool_recognizes_anthropic_native_shape():
    """By the time an Anthropic Messages API response reaches the agentic-loop
    gate, the OpenAI-shaped tool this guardrail injects (type: "function")
    has already been transformed into Anthropic's native tool shape
    (type: "custom", top-level "name", no nested "function" object)."""
    anthropic_native_tools = [
        {
            "type": "custom",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "input_schema": {"type": "object", "properties": {"hash": {"type": "string"}}},
        }
    ]
    assert has_headroom_retrieve_tool(anthropic_native_tools)
    assert not has_headroom_retrieve_tool([{"type": "custom", "name": "some_other_tool"}])


@pytest.mark.asyncio
async def test_apply_guardrail_bypass_header_skips_compression(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )
    request_data = {"proxy_server_request": {"headers": {"x-headroom-bypass": "true"}}}

    with patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post:
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
        )
        mock_post.assert_not_called()

    assert result.get("structured_messages") == ORIGINAL_MESSAGES


@pytest.mark.asyncio
async def test_apply_guardrail_response_type_passthrough(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(
        texts=["some response text"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post:
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="response",
        )
        mock_post.assert_not_called()

    assert result is inputs


@pytest.mark.asyncio
async def test_apply_guardrail_empty_structured_messages_passthrough(
    guardrail: HeadroomGuardrail,
):
    inputs = GenericGuardrailAPIInputs(texts=["hello"])

    with patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post:
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="request",
        )
        mock_post.assert_not_called()

    assert result is inputs


@pytest.mark.asyncio
async def test_apply_guardrail_http_error_raises():
    guardrail = _make_guardrail()
    mock_response = _make_compress_response([], status=500)
    mock_response.text = "Internal Server Error"

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )

    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_apply_guardrail_transport_error_raises():
    guardrail = _make_guardrail()

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("Connection refused"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )

    assert exc_info.value.status_code == 502
    assert "unreachable" in str(exc_info.value.detail)


def _make_http_status_error(status: int, body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", f"{FAKE_API_BASE}/v1/compress")
    response = httpx.Response(status, request=request, text=body)
    return httpx.HTTPStatusError(
        f"Server error '{status}' for url",
        request=request,
        response=response,
    )


@pytest.mark.asyncio
async def test_apply_guardrail_http_status_error_raises():
    """Regression test: litellm's async httpx client calls raise_for_status()
    internally, so a non-2xx /v1/compress response surfaces as
    httpx.HTTPStatusError, not as a returned MagicMock with status_code set.
    A prior version of _call_compress only checked response.status_code and
    never caught this exception, so it went unhandled instead of blocking
    the request per fail_closed policy."""
    guardrail = _make_guardrail()

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        side_effect=_make_http_status_error(500, "headroom internal error"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )

    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_apply_guardrail_http_status_error_fail_open_forwards_uncompressed():
    guardrail = _make_guardrail(unreachable_fallback="fail_open")

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        side_effect=_make_http_status_error(500, "headroom internal error"),
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="request",
        )

    assert result["structured_messages"] == ORIGINAL_MESSAGES


@pytest.mark.asyncio
async def test_apply_guardrail_transport_error_fail_open_forwards_uncompressed():
    guardrail = _make_guardrail(unreachable_fallback="fail_open")

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("Connection refused"),
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="request",
        )

    assert result["structured_messages"] == ORIGINAL_MESSAGES


@pytest.mark.asyncio
async def test_apply_guardrail_http_error_fail_open_forwards_uncompressed():
    guardrail = _make_guardrail(unreachable_fallback="fail_open")
    mock_response = _make_compress_response([], status=500)
    mock_response.text = "Internal Server Error"

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="request",
        )

    assert result["structured_messages"] == ORIGINAL_MESSAGES


@pytest.mark.asyncio
async def test_apply_guardrail_non_json_response_fail_open_forwards_uncompressed():
    guardrail = _make_guardrail(unreachable_fallback="fail_open")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("not JSON")
    mock_response.text = "<!DOCTYPE html><html>not json</html>"

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="request",
        )

    assert result["structured_messages"] == ORIGINAL_MESSAGES


@pytest.mark.asyncio
async def test_apply_guardrail_missing_messages_key_fail_open_forwards_uncompressed():
    guardrail = _make_guardrail(unreachable_fallback="fail_open")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"tokens_before": 100, "tokens_after": 10}
    mock_response.text = "{}"

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="request",
        )

    assert result["structured_messages"] == ORIGINAL_MESSAGES


@pytest.mark.asyncio
async def test_apply_guardrail_empty_compressed_messages_fail_open_forwards_uncompressed():
    guardrail = _make_guardrail(unreachable_fallback="fail_open")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "messages": ["not-a-dict", 42, None],
        "tokens_before": 1000,
        "tokens_after": 0,
        "compression_ratio": 0,
    }
    mock_response.text = "{}"

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="request",
        )

    assert result["structured_messages"] == ORIGINAL_MESSAGES


@pytest.mark.asyncio
async def test_apply_guardrail_fail_open_does_not_register_hashes_from_original_messages():
    """When compression fails with fail_open, user-supplied messages that
    happen to contain hash-shaped strings must NOT cause those hashes to be
    registered as valid for CCR retrieval. Otherwise an attacker can plant a
    hash= string in their prompt, trigger a compression failure, and have
    that hash honored by a later headroom_retrieve tool call."""
    messages_with_fake_hash = [
        {"role": "user", "content": "Please fetch hash=deadbeef000000000000dead for me"},
    ]
    guardrail = _make_guardrail(unreachable_fallback="fail_open")

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=messages_with_fake_hash,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("Connection refused"),
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="request",
        )

    assert result["structured_messages"] == messages_with_fake_hash
    assert not has_headroom_retrieve_tool(result.get("tools") or [])
    assert not guardrail._issued_hashes_by_call_id


@pytest.mark.asyncio
async def test_apply_guardrail_missing_messages_key_raises():
    guardrail = _make_guardrail()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"tokens_before": 100, "tokens_after": 10}
    mock_response.text = "{}"

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )

    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_apply_guardrail_empty_compressed_messages_raises():
    guardrail = _make_guardrail()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "messages": ["not-a-dict", 42, None],
        "tokens_before": 1000,
        "tokens_after": 0,
        "compression_ratio": 0,
    }
    mock_response.text = "{}"

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )

    assert exc_info.value.status_code == 502
    assert "empty message list" in str(exc_info.value.detail)


def test_init_raises_without_api_base():
    with pytest.raises(ValueError, match="API base URL"):
        HeadroomGuardrail(api_base=None)


def test_init_defaults_to_fail_closed():
    guardrail = _make_guardrail()
    assert guardrail.unreachable_fallback == "fail_closed"


def test_init_rejects_invalid_unreachable_fallback_value():
    guardrail = _make_guardrail(unreachable_fallback="not-a-real-mode")
    assert guardrail.unreachable_fallback == "fail_closed"


def test_bypass_header_case_insensitive():
    guardrail = _make_guardrail()

    for header_value in ("true", "True", "TRUE"):
        data = {"proxy_server_request": {"headers": {"x-headroom-bypass": header_value}}}
        assert guardrail._should_bypass(data) is True

    data = {"proxy_server_request": {"headers": {"x-headroom-bypass": "false"}}}
    assert guardrail._should_bypass(data) is False

    data = {"proxy_server_request": {"headers": {}}}
    assert guardrail._should_bypass(data) is False

    data = {}
    assert guardrail._should_bypass(data) is False


@pytest.mark.asyncio
async def test_apply_guardrail_sends_model_from_config():
    guardrail = _make_guardrail(model="gpt-4o-mini")
    mock_response = _make_compress_response(COMPRESSED_MESSAGES)

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_post:
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    call_kwargs = mock_post.call_args
    sent_payload = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
    assert sent_payload.get("model") == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_apply_guardrail_sends_model_from_request_data_when_no_config_model():
    guardrail = _make_guardrail()
    mock_response = _make_compress_response(COMPRESSED_MESSAGES)

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_post:
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4o"},
            input_type="request",
        )

    call_kwargs = mock_post.call_args
    sent_payload = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
    assert sent_payload.get("model") == "gpt-4o"


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_async_build_agentic_loop_plan_filters_non_headroom_tool_calls(
    guardrail: HeadroomGuardrail,
):
    """Mixed headroom_retrieve + client tool: follow-up only echoes headroom tool_calls."""
    hash_value = "b573993006976af767214fac"
    response = _make_openai_response_with_tool_call(
        tool_name=HEADROOM_RETRIEVE_TOOL_NAME,
        arguments={"hash": hash_value},
        tool_id="call_hr",
    )
    term_fn = MagicMock()
    term_fn.name = "terminal"
    term_fn.arguments = json.dumps({"command": "date"})
    term_tc = MagicMock()
    term_tc.id = "call_term"
    term_tc.type = "function"
    term_tc.function = term_fn
    response.choices[0].message.tool_calls.append(term_tc)

    guardrail._issued_hashes_by_call_id["call-1"] = (
        frozenset({hash_value}),
        time.monotonic() + 999,
    )
    mock_retrieve = _make_retrieve_response("expanded text")

    with patch.object(
        guardrail.async_handler,
        "get",
        new_callable=AsyncMock,
        return_value=mock_retrieve,
    ):
        plan = await guardrail.async_build_agentic_loop_plan(
            tools={
                "tool_calls": [
                    {
                        "id": "call_hr",
                        "type": "function",
                        "name": HEADROOM_RETRIEVE_TOOL_NAME,
                        "arguments": {"hash": hash_value},
                    }
                ]
            },
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=None,
            stream=True,
            kwargs={"litellm_call_id": "call-1"},
        )

    assert plan.run_agentic_loop is True
    assert plan.request_patch is not None
    follow = plan.request_patch.messages or []
    assistant = next(m for m in follow if m.get("role") == "assistant")
    tool_calls = assistant.get("tool_calls") or []
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "call_hr"
    tool_msgs = [m for m in follow if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_hr"


async def test_async_should_run_agentic_loop_detects_anthropic_content_block_format(
    guardrail: HeadroomGuardrail,
):
    # Anthropic's native tool format (type: "custom", top-level "name") --
    # by the time a Messages API response reaches this gate, the OpenAI-shaped
    # tool this guardrail injects has already been transformed into this shape.
    retrieve_tool_def = [
        {
            "type": "custom",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "input_schema": {"type": "object", "properties": {"hash": {"type": "string"}}},
        }
    ]

    response = MagicMock()
    response.choices = None
    response.content = [
        {
            "type": "tool_use",
            "id": "toolu_abc",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "input": {"hash": "b573993006976af767214fac"},
        }
    ]

    should_run, ctx = await guardrail.async_should_run_agentic_loop(
        response=response,
        model="claude-sonnet-4-6",
        messages=[],
        tools=retrieve_tool_def,
        stream=False,
        custom_llm_provider="anthropic",
        kwargs={},
    )

    assert should_run is True
    assert len(ctx["tool_calls"]) == 1
    assert ctx["tool_calls"][0]["arguments"]["hash"] == "b573993006976af767214fac"


@pytest.mark.asyncio
async def test_async_should_run_agentic_loop_detects_anthropic_response_as_plain_dict(
    guardrail: HeadroomGuardrail,
):
    """AnthropicMessagesResponse is a TypedDict -- real Messages API responses
    are plain dicts at runtime, not objects with attribute access. A
    MagicMock-only test would pass even if detection used bare getattr() and
    silently treated every real response as having no tool calls."""
    retrieve_tool_def = [
        {
            "type": "custom",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "input_schema": {"type": "object", "properties": {"hash": {"type": "string"}}},
        }
    ]
    response = {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": HEADROOM_RETRIEVE_TOOL_NAME,
                "input": {"hash": "b573993006976af767214fac"},
            }
        ]
    }

    should_run, ctx = await guardrail.async_should_run_agentic_loop(
        response=response,
        model="claude-sonnet-4-6",
        messages=[],
        tools=retrieve_tool_def,
        stream=False,
        custom_llm_provider="anthropic",
        kwargs={},
    )

    assert should_run is True
    assert len(ctx["tool_calls"]) == 1
    assert ctx["tool_calls"][0]["arguments"]["hash"] == "b573993006976af767214fac"


@pytest.mark.asyncio
async def test_async_should_run_agentic_loop_detects_responses_api_output_format(
    guardrail: HeadroomGuardrail,
):
    retrieve_tool_def = [{"type": "function", "function": {"name": HEADROOM_RETRIEVE_TOOL_NAME}}]

    response = MagicMock()
    response.choices = None
    response.content = None
    response.output = [
        {
            "type": "function_call",
            "id": "fc_abc123",
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "arguments": json.dumps({"hash": "b573993006976af767214fac"}),
        }
    ]

    should_run, ctx = await guardrail.async_should_run_agentic_loop(
        response=response,
        model="gpt-4o",
        messages=[],
        tools=retrieve_tool_def,
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )

    assert should_run is True
    assert len(ctx["tool_calls"]) == 1
    assert ctx["tool_calls"][0]["arguments"]["hash"] == "b573993006976af767214fac"


@pytest.mark.asyncio
async def test_apply_guardrail_litellm_timeout_raises_when_fail_closed():
    guardrail = _make_guardrail()

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        side_effect=litellm.Timeout(
            message="Connection timed out after 10 seconds.",
            model="default-model-name",
            llm_provider="litellm-httpx-handler",
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )

    assert exc_info.value.status_code == 502
    assert "unreachable" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_apply_guardrail_litellm_timeout_fail_open_forwards_uncompressed():
    guardrail = _make_guardrail(unreachable_fallback="fail_open")

    inputs = GenericGuardrailAPIInputs(
        texts=["hello"],
        structured_messages=ORIGINAL_MESSAGES,
    )

    with patch.object(
        guardrail.async_handler,
        "post",
        new_callable=AsyncMock,
        side_effect=litellm.Timeout(
            message="Connection timed out after 10 seconds.",
            model="default-model-name",
            llm_provider="litellm-httpx-handler",
        ),
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="request",
        )

    assert result["structured_messages"] == ORIGINAL_MESSAGES
