"""
Unit tests for the Headroom Interception callback (server-side, non-guardrail
integration mode for Headroom compression).

Tests cover:
- async_pre_call_deployment_hook compresses for completion/acompletion and
  anthropic_messages/aanthropic_messages; no-ops for responses/aresponses and
  unrelated call types
- the _agentic_loop_depth guard prevents double-compression of the CCR follow-up call
- compression savings land in the shape compression_savings.py reads
- CCR round trip in "openai" tool-call format (tool injected into `tools`)
- CCR round trip in "hermes" tool-call format (inline <tool_call> text tags,
  no `tools` kwarg touched, system prompt gets the tool declaration)
- hash validation stays scoped per litellm_call_id
- unreachable_fallback="fail_closed"/"fail_open" behavior
- initialize_from_proxy_config parameter resolution
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.integrations.headroom_interception.client import HeadroomUnreachableError
from litellm.integrations.headroom_interception.handler import HeadroomInterceptionLogger
from litellm.proxy.spend_tracking.compression_savings import extract_compression_saved_tokens
from litellm.types.utils import CallTypes

FAKE_API_BASE = "https://headroom.example.com"
FAKE_API_KEY = "test-key"

ORIGINAL_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "A" * 5000},
]
COMPRESSED_MESSAGES_WITH_HASH = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summary. Retrieve more: <<ccr:b573993006aa,string,49.0KB>>"},
]


def _make_logger(**kwargs) -> HeadroomInterceptionLogger:
    defaults = dict(api_base=FAKE_API_BASE, api_key=FAKE_API_KEY)
    defaults.update(kwargs)
    return HeadroomInterceptionLogger(**defaults)


def _make_compress_response(messages: list, status: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = {
        "messages": messages,
        "tokens_before": 1000,
        "tokens_after": 100,
        "tokens_saved": 900,
        "compression_ratio": 0.1,
    }
    mock.text = ""
    return mock


def _make_retrieve_response(original_content: str, status: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = {"original_content": original_content}
    mock.text = original_content
    return mock


def _make_openai_chat_response_with_tool_call(tool_name: str, arguments: dict, tool_id: str = "call_abc") -> MagicMock:
    import json

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


def _make_hermes_text_response(text: str) -> MagicMock:
    message = MagicMock()
    message.content = text
    message.tool_calls = None

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def logger() -> HeadroomInterceptionLogger:
    return _make_logger()


@pytest.mark.asyncio
async def test_pre_call_hook_compresses_for_chat_completions(logger: HeadroomInterceptionLogger):
    mock_response = _make_compress_response(COMPRESSED_MESSAGES_WITH_HASH)
    with patch.object(logger._client.async_handler, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await logger.async_pre_call_deployment_hook(
            kwargs={"messages": ORIGINAL_MESSAGES, "model": "gpt-4o"},
            call_type=CallTypes.acompletion,
        )

    assert result is not None
    assert result["messages"] == COMPRESSED_MESSAGES_WITH_HASH
    assert any(
        t.get("function", {}).get("name") == "headroom_retrieve" for t in result["tools"] if isinstance(t, dict)
    )


@pytest.mark.asyncio
async def test_pre_call_hook_compresses_for_anthropic_messages(logger: HeadroomInterceptionLogger):
    mock_response = _make_compress_response(COMPRESSED_MESSAGES_WITH_HASH)
    with patch.object(logger._client.async_handler, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await logger.async_pre_call_deployment_hook(
            kwargs={"messages": ORIGINAL_MESSAGES, "model": "claude-3-5-sonnet"},
            call_type=CallTypes.anthropic_messages,
        )

    assert result is not None
    assert result["messages"] == COMPRESSED_MESSAGES_WITH_HASH


@pytest.mark.asyncio
async def test_pre_call_hook_noop_for_responses_api(logger: HeadroomInterceptionLogger):
    """Responses API is explicitly out of scope: `input` isn't message-shaped."""
    with patch.object(logger._client.async_handler, "post", new_callable=AsyncMock) as mock_post:
        result = await logger.async_pre_call_deployment_hook(
            kwargs={"input": "hello", "model": "gpt-4o"},
            call_type=CallTypes.aresponses,
        )

    assert result is None
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_pre_call_hook_noop_for_unrelated_call_type(logger: HeadroomInterceptionLogger):
    with patch.object(logger._client.async_handler, "post", new_callable=AsyncMock) as mock_post:
        result = await logger.async_pre_call_deployment_hook(
            kwargs={"input": "hello world"},
            call_type=CallTypes.aembedding,
        )

    assert result is None
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_pre_call_hook_skips_when_agentic_loop_depth_positive(logger: HeadroomInterceptionLogger):
    """Regression guard: without this check the CCR follow-up call itself would
    get re-compressed, corrupting the retrieval round trip."""
    with patch.object(logger._client.async_handler, "post", new_callable=AsyncMock) as mock_post:
        result = await logger.async_pre_call_deployment_hook(
            kwargs={"messages": ORIGINAL_MESSAGES, "model": "gpt-4o", "_agentic_loop_depth": 1},
            call_type=CallTypes.acompletion,
        )

    assert result is None
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_pre_call_hook_records_compression_savings_readable_by_spend_tracking(
    logger: HeadroomInterceptionLogger,
):
    """Regression test for the spend-tracking wiring: assert against the real
    extract_compression_saved_tokens(), not just the written shape, so this
    fails if the key/shape ever drifts from what spend tracking reads."""
    mock_response = _make_compress_response(COMPRESSED_MESSAGES_WITH_HASH)
    with patch.object(logger._client.async_handler, "post", new_callable=AsyncMock, return_value=mock_response):
        kwargs = {"messages": ORIGINAL_MESSAGES, "model": "gpt-4o"}
        result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=CallTypes.acompletion)

    assert result is not None
    metadata = result["litellm_metadata"]
    assert extract_compression_saved_tokens(metadata) == 900


@pytest.mark.asyncio
async def test_ccr_openai_format_round_trip(logger: HeadroomInterceptionLogger):
    original_content = "This is the full compressed content."
    mock_retrieve = _make_retrieve_response(original_content)
    response = _make_openai_chat_response_with_tool_call(
        tool_name="headroom_retrieve",
        arguments={"hash": "b573993006976af767214fac"},
        tool_id="call_abc",
    )
    tools = [{"type": "function", "function": {"name": "headroom_retrieve"}}]

    should_run, ctx = await logger.async_should_run_agentic_loop(
        response=response,
        model="gpt-4o",
        messages=[],
        tools=tools,
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )
    assert should_run is True

    logger._client._issued_hashes_by_call_id["call-1"] = (
        frozenset({"b573993006976af767214fac"}),
        time.monotonic() + 999,
    )
    with patch.object(logger._client.async_handler, "get", new_callable=AsyncMock, return_value=mock_retrieve):
        plan = await logger.async_build_agentic_loop_plan(
            tools=ctx,
            model="gpt-4o",
            messages=[{"role": "user", "content": "What does it say?"}],
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=None,
            stream=False,
            kwargs={"litellm_call_id": "call-1"},
        )

    follow_up = plan.request_patch.messages
    tool_message = next(m for m in follow_up if m.get("role") == "tool")
    assert tool_message["content"] == original_content
    assert tool_message["tool_call_id"] == "call_abc"
    # openai format never touches the system prompt
    assert not any("headroom_retrieve" in str(m.get("content", "")) for m in follow_up if m.get("role") == "system")


@pytest.mark.asyncio
async def test_hermes_injects_tool_declaration_into_new_system_message(logger: HeadroomInterceptionLogger):
    logger_hermes = _make_logger(tool_call_format="hermes")
    mock_response = _make_compress_response(COMPRESSED_MESSAGES_WITH_HASH)
    with patch.object(logger_hermes._client.async_handler, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await logger_hermes.async_pre_call_deployment_hook(
            kwargs={"messages": [{"role": "user", "content": "hi"}], "model": "local-hermes"},
            call_type=CallTypes.acompletion,
        )

    assert result is not None
    assert "tools" not in result  # hermes never populates the `tools` kwarg
    system_message = result["messages"][0]
    assert system_message["role"] == "system"
    assert "headroom_retrieve" in system_message["content"]
    assert "<tool_call>" in system_message["content"]


@pytest.mark.asyncio
async def test_hermes_prepends_to_existing_system_message(logger: HeadroomInterceptionLogger):
    logger_hermes = _make_logger(tool_call_format="hermes")
    compressed_with_original_system_message = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Summary. Retrieve more: <<ccr:b573993006aa,string,49.0KB>>"},
    ]
    mock_response = _make_compress_response(compressed_with_original_system_message)
    with patch.object(logger_hermes._client.async_handler, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await logger_hermes.async_pre_call_deployment_hook(
            kwargs={
                "messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "hi"}],
                "model": "local-hermes",
            },
            call_type=CallTypes.acompletion,
        )

    assert result is not None
    assert len(result["messages"]) == 2
    system_message = result["messages"][0]
    assert "headroom_retrieve" in system_message["content"]
    assert "You are terse." in system_message["content"]


@pytest.mark.asyncio
async def test_hermes_ccr_round_trip_multiple_tool_calls(logger: HeadroomInterceptionLogger):
    logger_hermes = _make_logger(tool_call_format="hermes")
    logger_hermes._client._issued_hashes_by_call_id["call-1"] = (
        frozenset({"aaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbb"}),
        time.monotonic() + 999,
    )
    raw_text = (
        "Let me look those up.\n"
        '<tool_call>\n{"name": "headroom_retrieve", "arguments": {"hash": "aaaaaaaaaaaaaaaaaaaaaaaa"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "headroom_retrieve", "arguments": {"hash": "bbbbbbbbbbbbbbbbbbbbbbbb"}}\n</tool_call>'
    )
    response = _make_hermes_text_response(raw_text)

    should_run, ctx = await logger_hermes.async_should_run_agentic_loop(
        response=response,
        model="local-hermes",
        messages=[],
        tools=None,
        stream=False,
        custom_llm_provider="hosted_vllm",
        kwargs={},
    )
    assert should_run is True
    assert len(ctx["tool_calls"]) == 2

    mock_retrieve_a = _make_retrieve_response("Content A")
    mock_retrieve_b = _make_retrieve_response("Content B")
    with patch.object(
        logger_hermes._client.async_handler,
        "get",
        new_callable=AsyncMock,
        side_effect=[mock_retrieve_a, mock_retrieve_b],
    ):
        plan = await logger_hermes.async_build_agentic_loop_plan(
            tools=ctx,
            model="local-hermes",
            messages=[{"role": "user", "content": "look these up"}],
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=None,
            stream=False,
            kwargs={"litellm_call_id": "call-1"},
        )

    follow_up = plan.request_patch.messages
    assistant_message = next(m for m in follow_up if m.get("role") == "assistant")
    assert assistant_message["content"] == raw_text  # echoed verbatim

    tool_messages = [m for m in follow_up if m.get("role") == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[0]["content"] == "<tool_response>\nContent A\n</tool_response>"
    assert tool_messages[1]["content"] == "<tool_response>\nContent B\n</tool_response>"


@pytest.mark.asyncio
async def test_hermes_skips_malformed_tool_call_block_but_processes_valid_ones(logger: HeadroomInterceptionLogger):
    logger_hermes = _make_logger(tool_call_format="hermes")
    raw_text = (
        "<tool_call>\nnot valid json\n</tool_call>\n"
        '<tool_call>\n{"name": "headroom_retrieve", "arguments": {"hash": "cccccccccccccccccccccccc"}}\n</tool_call>'
    )
    response = _make_hermes_text_response(raw_text)

    should_run, ctx = await logger_hermes.async_should_run_agentic_loop(
        response=response,
        model="local-hermes",
        messages=[],
        tools=None,
        stream=False,
        custom_llm_provider="hosted_vllm",
        kwargs={},
    )

    assert should_run is True
    assert len(ctx["tool_calls"]) == 1
    assert ctx["tool_calls"][0]["arguments"]["hash"] == "cccccccccccccccccccccccc"


@pytest.mark.asyncio
async def test_hermes_no_tool_call_tags_does_not_run_agentic_loop(logger: HeadroomInterceptionLogger):
    logger_hermes = _make_logger(tool_call_format="hermes")
    response = _make_hermes_text_response("Just a normal answer, no tool calls here.")

    should_run, ctx = await logger_hermes.async_should_run_agentic_loop(
        response=response,
        model="local-hermes",
        messages=[],
        tools=None,
        stream=False,
        custom_llm_provider="hosted_vllm",
        kwargs={},
    )

    assert should_run is False
    assert ctx == {}


@pytest.mark.asyncio
async def test_hash_rejected_when_not_issued_for_this_call_id(logger: HeadroomInterceptionLogger):
    response = _make_openai_chat_response_with_tool_call(
        tool_name="headroom_retrieve",
        arguments={"hash": "deadbeef000000000000dead"},
        tool_id="call_xyz",
    )
    assert not logger._client._issued_hashes_by_call_id

    with patch.object(logger._client.async_handler, "get", new_callable=AsyncMock) as mock_get:
        plan = await logger.async_build_agentic_loop_plan(
            tools={
                "tool_calls": [
                    {"id": "call_xyz", "name": "headroom_retrieve", "arguments": {"hash": "deadbeef000000000000dead"}}
                ]
            },
            model="gpt-4o",
            messages=[{"role": "user", "content": "fetch it"}],
            response=response,
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={},
            logging_obj=None,
            stream=False,
            kwargs={"litellm_call_id": "call-unknown"},
        )

    mock_get.assert_not_called()
    follow_up = plan.request_patch.messages
    tool_message = next(m for m in follow_up if m.get("role") == "tool")
    assert "was not produced by the current request" in tool_message["content"]


@pytest.mark.asyncio
async def test_unreachable_fail_closed_raises_headroom_unreachable_error(logger: HeadroomInterceptionLogger):
    """Regression test for the error-handling reconciliation: a plain litellm
    exception (not fastapi.HTTPException) must propagate uncaught from the
    SDK-layer pre-call hook."""
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.text = "service unavailable"

    with patch.object(logger._client.async_handler, "post", new_callable=AsyncMock, return_value=mock_response):
        with pytest.raises(HeadroomUnreachableError):
            await logger.async_pre_call_deployment_hook(
                kwargs={"messages": ORIGINAL_MESSAGES, "model": "gpt-4o"},
                call_type=CallTypes.acompletion,
            )


@pytest.mark.asyncio
async def test_unreachable_fail_open_forwards_uncompressed(logger: HeadroomInterceptionLogger):
    logger_fail_open = _make_logger(unreachable_fallback="fail_open")
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.text = "service unavailable"

    with patch.object(
        logger_fail_open._client.async_handler, "post", new_callable=AsyncMock, return_value=mock_response
    ):
        result = await logger_fail_open.async_pre_call_deployment_hook(
            kwargs={"messages": ORIGINAL_MESSAGES, "model": "gpt-4o"},
            call_type=CallTypes.acompletion,
        )

    # compression_applied is False on fail_open (no compression happened),
    # so the pre-call hook is a no-op and the original messages are untouched.
    assert result is None


def test_init_raises_without_api_base():
    with pytest.raises(ValueError):
        HeadroomInterceptionLogger(api_base=None, api_key=FAKE_API_KEY)


def test_initialize_from_proxy_config_reads_litellm_settings():
    logger = HeadroomInterceptionLogger.initialize_from_proxy_config(
        litellm_settings={
            "headroom_interception_params": {
                "api_base": FAKE_API_BASE,
                "api_key": FAKE_API_KEY,
                "tool_call_format": "hermes",
            }
        },
        callback_specific_params={},
    )
    assert logger._client.headroom_api_base == FAKE_API_BASE
    assert logger._client.tool_call_format == "hermes"


def test_initialize_from_proxy_config_reads_callback_specific_params():
    logger = HeadroomInterceptionLogger.initialize_from_proxy_config(
        litellm_settings={},
        callback_specific_params={
            "headroom_interception": {"api_base": FAKE_API_BASE, "api_key": FAKE_API_KEY}
        },
    )
    assert logger._client.headroom_api_base == FAKE_API_BASE
    assert logger._client.tool_call_format == "openai"
