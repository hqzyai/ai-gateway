"""
Tests for the provider-agnostic chat completion agentic loop dispatcher
(`litellm/litellm_core_utils/chat_completion_agentic_loop.py`) and the
code-interpreter interception integration that drives it.

The load-bearing regression here protects a reviewer requirement: the internal
agentic/interception control fields must NEVER reach the outbound provider HTTP
request body. The relevant fields are:

    _agentic_loop_depth
    _agentic_loop_fingerprints
    _agentic_loop_api_surface
    max_agentic_loops
    _code_interpreter_interception_active
    _code_interpreter_interception_sandbox_key
    _code_interpreter_interception_converted_stream

A scrubber in gpt_transformation.py used to strip these. That scrubber was
removed, so `test_internal_control_fields_never_leak_into_provider_body` proves
they stay out of the body even without it.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.code_interpreter_interception.handler import (
    CodeInterpreterInterceptionLogger,
)
from litellm.litellm_core_utils.chat_completion_agentic_loop import (
    AgenticLoopRoutingContext,
    get_agentic_loop_routing_context,
    maybe_run_chat_completion_agentic_loop,
    reset_agentic_loop_routing_context,
    set_agentic_loop_routing_context,
)
from litellm.types.integrations.custom_logger import (
    AgenticLoopPlan,
    AgenticLoopRequestPatch,
)
from litellm.types.utils import (
    Choices,
    Function,
    ChatCompletionMessageToolCall,
    Message,
    ModelResponse,
    Usage,
)

# The internal control fields that must never reach a provider request body.
_INTERNAL_CONTROL_FIELDS = (
    "_agentic_loop_depth",
    "_agentic_loop_fingerprints",
    "_agentic_loop_api_surface",
    "max_agentic_loops",
    "_code_interpreter_interception_active",
    "_code_interpreter_interception_sandbox_key",
    "_code_interpreter_interception_converted_stream",
    "litellm_metadata",
)


@pytest.fixture
def restore_callbacks():
    """Save/restore litellm.callbacks so a registered fake logger never pollutes
    other tests in the suite."""
    saved = list(litellm.callbacks)
    try:
        yield
    finally:
        litellm.callbacks = saved


class _SandboxResult:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.error = None


class FakeSandboxConfig:
    """Injected sandbox so the interception loop runs no real network / E2B."""

    def __init__(self) -> None:
        self.created = 0
        self.deleted = 0
        self.run_codes: List[str] = []

    async def acreate_sandbox(self) -> Any:
        self.created += 1
        return MagicMock(id="sandbox-123")

    async def arun_code(self, container: Any, code: str) -> _SandboxResult:
        self.run_codes.append(code)
        return _SandboxResult(stdout="42\n")

    async def adelete_sandbox(self, container: Any) -> None:
        self.deleted += 1


def _tool_call_model_response(
    usage: Optional[Usage] = None,
    tool_name: str = "litellm_code_execution",
) -> ModelResponse:
    response = ModelResponse(
        choices=[
            Choices(
                finish_reason="tool_calls",
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_abc",
                            type="function",
                            function=Function(
                                name=tool_name,
                                arguments='{"code": "print(6*7)"}',
                            ),
                        )
                    ],
                ),
            )
        ]
    )
    if usage is not None:
        setattr(response, "usage", usage)
    return response


def _plain_model_response(content: str = "The answer is 42", usage: Optional[Usage] = None) -> ModelResponse:
    response = ModelResponse(
        choices=[
            Choices(
                finish_reason="stop",
                message=Message(role="assistant", content=content),
            )
        ]
    )
    if usage is not None:
        setattr(response, "usage", usage)
    return response


def _raw_response_for(model_response: ModelResponse) -> MagicMock:
    """Wrap a ModelResponse as the OpenAI `with_raw_response.create` return value
    (an object exposing `.headers` and `.parse()` -> something with model_dump)."""
    parsed = MagicMock()
    parsed.model_dump.return_value = model_response.model_dump()
    raw = MagicMock()
    raw.headers = {}
    raw.parse.return_value = parsed
    return raw


# ---------------------------------------------------------------------------
# A) PROVIDER-PAYLOAD REGRESSION
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_internal_control_fields_never_leak_into_provider_body(restore_callbacks):
    """Drive a real acompletion with a native code_interpreter tool through the
    interception logger + agentic loop, capturing every outbound OpenAI request
    body. None of the internal control fields may appear at top-level or inside
    extra_body on ANY of the captured calls."""
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandboxConfig())
    litellm.callbacks = [logger]

    # First create -> model emits a code_execution tool call (triggers the loop).
    # Second create -> model returns a plain answer (loop terminates).
    create = AsyncMock(
        side_effect=[
            _raw_response_for(_tool_call_model_response()),
            _raw_response_for(_plain_model_response()),
        ]
    )
    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create = create

    response = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "what is 6*7?"}],
        tools=[{"type": "code_interpreter"}],
        tool_choice={"type": "code_interpreter"},
        api_key="sk-test",
        client=mock_client,
    )

    # The loop must have actually fired (sanity: two provider calls).
    assert create.await_count == 2, (
        f"expected the agentic loop to issue a follow-up provider call; got {create.await_count} call(s)"
    )

    for idx, call in enumerate(create.await_args_list):
        body = call.kwargs
        extra_body = body.get("extra_body") or {}
        for field in _INTERNAL_CONTROL_FIELDS:
            assert field not in body, (
                f"provider call #{idx}: internal field {field!r} leaked into "
                f"top-level request body: {sorted(body.keys())}"
            )
            assert field not in extra_body, (
                f"provider call #{idx}: internal field {field!r} leaked into extra_body: {sorted(extra_body.keys())}"
            )
        # The native code_interpreter tool must have been swapped for the
        # function tool, never sent raw to OpenAI as a chat-completions request.
        for tool in body.get("tools") or []:
            assert tool.get("type") != "code_interpreter"

    # The final response is the post-loop answer, not the tool-call turn.
    assert response.choices[0].message.content == "The answer is 42"


class _UsageSpyLogger(CustomLogger):
    """Records every success-logging event the way a spend-log callback would."""

    def __init__(self) -> None:
        super().__init__()
        self.events: List[Tuple[Optional[str], Optional[int], Optional[int]]] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        usage = getattr(response_obj, "usage", None)
        self.events.append(
            (
                kwargs.get("litellm_call_id"),
                getattr(usage, "prompt_tokens", None),
                getattr(usage, "completion_tokens", None),
            )
        )


async def _settle_async_logging(spy: _UsageSpyLogger, expected: int = 1, timeout: float = 5.0) -> None:
    """Wait for the expected success events, then let any extra ones land."""
    import asyncio
    import time

    deadline = time.monotonic() + timeout
    while len(spy.events) < expected and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.5)


@pytest.mark.asyncio
async def test_agentic_loop_bills_every_turn_exactly_once(restore_callbacks):
    """Regression: an agentic loop issues N provider calls for one client request,
    but only the last response reaches the outer logging object.

    Before the usage fold, the pre-loop turn's tokens were billed by nobody and
    the follow-up's tokens were billed twice (once by the nested acompletion's
    own logging, once by the outer one). Both halves are pinned here: the spend
    event must fire once, and its token counts must be the sum over every
    provider call actually made."""
    logger = CodeInterpreterInterceptionLogger(sandbox_config=FakeSandboxConfig())
    spy = _UsageSpyLogger()
    litellm.callbacks = [logger, spy]

    create = AsyncMock(
        side_effect=[
            _raw_response_for(
                _tool_call_model_response(usage=Usage(prompt_tokens=1000, completion_tokens=20, total_tokens=1020))
            ),
            _raw_response_for(
                _plain_model_response(usage=Usage(prompt_tokens=1500, completion_tokens=30, total_tokens=1530))
            ),
        ]
    )
    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create = create

    response = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "what is 6*7?"}],
        tools=[{"type": "code_interpreter"}],
        tool_choice={"type": "code_interpreter"},
        api_key="sk-test",
        client=mock_client,
    )

    assert create.await_count == 2, (
        f"expected the agentic loop to issue a follow-up provider call; got {create.await_count} call(s)"
    )

    assert response.usage.prompt_tokens == 2500
    assert response.usage.completion_tokens == 50
    assert response.usage.total_tokens == 2550

    await _settle_async_logging(spy)

    assert len(spy.events) == 1, f"expected exactly one spend event for the whole loop; got {spy.events}"
    _call_id, logged_prompt_tokens, logged_completion_tokens = spy.events[0]
    assert logged_prompt_tokens == 2500
    assert logged_completion_tokens == 50


class _ToolCallGatedLogger(CustomLogger):
    """Gate that fires only while the model is still emitting tool calls, so the
    loop runs exactly one follow-up turn and then terminates naturally."""

    def __init__(self, follow_up_messages: List[Dict[str, Any]]) -> None:
        super().__init__()
        self._follow_up_messages = follow_up_messages

    @staticmethod
    def _tool_calls_of(response: Any) -> Optional[List[Any]]:
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        return getattr(getattr(choices[0], "message", None), "tool_calls", None)

    async def async_should_run_agentic_loop(
        self,
        response: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        stream: bool,
        custom_llm_provider: str,
        kwargs: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        tool_calls = self._tool_calls_of(response)
        if not tool_calls:
            return False, {}
        return True, {"tool_calls": [{"id": "call_abc", "name": "litellm_code_execution"}]}

    async def async_build_agentic_loop_plan(
        self,
        tools: Dict[str, Any],
        model: str,
        messages: List[Dict[str, Any]],
        response: Any,
        anthropic_messages_provider_config: Any,
        anthropic_messages_optional_request_params: Dict[str, Any],
        logging_obj: Any,
        stream: bool,
        kwargs: Dict[str, Any],
    ) -> AgenticLoopPlan:
        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=AgenticLoopRequestPatch(
                model=model,
                messages=self._follow_up_messages,
                max_tokens=200,
                optional_params={},
                kwargs={},
            ),
        )


def _openai_style_stream_chunks(
    completion_id: str, prompt_tokens: int, completion_tokens: int, emit_tool_call: bool
) -> List[bytes]:
    """SSE for one streamed turn, ending with a usage-carrying chunk."""
    import json as _json

    envelope = {"id": completion_id, "object": "chat.completion.chunk", "created": 0, "model": "m"}
    if emit_tool_call:
        first_delta: Dict[str, Any] = {
            "role": "assistant",
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "litellm_code_execution",
                        "arguments": _json.dumps({"code": "print(6*7)"}),
                    },
                }
            ],
        }
        final_reason = "tool_calls"
    else:
        first_delta = {"role": "assistant", "content": "The answer is 42"}
        final_reason = "stop"

    payloads = [
        {**envelope, "choices": [{"index": 0, "delta": first_delta, "finish_reason": None}]},
        {**envelope, "choices": [{"index": 0, "delta": {}, "finish_reason": final_reason}]},
        {
            **envelope,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
    ]
    return [f"data: {_json.dumps(p)}\n\n".encode() for p in payloads] + [b"data: [DONE]\n\n"]


class _FakeUpstreamResponse:
    """Minimal httpx-response stand-in for the shared httpx LLM handler."""

    status_code = 200
    headers: Dict[str, str] = {}
    text = ""

    def __init__(self, sse_lines: List[bytes]) -> None:
        self._sse_lines = sse_lines

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self, chunk_size: Optional[int] = None):
        for line in self._sse_lines:
            yield line

    async def aiter_lines(self):
        for line in self._sse_lines:
            yield line.decode().rstrip("\n")


class _FakeUpstreamJsonResponse:
    """Non-streaming httpx-response stand-in; the agentic follow-up never streams."""

    status_code = 200
    headers: Dict[str, str] = {}
    text = ""

    def __init__(self, completion_id: str, prompt_tokens: int, completion_tokens: int) -> None:
        self._body = {
            "id": completion_id,
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "The answer is 42"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._body


@pytest.mark.asyncio
async def test_streaming_agentic_loop_bills_every_turn_under_a_distinct_request_id(restore_callbacks):
    """Regression: on the streaming path the pre-loop turn is billed by its own
    stream wrapper before the loop even runs, so the non-streaming fix (fold the
    parent's tokens into the follow-up) would double-count it.

    The follow-up must therefore bill itself, and it must do so under its own
    litellm_call_id: SpendLogs keys rows by request id and inserts them with
    skip_duplicates=True, so reusing the parent's id silently drops this turn's
    row and the tokens vanish from the bill. Both properties are pinned here,
    together with the end-to-end equality billed == sent."""
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    turn_usage = [(1000, 20), (1500, 30)]
    # A plain gate keeps the request genuinely streaming; the code-interpreter
    # logger converts streaming requests to non-streaming, which is the other branch.
    logger = _ToolCallGatedLogger(
        follow_up_messages=[
            {"role": "user", "content": "what is 6*7?"},
            {"role": "user", "content": "tool result"},
        ]
    )
    spy = _UsageSpyLogger()
    litellm.callbacks = [logger, spy]

    post = AsyncMock(
        side_effect=[
            _FakeUpstreamResponse(_openai_style_stream_chunks("chatcmpl-1", 1000, 20, emit_tool_call=True)),
            _FakeUpstreamJsonResponse("chatcmpl-2", 1500, 30),
        ]
    )
    mock_client = AsyncMock(spec=AsyncHTTPHandler)
    mock_client.post = post

    stream = await litellm.acompletion(
        model="hosted_vllm/m",
        messages=[{"role": "user", "content": "what is 6*7?"}],
        api_key="sk-test",
        api_base="http://localhost:9999",
        stream=True,
        stream_options={"include_usage": True},
        client=mock_client,
    )
    async for _chunk in stream:
        pass

    assert post.await_count == 2, (
        f"expected the agentic loop to issue a follow-up provider call; got {post.await_count} call(s)"
    )

    await _settle_async_logging(spy, expected=2)

    sent_prompt = sum(prompt for prompt, _completion in turn_usage)
    sent_completion = sum(completion for _prompt, completion in turn_usage)
    billed_prompt = sum(event[1] for event in spy.events)
    billed_completion = sum(event[2] for event in spy.events)

    assert (billed_prompt, billed_completion) == (sent_prompt, sent_completion), (
        f"billed tokens must equal tokens sent upstream; sent={(sent_prompt, sent_completion)} "
        f"billed={(billed_prompt, billed_completion)} events={spy.events}"
    )

    call_ids = [event[0] for event in spy.events]
    assert len(set(call_ids)) == len(call_ids), (
        f"each turn needs its own request id or SpendLogs drops one as a duplicate; got {call_ids}"
    )


# ---------------------------------------------------------------------------
# B) DISPATCHER UNIT TESTS
# ---------------------------------------------------------------------------


class _LoggingStub:
    """Minimal logging_obj: dispatcher only reads dynamic_success_callbacks and
    litellm_call_id off it."""

    litellm_call_id = "call-test"
    dynamic_success_callbacks: List[Any] = []


class _GateOnlyLogger(CustomLogger):
    """Overrides the gate to fire, but builds a plan from request_patch."""

    def __init__(self, plan: AgenticLoopPlan, tool_calls: Dict[str, Any]) -> None:
        super().__init__()
        self._plan = plan
        self._tool_calls = tool_calls
        self.cleanup_calls = 0

    async def async_should_run_agentic_loop(
        self,
        response: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        stream: bool,
        custom_llm_provider: str,
        kwargs: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        return True, self._tool_calls

    async def async_build_agentic_loop_plan(
        self,
        tools: Dict[str, Any],
        model: str,
        messages: List[Dict[str, Any]],
        response: Any,
        anthropic_messages_provider_config: Any,
        anthropic_messages_optional_request_params: Dict[str, Any],
        logging_obj: Any,
        stream: bool,
        kwargs: Dict[str, Any],
    ) -> AgenticLoopPlan:
        return self._plan

    async def async_agentic_loop_cleanup_hook(self, plan: AgenticLoopPlan, kwargs: Dict[str, Any]) -> None:
        self.cleanup_calls += 1


class _ToolCallOnlyLogger(_GateOnlyLogger):
    async def async_should_run_agentic_loop(
        self,
        response: Any,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        stream: bool,
        custom_llm_provider: str,
        kwargs: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        choices = getattr(response, "choices", None)
        finish_reason = getattr(choices[0], "finish_reason", None) if isinstance(choices, list) and choices else None
        if finish_reason != "tool_calls":
            return False, {}
        return await super().async_should_run_agentic_loop(
            response=response,
            model=model,
            messages=messages,
            tools=tools,
            stream=stream,
            custom_llm_provider=custom_llm_provider,
            kwargs=kwargs,
        )


def _patched_messages() -> List[Dict[str, Any]]:
    return [
        {"role": "user", "content": "what is 6*7?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "litellm_code_execution",
                        "arguments": '{"code": "print(6*7)"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": "42\n"},
    ]


@pytest.mark.asyncio
async def test_dispatcher_returns_none_when_no_callback_gates(restore_callbacks):
    """No callback overrides the gate -> dispatcher returns None so the caller
    keeps the original response untouched."""
    litellm.callbacks = []

    result = await maybe_run_chat_completion_agentic_loop(
        response=_plain_model_response(),
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        kwargs={},
        logging_obj=_LoggingStub(),
        custom_llm_provider="openai",
        stream=False,
    )

    assert result is None


@pytest.mark.asyncio
async def test_dispatcher_runs_followup_with_incremented_depth_and_patched_messages(
    restore_callbacks,
):
    """A gating logger with a request_patch -> the dispatcher calls
    litellm.acompletion exactly once with _agentic_loop_depth == 1 and the
    patched messages. Loop-control state rides as litellm-level kwargs and is
    mirrored into litellm_metadata; the provider-surface transient
    _agentic_loop_api_surface is never forwarded. (Provider-body stripping of
    these litellm-level kwargs is asserted separately in test A.)"""
    followup = _plain_model_response("done")
    plan = AgenticLoopPlan(
        run_agentic_loop=True,
        request_patch=AgenticLoopRequestPatch(messages=_patched_messages()),
    )
    logger = _GateOnlyLogger(plan=plan, tool_calls={"tool_calls": [{"id": "call_abc"}]})
    litellm.callbacks = [logger]

    acompletion_mock = AsyncMock(return_value=followup)
    with patch.object(litellm, "acompletion", acompletion_mock):
        result = await maybe_run_chat_completion_agentic_loop(
            response=_tool_call_model_response(),
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "what is 6*7?"}],
            optional_params={"temperature": 0.1},
            kwargs={"_code_interpreter_interception_active": True},
            logging_obj=_LoggingStub(),
            custom_llm_provider="openai",
            stream=False,
        )

    assert result is followup
    acompletion_mock.assert_awaited_once()
    call_kwargs = acompletion_mock.await_args.kwargs

    assert call_kwargs["_agentic_loop_depth"] == 1
    assert call_kwargs["messages"] == _patched_messages()
    # Preserved non-internal optional param survives the rerun.
    assert call_kwargs["temperature"] == 0.1
    # Loop-control state is carried at the litellm level for the follow-up.
    assert call_kwargs["max_agentic_loops"] >= 1
    assert "_agentic_loop_fingerprints" in call_kwargs
    # Interception markers are mirrored into litellm_metadata for the follow-up.
    assert call_kwargs["litellm_metadata"]["_code_interpreter_interception_active"] is True
    # The transient surface marker is NOT forwarded to the follow-up call.
    assert "_agentic_loop_api_surface" not in call_kwargs
    # Cleanup hook always runs.
    assert logger.cleanup_calls == 1


@pytest.mark.asyncio
async def test_dispatcher_routes_followup_through_original_router_model_group(
    restore_callbacks,
):
    followup = _plain_model_response("fallback answer")
    plan = AgenticLoopPlan(
        run_agentic_loop=True,
        request_patch=AgenticLoopRequestPatch(messages=_patched_messages()),
    )
    logger = _GateOnlyLogger(plan=plan, tool_calls={"tool_calls": [{"id": "call_abc"}]})
    litellm.callbacks = [logger]
    router_acompletion = AsyncMock(return_value=followup)
    direct_acompletion = AsyncMock()
    routing_token = set_agentic_loop_routing_context(
        model_group="primary-model-group",
        acompletion=router_acompletion,
    )

    try:
        with patch.object(litellm, "acompletion", direct_acompletion):
            result = await maybe_run_chat_completion_agentic_loop(
                response=_tool_call_model_response(),
                model="provider/deployment-model",
                messages=[{"role": "user", "content": "what is 6*7?"}],
                optional_params={
                    "api_key": "deployment-secret",
                    "base_url": "https://deployment.invalid",
                    "temperature": 0.1,
                },
                kwargs={"client": object(), "model_info": {"id": "deployment-id"}},
                logging_obj=_LoggingStub(),
                custom_llm_provider="provider",
                stream=False,
            )
    finally:
        reset_agentic_loop_routing_context(routing_token)

    assert result is followup
    direct_acompletion.assert_not_awaited()
    router_acompletion.assert_awaited_once()
    call_kwargs = router_acompletion.await_args.kwargs
    assert call_kwargs["model"] == "primary-model-group"
    assert call_kwargs["messages"] == _patched_messages()
    assert call_kwargs["temperature"] == 0.1
    assert "api_key" not in call_kwargs
    assert "base_url" not in call_kwargs
    assert "client" not in call_kwargs
    assert "model_info" not in call_kwargs


@pytest.mark.asyncio
async def test_router_sets_and_clears_agentic_followup_context():
    router = litellm.Router(model_list=[])
    response = _plain_model_response("done")
    observed_contexts: list[AgenticLoopRoutingContext | None] = []

    async def routed_call(**kwargs: object) -> ModelResponse:
        observed_contexts.append(get_agentic_loop_routing_context())
        return response

    try:
        with patch.object(router, "async_function_with_fallbacks", side_effect=routed_call):
            result = await router.acompletion(
                model="primary-model-group",
                messages=[{"role": "user", "content": "hi"}],
            )
    finally:
        router.discard()

    assert result is response
    assert len(observed_contexts) == 1
    assert observed_contexts[0] is not None
    assert observed_contexts[0].model_group == "primary-model-group"
    assert get_agentic_loop_routing_context() is None


@pytest.mark.asyncio
async def test_agentic_followup_uses_router_fallback_after_primary_rate_limit(
    restore_callbacks,
):
    router = litellm.Router(
        model_list=[
            {
                "model_name": "primary-model-group",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "primary-key",
                    "mock_response": "litellm.RateLimitError",
                },
            },
            {
                "model_name": "fallback-model-group",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "fallback-key",
                    "mock_response": "fallback answer",
                },
            },
        ],
        fallbacks=[{"primary-model-group": ["fallback-model-group"]}],
        num_retries=0,
    )
    plan = AgenticLoopPlan(
        run_agentic_loop=True,
        request_patch=AgenticLoopRequestPatch(messages=_patched_messages()),
    )
    logger = _ToolCallOnlyLogger(plan=plan, tool_calls={"tool_calls": [{"id": "call_abc"}]})
    litellm.callbacks = [logger]
    routing_token = set_agentic_loop_routing_context(
        model_group="primary-model-group",
        acompletion=router.acompletion,
    )

    try:
        result = await maybe_run_chat_completion_agentic_loop(
            response=_tool_call_model_response(),
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "what is 6*7?"}],
            optional_params={"api_key": "stale-deployment-key"},
            kwargs={},
            logging_obj=_LoggingStub(),
            custom_llm_provider="openai",
            stream=False,
        )
    finally:
        reset_agentic_loop_routing_context(routing_token)
        router.discard()

    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "fallback answer"


@pytest.mark.asyncio
async def test_dispatcher_raises_when_depth_reaches_max_agentic_loops(
    restore_callbacks,
):
    """depth >= max_agentic_loops -> ValueError mentioning max_agentic_loops,
    before any follow-up call is attempted."""
    logger = _GateOnlyLogger(
        plan=AgenticLoopPlan(run_agentic_loop=True),
        tool_calls={"tool_calls": [{"id": "call_abc"}]},
    )
    litellm.callbacks = [logger]

    acompletion_mock = AsyncMock()
    with patch.object(litellm, "acompletion", acompletion_mock):
        with pytest.raises(ValueError, match="max_agentic_loops"):
            await maybe_run_chat_completion_agentic_loop(
                response=_tool_call_model_response(),
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={},
                kwargs={"_agentic_loop_depth": 3, "max_agentic_loops": 3},
                logging_obj=_LoggingStub(),
                custom_llm_provider="openai",
                stream=False,
            )

    acompletion_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_headroom_safety_limit_preserves_final_turn_usage(restore_callbacks):
    logger = _GateOnlyLogger(
        plan=AgenticLoopPlan(run_agentic_loop=True),
        tool_calls={"tool_calls": [{"id": "call_abc"}]},
    )
    litellm.callbacks = [logger]
    usage = Usage(prompt_tokens=80_000, completion_tokens=400, total_tokens=80_400)
    response = _tool_call_model_response(usage=usage, tool_name="headroom_retrieve")

    acompletion_mock = AsyncMock()
    with patch.object(litellm, "acompletion", acompletion_mock):
        result = await maybe_run_chat_completion_agentic_loop(
            response=response,
            model="qwen3.8-max",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            kwargs={"_agentic_loop_depth": 3, "max_agentic_loops": 3},
            logging_obj=_LoggingStub(),
            custom_llm_provider="openai",
            stream=False,
        )

    assert isinstance(result, ModelResponse)
    assert result.usage.prompt_tokens == 80_000
    assert result.usage.completion_tokens == 400
    assert result.usage.total_tokens == 80_400
    assert result.choices[0].finish_reason == "stop"
    assert result.choices[0].message.tool_calls is None
    acompletion_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatcher_raises_on_repeated_tool_call_fingerprint(restore_callbacks):
    """A tool_calls fingerprint already present in _agentic_loop_fingerprints ->
    ValueError about the repeated fingerprint (cycle guard), with no follow-up
    call."""
    import json

    # The dispatcher fingerprints the whole value the gate returns as its second
    # tuple element, so the seeded fingerprint must mirror that dict exactly.
    gate_tool_calls = {"tool_calls": [{"id": "call_abc", "name": "litellm_code_execution"}]}
    fingerprint = json.dumps(gate_tool_calls, sort_keys=True, default=str)

    logger = _GateOnlyLogger(
        plan=AgenticLoopPlan(run_agentic_loop=True),
        tool_calls=gate_tool_calls,
    )
    litellm.callbacks = [logger]

    acompletion_mock = AsyncMock()
    with patch.object(litellm, "acompletion", acompletion_mock):
        with pytest.raises(ValueError, match="fingerprint"):
            await maybe_run_chat_completion_agentic_loop(
                response=_tool_call_model_response(),
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={},
                kwargs={
                    "_agentic_loop_depth": 0,
                    "max_agentic_loops": 3,
                    "_agentic_loop_fingerprints": [fingerprint],
                },
                logging_obj=_LoggingStub(),
                custom_llm_provider="openai",
                stream=False,
            )

    acompletion_mock.assert_not_awaited()


def test_followup_metadata_drops_measurements_the_first_turn_already_reported():
    """Regression: an agentic loop bills several turns for one client request. A
    measurement the interception took once for the whole request (what
    compression saved, whether retrieval fired) must ride on exactly one spend
    row -- carrying it onto the follow-up turn made the daily savings aggregate
    sum the same number twice."""
    from litellm.litellm_core_utils.chat_completion_agentic_loop import (
        _add_agentic_loop_metadata,
    )

    savings = {"tokens_before": 100, "tokens_after": 40, "tokens_saved": 60}
    first_turn_metadata = {
        "compression_savings": savings,
        "compression_retrieval": {"requested_keys": 1},
        "user_api_key_alias": "keep-me",
    }
    kwargs_for_followup: Dict[str, Any] = {
        "litellm_metadata": first_turn_metadata,
        "_agentic_loop_depth": 1,
        "max_agentic_loops": 3,
    }

    _add_agentic_loop_metadata(kwargs_for_followup)

    followup_metadata = kwargs_for_followup["litellm_metadata"]
    assert "compression_savings" not in followup_metadata
    assert "compression_retrieval" not in followup_metadata
    assert followup_metadata["user_api_key_alias"] == "keep-me"
    assert followup_metadata["_agentic_loop_depth"] == 1

    # The first turn's own row must keep reporting them.
    assert first_turn_metadata["compression_savings"] is savings
    assert "compression_retrieval" in first_turn_metadata
