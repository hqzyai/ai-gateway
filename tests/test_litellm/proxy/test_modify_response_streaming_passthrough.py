"""Regression test for the ModifyResponseException streaming passthrough.

When a guardrail blocks a *streaming* request pre-call by raising
``ModifyResponseException``, the chat-completion route streams the violation
message back as a 200 by building a ``CustomStreamWrapper``. The logging object
must be read from ``e.request_data`` (the processor's data, which carries
``litellm_logging_obj``) and NOT from the outer request body returned by
``_read_request_body`` -- the two diverge at ``function_setup`` and only the
processor copy gets ``litellm_logging_obj`` attached.

Reading it from the outer body passed ``logging_obj=None`` to
``CustomStreamWrapper.__init__``, which dereferences
``logging_obj.model_call_details`` and 500s with
``AttributeError: 'NoneType' object has no attribute 'model_call_details'``.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, Response

from litellm.exceptions import RejectedRequestError
from litellm.integrations.custom_guardrail import ModifyResponseException
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.tool_permission import ToolPermissionGuardrail
from litellm.proxy.proxy_server import async_data_generator, chat_completion
from litellm.types.utils import ModelResponseStream
from litellm.utils import ModelResponseIterator


async def _run_streaming_block_and_get_wrapper(exception):
    """Drive chat_completion's streaming guardrail-passthrough handler for the
    given pre-call block exception and return the patched CustomStreamWrapper.

    The outer request body (what _read_request_body returns) is a streaming
    request that does NOT carry litellm_logging_obj -- mirroring production,
    where the outer body diverges from the processor's data at function_setup.
    Only the processor copy (exposed as exception.request_data) carries it.
    """
    request = MagicMock(spec=Request)
    fastapi_response = MagicMock(spec=Response)
    user_api_key_dict = UserAPIKeyAuth()
    outer_body = {"model": "gpt-4o", "messages": [], "stream": True}

    with patch(
        "litellm.proxy.proxy_server._read_request_body",
        new_callable=AsyncMock,
        return_value=outer_body,
    ), patch(
        "litellm.proxy.proxy_server.ProxyBaseLLMRequestProcessing.base_process_llm_request",
        new_callable=AsyncMock,
        side_effect=exception,
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj"
    ) as mock_proxy_logging, patch(
        "litellm.proxy.proxy_server.select_data_generator",
        return_value=iter([]),
    ), patch(
        "litellm.CustomStreamWrapper"
    ) as mock_csw:
        mock_proxy_logging.post_call_failure_hook = AsyncMock()

        await chat_completion(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
        )

    return mock_csw


@pytest.mark.asyncio
async def test_streaming_modify_response_uses_request_data_logging_obj():
    sentinel_logging_obj = MagicMock(name="litellm_logging_obj")
    exception = ModifyResponseException(
        message="blocked by guardrail",
        model="gpt-4o",
        request_data={
            "model": "gpt-4o",
            "stream": True,
            "litellm_logging_obj": sentinel_logging_obj,
        },
        guardrail_name="test-guardrail",
    )

    mock_csw = await _run_streaming_block_and_get_wrapper(exception)

    # The wrapper must be built with the logging object from e.request_data,
    # NOT None (which is what the outer body would have yielded).
    mock_csw.assert_called_once()
    assert mock_csw.call_args.kwargs["logging_obj"] is sentinel_logging_obj


@pytest.mark.asyncio
async def test_tool_permission_stream_returns_guardrail_message_instead_of_server_error(monkeypatch):
    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", "true")
    guardrail = ToolPermissionGuardrail(
        guardrail_name="只读工具白名单",
        rules=[{"id": "read-only", "tool_name": "^read_file$", "decision": "allow"}],
        default_action="deny",
        on_disallowed_action="block",
    )
    request_data = {"model": "test-model", "stream": True}
    auth = UserAPIKeyAuth()

    async def upstream():
        yield ModelResponseStream(
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-denied",
                                "type": "function",
                                "function": {"name": "write_file", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        )

    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as logging:
        logging.needs_iterator_wrap.return_value = True
        logging.needs_per_chunk_streaming_hook.return_value = False
        logging.async_post_call_streaming_iterator_hook = guardrail.async_post_call_streaming_iterator_hook
        logging.post_call_failure_hook = AsyncMock()
        output = [line async for line in async_data_generator(upstream(), auth, request_data)]

    assert output[-1] == "data: [DONE]\n\n"
    assert len(output) == 2
    payload = json.loads(output[0][6:])
    assert "error" not in payload
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["choices"][0]["delta"]["content"] == (
        "触发护栏「只读工具白名单」，内容已拦截：工具调用不符合权限规则"
    )
    assert not payload["choices"][0]["delta"].get("tool_calls")
    logging.post_call_failure_hook.assert_awaited_once()
    assert isinstance(logging.post_call_failure_hook.call_args.kwargs["original_exception"], ModifyResponseException)


@pytest.mark.asyncio
@pytest.mark.parametrize("managed,finish_reason", [("true", "stop"), ("false", "content_filter")])
@pytest.mark.parametrize("stream", [False, True])
async def test_guardrail_reply_finish_reason_matches_managed_mode(monkeypatch, managed, finish_reason, stream):
    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", managed)
    message = "触发护栏「工具权限」，内容已拦截：工具调用不符合权限规则"
    data = {"model": "test-model", "stream": stream, "messages": [], "litellm_logging_obj": MagicMock()}
    exception = ModifyResponseException(
        message=message,
        model="test-model",
        request_data=data,
        guardrail_name="工具权限",
    )
    if stream:
        with patch("litellm.utils.ModelResponseIterator", wraps=ModelResponseIterator) as iterator:
            await _run_streaming_block_and_get_wrapper(exception)
        response = iterator.call_args.kwargs["model_response"]
    else:
        with (
            patch("litellm.proxy.proxy_server._read_request_body", new_callable=AsyncMock, return_value=data),
            patch(
                "litellm.proxy.proxy_server.ProxyBaseLLMRequestProcessing.base_process_llm_request",
                new_callable=AsyncMock,
                side_effect=exception,
            ),
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as logging,
        ):
            logging.post_call_failure_hook = AsyncMock()
            response = await chat_completion(
                request=MagicMock(spec=Request),
                fastapi_response=Response(),
                user_api_key_dict=UserAPIKeyAuth(),
            )
        logging.post_call_failure_hook.assert_awaited_once()
    assert response.choices[0].finish_reason == finish_reason
    assert response.choices[0].message.content == message
    assert not response.choices[0].message.tool_calls


@pytest.mark.asyncio
async def test_streaming_rejected_request_uses_request_data_logging_obj():
    # RejectedRequestError gets the identical fix in its own streaming
    # passthrough handler, so it needs the same regression guard.
    sentinel_logging_obj = MagicMock(name="litellm_logging_obj")
    exception = RejectedRequestError(
        message="rejected by guardrail",
        model="gpt-4o",
        llm_provider="openai",
        request_data={
            "model": "gpt-4o",
            "stream": True,
            "litellm_logging_obj": sentinel_logging_obj,
        },
    )

    mock_csw = await _run_streaming_block_and_get_wrapper(exception)

    mock_csw.assert_called_once()
    assert mock_csw.call_args.kwargs["logging_obj"] is sentinel_logging_obj
