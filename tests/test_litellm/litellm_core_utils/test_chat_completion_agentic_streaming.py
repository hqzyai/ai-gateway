"""Tests for chat-completions agentic streaming wrapper."""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.litellm_core_utils.chat_completion_agentic_streaming import (
    AgenticChatCompletionStreamingIterator,
    maybe_wrap_chat_completion_stream,
)
from litellm.types.utils import Choices, Message, ModelResponse, ModelResponseStream


async def _aiter_chunks(chunks: list[object]) -> AsyncIterator[object]:
    for chunk in chunks:
        yield chunk


def _stream_chunk(
    *, content: str | None = None, tool_calls: list | None = None, finish: str | None = None
) -> ModelResponseStream:
    delta_kwargs: dict[str, Any] = {}
    if content is not None:
        delta_kwargs["content"] = content
    if tool_calls is not None:
        delta_kwargs["tool_calls"] = tool_calls
    choice = MagicMock()
    choice.index = 0
    choice.delta = MagicMock(**delta_kwargs)
    choice.delta.get = lambda k, default=None: delta_kwargs.get(k, default)
    choice.finish_reason = finish
    chunk = MagicMock(spec=ModelResponseStream)
    chunk.id = "chatcmpl-test"
    chunk.object = "chat.completion.chunk"
    chunk.created = 1
    chunk.model = "gpt-4o"
    chunk.choices = [choice]
    chunk.model_dump = MagicMock(
        return_value={
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": delta_kwargs,
                    "finish_reason": finish,
                }
            ],
        }
    )
    return chunk


@pytest.mark.asyncio
async def test_maybe_wrap_skips_when_no_agentic_hook():
    stream = _aiter_chunks([_stream_chunk(content="hi", finish="stop")])
    with patch(
        "litellm.litellm_core_utils.chat_completion_agentic_streaming.chat_completion_has_agentic_hook",
        return_value=False,
    ):
        wrapped = maybe_wrap_chat_completion_stream(
            stream,
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            kwargs={},
            logging_obj=MagicMock(),
            custom_llm_provider="openai",
            stream=True,
        )
    assert wrapped is stream


@pytest.mark.asyncio
async def test_agentic_stream_replays_original_when_hook_returns_none():
    chunks = [
        _stream_chunk(content="hello"),
        _stream_chunk(content=None, finish="stop"),
    ]
    rebuilt = ModelResponse(
        id="chatcmpl-test",
        choices=[Choices(finish_reason="stop", index=0, message=Message(content="hello", role="assistant"))],
        created=1,
        model="gpt-4o",
        object="chat.completion",
    )

    with (
        patch(
            "litellm.litellm_core_utils.chat_completion_agentic_streaming.stream_chunk_builder",
            return_value=rebuilt,
            create=True,
        ),
        patch(
            "litellm.litellm_core_utils.chat_completion_agentic_streaming.maybe_run_chat_completion_agentic_loop",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "litellm.main.stream_chunk_builder",
            return_value=rebuilt,
        ),
    ):
        iterator = AgenticChatCompletionStreamingIterator(
            completion_stream=_aiter_chunks(chunks),
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"tools": []},
            kwargs={},
            logging_obj=MagicMock(),
            custom_llm_provider="openai",
        )
        out = [c async for c in iterator]

    assert out == chunks


@pytest.mark.asyncio
async def test_agentic_stream_strips_headroom_when_hook_returns_none():
    original = [
        _stream_chunk(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_hr",
                    "type": "function",
                    "function": {"name": "headroom_retrieve", "arguments": '{"hash":"abc"}'},
                }
            ],
            finish="tool_calls",
        )
    ]
    rebuilt = ModelResponse(
        id="chatcmpl-test",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        {
                            "id": "call_hr",
                            "type": "function",
                            "function": {"name": "headroom_retrieve", "arguments": '{"hash":"abc"}'},
                        }
                    ],
                ),
            )
        ],
        created=1,
        model="gpt-4o",
        object="chat.completion",
    )

    with (
        patch(
            "litellm.main.stream_chunk_builder",
            return_value=rebuilt,
        ),
        patch(
            "litellm.litellm_core_utils.chat_completion_agentic_streaming.maybe_run_chat_completion_agentic_loop",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        iterator = AgenticChatCompletionStreamingIterator(
            completion_stream=_aiter_chunks(original),
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"tools": []},
            kwargs={},
            logging_obj=MagicMock(),
            custom_llm_provider="openai",
        )
        out = [c async for c in iterator]

    assert len(out) == 1
    assert out[0] is not original[0]
    delta = out[0].choices[0].delta
    content = getattr(delta, "content", None) or (delta.get("content") if isinstance(delta, dict) else None)
    assert content is not None
    assert "Proxy-owned tool" in content
    finish = out[0].choices[0].finish_reason
    assert finish == "stop"


@pytest.mark.asyncio
async def test_agentic_stream_yields_follow_up_instead_of_original_tool_call():
    original = [
        _stream_chunk(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_hr",
                    "type": "function",
                    "function": {"name": "headroom_retrieve", "arguments": '{"hash":"abc"}'},
                }
            ],
            finish="tool_calls",
        )
    ]
    rebuilt = ModelResponse(
        id="chatcmpl-test",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        {
                            "id": "call_hr",
                            "type": "function",
                            "function": {"name": "headroom_retrieve", "arguments": '{"hash":"abc"}'},
                        }
                    ],
                ),
            )
        ],
        created=1,
        model="gpt-4o",
        object="chat.completion",
    )
    follow_up = ModelResponse(
        id="chatcmpl-follow",
        choices=[Choices(finish_reason="stop", index=0, message=Message(content="retrieved answer", role="assistant"))],
        created=2,
        model="gpt-4o",
        object="chat.completion",
    )

    with (
        patch(
            "litellm.main.stream_chunk_builder",
            return_value=rebuilt,
        ),
        patch(
            "litellm.litellm_core_utils.chat_completion_agentic_streaming.maybe_run_chat_completion_agentic_loop",
            new_callable=AsyncMock,
            return_value=follow_up,
        ),
    ):
        iterator = AgenticChatCompletionStreamingIterator(
            completion_stream=_aiter_chunks(original),
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"tools": []},
            kwargs={},
            logging_obj=MagicMock(),
            custom_llm_provider="openai",
        )
        out = [c async for c in iterator]

    assert len(out) == 1
    assert out[0] is not original[0]
    delta = out[0].choices[0].delta
    assert getattr(delta, "content", None) == "retrieved answer" or (
        isinstance(delta, dict) and delta.get("content") == "retrieved answer"
    )
