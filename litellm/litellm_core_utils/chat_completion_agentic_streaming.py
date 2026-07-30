"""
Agentic streaming wrapper for OpenAI chat completions.

Mirrors Anthropic's AgenticAnthropicStreamingIterator for the chat-completions
surface: buffer the upstream stream, rebuild a ModelResponse, run agentic hooks
(e.g. Headroom CCR), then either replay the original chunks or stream the
follow-up response.

Chunks are buffered without yielding until the agentic decision is made so
clients that execute tool calls at end-of-stream (e.g. Hermes) never see
proxy-owned tools like ``headroom_retrieve``.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Union, cast

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.chat_completion_agentic_loop import (
    AgenticLoopRoutingContext,
    get_agentic_loop_routing_context,
    maybe_run_chat_completion_agentic_loop,
    sanitize_response_hiding_proxy_owned_tools,
)
from litellm.types.utils import ModelResponse, ModelResponseStream


def chat_completion_has_agentic_hook(logging_obj: object) -> bool:
    """True when any callback overrides async_should_run_agentic_loop."""
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.llms.custom_httpx.llm_http_handler import _custom_logger_callbacks

    base_func = CustomLogger.async_should_run_agentic_loop
    for cb in _custom_logger_callbacks(logging_obj):
        cb_func = getattr(type(cb), "async_should_run_agentic_loop", base_func)
        if getattr(cb_func, "__func__", cb_func) is not getattr(base_func, "__func__", base_func):
            return True
    return False


def _chunk_as_dict(chunk: object) -> dict[str, Any]:
    if isinstance(chunk, dict):
        return chunk
    model_dump = getattr(chunk, "model_dump", None)
    if callable(model_dump):
        return cast(  # cast-ok: Pydantic model_dump returns a string-keyed dictionary
            dict[str, Any], model_dump(exclude_none=True, exclude_unset=True)
        )
    dict_method = getattr(chunk, "dict", None)
    if callable(dict_method):
        return cast(dict[str, Any], dict_method())  # cast-ok: legacy Pydantic dict returns a string-keyed dictionary
    return dict(chunk)  # pyright: ignore[reportArgumentType,reportCallIssue]  # provider chunks may implement mapping at runtime


class AgenticChatCompletionStreamingIterator:
    """
    Buffer-then-decide async iterator over chat-completion stream chunks.

    Phase 1: drain upstream CustomStreamWrapper / async iterator, collecting chunks.
    Phase 2: rebuild ModelResponse, run maybe_run_chat_completion_agentic_loop.
    Phase 3: yield either follow-up stream chunks or the original buffered chunks.
    """

    def __init__(
        self,
        completion_stream: Any,
        *,
        model: str,
        messages: list,
        optional_params: dict[str, Any],
        kwargs: dict[str, Any],
        logging_obj: object,
        custom_llm_provider: str,
        stream: bool = True,
        routing_context: AgenticLoopRoutingContext | None = None,
    ):
        self._inner = completion_stream
        self._model = model
        self._messages = messages
        self._optional_params = optional_params
        self._kwargs = kwargs
        self._logging_obj = logging_obj
        self._custom_llm_provider = custom_llm_provider
        self._stream = stream
        self._routing_context = routing_context

        self._collected_chunks: list[object] = []
        self._output_iterator: AsyncIterator[object] | None = None
        self._started = False
        self._done = False

    def __aiter__(self) -> AgenticChatCompletionStreamingIterator:
        return self

    async def __anext__(self) -> object:
        if not self._started:
            self._started = True
            await self._drain_and_process()

        if self._output_iterator is None:
            raise StopAsyncIteration

        try:
            return await self._output_iterator.__anext__()
        except StopAsyncIteration:
            self._done = True
            raise

    async def aclose(self) -> None:
        from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
            aclose_if_supported,
        )

        await aclose_if_supported(self._inner)
        await aclose_if_supported(self._output_iterator)

    async def _drain_and_process(self) -> None:
        try:
            async for chunk in self._inner:
                self._collected_chunks.append(chunk)
        except Exception:  # noqa: BLE001  # stream wrappers must fail open for provider-specific exceptions
            verbose_logger.exception("AgenticChatCompletionStreamingIterator: error draining upstream stream")
            self._output_iterator = _list_aiter(self._collected_chunks)
            return

        if not self._collected_chunks:
            self._output_iterator = _list_aiter([])
            return

        rebuilt = self._rebuild_model_response()
        if rebuilt is None:
            self._output_iterator = _list_aiter(self._collected_chunks)
            return

        try:
            looped = await maybe_run_chat_completion_agentic_loop(
                response=rebuilt,
                model=self._model,
                messages=self._messages,
                optional_params=self._optional_params,
                kwargs=self._kwargs,
                logging_obj=self._logging_obj,
                custom_llm_provider=self._custom_llm_provider,
                stream=self._stream,
                routing_context=self._routing_context,
            )
        except Exception:  # noqa: BLE001  # agentic callbacks may raise provider-specific exceptions
            verbose_logger.exception(
                "AgenticChatCompletionStreamingIterator: agentic hook failed; stripping proxy-owned tools if present"
            )
            self._output_iterator = _safe_output_after_agentic_miss(
                rebuilt=rebuilt,
                collected_chunks=self._collected_chunks,
            )
            return

        if looped is None:
            self._output_iterator = _safe_output_after_agentic_miss(
                rebuilt=rebuilt,
                collected_chunks=self._collected_chunks,
            )
            return

        self._output_iterator = _response_as_chunk_aiter(looped)

    def _rebuild_model_response(self) -> ModelResponse | None:
        from litellm.main import stream_chunk_builder

        chunk_dicts = [_chunk_as_dict(c) for c in self._collected_chunks]
        try:
            rebuilt = stream_chunk_builder(chunks=chunk_dicts, messages=self._messages)
        except Exception:  # noqa: BLE001  # chunk builders may raise provider-specific exceptions
            verbose_logger.exception("AgenticChatCompletionStreamingIterator: stream_chunk_builder failed")
            return None
        if isinstance(rebuilt, ModelResponse):
            return rebuilt
        return None


def _safe_output_after_agentic_miss(
    *,
    rebuilt: ModelResponse,
    collected_chunks: list[object],
) -> AsyncIterator[object]:
    """Prefer stripping proxy-owned tool calls over replaying them to the client."""
    sanitized = sanitize_response_hiding_proxy_owned_tools(rebuilt)
    if sanitized is not None:
        verbose_logger.warning(
            "AgenticChatCompletionStreamingIterator: agentic loop missed; "
            "stripped proxy-owned tool calls instead of replaying"
        )
        return _response_as_chunk_aiter(sanitized)
    return _list_aiter(collected_chunks)


async def _list_aiter(items: list[object]) -> AsyncIterator[object]:
    for item in items:
        yield item


def _response_as_chunk_aiter(response: object) -> AsyncIterator[object]:
    if hasattr(response, "__aiter__"):
        return cast(  # cast-ok: hasattr verifies the async iterator protocol at runtime
            AsyncIterator[object], response.__aiter__()
        )

    if isinstance(response, ModelResponseStream):
        return _list_aiter([response])

    if isinstance(response, ModelResponse):
        from litellm.llms.base_llm.base_model_iterator import (
            convert_model_response_to_streaming,
        )

        return _list_aiter([convert_model_response_to_streaming(response)])

    return _list_aiter([response])


def maybe_wrap_chat_completion_stream(
    response: object,
    *,
    model: str,
    messages: list,
    optional_params: dict[str, Any],
    kwargs: dict[str, Any],
    logging_obj: object,
    custom_llm_provider: str,
    stream: bool,
) -> Union[object, AgenticChatCompletionStreamingIterator]:
    """Wrap a streaming chat-completions response when agentic hooks are registered."""
    if not stream:
        return response
    if not hasattr(response, "__aiter__"):
        return response
    if not chat_completion_has_agentic_hook(logging_obj):
        return response
    return AgenticChatCompletionStreamingIterator(
        completion_stream=response,
        model=model,
        messages=messages,
        optional_params=optional_params,
        kwargs=kwargs,
        logging_obj=logging_obj,
        custom_llm_provider=custom_llm_provider or "openai",
        stream=True,
        routing_context=get_agentic_loop_routing_context(),
    )
