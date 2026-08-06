"""
Fold every agentic-loop turn's token usage into the response the caller logs.

An agentic loop answers one client request with several upstream calls, but
only the final response reaches the outer ``@client`` wrapper. That wrapper is
what builds the spend-log row and computes cost, so without folding, every turn
except the last is billed as if it were free.

The three chat surfaces carry usage in three different shapes, so the fold
dispatches on the response type rather than on the call site.
"""

from typing import cast

from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
    AnthropicUsage,
)
from litellm.types.llms.openai import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseAPIUsage,
    ResponsesAPIResponse,
)
from litellm.types.utils import ModelResponse, Usage

_ANTHROPIC_TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def accumulate_agentic_loop_usage(previous_response: object, followup_response: object) -> None:
    """
    Add ``previous_response``'s tokens into ``followup_response``'s usage, in place.

    In place because the follow-up response object is the one that propagates
    back to the outer wrapper, which reads usage off it for both the spend-log
    row and the cost calculation.
    """
    match (previous_response, followup_response):
        case (ModelResponse() as previous, ModelResponse() as followup):
            _accumulate_chat_usage(previous=previous, followup=followup)
        case (ResponsesAPIResponse() as previous, ResponsesAPIResponse() as followup):
            _accumulate_responses_usage(previous=previous, followup=followup)
        case (dict(), dict()):
            _accumulate_anthropic_usage(
                previous=cast(AnthropicMessagesResponse, previous_response),
                followup=cast(AnthropicMessagesResponse, followup_response),
            )
        case _:
            return


def _accumulate_chat_usage(previous: ModelResponse, followup: ModelResponse) -> None:
    from litellm.cost_calculator import BaseTokenUsageProcessor

    previous_usage = getattr(previous, "usage", None)
    followup_usage = getattr(followup, "usage", None)
    if not isinstance(previous_usage, Usage) or not isinstance(followup_usage, Usage):
        return
    setattr(
        followup,
        "usage",
        BaseTokenUsageProcessor.combine_usage_objects([previous_usage, followup_usage]),
    )


def _accumulate_responses_usage(previous: ResponsesAPIResponse, followup: ResponsesAPIResponse) -> None:
    previous_usage = previous.usage
    followup_usage = followup.usage
    if previous_usage is None or followup_usage is None:
        return
    followup.usage = ResponseAPIUsage(
        input_tokens=previous_usage.input_tokens + followup_usage.input_tokens,
        output_tokens=previous_usage.output_tokens + followup_usage.output_tokens,
        total_tokens=previous_usage.total_tokens + followup_usage.total_tokens,
        input_tokens_details=_combine_input_details(
            previous_usage.input_tokens_details, followup_usage.input_tokens_details
        ),
        output_tokens_details=_combine_output_details(
            previous_usage.output_tokens_details, followup_usage.output_tokens_details
        ),
    )


def _combine_input_details(
    previous: InputTokensDetails | None, followup: InputTokensDetails | None
) -> InputTokensDetails | None:
    if previous is None or followup is None:
        return followup or previous
    return InputTokensDetails(
        audio_tokens=_optional_sum(previous.audio_tokens, followup.audio_tokens),
        cached_tokens=previous.cached_tokens + followup.cached_tokens,
        text_tokens=_optional_sum(previous.text_tokens, followup.text_tokens),
    )


def _combine_output_details(
    previous: OutputTokensDetails | None, followup: OutputTokensDetails | None
) -> OutputTokensDetails | None:
    if previous is None or followup is None:
        return followup or previous
    return OutputTokensDetails(
        reasoning_tokens=_optional_sum(previous.reasoning_tokens, followup.reasoning_tokens),
        text_tokens=_optional_sum(previous.text_tokens, followup.text_tokens),
    )


def _accumulate_anthropic_usage(previous: AnthropicMessagesResponse, followup: AnthropicMessagesResponse) -> None:
    previous_usage = previous.get("usage")
    followup_usage = followup.get("usage")
    if previous_usage is None or followup_usage is None:
        return
    previous_tokens = dict(previous_usage)
    followup_tokens = dict(followup_usage)
    followup["usage"] = cast(
        AnthropicUsage,
        {
            **followup_tokens,
            **{
                key: _int_or_zero(previous_tokens.get(key)) + _int_or_zero(followup_tokens.get(key))
                for key in _ANTHROPIC_TOKEN_KEYS
                if key in previous_tokens or key in followup_tokens
            },
        },
    )


def _optional_sum(previous: int | None, followup: int | None) -> int | None:
    if previous is None and followup is None:
        return None
    return (previous or 0) + (followup or 0)


def _int_or_zero(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value
