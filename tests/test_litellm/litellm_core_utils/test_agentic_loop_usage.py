"""
Tests for `litellm/litellm_core_utils/agentic_loop_usage.py`.

An agentic loop answers one client request with several upstream calls, but only
the last response reaches the outer logging object. These tests pin the fold
that carries the earlier turns' tokens onto that last response, on each of the
three chat surfaces.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.litellm_core_utils.agentic_loop_usage import accumulate_agentic_loop_usage
from litellm.types.llms.openai import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseAPIUsage,
    ResponsesAPIResponse,
)
from litellm.types.utils import (
    Choices,
    Message,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
)


def _chat_response(usage: Usage | None) -> ModelResponse:
    response = ModelResponse(
        choices=[Choices(finish_reason="stop", index=0, message=Message(role="assistant", content="hi"))]
    )
    if usage is None:
        if hasattr(response, "usage"):
            delattr(response, "usage")
    else:
        setattr(response, "usage", usage)
    return response


def _responses_response(usage: ResponseAPIUsage | None) -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_1",
        created_at=0,
        model="gpt-4o",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=usage,
    )


def test_chat_usage_is_summed_across_turns():
    previous = _chat_response(Usage(prompt_tokens=1000, completion_tokens=20, total_tokens=1020))
    followup = _chat_response(Usage(prompt_tokens=1500, completion_tokens=30, total_tokens=1530))

    accumulate_agentic_loop_usage(previous_response=previous, followup_response=followup)

    assert followup.usage.prompt_tokens == 2500
    assert followup.usage.completion_tokens == 50
    assert followup.usage.total_tokens == 2550


def test_chat_usage_fold_does_not_mutate_the_earlier_turn():
    previous = _chat_response(Usage(prompt_tokens=1000, completion_tokens=20, total_tokens=1020))
    followup = _chat_response(Usage(prompt_tokens=1500, completion_tokens=30, total_tokens=1530))

    accumulate_agentic_loop_usage(previous_response=previous, followup_response=followup)

    assert previous.usage.prompt_tokens == 1000
    assert previous.usage.completion_tokens == 20


def test_chat_cached_prompt_tokens_are_summed():
    previous = _chat_response(
        Usage(
            prompt_tokens=1000,
            completion_tokens=20,
            total_tokens=1020,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=800),
        )
    )
    followup = _chat_response(
        Usage(
            prompt_tokens=1500,
            completion_tokens=30,
            total_tokens=1530,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=1200),
        )
    )

    accumulate_agentic_loop_usage(previous_response=previous, followup_response=followup)

    assert followup.usage.prompt_tokens_details.cached_tokens == 2000


def test_chat_fold_is_a_no_op_when_a_turn_has_no_usage():
    previous = _chat_response(None)
    followup = _chat_response(Usage(prompt_tokens=1500, completion_tokens=30, total_tokens=1530))

    accumulate_agentic_loop_usage(previous_response=previous, followup_response=followup)

    assert followup.usage.prompt_tokens == 1500


def test_anthropic_usage_is_summed_across_turns():
    previous = {
        "id": "msg_1",
        "type": "message",
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 20,
            "cache_read_input_tokens": 700,
            "cache_creation_input_tokens": 100,
        },
    }
    followup = {
        "id": "msg_2",
        "type": "message",
        "usage": {
            "input_tokens": 1500,
            "output_tokens": 30,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 50,
        },
    }

    accumulate_agentic_loop_usage(previous_response=previous, followup_response=followup)

    assert followup["usage"] == {
        "input_tokens": 2500,
        "output_tokens": 50,
        "cache_read_input_tokens": 1600,
        "cache_creation_input_tokens": 150,
    }
    assert previous["usage"]["input_tokens"] == 1000


def test_anthropic_fold_keeps_keys_absent_from_both_turns_absent():
    previous = {"id": "msg_1", "usage": {"input_tokens": 1000, "output_tokens": 20}}
    followup = {"id": "msg_2", "usage": {"input_tokens": 1500, "output_tokens": 30}}

    accumulate_agentic_loop_usage(previous_response=previous, followup_response=followup)

    assert followup["usage"] == {"input_tokens": 2500, "output_tokens": 50}


def test_anthropic_fold_is_a_no_op_when_a_turn_has_no_usage():
    previous = {"id": "msg_1"}
    followup = {"id": "msg_2", "usage": {"input_tokens": 1500, "output_tokens": 30}}

    accumulate_agentic_loop_usage(previous_response=previous, followup_response=followup)

    assert followup["usage"] == {"input_tokens": 1500, "output_tokens": 30}


def test_responses_usage_is_summed_across_turns():
    previous = _responses_response(
        ResponseAPIUsage(
            input_tokens=1000,
            output_tokens=20,
            total_tokens=1020,
            input_tokens_details=InputTokensDetails(cached_tokens=800),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=5),
        )
    )
    followup = _responses_response(
        ResponseAPIUsage(
            input_tokens=1500,
            output_tokens=30,
            total_tokens=1530,
            input_tokens_details=InputTokensDetails(cached_tokens=1200),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=7),
        )
    )

    accumulate_agentic_loop_usage(previous_response=previous, followup_response=followup)

    assert followup.usage.input_tokens == 2500
    assert followup.usage.output_tokens == 50
    assert followup.usage.total_tokens == 2550
    assert followup.usage.input_tokens_details.cached_tokens == 2000
    assert followup.usage.output_tokens_details.reasoning_tokens == 12


def test_responses_fold_is_a_no_op_when_a_turn_has_no_usage():
    previous = _responses_response(None)
    followup = _responses_response(ResponseAPIUsage(input_tokens=1500, output_tokens=30, total_tokens=1530))

    accumulate_agentic_loop_usage(previous_response=previous, followup_response=followup)

    assert followup.usage.input_tokens == 1500


@pytest.mark.parametrize(
    "previous, followup",
    [
        (None, None),
        ("not a response", "also not a response"),
        (_chat_response(Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)), {"usage": {"input_tokens": 5}}),
    ],
)
def test_mismatched_or_unknown_surfaces_are_left_alone(previous, followup):
    accumulate_agentic_loop_usage(previous_response=previous, followup_response=followup)
