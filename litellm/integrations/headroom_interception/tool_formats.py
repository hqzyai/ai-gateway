"""
Format-aware handling of the ``headroom_retrieve`` tool that Headroom compression
injects to let a model ask for content that was compressed away.

Two axes are handled independently:

* API surface (chat completions / Responses API / Anthropic Messages) -- detected
  from the response shape, same as the rest of litellm's cross-surface helpers.
* Tool-call format (``openai`` native structured tool calling / ``hermes`` inline
  ``<tool_call>`` text tags for models without native tool calling) -- selected by
  config, since it depends on which model is being served, not on the response.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, List, Literal, Optional, cast

from typing_extensions import assert_never

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    add_system_prompt_to_messages,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    NormalizedToolCall,
    get_attribute_or_key,
    get_tool_calls_from_response,
    has_tool_with_name,
)
from litellm.types.llms.openai import AllMessageValues

HEADROOM_RETRIEVE_TOOL_NAME = "headroom_retrieve"
# Headroom's real compression marker is "<<ccr:{12-hex-hash},{type},{size}>>",
# e.g. "<<ccr:97cddfc1c993,string,49.0KB>>" -- confirmed against live traffic
# from the actual Headroom compression service (not "hash=..." as originally
# assumed; that pattern never matched anything Headroom actually emits, so CCR
# retrieval always rejected the model's hash as "not produced by this call").
_HASH_PATTERN = re.compile(r"<<ccr:([a-f0-9]{12}),")
_HERMES_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

ToolCallFormat = Literal["openai", "hermes"]


def extract_hashes_from_messages(messages: list[dict[str, object]]) -> list[str]:
    hashes: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            hashes.extend(_HASH_PATTERN.findall(content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        hashes.extend(_HASH_PATTERN.findall(text))
    return hashes


def _headroom_retrieve_tool_spec() -> dict[str, object]:
    return {
        "name": HEADROOM_RETRIEVE_TOOL_NAME,
        "description": (
            "Retrieve original content that was compressed by Headroom. "
            "Call this when you encounter a compression marker containing a hash."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hash": {
                    "type": "string",
                    "description": "The 12-character hex hash from the compression marker, e.g. the "
                    "'97cddfc1c993' in '<<ccr:97cddfc1c993,string,49.0KB>>'.",
                },
                "query": {
                    "type": "string",
                    "description": "Optional search query for BM25-ranked retrieval.",
                },
            },
            "required": ["hash"],
        },
    }


def _build_openai_retrieve_tool() -> dict[str, object]:
    return {"type": "function", "function": _headroom_retrieve_tool_spec()}


def has_headroom_retrieve_tool(tools: object) -> bool:
    return has_tool_with_name(tools, HEADROOM_RETRIEVE_TOOL_NAME)


def _merge_openai_retrieve_tool(tools: Optional[list[object]]) -> list[object]:
    if tools is None:
        return [_build_openai_retrieve_tool()]
    if has_headroom_retrieve_tool(tools):
        return tools
    return list(tools) + [_build_openai_retrieve_tool()]


def _hermes_tool_declaration() -> str:
    tool_json = json.dumps(_headroom_retrieve_tool_spec(), indent=2)
    return (
        "You have access to the following function:\n\n"
        f"<tools>\n{tool_json}\n</tools>\n\n"
        "When you need to call this function, respond with:\n"
        "<tool_call>\n"
        f'{{"name": "{HEADROOM_RETRIEVE_TOOL_NAME}", "arguments": {{"hash": "...", "query": "..."}}}}\n'
        "</tool_call>\n\n"
        "You may emit multiple <tool_call> blocks in one turn. Wait for the matching "
        "<tool_response> before continuing your answer."
    )


def inject_retrieve_tool(
    messages: list[dict[str, object]],
    tools: Optional[list[object]],
    tool_call_format: ToolCallFormat,
) -> tuple[list[dict[str, object]], Optional[list[object]]]:
    """
    Make the ``headroom_retrieve`` tool callable by the model, in whichever
    format it understands. OpenAI-format models get it via the ``tools`` API
    field; hermes-format models get it as text in the system prompt, since
    they have no native ``tools`` field to populate.
    """
    match tool_call_format:
        case "openai":
            return messages, _merge_openai_retrieve_tool(tools)
        case "hermes":
            patched = add_system_prompt_to_messages(
                messages=cast(List[AllMessageValues], messages),
                system_prompt=_hermes_tool_declaration(),
                merge_with_first_system=True,
            )
            return cast(List[dict[str, object]], patched), tools
        case _ as unreachable:
            assert_never(unreachable)


def _extract_openai_retrieve_tool_calls(response: object) -> list[NormalizedToolCall]:
    return [tc for tc in get_tool_calls_from_response(response) if tc["name"] == HEADROOM_RETRIEVE_TOOL_NAME]


def _get_chat_completion_text(response: object) -> Optional[str]:
    choices = get_attribute_or_key(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        return None
    message = get_attribute_or_key(choices[0], "message", None)
    if message is None:
        return None
    content = get_attribute_or_key(message, "content", None)
    return content if isinstance(content, str) else None


def _extract_hermes_tool_calls(response: object) -> list[NormalizedToolCall]:
    text = _get_chat_completion_text(response)
    if not text:
        return []

    tool_calls: list[NormalizedToolCall] = []
    for match in _HERMES_TOOL_CALL_PATTERN.finditer(text):
        try:
            payload: Any = json.loads(match.group(1))
        except ValueError:
            verbose_logger.debug("HeadroomInterception: skipping malformed hermes <tool_call> block")
            continue
        if not isinstance(payload, dict):
            continue

        name = payload.get("name")
        if name != HEADROOM_RETRIEVE_TOOL_NAME:
            continue

        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        tool_calls.append(NormalizedToolCall(id=f"hermes_{uuid.uuid4().hex[:12]}", name=name, arguments=arguments))
    return tool_calls


def extract_retrieve_tool_calls(
    response: object,
    tools: object,
    tool_call_format: ToolCallFormat,
) -> list[NormalizedToolCall]:
    match tool_call_format:
        case "openai":
            if not has_headroom_retrieve_tool(tools):
                return []
            return _extract_openai_retrieve_tool_calls(response)
        case "hermes":
            return _extract_hermes_tool_calls(response)
        case _ as unreachable:
            assert_never(unreachable)


def _is_responses_api_response(response: object) -> bool:
    # Real response objects can be plain dicts at runtime (e.g. TypedDict-based
    # response types), so getattr alone would silently miss the key -- use the
    # same dict-or-object accessor as the tool-call extractors.
    return isinstance(get_attribute_or_key(response, "output", None), list)


def _is_anthropic_messages_response(response: object) -> bool:
    return isinstance(get_attribute_or_key(response, "content", None), list)


def _build_assistant_message_from_response(response: object) -> dict[str, object]:
    choices = get_attribute_or_key(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        return {"role": "assistant", "content": None, "tool_calls": []}
    message = get_attribute_or_key(choices[0], "message", None)
    if message is None:
        return {"role": "assistant", "content": None, "tool_calls": []}
    content = get_attribute_or_key(message, "content", None)
    tool_calls = get_attribute_or_key(message, "tool_calls", None)
    raw_tool_calls: list[dict[str, object]] = []
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            fn = get_attribute_or_key(tc, "function", None)
            raw_tool_calls.append(
                {
                    "id": get_attribute_or_key(tc, "id", None),
                    "type": "function",
                    "function": {
                        "name": get_attribute_or_key(fn, "name", None) if fn else None,
                        "arguments": get_attribute_or_key(fn, "arguments", "{}") if fn else "{}",
                    },
                }
            )
    return {"role": "assistant", "content": content, "tool_calls": raw_tool_calls}


def _build_anthropic_followup_messages(
    retrieved: list[tuple[NormalizedToolCall, str]],
) -> list[dict[str, object]]:
    """Build Anthropic Messages API follow-up messages for a tool round-trip.

    Anthropic requires the tool_use block to be echoed back in an assistant
    message, paired with a tool_result block in a user message keyed by the
    same tool_use_id -- it does not accept chat-style tool-role messages.
    """
    assistant_message: dict[str, object] = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_call["id"],
                "name": tool_call["name"],
                "input": tool_call["arguments"],
            }
            for tool_call, _ in retrieved
        ],
    }
    user_message: dict[str, object] = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tool_call["id"], "content": content}
            for tool_call, content in retrieved
        ],
    }
    return [assistant_message, user_message]


def _build_responses_followup_items(
    retrieved: list[tuple[NormalizedToolCall, str]],
) -> list[dict[str, object]]:
    """Build Responses API input items for a tool round-trip.

    The Responses API does not accept chat-style assistant/tool messages as
    follow-up input; it requires the model's function_call to be echoed back
    paired with a function_call_output keyed by the same call_id.
    """
    items: list[dict[str, object]] = []
    for tool_call, content in retrieved:
        call_id = tool_call["id"]
        items.append(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": tool_call["name"],
                "arguments": json.dumps(tool_call["arguments"]),
            }
        )
        items.append({"type": "function_call_output", "call_id": call_id, "output": content})
    return items


def _build_hermes_followup_messages(
    response: object,
    messages: list[dict[str, object]],
    retrieved: list[tuple[NormalizedToolCall, str]],
) -> list[dict[str, object]]:
    """Build hermes-format follow-up messages.

    The assistant turn echoes the model's raw text verbatim (including the
    <tool_call> block it emitted), matching the Hermes multi-turn convention
    of feeding the model's own prior turn back unmodified. One role="tool"
    message per retrieved call, content wrapped in <tool_response> -- this is
    how vLLM's built-in Hermes-family chat templates render role="tool".
    """
    raw_text = _get_chat_completion_text(response) or ""
    assistant_message: dict[str, object] = {"role": "assistant", "content": raw_text}
    tool_messages: list[dict[str, object]] = [
        {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": f"<tool_response>\n{content}\n</tool_response>",
        }
        for tool_call, content in retrieved
    ]
    return list(messages) + [assistant_message] + tool_messages


def build_followup_messages(
    response: object,
    messages: list[dict[str, object]],
    retrieved: list[tuple[NormalizedToolCall, str]],
    tool_call_format: ToolCallFormat,
) -> list[dict[str, object]]:
    if _is_responses_api_response(response):
        return list(messages) + _build_responses_followup_items(retrieved)
    if _is_anthropic_messages_response(response):
        return list(messages) + _build_anthropic_followup_messages(retrieved)

    match tool_call_format:
        case "openai":
            assistant_message = _build_assistant_message_from_response(response)
            tool_results = [
                {"role": "tool", "tool_call_id": tool_call["id"], "content": content}
                for tool_call, content in retrieved
            ]
            return list(messages) + [assistant_message] + tool_results
        case "hermes":
            return _build_hermes_followup_messages(response, messages, retrieved)
        case _ as unreachable:
            assert_never(unreachable)
