"""User-facing masking for content-filter regex matches.

LiteLLM's default ``[{pattern_name}_REDACTED]`` placeholder exposes internal
rule identifiers.  Keep useful, non-sensitive context where appropriate while
never returning the configured pattern name to the caller.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, JsonValue, TypeAdapter

from litellm.types.llms.openai import ChatCompletionToolCallChunk
from litellm.types.proxy.guardrails.guardrail_hooks.litellm_content_filter import ContentFilterDetection
from litellm.types.utils import ChatCompletionDeltaToolCall, ChatCompletionMessageToolCall, Function

if TYPE_CHECKING:
    from .content_filter import ContentFilterGuardrail

Call = TypeVar("Call", ChatCompletionToolCallChunk, ChatCompletionMessageToolCall)
from .sensitive_normalization import source_spans, structured, view

SensitiveSource = tuple[str, str]


def merge_tool_fragments(parts: Sequence[ChatCompletionDeltaToolCall]) -> tuple[ChatCompletionDeltaToolCall, ...]:
    return tuple(
        ChatCompletionDeltaToolCall(
            index=index,
            id="".join(part.id or "" for part in parts if part.index == index) or None,
            type=next((part.type for part in parts if part.index == index and part.type), None),
            function=Function(
                name="".join(part.function.name or "" for part in parts if part.index == index) or None,
                arguments="".join(part.function.arguments for part in parts if part.index == index),
            ),
        )
        for index in sorted(frozenset(part.index for part in parts))
    )


def text_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(
            text for item in TypeAdapter(tuple[object, ...]).validate_python(value) for text in text_values(item)
        )
    if isinstance(value, Mapping):
        return tuple(
            text
            for key, item in TypeAdapter(dict[str, object]).validate_python(value).items()
            if key in ("content", "text", "arguments")
            for text in text_values(item)
        )
    return ()


def sensitive_sources(guardrail: ContentFilterGuardrail, data: Mapping[str, object]) -> tuple[SensitiveSource, ...]:
    return tuple(
        dict.fromkeys(
            (text[start:end], str(pattern.get("pattern_name") or "sensitive"))
            for text in text_values(data.get("messages"))
            for pattern in guardrail.compiled_patterns
            if getattr(pattern.get("action"), "value", pattern.get("action")) == "MASK"
            for start, end in guardrail._find_pattern_spans(text, pattern)  # pyright: ignore[reportPrivateUsage]  # Retain the native content filter matcher.
            if end > start
        )
    )


def current_turn_start(inputs: Mapping[str, object], count: int) -> int:
    messages = inputs.get("structured_messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], Mapping):
        return max(0, count - 1)
    content = TypeAdapter(dict[str, object]).validate_python(messages[-1]).get("content")
    if isinstance(content, str):
        return max(0, count - 1)
    parts = TypeAdapter(tuple[object, ...]).validate_python(content) if isinstance(content, list) else ()
    latest = sum(
        1
        for part in parts
        if isinstance(part, Mapping)
        for item in (TypeAdapter(dict[str, object]).validate_python(part),)
        if item.get("type") in ("text", "input_text") and isinstance(item.get("text"), str)
    )
    return max(0, count - latest)


def _prefix_mask(value: str, *, prefix: int) -> str:
    if len(value) <= prefix:
        return "****"
    return f"{value[:prefix]}****"


def mask_sensitive_value(value: str, pattern_name: str) -> str:
    """Mask one matched value without exposing the internal pattern name."""
    normalized = pattern_name.lower()

    # These presets include a label in their regex span; retain it for readability.
    labeled = (
        ("cn_bank_labeled", r"((?:银行卡号|银行卡|银行账号|卡号)\s*[:：=]?\s*)(.+)", 4),
        ("cn_landline_labeled", r"((?:联系电话|电话|座机)\s*[:：=]?\s*)(.+)", 3),
        ("cn_account", r"((?:微信号|微信|QQ|账号)\s*[:：=]\s*)(.+)", 1),
    )
    for name, pattern, prefix in labeled:
        if normalized != name:
            continue
        match = re.fullmatch(pattern, value, re.IGNORECASE | re.DOTALL)
        if match:
            canonical, _ = view(match[2], True)
            return match[1] + _prefix_mask(canonical, prefix=prefix)

    if structured(normalized):
        value, _ = view(value, True)
        if normalized == "cn_mobile":
            value = re.sub(r"^(?:\+86|0086|86)(?=1[3-9][0-9]{9}$)", "", value)
    compact = "".join(char for char in value if char.isalnum())

    if any(token in normalized for token in ("mobile", "phone", "手机号", "电话号码")):
        # Keep only the carrier prefix; never expose a trailing subscriber segment.
        return _prefix_mask(compact, prefix=3)
    if normalized == "cn_id" or any(token in normalized for token in ("identity", "id_card", "身份证")):
        return _prefix_mask(compact, prefix=3)
    if normalized == "cn_uscc" or "统一社会信用" in normalized:
        return _prefix_mask(compact, prefix=2)
    if any(token in normalized for token in ("credit_card", "bank_card", "银行卡")):
        return _prefix_mask(compact, prefix=4)
    if ("email" in normalized or "邮箱" in normalized) and "@" in value:
        local, _domain = value.rsplit("@", 1)
        return f"{_prefix_mask(local, prefix=1)}@***"
    return "****"


def mask_sensitive_spans(
    text: str,
    spans: Sequence[tuple[int, int]],
    pattern_name: str,
) -> str:
    """Replace spans right-to-left so their original offsets remain valid."""
    masked = text
    for start, end in sorted(spans, reverse=True):
        masked = masked[:start] + mask_sensitive_value(masked[start:end], pattern_name) + masked[end:]
    return masked


def mask_sensitive_fragments(
    text: str,
    sources: Sequence[SensitiveSource],
    detections: list[ContentFilterDetection]  # mutable-ok: Native matcher appends to the supplied log.
    | None = None,
) -> str:
    """Mask exact sensitive identifiers with type-aware display formats.

    Derived attributes and short substrings are intentionally left alone. A
    birth year, region code, card BIN, or carrier prefix can be the substance of
    a legitimate explanation and must not be destroyed by generic substring
    replacement. Strict policies should block the response instead.
    """
    masked = text
    for source, pattern_name in sources:
        if not source:
            continue
        spans = source_spans(masked, source, pattern_name)
        masked = mask_sensitive_spans(masked, spans, pattern_name)
        if detections is not None:
            detections.extend(
                {  # mutable-ok: Native ContentFilterDetection dictionary.
                    "type": "pattern",
                    "pattern_name": pattern_name,
                    "action": "MASK",
                }
                for _ in spans
            )
    return masked


def mask_history_text(
    guardrail: ContentFilterGuardrail,
    text: str,
    detections: list[  # mutable-ok: Native matcher appends to the supplied log.
        ContentFilterDetection
    ],
) -> str:
    """Redact historical content without replaying historical BLOCK rules."""
    for pattern in guardrail.compiled_patterns:
        action = pattern.get("action")
        if getattr(action, "value", action) != "MASK":
            continue
        spans = guardrail._find_pattern_spans(text, pattern)  # pyright: ignore[reportPrivateUsage]  # Retain the native content filter matcher.
        if spans:
            text = mask_sensitive_spans(text, spans, TypeAdapter(str).validate_python(pattern["pattern_name"]))
            detections.extend(
                {  # mutable-ok: Native ContentFilterDetection log entry schema.
                    "type": "pattern",
                    "pattern_name": pattern["pattern_name"],
                    "action": "MASK",
                }
                for _ in spans
            )
    for keyword, (action, description) in guardrail.blocked_words.items():
        if getattr(action, "value", action) == "MASK" and re.search(
            guardrail._keyword_to_regex_pattern(keyword),
            text,
            re.IGNORECASE,  # pyright: ignore[reportPrivateUsage]  # Preserve native keyword boundary matching.
        ):
            text = guardrail._handle_blocked_word_match(  # pyright: ignore[reportPrivateUsage]  # Preserve native detection records when masking history.
                keyword, action, description, text, detections
            )
    return text


def mask_history_arguments(
    guardrail: ContentFilterGuardrail,
    calls: list[  # mutable-ok: Native adapter passes mutable tool-call lists; copied before redaction.
        Call
    ],
    detections: list[  # mutable-ok: Native matcher appends to the supplied log.
        ContentFilterDetection
    ],
) -> list[Call]:  # mutable-ok: Native request adapter requires the copied tool-call list.
    """Keep historical call IDs/names intact; redact argument values only."""
    import json
    from copy import deepcopy

    def redact(value: JsonValue) -> JsonValue:
        if isinstance(value, str):
            return mask_history_text(guardrail, value, detections)
        if isinstance(value, dict):
            return {  # mutable-ok: Native tool argument JSON object.
                key: redact(item) for key, item in value.items()
            }
        if isinstance(value, list):
            return [  # mutable-ok: Native tool argument JSON array.
                redact(item) for item in value
            ]
        return value

    result = deepcopy(calls)
    for call in result:
        arguments = call["function"].get("arguments") if isinstance(call, dict) else call.function.arguments
        if not isinstance(arguments, str):
            continue
        try:
            parsed = TypeAdapter[JsonValue](JsonValue).validate_json(arguments)
        except ValueError:
            masked = mask_history_text(guardrail, arguments, detections)
        else:
            transformed = redact(parsed)
            masked = arguments if transformed == parsed else json.dumps(transformed, ensure_ascii=False)
        if isinstance(call, dict):
            call["function"]["arguments"] = masked
        else:
            call.function.arguments = masked
    return result


def contains_sensitive_value(value: object, sources: Sequence[SensitiveSource]) -> bool:
    """Return whether a structured value contains an exact sensitive identifier."""
    if isinstance(value, str):
        # Tool arguments are commonly a JSON string, including escaped newlines
        # and Unicode. Inspect decoded values rather than the wire encoding.
        try:
            decoded = TypeAdapter[JsonValue](JsonValue).validate_json(value)
        except ValueError:
            decoded = None
        if isinstance(decoded, (dict, list)) and contains_sensitive_value(decoded, sources):
            return True
        return any(source and source_spans(value, source, pattern_name) for source, pattern_name in sources)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return contains_sensitive_value(str(value), sources)
    if isinstance(value, list):
        return any(
            contains_sensitive_value(item, sources) for item in TypeAdapter(tuple[object, ...]).validate_python(value)
        )
    if isinstance(value, dict):
        return any(
            contains_sensitive_value(item, sources)
            for item in TypeAdapter(dict[str, object]).validate_python(value).values()
        )
    if isinstance(value, BaseModel):
        return contains_sensitive_value(value.model_dump(), sources)
    return False
