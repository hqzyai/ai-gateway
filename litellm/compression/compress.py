"""
Main compress() function — normalizes input messages, orchestrates BM25/embedding
scoring, message stubbing, and retrieval tool injection.
"""

import math
import re
from collections.abc import Mapping, Sequence, Set as AbstractSet
from typing import Any, Final, Optional, cast

from litellm.caching.dual_cache import DualCache
from litellm.compression.chunk_selection import select_chunks_to_budget
from litellm.compression.message_stubbing import (
    extract_key,
    stub_message,
)
from litellm.compression.retrieval_tool import build_retrieval_tool
from litellm.compression.scoring.bm25 import bm25_score_messages
from litellm.litellm_core_utils.token_counter import token_counter
from litellm.types.compression import CompressedResult
from litellm.types.utils import CallTypes, SelectTokenizerResponse

# CallTypes that produce Anthropic-shaped messages (structured content blocks).
# Everything else is treated as OpenAI chat-completions shape.
_ANTHROPIC_CALL_TYPES: Final = frozenset({CallTypes.anthropic_messages.value})
# CallTypes that are valid targets for compression.  Compression operates on
# message-shaped inputs, so we only accept call types whose payload is a list
# of role/content messages.
_SUPPORTED_CALL_TYPES: Final = frozenset(
    {
        CallTypes.completion.value,
        CallTypes.acompletion.value,
        CallTypes.anthropic_messages.value,
    }
)
_RECENT_WORKING_SET_UNITS = 4
_MESSAGE_OVERHEAD_TOKENS = 16
_QUERY_TARGET_CHARS = 2000
_QUERY_MESSAGE_CAP_CHARS = 4000
_QUERY_MESSAGE_SKIP_CHARS = 20000
_HISTORY_COVERAGE_BUCKETS = 4
_HISTORY_COVERAGE_BUDGET_RATIO = 0.2
_RECENT_WORKING_SET_BUDGET_RATIO = 0.4
_RELEVANCE_WEIGHT = 0.45
_RECENCY_WEIGHT = 0.35
_CONVERSATIONAL_ROLE_WEIGHT = 0.2

CandidateUnit = tuple[float, tuple[int, ...], bool]


def _normalize_call_type(call_type: CallTypes | str) -> str:
    """Return the string value for a ``CallTypes`` enum or a raw string."""
    if isinstance(call_type, CallTypes):
        return call_type.value
    return call_type


def _is_anthropic_call_type(call_type: str) -> bool:
    return call_type in _ANTHROPIC_CALL_TYPES


def _build_retrieval_tools(keys: Sequence[str], call_type: str) -> Sequence[Mapping[str, Any]]:
    """
    Build retrieval tool definitions in the target request schema.

    - Chat-completions call types: keep OpenAI function-tool schema.
    - Anthropic messages call type: remap to Anthropic's custom tool schema.
    """
    if not keys:
        return []

    openai_tools: Final = [build_retrieval_tool(keys)]
    if not _is_anthropic_call_type(call_type):
        return openai_tools

    # Lazy import to avoid introducing provider transformation imports during
    # module import for non-Anthropic call paths.
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    anthropic_tools, _mcp_servers = AnthropicConfig()._map_tools(openai_tools)
    return cast(list[dict], anthropic_tools)


def _content_to_text(content: Any) -> str:
    """
    Convert OpenAI/Anthropic message content blocks to plain text.

    Text extraction policy:
    - Include text-bearing fields only (`text` blocks + string values).
    - For `tool_result`, expand into nested `content` items.
    - Ignore non-textual blocks (images/documents/tool metadata/thinking metadata).

    Implemented iteratively (stack-based) to avoid unbounded recursion.
    """
    parts: list[str] = []
    stack: list[Any] = [content]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, list):
            # Push list items in reverse order so they are processed left-to-right.
            for element in reversed(item):
                stack.append(element)
        elif isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "text":
                parts.append(str(item.get("text", "")))
            elif item_type == "tool_result":
                stack.append(item.get("content", ""))
    return " ".join(parts)


def _normalize_messages_for_compression(
    messages: Sequence[Mapping[str, Any]],
    call_type: str,
) -> tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]:
    """
    Normalize each original message to a text-surrogate content for scoring.

    Returns:
        (normalized_messages, original_messages_copy)
    """
    if call_type not in _SUPPORTED_CALL_TYPES:
        raise ValueError(
            f"Unsupported call_type={call_type!r} for compression. Expected one of: {sorted(_SUPPORTED_CALL_TYPES)}."
        )

    original_messages = [dict(m) for m in messages]

    normalized_messages = []
    for msg in original_messages:
        normalized_messages.append(
            {
                **msg,
                "content": _content_to_text(msg.get("content", "")),
            }
        )
    return normalized_messages, original_messages


def _build_relevance_query(messages: Sequence[Mapping[str, Any]]) -> str:
    """
    Build the BM25/embedding relevance query from recent user messages.

    Walks user messages newest-first, accumulating text until enough signal is
    collected.  A lone trailing instruction like "now produce the patch" carries
    no relevance signal, so earlier user messages (the actual task statement)
    are pulled in until the query reaches a useful size.
    """
    user_texts = [_content_to_text(msg.get("content", "")) for msg in reversed(messages) if msg.get("role") == "user"]

    collected: list[str] = []
    total_chars = 0
    for text in user_texts:
        if not text or len(text) > _QUERY_MESSAGE_SKIP_CHARS:
            continue
        capped = text[:_QUERY_MESSAGE_CAP_CHARS]
        collected.append(capped)
        total_chars += len(capped)
        if total_chars >= _QUERY_TARGET_CHARS:
            break

    if collected:
        return " ".join(reversed(collected))
    return next((t[:_QUERY_MESSAGE_CAP_CHARS] for t in user_texts if t), "")


def _extract_tool_use_ids(content: Any) -> Sequence[str]:
    if not isinstance(content, list):
        return []
    tool_use_ids = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "tool_use":
            continue
        tool_use_id = part.get("id")
        if isinstance(tool_use_id, str) and tool_use_id:
            tool_use_ids.append(tool_use_id)
    return tool_use_ids


def _extract_tool_result_ids(content: Any) -> AbstractSet[str]:
    if not isinstance(content, list):
        return set()
    tool_result_ids = set()
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "tool_result":
            continue
        tool_use_id = part.get("tool_use_id")
        if isinstance(tool_use_id, str) and tool_use_id:
            tool_result_ids.add(tool_use_id)
    return tool_result_ids


def _is_skill_tool_name(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    return any(segment in {"skill", "skills"} for segment in re.split(r"[^a-z0-9]+", name.lower()))


def _message_calls_skill_tool(message: Mapping[str, Any]) -> bool:
    raw_tool_calls = message.get("tool_calls")
    if isinstance(raw_tool_calls, list):
        for tool_call in raw_tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if isinstance(function, dict) and _is_skill_tool_name(function.get("name")):
                return True

    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, dict) and part.get("type") == "tool_use" and _is_skill_tool_name(part.get("name"))
        for part in content
    )


def _get_skill_tool_exchange_indices(
    messages: Sequence[Mapping[str, Any]], tool_exchange_spans: Sequence[AbstractSet[int]]
) -> frozenset[int]:
    return frozenset(
        idx
        for span in tool_exchange_spans
        if any(_message_calls_skill_tool(messages[span_idx]) for span_idx in span)
        for idx in span
    )


def _extract_anthropic_tool_exchange_spans(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[Sequence[AbstractSet[int]], Optional[str]]:
    """
    Return atomic 2-message spans for Anthropic tool exchanges.

    Each assistant message containing `tool_use` must be immediately followed by a
    user message containing matching `tool_result` blocks for all tool_use ids.
    """
    spans = []
    i = 0
    while i < len(messages):
        current = messages[i]
        if current.get("role") != "assistant":
            i += 1
            continue

        tool_use_ids = _extract_tool_use_ids(current.get("content"))
        if not tool_use_ids:
            i += 1
            continue

        if i + 1 >= len(messages):
            return [], "invalid_anthropic_tool_sequence"

        next_msg = messages[i + 1]
        if next_msg.get("role") != "user":
            return [], "invalid_anthropic_tool_sequence"

        tool_result_ids = _extract_tool_result_ids(next_msg.get("content"))
        if not tool_result_ids:
            return [], "invalid_anthropic_tool_sequence"

        for tool_use_id in tool_use_ids:
            if tool_use_id not in tool_result_ids:
                return [], "invalid_anthropic_tool_sequence"

        spans.append({i, i + 1})
        i += 2

    return spans, None


def _extract_openai_tool_call_ids(tool_calls: Any) -> Optional[Sequence[str]]:
    if not isinstance(tool_calls, list):
        return None
    tool_call_ids = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            return None
        tool_call_id = tool_call.get("id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return None
        tool_call_ids.append(tool_call_id)
    if len(tool_call_ids) != len(set(tool_call_ids)):
        return None
    return tool_call_ids


def _extract_openai_tool_exchange_spans(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[Sequence[AbstractSet[int]], Optional[str]]:
    spans = []
    covered_tool_indices = set()
    i = 0
    while i < len(messages):
        current = messages[i]
        if current.get("role") != "assistant":
            i += 1
            continue

        raw_tool_calls = current.get("tool_calls")
        if raw_tool_calls is None:
            i += 1
            continue

        tool_call_ids = _extract_openai_tool_call_ids(raw_tool_calls)
        if tool_call_ids is None:
            return [], "invalid_openai_tool_sequence"
        if not tool_call_ids:
            i += 1
            continue

        j = i + 1
        tool_result_ids = set()
        tool_indices = set()
        while j < len(messages) and messages[j].get("role") == "tool":
            tool_call_id = messages[j].get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                return [], "invalid_openai_tool_sequence"
            if tool_call_id in tool_result_ids:
                return [], "invalid_openai_tool_sequence"
            tool_result_ids.add(tool_call_id)
            tool_indices.add(j)
            j += 1

        if tool_result_ids != set(tool_call_ids):
            return [], "invalid_openai_tool_sequence"

        spans.append({i, *tool_indices})
        covered_tool_indices.update(tool_indices)
        i = j

    if any(message.get("role") == "tool" and idx not in covered_tool_indices for idx, message in enumerate(messages)):
        return [], "invalid_openai_tool_sequence"

    return spans, None


def get_protected_indices(messages: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    """
    Return indices of messages that must never be compressed:
    - All system and developer messages
    - The last user message
    - The last assistant message

    The last user message is what the model is being asked to act on right now,
    so compressing it replaces the live instruction with a marker. Compression
    guardrails share this policy; see the Headroom guardrail.
    """
    system_indices: Final = tuple(
        index for index, msg in enumerate(messages) if msg.get("role", "") in ("system", "developer")
    )
    last_user: Final = tuple(index for index, msg in enumerate(messages) if msg.get("role", "") == "user")[-1:]
    last_assistant: Final = tuple(index for index, msg in enumerate(messages) if msg.get("role", "") == "assistant")[
        -1:
    ]
    return system_indices + last_user + last_assistant


def _normalize_scores(scores: list[float]) -> list[float]:
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0
    score_range = max_score - min_score
    if score_range == 0:
        return [0.0] * len(scores)
    return [(score - min_score) / score_range for score in scores]


def _hybrid_retention_scores(messages: list[dict], relevance_scores: list[float]) -> list[float]:
    normalized_relevance = _normalize_scores(relevance_scores)
    message_count = max(len(messages), 1)
    return [
        _RELEVANCE_WEIGHT * relevance
        + _RECENCY_WEIGHT * ((idx + 1) / message_count)
        + _CONVERSATIONAL_ROLE_WEIGHT * (1.0 if message.get("role") in ("user", "assistant") else 0.0)
        for idx, (message, relevance) in enumerate(zip(messages, normalized_relevance))
    ]


def _combine_scores(
    bm25_scores: list[float],
    emb_scores: list[float],
    bm25_weight: float = 0.4,
) -> list[float]:
    """Weighted average of BM25 and embedding scores, with min-max normalization."""

    norm_bm25 = _normalize_scores(bm25_scores)
    norm_emb = _normalize_scores(emb_scores)
    emb_weight = 1.0 - bm25_weight

    return [bm25_weight * b + emb_weight * e for b, e in zip(norm_bm25, norm_emb)]


def _count_message_tokens(
    model: str,
    messages: list[Any],
    custom_tokenizer: Optional[SelectTokenizerResponse],
    token_count_multiplier: float,
) -> int:
    if custom_tokenizer is None:
        token_count = token_counter(model=model, messages=messages)
    else:
        token_count = token_counter(model=model, messages=messages, custom_tokenizer=custom_tokenizer)
    return math.ceil(token_count * token_count_multiplier)


def _truncate_message_to_budget(
    message: dict,
    max_tokens: int,
    model: str,
    custom_tokenizer: Optional[SelectTokenizerResponse],
    token_count_multiplier: float,
    query: str,
) -> Optional[dict]:
    source = _content_to_text(message.get("content", ""))
    if not source:
        return None

    def count_tokens(text: str) -> int:
        if custom_tokenizer is None:
            token_count = token_counter(model=model, text=text)
        else:
            token_count = token_counter(model=model, text=text, custom_tokenizer=custom_tokenizer)
        return math.ceil(token_count * token_count_multiplier)

    selected = select_chunks_to_budget(
        content=source,
        query=query,
        max_tokens=max_tokens - _MESSAGE_OVERHEAD_TOKENS,
        count_tokens=count_tokens,
    )
    if selected is None:
        return None
    return {**message, "content": selected}


def _select_kept_indices_for_budget(
    normalized_messages: list[dict],
    original_messages: list[dict],
    retention_scores: list[float],
    compression_target: int,
    model: str,
    initial_kept_indices: set[int],
    tool_exchange_spans: list[set[int]],
    custom_tokenizer: Optional[SelectTokenizerResponse],
    token_count_multiplier: float,
    query: str,
) -> tuple[set[int], dict[int, dict]]:
    kept_indices = set(initial_kept_indices)
    current_tokens = sum(
        _count_message_tokens(
            model=model,
            messages=[original_messages[i]],
            custom_tokenizer=custom_tokenizer,
            token_count_multiplier=token_count_multiplier,
        )
        for i in kept_indices
    )

    span_id_by_index = {idx: span_id for span_id, span in enumerate(tool_exchange_spans) for idx in span}
    single_message_units: list[CandidateUnit] = [
        (retention_scores[idx], (idx,), True)
        for idx in range(len(normalized_messages))
        if idx not in span_id_by_index and idx not in kept_indices
    ]
    tool_span_units: list[CandidateUnit] = [
        (
            max(retention_scores[idx] for idx in span),
            tuple(sorted(span)),
            False,
        )
        for span in tool_exchange_spans
        if not any(idx in kept_indices for idx in span)
    ]
    candidate_units = sorted([*single_message_units, *tool_span_units], key=lambda unit: min(unit[1]))
    recent_units = sorted(candidate_units, key=lambda unit: max(unit[1]), reverse=True)[:_RECENT_WORKING_SET_UNITS]
    recent_indices = {unit[1] for unit in recent_units}
    older_units = [unit for unit in candidate_units if unit[1] not in recent_indices]
    coverage_bucket_size = max(math.ceil(len(older_units) / _HISTORY_COVERAGE_BUCKETS), 1)
    coverage_units = [
        max(bucket, key=lambda unit: unit[0])
        for start in range(0, len(older_units), coverage_bucket_size)
        if (bucket := older_units[start : start + coverage_bucket_size])
    ][:_HISTORY_COVERAGE_BUCKETS]

    available_tokens = max(compression_target - current_tokens, 0)
    coverage_limit = current_tokens + math.floor(available_tokens * _HISTORY_COVERAGE_BUDGET_RATIO)
    kept_indices, truncated_overrides, current_tokens = _fit_candidate_units(
        candidate_units=coverage_units,
        original_messages=original_messages,
        model=model,
        custom_tokenizer=custom_tokenizer,
        token_count_multiplier=token_count_multiplier,
        kept_indices=kept_indices,
        truncated_overrides={},
        current_tokens=current_tokens,
        token_limit=coverage_limit,
        allow_truncation=False,
        query=query,
    )
    recent_limit = current_tokens + math.floor(available_tokens * _RECENT_WORKING_SET_BUDGET_RATIO)
    kept_indices, truncated_overrides, current_tokens = _fit_candidate_units(
        candidate_units=recent_units,
        original_messages=original_messages,
        model=model,
        custom_tokenizer=custom_tokenizer,
        token_count_multiplier=token_count_multiplier,
        kept_indices=kept_indices,
        truncated_overrides=truncated_overrides,
        current_tokens=current_tokens,
        token_limit=min(recent_limit, compression_target),
        allow_truncation=False,
        query=query,
    )
    kept_indices, truncated_overrides, _ = _fit_candidate_units(
        candidate_units=sorted(candidate_units, key=lambda unit: unit[0], reverse=True),
        original_messages=original_messages,
        model=model,
        custom_tokenizer=custom_tokenizer,
        token_count_multiplier=token_count_multiplier,
        kept_indices=kept_indices,
        truncated_overrides=truncated_overrides,
        current_tokens=current_tokens,
        token_limit=compression_target,
        allow_truncation=True,
        query=query,
    )
    return kept_indices, truncated_overrides


def _fit_candidate_units(
    candidate_units: list[CandidateUnit],
    original_messages: list[dict],
    model: str,
    custom_tokenizer: Optional[SelectTokenizerResponse],
    token_count_multiplier: float,
    kept_indices: set[int],
    truncated_overrides: dict[int, dict],
    current_tokens: int,
    token_limit: int,
    allow_truncation: bool,
    query: str,
) -> tuple[set[int], dict[int, dict], int]:
    selected_indices = set(kept_indices)
    selected_overrides = dict(truncated_overrides)
    truncation_used = False

    for _score, indices, can_truncate in candidate_units:
        if any(idx in selected_indices for idx in indices):
            continue
        message_tokens = sum(
            _count_message_tokens(
                model=model,
                messages=[original_messages[idx]],
                custom_tokenizer=custom_tokenizer,
                token_count_multiplier=token_count_multiplier,
            )
            for idx in indices
        )
        remaining = token_limit - current_tokens

        if remaining <= 0:
            break

        if current_tokens + message_tokens <= token_limit:
            selected_indices.update(indices)
            current_tokens += message_tokens
        elif (
            allow_truncation
            and not truncation_used
            and can_truncate
            and len(indices) == 1
            and remaining >= max(100, token_limit // 4)
        ):
            idx = indices[0]
            truncated = _truncate_message_to_budget(
                message=original_messages[idx],
                max_tokens=remaining,
                model=model,
                custom_tokenizer=custom_tokenizer,
                token_count_multiplier=token_count_multiplier,
                query=query,
            )
            if truncated is not None:
                truncated_tokens = _count_message_tokens(
                    model=model,
                    messages=[truncated],
                    custom_tokenizer=custom_tokenizer,
                    token_count_multiplier=token_count_multiplier,
                )
                selected_overrides[idx] = truncated
                selected_indices.add(idx)
                current_tokens += truncated_tokens
                truncation_used = True

    return selected_indices, selected_overrides, current_tokens


def _get_dropped_tool_span_indices(kept_indices: set[int], tool_exchange_spans: list[set[int]]) -> set[int]:
    dropped_tool_span_indices: set[int] = set()
    for span in tool_exchange_spans:
        if not any(idx in kept_indices for idx in span):
            dropped_tool_span_indices.update(span)
    return dropped_tool_span_indices


def compress(
    messages: list[dict],
    model: str,
    call_type: CallTypes | str = CallTypes.completion,
    compression_trigger: int = 200_000,
    compression_target: Optional[int] = None,
    embedding_model: Optional[str] = None,
    embedding_model_params: Optional[dict[str, Any]] = None,
    compression_cache: Optional[DualCache] = None,
    custom_tokenizer: Optional[SelectTokenizerResponse] = None,
    token_count_multiplier: float = 1.0,
) -> CompressedResult:
    """
    Compress a list of messages by replacing lower-priority content with stubs.

    Messages below ``compression_trigger`` tokens pass through unchanged.
    Messages above are scored with BM25 (and optionally embeddings), then
    selected using relevance, recency, role priority, recent-working-set, and
    historical-coverage signals. Originals are cached and a retrieval tool is
    injected so the model can recover dropped content on demand.

    Parameters:
        messages: The conversation messages to (potentially) compress.
        model: The LLM model name — used for token counting.
        call_type: The LiteLLM call type whose message schema these messages
            follow.  Supported values:
            - ``CallTypes.completion`` / ``CallTypes.acompletion`` — OpenAI
              chat-completions shape (default)
            - ``CallTypes.anthropic_messages`` — Anthropic Messages shape
              (structured content blocks + atomic tool exchanges)
        compression_trigger: Only compress if input exceeds this token count.
        compression_target: Target token count after compression.
            Defaults to 70% of ``compression_trigger``.
        embedding_model: If provided, use BM25 + embeddings for scoring.
            If ``None``, BM25 only.
        embedding_model_params: Optional kwargs forwarded to
            ``litellm.embedding()`` when ``embedding_model`` is set.
        compression_cache: Passed through to ``litellm.embedding()`` for
            cross-turn caching of embedding vectors.
        custom_tokenizer: Resolved tokenizer used consistently for message
            selection and compression accounting.
        token_count_multiplier: Model-specific calibration applied to every
            compression token count.

    Returns:
        A ``CompressedResult`` dict containing compressed messages, token
        counts, a cache of original content, and the retrieval tool definition.
    """
    call_type_str: Final = _normalize_call_type(call_type)
    normalized_messages, original_messages = _normalize_messages_for_compression(
        messages=messages,
        call_type=call_type_str,
    )

    if compression_target is None:
        compression_target = compression_trigger * 7 // 10

    effective_token_count_multiplier = max(token_count_multiplier, 1.0)
    original_tokens = _count_message_tokens(
        model=model,
        messages=cast(list[Any], original_messages),
        custom_tokenizer=custom_tokenizer,
        token_count_multiplier=effective_token_count_multiplier,
    )

    # Pass through if below trigger
    if original_tokens <= compression_trigger:
        return CompressedResult(
            messages=original_messages,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            compression_ratio=0.0,
            cache={},
            tools=[],
            compression_skipped_reason="below_trigger",
        )

    # Extract query for relevance scoring
    query = _build_relevance_query(normalized_messages)

    # Score each message
    bm25_scores: Final = bm25_score_messages(query, normalized_messages)

    if embedding_model:
        from litellm.compression.scoring.embedding_scorer import (
            embedding_score_messages,
        )

        emb_scores: Final = embedding_score_messages(
            query,
            normalized_messages,
            model=embedding_model,
            cache=compression_cache,
            embedding_model_params=embedding_model_params,
        )
        relevance_scores = _combine_scores(bm25_scores, emb_scores, bm25_weight=0.4)
    else:
        relevance_scores = bm25_scores

    retention_scores = _hybrid_retention_scores(messages=normalized_messages, relevance_scores=relevance_scores)

    # Protected messages are never compressed
    protected_indices = get_protected_indices(normalized_messages)
    kept_indices: set[int] = set(protected_indices)

    if _is_anthropic_call_type(call_type_str):
        tool_exchange_spans, tool_sequence_error = _extract_anthropic_tool_exchange_spans(original_messages)
    else:
        tool_exchange_spans, tool_sequence_error = _extract_openai_tool_exchange_spans(original_messages)

    if tool_sequence_error is not None:
        return CompressedResult(
            messages=original_messages,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            compression_ratio=0.0,
            cache={},
            tools=[],
            compression_skipped_reason=tool_sequence_error,
        )

    kept_indices.update(_get_skill_tool_exchange_indices(original_messages, tool_exchange_spans))

    for span in tool_exchange_spans:
        if any(idx in kept_indices for idx in span):
            kept_indices.update(span)

    kept_indices, truncated_overrides = _select_kept_indices_for_budget(
        normalized_messages=normalized_messages,
        original_messages=original_messages,
        retention_scores=retention_scores,
        compression_target=compression_target,
        model=model,
        initial_kept_indices=kept_indices,
        tool_exchange_spans=tool_exchange_spans,
        custom_tokenizer=custom_tokenizer,
        token_count_multiplier=effective_token_count_multiplier,
        query=query,
    )

    # Build compressed messages and cache
    compressed_messages: list[dict] = []
    cache: dict[str, str] = {}
    used_keys: set[str] = set()
    dropped_tool_span_indices = _get_dropped_tool_span_indices(
        kept_indices=kept_indices, tool_exchange_spans=tool_exchange_spans
    )

    for i, msg in enumerate(original_messages):
        if i in dropped_tool_span_indices:
            continue
        if i in kept_indices:
            # Use the truncated version if we made one, otherwise the original
            compressed_messages.append(truncated_overrides.get(i, msg))
        else:
            key = extract_key(normalized_messages[i], fallback_index=i, used_keys=used_keys)
            content = _content_to_text(msg.get("content", ""))
            cache[key] = content
            compressed_messages.append(stub_message(msg, key))

    # Build retrieval tool in the target request schema
    tools: Final = _build_retrieval_tools(list(cache.keys()), call_type=call_type_str)

    compressed_tokens = _count_message_tokens(
        model=model,
        messages=cast(list[Any], compressed_messages),
        custom_tokenizer=custom_tokenizer,
        token_count_multiplier=effective_token_count_multiplier,
    )

    return CompressedResult(
        messages=compressed_messages,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=(round(1 - (compressed_tokens / original_tokens), 4) if original_tokens > 0 else 0.0),
        cache=cache,
        tools=tools,
    )
