"""
Compression Interception Handler

CustomLogger that compresses inbound Anthropic Messages requests and fulfills
litellm_content_retrieve tool calls server-side via the typed agentic loop plan.
"""

import json
import math
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, cast

import litellm
from tokenizers import Tokenizer
from litellm._logging import verbose_logger
from litellm.compression import compress
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.prompt_templates.factory import (
    get_tool_calls_from_response,
)
from litellm.litellm_core_utils.token_counter import token_counter
from litellm.types.integrations.compression_interception import (
    CompressionInterceptionConfig,
    CompressionRetrievalMetadata,
    CompressionSavingsMetadata,
)
from litellm.types.integrations.custom_logger import (
    CHAT_COMPLETION_AGENTIC_SURFACE,
    AgenticLoopPlan,
    AgenticLoopRequestPatch,
)
from litellm.types.utils import (
    CallTypes,
    CustomHuggingfaceTokenizer,
    SelectTokenizerResponse,
)
from litellm.utils import _select_tokenizer

LITELLM_CONTENT_RETRIEVE_TOOL_NAME = "litellm_content_retrieve"
_CACHE_TTL_SECONDS = 15 * 60
_SUPPORTED_CALL_TYPES = frozenset(
    {
        CallTypes.completion,
        CallTypes.acompletion,
        CallTypes.anthropic_messages,
    }
)


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _compression_token_count_multiplier(kwargs: dict[str, Any]) -> float:
    model_info = kwargs.get("model_info")
    if not isinstance(model_info, dict):
        return 1.0
    value = model_info.get("compression_token_count_multiplier")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 1.0
    return max(float(value), 1.0)


@lru_cache(maxsize=32)
def _load_local_tokenizer(tokenizer_json_path: str) -> SelectTokenizerResponse:
    tokenizer = Tokenizer.from_file(tokenizer_json_path)
    return {"type": "huggingface_tokenizer", "tokenizer": tokenizer}


def _local_tokenizer_path(tokenizer_json_path: str) -> Path:
    configured_path = Path(tokenizer_json_path).expanduser()
    if configured_path.is_absolute():
        return configured_path.resolve()
    package_root = Path(litellm.__file__).resolve().parent
    return (package_root / configured_path).resolve()


def _compression_savings_from_counts(
    original_tokens: object, compressed_tokens: object
) -> CompressionSavingsMetadata | None:
    if isinstance(original_tokens, bool) or not isinstance(original_tokens, int):
        return None
    if isinstance(compressed_tokens, bool) or not isinstance(compressed_tokens, int):
        return None
    if compressed_tokens < 0 or original_tokens < compressed_tokens:
        return None
    return CompressionSavingsMetadata(
        tokens_before=original_tokens,
        tokens_after=compressed_tokens,
        tokens_saved=original_tokens - compressed_tokens,
        source="compression_interception",
    )


def _record_compression_savings(kwargs: dict[str, object], savings: CompressionSavingsMetadata) -> None:
    """
    Attach savings to the request's litellm metadata so they land in the
    SpendLog row's metadata JSON under ``compression_savings``.

    ``/v1/messages`` requests carry proxy metadata under ``litellm_metadata``
    (the ``metadata`` key is Anthropic's own API field). The existing dict is
    updated in place because the proxy and the logging object hold references
    to the same object; replacing it would orphan writes made through those
    references.
    """
    existing = kwargs.get("litellm_metadata")
    if isinstance(existing, dict):
        existing["compression_savings"] = savings
        return
    kwargs["litellm_metadata"] = {"compression_savings": savings}


def _record_compression_retrieval(kwargs: dict[str, object], retrieval: CompressionRetrievalMetadata) -> None:
    existing = kwargs.get("litellm_metadata")
    if isinstance(existing, dict):
        existing["compression_retrieval"] = retrieval
        return
    kwargs["litellm_metadata"] = {"compression_retrieval": retrieval}


def _compression_retrieval_metadata(
    requested_keys: int,
    cache_hits: int = 0,
    cache_misses: int = 0,
    second_call_success: bool = False,
) -> CompressionRetrievalMetadata:
    return CompressionRetrievalMetadata(
        source="compression_interception",
        retrieval_requested=True,
        requested_keys=requested_keys,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        second_call_success=second_call_success,
    )


class CompressionInterceptionLogger(CustomLogger):
    """
    CustomLogger that implements transparent prompt compression + retrieval loops.

    Flow:
    1. Compress inbound /v1/messages requests in pre-call hook.
    2. Inject litellm_content_retrieve tool and persist compressed cache by call_id.
    3. Detect retrieval tool_use blocks in first model response.
    4. Build typed rerun plan with tool_result blocks from the compressed cache.
    """

    def __init__(
        self,
        enabled: bool = True,
        compression_trigger: int = 200_000,
        compression_target: Optional[int] = None,
        context_window_tokens: Optional[int] = None,
        safety_buffer_tokens: int = 4096,
        embedding_model: Optional[str] = None,
        embedding_model_params: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.enabled = enabled
        self.compression_trigger = compression_trigger
        self.compression_target = compression_target
        self.context_window_tokens = context_window_tokens
        self.safety_buffer_tokens = safety_buffer_tokens
        self.embedding_model = embedding_model
        self.embedding_model_params = embedding_model_params
        self._compression_cache_by_call_id: dict[str, tuple[dict[str, str], float]] = {}

    @classmethod
    def from_config_yaml(cls, config: CompressionInterceptionConfig) -> "CompressionInterceptionLogger":
        return cls(
            enabled=bool(config.get("enabled", True)),
            compression_trigger=int(config.get("compression_trigger", 200_000)),
            compression_target=config.get("compression_target"),
            context_window_tokens=config.get("context_window_tokens"),
            safety_buffer_tokens=int(config.get("safety_buffer_tokens", 4096)),
            embedding_model=config.get("embedding_model"),
            embedding_model_params=config.get("embedding_model_params"),
        )

    @staticmethod
    def initialize_from_proxy_config(
        litellm_settings: dict[str, Any],
        callback_specific_params: dict[str, Any],
    ) -> "CompressionInterceptionLogger":
        compression_params: CompressionInterceptionConfig = {}
        if "compression_interception_params" in litellm_settings:
            compression_params = litellm_settings["compression_interception_params"]
        elif "compression_interception" in callback_specific_params and isinstance(
            callback_specific_params["compression_interception"], dict
        ):
            compression_params = cast(
                CompressionInterceptionConfig,
                callback_specific_params["compression_interception"],
            )
        return CompressionInterceptionLogger.from_config_yaml(compression_params)

    async def async_pre_call_deployment_hook(
        self, kwargs: dict[str, Any], call_type: Optional[CallTypes]
    ) -> Optional[dict]:
        if not self.enabled:
            return None
        compression_call_type = call_type or CallTypes.anthropic_messages
        if compression_call_type not in _SUPPORTED_CALL_TYPES:
            return None
        if int(kwargs.get("_agentic_loop_depth", 0) or 0) > 0:
            return None

        messages = kwargs.get("messages")
        model = kwargs.get("model")
        if not isinstance(messages, list) or not isinstance(model, str):
            return None

        if self._has_retrieval_tool(kwargs.get("tools")):
            return None

        self._prune_expired_cache()
        custom_tokenizer = self._resolve_custom_tokenizer(kwargs=kwargs, model=model)
        token_count_multiplier = _compression_token_count_multiplier(kwargs)
        compression_trigger, compression_target = self._compression_limits(
            kwargs=kwargs,
            model=model,
            custom_tokenizer=custom_tokenizer,
            token_count_multiplier=token_count_multiplier,
        )

        compressed = compress(  # type: ignore
            messages=messages,
            model=model,
            call_type=compression_call_type,
            compression_trigger=compression_trigger,
            compression_target=compression_target,
            embedding_model=self.embedding_model,
            embedding_model_params=self.embedding_model_params,
            custom_tokenizer=custom_tokenizer,
            token_count_multiplier=token_count_multiplier,
        )

        cache = cast(dict[str, str], compressed.get("cache", {}))
        skip_reason = cast(Optional[str], compressed.get("compression_skipped_reason"))
        compressed_tools = cast(list[dict[str, Any]], compressed.get("tools", []))
        compressed_messages = cast(list[dict[str, Any]], compressed.get("messages", messages))
        compression_applied = skip_reason is None and compressed_messages != messages

        # Only mutate kwargs when compression actually produced a result.
        # If compression was a no-op (below trigger, invalid tool sequence, etc.),
        # leave ``messages`` and ``tools`` untouched — injecting an empty
        # ``tools: []`` onto a request that originally had no tools breaks
        # Anthropic Messages requests.
        if compression_applied:
            kwargs["messages"] = compressed_messages
            if compressed_tools:
                kwargs["tools"] = self._merge_tools(
                    existing_tools=cast(Optional[list[dict[str, Any]]], kwargs.get("tools")),
                    compressed_tools=compressed_tools,
                )
            call_id = cast(Optional[str], kwargs.get("litellm_call_id"))
            if cache:
                if not call_id:
                    call_id = str(uuid.uuid4())
                    kwargs["litellm_call_id"] = call_id
                self._compression_cache_by_call_id[call_id] = (cache, time.time())
            savings = _compression_savings_from_counts(
                original_tokens=compressed.get("original_tokens"),
                compressed_tokens=compressed.get("compressed_tokens"),
            )
            if savings is not None:
                _record_compression_savings(kwargs=kwargs, savings=savings)
            verbose_logger.debug(
                "CompressionInterception: compressed request [call_id=%s original=%d compressed=%d cached_keys=%d]",
                call_id,
                compressed.get("original_tokens"),
                compressed.get("compressed_tokens"),
                len(cache),
            )
        elif skip_reason is not None:
            verbose_logger.debug(
                "CompressionInterception: compression skipped [reason=%s original=%d compressed=%d]",
                skip_reason,
                compressed.get("original_tokens"),
                compressed.get("compressed_tokens"),
            )

        return kwargs

    def _compression_limits(
        self,
        kwargs: dict[str, Any],
        model: str,
        custom_tokenizer: Optional[SelectTokenizerResponse] = None,
        token_count_multiplier: float = 1.0,
    ) -> tuple[int, Optional[int]]:
        requested_output_tokens = next(
            (
                value
                for key in ("max_tokens", "max_completion_tokens", "max_output_tokens")
                if (value := _positive_int(kwargs.get(key))) is not None
            ),
            None,
        )
        context_window_tokens = (
            self.context_window_tokens or self._model_info_context_window(kwargs) or self._model_context_window(model)
        )
        if requested_output_tokens is None or context_window_tokens is None:
            return self.compression_trigger, self.compression_target

        tool_schema_tokens = self._tool_schema_tokens(
            model=model,
            tools=kwargs.get("tools"),
            custom_tokenizer=custom_tokenizer,
            token_count_multiplier=token_count_multiplier,
        )
        budget_target = max(
            context_window_tokens - requested_output_tokens - tool_schema_tokens - max(self.safety_buffer_tokens, 0),
            1,
        )
        compression_target = (
            min(self.compression_target, budget_target) if self.compression_target is not None else budget_target
        )
        return min(self.compression_trigger, compression_target), compression_target

    @staticmethod
    def _model_info_context_window(kwargs: dict[str, Any]) -> int | None:
        model_info = kwargs.get("model_info")
        if not isinstance(model_info, dict):
            return None
        return _positive_int(model_info.get("max_input_tokens"))

    @staticmethod
    def _model_context_window(model: str) -> int | None:
        try:
            model_info = litellm.get_model_info(model=model)
        except Exception:
            return None
        return _positive_int(model_info.get("max_input_tokens"))

    @staticmethod
    def _tool_schema_tokens(
        model: str,
        tools: object,
        custom_tokenizer: Optional[SelectTokenizerResponse] = None,
        token_count_multiplier: float = 1.0,
    ) -> int:
        if not isinstance(tools, list) or not tools:
            return 0
        try:
            serialized_tools = json.dumps(tools, ensure_ascii=False, separators=(",", ":"), default=str)
            if custom_tokenizer is None:
                token_count = token_counter(model=model, text=serialized_tools)
            else:
                token_count = token_counter(model=model, text=serialized_tools, custom_tokenizer=custom_tokenizer)
            return math.ceil(token_count * max(token_count_multiplier, 1.0))
        except (KeyError, TypeError, ValueError):
            return 0

    @staticmethod
    def _resolve_custom_tokenizer(
        kwargs: dict[str, Any],
        model: str,
    ) -> Optional[SelectTokenizerResponse]:
        model_info = kwargs.get("model_info")
        if not isinstance(model_info, dict):
            return None
        tokenizer_config = model_info.get("custom_tokenizer")
        if not isinstance(tokenizer_config, dict):
            return None
        try:
            tokenizer_json_path = tokenizer_config.get("tokenizer_json_path")
            if isinstance(tokenizer_json_path, str) and tokenizer_json_path:
                return _load_local_tokenizer(str(_local_tokenizer_path(tokenizer_json_path)))
            if not isinstance(tokenizer_config.get("identifier"), str):
                return None
            if not isinstance(tokenizer_config.get("revision"), str):
                return None
            auth_token = tokenizer_config.get("auth_token")
            if auth_token is not None and not isinstance(auth_token, str):
                return None
            return _select_tokenizer(
                model=model,
                custom_tokenizer=cast(CustomHuggingfaceTokenizer, tokenizer_config),
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            verbose_logger.exception(
                "CompressionInterception: failed to load model_info.custom_tokenizer for model=%s",
                model,
            )
            return None

    async def async_should_run_agentic_loop(
        self,
        response: Any,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]],
        stream: bool,
        custom_llm_provider: str,
        kwargs: dict,
    ) -> tuple[bool, dict]:
        if not self.enabled:
            return False, {}
        if not self._has_retrieval_tool(tools):
            return False, {}

        tool_calls, thinking_blocks = self._extract_retrieval_tool_calls(response=response)
        if not tool_calls:
            return False, {}

        call_id = kwargs.get("litellm_call_id")
        _record_compression_retrieval(
            kwargs=kwargs,
            retrieval=_compression_retrieval_metadata(requested_keys=len(tool_calls)),
        )
        verbose_logger.info(
            "CompressionInterception: retrieval requested [call_id=%s requested_keys=%d]",
            call_id if isinstance(call_id, str) and call_id else "unknown",
            len(tool_calls),
        )

        return True, {
            "tool_calls": tool_calls,
            "thinking_blocks": thinking_blocks,
            "tool_type": "compression_retrieval",
        }

    async def async_build_agentic_loop_plan(
        self,
        tools: dict,
        model: str,
        messages: list[dict],
        response: Any,
        anthropic_messages_provider_config: Any,
        anthropic_messages_optional_request_params: dict,
        logging_obj: Any,
        stream: bool,
        kwargs: dict,
    ) -> AgenticLoopPlan:
        self._prune_expired_cache()
        tool_calls = cast(list[dict[str, Any]], tools.get("tool_calls", []))
        thinking_blocks = cast(list[dict[str, Any]], tools.get("thinking_blocks", []))

        call_id = self._resolve_call_id(logging_obj=logging_obj, kwargs=kwargs)
        cache = self._get_cache(call_id=call_id)
        retrieval_resolutions = tuple(self._resolve_retrieval_content(tool_call, cache) for tool_call in tool_calls)
        retrieval_results = [content for content, _hit in retrieval_resolutions]
        cache_hits = sum(1 for _content, hit in retrieval_resolutions if hit)
        cache_misses = len(retrieval_resolutions) - cache_hits
        _record_compression_retrieval(
            kwargs=kwargs,
            retrieval=_compression_retrieval_metadata(
                requested_keys=len(tool_calls),
                cache_hits=cache_hits,
                cache_misses=cache_misses,
            ),
        )
        verbose_logger.info(
            "CompressionInterception: cache lookup completed "
            "[call_id=%s requested_keys=%d cache_hits=%d cache_misses=%d]",
            call_id or "unknown",
            len(tool_calls),
            cache_hits,
            cache_misses,
        )

        if kwargs.get("_agentic_loop_api_surface") == CHAT_COMPLETION_AGENTIC_SURFACE:
            assistant_message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", LITELLM_CONTENT_RETRIEVE_TOOL_NAME),
                            "arguments": json.dumps(tc.get("input", {}), ensure_ascii=False),
                        },
                    }
                    for tc in tool_calls
                ],
            }
            tool_messages = [
                {
                    "role": "tool",
                    "tool_call_id": tool_calls[i].get("id"),
                    "content": retrieval_results[i],
                }
                for i in range(len(tool_calls))
            ]
            follow_up_messages = messages + [assistant_message] + tool_messages
        else:
            assistant_message = {
                "role": "assistant",
                "content": thinking_blocks
                + [
                    {
                        "type": "tool_use",
                        "id": tc.get("id"),
                        "name": tc.get("name", LITELLM_CONTENT_RETRIEVE_TOOL_NAME),
                        "input": tc.get("input", {}),
                    }
                    for tc in tool_calls
                ],
            }
            user_message = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_calls[i].get("id"),
                        "content": retrieval_results[i],
                    }
                    for i in range(len(tool_calls))
                ],
            }
            follow_up_messages = messages + [assistant_message, user_message]

        max_tokens = cast(
            Optional[int],
            anthropic_messages_optional_request_params.get("max_tokens") or kwargs.get("max_tokens"),
        )
        optional_params_without_max_tokens = {
            k: v for k, v in anthropic_messages_optional_request_params.items() if k != "max_tokens"
        }

        full_model_name = model
        if logging_obj is not None:
            agentic_params = logging_obj.model_call_details.get("agentic_loop_params", {})
            full_model_name = cast(str, agentic_params.get("model", model))

        request_patch = AgenticLoopRequestPatch(
            model=full_model_name,
            messages=follow_up_messages,
            max_tokens=max_tokens,
            optional_params=optional_params_without_max_tokens,
            kwargs=self._prepare_followup_kwargs(kwargs=kwargs),
        )

        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=request_patch,
            metadata={
                "tool_type": "compression_retrieval",
                "call_id": call_id or "",
                "requested_keys": len(tool_calls),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
            },
        )

    async def async_post_agentic_loop_response_hook(
        self, response: object, plan: AgenticLoopPlan, kwargs: dict[str, object]
    ) -> object:
        metadata = plan.metadata
        if metadata.get("tool_type") != "compression_retrieval":
            return response

        requested_keys = int(metadata.get("requested_keys", 0))
        cache_hits = int(metadata.get("cache_hits", 0))
        cache_misses = int(metadata.get("cache_misses", 0))
        call_id = str(metadata.get("call_id") or "unknown")
        _record_compression_retrieval(
            kwargs=kwargs,
            retrieval=_compression_retrieval_metadata(
                requested_keys=requested_keys,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                second_call_success=True,
            ),
        )
        verbose_logger.info(
            "CompressionInterception: retrieval follow-up succeeded "
            "[call_id=%s requested_keys=%d cache_hits=%d cache_misses=%d]",
            call_id,
            requested_keys,
            cache_hits,
            cache_misses,
        )
        return response

    def _prune_expired_cache(self) -> None:
        now = time.time()
        self._compression_cache_by_call_id = {
            call_id: (cache, created_at)
            for call_id, (
                cache,
                created_at,
            ) in self._compression_cache_by_call_id.items()
            if now - created_at <= _CACHE_TTL_SECONDS
        }

    def _get_cache(self, call_id: Optional[str]) -> dict[str, str]:
        if not call_id:
            return {}
        cache_entry = self._compression_cache_by_call_id.get(call_id)
        if cache_entry is None:
            return {}
        return cache_entry[0]

    def _resolve_call_id(self, logging_obj: Any, kwargs: dict[str, Any]) -> Optional[str]:
        if logging_obj is not None:
            logging_call_id = getattr(logging_obj, "litellm_call_id", None)
            if isinstance(logging_call_id, str) and logging_call_id:
                return logging_call_id
        kwargs_call_id = kwargs.get("litellm_call_id")
        return cast(Optional[str], kwargs_call_id if isinstance(kwargs_call_id, str) else None)

    def _resolve_retrieval_content(self, tool_call: dict[str, Any], cache: dict[str, str]) -> tuple[str, bool]:
        raw_input = tool_call.get("input", {})
        key = ""
        if isinstance(raw_input, dict):
            key = str(raw_input.get("key", "") or "")
        if not key:
            return "No retrieval key provided.", False
        if key in cache:
            return cache[key], True
        return f"[compressed content key '{key}' not found]", False

    def _extract_retrieval_tool_calls(self, response: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        tool_calls = [
            {
                "id": tool_call["id"],
                "type": "tool_use",
                "name": LITELLM_CONTENT_RETRIEVE_TOOL_NAME,
                "input": tool_call["arguments"],
            }
            for tool_call in get_tool_calls_from_response(response)
            if tool_call["name"] == LITELLM_CONTENT_RETRIEVE_TOOL_NAME
        ]

        if isinstance(response, dict):
            content = response.get("content", [])
        else:
            content = getattr(response, "content", []) or []

        if not isinstance(content, list):
            return tool_calls, []

        thinking_blocks: list[dict[str, Any]] = []

        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type in ("thinking", "redacted_thinking"):
                    thinking_blocks.append(block)
            else:
                block_type = getattr(block, "type", None)
                if block_type == "thinking":
                    thinking_blocks.append(
                        {
                            "type": "thinking",
                            "thinking": getattr(block, "thinking", ""),
                            "signature": getattr(block, "signature", ""),
                        }
                    )
                elif block_type == "redacted_thinking":
                    thinking_blocks.append(
                        {
                            "type": "redacted_thinking",
                            "data": getattr(block, "data", ""),
                        }
                    )

        return tool_calls, thinking_blocks

    def _prepare_followup_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        internal_keys = {"litellm_logging_obj"}
        return {
            k: v for k, v in kwargs.items() if not k.startswith("_compression_interception") and k not in internal_keys
        }

    def _has_retrieval_tool(self, tools: Any) -> bool:
        if not isinstance(tools, list):
            return False
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if tool.get("type") == "function" and isinstance(function, dict):
                if function.get("name") == LITELLM_CONTENT_RETRIEVE_TOOL_NAME:
                    return True
            if tool.get("type") == "custom" and tool.get("name") == LITELLM_CONTENT_RETRIEVE_TOOL_NAME:
                return True
        return False

    def _merge_tools(
        self,
        existing_tools: Optional[list[dict[str, Any]]],
        compressed_tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged = list(existing_tools or [])
        if self._has_retrieval_tool(merged):
            return merged
        merged.extend(compressed_tools)
        return merged
