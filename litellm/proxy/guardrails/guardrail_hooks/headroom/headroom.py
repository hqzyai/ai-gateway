from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Literal, Optional

import httpx
from fastapi import HTTPException

import litellm
from httpx import Response as HttpxResponse
from litellm.proxy.spend_tracking.compression_savings import HEADROOM_GUARDRAIL_PROVIDER
from typing_extensions import TypeGuard

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    get_attribute_or_key,
    get_tool_calls_from_response,
    has_tool_with_name,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]
    httpxSpecialProvider,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.integrations.custom_logger import AgenticLoopPlan, AgenticLoopRequestPatch
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

BYPASS_HEADER = "x-headroom-bypass"
HEADROOM_RETRIEVE_TOOL_NAME = "headroom_retrieve"
HEADROOM_CCR_SYSTEM_INSTRUCTION = (
    "CCR retrieval policy: messages may contain lean-ctx CCR markers for omitted original content. "
    "If omitted content could affect your answer or next action, you MUST call `headroom_retrieve` with the exact "
    "marker hash before calling `read_file`, `terminal`, search, or any other tool to re-read or re-fetch the same "
    "source. Do not guess from the compressed summary. Skip retrieval only when the omitted content is irrelevant."
)
_HASH_PATTERN = re.compile(r"hash=([a-f0-9]{24})")
_HASH_CACHE_TTL_SECONDS = 15 * 60
_TOOL_RESULT_UNWRAP_KEYS = frozenset({"content", "output", "stdout", "stderr", "result", "text"})
_TOOL_RESULT_UNWRAP_MIN_CHARS = 400


@dataclass(frozen=True, slots=True)
class HeadroomRetrievalResult:
    content: str
    succeeded: bool
    error: str | None = None


def _is_str_object_dict(value: object) -> TypeGuard[dict[str, object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, dict)


def unwrap_agent_tool_result_content(content: object) -> object:
    """Unwrap Hermes-style JSON tool payloads so lean-ctx CCR can compress them.

    Agents like Hermes often send tool results as ``{"content": "<file text>"}`` /
    ``{"output": "<shell text>"}``. lean-ctx's /v1/compress heavily compresses plain
    text tool bodies but barely touches those JSON envelopes. When a dominant text
    field is large enough, return that field alone for compression.
    """
    if not isinstance(content, str):
        return content
    try:
        parsed: object = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return content
    if not _is_str_object_dict(parsed):
        return content

    text_fields = tuple(
        (key, value)
        for key, value in parsed.items()
        if key in _TOOL_RESULT_UNWRAP_KEYS and isinstance(value, str) and len(value) >= _TOOL_RESULT_UNWRAP_MIN_CHARS
    )
    if not text_fields:
        return content

    _, best = max(text_fields, key=lambda item: len(item[1]))
    if len(best) * 10 >= len(content) * 7:
        return best

    other_substantial = False
    for key, value in parsed.items():
        if key in _TOOL_RESULT_UNWRAP_KEYS and isinstance(value, str) and value == best:
            continue
        if isinstance(value, str) and len(value) >= _TOOL_RESULT_UNWRAP_MIN_CHARS:
            other_substantial = True
            break
        if isinstance(value, (list, dict)):
            try:
                encoded = json.dumps(value, separators=(",", ":"))
            except (TypeError, ValueError):
                encoded = ""
            if len(encoded) >= _TOOL_RESULT_UNWRAP_MIN_CHARS:
                other_substantial = True
                break
    if other_substantial:
        return content
    return best


def prepare_messages_for_compression(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    def _prepare_one(msg: dict[str, object]) -> dict[str, object]:
        if msg.get("role") != "tool":
            return msg
        content = msg.get("content")
        unwrapped = unwrap_agent_tool_result_content(content)
        if unwrapped is content:
            return msg
        return {**msg, "content": unwrapped}

    return [_prepare_one(msg) for msg in messages]


def _is_object_list(value: object) -> TypeGuard[list[object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, list)


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


def _build_headroom_retrieve_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "description": (
                "Retrieve original content omitted by Headroom. When omitted content could affect the answer or "
                "next action, call this with the exact marker hash before using any other tool to re-read or "
                "re-fetch the same source."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hash": {
                        "type": "string",
                        "description": "The 24-character hex hash from the compression marker.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional search query for BM25-ranked retrieval.",
                    },
                },
                "required": ["hash"],
            },
        },
    }


def inject_headroom_ccr_instruction(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    system_index = next(
        (
            index
            for index, message in enumerate(messages)
            if message.get("role") in ("system", "developer") and isinstance(message.get("content"), str)
        ),
        None,
    )
    if system_index is None:
        return [{"role": "system", "content": HEADROOM_CCR_SYSTEM_INSTRUCTION}, *messages]

    system_content = messages[system_index]["content"]
    if isinstance(system_content, str) and HEADROOM_CCR_SYSTEM_INSTRUCTION in system_content:
        return messages
    return [
        (
            {**message, "content": f"{system_content}\n\n{HEADROOM_CCR_SYSTEM_INSTRUCTION}"}
            if index == system_index
            else message
        )
        for index, message in enumerate(messages)
    ]


def _resolve_call_id(logging_obj: object, request_state: dict[str, object]) -> Optional[str]:
    """Resolve the litellm_call_id shared by a request's pre-call hook and its
    agentic-loop hooks, so CCR hash validation can be scoped per call instead
    of trusting any hash-shaped string that shows up in message text."""
    logging_call_id = getattr(logging_obj, "litellm_call_id", None)
    if isinstance(logging_call_id, str) and logging_call_id:
        return logging_call_id
    kwargs_call_id = request_state.get("litellm_call_id")
    return kwargs_call_id if isinstance(kwargs_call_id, str) else None


def has_headroom_retrieve_tool(tools: object) -> bool:
    return has_tool_with_name(tools, HEADROOM_RETRIEVE_TOOL_NAME)


def _headroom_logging_responses(logging_obj: object) -> tuple[dict[str, object], ...]:
    model_call_details: object = getattr(logging_obj, "model_call_details", None)
    nested_litellm_params: object = (
        model_call_details.get("litellm_params") if _is_str_object_dict(model_call_details) else None
    )
    direct_litellm_params: object = getattr(logging_obj, "litellm_params", None)
    containers: tuple[object, ...] = (direct_litellm_params, nested_litellm_params)
    responses: list[dict[str, object]] = []
    seen: set[int] = set()
    for container in containers:
        if not _is_str_object_dict(container):
            continue
        metadata = container.get("metadata")
        if not _is_str_object_dict(metadata):
            continue
        entries = metadata.get("standard_logging_guardrail_information")
        if not _is_object_list(entries):
            continue
        for entry in entries:
            if not _is_str_object_dict(entry) or entry.get("guardrail_provider") != HEADROOM_GUARDRAIL_PROVIDER:
                continue
            response = entry.get("guardrail_response")
            if not _is_str_object_dict(response) or id(response) in seen:
                continue
            seen.add(id(response))
            responses.append(response)
    return tuple(responses)


def _update_ccr_logging(logging_obj: object, values: dict[str, object]) -> None:
    for response in _headroom_logging_responses(logging_obj):
        response.update(values)


def _ccr_status(logging_obj: object) -> str | None:
    responses = _headroom_logging_responses(logging_obj)
    if not responses:
        return None
    status = responses[0].get("ccr_status")
    return status if isinstance(status, str) else None


def _response_used_fallback(response: object) -> bool:
    hidden_params: object = getattr(response, "_hidden_params", None)
    if not _is_str_object_dict(hidden_params):
        return False
    additional_headers = hidden_params.get("additional_headers")
    if not _is_str_object_dict(additional_headers):
        return False
    attempted = additional_headers.get("x-litellm-attempted-fallbacks", 0)
    if not isinstance(attempted, (int, str)):
        return False
    try:
        return int(attempted) > 0
    except (TypeError, ValueError):
        return False


def _extract_headroom_tool_calls(response: object) -> list[dict[str, object]]:
    return [
        {"id": tc["id"], "type": "function", "name": tc["name"], "arguments": tc["arguments"]}
        for tc in get_tool_calls_from_response(response)
        if tc["name"] == HEADROOM_RETRIEVE_TOOL_NAME
    ]


def _build_assistant_message_from_response(response: object) -> dict[str, object]:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        return {"role": "assistant", "content": None, "tool_calls": []}
    message = getattr(choices[0], "message", None)
    if message is None:
        return {"role": "assistant", "content": None, "tool_calls": []}
    content = getattr(message, "content", None)
    tool_calls = getattr(message, "tool_calls", None)
    raw_tool_calls: list[dict[str, object]] = []
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            raw_tool_calls.append(
                {
                    "id": getattr(tc, "id", None),
                    "type": "function",
                    "function": {
                        "name": getattr(fn, "name", None) if fn else None,
                        "arguments": getattr(fn, "arguments", "{}") if fn else "{}",
                    },
                }
            )
    return {"role": "assistant", "content": content, "tool_calls": raw_tool_calls}


def _is_responses_api_response(response: object) -> bool:
    # Real response objects can be plain dicts at runtime (e.g. TypedDict-based
    # response types), so getattr alone would silently miss the key -- use the
    # same dict-or-object accessor as the tool-call extractors.
    return isinstance(get_attribute_or_key(response, "output", None), list)


def _is_anthropic_messages_response(response: object) -> bool:
    return isinstance(get_attribute_or_key(response, "content", None), list)


def _build_anthropic_followup_messages(
    retrieved: list[tuple[dict[str, object], str]],
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
                "id": tool_call.get("id"),
                "name": tool_call.get("name"),
                "input": tool_call.get("arguments", {}),
            }
            for tool_call, _ in retrieved
        ],
    }
    user_message: dict[str, object] = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tool_call.get("id"), "content": content}
            for tool_call, content in retrieved
        ],
    }
    return [assistant_message, user_message]


def _build_responses_followup_items(
    retrieved: list[tuple[dict[str, object], str]],
) -> list[dict[str, object]]:
    """Build Responses API input items for a tool round-trip.

    The Responses API does not accept chat-style assistant/tool messages as
    follow-up input; it requires the model's function_call to be echoed back
    paired with a function_call_output keyed by the same call_id.
    """
    items: list[dict[str, object]] = []
    for tool_call, content in retrieved:
        call_id = tool_call.get("id")
        items.append(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": tool_call.get("name"),
                "arguments": json.dumps(tool_call.get("arguments", {})),
            }
        )
        items.append({"type": "function_call_output", "call_id": call_id, "output": content})
    return items


class HeadroomGuardrail(CustomGuardrail):
    @classmethod
    def get_supported_event_hooks(cls) -> List[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        guardrail_name: str | None = None,
        event_hook: GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None = None,
        default_on: bool = False,
        unreachable_fallback: str | None = None,
        enable_ccr: bool = True,
    ):
        self.headroom_api_base = (api_base or get_secret_str("HEADROOM_API_BASE") or "").rstrip("/")
        if not self.headroom_api_base:
            raise ValueError(
                "Headroom guardrail requires an API base URL. "
                "Set `api_base` in the guardrail config or HEADROOM_API_BASE env var."
            )
        self.headroom_api_key = api_key or get_secret_str("HEADROOM_API_KEY")
        self.headroom_model = model
        self.unreachable_fallback: Literal["fail_closed", "fail_open"] = (
            "fail_open" if unreachable_fallback == "fail_open" else "fail_closed"
        )
        self.enable_ccr = enable_ccr
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )
        self._issued_hashes_by_call_id: dict[str, tuple[frozenset[str], float]] = {}
        self._ccr_logging_objects_by_call_id: dict[str, object] = {}
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
            supported_event_hooks=list(self.get_supported_event_hooks()),
        )

    def _should_bypass(self, request_data: dict) -> bool:
        psr = request_data.get("proxy_server_request")
        if not _is_str_object_dict(psr):
            return False
        headers = psr.get("headers")
        if not _is_str_object_dict(headers):
            return False
        value = headers.get(BYPASS_HEADER)
        return str(value).lower() == "true"

    def _request_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.headroom_api_key:
            headers["Authorization"] = f"Bearer {self.headroom_api_key}"
        return headers

    def _prune_expired_hashes(self) -> None:
        now = time.monotonic()
        self._issued_hashes_by_call_id = {
            call_id: (hashes, expiry)
            for call_id, (hashes, expiry) in self._issued_hashes_by_call_id.items()
            if expiry > now
        }

    def _handle_compress_failure(
        self,
        messages: list[dict[str, object]],
        error: str,
        detail: dict[str, object],
    ) -> list[dict[str, object]]:
        if self.unreachable_fallback == "fail_open":
            verbose_proxy_logger.critical(
                "Headroom: %s; fail_open configured, forwarding request uncompressed. detail=%s",
                error,
                detail,
            )
            return messages
        raise HTTPException(status_code=502, detail={"error": error, **detail})

    async def _call_compress(
        self,
        messages: list[dict[str, object]],
        model: str | None,
    ) -> tuple[list[dict[str, object]], bool, dict[str, object]]:
        payload: dict[str, object] = {"messages": messages}
        if model:
            payload["model"] = model

        try:
            raw_response: HttpxResponse | None = await self.async_handler.post(  # pyright: ignore[reportUnknownMemberType]
                url=f"{self.headroom_api_base}/v1/compress",
                json=payload,
                headers=self._request_headers(),
            )
        except httpx.HTTPStatusError as e:
            return (
                self._handle_compress_failure(
                    messages,
                    "Headroom compression service returned an error",
                    {"status_code": e.response.status_code, "body": e.response.text},
                ),
                False,
                {},
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.TransportError, litellm.Timeout) as e:
            return (
                self._handle_compress_failure(
                    messages,
                    "Headroom compression service unreachable",
                    {"detail": str(e)},
                ),
                False,
                {},
            )
        if raw_response is None:
            return (
                self._handle_compress_failure(
                    messages,
                    "Headroom compression service returned no response",
                    {},
                ),
                False,
                {},
            )
        response: HttpxResponse = raw_response

        if response.status_code != 200:
            return (
                self._handle_compress_failure(
                    messages,
                    "Headroom compression service returned an error",
                    {"status_code": response.status_code, "body": response.text},
                ),
                False,
                {},
            )

        try:
            body: object = response.json()
        except ValueError:
            return (
                self._handle_compress_failure(
                    messages,
                    "Headroom compression service returned non-JSON response",
                    {"body": response.text[:500]},
                ),
                False,
                {},
            )
        if not _is_str_object_dict(body):
            return (
                self._handle_compress_failure(
                    messages,
                    "Headroom compression service returned unexpected response shape",
                    {"body": response.text[:500]},
                ),
                False,
                {},
            )

        compressed_messages = body.get("messages")
        if not _is_object_list(compressed_messages):
            return (
                self._handle_compress_failure(
                    messages,
                    "Headroom compression service response missing 'messages'",
                    {"body": response.text},
                ),
                False,
                {},
            )

        filtered = [item for item in compressed_messages if _is_str_object_dict(item)]
        if not filtered:
            return (
                self._handle_compress_failure(
                    messages,
                    "Headroom compression service returned empty message list",
                    {"body": response.text},
                ),
                False,
                {},
            )

        verbose_proxy_logger.debug(
            "Headroom: compressed %s tokens -> %s tokens (ratio %.2f)",
            body.get("tokens_before", "?"),
            body.get("tokens_after", "?"),
            body.get("compression_ratio", 0),
        )

        stats = {
            key: body[key]
            for key in (
                "tokens_before",
                "tokens_after",
                "tokens_saved",
                "compression_ratio",
                "transforms_applied",
            )
            if key in body
        }
        return filtered, True, stats

    async def _call_retrieve(self, hash_value: str, query: str | None = None) -> HeadroomRetrievalResult:
        params: dict[str, str] = {}
        if query:
            params["query"] = query

        try:
            raw_response: HttpxResponse | None = await self.async_handler.get(  # pyright: ignore[reportUnknownMemberType]
                url=f"{self.headroom_api_base}/v1/retrieve/{hash_value}",
                params=params,
                headers=self._request_headers(),
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.TransportError, litellm.Timeout) as e:
            verbose_proxy_logger.warning("Headroom: retrieve failed for hash=%s: %s", hash_value, e)
            return HeadroomRetrievalResult(
                content=f"[Headroom: retrieval failed for hash={hash_value}]",
                succeeded=False,
                error="transport_error",
            )

        if raw_response is None or raw_response.status_code == 404:
            return HeadroomRetrievalResult(
                content=f"[Headroom: hash={hash_value} not found or expired]",
                succeeded=False,
                error="not_found_or_expired",
            )

        if raw_response.status_code != 200:
            verbose_proxy_logger.warning(
                "Headroom: retrieve returned %s for hash=%s",
                raw_response.status_code,
                hash_value,
            )
            return HeadroomRetrievalResult(
                content=f"[Headroom: retrieval error {raw_response.status_code} for hash={hash_value}]",
                succeeded=False,
                error=f"http_{raw_response.status_code}",
            )

        try:
            body: object = raw_response.json()
        except ValueError:
            return HeadroomRetrievalResult(content=raw_response.text, succeeded=True)

        if _is_str_object_dict(body):
            original_content = body.get("original_content")
            if isinstance(original_content, str):
                return HeadroomRetrievalResult(content=original_content, succeeded=True)

        return HeadroomRetrievalResult(content=str(body), succeeded=True)

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        if input_type != "request":
            return inputs

        if self._should_bypass(request_data):
            verbose_proxy_logger.debug("Headroom: %s header set; skipping compression", BYPASS_HEADER)
            return inputs

        structured_messages = inputs.get("structured_messages")
        if not _is_object_list(structured_messages) or not structured_messages:
            return inputs

        messages = [m for m in structured_messages if _is_str_object_dict(m)]
        if not messages:
            return inputs

        messages = prepare_messages_for_compression(messages)

        model = self.headroom_model or request_data.get("model")
        start_time = time.time()
        compressed, compression_succeeded, stats = await self._call_compress(
            messages=messages,
            model=model if isinstance(model, str) else None,
        )
        end_time = time.time()

        if not compression_succeeded:
            return {**inputs, "structured_messages": compressed}  # pyright: ignore[reportReturnType]

        hashes = extract_hashes_from_messages(compressed)
        unique_hashes = frozenset(hashes)
        messages_for_model = inject_headroom_ccr_instruction(compressed) if hashes and self.enable_ccr else compressed
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response={
                **stats,
                "messages_before": [{**m} for m in messages],
                "messages_after": [{**m} for m in messages_for_model],
                "ccr_enabled": self.enable_ccr,
                "ccr_status": "not_triggered",
                "ccr_hashes_issued": len(unique_hashes),
                "ccr_hashes_requested": 0,
                "ccr_hashes_retrieved": 0,
                "ccr_retrieved_chars": 0,
                "ccr_fallback_used": False,
            },
            request_data=request_data,
            guardrail_status="success",
            guardrail_provider=HEADROOM_GUARDRAIL_PROVIDER,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
        )

        if not hashes or not self.enable_ccr:
            return {**inputs, "structured_messages": messages_for_model}  # pyright: ignore[reportReturnType]

        self._prune_expired_hashes()
        call_id = _resolve_call_id(logging_obj, request_data)
        if not call_id:
            call_id = str(uuid.uuid4())
            request_data["litellm_call_id"] = call_id
        self._issued_hashes_by_call_id[call_id] = (unique_hashes, time.monotonic() + _HASH_CACHE_TTL_SECONDS)

        existing_tools = inputs.get("tools")
        retrieve_tool = _build_headroom_retrieve_tool()
        if isinstance(existing_tools, list) and not has_headroom_retrieve_tool(existing_tools):
            merged_tools: list[object] = list(existing_tools) + [retrieve_tool]
        elif existing_tools is None:
            merged_tools = [retrieve_tool]
        else:
            merged_tools = list(existing_tools) if isinstance(existing_tools, list) else [retrieve_tool]

        return {**inputs, "structured_messages": messages_for_model, "tools": merged_tools}  # pyright: ignore[reportReturnType]

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
        if not self.enable_ccr:
            return False, {}

        # Gate on the response, not the request tool list. Guardrail-injected
        # tools may not always be reflected in optional_params.tools by the time
        # the agentic loop runs (streaming rebuild path). Hash scoping in
        # async_build_agentic_loop_plan still rejects forged hashes.
        tool_calls = _extract_headroom_tool_calls(response)
        if not tool_calls:
            return False, {}

        return True, {"tool_calls": tool_calls}

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
        tool_calls: list[dict[str, object]] = tools.get("tool_calls", [])  # type: ignore[assignment]

        self._prune_expired_hashes()
        call_id = _resolve_call_id(logging_obj, kwargs)
        valid_hashes = self._issued_hashes_by_call_id.get(call_id, (frozenset(), 0.0))[0] if call_id else frozenset()

        retrieved_results: list[tuple[dict[str, object], HeadroomRetrievalResult]] = []
        for tc in tool_calls:
            arguments = tc.get("arguments", {})
            hash_value = arguments.get("hash", "") if isinstance(arguments, dict) else ""
            query = arguments.get("query") if isinstance(arguments, dict) else None
            # A hash is only honored if it was issued by *this request's own*
            # Headroom /v1/compress call, scoped by litellm_call_id. Scoping by
            # message text alone is forgeable -- an attacker can plant a
            # hash-shaped string in their own prompt, and a hash issued for one
            # request would validate for any other request that echoes it back.
            if str(hash_value) not in valid_hashes:
                verbose_proxy_logger.warning(
                    "Headroom CCR: rejecting hash=%s not produced by current request compression",
                    hash_value,
                )
                result = HeadroomRetrievalResult(
                    content=f"[Headroom: hash={hash_value} was not produced by the current request]",
                    succeeded=False,
                    error="invalid_request_hash",
                )
            else:
                result = await self._call_retrieve(
                    hash_value=str(hash_value),
                    query=str(query) if query else None,
                )
            verbose_proxy_logger.debug("Headroom CCR: retrieved hash=%s (%d chars)", hash_value, len(result.content))
            retrieved_results.append((tc, result))

        retrieved = [(tc, result.content) for tc, result in retrieved_results]
        successful_results = tuple(result for _, result in retrieved_results if result.succeeded)
        failed_results = tuple(result for _, result in retrieved_results if not result.succeeded)

        if _is_responses_api_response(response):
            follow_up_messages = list(messages) + _build_responses_followup_items(retrieved)
        elif _is_anthropic_messages_response(response):
            follow_up_messages = list(messages) + _build_anthropic_followup_messages(retrieved)
        else:
            assistant_message = _build_assistant_message_from_response(response)
            # When the model mixes headroom_retrieve with client-owned tools in one
            # turn, only echo headroom tool_calls into the follow-up. Otherwise the
            # provider rejects the request for missing tool results on e.g. terminal.
            headroom_ids = {tc.get("id") for tc, _ in retrieved}
            raw_tool_calls = assistant_message.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                assistant_message = {
                    **assistant_message,
                    "tool_calls": [
                        tc for tc in raw_tool_calls if isinstance(tc, dict) and tc.get("id") in headroom_ids
                    ],
                }
            tool_results = [
                {"role": "tool", "tool_call_id": tc.get("id"), "content": content} for tc, content in retrieved
            ]
            follow_up_messages = list(messages) + [assistant_message] + tool_results

        max_tokens: Optional[int] = anthropic_messages_optional_request_params.get("max_tokens") or kwargs.get(
            "max_tokens"
        )
        optional_params_without_max_tokens = {
            k: v for k, v in anthropic_messages_optional_request_params.items() if k != "max_tokens"
        }

        full_model_name = model
        if logging_obj is not None:
            agentic_params = getattr(logging_obj, "model_call_details", {}).get("agentic_loop_params", {})
            candidate = agentic_params.get("model", model)
            if isinstance(candidate, str) and candidate:
                full_model_name = candidate
        if call_id and logging_obj is not None:
            self._ccr_logging_objects_by_call_id[call_id] = logging_obj
            _update_ccr_logging(
                logging_obj,
                {
                    "ccr_status": "retrieve_failed" if failed_results else "retrieve_success",
                    "ccr_hashes_requested": len(tool_calls),
                    "ccr_hashes_retrieved": len(successful_results),
                    "ccr_retrieved_chars": sum(len(result.content) for result in successful_results),
                    "ccr_followup_model": full_model_name,
                    "ccr_error": failed_results[0].error if failed_results else None,
                },
            )

        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=AgenticLoopRequestPatch(
                model=full_model_name,
                messages=follow_up_messages,
                max_tokens=max_tokens,
                optional_params=optional_params_without_max_tokens,
                kwargs={
                    k: v for k, v in kwargs.items() if not k.startswith("_headroom") and k != "litellm_logging_obj"
                },
            ),
            metadata={"tool_type": "headroom_ccr", "ccr_call_id": call_id},
        )

    async def async_post_agentic_loop_response_hook(
        self,
        response: Any,
        plan: AgenticLoopPlan,
        kwargs: dict,
    ) -> Any:
        call_id = plan.metadata.get("ccr_call_id")
        logging_obj = self._ccr_logging_objects_by_call_id.get(call_id) if isinstance(call_id, str) else None
        if logging_obj is not None and _ccr_status(logging_obj) == "retrieve_success":
            _update_ccr_logging(
                logging_obj,
                {
                    "ccr_status": "completed",
                    "ccr_fallback_used": _response_used_fallback(response),
                    "ccr_error": None,
                },
            )
        return response

    async def async_agentic_loop_cleanup_hook(
        self,
        plan: AgenticLoopPlan,
        kwargs: dict,
    ) -> None:
        call_id = plan.metadata.get("ccr_call_id")
        if not isinstance(call_id, str):
            return
        logging_obj = self._ccr_logging_objects_by_call_id.pop(call_id, None)
        if logging_obj is not None and _ccr_status(logging_obj) == "retrieve_success":
            _update_ccr_logging(
                logging_obj,
                {"ccr_status": "followup_failed", "ccr_error": "followup_failed"},
            )
        self._issued_hashes_by_call_id.pop(call_id, None)

    @staticmethod
    def get_config_model() -> type[GuardrailConfigModel[object]] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.headroom import (
            HeadroomGuardrailConfigModel,
        )

        return HeadroomGuardrailConfigModel
