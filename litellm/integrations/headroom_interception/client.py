"""
HTTP client for the external Headroom compression service, shared by both
integration paths: the ``headroom`` guardrail (litellm/proxy/guardrails/
guardrail_hooks/headroom/) and the ``headroom_interception`` server-side
callback (handler.py in this package).

This class has no framework inheritance (not a CustomLogger, not a
CustomGuardrail) on purpose -- it's a single-purpose HTTP + compress-cache-
retrieve (CCR) unit that both adapters compose, rather than one adapter
inheriting from or wrapping the other.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal, Optional

import httpx
from typing_extensions import TypeGuard

import litellm
from litellm._logging import verbose_logger
from litellm.exceptions import BadGatewayError
from litellm.integrations.headroom_interception.tool_formats import (
    ToolCallFormat,
    build_followup_messages,
    extract_hashes_from_messages,
    extract_retrieve_tool_calls,
    inject_retrieve_tool,
)
from litellm.litellm_core_utils.prompt_templates.factory import NormalizedToolCall
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.headroom_interception import HeadroomCompressResult

_HASH_CACHE_TTL_SECONDS = 15 * 60


def _is_str_object_dict(value: object) -> TypeGuard[dict[str, object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, dict)


def _is_object_list(value: object) -> TypeGuard[list[object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, list)


def resolve_call_id(logging_obj: object, state: dict[str, object]) -> Optional[str]:
    """Resolve the litellm_call_id shared by a request's pre-call hook and its
    agentic-loop hooks, so CCR hash validation can be scoped per call instead
    of trusting any hash-shaped string that shows up in message text."""
    logging_call_id = getattr(logging_obj, "litellm_call_id", None)
    if isinstance(logging_call_id, str) and logging_call_id:
        return logging_call_id
    state_call_id = state.get("litellm_call_id")
    return state_call_id if isinstance(state_call_id, str) else None


class HeadroomUnreachableError(BadGatewayError):
    """Raised on unreachable_fallback="fail_closed" when the Headroom compression
    service errors, times out, or returns an unusable response. A plain litellm
    exception (not fastapi.HTTPException) since HeadroomClient runs both inside
    the proxy and, via HeadroomInterceptionLogger, deep in the SDK layer where a
    FastAPI type would be a leaky abstraction."""

    def __init__(self, error: str, detail: dict[str, object]):
        self.error = error
        self.detail = detail
        super().__init__(message=f"{error}: {detail}", llm_provider="headroom", model=None)


class HeadroomClient:
    def __init__(
        self,
        *,
        api_base: str,
        api_key: Optional[str],
        model: Optional[str],
        unreachable_fallback: Literal["fail_closed", "fail_open"],
        tool_call_format: ToolCallFormat,
        httpx_special_provider: httpxSpecialProvider,
    ) -> None:
        self.headroom_api_base = api_base.rstrip("/")
        self.headroom_api_key = api_key
        self.headroom_model = model
        self.unreachable_fallback = unreachable_fallback
        self.tool_call_format = tool_call_format
        self.async_handler: AsyncHTTPHandler = get_async_httpx_client(llm_provider=httpx_special_provider)
        self._issued_hashes_by_call_id: dict[str, tuple[frozenset[str], float]] = {}

    def _request_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.headroom_api_key:
            headers["Authorization"] = f"Bearer {self.headroom_api_key}"
        return headers

    def _prune_expired_hashes(self) -> None:
        now = time.monotonic()
        self._issued_hashes_by_call_id = {
            call_id: (hashes, expiry) for call_id, (hashes, expiry) in self._issued_hashes_by_call_id.items() if expiry > now
        }

    def _handle_compress_failure(
        self,
        messages: list[dict[str, object]],
        error: str,
        detail: dict[str, object],
    ) -> list[dict[str, object]]:
        if self.unreachable_fallback == "fail_open":
            verbose_logger.critical(
                "Headroom: %s; fail_open configured, forwarding request uncompressed. detail=%s",
                error,
                detail,
            )
            return messages
        raise HeadroomUnreachableError(error=error, detail=detail)

    async def _call_compress(
        self,
        messages: list[dict[str, object]],
        model: Optional[str],
    ) -> tuple[list[dict[str, object]], bool, dict[str, object]]:
        payload: dict[str, object] = {"messages": messages}
        if model:
            payload["model"] = model

        try:
            raw_response: Optional[httpx.Response] = await self.async_handler.post(  # pyright: ignore[reportUnknownVariableType]
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
                self._handle_compress_failure(messages, "Headroom compression service returned no response", {}),
                False,
                {},
            )
        response: httpx.Response = raw_response

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

        verbose_logger.debug(
            "Headroom: compressed %s tokens -> %s tokens (ratio %.2f)",
            body.get("tokens_before", "?"),
            body.get("tokens_after", "?"),
            body.get("compression_ratio", 0),
        )

        stats = {
            key: body[key]
            for key in ("tokens_before", "tokens_after", "tokens_saved", "compression_ratio", "transforms_applied")
            if key in body
        }
        return filtered, True, stats

    async def _call_retrieve(self, hash_value: str, query: Optional[str] = None) -> str:
        params: dict[str, str] = {}
        if query:
            params["query"] = query

        try:
            raw_response: Optional[httpx.Response] = await self.async_handler.get(  # pyright: ignore[reportUnknownVariableType]
                url=f"{self.headroom_api_base}/v1/retrieve/{hash_value}",
                params=params,
                headers=self._request_headers(),
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.TransportError, litellm.Timeout) as e:
            verbose_logger.warning("Headroom: retrieve failed for hash=%s: %s", hash_value, e)
            return f"[Headroom: retrieval failed for hash={hash_value}]"

        if raw_response is None or raw_response.status_code == 404:
            return f"[Headroom: hash={hash_value} not found or expired]"

        if raw_response.status_code != 200:
            verbose_logger.warning("Headroom: retrieve returned %s for hash=%s", raw_response.status_code, hash_value)
            return f"[Headroom: retrieval error {raw_response.status_code} for hash={hash_value}]"

        try:
            body: object = raw_response.json()
        except ValueError:
            return raw_response.text

        if _is_str_object_dict(body):
            original_content = body.get("original_content")
            if isinstance(original_content, str):
                return original_content

        return str(body)

    async def compress_and_inject_retrieve_tool(
        self,
        *,
        messages: list[dict[str, object]],
        tools: Optional[list[object]],
        model: Optional[str],
        call_id: Optional[str],
    ) -> HeadroomCompressResult:
        self._prune_expired_hashes()
        compressed, succeeded, stats = await self._call_compress(
            messages=messages,
            model=self.headroom_model or model,
        )
        if not succeeded:
            return HeadroomCompressResult(
                messages=compressed,
                tools=tools,
                compression_applied=False,
                retrieve_tool_injected=False,
                call_id=call_id,
                stats={},
            )

        hashes = extract_hashes_from_messages(compressed)
        if not hashes:
            return HeadroomCompressResult(
                messages=compressed,
                tools=tools,
                compression_applied=True,
                retrieve_tool_injected=False,
                call_id=call_id,
                stats=stats,
            )

        resolved_call_id = call_id or str(uuid.uuid4())
        self._issued_hashes_by_call_id[resolved_call_id] = (
            frozenset(hashes),
            time.monotonic() + _HASH_CACHE_TTL_SECONDS,
        )
        patched_messages, patched_tools = inject_retrieve_tool(compressed, tools, self.tool_call_format)
        return HeadroomCompressResult(
            messages=patched_messages,
            tools=patched_tools,
            compression_applied=True,
            retrieve_tool_injected=True,
            call_id=resolved_call_id,
            stats=stats,
        )

    async def should_run_agentic_loop(
        self,
        *,
        response: object,
        tools: object,
    ) -> tuple[bool, list[NormalizedToolCall]]:
        tool_calls = extract_retrieve_tool_calls(response, tools, self.tool_call_format)
        return bool(tool_calls), tool_calls

    async def build_agentic_loop_followup(
        self,
        *,
        tool_calls: list[NormalizedToolCall],
        response: object,
        messages: list[dict[str, object]],
        call_id: Optional[str],
    ) -> list[dict[str, object]]:
        self._prune_expired_hashes()
        valid_hashes = self._issued_hashes_by_call_id.get(call_id, (frozenset(), 0.0))[0] if call_id else frozenset()

        retrieved: list[tuple[NormalizedToolCall, str]] = []
        for tool_call in tool_calls:
            hash_value = str(tool_call["arguments"].get("hash", "") or "")
            query = tool_call["arguments"].get("query")
            # A hash is only honored if it was issued by *this request's own*
            # Headroom /v1/compress call, scoped by litellm_call_id. Scoping by
            # message text alone is forgeable -- an attacker can plant a
            # hash-shaped string in their own prompt, and a hash issued for one
            # request would validate for any other request that echoes it back.
            if hash_value not in valid_hashes:
                verbose_logger.warning(
                    "Headroom CCR: rejecting hash=%s not produced by current request compression",
                    hash_value,
                )
                content = f"[Headroom: hash={hash_value} was not produced by the current request]"
            else:
                content = await self._call_retrieve(hash_value=hash_value, query=str(query) if query else None)
            verbose_logger.debug("Headroom CCR: retrieved hash=%s (%d chars)", hash_value, len(content))
            retrieved.append((tool_call, content))

        return build_followup_messages(response, messages, retrieved, self.tool_call_format)
