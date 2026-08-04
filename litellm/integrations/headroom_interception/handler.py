"""
Headroom Interception Handler

CustomLogger that transparently compresses inbound chat completions and
Anthropic Messages requests via the external Headroom compression service, and
fulfills headroom_retrieve tool calls server-side via the typed agentic loop
plan (compress-cache-retrieve, "CCR"). Registered as a plain
litellm_settings.callbacks entry -- no guardrails: config required, so
compression is transparent to the calling client.

Responses API (/v1/responses) is intentionally out of scope: its `input`
kwarg isn't message-shaped like `messages`, and normalizing it is separate,
unrequested work.
"""

from typing import Any, Dict, List, Optional, Tuple, cast

from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.headroom_interception.client import HeadroomClient, resolve_call_id
from litellm.litellm_core_utils.prompt_templates.factory import NormalizedToolCall
from litellm.llms.custom_httpx.http_handler import httpxSpecialProvider
from litellm.secret_managers.main import get_secret_str
from litellm.types.integrations.custom_logger import AgenticLoopPlan, AgenticLoopRequestPatch
from litellm.types.integrations.headroom_interception import (
    HeadroomInterceptionConfig,
    HeadroomInterceptionSavingsMetadata,
)
from litellm.types.utils import CallTypes

_COMPRESSIBLE_CALL_TYPES = (
    CallTypes.completion,
    CallTypes.acompletion,
    CallTypes.anthropic_messages,
    CallTypes.aanthropic_messages,
)


def _savings_from_stats(stats: dict[str, object]) -> Optional[HeadroomInterceptionSavingsMetadata]:
    tokens_before = stats.get("tokens_before")
    tokens_after = stats.get("tokens_after")
    if isinstance(tokens_before, bool) or not isinstance(tokens_before, int):
        return None
    if isinstance(tokens_after, bool) or not isinstance(tokens_after, int):
        return None
    if tokens_after < 0 or tokens_before < tokens_after:
        return None
    return HeadroomInterceptionSavingsMetadata(
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_saved=tokens_before - tokens_after,
        source="headroom_interception",
    )


def _record_compression_savings(kwargs: dict[str, object], savings: HeadroomInterceptionSavingsMetadata) -> None:
    """
    Attach savings to the request's litellm metadata so they land in the
    SpendLog row's metadata JSON under ``compression_savings`` -- the same
    key/shape ``compression_interception`` writes, so
    ``compression_savings.py``'s spend aggregation picks it up with no changes.
    """
    existing = kwargs.get("litellm_metadata")
    if isinstance(existing, dict):
        existing["compression_savings"] = savings
        return
    kwargs["litellm_metadata"] = {"compression_savings": savings}


class HeadroomInterceptionLogger(CustomLogger):
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        unreachable_fallback: Optional[str] = None,
        tool_call_format: Optional[str] = None,
    ) -> None:
        super().__init__()
        resolved_api_base = (api_base or get_secret_str("HEADROOM_API_BASE") or "").rstrip("/")
        if not resolved_api_base:
            raise ValueError(
                "Headroom interception requires an API base URL. "
                "Set `api_base` in headroom_interception_params or the HEADROOM_API_BASE env var."
            )
        self._client = HeadroomClient(
            api_base=resolved_api_base,
            api_key=api_key or get_secret_str("HEADROOM_API_KEY"),
            model=model,
            unreachable_fallback="fail_open" if unreachable_fallback == "fail_open" else "fail_closed",
            tool_call_format="hermes" if tool_call_format == "hermes" else "openai",
            httpx_special_provider=httpxSpecialProvider.LoggingCallback,
        )

    @classmethod
    def from_config_yaml(cls, config: HeadroomInterceptionConfig) -> "HeadroomInterceptionLogger":
        return cls(
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
            model=config.get("model"),
            unreachable_fallback=config.get("unreachable_fallback"),
            tool_call_format=config.get("tool_call_format"),
        )

    @staticmethod
    def initialize_from_proxy_config(
        litellm_settings: Dict[str, Any],
        callback_specific_params: Dict[str, Any],
    ) -> "HeadroomInterceptionLogger":
        params: HeadroomInterceptionConfig = {}
        if "headroom_interception_params" in litellm_settings:
            params = litellm_settings["headroom_interception_params"]
        elif "headroom_interception" in callback_specific_params and isinstance(
            callback_specific_params["headroom_interception"], dict
        ):
            params = cast(HeadroomInterceptionConfig, callback_specific_params["headroom_interception"])
        return HeadroomInterceptionLogger.from_config_yaml(params)

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Optional[CallTypes]
    ) -> Optional[dict]:
        if call_type not in _COMPRESSIBLE_CALL_TYPES:
            return None
        if int(kwargs.get("_agentic_loop_depth", 0) or 0) > 0:
            return None

        raw_messages = kwargs.get("messages")
        if not isinstance(raw_messages, list):
            return None
        messages = [m for m in raw_messages if isinstance(m, dict)]
        if not messages:
            return None

        raw_call_id = kwargs.get("litellm_call_id")
        call_id = raw_call_id if isinstance(raw_call_id, str) else None
        raw_tools = kwargs.get("tools")
        raw_model = kwargs.get("model")

        result = await self._client.compress_and_inject_retrieve_tool(
            messages=messages,
            tools=raw_tools if isinstance(raw_tools, list) else None,
            model=raw_model if isinstance(raw_model, str) else None,
            call_id=call_id,
        )
        if not result["compression_applied"]:
            return None

        kwargs["messages"] = result["messages"]
        if result["retrieve_tool_injected"]:
            if result["tools"] is not None:
                kwargs["tools"] = result["tools"]
            if result["call_id"] and not call_id:
                kwargs["litellm_call_id"] = result["call_id"]

        savings = _savings_from_stats(result["stats"])
        if savings is not None:
            _record_compression_savings(kwargs, savings)

        return kwargs

    async def async_should_run_agentic_loop(
        self,
        response: Any,
        model: str,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        stream: bool,
        custom_llm_provider: str,
        kwargs: Dict,
    ) -> Tuple[bool, Dict]:
        should_run, tool_calls = await self._client.should_run_agentic_loop(response=response, tools=tools)
        if not should_run:
            return False, {}
        return True, {"tool_calls": tool_calls}

    async def async_build_agentic_loop_plan(
        self,
        tools: Dict,
        model: str,
        messages: List[Dict],
        response: Any,
        anthropic_messages_provider_config: Any,
        anthropic_messages_optional_request_params: Dict,
        logging_obj: Any,
        stream: bool,
        kwargs: Dict,
    ) -> AgenticLoopPlan:
        tool_calls = cast(List[NormalizedToolCall], tools.get("tool_calls", []))
        call_id = resolve_call_id(logging_obj, kwargs)

        follow_up_messages = await self._client.build_agentic_loop_followup(
            tool_calls=tool_calls,
            response=response,
            messages=messages,
            call_id=call_id,
        )

        max_tokens = cast(
            Optional[int],
            anthropic_messages_optional_request_params.get("max_tokens") or kwargs.get("max_tokens"),
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

        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=AgenticLoopRequestPatch(
                model=full_model_name,
                messages=follow_up_messages,
                max_tokens=max_tokens,
                optional_params=optional_params_without_max_tokens,
                kwargs={
                    k: v
                    for k, v in kwargs.items()
                    if not k.startswith("_headroom_interception") and k != "litellm_logging_obj"
                },
            ),
            metadata={"tool_type": "headroom_ccr", "call_id": call_id or ""},
        )
