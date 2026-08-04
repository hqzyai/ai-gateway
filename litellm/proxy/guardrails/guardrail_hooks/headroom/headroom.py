from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, List, Literal, Optional, cast

from fastapi import HTTPException
from typing_extensions import TypeGuard

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.integrations.headroom_interception.client import (
    HeadroomClient,
    HeadroomUnreachableError,
    resolve_call_id,
)
from litellm.integrations.headroom_interception.tool_formats import (
    HEADROOM_RETRIEVE_TOOL_NAME,
    extract_hashes_from_messages,
    has_headroom_retrieve_tool,
)
from litellm.litellm_core_utils.prompt_templates.factory import NormalizedToolCall
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    httpxSpecialProvider,
)
from litellm.proxy.spend_tracking.compression_savings import HEADROOM_GUARDRAIL_PROVIDER
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.integrations.custom_logger import AgenticLoopPlan, AgenticLoopRequestPatch
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

BYPASS_HEADER = "x-headroom-bypass"

# Re-exported for backwards compatibility -- these used to be defined in this
# module; they now live in litellm.integrations.headroom_interception, shared
# with the headroom_interception callback.
__all__ = [
    "HeadroomGuardrail",
    "HEADROOM_RETRIEVE_TOOL_NAME",
    "extract_hashes_from_messages",
    "has_headroom_retrieve_tool",
]


def _is_str_object_dict(value: object) -> TypeGuard[dict[str, object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, dict)


def _is_object_list(value: object) -> TypeGuard[list[object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, list)


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
        tool_call_format: str | None = None,
    ):
        resolved_api_base = (api_base or get_secret_str("HEADROOM_API_BASE") or "").rstrip("/")
        if not resolved_api_base:
            raise ValueError(
                "Headroom guardrail requires an API base URL. "
                "Set `api_base` in the guardrail config or HEADROOM_API_BASE env var."
            )
        self._client = HeadroomClient(
            api_base=resolved_api_base,
            api_key=api_key or get_secret_str("HEADROOM_API_KEY"),
            model=model,
            unreachable_fallback="fail_open" if unreachable_fallback == "fail_open" else "fail_closed",
            tool_call_format="hermes" if tool_call_format == "hermes" else "openai",
            httpx_special_provider=httpxSpecialProvider.GuardrailCallback,
        )
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
            supported_event_hooks=list(self.get_supported_event_hooks()),
        )

    @property
    def async_handler(self) -> AsyncHTTPHandler:
        return self._client.async_handler

    @property
    def unreachable_fallback(self) -> Literal["fail_closed", "fail_open"]:
        return self._client.unreachable_fallback

    @property
    def _issued_hashes_by_call_id(self) -> dict[str, tuple[frozenset[str], float]]:
        return self._client._issued_hashes_by_call_id

    def _should_bypass(self, request_data: dict) -> bool:
        psr = request_data.get("proxy_server_request")
        if not _is_str_object_dict(psr):
            return False
        headers = psr.get("headers")
        if not _is_str_object_dict(headers):
            return False
        value = headers.get(BYPASS_HEADER)
        return str(value).lower() == "true"

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

        model = request_data.get("model")
        existing_tools = inputs.get("tools")
        call_id = resolve_call_id(logging_obj, request_data)

        start_time = time.time()
        try:
            result = await self._client.compress_and_inject_retrieve_tool(
                messages=messages,
                tools=list(existing_tools) if isinstance(existing_tools, list) else None,
                model=model if isinstance(model, str) else None,
                call_id=call_id,
            )
        except HeadroomUnreachableError as e:
            raise HTTPException(status_code=502, detail={"error": e.error, **e.detail}) from e
        end_time = time.time()

        if not result["compression_applied"]:
            return {**inputs, "structured_messages": result["messages"]}  # pyright: ignore[reportReturnType]

        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response=result["stats"],
            request_data=request_data,
            guardrail_status="success",
            guardrail_provider=HEADROOM_GUARDRAIL_PROVIDER,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
        )

        if not result["retrieve_tool_injected"]:
            return {**inputs, "structured_messages": result["messages"]}  # pyright: ignore[reportReturnType]

        if result["call_id"] and not call_id:
            request_data["litellm_call_id"] = result["call_id"]

        return {
            **inputs,
            "structured_messages": result["messages"],
            "tools": result["tools"],
        }  # pyright: ignore[reportReturnType]

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
        should_run, tool_calls = await self._client.should_run_agentic_loop(response=response, tools=tools)
        if not should_run:
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
        tool_calls = cast(List[NormalizedToolCall], tools.get("tool_calls", []))
        call_id = resolve_call_id(logging_obj, kwargs)

        follow_up_messages = await self._client.build_agentic_loop_followup(
            tool_calls=tool_calls,
            response=response,
            messages=messages,
            call_id=call_id,
        )

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
            metadata={"tool_type": "headroom_ccr"},
        )

    @staticmethod
    def get_config_model() -> type[GuardrailConfigModel[object]] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.headroom import (
            HeadroomGuardrailConfigModel,
        )

        return HeadroomGuardrailConfigModel
