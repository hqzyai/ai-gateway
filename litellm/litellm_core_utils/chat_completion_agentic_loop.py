# this is a patch to allow for agentic loops covering llm_http_handler.py and openai sdk based calling flows for the .completion() api

import json
import uuid
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import cast

from litellm._internal_context import suppressed_sub_call_billing
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.agentic_loop_usage import accumulate_agentic_loop_usage
from litellm.types.integrations.custom_logger import (
    CHAT_COMPLETION_AGENTIC_SURFACE,
    NON_CODE_INTERPRETER_INTERCEPTION_INTERNAL_PREFIXES,
    AgenticLoopPlan,
    AgenticLoopRequestPatch,
    is_interception_internal_key,
    without_request_scoped_metadata,
)
from litellm.types.utils import Choices, Message, ModelResponse
from litellm.utils import CustomStreamWrapper

_FOLLOWUP_INTERNAL_PARAMS = frozenset(
    (
        "acompletion",
        "litellm_logging_obj",
        "custom_llm_provider",
        "model_alias_map",
        "stream_response",
        "custom_prompt_dict",
        "_agentic_loop_api_surface",
    )
)

_PROXY_OWNED_TOOL_NAMES = frozenset({"headroom_retrieve"})
_ROUTER_DEPLOYMENT_PARAMS = frozenset(
    {
        "api_base",
        "api_key",
        "api_version",
        "base_url",
        "client",
        "deployment_id",
        "model_info",
        "model_list",
        "specific_deployment",
    }
)


@dataclass(frozen=True, slots=True)
class AgenticLoopRoutingContext:
    model_group: str
    acompletion: Callable[..., Awaitable[object]]


_AGENTIC_LOOP_ROUTING_CONTEXT: ContextVar[AgenticLoopRoutingContext | None] = ContextVar(
    "litellm_agentic_loop_routing_context",
    default=None,
)


def set_agentic_loop_routing_context(
    model_group: str,
    acompletion: Callable[..., Awaitable[object]],
) -> Token[AgenticLoopRoutingContext | None]:
    return _AGENTIC_LOOP_ROUTING_CONTEXT.set(
        AgenticLoopRoutingContext(model_group=model_group, acompletion=acompletion)
    )


def reset_agentic_loop_routing_context(token: Token[AgenticLoopRoutingContext | None]) -> None:
    _AGENTIC_LOOP_ROUTING_CONTEXT.reset(token)


def get_agentic_loop_routing_context() -> AgenticLoopRoutingContext | None:
    return _AGENTIC_LOOP_ROUTING_CONTEXT.get()


def _tool_call_name(tool_call: object) -> str | None:
    if isinstance(tool_call, dict):
        function = tool_call.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            return name if isinstance(name, str) else None
        name = tool_call.get("name")
        return name if isinstance(name, str) else None
    function = getattr(tool_call, "function", None)
    name = getattr(function, "name", None) if function is not None else getattr(tool_call, "name", None)
    return name if isinstance(name, str) else None


def sanitize_response_hiding_proxy_owned_tools(response: ModelResponse) -> ModelResponse | None:
    """
    Strip proxy-owned tool calls (e.g. headroom_retrieve) so clients that do not
    implement them never see a leaked tool_call when the agentic follow-up fails.
    Returns None when there is nothing to strip.
    """
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        return None
    message = getattr(choices[0], "message", None)
    if message is None:
        return None
    tool_calls = getattr(message, "tool_calls", None)
    if not isinstance(tool_calls, list) or not tool_calls:
        return None

    kept = [tc for tc in tool_calls if _tool_call_name(tc) not in _PROXY_OWNED_TOOL_NAMES]
    stripped = [tc for tc in tool_calls if _tool_call_name(tc) in _PROXY_OWNED_TOOL_NAMES]
    if not stripped:
        return None

    content = getattr(message, "content", None)
    if not kept:
        notice = (
            "[LiteLLM] Proxy-owned tool call(s) could not be completed (agentic follow-up failed). Retry the request."
        )
        new_content = notice if not content else f"{content}\n{notice}"
        new_message = Message(content=new_content, role="assistant")
        finish_reason = "stop"
    else:
        new_message = Message(content=content, role="assistant", tool_calls=kept)
        finish_reason = "tool_calls"

    return ModelResponse(
        id=getattr(response, "id", "chatcmpl-sanitized"),
        choices=[Choices(finish_reason=finish_reason, index=0, message=new_message)],
        created=getattr(response, "created", 0),
        model=getattr(response, "model", None),
        object="chat.completion",
    )


def _gate_overridden(callback: CustomLogger) -> bool:
    base = CustomLogger.async_should_run_agentic_loop
    func = type(callback).async_should_run_agentic_loop
    return getattr(func, "__func__", func) is not getattr(base, "__func__", base)


def _build_plan_overridden(callback: CustomLogger) -> bool:
    base = CustomLogger.async_build_agentic_loop_plan
    func = type(callback).async_build_agentic_loop_plan
    return getattr(func, "__func__", func) is not getattr(base, "__func__", base)


def _post_hook_overridden(callback: CustomLogger) -> bool:
    base = CustomLogger.async_post_agentic_loop_response_hook
    func = type(callback).async_post_agentic_loop_response_hook
    return getattr(func, "__func__", func) is not getattr(base, "__func__", base)


def _coerce_int(value: object, default: int) -> int:
    return int(value) if isinstance(value, (int, str)) else default


def _agentic_loop_settings(kwargs: dict[str, object]) -> tuple[int, int, list[str]]:
    depth = _coerce_int(kwargs.get("_agentic_loop_depth"), 0)
    max_loops = max(_coerce_int(kwargs.get("max_agentic_loops"), 3), 1)
    raw_fingerprints = kwargs.get("_agentic_loop_fingerprints")
    fingerprints = [str(fp) for fp in raw_fingerprints] if isinstance(raw_fingerprints, list) else []
    return depth, max_loops, fingerprints


def _fingerprint_tools(tool_calls: object) -> str:
    try:
        return json.dumps(tool_calls, sort_keys=True, default=str)
    except Exception:
        return str(tool_calls)


def _check_agentic_loop_safety(
    tool_calls: object,
    fingerprints: list[str],
    depth: int,
    max_loops: int,
    model: str,
) -> str:
    fingerprint = _fingerprint_tools(tool_calls)
    if fingerprint in fingerprints:
        raise ValueError("Agentic loop detected repeated tool-call fingerprint; aborting rerun")
    if depth >= max_loops:
        raise ValueError(f"Exceeded max_agentic_loops={max_loops} for model={model}")
    return fingerprint


def _wrap_response_as_fake_stream(response: object) -> object:
    if getattr(response, "object", None) == "chat.completion.chunk":
        return response
    if not hasattr(response, "choices"):
        return response
    from litellm.llms.base_llm.base_model_iterator import (
        convert_model_response_to_streaming,
    )

    return convert_model_response_to_streaming(cast(ModelResponse, response))


def _add_agentic_loop_metadata(kwargs_for_followup: dict[str, object]) -> None:
    existing = kwargs_for_followup.get("litellm_metadata")
    metadata = without_request_scoped_metadata(existing) if isinstance(existing, dict) else {}
    for key, value in kwargs_for_followup.items():
        if key.startswith("_agentic_loop") or key == "max_agentic_loops" or is_interception_internal_key(key):
            metadata[key] = value
    kwargs_for_followup["litellm_metadata"] = metadata


def _filter_followup_kwargs(source: dict[str, object]) -> dict[str, object]:
    return {
        k: v
        for k, v in source.items()
        if not is_interception_internal_key(k, prefixes=NON_CODE_INTERPRETER_INTERCEPTION_INTERNAL_PREFIXES)
        and k not in _FOLLOWUP_INTERNAL_PARAMS
    }


async def _execute_chat_completion_agentic_plan(
    *,
    plan: AgenticLoopPlan,
    callback: CustomLogger,
    model: str,
    optional_params: dict[str, object],
    kwargs: dict[str, object],
    logging_obj: object,
    custom_llm_provider: str,
    depth: int,
    max_loops: int,
    fingerprints: list[str],
    fingerprint: str,
    routing_context: AgenticLoopRoutingContext | None,
    previous_response: ModelResponse,
    parent_already_billed: bool,
) -> object:
    """
    Run one agentic follow-up turn, keeping billed tokens equal to sent tokens.

    ``parent_already_billed`` is what decides how. On the non-streaming path the
    parent turn's response never reaches a logging object of its own (the outer
    wrapper logs whatever this returns), so the parent's tokens are folded into
    the follow-up response and the follow-up's own billing event is suppressed:
    one event, carrying every turn. On the streaming path the parent turn was
    already billed by its own stream wrapper before this runs, so folding would
    double-count it; instead the follow-up bills itself under a fresh
    ``litellm_call_id``, because SpendLogs keys rows by request id and writes
    them with ``skip_duplicates=True`` -- reusing the parent's id would silently
    drop this turn.
    """
    import litellm

    patch = plan.request_patch or AgenticLoopRequestPatch()
    if patch.messages is None:
        raise ValueError("Agentic loop plan missing patched messages")

    optional_params_for_followup = {**optional_params, **patch.optional_params}
    if patch.tools is not None:
        optional_params_for_followup["tools"] = patch.tools
    if "tool_choice" not in patch.optional_params:
        optional_params_for_followup.pop("tool_choice", None)

    kwargs_for_followup = _filter_followup_kwargs(kwargs)
    kwargs_for_followup.update(
        {k: v for k, v in _filter_followup_kwargs(patch.kwargs).items() if k not in optional_params_for_followup}
    )
    kwargs_for_followup["_agentic_loop_depth"] = depth + 1
    kwargs_for_followup["max_agentic_loops"] = max_loops
    kwargs_for_followup["_agentic_loop_fingerprints"] = fingerprints + [fingerprint]
    if parent_already_billed:
        kwargs_for_followup["litellm_call_id"] = str(uuid.uuid4())
    _add_agentic_loop_metadata(kwargs_for_followup)

    followup: Callable[..., Awaitable[object]]
    if routing_context is None:
        full_model_name = patch.model or model
        if "/" not in full_model_name:
            full_model_name = f"{custom_llm_provider}/{full_model_name}"
        followup = litellm.acompletion
    else:
        full_model_name = patch.model or routing_context.model_group
        followup = routing_context.acompletion
        optional_params_for_followup = {
            key: value for key, value in optional_params_for_followup.items() if key not in _ROUTER_DEPLOYMENT_PARAMS
        }
        kwargs_for_followup = {
            key: value for key, value in kwargs_for_followup.items() if key not in _ROUTER_DEPLOYMENT_PARAMS
        }

    try:
        with nullcontext() if parent_already_billed else suppressed_sub_call_billing():
            response_followup = await followup(
                model=full_model_name,
                messages=patch.messages,
                **optional_params_for_followup,
                **kwargs_for_followup,
            )
        if _post_hook_overridden(callback):
            try:
                response_followup = await callback.async_post_agentic_loop_response_hook(
                    response=response_followup, plan=plan, kwargs=kwargs
                )
            except Exception as e:
                _call_id = getattr(logging_obj, "litellm_call_id", "unknown")
                verbose_logger.exception(
                    "LiteLLM.AgenticHookError: Exception in "
                    "async_post_agentic_loop_response_hook [call_id=%s model=%s]: %s",
                    _call_id,
                    model,
                    str(e),
                )
        if not parent_already_billed:
            accumulate_agentic_loop_usage(
                previous_response=previous_response,
                followup_response=response_followup,
            )
        if kwargs.get("_code_interpreter_interception_converted_stream") and not depth:
            return _wrap_response_as_fake_stream(response_followup)
        return response_followup
    finally:
        try:
            await callback.async_agentic_loop_cleanup_hook(plan=plan, kwargs=kwargs)
        except Exception as e:
            _call_id = getattr(logging_obj, "litellm_call_id", "unknown")
            verbose_logger.exception(
                "LiteLLM.AgenticHookError: Exception in async_agentic_loop_cleanup_hook [call_id=%s model=%s]: %s",
                _call_id,
                model,
                str(e),
            )


async def maybe_run_chat_completion_agentic_loop(
    *,
    response: ModelResponse,
    model: str,
    messages: list,
    optional_params: dict,
    kwargs: dict,
    logging_obj: object,
    custom_llm_provider: str,
    stream: bool,
    routing_context: AgenticLoopRoutingContext | None = None,
) -> ModelResponse | CustomStreamWrapper | None:
    import litellm

    callbacks = litellm.callbacks + (getattr(logging_obj, "dynamic_success_callbacks", None) or [])
    depth, max_loops, fingerprints = _agentic_loop_settings(kwargs)
    tools = optional_params.get("tools", [])

    for callback in callbacks:
        if not isinstance(callback, CustomLogger):
            continue

        if not _gate_overridden(callback):
            continue

        hook_kwargs = {
            **kwargs,
            "_agentic_loop_api_surface": CHAT_COMPLETION_AGENTIC_SURFACE,
            "custom_llm_provider": custom_llm_provider,
        }
        try:
            should_run, tool_calls = await callback.async_should_run_agentic_loop(
                response=response,
                model=model,
                messages=messages,
                tools=tools,
                stream=stream,
                custom_llm_provider=custom_llm_provider,
                kwargs=hook_kwargs,
            )
        except Exception as e:
            verbose_logger.exception(
                "LiteLLM.AgenticHookError: Exception in chat completion agentic gate: %s",
                str(e),
            )
            continue

        if not should_run:
            continue

        fingerprint = _check_agentic_loop_safety(
            tool_calls=tool_calls,
            fingerprints=fingerprints,
            depth=depth,
            max_loops=max_loops,
            model=model,
        )

        try:
            if not _build_plan_overridden(callback):
                return await callback.async_run_agentic_loop(
                    tools=tool_calls,
                    model=model,
                    messages=messages,
                    response=response,
                    anthropic_messages_provider_config=None,
                    anthropic_messages_optional_request_params=optional_params,
                    logging_obj=logging_obj,
                    stream=stream,
                    kwargs=hook_kwargs,
                )

            plan = await callback.async_build_agentic_loop_plan(
                tools=tool_calls,
                model=model,
                messages=messages,
                response=response,
                anthropic_messages_provider_config=None,
                anthropic_messages_optional_request_params=optional_params,
                logging_obj=logging_obj,
                stream=stream,
                kwargs=hook_kwargs,
            )

            if plan.response_override is not None:
                return plan.response_override
            if plan.terminate:
                return response
            if not plan.run_agentic_loop:
                continue

            return await _execute_chat_completion_agentic_plan(
                plan=plan,
                callback=callback,
                model=model,
                optional_params=optional_params,
                kwargs=kwargs,
                logging_obj=logging_obj,
                custom_llm_provider=custom_llm_provider,
                depth=depth,
                max_loops=max_loops,
                fingerprints=fingerprints,
                fingerprint=fingerprint,
                routing_context=routing_context or get_agentic_loop_routing_context(),
                previous_response=response,
                parent_already_billed=stream,
            )
        except Exception as e:
            verbose_logger.exception(
                "LiteLLM.AgenticHookError: Exception in chat completion agentic hooks: %s",
                str(e),
            )
            if isinstance(response, ModelResponse):
                sanitized = sanitize_response_hiding_proxy_owned_tools(response)
                if sanitized is not None:
                    return sanitized

    if kwargs.get("_code_interpreter_interception_converted_stream") and not depth and hasattr(response, "choices"):
        return cast(
            "ModelResponse | CustomStreamWrapper",
            _wrap_response_as_fake_stream(response),
        )
    if isinstance(response, ModelResponse):
        sanitized = sanitize_response_hiding_proxy_owned_tools(response)
        if sanitized is not None:
            return sanitized
    return None
