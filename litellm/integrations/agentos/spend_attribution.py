"""Rewrite LiteLLM spend logs to store the public model group name."""

from __future__ import annotations

import json
from collections.abc import Mapping

from litellm.integrations.agentos.routing_hops import (
    apply_routing_hops_to_spend_payload,
    as_mapping,
    canonical_model_group_name,
)
from litellm.proxy._types import SpendLogsPayload


def logical_model_alias_from_kwargs(kwargs: Mapping[str, object] | None) -> str | None:
    """Compatibility entry point for canonical group attribution."""
    return canonical_model_group_name(kwargs)


def provider_identity_from_kwargs(kwargs: Mapping[str, object] | None) -> str | None:
    """Return the catalog provider rather than the OpenAI-compatible transport."""

    litellm_params = as_mapping(as_mapping(kwargs).get("litellm_params"))
    model_info = as_mapping(litellm_params.get("model_info"))
    if not model_info:
        metadata = as_mapping(litellm_params.get("metadata")) or as_mapping(as_mapping(kwargs).get("metadata"))
        model_info = as_mapping(metadata.get("model_info"))
    provider = model_info.get("litellm_provider")
    if isinstance(provider, str) and provider.strip():
        return provider.strip().casefold()
    upstream = model_info.get("agentos_upstream_model")
    if isinstance(upstream, str) and "/" in upstream:
        prefix = upstream.split("/", 1)[0].strip().casefold()
        if prefix:
            return prefix
    return None


def apply_logical_model_alias_to_spend_payload(
    kwargs: Mapping[str, object] | None,
    payload: SpendLogsPayload,
) -> SpendLogsPayload:
    """Set LiteLLM model/model_group columns to the canonical public name."""

    alias = logical_model_alias_from_kwargs(kwargs)
    if alias is not None:
        payload["model"] = alias
        payload["model_group"] = alias
    params = as_mapping(as_mapping(kwargs).get("litellm_params"))
    info = as_mapping(params.get("model_info"))
    if not info:
        info = as_mapping(
            (as_mapping(params.get("metadata")) or as_mapping(as_mapping(kwargs).get("metadata"))).get("model_info")
        )
    upstream = info.get("agentos_upstream_model")
    metadata = dict(  # mutable-ok: Update the native SpendLogs metadata dictionary before JSON serialization.
        as_mapping(as_mapping(payload).get("metadata"))
    )
    if isinstance(upstream, str) and upstream.strip():
        metadata["agentos_upstream_model"] = upstream.strip()
    provider = provider_identity_from_kwargs(kwargs)
    if provider is not None:
        transport = as_mapping(payload).get("custom_llm_provider")
        if isinstance(transport, str) and transport.strip() and transport.casefold() != provider:
            metadata.setdefault("agentos_transport_provider", transport.strip().casefold())
        metadata["agentos_provider"] = provider
        payload["custom_llm_provider"] = provider
    payload["metadata"] = json.dumps(metadata, ensure_ascii=False, default=str)
    status = str(as_mapping(payload).get("status") or "success")
    return apply_routing_hops_to_spend_payload(kwargs, payload, success=status != "failure")
