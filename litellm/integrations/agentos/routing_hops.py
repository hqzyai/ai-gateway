"""Build routing hop trail for Spend Log metadata (retry / fallback visibility)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType

from pydantic import TypeAdapter

from litellm.proxy._types import SpendLogsPayload


def as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, str):
        try:
            return TypeAdapter(Mapping[str, object]).validate_json(value)
        except ValueError:
            return MappingProxyType(
                dict()  # mutable-ok: MappingProxyType freezes this native dictionary immediately.
            )
    if isinstance(value, Mapping):
        return TypeAdapter(Mapping[str, object]).validate_python(value)
    return MappingProxyType(
        dict()  # mutable-ok: MappingProxyType freezes this native dictionary immediately.
    )


def _request_metadata(kwargs: Mapping[str, object] | None) -> Mapping[str, object]:
    params = as_mapping(as_mapping(kwargs).get("litellm_params"))
    return (
        as_mapping(params.get("metadata"))
        or as_mapping(as_mapping(kwargs).get("metadata"))
        or as_mapping(as_mapping(kwargs).get("litellm_metadata"))
    )


def _model_info(kwargs: Mapping[str, object] | None, entry: Mapping[str, object] | None = None) -> Mapping[str, object]:
    if entry:
        params = as_mapping(entry.get("litellm_params"))
        info = as_mapping(params.get("model_info")) or as_mapping(entry.get("model_info"))
        return info or as_mapping(as_mapping(entry.get("metadata")).get("model_info"))
    params = as_mapping(as_mapping(kwargs).get("litellm_params"))
    return as_mapping(params.get("model_info")) or as_mapping(_request_metadata(kwargs).get("model_info"))


def _first_str(*values: object) -> str | None:
    return next((value.strip() for value in values if isinstance(value, str) and value.strip()), None)


def canonical_model_group_name(
    kwargs: Mapping[str, object] | None = None, *, model_info: Mapping[str, object] | None = None
) -> str | None:
    info = model_info if model_info is not None else _model_info(kwargs)
    return _first_str(info.get("agentos_model_group_name"), info.get("agentos_model_alias"))


def _hop_from_attempt(
    *,
    seq: int,
    outcome: str,
    kwargs: Mapping[str, object] | None = None,
    entry: Mapping[str, object] | None = None,
    error_type: str | None = None,
    error: str | None = None,
) -> Mapping[str, object]:
    info = _model_info(kwargs, entry)
    params = as_mapping(as_mapping(entry if entry else kwargs).get("litellm_params"))
    fields = (
        ("seq", seq),
        ("outcome", outcome),
        ("deployment_id", info.get("id") or as_mapping(entry).get("model_id")),
        (
            "upstream",
            _first_str(
                params.get("model"),
                info.get("agentos_upstream_model"),
                info.get("base_model"),
                as_mapping(entry).get("model"),
                None if entry else as_mapping(kwargs).get("model"),
            ),
        ),
        ("api_base", params.get("api_base") or (None if entry else as_mapping(kwargs).get("api_base"))),
        ("model_group_name", canonical_model_group_name(model_info=info)),
        ("routing_strategy", info.get("agentos_routing_strategy")),
        ("provider_account_id", info.get("agentos_provider_account_id")),
        ("error_type", error_type or None),
        ("error", error[:200] if error else None),
    )
    return MappingProxyType(
        dict(  # mutable-ok: Immediately frozen hop dictionary.
            (key, value) for key, value in fields if value is not None
        )
    )


def build_routing_hops(
    kwargs: Mapping[str, object] | None, *, success: bool = True
) -> tuple[Mapping[str, object], ...]:
    previous = _request_metadata(kwargs).get("previous_models")
    entries = (
        tuple(
            TypeAdapter(Mapping[str, object]).validate_python(item)
            for item in TypeAdapter(tuple[object, ...]).validate_python(previous)
            if isinstance(item, dict)
        )
        if isinstance(previous, list)
        else ()
    )
    failed = tuple(
        _hop_from_attempt(
            seq=index,
            outcome="failed",
            entry=entry,
            error_type=str(entry.get("exception_type") or "") or None,
            error=str(entry.get("exception_string") or entry.get("exception") or "") or None,
        )
        for index, entry in enumerate(entries, 1)
    )
    return (*failed, _hop_from_attempt(seq=len(failed) + 1, outcome="success" if success else "failed", kwargs=kwargs))


def apply_routing_hops_to_spend_payload(
    kwargs: Mapping[str, object] | None,
    payload: SpendLogsPayload,
    *,
    success: bool = True,
) -> SpendLogsPayload:
    hops = build_routing_hops(kwargs, success=success)
    fallback = _first_str(
        as_mapping(payload).get("model"), as_mapping(payload).get("model_group"), as_mapping(kwargs).get("model")
    )
    serialized_hops = tuple(
        dict(hop, upstream=fallback)  # mutable-ok: json.dumps cannot serialize MappingProxyType.
        if fallback and not hop.get("upstream") and not hop.get("model_group_name")
        else dict(  # mutable-ok: Materialize the hop only at the native SpendLogs JSON boundary.
            hop
        )
        for hop in hops
    )
    meta = as_mapping(as_mapping(payload).get("metadata"))
    nested = as_mapping(meta.get("spend_logs_metadata"))
    strategy = hops[-1].get("routing_strategy")
    merged = dict(  # mutable-ok: Assemble the native SpendLogs metadata JSON dictionary.
        meta,
        agentos_routing_hops=serialized_hops,
        spend_logs_metadata=dict(  # mutable-ok: Native nested metadata JSON dictionary.
            nested, agentos_routing_hops=serialized_hops
        ),
    )
    if strategy:
        merged["agentos_routing_strategy"] = strategy
    payload["metadata"] = json.dumps(merged, ensure_ascii=False, default=str)
    return payload
