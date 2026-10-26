"""Enforce the AgentOS product identity for every gateway chat entrypoint."""

from __future__ import annotations

import re

from pydantic import TypeAdapter

from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

REQUEST_SNAPSHOT_METADATA_KEY = "agentos_request"
_SENSITIVE_KEYS = frozenset(
    (
        "api_key",
        "authorization",
        "client_secret",
        "credentials",
        "key",
        "master_key",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "secret_access_key",
        "secret_key",
        "token",
        "access_token",
        "x-api-key",
    )
)
_SENSITIVE_KEY_SUFFIXES = (
    "_api_key",
    "_access_token",
    "_refresh_token",
    "_client_secret",
    "_private_key",
    "_secret_access_key",
    "_secret_key",
)
_SENSITIVE_KEY_MARKERS = (
    "password",
    "passwd",
    "authorization",
    "cookie",
    "credential",
    "secret",
    "bearer",
    "clientsecret",
    "privatekey",
    "apikey",
    "accesskey",
    "sessionkey",
)
_INTERNAL_REQUEST_KEYS = frozenset(
    (
        "metadata",
        "headers",
        "extra_headers",
        "proxy_server_request",
        "litellm_logging_obj",
        "logging_obj",
    )
)

AGENTOS_IDENTITY_MESSAGE = (
    "You are AgentOS, an intelligent AI assistant. "
    "When asked who you are, identify yourself only as AgentOS. "
    "Do not claim to be or mention any underlying model provider, model family, "
    "runtime, framework, or implementation."
)


def apply_agentos_identity(
    data: dict[str, object],  # mutable-ok: Native pre-call request is updated in place.
) -> dict[str, object]:  # mutable-ok: Native pre-call hooks update and return the same mutable request dictionary.
    """Return request data with the product identity as the first system message."""

    messages = data.get("messages")
    if not isinstance(messages, list):
        return data
    first = (
        TypeAdapter(dict[str, object]).validate_python(messages[0])
        if messages and isinstance(messages[0], dict)
        else None
    )
    if first is not None and first.get("role") == "system":
        content = first.get("content")
        if isinstance(content, str):
            if not content.startswith(AGENTOS_IDENTITY_MESSAGE):
                messages[
                    0
                ] = {  # mutable-ok: Native message adapters require dictionary messages, preserving existing fields.
                    **first,
                    "content": f"{AGENTOS_IDENTITY_MESSAGE}\n\n{content}".strip(),
                }
        elif isinstance(content, list):
            first_part = (
                TypeAdapter(dict[str, object]).validate_python(content[0])
                if content and isinstance(content[0], dict)
                else None
            )
            if first_part is None or first_part.get("text") != AGENTOS_IDENTITY_MESSAGE:
                messages[0] = {  # mutable-ok: Native multipart message adapter dictionary contract.
                    **first,
                    "content": [  # mutable-ok: Native multipart content list.
                        {"type": "text", "text": AGENTOS_IDENTITY_MESSAGE},  # mutable-ok: Native content block.
                        *content,
                    ],
                }
        else:
            messages[0] = {  # mutable-ok: Native message dictionary contract.
                **first,
                "content": AGENTOS_IDENTITY_MESSAGE,
            }
        return data
    data["messages"] = [  # mutable-ok: Native completion requests require a mutable message list.
        {"role": "system", "content": AGENTOS_IDENTITY_MESSAGE},  # mutable-ok: Native chat message dictionary contract.
        *messages,
    ]
    return data


def persist_request_snapshot(
    data: dict[str, object],  # mutable-ok: Native callback metadata is updated in place.
) -> dict[str, object]:  # mutable-ok: Native callback request metadata is updated in place.
    """Persist a safe request-body snapshot for success and failure spend logs."""

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}  # mutable-ok: Native logging callbacks populate a shared request metadata dictionary.
        data["metadata"] = metadata
    metadata[REQUEST_SNAPSHOT_METADATA_KEY] = _sanitize_request_value(
        {  # mutable-ok: Native snapshot JSON object.
            key: value
            for key, value in data.items()
            if key not in _INTERNAL_REQUEST_KEYS and not _is_sensitive_key(key)
        }
    )
    return data


def _sanitize_request_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [  # mutable-ok: Native snapshot JSON array.
            _sanitize_request_value(item) for item in TypeAdapter(tuple[object, ...]).validate_python(value)
        ]
    if isinstance(value, dict):
        return {  # mutable-ok: Native request snapshot JSON objects retain their dictionary representation.
            str(key): "[REDACTED]" if _is_sensitive_key(key) else _sanitize_request_value(item)
            for key, item in TypeAdapter(dict[object, object]).validate_python(value).items()
        }
    return str(value)


def _is_sensitive_key(key: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
    compact = normalized.replace("_", "")
    return (
        normalized in _SENSITIVE_KEYS
        or normalized.endswith(_SENSITIVE_KEY_SUFFIXES)
        or any(marker in compact for marker in _SENSITIVE_KEY_MARKERS)
        or normalized.endswith("_token")
        or compact.endswith("token")
    )


def prepare_agentos_request(
    data: dict[str, object],  # mutable-ok: Native pre-call request is updated in place.
) -> dict[str, object]:  # mutable-ok: Native pre-call hook passes through and updates the original request.
    """Apply runtime request mutations and capture the exact outbound body."""

    return persist_request_snapshot(apply_agentos_identity(data))


class AgentOSIdentityCallback(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],  # mutable-ok: Native CustomLogger pre-call request contract.
        call_type: str,
    ) -> dict[str, object] | None:  # mutable-ok: Native CustomLogger returns the same request after in-place updates.
        if call_type not in ("completion", "acompletion"):
            return None
        return prepare_agentos_request(data)


agentos_identity_callback = AgentOSIdentityCallback()
