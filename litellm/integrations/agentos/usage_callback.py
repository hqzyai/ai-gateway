from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import HTTPException
from prometheus_client import Gauge
from pydantic import TypeAdapter

from litellm.integrations.agentos.routing_hops import as_mapping, canonical_model_group_name
from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("agentos.usage_callback")
_ATTRIBUTION_KEY = "_agentos_attribution"
_OUTBOX_PENDING = Gauge(
    "agentos_usage_outbox_pending",
    "Pending AgentOS usage events waiting for BFF delivery.",
)
_OUTBOX_DEAD_LETTER = Gauge(
    "agentos_usage_outbox_dead_letter",
    "AgentOS usage events rejected permanently and awaiting operator action.",
)
_OUTBOX_OLDEST_PENDING_AGE = Gauge(
    "agentos_usage_outbox_oldest_pending_age_seconds",
    "Age in seconds of the oldest pending AgentOS usage event.",
)


class PermanentUsageDeliveryError(RuntimeError):
    pass


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, dict):
        return TypeAdapter(dict[str, object]).validate_python(value)
    for method in ("model_dump", "dict"):
        converter = getattr(value, method, None)
        if callable(converter):
            converted = converter()
            if isinstance(converted, dict):
                return TypeAdapter(dict[str, object]).validate_python(converted)
    return as_mapping(None)


def _timestamp(value: object) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = datetime.now(UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _duration_ms(started_at: str, completed_at: str) -> float:
    started = datetime.fromisoformat(started_at)
    completed = datetime.fromisoformat(completed_at)
    return max(0.0, (completed - started).total_seconds() * 1000)


def _seal_attribution(payload: Mapping[str, object], token: str) -> Mapping[str, object]:
    trusted = {  # mutable-ok: Canonical json.dumps dictionary encoding is part of the existing HMAC signature contract.
        key: payload.get(key)
        for key in (
            "model",
            "org_id",
            "team_id",
            "user_id",
            "profile_id",
            "key_type",
            "key_hash",
            "agent_instance_id",
            "runtime_session_id",
            "skill_key",
        )
    }
    body = json.dumps(trusted, sort_keys=True, separators=(",", ":")).encode()
    signature = hmac.new(token.encode(), body, hashlib.sha256).hexdigest()
    return {  # mutable-ok: Native request metadata carries this signed JSON dictionary.
        "payload": trusted,
        "signature": signature,
    }


def _verified_attribution(kwargs: Mapping[str, object], token: str) -> Mapping[str, object]:
    if not token:
        return as_mapping(None)
    standard = _mapping(kwargs.get("standard_logging_object"))
    standard_metadata = _mapping(standard.get("metadata"))
    requester_metadata = _mapping(standard_metadata.get("requester_metadata"))
    request_metadata = _mapping(_mapping(kwargs.get("litellm_params")).get("metadata"))
    candidates = (
        standard_metadata.get(_ATTRIBUTION_KEY),
        requester_metadata.get(_ATTRIBUTION_KEY),
        request_metadata.get(_ATTRIBUTION_KEY),
    )
    for candidate in candidates:
        sealed = _mapping(candidate)
        payload = _mapping(sealed.get("payload"))
        signature = sealed.get("signature")
        if not payload or not isinstance(signature, str):
            continue
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        expected = hmac.new(token.encode(), body, hashlib.sha256).hexdigest()
        if hmac.compare_digest(signature, expected):
            return payload
    return as_mapping(None)


def _has_billable_attribution(kwargs: Mapping[str, object], token: str) -> bool:
    """Exclude master-key health/admin calls that do not belong to a billable AgentOS identity."""

    if _verified_attribution(kwargs, token):
        return True
    standard = _mapping(kwargs.get("standard_logging_object"))
    standard_metadata = _mapping(standard.get("metadata"))
    key_metadata = _mapping(standard_metadata.get("user_api_key_metadata"))
    return key_metadata.get("key_type") in ("profile", "virtual", "evaluation", "chat_share", "agent_api")


def build_usage_payload(
    kwargs: Mapping[str, object],
    response_obj: object,
    start_time: object,
    end_time: object,
    *,
    internal_token: str = "",
) -> dict[  # mutable-ok: Native failure callback adds outcome fields to this JSON payload.
    str, object
]:
    response = _mapping(response_obj)
    standard = _mapping(kwargs.get("standard_logging_object"))
    standard_metadata = _mapping(standard.get("metadata"))
    # Static owner/key identity must come from LiteLLM's authenticated standard
    # logging object. Request metadata is caller-controlled, so task/Skill
    # attribution stays empty until Hermes sends a separately authenticated
    # task-scoped identity.
    trusted_attribution = _verified_attribution(kwargs, internal_token)
    key_metadata = trusted_attribution or _mapping(standard_metadata.get("user_api_key_metadata"))
    usage = _mapping(response.get("usage"))
    if not usage:
        usage = _mapping(standard.get("usage"))
    prompt_details = _mapping(usage.get("prompt_tokens_details"))

    def first(*values: object) -> object:
        return next((value for value in values if value not in (None, "")), None)

    call_id = first(
        kwargs.get("litellm_call_id"),
        standard.get("id"),
        standard.get("request_id"),
        kwargs.get("request_id"),
        response.get("id"),
    )
    if call_id is None:
        raise ValueError("LiteLLM callback has no call id")
    litellm_params = _mapping(kwargs.get("litellm_params"))
    model_info = _mapping(litellm_params.get("model_info")) or _mapping(standard.get("model_info"))
    model_group_name = first(
        canonical_model_group_name(model_info=model_info),
        standard.get("model_group"),
        kwargs.get("model"),
        trusted_attribution.get("model"),
    )
    model = first(response.get("model"), standard.get("model"), kwargs.get("model"), model_group_name)
    if model_group_name is None or model is None:
        raise ValueError("LiteLLM callback has no model identity")
    deployment_id = first(
        model_info.get("id"),
        standard.get("model_id"),
        standard_metadata.get("model_id"),
    )
    upstream_model = first(model_info.get("agentos_upstream_model"), litellm_params.get("model"), model)
    started_at = _timestamp(start_time)
    completed_at = _timestamp(end_time)
    completion_started = first(
        standard.get("completion_start_time"),
        standard.get("completionStartTime"),
        kwargs.get("completion_start_time"),
    )
    first_token_latency_ms = (
        _duration_ms(started_at, _timestamp(completion_started)) if completion_started is not None else None
    )
    response_cost = first(
        standard.get("response_cost"),
        kwargs.get("response_cost"),
        litellm_params.get("response_cost"),
    )
    try:
        provider_cost = TypeAdapter(float).validate_python(response_cost) if response_cost is not None else None
    except (TypeError, ValueError):
        provider_cost = None
    fallback_depth = first(
        standard_metadata.get("fallback_depth"),
        standard.get("fallback_depth"),
        kwargs.get("fallback_depth"),
        0,
    )
    silent_experiment = bool(
        first(
            standard_metadata.get("is_silent_experiment"),
            standard.get("is_silent_experiment"),
            kwargs.get("is_silent_experiment"),
            False,
        )
    )
    logical_request_id = (
        first(
            standard_metadata.get("logical_request_id"),
            standard.get("request_id"),
            kwargs.get("request_id"),
            standard.get("litellm_call_id"),
            call_id,
        )
        if deployment_id is not None
        else first(
            standard_metadata.get("logical_request_id"),
            standard.get("request_id"),
            kwargs.get("request_id"),
            standard.get("id"),
            call_id,
        )
    )
    usage_call_id = str(call_id)
    attempt_id = usage_call_id
    if deployment_id is not None:
        attempt_digest = hashlib.sha256(
            "|".join(
                (
                    usage_call_id,
                    str(deployment_id),
                    started_at,
                    str(standard.get("id") or ""),
                    str(fallback_depth),
                    str(silent_experiment),
                )
            ).encode()
        ).hexdigest()[:24]
        attempt_id = f"{usage_call_id}:{attempt_digest}"
        usage_call_id = attempt_id

    def optional_int(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, (str, bytes, int, float)):
            return int(value)
        raise TypeError("Invalid usage integer")

    key_type = first(
        key_metadata.get("key_type"),
        "profile" if key_metadata.get("profile_id") else None,
    )
    profile_id = key_metadata.get("profile_id")
    agent_instance_id = key_metadata.get("agent_instance_id")
    task_run_id = None
    skill_key = key_metadata.get("skill_key") if key_type == "evaluation" else None
    runtime_session_id = first(
        trusted_attribution.get("runtime_session_id"),
        standard.get("session_id"),
        standard_metadata.get("session_id"),
        standard_metadata.get("litellm_session_id"),
    )
    if key_type in ("virtual", "evaluation"):
        profile_id = agent_instance_id = task_run_id = skill_key = None
        if key_type == "evaluation":
            skill_key = key_metadata.get("skill_key")

    payload = {  # mutable-ok: Native callback payload must be a JSON dictionary and receives an optional session field.
        "call_id": usage_call_id,
        "model": str(model),
        "model_group_name": str(model_group_name),
        "logical_request_id": str(logical_request_id),
        "attempt_id": attempt_id,
        "logical_model_id": model_info.get("agentos_logical_model_id"),
        "deployment_id": str(deployment_id) if deployment_id is not None else None,
        "provider_account_id": model_info.get("agentos_provider_account_id"),
        "credential_version": model_info.get("agentos_credential_version"),
        "upstream_model": str(upstream_model),
        "routing_policy_version": optional_int(model_info.get("agentos_routing_policy_version")),
        "routing_strategy": model_info.get("agentos_routing_strategy"),
        "error_type": standard.get("error_type"),
        "fallback_depth": optional_int(fallback_depth) or 0,
        "latency_ms": _duration_ms(started_at, completed_at),
        "first_token_latency_ms": first_token_latency_ms,
        "provider_cost": provider_cost,
        "is_silent_experiment": silent_experiment,
        "input_tokens": optional_int(first(usage.get("prompt_tokens"), standard.get("prompt_tokens"))),
        "output_tokens": optional_int(first(usage.get("completion_tokens"), standard.get("completion_tokens"))),
        "cached_tokens": optional_int(
            first(
                prompt_details.get("cached_tokens"),
                usage.get("cache_read_input_tokens"),
                standard.get("cache_read_input_tokens"),
                0,
            )
        ),
        "started_at": started_at,
        "completed_at": completed_at,
        "call_status": "success",
        "org_id": key_metadata.get("org_id"),
        "team_id": key_metadata.get("team_id"),
        "user_id": key_metadata.get("user_id"),
        "profile_id": profile_id,
        "key_type": key_type,
        "key_hash": trusted_attribution.get("key_hash") or standard_metadata.get("user_api_key_hash"),
        "agent_instance_id": agent_instance_id,
        "task_run_id": task_run_id,
        "skill_key": skill_key,
    }
    if runtime_session_id is not None:
        payload["runtime_session_id"] = str(runtime_session_id)
    return payload


def build_preflight_payload(user_api_key_dict: object, data: Mapping[str, object]) -> Mapping[str, object] | None:
    auth = _mapping(user_api_key_dict)
    key_metadata = _mapping(auth.get("metadata"))
    if not any(key_metadata.get(key) for key in ("org_id", "team_id", "user_id", "profile_id")):
        return None
    request_metadata = _mapping(data.get("metadata"))
    key_hash = auth.get("api_key")
    if isinstance(key_hash, str) and key_hash.lower().startswith(("sk-", "bearer ")):
        key_hash = None
    key_type = key_metadata.get("key_type") or ("profile" if key_metadata.get("profile_id") else "virtual")
    profile_id = key_metadata.get("profile_id")
    agent_instance_id = key_metadata.get("agent_instance_id")
    if key_type in ("virtual", "evaluation"):
        profile_id = agent_instance_id = None
    payload = {  # mutable-ok: Native preflight JSON dictionary receives the evaluation-only skill field before signing.
        "model": str(data.get("model") or "unknown"),
        "org_id": key_metadata.get("org_id"),
        "team_id": key_metadata.get("team_id"),
        "user_id": key_metadata.get("user_id"),
        "profile_id": profile_id,
        "key_type": key_type,
        "key_hash": key_hash,
        "agent_instance_id": agent_instance_id,
        "runtime_session_id": request_metadata.get("session_id") or request_metadata.get("litellm_session_id"),
    }
    if key_type == "evaluation":
        payload["skill_key"] = key_metadata.get("skill_key")
    return payload


class AgentOSUsageCallback(CustomLogger):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # Native CustomLogger constructor is not fully annotated.
        self.url = os.environ.get(
            "AGENTOS_USAGE_CALLBACK_URL",
            "http://agentos-server:8080/internal/ai-gateway/usage",
        )
        self.preflight_url = os.environ.get(
            "AGENTOS_USAGE_PREFLIGHT_URL",
            "http://agentos-server:8080/internal/ai-gateway/preflight",
        )
        self.token = os.environ.get("AGENTOS_BFF_INTERNAL_TOKEN", "")
        self.outbox_path = os.environ.get("AGENTOS_USAGE_OUTBOX_PATH", "")
        self.flush_interval_seconds = max(
            1,
            int(os.environ.get("AGENTOS_USAGE_OUTBOX_FLUSH_SECONDS", "5")),
        )
        self._stop = threading.Event()
        if self.token:
            if not self.outbox_path:
                raise RuntimeError("AGENTOS_USAGE_OUTBOX_PATH is required when usage callbacks are enabled")
            self._init_outbox()
            self._refresh_outbox_metrics()
            threading.Thread(
                target=self._drain_loop,
                name="agentos-usage-outbox",
                daemon=True,
            ).start()

    async def async_pre_call_hook(
        self,
        user_api_key_dict: object,
        cache: object,
        data: dict[str, object],  # mutable-ok: LiteLLM's pre-call hook mutates the original request metadata in place.
        call_type: str,
    ) -> dict[str, object] | None:  # mutable-ok: Native pre-call API returns the original mutable request dictionary.
        del cache, call_type
        payload = build_preflight_payload(user_api_key_dict, data)
        if payload is None:
            return data
        if not self.token:
            raise HTTPException(status_code=503, detail="AgentOS usage preflight is not configured")
        await asyncio.to_thread(self._preflight, payload)
        metadata = dict(  # mutable-ok: Replace caller-controlled attribution in the native request dictionary.
            _mapping(data.get("metadata"))
        )
        metadata[_ATTRIBUTION_KEY] = _seal_attribution(payload, self.token)
        data["metadata"] = metadata
        return data

    async def async_log_success_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        if not _has_billable_attribution(kwargs, self.token):
            return
        try:
            payload = build_usage_payload(
                kwargs,
                response_obj,
                start_time,
                end_time,
                internal_token=self.token,
            )
            await self._send(payload)
        except Exception:
            logger.exception("AgentOS usage callback failed")

    async def async_log_failure_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        if not _has_billable_attribution(kwargs, self.token):
            return
        try:
            payload = build_usage_payload(
                kwargs,
                response_obj,
                start_time,
                end_time,
                internal_token=self.token,
            )
            exception = kwargs.get("exception") or _mapping(kwargs.get("litellm_params")).get("exception")
            payload["call_status"] = "timeout" if "timeout" in str(exception).casefold() else "failed"
            payload["error_type"] = (
                type(exception).__name__ if isinstance(exception, BaseException) else "UpstreamError"
            )
            await self._send(payload)
        except Exception:
            logger.exception("AgentOS failed-call usage callback failed")

    async def _send(self, payload: Mapping[str, object]) -> None:
        if not self.token:
            raise RuntimeError("AgentOS usage callback is not configured")
        await asyncio.to_thread(self._enqueue, payload)

    def _post_with_retries(self, payload: Mapping[str, object]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        error: Exception | None = None
        for attempt in range(1, 4):
            request = Request(
                self.url,
                data=body,
                headers={  # mutable-ok: urllib.request.Request accepts a native header dictionary.
                    "Content-Type": "application/json",
                    "X-AgentOS-Internal-Token": self.token,
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=5) as response:  # pyright: ignore[reportAny]  # urllib's native response context manager is untyped.
                    result = TypeAdapter(dict[str, object]).validate_json(response.read() or b"{}")  # pyright: ignore[reportAny]  # Validate the native HTTP response at the JSON boundary.
                usage = _mapping(result.get("usage"))
                if usage.get("scoring_status") == "quarantined":
                    logger.warning(
                        "AgentOS quarantined LiteLLM call %s: %s",
                        payload["call_id"],
                        usage.get("quarantine_reason"),
                    )
                return
            except HTTPError as exc:
                detail = exc.read().decode(errors="replace")
                if exc.code < 500:
                    raise PermanentUsageDeliveryError(
                        f"AgentOS usage callback rejected ({exc.code}): {detail}"
                    ) from exc
                error = exc
            except (URLError, TimeoutError, OSError) as exc:
                error = exc
            if attempt < 3:
                time.sleep(0.25 * attempt)
        raise RuntimeError(f"AgentOS usage callback exhausted retries: {error}")

    def _connect_outbox(self) -> sqlite3.Connection:
        path = Path(self.outbox_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _init_outbox(self) -> None:
        with self._connect_outbox() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_outbox (
                    call_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _enqueue(self, payload: Mapping[str, object]) -> None:
        call_id = str(payload["call_id"])
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        now = time.time()
        with self._connect_outbox() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO usage_outbox (
                    call_id, payload, status, attempts, next_attempt_at,
                    last_error, created_at, updated_at
                ) VALUES (?, ?, 'pending', 0, 0, NULL, ?, ?)
                """,
                (call_id, encoded, now, now),
            )
            row = TypeAdapter[tuple[str] | None](tuple[str] | None).validate_python(
                connection.execute(
                    "SELECT payload FROM usage_outbox WHERE call_id = ?",
                    (call_id,),
                ).fetchone()
            )
        if row is None or row[0] != encoded:
            raise RuntimeError(f"usage outbox call_id collision: {call_id}")
        self._refresh_outbox_metrics()

    def _flush_pending(self, limit: int = 100) -> int:
        now = time.time()
        with self._connect_outbox() as connection:
            rows = TypeAdapter(tuple[tuple[str, str, int], ...]).validate_python(
                connection.execute(
                    """
                SELECT call_id, payload, attempts
                FROM usage_outbox
                WHERE status = 'pending' AND next_attempt_at <= ?
                ORDER BY created_at, call_id
                LIMIT ?
                """,
                    (now, limit),
                ).fetchall()
            )
        delivered = 0
        for call_id, encoded, attempts in rows:
            try:
                self._post_with_retries(TypeAdapter(dict[str, object]).validate_json(encoded))
            except PermanentUsageDeliveryError as exc:
                self._mark_outbox(call_id, "dead_letter", attempts + 1, str(exc), now)
                logger.error("AgentOS usage outbox dead-lettered %s: %s", call_id, exc)
            except Exception as exc:
                delay = min(300, 2 << min(attempts, 7))
                self._mark_outbox(call_id, "pending", attempts + 1, str(exc), now + delay)
                logger.warning("AgentOS usage outbox will retry %s: %s", call_id, exc)
            else:
                with self._connect_outbox() as connection:
                    connection.execute("DELETE FROM usage_outbox WHERE call_id = ?", (call_id,))
                delivered += 1
        self._refresh_outbox_metrics()
        return delivered

    def _mark_outbox(
        self,
        call_id: str,
        status: str,
        attempts: int,
        error: str,
        next_attempt_at: float,
    ) -> None:
        with self._connect_outbox() as connection:
            connection.execute(
                """
                UPDATE usage_outbox
                SET status = ?, attempts = ?, next_attempt_at = ?,
                    last_error = ?, updated_at = ?
                WHERE call_id = ?
                """,
                (status, attempts, next_attempt_at, error[:2000], time.time(), call_id),
            )

    def _refresh_outbox_metrics(self) -> None:
        now = time.time()
        with self._connect_outbox() as connection:
            pending, dead_letter, oldest_pending = TypeAdapter(
                tuple[int | None, int | None, float | None]
            ).validate_python(
                connection.execute(
                    """
                SELECT
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN status = 'dead_letter' THEN 1 ELSE 0 END),
                    MIN(CASE WHEN status = 'pending' THEN created_at END)
                FROM usage_outbox
                """
                ).fetchone()
            )
        _OUTBOX_PENDING.set(int(pending or 0))
        _OUTBOX_DEAD_LETTER.set(int(dead_letter or 0))
        _OUTBOX_OLDEST_PENDING_AGE.set(max(0.0, now - float(oldest_pending)) if oldest_pending else 0.0)

    def _drain_loop(self) -> None:
        while not self._stop.wait(self.flush_interval_seconds):
            try:
                self._flush_pending()
            except Exception:
                logger.exception("AgentOS usage outbox drain failed")

    def _preflight(self, payload: Mapping[str, object]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        error: Exception | None = None
        for attempt in range(1, 4):
            request = Request(
                self.preflight_url,
                data=body,
                headers={  # mutable-ok: urllib.request.Request accepts a native header dictionary.
                    "Content-Type": "application/json",
                    "X-AgentOS-Internal-Token": self.token,
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=5):
                    return
            except HTTPError as exc:
                detail = exc.read().decode(errors="replace")
                if exc.code < 500:
                    raise HTTPException(
                        status_code=429 if exc.code == 409 else exc.code,
                        detail=f"AgentOS usage preflight rejected: {detail}",
                    ) from exc
                error = exc
            except (URLError, TimeoutError, OSError) as exc:
                error = exc
            if attempt < 3:
                time.sleep(0.25 * attempt)
        raise HTTPException(status_code=503, detail=f"AgentOS usage preflight unavailable: {error}")


agentos_usage_callback = AgentOSUsageCallback()
