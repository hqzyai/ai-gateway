from __future__ import annotations
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import pytest
from litellm.integrations.agentos import usage_callback as callback_module
from litellm.integrations.agentos import usage_outbox_admin


def test_attribution_signature_matches_original_patch_and_rejects_tampering():
    token = "regression-test-token"
    payload = {
        "model": "模型A", "org_id": "org", "team_id": "team", "user_id": "user",
        "profile_id": "p", "key_type": "profile", "key_hash": "hash",
        "agent_instance_id": "instance", "runtime_session_id": "s", "skill_key": None,
        "request": "not signed",
    }
    sealed = callback_module._seal_attribution(payload, token)
    assert sealed["signature"] == "8334fa8a2df0f85f4b3fc03d046dfb8a89efc4b15eb98914c5470f6fc9c37715"
    assert "request" not in sealed["payload"]
    request = {"litellm_params": {"metadata": {"_agentos_attribution": sealed}}}
    assert callback_module._verified_attribution(request, token) == sealed["payload"]
    tampered = {**sealed, "payload": {**sealed["payload"], "org_id": "other-org"}}
    assert not callback_module._verified_attribution({"litellm_params": {"metadata": {"_agentos_attribution": tampered}}}, token)
    assert not callback_module._verified_attribution(request, "wrong-token")


def _callback_module(monkeypatch):
    gauges = {}

    class Gauge:
        def __init__(self):
            self.value = 0

        def set(self, value):
            self.value = value

    for field, name in (
        ("_OUTBOX_PENDING", "agentos_usage_outbox_pending"),
        ("_OUTBOX_DEAD_LETTER", "agentos_usage_outbox_dead_letter"),
        ("_OUTBOX_OLDEST_PENDING_AGE", "agentos_usage_outbox_oldest_pending_age_seconds"),
    ):
        gauge = Gauge()
        gauges[name] = gauge
        monkeypatch.setattr(callback_module, field, gauge)
    monkeypatch.setitem(callback_module.__dict__, "_test_gauges", gauges)
    monkeypatch.delenv("AGENTOS_BFF_INTERNAL_TOKEN", raising=False)
    monkeypatch.setenv("AGENTOS_USAGE_OUTBOX_FLUSH_SECONDS", "3600")
    return callback_module


def _outbox_admin_module():
    return usage_outbox_admin


def test_callback_builds_whitelisted_usage_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    callback = _callback_module(monkeypatch)
    now = datetime.now(UTC)

    payload = callback.build_usage_payload(
        {
            "model": "qwen",
            "litellm_call_id": "call-1",
            "messages": [{"role": "user", "content": "must not leave gateway"}],
            "standard_logging_object": {
                "id": "shared-request-id",
                "model": "provider/qwen",
                "model_group": "qwen",
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "metadata": {
                    "user_api_key_hash": "hashed-key-1",
                    "user_api_key_metadata": {
                        "org_id": "org-1",
                        "team_id": "team-1",
                        "user_id": "user-1",
                        "profile_id": "profile-1",
                        "agent_instance_id": "instance-1",
                        "key_type": "profile",
                    },
                    "requester_metadata": {
                        "task_run_id": "task-1",
                        "skill_key": "search",
                    },
                },
            },
            "litellm_params": {
                "metadata": {
                    "user_api_key": "sk-must-not-leak",
                }
            },
        },
        {
            "id": "provider-response-1",
            "model": "provider/qwen",
            "choices": [{"message": {"content": "must not leave gateway"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 10},
            },
        },
        now,
        now,
    )

    assert payload == {
        "call_id": "call-1",
        "model": "provider/qwen",
        "model_group_name": "qwen",
        "logical_request_id": "shared-request-id",
        "attempt_id": "call-1",
        "logical_model_id": None,
        "deployment_id": None,
        "provider_account_id": None,
        "credential_version": None,
        "upstream_model": "provider/qwen",
        "routing_policy_version": None,
        "routing_strategy": None,
        "error_type": None,
        "fallback_depth": 0,
        "latency_ms": 0.0,
        "first_token_latency_ms": None,
        "provider_cost": None,
        "is_silent_experiment": False,
        "input_tokens": 100,
        "output_tokens": 20,
        "cached_tokens": 10,
        "started_at": now.isoformat(),
        "completed_at": now.isoformat(),
        "call_status": "success",
        "org_id": "org-1",
        "team_id": "team-1",
        "user_id": "user-1",
        "profile_id": "profile-1",
        "key_type": "profile",
        "key_hash": "hashed-key-1",
        "agent_instance_id": "instance-1",
        "task_run_id": None,
        "skill_key": None,
    }
    assert "sk-must-not-leak" not in json.dumps(payload)
    assert "must not leave gateway" not in json.dumps(payload)


def test_callback_keeps_virtual_owner_but_drops_profile_task_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = _callback_module(monkeypatch)
    now = datetime.now(UTC)

    payload = callback.build_usage_payload(
        {
            "litellm_call_id": "call-virtual",
            "standard_logging_object": {
                "model": "provider/qwen",
                "model_group": "qwen",
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "metadata": {
                    "user_api_key_hash": "hashed-virtual-key",
                    "user_api_key_metadata": {
                        "org_id": "org-1",
                        "team_id": "team-1",
                        "user_id": "user-1",
                        "key_type": "virtual",
                    },
                    "requester_metadata": {
                        "profile_id": "forged-profile",
                        "agent_instance_id": "forged-instance",
                        "task_run_id": "forged-task",
                        "skill_key": "forged-skill",
                    },
                },
            },
        },
        {},
        now,
        now,
    )

    assert payload["key_type"] == "virtual"
    assert payload["key_hash"] == "hashed-virtual-key"
    assert payload["org_id"] == "org-1"
    assert payload["team_id"] == "team-1"
    assert payload["user_id"] == "user-1"
    assert payload["profile_id"] is None
    assert payload["agent_instance_id"] is None
    assert payload["task_run_id"] is None
    assert payload["skill_key"] is None


def test_callback_emits_physical_routing_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    callback = _callback_module(monkeypatch)
    now = datetime.now(UTC)

    payload = callback.build_usage_payload(
        {
            "litellm_call_id": "request-1",
            "litellm_params": {
                "model": "agentos-lm-lm_1",
                "model_info": {
                    "id": "md_1",
                    "agentos_logical_model_id": "lm_1",
                    "agentos_model_alias": "deepseek-v4-pro",
                    "agentos_provider_account_id": "pa_1",
                    "agentos_credential_version": 3,
                    "agentos_upstream_model": "openai/deepseek-v4-pro",
                    "agentos_routing_policy_version": 2,
                    "agentos_routing_strategy": "latency-based-routing",
                },
            },
            "standard_logging_object": {
                "id": "request-1",
                "model": "openai/deepseek-v4-pro",
                "model_group": "agentos-lm-lm_1",
                "response_cost": 0.012,
                "metadata": {
                    "fallback_depth": 1,
                    "is_silent_experiment": True,
                },
            },
        },
        {},
        now,
        now,
    )

    assert payload["logical_request_id"] == "request-1"
    assert payload["call_id"].startswith("request-1:")
    assert payload["attempt_id"] == payload["call_id"]
    assert payload["model_group_name"] == "deepseek-v4-pro"
    assert payload["model"] == "openai/deepseek-v4-pro"
    assert payload["logical_model_id"] == "lm_1"
    assert payload["deployment_id"] == "md_1"
    assert payload["provider_account_id"] == "pa_1"
    assert payload["credential_version"] == 3
    assert payload["upstream_model"] == "openai/deepseek-v4-pro"
    assert payload["routing_policy_version"] == 2
    assert payload["routing_strategy"] == "latency-based-routing"
    assert payload["fallback_depth"] == 1
    assert payload["provider_cost"] == 0.012
    assert payload["is_silent_experiment"] is True


@pytest.mark.parametrize("source", ["params", "standard"])
def test_callback_prefers_canonical_model_group_name(monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    callback = _callback_module(monkeypatch)
    now = datetime.now(UTC)
    info = {
        "agentos_model_group_name": " deepseek ",
        "agentos_model_alias": "legacy-group",
        "api_key": "secret-must-not-leak",
    }
    payload = callback.build_usage_payload(
        {
            "litellm_call_id": "native-group-call",
            "litellm_params": {"model_info": info if source == "params" else {}},
            "standard_logging_object": {
                "model": "openai/upstream",
                "model_group": "fallback-group",
                "model_info": info if source == "standard" else {},
            },
        },
        {},
        now,
        now,
    )
    assert payload["model_group_name"] == "deepseek"
    assert "model_alias" not in payload
    assert "secret-must-not-leak" not in json.dumps(payload)


def test_callback_uses_distinct_idempotency_keys_for_fallback_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = _callback_module(monkeypatch)
    now = datetime.now(UTC)

    def attempt(deployment_id: str):
        return callback.build_usage_payload(
            {
                "litellm_call_id": "logical-request-1",
                "litellm_params": {
                    "model": "openai/deepseek",
                    "model_info": {
                        "id": deployment_id,
                        "agentos_logical_model_id": "lm_1",
                        "agentos_routing_policy_version": 2,
                    },
                },
                "standard_logging_object": {
                    "model": "openai/deepseek",
                    "model_group": "agentos-lm-lm_1",
                },
            },
            {},
            now,
            now,
        )

    first = attempt("md_1")
    fallback = attempt("md_2")

    assert first["logical_request_id"] == fallback["logical_request_id"] == "logical-request-1"
    assert first["call_id"] != fallback["call_id"]
    assert first["attempt_id"] == first["call_id"]
    assert fallback["attempt_id"] == fallback["call_id"]


def test_callback_preserves_signed_evaluation_skill_without_employee_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = _callback_module(monkeypatch)
    now = datetime.now(UTC)
    preflight = callback.build_preflight_payload(
        {
            "api_key": "hashed-evaluation-key",
            "metadata": {
                "org_id": "org-1",
                "team_id": "team-1",
                "key_type": "evaluation",
                "skill_key": "general",
            },
        },
        {"model": "qwen", "messages": [{"role": "user", "content": "private"}]},
    )
    sealed = callback._seal_attribution(preflight, "internal-token")
    payload = callback.build_usage_payload(
        {
            "litellm_call_id": "evaluation-call",
            "litellm_params": {"metadata": {"_agentos_attribution": sealed}},
        },
        {},
        now,
        now,
        internal_token="internal-token",
    )

    assert preflight["skill_key"] == "general"
    assert payload["key_type"] == "evaluation"
    assert payload["skill_key"] == "general"
    assert payload["user_id"] is None
    assert payload["profile_id"] is None
    assert "private" not in json.dumps(preflight)


def test_callback_ignores_caller_supplied_key_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    callback = _callback_module(monkeypatch)
    now = datetime.now(UTC)

    payload = callback.build_usage_payload(
        {
            "model": "qwen",
            "litellm_call_id": "call-forged-owner",
            "standard_logging_object": {
                "model": "provider/qwen",
                "model_group": "qwen",
                "metadata": {
                    "requester_metadata": {
                        "org_id": "org-victim",
                        "team_id": "team-victim",
                        "user_id": "user-victim",
                    }
                },
            },
            "litellm_params": {
                "metadata": {
                    "user_api_key_hash": "forged-key-hash",
                    "user_api_key_metadata": {
                        "org_id": "org-victim",
                        "team_id": "team-victim",
                        "user_id": "user-victim",
                        "key_type": "virtual",
                    },
                }
            },
        },
        {},
        now,
        now,
    )

    assert payload["key_type"] is None
    assert payload["key_hash"] is None
    assert payload["org_id"] is None
    assert payload["team_id"] is None
    assert payload["user_id"] is None


def test_callback_links_trusted_litellm_session_and_builds_preflight_without_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = _callback_module(monkeypatch)
    now = datetime.now(UTC)
    payload = callback.build_usage_payload(
        {
            "litellm_call_id": "call-session",
            "standard_logging_object": {
                "model": "provider/qwen",
                "model_group": "qwen",
                "session_id": "hermes-session-1",
                "metadata": {
                    "user_api_key_hash": "hashed-key-1",
                    "user_api_key_metadata": {
                        "org_id": "org-1",
                        "team_id": "team-1",
                        "user_id": "user-1",
                        "profile_id": "profile-1",
                        "agent_instance_id": "instance-1",
                        "key_type": "profile",
                    },
                },
            },
        },
        {},
        now,
        now,
    )
    preflight = callback.build_preflight_payload(
        {
            "api_key": "hashed-key-1",
            "metadata": {
                "org_id": "org-1",
                "team_id": "team-1",
                "user_id": "user-1",
                "profile_id": "profile-1",
                "agent_instance_id": "instance-1",
                "key_type": "profile",
            },
        },
        {
            "model": "qwen",
            "metadata": {"session_id": "hermes-session-1"},
            "messages": [{"role": "user", "content": "must stay inside gateway"}],
        },
    )

    assert payload["runtime_session_id"] == "hermes-session-1"
    assert preflight == {
        "model": "qwen",
        "org_id": "org-1",
        "team_id": "team-1",
        "user_id": "user-1",
        "profile_id": "profile-1",
        "key_type": "profile",
        "key_hash": "hashed-key-1",
        "agent_instance_id": "instance-1",
        "runtime_session_id": "hermes-session-1",
    }
    assert "must stay inside gateway" not in json.dumps(preflight)


def test_callback_uses_signed_preflight_attribution_for_failed_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    callback = _callback_module(monkeypatch)
    now = datetime.now(UTC)
    attribution = {
        "model": "qwen",
        "org_id": "org-1",
        "team_id": "team-1",
        "user_id": "user-1",
        "profile_id": "profile-1",
        "key_type": "profile",
        "key_hash": "hashed-key-1",
        "agent_instance_id": "instance-1",
        "runtime_session_id": "session-1",
    }
    sealed = callback._seal_attribution(attribution, "internal-token")

    payload = callback.build_usage_payload(
        {
            "litellm_call_id": "failed-call-1",
            "litellm_params": {"metadata": {"_agentos_attribution": sealed}},
        },
        {},
        now,
        now,
        internal_token="internal-token",
    )

    assert payload["model"] == "qwen"
    assert payload["model_group_name"] == "qwen"
    assert payload["org_id"] == "org-1"
    assert payload["profile_id"] == "profile-1"
    assert payload["key_hash"] == "hashed-key-1"
    assert payload["runtime_session_id"] == "session-1"


@pytest.mark.asyncio
async def test_preflight_overwrites_untrusted_attribution_with_signed_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    callback = _callback_module(monkeypatch)
    monkeypatch.setenv("AGENTOS_BFF_INTERNAL_TOKEN", "internal-token")
    monkeypatch.setenv("AGENTOS_USAGE_OUTBOX_PATH", str(tmp_path / "usage-outbox.sqlite3"))
    monkeypatch.setenv("AGENTOS_USAGE_OUTBOX_FLUSH_SECONDS", "3600")
    handler = callback.AgentOSUsageCallback()
    monkeypatch.setattr(handler, "_preflight", lambda payload: None)
    data = {
        "model": "qwen",
        "metadata": {"session_id": "session-1", "_agentos_attribution": {"profile_id": "forged"}},
        "messages": [{"role": "user", "content": "must stay private"}],
    }

    result = await handler.async_pre_call_hook(
        {
            "api_key": "hashed-key-1",
            "metadata": {
                "org_id": "org-1",
                "team_id": "team-1",
                "user_id": "user-1",
                "profile_id": "profile-1",
                "agent_instance_id": "instance-1",
                "key_type": "profile",
            },
        },
        None,
        data,
        "completion",
    )

    trusted = callback._verified_attribution(
        {"litellm_params": {"metadata": result["metadata"]}},
        "internal-token",
    )
    assert trusted["profile_id"] == "profile-1"
    assert "must stay private" not in json.dumps(result["metadata"])


def test_callback_posts_with_internal_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    callback = _callback_module(monkeypatch)
    monkeypatch.setenv("AGENTOS_BFF_INTERNAL_TOKEN", "internal-token")
    monkeypatch.setenv("AGENTOS_USAGE_OUTBOX_PATH", str(tmp_path / "usage-outbox.sqlite3"))
    handler = callback.AgentOSUsageCallback()
    captured = {}

    class Response:
        def read(self) -> bytes:
            return b'{"usage":{"scoring_status":"scored"},"created":true}'

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

    def fake_urlopen(request, timeout: int):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(callback, "urlopen", fake_urlopen)
    handler._post_with_retries({"call_id": "call-1"})

    assert captured["timeout"] == 5
    assert captured["request"].get_header("X-agentos-internal-token") == "internal-token"


@pytest.mark.asyncio
async def test_callback_send_durably_enqueues_without_inline_delivery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    callback = _callback_module(monkeypatch)
    monkeypatch.setenv("AGENTOS_BFF_INTERNAL_TOKEN", "internal-token")
    monkeypatch.setenv("AGENTOS_USAGE_OUTBOX_PATH", str(tmp_path / "usage-outbox.sqlite3"))
    monkeypatch.setenv("AGENTOS_USAGE_OUTBOX_FLUSH_SECONDS", "3600")
    handler = callback.AgentOSUsageCallback()
    monkeypatch.setattr(
        handler,
        "_flush_pending",
        lambda *args: pytest.fail(f"inline outbox flush: {args}"),
    )
    monkeypatch.setattr(
        handler,
        "_post_with_retries",
        lambda payload: pytest.fail(f"inline network delivery: {payload}"),
    )

    await handler._send({"call_id": "queued-call"})

    with handler._connect_outbox() as connection:
        row = connection.execute(
            "SELECT payload, status, attempts FROM usage_outbox WHERE call_id = ?",
            ("queued-call",),
        ).fetchone()
    handler._stop.set()
    assert row == ('{"call_id":"queued-call"}', "pending", 0)


def test_callback_persists_failed_delivery_and_retries_after_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    callback = _callback_module(monkeypatch)
    outbox = tmp_path / "usage-outbox.sqlite3"
    monkeypatch.setenv("AGENTOS_BFF_INTERNAL_TOKEN", "internal-token")
    monkeypatch.setenv("AGENTOS_USAGE_OUTBOX_PATH", str(outbox))
    monkeypatch.setenv("AGENTOS_USAGE_OUTBOX_FLUSH_SECONDS", "3600")
    first = callback.AgentOSUsageCallback()

    def unavailable(payload):
        raise RuntimeError(f"unavailable: {payload['call_id']}")

    monkeypatch.setattr(first, "_post_with_retries", unavailable)
    first._enqueue({"call_id": "durable-call"})
    assert first._flush_pending() == 0
    first._stop.set()

    second = callback.AgentOSUsageCallback()
    delivered = []
    monkeypatch.setattr(second, "_post_with_retries", lambda payload: delivered.append(payload))
    with second._connect_outbox() as connection:
        connection.execute("UPDATE usage_outbox SET next_attempt_at = 0 WHERE call_id = 'durable-call'")
    assert second._flush_pending() == 1
    assert delivered == [{"call_id": "durable-call"}]
    with second._connect_outbox() as connection:
        assert connection.execute("SELECT COUNT(*) FROM usage_outbox").fetchone()[0] == 0


def test_callback_dead_letters_permanent_4xx_and_updates_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    callback = _callback_module(monkeypatch)
    monkeypatch.setenv("AGENTOS_BFF_INTERNAL_TOKEN", "internal-token")
    monkeypatch.setenv("AGENTOS_USAGE_OUTBOX_PATH", str(tmp_path / "usage-outbox.sqlite3"))
    monkeypatch.setenv("AGENTOS_USAGE_OUTBOX_FLUSH_SECONDS", "3600")
    handler = callback.AgentOSUsageCallback()

    def rejected(request, timeout: int):
        del request, timeout
        raise callback.HTTPError(
            handler.url,
            400,
            "bad request",
            {},
            BytesIO(b"invalid usage"),
        )

    monkeypatch.setattr(callback, "urlopen", rejected)
    handler._enqueue({"call_id": "rejected-call", "private": "must-not-print"})

    assert handler._flush_pending() == 0
    with handler._connect_outbox() as connection:
        row = connection.execute(
            "SELECT status, attempts FROM usage_outbox WHERE call_id = ?",
            ("rejected-call",),
        ).fetchone()
    assert row == ("dead_letter", 1)
    gauges = callback.__dict__["_test_gauges"]
    assert gauges["agentos_usage_outbox_pending"].value == 0
    assert gauges["agentos_usage_outbox_dead_letter"].value == 1
    assert gauges["agentos_usage_outbox_oldest_pending_age_seconds"].value == 0


def test_dead_letter_can_be_requeued_once_and_delivered_without_duplicate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    callback = _callback_module(monkeypatch)
    admin = _outbox_admin_module()
    outbox = tmp_path / "usage-outbox.sqlite3"
    monkeypatch.setenv("AGENTOS_BFF_INTERNAL_TOKEN", "internal-token")
    monkeypatch.setenv("AGENTOS_USAGE_OUTBOX_PATH", str(outbox))
    monkeypatch.setenv("AGENTOS_USAGE_OUTBOX_FLUSH_SECONDS", "3600")
    handler = callback.AgentOSUsageCallback()
    handler._enqueue(
        {
            "call_id": "manual-requeue-call",
            "secret": "must-never-appear-in-operator-output",
        }
    )
    handler._mark_outbox(
        "manual-requeue-call",
        "dead_letter",
        1,
        "permanent rejection",
        0,
    )

    assert admin.main(["--database", str(outbox), "status"]) == 0
    output = capsys.readouterr().out
    assert json.loads(output) == {
        "pending": 0,
        "dead_letter": 1,
        "oldest_pending_age_seconds": 0.0,
    }
    assert "must-never-appear" not in output

    assert admin.requeue_dead_letters(outbox, call_id="manual-requeue-call") == 1
    assert admin.requeue_dead_letters(outbox, call_id="manual-requeue-call") == 0
    delivered = []
    monkeypatch.setattr(handler, "_post_with_retries", lambda payload: delivered.append(payload))

    assert handler._flush_pending() == 1
    assert handler._flush_pending() == 0
    assert delivered == [
        {
            "call_id": "manual-requeue-call",
            "secret": "must-never-appear-in-operator-output",
        }
    ]


@pytest.mark.asyncio
async def test_callback_forwards_failed_call_without_inventing_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    callback = _callback_module(monkeypatch)
    handler = callback.AgentOSUsageCallback()
    captured = []

    async def capture(payload):
        captured.append(payload)

    monkeypatch.setattr(handler, "_send", capture)
    now = datetime.now(UTC)
    await handler.async_log_failure_event(
        {
            "model": "qwen",
            "litellm_call_id": "call-timeout",
            "exception": TimeoutError("provider timeout"),
            "standard_logging_object": {
                "id": "call-timeout",
                "model": "provider/qwen",
                "model_group": "qwen",
                "metadata": {
                    "user_api_key_hash": "hashed-key-1",
                    "user_api_key_metadata": {
                        "org_id": "org-1",
                        "team_id": "team-1",
                        "user_id": "user-1",
                        "profile_id": "profile-1",
                        "key_type": "profile",
                    },
                },
            },
            "litellm_params": {"metadata": {}},
        },
        {},
        now,
        now,
    )

    assert captured[0]["call_status"] == "timeout"
    assert captured[0]["input_tokens"] is None
    assert captured[0]["output_tokens"] is None


@pytest.mark.asyncio
async def test_callback_skips_unattributed_master_key_canary(monkeypatch: pytest.MonkeyPatch) -> None:
    callback = _callback_module(monkeypatch)
    handler = callback.AgentOSUsageCallback()
    captured = []

    async def capture(payload):
        captured.append(payload)

    monkeypatch.setattr(handler, "_send", capture)
    now = datetime.now(UTC)
    kwargs = {
        "model": "qwen3.6-27b-nvfp4",
        "litellm_call_id": "model-canary-call",
        "standard_logging_object": {
            "model": "qwen3.6-27b-nvfp4",
            "model_group": "qwen3.6-27b-nvfp4",
            "metadata": {"requester_metadata": {"agentos_source": "model_canary"}},
        },
    }
    await handler.async_log_success_event(kwargs, {}, now, now)
    await handler.async_log_failure_event(kwargs | {"exception": TimeoutError("probe timeout")}, {}, now, now)

    assert captured == []


def test_chat_share_callback_preserves_share_identity_without_employee_debit(monkeypatch):
    callback = _callback_module(monkeypatch)
    identity = {"key_type": "chat_share", "org_id": "default", "team_id": "team", "agent_instance_id": "instance"}
    kwargs = {"standard_logging_object": {"metadata": {"user_api_key_metadata": identity, "user_api_key_hash": "hash"}}}
    assert callback._has_billable_attribution(kwargs, "")
    payload = callback.build_preflight_payload({"metadata": identity, "api_key": "hash"}, {"model": "test"})
    assert payload["key_type"] == "chat_share"
    assert payload["agent_instance_id"] == "instance"
    assert payload["user_id"] is None and payload["profile_id"] is None
