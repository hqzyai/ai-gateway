import json
import pytest
from litellm.integrations.agentos import spend_attribution


def _spend_module(monkeypatch):
    return spend_attribution, None


def test_spend_payload_rewrites_model_fields_to_logical_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = _spend_module(monkeypatch)
    payload = module.apply_logical_model_alias_to_spend_payload(
        {
            "litellm_params": {
                "model_info": {"agentos_model_alias": "xiaomi"},
            }
        },
        {"model": "agentos-lm-lm_1", "model_group": "agentos-lm-lm_1"},
    )
    assert payload["model"] == "xiaomi"
    assert payload["model_group"] == "xiaomi"
    assert "agentos_routing_hops" in payload["metadata"]


@pytest.mark.parametrize("legacy", [None, "old-alias"])
def test_spend_prefers_native_model_group_name(monkeypatch: pytest.MonkeyPatch, legacy: str | None) -> None:
    module, _ = _spend_module(monkeypatch)
    payload = module.apply_logical_model_alias_to_spend_payload(
        {
            "litellm_params": {
                "model_info": {
                    "agentos_model_group_name": " deepseek ",
                    "agentos_model_alias": legacy,
                    "litellm_provider": "siliconflow",
                }
            }
        },
        {"model": "upstream", "model_group": "upstream", "custom_llm_provider": "openai"},
    )
    assert payload["model"] == payload["model_group"] == "deepseek"
    metadata = json.loads(payload["metadata"])
    assert metadata["agentos_provider"] == "siliconflow"
    assert metadata["agentos_transport_provider"] == "openai"
    hop = metadata["agentos_routing_hops"][0]
    assert hop["model_group_name"] == "deepseek"
    assert "model_alias" not in hop


def test_routing_hops_canonicalize_legacy_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = _spend_module(monkeypatch)
    payload = module.apply_logical_model_alias_to_spend_payload(
        {
            "litellm_params": {
                "model_info": {"agentos_model_group_name": "deepseek"},
                "metadata": {
                    "previous_models": [
                        {
                            "litellm_params": {
                                "api_key": "secret-must-not-leak",
                                "model_info": {"agentos_model_alias": "legacy-group"},
                            }
                        }
                    ],
                },
            }
        },
        {"model": "deepseek"},
    )
    hops = json.loads(payload["metadata"])["agentos_routing_hops"]
    assert [hop["model_group_name"] for hop in hops] == ["legacy-group", "deepseek"]
    assert all("model_alias" not in hop for hop in hops)
    assert "secret-must-not-leak" not in payload["metadata"]


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("provider", [None, "dashscope"])
def test_routing_annotation_preserves_compression_recall_and_cache_usage(
    monkeypatch: pytest.MonkeyPatch, serialized: bool, provider: str | None
) -> None:
    module, _ = _spend_module(monkeypatch)
    metadata = {
        "guardrail_information": [
            {
                "guardrail_provider": "headroom",
                "guardrail_response": {
                    "tokens_before": 1000,
                    "headroom_compressed_tokens": 400,
                    "ccr_initial_input_tokens": 400,
                    "ccr_followup_input_tokens": 800,
                    "tokens_after": 1200,
                    "tokens_saved": -200,
                    "ccr_status": "retrieved",
                },
            }
        ],
        "compression_gross_saved_tokens": 600,
        "compression_extra_input_tokens": 800,
        "compression_saved_tokens": -200,
        "additional_usage_values": {"cache_read_input_tokens": 256},
        "spend_logs_metadata": {"hermes_session_id": "session-test"},
    }
    expected = json.loads(json.dumps(metadata))
    payload = module.apply_logical_model_alias_to_spend_payload(
        {"litellm_params": {"model_info": {"agentos_model_group_name": "qwen", "litellm_provider": provider}}},
        {
            "model": "agentos-lm-lm_1",
            "custom_llm_provider": "openai",
            "metadata": json.dumps(metadata) if serialized else metadata,
        },
    )
    actual = json.loads(payload["metadata"])
    for key, value in expected.items():
        if key != "spend_logs_metadata":
            assert actual[key] == value
    assert actual["spend_logs_metadata"]["hermes_session_id"] == "session-test"
    assert actual["agentos_routing_hops"][-1]["model_group_name"] == "qwen"
    if provider:
        assert actual["agentos_provider"] == provider
        assert actual["agentos_transport_provider"] == "openai"


def test_spend_persists_actual_model_separately_from_group(monkeypatch):
    module, _ = _spend_module(monkeypatch)
    payload = module.apply_logical_model_alias_to_spend_payload(
        {
            "litellm_params": {
                "model_info": {
                    "agentos_model_alias": "shared-group",
                    "agentos_upstream_model": "dashscope/qwen3.8-flash",
                    "litellm_provider": "dashscope",
                }
            }
        },
        {"model": "shared-group", "metadata": {}},
    )
    metadata = json.loads(payload["metadata"])
    assert metadata["agentos_upstream_model"] == "dashscope/qwen3.8-flash"
    assert payload["model"] == "shared-group"
