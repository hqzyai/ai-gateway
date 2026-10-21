"""
Tests for model-cost-map loading: the model-count integrity check (which must
count actual model entries, not reserved meta keys) and the extraction of the
``fallback_generalizations`` block out of the raw map.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.litellm_core_utils.fallback_generalizations import (
    get_fallback_generalization_rules,
    match_capability_generalizations,
    match_routing_generalization,
    set_fallback_generalizations,
)
from litellm.litellm_core_utils.get_model_cost_map import (
    FALLBACK_GENERALIZATIONS_KEY,
    GetModelCostMap,
    _count_model_entries,
    _finalize_model_cost_map,
    get_model_cost_map,
    get_model_cost_map_source_info,
)


def _make_models(n: int) -> dict:
    return {
        f"model-{i}": {"litellm_provider": "openai", "mode": "chat"} for i in range(n)
    }


def test_count_model_entries_excludes_reserved_keys():
    m = _make_models(3)
    m["sample_spec"] = {"foo": "bar"}
    m[FALLBACK_GENERALIZATIONS_KEY] = {"rules": []}
    assert _count_model_entries(m) == 3


def test_validation_rejects_truly_shrunk_file_even_with_meta_keys():
    """A file with only a handful of real models must be rejected as corrupt,
    and the extra meta keys must not inflate the count past the minimum."""
    shrunk = _make_models(5)
    shrunk["sample_spec"] = {"foo": "bar"}
    shrunk[FALLBACK_GENERALIZATIONS_KEY] = {"rules": [{"name": "x"}]}

    assert (
        GetModelCostMap.validate_model_cost_map(
            fetched_map=shrunk,
            backup_model_count=2000,
            min_model_count=50,
        )
        is False
    )


def test_validation_accepts_healthy_file_with_meta_keys():
    healthy = _make_models(2000)
    healthy["sample_spec"] = {"foo": "bar"}
    healthy[FALLBACK_GENERALIZATIONS_KEY] = {"rules": []}

    assert (
        GetModelCostMap.validate_model_cost_map(
            fetched_map=healthy,
            backup_model_count=2000,
            min_model_count=50,
        )
        is True
    )


def test_validation_rejects_significant_shrink_vs_backup():
    # 600 real models vs a 2000-model backup is below the 50% shrink threshold.
    shrunk = _make_models(600)
    shrunk[FALLBACK_GENERALIZATIONS_KEY] = {"rules": []}
    assert (
        GetModelCostMap.validate_model_cost_map(
            fetched_map=shrunk,
            backup_model_count=2000,
            min_model_count=50,
            max_shrink_ratio=0.5,
        )
        is False
    )


def test_finalize_pops_key_and_installs_rules():
    previous = list(get_fallback_generalization_rules())
    try:
        raw = _make_models(2)
        raw[FALLBACK_GENERALIZATIONS_KEY] = {
            "rules": [
                {
                    "name": "rule",
                    "pattern": r"^widget-",
                    "model_info": {"litellm_provider": "openai"},
                }
            ]
        }
        finalized = _finalize_model_cost_map(raw)

        # The reserved key is removed from the returned model map ...
        assert FALLBACK_GENERALIZATIONS_KEY not in finalized
        # ... and its rules are installed into the generalizations module.
        assert match_routing_generalization("widget-9") == "openai"
    finally:
        set_fallback_generalizations(previous)


def test_finalize_with_no_block_clears_rules():
    previous = list(get_fallback_generalization_rules())
    try:
        set_fallback_generalizations(
            [{"name": "stale", "pattern": r"^x", "model_info": {"a": 1}}]
        )
        _finalize_model_cost_map(_make_models(2))
        assert match_capability_generalizations("x-1") is None
    finally:
        set_fallback_generalizations(previous)


def test_finalize_limits_seed_providers_and_preserves_aliases_and_pricing() -> None:
    previous = list(get_fallback_generalization_rules())
    try:
        raw = {
            "siliconflow/Qwen/Qwen3-8B": {"litellm_provider": "siliconflow", "input_cost_per_token": 0},
            "doubao-seed-2-0-pro-260215": {
                "litellm_provider": "volcengine",
                "aliases": ["volcengine/doubao-seed-2-0-pro-260215"],
                "tiered_pricing": [{"input_cost_per_token": 1e-6}],
            },
            "dashscope/qwen-image-3.0-pro": {"litellm_provider": "dashscope", "output_cost_per_image": 0.1},
            "other/model": {"litellm_provider": "openai", "aliases": ["siliconflow/unrelated"]},
            "dashscope/wrong-provider": {"litellm_provider": "openrouter"},
            "sample_spec": {"litellm_provider": "dashscope"},
            "unknown": {"mode": "chat"},
        }

        result = _finalize_model_cost_map(raw)

        assert set(result) == {
            "siliconflow/Qwen/Qwen3-8B",
            "doubao-seed-2-0-pro-260215",
            "volcengine/doubao-seed-2-0-pro-260215",
            "dashscope/qwen-image-3.0-pro",
        }
        assert result["siliconflow/Qwen/Qwen3-8B"]["input_cost_per_token"] == 0
        assert result["dashscope/qwen-image-3.0-pro"]["output_cost_per_image"] == 0.1
        assert result["doubao-seed-2-0-pro-260215"]["tiered_pricing"] == [{"input_cost_per_token": 1e-6}]
        assert result["volcengine/doubao-seed-2-0-pro-260215"] is result["doubao-seed-2-0-pro-260215"]
        assert "aliases" not in result["doubao-seed-2-0-pro-260215"]
    finally:
        set_fallback_generalizations(previous)


@pytest.mark.parametrize("source", ["default", "local", "remote", "network_failure", "invalid_remote"])
def test_loading_and_reloading_only_initializes_three_providers(source: str, monkeypatch: pytest.MonkeyPatch) -> None:
    previous = list(get_fallback_generalization_rules())
    backup = GetModelCostMap.load_local_model_cost_map()
    expected_models = {
        name
        for name, info in backup.items()
        if info.get("litellm_provider") in {"siliconflow", "volcengine", "dashscope"}
    }
    remote_get = Mock(
        return_value=httpx.Response(
            200,
            json={} if source == "invalid_remote" else backup,
            request=httpx.Request("GET", "https://pricing.example/models.json"),
        ),
        side_effect=httpx.ConnectError("unavailable") if source == "network_failure" else None,
    )
    monkeypatch.setattr(httpx, "get", remote_get)
    if source == "default":
        monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    else:
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True" if source == "local" else "False")
    try:
        initial = get_model_cost_map("https://pricing.example/models.json")
        reloaded = get_model_cost_map("https://pricing.example/models.json")

        assert initial == reloaded
        assert expected_models <= initial.keys()
        assert {info["litellm_provider"] for info in initial.values()} == {"siliconflow", "volcengine", "dashscope"}
        assert "sample_spec" not in initial
        assert remote_get.call_count == (0 if source in {"default", "local"} else 2)
        assert get_model_cost_map_source_info()["source"] == ("remote" if source == "remote" else "local")
        assert get_model_cost_map_source_info()["is_env_forced"] is (source == "local")
    finally:
        set_fallback_generalizations(previous)


def test_shipped_backup_contains_all_selected_provider_pricing() -> None:
    source = json.loads((Path(__file__).resolve().parents[3] / "model_prices_and_context_window.json").read_text())
    backup = GetModelCostMap.load_local_model_cost_map()
    selected_providers = {"siliconflow", "volcengine", "dashscope"}

    assert {name: info for name, info in backup.items() if info.get("litellm_provider") in selected_providers} == {
        name: info for name, info in source.items() if info.get("litellm_provider") in selected_providers
    }


def test_shipped_backup_carries_the_claude_routing_rules():
    """The bundled backup must ship the Claude routing rules so a fresh install
    (or an offline fallback) routes unknown Claude models without code changes.
    Bedrock-syntax ids must hit the bedrock rule before the bare-id Anthropic rule."""
    backup = GetModelCostMap.load_local_model_cost_map()
    rules = backup.get(FALLBACK_GENERALIZATIONS_KEY, {}).get("rules", [])
    names = [r.get("name") for r in rules]
    assert names.index("bedrock-claude-ids") < names.index("anthropic-claude-ids")

    previous = list(get_fallback_generalization_rules())
    try:
        set_fallback_generalizations(rules)
        assert match_routing_generalization("claude-opus-4-9") == "anthropic"
        assert match_routing_generalization("global.anthropic.claude-opus-4-9") == "bedrock"
    finally:
        set_fallback_generalizations(previous)


def test_shipped_routing_rules_never_match_through_an_unrecognized_namespace():
    """Routing rules decide ``litellm_provider`` for otherwise-unknown ids, and the
    proxy's wildcard access check (``can_key_call_model`` with a ``bedrock/*`` key)
    trusts that inference: it rebuilds ``{provider}/{model}`` and matches it against
    the key's patterns. A routing pattern that matches as a substring lets
    ``bedrockz/anthropic.claude-...`` resolve to bedrock and slip through a
    ``bedrock/*`` key, so every shipped routing rule must anchor to the start of
    the name and never match an id carrying an unrecognized namespace prefix."""
    backup = GetModelCostMap.load_local_model_cost_map()
    rules = backup[FALLBACK_GENERALIZATIONS_KEY]["rules"]

    routing_rules = [r for r in rules if "litellm_provider" in r["model_info"]]
    assert routing_rules
    assert all(r["pattern"].startswith("^") for r in routing_rules)

    previous = list(get_fallback_generalization_rules())
    try:
        set_fallback_generalizations(rules)
        for bedrock_id in [
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-v2:1",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "us-gov.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "global.anthropic.claude-fable-5-20260120-v1:0",
        ]:
            assert match_routing_generalization(bedrock_id) == "bedrock", bedrock_id
        for namespaced in [
            "bedrockz/anthropic.claude-3-5-sonnet-20240620",
            "bedrockz/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "bedrockz/claude-3-5-sonnet-20240620",
        ]:
            assert match_routing_generalization(namespaced) is None, namespaced
    finally:
        set_fallback_generalizations(previous)


def test_shipped_backup_marks_claude_4_6_plus_adaptive_not_4_0():
    """Adaptive thinking is data, not code. The bundled backup must carry
    supports_adaptive_thinking on genuine Claude >= 4.6 entries (every provider
    route) and on the version-gated anthropic-claude-adaptive-thinking rule for
    unmapped future Claudes, while leaving the dated Claude 4.0 names
    ("...-4-20250514") unflagged so a date can never be mistaken for a 4.6+ minor
    version. The version-neutral claude-family-baseline capability rule must not flag
    it, so an unmapped sub-4.6 name resolves but stays non-adaptive. The adaptive rule
    carries only its delta; capability unioning stacks it onto the baseline, so the
    baseline block is never duplicated across rules and no rule needs ``extends``."""
    backup = GetModelCostMap.load_local_model_cost_map()

    rules = backup[FALLBACK_GENERALIZATIONS_KEY]["rules"]
    baseline_rule = next(r for r in rules if r.get("name") == "claude-family-baseline")
    adaptive_rule = next(r for r in rules if r.get("name") == "claude-adaptive-thinking")
    assert "supports_adaptive_thinking" not in baseline_rule["model_info"]
    assert "litellm_provider" not in baseline_rule["model_info"]
    assert adaptive_rule["model_info"] == {"supports_adaptive_thinking": True}
    assert all("extends" not in r for r in rules)

    for adaptive in [
        "anthropic.claude-opus-4-8",
        "vertex_ai/claude-opus-4-6@default",
        "us.anthropic.claude-sonnet-4-6",
        "openrouter/anthropic/claude-opus-4.7",
        "azure_ai/claude-opus-4-7",
    ]:
        assert backup[adaptive]["supports_adaptive_thinking"] is True, adaptive

    for non_adaptive in [
        "claude-opus-4-20250514",
        "us.anthropic.claude-opus-4-20250514-v1:0",
        "claude-opus-4-5",
    ]:
        assert "supports_adaptive_thinking" not in backup[non_adaptive], non_adaptive
