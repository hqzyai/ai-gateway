import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

import litellm
from litellm.cost_calculator import cost_per_token
from litellm.llms.volcengine.chat.transformation import (
    VolcEngineChatConfig as VolcEngineConfig,
)
from litellm.types.utils import PromptTokensDetailsWrapper, Usage
from litellm.utils import get_optional_params


class TestVolcEngineConfig:
    def test_reasoning_effort_passthrough(self):
        model = "doubao-seed-2-0-pro-260215"
        config = VolcEngineConfig()

        assert "reasoning_effort" in config.get_supported_openai_params(model=model)

        mapped_params = get_optional_params(
            model=model,
            custom_llm_provider="volcengine",
            reasoning_effort="high",
            drop_params=False,
        )

        assert mapped_params["reasoning_effort"] == "high"

    def test_get_optional_params(self):
        config = VolcEngineConfig()
        supported_params = config.get_supported_openai_params(model="doubao-seed-1.6")
        assert "thinking" in supported_params

        # Test thinking disabled - should appear in extra_body
        mapped_params = config.map_openai_params(
            non_default_params={
                "thinking": {"type": "disabled"},
            },
            optional_params={},
            model="doubao-seed-1.6",
            drop_params=False,
        )

        # Fixed: thinking disabled should appear in extra_body
        assert mapped_params == {"extra_body": {"thinking": {"type": "disabled"}}}

        e2e_mapped_params = get_optional_params(
            model="doubao-seed-1.6",
            custom_llm_provider="volcengine",
            thinking={"type": "enabled"},
            drop_params=False,
        )

        assert "thinking" in e2e_mapped_params["extra_body"] and e2e_mapped_params[
            "extra_body"
        ]["thinking"] == {
            "type": "enabled",
        }

    def test_thinking_parameter_handling(self):
        """Test comprehensive thinking parameter handling scenarios"""
        config = VolcEngineConfig()

        # Test 1: thinking enabled - should appear in extra_body
        result_enabled = config.map_openai_params(
            non_default_params={"thinking": {"type": "enabled"}},
            optional_params={},
            model="doubao-seed-1.6",
            drop_params=False,
        )
        assert result_enabled == {"extra_body": {"thinking": {"type": "enabled"}}}

        # Test 2: thinking None - should NOT appear in extra_body
        result_none = config.map_openai_params(
            non_default_params={"thinking": None},
            optional_params={},
            model="doubao-seed-1.6",
            drop_params=False,
        )
        assert result_none == {}

        # Test 3: thinking with custom value - should NOT appear in extra_body (invalid value)
        result_custom = config.map_openai_params(
            non_default_params={"thinking": "custom_mode"},
            optional_params={},
            model="doubao-seed-1.6",
            drop_params=False,
        )
        assert result_custom == {}

        # Test 4: thinking disabled - should appear in extra_body with original structure
        result_disabled = config.map_openai_params(
            non_default_params={"thinking": {"type": "disabled"}},
            optional_params={},
            model="doubao-seed-1.6",
            drop_params=False,
        )
        assert result_disabled == {"extra_body": {"thinking": {"type": "disabled"}}}

        # Test 5: No thinking parameter - should return empty dict
        result_no_thinking = config.map_openai_params(
            non_default_params={},
            optional_params={},
            model="doubao-seed-1.6",
            drop_params=False,
        )
        assert result_no_thinking == {}

        # Test 6: invalid thinking type - should NOT appear in extra_body (invalid type)
        result_no_thinking = config.map_openai_params(
            non_default_params={"thinking": {"type": "invalid_type"}},
            optional_params={},
            model="doubao-seed-1.6",
            drop_params=False,
        )
        assert result_no_thinking == {}

        # Test 7: invalid thinking type - should NOT appear in extra_body (value is None)
        result_no_thinking = config.map_openai_params(
            non_default_params={"thinking": {"type": None}},
            optional_params={},
            model="doubao-seed-1.6",
            drop_params=False,
        )
        assert result_no_thinking == {}

    def test_e2e_completion(self):
        from openai import OpenAI

        from litellm import completion
        from litellm.types.utils import ModelResponse

        client = OpenAI(api_key="test_api_key")

        mock_raw_response = MagicMock()
        mock_raw_response.headers = {
            "x-request-id": "123",
            "openai-organization": "org-123",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "99",
        }
        mock_raw_response.parse.return_value = ModelResponse()

        with patch.object(
            client.chat.completions.with_raw_response, "create", mock_raw_response
        ) as mock_create:
            completion(
                model="volcengine/doubao-seed-1.6",
                messages=[
                    {
                        "role": "system",
                        "content": "**Tell me your model detail information.**",
                    }
                ],
                user="guest",
                stream=True,
                thinking={"type": "disabled"},
                client=client,
            )

            mock_create.assert_called_once()
            print(mock_create.call_args.kwargs)
            # Fixed: thinking disabled should appear in extra_body with original structure
            assert (
                "extra_body" in mock_create.call_args.kwargs
                and "thinking" in mock_create.call_args.kwargs.get("extra_body", {})
                and mock_create.call_args.kwargs.get("extra_body", {})["thinking"]
                == {"type": "disabled"}
            )


ARK_CHAT_TIERS = {
    "volcengine/doubao-seed-2-0-pro-260215": (
        (3.2e-06, 1.6e-05, 6.4e-07),
        (4.8e-06, 2.4e-05, 9.6e-07),
        (9.6e-06, 4.8e-05, 1.92e-06),
    ),
    "volcengine/doubao-seed-2-0-code-preview-260215": (
        (3.2e-06, 1.6e-05, 6.4e-07),
        (4.8e-06, 2.4e-05, 9.6e-07),
        (9.6e-06, 4.8e-05, 1.92e-06),
    ),
    "volcengine/doubao-seed-2-0-lite-260215": (
        (6e-07, 3.6e-06, 1.2e-07),
        (9e-07, 5.4e-06, 1.8e-07),
        (1.8e-06, 1.08e-05, 3.6e-07),
    ),
    "volcengine/doubao-seed-2-0-mini-260215": (
        (2e-07, 2e-06, 4e-08),
        (4e-07, 4e-06, 8e-08),
        (8e-07, 8e-06, 1.6e-07),
    ),
}

ARK_TIER_BOUNDARIES = ((1000, 0), (32768, 0), (32769, 1), (131072, 1), (131073, 2), (262144, 2))


@pytest.mark.parametrize("model", sorted(ARK_CHAT_TIERS))
@pytest.mark.parametrize(("prompt_tokens", "tier"), ARK_TIER_BOUNDARIES)
def test_seed_2_0_chat_cost_matches_ark_input_length_tier(model: str, prompt_tokens: int, tier: int) -> None:
    """doubao-seed-2.0 bills the whole request at the tier its prompt length falls in.

    These entries used to carry only ``tiered_pricing``, which no volcengine cost path
    reads, so every request tracked as $0. Ark's boundaries are 32768 and 131072 prompt
    tokens inclusive, so a prompt sitting exactly on a boundary stays in the lower tier.
    """
    input_rate, output_rate, _ = ARK_CHAT_TIERS[model][tier]
    completion_tokens = 1000

    prompt_cost, completion_cost = cost_per_token(
        model=model,
        custom_llm_provider="volcengine",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    assert prompt_cost == pytest.approx(input_rate * prompt_tokens)
    assert completion_cost == pytest.approx(output_rate * completion_tokens)


@pytest.mark.parametrize("model", sorted(ARK_CHAT_TIERS))
@pytest.mark.parametrize(("prompt_tokens", "tier"), ((1000, 0), (50000, 1), (200000, 2)))
def test_seed_2_0_chat_cached_prompt_uses_the_same_tier(model: str, prompt_tokens: int, tier: int) -> None:
    """Cache-read tokens are billed at the tier's own ContextSessionHit rate."""
    input_rate, _, cache_rate = ARK_CHAT_TIERS[model][tier]
    cached_tokens = 500

    prompt_cost, _ = cost_per_token(
        model=model,
        custom_llm_provider="volcengine",
        usage_object=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=100,
            total_tokens=prompt_tokens + 100,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens),
        ),
        prompt_tokens=prompt_tokens,
        completion_tokens=100,
    )

    expected = input_rate * (prompt_tokens - cached_tokens) + cache_rate * cached_tokens
    assert prompt_cost == pytest.approx(expected)


@pytest.mark.parametrize("model", sorted(ARK_CHAT_TIERS))
def test_seed_2_0_tiered_pricing_agrees_with_above_threshold_fields(model: str) -> None:
    """The two representations must not drift.

    ``tiered_pricing`` drives the proxy's pre-call budget reservation while the
    ``_above_<N>_tokens`` fields drive actual spend, so a rate present in one and not
    the other would reserve and bill different amounts for the same request.
    """
    entry = litellm.get_model_info(model=model, custom_llm_provider="volcengine")
    tiers = entry["tiered_pricing"]
    boundaries = (0, 32768, 131072)

    assert [tier["range"][0] for tier in tiers] == list(boundaries)

    for index, threshold in enumerate(boundaries):
        suffix = "" if index == 0 else f"_above_{threshold}_tokens"
        assert tiers[index]["input_cost_per_token"] == entry[f"input_cost_per_token{suffix}"]
        assert tiers[index]["output_cost_per_token"] == entry[f"output_cost_per_token{suffix}"]


@pytest.mark.parametrize(
    ("model", "expected_cost_per_image"),
    [
        ("volcengine/doubao-seedream-4-0-250828", 0.2),
        ("volcengine/doubao-seedream-4-5-251128", 0.25),
    ],
)
def test_seedream_4_x_models_are_priced(model: str, expected_cost_per_image: float) -> None:
    """These shipped without a price-map entry, so every generation tracked as $0."""
    entry = litellm.get_model_info(model=model, custom_llm_provider="volcengine")

    assert entry["mode"] == "image_generation"
    assert entry["output_cost_per_image"] == pytest.approx(expected_cost_per_image)
    assert entry["input_cost_per_image"] == 0.0
