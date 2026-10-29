import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

import pytest

import litellm
from litellm.proxy.spend_tracking.savings import compute_savings_spend, recorded_savings


@pytest.mark.parametrize("actual_cost, expected", [(None, 0.008), (0.003, 0.007)])
def test_recorded_cache_savings_use_saved_prices(actual_cost, expected):
    metadata = {
        "usage_object": {"prompt_tokens": 2000, "prompt_tokens_details": {"cached_tokens": 1000}},
        "model_map_information": {
            "model_map_value": {
                "input_cost_per_token": 0.00001,
                "cache_read_input_token_cost": 0.000002,
            }
        },
        "cost_breakdown": {} if actual_cost is None else {"cache_read_cost": actual_cost},
    }
    assert recorded_savings(metadata)["prompt_caching_savings_spend"] == pytest.approx(expected)


def test_recorded_unknown_prices_are_not_zero_and_empty_usage_is_zero():
    assert recorded_savings({"usage_object": {"cache_read_input_tokens": 100}})["prompt_caching_savings_spend"] is None
    assert all(value == 0 for value in recorded_savings({}).values())


def test_recorded_snapshot_wins_over_changed_prices():
    assert (
        recorded_savings(
            {
                "optimization_savings": {"prompt_caching_savings_spend": 0.25},
                "usage_object": {"cache_read_input_tokens": 100},
            }
        )["prompt_caching_savings_spend"]
        == 0.25
    )


def test_recorded_signed_compression_and_long_context_prices():
    amounts = recorded_savings(
        {
            "compression_savings": {"tokens_saved": -10},
            "usage_object": {"prompt_tokens": 200000, "cache_read_input_tokens": 100},
            "model_map_information": {
                "model_map_value": {
                    "input_cost_per_token": 0.01,
                    "input_cost_per_token_above_128k_tokens": 0.02,
                    "cache_read_input_token_cost": 0.001,
                }
            },
        }
    )
    assert amounts["compression_savings_spend"] == pytest.approx(-0.2)
    assert amounts["compression_extra_input_spend"] == pytest.approx(0.2)
    assert amounts["prompt_caching_savings_spend"] == pytest.approx(1.9)


def _anthropic_costs(model: str) -> tuple[float, float]:
    info = litellm.get_model_info(model=model, custom_llm_provider="anthropic")
    input_cost = info["input_cost_per_token"] or 0.0
    cache_read_cost = info.get("cache_read_input_token_cost") or input_cost
    return input_cost, cache_read_cost


def test_compression_savings_priced_at_input_rate():
    input_cost, _ = _anthropic_costs("claude-sonnet-5")
    result = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=4389,
        cache_read_input_tokens=0,
    )
    assert result.compression == pytest.approx(4389 * input_cost)
    assert result.compression > 0
    assert result.prompt_caching == 0.0


def test_prompt_caching_savings_priced_at_input_minus_cache_read():
    input_cost, cache_read_cost = _anthropic_costs("claude-sonnet-5")
    # A model that supports prompt caching must charge less to read from cache;
    # otherwise this test is asserting nothing.
    assert cache_read_cost < input_cost
    result = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=0,
        cache_read_input_tokens=8200,
    )
    assert result.prompt_caching == pytest.approx(8200 * (input_cost - cache_read_cost))
    assert result.prompt_caching > 0
    assert result.compression == 0.0


def test_unknown_model_fails_open_to_zero():
    result = compute_savings_spend(
        model="totally-made-up-model-xyz",
        custom_llm_provider="anthropic",
        compression_saved_tokens=1000,
        cache_read_input_tokens=1000,
    )
    assert result.compression == 0.0
    assert result.prompt_caching == 0.0


def test_missing_model_fails_open_to_zero():
    result = compute_savings_spend(
        model=None,
        custom_llm_provider=None,
        compression_saved_tokens=1000,
        cache_read_input_tokens=1000,
    )
    assert result.compression == 0.0
    assert result.prompt_caching == 0.0


def test_negative_compression_tokens_produce_negative_net_savings():
    input_cost, _ = _anthropic_costs("claude-sonnet-5")
    result = compute_savings_spend(
        model="claude-sonnet-5",
        custom_llm_provider="anthropic",
        compression_saved_tokens=-500,
        compression_gross_saved_tokens=1000,
        compression_extra_input_tokens=1500,
        cache_read_input_tokens=-500,
    )
    assert result.compression == pytest.approx(-500 * input_cost)
    assert result.compression_gross == pytest.approx(1000 * input_cost)
    assert result.compression_extra_input == pytest.approx(1500 * input_cost)
    assert result.prompt_caching == 0.0
