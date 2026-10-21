"""
Test that models not in the cost map do NOT bypass budget enforcement.

Regression test for the bug where unmapped models got fallback costs of 0,
causing _is_model_cost_zero() to return True and skip all budget checks.

See: https://github.com/BerriAI/litellm/issues/24770
"""

import copy

import pytest
from pydantic import JsonValue

import litellm
from litellm.proxy.auth.auth_checks import _is_model_cost_zero
from litellm.router import Router


class TestUnmappedModelBudgetEnforcement:
    """Unmapped models must NOT bypass budget checks."""

    def setup_method(self):
        """Snapshot litellm.model_cost before each test."""
        self._saved_model_cost = litellm.model_cost
        litellm.model_cost = copy.deepcopy(self._saved_model_cost)
        litellm.utils.invalidate_model_cost_cache()

    def teardown_method(self):
        """Restore litellm.model_cost after each test."""
        litellm.model_cost = self._saved_model_cost
        litellm.utils.invalidate_model_cost_cache()

    def test_unmapped_model_enforces_budget(self):
        """A model not in litellm.model_cost should have budget enforced."""
        router = Router(
            model_list=[
                {
                    "model_name": "custom-model",
                    "litellm_params": {
                        "model": "openai/totally-nonexistent-model-xyz",
                        "api_key": "sk-fake",
                    },
                },
            ]
        )
        result = _is_model_cost_zero(model="custom-model", llm_router=router)
        assert result is False, "Unmapped model should enforce budget (return False), not bypass it (return True)"

    @pytest.mark.parametrize(
        ("pricing", "expected_free"),
        [
            ({"output_cost_per_image": 2.0}, False),
            ({"input_cost_per_image": 0.02}, False),
            ({"input_cost_per_pixel": 0.000001}, False),
            ({"output_cost_per_second": 0.1}, False),
            ({"output_cost_per_image_above_16384_tokens": 0.6}, False),
            (
                {"image_resolution_pricing": {"2K": {"output_cost_per_image": 0.2}}},
                False,
            ),
            ({"output_cost_per_image": 0.0}, True),
        ],
    )
    def test_non_token_pricing_enforces_budget(self, pricing: dict[str, JsonValue], expected_free: bool) -> None:
        router = Router(
            model_list=[
                {
                    "model_name": "image-model",
                    "litellm_params": {
                        "model": "siliconflow/Kwai-Kolors/Kolors",
                        "api_key": "sk-test",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {"mode": "image_generation", **pricing},
                },
            ]
        )

        assert _is_model_cost_zero("image-model", router) is expected_free

    @pytest.mark.parametrize(
        ("pricing", "expected_free"),
        [
            ({"output_cost_per_video": 2.0}, False),
            ({"output_cost_per_video_per_second": 0.2}, False),
            ({"output_cost_per_second": 0.1}, False),
            ({"output_cost_per_second_480p": 0.1}, False),
            ({"output_cost_per_second_720p": 0.2}, False),
            ({"output_cost_per_second_1080p": 0.4}, False),
            ({"output_cost_per_video_token": 0.000037}, False),
            ({"video_token_pricing": {"no_video_input": 0.000037, "video_input": 0.000022}}, False),
            ({"output_cost_per_video": 0.0}, True),
        ],
    )
    def test_video_pricing_enforces_budget(self, pricing: dict[str, JsonValue], expected_free: bool) -> None:
        router = Router(
            model_list=[
                {
                    "model_name": "video-model",
                    "litellm_params": {
                        "model": "openai/budget-test-video",
                        "api_key": "sk-test",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {"mode": "video_generation", **pricing},
                },
            ]
        )

        assert _is_model_cost_zero("video-model", router) is expected_free

    def test_paid_image_deployment_in_otherwise_free_group_enforces_budget(self) -> None:
        router = Router(
            model_list=[
                {
                    "model_name": "image-model",
                    "litellm_params": {
                        "model": "siliconflow/Kwai-Kolors/Kolors",
                        "api_key": "sk-test",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {
                        "id": f"image-deployment-{index}",
                        "output_cost_per_image": image_cost,
                    },
                }
                for index, image_cost in enumerate((0.0, 2.0))
            ]
        )

        assert _is_model_cost_zero("image-model", router) is False

    def test_cost_map_update_invalidates_free_image_budget_bypass(self) -> None:
        from litellm.utils import invalidate_model_cost_cache

        model = "siliconflow/Kwai-Kolors/Kolors"
        router = Router(
            model_list=[
                {
                    "model_name": "image-model",
                    "litellm_params": {
                        "model": model,
                        "api_key": "sk-test",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
            ]
        )
        assert _is_model_cost_zero("image-model", router) is True

        litellm.model_cost = {
            **litellm.model_cost,
            model: {**litellm.model_cost[model], "output_cost_per_image": 2.0},
        }
        invalidate_model_cost_cache()

        assert _is_model_cost_zero("image-model", router) is False

    def test_explicitly_free_model_bypasses_budget(self):
        """A model with explicit cost=0 in model_info should bypass budget."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-model",
                    "litellm_params": {
                        "model": "ollama/llama2",
                        "api_base": "http://localhost:11434",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {
                        "id": "free-model-id",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
            ]
        )
        result = _is_model_cost_zero(model="free-model", llm_router=router)
        assert result is True, "Explicitly free model should bypass budget (return True)"

    def test_known_paid_model_enforces_budget(self):
        """A model in the cost map with non-zero costs should enforce budget."""
        router = Router(
            model_list=[
                {
                    "model_name": "paid-model",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "sk-fake",
                    },
                },
            ]
        )
        result = _is_model_cost_zero(model="paid-model", llm_router=router)
        assert result is False, "Known paid model should enforce budget (return False)"

    def test_unmapped_model_with_litellm_params_pricing(self):
        """A model with cost=0 in litellm_params (not model_info) should bypass budget."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-via-params",
                    "litellm_params": {
                        "model": "openai/nonexistent-but-free-model",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
            ]
        )
        result = _is_model_cost_zero(model="free-via-params", llm_router=router)
        assert result is True, "Model with explicit cost=0 in litellm_params should bypass budget"

    def test_cache_invalidates_on_in_place_pricing_update(self):
        """
        Regression test for the stale-cache bug surfaced in PR review:
        upgrading an explicitly free deployment to paid via ``upsert_deployment``
        (same deployment count, same router instance) must invalidate the
        cached ``_is_model_cost_zero=True`` answer so budget checks resume
        immediately — not after the next proxy restart.
        """
        from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

        router = Router(
            model_list=[
                {
                    "model_name": "ramping-model",
                    "litellm_params": {
                        "model": "openai/ramping-deploy",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {
                        "id": "ramping-deploy-id",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
            ]
        )
        # Warm the cache as zero-cost.
        assert _is_model_cost_zero(model="ramping-model", llm_router=router) is True
        assert router._zero_cost_cache.get("ramping-model") is True

        # In-place pricing update: same deployment count, same router id,
        # same model name. The pre-fix cache key was
        # ``(id(router), len(model_list), model_name)`` and would not change.
        router.upsert_deployment(
            deployment=Deployment(
                model_name="ramping-model",
                litellm_params=LiteLLM_Params(
                    model="openai/ramping-deploy",
                    api_key="sk-fake",
                    input_cost_per_token=0.000002,
                    output_cost_per_token=0.000008,
                ),
                model_info=ModelInfo(
                    id="ramping-deploy-id",
                    input_cost_per_token=0.000002,
                    output_cost_per_token=0.000008,
                ),
            )
        )

        # Cache must have been cleared by ``_invalidate_model_group_info_cache``.
        assert router._zero_cost_cache == {}
        # Subsequent call sees the new pricing and enforces budget.
        assert _is_model_cost_zero(model="ramping-model", llm_router=router) is False

    def test_handles_router_without_zero_cost_cache_attribute(self):
        """Tolerate router-like objects (e.g. ``MagicMock`` stand-ins) that
        do not expose ``_zero_cost_cache`` — the auth check must still
        compute a correct answer, just without caching."""
        from unittest.mock import MagicMock

        from litellm.types.router import ModelGroupInfo

        mock_router = MagicMock(spec=Router)
        mock_router.model_list = []
        mock_router.get_model_group_info.return_value = ModelGroupInfo(
            model_group="paid-model",
            providers=["openai"],
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        # Strip the attribute so the helper falls back to the no-cache path.
        del mock_router._zero_cost_cache

        result = _is_model_cost_zero(model="paid-model", llm_router=mock_router)
        assert result is False
