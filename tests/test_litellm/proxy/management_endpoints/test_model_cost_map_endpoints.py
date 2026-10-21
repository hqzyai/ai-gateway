from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints import model_cost_map_endpoints as endpoints


def test_openai_pricing_is_only_loaded_from_manual_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    base = get_model_cost_map("")
    assert {info["litellm_provider"] for info in base.values()} == {"siliconflow", "volcengine", "dashscope"}
    assert not any(name.startswith("openai/") for name in base)
    manual_rule = {
        "openai/custom-model": {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
        }
    }

    initial = endpoints.merge_model_cost_map(base, manual_rule)
    reloaded = endpoints.merge_model_cost_map(get_model_cost_map(""), manual_rule)

    assert initial == reloaded
    assert len(reloaded) == len(base) + 1
    assert reloaded["openai/custom-model"] == manual_rule["openai/custom-model"]


def test_merge_model_cost_map_preserves_upstream_metadata_and_adds_models():
    base = {
        "volcengine/model": {"mode": "video_generation", "source": "upstream", "unmodified": True},
        "other/model": {"input_cost_per_token": 1e-6},
    }
    overrides = {
        "volcengine/model": {"video_token_pricing": {"video_input_1080p": 0.000031}},
        "custom/model": {"mode": "image_generation", "output_cost_per_image": 0.25},
    }

    merged = endpoints.merge_model_cost_map(base, overrides)

    assert merged == {
        "volcengine/model": {
            "mode": "video_generation",
            "source": "upstream",
            "unmodified": True,
            "video_token_pricing": {"video_input_1080p": 0.000031},
        },
        "other/model": {"input_cost_per_token": 1e-6},
        "custom/model": {"mode": "image_generation", "output_cost_per_image": 0.25},
    }
    assert base["volcengine/model"] == {"mode": "video_generation", "source": "upstream", "unmodified": True}


@pytest.mark.asyncio
async def test_upsert_override_persists_and_applies_rule(monkeypatch):
    prisma_client = object()
    set_param = AsyncMock()
    schedule_reload = AsyncMock()
    apply_cost_map = MagicMock()
    repository = MagicMock()
    repository.set_param = set_param
    monkeypatch.setattr(endpoints, "_get_prisma_client", lambda: prisma_client)
    monkeypatch.setattr(
        endpoints, "load_model_cost_map_overrides", AsyncMock(return_value={"existing": {"mode": "chat"}})
    )
    monkeypatch.setattr(endpoints, "_schedule_cluster_reload", schedule_reload)
    monkeypatch.setattr(endpoints, "_apply_cost_map", apply_cost_map)
    monkeypatch.setattr(
        endpoints,
        "_load_latest_model_cost_map",
        lambda: {"volcengine/model": {"mode": "video_generation"}},
    )
    monkeypatch.setattr(endpoints, "ConfigRepository", lambda _: repository)

    response = await endpoints.upsert_model_cost_map_override(
        data=endpoints.ModelCostMapOverrideRequest(
            model_name="volcengine/model",
            values={"video_token_pricing": {"video_input_1080p": 0.000031}},
        ),
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    set_param.assert_awaited_once_with(
        "model_cost_map_overrides",
        {
            "existing": {"mode": "chat"},
            "volcengine/model": {"video_token_pricing": {"video_input_1080p": 0.000031}},
        },
    )
    schedule_reload.assert_awaited_once_with(prisma_client)
    apply_cost_map.assert_called_once_with(
        {
            "existing": {"mode": "chat"},
            "volcengine/model": {
                "mode": "video_generation",
                "video_token_pricing": {"video_input_1080p": 0.000031},
            },
        }
    )
    assert response.override_count == 2


@pytest.mark.asyncio
async def test_upsert_override_rejects_non_admin():
    with pytest.raises(HTTPException) as error:
        await endpoints.upsert_model_cost_map_override(
            data=endpoints.ModelCostMapOverrideRequest(
                model_name="model",
                values={"input_cost_per_token": 1e-6},
            ),
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER),
        )

    assert getattr(error.value, "status_code", None) == 403
