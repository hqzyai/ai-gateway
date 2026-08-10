from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.common_utils.periodic_reload_schedule import (
    MODEL_COST_MAP_RELOAD_PARAM_NAME,
    pod_reload_is_due,
    read_reload_schedule,
    utc_now,
)
from litellm.proxy.management_endpoints import model_cost_map_endpoints as endpoints


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
    schedule_reload = AsyncMock(return_value=7)
    adopt_revision = MagicMock()
    apply_cost_map = MagicMock()
    repository = MagicMock()
    repository.set_param = set_param
    monkeypatch.setattr(endpoints, "_get_prisma_client", lambda: prisma_client)
    monkeypatch.setattr(
        endpoints, "load_model_cost_map_overrides", AsyncMock(return_value={"existing": {"mode": "chat"}})
    )
    monkeypatch.setattr(endpoints, "_schedule_cluster_reload", schedule_reload)
    monkeypatch.setattr(endpoints, "_adopt_reload_revision", adopt_revision)
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
    # The serving pod already holds the new prices, so it adopts the revision it published
    # instead of reading its own announcement back and refetching on the next poll.
    adopt_revision.assert_called_once_with(7)
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


class _FakeConfigRow:
    def __init__(self, param_name: str) -> None:
        self.param_name = param_name
        self.param_value: object = None
        self.reload_revision = 0
        self.last_run_at = None


class _FakeConfigTable:
    """In-memory stand-in for the LiteLLM_Config row.

    Models the one prisma behaviour this mechanism depends on: the atomic
    ``{"increment": n}`` write on reload_revision. A MagicMock records the call but never
    advances the counter, so it cannot show whether another pod would notice the change.
    """

    def __init__(self) -> None:
        self.rows: dict[str, _FakeConfigRow] = {}  # mutable-ok: models a mutable database table

    async def find_unique(self, where):
        return self.rows.get(where["param_name"])

    async def upsert(self, where, data):
        param_name = where["param_name"]
        existing = self.rows.get(param_name)
        row = existing if existing is not None else _FakeConfigRow(param_name)
        self.rows[param_name] = row
        for field, value in (data["update"] if existing is not None else data["create"]).items():
            increment = value.get("increment") if isinstance(value, dict) else None
            setattr(row, field, getattr(row, field, 0) + increment if increment is not None else value)
        return row


def _prisma_with(table: _FakeConfigTable) -> MagicMock:
    prisma_client = MagicMock()
    prisma_client.db.litellm_config = table
    return prisma_client


@pytest.mark.asyncio
async def test_schedule_cluster_reload_bumps_the_shared_revision():
    """Regression: the cross-pod signal has to be the counter the pollers actually read.

    This used to write a ``force_reload`` flag into the config JSON. Nothing reads that for the
    model cost map any more, so the announcement went nowhere and every other pod kept serving
    the pre-override prices.
    """
    table = _FakeConfigTable()
    prisma_client = _prisma_with(table)

    first = await endpoints._schedule_cluster_reload(prisma_client)
    second = await endpoints._schedule_cluster_reload(prisma_client)

    assert (first, second) == (1, 2)
    assert table.rows[MODEL_COST_MAP_RELOAD_PARAM_NAME].reload_revision == 2


@pytest.mark.asyncio
async def test_saved_override_makes_an_unaware_pod_reload():
    """Ties the writer to the real poller predicate so the two halves cannot drift apart again.

    Overrides are stored in the database but applied to each pod's in-process litellm.model_cost,
    so a save that does not reach the other pods leaves them pricing requests with stale rates.
    """
    table = _FakeConfigTable()
    prisma_client = _prisma_with(table)
    now = utc_now()

    await endpoints._schedule_cluster_reload(prisma_client)

    schedule = await read_reload_schedule(prisma_client, MODEL_COST_MAP_RELOAD_PARAM_NAME)
    assert schedule is not None

    def _is_due(pod_applied_revision: int) -> bool:
        return pod_reload_is_due(
            schedule=schedule,
            pod_applied_revision=pod_applied_revision,
            pod_data_loaded_at=now,
            current_time=now,
            description="Model cost map",
        )

    # A pod that has not seen this save reloads; once it has, it stops.
    assert _is_due(0) is True
    assert _is_due(schedule.reload_revision) is False
