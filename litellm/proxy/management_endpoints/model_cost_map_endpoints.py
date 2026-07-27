from typing import Annotated, TypeAlias

import litellm
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, field_validator

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.utils import invalidate_config_param
from litellm.repositories.config_repository import ConfigRepository
from litellm.utils import invalidate_model_cost_cache


ModelCostValues: TypeAlias = dict[str, JsonValue]
ModelCostOverrides: TypeAlias = dict[str, ModelCostValues]
ModelCostMap: TypeAlias = dict[str, ModelCostValues]

_CONFIG_NAME = "model_cost_map_overrides"
_OVERRIDES_ADAPTER = TypeAdapter(ModelCostOverrides)
_MODEL_COST_MAP_ADAPTER = TypeAdapter(ModelCostMap)
UserAuth: TypeAlias = Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]


class ModelCostMapReloadConfig(BaseModel):
    interval_hours: int | None = None
    force_reload: bool = False


class ConfigParamValue(BaseModel):
    param_value: object

    model_config = ConfigDict(from_attributes=True)


class ModelCostMapOverrideRequest(BaseModel):
    model_name: str = Field(min_length=1, max_length=512)
    values: ModelCostValues

    model_config = ConfigDict(extra="forbid")

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Model name is required")
        return normalized

    @field_validator("values")
    @classmethod
    def validate_values(cls, value: ModelCostValues) -> ModelCostValues:
        if not value:
            raise ValueError("At least one pricing value is required")
        return value


class ModelCostMapOverrideResponse(BaseModel):
    model_name: str
    values: ModelCostValues


class ModelCostMapOverrideListResponse(BaseModel):
    overrides: tuple[ModelCostMapOverrideResponse, ...]


class ModelCostMapOverrideMutationResponse(BaseModel):
    status: str
    model_name: str
    override_count: int


router = APIRouter()


def parse_model_cost_map_overrides(value: object) -> ModelCostOverrides:
    if value is None:
        return {}
    return _OVERRIDES_ADAPTER.validate_python(value)


def merge_model_cost_map(
    base: object,
    overrides: ModelCostOverrides,
) -> ModelCostMap:
    validated_base = _MODEL_COST_MAP_ADAPTER.validate_python(base)
    merged_overrides = {
        model_name: {
            **validated_base.get(model_name, {}),
            **values,
        }
        for model_name, values in overrides.items()
    }
    return {**validated_base, **merged_overrides}


async def load_model_cost_map_overrides(prisma_client: object) -> ModelCostOverrides:
    config = await ConfigRepository(prisma_client).get_param(_CONFIG_NAME)
    raw_value = ConfigParamValue.model_validate(config).param_value if config is not None else None
    return parse_model_cost_map_overrides(raw_value)


async def _schedule_cluster_reload(prisma_client: object) -> None:
    repository = ConfigRepository(prisma_client)
    existing = await repository.get_param("model_cost_map_reload_config")
    reload_config = (
        ModelCostMapReloadConfig.model_validate(ConfigParamValue.model_validate(existing).param_value)
        if existing is not None
        else ModelCostMapReloadConfig()
    )
    await repository.set_param(
        "model_cost_map_reload_config",
        {
            "interval_hours": reload_config.interval_hours,
            "force_reload": True,
        },
    )
    await invalidate_config_param("model_cost_map_reload_config")


def _require_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Proxy admin role required")


def _require_admin_reader(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role not in {
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    }:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")


def _get_prisma_client() -> object:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database connection not available"
        )
    return prisma_client


def _apply_cost_map(cost_map: ModelCostMap) -> None:
    litellm.model_cost = cost_map
    invalidate_model_cost_cache()
    litellm.add_known_models(model_cost_map=cost_map)


def _load_latest_model_cost_map() -> ModelCostMap:
    from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map_for_validation

    return _MODEL_COST_MAP_ADAPTER.validate_python(get_model_cost_map_for_validation(litellm.model_cost_map_url))


@router.get(
    "/model/cost_map/overrides",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ModelCostMapOverrideListResponse,
)
async def list_model_cost_map_overrides(
    user_api_key_dict: UserAuth,
) -> ModelCostMapOverrideListResponse:
    _require_admin_reader(user_api_key_dict)
    overrides = await load_model_cost_map_overrides(_get_prisma_client())
    return ModelCostMapOverrideListResponse(
        overrides=tuple(
            ModelCostMapOverrideResponse(model_name=model_name, values=values)
            for model_name, values in sorted(overrides.items())
        )
    )


@router.put(
    "/model/cost_map/overrides",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ModelCostMapOverrideMutationResponse,
)
async def upsert_model_cost_map_override(
    data: ModelCostMapOverrideRequest,
    user_api_key_dict: UserAuth,
) -> ModelCostMapOverrideMutationResponse:
    _require_proxy_admin(user_api_key_dict)
    prisma_client = _get_prisma_client()
    existing = await load_model_cost_map_overrides(prisma_client)
    updated = {**existing, data.model_name: data.values}
    await ConfigRepository(prisma_client).set_param(_CONFIG_NAME, updated)
    await _schedule_cluster_reload(prisma_client)
    _apply_cost_map(merge_model_cost_map(_load_latest_model_cost_map(), updated))
    return ModelCostMapOverrideMutationResponse(
        status="success",
        model_name=data.model_name,
        override_count=len(updated),
    )


@router.delete(
    "/model/cost_map/overrides/{model_name:path}",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ModelCostMapOverrideMutationResponse,
)
async def delete_model_cost_map_override(
    model_name: str,
    user_api_key_dict: UserAuth,
) -> ModelCostMapOverrideMutationResponse:
    _require_proxy_admin(user_api_key_dict)
    prisma_client = _get_prisma_client()
    existing = await load_model_cost_map_overrides(prisma_client)
    if model_name not in existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pricing override not found")
    updated = {name: values for name, values in existing.items() if name != model_name}
    repository = ConfigRepository(prisma_client)
    if updated:
        await repository.set_param(_CONFIG_NAME, updated)
    else:
        await repository.delete_param(_CONFIG_NAME)
    await _schedule_cluster_reload(prisma_client)
    _apply_cost_map(merge_model_cost_map(_load_latest_model_cost_map(), updated))
    return ModelCostMapOverrideMutationResponse(
        status="success",
        model_name=model_name,
        override_count=len(updated),
    )
