import asyncio
import hmac

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict

from litellm.router_strategy.decision_observations import KEY, LIMIT, Decision, recorder


async def require_master_key(request: Request) -> None:
    from litellm.proxy.proxy_server import master_key

    expected = f"Bearer {master_key}" if master_key else ""
    provided = request.headers.get("authorization", "")
    if not expected or not hmac.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(status_code=403, detail="Routing observations require gateway master key")


router = APIRouter()


class RoutingDecisionsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    decisions: tuple[Decision, ...] = ()
    available: bool


@router.get("/agentos/routing-decisions", dependencies=(Depends(require_master_key),))
async def routing_decisions(
    model_group: str = Query(min_length=1, max_length=255),
    config_version: int | None = Query(default=None, ge=1),
) -> RoutingDecisionsResponse:
    try:
        raw = await asyncio.to_thread(recorder.client.lrange, KEY.format(model_group=model_group), 0, LIMIT - 1)
        decisions = tuple(Decision.model_validate_json(item) for item in reversed(raw))
    except Exception:
        return RoutingDecisionsResponse(available=False)
    return RoutingDecisionsResponse(
        decisions=tuple(item for item in decisions if config_version is None or item.config_version == config_version),
        available=True,
    )
